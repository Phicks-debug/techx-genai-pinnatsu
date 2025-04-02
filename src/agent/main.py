import os
import time
import json
import boto3
import signal
import asyncio
import nats

from nats.js.api import ConsumerConfig, AckPolicy
from nats.errors import TimeoutError, ConnectionClosedError, SlowConsumerError

from .._lib.logger import CustomLogger
from typing import Optional


# NATS client and JetStream context
nc, js, sub = None, None, None

# Initialize Bedrock client
client = boto3.client(service_name="bedrock-runtime", region_name="us-west-2")
model_id = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"

# Flag to control server shutdown
running = True

# Initialize logger
logger = CustomLogger()


async def connect_to_nats():
    """Connect to NATS server and get JetStream context"""
    global nc, js
    retry_count = 0
    max_retries = 5
    retry_delay = 2

    while retry_count < max_retries and running:
        try:
            nc = await nats.connect(
                servers=os.getenv("NATS_URL", "nats://localhost:4222"),
                reconnect_time_wait=2,
                max_reconnect_attempts=10,
            )
            logger.info(f"Connected to NATS server at {nc.connected_url.netloc}...")

            # Get JetStream context
            js = nc.jetstream(timeout=5)
            logger.info("Obtained JetStream context.")

            # Ensure JetStream streams exist
            try:
                # Attempt to get stream info to verify existence
                await js.stream_info("invoke_stream_bedrock_model")
                await js.stream_info("responses_stream_bedrock_model")
                logger.info("Verified required JetStream streams exist.")
            except Exception as e:
                warning = f"Could not verify JetStream streams (may not exist yet or permissions issue): {e}"  # noqa: E501
                logger.warning(warning)
                return False
                # Decide if this is critical - maybe proceed cautiously

            # Return success
            return True

        except Exception as e:
            retry_count += 1

            # Log the error with retry attempt
            error = f"Failed to connect to JetStream ({retry_count}/{max_retries}): {e}"
            logger.error(error)

            # Close the connection if it was partially established
            if nc and not nc.is_closed:
                await nc.close()
            if retry_count < max_retries and running:
                logger.info(f"Retrying in {retry_delay:.1f} seconds...")
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 1.5, 30)
            else:
                break  # Exit loop if max retries or not running

    logger.error("Failed to connect to NATS after maximum retries.")
    return False


async def publish_to_client(
    connection_id: str, request_id: str, subject: str, content: dict
):
    """Publishes a response chunk back to the client via NATS JetStream."""
    global js
    if not js:
        logger.error("Cannot publish response: JetStream context not available.")
        return False

    try:
        # Publish the message using Jetstream
        ack = await js.publish(
            subject=subject,
            payload=json.dumps({"request_id": request_id, "chunk": content}).encode(),
            timeout=2,
            stream="responses_stream_bedrock_model",
        )
        debug = f"RequestID: {request_id} Published response chunk (Stream: {ack.stream}, Seq: {ack.seq})"  # noqa: E501
        logger.debug(debug)
        return True
    except TimeoutError:
        logger.error(f"Timeout publishing response chunk to NATS subject {subject}")
        return False
    except ConnectionClosedError:
        warning = f"NATS connection closed while publishing response chunk to NAT subject {subject} for connection {connection_id}"  # noqa: E501
        logger.warning(warning)
    except SlowConsumerError:
        error = f"Slow consumer detected for {connection_id} on subject {subject}."
        logger.error(error)
    except Exception as e:
        logger.error(f"Failed to publish response chunk to NATS subject {subject}: {e}")
        return False


async def invoke_model_and_response_stream(msg):
    """
    Handles an incoming 'invoke' message, calls Bedrock, streams response chunks
    back via NATS JetStream, and ACKs the original message upon completion.
    """
    message_data = msg.data.decode()
    ack_policy = "ack"  # Default to ack, change on failure

    # Parse the incoming message
    try:
        data = json.loads(message_data)
    except json.JSONDecodeError:
        error = "Failed to decode JSON from invoke message. Terminating message."
        logger.error(error)
        await msg.term()
        return

    # Extract necessary info
    messages = data.get("messages")
    request_id = data.get("request_id")
    connection_id = data.get("connection_id")

    # Generate request_id if needed
    if not request_id:
        request_id = f"natsproc_{time.time_ns()}"
        warning = f"Missing 'request_id' in invoke message. Generated: {request_id}"
        logger.warning(warning)

    # Log the received invoke request
    logger.info(f"RequestID: {request_id} Received invoke request")

    if not connection_id:
        # Log error and terminate message if connection_id is missing
        error = "Missing 'connection_id' in invoke message. Cannot route response. Terminating message."  # noqa: E501
        logger.error(error)
        ack_policy = "term"
        return

    if not messages:
        # Log warning and terminate message if messages are missing
        warning = f"RequestID: {request_id} Received invoke request with no 'messages'. Terminating message."  # noqa: E501
        logger.warning(warning)
        ack_policy = "term"
        return

    try:
        info = f"RequestID: {request_id} Staring process streaming for connection {connection_id}"  # noqa: E501
        logger.info(info)

        # Prepare payload for Bedrock
        payload = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 128000,
            "temperature": 0.3,
            "top_k": 40,
            "top_p": 0.99,
            "messages": messages,
            "system": "",  # Add system prompt if needed
        }

        # Initialize variables for response processing
        start_time = time.time()
        time_taken = 0
        is_fist_token, bedrock_error, publish_error = False, False, False

        try:
            # Invoke model with streaming response
            stream = client.invoke_model_with_response_stream(
                modelId=model_id,
                body=json.dumps(payload),
                accept="application/json",
                contentType="application/json",
                trace="ENABLED",
            )

            # Process the streaming response
            for e in stream["body"]:
                chunk = json.loads(e["chunk"]["bytes"])

                # Log FTT
                if not is_fist_token:
                    time_taken = time.time() - start_time
                    is_fist_token = True

                # Stream response chunk back via NATS JetStream
                success = await publish_to_client(
                    connection_id=connection_id,
                    request_id=request_id,
                    subject=f"responses.{connection_id}",
                    content=chunk,
                )
                if not success:
                    # Stop processing this request and NAK the invoke message
                    critical = f"RequestID: {request_id} Failed critical publish to client {connection_id}. Stopping stream."  # noqa: E501
                    logger.critical(critical)
                    publish_error = True
                    ack_policy = "nack"
                    break

                # --- Console output (keep for debugging) ---
                if chunk["type"] == "message_start":
                    pass
                elif chunk["type"] == "content_block_start":
                    if chunk["content_block"]["type"] == "thinking":
                        # cprint(f"\n[{request_id}] THINKING:", "yellow")
                        pass
                    elif chunk["content_block"]["type"] == "text":
                        # cprint(f"\n\n[{request_id}] ANSWER:", "yellow")
                        pass
                elif chunk["type"] == "content_block_delta":
                    if chunk["delta"]["type"] == "thinking_delta":
                        # cprint(chunk["delta"]["thinking"], "cyan", end="", flush=True)
                        pass
                    elif chunk["delta"]["type"] == "text_delta":
                        # cprint(chunk["delta"]["text"], "green", end="", flush=True)
                        pass
                elif chunk["type"] == "message_delta":
                    pass
                elif chunk["type"] == "message_stop":
                    pass
                else:
                    pass

            final_time = time.time() - start_time
            info = f"RequestID: {request_id} Time to First Token: {time_taken:.2f}s | Total time: {final_time:.2f}s"  # noqa: E501
            logger.info(info)

        except Exception as bedrock_e:
            # Handle any errors that occur during Bedrock invocation or streaming
            error = f"RequestID: {request_id} Error during Bedrock invocation or streaming for connection {connection_id}: {bedrock_e}"  # noqa: E501
            logger.error(error, exc_info=True)
            bedrock_error = True
            ack_policy = "nack"

        # Final log for successful completion (if no errors occurred)
        if not bedrock_error and not publish_error:
            info = f"RequestID: {request_id} Successfully completed processing and streaming for connection {connection_id}"  # noqa: E501
            logger.info(info)
            ack_policy = "ack"

    # Handle Exceptions for the whole message processing
    except json.JSONDecodeError as json_e:
        error = f"RequestID: {request_id} Failed to decode incoming NATS message: {json_e}. Message Data: '{message_data}'. Terminating."  # noqa: E501
        logger.error(error, exc_info=True)
        ack_policy = "term"
    except Exception as e:
        error = f"RequestID: {request_id} Unexpected error processing message: {e}. Message Data: '{message_data}'. Nacking."  # noqa: E501
        logger.error(error, exc_info=True)
        ack_policy = "nack"
    finally:
        # Acknowledge the message based on the determined policy
        if ack_policy == "ack":
            # Positive acknowledgment, message won't be redelivered
            await msg.ack()
            logger.debug(f"RequestID: {request_id} Acknowledged (ack) message.")
        elif ack_policy == "nack":
            # Negative acknowledgment, allow redelivery based on stream config
            await msg.nak()
            logger.warning(f"RequestID: {request_id} Acknowledged (nak) message.")
        elif ack_policy == "term":
            # Terminal acknowledgment, message won't be redelivered
            await msg.term()
            logger.error(f"RequestID: {request_id} Acknowledged (term) message.")


async def start_subscription():
    """Subscribe to 'invoke' subject using JetStream durable consumer."""
    global js, sub, running
    if not js:
        logger.error("JetStream context not available, cannot subscribe.")
        return False

    try:
        # Subscribe to the 'invoke' subject
        # Use a durable consumer name so processing can resume after restarts
        # Create subscription options with consumer configuration
        subscribe_opts = ConsumerConfig(
            durable_name="agent-worker",
            ack_policy=AckPolicy.EXPLICIT,
            ack_wait=60 * 5,
            max_deliver=2,
        )

        # Subscribe with the options
        sub = await js.subscribe(
            subject="invoke",
            cb=invoke_model_and_response_stream,
            manual_ack=True,
            stream="invoke_stream_bedrock_model",
            config=subscribe_opts,
        )
        logger.info("Subscribed to 'invoke' subject with durable consumer 'agent'")
        return True
    except Exception as e:
        error = f"Failed to subscribe to JetStream subject 'invoke': {e}"
        logger.error(error, exc_info=True)
        return False


async def shutdown():
    """Clean shutdown of the server"""
    global running, nc, sub
    if not running:
        return

    logger.info("Shutting down agent...")
    running = False  # Signal loops to stop

    # Unsubscribe from JetStream subject
    # Drain will handle stopping message delivery, but explicit unsub is clearer
    if sub:
        try:
            await sub.unsubscribe()
            logger.info("Unsubscribed from 'invoke' subject.")
            sub = None  # Clear subscription object
        except Exception as e:
            logger.error(f"Error during unsubscribe: {e}")

    # Drain NATS connection
    if nc and nc.is_connected:
        try:
            info = "Draining NATS connection (allowing in-progress messages to be acked)..."  # noqa: E501
            logger.info(info)
            await nc.drain()
            logger.info("NATS connection drained and closed")
        except ConnectionClosedError:
            logger.warning("NATS connection already closed before drain completed.")
        except Exception as e:
            logger.error(f"Error draining NATS connection: {e}")
    elif nc:
        logger.warning("NATS connection was already closed or not established.")

    # Give tasks a final moment
    await asyncio.sleep(0.1)
    logger.info("Shutdown complete.")


async def main():
    """Main function to run the agent"""
    global running

    # Set up signal handlers
    loop = asyncio.get_running_loop()
    shutdown_task: Optional[asyncio.Task] = None

    def signal_handler():
        nonlocal shutdown_task
        if shutdown_task is None or shutdown_task.done():
            logger.info("Received shutdown signal.")
            # Use call_soon_threadsafe if handler might be called from different context
            # loop.call_soon_threadsafe(asyncio.create_task, shutdown())
            shutdown_task = asyncio.create_task(shutdown())  # Create task in the loop
        else:
            logger.info("Shutdown already in progress.")

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, signal_handler)
        except NotImplementedError:
            # Windows compatibility
            signal.signal(sig, lambda s, f: loop.call_soon_threadsafe(signal_handler))

    logger.info("Starting agent...")

    if await connect_to_nats():
        if await start_subscription():
            # Keep server alive until shutdown signal
            while running:
                try:
                    # Check connection health periodically
                    if not nc or nc.is_closed:
                        error = "NATS connection lost unexpectedly. Initiating shutdown process."  # noqa: E501
                        logger.error(error)
                        if shutdown_task is None or shutdown_task.done():
                            shutdown_task = asyncio.create_task(shutdown())
                    await asyncio.sleep(1)  # Keep main alive
                except asyncio.CancelledError:
                    logger.info("Main loop cancelled.")
                    if running:  # If cancelled but not via signal handler
                        if shutdown_task is None or shutdown_task.done():
                            shutdown_task = asyncio.create_task(shutdown())
                    break
        else:
            logger.error("Failed to start NATS subscription. Shutting down.")
            await shutdown()  # Attempt clean shutdown even if subscription failed
    else:
        logger.error("Failed to connect to NATS. Agent cannot start.")
        # No need to call shutdown here as resources weren't fully acquired

    # Wait for shutdown task if it was initiated
    if shutdown_task and not shutdown_task.done():
        logger.info("Waiting for shutdown task to complete...")
        try:
            await shutdown_task
        except Exception as e:
            logger.error(f"Error during final shutdown wait: {e}")


if __name__ == "__main__":
    asyncio.run(main())
