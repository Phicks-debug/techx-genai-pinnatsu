import os
import uuid
import json
import logging
import asyncio

import nats
from nats.js.api import StreamConfig, RetentionPolicy, StorageType
from nats.errors import TimeoutError, ConnectionClosedError, SlowConsumerError

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager


# NATS client and JetStream context
nc = None
js = None

# Store active connections
active_connections = []

logger = logging.getLogger("uvicorn")


async def setup_jetstream():
    """Creates or updates the necessary JetStream streams."""
    global js
    try:
        # JetStream Management API
        jsm = nc.jsm()

        # Define Stream for 'invoke' messages
        invoke_stream_cfg = StreamConfig(
            name="invoke_stream_bedrock_model",
            subjects=["invoke"],
            storage=StorageType.MEMORY,
            retention=RetentionPolicy.INTEREST,
        )

        # Log stream creation/update
        name = invoke_stream_cfg.name
        subjects = invoke_stream_cfg.subjects
        info = f"Creating/Updating JetStream stream: '{name}' with subjects: {subjects}"
        logger.info(info)

        # Create or update the stream
        await jsm.add_stream(config=invoke_stream_cfg)

        # Define Stream for 'responses' messages
        responses_stream_cfg = StreamConfig(
            name="responses_stream_bedrock_model",
            subjects=["responses.*"],
            storage=StorageType.MEMORY,
            retention=RetentionPolicy.INTEREST,
        )

        # Log stream creation/update
        name = responses_stream_cfg.name
        subjects = responses_stream_cfg.subjects
        info = f"Creating/Updating JetStream stream: '{name}' with subjects: {subjects}"
        logger.info(info)

        # Create or update the stream
        await jsm.add_stream(config=responses_stream_cfg)

        # Log successful setup of streams
        logger.info("JetStream streams configured successfully.")

    except Exception as e:
        logger.error(f"Failed to setup JetStream streams: {e}")
        # Depending on requirements, you might want to raise this exception
        # or prevent the application from starting fully.
        raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Connect to NATS server and setup JetStream
    global nc, js
    try:
        nc = await nats.connect(
            servers=os.getenv("NATS_URL", "nats://localhost:4222"),
            reconnect_time_wait=2,
            max_reconnect_attempts=10,
        )
        logger.info(f"Connected to NATS server at '{nc.connected_url.netloc}...'")

        # Get JetStream context
        js = nc.jetstream(timeout=5)
        logger.info("Obtained JetStream context.")

        # Ensure JetStream streams exist
        await setup_jetstream()

    except Exception as e:
        logger.error(f"Failed during NATS/JetStream startup: {e}")
        # Handle connection/setup failure appropriately
        if nc and nc.is_connected:
            await nc.drain()
            logger.info("Disconnected from NATS server due to startup failure.")
        else:
            logger.info("Failed to connect to NATS server...")
        raise

    yield

    # Shutdown: Disconnect from NATS server
    if nc and nc.is_connected:
        logger.info("Draining and disconnecting from NATS server...")
        await nc.drain()
        logger.info("Disconnected from NATS server")
    elif nc:
        logger.warning("NATS connection was already closed or not established.")


# Initialize FastAPI app with lifespan management
app = FastAPI(lifespan=lifespan)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global active_connections, js

    # Accept the websocket connection
    await websocket.accept()

    # Generate a unique connection_id for this websocket
    connection_id = str(id(websocket))

    # Store connection with its ID
    active_connections.append({"id": connection_id, "websocket": websocket})
    logger.info(f"New websocket connection established: {connection_id}")

    # Define the response subject for this connection
    sub, nats_task = None, None
    response_subject = f"responses.{connection_id}"

    try:
        if not js:
            # Should not happen if lifespan completed successfully, but good check
            raise RuntimeError("JetStream context not available.")

        # Subscribe to the response subject using JetStream
        # Create an ephemeral, push-based consumer tied to this specific subject
        sub = await js.subscribe(
            subject=response_subject,
            durable=None,  # Ephemeral consumer (deleted when subscription ends)
            deliver_policy="new",  # Only deliver messages created after starts
            stream="responses_stream_bedrock_model",
        )

        # Log subscription creation
        logger.info(f"JetStream subscription created for subject: {response_subject}")

        # Start a task to process incoming NATS messages directed to this connection
        nats_task = asyncio.create_task(process_nats_messages(sub, websocket))

        # Ensure the task is running
        if nats_task.done():
            w = f"NATS processing task {connection_id} completed unexpectedly at start"
            logger.warning(w)
            if nats_task.exception():
                raise nats_task.exception()  # Propagate error

        while True:
            # Receive message from WebSocket client
            data = await websocket.receive_text()
            message = json.loads(data)

            # Add connection_id to the message before publishing to NATS
            message["connection_id"] = connection_id

            # Generate a request_id if not provided (using UUID5 for consistency)
            req_id = uuid.uuid5(namespace=uuid.UUID(int=0), name="msg").hex
            message["request_id"] = f"req_{req_id}"

            # Publish message to NATS using JetStream
            try:
                ack = await js.publish(
                    subject="invoke",
                    payload=json.dumps(message).encode(),
                    timeout=2,
                    stream="invoke_stream_bedrock_model",
                )

                # Log successful publish
                info = f"Published to JetStream subject 'invoke', stream={ack.stream}, seq={ack.seq}"  # noqa: E501
                logger.debug(info)
            except Exception as e:
                logger.error(f"Failed to publish message to JetStream 'invoke': {e}")
                # Decide how to handle publish failure (e.g., notify client, retry?)

    except WebSocketDisconnect:
        logger.warning(f"WebSocket client disconnected: {connection_id}")
    except Exception as e:
        logger.error(f"Error in WebSocket connection {connection_id}: {e}")
    finally:
        # Remove the connection from active_connections
        active_connections = [
            connection
            for connection in active_connections
            if connection["id"] != connection_id
        ]

        # Clean up subscription and task
        if sub:
            try:
                # Unsubscribe efficiently cleans up ephemeral consumers
                await sub.unsubscribe()
                logger.info(f"Unsubscribed JetStream consumer for {response_subject}")
            except Exception as e:
                error = f"Error unsubscribing JetStream consumer for {response_subject}: {e}"  # noqa: E501
                logger.error(error)
        if nats_task and not nats_task.done():
            nats_task.cancel()
            try:
                await nats_task  # Wait for cancellation to complete
            except asyncio.CancelledError:
                logger.info(f"NATS processing task for {connection_id} cancelled.")
            except Exception as e:
                error = f"Error during NATS task cancellation for {connection_id}: {e}"
                logger.error(error)


async def process_nats_messages(subscription, websocket):
    """Process messages from NATS JetStream subscription and send to WebSocket"""
    conn_id = websocket.headers.get("sec-websocket-key", "unknown")
    try:
        # Handle message processing and WebSocket communication with acknowledgments
        async for msg in subscription.messages:
            await handle_nats_message(msg, websocket)

        # Log successful message handling
        debug = f"Receive messages from NATS JetStream for {conn_id}."
        logger.debug(debug)

    except TimeoutError:
        warning = f"NATS subscription messages iterator timed out for {conn_id}. This might be normal if inactive."  # noqa: E501
        logger.warning(warning)
    except ConnectionClosedError:
        warning = f"NATS connection closed while processing messages for {conn_id}."
        logger.warning(warning)
    except SlowConsumerError:
        error = f"Slow consumer detected for {conn_id} on subject {subscription.subject}."  # noqa: E501
        logger.error(error)
    except Exception as e:
        error = f"Unexpected error processing NATS messages for {conn_id}: {e}"
        logger.error(error)


async def handle_nats_message(msg, websocket):
    """Handle messages from NATS, send to WebSocket client, and ACKNOWLEDGE."""
    conn_id = websocket.headers.get("sec-websocket-key", "unknown")
    try:
        # Send message to the specific WebSocket client
        await websocket.send_text(msg.data.decode())

        # Acknowledge the message AFTER successful processing
        await msg.ack()

        # Log successful message handling
        debug = f"Message sent to WebSocket {conn_id} and acknowledged (ack) to JetStream."  # noqa: E501
        logger.debug(debug)
    except Exception as e:
        logger.error(f"Error sending message to WebSocket {conn_id} or acking: {e}")
        # Decide on ack policy on failure.
        # - msg.nak(): Negative acknowledgment, might trigger redelivery.
        # - msg.term(): Terminal acknowledgment, message won't be redelivered.
        # - Do nothing (message will likely be redelivered after ack_wait timeout)
        # Choosing to do nothing here, letting it potentially redeliver.
        # Consider adding msg.nak() if redelivery is desired sooner.
        # try:
        #     await msg.nak(delay=5) # Example: nack with 5s redelivery delay
        # except Exception as nak_err:
        #     logger.error(f"Failed to NACK message for {conn_id}: {nak_err}")


async def broadcast(message):
    """Broadcast message to all connected WebSocket clients"""
    global active_connections
    disconnected = []
    msg_str = json.dumps(message)
    for connection in active_connections:
        try:
            await connection["websocket"].send_text(msg_str)
        except Exception:
            disconnected.append(connection)

    # Remove disconnected clients
    for conn in disconnected:
        if conn in active_connections:
            active_connections.remove(conn)
            info = f"Removed disconnected client during broadcast: {conn.get('id', 'unknown')}"  # noqa: E501
            logger.info(info)
