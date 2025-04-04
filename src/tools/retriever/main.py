import os
import time
import json
import boto3
import signal
import asyncio
import nats

from nats.js.api import ConsumerConfig, AckPolicy
from nats.errors import ConnectionClosedError

from qdrant_client import QdrantClient

from typing import Optional, List
from src._lib.logger import CustomLogger

logger = CustomLogger()

# NATS client and JetStream context
nc, js, sub = None, None, None

# Flag to control server shutdown
running = True

# Initialize AWS clients
bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    region_name=os.getenv("EMBED_MODEL_REGION"),
)
rerank_client = boto3.client(
    service_name="bedrock-runtime",
    region_name=os.getenv("RERANK_MODEL_REGION"),
)

# Initialize Qdrant client
qdrant_client = QdrantClient(
    host=os.getenv("QDRANT_HOST", "172.30.34.4"),
    port=os.getenv("QDRANT_PORT", "6334"),
    api_key=os.getenv("QDRANT_API_KEY", "IPsIIBLEeL0Z1pNQDatUwyPM93FXkByg"),
)

# Model IDs
EMBEDDING_MODEL_ID = os.getenv("EMBEDDING_MODEL_ID", "cohere.embed-multilingual-v3")
RERANKING_MODEL_ID = os.getenv("RERANKING_MODEL_ID", "cohere.rerank-v3-5:0")

# Retrieval settings
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "vpb-schema-collection")
RETRIEVE_TOP_K = int(os.getenv("RETRIEVE_TOP_K", "5"))
RETRIEVE_THRESHOLD = float(os.getenv("RETRIEVE_THRESHOLD", "0.7"))
ENABLE_RERANKING = os.getenv("ENABLE_RERANKING", "true").lower() == "true"
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "3"))
EMBEDDING_TYPE = "float"

# NATS settings
NATS_URL = os.getenv("NATS_URL", "nats://localhost:4222")


async def connect_to_nats():
    """Connect to NATS server and get JetStream context"""
    global nc, js
    retry_count = 0
    max_retries = 5
    retry_delay = 2

    while retry_count < max_retries and running:
        try:
            nc = await nats.connect(
                servers=NATS_URL,
                reconnect_time_wait=2,
                max_reconnect_attempts=10,
            )
            logger.info(f"Connected to NATS server at {nc.connected_url.netloc}...")

            # Get JetStream context
            js = nc.jetstream(timeout=5)
            logger.info("Obtained JetStream context.")

            # Setup needed streams
            jsm = nc.jsm()
            try:
                # Create retrieve stream if not exists
                await jsm.add_stream(
                    name="retrieve_stream",
                    subjects=["retrieve"],
                    storage="memory",
                    retention="workqueue",
                )
                logger.info("Created retrieve_stream")
            except Exception as e:
                if "stream name already in use" not in str(e):
                    logger.error(f"Error creating retrieve stream: {e}")
                    return False

            try:
                await jsm.add_stream(
                    name="responses_stream_retrieve",
                    subjects=["retrieve.responses.*"],
                    storage="memory",
                    retention="workqueue",
                )
                logger.info("Created responses_stream_retrieve")
            except Exception as e:
                if "stream name already in use" not in str(e):
                    logger.error(f"Error creating responses stream: {e}")
                    return False

            return True

        except Exception as e:
            retry_count += 1
            error = f"Failed to connect to NATS ({retry_count}/{max_retries}): {e}"
            logger.error(error)

            if nc and not nc.is_closed:
                await nc.close()

            if retry_count < max_retries and running:
                logger.info(f"Retrying in {retry_delay:.1f} seconds...")
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 1.5, 30)
            else:
                break

    logger.error("Failed to connect to NATS after maximum retries.")
    return False


async def publish_to_client(
    connection_id: str, request_id: str, subject: str, content: dict
):
    global js
    if not js:
        logger.error(
            f"RequestID: {request_id} Cannot publish response: JetStream not available."
        )
        return False
    try:
        payload = {"request_id": request_id, "response": content}
        ack = await js.publish(
            subject=f"retrieve.responses.{connection_id}",
            payload=json.dumps(payload).encode(),
            timeout=2,
            stream="responses_stream_retrieve",
        )
        logger.debug(
            f"RequestID: {request_id} Published response (Stream: {ack.stream}"
            f", Seq: {ack.seq})"
        )
        return True
    except Exception as e:
        logger.error(f"RequestID: {request_id} Failed to publish response: {str(e)}")
        return False


async def generate_embedding(text: str):
    """Generate embeddings for text using Bedrock."""
    try:
        response = bedrock_client.invoke_model(
            modelId=EMBEDDING_MODEL_ID,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(
                {
                    "texts": [text],
                    "input_type": "search_query",
                    "embedding_types": [EMBEDDING_TYPE],
                }
            ),
        )

        embedding_body = json.loads(response["body"].read().decode())
        embedding = embedding_body["embeddings"][EMBEDDING_TYPE][0]

        if not embedding or not isinstance(embedding, list):
            raise ValueError("Failed to extract embedding from Bedrock response")

        logger.info(
            f"Embedding generated successfully (first few dims): {embedding[:5]}"
        )
        return embedding

    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        return None


async def search_qdrant(embedding: List[float], query_text: str):
    """Search Qdrant using the embedding."""
    try:
        if not qdrant_client:
            raise RuntimeError("Qdrant client not initialized")

        logger.info(f"Performing search in Qdrant collection: {COLLECTION_NAME}")
        search_result = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=embedding,
            limit=RETRIEVE_TOP_K,
            score_threshold=RETRIEVE_THRESHOLD,
            with_payload=True,
        )

        logger.info(f"Found {len(search_result)} results")

        # If reranking is enabled and we have results
        if search_result and ENABLE_RERANKING:
            reranked_results = await rerank_results(search_result, query_text)
            return reranked_results

        return search_result

    except Exception as e:
        logger.error(f"Error searching Qdrant: {e}")
        return []


async def rerank_results(search_results, query_text: str):
    """Rerank search results using Bedrock reranking model."""
    try:
        # Extract texts from search results
        documents = [result.payload.get("text", "") for result in search_results]

        # Call reranking model
        response = rerank_client.invoke_model(
            modelId=RERANKING_MODEL_ID,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(
                {
                    "query": query_text,
                    "documents": documents,
                    "top_n": min(RERANK_TOP_N, len(documents)),
                    "api_version": 2,
                }
            ),
        )

        rerank_body = json.loads(response["body"].read().decode())
        logger.info(f"Reranked documents: {rerank_body['results']}")

        # Map the index from the reranked documents to the original search result
        reranked_documents = [
            search_results[result["index"]] for result in rerank_body["results"]
        ]

        return reranked_documents

    except Exception as e:
        logger.error(f"Error reranking results: {e}")
        # On error, return original results
        return search_results


def format_results(documents):
    """Format results for inclusion in prompt."""
    formatted_schemas = []

    for doc in documents:
        payload = doc.payload
        formatted_doc = (
            "<schema>\n"
            f"<content>{payload.get('text', '')}</content>\n"
            f"<title>{payload.get('title', '')}</title>\n"
            "</schema>"
        )
        formatted_schemas.append(formatted_doc)

    return "<schemas>\n" + "\n\n".join(formatted_schemas) + "\n</schemas>"


async def process_retrieve_request(msg):
    message_data = msg.data.decode()
    logger.debug(f"Received retrieve request: {message_data}")
    ack_policy = "ack"  # Default to ack, change on failure

    try:
        data = json.loads(message_data)
        query_text = data.get("query")
        request_id = data.get("request_id")
        connection_id = data.get("connection_id")

        # Validate required fields
        if not all([query_text, request_id, connection_id]):
            missing_fields = [
                f for f in ["query", "request_id", "connection_id"] if not data.get(f)
            ]
            raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")

        logger.info(
            f"RequestID: {request_id} Processing retrieve request for query: "
            f"{query_text[:50]}..."
        )

        # Start timing
        start_time = time.time()

        # Generate embedding
        embedding = await generate_embedding(query_text)
        if not embedding:
            raise RuntimeError(f"RequestID: {request_id} Failed to generate embedding.")

        logger.debug(
            f"RequestID: {request_id} Generated embedding for query: {query_text[:30]}"
        )

        # Search Qdrant
        search_results = await search_qdrant(embedding, query_text)
        logger.debug(
            f"RequestID: {request_id} Qdrant results: {len(search_results)} documents"
        )

        # Format schema
        formatted_schema = format_results(search_results)

        # Create formatted prompt with context
        formatted_prompt = (
            f"Here is the retrieved schema from the knowledge base:\n\n"
            f"{formatted_schema}\n\n"
        )

        total_time = time.time() - start_time
        logger.info(f"RequestID: {request_id} Retrieval completed in {total_time:.2f}s")

        # Prepare response content
        content = {
            "formatted_prompt": formatted_prompt,
            "query": query_text,
            "processing_time": total_time,
        }

        logger.debug(f"RequestID: {request_id} Publishing response: {content}")

        # Publish response with formatted documents
        success = await publish_to_client(
            connection_id=connection_id,
            request_id=request_id,
            subject=f"retrieve.responses.{connection_id}",
            content=content,
        )

        if not success:
            raise RuntimeError(
                f"RequestID: {request_id} Failed to publish response to client."
            )

    except json.JSONDecodeError as e:
        logger.error(
            f"Failed to decode JSON from message: {message_data}. Error: {str(e)}"
        )
        ack_policy = "term"
    except ValueError as e:
        logger.error(f"Invalid request data: {str(e)}")
        ack_policy = "term"
    except RuntimeError as e:
        logger.error(str(e))
        ack_policy = "nack"
    except Exception as e:
        logger.error(f"Unexpected error during retrieval: {str(e)}", exc_info=True)
        ack_policy = "nack"
    finally:
        # Acknowledge the message based on the determined policy
        if ack_policy == "ack":
            await msg.ack()
            logger.debug(f"RequestID: {request_id} Acknowledged (ack) message.")
        elif ack_policy == "nack":
            await msg.nak()
            logger.warning(f"RequestID: {request_id} Acknowledged (nak) message.")
        elif ack_policy == "term":
            await msg.term()
            logger.error(f"RequestID: {request_id} Acknowledged (term) message.")


async def start_subscription():
    """Subscribe to 'retrieve' subject using JetStream durable consumer."""
    global js, sub, running
    if not js:
        logger.error("JetStream context not available, cannot subscribe.")
        return False

    try:
        # Create consumer configuration
        subscribe_opts = ConsumerConfig(
            durable_name="retrieve-worker",
            ack_policy=AckPolicy.EXPLICIT,
            ack_wait=60 * 5,
            max_deliver=2,
        )

        # Subscribe with the options and specify stream
        sub = await js.subscribe(
            subject="retrieve",
            stream="retrieve_stream",
            cb=process_retrieve_request,
            manual_ack=True,
            config=subscribe_opts,
        )
        logger.info(
            "Subscribed to 'retrieve' subject with durable consumer 'retrieve-worker'"
        )
        return True

    except Exception as e:
        error = f"Failed to subscribe to JetStream subject 'retrieve': {e}"
        logger.error(error, exc_info=True)
        return False


async def shutdown():
    """Clean shutdown of the server"""
    global running, nc, sub
    if not running:
        return

    logger.info("Shutting down retrieve service...")
    running = False

    # Unsubscribe from JetStream subject
    if sub:
        try:
            await sub.unsubscribe()
            logger.info("Unsubscribed from 'retrieve' subject.")
            sub = None
        except Exception as e:
            logger.error(f"Error during unsubscribe: {e}")

    # Drain NATS connection
    if nc and nc.is_connected:
        try:
            info = "Draining NATS connection (allowing in-progress to be acked)..."
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
    """Main function to run the retrieve service"""
    global running

    # Set up signal handlers
    loop = asyncio.get_running_loop()
    shutdown_task: Optional[asyncio.Task] = None

    def signal_handler():
        nonlocal shutdown_task
        if shutdown_task is None or shutdown_task.done():
            logger.info("Received shutdown signal.")
            shutdown_task = asyncio.create_task(shutdown())
        else:
            logger.info("Shutdown already in progress.")

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, signal_handler)
        except NotImplementedError:
            signal.signal(sig, lambda s, f: loop.call_soon_threadsafe(signal_handler))

    logger.info("Starting retrieve service...")

    if await connect_to_nats():
        if await start_subscription():
            # Keep server alive until shutdown signal
            while running:
                try:
                    # Check connection health periodically
                    if not nc or nc.is_closed:
                        error = "NATS connection lost unexpectedly. Initiating shutdown"
                        logger.error(error)
                        if shutdown_task is None or shutdown_task.done():
                            shutdown_task = asyncio.create_task(shutdown())
                    await asyncio.sleep(1)
                except asyncio.CancelledError:
                    logger.info("Main loop cancelled.")
                    if running:
                        if shutdown_task is None or shutdown_task.done():
                            shutdown_task = asyncio.create_task(shutdown())
                    break
        else:
            logger.error("Failed to start NATS subscription. Shutting down.")
            await shutdown()
    else:
        logger.error("Failed to connect to NATS. Retrieve service cannot start.")

    if shutdown_task and not shutdown_task.done():
        logger.info("Waiting for shutdown task to complete...")
        try:
            await shutdown_task
        except Exception as e:
            logger.error(f"Error during final shutdown wait: {e}")


if __name__ == "__main__":
    asyncio.run(main())
