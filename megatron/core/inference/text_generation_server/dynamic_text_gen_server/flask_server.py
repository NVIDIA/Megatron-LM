# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import asyncio
import logging
import multiprocessing
import socket
from contextlib import contextmanager

try:
    from flask import Flask
    from hypercorn.asyncio import serve
    from hypercorn.config import Config

    HAS_FLASK = True
except ImportError as e:
    HAS_FLASK = False

import megatron.core.inference.text_generation_server.dynamic_text_gen_server.endpoints as endpoints
from megatron.core.inference.inference_client import InferenceClient
from megatron.core.tokenizers.megatron_tokenizer import MegatronTokenizer
from megatron.core.utils import trace_async_exceptions

logger = logging.getLogger(__name__)


@contextmanager
def temp_log_level(level, logger=None):
    """Enables temporarily overriding the logging level."""
    logger = logger or logging.getLogger()
    old_level = logger.level
    logger.setLevel(level)
    try:
        yield
    finally:
        logger.setLevel(old_level)


async def _run_flask_logic(coordinator_addr, tokenizer, rank, flask_port):
    """Internal function to run the Flask server logic."""
    if not HAS_FLASK:
        raise RuntimeError(f"Flask not available")

    try:
        hostname = socket.gethostname()
    except Exception as e:
        logger.warning(f"Could not get hostname: {e}")
        hostname = "0.0.0.0"

    inference_client = InferenceClient(coordinator_addr)
    await inference_client.start()
    logger.info(f"Rank {rank}: InferenceClient connected in background process.")

    app = Flask(__name__)

    # Store client and tokenizer in app config for Blueprints
    app.config['client'] = inference_client
    app.config['tokenizer'] = tokenizer

    for endpoint in endpoints.__all__:
        app.register_blueprint(endpoint)

    config = Config()
    config.wsgi_max_body_size = 1024 * 1024 * 1024  # 1 GiB
    config.bind = [f"0.0.0.0:{flask_port}"]

    with temp_log_level(logging.INFO, logger):
        logger.info(f"Starting Flask server on http://{hostname}:{flask_port}")

    try:
        await serve(app, config)
    finally:
        await inference_client.stop()
        logger.info(f"Rank {rank}: Flask server shut down.")


def _worker_entrypoint(
    coordinator_addr: str, tokenizer: MegatronTokenizer, rank: int, flask_port: int
):
    """Synchronous wrapper to bootstrap the asyncio loop in the new process."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(_run_flask_logic(coordinator_addr, tokenizer, rank, flask_port))
    except KeyboardInterrupt:
        pass
    finally:
        loop.close()


@trace_async_exceptions
async def run_flask_server(
    coordinator_addr: str, tokenizer: MegatronTokenizer, rank: int, flask_port: int
):
    """Initializes and runs the async Flask server in a separate process."""

    ctx = multiprocessing.get_context("spawn")

    p = ctx.Process(
        target=_worker_entrypoint, args=(coordinator_addr, tokenizer, rank, flask_port), daemon=True
    )

    p.start()
    logger.info(f"Rank {rank}: Launched Flask server in process PID {p.pid}")

    try:
        while p.is_alive():
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        logger.info(f"Rank {rank}: Stopping Flask background process...")
        p.terminate()
        p.join()
