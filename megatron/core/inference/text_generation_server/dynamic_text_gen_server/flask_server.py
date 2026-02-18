# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import asyncio
import logging
import socket
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

try:
    from flask import Flask
    from hypercorn.asyncio import serve
    from hypercorn.config import Config
    from hypercorn.middleware import AsyncioWSGIMiddleware

    HAS_FLASK = True
except ImportError as e:
    HAS_FLASK = False

import megatron.core.inference.text_generation_server.dynamic_text_gen_server.endpoints as endpoints
from megatron.core.inference.inference_client import InferenceClient
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


@trace_async_exceptions
async def run_flask_server_on_client(
    client: InferenceClient,
    tokenizer,
    flask_port: int,
    parsers: list[str] = None,
    verbose: bool = False,
):
    """Initializes and runs the async Flask server using the provided InferenceClient."""
    if not HAS_FLASK:
        raise RuntimeError(f"Flask not available")

    try:
        hostname = socket.gethostname()
    except Exception as e:
        logger.warning(f"Could not get hostname: {e}")
        hostname = "0.0.0.0"

    app = Flask(__name__)

    # Store client and tokenizer in app config for Blueprints to use
    app.config['client'] = client
    app.config['tokenizer'] = tokenizer
    app.config['parsers'] = parsers
    app.config['verbose'] = verbose

    # Register all blueprints from the 'endpoints' package
    for endpoint in endpoints.__all__:
        app.register_blueprint(endpoint)

    @app.route('/')
    def health_check():
        return "Megatron Dynamic Inference Server is running."

    loop = asyncio.get_event_loop()

    config = Config()
    config.keep_alive_timeout = 30.0
    config.wsgi_max_body_size = 2**30  # 1 GB
    config.bind = [f"0.0.0.0:{flask_port}"]
    config.backlog = 8192
    config.keep_alive_timeout = 30.0

    # Force logging level to INFO to ensure that hostname is printed
    with temp_log_level(logging.INFO, logger):
        logger.info(f"Starting Flask server on http://{hostname}:{flask_port}")
        logger.info(f"Using tokenizer: {type(tokenizer)}")
        logger.info(f"Using parsers: {parsers}")

    loop.set_default_executor(ThreadPoolExecutor(max_workers=8192))
    await serve(AsyncioWSGIMiddleware(app), config)


@trace_async_exceptions
async def run_flask_server(
    coordinator_addr: str,
    tokenizer,
    rank: int,
    flask_port: int,
    parsers: list[str] = None,
    verbose: bool = False,
):
    """Initializes and runs the async Flask server
    starting an InferenceClient with the provided coordinator address."""
    inference_client = InferenceClient(coordinator_addr)
    await inference_client.start()
    logger.info(f"Rank {rank}: InferenceClient connected.")
    try:
        await run_flask_server_on_client(inference_client, tokenizer, flask_port, parsers, verbose)
    finally:
        await inference_client.stop()
        logger.info(f"Rank {rank}: Flask server and client shut down.")
