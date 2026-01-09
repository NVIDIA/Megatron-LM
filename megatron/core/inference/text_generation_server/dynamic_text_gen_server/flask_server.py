# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import logging
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
async def run_flask_server(coordinator_port: int, tokenizer, rank: int, flask_port: int):
    """Initializes and runs the async Flask server."""
    if not HAS_FLASK:
        raise RuntimeError(f"Flask not available")

    try:
        hostname = socket.gethostname()
    except Exception as e:
        logger.warning(f"Could not get hostname: {e}")
        hostname = "0.0.0.0"

    inference_client = InferenceClient(coordinator_port)
    await inference_client.start()
    logger.info(f"Rank {rank}: InferenceClient connected.")

    app = Flask(__name__)

    # Store client and tokenizer in app config for Blueprints to use
    app.config['client'] = inference_client
    app.config['tokenizer'] = tokenizer

    # Register all blueprints from the 'endpoints' package
    for endpoint in endpoints.__all__:
        app.register_blueprint(endpoint)

    @app.route('/')
    def health_check():
        return "Megatron Dynamic Inference Server is running."

    config = Config()
    config.bind = [f"0.0.0.0:{flask_port}"]

    # Force logging level to INFO to ensure that hostname is printed
    with temp_log_level(logging.INFO, logger):
        logger.info(f"Starting Flask server on http://{hostname}:{flask_port}")

    try:
        await serve(app, config)
    finally:
        await inference_client.stop()
        logger.info(f"Rank {rank}: Flask server and client shut down.")
