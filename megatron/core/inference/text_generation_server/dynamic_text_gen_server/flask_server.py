# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import asyncio
import logging
import multiprocessing as mp
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

# Global reference to manage the background Flask process
_FLASK_PROCESS = None


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
async def _run_flask_server(
    coordinator_addr: str,
    tokenizer,
    rank: int,
    flask_port: int,
    parsers: list[str] = None,
    verbose: bool = False,
):
    """
    Initializes and runs the async Flask server. Automatically starts and
    manages its own InferenceClient connected to the provided coordinator address.
    """
    if not HAS_FLASK:
        raise RuntimeError(f"Flask not available")

    # Create and start the client locally inside this process
    inference_client = InferenceClient(coordinator_addr)
    await inference_client.start()
    logger.info(f"Rank {rank}: InferenceClient connected.")

    try:
        try:
            hostname = socket.gethostname()
        except Exception as e:
            logger.warning(f"Could not get hostname: {e}")
            hostname = "0.0.0.0"

        app = Flask(__name__)

        # Store client and tokenizer in app config for Blueprints to use
        app.config['client'] = inference_client
        app.config['tokenizer'] = tokenizer
        app.config['parsers'] = parsers
        app.config['verbose'] = verbose

        # Register all blueprints from the 'endpoints' package
        for endpoint in endpoints.__all__:
            app.register_blueprint(endpoint)

        loop = asyncio.get_event_loop()

        config = Config()
        config.wsgi_max_body_size = 2**30  # 1 GB; needed for large prompts.
        config.keep_alive_timeout = 30.0  # Keep connection alive between long-running requests.
        config.backlog = 2**14  # Expect high load; ensure we do not drop connections.
        config.h2_max_concurrent_streams = (
            2**14
        )  # Allow many concurrent streams for HTTP/2 clients.
        config.bind = [f"0.0.0.0:{flask_port}"]

        with temp_log_level(logging.INFO, logger):
            logger.info(f"Starting Flask server on http://{hostname}:{flask_port}")
            logger.info(f"Using tokenizer: {type(tokenizer)}")
            logger.info(f"Using parsers: {parsers}")

        loop.set_default_executor(ThreadPoolExecutor(max_workers=8192))
        await serve(AsyncioWSGIMiddleware(app, max_body_size=config.wsgi_max_body_size), config)

    finally:
        # Gracefully shut down the client when the server stops
        inference_client.stop()
        logger.info(f"Rank {rank}: Flask server and client shut down.")


def _flask_process_worker(
    coordinator_addr: str,
    tokenizer,
    rank: int,
    flask_port: int,
    parsers: list[str] = None,
    verbose: bool = False,
):
    """Synchronous worker function that sets up a new event loop for the separate process."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(
            _run_flask_server(coordinator_addr, tokenizer, rank, flask_port, parsers, verbose)
        )
    except KeyboardInterrupt:
        logger.info(f"Rank {rank}: Flask server process interrupted.")
    finally:
        pending = asyncio.all_tasks(loop)
        for task in pending:
            task.cancel()
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        loop.close()


def start_flask_server(
    coordinator_addr: str,
    tokenizer,
    rank: int,
    flask_port: int,
    parsers: list[str] = None,
    verbose: bool = False,
):
    """Spawns and starts a separate process to run the Flask frontend transparently."""
    global _FLASK_PROCESS

    if _FLASK_PROCESS is not None and _FLASK_PROCESS.is_alive():
        logger.warning("Flask server process is already running.")
        return

    _FLASK_PROCESS = mp.Process(
        target=_flask_process_worker,
        args=(coordinator_addr, tokenizer, rank, flask_port, parsers, verbose),
        daemon=True,
    )
    _FLASK_PROCESS.start()
    logger.info(f"Started Flask frontend in separate process (PID: {_FLASK_PROCESS.pid})")


def stop_flask_server():
    """Terminates the background Flask server process with a timeout."""
    global _FLASK_PROCESS

    if _FLASK_PROCESS is not None and _FLASK_PROCESS.is_alive():
        logger.info(f"Terminating Flask frontend process (PID: {_FLASK_PROCESS.pid})")
        _FLASK_PROCESS.terminate()

        # Wait up to 3 seconds for it to shut down gracefully
        _FLASK_PROCESS.join(timeout=3)

        # If it's still alive, it's ignoring SIGTERM. Force kill it.
        if _FLASK_PROCESS.is_alive():
            logger.warning(
                f"Flask process (PID: {_FLASK_PROCESS.pid}) refused to terminate. Force killing."
            )
            _FLASK_PROCESS.kill()
            _FLASK_PROCESS.join()

        _FLASK_PROCESS = None
        logger.info("Flask frontend process terminated.")
