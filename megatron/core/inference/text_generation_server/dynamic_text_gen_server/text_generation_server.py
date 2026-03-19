# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import asyncio
import logging
import multiprocessing as mp
import socket
from contextlib import contextmanager
from typing import List, Optional

try:
    from hypercorn.asyncio import serve
    from hypercorn.config import Config
    from quart import Quart

    HAS_BACKEND = True
except ImportError as e:
    HAS_BACKEND = False

import megatron.core.inference.text_generation_server.dynamic_text_gen_server.endpoints as endpoints
from megatron.core.inference.inference_client import InferenceClient
from megatron.core.utils import trace_async_exceptions

logger = logging.getLogger(__name__)

# Global reference to manage the background server processes
_SERVER_PROCESSES: List[mp.Process] = []
_SHARED_SOCKET = None


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
async def _run_text_gen_server(
    coordinator_addr: str,
    tokenizer,
    rank: int,
    server_port: int,
    parsers: Optional[List[str]] = None,
    verbose: bool = False,
    fd: Optional[int] = None,
):
    """
    Initializes and runs the async web server. Automatically starts and
    manages its own InferenceClient connected to the provided coordinator address.
    """
    if not HAS_BACKEND:
        raise RuntimeError(f"Web backend framework (Quart) not available")

    # Create and start the client locally inside this process
    inference_client = InferenceClient(coordinator_addr, deserialize=True)
    inference_client.start()
    logger.info(f"Rank {rank}: InferenceClient connected.")

    try:
        try:
            hostname = socket.gethostname()
        except Exception as e:
            logger.warning(f"Could not get hostname: {e}")
            hostname = "0.0.0.0"

        app = Quart(__name__)

        # Quart native way to handle max body size (1 GB; needed for large prompts)
        app.config['MAX_CONTENT_LENGTH'] = 2**30

        # Store client and tokenizer in app config for Blueprints to use
        app.config['client'] = inference_client
        app.config['tokenizer'] = tokenizer
        app.config['parsers'] = parsers
        app.config['verbose'] = verbose

        # Register all blueprints from the 'endpoints' package
        for endpoint in endpoints.__all__:
            app.register_blueprint(endpoint)

        config = Config()
        config.keep_alive_timeout = 30.0  # Keep connection alive between long-running requests.
        config.backlog = 2**14  # Expect high load; ensure we do not drop connections.
        config.h2_max_concurrent_streams = (
            2**14
        )  # Allow many concurrent streams for HTTP/2 clients.

        if fd is not None:
            config.bind = [f"fd://{fd}"]
        else:
            config.bind = [f"0.0.0.0:{server_port}"]

        with temp_log_level(logging.INFO, logger):
            logger.info(f"Starting text generation server on http://{hostname}:{server_port}")
            logger.info(f"Using tokenizer: {type(tokenizer)}")
            logger.info(f"Using parsers: {parsers}")

        # Quart is natively ASGI, so we can serve the app directly
        await serve(app, config)

    finally:
        # Gracefully shut down the client when the server stops
        inference_client.stop()
        logger.info(f"Rank {rank}: Web server and client shut down.")


def _server_process_worker(
    coordinator_addr: str,
    tokenizer,
    rank: int,
    server_port: int,
    parsers: Optional[List[str]] = None,
    verbose: bool = False,
    fd: Optional[int] = None,
):
    """Synchronous worker function that sets up a new event loop for the separate process."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(
            _run_text_gen_server(
                coordinator_addr, tokenizer, rank, server_port, parsers, verbose, fd
            )
        )
    except KeyboardInterrupt:
        logger.info(f"Rank {rank}: text gen server process interrupted.")
    finally:
        pending = asyncio.all_tasks(loop)
        for task in pending:
            task.cancel()
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        loop.close()


def start_text_gen_server(
    coordinator_addr: str,
    tokenizer,
    rank: int,
    server_port: int,
    parsers: Optional[List[str]] = None,
    verbose: bool = False,
    num_replicas: int = 4,
):
    """Start the text generation server."""
    global _SERVER_PROCESSES
    global _SHARED_SOCKET

    if _SERVER_PROCESSES:
        logger.warning("Text gen server processes are already running.")
        return

    _SHARED_SOCKET = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    _SHARED_SOCKET.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    if hasattr(socket, 'SO_REUSEPORT'):
        try:
            _SHARED_SOCKET.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        except OSError:
            pass

    _SHARED_SOCKET.bind(("0.0.0.0", server_port))
    _SHARED_SOCKET.setblocking(False)

    _SHARED_SOCKET.set_inheritable(True)
    fd = _SHARED_SOCKET.fileno()

    for i in range(num_replicas):
        p = mp.Process(
            target=_server_process_worker,
            args=(coordinator_addr, tokenizer, rank, server_port, parsers, verbose, fd),
            daemon=True,
        )
        p.start()
        _SERVER_PROCESSES.append(p)
        logger.info(f"Started text gen frontend replica {i+1}/{num_replicas} (PID: {p.pid})")


def stop_text_gen_server():
    """Stop the text generation server."""
    global _SERVER_PROCESSES
    global _SHARED_SOCKET

    if not _SERVER_PROCESSES:
        return

    logger.info(f"Terminating {len(_SERVER_PROCESSES)} Text Gen frontend processes...")

    for p in _SERVER_PROCESSES:
        if p.is_alive():
            p.terminate()

    for p in _SERVER_PROCESSES:
        p.join(timeout=3)
        if p.is_alive():
            p.kill()
            p.join()

    # Clean up the master socket
    if _SHARED_SOCKET is not None:
        _SHARED_SOCKET.close()
        _SHARED_SOCKET = None

    _SERVER_PROCESSES = []
    logger.info("All text gen frontend processes terminated.")
