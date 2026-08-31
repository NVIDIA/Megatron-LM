# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import asyncio
import copy
import logging
import multiprocessing as mp
import socket
from concurrent.futures import ThreadPoolExecutor
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
from megatron.core.inference.config import MultimodalPromptConfig
from megatron.core.inference.inference_client import InferenceClient
from megatron.core.utils import trace_async_exceptions

logger = logging.getLogger(__name__)

# Global reference to manage the background server processes
_SERVER_PROCESSES: List[mp.Process] = []


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
    hostname: Optional[str] = None,
    chat_template: Optional[str] = None,
    multimodal_prompt_config: Optional[MultimodalPromptConfig] = None,
):
    """
    Initializes and runs the async web server. Automatically starts and
    manages its own InferenceClient connected to the provided coordinator address.
    """
    if not HAS_BACKEND:
        raise RuntimeError(f"Web backend framework (Quart) not available")

    # Create and start the client locally inside this process
    inference_client = InferenceClient(coordinator_addr, deserialize=False)
    inference_client.start()
    logger.info(f"Rank {rank}: InferenceClient connected.")

    try:
        # Bind what the caller asked for -- None means every interface, which is
        # not the single address gethostname() resolves to. The resolved name is
        # for the log line only.
        bind_host = hostname
        if hostname is None:
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
        app.config['chat_template'] = chat_template
        app.config['multimodal_prompt_config'] = (
            multimodal_prompt_config or MultimodalPromptConfig()
        )

        # Applying the chat template is synchronous and O(prompt); on the event loop it
        # stalls every other request this replica owns, including delivery of responses
        # that already finished. One worker is enough - the point is the yield, not
        # throughput. The copy is required: HF tokenizers are not thread-safe.
        app.config['tokenize_executor'] = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="tokenize"
        )
        app.config['tokenize_tokenizer'] = copy.deepcopy(tokenizer)

        # Register all blueprints from the 'endpoints' package
        for endpoint in endpoints.__all__:
            app.register_blueprint(endpoint)

        config = Config()
        config.keep_alive_timeout = 30.0  # Keep connection alive between long-running requests.
        config.backlog = 2**14  # Expect high load; ensure we do not drop connections.
        config.h2_max_concurrent_streams = (
            2**14
        )  # Allow many concurrent streams for HTTP/2 clients.

        # Held for this worker's lifetime; closing it would drop the listener.
        own_socket = _bind_reuseport_socket(server_port, bind_host)
        config.bind = [f"fd://{own_socket.fileno()}"]

        with temp_log_level(logging.INFO, logger):
            logger.info(f"Starting text generation server on http://{hostname}:{server_port}")
            logger.info(f"Using tokenizer: {type(tokenizer)}")
            logger.info(f"Using parsers: {parsers}")

        try:
            # Quart is natively ASGI, so we can serve the app directly
            await serve(app, config)
        finally:
            own_socket.close()

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
    hostname: Optional[str] = None,
    chat_template: Optional[str] = None,
    multimodal_prompt_config: Optional[MultimodalPromptConfig] = None,
):
    """Synchronous worker function that sets up a new event loop for the separate process."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(
            _run_text_gen_server(
                coordinator_addr,
                tokenizer,
                rank,
                server_port,
                parsers,
                verbose,
                hostname,
                chat_template,
                multimodal_prompt_config,
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


def _bind_reuseport_socket(server_port: int, hostname: Optional[str]) -> socket.socket:
    """Bind this worker's own socket on the shared port, with SO_REUSEPORT.

    Unlike inheriting one fd, this gives every worker its own accept queue and
    lets the kernel hash each connection's 4-tuple across them. It is a large
    improvement on sharing (measured 604x -> 2x spread at 32 replicas) but is
    still hashing, not balancing: it cannot see that a replica is already busy,
    so the spread is statistical rather than exact.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    # Required on every socket sharing the port; without it the second bind fails.
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    sock.bind((hostname if hostname is not None else "0.0.0.0", server_port))
    sock.setblocking(False)
    return sock


def _reserve_port(hostname: Optional[str]) -> int:
    """Pick a free port for the replicas to bind individually.

    Replicas each bind the port themselves, so the parent cannot hold the socket
    and hand out its fd; it binds only long enough to learn a free port. The gap
    before the replicas bind is a small race with unrelated processes, which is
    why an explicit port is preferred when one is available.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind((hostname if hostname is not None else "0.0.0.0", 0))
        return probe.getsockname()[1]


def start_text_gen_server(
    coordinator_addr: str,
    tokenizer,
    rank: int,
    server_port: int,
    parsers: Optional[List[str]] = None,
    verbose: bool = False,
    num_replicas: int = 4,
    hostname: Optional[str] = None,
    sock: Optional[socket.socket] = None,
    chat_template: Optional[str] = None,
    multimodal_prompt_config: Optional[MultimodalPromptConfig] = None,
) -> Optional[str]:
    """Start the text generation server.

    Every replica binds its own socket on ``server_port`` with SO_REUSEPORT, so
    each gets its own accept queue and the kernel spreads connections across
    them. Sharing one inherited socket does not balance -- replicas race to
    accept from a single queue and whichever is already running keeps winning,
    which concentrates most traffic on a handful of them as replica count grows.

    Call this on every rank that should host a frontend. Frontend work (chat
    template, detokenize, parsers, JSON) is CPU-bound, so hosting on a single
    rank confines it to that rank's CPU allocation and leaves the rest of the
    job's cores unused. Each caller gets its own URL back; collecting them and
    spreading requests over the result is the caller's business.

    Args:
        server_port: Port to listen on. Overridden by ``sock`` when given; 0
            asks the OS to choose a free one.
        sock: A socket the caller already bound, used only to fix the port.
            Replicas bind that port themselves, so it is closed here rather than
            shared with them.
        chat_template: Chat template to apply, as a file path or an inline
            template string. None falls back to the tokenizer's own template.

    Returns:
        The base URL this rank serves on, or None if the server was already
        running.
    """
    global _SERVER_PROCESSES

    if _SERVER_PROCESSES:
        logger.warning("Text gen server processes are already running.")
        return None

    if sock is not None:
        # Take the port and release the socket: replicas each bind their own with
        # SO_REUSEPORT, which one shared socket cannot provide.
        server_port = sock.getsockname()[1]
        if server_port == 0:
            raise ValueError(
                "socket must be bound to a real port before being passed to start_text_gen_server"
            )
        sock.close()
    elif server_port == 0:
        server_port = _reserve_port(hostname)

    for i in range(num_replicas):
        p = mp.Process(
            target=_server_process_worker,
            args=(
                coordinator_addr,
                tokenizer,
                rank,
                server_port,
                parsers,
                verbose,
                hostname,
                chat_template,
                multimodal_prompt_config,
            ),
            daemon=True,
        )
        p.start()
        _SERVER_PROCESSES.append(p)
        logger.info(
            f"Started text gen frontend replica {i+1}/{num_replicas} "
            f"on port {server_port} (PID: {p.pid})"
        )

    return f"http://{hostname or socket.gethostname()}:{server_port}"


def _terminate(processes: List[mp.Process], what: str):
    """Terminate a group of worker processes, escalating to kill if needed."""
    if not processes:
        return
    logger.info(f"Terminating {len(processes)} {what} processes...")
    for p in processes:
        if p.is_alive():
            p.terminate()
    for p in processes:
        p.join(timeout=3)
        if p.is_alive():
            p.kill()
            p.join()


def stop_text_gen_server():
    """Stop this rank's frontend replica processes."""
    global _SERVER_PROCESSES

    if not _SERVER_PROCESSES:
        return

    _terminate(_SERVER_PROCESSES, "Text Gen frontend")
    _SERVER_PROCESSES = []
    logger.info("All text gen frontend processes terminated.")
