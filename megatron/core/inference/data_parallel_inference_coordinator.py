# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import asyncio
import errno
import faulthandler
import logging
import signal
import socket
from collections import deque
from enum import Enum, auto
from multiprocessing import Event
from multiprocessing.connection import Connection
from typing import List, Optional, Tuple

import torch

from megatron.core.inference.async_zmq_socket import AsyncZmqSendRecv
from megatron.core.inference.headers import Headers, UnknownHeaderError
from megatron.core.utils import get_asyncio_loop, trace_async_exceptions

try:
    import zmq
    import zmq.asyncio

    HAVE_ZMQ = True
except:
    HAVE_ZMQ = False

# Register faulthandler to emit stack traces upon process kill.
faulthandler.enable()
faulthandler.register(signal.SIGTERM, all_threads=False, chain=True)
faulthandler.register(signal.SIGINT, all_threads=False, chain=True)


class DataParallelInferenceCoordinator:
    """
    Coordinates inference requests between clients and distributed model engines.

    This class acts as a central server. It uses a ZMQ ROUTER socket to manage
    communication flows between multiple clients and multiple data parallel ranks.

    The coordinator's main responsibilities are:
    1.  **Worker Registration**: It waits for a specified number of data parallel ranks
        (representing distributed model instances) to connect and register themselves.
    2.  **Client Connection**: It accepts connections from external clients, like
        `InferenceClient`, and performs a simple handshake.
    3.  **Request Forwarding**: It receives inference requests from clients, assigns a
        unique server-side request ID, tokenizes the prompt, and forwards the request
        to one of the available data parallel rank using a round-robin scheduling
        strategy.
    4.  **Response Routing**: It receives completed results from
        the data parallel ranks and routes them back to the original client that made the
        request.
    5.  **Control Signal Broadcasting**: It relays control signals (e.g., PAUSE, STOP)
        from a client to all connected data parallel ranks.

    Sends are decoupled from the main recv loop via a send queue pattern:
    ``_isend()`` enqueues send futures, and a background ``_send_task`` drains
    and awaits them.

    Attributes:
        router_socket (zmq.asyncio.Socket): The central async ZMQ ROUTER socket for
            all communication.
        data_parallel_size (int): The number of data parallel workers to expect.
        identities_of_data_parallel_ranks (deque): A deque holding the ZMQ
            identities of connected TP-coordinators, used for round-robin scheduling.
        request_id_to_client_id (dict): Maps server-side request IDs to the ZMQ
            identity of the client that initiated the request.
        request_id_to_client_request_id (dict): Maps server-side request IDs to the
            original request ID provided by the client.
        next_request_id (int): A counter for generating unique server-side request IDs.
    """

    class CoordinatorState(Enum):
        """State machine for the coordinator."""

        RUNNING = auto()
        PAUSED = auto()
        SUSPENDED = auto()
        STOPPING = auto()

    def __init__(
        self,
        pipe_connection: Connection,
        data_parallel_size: int,
        tokenizer,
        inference_coordinator_port: int | None = None,
        deterministic_mode: bool = False,
    ):
        """
        Initializes the inference coordinator.

        This sets up the async ZMQ context and a ROUTER socket, binding it to the
        given port. Worker registration is deferred to the ``_recv_task`` which
        handles ENGINE_CONNECT messages.

        Args:
            pipe_connection (Connection): A connecting pipe to the parent process.
            data_parallel_size (int): The number of TP-coordinator workers that are
                expected to connect.
            tokenizer: The tokenizer to use for prompt tokenization and detokenization.
            inference_coordinator_port (Optional[int]): The TCP port number to bind the server to.
            deterministic_mode (bool): Whether to enable deterministic scheduling.
        """
        assert HAVE_ZMQ, (
            "please install the pyzmq library to use DataParallelInferenceCoordinator\n"
            "pip install pyzmq"
        )
        self.pipe_connection = pipe_connection
        self.data_parallel_size = data_parallel_size
        self.deterministic_mode = deterministic_mode
        self.ready_event = None  # Set in start()

        self.context = zmq.asyncio.Context.instance()

        # This is the central router socket
        # 1. data parallel ranks connect to this socket to register themselves
        # 2. Users connect to this socket and submit their requests. We transmit them to
        #    data parallel ranks in a round robin fashion
        # 3. data parallel ranks return completed requests to this socket. We route them back to
        #    the user that had submitted the request originally.

        # Get local IP.
        local_ip = socket.gethostname()

        self.router_socket = self.context.socket(zmq.ROUTER)
        self._zmq = AsyncZmqSendRecv()
        # Raise error if the other side of the connection has dropped.
        self.router_socket.setsockopt(zmq.ROUTER_MANDATORY, 1)
        is_bound = False
        if inference_coordinator_port is not None:
            try:
                self.router_socket.bind(f"tcp://{local_ip}:{inference_coordinator_port}")
                is_bound = True
            except zmq.error.ZMQError as e:
                if e.errno == errno.EADDRINUSE:
                    logging.warning(
                        f"Port {inference_coordinator_port} is already in use. "
                        "Binding to a random available port instead."
                    )
            except Exception:
                logging.warning(
                    f"Unknown error when binding to port {inference_coordinator_port}. "
                    "Attempting to bind to a random available port instead."
                )
        if not is_bound:
            self.router_socket.bind_to_random_port(f"tcp://{local_ip}")
        self.addr = self.router_socket.getsockopt_string(zmq.LAST_ENDPOINT)

        # Send the address to the parent process.
        self.pipe_connection.send(self.addr)
        self.pipe_connection.close()

        self.identities_of_data_parallel_ranks = deque([])
        self._round_robin_idx = 0

        self.request_id_to_client_id = {}
        self.request_id_to_client_request_id = {}

        self.next_request_id = 0
        self.tokenizer = tokenizer
        self.state = self.CoordinatorState.RUNNING

    def get_next_data_parallel_rank(self):
        """
        Selects the next data parallel rank using round-robin scheduling.

        Returns:
            bytes: The ZMQ identity of the next data parallel rank to receive a request.
        """
        identities = self.identities_of_data_parallel_ranks
        if not identities:
            raise RuntimeError("No engines connected")
        idx = self._round_robin_idx % len(identities)
        self._round_robin_idx = idx + 1
        return identities[idx]

    def _remove_engine(self, identity):
        """Remove a disconnected engine from the routing pool."""
        self.identities_of_data_parallel_ranks.remove(identity)
        logging.warning(
            "Coordinator: removed engine %s (now %d engines)",
            identity,
            len(self.identities_of_data_parallel_ranks),
        )

    def start(self, ready_event: Event, loop: Optional[asyncio.AbstractEventLoop] = None):
        """
        Starts the main event loop for the coordinator.

        Creates background tasks for receiving, sending, and startup send buffering.
        Returns immediately — the actual work happens in the background tasks driven
        by the event loop.

        Args:
            ready_event (Event): Set when all engines have connected.
            loop (Optional[asyncio.AbstractEventLoop]): The event loop to use.
        """
        logging.info("Inference Coordinator: waiting for connections from data parallel ranks...")
        loop = get_asyncio_loop(loop)

        self.ready_event = ready_event

        # Attempt to connect, and do not allow any sends until we are connected.
        self.is_running = asyncio.Event()
        self._startup_sends = []

        if self.data_parallel_size == 0:
            self.is_running.set()
            if self.ready_event is not None:
                self.ready_event.set()

        self.startup_sends_task = loop.create_task(self._startup_sends_task())
        self.send_task = loop.create_task(self._zmq.send_task(on_send_error=self._on_send_error))
        self.recv_task = loop.create_task(self._recv_task())

    @trace_async_exceptions
    async def _recv_task(self):
        """Main loop of the inference coordinator.

        Listens for incoming messages and dispatches based on header type.
        Handles engine registration (ENGINE_CONNECT), client handshake
        (CLIENT_CONNECT), request forwarding, control signals, and replies.
        """
        known_clients = set()
        while True:
            identity, header, data = await self._irecv()

            if header == Headers.ENGINE_CONNECT:
                if identity in self.identities_of_data_parallel_ranks:
                    logging.warning("Coordinator: duplicate ENGINE_CONNECT from %s, ignoring.", identity)
                    continue
                self.identities_of_data_parallel_ranks.append(identity)
                logging.info(f"Inference Coordinator: Data parallel rank connected: {identity}")
                if len(self.identities_of_data_parallel_ranks) == self.data_parallel_size:
                    # In deterministic mode, sort identities for consistent scheduling order.
                    if self.deterministic_mode:
                        self.identities_of_data_parallel_ranks = deque(
                            sorted(self.identities_of_data_parallel_ranks)
                        )
                    self.is_running.set()
                    if self.ready_event is not None:
                        self.ready_event.set()
                    logging.info("Inference Coordinator: Connected with data parallel ranks...")

            elif header == Headers.CLIENT_CONNECT:
                if identity in known_clients:
                    logging.info(
                        f"Client {identity} sent a duplicate connect request. Ignoring .."
                    )
                    continue
                known_clients.add(identity)
                # Due to the `startup_sends` logic, this will not be sent until we are connected.
                self._isend(identity, Headers.ACK)

            elif header == Headers.SUBMIT_REQUEST:
                # Message from a known client
                if identity not in known_clients:
                    logging.info(
                        f"Received message from unknown client {identity}. Ignoring."
                    )
                    continue
                # this is a message from a client.
                # route it to a data parallel rank
                client_request_id, prompt, sampling_params = data
                # map client request_id to server request_id
                # necessary because multiple clients might have the same request_id.
                request_id = self.next_request_id
                self.next_request_id += 1
                self.request_id_to_client_id[request_id] = identity
                self.request_id_to_client_request_id[request_id] = client_request_id

                # Serialize prompt.
                if isinstance(prompt, (str, list)):
                    pass
                elif isinstance(prompt, torch.Tensor):
                    prompt = prompt.tolist()
                else:
                    raise Exception("specialize for <%s> prompt." % type(prompt).__name__)

                next_data_parallel_rank_identity = self.get_next_data_parallel_rank()
                self._isend(
                    next_data_parallel_rank_identity,
                    Headers.SUBMIT_REQUEST,
                    [request_id, prompt, sampling_params],
                )

            elif header in (
                Headers.PAUSE,
                Headers.UNPAUSE,
                Headers.SUSPEND,
                Headers.RESUME,
                Headers.INCREMENT_STALENESS,
                Headers.STOP,
            ):
                # Start by checking the current state against the control signal.
                if identity not in known_clients:
                    logging.warning("Coordinator: ignoring signal from unknown client.")
                    continue

                if header == Headers.PAUSE:
                    idem_states = (self.CoordinatorState.PAUSED, self.CoordinatorState.SUSPENDED)
                    if self.state == self.CoordinatorState.RUNNING:
                        self.state = self.CoordinatorState.PAUSED
                    elif self.state in idem_states:
                        # Already paused/suspended, ignore redundant PAUSE.
                        continue
                    else:
                        logging.warning("Coordinator: ignoring PAUSE in state %s", self.state)
                        continue
                elif header == Headers.UNPAUSE:
                    if self.state != self.CoordinatorState.PAUSED:
                        logging.warning("Coordinator: ignoring UNPAUSE in state %s", self.state)
                        continue
                    self.state = self.CoordinatorState.RUNNING
                elif header == Headers.SUSPEND:
                    if self.state != self.CoordinatorState.PAUSED:
                        logging.warning("Coordinator: ignoring SUSPEND in state %s", self.state)
                        continue
                    self.state = self.CoordinatorState.SUSPENDED
                elif header == Headers.RESUME:
                    if self.state != self.CoordinatorState.SUSPENDED:
                        logging.warning("Coordinator: ignoring RESUME in state %s", self.state)
                        continue
                    self.state = self.CoordinatorState.PAUSED
                elif header == Headers.STOP:
                    good_states = (self.CoordinatorState.PAUSED, self.CoordinatorState.SUSPENDED)
                    if self.state not in good_states:
                        logging.warning("Coordinator: ignoring STOP in state %s", self.state)
                        continue
                    self.state = self.CoordinatorState.STOPPING

                # Broadcast the control signal to all data parallel ranks.
                for data_parallel_rank_id in list(self.identities_of_data_parallel_ranks):
                    self._isend(data_parallel_rank_id, header)

            elif header == Headers.ENGINE_REPLY:
                # This is the output of a single engine step on some data parallel rank.
                if identity not in self.identities_of_data_parallel_ranks:
                    logging.warning("Coordinator: ENGINE_REPLY from unknown engine %s, ignoring.", identity)
                    continue
                finished_request_records = data

                for finished_request_record in finished_request_records:
                    self.detokenize(finished_request_record)
                    fid = finished_request_record["requests"][0]["request_id"]
                    client_identity = self.request_id_to_client_id[fid]
                    client_request_identity = self.request_id_to_client_request_id[fid]
                    del self.request_id_to_client_id[fid]
                    del self.request_id_to_client_request_id[fid]

                    self._isend(
                        client_identity,
                        Headers.ENGINE_REPLY,
                        [client_request_identity, finished_request_record],
                    )

            elif header == Headers.SHUTDOWN:
                if identity not in known_clients:
                    logging.warning("Coordinator: ignoring signal from unknown client.")
                    continue
                break

            elif header == Headers.DISCONNECT:
                if identity in self.identities_of_data_parallel_ranks:
                    self._remove_engine(identity)

            else:
                raise UnknownHeaderError(header)

    def _on_send_error(self, e, identity):
        """Handle send errors from the background send task."""
        if isinstance(e, zmq.error.ZMQError) and e.errno == zmq.EHOSTUNREACH:
            if identity is not None and identity in self.identities_of_data_parallel_ranks:
                self._remove_engine(identity)
            else:
                logging.warning("Coordinator: send failed, recipient unreachable (identity=%s)", identity)
            return True
        return False

    @trace_async_exceptions
    async def _startup_sends_task(self):
        """Before all engines are connected, we queue up sends for later."""
        await self.is_running.wait()
        for (identity, header, data) in self._startup_sends:
            self._isend(identity, header, data)
        self._startup_sends = None

    def _isend(
        self, identity: bytes, header: Headers, data: Optional[List] = None
    ):
        """
        Asynchronously send a signal via the ROUTER socket.

        Args:
            identity (bytes): The ZMQ identity of the recipient.
            header (Headers): The signal header to send.
            data (Optional[List]): The data payload to send.
        """
        # If we have not connected yet, wait on sends.
        if not self.is_running.is_set():
            if header not in [Headers.ENGINE_CONNECT, Headers.CLIENT_CONNECT]:
                self._startup_sends.append((identity, header, data))
                return

        self._zmq.isend(self.router_socket, header, data, identity=identity)

    async def _irecv(
        self, deserialize: bool = True
    ) -> Tuple[Optional[bytes], Headers, List | bytes | None]:
        """
        Asynchronously receive a signal from the ROUTER socket.

        Returns:
            identity (Optional[bytes]): The source of the signal.
            header (Headers): The signal header received.
            data (List | bytes | None): The data payload received.
        """
        return await self._zmq.irecv(
            self.router_socket, socket_uses_identity=True, deserialize=deserialize
        )

    def detokenize(self, finished_request_record):
        """
        Detokenizes the generated tokens in the finished request record.

        This method uses the coordinator's tokenizer to convert the list of
        generated token IDs back into human-readable text.

        Args:
            finished_request_record (dict): The record containing the generated
                tokens to be detokenized. It is modified in place.
        """
        for request in finished_request_record["requests"]:
            if request["prompt"] is None:
                request["prompt"] = self.tokenizer.detokenize(request["prompt_tokens"][1])
            request["generated_text"] = self.tokenizer.detokenize(request["generated_tokens"])

    @classmethod
    def entrypoint(
        cls,
        pipe_connection: Connection,
        ready_event: Event,
        data_parallel_size: int,
        tokenizer,
        inference_coordinator_port: int | None = None,
        deterministic_mode: bool = False,
    ):
        """
        Class method to instantiate and run the coordinator, for use in a separate process.

        This method initializes the coordinator, starts the background tasks, and
        runs the event loop forever until interrupted or stopped.

        Args:
            pipe_connection (Connection): A connecting pipe to the parent process.
            ready_event (Event): A threading or multiprocessing event object that is set()
                once the coordinator is ready to accept connections.
            inference_coordinator_port (int): The port to bind to.
            data_parallel_size (int): The number of expected TP-coordinators.
            deterministic_mode (bool): Whether to enable deterministic scheduling.
        """
        coordinator = cls(
            pipe_connection,
            data_parallel_size,
            tokenizer,
            inference_coordinator_port,
            deterministic_mode=deterministic_mode,
        )
        loop = get_asyncio_loop()
        coordinator.start(ready_event, loop=loop)
        try:
            loop.run_forever()
        except KeyboardInterrupt:
            logging.info("Coordinator process interrupted. Exiting...")
        finally:
            coordinator.stop()
        logging.info("Inference Coordinator: shut down successfully.")

    def stop(self):
        """
        Stops the inference coordinator, performing any necessary cleanup operations.
        """
        self._zmq.shutdown()
        self.recv_task.cancel()
        self.startup_sends_task.cancel()
        self.router_socket.close()
        self.context.term()
