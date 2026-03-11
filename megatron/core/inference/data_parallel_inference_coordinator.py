# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import asyncio
import faulthandler
import json
import logging
import signal
from collections import deque
from enum import Enum, auto
from multiprocessing import Event
from multiprocessing.connection import Connection
from typing import Optional

import torch

from megatron.core.inference.async_zmq_communicator import AsyncZmqEndpoint
from megatron.core.inference.config import PrefixCachingCoordinatorPolicy
from megatron.core.inference.headers import Headers, UnknownHeaderError
from megatron.core.inference.inference_request import compute_block_hashes_batched
from megatron.core.utils import get_asyncio_loop, trace_async_exceptions

# Register faulthandler to emit stack traces upon process kill.
faulthandler.enable()
faulthandler.register(signal.SIGTERM, all_threads=False, chain=True)
faulthandler.register(signal.SIGINT, all_threads=False, chain=True)


class DataParallelInferenceCoordinator(AsyncZmqEndpoint):
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

    Attributes:
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
        ready_event: Event,
        data_parallel_size: int,
        tokenizer,
        inference_coordinator_port: int | None = None,
        deterministic_mode: bool = False,
        block_size_tokens: int | None = None,
        enable_prefix_caching: bool = False,
        prefix_caching_coordinator_policy: PrefixCachingCoordinatorPolicy = (
            PrefixCachingCoordinatorPolicy.FIRST_PREFIX_BLOCK
        ),
        schedule_output_path: str | None = None,
    ):
        """
        Initializes the inference coordinator.

        This sets up the async ZMQ context and a ROUTER socket, binding it to the
        given port. Worker registration is deferred to the `_recv_task` which
        handles ENGINE_CONNECT messages.

        Args:
            pipe_connection (Connection): A connecting pipe to the parent process.
            ready_event (Event): Set when all engines have connected.
            data_parallel_size (int): The number of TP-coordinator workers that are
                expected to connect.
            tokenizer: The tokenizer to use for prompt tokenization and detokenization.
            inference_coordinator_port (Optional[int]): The TCP port number to bind the server to.
            deterministic_mode (bool): Whether to enable deterministic scheduling.
        """
        self.pipe_connection = pipe_connection
        self.data_parallel_size = data_parallel_size
        self.deterministic_mode = deterministic_mode
        self.ready_event = ready_event

        super().__init__("ROUTER", bind=True, bind_port=inference_coordinator_port)

        # Send the address to the parent process.
        self.pipe_connection.send(self.address)
        self.pipe_connection.close()

        self.identities_of_data_parallel_ranks = deque([])
        self._round_robin_idx = 0

        self.request_id_to_client_id = {}
        self.request_id_to_client_request_id = {}

        self.next_request_id = 0
        self.tokenizer = tokenizer
        self.state = self.CoordinatorState.RUNNING
        self.is_shutdown = False

        # Prefix caching state for routing.
        self.block_size_tokens = block_size_tokens
        self.enable_prefix_caching = enable_prefix_caching
        self.prefix_caching_coordinator_policy = prefix_caching_coordinator_policy
        self.hash_to_rank_info = {}  # Dict[int, Dict[bytes, int]]: hash → {rank → timestamp}
        self._assignment_counter = 0

        # Schedule recording.
        self.schedule_output_path = schedule_output_path
        self.schedule_records = [] if schedule_output_path else None
        self.identity_to_rank_index = {}

        # Set by shutdown() to signal the entrypoint to exit the event loop.
        self._shutdown_event = asyncio.Event()

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
        # Clamp round-robin index so it doesn't skip an engine after removal.
        n = len(self.identities_of_data_parallel_ranks)
        if n > 0:
            self._round_robin_idx = self._round_robin_idx % n
        else:
            self._round_robin_idx = 0
        for rank_info in self.hash_to_rank_info.values():
            rank_info.pop(identity, None)
        logging.warning(
            "Coordinator: removed engine %s (now %d engines)",
            identity,
            n,
        )

    def compute_request_hashes(self, prompt):
        """Compute block hashes for a prompt on CPU.

        Args:
            prompt: Either a string (to be tokenized) or a list of token IDs.

        Returns:
            List of integer block hashes, or empty list if prefix caching is disabled.
        """
        if not self.enable_prefix_caching or self.block_size_tokens is None:
            return []
        if isinstance(prompt, str):
            tokens = self.tokenizer.tokenize(prompt)
        else:
            tokens = list(prompt)
        token_tensor = torch.tensor(tokens, dtype=torch.int64)
        return compute_block_hashes_batched(token_tensor, self.block_size_tokens)

    def get_best_data_parallel_rank(self, request_hashes):
        """Select the best DP rank based on prefix cache affinity.

        Iterates request hashes in reverse order and picks the rank that cached
        the longest matching prefix (the furthest hash found). Since hashes are
        parent-chained, finding hash[i] in a rank guarantees hash[0..i-1] are
        also present. Among ranks that share the longest match, the most recently
        assigned rank (highest timestamp) is preferred. Falls back to round-robin
        when no rank matches.

        Args:
            request_hashes: List of block hashes for the request.

        Returns:
            bytes: The ZMQ identity of the selected data parallel rank.
        """
        if (
            not self.enable_prefix_caching
            or not request_hashes
            or self.prefix_caching_coordinator_policy == PrefixCachingCoordinatorPolicy.ROUND_ROBIN
        ):
            return self.get_next_data_parallel_rank()

        # Reverse scan: first match is the longest prefix (parent-chained hashes).
        for h in reversed(request_hashes):
            rank_info = self.hash_to_rank_info.get(h)
            if rank_info:
                # Pick the most recently assigned rank.
                best_rank = max(rank_info, key=rank_info.get)
                return best_rank

        return self.get_next_data_parallel_rank()

    def _update_rank_hashes(self, rank_identity, request_hashes):
        """Record that a rank owns the given hashes.

        Args:
            rank_identity: ZMQ identity of the target rank.
            request_hashes: List of block hashes assigned to this rank.
        """
        self._assignment_counter += 1
        ts = self._assignment_counter
        for h in request_hashes:
            self.hash_to_rank_info.setdefault(h, {})[rank_identity] = ts

    def start(self, loop: Optional[asyncio.AbstractEventLoop] = None):
        """
        Starts the main event loop for the coordinator.

        Creates background tasks for receiving, sending, and startup send buffering.
        Returns immediately — the actual work happens in the background tasks driven
        by the event loop.

        Args:
            loop (Optional[asyncio.AbstractEventLoop]): The event loop to use.
        """
        logging.info("Inference Coordinator: waiting for connections from data parallel ranks...")

        if self.data_parallel_size == 0:
            self.is_running.set()
            if self.ready_event is not None:
                self.ready_event.set()

        super().start(loop, set_running=False)

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
                    logging.warning(
                        "Coordinator: duplicate ENGINE_CONNECT from %s, ignoring.", identity
                    )
                    continue
                self.identities_of_data_parallel_ranks.append(identity)
                logging.info(f"Inference Coordinator: Data parallel rank connected: {identity}")
                if len(self.identities_of_data_parallel_ranks) == self.data_parallel_size:
                    # In deterministic mode, sort identities for consistent scheduling order.
                    if self.deterministic_mode:
                        self.identities_of_data_parallel_ranks = deque(
                            sorted(self.identities_of_data_parallel_ranks)
                        )
                    # Rebuild rank index mapping now that all engines are known.
                    sorted_ids = sorted(self.identities_of_data_parallel_ranks)
                    self.identity_to_rank_index = {
                        ident: idx for idx, ident in enumerate(sorted_ids)
                    }
                    self.is_running.set()
                    if self.ready_event is not None:
                        self.ready_event.set()
                    logging.info("Inference Coordinator: Connected with data parallel ranks...")

            elif header == Headers.CLIENT_CONNECT:
                if identity in known_clients:
                    logging.info(f"Client {identity} sent a duplicate connect request. Ignoring ..")
                    continue
                known_clients.add(identity)
                # Due to the `startup_sends` logic, this will not be sent until we are connected.
                self._isend(Headers.ACK, identity=identity)

            elif header == Headers.SUBMIT_REQUEST:
                # Message from a known client
                if identity not in known_clients:
                    logging.info(f"Received message from unknown client {identity}. Ignoring.")
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

                request_hashes = self.compute_request_hashes(prompt)
                if (
                    self.prefix_caching_coordinator_policy
                    == PrefixCachingCoordinatorPolicy.FIRST_PREFIX_BLOCK
                ):
                    request_hashes = request_hashes[:1]

                next_data_parallel_rank_identity = self.get_best_data_parallel_rank(request_hashes)
                self._isend(
                    Headers.SUBMIT_REQUEST,
                    [request_id, prompt, sampling_params],
                    identity=next_data_parallel_rank_identity,
                )

                if request_hashes:
                    self._update_rank_hashes(next_identity, request_hashes)
                if self.schedule_records is not None:
                    self.schedule_records.append(
                        {
                            "request_id": request_id,
                            "rank_index": self.identity_to_rank_index[next_identity],
                            "num_hashes": len(request_hashes),
                        }
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
                    self._isend(header, identity=data_parallel_rank_id)

                # STOP affects engines; reset coordinator to RUNNING to allow future engines.
                if header == Headers.STOP:
                    self.state = self.CoordinatorState.RUNNING

            elif header == Headers.ENGINE_REPLY:
                # This is the output of a single engine step on some data parallel rank.
                if identity not in self.identities_of_data_parallel_ranks:
                    logging.warning(
                        "Coordinator: ENGINE_REPLY from unknown engine %s, ignoring.", identity
                    )
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
                        Headers.ENGINE_REPLY,
                        [client_request_identity, finished_request_record],
                        identity=client_identity,
                    )

            elif header == Headers.SHUTDOWN:
                if identity not in known_clients:
                    logging.warning("Coordinator: ignoring signal from unknown client.")
                    continue
                await self.shutdown()
                return

            elif header == Headers.DISCONNECT:
                if identity in self.identities_of_data_parallel_ranks:
                    self._remove_engine(identity)

            else:
                raise UnknownHeaderError(header)

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
        block_size_tokens: int | None = None,
        enable_prefix_caching: bool = False,
        prefix_caching_coordinator_policy: PrefixCachingCoordinatorPolicy = (
            PrefixCachingCoordinatorPolicy.FIRST_PREFIX_BLOCK
        ),
        schedule_output_path: str | None = None,
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
            block_size_tokens (Optional[int]): Token block size for prefix caching hashing.
            enable_prefix_caching (bool): Whether prefix caching is enabled.
            prefix_caching_coordinator_policy (PrefixCachingCoordinatorPolicy): Routing policy.
            schedule_output_path (Optional[str]): Path to write scheduling decisions JSON.
        """
        coordinator = cls(
            pipe_connection,
            ready_event,
            data_parallel_size,
            tokenizer,
            inference_coordinator_port,
            deterministic_mode=deterministic_mode,
            block_size_tokens=block_size_tokens,
            enable_prefix_caching=enable_prefix_caching,
            prefix_caching_coordinator_policy=prefix_caching_coordinator_policy,
            schedule_output_path=schedule_output_path,
        )
        loop = get_asyncio_loop()
        coordinator.start(loop=loop)
        try:
            loop.run_until_complete(coordinator._shutdown_event.wait())
        except KeyboardInterrupt:
            logging.info("Coordinator process interrupted. Exiting...")
        finally:
            loop.run_until_complete(coordinator.shutdown())
        logging.info("Inference Coordinator: shut down successfully.")

    async def shutdown(self):
        """Stops the inference coordinator, performing any necessary cleanup operations."""
        if self.is_shutdown:
            return
        self.is_shutdown = True
        if self.schedule_output_path and self.schedule_records:
            schedule_data = {
                "policy": self.prefix_caching_coordinator_policy.value,
                "data_parallel_size": self.data_parallel_size,
                "num_requests": len(self.schedule_records),
                "records": self.schedule_records,
            }
            with open(self.schedule_output_path, "w") as f:
                json.dump(schedule_data, f, indent=2)
        await super().shutdown()
        self._ctx.term()
        self._shutdown_event.set()
