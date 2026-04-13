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

import numpy as np
import torch

from megatron.core.inference.async_zmq_communicator import AsyncZmqEndpoint
from megatron.core.inference.config import PrefixCachingCoordinatorPolicy
from megatron.core.inference.headers import Headers, UnknownHeaderError
from megatron.core.inference.inference_request import compute_block_hashes_batched
from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    TextGenerationController,
)
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
        max_requests,
        inference_coordinator_port: int | None = None,
        deterministic_mode: bool = False,
        block_size_tokens: int | None = None,
        enable_prefix_caching: bool = False,
        prefix_caching_coordinator_policy: PrefixCachingCoordinatorPolicy = (
            PrefixCachingCoordinatorPolicy.FIRST_PREFIX_BLOCK
        ),
        prefix_caching_routing_alpha: float = 0.5,
        schedule_output_path: str | None = None,
        hostname: str | None = None,
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
            prefix_caching_routing_alpha (float): Weight for prefix-aware routing score:
                score = alpha * match + (1 - alpha) * normalized_load.
            max_requests (int): Max concurrent requests per rank, used to
                compute normalized_load for prefix-aware scoring.
            deterministic_mode (bool): Whether to enable deterministic scheduling.
        """
        self.pipe_connection = pipe_connection
        self.data_parallel_size = data_parallel_size
        self.deterministic_mode = deterministic_mode
        self.ready_event = ready_event

        super().__init__(
            "ROUTER", bind=True, bind_host=hostname, bind_port=inference_coordinator_port
        )

        # Send the address to the parent process.
        self.pipe_connection.send(self.address)
        self.pipe_connection.close()

        self.identities_of_data_parallel_ranks = deque([])
        self._round_robin_idx = 0

        self.request_id_to_client_id = {}
        self.request_id_to_client_request_id = {}
        self.request_id_to_rank = {}  # Maps request_id → rank identity for pending count tracking

        self.next_request_id = 0
        self.tokenizer = tokenizer
        self.state = self.CoordinatorState.RUNNING

        # Prefix caching state for routing.
        self.block_size_tokens = block_size_tokens
        self.enable_prefix_caching = enable_prefix_caching
        self.prefix_caching_coordinator_policy = prefix_caching_coordinator_policy
        self.prefix_caching_routing_alpha = prefix_caching_routing_alpha
        self.max_requests = max_requests
        assert self.max_requests is not None and self.max_requests > 0

        # Schedule recording.
        self.schedule_output_path = schedule_output_path
        self.schedule_records = [] if schedule_output_path else None
        self.identity_to_rank_index = {}

        # Set by shutdown() to signal the entrypoint to exit the event loop.
        self._shutdown_event = asyncio.Event()

        # Numpy arrays for vectorized scoring (indexed by rank index).
        # Starts empty; populated by _register_rank_identity as engines connect.
        self._identities_list = []  # rank_index → identity
        self._pending_counts = np.zeros(0, dtype=np.int32)
        self._quorum_reached = False

        # Hash → {rank_idx: timestamp} dict for prefix cache affinity routing.
        # Each key is a block hash; each value maps rank indices to assignment
        # timestamps (positive int).  Missing entries are implicitly zero.
        self._hash_table: dict[int, dict[int, int]] = {}
        self._hash_assignment_counter = 0

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

    def _register_rank_identity(self, identity):
        """Register a new rank identity in the scoring data structures.

        Called when a rank dynamically connects to a running coordinator
        (e.g. in tests that spawn the coordinator with data_parallel_size=0
        and let engines register after the fact).
        """
        if identity in self.identity_to_rank_index:
            return
        new_idx = len(self._identities_list)
        self.identity_to_rank_index[identity] = new_idx
        self._identities_list.append(identity)
        self._pending_counts = np.append(self._pending_counts, np.int32(0))
        logging.info(
            "Coordinator: registered engine %s as rank index %d (now %d engines)",
            identity,
            new_idx,
            len(self._identities_list),
        )

    def _remove_engine(self, identity):
        """Remove a disconnected engine from the routing and scoring pools.

        Prunes the engine from all data structures: the round-robin deque,
        the scoring arrays (_identities_list, _pending_counts), the
        identity_to_rank_index map, and the prefix-cache _hash_table.
        Indices above the removed rank are shifted down by one so that the
        _hash_table, _pending_counts, and _identities_list stay consistent.
        """
        self.identities_of_data_parallel_ranks.remove(identity)
        # Clamp round-robin index so it doesn't skip an engine after removal.
        n = len(self.identities_of_data_parallel_ranks)
        if n > 0:
            self._round_robin_idx = self._round_robin_idx % n
        else:
            self._round_robin_idx = 0

        rank_idx = self.identity_to_rank_index.pop(identity, None)
        if rank_idx is not None:
            # Remove from the parallel scoring arrays.
            self._identities_list.pop(rank_idx)
            self._pending_counts = np.delete(self._pending_counts, rank_idx)

            # Rebuild identity_to_rank_index since indices after rank_idx shifted.
            self.identity_to_rank_index = {
                ident: idx for idx, ident in enumerate(self._identities_list)
            }

            # Remap _hash_table: drop the removed rank, shift higher indices down.
            remapped = {}
            for h, rank_info in self._hash_table.items():
                new_info = {}
                for ridx, ts in rank_info.items():
                    if ridx < rank_idx:
                        new_info[ridx] = ts
                    elif ridx > rank_idx:
                        new_info[ridx - 1] = ts
                    # ridx == rank_idx → dropped
                if new_info:
                    remapped[h] = new_info
            self._hash_table = remapped

        logging.warning("Coordinator: removed engine %s (now %d engines)", identity, n)

    def _handle_unreachable_identity(self, identity):
        """Override: prune the dead engine instead of just logging.

        EHOSTUNREACH means the peer DEALER socket on the other side has gone away. A
        crashed engine never sends DISCONNECT, so without this hook the coordinator would
        keep round-robining requests to a dead identity forever.
        """
        super()._handle_unreachable_identity(identity)
        if identity in self.identities_of_data_parallel_ranks:
            self._remove_engine(identity)

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
        """Select the best DP rank based on prefix cache affinity and load.

        Uses a scoring function: score = alpha * match + (1 - alpha) * normalized_load
        where *match* is a policy-dependent affinity score in [0, 1] (binary for
        ``first_prefix_block``, normalized prefix depth for ``longest_prefix``)
        and normalized_load = free_slots / max_requests (higher means more free
        capacity).

        Args:
            request_hashes: List of block hashes for the request.

        Returns:
            bytes: The ZMQ identity of the selected data parallel rank.
        """
        if self.prefix_caching_coordinator_policy == PrefixCachingCoordinatorPolicy.ROUND_ROBIN:
            return self.get_next_data_parallel_rank()

        if not self.enable_prefix_caching or not request_hashes:
            return self.get_next_data_parallel_rank()

        match, recency = self._match_vector(request_hashes)

        alpha = self.prefix_caching_routing_alpha

        # Vectorized score: alpha * match + (1-alpha) * free_capacity_fraction.
        free_slots = np.maximum(0, self.max_requests - self._pending_counts).astype(np.float64)
        scores = alpha * match + (1.0 - alpha) * (free_slots / self.max_requests)

        # Tiebreak: highest score, then highest recency, then lowest rank index.
        n_ranks = len(self._identities_list)
        order = np.lexsort((np.arange(n_ranks), -recency, -scores))
        best_idx = int(order[0])
        return self._identities_list[best_idx]

    def _update_rank_hashes(self, rank_identity, request_hashes):
        """Record that a rank owns the given hashes.

        Args:
            rank_identity: ZMQ identity of the target rank.
            request_hashes: List of block hashes assigned to this rank.
        """
        rank_idx = self.identity_to_rank_index[rank_identity]
        self._hash_assignment_counter += 1
        ts = self._hash_assignment_counter
        for h in request_hashes:
            self._hash_table.setdefault(h, {})[rank_idx] = ts

    def _match_vector(self, hashes):
        """Return ``(match, recency)`` vectors of shape ``(n_ranks,)``.

        *match* is binary depth: ``(depth + 1) / len(hashes)`` for ranks that
        have the deepest cached block, 0 otherwise.  *recency* is the raw
        assignment timestamp for each matching rank (0 for non-matching ranks).

        For ``FIRST_PREFIX_BLOCK`` the caller already truncates *hashes* to a
        single element, so the same logic yields a binary 0/1 match score.
        """
        n_ranks = len(self._identities_list)
        n = len(hashes)
        zeros = np.zeros(n_ranks, dtype=np.float64)
        if n == 0:
            return zeros, zeros.copy()
        for i in range(n - 1, -1, -1):
            row = self._hash_table.get(hashes[i])
            if row is None:
                continue
            rank_idxs = np.fromiter(row.keys(), dtype=np.intp)
            present = np.zeros(n_ranks, dtype=bool)
            present[rank_idxs] = True
            recency = np.zeros(n_ranks, dtype=np.float64)
            recency[rank_idxs] = np.fromiter(row.values(), dtype=np.float64)
            if present.any():
                return present.astype(np.float64) * ((i + 1.0) / n), recency
        return zeros, zeros.copy()

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
                self._register_rank_identity(identity)
                logging.info(f"Inference Coordinator: Data parallel rank connected: {identity}")
                if len(self.identities_of_data_parallel_ranks) == self.data_parallel_size:
                    if not self._quorum_reached:
                        self._quorum_reached = True
                        # In deterministic mode, sort identities for consistent scheduling.
                        if self.deterministic_mode:
                            self.identities_of_data_parallel_ranks = deque(
                                sorted(self.identities_of_data_parallel_ranks)
                            )
                        # Rebuild all scoring data structures in sorted order.
                        sorted_ids = sorted(self.identities_of_data_parallel_ranks)
                        self.identity_to_rank_index = {
                            ident: idx for idx, ident in enumerate(sorted_ids)
                        }
                        self._identities_list = list(sorted_ids)
                        self._pending_counts = np.zeros(len(sorted_ids), dtype=np.int32)
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

                # Guard against the case where every engine has disconnected while requests
                # are still arriving. Without this, get_best_data_parallel_rank would index
                # into an empty pool and crash.
                if not self.identities_of_data_parallel_ranks:
                    logging.error(
                        "Coordinator: no engines available for request %d; dropping.", request_id
                    )
                    del self.request_id_to_client_id[request_id]
                    del self.request_id_to_client_request_id[request_id]
                    continue

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

                self.request_id_to_rank[request_id] = next_data_parallel_rank_identity
                self._pending_counts[
                    self.identity_to_rank_index[next_data_parallel_rank_identity]
                ] += 1
                if request_hashes:
                    self._update_rank_hashes(next_data_parallel_rank_identity, request_hashes)
                if self.schedule_records is not None:
                    self.schedule_records.append(
                        {
                            "request_id": request_id,
                            "rank_index": self.identity_to_rank_index[
                                next_data_parallel_rank_identity
                            ],
                            "num_hashes": len(request_hashes),
                        }
                    )

            elif header in (
                Headers.PAUSE,
                Headers.UNPAUSE,
                Headers.SUSPEND,
                Headers.RESUME,
                Headers.SET_GENERATION_EPOCH,
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
                # Forward data so data-bearing signals (e.g. SET_GENERATION_EPOCH)
                # retain their arguments.
                for data_parallel_rank_id in list(self.identities_of_data_parallel_ranks):
                    self._isend(header, data, identity=data_parallel_rank_id)

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
                finished_requests = data

                for finished_request in finished_requests:
                    self.detokenize(finished_request)
                    fid = finished_request["request_id"]
                    client_identity = self.request_id_to_client_id[fid]
                    client_request_identity = self.request_id_to_client_request_id[fid]
                    del self.request_id_to_client_id[fid]
                    del self.request_id_to_client_request_id[fid]
                    assigned_rank = self.request_id_to_rank.pop(fid, None)
                    if assigned_rank is not None:
                        idx = self.identity_to_rank_index.get(assigned_rank)
                        if idx is not None:
                            assert self._pending_counts[idx] >= 1
                            self._pending_counts[idx] -= 1

                    self._isend(
                        Headers.ENGINE_REPLY,
                        [client_request_identity, finished_request],
                        identity=client_identity,
                    )

            elif header == Headers.SHUTDOWN:
                if identity not in known_clients:
                    logging.warning("Coordinator: ignoring signal from unknown client.")
                    continue
                # Signal the entrypoint to exit; its finally block calls shutdown()
                # cleanly from outside this task. Calling shutdown() from inside
                # _recv_task would self-cancel and skip socket/context cleanup.
                self._shutdown_event.set()
                return

            elif header == Headers.DISCONNECT:
                if identity in self.identities_of_data_parallel_ranks:
                    self._remove_engine(identity)

            else:
                raise UnknownHeaderError(header)

    def detokenize(self, finished_request):
        """
        Detokenizes the generated tokens in the finished request.

        This method uses the coordinator's tokenizer to convert the list of
        generated token IDs back into human-readable text.

        Args:
            finished_request (dict): The serialized merged request containing the
                generated tokens to be detokenized. It is modified in place.
        """
        if finished_request["prompt"] is None:
            finished_request["prompt"] = TextGenerationController.detokenize(
                self.tokenizer, finished_request["prompt_tokens"][1], remove_EOD=False
            )
        detokenize_stop_sequence = (finished_request.get("sampling_params", {}) or {}).get(
            "detokenize_stop_sequence", False
        )
        finished_request["generated_text"] = TextGenerationController.detokenize(
            self.tokenizer,
            finished_request["generated_tokens"],
            remove_EOD=not detokenize_stop_sequence,
        )

    @classmethod
    def entrypoint(
        cls,
        pipe_connection: Connection,
        ready_event: Event,
        data_parallel_size: int,
        tokenizer,
        max_requests,
        inference_coordinator_port: int | None = None,
        deterministic_mode: bool = False,
        block_size_tokens: int | None = None,
        enable_prefix_caching: bool = False,
        prefix_caching_coordinator_policy: PrefixCachingCoordinatorPolicy = (
            PrefixCachingCoordinatorPolicy.FIRST_PREFIX_BLOCK
        ),
        prefix_caching_routing_alpha: float = 0.5,
        schedule_output_path: str | None = None,
        hostname: str | None = None,
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
            prefix_caching_routing_alpha (float): Weight for prefix-aware routing score.
            max_requests (int): Max concurrent requests per rank.
        """
        coordinator = cls(
            pipe_connection,
            ready_event,
            data_parallel_size,
            tokenizer,
            max_requests,
            inference_coordinator_port,
            deterministic_mode=deterministic_mode,
            block_size_tokens=block_size_tokens,
            enable_prefix_caching=enable_prefix_caching,
            prefix_caching_coordinator_policy=prefix_caching_coordinator_policy,
            prefix_caching_routing_alpha=prefix_caching_routing_alpha,
            schedule_output_path=schedule_output_path,
            hostname=hostname,
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
        """Stops the inference coordinator, performing any necessary cleanup operations.

        Context termination is handled by `AsyncZmqEndpoint.shutdown()` via `_owns_ctx`
        since no external context is passed to this endpoint.
        """
        if self.is_shutdown:
            return
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
        self._shutdown_event.set()
