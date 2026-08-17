# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import errno
import faulthandler
import json
import logging
import signal
import socket
import time
from collections import deque
from multiprocessing import Event
from multiprocessing.connection import Connection

import numpy as np
import torch

from megatron.core.inference.config import (
    PrefixCachingCoordinatorPolicy,
    PrefixCachingEvictionPolicy,
)
from megatron.core.inference.headers import Headers, UnknownHeaderError
from megatron.core.inference.inference_request import compute_block_hashes_batched
from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    TextGenerationController,
)

from .handlers import HANDLERS
from .state import CoordinatorState

try:
    import zmq

    HAVE_ZMQ = True
except:
    HAVE_ZMQ = False

try:
    import msgpack

    HAVE_MSGPACK = True
except:
    HAVE_MSGPACK = False

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
        to one of the available data parallel ranks using load-balanced (and,
        when prefix caching is enabled, prefix-affinity-aware) routing.
    4.  **Response Routing**: It receives completed results from
        the data parallel ranks and routes them back to the original client that made the
        request.
    5.  **Control Signal Broadcasting**: It relays control signals (e.g., PAUSE, STOP)
        from a client to all connected data parallel ranks.

    Message handling is split out into handlers.py: the event loop in start()
    dispatches each message to the handler registered for its header, so
    supporting a new message type requires no changes here.

    Attributes:
        router_socket (zmq.Socket): The central ZMQ ROUTER socket for all communication.
        data_parallel_size (int): The number of data parallel workers to expect.
        identities_of_data_parallel_ranks (deque): A deque holding the ZMQ
            identities of connected data parallel instances, used for request routing.
        request_id_to_client_id (dict): Maps server-side request IDs to the ZMQ
            identity of the client that initiated the request.
        request_id_to_client_request_id (dict): Maps server-side request IDs to the
            original request ID provided by the client.
        next_request_id (int): A counter for generating unique server-side request IDs.
    """

    # Exposed as a class attribute for backwards compatibility; the canonical
    # definition lives in state.py.
    CoordinatorState = CoordinatorState

    def __init__(
        self,
        pipe_connection: Connection,
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
        prefix_caching_eviction_policy: PrefixCachingEvictionPolicy = (
            PrefixCachingEvictionPolicy.LRU
        ),
        prefix_cache_ttl_seconds: float = 300.0,
        schedule_output_path: str | None = None,
        hostname: str | None = None,
    ):
        """
        Initializes the inference coordinator.

        This sets up the ZMQ context and a ROUTER socket, binding it to the given
        port. It then enters a blocking loop to wait for all expected data parallel
        ranks to connect before proceeding.

        Args:
            pipe_connection (Connection): A connecting pipe to the parent process.
            data_parallel_size (int): The number of data parallel instances that are
                expected to connect.
            tokenizer: The tokenizer to use for prompt tokenization and detokenization.
            inference_coordinator_port (Optional[int]): The TCP port number to bind the server to.
            prefix_caching_routing_alpha (float): Weight for prefix-aware routing score:
                score = alpha * match + (1 - alpha) * normalized_load.
            max_requests (int): Max concurrent requests per rank, used to
                compute normalized_load for prefix-aware scoring.
        """
        assert HAVE_ZMQ, (
            "please install the pyzmq library to use DataParallelInferenceCoordinator\n"
            "pip install pyzmq"
        )
        assert HAVE_MSGPACK, (
            "please install the messagepack library to use DataParallelInferenceCoordinator\n"
            "pip install msgpack"
        )
        self.pipe_connection = pipe_connection
        self.data_parallel_size = data_parallel_size
        self.context = zmq.Context()

        # This is the central router socket
        # 1. data parallel ranks connect to this socket to register themselves
        # 2. Users connect to this socket and submit their requests. We transmit them to
        #    data parallel ranks in a round robin fashion
        # 3. data parallel ranks return completed requests to this socket. We route them back to
        #    the user that had submitted the request originally.

        # Get local IP.
        local_ip = hostname or socket.gethostname()

        self.router_socket = self.context.socket(zmq.ROUTER)
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

        logging.info("Inference Coordinator: waiting for connections from data parallel ranks...")
        # First wait for all data parallel ranks to establish connections.
        self.identities_of_data_parallel_ranks = deque([])
        # time.sleep(5)  # Give data parallel ranks time to spawn and connect.
        for _ in range(data_parallel_size):
            identity, _ = self.router_socket.recv_multipart()
            assert identity not in self.identities_of_data_parallel_ranks
            self.identities_of_data_parallel_ranks.append(identity)
        logging.info("Inference Coordinator: Connected with data parallel ranks...")

        # In deterministic mode, sort identities for consistent scheduling order.
        if deterministic_mode:
            self.identities_of_data_parallel_ranks = deque(
                sorted(self.identities_of_data_parallel_ranks)
            )

        self.request_id_to_client_id = {}
        self.request_id_to_client_request_id = {}
        self.request_id_to_rank = {}  # Maps request_id → rank identity for pending count tracking
        self.removed_engine_identities = set()
        self.client_request_to_request_id = {}

        self.next_request_id = 0
        self.tokenizer = tokenizer
        self.state = CoordinatorState.RUNNING

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

        # Deterministic rank index mapping (sorted identity -> 0-based index).
        sorted_identities = sorted(self.identities_of_data_parallel_ranks)
        self.identity_to_rank_index = {
            identity: idx for idx, identity in enumerate(sorted_identities)
        }

        # Numpy arrays for vectorized scoring (indexed by rank index).
        n_ranks = len(sorted_identities)
        self._identities_list = list(sorted_identities)  # rank_index → identity
        self._pending_counts = np.zeros(n_ranks, dtype=np.int32)

        # Hash → {rank_idx: [last_touch, refcount]} for prefix cache affinity
        # routing. This models what each engine holds; it is a prediction, since
        # the engines are never asked. Entries are retired the way the engine
        # retires blocks, per prefix_caching_eviction_policy:
        #   REF_ZERO -- the engine deregisters at refcount 0, so drop the entry
        #               when the last request holding it completes.
        #   LRU      -- the engine keeps released blocks and evicts under memory
        #               pressure, which the coordinator cannot observe, so
        #               approximate it by age: drop entries untouched for
        #               prefix_cache_ttl_seconds.
        # Without either, the table only grows and grows steadily more optimistic,
        # claiming hits on blocks the engine evicted long ago.
        self._hash_table: dict[int, dict[int, list]] = {}
        # (touch_time, hash, rank_idx) in insertion order, so TTL expiry is a
        # sweep from the front costing only what it actually evicts. An entry
        # refreshed since it was queued is skipped by comparing timestamps.
        self._hash_expiry: deque = deque()
        # request_id → hashes, kept only under REF_ZERO, where completion is what
        # releases the blocks.
        self.request_id_to_hashes: dict[int, list] = {}
        self.prefix_caching_eviction_policy = prefix_caching_eviction_policy
        self.prefix_cache_ttl_seconds = prefix_cache_ttl_seconds

        # Clients that have completed the CONNECT handshake.
        self.known_clients = set()

        # Header -> handler dispatch table, sourced from the handler registry.
        self._handlers = dict(HANDLERS)

    def get_least_loaded_data_parallel_rank(self):
        """
        Selects the data parallel rank with the fewest in-flight requests.

        Ties are broken by lowest rank index for deterministic behavior.

        Returns:
            bytes: The ZMQ identity of the least-loaded data parallel rank.
        """
        if not self._identities_list:
            raise RuntimeError("No engines connected")
        best_idx = int(np.argmin(self._pending_counts))
        return self._identities_list[best_idx]

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
        """Remove a disconnected engine from all routing bookkeeping.
        Called both during shutdown and when an engine becomes unreachable mid-operation
        (e.g. zmq.EHOSTUNREACH in _send_to_engine). The O(n) index-shifting and hash-table
        rebuild are acceptable because the number of connected engines is small; optimize
        only if dynamic registration/deregistration at high engine counts becomes a use case.
        """
        self.identities_of_data_parallel_ranks.remove(identity)
        self.removed_engine_identities.add(identity)
        idx = self.identity_to_rank_index.pop(identity, None)
        if idx is None:
            return
        self._identities_list.pop(idx)
        self._pending_counts = np.delete(self._pending_counts, idx)
        # Shift indices for engines that came after the removed slot.
        for ident in self.identity_to_rank_index:
            if self.identity_to_rank_index[ident] > idx:
                self.identity_to_rank_index[ident] -= 1
        # Drop hash-table entries for the removed rank; shift indices above it.
        new_hash_table = {}
        for h, rank_ts in self._hash_table.items():
            # h is hash index
            # rank_ts is a dict mapping rank_idx → timestamp
            new_row = {}
            for r, ts in rank_ts.items():
                if r == idx:
                    # skip this rank as it is removed
                    continue
                new_r = r - 1 if r > idx else r
                new_row[new_r] = ts
            if new_row:
                new_hash_table[h] = new_row
        self._hash_table = new_hash_table
        logging.warning(
            "Coordinator: removed engine %s (now %d engines)",
            identity,
            len(self.identities_of_data_parallel_ranks),
        )

    def _send_to_engine(self, identity, frames):
        """Send a message to an engine, removing it from the pool if unreachable.

        Args:
            identity: ZMQ identity of the target engine.
            frames (list): Raw frames to send, metadata frame first.

        Returns:
            True if the send succeeded, False if the engine was unreachable and removed.
        """
        try:
            self.router_socket.send_multipart([identity, *frames])
            return True
        except zmq.error.ZMQError as e:
            if e.errno == zmq.EHOSTUNREACH:
                self._remove_engine(identity)
                return False
            raise

    def _broadcast_to_engines(self, payload):
        """Send a deserialized payload to every connected data parallel rank."""
        serialized = msgpack.packb(payload, use_bin_type=True)
        for data_parallel_rank_id in list(self.identities_of_data_parallel_ranks):
            self._send_to_engine(data_parallel_rank_id, [serialized])

    def compute_request_hashes(self, prompt):
        """Compute block hashes for a prompt on CPU.

        Callers decide whether hashes are wanted at all: computing them requires
        the decoded prompt, and decoding it is the cost the caller is usually
        trying to avoid. See handle_submit_request.

        Args:
            prompt: Either a string (to be tokenized) or a list of token IDs.

        Returns:
            List of integer block hashes, one per complete block.
        """
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
        if self.prefix_caching_coordinator_policy == PrefixCachingCoordinatorPolicy.LOAD_BALANCED:
            return self.get_least_loaded_data_parallel_rank()

        # Without prefix caching (or when the request has no hashes to match on)
        # fall back to load-balanced routing.
        if not self.enable_prefix_caching or not request_hashes:
            return self.get_least_loaded_data_parallel_rank()

        if self.prefix_caching_coordinator_policy == PrefixCachingCoordinatorPolicy.LONGEST_PREFIX:
            # Cost of sending here: the prefill this rank would still have to
            # compute, scaled by how contended it is. Multiplicative rather than a
            # weighted sum because prefill saved on a rank that cannot start it
            # promptly is not saved.
            #
            # 1 + load, so an idle rank is charged for the prefill it would still
            # have to do rather than scoring zero on any prefix. Reading it as
            # queue position: this request is the (load + 1)-th on that rank, so
            # the product approximates when its prefill would finish. Affinity
            # therefore decides from the first request, including while every rank
            # is idle.
            depth = self._prefix_depth_vector(request_hashes)
            remaining_blocks = np.maximum(0, len(request_hashes) - depth)
            cost = remaining_blocks.astype(np.float64) * (1.0 + self._pending_counts)
            # Tiebreak: lowest cost, then least loaded, then lowest rank index.
            n_ranks = len(self._identities_list)
            order = np.lexsort((np.arange(n_ranks), self._pending_counts, cost))
            return self._identities_list[int(order[0])]

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
        """Record that a rank owns the given hashes, and take a reference to them.

        Args:
            rank_identity: ZMQ identity of the target rank.
            request_hashes: List of block hashes assigned to this rank.
        """
        rank_idx = self.identity_to_rank_index[rank_identity]
        now = time.monotonic()
        for h in request_hashes:
            entry = self._hash_table.setdefault(h, {}).get(rank_idx)
            if entry is None:
                self._hash_table[h][rank_idx] = [now, 1]
            else:
                entry[0] = now
                entry[1] += 1
            # One timestamp for the whole call keeps the deque sorted.
            self._hash_expiry.append((now, h, rank_idx))

    def _release_rank_hashes(self, request_id, rank_identity):
        """Drop the references a finished request held, under REF_ZERO.

        Mirrors the engine deregistering blocks once no request holds them. Blocks
        shared with a still-running request keep a nonzero count and survive.

        Args:
            request_id: Server-side request id that just completed.
            rank_identity: ZMQ identity of the rank that served it.
        """
        hashes = self.request_id_to_hashes.pop(request_id, None)
        if not hashes:
            return
        rank_idx = self.identity_to_rank_index.get(rank_identity)
        if rank_idx is None:  # engine already removed; its column is gone
            return
        for h in hashes:
            row = self._hash_table.get(h)
            entry = row.get(rank_idx) if row is not None else None
            if entry is None:
                continue
            entry[1] -= 1
            if entry[1] <= 0:
                del row[rank_idx]
                if not row:
                    del self._hash_table[h]

    def _expire_rank_hashes(self, now):
        """Drop entries untouched for the TTL, under LRU.

        The deque is insertion-ordered, so expired entries form a prefix of it and
        the sweep stops at the first live one -- the cost is what it evicts, not
        the table size. A stale queue entry whose table entry was refreshed since
        is recognized by its older timestamp and skipped.
        """
        ttl = self.prefix_cache_ttl_seconds
        while self._hash_expiry:
            ts, h, rank_idx = self._hash_expiry[0]
            if now - ts <= ttl:
                break
            self._hash_expiry.popleft()
            row = self._hash_table.get(h)
            entry = row.get(rank_idx) if row is not None else None
            if entry is not None and entry[0] <= ts:
                del row[rank_idx]
                if not row:
                    del self._hash_table[h]

    def _prefix_depth_vector(self, hashes):
        """Return per-rank contiguous prefix depth, in blocks.

        Prefix cache reuse requires an unbroken chain from the first block -- each
        block hash chains the previous one's digest -- so a rank only benefits up
        to its first miss. Walking forward and dropping ranks as they miss gives
        each rank its true depth, and the loop exits as soon as no rank is left.
        """
        n_ranks = len(self._identities_list)
        depth = np.zeros(n_ranks, dtype=np.int64)
        alive = np.ones(n_ranks, dtype=bool)
        for h in hashes:
            row = self._hash_table.get(h)
            if row is None:
                break
            present = np.zeros(n_ranks, dtype=bool)
            present[np.fromiter(row.keys(), dtype=np.intp, count=len(row))] = True
            alive &= present
            if not alive.any():
                break
            depth += alive
        return depth

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
            # Values are [last_touch, refcount]; recency is the touch time.
            recency[rank_idxs] = np.fromiter(
                (e[0] for e in row.values()), dtype=np.float64, count=len(row)
            )
            if present.any():
                return present.astype(np.float64) * ((i + 1.0) / n), recency
        return zeros, zeros.copy()

    def start(self):
        """
        Starts the main event loop for the coordinator.

        This method runs an infinite loop, continuously listening for incoming
        messages on the ZMQ ROUTER socket. It reads the message header and
        dispatches to the handler registered for it (see handlers.py).
        A handler that returns a truthy value stops the loop.
        """
        # Todo [Siddharth]: Make this more robust to handle invalid messages.
        while True:
            # Messages are one or more frames. frames[0] is the metadata frame:
            # a header plus whatever the coordinator needs to route the message.
            # Any later frames are opaque payload bodies, forwarded without ever
            # being decoded here -- that is what keeps this loop's cost
            # independent of prompt length.
            sender_identity, *frames = self.router_socket.recv_multipart()

            # An empty payload is a data parallel rank (re-)registering itself.
            if frames[0] == b"":
                self._handle_rank_registration(sender_identity)
                continue

            metadata = msgpack.unpackb(frames[0], raw=False)
            header = Headers(metadata[0])

            handler = self._handlers.get(header)
            if handler is None:
                raise UnknownHeaderError(header)
            if handler(self, sender_identity, metadata, frames[1:]):
                break

    def _handle_rank_registration(self, sender_identity):
        """Register a data parallel rank that connected to a running coordinator."""
        if sender_identity not in self.identities_of_data_parallel_ranks:
            self.identities_of_data_parallel_ranks.append(sender_identity)
            self._register_rank_identity(sender_identity)

    def detokenize(self, finished_request):
        """
        Detokenizes the generated tokens in the finished request.

        This method uses the coordinator's tokenizer to convert the list of
        generated token IDs back into human-readable text.

        Args:
            finished_request (dict): The serialized merged request containing the
                generated tokens to be detokenized. It is modified in place.
        """
        if not finished_request["sampling_params"]["detokenize_generations"]:
            return

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
        prefix_caching_eviction_policy: PrefixCachingEvictionPolicy = (
            PrefixCachingEvictionPolicy.LRU
        ),
        prefix_cache_ttl_seconds: float = 300.0,
        schedule_output_path: str | None = None,
        hostname: str | None = None,
    ):
        """
        Class method to instantiate and run the coordinator, for use in a separate process.

        This method initializes the coordinator, signals a `ready_event` to indicate
        that it is fully initialized and listening, and then starts the main event loop.

        Args:
            pipe_connection (Connection): A connecting pipe to the parent process.
            ready_event (Event): A threading or multiprocessing event object that is set()
                once the coordinator is ready to accept connections.
            inference_coordinator_port (int): The port to bind to.
            data_parallel_size (int): The number of expected data parallel instances.
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
            data_parallel_size,
            tokenizer,
            max_requests,
            inference_coordinator_port,
            deterministic_mode=deterministic_mode,
            block_size_tokens=block_size_tokens,
            enable_prefix_caching=enable_prefix_caching,
            prefix_caching_coordinator_policy=prefix_caching_coordinator_policy,
            prefix_caching_routing_alpha=prefix_caching_routing_alpha,
            prefix_caching_eviction_policy=prefix_caching_eviction_policy,
            prefix_cache_ttl_seconds=prefix_cache_ttl_seconds,
            schedule_output_path=schedule_output_path,
            hostname=hostname,
        )
        ready_event.set()
        try:
            coordinator.start()
        except KeyboardInterrupt:
            logging.info("Coordinator process interrupted. Exiting...")
        coordinator.stop()
        logging.info("Inference Coordinator: shut down successfully.")

    def stop(self):
        """
        Stops the inference coordinator, performing any necessary cleanup operations.
        """
        if self.schedule_output_path and self.schedule_records:
            schedule_data = {
                "policy": self.prefix_caching_coordinator_policy.value,
                "data_parallel_size": self.data_parallel_size,
                "num_requests": len(self.schedule_records),
                "records": self.schedule_records,
            }
            with open(self.schedule_output_path, "w") as f:
                json.dump(schedule_data, f, indent=2)
        self.router_socket.close()
        self.context.term()
