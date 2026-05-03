# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import errno
import faulthandler
import json
import logging
import signal
import socket
from collections import deque
from enum import Enum, auto
from multiprocessing import Event
from multiprocessing.connection import Connection
from typing import Any, Optional, Tuple

import numpy as np
import torch
from megatron.core.inference.config import PrefixCachingCoordinatorPolicy
from megatron.core.inference.headers import Headers
from megatron.core.inference.inference_request import compute_block_hashes_batched
from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    TextGenerationController,
)

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
if hasattr(faulthandler, "register"):
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

    Attributes:
        router_socket (zmq.Socket): The central ZMQ ROUTER socket for all communication.
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

        This sets up the ZMQ context and a ROUTER socket, binding it to the given
        port. It then enters a blocking loop to wait for all expected data parallel
        ranks to connect before proceeding.

        Args:
            pipe_connection (Connection): A connecting pipe to the parent process.
            data_parallel_size (int): The number of TP-coordinator workers that are
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
        while len(self.identities_of_data_parallel_ranks) < data_parallel_size:
            frames = self.router_socket.recv_multipart()
            if len(frames) != 2:
                self._log_protocol_error(
                    "client_error",
                    f"invalid registration frame count: expected 2, got {len(frames)}",
                    "ENGINE_REGISTRATION",
                )
                continue
            identity, _ = frames
            if identity in self.identities_of_data_parallel_ranks:
                self._log_protocol_error(
                    "client_error",
                    f"duplicate engine identity during registration: {identity}",
                    "ENGINE_REGISTRATION",
                )
                continue
            self.identities_of_data_parallel_ranks.append(identity)
        logging.info("Inference Coordinator: Connected with data parallel ranks...")

        # In deterministic mode, sort identities for consistent scheduling order.
        if deterministic_mode:
            self.identities_of_data_parallel_ranks = deque(
                sorted(self.identities_of_data_parallel_ranks)
            )
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

        # Deterministic rank index mapping (sorted identity -> 0-based index).
        sorted_identities = sorted(self.identities_of_data_parallel_ranks)
        self.identity_to_rank_index = {
            identity: idx for idx, identity in enumerate(sorted_identities)
        }

        # Numpy arrays for vectorized scoring (indexed by rank index).
        n_ranks = len(sorted_identities)
        self._identities_list = list(sorted_identities)  # rank_index → identity
        self._pending_counts = np.zeros(n_ranks, dtype=np.int32)

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
        """Remove a disconnected engine from the routing pool."""
        self.identities_of_data_parallel_ranks.remove(identity)
        logging.warning(
            "Coordinator: removed engine %s (now %d engines)",
            identity,
            len(self.identities_of_data_parallel_ranks),
        )

    def _send_to_engine(self, identity, payload):
        """Send payload to an engine, removing it from the pool if unreachable.

        Returns:
            True if the send succeeded, False if the engine was unreachable and removed.
        """
        try:
            self.router_socket.send_multipart([identity, payload])
            return True
        except zmq.error.ZMQError as e:
            if e.errno == zmq.EHOSTUNREACH:
                self._remove_engine(identity)
                return False
            raise

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

    @staticmethod
    def _message_type_from_payload(msg: Any) -> Optional[str]:
        """Best-effort extraction of message type for logs."""
        if not isinstance(msg, (list, tuple)) or len(msg) == 0:
            return None

        raw_type = msg[0]
        if isinstance(raw_type, Headers):
            return raw_type.name

        try:
            return Headers(raw_type).name
        except Exception:
            return str(raw_type)

    @staticmethod
    def _log_protocol_error(error_type: str, reason: str, message_type: Optional[str] = None):
        """Log concise structured protocol errors."""
        fields = {
            "error_type": error_type,
            "message_type": message_type,
            "reason": reason,
        }
        log_fn = logging.warning if error_type == "client_error" else logging.error
        log_fn("Coordinator protocol error: %s", fields)

    @staticmethod
    def validate_message(msg: Any) -> Tuple[bool, Optional[str]]:
        """Validate inbound coordinator payloads.

        Returns:
            (True, None) if valid.
            (False, reason) if invalid.
        """
        if msg is None:
            return False, "payload is None"
        if not isinstance(msg, (list, tuple)):
            return False, "payload must be a list or tuple"
        if len(msg) == 0:
            return False, "payload must be non-empty"

        raw_header = msg[0]
        try:
            header = raw_header if isinstance(raw_header, Headers) else Headers(raw_header)
        except Exception:
            return False, "unknown message type"

        if header == Headers.CONNECT:
            if len(msg) != 1:
                return False, "CONNECT payload must contain only header"
            return True, None

        if header == Headers.SUBMIT_REQUEST:
            if len(msg) != 4:
                return False, "SUBMIT_REQUEST payload must have 4 fields"
            if not isinstance(msg[1], int):
                return False, "SUBMIT_REQUEST request_id must be int"
            if not isinstance(msg[2], (str, list, torch.Tensor)):
                return False, "SUBMIT_REQUEST prompt must be str, list, or Tensor"
            if not isinstance(msg[3], dict):
                return False, "SUBMIT_REQUEST sampling_params must be dict"
            return True, None

        if header == Headers.ENGINE_REPLY:
            if len(msg) != 2:
                return False, "ENGINE_REPLY payload must have 2 fields"
            if not isinstance(msg[1], list):
                return False, "ENGINE_REPLY finished_requests must be list"
            return True, None

        if header in (
            Headers.PAUSE,
            Headers.UNPAUSE,
            Headers.SUSPEND,
            Headers.RESUME,
            Headers.STOP,
            Headers.DISCONNECT,
            Headers.SHUTDOWN,
        ):
            if len(msg) != 1:
                return False, f"{header.name} payload must contain only header"
            return True, None

        if header == Headers.SET_GENERATION_EPOCH:
            if len(msg) < 2:
                return False, "SET_GENERATION_EPOCH payload must include epoch value"
            if not isinstance(msg[1], int):
                return False, "SET_GENERATION_EPOCH epoch value must be int"
            return True, None
            # Allow for re-registration if connecting to a running coordinator.
            if serialized_payload == b"":
                if sender_identity not in self.identities_of_data_parallel_ranks:
                    self.identities_of_data_parallel_ranks.append(sender_identity)
                    self._register_rank_identity(sender_identity)
                continue

        return False, "unknown message type"

    def handle_message(self, msg: Any, sender_identity: bytes, known_clients: set[bytes]) -> bool:
        """Validate and safely process one inbound message.

        Returns:
            True if the coordinator should stop; otherwise False.
        """
        message_type = self._message_type_from_payload(msg)
        is_valid, reason = self.validate_message(msg)
        if not is_valid:
            self._log_protocol_error("client_error", reason or "invalid payload", message_type)
            return False

        try:
            return self._process_valid_message(msg, sender_identity, known_clients)
        except Exception as e:
            self._log_protocol_error("internal_error", str(e), message_type)
            return False

    def _process_valid_message(self, deserialized_payload, sender_identity, known_clients):
        """Process one already-validated payload."""
        header = deserialized_payload[0]
        if not isinstance(header, Headers):
            header = Headers(header)

        if header == Headers.CONNECT:
            if sender_identity in known_clients:
                self._log_protocol_error(
                    "client_error",
                    f"duplicate CONNECT from client {sender_identity}",
                    header.name,
                )
                return False

            known_clients.add(sender_identity)
            self.router_socket.send_multipart(
                [sender_identity, msgpack.packb([Headers.CONNECT_ACK.value], use_bin_type=True)]
            )
            return False

        if header == Headers.SUBMIT_REQUEST:
            # Message from a known client.
            if sender_identity not in known_clients:
                self._log_protocol_error(
                    "client_error",
                    f"SUBMIT_REQUEST from unknown client {sender_identity}",
                    header.name,
                )
                return False

            client_request_id = deserialized_payload[1]
            prompt = deserialized_payload[2]
            sampling_params = deserialized_payload[3]

            # Map client request_id to server request_id.
            request_id = self.next_request_id
            self.next_request_id += 1
            self.request_id_to_client_id[request_id] = sender_identity
            self.request_id_to_client_request_id[request_id] = client_request_id
            request_succeeded = False
            try:
                # Serialize prompt.
                try:
                    if isinstance(prompt, (str, list)):
                        pass
                    elif isinstance(prompt, torch.Tensor):
                        prompt = prompt.tolist()
                    else:
                        self._log_protocol_error(
                            "client_error",
                            f"unsupported prompt type: {type(prompt).__name__}",
                            header.name,
                        )
                        return False

                    if isinstance(prompt, list) and any(
                        not isinstance(token, int) or isinstance(token, bool) for token in prompt
                    ):
                        self._log_protocol_error(
                            "client_error",
                            "invalid SUBMIT_REQUEST prompt list contents",
                            header.name,
                        )
                        return False

                    payload = msgpack.packb(
                        [Headers.SUBMIT_REQUEST.value, request_id, prompt, sampling_params],
                        use_bin_type=True,
                    )
                except (TypeError, ValueError) as e:
                    self._log_protocol_error(
                        "client_error",
                        f"invalid SUBMIT_REQUEST payload: {e}",
                        header.name,
                    )
                    return False

                request_hashes = self.compute_request_hashes(prompt)

                if (
                    self.prefix_caching_coordinator_policy
                    == PrefixCachingCoordinatorPolicy.FIRST_PREFIX_BLOCK
                ):
                    request_hashes = request_hashes[:1]

                # Account for the fact that some engines may have died.
                for _ in range(len(self.identities_of_data_parallel_ranks)):
                    next_identity = self.get_best_data_parallel_rank(request_hashes)
                    if self._send_to_engine(next_identity, payload):
                        break
                else:
                    # If all engines have died, we are in an abnormal state, and must exit cleanly.
                    logging.error("Coordinator: no reachable engines for request %d", request_id)
                    return True

                self.request_id_to_rank[request_id] = next_identity
                self._pending_counts[self.identity_to_rank_index[next_identity]] += 1
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
                request_succeeded = True
                return False
            finally:
                if not request_succeeded:
                    self.request_id_to_client_id.pop(request_id, None)
                    self.request_id_to_client_request_id.pop(request_id, None)

        if header in (
            Headers.PAUSE,
            Headers.UNPAUSE,
            Headers.SUSPEND,
            Headers.RESUME,
            Headers.SET_GENERATION_EPOCH,
            Headers.STOP,
        ):
            # Start by checking the current state against the control signal.
            if sender_identity not in known_clients:
                self._log_protocol_error(
                    "client_error",
                    f"{header.name} signal from unknown client {sender_identity}",
                    header.name,
                )
                return False

            if header == Headers.PAUSE:
                idem_states = (self.CoordinatorState.PAUSED, self.CoordinatorState.SUSPENDED)
                if self.state == self.CoordinatorState.RUNNING:
                    self.state = self.CoordinatorState.PAUSED
                elif self.state in idem_states:
                    # Already paused/suspended, ignore redundant PAUSE.
                    return False
                else:
                    self._log_protocol_error(
                        "client_error",
                        f"ignoring PAUSE in state {self.state}",
                        header.name,
                    )
                    return False
            elif header == Headers.UNPAUSE:
                if self.state != self.CoordinatorState.PAUSED:
                    self._log_protocol_error(
                        "client_error",
                        f"ignoring UNPAUSE in state {self.state}",
                        header.name,
                    )
                    return False
                self.state = self.CoordinatorState.RUNNING
            elif header == Headers.SUSPEND:
                if self.state != self.CoordinatorState.PAUSED:
                    self._log_protocol_error(
                        "client_error",
                        f"ignoring SUSPEND in state {self.state}",
                        header.name,
                    )
                    return False
                self.state = self.CoordinatorState.SUSPENDED
            elif header == Headers.RESUME:
                if self.state != self.CoordinatorState.SUSPENDED:
                    self._log_protocol_error(
                        "client_error",
                        f"ignoring RESUME in state {self.state}",
                        header.name,
                    )
                    return False
                self.state = self.CoordinatorState.PAUSED
            elif header == Headers.STOP:
                good_states = (self.CoordinatorState.PAUSED, self.CoordinatorState.SUSPENDED)
                if self.state not in good_states:
                    self._log_protocol_error(
                        "client_error",
                        f"ignoring STOP in state {self.state}",
                        header.name,
                    )
                    return False
                self.state = self.CoordinatorState.STOPPING

            # Broadcast the control signal if we're in a good state.
            # Forward the full deserialized payload so that data-bearing
            # signals (e.g. SET_GENERATION_EPOCH) retain their arguments.
            broadcast_payload = msgpack.packb(deserialized_payload, use_bin_type=True)
            for data_parallel_rank_id in list(self.identities_of_data_parallel_ranks):
                self._send_to_engine(data_parallel_rank_id, broadcast_payload)

            # STOP affects engines; reset coordinator to RUNNING to allow future engines.
            if header == Headers.STOP:
                self.state = self.CoordinatorState.RUNNING
            return False

        if header == Headers.ENGINE_REPLY:
            # This is the output of a single engine step on some data parallel rank.
            if sender_identity not in self.identities_of_data_parallel_ranks:
                self._log_protocol_error(
                    "client_error", "ENGINE_REPLY from unknown engine identity", header.name
                )
                return False

            finished_requests = deserialized_payload[1]
            for finished_request in finished_requests:
                if not isinstance(finished_request, dict):
                    self._log_protocol_error(
                        "client_error", "ENGINE_REPLY item must be dict", header.name
                    )
                    continue

                fid = finished_request.get("request_id")
                if not isinstance(fid, int):
                    self._log_protocol_error(
                        "client_error", "ENGINE_REPLY item missing valid request_id", header.name
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

                # Broadcast the control signal if we're in a good state.
                # Forward the full deserialized payload so that data-bearing
                # signals (e.g. SET_GENERATION_EPOCH) retain their arguments.
                broadcast_payload = msgpack.packb(deserialized_payload, use_bin_type=True)
                for data_parallel_rank_id in list(self.identities_of_data_parallel_ranks):
                    self._send_to_engine(data_parallel_rank_id, broadcast_payload)

                # STOP affects engines; reset coordinator to RUNNING to allow future engines.
                if header == Headers.STOP:
                    self.state = self.CoordinatorState.RUNNING

            elif header == Headers.ENGINE_REPLY:
                # This is the output of a single engine step on some data parallel rank.
                assert sender_identity in self.identities_of_data_parallel_ranks
                finished_requests = deserialized_payload[1]

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

                    self.router_socket.send_multipart(
                        [
                            client_identity,
                            msgpack.packb(
                                [header.value, client_request_identity, finished_request],
                                use_bin_type=True,
                            ),
                        ]
                    )
                    continue

                client_identity = self.request_id_to_client_id.get(fid)
                client_request_identity = self.request_id_to_client_request_id.get(fid)
                if client_identity is None or client_request_identity is None:
                    self._log_protocol_error(
                        "client_error", f"unknown completed request_id: {fid}", header.name
                    )
                    continue

                try:
                    self.detokenize(finished_request)
                except Exception as e:
                    self._log_protocol_error(
                        "client_error",
                        f"failed to detokenize finished request: {e}",
                        header.name,
                    )
                    continue

                del self.request_id_to_client_id[fid]
                del self.request_id_to_client_request_id[fid]

                self.router_socket.send_multipart(
                    [
                        client_identity,
                        msgpack.packb(
                            [header.value, client_request_identity, finished_request],
                            use_bin_type=True,
                        ),
                    ]
                )
            return False

        if header == Headers.SHUTDOWN:
            if sender_identity not in known_clients:
                self._log_protocol_error(
                    "client_error",
                    f"SHUTDOWN signal from unknown client {sender_identity}",
                    header.name,
                )
                return False
            return True

        if header == Headers.DISCONNECT:
            if sender_identity in self.identities_of_data_parallel_ranks:
                self._remove_engine(sender_identity)
            return False

        self._log_protocol_error("client_error", "unknown message type", str(header))
        return False

    def start(self):
        """
        Starts the main event loop for the coordinator.

        This method runs an infinite loop, continuously listening for incoming
        messages on the ZMQ ROUTER socket. It parses the message header to
        determine the message type and takes appropriate action, such as
        handling new client connections, forwarding requests, broadcasting
        control signals, or processing replies from the engines.
        """
        known_clients = set()
        while True:
            frames = self.router_socket.recv_multipart()
            if len(frames) != 2:
                self._log_protocol_error(
                    "client_error",
                    f"invalid multipart frame count: expected 2, got {len(frames)}",
                )
                continue
            sender_identity, serialized_payload = frames

            # Allow for re-registration if connecting to a running coordinator.
            if serialized_payload == b"":
                if sender_identity not in self.identities_of_data_parallel_ranks:
                    self.identities_of_data_parallel_ranks.append(sender_identity)
                continue

            try:
                deserialized_payload = msgpack.unpackb(serialized_payload, raw=False)
            except Exception as e:
                self._log_protocol_error("client_error", f"failed to decode payload: {e}")
                continue

            if self.handle_message(deserialized_payload, sender_identity, known_clients):
                break

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

        This method initializes the coordinator, signals a `ready_event` to indicate
        that it is fully initialized and listening, and then starts the main event loop.

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
