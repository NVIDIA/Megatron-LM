# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import asyncio
import socket
import struct

import torch.distributed as dist

try:
    import zmq

    HAVE_ZMQ = True
except ImportError:
    from unittest.mock import MagicMock

    zmq = MagicMock()
    HAVE_ZMQ = False


_SETUP_MAGIC = b"AZQS"
_DATA_MAGIC = b"AZQD"
_ERROR_MAGIC = b"AZQE"


class ZMQCollectiveError(RuntimeError):
    """Raised when a tagged ZMQ collective observes a protocol mismatch."""


class AsyncZMQCommunicator:
    """
    An asyncio-friendly communicator abstraction using ZMQ.
    Can be used to implement collective operations like all-reduce,
    and bcast which are asyncio friendly on top of ZMQ sockets.
    Only to be used with small amounts of data (e.g., 1 integer)
    on the CPU.
    """

    def __init__(
        self,
        zmq_context: zmq.Context,
        process_group: dist.ProcessGroup,
        hostname: str | None = None,
    ):
        """
        Constructor for AsyncZMQCommunicator. Sets up ZMQ sockets
        for communication among ranks in the given process group.
        Args:
            zmq_context (zmq.Context): ZMQ context to create sockets.
            process_group (dist.ProcessGroup): Process group for communication.
            hostname (str | None): Hostname or IP address to use for ZMQ socket binding.
                If None, defaults to socket.gethostname().
        """
        self.rank = dist.get_rank(process_group)
        self.world_size = dist.get_world_size(process_group)
        self.is_leader = self.rank == 0
        # Get the global rank of the leader (first rank in the process group)
        src_rank = dist.get_process_group_ranks(process_group)[0]
        self._async_collective_step = 0
        self._sync_collective_step = 0
        self.protocol_mismatch_count = 0

        if self.is_leader:
            local_ip = hostname or socket.gethostname()
            self.gather_sock = zmq_context.socket(zmq.PULL)
            self.gather_sock.bind_to_random_port(f"tcp://{local_ip}")
            gather_socket_addr = self.gather_sock.getsockopt_string(zmq.LAST_ENDPOINT)

            # Share the socket addresses with all peers
            dist.broadcast_object_list([gather_socket_addr], src=src_rank, group=process_group)

            self.result_push_socks = {}
            while len(self.result_push_socks) < self.world_size - 1:
                peer_rank, result_socket_addr = self._unpack_setup_message(self.gather_sock.recv())
                result_push_sock = zmq_context.socket(zmq.PUSH)
                result_push_sock.connect(result_socket_addr)
                self.result_push_socks[peer_rank] = result_push_sock

        else:
            bcast_output = [None]
            dist.broadcast_object_list(bcast_output, src=src_rank, group=process_group)
            [gather_socket_addr] = bcast_output
            self.gather_sock = zmq_context.socket(zmq.PUSH)
            self.gather_sock.connect(gather_socket_addr)
            local_ip = hostname or socket.gethostname()
            self.result_recv_sock = zmq_context.socket(zmq.PULL)
            self.result_recv_sock.bind_to_random_port(f"tcp://{local_ip}")
            result_socket_addr = self.result_recv_sock.getsockopt_string(zmq.LAST_ENDPOINT)
            self.gather_sock.send(self._pack_setup_message(self.rank, result_socket_addr))

        # Prevent a fast peer from sending the first collective before the leader has collected
        # all per-rank result sockets. ZMQ preserves per-peer ordering, not global ordering.
        dist.barrier(group=process_group)

    @staticmethod
    def _pack_setup_message(rank: int, socket_addr: str) -> bytes:
        addr = socket_addr.encode("utf-8")
        return struct.pack(f"!4siH{len(addr)}s", _SETUP_MAGIC, rank, len(addr), addr)

    @staticmethod
    def _unpack_setup_message(msg: bytes) -> tuple[int, str]:
        magic, rank, addr_len = struct.unpack("!4siH", msg[:10])
        if magic != _SETUP_MAGIC:
            raise ZMQCollectiveError("Unexpected ZMQ communicator setup message")
        addr = struct.unpack(f"!{addr_len}s", msg[10 : 10 + addr_len])[0].decode("utf-8")
        return rank, addr

    @staticmethod
    def _pack_values_message(phase: str, step_id: int, values: tuple[int, ...]) -> bytes:
        phase_bytes = phase.encode("utf-8")
        n = len(values)
        return struct.pack(
            f"!4sHqH{len(phase_bytes)}s{n}i",
            _DATA_MAGIC,
            len(phase_bytes),
            step_id,
            n,
            phase_bytes,
            *values,
        )

    @staticmethod
    def _unpack_values_message(
        msg: bytes, *, expected_phase: str, expected_step_id: int, expected_count: int
    ) -> tuple[int, ...]:
        if len(msg) < 16:
            raise ZMQCollectiveError("Malformed ZMQ collective message")

        magic = msg[:4]
        if magic == _ERROR_MAGIC:
            raise ZMQCollectiveError(AsyncZMQCommunicator._unpack_error_message(msg))
        if magic != _DATA_MAGIC:
            raise ZMQCollectiveError("Unexpected ZMQ collective message type")

        _, phase_len, step_id, value_count = struct.unpack("!4sHqH", msg[:16])
        phase_start = 16
        phase_end = phase_start + phase_len
        phase = struct.unpack(f"!{phase_len}s", msg[phase_start:phase_end])[0].decode("utf-8")
        if phase != expected_phase:
            raise ZMQCollectiveError(
                f"ZMQ collective phase mismatch: expected {expected_phase}, got {phase}"
            )
        if step_id != expected_step_id:
            raise ZMQCollectiveError(
                f"ZMQ collective step mismatch for {phase}: expected {expected_step_id}, got {step_id}"
            )
        if value_count != expected_count:
            raise ZMQCollectiveError(
                f"ZMQ collective value_count mismatch for {phase} step {step_id}: "
                f"expected {expected_count}, got {value_count}"
            )

        values_start = phase_end
        values_end = values_start + (4 * value_count)
        if len(msg) != values_end:
            raise ZMQCollectiveError(
                f"Malformed ZMQ collective payload for {phase} step {step_id}: "
                f"expected {values_end} bytes, got {len(msg)}"
            )
        return struct.unpack(f"!{value_count}i", msg[values_start:values_end])

    def _unpack_collective_values_message(
        self, msg: bytes, *, expected_phase: str, expected_step_id: int, expected_count: int
    ) -> tuple[int, ...]:
        try:
            return self._unpack_values_message(
                msg,
                expected_phase=expected_phase,
                expected_step_id=expected_step_id,
                expected_count=expected_count,
            )
        except ZMQCollectiveError:
            self.protocol_mismatch_count += 1
            raise

    @staticmethod
    def _pack_error_message(message: str) -> bytes:
        encoded = message.encode("utf-8")
        return struct.pack(f"!4sI{len(encoded)}s", _ERROR_MAGIC, len(encoded), encoded)

    @staticmethod
    def _unpack_error_message(msg: bytes) -> str:
        _, message_len = struct.unpack("!4sI", msg[:8])
        return struct.unpack(f"!{message_len}s", msg[8 : 8 + message_len])[0].decode("utf-8")

    def _next_collective_tag(
        self, *, phase: str | None, step_id: int | None, sync: bool
    ) -> tuple[str, int]:
        if phase is None:
            phase = "sync_all_reduce_max" if sync else "all_reduce_max"
        if step_id is None:
            if sync:
                step_id = self._sync_collective_step
                self._sync_collective_step += 1
            else:
                step_id = self._async_collective_step
                self._async_collective_step += 1
        return phase, step_id

    def _send_result_to_peers(self, msg: bytes) -> None:
        for peer_rank in sorted(self.result_push_socks):
            self.result_push_socks[peer_rank].send(msg)

    def _send_error_to_peers(self, error: Exception) -> None:
        self._send_result_to_peers(self._pack_error_message(str(error)))

    async def all_reduce_max(
        self,
        *local_vals: int,
        async_op=True,
        phase: str | None = None,
        step_id: int | None = None,
    ) -> int | tuple[int, ...]:
        """Element-wise all-reduce max of one or more integers.

        Packs all values into a single message so the communication cost
        is independent of the number of values.

        Returns a single int when called with one argument, otherwise a tuple.
        """
        n = len(local_vals)
        if n == 0:
            raise ValueError("all_reduce_max requires at least one value")

        if self.world_size <= 1:
            return local_vals[0] if n == 1 else local_vals

        phase, step_id = self._next_collective_tag(phase=phase, step_id=step_id, sync=False)
        payload = self._pack_values_message(phase, step_id, local_vals)

        if self.is_leader:
            rows = [local_vals]

            while len(rows) < self.world_size:
                try:
                    if async_op:
                        msg = self.gather_sock.recv(flags=zmq.NOBLOCK)
                    else:
                        msg = self.gather_sock.recv()
                    rows.append(
                        self._unpack_collective_values_message(
                            msg,
                            expected_phase=phase,
                            expected_step_id=step_id,
                            expected_count=n,
                        )
                    )
                except zmq.Again:
                    await asyncio.sleep(0.001)
                except Exception as error:
                    self._send_error_to_peers(error)
                    raise

            maxes = tuple(max(row[i] for row in rows) for i in range(n))
            self._send_result_to_peers(self._pack_values_message(phase, step_id, maxes))
            if not async_op:
                await asyncio.sleep(
                    0
                )  # Yield control once to ensure that other coroutines can run.
                # This might be needed for colocated RL.
            return maxes[0] if n == 1 else maxes

        else:
            self.gather_sock.send(payload)

            while True:
                try:
                    if async_op:
                        msg = self.result_recv_sock.recv(flags=zmq.NOBLOCK)
                    else:
                        msg = self.result_recv_sock.recv()
                    result = self._unpack_collective_values_message(
                        msg,
                        expected_phase=phase,
                        expected_step_id=step_id,
                        expected_count=n,
                    )
                    if not async_op:
                        await asyncio.sleep(
                            0
                        )  # Yield control once to ensure that other coroutines can run.
                        # This might be needed for colocated RL.
                    return result[0] if n == 1 else result
                except zmq.Again:
                    await asyncio.sleep(0.001)

    def sync_all_reduce_max(
        self, *local_vals: int, phase: str | None = None, step_id: int | None = None
    ) -> int | tuple[int, ...]:
        """Synchronous (non-asyncio) variant of all_reduce_max.

        Uses blocking ZMQ sends/recvs so it can be called from synchronous
        call sites that need a CPU-only MAX reduction across the process
        group. Intended for tiny payloads (e.g. a few integers) that would
        otherwise force a NCCL AllReduce kernel on the compute stream.

        Note: when called from inside a running asyncio event loop, the
        blocking recv will pause other coroutines on this rank until all
        peers respond. This is acceptable here because every rank reaches
        the call simultaneously and the message size is trivial.

        Returns a single int when called with one argument, otherwise a tuple.
        """
        n = len(local_vals)
        if n == 0:
            raise ValueError("sync_all_reduce_max requires at least one value")

        if self.world_size <= 1:
            return local_vals[0] if n == 1 else local_vals

        phase, step_id = self._next_collective_tag(phase=phase, step_id=step_id, sync=True)
        payload = self._pack_values_message(phase, step_id, local_vals)

        if self.is_leader:
            rows = [local_vals]
            try:
                while len(rows) < self.world_size:
                    msg = self.gather_sock.recv()
                    rows.append(
                        self._unpack_collective_values_message(
                            msg,
                            expected_phase=phase,
                            expected_step_id=step_id,
                            expected_count=n,
                        )
                    )
            except Exception as error:
                self._send_error_to_peers(error)
                raise
            maxes = tuple(max(row[i] for row in rows) for i in range(n))
            self._send_result_to_peers(self._pack_values_message(phase, step_id, maxes))
            return maxes[0] if n == 1 else maxes
        else:
            self.gather_sock.send(payload)
            msg = self.result_recv_sock.recv()
            result = self._unpack_collective_values_message(
                msg,
                expected_phase=phase,
                expected_step_id=step_id,
                expected_count=n,
            )
            return result[0] if n == 1 else result

    def close(self):
        """
        Close the ZMQ sockets.
        """
        # linger=0: discard unsent messages immediately on close rather than blocking until sent.
        # The ZMQ default is to not allow `close` until all messages have been successfully sent.
        self.gather_sock.close(linger=0)
        if self.is_leader:
            for sock in self.result_push_socks.values():
                sock.close(linger=0)
        else:
            self.result_recv_sock.close(linger=0)
