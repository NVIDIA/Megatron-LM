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


class AsyncZMQCommunicator:
    """
    An asyncio-friendly communicator abstraction using ZMQ.
    Can be used to implement collective operations like all-reduce,
    and bcast which are asyncio friendly on top of ZMQ sockets.
    Only to be used with small amounts of data (e.g., 1 integer)
    on the CPU.
    """

    def __init__(self, zmq_context: zmq.Context, process_group: dist.ProcessGroup):
        """
        Constructor for AsyncZMQCommunicator. Sets up ZMQ sockets
        for communication among ranks in the given process group.
        Args:
            zmq_context (zmq.Context): ZMQ context to create sockets.
            process_group (dist.ProcessGroup): Process group for communication.
        """
        self.rank = dist.get_rank(process_group)
        self.world_size = dist.get_world_size(process_group)
        self.is_leader = self.rank == 0
        # Get the global rank of the leader (first rank in the process group)
        src_rank = dist.get_process_group_ranks(process_group)[0]

        if self.is_leader:
            local_ip = socket.gethostname()
            self.gather_sock = zmq_context.socket(zmq.PULL)
            self.gather_sock.bind_to_random_port(f"tcp://{local_ip}")
            gather_socket_addr = self.gather_sock.getsockopt_string(zmq.LAST_ENDPOINT)

            self.bcast_sock = zmq_context.socket(zmq.PUB)
            self.bcast_sock.bind_to_random_port(f"tcp://{local_ip}")
            bcast_socket_addr = self.bcast_sock.getsockopt_string(zmq.LAST_ENDPOINT)

            # Share the socket addresses with all peers
            dist.broadcast_object_list(
                [gather_socket_addr, bcast_socket_addr], src=src_rank, group=process_group
            )

        else:
            bcast_output = [None, None]
            dist.broadcast_object_list(bcast_output, src=src_rank, group=process_group)
            gather_socket_addr, bcast_socket_addr = bcast_output
            self.gather_sock = zmq_context.socket(zmq.PUSH)
            self.gather_sock.connect(gather_socket_addr)
            self.bcast_sock = zmq_context.socket(zmq.SUB)
            self.bcast_sock.connect(bcast_socket_addr)
            self.bcast_sock.setsockopt_string(zmq.SUBSCRIBE, "")

    async def all_reduce_max(self, local_val: int) -> int:
        """
        Asyncio friendly all reduce max operation. Gathers on rank 0, computes max,
        and broadcasts the result.
        """
        if self.world_size <= 1:
            return local_val

        payload = struct.pack('!i', local_val)

        if self.is_leader:
            # Rank 0: Gather -> Max -> Broadcast
            values = [local_val]

            # Non-blocking gather from N-1 peers
            while len(values) < self.world_size:
                try:
                    msg = self.gather_sock.recv(flags=zmq.NOBLOCK)
                    values.append(struct.unpack('!i', msg)[0])
                except zmq.Again:
                    await asyncio.sleep(0.001)  # Yield to event loop

            max_val = max(values)
            self.bcast_sock.send(struct.pack('!i', max_val))
            return max_val

        else:
            # Worker: Send -> Wait for Broadcast
            self.gather_sock.send(payload)

            while True:
                try:
                    msg = self.bcast_sock.recv(flags=zmq.NOBLOCK)
                    return struct.unpack('!i', msg)[0]
                except zmq.Again:
                    await asyncio.sleep(0.001)  # Yield to event loop

    async def all_reduce_max_pair(self, val_a: int, val_b: int) -> tuple[int, int]:
        """Element-wise all-reduce max of two integers.

        Packs both values into a single message so the communication cost
        is identical to a single-integer all-reduce.
        """
        if self.world_size <= 1:
            return val_a, val_b

        payload = struct.pack('!ii', val_a, val_b)

        if self.is_leader:
            pairs = [(val_a, val_b)]

            while len(pairs) < self.world_size:
                try:
                    msg = self.gather_sock.recv(flags=zmq.NOBLOCK)
                    pairs.append(struct.unpack('!ii', msg))
                except zmq.Again:
                    await asyncio.sleep(0.001)

            max_a = max(p[0] for p in pairs)
            max_b = max(p[1] for p in pairs)
            self.bcast_sock.send(struct.pack('!ii', max_a, max_b))
            return max_a, max_b

        else:
            self.gather_sock.send(payload)

            while True:
                try:
                    msg = self.bcast_sock.recv(flags=zmq.NOBLOCK)
                    return struct.unpack('!ii', msg)
                except zmq.Again:
                    await asyncio.sleep(0.001)

    def close(self):
        """
        Close the ZMQ sockets.
        """
        self.gather_sock.close(linger=0)
        self.bcast_sock.close(linger=0)
