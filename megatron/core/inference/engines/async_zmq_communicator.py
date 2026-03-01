# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

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

    async def all_reduce_max(self, *local_vals: int) -> int | tuple[int, ...]:
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

        fmt = f'!{n}i'
        payload = struct.pack(fmt, *local_vals)

        if self.is_leader:
            rows = [local_vals]

            for _ in range(self.world_size - 1):
                msg = await self.gather_sock.recv()
                rows.append(struct.unpack(fmt, msg))

            maxes = tuple(max(row[i] for row in rows) for i in range(n))
            await self.bcast_sock.send(struct.pack(fmt, *maxes))
            return maxes[0] if n == 1 else maxes

        else:
            await self.gather_sock.send(payload)
            msg = await self.bcast_sock.recv()
            result = struct.unpack(fmt, msg)
            return result[0] if n == 1 else result

    def close(self):
        """
        Close the ZMQ sockets.
        """
        # linger=0: discard unsent messages immediately on close rather than blocking until sent.
        # The ZMQ default is to not allow `close` until all messages have been successfully sent.
        self.gather_sock.close(linger=0)
        self.bcast_sock.close(linger=0)
