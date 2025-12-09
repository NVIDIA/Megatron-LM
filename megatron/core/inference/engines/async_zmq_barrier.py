import asyncio
import zmq
import struct


class AsyncZMQBarrier:
    def __init__(
        self, 
        zmq_context: zmq.Context, 
        rank: int, 
        world_size: int, 
        leader_ip: str = "127.0.0.1", 
        gather_port: int = 15000,
        bcast_port: int = 15001
    ):
        """
        An async-friendly synchronization barrier using ZMQ.
        """
        self.rank = rank
        self.world_size = world_size
        self.is_leader = (rank == 0)
        
        if self.is_leader:
            self.gather_sock = zmq_context.socket(zmq.PULL)
            self.gather_sock.bind(f"tcp://*:{gather_port}")
            
            self.bcast_sock = zmq_context.socket(zmq.PUB)
            self.bcast_sock.bind(f"tcp://*:{bcast_port}")
        else:
            self.gather_sock = zmq_context.socket(zmq.PUSH)
            self.gather_sock.connect(f"tcp://{leader_ip}:{gather_port}")
            
            self.bcast_sock = zmq_context.socket(zmq.SUB)
            self.bcast_sock.connect(f"tcp://{leader_ip}:{bcast_port}")
            self.bcast_sock.setsockopt_string(zmq.SUBSCRIBE, "")

    async def all_reduce_max(self, local_val: int) -> int:
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
                    await asyncio.sleep(0.001) # Yield to event loop
            
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
                    await asyncio.sleep(0.001) # Yield to event loop

    def close(self):
        self.gather_sock.close()
        self.bcast_sock.close()