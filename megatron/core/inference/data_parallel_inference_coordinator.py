# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import asyncio
import faulthandler
import logging
import signal
from collections import deque
from itertools import cycle
from multiprocessing import Event
from typing import List, Optional, Tuple

import torch

from megatron.core.inference.headers import Headers
from megatron.core.utils import get_asyncio_loop, trace_async_exceptions

try:
    import zmq
    import zmq.asyncio

    HAVE_ZMQ = True
except:
    HAVE_ZMQ = False

try:
    import msgpack

    HAVE_MSGPACK = True
except:
    HAVE_MSGPACK = False


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

    def __init__(
        self, ready_event: Event, inference_coordinator_port: int, data_parallel_size: int
    ):
        """
        Initializes the inference coordinator.

        This sets up the ZMQ context and a ROUTER socket, binding it to the given
        port. It then enters a blocking loop to wait for all expected data parallel
        ranks to connect before proceeding.

        Args:
            ready_event (Event): A threading or multiprocessing event object that is set()
                once the coordinator is ready to accept connections.
            inference_coordinator_port (int): The TCP port number to bind the server to.
            data_parallel_size (int): The number of TP-coordinator workers that are
                expected to connect.
        """
        assert HAVE_ZMQ, (
            "please install the pyzmq library to use DataParallelInferenceCoordinator\n"
            "pip install pyzmq"
        )
        assert HAVE_MSGPACK, (
            "please install the messagepack library to use DataParallelInferenceCoordinator\n"
            "pip install msgpack"
        )
        self.ready_event = ready_event
        self.data_parallel_size = data_parallel_size

        self.context = zmq.asyncio.Context.instance()

        # This is the central router socket
        # 1. data parallel ranks connect to this socket to register themselves
        # 2. Users connect to this socket and submit their requests. We transmit them to
        #    data parallel ranks in a round robin fashion
        # 3. data parallel ranks return completed requests to this socket. We route them back to
        #    the user that had submitted the request originally.

        self.router_socket = self.context.socket(zmq.ROUTER)
        self.socket_uses_identity = True
        self.router_socket.bind(f"tcp://0.0.0.0:{inference_coordinator_port}")

    def get_next_data_parallel_rank(self):
        """
        Selects the next data parallel rank using round-robin scheduling.

        Returns:
            bytes: The ZMQ identity of the next data parallel rank to receive a request.
        """
        return next(self.data_parallel_rank_iterator)

    def start(self, loop: Optional[asyncio.AbstractEventLoop] = None):
        """
        Starts the main event loop for the coordinator.

        This method runs an infinite loop, continuously listening for incoming
        messages on the ZMQ ROUTER socket. It parses the message header to
        determine the message type and takes appropriate action, such as
        handling new client connections, forwarding requests, broadcasting
        control signals, or processing replies from the engines.
        """
        logging.info("Inference Coordinator: waiting for connections from data parallel ranks...")
        loop = get_asyncio_loop(loop)

        self.ready_event.clear()
        self.identities_of_data_parallel_ranks = deque([])
        self.data_parallel_rank_iterator = cycle([])
        self.known_clients = set()
        self.request_id_to_client_id = {}
        self.request_id_to_client_request_id = {}
        self.next_request_id = 0

        self._send_awaitables = asyncio.Queue()
        self.send_task = loop.create_task(self._send_task())
        self.recv_task = loop.create_task(self._recv_task())

    @trace_async_exceptions
    async def _recv_task(self):
        """Main loop of the inference coordinator."""

        print("Inference Coordinator: waiting for connections from data parallel ranks...")
        # First wait for all data parallel ranks to establish connections.
        for _ in range(self.data_parallel_size):
            identity, header, _ = await self._irecv()
            assert header == Headers.CONNECT
            assert identity not in self.identities_of_data_parallel_ranks
            self.identities_of_data_parallel_ranks.append(identity)
            print(f"Inference Coordinator: Data parallel rank connected: {identity}")
        print("All data parallel ranks connected.")
        logging.info("Inference Coordinator: Connected with data parallel ranks...")
        self.data_parallel_rank_iterator = cycle(self.identities_of_data_parallel_ranks)
        self.ready_event.set()
        print("Inference Coordinator: Ready to accept client connections.")

        # Todo [Siddharth]: Make this more robust to handle invalid messages.
        while True:
            identity, header, data = await self._irecv()

            if header == Headers.CONNECT:
                if identity in self.known_clients:
                    logging.info(
                        f"Client {identity} sent a duplicate connect request. Ignoring .."
                    )
                    continue

                self.known_clients.add(identity)
                self._isend(identity, Headers.ACK)

            elif header == Headers.SUBMIT_REQUEST:
                # ToDo [Siddharth]: We might want to tokenize the prompt on the
                # assigned data parallel rank for this process instead
                # of the coordinator.

                # Message from a known client
                if identity not in self.known_clients:
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

                next_data_parallel_rank_identity = self.get_next_data_parallel_rank()
                self._isend(
                    next_data_parallel_rank_identity,
                    Headers.SUBMIT_REQUEST,
                    [request_id, prompt, sampling_params],
                )
            elif header in [Headers.PAUSE, Headers.UNPAUSE, Headers.STOP]:
                # control signals for the engine
                # broadcast to all data parallel ranks
                if identity not in self.known_clients:
                    continue
                for data_parallel_rank_id in self.identities_of_data_parallel_ranks:
                    self._isend(data_parallel_rank_id, header)
            elif header == Headers.ENGINE_REPLY:
                # This is the output of a single engine step on some data parallel rank.
                assert identity in self.identities_of_data_parallel_ranks
                finished_requests = data

                for finished_request in finished_requests:
                    fid = finished_request["request_id"]
                    client_identity = self.request_id_to_client_id[fid]
                    client_request_identity = self.request_id_to_client_request_id[fid]
                    del self.request_id_to_client_id[fid]
                    del self.request_id_to_client_request_id[fid]

                    self._isend(
                        client_identity,
                        Headers.ENGINE_REPLY,
                        [client_request_identity, finished_request],
                    )

    @trace_async_exceptions
    async def _send_task(self):
        """Pop futures of sends out of a queue and await them.

        For explanation why this works, refer to the documentation for zmq.asyncio:
            'Returns a Future that resolves when sending is complete.'
        """
        while True:
            await (await self._send_awaitables.get())
            self._send_awaitables.task_done()

    def _isend(
        self, identity: bytes, header: Headers, data: Optional[List] = None
    ) -> asyncio.Future:
        """
        Asynchronously send a signal to the inference coordinator.

        Args:
            identity (bytes): The ZMQ identity of the recipient.
            header (Headers): The signal header to send.
            data (Optional[List]): The data payload to send.
        """
        to_send = [identity, header.value.to_bytes()]
        if data is not None:
            to_send.append(msgpack.packb(data, use_bin_type=True))
        send_awaitable = self.router_socket.send_multipart(to_send)
        self._send_awaitables.put_nowait(send_awaitable)

    async def _irecv(
        self, deserialize: bool = True
    ) -> Tuple[Optional[bytes], Headers, List | bytes | None]:
        """
        Asynchronously receive a signal from the inference coordinator.

        Returns:
            identity (Optional[bytes]): The source of the signal.
            header (Headers): The signal header received.
            data (List | bytes | None): The data payload received.
        """
        raw = await self.router_socket.recv_multipart()
        if self.socket_uses_identity:
            identity, header, *rest = raw
        else:
            header, *rest = raw
            identity = None

        header = Headers(int.from_bytes(header))
        data = rest[0] if rest else None

        if deserialize:
            message = msgpack.unpackb(data, raw=False) if data is not None else None
        else:
            message = data

        return identity, header, message

    @classmethod
    def entrypoint(
        cls, ready_event: Event, inference_coordinator_port: int, data_parallel_size: int
    ):
        """
        Class method to instantiate and run the coordinator, for use in a separate process.

        This method initializes the coordinator, signals a `ready_event` to indicate
        that it is fully initialized and listening, and then starts the main event loop.

        Args:
            ready_event (Event): A threading or multiprocessing event object that is set()
                once the coordinator is ready to accept connections.
            inference_coordinator_port (int): The port to bind to.
            data_parallel_size (int): The number of expected TP-coordinators.
        """
        # Register faulthandler to emit stack traces upon process kill.
        faulthandler.enable()
        faulthandler.register(signal.SIGTERM, all_threads=False, chain=True)
        faulthandler.register(signal.SIGINT, all_threads=False, chain=True)

        print("Inference Coordinator: Initializing coordinator...")
        coordinator = cls(ready_event, inference_coordinator_port, data_parallel_size)
        print("Inference Coordinator: Starting coordinator...")
        loop = get_asyncio_loop()
        coordinator.start(loop=loop)
        print("Inference Coordinator: Coordinator started.")
        try:
            loop.run_forever()
        except KeyboardInterrupt:
            logging.info("Coordinator process interrupted. Exiting...")
        finally:
            coordinator.stop()

    def stop(self):
        """
        Stops the inference coordinator, performing any necessary cleanup operations.
        """
        self.router_socket.close()
        self.context.term()
        self.send_task.cancel()
        self.recv_task.cancel()
