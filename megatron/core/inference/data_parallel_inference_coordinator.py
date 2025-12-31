# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import faulthandler
import logging
import signal
from collections import deque
from itertools import cycle
from multiprocessing import Event

import torch

from megatron.core.inference.headers import Headers, UnknownHeaderError

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

    def __init__(self, inference_coordinator_port: int, data_parallel_size: int):
        """
        Initializes the inference coordinator.

        This sets up the ZMQ context and a ROUTER socket, binding it to the given
        port. It then enters a blocking loop to wait for all expected data parallel
        ranks to connect before proceeding.

        Args:
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
        self.context = zmq.Context()

        # This is the central router socket
        # 1. data parallel ranks connect to this socket to register themselves
        # 2. Users connect to this socket and submit their requests. We transmit them to
        #    data parallel ranks in a round robin fashion
        # 3. data parallel ranks return completed requests to this socket. We route them back to
        #    the user that had submitted the request originally.

        self.router_socket = self.context.socket(zmq.ROUTER)
        self.router_socket.bind(f"tcp://0.0.0.0:{inference_coordinator_port}")
        self.data_parallel_size = data_parallel_size

        logging.info("Inference Coordinator: waiting for connections from data parallel ranks...")
        # First wait for all data parallel ranks to establish connections.
        self.identities_of_data_parallel_ranks = deque([])
        # time.sleep(5)  # Give data parallel ranks time to spawn and connect.
        for _ in range(data_parallel_size):
            identity, _ = self.router_socket.recv_multipart()
            assert identity not in self.identities_of_data_parallel_ranks
            self.identities_of_data_parallel_ranks.append(identity)
        logging.info("Inference Coordinator: Connected with data parallel ranks...")
        self.data_parallel_rank_iterator = cycle(self.identities_of_data_parallel_ranks)
        self.data_parallel_pause_acks = set()
        self.data_parallel_stop_acks = set()

        self.request_id_to_client_id = {}
        self.request_id_to_client_request_id = {}

        self.next_request_id = 0

    def get_next_data_parallel_rank(self):
        """
        Selects the next data parallel rank using round-robin scheduling.

        Returns:
            bytes: The ZMQ identity of the next data parallel rank to receive a request.
        """
        return next(self.data_parallel_rank_iterator)

    def start(self):
        """
        Starts the main event loop for the coordinator.

        This method runs an infinite loop, continuously listening for incoming
        messages on the ZMQ ROUTER socket. It parses the message header to
        determine the message type and takes appropriate action, such as
        handling new client connections, forwarding requests, broadcasting
        control signals, or processing replies from the engines.
        """
        # Todo [Siddharth]: Make this more robust to handle invalid messages.
        known_clients = set()
        while True:
            sender_identity, serialized_payload = self.router_socket.recv_multipart()
            deserialized_payload = msgpack.unpackb(serialized_payload, raw=False)
            header = Headers(deserialized_payload[0])

            if header == Headers.CONNECT:
                if sender_identity in known_clients:
                    logging.info(
                        f"Client {sender_identity} sent a duplicate connect request. Ignoring .."
                    )
                    continue

                # print(f"New client connected: {sender_identity}")
                known_clients.add(sender_identity)
                self.router_socket.send_multipart(
                    [sender_identity, msgpack.packb([Headers.CONNECT_ACK.value], use_bin_type=True)]
                )

            elif header == Headers.SUBMIT_REQUEST:
                # ToDo [Siddharth]: We might want to tokenize the prompt on the
                # assigned data parallel rank for this process instead
                # of the coordinator.

                # Message from a known client
                if sender_identity not in known_clients:
                    logging.info(
                        f"Received message from unknown client {sender_identity}. Ignoring."
                    )
                    continue
                # this is a message from a client.
                # route it to a data parallel rank
                client_request_id, prompt, sampling_params = deserialized_payload[1:]
                # map client request_id to server request_id
                # necessary because multiple clients might have the same request_id.
                request_id = self.next_request_id
                self.next_request_id += 1
                self.request_id_to_client_id[request_id] = sender_identity
                self.request_id_to_client_request_id[request_id] = client_request_id

                # Serialize prompt.
                if isinstance(prompt, (str, list)):
                    pass
                elif isinstance(prompt, torch.Tensor):
                    prompt = prompt.tolist()
                else:
                    raise Exception("specialize for <%s> prompt." % type(prompt).__name__)

                next_data_parallel_rank_identity = self.get_next_data_parallel_rank()
                self.router_socket.send_multipart(
                    [
                        next_data_parallel_rank_identity,
                        msgpack.packb(
                            [Headers.SUBMIT_REQUEST.value, request_id, prompt, sampling_params],
                            use_bin_type=True,
                        ),
                    ]
                )
            elif header in [
                Headers.PAUSE,
                Headers.UNPAUSE,
                Headers.SUSPEND,
                Headers.RESUME,
                Headers.STOP,
            ]:
                # control signals for the engine
                # broadcast to all data parallel ranks
                if sender_identity not in known_clients:
                    continue
                for data_parallel_rank_id in self.identities_of_data_parallel_ranks:
                    self.router_socket.send_multipart(
                        [data_parallel_rank_id, msgpack.packb([header.value], use_bin_type=True)]
                    )
                if header == Headers.UNPAUSE:
                    self.data_parallel_pause_acks = set()
            elif header == Headers.PAUSE_ACK:
                # control signal ack from the engine
                assert sender_identity in self.identities_of_data_parallel_ranks
                assert sender_identity not in self.data_parallel_pause_acks
                self.data_parallel_pause_acks.add(sender_identity)
                # route to all clients only once we have gotten an ack from all data parallel ranks
                if len(self.data_parallel_pause_acks) == self.data_parallel_size:
                    for client_id in known_clients:
                        self.router_socket.send_multipart(
                            [
                                client_id,
                                msgpack.packb([header.value, sender_identity], use_bin_type=True),
                            ]
                        )
                    for data_parallel_rank_id in self.identities_of_data_parallel_ranks:
                        self.router_socket.send_multipart(
                            [
                                data_parallel_rank_id,
                                msgpack.packb([Headers.PAUSE_ACK.value], use_bin_type=True),
                            ]
                        )
            elif header == Headers.STOP_ACK:
                # control signal ack from the engine
                assert sender_identity in self.identities_of_data_parallel_ranks
                assert sender_identity not in self.data_parallel_stop_acks
                self.data_parallel_stop_acks.add(sender_identity)
                # route to all clients only once we have gotten an ack from all data parallel ranks
                if len(self.data_parallel_stop_acks) == self.data_parallel_size:
                    for client_id in known_clients:
                        self.router_socket.send_multipart(
                            [
                                client_id,
                                msgpack.packb([header.value, sender_identity], use_bin_type=True),
                            ]
                        )
                    for data_parallel_rank_id in self.identities_of_data_parallel_ranks:
                        self.router_socket.send_multipart(
                            [
                                data_parallel_rank_id,
                                msgpack.packb([Headers.STOP_ACK.value], use_bin_type=True),
                            ]
                        )
                    break  # Exit the main loop after STOP_ACKs have been processed.
            elif header == Headers.ENGINE_REPLY:
                # This is the output of a single engine step on some data parallel rank.
                assert sender_identity in self.identities_of_data_parallel_ranks
                finished_request_records = deserialized_payload[1]

                for finished_request_record in finished_request_records:
                    fid = finished_request_record["requests"][0]["request_id"]
                    client_identity = self.request_id_to_client_id[fid]
                    client_request_identity = self.request_id_to_client_request_id[fid]
                    del self.request_id_to_client_id[fid]
                    del self.request_id_to_client_request_id[fid]

                    self.router_socket.send_multipart(
                        [
                            client_identity,
                            msgpack.packb(
                                [header.value, client_request_identity, finished_request_record],
                                use_bin_type=True,
                            ),
                        ]
                    )

            else:
                raise UnknownHeaderError(header)

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
        coordinator = cls(inference_coordinator_port, data_parallel_size)
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
        self.router_socket.close()
        self.context.term()
