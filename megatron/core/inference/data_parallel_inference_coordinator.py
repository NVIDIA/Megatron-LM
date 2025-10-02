# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import logging
from collections import deque
from itertools import cycle, repeat
from typing import List, Tuple

import torch

from megatron.core.inference.headers import Headers
from megatron.core.inference.inference_request import DynamicInferenceRequest

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
        tokenizer: The tokenizer object for encoding prompts.
        router_socket (zmq.Socket): The central ZMQ ROUTER socket for all communication.
        data_parallel_size (int): The number of data parallel workers to expect.
        identities_of_data_parallel_ranks (deque): A deque holding the ZMQ
            identities of connected TP-coordinators, used for round-robin scheduling.
        request_id_to_client_id (dict): Maps server-side request IDs to the ZMQ
            identity of the client that initiated the request.
        request_id_to_client_request_id (dict): Maps server-side request IDs to the
            original request ID provided by the client.
        next_request_id (int): A counter for generating unique server-side request IDs.
        requests (dict): A store for active `DynamicInferenceRequest` objects, keyed by
            server-side request ID.
    """

    def __init__(self, tokenizer, inference_coordinator_port: int, data_parallel_size: int):
        """
        Initializes the inference coordinator.

        This sets up the ZMQ context and a ROUTER socket, binding it to the given
        port. It then enters a blocking loop to wait for all expected data parallel
        ranks to connect before proceeding.

        Args:
            tokenizer: An object with `tokenize`, `detokenize`, and `bos` attributes,
                used for processing text.
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
        self.tokenizer = tokenizer

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

        self.request_id_to_client_id = {}
        self.request_id_to_client_request_id = {}

        self.next_request_id = 0
        self.requests = {}

    def get_next_data_parallel_rank(self):
        """
        Selects the next data parallel rank using round-robin scheduling.

        Returns:
            bytes: The ZMQ identity of the next data parallel rank to receive a request.
        """
        return next(self.data_parallel_rank_iterator)

    def tokenize_prompt(
        self, prompt: str, add_BOS: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Utility to tokenize the input prompts

        Args:
            prompt (str): The input prompt

        Returns:
            torch.Tensor: Returns the tokenized prompt
        """
        prompt_tokens = self.tokenizer.tokenize(prompt)

        if add_BOS:
            prompt_tokens = [self.tokenizer.bos] + prompt_tokens

        return prompt_tokens

    def postprocess(
        self,
        request_ids: List[int],
        finished_request_ids: List[int],
        generated_tokens: List[int],
        log_probs: List[int],
        chunked_prefill_request_id: int = -1,
        materialize_only_last_token_logits: bool = True,
    ):
        """
        Processes replies from the engine, appending tokens and handling finished requests.

        For each generated token, this method appends it to the corresponding active
        request. If a request is marked as finished, it detokenizes the full
        sequence, sends the final result back to the original client, and cleans
        up the request state.

        Args:
            request_ids (List[int]): A list of request IDs that have new tokens.
            finished_request_ids (List[int]): A list of request IDs that have completed
                generation in this step.
            generated_tokens (List[int]): The list of new tokens, one for each ID in
                `request_ids`.
            log_probs (List[int]): Log probabilities for each token.
            chunked_prefill_request_id (int): The request ID currently undergoing chunked prefill,
                -1 if no chunked prefill is active.
        """
        # Todo [Siddharth]: This is duplicated logic from the engine.
        # We should refactor this to avoid duplication.
        log_probs_iter = log_probs if log_probs else repeat(None)
        for request_id, token, request_log_probs in zip(
            request_ids, generated_tokens, log_probs_iter
        ):
            request: DynamicInferenceRequest = self.requests[request_id]
            # Handle chunked prefill similar to the engine logic
            if chunked_prefill_request_id == -1 or request_id != chunked_prefill_request_id:
                request.generated_tokens.append(token)

                if request_log_probs is not None:
                    if not request.prompt_log_probs:
                        request.prompt_log_probs = []
                    if not request.generated_log_probs:
                        request.generated_log_probs = []
                    # If the request log probs span > 1 token we are in prefill
                    if len(request_log_probs) > 1:
                        request.prompt_log_probs.extend(request_log_probs)
                    else:
                        if (
                            # If it is a chunked prefill request
                            len(request.prompt_log_probs) > 0
                            # And we are missing the last token for prefill
                            and len(request.prompt_log_probs) < len(request.prompt_tokens)
                            # And we need to track full prefill
                            and not materialize_only_last_token_logits
                        ):
                            assert (
                                len(request.prompt_log_probs) == len(request.prompt_tokens) - 1
                            ), "Prompt log probs length is not equal to prompt tokens length - 1"
                            request.prompt_log_probs.extend(request_log_probs)
                        else:
                            request.generated_log_probs.extend(request_log_probs)
            else:
                # This is the chunked prefill request, handle log probs but don't append tokens
                if request_log_probs is not None:
                    if materialize_only_last_token_logits:
                        # Here we discard intermediate log probs,
                        # as we only materialize the last token log probs
                        request.prompt_log_probs = []
                        request.generated_log_probs = []
                    else:
                        # Otherwise, we gather log probs for all tokens
                        if not request.prompt_log_probs:
                            request.prompt_log_probs = []
                        request.prompt_log_probs.extend(request_log_probs)
                        request.generated_log_probs = []

        if finished_request_ids:
            for fid in finished_request_ids:
                if fid == chunked_prefill_request_id:
                    continue  # skip chunked prefill request, this is not a finished request
                request = self.requests.pop(fid)
                request.generated_length = len(request.generated_tokens)
                request.generated_text = self.tokenizer.detokenize(request.generated_tokens)

                client_identity = self.request_id_to_client_id[fid]
                client_request_identity = self.request_id_to_client_request_id[fid]
                del self.request_id_to_client_id[fid]
                del self.request_id_to_client_request_id[fid]
                self.router_socket.send_multipart(
                    [
                        client_identity,
                        msgpack.packb(
                            [client_request_identity, request.serializable()], use_bin_type=True
                        ),
                    ]
                )

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
                    [sender_identity, msgpack.packb([Headers.ACK.value], use_bin_type=True)]
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

                # tokenize the prompt if it is a string.
                if isinstance(prompt, str):
                    prompt_tokens = self.tokenize_prompt(prompt)
                else:
                    prompt_tokens = prompt  # no error handling here as it is done in the engine.

                self.requests[request_id] = DynamicInferenceRequest(
                    request_id=request_id, prompt=prompt, prompt_tokens=prompt_tokens
                )

                next_data_parallel_rank_identity = self.get_next_data_parallel_rank()
                self.router_socket.send_multipart(
                    [
                        next_data_parallel_rank_identity,
                        msgpack.packb(
                            [
                                Headers.SUBMIT_REQUEST.value,
                                request_id,
                                prompt_tokens,
                                sampling_params,
                            ],
                            use_bin_type=True,
                        ),
                    ]
                )
            elif header in [Headers.PAUSE, Headers.UNPAUSE, Headers.STOP]:
                # control signals for the engine
                # broadcast to all data parallel ranks
                if sender_identity not in known_clients:
                    continue
                for data_parallel_rank_id in self.identities_of_data_parallel_ranks:
                    self.router_socket.send_multipart(
                        [data_parallel_rank_id, msgpack.packb([header.value], use_bin_type=True)]
                    )
            elif header == Headers.ENGINE_REPLY:
                # This is the output of a single engine step on some data parallel rank.
                assert sender_identity in self.identities_of_data_parallel_ranks
                (
                    request_ids,
                    finished_request_ids,
                    generated_tokens,
                    logprobs,
                    chunked_prefill_request_id,
                    materialize_only_last_token_logits,
                ) = deserialized_payload[1:]
                self.postprocess(
                    request_ids,
                    finished_request_ids,
                    generated_tokens,
                    logprobs,
                    chunked_prefill_request_id,
                    materialize_only_last_token_logits,
                )

    @classmethod
    def entrypoint(
        cls, ready_event, tokenizer, inference_coordinator_port: int, data_parallel_size: int
    ):
        """
        Class method to instantiate and run the coordinator, for use in a separate process.

        This method initializes the coordinator, signals a `ready_event` to indicate
        that it is fully initialized and listening, and then starts the main event loop.

        Args:
            ready_event: A threading or multiprocessing event object that is set()
                once the coordinator is ready to accept connections.
            tokenizer: The tokenizer object.
            inference_coordinator_port (int): The port to bind to.
            data_parallel_size (int): The number of expected TP-coordinators.
        """
        tokenizer = tokenizer
        coordinator = cls(tokenizer, inference_coordinator_port, data_parallel_size)
        ready_event.set()
        try:
            coordinator.start()
        except KeyboardInterrupt:
            logging.info("Coordinator process interrupted. Exiting...")
            coordinator.stop()

    def stop(self):
        """
        Stops the inference coordinator, performing any necessary cleanup operations.
        """
        self.router_socket.close()
        self.context.term()
