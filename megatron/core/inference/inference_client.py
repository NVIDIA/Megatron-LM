# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import asyncio
import logging
import os
import time
from typing import List, Union

from megatron.core.inference.inference_request import DynamicInferenceRequest
from megatron.core.inference.sampling_params import SamplingParams

from .headers import Headers

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

from .headers import Headers


class InferenceClient:
    """
    An asynchronous client for communicating with an inference coordinator service.

    This client uses ZeroMQ (ZMQ) for messaging and MessagePack (msgpack) for
    serialization. It is designed to work within an asyncio event loop. It can
    submit inference requests, listen for completed results, and send control
    signals (e.g., pause, stop) to the inference engines.

    The client operates by connecting a ZMQ DEALER socket to the inference
    coordinator's ROUTER socket. Requests are sent with a unique ID, and an
    `asyncio.Future` is created for each request. A background task listens for
    replies from the coordinator, and when a reply is received, it resolves the
    corresponding future with the result.

    Attributes:
        context (zmq.Context): The ZeroMQ context.
        socket (zmq.Socket): The ZMQ DEALER socket used for communication.
        completion_futures (dict[int, asyncio.Future]): A dictionary mapping
            request IDs to the asyncio Future objects that will hold the results.
        next_request_id (int): A counter for generating unique request IDs.
        listener_task (asyncio.Task): The background task that listens for
            completed requests.
    """

    def __init__(self, inference_coordinator_port: int):
        """
        Initializes the InferenceClient.

        Args:
            inference_coordinator_port (int): The port number on which the
                inference coordinator is listening.
        """
        assert (
            HAVE_ZMQ
        ), "please install the pyzmq library to use InferenceClient - pip install pyzmq"
        assert (
            HAVE_MSGPACK
        ), "please install the messagepack library to use InferenceClient - pip install msgpack"
        self.context = zmq.Context()
        socket = self.context.socket(zmq.DEALER)
        inference_coordinator_address = os.getenv('MASTER_ADDR', '127.0.0.1')
        socket.connect(f"tcp://{inference_coordinator_address}:{inference_coordinator_port}")

        self.socket = socket
        self.completion_futures = {}
        self.request_submission_times = {}
        self.next_request_id = 0

    def add_request(
        self, prompt: Union[str, List[int]], sampling_params: SamplingParams
    ) -> asyncio.Future:
        """
        Submits a new inference request to the coordinator.

        This method sends the prompt and sampling parameters to the inference
        coordinator. It immediately returns an asyncio.Future, which can be
        awaited to get the result of the inference request when it is complete.

        Args:
            prompt (str): The input prompt to send to the language model.
            sampling_params: An object containing the sampling parameters for
                text generation (e.g., temperature, top_p). It must have a
                `serializable()` method.

        Returns:
            asyncio.Future: A future that will be resolved with a
            `DynamicInferenceRequest` object containing the completed result.
        """
        request_id = self.next_request_id
        self.next_request_id += 1
        payload = [Headers.SUBMIT_REQUEST.value, request_id, prompt, sampling_params.serializable()]
        payload_serialized = msgpack.packb(payload, use_bin_type=True)
        self.socket.send(payload_serialized)
        assert request_id not in self.completion_futures
        self.completion_futures[request_id] = asyncio.get_event_loop().create_future()
        self.request_submission_times[request_id] = time.perf_counter()
        return self.completion_futures[request_id]

    async def _listen_for_completed_requests(self):
        """
        Listens for completed inference requests from the coordinator.

        This coroutine runs in an infinite loop, continuously polling the socket
        for replies. When a reply is received, it unpacks the message, finds the
        corresponding Future using the request ID, and sets the result.

        This method is started as a background task by the `start()` method.
        """
        while True:
            try:
                request_id, reply = msgpack.unpackb(self.socket.recv(flags=zmq.NOBLOCK), raw=False)
                reply['latency'] = time.perf_counter() - self.request_submission_times.pop(
                    request_id
                )
                completion_future = self.completion_futures.pop(request_id)
                completion_future.set_result(DynamicInferenceRequest.deserialize(reply))
            except zmq.Again:
                await asyncio.sleep(0.005)
                continue
            except KeyboardInterrupt:
                break

    def _connect_with_inference_coordinator(self):
        """
        Performs the initial handshake with the inference coordinator.

        Sends a CONNECT signal and waits for an ACK reply to ensure the
        connection is established and acknowledged by the coordinator.
        """
        payload = [Headers.CONNECT.value]
        self.socket.send(msgpack.packb(payload, use_bin_type=True))
        reply = msgpack.unpackb(self.socket.recv(), raw=False)[0]
        assert Headers(reply) == Headers.ACK

    async def start(self):
        """
        Connects to the coordinator and starts the background listener task.

        This method must be awaited before submitting any requests. It handles
        the initial handshake and spawns the `listen_for_completed_requests`
        coroutine.
        """
        logging.info("Client: Connecting to InferenceCoordinator...")
        self._connect_with_inference_coordinator()
        self.listener_task = asyncio.create_task(self._listen_for_completed_requests())

    def _send_signal_to_engines(self, signal):
        """
        Sends a generic control signal to the inference coordinator.

        Args:
            signal: The signal to send, typically a value from the `Headers` enum.
        """
        payload = [signal.value]
        payload_serialized = msgpack.packb(payload, use_bin_type=True)
        self.socket.send(payload_serialized)

    def pause_engines(self):
        """Sends a signal to pause all inference engines."""
        self._send_signal_to_engines(Headers.PAUSE)

    def unpause_engines(self):
        """Sends a signal to unpause all inference engines."""
        self._send_signal_to_engines(Headers.UNPAUSE)

    def stop_engines(self):
        """Sends a signal to gracefully stop all inference engines."""
        self._send_signal_to_engines(Headers.STOP)

    def stop(self):
        """
        Stops the client and cleans up all resources.

        This method cancels the background listener task, closes the ZMQ socket,
        and terminates the ZMQ context. It should be called when the client is
        no longer needed to ensure a graceful shutdown.
        """
        self.listener_task.cancel()
        self.socket.close()
        self.context.term()
