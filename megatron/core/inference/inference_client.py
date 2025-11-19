# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import asyncio
import logging
import os
import time
from typing import List, Optional, Union

from megatron.core.inference.headers import Headers
from megatron.core.inference.inference_request import DynamicInferenceRequest
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.utils import get_asyncio_loop, trace_async_exceptions

from .headers import Headers

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
        self.context = zmq.asyncio.Context.instance()
        socket = self.context.socket(zmq.DEALER)
        inference_coordinator_address = os.getenv('MASTER_ADDR', '127.0.0.1')
        socket.connect(f"tcp://{inference_coordinator_address}:{inference_coordinator_port}")

        self.socket = socket
        self.socket_uses_identity = False

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
        self._isend(Headers.SUBMIT_REQUEST, [request_id, prompt, sampling_params.serializable()])
        assert request_id not in self.completion_futures
        self.completion_futures[request_id] = get_asyncio_loop().create_future()
        self.request_submission_times[request_id] = time.perf_counter()
        return self.completion_futures[request_id]

    @trace_async_exceptions
    async def _recv_task(self):
        """
        Listen for packets from the coordinator.

        This coroutine runs in an infinite loop, continuously polling the socket
        for data.
        When a request reply is received, it unpacks the message, finds the
        corresponding Future using the request ID, and sets the result.
        Other control packets are handled appropriately.

        This method is started as a background task by the `start()` method.
        """
        while True:
            try:
                _, header, data = await self._irecv()

                assert header == Headers.ACK or self.initial_reply
                if header == Headers.ENGINE_REPLY:
                    request_id, reply = data
                    reply['latency'] = time.perf_counter() - self.request_submission_times.pop(
                        request_id
                    )
                    completion_future = self.completion_futures.pop(request_id)
                    completion_future.set_result(DynamicInferenceRequest.deserialize(reply))
                elif header == Headers.ACK:
                    self.initial_reply = True
            except asyncio.CancelledError:
                break

    def start(self, loop: Optional[asyncio.AbstractEventLoop] = None):
        """
        Connects to the coordinator and starts the background tasks.

        This method must be completed before submitting any requests. It handles
        the initial handshake and spawns background tasks.
        """
        logging.info("Client: Connecting to InferenceCoordinator...")
        loop = get_asyncio_loop(loop)

        self.completion_futures = {}
        self.request_submission_times = {}
        self.next_request_id = 0
        self._send_awaitables = asyncio.Queue()

        self.initial_reply = False
        self._isend(Headers.CONNECT)

        self.send_task = loop.create_task(self._send_task())
        self.recv_task = loop.create_task(self._recv_task())

    @trace_async_exceptions
    async def _send_task(self):
        """Pop futures of sends out of a queue and await them.

        For explanation why this works, refer to the documentation for zmq.asyncio:
            'Returns a Future that resolves when sending is complete.'
        """
        while True:
            await (await self._send_awaitables.get())
            self._send_awaitables.task_done()

    def _isend(self, header: Headers, data: Optional[List] = None) -> asyncio.Future:
        """
        Asynchronously send a signal to the inference coordinator.

        Args:
            header (Headers): The signal header to send.
            data (Optional[List]): The data payload to send.
        """
        to_send = [header.value.to_bytes()]
        if data is not None:
            to_send.append(msgpack.packb(data, use_bin_type=True))
        send_awaitable = self.socket.send_multipart(to_send)
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
        raw = await self.socket.recv_multipart()
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

    def pause_engines(self):
        """Sends a signal to pause all inference engines."""
        self._isend(Headers.PAUSE)

    def unpause_engines(self):
        """Sends a signal to unpause all inference engines."""
        self._isend(Headers.UNPAUSE)

    def stop_engines(self):
        """Sends a signal to gracefully stop all inference engines."""
        self._isend(Headers.STOP)

    def stop(self):
        """
        Stops the client and cleans up all resources.

        This method cancels the background listener task, closes the ZMQ socket,
        and terminates the ZMQ context. It should be called when the client is
        no longer needed to ensure a graceful shutdown.
        """
        self.recv_task.cancel()
        self.send_task.cancel()
        self.socket.close()
        self.context.term()
