# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import asyncio
import logging
import time
from typing import List, Optional, Tuple, Union

from megatron.core.inference.async_zmq_socket import AsyncZmqSendRecv
from megatron.core.inference.inference_request import DynamicInferenceRequestRecord
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.utils import get_asyncio_loop, trace_async_exceptions

from .headers import Headers

try:
    import zmq
    import zmq.asyncio

    HAVE_ZMQ = True
except:
    HAVE_ZMQ = False

class InferenceClient:
    """
    An asynchronous client for communicating with an inference coordinator service.

    This client uses ZeroMQ (ZMQ) for messaging and MessagePack (msgpack) for
    serialization. It is designed to work within an asyncio event loop. It can
    submit inference requests, listen for completed results, and send control
    signals (e.g., pause, stop) to the inference engines.

    The client operates by connecting an async ZMQ DEALER socket to the inference
    coordinator's ROUTER socket. Sends are decoupled from callers via a send queue
    pattern: `_isend()` enqueues send futures, and a background `_send_task`
    drains and awaits them. This makes `add_request()` and signal methods
    synchronous.

    Attributes:
        context (zmq.asyncio.Context): The async ZeroMQ context.
        socket (zmq.asyncio.Socket): The async ZMQ DEALER socket used for
            communication.
        completion_futures (dict[int, asyncio.Future]): A dictionary mapping
            request IDs to the asyncio Future objects that will hold the results.
        next_request_id (int): A counter for generating unique request IDs.
    """

    def __init__(self, inference_coordinator_address: str):
        """
        Initializes the InferenceClient.

        Args:
            inference_coordinator_address (str): The address on which the
                inference coordinator is listening.
        """
        assert (
            HAVE_ZMQ
        ), "please install the pyzmq library to use InferenceClient - pip install pyzmq"
        self.context = zmq.asyncio.Context.instance()
        socket = self.context.socket(zmq.DEALER)
        socket.connect(inference_coordinator_address)

        self.socket = socket
        self._zmq = AsyncZmqSendRecv()

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
                `serialize()` method.

        Returns:
            asyncio.Future: A future that will be resolved with a
            `DynamicInferenceRequestRecord` object containing the completed result.
        """
        request_id = self.next_request_id
        self.next_request_id += 1
        self._isend(Headers.SUBMIT_REQUEST, [request_id, prompt, sampling_params.serialize()])
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

                assert header == Headers.ACK or self.is_running.is_set()
                if header == Headers.ENGINE_REPLY:
                    request_id, reply = data
                    reply['latency'] = time.perf_counter() - self.request_submission_times.pop(
                        request_id
                    )
                    completion_future = self.completion_futures.pop(request_id)
                    if completion_future.done():
                        logging.warning(f"Client: The future for {request_id} has been cancelled!")
                        continue
                    completed_request = DynamicInferenceRequestRecord.deserialize(reply)
                    completion_future.get_loop().call_soon_threadsafe(
                        completion_future.set_result, completed_request
                    )
                elif header == Headers.ACK:
                    self.is_running.set()
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

        # Attempt to connect, and do not allow any sends until we are connected.
        self.is_running = asyncio.Event()
        self._startup_sends = []
        self._isend(Headers.CLIENT_CONNECT)

        self.startup_sends_task = loop.create_task(self._startup_sends_task())
        self.send_task = loop.create_task(self._zmq.send_task())
        self.recv_task = loop.create_task(self._recv_task())

    @trace_async_exceptions
    async def _startup_sends_task(self):
        """Before a connection is established, we queue up sends for later."""
        await self.is_running.wait()
        for (header, data) in self._startup_sends:
            self._isend(header, data)
        self._startup_sends = None

    def _isend(self, header: Headers, data: Optional[List] = None):
        """
        Asynchronously send a signal to the inference coordinator.

        Args:
            header (Headers): The signal header to send.
            data (Optional[List]): The data payload to send.
        """
        # If we have not connected yet, wait on sends.
        if not self.is_running.is_set():
            if header not in [Headers.ENGINE_CONNECT, Headers.CLIENT_CONNECT]:
                self._startup_sends.append((header, data))
                return

        self._zmq.isend(self.socket, header, data)

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
        return await self._zmq.irecv(self.socket, deserialize=deserialize)

    def pause_engines(self):
        """Sends PAUSE to all engines via coordinator."""
        self._isend(Headers.PAUSE)

    def unpause_engines(self):
        """Sends UNPAUSE to all engines via coordinator."""
        self._isend(Headers.UNPAUSE)

    def increment_staleness(self):
        """Sends a signal to increment staleness on all in-flight requests."""
        self._isend(Headers.INCREMENT_STALENESS)

    def suspend_engines(self):
        """Sends SUSPEND to all engines via coordinator."""
        self._isend(Headers.SUSPEND)

    def resume_engines(self):
        """Sends RESUME to all engines via coordinator."""
        self._isend(Headers.RESUME)

    def stop_engines(self):
        """Sends STOP to all engines via coordinator. Requires PAUSED or SUSPENDED.

        Callers should await engine.stopped for confirmation.
        Does not affect the coordinator.
        """
        self._isend(Headers.STOP)

    def shutdown_coordinator(self):
        """Tells the coordinator process to exit its main loop.

        Does not affect the engines.
        """
        self._isend(Headers.SHUTDOWN)

    def stop(self):
        """
        Stops the client and cleans up all resources.

        This method cancels the background listener task, closes the ZMQ socket,
        and terminates the ZMQ context. It should be called when the client is
        no longer needed to ensure a graceful shutdown.
        """
        self.recv_task.cancel()
        self.startup_sends_task.cancel()
        self._zmq.shutdown()
        # Wake up any listeners.
        for future in self.completion_futures.values():
            if not future.done():
                future.cancel()
        self.completion_futures.clear()
        self.socket.close(linger=0)
        self.context.term()
