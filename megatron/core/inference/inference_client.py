# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import asyncio
import logging
import time
from typing import Awaitable, List, Optional, Union

from megatron.core.inference.inference_request import DynamicInferenceRequestRecord
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.utils import get_asyncio_loop, trace_async_exceptions

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
        assert (
            HAVE_MSGPACK
        ), "please install the messagepack library to use InferenceClient - pip install msgpack"
        self.context = zmq.Context()
        socket = self.context.socket(zmq.DEALER)
        socket.connect(inference_coordinator_address)

        self._loop = None
        self.running = asyncio.Event()
        self.paused = asyncio.Event()
        self.stopped = asyncio.Event()

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
                `serialize()` method.

        Returns:
            asyncio.Future: A future that will be resolved with a
            `DynamicInferenceRequestRecord` object containing the completed result.
        """
        if not self.running.is_set():
            raise RuntimeError("InferenceClient is not currently running.")
        request_id = self.next_request_id
        self.next_request_id += 1
        payload = [Headers.SUBMIT_REQUEST.value, request_id, prompt, sampling_params.serialize()]
        payload_serialized = msgpack.packb(payload, use_bin_type=True)
        self.socket.send(payload_serialized)
        assert request_id not in self.completion_futures
        self.completion_futures[request_id] = asyncio.get_running_loop().create_future()
        self.request_submission_times[request_id] = time.perf_counter()
        return self.completion_futures[request_id]

    @trace_async_exceptions
    async def _recv_task(self):
        """
        Listens for completed inference requests from the coordinator.

        This coroutine runs in an infinite loop, continuously polling the socket
        for data.
        When a request reply is received, it unpacks the message, finds the
        corresponding Future using the request ID, and sets the result.
        Other control packets are handled appropriately.

        This method is started as a background task by the `start()` method.
        """
        while True:
            try:
                data = msgpack.unpackb(self.socket.recv(flags=zmq.NOBLOCK), raw=False)
                header = Headers(data[0])
                if header == Headers.ENGINE_REPLY:
                    request_id, reply = data[1:]
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
                elif header == Headers.PAUSE_ACK:
                    self.paused.set()
                elif header == Headers.STOP_ACK:
                    self.stopped.set()
            except zmq.Again:
                await asyncio.sleep(0.005)
                continue
            except KeyboardInterrupt:
                break

    def _connect_with_inference_coordinator(self):
        """
        Performs the initial handshake with the inference coordinator.

        Sends a CONNECT signal and waits for a CONNECT_ACK reply to ensure the
        connection is established and acknowledged by the coordinator.
        """
        payload = [Headers.CONNECT.value]
        self.socket.send(msgpack.packb(payload, use_bin_type=True))
        reply = msgpack.unpackb(self.socket.recv(), raw=False)[0]
        assert Headers(reply) == Headers.CONNECT_ACK

    async def start(self, loop: Optional[asyncio.AbstractEventLoop] = None):
        """
        Connects to the coordinator and starts the background listener task.

        This method must be awaited before submitting any requests. It handles
        the initial handshake and spawns the `listen_for_completed_requests`
        coroutine.
        """
        logging.info("Client: Connecting to InferenceCoordinator...")
        self._loop = get_asyncio_loop(loop)
        self.running.set()
        self.paused.clear()
        self.stopped.clear()
        self._connect_with_inference_coordinator()
        self.listener_task = self._loop.create_task(self._recv_task())

    def _send_signal_to_engines(self, signal):
        """
        Sends a generic control signal to the inference coordinator.

        Args:
            signal: The signal to send, typically a value from the `Headers` enum.
        """
        payload = [signal.value]
        payload_serialized = msgpack.packb(payload, use_bin_type=True)
        self.socket.send(payload_serialized)

    def pause_engines(self) -> Awaitable:
        """Sends a signal to pause all inference engines.

        The signal first propagates thru the coordinator to all engines.
        All engines acknowledge this signal and clear their `running` flags.
        The coordinator awaits all acknowledgements before forwarding the ACK
            back to the client, as well as to the engines.
        The engines set their `paused` flags upon seeing the ACK.

        Returns:
            Awaitable: An awaitable that resolves when all engines have paused.
        """
        self._send_signal_to_engines(Headers.PAUSE)
        return self.paused.wait()

    def unpause_engines(self) -> None:
        """Sends a signal to unpause all inference engines."""
        self.paused.clear()
        self.running.set()
        self._send_signal_to_engines(Headers.UNPAUSE)

    def suspend_engines(self):
        """Sends a signal to pause all inference engines."""
        self._send_signal_to_engines(Headers.PAUSE)
        self._send_signal_to_engines(Headers.SUSPEND)

    def resume_engines(self):
        """Sends a signal to unpause all inference engines."""
        self._send_signal_to_engines(Headers.RESUME)
        self._send_signal_to_engines(Headers.UNPAUSE)

    def stop_engines(self) -> Awaitable:
        """Sends a signal to gracefully stop all inference engines.

        The signal first propagates thru the coordinator to all engines.
        All engines acknowledge this signal and clear their `running` flags.
        The coordinator awaits all acknowledgements before forwarding the ACK
            back to the client, as well as to the engines.
        The engines set their `stopped` flags upon seeing the ACK.

        Returns:
            Awaitable: An awaitable that resolves when all engines have stopped.
        """
        self._send_signal_to_engines(Headers.STOP)
        self.running.clear()
        return self.stopped.wait()

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
