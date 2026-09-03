# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import asyncio
import functools
import logging
import time
from typing import List, Optional, Union

from megatron.core.inference.async_stream import AsyncStream
from megatron.core.inference.inference_request import (
    DynamicInferenceRequest,
    serialize_multimodal_data,
)
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


class InferenceRequestError(RuntimeError):
    """Terminal request failure reported by the inference coordinator."""

    def __init__(self, reason: str, *, source_safe: bool = False):
        super().__init__(reason)
        self.source_safe = source_safe


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

    def __init__(self, inference_coordinator_address: str, deserialize: bool = False):
        """
        Initializes the InferenceClient.

        Args:
            inference_coordinator_address (str): The address on which the
                inference coordinator is listening.
            deserialize (bool): If True, deserialize completed requests
                into DynamicInferenceRequest objects. If False (default), return
                the raw serialized dict for lower overhead.
        """
        assert (
            HAVE_ZMQ
        ), "please install the pyzmq library to use InferenceClient - pip install pyzmq"
        assert (
            HAVE_MSGPACK
        ), "please install the messagepack library to use InferenceClient - pip install msgpack"
        self.context = zmq.Context()
        socket = self.context.socket(zmq.DEALER)

        # Prevent socket.send() from thread-blocking at >1000 concurrent requests
        socket.setsockopt(zmq.SNDHWM, 0)
        socket.setsockopt(zmq.RCVHWM, 0)

        socket.connect(inference_coordinator_address)

        self._loop = None
        self.socket = socket
        self.deserialize = deserialize
        self.completion_futures = {}
        self.request_submission_times = {}
        self.next_request_id = 0
        self.streams: dict[int, AsyncStream[dict]] = {}
        self.aborted_request_ids: set[int] = set()
        # Resolves when abort cleanup makes transferred state safe to reuse.
        self.abort_futures: dict[int, asyncio.Future] = {}
        # Background socket receiver for request, stream, and abort replies.
        self.listener_task: asyncio.Task | None = None

    def add_request(
        self,
        prompt: Union[str, List[int]],
        sampling_params: SamplingParams,
        *,
        multi_modal_data=None,
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
            multi_modal_data: Optional vLLM-style modality dictionary.

                Images:
                    ``"image"`` accepts raw image bytes, a list of raw image
                    bytes, or a preprocessed image tensor dictionary.
                Video:
                    ``"video"`` accepts raw video bytes, a list of raw video
                    bytes, or a preprocessed video tensor dictionary.
                Audio:
                    Audio does not yet have any supported data preprocessing
                    or modeling formats.

        Returns:
            asyncio.Future: A future that will be resolved with a
            `DynamicInferenceRequest` object (if deserialize=True) or a raw
            serialized dict (if deserialize=False) containing the completed result.
        """
        return self.add_request_with_id(prompt, sampling_params, multi_modal_data=multi_modal_data)[
            1
        ]

    def add_request_with_id(
        self,
        prompt: Union[str, List[int]],
        sampling_params: SamplingParams,
        *,
        multi_modal_data=None,
    ) -> tuple[int, asyncio.Future]:
        """Submit a request and return its id alongside its completion future.

        Same submission as add_request, which delegates here. The id is what
        abort_request takes, so a caller that may need to cancel -- an HTTP
        handler whose client can disconnect mid-generation, for instance -- has
        to use this form. With only the future in hand there is no way to name
        the request to the coordinator, and cancelling the future alone leaves
        the engine generating.

        Args:
            prompt: A string or list of token IDs.
            sampling_params: Sampling parameters for the request.
            multi_modal_data: Optional vLLM-style modality dictionary; see
                add_request.

        Returns:
            tuple[int, asyncio.Future]: The request id and its completion future.
        """
        request_id = self.next_request_id
        self.next_request_id += 1
        payload = [
            Headers.SUBMIT_REQUEST.value,
            request_id,
            prompt,
            sampling_params.serialize(),
            serialize_multimodal_data(multi_modal_data),
        ]
        return request_id, self._submit_request(payload, request_id)

    def _make_kv_handoff_request(
        self,
        prompt: Union[str, List[int]],
        sampling_params: SamplingParams,
        kv_meta: dict,
        src_block_ids: List[int],
    ) -> tuple[int, list]:
        """Allocate an ID and build a decode request carrying remote KV metadata."""
        request_id = self.next_request_id
        self.next_request_id += 1
        payload = [
            Headers.SUBMIT_REQUEST_WITH_KV.value,
            request_id,
            prompt,
            sampling_params.serialize(),
            kv_meta,
            list(src_block_ids),
        ]
        return request_id, payload

    def add_request_with_kv_handoff(
        self,
        prompt: Union[str, List[int]],
        sampling_params: SamplingParams,
        kv_meta: dict,
        src_block_ids: List[int],
    ) -> asyncio.Future:
        """Submit a request with remote KV metadata.

        The decode engine allocates local blocks, pulls the KV from the
        prefill peer described by ``kv_meta``, then begins generation.

        Args:
            prompt: A string or list of token IDs.
            sampling_params: Sampling parameters for the decode request.
            kv_meta: Metadata identifying the remote KV buffers.
            src_block_ids: Remote block IDs containing the request's KV state.

        Returns:
            asyncio.Future: A future that resolves to the completed request.
        """
        request_id, payload = self._make_kv_handoff_request(
            prompt, sampling_params, kv_meta, src_block_ids
        )
        return self._submit_request(payload, request_id)

    def add_request_with_kv_handoff_streaming(
        self,
        prompt: Union[str, List[int]],
        sampling_params: SamplingParams,
        kv_meta: dict,
        src_block_ids: List[int],
    ) -> AsyncStream[dict]:
        """Submit a streaming request with remote KV metadata.

        Returns the same per-step partial/final iterator as
        :meth:`add_request_streaming`.

        Args:
            prompt: A string or list of token IDs.
            sampling_params: Sampling parameters for the decode request.
            kv_meta: Metadata identifying the remote KV buffers.
            src_block_ids: Remote block IDs containing the request's KV state.

        Returns:
            AsyncStream[dict]: Per-step partial and final reply frames.
        """
        sampling_params.streaming = True
        request_id, payload = self._make_kv_handoff_request(
            prompt, sampling_params, kv_meta, src_block_ids
        )
        return self._submit_stream(payload, request_id)

    def release_handoff(self, request_id: int) -> None:
        """Tell the coordinator to release the KV blocks pinned for `request_id`.

        Fire-and-forget. The coordinator broadcasts RELEASE_KV to every engine;
        engines without that request_id ignore the message.
        """
        payload = [Headers.RELEASE_KV.value, int(request_id)]
        self.socket.send(msgpack.packb(payload, use_bin_type=True))

    def abort_request(self, request_id: int) -> None:
        """Cancel an in-flight request and close its local response stream."""

        self._send_abort(request_id)

    def abort_request_and_wait(self, request_id: int) -> asyncio.Future:
        """Cancel a request and return its source-safety acknowledgement."""

        request_id = int(request_id)
        existing = self.abort_futures.get(request_id)
        if existing is not None:
            return existing
        abort_future = self._new_abort_future(request_id)
        self._send_abort(request_id)
        return abort_future

    def _send_abort(self, request_id: int) -> None:
        request_id = int(request_id)
        stream = self.streams.pop(request_id, None)
        future = self.completion_futures.pop(request_id, None)
        abort_future = self.abort_futures.get(request_id)
        if stream is None and future is None and abort_future is None:
            # Already completed (or never submitted): _submit_request and
            # _submit_stream register synchronously and _recv_task pops only
            # immediately before delivering, so absence from both means the
            # reply has been consumed. No further ENGINE_REPLY will arrive to
            # prune aborted_request_ids, and the coordinator has already
            # dropped its mapping, so recording the id would leak an entry
            # nothing ever removes and the ABORT_REQUEST send would be wasted.
            return
        self.aborted_request_ids.add(request_id)
        if stream is not None:
            stream.finish()
        if future is not None and not future.done():
            future.cancel()
        self.request_submission_times.pop(request_id, None)
        payload = [Headers.ABORT_REQUEST.value, request_id]
        self.socket.send(msgpack.packb(payload, use_bin_type=True))

    def _new_abort_future(self, request_id: int) -> asyncio.Future:
        """Create a future for the request's source-safety acknowledgement."""

        future = asyncio.get_running_loop().create_future()
        self.abort_futures[request_id] = future
        future.add_done_callback(functools.partial(self._discard_abort_future, request_id))
        return future

    def _discard_abort_future(self, request_id: int, future: asyncio.Future) -> None:
        """Remove a completed acknowledgement without discarding a newer waiter."""

        if self.abort_futures.get(request_id) is future:
            self.abort_futures.pop(request_id)

    def add_request_streaming(
        self,
        prompt: Union[str, List[int]],
        sampling_params: SamplingParams,
        *,
        multi_modal_data=None,
    ) -> AsyncStream[dict]:
        """Submit a streaming inference request.

        Used by Dynamo directly and by the OpenAI-compatible HTTP frontend.

        Returns an async iterator that yields incremental output dictionaries:

        - ``{"partial": {"request_id": int, "new_tokens": list[int]}}`` whenever
          the request's streaming interval is reached, in order.
        - ``{"final": <full reply dict or DynamicInferenceRequest>}`` exactly once
          at the end. The iterator then stops.

        ``sampling_params.streaming`` is forced to True before submission so the
        engine knows to emit ENGINE_REPLY_PARTIAL frames for this request.

        Args:
            prompt: A string or list of token IDs.
            sampling_params: Sampling parameters. ``streaming`` is set to True
                in-place.
            multi_modal_data: Optional vLLM-style modality dictionary.

                Images:
                    ``"image"`` accepts raw image bytes, a list of raw image
                    bytes, or a preprocessed image tensor dictionary.
                Video:
                    ``"video"`` accepts raw video bytes, a list of raw video
                    bytes, or a preprocessed video tensor dictionary.
                Audio:
                    Audio does not yet have any supported data preprocessing
                    or modeling formats.

        Returns:
            AsyncStream[dict]: Per-step partial and final reply frames.
        """
        sampling_params.streaming = True
        request_id = self.next_request_id
        self.next_request_id += 1
        payload = [
            Headers.SUBMIT_REQUEST.value,
            request_id,
            prompt,
            sampling_params.serialize(),
            serialize_multimodal_data(multi_modal_data),
        ]
        return self._submit_stream(payload, request_id)

    def _submit_request(self, payload: list, request_id: int) -> asyncio.Future:
        """Send a prepared request and register its completion future."""
        self.socket.send(msgpack.packb(payload, use_bin_type=True))
        assert request_id not in self.completion_futures
        future = asyncio.get_running_loop().create_future()
        self.completion_futures[request_id] = future
        self.request_submission_times[request_id] = time.perf_counter()
        return future

    def _submit_stream(self, payload: list, request_id: int) -> AsyncStream[dict]:
        """Send a prepared streaming request and register its response stream."""
        self.socket.send(msgpack.packb(payload, use_bin_type=True))
        stream = AsyncStream(
            request_id, functools.partial(self.abort_request, request_id), loop=self._loop
        )
        self.streams[request_id] = stream
        self.request_submission_times[request_id] = time.perf_counter()
        return stream

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
                    if request_id in self.aborted_request_ids:
                        self.aborted_request_ids.discard(request_id)
                        continue
                    submitted = self.request_submission_times.pop(request_id, None)
                    if submitted is not None:
                        reply['latency'] = time.perf_counter() - submitted
                    # Streaming path: deliver final reply + sentinel and stop.
                    if request_id in self.streams:
                        stream = self.streams.pop(request_id)
                        completed_request = (
                            DynamicInferenceRequest.deserialize(reply)
                            if self.deserialize
                            else reply
                        )
                        stream.put({"final": completed_request})
                        stream.finish()
                        continue
                    completion_future = self.completion_futures.pop(request_id, None)
                    if completion_future is None:
                        logging.warning(
                            "Client: ignoring late ENGINE_REPLY for request %d", request_id
                        )
                        continue
                    if completion_future.done():
                        logging.warning(f"Client: The future for {request_id} has been cancelled!")
                        continue
                    completed_request = (
                        DynamicInferenceRequest.deserialize(reply) if self.deserialize else reply
                    )
                    completion_future.set_result(completed_request)
                elif header == Headers.ENGINE_REPLY_PARTIAL:
                    request_id, partial = data[1:]
                    stream = self.streams.get(request_id)
                    if stream is not None:
                        stream.put({"partial": partial})
                elif header == Headers.REQUEST_ERROR:
                    request_id, reason, source_safe = data[1:]
                    self.request_submission_times.pop(request_id, None)
                    error = InferenceRequestError(str(reason), source_safe=bool(source_safe))
                    abort_future = self.abort_futures.get(request_id)
                    if source_safe:
                        self.aborted_request_ids.discard(request_id)
                        if abort_future is not None and not abort_future.done():
                            abort_future.set_result(True)
                    stream = self.streams.pop(request_id, None)
                    if stream is not None:
                        stream.finish(exception=error)
                        continue
                    future = self.completion_futures.pop(request_id, None)
                    if future is not None and not future.done():
                        future.set_exception(error)
                elif header == Headers.REQUEST_ABORTED:
                    request_id, source_safe = int(data[1]), bool(data[2])
                    if source_safe:
                        self.aborted_request_ids.discard(request_id)
                    future = self.abort_futures.get(request_id)
                    if future is None:
                        continue
                    if source_safe and not future.done():
                        future.set_result(True)
            except zmq.Again:
                await asyncio.sleep(0.005)
                continue
            except KeyboardInterrupt:
                break

    def _connect_with_inference_coordinator(self, timeout_seconds: Optional[float] = None):
        """
        Performs the initial handshake with the inference coordinator.

        Sends a CONNECT signal and waits for a CONNECT_ACK reply to ensure the
        connection is established and acknowledged by the coordinator.
        """
        payload = [Headers.CONNECT.value]
        self.socket.send(msgpack.packb(payload, use_bin_type=True))
        if timeout_seconds is not None and not self.socket.poll(
            timeout=max(0, int(timeout_seconds * 1000))
        ):
            raise TimeoutError("Timed out connecting to the Megatron inference coordinator")
        reply = msgpack.unpackb(self.socket.recv(), raw=False)
        assert Headers(reply[0]) == Headers.CONNECT_ACK

    def start(
        self,
        loop: Optional[asyncio.AbstractEventLoop] = None,
        connect_timeout_seconds: Optional[float] = None,
    ):
        """
        Connects to the coordinator and starts the background listener task.

        This must be called before submitting any requests. It handles
        the initial handshake and spawns the `listen_for_completed_requests`
        coroutine.
        """
        logging.info("Client: Connecting to InferenceCoordinator...")
        self._loop = get_asyncio_loop(loop)
        self._connect_with_inference_coordinator(connect_timeout_seconds)
        self.listener_task = self._loop.create_task(self._recv_task())

    def _send_signal_to_engines(self, signal, *args):
        """
        Sends a generic control signal to the inference coordinator.

        Args:
            signal: The signal to send, typically a value from the `Headers` enum.
            *args: Optional extra values to include in the payload.
        """
        payload = [signal.value, *args]
        payload_serialized = msgpack.packb(payload, use_bin_type=True)
        self.socket.send(payload_serialized)

    def pause_engines(self):
        """Sends PAUSE to all engines via coordinator.

        The coordinator broadcasts PAUSE. Each engine reaches EP consensus,
        then synchronizes via a world-wide barrier before transitioning to
        PAUSED. Callers should await engine.paused for confirmation.
        """
        self._send_signal_to_engines(Headers.PAUSE)

    def unpause_engines(self) -> None:
        """Sends UNPAUSE to all engines. No synchronization needed."""
        self._send_signal_to_engines(Headers.UNPAUSE)

    def start_cuda_profiler(self) -> None:
        """Sends START_CUDA_PROFILER to all engines via coordinator.

        Each engine calls ``torch.cuda.profiler.start()`` (cudaProfilerStart) on
        its next loop iteration, so an outer ``nsys profile --capture-range=
        cudaProfilerApi`` begins recording. No synchronization needed.
        """
        self._send_signal_to_engines(Headers.START_CUDA_PROFILER)

    def stop_cuda_profiler(self) -> None:
        """Sends STOP_CUDA_PROFILER to all engines (cudaProfilerStop)."""
        self._send_signal_to_engines(Headers.STOP_CUDA_PROFILER)

    def set_generation_epoch(self, generation_epoch: int):
        """Sends a signal to stamp all in-flight requests with the given generation epoch.

        Args:
            generation_epoch: The current generation epoch number.
        """
        self._send_signal_to_engines(Headers.SET_GENERATION_EPOCH, generation_epoch)

    def suspend_engines(self):
        """Sends SUSPEND to all engines via coordinator. Requires PAUSED.

        Callers should await engine.suspended for confirmation.
        """
        self._send_signal_to_engines(Headers.SUSPEND)

    def resume_engines(self):
        """Sends RESUME to all engines via coordinator. Requires SUSPENDED.

        Callers should await engine.paused (or engine.running after UNPAUSE) for confirmation.
        """
        self._send_signal_to_engines(Headers.RESUME)

    def stop_engines(self):
        """Sends STOP to all engines via coordinator. Requires PAUSED or SUSPENDED.

        Callers should await engine.stopped for confirmation.
        Does not affect the coordinator.
        """
        self._send_signal_to_engines(Headers.STOP)

    def shutdown_coordinator(self):
        """Tells the coordinator process to exit its main loop.

        Does not affect the engines.
        """
        self._send_signal_to_engines(Headers.SHUTDOWN)

    def stop(self):
        """
        Stops the client and cleans up all resources.

        This method cancels the background listener task, closes the ZMQ socket,
        and terminates the ZMQ context. It should be called when the client is
        no longer needed to ensure a graceful shutdown.
        """
        if self.listener_task is not None and not self.listener_task.done():
            self.listener_task.cancel()
        # Wake up any listeners.
        for future in self.completion_futures.values():
            if not future.done():
                future.cancel()
        self.completion_futures.clear()
        # Terminate any open streaming iterators.
        for stream in self.streams.values():
            stream.finish()
        self.streams.clear()
        for future in self.abort_futures.values():
            if not future.done():
                future.cancel()
        self.abort_futures.clear()
        self.aborted_request_ids.clear()
        self.socket.close(linger=0)
        self.context.term()
