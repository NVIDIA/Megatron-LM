# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import asyncio
import functools
import logging
import time
from typing import List, Optional, Union

import torch

from megatron.core.inference.async_stream import AsyncStream
from megatron.core.inference.config import routes_on_prefix
from megatron.core.inference.inference_request import (
    DynamicInferenceRequest,
    compute_block_hashes_batched,
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

    def __init__(
        self,
        inference_coordinator_address: str,
        deserialize: bool = False,
        block_size_tokens: Optional[int] = None,
        prefix_caching_coordinator_policy=None,
    ):
        """
        Initializes the InferenceClient.

        Args:
            inference_coordinator_address (str): The address on which the
                inference coordinator is listening.
            deserialize (bool): If True, deserialize completed requests
                into DynamicInferenceRequest objects. If False (default), return
                the raw serialized dict for lower overhead.
            block_size_tokens (Optional[int]): Token block size to hash prompts
                on. Must match the engine's KV block size, or the hashes name
                blocks the engine never cached. None disables client-side
                hashing, leaving it to the coordinator.
            prefix_caching_coordinator_policy: The coordinator's routing policy,
                which decides whether anyone reads the hashes at all.
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
        self.block_size_tokens = block_size_tokens
        self.prefix_caching_coordinator_policy = prefix_caching_coordinator_policy

    def _block_hashes(self, prompt, serialized_multi_modal_data):
        """Hash the prompt into per-block routing hashes.

        Hashing here rather than at the coordinator is the point: the tokens are
        already in hand, clients are many against the coordinator's one serial
        loop, and hashing there would mean decoding the prompt frame that the
        metadata/body split exists to avoid.

        Multimodal prompts are salted with the media key so that equal token
        placeholders backed by different media cannot share KV. The key is reused
        from the already-serialized media rather than recomputed: deriving it
        digests the media itself, which runs to hundreds of MB for video.

        Returns None when this client cannot hash -- a string prompt, which needs
        a tokenizer it does not have. The coordinator hashes those itself, so
        None means "unhashed", distinct from an empty list meaning "no complete
        blocks".
        """
        if self.block_size_tokens is None or not routes_on_prefix(
            self.prefix_caching_coordinator_policy
        ):
            return []
        if isinstance(prompt, str):
            return None
        tokens = prompt.tolist() if isinstance(prompt, torch.Tensor) else list(prompt)
        cache_salt = (
            serialized_multi_modal_data.get("media_cache_key")
            if isinstance(serialized_multi_modal_data, dict)
            else None
        )
        return compute_block_hashes_batched(
            torch.tensor(tokens, dtype=torch.int64), self.block_size_tokens, cache_salt=cache_salt
        )

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
        request_id = self.next_request_id
        self.next_request_id += 1
        # Serialized once and used twice: it goes on the wire, and its media key
        # salts the hashes. Deriving that key digests the media itself, so doing
        # it a second time would mean re-hashing up to hundreds of MB of video.
        serialized_multi_modal_data = serialize_multimodal_data(multi_modal_data)
        frames = [
            msgpack.packb(
                [
                    Headers.SUBMIT_REQUEST.value,
                    request_id,
                    sampling_params.serialize(),
                    serialized_multi_modal_data,
                ],
                use_bin_type=True,
            ),
            # The prompt travels in its own frame so the coordinator can route the
            # request without decoding it -- at long prompts that decode dominates
            # its per-request cost, and it is one serial loop shared by every rank.
            #
            # Multimodal data stays in the metadata frame: the coordinator reads
            # `media_cache_key` out of it for routing. It is the one field here not
            # bounded by construction, so it could later move to its own body frame,
            # but that is a change to multimodal routing rather than to the wire
            # format, and is left alone here.
            self._pack_prompt(prompt),
            msgpack.packb(
                self._block_hashes(prompt, serialized_multi_modal_data), use_bin_type=True
            ),
        ]
        return self._submit_request(frames, request_id)

    @staticmethod
    def _pack_prompt(prompt):
        """Pack a prompt into its own frame.

        Coercion happens here rather than at the coordinator: the coordinator no
        longer decodes the prompt, and clients are many against its one serial
        loop, so this is both the only place that still sees the object and the
        cheaper place to normalize it.
        """
        if isinstance(prompt, torch.Tensor):
            prompt = prompt.tolist()
        elif not isinstance(prompt, (str, list)):
            raise TypeError(f"unsupported prompt type {type(prompt).__name__}")
        return msgpack.packb(prompt, use_bin_type=True)

    def _make_kv_handoff_request(
        self,
        prompt: Union[str, List[int]],
        sampling_params: SamplingParams,
        kv_meta: dict,
        src_block_ids: List[int],
    ) -> tuple[int, list]:
        """Allocate an ID and build a decode request carrying remote KV metadata.

        Framed as [metadata, prompt, src_block_ids].

        Nothing whose size follows the sequence length belongs in the metadata
        frame, since that frame is the only one the coordinator decodes.
        ``src_block_ids`` names one block per block_size_tokens of prompt, so it
        grows with the prompt and travels as its own body. ``kv_meta`` stays in
        the metadata: it is the peer's NIXL agent/layout export, bounded by TP
        size and num_speculative_tokens, not by prompt length.
        """
        request_id = self.next_request_id
        self.next_request_id += 1
        frames = [
            msgpack.packb(
                [
                    Headers.SUBMIT_REQUEST_WITH_KV.value,
                    request_id,
                    sampling_params.serialize(),
                    kv_meta,
                ],
                use_bin_type=True,
            ),
            self._pack_prompt(prompt),
            msgpack.packb(list(src_block_ids), use_bin_type=True),
        ]
        return request_id, frames

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
        request_id, frames = self._make_kv_handoff_request(
            prompt, sampling_params, kv_meta, src_block_ids
        )
        return self._submit_request(frames, request_id)

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
        request_id, frames = self._make_kv_handoff_request(
            prompt, sampling_params, kv_meta, src_block_ids
        )
        return self._submit_stream(frames, request_id)

    def release_handoff(self, request_id: int) -> None:
        """Tell the coordinator to release the KV blocks pinned for `request_id`.

        Fire-and-forget. The coordinator broadcasts RELEASE_KV to every engine;
        engines without that request_id ignore the message.
        """
        payload = [Headers.RELEASE_KV.value, int(request_id)]
        self.socket.send(msgpack.packb(payload, use_bin_type=True))

    def abort_request(self, request_id: int) -> None:
        """Cancel an in-flight request and close its local response stream."""
        request_id = int(request_id)
        self.aborted_request_ids.add(request_id)
        stream = self.streams.pop(request_id, None)
        if stream is not None:
            stream.finish()
        future = self.completion_futures.pop(request_id, None)
        if future is not None and not future.done():
            future.cancel()
        self.request_submission_times.pop(request_id, None)
        payload = [Headers.ABORT_REQUEST.value, request_id]
        self.socket.send(msgpack.packb(payload, use_bin_type=True))

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
        # Serialized once and used twice: it goes on the wire, and its media key
        # salts the hashes. Deriving that key digests the media itself, so doing
        # it a second time would mean re-hashing up to hundreds of MB of video.
        serialized_multi_modal_data = serialize_multimodal_data(multi_modal_data)
        frames = [
            msgpack.packb(
                [
                    Headers.SUBMIT_REQUEST.value,
                    request_id,
                    sampling_params.serialize(),
                    serialized_multi_modal_data,
                ],
                use_bin_type=True,
            ),
            # The prompt travels in its own frame so the coordinator can route the
            # request without decoding it -- at long prompts that decode dominates
            # its per-request cost, and it is one serial loop shared by every rank.
            #
            # Multimodal data stays in the metadata frame: the coordinator reads
            # `media_cache_key` out of it for routing. It is the one field here not
            # bounded by construction, so it could later move to its own body frame,
            # but that is a change to multimodal routing rather than to the wire
            # format, and is left alone here.
            self._pack_prompt(prompt),
            msgpack.packb(
                self._block_hashes(prompt, serialized_multi_modal_data), use_bin_type=True
            ),
        ]
        return self._submit_stream(frames, request_id)

    def _submit_request(self, frames: list, request_id: int) -> asyncio.Future:
        """Send a prepared request and register its completion future."""
        self.socket.send_multipart(frames)
        assert request_id not in self.completion_futures
        future = asyncio.get_running_loop().create_future()
        self.completion_futures[request_id] = future
        self.request_submission_times[request_id] = time.perf_counter()
        return future

    def _submit_stream(self, frames: list, request_id: int) -> AsyncStream[dict]:
        """Send a prepared streaming request and register its response stream."""
        self.socket.send_multipart(frames)
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
                # frames[0] is metadata; a reply body, when present, follows it.
                frames = self.socket.recv_multipart(flags=zmq.NOBLOCK)
                data = msgpack.unpackb(frames[0], raw=False)
                header = Headers(data[0])
                if header == Headers.ENGINE_REPLY:
                    request_id = data[1]
                    if request_id in self.aborted_request_ids:
                        self.aborted_request_ids.discard(request_id)
                        continue
                    reply = msgpack.unpackb(frames[1], raw=False)
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
                    completion_future = self.completion_futures.pop(request_id)
                    if completion_future.done():
                        logging.warning(f"Client: The future for {request_id} has been cancelled!")
                        continue
                    completed_request = (
                        DynamicInferenceRequest.deserialize(reply) if self.deserialize else reply
                    )
                    completion_future.set_result(completed_request)
                elif header == Headers.ENGINE_REPLY_PARTIAL:
                    request_id = data[1]
                    stream = self.streams.get(request_id)
                    if stream is not None:
                        stream.put({"partial": msgpack.unpackb(frames[1], raw=False)})
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
        reply = msgpack.unpackb(self.socket.recv_multipart()[0], raw=False)
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
        if hasattr(self, 'listener_task') and not self.listener_task.done():
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
        self.aborted_request_ids.clear()
        self.socket.close(linger=0)
        self.context.term()
