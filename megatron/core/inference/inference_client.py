# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import asyncio
import logging
import time
from typing import Optional, Union

from megatron.core.inference.async_zmq_communicator import AsyncZmqEndpoint
from megatron.core.inference.inference_request import DynamicInferenceRequest
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.utils import get_asyncio_loop, trace_async_exceptions

from .headers import Headers


class InferenceClient(AsyncZmqEndpoint):
    """
    An asynchronous client for communicating with an inference coordinator service.

    This client submits inference requests, listens for completed results, and
    sends control signals to the inference engines through the coordinator.
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
        super().__init__("DEALER", connect=inference_coordinator_address)
        self.deserialize = deserialize

    def add_request(
        self, prompt: Union[str, list[int]], sampling_params: SamplingParams
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
            `DynamicInferenceRequest` object (if deserialize=True) or a raw
            serialized dict (if deserialize=False) containing the completed result.
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
        """Listen for replies from the coordinator and resolve futures."""
        while True:
            try:
                _, header, data = await self._irecv()

                if header == Headers.ENGINE_REPLY:
                    request_id, reply = data
                    reply['latency'] = time.perf_counter() - self.request_submission_times.pop(
                        request_id
                    )
                    completion_future = self.completion_futures.pop(request_id)
                    if completion_future.done():
                        logging.warning(f"Client: The future for {request_id} has been cancelled!")
                        continue
                    completed_request = (
                        DynamicInferenceRequest.deserialize(reply) if self.deserialize else reply
                    )
                    completion_future.set_result(completed_request)
                elif header == Headers.ACK:
                    pass  # Coordinator acknowledged our connect; no action needed.
            except asyncio.CancelledError:
                break

    def start(self, loop: Optional[asyncio.AbstractEventLoop] = None):
        """Connects to the coordinator and starts the background tasks."""
        logging.info("Client: Connecting to InferenceCoordinator...")

        self.completion_futures = {}
        self.request_submission_times = {}
        self.next_request_id = 0

        self._isend(Headers.CLIENT_CONNECT)
        super().start(loop)

    def pause_engines(self):
        """Sends PAUSE to all engines via coordinator.

        Callers should await engine.paused for confirmation.
        """
        self._isend(Headers.PAUSE)

    def unpause_engines(self):
        """Sends UNPAUSE to all engines."""
        self._isend(Headers.UNPAUSE)

    def increment_staleness(self):
        """Sends a signal to increment staleness on all in-flight requests."""
        self._isend(Headers.INCREMENT_STALENESS)

    def suspend_engines(self):
        """Sends SUSPEND to all engines via coordinator. Requires PAUSED.

        Callers should await engine.suspended for confirmation.
        """
        self._isend(Headers.SUSPEND)

    def resume_engines(self):
        """Sends RESUME to all engines via coordinator. Requires SUSPENDED.

        Callers should await engine.resumed for confirmation.
        """
        self._isend(Headers.RESUME)

    def stop_engines(self):
        """Sends STOP to all engines via coordinator. Requires PAUSED or SUSPENDED.

        Callers should await engine.stopped for confirmation.
        """
        self._isend(Headers.STOP)

    def shutdown_coordinator(self):
        """Tells the coordinator process to exit its main loop."""
        self._isend(Headers.SHUTDOWN)

    async def shutdown(self):
        """
        Stops the client and cleans up all resources.

        This method awaits background tasks, closes the ZMQ sockets,
        and terminates the ZMQ context. It should be called when the client is
        no longer needed to ensure a graceful shutdown.
        """
        await super().shutdown()
        # Wake up any listeners.
        for future in self.completion_futures.values():
            if not future.done():
                future.cancel()
        self.completion_futures.clear()
        self._ctx.term()
