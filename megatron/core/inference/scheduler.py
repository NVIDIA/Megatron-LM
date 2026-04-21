# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
import functools
import time
import typing
import warnings
from collections import OrderedDict
from typing import Dict, Optional, Type, Union

import torch

from megatron.core.inference.async_stream import AsyncStream
from megatron.core.inference.inference_request import InferenceRequest, Status
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.inference.utils import Counter


class Scheduler:
    """Scheduler for handling requests to inference engine

    This class is responsible for handing of all the incomign requests

    Args:
        max_batch_size (int): The max batch size that we can pass to the
            inference engine at a time.
        request_type (InferenceRequest): The class to use for instantiating new requests.
    """

    def __init__(self, max_batch_size):
        self.max_batch_size = max_batch_size
        self.requests: Dict[str, InferenceRequest] = OrderedDict()
        self.streams: Dict[str, AsyncStream] = OrderedDict()
        self.active_request_pool: Dict[str, InferenceRequest] = OrderedDict()
        self.waiting_request_pool: Dict[str, InferenceRequest] = OrderedDict()
        self.completed_request_pool: Dict[str, InferenceRequest] = OrderedDict()
        self.request_counter = Counter()

    def get_new_request_id(self) -> str:
        """Gets a new request id"""
        request_id = str(next(self.request_counter))
        return request_id

    def add_request(
        self,
        prompt: Optional[str] = None,
        prompt_tokens: Optional[torch.Tensor] = None,
        encoder_prompt: Optional[str] = None,
        sampling_params: Optional[SamplingParams] = None,
        arrival_time: Optional[float] = None,
        streaming: bool = False,
        inference_request: Optional[InferenceRequest] = None,
        *,
        inference_parameters: Optional[SamplingParams] = None,
    ) -> str:
        """Add an incoming request

        This method will add the request to either the active pool or the waiting pool
        depending on the batch size.

        Args:
            prompt (str): Input prompt string
            prompt_tokens (torch.Tensor): A torch tensor having the input prompts tokenized
            encoder_prompt (str): Encoder input string
            sampling_params (SamplingParams): The sampling parameters
            arrival_time (float, optional): The incoming request time. Defaults to None.
            streaming (bool, optional): Whether to asynchronously stream tokens for this request.
            inference_request (InferenceRequest, optional): A fully constructed request.
                Defaults to None.

        Returns:
            The request_id for the new request.
        """
        status = (
            Status.ACTIVE_BUT_NOT_GENERATING_TOKENS
            if len(self.active_request_pool) < self.max_batch_size
            else Status.WAITING_IN_QUEUE
        )

        # Deprecation warning for `inference_parameters`.
        if inference_parameters is not None:
            warnings.warn(
                "`inference_parameters` has been renamed to `sampling_params`, and the "
                "previous name will be removed in `megatron-core` 0.13."
            )
            if sampling_params is None:
                sampling_params = inference_parameters

        if inference_request is None:
            assert prompt is not None
            assert prompt_tokens is not None

            request_id = self.get_new_request_id()

            if arrival_time is None:
                arrival_time = time.time()

            inference_request = InferenceRequest(
                request_id=request_id,
                prompt=prompt,
                sampling_params=sampling_params,
                arrival_time=arrival_time,
                prompt_tokens=prompt_tokens,
                status=status,
                encoder_prompt=encoder_prompt,
            )
        else:
            request_id = inference_request.request_id
            inference_request.status = status
            if inference_request.arrival_time is None:
                inference_request.arrival_time = time.time()

        self.requests[request_id] = inference_request

        if streaming:
            abort_request = functools.partial(self.abort_request, request_id=request_id)
            self.streams[request_id] = AsyncStream(request_id, abort_request)

        if status == status.ACTIVE_BUT_NOT_GENERATING_TOKENS:
            self.active_request_pool[request_id] = inference_request
        else:
            self.waiting_request_pool[request_id] = inference_request

        return request_id

    def num_requests_pending(self) -> int:
        """Get the number of requests pending.

        This method returns the number of active + waiting requests.
        """
        return len(self.active_request_pool) + len(self.waiting_request_pool)

    def have_requests_pending(self) -> bool:
        """Method to check if there are requests pending.

        This method returns False only when there are no active requests or waiting requests.
        """
        return self.num_requests_pending() > 0

    def add_earliest_waiting_request_to_active_pool(self):
        """Utility to add the waiting request to active pool

        This method will add the earliest request (FIFO) that is in the waiting request
        pool to the active request pool.
        """
        assert (
            len(self.active_request_pool) < self.max_batch_size
        ), "Active request pool is already full. Cant add any more requests"
        if len(self.waiting_request_pool) > 0:
            (earliest_waiting_request_request_id, earliest_waiting_request) = (
                self.waiting_request_pool.popitem(last=False)
            )
            earliest_waiting_request.status = Status.ACTIVE_BUT_NOT_GENERATING_TOKENS
            self.active_request_pool[earliest_waiting_request_request_id] = earliest_waiting_request

    def update_requests_pools(
        self, result_dict: Optional[typing.OrderedDict[str, InferenceRequest]] = None
    ):
        """Update request pool status

        This method will full up the active request pool, if it has less than max batch size
        elements from the waiting request pool.
        If provided with a request dict, it will put the completed requests into the completed
        request pool and add waiting request into active pool.

        Args:
            result (typing.OrderedDict[str, InferenceRequest], optional): The result returned
                by the engine. A dictionary with keys as the request ids, and values as the
                requests. Defaults to None
        """
        for result_request_id in list(result_dict.keys()):
            active_request = self.active_request_pool[result_request_id]

            # If a request has completed put it into the completed request pool.
            if active_request.status == Status.COMPLETED:
                completed_request = self.active_request_pool.pop(result_request_id)
                self.completed_request_pool[result_request_id] = completed_request

        # If the active request pool is not full, add waiting requests in FIFO order
        while (
            len(self.active_request_pool) < self.max_batch_size
            and len(self.waiting_request_pool) > 0
        ):
            self.add_earliest_waiting_request_to_active_pool()

    def abort_request(
        self,
        request_id: str,
        *,
        exception: Optional[Union[BaseException, Type[BaseException]]] = None,
    ):
        """Cancels the given request"""
        stream = self.streams.get(request_id, None)
        if stream is not None:
            stream.finish(exception=exception)
