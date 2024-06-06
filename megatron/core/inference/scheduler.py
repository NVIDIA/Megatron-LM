import time
import typing
from collections import OrderedDict
from typing import Dict, List

import torch

from megatron.core.inference.common_inference_params import CommonInferenceParams
from megatron.core.inference.inference_request import InferenceRequest, Status
from megatron.core.inference.utils import Counter


class Scheduler:
    def __init__(self, max_batch_size: int):
        """Scheduler for handling requests to inference engine

        This class is responsible for handing of all the incomign requests

        Args:
            max_batch_size (int): The max batch size that we can pass to the inference engine at a time. 
        """
        self.max_batch_size = max_batch_size
        self.active_request_pool: Dict[int, InferenceRequest] = OrderedDict()
        self.waiting_request_pool: Dict[int, InferenceRequest] = OrderedDict()
        self.completed_request_pool: Dict[int, InferenceRequest] = OrderedDict()
        self.request_counter = Counter()

    def add_request(
        self,
        prompt: str,
        prompt_tokens: torch.Tensor,
        inference_parameters: CommonInferenceParams,
        arrival_time: float = None,
    ):
        """Add an incoming request

        This method will add the request to either the active pool or the waiting pool depending on the batch size. 

        Args:
            prompt (str): Input prompt string
            prompt_tokens (torch.Tensor): A torch tensor having the input prompts tokenized
            inference_parameters (CommonInferenceParams): The inference parameters
            arrival_time (float, optional): The incoming request time. Defaults to None.
        """
        request_id = str(next(self.request_counter))

        if arrival_time is None:
            arrival_time = time.time()

        status = (
            Status.ACTIVE_BUT_NOT_GENERATING_TOKENS
            if len(self.active_request_pool) < self.max_batch_size
            else Status.WAITING_IN_QUEUE
        )

        inference_request = InferenceRequest(
            request_id=request_id,
            prompt=prompt,
            inference_parameters=inference_parameters,
            arrival_time=arrival_time,
            prompt_tokens=prompt_tokens,
            status=status,
        )

        if status == status.ACTIVE_BUT_NOT_GENERATING_TOKENS:
            self.active_request_pool[request_id] = inference_request
        else:
            self.waiting_request_pool[request_id] = inference_request

    def have_requests_pending(self) -> bool:
        """Method to check if there are requests pending

        This method returns False only when there are no active requests or waiting requests. 
        """
        num_requests_pending = len(self.active_request_pool) + len(self.waiting_request_pool)
        return num_requests_pending > 0

    def add_earliest_waiting_request_to_active_pool(self):
        """Utility to add the waiting request to active pool

        This method will add the earliest request (FIFO) that is in the waiting request pool to the active request pool.
        """
        assert (
            len(self.active_request_pool) < self.max_batch_size
        ), "Active request pool is already full. Cant add any more requests"
        if len(self.waiting_request_pool) > 0:
            (
                earliest_waiting_request_request_id,
                earliest_waiting_request,
            ) = self.waiting_request_pool.popitem(last=False)
            earliest_waiting_request.status = Status.ACTIVE_BUT_NOT_GENERATING_TOKENS
            self.active_request_pool[earliest_waiting_request_request_id] = earliest_waiting_request

    def update_requests_pools(self, result_dict: typing.OrderedDict[int, InferenceRequest] = None):
        """Update request pool status

        This method will full up the active request pool, if it has less than max batch size elements from the waiting request pool. 
        If provided with a request dict, it will put the completed requests into the completed request pool and add waiting request into active pool.  

        Args:
            result (typing.OrderedDict[int, InferenceRequest], optional): The result returned by the engine. A dictionary with keys as the request ids, and values as the requests. Defaults to None
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
