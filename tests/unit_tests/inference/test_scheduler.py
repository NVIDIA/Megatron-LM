from typing import Dict

import torch

from megatron.core.inference.common_inference_params import CommonInferenceParams
from megatron.core.inference.inference_request import InferenceRequest, Status
from megatron.core.inference.scheduler import Scheduler


class TestScheduler:

    def setup_method(self, method):
        self.max_batch_size = 4
        self.scheduler = Scheduler(max_batch_size=self.max_batch_size)
        assert (
            len(self.scheduler.active_request_pool) == 0
        ), "Active request pool should be empty on initalization"
        assert (
            len(self.scheduler.waiting_request_pool) == 0
        ), "Waiting request pool should be empty on initalization"
        assert (
            len(self.scheduler.completed_request_pool) == 0
        ), "Completed request pool should be empty on initalization"

    def test_scheduler(self):
        prompt = "sample prompt"
        prompt_tokens = torch.randn(5)
        inference_parameters = CommonInferenceParams()

        for i in range(self.max_batch_size):
            self.scheduler.add_request(prompt, prompt_tokens, inference_parameters)
            assert (
                len(self.scheduler.active_request_pool) == i + 1
            ), f"Active request pool should have {i+1} requests, but it has only {len(self.scheduler.active_request_pool)}"

        self.scheduler.add_request(prompt, prompt_tokens, inference_parameters)
        assert (
            len(self.scheduler.waiting_request_pool) == 1
        ), f"Waiting request pool should have 1 request but it has {len(self.scheduler.waiting_request_pool)} requests"

        waiting_request: InferenceRequest = list(self.scheduler.waiting_request_pool.values())[0]
        assert (
            waiting_request.status == Status.WAITING_IN_QUEUE
        ), f"Status should be WAITING_IN_QUEUE, but its {waiting_request.status} for the waiting request"

        assert (
            self.scheduler.have_requests_pending()
        ), "Scheduler should have requests pending, but it seems to be having no requests"

        active_request_dict: Dict[int, InferenceRequest] = self.scheduler.active_request_pool
        for request_id, request in active_request_dict.items():
            # Mark every even request compelted
            if int(request_id) % 2 == 0:
                request.status = Status.COMPLETED

        self.scheduler.update_requests_pools(active_request_dict)
        assert (
            len(self.scheduler.active_request_pool) == 3
        ), f"Active request pool should have 3 requests, but it has {len(self.scheduler.active_request_pool)}"

        assert (
            len(self.scheduler.waiting_request_pool) == 0
        ), f"Waiting request pool should be empty but it has {len(self.scheduler.waiting_request_pool)} requests"

        assert (
            len(self.scheduler.completed_request_pool) == 2
        ), f"Completed request pool should have 2 requests but it has {len(self.scheduler.completed_request_pool)} requests "

        active_request_dict: Dict[int, InferenceRequest] = self.scheduler.active_request_pool
        for request_id, request in active_request_dict.items():
            # Mark all requests compelted
            request.status = Status.COMPLETED

        self.scheduler.update_requests_pools(active_request_dict)
        assert (
            len(self.scheduler.active_request_pool) == 0
        ), f"Active request pool should be empty, but it has {len(self.scheduler.active_request_pool)}"

        assert (
            len(self.scheduler.waiting_request_pool) == 0
        ), f"Waiting request pool should be empty but it has {len(self.scheduler.waiting_request_pool)} requests"

        assert (
            len(self.scheduler.completed_request_pool) == 5
        ), f"Completed request pool should have 5 requests but it has {len(self.scheduler.completed_request_pool)} requests "

        assert (
            self.scheduler.have_requests_pending() == False
        ), "Scheduler should not have any requests pending"
