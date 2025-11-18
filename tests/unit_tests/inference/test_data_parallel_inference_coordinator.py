# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import asyncio
import random
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pytest
import torch.distributed as dist
from tqdm import tqdm

from megatron.core.inference.data_parallel_inference_coordinator import (
    DataParallelInferenceCoordinator,
)
from megatron.core.inference.engines.dynamic_engine import DynamicInferenceEngine
from megatron.core.inference.inference_client import InferenceClient
from megatron.core.inference.inference_request import DynamicInferenceRequest, Status
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.utils import get_asyncio_loop
from tests.unit_tests.test_utilities import Utils

try:
    import zmq

    HAVE_ZMQ = True
except Exception:
    HAVE_ZMQ = False


class DummyContext:
    """Dummy inference context."""

    def __init__(self):
        self.active_cnt = 0

    def get_active_request_count(self) -> int:
        return self.active_cnt


class DummyEngine(DynamicInferenceEngine):
    """Dummy inference engine that only implements coordinator-related methods."""

    def __init__(self):
        """We cannot call super().__init__() because it requires complex setup."""
        self.waiting_request_ids = deque()
        self.requests: Dict[int, DynamicInferenceRequest] = {}
        self.request_completion_futures: Dict[int, asyncio.Future] = {}
        self.paused = False
        self.stopped = False
        self._loop = get_asyncio_loop()
        self.context = DummyContext()

    def add_request(
        self, request_id: int, prompt: str, sampling_params: Optional[SamplingParams] = None
    ) -> asyncio.Future[DynamicInferenceRequest]:
        """Dummy add_request."""

        self.requests[request_id] = DynamicInferenceRequest(
            prompt=prompt,
            request_id=request_id,
            sampling_params=sampling_params,
            status=Status.WAITING_IN_QUEUE,
        )
        self.waiting_request_ids.append(request_id)

        fut = self._loop.create_future()
        self.request_completion_futures[request_id] = fut
        return fut

    async def async_step(
        self, *, verbose: Optional[bool] = False
    ) -> Tuple[List[DynamicInferenceRequest], List[DynamicInferenceRequest], float]:
        """Dummy async_step."""
        # Finish "active" requests.
        finished_requests = []
        to_remove = []
        for request_id, request in self.requests.items():
            if request.status == Status.ACTIVE_AND_GENERATING_TOKENS:
                request.status = Status.COMPLETED
                self.context.active_cnt -= 1
                finished_requests.append(request)
                self.request_completion_futures[request_id].set_result(request)
                to_remove.append(request_id)
        for request_id in to_remove:
            del self.requests[request_id]

        # Activate queued requests. They will "process" for 1 step.
        active_requests = []
        while self.waiting_request_ids:
            request_id = self.waiting_request_ids.popleft()
            request = self.requests[request_id]
            request.status = Status.ACTIVE_AND_GENERATING_TOKENS
            self.context.active_cnt += 1
            active_requests.append(request)

        return {
            "active_requests": active_requests,
            "finished_requests": finished_requests,
            "step_time": 0.01,
            "cuda_graph_request_count": 1,
        }


@dataclass
class CoordinatorTestConfig:
    """Test configuration args."""

    port: int = 46581

    num_requests: int = 10**1
    min_time_offset: float = 10 ** (-4)
    max_time_offset: float = 10 ** (-3)

    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1


@dataclass
class CoordinatorTestEnv:
    """Test environment, including requests."""

    config: CoordinatorTestConfig
    requests: List[Tuple]
    engine: DummyEngine


class TestCoordinator:

    @classmethod
    def _build_requests(cls, test_config: CoordinatorTestConfig) -> List[Tuple]:
        ret = []

        for _ in range(test_config.num_requests):
            arrival_delta = random.uniform(test_config.min_time_offset, test_config.max_time_offset)
            ret.append(("Hello world!", SamplingParams(), arrival_delta))
        return ret

    @classmethod
    def _build_test_env(cls, test_config):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=test_config.tensor_model_parallel_size,
            pipeline_model_parallel_size=test_config.pipeline_model_parallel_size,
        )
        requests = cls._build_requests(test_config)
        engine = DummyEngine()
        return CoordinatorTestEnv(config=test_config, requests=requests, engine=engine)

    @classmethod
    async def _run_test(cls, **test_config_kwargs):
        # Test environment.
        test_config = CoordinatorTestConfig(**test_config_kwargs)
        env = cls._build_test_env(test_config)

        await env.engine.start_listening_to_data_parallel_coordinator(
            inference_coordinator_port=test_config.port, launch_inference_coordinator=True
        )

        if dist.get_rank() == 0:
            client = InferenceClient(test_config.port)
            await client.start()
            futures = []
            for request in tqdm(env.requests, "add_requests"):
                prompt, sampling_params, arrival_delta = request
                await asyncio.sleep(arrival_delta)
                fut = client.add_request(prompt=prompt, sampling_params=sampling_params)
                futures.append(fut)
            results: List[DynamicInferenceRequest] = await asyncio.gather(*futures)

            client.stop_engines()
            client.stop()

        await env.engine.engine_loop_task

        return env

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.internal
    @pytest.mark.skipif(not HAVE_ZMQ, reason="pyzmq is required for this test")
    @pytest.mark.asyncio
    async def test_simple(self):
        """Simple test with no TP or PP."""
        env = await self._run_test(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)

    @pytest.mark.internal
    @pytest.mark.skipif(not HAVE_ZMQ, reason="pyzmq is required for this test")
    @pytest.mark.asyncio
    async def test_tp(self):
        """Simple test with no TP or PP."""
        env = await self._run_test(tensor_model_parallel_size=2, pipeline_model_parallel_size=1)

    @pytest.mark.internal
    @pytest.mark.skipif(not HAVE_ZMQ, reason="pyzmq is required for this test")
    @pytest.mark.asyncio
    async def test_throughput(self):
        """Throughput test with no TP or PP."""
        start = time.time()
        env = await self._run_test(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            num_requests=10**3,
            min_time_offset=0.0,
            max_time_offset=0.0,
        )
        end = time.time()
        if dist.get_rank() == 0:
            print(f"Throughput test time: {end - start} seconds.")


if __name__ == "__main__":
    test = TestCoordinator()
    test.test_simple()
    test.test_tp()
    test.teardown_method(None)
    print("~~~")
    print("success.")
