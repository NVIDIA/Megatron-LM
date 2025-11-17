# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import asyncio
import random
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pytest
import torch.distributed as dist
from tqdm import tqdm

from megatron.core.inference.data_parallel_inference_coordinator import (
    DataParallelInferenceCoordinator,
)
from megatron.core.inference.engines.dynamic_engine import DynamicInferenceEngine
from megatron.core.inference.inference_client import InferenceClient
from megatron.core.inference.inference_request import (
    DynamicInferenceRequest,
    DynamicInferenceRequestRecord,
    Status,
)
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.utils import get_asyncio_loop
from tests.unit_tests.test_utilities import Utils

try:
    import zmq

    HAVE_ZMQ = True
except Exception:
    HAVE_ZMQ = False

IS_ZMQ_FLAKY = False


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
        self.request_records: Dict[int, DynamicInferenceRequestRecord] = {}
        self.request_completion_futures: Dict[int, asyncio.Future] = {}
        self.paused = False
        self.stopped = False
        self.suspend_signal = False
        self.is_suspended = False
        self._loop = get_asyncio_loop()
        self.context = DummyContext()

    def add_request(
        self, request_id: int, prompt: str, sampling_params: Optional[SamplingParams] = None
    ) -> asyncio.Future[DynamicInferenceRequestRecord]:
        """Dummy add_request."""

        self.request_records[request_id] = DynamicInferenceRequestRecord.from_request(
            DynamicInferenceRequest(
                prompt=prompt,
                request_id=request_id,
                sampling_params=sampling_params,
                status=Status.WAITING_IN_QUEUE,
            )
        )
        self.waiting_request_ids.append(request_id)

        fut = self._loop.create_future()
        self.request_completion_futures[request_id] = fut
        return fut

    async def async_step(self, *, verbose: Optional[bool] = False) -> Dict:
        """Dummy async_step."""
        # Finish "active" requests.
        finished_request_records = []
        to_remove = []
        for request_id, record in self.request_records.items():
            if record[-1].status == Status.ACTIVE_AND_GENERATING_TOKENS:
                record[-1].status = Status.COMPLETED
                self.context.active_cnt -= 1
                finished_request_records.append(record)
                self.request_completion_futures[request_id].set_result(record)
                to_remove.append(request_id)
        for request_id in to_remove:
            del self.request_records[request_id]

        # Activate queued requests. They will "process" for 1 step.
        active_request_ids = []
        while self.waiting_request_ids:
            request_id = self.waiting_request_ids.popleft()
            record = self.request_records[request_id]
            record[-1].status = Status.ACTIVE_AND_GENERATING_TOKENS
            self.context.active_cnt += 1
            active_request_ids.append(request_id)

        return {
            "active_request_ids": active_request_ids,
            "finished_request_records": finished_request_records,
            "step_time": 0.01,
            "cuda_graph_request_count": 1,
        }


@dataclass
class CoordinatorTestConfig:
    """Test configuration args."""

    port: int = 46581
    launch_inference_coordinator: bool = True
    stop_engines: bool = True
    verify_results: bool = True

    num_requests: int = 10**1
    min_time_offset: float = 10 ** (-4)
    max_time_offset: float = 10 ** (-3)
    num_iterations: int = 1

    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1


@dataclass
class CoordinatorTestEnv:
    """Test environment, including requests."""

    config: CoordinatorTestConfig
    requests: List[Tuple]
    engine: DummyEngine
    responses: List[List[DynamicInferenceRequest]] = field(default_factory=list)
    timing_data: Dict[str, Optional[float]] = field(
        default_factory=lambda: {
            "start_time": None,
            "init_time": None,
            "done_time": None,
            "stop_time": None,
        }
    )


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

        # Connect each engine to their respective processes.
        env.timing_data["start_time"] = time.time()
        await env.engine.start_listening_to_data_parallel_coordinator(
            inference_coordinator_port=test_config.port,
            launch_inference_coordinator=test_config.launch_inference_coordinator,
        )

        if dist.get_rank() == 0:
            client = InferenceClient(test_config.port)
            await client.start()
            env.timing_data["init_time"] = time.time()

            all_results = []
            for _ in range(test_config.num_iterations):
                futures = []
                for request in tqdm(env.requests, "add_requests"):
                    prompt, sampling_params, arrival_delta = request
                    await asyncio.sleep(arrival_delta)
                    fut = client.add_request(prompt=prompt, sampling_params=sampling_params)
                    futures.append(fut)
                results: List[DynamicInferenceRequestRecord] = await asyncio.gather(*futures)
                all_results.append(results)
            env.timing_data["done_time"] = time.time()

            if test_config.stop_engines:
                client.stop_engines()
            client.stop()

        if test_config.stop_engines:
            await env.engine.engine_loop_task
        env.timing_data["stop_time"] = time.time()

        if dist.get_rank() == 0:
            env.responses = all_results
            if test_config.verify_results:
                for batch in all_results:
                    for result in batch:
                        assert result.status == Status.COMPLETED

        return env

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.internal
    @pytest.mark.skipif(IS_ZMQ_FLAKY, reason="pyzmq is flaky in CI")
    @pytest.mark.skipif(not HAVE_ZMQ, reason="pyzmq is required for this test")
    @pytest.mark.asyncio
    async def test_simple(self):
        """Simple test with no TP or PP."""
        env = await self._run_test(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)

    @pytest.mark.internal
    @pytest.mark.skipif(IS_ZMQ_FLAKY, reason="pyzmq is flaky in CI")
    @pytest.mark.skipif(not HAVE_ZMQ, reason="pyzmq is required for this test")
    @pytest.mark.asyncio
    async def test_tp(self):
        """Simple test with TP, but no PP."""
        env = await self._run_test(tensor_model_parallel_size=2, pipeline_model_parallel_size=1)

    @pytest.mark.internal
    @pytest.mark.skipif(IS_ZMQ_FLAKY, reason="pyzmq is flaky in CI")
    @pytest.mark.skipif(not HAVE_ZMQ, reason="pyzmq is required for this test")
    @pytest.mark.asyncio
    async def test_pp(self):
        """Simple test with no TP, but PP."""
        env = await self._run_test(tensor_model_parallel_size=1, pipeline_model_parallel_size=2)

    @pytest.mark.internal
    @pytest.mark.skipif(IS_ZMQ_FLAKY, reason="pyzmq is flaky in CI")
    @pytest.mark.skipif(not HAVE_ZMQ, reason="pyzmq is required for this test")
    @pytest.mark.asyncio
    async def test_tp_pp(self):
        """Simple test with both TP and PP."""
        env = await self._run_test(tensor_model_parallel_size=2, pipeline_model_parallel_size=2)

    @pytest.mark.internal
    @pytest.mark.skipif(IS_ZMQ_FLAKY, reason="pyzmq is flaky in CI")
    @pytest.mark.skipif(not HAVE_ZMQ, reason="pyzmq is required for this test")
    @pytest.mark.asyncio
    async def test_throughput(self):
        """Throughput test with no TP or PP."""
        env = await self._run_test(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            num_requests=10**4,
            num_iterations=10,
            min_time_offset=0.0,
            max_time_offset=0.0,
        )
        if dist.get_rank() == 0:
            init_duration = (env.timing_data["init_time"] - env.timing_data["start_time"]) * 10**3
            golden_init_duration = 4445.64  # ms
            run_duration = (env.timing_data["done_time"] - env.timing_data["init_time"]) * 10**3
            golden_run_duration = 3088.87  # ms
            stop_duration = (env.timing_data["stop_time"] - env.timing_data["done_time"]) * 10**3
            golden_stop_duration = 129.57  # ms

            # Print current results.
            print(f"Initialization time: {init_duration:.2f} ms")
            print(f"Run time: {run_duration:.2f} ms")
            print(f"Stop time: {stop_duration:.2f} ms")

            # Check against golden values.
            def clamp_to_golden_value(value, golden_value, delta=0.1):
                return value > golden_value * (1 - delta) and value < golden_value * (1 + delta)

            assert clamp_to_golden_value(init_duration, golden_init_duration, delta=0.5), (
                f"WARNING: Init duration {init_duration:.2f}s deviates from "
                f"golden value {golden_init_duration:.2f}s"
            )
            assert clamp_to_golden_value(run_duration, golden_run_duration, delta=0.2), (
                f"WARNING: Run duration {run_duration:.2f}s deviates from "
                f"golden value {golden_run_duration:.2f}s"
            )
            assert clamp_to_golden_value(stop_duration, golden_stop_duration, delta=0.3), (
                f"WARNING: Stop duration {stop_duration:.2f}s deviates from "
                f"golden value {golden_stop_duration:.2f}s"
            )

            # Print summary.
            print(
                f"ZMQ throughput is approximately "
                f"{env.config.num_requests * env.config.num_iterations / (run_duration):.2f} "
                f"requests/ms"
            )


if __name__ == "__main__":
    test = TestCoordinator()
    asyncio.run(test.test_simple())
    test.test_tp()
    test.test_pp()
    test_test.tp_pp()
    test_test.throughput()
    test.teardown_method(None)
    print("~~~")
    print("success.")
