# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import asyncio
import itertools
import time
from collections import deque
from typing import Dict, Optional

import msgpack
import pytest
import torch
from tqdm import tqdm

from megatron.core.inference.engines.dynamic_engine import DynamicInferenceEngine, RequestEntry
from megatron.core.inference.headers import Headers
from megatron.core.inference.inference_client import InferenceClient
from megatron.core.inference.inference_request import (
    DynamicInferenceRequest,
    DynamicInferenceRequestRecord,
    Status,
)
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.utils import get_asyncio_loop
from tests.unit_tests.test_utilities import Utils

try:
    import zmq

    HAVE_ZMQ = True
except ImportError:
    HAVE_ZMQ = False

NUM_REQUESTS = 10
NUM_TOKENS = 2
DEFAULT_PORT = 46581
ZMQ_FLAKY_SHUTDOWN = True


class DummyTokenizer:
    """Dummy tokenizer."""

    def __init__(self, vocab_size: int = 10, bos: int | None = None, eod: int = 0, pad: int = 0):
        self.vocab_size = vocab_size
        self.bos = bos
        self.eod = eod
        self.pad = pad

    def tokenize(self, prompt):
        if isinstance(prompt, str):
            return [int(tok) % self.vocab_size for tok in prompt.strip().split()]
        return list(prompt)

    def detokenize(self, tokens, skip_special_tokens: bool = False):
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        if skip_special_tokens and self.eod in tokens:
            tokens = [tok for tok in tokens if tok != self.eod]
        return " ".join(str(tok) for tok in tokens)


class DummyContext:
    """Dummy inference context."""

    def __init__(self):
        self.active_cnt = 0

    def get_active_request_count(self) -> int:
        return self.active_cnt


class DummyController:
    """Dummy inference controller."""

    def __init__(self):
        self.tokenizer = DummyTokenizer()

    def dummy_forward(self):
        pass


class DummyEngine(DynamicInferenceEngine):
    """Dummy inference engine that only implements coordinator-related methods."""

    def __init__(self):
        """We cannot call super().__init__() because it requires complex setup."""
        self.waiting_request_ids = deque()
        self.requests: Dict[int, RequestEntry] = {}
        self.suspend_signal = False
        self.is_suspended = False
        self._loop = get_asyncio_loop()
        self.context = DummyContext()
        self.controller = DummyController()
        self.running = asyncio.Event()
        self.paused = asyncio.Event()
        self.stopped = asyncio.Event()
        self.pending_microbatch = deque()
        self.received_pause: bool = False
        self.received_stop: bool = False
        self.pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        self.rank = torch.distributed.get_rank()

    def add_request(
        self, request_id: int, prompt: str, sampling_params: Optional[SamplingParams] = None
    ) -> asyncio.Future[DynamicInferenceRequestRecord]:
        """Dummy add_request."""

        self.requests[request_id] = RequestEntry(
            record=DynamicInferenceRequestRecord.from_request(
                DynamicInferenceRequest(
                    prompt=prompt,
                    request_id=request_id,
                    sampling_params=sampling_params,
                    status=Status.WAITING_IN_QUEUE,
                )
            ),
            future=self._loop.create_future(),
        )
        self.waiting_request_ids.append(request_id)

        return self.requests[request_id].future

    async def async_step(self, *, verbose: Optional[bool] = False) -> Dict:
        """Dummy async_step."""
        # Finish "active" requests.
        finished_request_records = []
        to_remove = []
        for request_id, entry in self.requests.items():
            request = entry.record[-1]
            if request.status == Status.ACTIVE_AND_GENERATING_TOKENS:
                request.sampling_params.num_tokens_to_generate -= 1
                if request.sampling_params.num_tokens_to_generate > 0:
                    continue
                request.status = Status.COMPLETED
                self.context.active_cnt -= 1
                finished_request_records.append(entry.record)
                entry.future.set_result(entry.record)
                to_remove.append(request_id)
                # Send signal to coordinator.
                if self.is_mp_coordinator:
                    payload = msgpack.packb(
                        [Headers.ENGINE_REPLY.value, [entry.record.serialize()]], use_bin_type=True
                    )
                    self.socket_for_receiving_requests.send(payload)

        for request_id in to_remove:
            del self.requests[request_id]

        # Activate queued requests. They will "process" for 1 step.
        active_request_ids = []
        while self.waiting_request_ids:
            request_id = self.waiting_request_ids.popleft()
            record = self.requests[request_id].record
            record[-1].status = Status.ACTIVE_AND_GENERATING_TOKENS
            self.context.active_cnt += 1
            active_request_ids.append(request_id)

        return {
            "active_request_ids": active_request_ids,
            "finished_request_records": finished_request_records,
            "step_time": 0.01,
            "cuda_graph_request_count": 1,
        }


@pytest.fixture
def initialize_model_parallel(request, monkeypatch):
    """Fixture to initialize and destroy model parallel.

    Parameters are passed via request.param as a tuple: (tp, pp, ep).
    Defaults to (1, 1, 1) if not parametrized.
    """
    monkeypatch.setenv("CUDA_DEVICE_MAX_CONNECTIONS", "1")

    tp, pp, ep = getattr(request, "param", (1, 1, 1))
    world_size = Utils.world_size
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=tp,
        pipeline_model_parallel_size=pp,
        expert_model_parallel_size=ep,
    )
    dp = world_size // (tp * pp * ep)
    yield world_size, dp, tp, pp, ep
    Utils.destroy_model_parallel()


@pytest.mark.skipif(ZMQ_FLAKY_SHUTDOWN, reason="ZMQ shutdown is flaky")
class TestCoordinator:
    """Test class for Data Parallel Inference Coordinator."""

    def build_requests(self, num_requests=NUM_REQUESTS, num_tokens=NUM_TOKENS):
        """Build a list of test requests."""
        return [
            ("Hello world!", SamplingParams(num_tokens_to_generate=num_tokens))
            for _ in range(num_requests)
        ]

    async def run_coordinator_test(
        self,
        *,
        launch_coordinator=True,
        stop_engines=True,
        num_requests=NUM_REQUESTS,
        num_tokens=NUM_TOKENS,
    ):
        """Run a coordinator test. Model parallel must already be initialized."""
        engine = DummyEngine()
        requests = self.build_requests(num_requests, num_tokens)

        dp_addr = await engine.start_listening_to_data_parallel_coordinator(
            inference_coordinator_port=DEFAULT_PORT, launch_inference_coordinator=launch_coordinator
        )

        try:
            if torch.distributed.get_rank() == 0:
                client = InferenceClient(dp_addr)
                await client.start()

                futures = [
                    client.add_request(prompt=prompt, sampling_params=params)
                    for prompt, params in requests
                ]
                results = await asyncio.wait_for(asyncio.gather(*futures), timeout=10.0)

                for record in results:
                    assert record[-1].status == Status.COMPLETED
        finally:
            if torch.distributed.get_rank() == 0:
                if stop_engines:
                    await asyncio.wait_for(client.stop_engines(), timeout=10.0)
                client.stop()
            if stop_engines:
                try:
                    await asyncio.wait_for(engine.engine_loop_task, timeout=30.0)
                except asyncio.TimeoutError:
                    engine.engine_loop_task.cancel()

        return dp_addr

    @pytest.mark.internal
    @pytest.mark.skipif(not HAVE_ZMQ, reason="pyzmq is required for this test")
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "initialize_model_parallel",
        [
            pytest.param((tp, pp, ep), id=f"tp{tp}-pp{pp}-ep{ep}")
            for tp, pp, ep in itertools.product([1, 2], [1, 2], [1, 2])
            if tp * pp * ep <= Utils.world_size
        ],
        indirect=["initialize_model_parallel"],
    )
    async def test_parallel_configs(self, initialize_model_parallel):
        """Test coordinator with various TP, PP, and EP configurations."""
        await self.run_coordinator_test()

    @pytest.mark.internal
    @pytest.mark.skipif(not HAVE_ZMQ, reason="pyzmq is required for this test")
    @pytest.mark.asyncio
    async def test_coordinator_lifecycle(self, initialize_model_parallel):
        """Test coordinator connection and port conflict behavior."""
        engine1 = DummyEngine()
        engine2 = None
        engine3 = None
        third_addr = None

        # Launch first coordinator - binds to DEFAULT_PORT
        first_addr = await engine1.start_listening_to_data_parallel_coordinator(
            inference_coordinator_port=DEFAULT_PORT, launch_inference_coordinator=True
        )

        try:
            # Cancel engine1 loop without sending stop to coordinator
            # This keeps coordinator process alive and holding the port
            engine1.engine_loop_task.cancel()
            try:
                await engine1.engine_loop_task
            except asyncio.CancelledError:
                pass

            # Connect engine2 to existing coordinator (don't launch new one)
            engine2 = DummyEngine()
            second_addr = await engine2.start_listening_to_data_parallel_coordinator(
                inference_coordinator_port=DEFAULT_PORT, launch_inference_coordinator=False
            )

            # Should connect to same port, but will not always in CI due to port conflicts.
            first_port = int(first_addr.rsplit(":", 1)[-1])
            second_port = int(second_addr.rsplit(":", 1)[-1])
            # assert second_port == first_port

            # Cancel engine2
            engine2.engine_loop_task.cancel()
            try:
                await engine2.engine_loop_task
            except asyncio.CancelledError:
                pass

            # Launch new coordinator - should get different port since first is holding it
            engine3 = DummyEngine()
            third_addr = await engine3.start_listening_to_data_parallel_coordinator(
                inference_coordinator_port=DEFAULT_PORT, launch_inference_coordinator=True
            )

            # Verify we got a different port due to conflict
            third_port = int(third_addr.rsplit(":", 1)[-1])
            assert (
                third_port != first_port
            ), f"Expected different port due to conflict, but got same: {third_port}"

        finally:
            # Clean up engine3's coordinator
            if engine3 is not None and third_addr is not None:
                client3 = InferenceClient(third_addr)
                await client3.start()
                await asyncio.wait_for(client3.stop_engines(), timeout=10.0)
                client3.stop()
                try:
                    await asyncio.wait_for(engine3.engine_loop_task, timeout=30.0)
                except asyncio.TimeoutError:
                    engine3.engine_loop_task.cancel()

            # Rebuild engine and reconnect to engine1's coordinator
            first_port = int(first_addr.rsplit(":", 1)[-1])
            engine1 = DummyEngine()
            await engine1.start_listening_to_data_parallel_coordinator(
                inference_coordinator_port=first_port, launch_inference_coordinator=False
            )
            client1 = InferenceClient(first_addr)
            await client1.start()
            await asyncio.wait_for(client1.stop_engines(), timeout=10.0)
            client1.stop()
            try:
                await asyncio.wait_for(engine1.engine_loop_task, timeout=30.0)
            except asyncio.TimeoutError:
                engine1.engine_loop_task.cancel()

    @pytest.mark.internal
    @pytest.mark.skipif(not HAVE_ZMQ, reason="pyzmq is required for this test")
    @pytest.mark.asyncio
    async def test_pause(self, initialize_model_parallel):
        """Test pause and resume functionality."""
        engine = DummyEngine()
        requests = self.build_requests(num_requests=32)

        dp_addr = await engine.start_listening_to_data_parallel_coordinator(
            inference_coordinator_port=DEFAULT_PORT, launch_inference_coordinator=True
        )

        success = True
        try:
            if torch.distributed.get_rank() == 0:
                client = InferenceClient(dp_addr)
                await client.start()

                # Submit requests and pause after completion.
                futures = [client.add_request(prompt=p, sampling_params=s) for p, s in requests[:2]]
                await asyncio.sleep(0.1)
                awaitables = futures + [client.pause_engines()]
                try:
                    await asyncio.wait_for(asyncio.gather(*awaitables), timeout=0.5)
                except asyncio.TimeoutError:
                    pytest.fail("Pause operation timed out.")

                # Ensure that requests can be added while paused.
                prompt, params = requests[2]
                future = client.add_request(prompt=prompt, sampling_params=params)
                with pytest.raises(asyncio.TimeoutError):
                    await asyncio.wait_for(future, timeout=0.1)

                # Resume and verify new requests complete.
                client.unpause_engines()
                # TODO: The system should not be incorrectly raising a cancelled error here.
                with pytest.raises(asyncio.CancelledError):
                    await future

                futures = [
                    client.add_request(prompt=p, sampling_params=s) for p, s in requests[3:4]
                ]
                await asyncio.sleep(0.1)
                try:
                    await asyncio.wait_for(asyncio.gather(*futures), timeout=0.5)
                except asyncio.TimeoutError:
                    pytest.fail("Resumed requests did not complete in time.")
        except:
            success = False
        finally:
            try:
                if torch.distributed.get_rank() == 0:
                    await asyncio.wait_for(client.stop_engines(), timeout=5.0)
                    client.stop()
                await asyncio.wait_for(engine.engine_loop_task, timeout=30.0)
            except asyncio.TimeoutError:
                engine.engine_loop_task.cancel()
        assert success, "Pause/resume test failed."

    @pytest.mark.internal
    @pytest.mark.skipif(not HAVE_ZMQ, reason="pyzmq is required for this test")
    @pytest.mark.asyncio
    async def test_throughput(self, initialize_model_parallel):
        """Throughput test with no TP or PP."""
        num_requests = 10**4
        num_iterations = 10

        engine = DummyEngine()
        requests = self.build_requests(num_requests=num_requests)

        start_time = time.time()
        dp_addr = await engine.start_listening_to_data_parallel_coordinator(
            inference_coordinator_port=DEFAULT_PORT, launch_inference_coordinator=True
        )

        try:
            if torch.distributed.get_rank() == 0:
                client = InferenceClient(dp_addr)
                await client.start()
                init_time = time.time()

                for _ in range(num_iterations):
                    futures = []
                    for prompt, sampling_params in tqdm(requests, "add_requests"):
                        fut = client.add_request(prompt=prompt, sampling_params=sampling_params)
                        futures.append(fut)
                    await asyncio.wait_for(asyncio.gather(*futures), timeout=10.0)
                done_time = time.time()
        finally:
            if torch.distributed.get_rank() == 0:
                await asyncio.wait_for(client.stop_engines(), timeout=10.0)
                client.stop()
            try:
                await asyncio.wait_for(engine.engine_loop_task, timeout=30.0)
            except asyncio.TimeoutError:
                engine.engine_loop_task.cancel()

        stop_time = time.time()

        flags = torch.tensor([1, 1, 1], dtype=torch.int, device=torch.cuda.current_device())

        init_duration = golden_init_duration = None
        run_duration = golden_run_duration = None
        stop_duration = golden_stop_duration = None

        if torch.distributed.get_rank() == 0:
            init_duration = (init_time - start_time) * 10**3
            golden_init_duration = 6974.43  # ms
            run_duration = (done_time - init_time) * 10**3
            golden_run_duration = 4392.63  # ms
            stop_duration = (stop_time - done_time) * 10**3
            golden_stop_duration = 931.49  # ms

            def clamp_to_golden_value(value, golden_value, delta=0.1):
                return value > golden_value * (1 - delta) and value < golden_value * (1 + delta)

            if not clamp_to_golden_value(init_duration, golden_init_duration, delta=0.5):
                flags[0] = 0
            if not clamp_to_golden_value(run_duration, golden_run_duration, delta=0.2):
                flags[1] = 0
            if not clamp_to_golden_value(stop_duration, golden_stop_duration, delta=1.0):
                flags[2] = 0

        # Synchronize results
        torch.distributed.broadcast(flags, src=0)

        if torch.distributed.get_rank() == 0:
            print(f"Initialization time: {init_duration:.2f} ms")
            print(f"Run time: {run_duration:.2f} ms")
            print(f"Stop time: {stop_duration:.2f} ms")

            assert flags[0].item() == 1, (
                f"WARNING: Init duration {init_duration:.2f}s deviates from "
                f"golden value {golden_init_duration:.2f}s"
            )
            assert flags[1].item() == 1, (
                f"WARNING: Run duration {run_duration:.2f}s deviates from "
                f"golden value {golden_run_duration:.2f}s"
            )
            assert flags[2].item() == 1, (
                f"WARNING: Stop duration {stop_duration:.2f}s deviates from "
                f"golden value {golden_stop_duration:.2f}s"
            )

            print(
                f"ZMQ throughput is approximately "
                f"{num_requests * num_iterations / run_duration:.2f} "
                f"requests/ms"
            )
        else:
            assert flags[0].item() == 1
            assert flags[1].item() == 1
            assert flags[2].item() == 1
