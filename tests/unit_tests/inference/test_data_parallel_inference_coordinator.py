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

from megatron.core.inference.engines.dynamic_engine import (
    DynamicInferenceEngine,
    EngineState,
    RequestEntry,
)
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
        self._loop = get_asyncio_loop()
        self.context = DummyContext()
        self.controller = DummyController()
        self.pending_microbatch = deque()
        self.pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        self.rank = torch.distributed.get_rank()

        # State machine (mirrors dynamic_engine.py reset()).
        self.state = EngineState.RUNNING
        self.running = asyncio.Event()
        self.running.set()
        self.paused = asyncio.Event()
        self.suspended = asyncio.Event()
        self.stopped = asyncio.Event()
        self._pending_signals = deque()
        self.resume_request_ids = None
        self.use_coordinator = False

        # Default for EP world size (overwritten during
        # start_listening_to_data_parallel_coordinator).
        self.ep_world_size = 1

        # CUDA events used by run_engine_with_coordinator for timing.
        self.step_start_event = torch.cuda.Event(enable_timing=True)
        self.step_end_event = torch.cuda.Event(enable_timing=True)
        self.step_count = 0

    async def run_engine_with_coordinator(self, *, loop=None):
        """Override to bypass @trace_async_exceptions for testability.

        In production, @trace_async_exceptions converts AssertionError to
        sys.exit(1) -> SystemExit.  In Python 3.12+, asyncio re-raises
        SystemExit from tasks in the main thread, killing the process.
        For tests, we let AssertionErrors propagate directly so
        pytest.raises can catch them.
        """
        return await DynamicInferenceEngine.run_engine_with_coordinator.__wrapped__(self, loop=loop)

    def suspend(self):
        pass

    def resume(self):
        pass

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


async def graceful_shutdown(engine, client=None, *, timeout=10.0):
    """Shut down the engine.

    First we try to pass PAUSE -> STOP into the coordinator via the client.
    If necessary, we escalate to engine.shutdown(immediate=True) and cancel all tasks.

    Args:
        engine: DummyEngine whose engine_loop_task to await.
        client: InferenceClient (only rank 0, None on other ranks).
        timeout: Per-stage timeout in seconds.
    """
    try:
        if client is not None:
            client.pause_engines()
            client.stop_engines()
        await asyncio.wait_for(engine.stopped.wait(), timeout=timeout)
    except asyncio.TimeoutError:
        await asyncio.wait_for(engine.shutdown(immediate=True), timeout=timeout)
    finally:
        proc = getattr(engine, 'inference_coordinator_process', None)
        if proc is not None and proc.is_alive():
            proc.terminate()
            proc.join(timeout=timeout)
        if client is not None:
            client.stop()


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
    dp = world_size // (tp * pp)
    yield world_size, dp, tp, pp, ep
    Utils.destroy_model_parallel()


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

        client = None
        try:
            if torch.distributed.get_rank() == 0:
                client = InferenceClient(dp_addr)
                client.start()

                futures = [
                    client.add_request(prompt=prompt, sampling_params=params)
                    for prompt, params in requests
                ]
                results = await asyncio.wait_for(asyncio.gather(*futures), timeout=10.0)

                for record in results:
                    assert record[-1].status == Status.COMPLETED
        finally:
            if stop_engines:
                await graceful_shutdown(engine, client)
            elif client is not None:
                client.stop()

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
        """Test coordinator connection, port conflicts, de-registration, and re-registration."""
        requests = self.build_requests(num_requests=4)
        engines = []
        client1 = None

        try:
            # Launch first coordinator - binds to DEFAULT_PORT
            engine1 = DummyEngine()
            engines.append(engine1)
            first_addr = await engine1.start_listening_to_data_parallel_coordinator(
                inference_coordinator_port=DEFAULT_PORT, launch_inference_coordinator=True
            )

            # Verify the engine can serve requests.
            if torch.distributed.get_rank() == 0:
                client1 = InferenceClient(first_addr)
                client1.start()

                futures = [
                    client1.add_request(prompt=p, sampling_params=s) for p, s in requests[:2]
                ]
                results = await asyncio.wait_for(asyncio.gather(*futures), timeout=5.0)
                for record in results:
                    assert record[-1].status == Status.COMPLETED

                # Pause the engines to force a synchronization point.
                client1.pause_engines()
            await asyncio.wait_for(engine1.paused.wait(), timeout=5.0)
            if torch.distributed.get_rank() == 0:
                client1.unpause_engines()
            await asyncio.wait_for(engine1.running.wait(), timeout=5.0)

            # Disconnect the engine (simulates unexpected departure).
            await engine1.shutdown(immediate=True)

            # Re-register a new engine with the same coordinator.
            engine2 = DummyEngine()
            engines.append(engine2)
            await engine2.start_listening_to_data_parallel_coordinator(
                inference_coordinator_port=DEFAULT_PORT, launch_inference_coordinator=False
            )

            # Verify the new engine can serve requests.
            if torch.distributed.get_rank() == 0:
                futures = [
                    client1.add_request(prompt=p, sampling_params=s) for p, s in requests[2:4]
                ]
                results = await asyncio.wait_for(asyncio.gather(*futures), timeout=5.0)
                for record in results:
                    assert record[-1].status == Status.COMPLETED

                # Pause the engines to force a synchronization point.
                client1.pause_engines()
            await asyncio.wait_for(engine2.paused.wait(), timeout=5.0)
            if torch.distributed.get_rank() == 0:
                client1.unpause_engines()
            await asyncio.wait_for(engine2.running.wait(), timeout=5.0)

            # Disconnect the engine (simulates unexpected departure).
            await engine2.shutdown(immediate=True)
            if client1 is not None:
                client1.stop()
                client1 = None

            # Spin up a new coordinator; verify it gets a new port.
            engine3 = DummyEngine()
            engines.append(engine3)
            third_addr = await engine3.start_listening_to_data_parallel_coordinator(
                inference_coordinator_port=DEFAULT_PORT, launch_inference_coordinator=True
            )

            first_port = int(first_addr.rsplit(":", 1)[-1])
            third_port = int(third_addr.rsplit(":", 1)[-1])
            assert (
                third_port != first_port
            ), f"Expected different port due to conflict, but got same: {third_port}"

        finally:
            # Shut down the last engine (which has the active engine loop).
            await graceful_shutdown(engines[-1], client1)
            # Kill any remaining coordinator processes (e.g. engine1's).
            for engine in engines:
                proc = getattr(engine, 'inference_coordinator_process', None)
                if proc is not None and proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=5.0)

    @pytest.mark.internal
    @pytest.mark.skipif(not HAVE_ZMQ, reason="pyzmq is required for this test")
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "initialize_model_parallel",
        [pytest.param((2, 2, 2), id="tp2-pp2-ep2")],
        indirect=["initialize_model_parallel"],
    )
    async def test_control_logic_lifecycle(self, initialize_model_parallel):
        """Comprehensive lifecycle test for the engine state machine."""
        # States where paused stays set: once set during PAUSE, it's only
        # cleared by UNPAUSE.  It remains set through SUSPEND/RESUME/STOP.
        PAUSED_FAMILY = {
            EngineState.PAUSED,
            EngineState.SUSPENDING,
            EngineState.SUSPENDED,
            EngineState.RESUMING,
            EngineState.STOPPING,
            EngineState.STOPPED,
        }

        def assert_state(eng, expected):
            """Assert engine state and all four event flags are consistent."""
            assert eng.state == expected, f"Expected state {expected}, got {eng.state}"
            assert eng.running.is_set() == (
                expected == EngineState.RUNNING
            ), f"running.is_set()={eng.running.is_set()} for state={expected}"
            assert eng.paused.is_set() == (
                expected in PAUSED_FAMILY
            ), f"paused.is_set()={eng.paused.is_set()} for state={expected}"
            assert eng.suspended.is_set() == (
                expected == EngineState.SUSPENDED
            ), f"suspended.is_set()={eng.suspended.is_set()} for state={expected}"
            assert eng.stopped.is_set() == (
                expected == EngineState.STOPPED
            ), f"stopped.is_set()={eng.stopped.is_set()} for state={expected}"

        requests = self.build_requests(num_requests=16)
        engine = DummyEngine()
        client = None
        doomed_futures = []

        try:
            dp_addr = await engine.start_listening_to_data_parallel_coordinator(
                inference_coordinator_port=DEFAULT_PORT, launch_inference_coordinator=True
            )

            if torch.distributed.get_rank() == 0:
                client = InferenceClient(dp_addr)
                client.start()

                await asyncio.wait_for(engine.running.wait(), timeout=5.0)
                assert_state(engine, EngineState.RUNNING)

                # Try to submit signals out of FSM order.
                client.suspend_engines()
                await asyncio.sleep(0.1)
                assert_state(engine, EngineState.RUNNING)
                client.resume_engines()
                await asyncio.sleep(0.1)
                assert_state(engine, EngineState.RUNNING)
                client.stop_engines()
                await asyncio.sleep(0.1)
                assert_state(engine, EngineState.RUNNING)

                # Submit and complete requests while running.
                futures = [client.add_request(prompt=p, sampling_params=s) for p, s in requests[:2]]
                results = await asyncio.wait_for(asyncio.gather(*futures), timeout=5.0)
                for record in results:
                    assert record[-1].status == Status.COMPLETED

                # Submit requests and immediately pause afterwards.
                paused_futures = [
                    client.add_request(prompt=p, sampling_params=s) for p, s in requests[2:5]
                ]
                client.pause_engines()
                await asyncio.wait_for(engine.paused.wait(), timeout=5.0)
                assert_state(engine, EngineState.PAUSED)

                # Try pausing again and see if it breaks.
                client.pause_engines()
                await asyncio.sleep(0.1)
                assert_state(engine, EngineState.PAUSED)

                # Requests submitted during PAUSE should queue, not complete.
                # Use asyncio.wait (not wait_for) so futures aren't cancelled.
                done, pending = await asyncio.wait(paused_futures, timeout=0.5)
                assert len(done) == 0, "No requests should complete while paused"
                assert len(pending) == 3

                # UNPAUSE and verify all in-flight requests complete.
                client.unpause_engines()
                await asyncio.wait_for(engine.running.wait(), timeout=5.0)
                results = await asyncio.wait_for(asyncio.gather(*pending), timeout=10.0)
                for record in results:
                    assert record[-1].status == Status.COMPLETED
                assert_state(engine, EngineState.RUNNING)

                # Engine processes new requests normally after unpause.
                futures = [
                    client.add_request(prompt=p, sampling_params=s) for p, s in requests[5:7]
                ]
                results = await asyncio.wait_for(asyncio.gather(*futures), timeout=5.0)
                for record in results:
                    assert record[-1].status == Status.COMPLETED

                # Suspend.
                client.pause_engines()
                await asyncio.wait_for(engine.paused.wait(), timeout=5.0)
                assert_state(engine, EngineState.PAUSED)

                client.suspend_engines()
                await asyncio.wait_for(engine.suspended.wait(), timeout=5.0)
                assert_state(engine, EngineState.SUSPENDED)

                # Try pausing again and see if it breaks.
                client.pause_engines()
                await asyncio.sleep(0.1)
                assert_state(engine, EngineState.SUSPENDED)

                # Try suspending again and see if it breaks.
                client.pause_engines()
                await asyncio.sleep(0.1)
                assert_state(engine, EngineState.SUSPENDED)

                # Resume.
                engine.paused.clear()
                client.resume_engines()
                await asyncio.wait_for(engine.paused.wait(), timeout=5.0)
                assert_state(engine, EngineState.PAUSED)
                assert not engine.suspended.is_set()

                # Engine processes requests after suspend/resume cycle.
                client.unpause_engines()
                await asyncio.wait_for(engine.running.wait(), timeout=5.0)

                futures = [
                    client.add_request(prompt=p, sampling_params=s) for p, s in requests[7:10]
                ]
                results = await asyncio.wait_for(asyncio.gather(*futures), timeout=5.0)
                for record in results:
                    assert record[-1].status == Status.COMPLETED

                # Submit requests that will be cancelled on STOP.
                client.pause_engines()
                await asyncio.wait_for(engine.paused.wait(), timeout=5.0)
                assert_state(engine, EngineState.PAUSED)

                doomed_futures = [
                    client.add_request(prompt=p, sampling_params=s) for p, s in requests[10:13]
                ]

                client.suspend_engines()
                await asyncio.wait_for(engine.suspended.wait(), timeout=5.0)
                assert_state(engine, EngineState.SUSPENDED)

                client.stop_engines()

            await asyncio.wait_for(engine.stopped.wait(), timeout=5.0)
            assert_state(engine, EngineState.STOPPED)

            if torch.distributed.get_rank() == 0:
                # Client should be stopped, so futures should be cancelled.
                for f in doomed_futures:
                    assert not f.done(), "Client futures should still be pending"
                client.stop()
                client = None
                for f in doomed_futures:
                    assert f.cancelled(), "Client futures should be cancelled after client.stop()"

        finally:
            await graceful_shutdown(engine, client)

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

        client = None
        try:
            if torch.distributed.get_rank() == 0:
                client = InferenceClient(dp_addr)
                client.start()
                init_time = time.time()

                for _ in range(num_iterations):
                    futures = []
                    for prompt, sampling_params in tqdm(requests, "add_requests"):
                        fut = client.add_request(prompt=prompt, sampling_params=sampling_params)
                        futures.append(fut)
                    await asyncio.wait_for(asyncio.gather(*futures), timeout=10.0)
                done_time = time.time()
        finally:
            await graceful_shutdown(engine, client)

        stop_time = time.time()

        flags = torch.tensor([1, 1, 1], dtype=torch.int, device=torch.cuda.current_device())

        init_duration = golden_init_duration = None
        run_duration = golden_run_duration = None
        stop_duration = golden_stop_duration = None

        if torch.distributed.get_rank() == 0:
            init_duration = (init_time - start_time) * 10**3
            golden_init_duration = 14066.31  # ms
            run_duration = (done_time - init_time) * 10**3
            golden_run_duration = 5059.90  # ms
            stop_duration = (stop_time - done_time) * 10**3
            golden_stop_duration = 80.73  # ms

            def clamp_to_golden_value(value, golden_value, delta=0.1):
                return value > golden_value * (1 - delta) and value < golden_value * (1 + delta)

            if not clamp_to_golden_value(init_duration, golden_init_duration, delta=2.0):
                flags[0] = 0
            if not clamp_to_golden_value(run_duration, golden_run_duration, delta=0.5):
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
