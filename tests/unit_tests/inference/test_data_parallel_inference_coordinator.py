# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import asyncio
import itertools
import multiprocessing
import os
import time
from collections import deque
from typing import Dict, Optional

import msgpack
import pytest
import torch
from tqdm import tqdm

from megatron.core.inference.data_parallel_inference_coordinator import (
    DataParallelInferenceCoordinator,
)
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
        for attr in self._STATE_EVENTS.values():
            setattr(self, attr, asyncio.Event())
        self.running.set()
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

        In production, @trace_async_exceptions converts AssertionError to sys.exit(1) -> SystemExit.
        In Python 3.12+, asyncio re-raises SystemExit from tasks in the main thread.
        For tests, we let AssertionErrors propagate directly so pytest.raises can catch them.
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
        await asyncio.sleep(0)

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


async def cleanup_engine(engine, client=None, timeout=30.0):
    """Disconnect an engine between tests. The coordinator stays alive.

    PAUSE is injected into each rank's _pending_signals rather than sent through the coordinator.
    This avoids state-mismatch bugs: the shared coordinator's state may differ from the engine's
    (e.g. the coordinator is PAUSED from a previous test while the engine is RUNNING).
    """
    task = getattr(engine, 'engine_loop_task', None)
    if task is not None and not task.done():
        # Inject PAUSE locally so every rank transitions to PAUSED without
        # relying on the coordinator's current state.
        engine._pending_signals.append(msgpack.packb([Headers.PAUSE.value], use_bin_type=True))
        await asyncio.wait_for(engine.wait_until(EngineState.PAUSED), timeout=timeout)

        sub = getattr(engine, 'model_parallel_num_msgs_subscriber_socket', None)
        if sub is not None:
            sub.setsockopt(zmq.RCVTIMEO, 1000)

        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass

    if client is not None:
        # Walk the coordinator back to RUNNING regardless of its current state
        # so the next test starts cleanly.  Each call is a no-op when the
        # coordinator is already in the target state (just logs a warning).
        client.resume_engines()  # SUSPENDED → PAUSED (no-op otherwise)
        client.unpause_engines()  # PAUSED    → RUNNING (no-op otherwise)
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


@pytest.fixture(scope="class")
def coordinator():
    """Launch a single coordinator process for the entire test class.

    Only rank 0 spawns the coordinator process.  Non-rank-0 processes use a
    placeholder address; the real address is broadcast inside each test's call
    to start_listening_to_data_parallel_coordinator (which broadcasts dp_addr
    from dp_src within the DP process group).

    The coordinator is spawned with data_parallel_size=0 so it doesn't block
    waiting for engines; engines register dynamically via the empty-payload
    re-registration path.
    """
    rank = int(os.environ.get("RANK", "0"))

    if rank == 0:
        spawn_context = multiprocessing.get_context('spawn')
        pipe_parent, pipe_child = spawn_context.Pipe()
        ready_event = spawn_context.Event()
        proc = spawn_context.Process(
            target=DataParallelInferenceCoordinator.entrypoint,
            args=(pipe_child, ready_event, 0, DummyTokenizer(), DEFAULT_PORT, False),
        )
        proc.start()

        # Wait for the coordinator to bind its socket and send the address.
        while not pipe_parent.poll(timeout=0.1):
            assert proc.is_alive(), "Coordinator process died during init"
        dp_addr = pipe_parent.recv()
        pipe_parent.close()
        ready_event.wait(timeout=10.0)
    else:
        proc = None
        # Placeholder: the engine setup broadcasts rank 0's actual address.
        dp_addr = f"tcp://localhost:{DEFAULT_PORT}"

    yield dp_addr

    # Only rank 0 tears down the coordinator process.
    if rank == 0 and proc is not None and proc.is_alive():
        ctx = zmq.Context()
        sock = ctx.socket(zmq.DEALER)
        sock.connect(dp_addr)
        sock.send(msgpack.packb([Headers.CONNECT.value], use_bin_type=True))
        sock.recv()  # CONNECT_ACK
        sock.send(msgpack.packb([Headers.SHUTDOWN.value], use_bin_type=True))
        sock.close(linger=1000)
        ctx.term()
        proc.join(timeout=10.0)
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=5.0)


class TestCoordinator:
    """Test class for Data Parallel Inference Coordinator."""

    def build_requests(self, num_requests=NUM_REQUESTS, num_tokens=NUM_TOKENS):
        """Build a list of test requests."""
        return [
            ("Hello world!", SamplingParams(num_tokens_to_generate=num_tokens))
            for _ in range(num_requests)
        ]

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
    async def test_parallel_configs(self, initialize_model_parallel, coordinator):
        """Test coordinator with various TP, PP, and EP configurations."""
        dp_addr = coordinator
        port = int(dp_addr.rsplit(":", 1)[-1])
        requests = self.build_requests()
        engine = DummyEngine()

        await engine.start_listening_to_data_parallel_coordinator(
            inference_coordinator_port=port, launch_inference_coordinator=False
        )

        # Ensure all engines are registered before submitting requests.
        await asyncio.get_event_loop().run_in_executor(None, torch.distributed.barrier)

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
            await asyncio.get_event_loop().run_in_executor(None, torch.distributed.barrier)
        finally:
            await cleanup_engine(engine, client)

    @pytest.mark.internal
    @pytest.mark.skipif(not HAVE_ZMQ, reason="pyzmq is required for this test")
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "initialize_model_parallel",
        [pytest.param((2, 2, 2), id="tp2-pp2-ep2")],
        indirect=["initialize_model_parallel"],
    )
    async def test_control_logic_lifecycle(self, initialize_model_parallel, coordinator):
        """Comprehensive lifecycle test for the engine state machine."""
        # States where paused stays set: once set during PAUSE, it's only cleared by UNPAUSE.
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

        dp_addr = coordinator
        port = int(dp_addr.rsplit(":", 1)[-1])
        requests = self.build_requests(num_requests=16)
        engine = DummyEngine()
        client = None
        doomed_futures = []

        try:
            await engine.start_listening_to_data_parallel_coordinator(
                inference_coordinator_port=port, launch_inference_coordinator=False
            )

            # Synchronize all ranks so every engine has registered.
            await asyncio.get_event_loop().run_in_executor(None, torch.distributed.barrier)

            if torch.distributed.get_rank() == 0:
                client = InferenceClient(dp_addr)
                client.start()

                await asyncio.wait_for(engine.wait_until(EngineState.RUNNING), timeout=5.0)
                assert_state(engine, EngineState.RUNNING)

                # Try to submit signals out of FSM order.
                # The coordinator's state machine filters these out.
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

                # Submit requests while RUNNING, then PAUSE before they drain.
                # These must survive the PAUSE (not be drained during PAUSING).
                pre_pause_futures = [
                    client.add_request(prompt=p, sampling_params=s) for p, s in requests[2:3]
                ]
                client.pause_engines()
                await asyncio.wait_for(engine.wait_until(EngineState.PAUSED), timeout=5.0)
                assert_state(engine, EngineState.PAUSED)

                # Pre-pause requests must NOT have been drained.
                done, pending = await asyncio.wait(pre_pause_futures, timeout=0.5)
                assert len(pending) > 0, "Pre-pause requests should not drain during PAUSING"

                # Try pausing again and see if it breaks.
                client.pause_engines()
                await asyncio.sleep(0.1)
                assert_state(engine, EngineState.PAUSED)

                # Requests submitted while PAUSED should queue, not complete.
                paused_futures = [
                    client.add_request(prompt=p, sampling_params=s) for p, s in requests[3:5]
                ]
                # Use asyncio.wait (not wait_for) so futures aren't cancelled.
                done, pending = await asyncio.wait(paused_futures, timeout=0.5)
                assert len(done) == 0, "No requests should complete while paused"
                assert len(pending) == 2

                # UNPAUSE and verify all in-flight requests (pre-pause + paused) complete.
                client.unpause_engines()
                await asyncio.wait_for(engine.wait_until(EngineState.RUNNING), timeout=5.0)
                all_queued = pre_pause_futures + paused_futures
                results = await asyncio.wait_for(asyncio.gather(*all_queued), timeout=10.0)
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
                await asyncio.wait_for(engine.wait_until(EngineState.PAUSED), timeout=5.0)
                assert_state(engine, EngineState.PAUSED)

                client.suspend_engines()
                await asyncio.wait_for(engine.wait_until(EngineState.SUSPENDED), timeout=5.0)
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
                await asyncio.wait_for(engine.wait_until(EngineState.PAUSED), timeout=5.0)
                assert_state(engine, EngineState.PAUSED)
                assert not engine.suspended.is_set()

                # Engine processes requests after suspend/resume cycle.
                client.unpause_engines()
                await asyncio.wait_for(engine.wait_until(EngineState.RUNNING), timeout=5.0)

                futures = [
                    client.add_request(prompt=p, sampling_params=s) for p, s in requests[7:10]
                ]
                results = await asyncio.wait_for(asyncio.gather(*futures), timeout=5.0)
                for record in results:
                    assert record[-1].status == Status.COMPLETED

                # Submit requests that will be cancelled on STOP.
                client.pause_engines()
                await asyncio.wait_for(engine.wait_until(EngineState.PAUSED), timeout=5.0)
                assert_state(engine, EngineState.PAUSED)

                doomed_futures = [
                    client.add_request(prompt=p, sampling_params=s) for p, s in requests[10:13]
                ]

            # Synchronize all ranks: rank 0 reaches here after confirming engine is at final PAUSED.
            # Non-rank-0 reaches here immediately (skipping the rank-0 block), but blocks.
            # After the barrier, all engine states are synchronized so it's safe to inject STOP.
            await asyncio.get_event_loop().run_in_executor(None, torch.distributed.barrier)

            if torch.distributed.get_rank() == 0:
                # Verify doomed futures are still pending.
                for f in doomed_futures:
                    assert not f.done(), "Client futures should still be pending"

            # Inject STOP directly so we don't kill the shared coordinator.
            engine._pending_signals.append(msgpack.packb([Headers.STOP.value], use_bin_type=True))

            await asyncio.wait_for(engine.wait_until(EngineState.STOPPED), timeout=60.0)
            assert_state(engine, EngineState.STOPPED)

        finally:
            # Let cleanup_engine handle client.stop() so it can first reset the
            # coordinator back to RUNNING (it sends RESUME + UNPAUSE).
            await cleanup_engine(engine, client)

        # cleanup_engine called client.stop() which cancels pending futures.
        if torch.distributed.get_rank() == 0:
            for f in doomed_futures:
                assert f.cancelled(), "Client futures should be cancelled after client.stop()"

    @pytest.mark.internal
    @pytest.mark.skipif(not HAVE_ZMQ, reason="pyzmq is required for this test")
    @pytest.mark.asyncio
    async def test_throughput(self, initialize_model_parallel, coordinator):
        """Throughput benchmark: measures ZMQ packet rate."""
        _, dp, _, _, _ = initialize_model_parallel
        num_requests = 10**4
        num_iterations = 10

        dp_addr = coordinator
        port = int(dp_addr.rsplit(":", 1)[-1])
        engine = DummyEngine()
        requests = self.build_requests(num_requests=num_requests)

        await engine.start_listening_to_data_parallel_coordinator(
            inference_coordinator_port=port, launch_inference_coordinator=False
        )

        # Ensure all engines are registered before submitting requests.
        await asyncio.get_event_loop().run_in_executor(None, torch.distributed.barrier)

        client = None
        try:
            if torch.distributed.get_rank() == 0:
                client = InferenceClient(dp_addr)
                client.start()

                start_time = time.time()
                for _ in range(num_iterations):
                    futures = []
                    for prompt, sampling_params in requests:
                        fut = client.add_request(prompt=prompt, sampling_params=sampling_params)
                        futures.append(fut)
                    await asyncio.wait_for(asyncio.gather(*futures), timeout=30.0)
                elapsed_ms = (time.time() - start_time) * 1e3
                total = num_requests * num_iterations // dp
                print(
                    f"ZMQ throughput: {total / elapsed_ms:.2f} requests/ms "
                    f"({total} reqs in {elapsed_ms:.0f} ms)"
                )
            await asyncio.get_event_loop().run_in_executor(None, torch.distributed.barrier)
        finally:
            await cleanup_engine(engine, client)
