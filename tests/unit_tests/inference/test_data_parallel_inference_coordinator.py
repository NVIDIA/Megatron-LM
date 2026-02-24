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
from megatron.core.inference.headers import EngineState, Headers
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

        # Defaults for DP/EP world sizes (overwritten during
        # start_listening_to_data_parallel_coordinator).
        self.dp_world_size = 1
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
        return await DynamicInferenceEngine.run_engine_with_coordinator.__wrapped__(
            self, loop=loop
        )

    def suspend(self):
        """No-op suspend — no real GPU state to offload."""

    def resume(self):
        """No-op resume — no real GPU state to onload."""

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
    """PAUSE then STOP the coordinator test stack.

    Handles the engine being in RUNNING (sends PAUSE first) or already
    PAUSED (skips straight to STOP).  Duplicate PAUSE is now safe
    (coordinator and engine are both idempotent), but this helper still
    checks engine.state to avoid unnecessary round-trips.

    client.stop() is deferred until after the engine has stopped so the
    DEALER socket stays alive for ZMQ message delivery.

    Args:
        engine: DummyEngine whose engine_loop_task to await.
        client: InferenceClient (only rank 0, None on other ranks).
        timeout: Max seconds to wait for each phase.
    """
    try:
        if client is not None:
            # Only PAUSE if the engine is still RUNNING.  If it's already
            # PAUSED (e.g. the test paused it), skip straight to STOP.
            if engine.state == EngineState.RUNNING:
                await asyncio.wait_for(client.pause_engines(), timeout=timeout)
            client.stop_engines()
        # All ranks wait for the engine to reach STOPPED.
        await asyncio.wait_for(engine.stopped.wait(), timeout=timeout)
    except asyncio.TimeoutError:
        engine.engine_loop_task.cancel()
        try:
            await engine.engine_loop_task
        except (asyncio.CancelledError, AssertionError):
            pass
    finally:
        # Engine loop's finally block already closed sockets.
        # Join/terminate the coordinator process.
        engine.stop()
        if client is not None:
            client.stop()


async def force_cleanup_engine(engine):
    """Emergency cleanup when the engine did not shut down cleanly."""
    if hasattr(engine, 'engine_loop_task') and not engine.engine_loop_task.done():
        engine.engine_loop_task.cancel()
        try:
            await engine.engine_loop_task
        except (asyncio.CancelledError, AssertionError):
            pass
    # Engine loop's finally block closes sockets on cancel.
    # stop() joins/terminates the coordinator process idempotently.
    engine.stop()


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
                await client.start()

                futures = [
                    client.add_request(prompt=prompt, sampling_params=params)
                    for prompt, params in requests
                ]
                results = await asyncio.wait_for(asyncio.gather(*futures), timeout=10.0)

                for record in results:
                    assert record[-1].status == Status.COMPLETED
        finally:
            if stop_engines:
                await graceful_shutdown(engine, client, timeout=30.0)
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
        engine1 = DummyEngine()
        engine2 = None
        engine3 = None
        third_addr = None

        # Launch first coordinator - binds to DEFAULT_PORT
        first_addr = await engine1.start_listening_to_data_parallel_coordinator(
            inference_coordinator_port=DEFAULT_PORT, launch_inference_coordinator=True
        )

        try:
            # Verify the engine can serve requests.
            client1 = None
            if torch.distributed.get_rank() == 0:
                client1 = InferenceClient(first_addr)
                await client1.start()

                futures = [
                    client1.add_request(prompt=p, sampling_params=s)
                    for p, s in requests[:2]
                ]
                results = await asyncio.wait_for(asyncio.gather(*futures), timeout=5.0)
                for record in results:
                    assert record[-1].status == Status.COMPLETED

            # Kill the engine unexpectedly.
            engine1.engine_loop_task.cancel()
            try:
                await engine1.engine_loop_task
            except asyncio.CancelledError:
                pass

            # Re-register a new engine.
            engine2 = DummyEngine()
            second_addr = await engine2.start_listening_to_data_parallel_coordinator(
                inference_coordinator_port=DEFAULT_PORT, launch_inference_coordinator=False
            )

            # Verify the new engine can serve requests.
            if torch.distributed.get_rank() == 0:
                futures = [
                    client1.add_request(prompt=p, sampling_params=s)
                    for p, s in requests[2:4]
                ]
                results = await asyncio.wait_for(asyncio.gather(*futures), timeout=5.0)
                for record in results:
                    assert record[-1].status == Status.COMPLETED

            # Kill the engine unexpectedly.
            engine2.engine_loop_task.cancel()
            try:
                await engine2.engine_loop_task
            except asyncio.CancelledError:
                pass
            # Stop the client.
            if client1 is not None:
                client1.stop()
                client1 = None

            # Spin up a new coordinator; verify it gets a new port.
            engine3 = DummyEngine()
            third_addr = await engine3.start_listening_to_data_parallel_coordinator(
                inference_coordinator_port=DEFAULT_PORT, launch_inference_coordinator=True
            )

            first_port = int(first_addr.rsplit(":", 1)[-1])
            third_port = int(third_addr.rsplit(":", 1)[-1])
            assert (
                third_port != first_port
            ), f"Expected different port due to conflict, but got same: {third_port}"

        finally:
            # Clean up engine3's coordinator via PAUSE → STOP.
            if engine3 is not None and third_addr is not None:
                client3 = None
                if torch.distributed.get_rank() == 0:
                    client3 = InferenceClient(third_addr)
                    await client3.start()
                await graceful_shutdown(engine3, client3, timeout=30.0)

            # Terminate engine1's orphaned coordinator process directly.
            engine1.stop()

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
    async def test_control_logic_lifecycle(self, initialize_model_parallel):
        """Comprehensive lifecycle test for the engine state machine.

        Walks through every state transition with assertions at each step,
        verifying state/event atomicity, request queuing during PAUSE,
        engine functionality after SUSPEND/RESUME, clean shutdown via
        PAUSE->STOP, and that invalid signals crash the engine.

        Lifecycle exercised:
          RUNNING -> requests -> PAUSE-with-in-flight -> UNPAUSE ->
          PAUSE -> idempotent-PAUSE -> requests-during-pause -> UNPAUSE ->
          requests -> PAUSE -> SUSPEND -> idempotent-PAUSE-while-SUSPENDED ->
          RESUME -> UNPAUSE -> requests ->
          PAUSE -> queued-requests -> SUSPEND -> STOP-from-SUSPENDED

        Event invariants checked after every transition:
          running.is_set()   <=> state == RUNNING
          paused.is_set()    <=> state in {PAUSED, ..., STOPPED}
          suspended.is_set() <=> state == SUSPENDED
          stopped.is_set()   <=> state == STOPPED
        """
        # States where paused stays set: once set during PAUSE, it's only
        # cleared by UNPAUSE.  It remains set through SUSPEND/RESUME/STOP.
        PAUSED_FAMILY = {
            EngineState.PAUSED, EngineState.SUSPENDING, EngineState.SUSPENDED,
            EngineState.RESUMING, EngineState.STOPPING, EngineState.STOPPED,
        }

        def assert_state(eng, expected):
            """Assert engine state and all four event flags are consistent."""
            assert eng.state == expected, (
                f"Expected state {expected}, got {eng.state}"
            )
            assert eng.running.is_set() == (expected == EngineState.RUNNING), (
                f"running.is_set()={eng.running.is_set()} for state={expected}"
            )
            assert eng.paused.is_set() == (expected in PAUSED_FAMILY), (
                f"paused.is_set()={eng.paused.is_set()} for state={expected}"
            )
            assert eng.suspended.is_set() == (expected == EngineState.SUSPENDED), (
                f"suspended.is_set()={eng.suspended.is_set()} for state={expected}"
            )
            assert eng.stopped.is_set() == (expected == EngineState.STOPPED), (
                f"stopped.is_set()={eng.stopped.is_set()} for state={expected}"
            )

        requests = self.build_requests(num_requests=16)
        engine = DummyEngine()
        dp_addr = await engine.start_listening_to_data_parallel_coordinator(
            inference_coordinator_port=DEFAULT_PORT, launch_inference_coordinator=True
        )

        client = None
        doomed_futures = []
        try:
            if torch.distributed.get_rank() == 0:
                client = InferenceClient(dp_addr)
                await client.start()

                # ── RUNNING ──────────────────────────────────────────
                await asyncio.sleep(0.1)
                assert_state(engine, EngineState.RUNNING)

                # Submit and complete requests while running.
                futures = [
                    client.add_request(prompt=p, sampling_params=s)
                    for p, s in requests[:2]
                ]
                results = await asyncio.wait_for(asyncio.gather(*futures), timeout=5.0)
                for record in results:
                    assert record[-1].status == Status.COMPLETED

                # ── PAUSE with in-flight requests ────────────────────
                # Submit requests and PAUSE in quick succession.  With
                # EP=1 + DummyEngine the requests and PAUSE may arrive in
                # the same schedule_requests batch, but the engine still
                # has to handle them correctly (add requests, then PAUSE).
                inflight_futures = [
                    client.add_request(
                        prompt="Hello world!",
                        sampling_params=SamplingParams(num_tokens_to_generate=100),
                    )
                    for _ in range(3)
                ]
                await asyncio.wait_for(client.pause_engines(), timeout=5.0)
                assert_state(engine, EngineState.PAUSED)

                # UNPAUSE and verify all in-flight requests complete.
                client.unpause_engines()
                results = await asyncio.wait_for(
                    asyncio.gather(*inflight_futures), timeout=10.0
                )
                for record in results:
                    assert record[-1].status == Status.COMPLETED
                assert_state(engine, EngineState.RUNNING)

                # ── PAUSE (1st) ──────────────────────────────────────
                await asyncio.wait_for(client.pause_engines(), timeout=5.0)
                assert_state(engine, EngineState.PAUSED)

                # ── Idempotent PAUSE while PAUSED ────────────────────
                # Client short-circuits (paused already set).
                await asyncio.wait_for(client.pause_engines(), timeout=5.0)
                assert_state(engine, EngineState.PAUSED)

                # Force PAUSE through to coordinator by clearing client
                # state.  Coordinator is in "paused" state and should
                # short-circuit PAUSE_ACK without bothering engines.
                client.paused.clear()
                await asyncio.wait_for(client.pause_engines(), timeout=5.0)
                assert_state(engine, EngineState.PAUSED)

                # Requests submitted during PAUSE should queue, not complete.
                # Use asyncio.wait (not wait_for) so futures aren't cancelled.
                paused_futures = [
                    client.add_request(prompt=p, sampling_params=s)
                    for p, s in requests[2:5]
                ]
                done, pending = await asyncio.wait(paused_futures, timeout=0.5)
                assert len(done) == 0, "No requests should complete while paused"
                assert len(pending) == 3

                # ── UNPAUSE (1st) ────────────────────────────────────
                client.unpause_engines()

                # Queued requests should now complete.
                results = await asyncio.wait_for(asyncio.gather(*pending), timeout=5.0)
                for record in results:
                    assert record[-1].status == Status.COMPLETED
                assert_state(engine, EngineState.RUNNING)

                # Engine processes new requests normally after unpause.
                futures = [
                    client.add_request(prompt=p, sampling_params=s)
                    for p, s in requests[5:7]
                ]
                results = await asyncio.wait_for(asyncio.gather(*futures), timeout=5.0)
                for record in results:
                    assert record[-1].status == Status.COMPLETED

                # ── PAUSE (2nd) ──────────────────────────────────────
                await asyncio.wait_for(client.pause_engines(), timeout=5.0)
                assert_state(engine, EngineState.PAUSED)

                # ── SUSPEND ──────────────────────────────────────────
                client.suspend_engines()
                await asyncio.wait_for(engine.suspended.wait(), timeout=5.0)
                assert_state(engine, EngineState.SUSPENDED)
                assert engine.suspended.is_set()

                # ── Idempotent PAUSE while SUSPENDED ─────────────────
                # Force PAUSE through to coordinator (clear client state).
                # Coordinator is in "suspended" state — short-circuits
                # PAUSE_ACK without bothering engines.
                client.paused.clear()
                await asyncio.wait_for(client.pause_engines(), timeout=5.0)
                assert_state(engine, EngineState.SUSPENDED)

                # ── RESUME ───────────────────────────────────────────
                # paused is already set and stays set through RESUMING.
                # Clear it manually so we can wait for the re-set that
                # confirms the RESUMING→PAUSED transition completed.
                engine.paused.clear()
                client.resume_engines()
                await asyncio.wait_for(engine.paused.wait(), timeout=5.0)
                assert_state(engine, EngineState.PAUSED)
                assert not engine.suspended.is_set()

                # ── UNPAUSE (2nd) ────────────────────────────────────
                client.unpause_engines()
                await asyncio.sleep(0.2)
                assert_state(engine, EngineState.RUNNING)

                # Engine processes requests after suspend/resume cycle.
                futures = [
                    client.add_request(prompt=p, sampling_params=s)
                    for p, s in requests[7:10]
                ]
                results = await asyncio.wait_for(asyncio.gather(*futures), timeout=5.0)
                for record in results:
                    assert record[-1].status == Status.COMPLETED

                # ── PAUSE (3rd) ──────────────────────────────────────
                await asyncio.wait_for(client.pause_engines(), timeout=5.0)
                assert_state(engine, EngineState.PAUSED)

                # Submit requests that will be cancelled on STOP.
                doomed_futures = [
                    client.add_request(prompt=p, sampling_params=s)
                    for p, s in requests[10:13]
                ]

                # ── SUSPEND (2nd) → STOP from SUSPENDED ─────────────
                client.suspend_engines()
                await asyncio.wait_for(engine.suspended.wait(), timeout=5.0)
                assert_state(engine, EngineState.SUSPENDED)

                # STOP directly from SUSPENDED — no RESUME needed.
                client.stop_engines()

            # All ranks: engine loop should exit cleanly.
            await asyncio.wait_for(engine.engine_loop_task, timeout=10.0)
            assert_state(engine, EngineState.STOPPED)

            # STOP should have cancelled all engine-side request state.
            assert len(engine.requests) == 0, "STOP should clear all requests"
            assert len(engine.waiting_request_ids) == 0, "STOP should clear waiting queue"

            if torch.distributed.get_rank() == 0:
                # Client futures are orphaned (no ENGINE_REPLY was sent).
                for f in doomed_futures:
                    assert not f.done(), "Client futures should still be pending"
                client.stop()
                for f in doomed_futures:
                    assert f.cancelled(), "Client futures should be cancelled after client.stop()"

        except Exception:
            await force_cleanup_engine(engine)
            if client is not None:
                try:
                    client.stop()
                except Exception:
                    pass
            raise

        # ── Coordinator signal gating ────────────────────────────────
        # STOP, SUSPEND, and RESUME sent from RUNNING are dropped by the
        # coordinator (it gates on its own state).  The engine never
        # receives them and stays healthy in RUNNING.
        #
        # Engine-side asserts (defense in depth) still exist but are not
        # reachable through normal coordinator operation.
        for signal_name, signal_fn in [
            ("STOP", lambda c: c.stop_engines()),
            ("SUSPEND", lambda c: c.suspend_engines()),
            ("RESUME", lambda c: c.resume_engines()),
        ]:
            engine = DummyEngine()
            dp_addr = await engine.start_listening_to_data_parallel_coordinator(
                inference_coordinator_port=DEFAULT_PORT,
                launch_inference_coordinator=True,
            )
            try:
                client = None
                if torch.distributed.get_rank() == 0:
                    client = InferenceClient(dp_addr)
                    await client.start()
                    await asyncio.sleep(0.1)
                    assert engine.state == EngineState.RUNNING
                    signal_fn(client)
                    # Signal should be dropped by coordinator.
                    await asyncio.sleep(0.3)
                    assert engine.state == EngineState.RUNNING, (
                        f"{signal_name} from RUNNING should be dropped by coordinator"
                    )
            finally:
                await graceful_shutdown(engine, client, timeout=30.0)

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
            await graceful_shutdown(engine, client, timeout=30.0)

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
