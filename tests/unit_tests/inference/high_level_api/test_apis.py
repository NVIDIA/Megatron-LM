# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for the high-level inference APIs (``MegatronLLM`` /
``MegatronAsyncLLM``). Tests run without torch/megatron init by stubbing
the engine pipeline; the worker-rank tests bypass ``__init__`` entirely
via ``cls.__new__``."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

import megatron.core.inference.apis._llm_base as base_mod
from megatron.core.inference.apis._llm_base import _CoordinatorRuntime, _MegatronLLMBase
from megatron.core.inference.apis.async_llm import MegatronAsyncLLM
from megatron.core.inference.apis.llm import MegatronLLM


@pytest.fixture
def mock_pipeline(monkeypatch):
    """Stub out the engine pipeline so the constructor runs without torch/megatron."""
    from megatron.core import parallel_state

    monkeypatch.setattr(base_mod, "DynamicInferenceContext", MagicMock())
    monkeypatch.setattr(base_mod, "GPTInferenceWrapper", MagicMock())
    monkeypatch.setattr(base_mod, "TextGenerationController", MagicMock())
    monkeypatch.setattr(base_mod, "DynamicInferenceEngine", MagicMock())
    # Bypass the EP-group initialization assert when no distributed setup
    # is in scope. Individual tests can override (e.g.,
    # ``test_ep_gt_1_requires_use_coordinator``).
    monkeypatch.setattr(parallel_state, "get_expert_model_parallel_world_size", lambda: 1)


@pytest.fixture
def fake_model_and_tokenizer():
    model = MagicMock()
    model.config = MagicMock()
    tokenizer = MagicMock()
    return model, tokenizer


def _make_worker_instance(cls):
    """Build a coordinator-mode worker-rank instance without running the
    constructor's engine/runtime setup."""
    obj = cls.__new__(cls)
    obj._engine = MagicMock()
    obj._context = MagicMock()
    obj._controller = MagicMock()
    obj._use_coordinator = True
    obj._is_primary_rank = False
    obj._loop_manager = None
    obj._coord_runtime = None
    obj._shutdown_called = False
    if cls is MegatronAsyncLLM:
        obj._serve_started = False
    return obj


def _make_primary_coord_instance(cls):
    """Primary-rank coord-mode instance with mocked runtime, for dispatch tests."""
    obj = cls.__new__(cls)
    obj._engine = MagicMock()
    obj._context = MagicMock()
    obj._controller = MagicMock()
    obj._use_coordinator = True
    obj._is_primary_rank = True
    obj._loop_manager = MagicMock()
    obj._loop_manager.run_sync = MagicMock(return_value=[])
    obj._loop_manager.run_async = AsyncMock(return_value=[])
    obj._coord_runtime = MagicMock()
    obj._coord_runtime.client = MagicMock()
    obj._coord_runtime.teardown = AsyncMock()
    obj._shutdown_called = False
    if cls is MegatronAsyncLLM:
        obj._serve_started = False
    return obj


class TestConstructorValidation:
    """Constructor-time validation for both ``MegatronLLM`` and ``MegatronAsyncLLM``."""

    @pytest.mark.parametrize(
        "extra_kwargs", [{"coordinator_host": "x"}, {"coordinator_port": 5000}]
    )
    def test_coordinator_host_or_port_without_use_coordinator_raises(
        self, mock_pipeline, fake_model_and_tokenizer, extra_kwargs
    ):
        model, tok = fake_model_and_tokenizer
        with pytest.raises(ValueError, match="coordinator_host/port require use_coordinator=True"):
            MegatronLLM(model=model, tokenizer=tok, use_coordinator=False, **extra_kwargs)

    def test_megatron_llm_direct_mode_succeeds(self, mock_pipeline, fake_model_and_tokenizer):
        model, tok = fake_model_and_tokenizer
        llm = MegatronLLM(model=model, tokenizer=tok)
        assert llm.is_primary_rank is True
        assert llm._use_coordinator is False

    def test_async_llm_requires_use_coordinator(self, mock_pipeline, fake_model_and_tokenizer):
        """``MegatronAsyncLLM`` rejects direct mode at ``__init__`` -- the
        engine's loop-bound primitives would collide with the caller's
        running asyncio loop."""
        model, tok = fake_model_and_tokenizer
        with pytest.raises(ValueError, match="requires use_coordinator=True"):
            MegatronAsyncLLM(model=model, tokenizer=tok)

    def test_ep_gt_1_requires_use_coordinator(
        self, mock_pipeline, fake_model_and_tokenizer, monkeypatch
    ):
        """Direct mode with expert_model_parallel_size > 1 must raise --
        EP routing requires the coordinator."""
        from megatron.core import parallel_state

        monkeypatch.setattr(parallel_state, "get_expert_model_parallel_world_size", lambda: 4)
        model, tok = fake_model_and_tokenizer
        with pytest.raises(ValueError, match="expert_model_parallel_size > 1"):
            MegatronLLM(model=model, tokenizer=tok, use_coordinator=False)


class TestLifecycleGuards:
    """Direct-mode lifecycle guards (``MegatronLLM`` only -- async direct is
    rejected at construction), and coordinator-mode worker-rank guards."""

    @pytest.mark.parametrize("method", ["pause", "unpause", "suspend", "resume"])
    def test_sync_lifecycle_raises_in_direct_mode(
        self, mock_pipeline, fake_model_and_tokenizer, method
    ):
        model, tok = fake_model_and_tokenizer
        llm = MegatronLLM(model=model, tokenizer=tok)
        with pytest.raises(RuntimeError, match="use_coordinator=True"):
            getattr(llm, method)()

    def test_sync_shutdown_is_noop_and_idempotent_in_direct_mode(
        self, mock_pipeline, fake_model_and_tokenizer
    ):
        model, tok = fake_model_and_tokenizer
        llm = MegatronLLM(model=model, tokenizer=tok)
        llm.shutdown()
        assert llm._shutdown_called is True
        llm.shutdown()  # second call is a no-op
        assert llm._shutdown_called is True
        llm.wait_for_shutdown()  # also a no-op

    def test_sync_generate_raises_on_worker_rank(self):
        llm = _make_worker_instance(MegatronLLM)
        with pytest.raises(RuntimeError, match="primary rank"):
            llm.generate("hello")

    @pytest.mark.asyncio
    async def test_async_generate_raises_on_worker_rank(self):
        llm = _make_worker_instance(MegatronAsyncLLM)
        with pytest.raises(RuntimeError, match="primary rank"):
            await llm.generate("hello")


class TestNormalizePrompts:
    """Input-shape normalization (str / list[int] / list[str] / list[list[int]])."""

    @staticmethod
    def _normalize(prompts):
        obj = _MegatronLLMBase.__new__(_MegatronLLMBase)
        return obj._normalize_prompts(prompts)

    @pytest.mark.parametrize(
        "prompts,expected",
        [
            ("abc", (["abc"], False)),
            ([1, 2, 3], ([[1, 2, 3]], False)),
            (["a", "b"], (["a", "b"], True)),
            ([[1, 2], [3, 4]], ([[1, 2], [3, 4]], True)),
            ([], ([], True)),
        ],
    )
    def test_valid_inputs(self, prompts, expected):
        assert self._normalize(prompts) == expected

    @pytest.mark.parametrize("bad_input", [{1, 2}, 1.5, [1.5], {"k": "v"}])
    def test_unsupported_inputs_raise_typeerror(self, bad_input):
        with pytest.raises(TypeError):
            self._normalize(bad_input)


class TestMegatronLLMDispatch:
    """Coord-mode primary-rank public methods dispatch through ``_loop_manager``."""

    @pytest.mark.parametrize("method", ["pause", "unpause", "suspend", "resume"])
    def test_lifecycle_dispatches_to_run_sync(self, method):
        llm = _make_primary_coord_instance(MegatronLLM)
        getattr(llm, method)()
        llm._loop_manager.run_sync.assert_called_once()

    def test_generate_dispatches_to_run_sync(self):
        llm = _make_primary_coord_instance(MegatronLLM)
        result = llm.generate("hello")
        llm._loop_manager.run_sync.assert_called_once()
        assert result == []

    def test_shutdown_dispatches_then_stops_loop_and_is_idempotent(self):
        llm = _make_primary_coord_instance(MegatronLLM)
        llm.shutdown()
        llm._loop_manager.run_sync.assert_called_once()
        llm._loop_manager.stop.assert_called_once()
        assert llm._shutdown_called is True

        llm._loop_manager.run_sync.reset_mock()
        llm._loop_manager.stop.reset_mock()
        llm.shutdown()  # second call no-ops
        llm._loop_manager.run_sync.assert_not_called()
        llm._loop_manager.stop.assert_not_called()

    def test_wait_for_shutdown_dispatches_to_run_sync(self):
        llm = _make_primary_coord_instance(MegatronLLM)
        llm.wait_for_shutdown()
        llm._loop_manager.run_sync.assert_called_once()


class TestMegatronAsyncLLMDispatch:
    """Coord-mode primary-rank public methods dispatch through ``_loop_manager.run_async``."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("method", ["pause", "unpause", "suspend", "resume"])
    async def test_lifecycle_dispatches_to_run_async(self, method):
        llm = _make_primary_coord_instance(MegatronAsyncLLM)
        await getattr(llm, method)()
        llm._loop_manager.run_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_dispatches_to_run_async(self):
        llm = _make_primary_coord_instance(MegatronAsyncLLM)
        # Single-string input -> generate unwraps via results[0], so the mock
        # must return a non-empty list.
        sentinel = MagicMock()
        llm._loop_manager.run_async = AsyncMock(return_value=[sentinel])
        result = await llm.generate("hello")
        llm._loop_manager.run_async.assert_called_once()
        assert result is sentinel

    @pytest.mark.asyncio
    async def test_shutdown_dispatches_then_stops_loop(self):
        llm = _make_primary_coord_instance(MegatronAsyncLLM)
        await llm.shutdown()
        llm._loop_manager.run_async.assert_called_once()
        llm._loop_manager.stop.assert_called_once()
        assert llm._shutdown_called is True

    @pytest.mark.asyncio
    async def test_wait_for_shutdown_dispatches_to_run_async(self):
        llm = _make_primary_coord_instance(MegatronAsyncLLM)
        await llm.wait_for_shutdown()
        llm._loop_manager.run_async.assert_called_once()


class TestCoordinatorRuntime:
    """Direct tests of ``_CoordinatorRuntime.setup`` and ``teardown``."""

    @pytest.mark.asyncio
    async def test_setup_primary_starts_inference_client(self, monkeypatch):
        engine = MagicMock()
        engine.start_listening_to_data_parallel_coordinator = AsyncMock(return_value="tcp://x:1")
        client_mock = MagicMock()
        client_class = MagicMock(return_value=client_mock)
        monkeypatch.setattr(
            "megatron.core.inference.inference_client.InferenceClient", client_class
        )
        cr = _CoordinatorRuntime(
            engine, is_primary=True, coordinator_host=None, coordinator_port=None
        )
        await cr.setup(loop=asyncio.get_event_loop())
        engine.start_listening_to_data_parallel_coordinator.assert_called_once()
        client_class.assert_called_once_with("tcp://x:1", deserialize=True)
        client_mock.start.assert_called_once()
        assert cr._client is client_mock

    @pytest.mark.asyncio
    async def test_setup_worker_skips_client(self):
        engine = MagicMock()
        engine.start_listening_to_data_parallel_coordinator = AsyncMock(return_value="tcp://x:1")
        cr = _CoordinatorRuntime(
            engine, is_primary=False, coordinator_host=None, coordinator_port=None
        )
        await cr.setup(loop=asyncio.get_event_loop())
        assert cr._client is None

    @pytest.mark.asyncio
    async def test_teardown_happy_path_shuts_down_client(self):
        engine = MagicMock()
        cr = _CoordinatorRuntime(
            engine, is_primary=True, coordinator_host=None, coordinator_port=None
        )
        client = MagicMock()
        cr._client = client
        await cr.teardown()
        client.shutdown_coordinator.assert_called_once()
        client.stop.assert_called_once()
        assert cr._client is None

    @pytest.mark.asyncio
    async def test_teardown_partial_setup_terminates_process(self):
        engine = MagicMock()
        proc = MagicMock()
        proc.is_alive.return_value = True
        engine.inference_coordinator_process = proc
        cr = _CoordinatorRuntime(
            engine, is_primary=True, coordinator_host=None, coordinator_port=None
        )
        # client never opened (None) — should fall through to proc.terminate
        await cr.teardown()
        proc.terminate.assert_called_once()
        proc.join.assert_called()

    @pytest.mark.asyncio
    async def test_teardown_worker_is_noop(self):
        engine = MagicMock()
        cr = _CoordinatorRuntime(
            engine, is_primary=False, coordinator_host=None, coordinator_port=None
        )
        await cr.teardown()  # no error, no work


class TestImplCoroutines:
    """Direct tests of the ``_*_impl`` coroutines on ``_MegatronLLMBase``."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "method,client_method",
        [
            ("_pause_impl", "pause_engines"),
            ("_unpause_impl", "unpause_engines"),
            ("_suspend_impl", "suspend_engines"),
            ("_resume_impl", "resume_engines"),
        ],
    )
    async def test_lifecycle_impl_signals_client_and_waits(self, method, client_method):
        llm = _make_primary_coord_instance(MegatronLLM)
        llm._engine.wait_until = AsyncMock()
        await getattr(llm, method)()
        getattr(llm._coord_runtime.client, client_method).assert_called_once()
        llm._engine.wait_until.assert_called_once()

    @pytest.mark.asyncio
    async def test_wait_for_shutdown_impl_awaits_engine_loop_task(self):
        llm = _make_primary_coord_instance(MegatronLLM)

        async def fake_task():
            return None

        llm._engine.engine_loop_task = fake_task()
        await llm._wait_for_shutdown_impl()

    @pytest.mark.asyncio
    async def test_generate_impl_coord_mode_gathers_client_futures(self):
        llm = _make_primary_coord_instance(MegatronLLM)
        f1, f2 = asyncio.Future(), asyncio.Future()
        f1.set_result("r1")
        f2.set_result("r2")
        llm._coord_runtime.client.add_request = MagicMock(side_effect=[f1, f2])
        out = await llm._generate_impl(["a", "b"], MagicMock())
        assert out == ["r1", "r2"]
        assert llm._coord_runtime.client.add_request.call_count == 2

    @pytest.mark.asyncio
    async def test_generate_impl_direct_mode_calls_engine_generate(self):
        llm = _make_primary_coord_instance(MegatronLLM)
        llm._use_coordinator = False  # exercise direct branch
        rec1, rec2 = MagicMock(), MagicMock()
        rec1.merge.return_value = "m1"
        rec2.merge.return_value = "m2"
        llm._engine.generate.return_value = [rec1, rec2]
        out = await llm._generate_impl(["a", "b"], MagicMock())
        assert out == ["m1", "m2"]
        llm._engine.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown_impl_primary_running_pauses_then_stops(self):
        from megatron.core.inference.engines.dynamic_engine import EngineState

        llm = _make_primary_coord_instance(MegatronLLM)
        llm._engine.state = EngineState.RUNNING
        llm._engine.wait_until = AsyncMock()
        await llm._shutdown_impl()
        # PAUSE then STOP, with wait_until called twice (PAUSED, STOPPED)
        llm._coord_runtime.client.pause_engines.assert_called_once()
        llm._coord_runtime.client.stop_engines.assert_called_once()
        assert llm._engine.wait_until.call_count == 2
        llm._coord_runtime.teardown.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_shutdown_impl_worker_awaits_engine_loop_task(self):
        llm = _make_primary_coord_instance(MegatronLLM)
        llm._is_primary_rank = False  # worker path
        llm._engine.engine_loop_task = asyncio.sleep(0)
        await llm._shutdown_impl()
        llm._coord_runtime.client.stop_engines.assert_not_called()


class TestConstructorCoordModeInit:
    """Cover ``_MegatronLLMBase.__init__`` coordinator branch + T1 teardown path."""

    def test_coord_mode_happy_path_wires_loop_manager_and_coord_runtime(
        self, mock_pipeline, fake_model_and_tokenizer, monkeypatch
    ):
        # Mock the runtime classes so init doesn't actually spawn threads/processes.
        fake_loop_mgr_inst = MagicMock()
        monkeypatch.setattr(
            base_mod, "_EventLoopManager", MagicMock(return_value=fake_loop_mgr_inst)
        )
        fake_coord_rt = MagicMock()
        monkeypatch.setattr(base_mod, "_CoordinatorRuntime", MagicMock(return_value=fake_coord_rt))
        monkeypatch.setattr(base_mod.dist, "get_rank", lambda: 0)

        model, tok = fake_model_and_tokenizer
        llm = MegatronLLM(model=model, tokenizer=tok, use_coordinator=True)

        fake_loop_mgr_inst.start.assert_called_once()
        # setup should have been scheduled via run_sync.
        fake_loop_mgr_inst.run_sync.assert_called_once()
        assert llm._loop_manager is fake_loop_mgr_inst
        assert llm._coord_runtime is fake_coord_rt
        assert llm.is_primary_rank is True

    def test_coord_mode_setup_failure_triggers_teardown_then_stops_loop(
        self, mock_pipeline, fake_model_and_tokenizer, monkeypatch
    ):
        fake_loop_mgr_inst = MagicMock()
        # First run_sync (setup) raises; second (teardown) is a no-op.
        fake_loop_mgr_inst.run_sync.side_effect = [RuntimeError("setup boom"), None]
        monkeypatch.setattr(
            base_mod, "_EventLoopManager", MagicMock(return_value=fake_loop_mgr_inst)
        )
        fake_coord_rt = MagicMock()
        monkeypatch.setattr(base_mod, "_CoordinatorRuntime", MagicMock(return_value=fake_coord_rt))
        monkeypatch.setattr(base_mod.dist, "get_rank", lambda: 0)

        model, tok = fake_model_and_tokenizer
        with pytest.raises(RuntimeError, match="setup boom"):
            MegatronLLM(model=model, tokenizer=tok, use_coordinator=True)

        # Two run_sync calls: setup (raised) + teardown (best-effort).
        assert fake_loop_mgr_inst.run_sync.call_count == 2
        fake_loop_mgr_inst.stop.assert_called_once()
