# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for the high-level inference APIs."""

import asyncio
import concurrent.futures
from unittest.mock import AsyncMock, MagicMock

import pytest

import megatron.core.inference.apis._llm_base as base_mod
import megatron.core.inference.apis.async_llm as async_llm_mod
import megatron.core.inference.apis.llm as llm_mod
from megatron.core.inference.apis._llm_base import _MegatronLLMBase
from megatron.core.inference.apis.async_llm import MegatronAsyncLLM
from megatron.core.inference.apis.llm import MegatronLLM
from megatron.core.inference.apis.serve_config import ServeConfig


@pytest.fixture
def mock_pipeline(monkeypatch):
    """Stub out the engine pipeline so the constructor runs without torch/megatron."""
    from megatron.core import parallel_state

    engine = MagicMock()
    engine.context = MagicMock()
    engine.controller = MagicMock()
    monkeypatch.setattr(base_mod, "build_dynamic_inference_engine", MagicMock(return_value=engine))
    # MegatronLLM / MegatronAsyncLLM default their inference_wrapper_cls to
    # None and resolve it in their public modules at call time.
    monkeypatch.setattr(llm_mod, "GPTInferenceWrapper", MagicMock())
    monkeypatch.setattr(async_llm_mod, "GPTInferenceWrapper", MagicMock())
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
    obj._async_shutdown_lock = base_mod.threading.Lock()
    obj._shutdown_future = None
    obj._shutdown_complete_future = None
    obj._serve_started = False
    return obj


class _ShutdownLoopManager:
    """Runtime-loop stand-in whose shutdown work finishes when ``teardown`` does.

    ``run_async`` serves the shutdown path without the fix; reusing it after ``stop()`` fails.
    """

    def __init__(self, teardown):
        self.teardown = teardown
        self.stops = 0

    def stop_when_complete(self, future):
        return base_mod._EventLoopManager.stop_when_complete(self, future)

    def submit(self, coro):
        coro.close()
        return self.teardown

    async def run_async(self, coro):
        coro.close()
        if self.stops:
            raise AssertionError("stopped runtime loop was reused")
        await asyncio.wrap_future(self.teardown)

    def stop(self):
        self.stops += 1


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
        model, tokenizer = fake_model_and_tokenizer
        llm = MegatronLLM(model=model, tokenizer=tokenizer, use_coordinator=False)
        assert llm.is_primary_rank is True
        assert llm._use_coordinator is False

        request = MagicMock()
        request.finalize_text.return_value = request
        llm._engine.generate.return_value = [request]
        assert llm.generate("hello")[0] is request
        request.finalize_text.assert_called_once_with(llm._controller.tokenizer)

    def test_async_llm_requires_use_coordinator(self, mock_pipeline, fake_model_and_tokenizer):
        """``MegatronAsyncLLM`` rejects direct mode at ``__init__`` -- the
        engine's loop-bound primitives would collide with the caller's
        running asyncio loop."""
        model, tok = fake_model_and_tokenizer
        with pytest.raises(ValueError, match="requires use_coordinator=True"):
            MegatronAsyncLLM(model=model, tokenizer=tok, use_coordinator=False)

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
        llm = MegatronLLM(model=model, tokenizer=tok, use_coordinator=False)
        with pytest.raises(RuntimeError, match="use_coordinator=True"):
            getattr(llm, method)()

    def test_sync_shutdown_is_noop_and_idempotent_in_direct_mode(
        self, mock_pipeline, fake_model_and_tokenizer
    ):
        model, tok = fake_model_and_tokenizer
        llm = MegatronLLM(model=model, tokenizer=tok, use_coordinator=False)
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

        llm._is_primary_rank = True
        request, future = MagicMock(), asyncio.get_running_loop().create_future()
        future.set_result(request)
        llm._coord_runtime = MagicMock()
        llm._coord_runtime.client.add_request.return_value = future
        assert await llm._generate_impl(["hello"], MagicMock()) == [request]
        request.finalize_text.assert_not_called()

    @pytest.mark.asyncio
    async def test_runtime_teardown_stops_client_when_shutdown_signal_fails(self):
        runtime = base_mod._CoordinatorRuntime.__new__(base_mod._CoordinatorRuntime)
        runtime._is_primary = True
        runtime._engine = MagicMock()
        runtime._client = MagicMock()
        runtime._client.shutdown_coordinator.side_effect = RuntimeError("coordinator gone")
        runtime._client.stop.side_effect = RuntimeError("client stop failed")
        client = runtime._client

        with pytest.raises(RuntimeError, match="coordinator gone"):
            await runtime.teardown()

        client.stop.assert_called_once_with()
        assert runtime._client is None

    @pytest.mark.asyncio
    async def test_engine_transition_failure_still_tears_down_runtime(self):
        llm = _make_worker_instance(MegatronAsyncLLM)
        llm._is_primary_rank = True
        llm._coord_runtime = MagicMock()
        llm._coord_runtime.teardown = AsyncMock()
        llm._engine.state = base_mod.EngineState.RUNNING
        llm._engine.wait_until = AsyncMock(side_effect=RuntimeError("transition failed"))

        with pytest.raises(RuntimeError, match="transition failed"):
            await llm._shutdown_impl()
        llm._coord_runtime.teardown.assert_awaited_once_with()

    @pytest.mark.parametrize("cls", [MegatronLLM, MegatronAsyncLLM])
    def test_shutdown_setup_failure_is_stable_and_cleans_up(self, cls):
        llm = _make_worker_instance(cls)
        llm._coord_runtime = MagicMock()
        llm._loop_manager = MagicMock()
        llm._stop_frontend_if_started = MagicMock(side_effect=RuntimeError("frontend failed"))
        llm._loop_manager.submit.side_effect = RuntimeError("submit failed")
        shutdown = llm.shutdown if cls is MegatronLLM else lambda: asyncio.run(llm.shutdown())

        with pytest.raises(RuntimeError, match="frontend failed"):
            shutdown()
        with pytest.raises(RuntimeError, match="frontend failed"):
            shutdown()
        llm._coord_runtime.teardown_now.assert_called_once_with()
        llm._loop_manager.stop.assert_called_once_with()

    @pytest.mark.asyncio
    async def test_concurrent_async_shutdown_callers_wait_for_same_teardown(self):
        llm = _make_worker_instance(MegatronAsyncLLM)
        teardown = concurrent.futures.Future()

        llm._loop_manager = _ShutdownLoopManager(teardown)
        first = asyncio.create_task(llm.shutdown())
        await asyncio.sleep(0)
        second = asyncio.create_task(llm.shutdown())
        await asyncio.sleep(0)

        assert not first.done()
        assert not second.done()
        teardown.set_result(None)
        await asyncio.gather(first, second)
        assert llm._loop_manager.stops == 1

    def test_async_shutdown_is_atomic_across_caller_threads(self):
        """A caller cannot observe shutdown-started before its shared future exists."""
        llm = _make_worker_instance(MegatronAsyncLLM)
        teardown = concurrent.futures.Future()
        frontend_entered = concurrent.futures.Future()
        release_frontend = concurrent.futures.Future()
        second_started = concurrent.futures.Future()

        def hold_first_caller():
            frontend_entered.set_result(None)
            release_frontend.result(timeout=10)

        def run_second_caller():
            second_started.set_result(None)
            return asyncio.run(llm.shutdown())

        llm._loop_manager = _ShutdownLoopManager(teardown)
        llm._stop_frontend_if_started = hold_first_caller
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            try:
                first = executor.submit(asyncio.run, llm.shutdown())
                frontend_entered.result(timeout=2)
                second = executor.submit(run_second_caller)
                second_started.result(timeout=2)
                with pytest.raises(concurrent.futures.TimeoutError):
                    second.result(timeout=0.5)
            finally:
                # Release the held caller even when a check fails, so the executor can exit.
                for future in (release_frontend, teardown):
                    if not future.done():
                        future.set_result(None)
            first.result(timeout=2)
            second.result(timeout=2)
        assert llm._loop_manager.stops == 1

    @pytest.mark.asyncio
    async def test_cancelled_async_shutdown_caller_does_not_leak_runtime(self):
        llm = _make_worker_instance(MegatronAsyncLLM)
        teardown = concurrent.futures.Future()

        llm._loop_manager = _ShutdownLoopManager(teardown)
        caller = asyncio.create_task(llm.shutdown())
        await asyncio.sleep(0)
        caller.cancel()
        with pytest.raises(asyncio.CancelledError):
            await caller

        assert llm._loop_manager.stops == 0
        teardown.set_result(None)
        await llm.shutdown()
        assert llm._loop_manager.stops == 1

    @pytest.mark.asyncio
    async def test_async_shutdown_propagates_teardown_error_after_runtime_join(self):
        llm = _make_worker_instance(MegatronAsyncLLM)
        teardown = concurrent.futures.Future()

        llm._loop_manager = _ShutdownLoopManager(teardown)
        teardown.set_exception(RuntimeError("teardown failed"))

        with pytest.raises(RuntimeError, match="teardown failed"):
            await llm.shutdown()
        assert llm._loop_manager.stops == 1

    @pytest.mark.asyncio
    async def test_wait_after_async_shutdown_does_not_reuse_stopped_loop(self):
        llm = _make_worker_instance(MegatronAsyncLLM)
        teardown = concurrent.futures.Future()
        teardown.set_result(None)

        llm._loop_manager = _ShutdownLoopManager(teardown)
        await llm.shutdown()
        await llm.wait_for_shutdown()

        sync_llm = _make_worker_instance(MegatronLLM)
        sync_llm._shutdown_called = True
        sync_llm._shutdown_complete_future = concurrent.futures.Future()
        sync_llm._shutdown_complete_future.set_result(None)
        sync_llm._loop_manager = MagicMock()
        sync_llm.shutdown()
        sync_llm.wait_for_shutdown()
        sync_llm._loop_manager.run_sync.assert_not_called()

    @pytest.mark.asyncio
    async def test_generate_after_shutdown_is_rejected(self):
        sync_llm = _make_worker_instance(MegatronLLM)
        sync_llm._is_primary_rank = True
        sync_llm._shutdown_called = True
        # A runtime loop that would accept work, so only the shutdown check can reject calls.
        sync_llm._loop_manager = MagicMock()
        with pytest.raises(RuntimeError, match="shut down"):
            sync_llm.generate("late")
        for method in ("pause", "unpause", "suspend", "resume"):
            with pytest.raises(RuntimeError, match="shut down"):
                getattr(sync_llm, method)()
        for method in (sync_llm.submit, sync_llm.run_sync):
            coro = asyncio.sleep(0)
            with pytest.raises(RuntimeError, match="shut down"):
                method(coro)
            coro.close()

        async_llm = _make_worker_instance(MegatronAsyncLLM)
        async_llm._is_primary_rank = True
        async_llm._shutdown_called = True
        async_llm._loop_manager = MagicMock()
        with pytest.raises(RuntimeError, match="shut down"):
            await async_llm.generate("late")
        for method in ("pause", "unpause", "suspend", "resume"):
            with pytest.raises(RuntimeError, match="shut down"):
                await getattr(async_llm, method)()

    def test_bridge_and_serve_raise_in_direct_mode(self, mock_pipeline, fake_model_and_tokenizer):
        model, tok = fake_model_and_tokenizer
        llm = MegatronLLM(model=model, tokenizer=tok, use_coordinator=False)
        with pytest.raises(ValueError, match="use_coordinator=True"):
            llm.serve(ServeConfig())

        async def coro():
            return 1  # pragma: no cover

        for method in (llm.run_sync, llm.submit):
            c = coro()
            with pytest.raises(RuntimeError, match="use_coordinator=True"):
                method(c)
            c.close()

    def test_sync_serve_nonblocking_worker_rank_noops(self):
        """Worker ranks skip the HTTP setup; ``blocking=False`` returns
        immediately without touching the runtime."""
        llm = _make_worker_instance(MegatronLLM)
        llm.serve(ServeConfig(), blocking=False)
        assert llm._serve_started is False

    def test_sync_serve_primary_rank_starts_frontend(self, monkeypatch):
        """Primary rank starts the HTTP frontend against the coordinator
        address and records ``_serve_started`` for shutdown teardown."""
        tgs = pytest.importorskip(
            "megatron.core.inference.text_generation_server.dynamic_text_gen_server"
            ".text_generation_server"
        )
        import torch.distributed as dist

        llm = _make_worker_instance(MegatronLLM)
        llm._is_primary_rank = True
        llm._coord_runtime = MagicMock()
        llm._coord_runtime.coord_addr = "tcp://coord:5555"

        started = {}
        monkeypatch.setattr(dist, "get_rank", lambda: 0)
        monkeypatch.setattr(tgs, "start_text_gen_server", lambda **kw: started.update(kw))

        sock = MagicMock()
        llm.serve(
            ServeConfig(
                port=1234,
                sock=sock,
                default_temperature=0.7,
                default_top_p=0.95,
                default_top_k=20,
                eval_mode=True,
            ),
            blocking=False,
        )
        assert llm._serve_started is True
        assert started["coordinator_addr"] == "tcp://coord:5555"
        assert started["server_port"] == 1234
        assert started["sock"] is sock
        assert started["default_temperature"] == 0.7
        assert started["default_top_p"] == 0.95
        assert started["default_top_k"] == 20
        assert started["eval_mode"] is True

    @pytest.mark.asyncio
    async def test_async_serve_primary_rank_starts_frontend(self, monkeypatch):
        """Async serving forwards sampling defaults to the HTTP frontend."""
        tgs = pytest.importorskip(
            "megatron.core.inference.text_generation_server.dynamic_text_gen_server"
            ".text_generation_server"
        )
        import torch.distributed as dist

        llm = _make_worker_instance(MegatronAsyncLLM)
        llm._is_primary_rank = True
        llm._coord_runtime = MagicMock()
        llm._coord_runtime.coord_addr = "tcp://coord:5555"

        started = {}
        monkeypatch.setattr(dist, "get_rank", lambda: 0)
        monkeypatch.setattr(tgs, "start_text_gen_server", lambda **kw: started.update(kw))

        sock = MagicMock()
        await llm.serve(
            ServeConfig(
                port=1234,
                sock=sock,
                default_temperature=0.7,
                default_top_p=0.95,
                default_top_k=20,
                eval_mode=True,
            ),
            blocking=False,
        )
        assert llm._serve_started is True
        assert started["coordinator_addr"] == "tcp://coord:5555"
        assert started["server_port"] == 1234
        assert started["sock"] is sock
        assert started["default_temperature"] == 0.7
        assert started["default_top_p"] == 0.95
        assert started["default_top_k"] == 20
        assert started["eval_mode"] is True


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
