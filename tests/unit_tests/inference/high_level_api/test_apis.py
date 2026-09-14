# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for the high-level inference APIs."""

import asyncio
import threading
from unittest.mock import MagicMock

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

    monkeypatch.setattr(base_mod, "DynamicInferenceContext", MagicMock())
    monkeypatch.setattr(base_mod, "GPTInferenceWrapper", MagicMock())
    monkeypatch.setattr(base_mod, "TextGenerationController", MagicMock())
    monkeypatch.setattr(base_mod, "DynamicInferenceEngine", MagicMock())
    # MegatronLLM / MegatronAsyncLLM default their inference_wrapper_cls to
    # None and resolve to base_mod.GPTInferenceWrapper at call time, so the
    # base_mod patch above is what steers them at construction time.
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


async def _no_teardown():
    """Stand-in for ``_shutdown_impl`` when there is no coordinator to tear down."""


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

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "cls, shutdown_races",
        [
            (MegatronLLM, False),
            (MegatronLLM, True),
            (MegatronAsyncLLM, False),
            (MegatronAsyncLLM, True),
        ],
        ids=["sync", "sync-shutdown-races", "async", "async-shutdown-races"],
    )
    async def test_serve_primary_rank_starts_frontend(self, monkeypatch, cls, shutdown_races):
        """Primary rank starts the HTTP frontend against the coordinator address with the
        configured defaults. ``serve()`` blocks its caller until the replicas listen, so only
        another thread's ``shutdown()`` can race it; that cannot stop replicas that do not
        exist yet, so the starter stops them itself on return."""
        tgs = pytest.importorskip(
            "megatron.core.inference.text_generation_server.dynamic_text_gen_server"
            ".text_generation_server"
        )
        import torch.distributed as dist

        llm = _make_worker_instance(cls)
        llm._is_primary_rank = True
        llm._coord_runtime = MagicMock(coord_addr="tcp://coord:5555")
        # A real runtime loop: embedders drive the async facade through run_sync on it.
        llm._loop_manager = base_mod._EventLoopManager()
        llm._loop_manager.start()
        monkeypatch.setattr(llm, "_shutdown_impl", _no_teardown)
        monkeypatch.setattr(dist, "get_rank", lambda: 0)

        started, calls = {}, []
        starting, stopping, release = threading.Event(), threading.Event(), threading.Event()

        def fake_start(**kw):
            started.update(kw)
            calls.append("start")
            if shutdown_races:
                starting.set()
                release.wait()  # the replicas are coming up until the test lets them

        def fake_stop():
            calls.append("stop")
            stopping.set()

        monkeypatch.setattr(tgs, "start_text_gen_server", fake_start)
        monkeypatch.setattr(tgs, "stop_text_gen_server", fake_stop)

        sock = MagicMock()
        config = ServeConfig(
            port=1234,
            sock=sock,
            default_temperature=0.7,
            default_top_p=0.95,
            default_top_k=20,
            eval_mode=True,
        )
        if cls is MegatronAsyncLLM:
            serving = asyncio.to_thread(llm.run_sync, llm.serve(config, blocking=False))
        else:
            serving = asyncio.to_thread(llm.serve, config, blocking=False)
        serving = asyncio.ensure_future(serving)
        if shutdown_races:
            await asyncio.to_thread(starting.wait)
            shutting_down = asyncio.ensure_future(
                llm.shutdown() if cls is MegatronAsyncLLM else asyncio.to_thread(llm.shutdown)
            )
            await asyncio.to_thread(stopping.wait)  # shutdown() found nothing to stop yet
            release.set()
            await shutting_down
        await serving
        llm._loop_manager.stop()

        forwarded = dict(
            coordinator_addr="tcp://coord:5555",
            server_port=1234,
            sock=sock,
            default_temperature=0.7,
            default_top_p=0.95,
            default_top_k=20,
            eval_mode=True,
        )
        assert {k: started[k] for k in forwarded} == forwarded
        # Without a race the frontend stays up. With one, shutdown()'s own stop is followed
        # by the starter's once the replicas actually exist.
        assert calls == (["start", "stop", "stop"] if shutdown_races else ["start"])
        assert llm._serve_started is (not shutdown_races)


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
