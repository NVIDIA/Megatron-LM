# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from types import SimpleNamespace

from megatron.training import async_utils


def test_init_persistent_async_worker_uses_strategy_modules(monkeypatch):
    calls = []
    monkeypatch.setattr(async_utils, "_async_calls_queue", None)

    class FakeAsyncCallsQueue:
        def __init__(self, persistent=False):
            calls.append(("queue", persistent))

        @staticmethod
        def warmup_persistent_caller(rank, **kwargs):
            calls.append(("warmup", rank, kwargs))

    monkeypatch.setattr(
        async_utils,
        "get_args",
        lambda: SimpleNamespace(
            async_strategy="mcore",
            async_ckpt_cpu_priority=3,
            async_ckpt_io_priority=4,
        ),
    )
    monkeypatch.setattr(
        async_utils,
        "get_async_strategy",
        lambda strategy: (
            "mcore",
            {
                "AsyncCallsQueue": FakeAsyncCallsQueue,
                "get_write_results_queue": lambda mode: calls.append(("results-queue", mode)),
            },
        ),
    )

    async_utils.init_persistent_async_worker(rank=0, mp_mode="fork")

    assert ("queue", True) in calls
    assert ("results-queue", "fork") in calls
    assert any(call[0] == "warmup" and call[2]["mp_mode"] == "fork" for call in calls)
    assert async_utils._async_calls_queue is not None


def test_get_save_and_finalize_callbacks_wraps_writer_and_finalize(monkeypatch):
    calls = []

    class FakeWriter:
        def get_save_function_and_args(self):
            return "save-fn", "preload-fn", ("arg",)

    class FakeAsyncRequest:
        def __init__(self, save_fn, save_args, finalize_fns, async_fn_kwargs, preload_fn):
            self.save_fn = save_fn
            self.save_args = save_args
            self.finalize_fns = finalize_fns
            self.async_fn_kwargs = async_fn_kwargs
            self.preload_fn = preload_fn

    monkeypatch.setattr(async_utils, "NVRxAsyncRequest", FakeAsyncRequest)
    monkeypatch.setattr(
        async_utils,
        "save_state_dict_async_finalize",
        lambda *items: calls.append(("finalize", items)),
    )

    request = async_utils.get_save_and_finalize_callbacks(FakeWriter(), ("state", "dict"))
    request.finalize_fns[0]()

    assert request.save_fn == "save-fn"
    assert request.save_args == ("arg",)
    assert request.preload_fn == "preload-fn"
    assert request.async_fn_kwargs == {}
    assert calls == [("finalize", ("state", "dict"))]
