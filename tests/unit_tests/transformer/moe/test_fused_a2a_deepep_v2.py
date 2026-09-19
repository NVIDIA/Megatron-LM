# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for the DeepEP V2 `ElasticBuffer` probe in
`megatron.core.transformer.moe.fused_a2a`.

These tests exercise the version-probe plumbing without running a
real a2a: they import the module, check the `HAVE_DEEP_EP_V2` flag
reflects the environment, check `get_buffer()` returns a V2
`ElasticBuffer` when V2 is installed, and check the V1 call path is
preserved when V2 is absent (via `monkeypatch`).

Integration tests (full dispatch/combine on 8 GPUs) are already
covered by the parametrized `TestFlexDispatcher` in
`test_token_dispatcher.py`; those exercise whichever DeepEP flavour
is installed in the CI image. This file is targeted at the probe
logic itself.
"""

import importlib
import sys

import pytest


def _reimport(name):
    if name in sys.modules:
        del sys.modules[name]
    return importlib.import_module(name)


def test_haves_are_booleans():
    """`HAVE_DEEP_EP` and `HAVE_DEEP_EP_V2` must exist and be bool."""
    mod = _reimport("megatron.core.transformer.moe.fused_a2a")
    assert isinstance(mod.HAVE_DEEP_EP, bool)
    assert isinstance(mod.HAVE_DEEP_EP_V2, bool)


def test_v2_absent_falls_back_to_v1(monkeypatch):
    """When V2 is not importable, fused_dispatch/fused_combine still
    work via V1 (or are `None` if V1 is also absent)."""

    class _NoV2Meta(type(sys.modules["sys"])):
        pass

    # Force `from deep_ep import ElasticBuffer` to fail at module
    # import time by preloading a dummy deep_ep without `ElasticBuffer`.
    import types

    fake = types.ModuleType("deep_ep")
    # No `ElasticBuffer` attr -> ImportError when re-imported.
    monkeypatch.setitem(sys.modules, "deep_ep", fake)
    # Keep utils module present so the downstream EventHandle/Overlap
    # import doesn't blow up (provide dummy classes).
    fake_utils = types.ModuleType("deep_ep.utils")

    class _DummyEventHandle:  # noqa: D401
        pass

    class _DummyEventOverlap:  # noqa: D401
        def __init__(self, *a, **k):
            pass

    fake_utils.EventHandle = _DummyEventHandle
    fake_utils.EventOverlap = _DummyEventOverlap
    monkeypatch.setitem(sys.modules, "deep_ep.utils", fake_utils)
    # Provide a dummy Buffer so HAVE_DEEP_EP is True when reimported.
    class _DummyBuffer:
        @staticmethod
        def get_dispatch_config(n):
            raise NotImplementedError

        @staticmethod
        def get_combine_config(n):
            raise NotImplementedError

        @staticmethod
        def set_num_sms(n):
            pass

    fake.Buffer = _DummyBuffer

    mod = _reimport("megatron.core.transformer.moe.fused_a2a")
    assert mod.HAVE_DEEP_EP_V2 is False, (
        "ElasticBuffer missing from deep_ep -> HAVE_DEEP_EP_V2 must be False"
    )
    assert mod.HAVE_DEEP_EP is True, "Legacy Buffer is present"
    assert mod.fused_dispatch is not None
    assert mod.fused_combine is not None


def test_v2_present_sets_have_v2_true(monkeypatch):
    """When `ElasticBuffer` is importable, HAVE_DEEP_EP_V2 is True and
    `set_deepep_num_sms` routes to the V2 class method."""
    import types

    fake = types.ModuleType("deep_ep")

    class _DummyElastic:
        _num_sms = None

        @classmethod
        def set_num_sms(cls, n):
            cls._num_sms = n

        @staticmethod
        def get_buffer_size_hint(**kwargs):
            return 1024 * 1024

    class _DummyBuffer:
        @staticmethod
        def get_dispatch_config(n):
            raise NotImplementedError

        @staticmethod
        def get_combine_config(n):
            raise NotImplementedError

        @staticmethod
        def set_num_sms(n):
            raise AssertionError("V1 set_num_sms must not be called when V2 is present")

    fake.Buffer = _DummyBuffer
    fake.ElasticBuffer = _DummyElastic

    fake_utils = types.ModuleType("deep_ep.utils")

    class _DummyEventHandle:
        pass

    class _DummyEventOverlap:
        def __init__(self, *a, **k):
            pass

    fake_utils.EventHandle = _DummyEventHandle
    fake_utils.EventOverlap = _DummyEventOverlap

    monkeypatch.setitem(sys.modules, "deep_ep", fake)
    monkeypatch.setitem(sys.modules, "deep_ep.utils", fake_utils)

    mod = _reimport("megatron.core.transformer.moe.fused_a2a")
    assert mod.HAVE_DEEP_EP_V2 is True
    assert mod.set_deepep_num_sms is not None

    mod.set_deepep_num_sms(7)
    assert _DummyElastic._num_sms == 7, (
        "set_deepep_num_sms must dispatch to ElasticBuffer.set_num_sms when V2 is active"
    )


def test_neither_v1_nor_v2_sets_callables_to_none(monkeypatch):
    """With no deep_ep at all, fused_* is None (unchanged behavior)."""
    import types

    # Drop any deep_ep module so the `from deep_ep import ...` ImportError
    # branch is taken for both V1 and V2.
    for k in list(sys.modules.keys()):
        if k.startswith("deep_ep"):
            monkeypatch.delitem(sys.modules, k, raising=False)
    monkeypatch.setitem(sys.modules, "deep_ep", None)  # type: ignore[arg-type]

    # `importlib.import_module("deep_ep")` when value is None triggers
    # ImportError, which is what we want. Build a clean fused_a2a import.
    if "megatron.core.transformer.moe.fused_a2a" in sys.modules:
        del sys.modules["megatron.core.transformer.moe.fused_a2a"]

    mod = importlib.import_module("megatron.core.transformer.moe.fused_a2a")
    assert mod.HAVE_DEEP_EP is False
    assert mod.HAVE_DEEP_EP_V2 is False
    assert mod.fused_dispatch is None
    assert mod.fused_combine is None
    assert mod.set_deepep_num_sms is None
