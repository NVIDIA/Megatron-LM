# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Loading the cuDNN frontend DSA namespace must not pin the caller's tensors.

``cudnn.DSA`` imports the kernel modules behind an entry point on first attribute access.
CUTLASS DSL 4.6 kernel decorators keep ``inspect.currentframe()`` in a local, so every
decoration leaves a self-referencing frame whose ``f_back`` chain reaches the frame that
triggered the import. With automatic GC disabled (``--manual-gc``), a first access from
inside a training step therefore held that step's activations until the next explicit
collection. These CPU tests model that frontend and check that MCore resolves every entry
point in one call and leaves nothing of the caller alive.
"""

import gc
import inspect
import re
import sys
import types
import weakref

import pytest

from megatron.core.transformer.experimental_attention_variant import dsa_cudnn_kernels


def _decorate_like_cutlass_dsl_4_6():
    """Mimic ``CuTeDSL.kernel`` of CUTLASS DSL 4.6: keep the own frame in a local."""
    cur_frame = inspect.currentframe()
    frame = cur_frame.f_back
    return frame.f_lineno


class _LazyDSANamespace:
    """``cudnn.DSA`` stand-in: imports (decorates) an entry point on first access."""

    def __init__(self, names):
        self.available = frozenset(names)
        self.loaded = []

    def __getattr__(self, name):
        if name not in self.available:
            raise AttributeError(f"DSA has no attribute {name!r}")
        self.loaded.append(name)
        _decorate_like_cutlass_dsl_4_6()

        def entry_point(*args, **kwargs):
            return None

        setattr(self, name, entry_point)
        return entry_point


class _Activation:
    """Stands for a tensor that is live on the Python stack."""


@pytest.fixture
def manual_gc():
    enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if enabled:
            gc.enable()


def _install(monkeypatch, namespace):
    monkeypatch.setitem(sys.modules, "cudnn", types.SimpleNamespace(DSA=namespace))
    monkeypatch.setattr(dsa_cudnn_kernels, "_cudnn_dsa", None)
    return namespace


def test_namespace_resolves_every_entry_point_once(monkeypatch):
    namespace = _install(monkeypatch, _LazyDSANamespace(dsa_cudnn_kernels._CUDNN_DSA_ENTRY_POINTS))
    dsa_cudnn_kernels._ensure_dsa_namespace()
    dsa_cudnn_kernels._ensure_dsa_namespace()
    assert dsa_cudnn_kernels._cudnn_dsa is namespace
    assert sorted(namespace.loaded) == sorted(dsa_cudnn_kernels._CUDNN_DSA_ENTRY_POINTS)


def test_entry_point_missing_from_the_frontend_is_left_to_its_call_site(monkeypatch):
    available = set(dsa_cudnn_kernels._CUDNN_DSA_ENTRY_POINTS) - {"SparseAttentionBackward"}
    namespace = _install(monkeypatch, _LazyDSANamespace(available))
    dsa_cudnn_kernels._ensure_dsa_namespace()
    assert dsa_cudnn_kernels._cudnn_dsa is namespace
    assert getattr(dsa_cudnn_kernels._cudnn_dsa, "SparseAttentionBackward", None) is None


def test_lazy_frontend_model_pins_the_caller_without_a_collection(manual_gc):
    """Positive control: the modelled frontend leaks exactly like the real one."""
    namespace = _LazyDSANamespace(["sparse_attention_backward_wrapper"])

    def backward_step():
        activation = _Activation()
        _ = namespace.sparse_attention_backward_wrapper
        return weakref.ref(activation)

    activation = backward_step()
    assert activation() is not None
    gc.collect()
    assert activation() is None


def test_first_use_inside_a_step_does_not_pin_the_step(monkeypatch, manual_gc):
    _install(monkeypatch, _LazyDSANamespace(dsa_cudnn_kernels._CUDNN_DSA_ENTRY_POINTS))

    def backward_step():
        activation = _Activation()
        dsa_cudnn_kernels._ensure_dsa_namespace()
        dsa_cudnn_kernels._cudnn_dsa.sparse_attention_backward_wrapper()
        return weakref.ref(activation)

    assert backward_step()() is None


def test_entry_point_list_covers_every_namespace_access():
    source = inspect.getsource(dsa_cudnn_kernels)
    used = set(re.findall(r"_cudnn_dsa\.([A-Za-z_]\w*)", source))
    used |= set(re.findall(r"getattr\(\s*_cudnn_dsa,\s*\"([A-Za-z_]\w*)\"", source))
    assert used
    assert used <= set(dsa_cudnn_kernels._CUDNN_DSA_ENTRY_POINTS)
