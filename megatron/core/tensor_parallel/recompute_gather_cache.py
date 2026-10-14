# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Reuse the sequence-parallel all-gather of a recomputed linear's input for its weight gradient.

Under selective activation recompute (``mla_up_proj``, ``mlp``, ``moe``, ...) the backward re-runs
a checkpointed forward to rebuild the autograd graph.  Every column-parallel Transformer Engine
linear inside that re-run gathers its sequence-parallel input along the first dimension for the
forward GEMM, saves the LOCAL input for the backward, and its backward gathers the same local
input AGAIN for the weight-gradient GEMM (``_Linear.backward`` / ``_LayerNormLinear.backward``:
``inputmat_total, handle = gather_along_first_dim(inputmat, tp_group, async_op=True)``).  The two
gathers have identical inputs and identical results; the first result is dropped right after the
forward GEMM because, in the general (non-recompute) case, keeping it alive until the backward
would hold a gathered activation per layer.  Inside a recompute re-run the backward follows
within the same layer's backward, so keeping the gathered input alive costs one tensor for a few
statements and saves one all-gather per recomputed linear per microbatch.

This module realises that with the smallest surface Megatron can offer without a Transformer
Engine change:

* :func:`recompute_phase` / :func:`in_recompute_phase` -- a context the checkpoint re-runs enter
  (``CheckpointFunction.backward`` and ``CheckpointWithoutOutput._recompute`` in ``random.py``).
* :func:`install` -- replaces the module-level name ``gather_along_first_dim`` that Transformer
  Engine's ``linear`` and ``layernorm_linear`` modules call by :func:`_cached_gather`.  Outside a
  recompute phase and for quantized gathers it is the original function.  Inside a recompute
  phase a gather is performed once and its result is kept, keyed by the identity of the local
  input (storage pointer, shape, strides, dtype, device) together with a strong reference to the
  local input (so its storage cannot be reused while the entry lives) and its version counter (so
  an in-place change invalidates it).  A later gather of the same unchanged local input -- the
  backward's -- returns the kept result instead of communicating, once (the entry is consumed).
  The caller receives a fresh view object of the kept tensor, so Transformer Engine's
  ``clear_tensor_data`` on its own reference cannot empty the kept one.
* A leftover entry (a recomputed linear whose backward never gathered) is dropped at the next
  recompute phase and counted; counters are printed at exit on rank 0.

Enabled by ``TransformerConfig.recompute_reuse_gathered_input``; ``install`` is called when a
Transformer Engine linear is built with that flag.  The clean long-term form is a Transformer
Engine option that hands a precomputed gathered input to the backward; this keeps the experiment
inside Megatron.
"""

import atexit
import contextlib
import threading
from typing import Any, Dict, Optional, Tuple

import torch

_state = threading.local()


def in_recompute_phase() -> bool:
    """True while a checkpoint function re-runs a forward to build the graph for its backward."""
    return getattr(_state, "depth", 0) > 0


@contextlib.contextmanager
def recompute_phase():
    """Mark the re-run of a checkpointed forward.

    Entering the outermost phase drops entries a previous re-run left unconsumed: their backward
    never gathered (or gathered through another path), so nothing will ever read them.
    """
    depth = getattr(_state, "depth", 0)
    if depth == 0 and _cache:
        _stats["unconsumed"] += len(_cache)
        _cache.clear()
    _state.depth = depth + 1
    try:
        yield
    finally:
        _state.depth -= 1


class _Entry:
    __slots__ = ("gathered", "local", "version")

    def __init__(self, gathered: torch.Tensor, local: torch.Tensor):
        self.gathered = gathered
        self.local = local  # keeps the local input's storage alive: the key cannot be reused
        self.version = local._version


_cache: Dict[Tuple, _Entry] = {}
_stats = {"stored": 0, "hits": 0, "unconsumed": 0, "invalid": 0, "bypassed": 0}
_originals: Dict[str, Any] = {}
_installed = False


def _key(t: torch.Tensor) -> Tuple:
    return (t.data_ptr(), tuple(t.shape), tuple(t.stride()), t.dtype, t.device.index)


def _cached_gather(original):
    def gather(inp, process_group, async_op: bool = False, quantizer=None, *args, **kwargs):
        if quantizer is not None or args or kwargs or not isinstance(inp, torch.Tensor):
            _stats["bypassed"] += 1
            return original(inp, process_group, async_op, quantizer, *args, **kwargs)
        key = _key(inp)
        entry = _cache.pop(key, None)
        if entry is not None:
            if (
                entry.local is inp or entry.local.data_ptr() == inp.data_ptr()
            ) and entry.version == inp._version and entry.gathered.untyped_storage().size() > 0:
                _stats["hits"] += 1
                return entry.gathered.view_as(entry.gathered), None
            _stats["invalid"] += 1
        if not in_recompute_phase():
            return original(inp, process_group, async_op, quantizer)
        out, handle = original(inp, process_group, async_op, quantizer)
        if handle is not None:
            # the kept result must be complete before its second use; the caller's own wait
            # happens before the GEMM anyway, so waiting here changes nothing but simplicity
            handle.wait()
            handle = None
        _cache[key] = _Entry(out, inp)
        _stats["stored"] += 1
        return out.view_as(out), handle

    gather.__wrapped__ = original
    return gather


_TE_MODULES = (
    "transformer_engine.pytorch.module.linear",
    "transformer_engine.pytorch.module.layernorm_linear",
)


def install() -> None:
    """Replace ``gather_along_first_dim`` in Transformer Engine's linear modules (idempotent)."""
    global _installed
    if _installed:
        return
    import importlib

    for name in _TE_MODULES:
        try:
            module = importlib.import_module(name)
        except ImportError as e:  # pragma: no cover - TE is required for these linears anyway
            raise RuntimeError(f"recompute_reuse_gathered_input requires Transformer Engine ({name})") from e
        fn = getattr(module, "gather_along_first_dim", None)
        if fn is None or not callable(fn):
            raise RuntimeError(
                f"recompute_reuse_gathered_input: {name} has no module-level gather_along_first_dim"
                " to reuse; disable the option or port the cache to this Transformer Engine"
            )
        _originals[name] = fn
        setattr(module, "gather_along_first_dim", _cached_gather(fn))
    _installed = True
    atexit.register(_report)


def stats() -> Dict[str, int]:
    """Counters: stored (gathers kept in a recompute phase), hits (backward gathers served from
    the cache), unconsumed (entries dropped at the next recompute phase), invalid (stale entries
    refused), bypassed (quantized or unexpected calls passed through)."""
    return dict(_stats)


def _report() -> None:
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            if torch.distributed.get_rank() != 0:
                return
    except Exception:  # pragma: no cover - reporting must never fail the process
        return
    print(f"recompute_gather_cache: {_stats}", flush=True)
