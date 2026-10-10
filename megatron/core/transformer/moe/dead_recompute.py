# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Skip the dead tail of an activation-recompute replay of a MoE layer.

Under activation recompute (``--recompute-granularity full`` or ``selective`` with ``moe``), the
backward pass re-runs the layer's forward to rebuild the autograd graph and the saved tensors.
The re-run's down projection (``linear_fc2``) and its expert-parallel ``combine`` are dead work:
nothing in the backward reads their outputs.  ``fc2`` is linear -- its backward needs its input
(the activation output) and its weight, never its output; the combine's backward is a cached
dispatch through the routing handle, which the dispatch produced.  With the routing weights
applied on the activation output (``bias_act_func``) the fc2 output feeds the combine alone, and
the combine output feeds the residual add alone.  Only the recomputed output values are garbage;
``torch.autograd.backward(outputs, args)`` starts from the graph, not from the values.

This module realises the skip with the smallest possible surface:

* :func:`recompute_phase` -- a context the reentrant ``CheckpointFunction.backward`` enters
  around its re-run of the forward.  It is the exact signal for "this forward builds the graph
  for a backward that already has its cotangent", independent of the grad mode or of which
  layers are checkpointed (``recompute_method='block'`` leaves some layers unchecked).
* :func:`skipping_dead_recompute` -- whether the MoE layer should skip its dead tail now:
  the config switch ``moe_skip_dead_recompute`` and :func:`in_recompute_phase`.
* :func:`skip_te_grouped_gemm` -- a context under which Transformer Engine's
  ``_GroupedLinear.forward`` allocates its output and saves its tensors exactly as it always
  does but launches no GEMM: the module-level ``general_grouped_gemm`` it calls is replaced by a
  no-op for the duration.  The backward is untouched.  (The clean long-term form is a
  ``skip_compute`` switch inside Transformer Engine; this keeps the experiment inside Megatron.)

The combine's skip is a ``skip_compute`` argument of ``fused_combine`` (``fused_a2a.py``): the
forward returns an uninitialised tensor of the combined shape, the backward is the same cached
dispatch.
"""

import contextlib
import threading

_state = threading.local()


def in_recompute_phase() -> bool:
    """True while a checkpoint function re-runs a forward to build the graph for its backward."""
    return getattr(_state, "depth", 0) > 0


@contextlib.contextmanager
def recompute_phase():
    """Mark the re-run of a checkpointed forward (entered by ``CheckpointFunction.backward``)."""
    _state.depth = getattr(_state, "depth", 0) + 1
    try:
        yield
    finally:
        _state.depth -= 1


def skipping_dead_recompute(config) -> bool:
    """Whether a MoE layer built with ``config`` skips its dead recompute tail right now."""
    return bool(getattr(config, "moe_skip_dead_recompute", False)) and in_recompute_phase()


def _noop_grouped_gemm(*args, **kwargs):
    return None


@contextlib.contextmanager
def skip_te_grouped_gemm():
    """Run a Transformer Engine ``GroupedLinear`` forward without its GEMM.

    ``_GroupedLinear.forward`` allocates ``out = torch.empty(...)`` and then calls the module-level
    ``general_grouped_gemm(weights, inputs, [out], ...)`` to fill it, then saves its tensors for the
    backward.  Replacing that name for the duration of one forward call leaves everything but the
    kernel launch in place: the output has the right shape/dtype/device (its values are never
    read), the saved tensors and the ``ctx`` are what they always are, the backward is unchanged.
    """
    try:
        from transformer_engine.pytorch.module import grouped_linear as te_grouped_linear
    except ImportError as e:  # pragma: no cover - TE is required for the grouped MLP anyway
        raise RuntimeError("moe_skip_dead_recompute requires Transformer Engine") from e
    if not hasattr(te_grouped_linear, "general_grouped_gemm"):
        raise RuntimeError(
            "moe_skip_dead_recompute: this Transformer Engine's grouped_linear module has no"
            " module-level general_grouped_gemm to skip; disable the option or port the skip"
        )
    real = te_grouped_linear.general_grouped_gemm
    te_grouped_linear.general_grouped_gemm = _noop_grouped_gemm
    try:
        yield
    finally:
        te_grouped_linear.general_grouped_gemm = real
