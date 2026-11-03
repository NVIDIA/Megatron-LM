# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Oracle cases for the GLU options and gated squared ReLU of the inference MoE GLU kernels.

PR #8566 adds ``glu_options`` (``GLUOptions``: clamp, linear offset, interleaved FC1 layout) and
``squared_relu_gate`` to ``padded_swiglu``, ``bounded_silu_mul``, ``swiglu_with_probs`` and
``weighted_silu_mul_bounded`` (plus ``weights_flat=None``). This module is stacked on that PR; the
kernels' default contract is in their own modules, whose helpers it reuses.

Contract (training's GLU, ``TEGroupedMLP``'s unfused ``glu`` and the fused
``clamped_weighted_swiglu``; ``_glu``):

* **SwiGLU with options:** ``silu(min(g, c)) * (clamp(u, -c, c) + offset) [* p]``, gate and linear
  read from ``[gate | linear]`` halves or from ``I``-wide interleaved blocks, evaluated in fp32
  (the clamp bound and offset as fp32 values) and rounded once. Tolerance class as the default
  SwiGLU: rounded once from fp32 per element and the harness policy per row. The clamp keeps NaN
  NaN, as ``torch.clamp`` does.
* **Interleaving** changes only where the inputs are read: bitwise the output of the same values
  laid out as halves.
* **Batch-invariant clamp:** bitwise training's fused ``clamped_weighted_swiglu`` (GPU, compiled);
  without weights (probabilities applied to the expert input), bitwise ``weighted_swiglu(y, 1)``.
* **Gated squared ReLU:** bitwise training's bf16 GLU with the live compiled ``squared_relu``
  (``relu(min(g, c))**2 * (clamp(u, -c, c) + offset)``, each step a bf16 op), times the fp32
  probability rounded to bf16 where the kernel applies one; NaN and infinities included.

Options: clamp 7.0 (GPT-OSS) and 7.3 (not bf16-representable), offset 1.0, interleave 1, 32 and
64, and all combined, with and without zero padding, weights, and live counts inside the buffer.
"""

from typing import Optional

import pytest
import torch

from megatron.core.inference.moe import activations
from tests.unit_tests.inference.oracles import oracle_utils as ou
from tests.unit_tests.inference.oracles.moe_activations import _common, _glu
from tests.unit_tests.inference.oracles.moe_activations import (
    test_silu_mul_bounded_kernel as silu_mul,
)
from tests.unit_tests.inference.oracles.moe_activations import test_swiglu_kernel as swiglu
from tests.unit_tests.inference.oracles.moe_activations import (
    test_swiglu_with_probs_kernel as with_probs,
)
from tests.unit_tests.inference.oracles.moe_activations import (
    test_weighted_silu_mul_bounded_kernel as weighted_kernel,
)
from tests.unit_tests.inference.oracles.moe_activations._common import DEVICE

pytestmark = _common.hardware(ou.HOPPER)

OPTIONS = {
    "clamp": dict(clamp=7.0),
    "clamp-not-bf16": dict(clamp=7.3),
    "offset": dict(offset=1.0),
    "interleave-32": dict(interleave=32),
    "all": dict(clamp=7.0, offset=1.0, interleave=64),
}
SWIGLU_OPTIONS = {
    **OPTIONS,
    "clamp-offset": dict(clamp=7.0, offset=1.0),
    "interleave-1": dict(interleave=1),
}


def glu_kwargs(
    clamp: Optional[float] = None,
    offset: float = 0.0,
    interleave: Optional[int] = None,
    squared_relu_gate: bool = False,
) -> dict:
    """Keyword arguments selecting the GLU options; empty for the default (main's API)."""
    if clamp is None and offset == 0.0 and interleave is None and not squared_relu_gate:
        return {}
    kwargs = dict(
        glu_options=activations.GLUOptions(
            clamp_value=clamp, linear_offset=offset, interleave_size=interleave
        )
    )
    if squared_relu_gate:
        kwargs["squared_relu_gate"] = True
    return kwargs


def interleave_halves(x: torch.Tensor, interleave: int) -> torch.Tensor:
    """The inverse of ``split_halves``: ``[gate | linear]`` rows rewritten in the interleaved
    FC1 layout with blocks of ``interleave`` columns."""
    n = x.shape[-1] // 2
    gate = x[..., :n].reshape(*x.shape[:-1], n // interleave, 1, interleave)
    linear = x[..., n:].reshape(*x.shape[:-1], n // interleave, 1, interleave)
    return torch.cat([gate, linear], dim=-2).reshape(x.shape).contiguous()


def gated_squared_relu_training(
    x: torch.Tensor,
    probs: Optional[torch.Tensor] = None,
    *,
    clamp: Optional[float] = None,
    offset: float = 0.0,
    interleave: Optional[int] = None,
) -> torch.Tensor:
    """Training's gated squared ReLU in bf16, with the live training ops.

    ``TEGroupedMLP``'s unfused ``glu`` is the only training path for ``activation_func =
    squared_relu`` with ``gated_linear_unit``: each step is a bf16 torch op (clamp, the
    compiled ``squared_relu``, ``linear + offset``, the product), and the probability multiply
    promotes to fp32 and casts back to bf16. PR #8566 claims bitwise parity with this.
    """
    from megatron.core.activations import squared_relu

    gate, linear = _glu.split_halves(x, interleave)
    if clamp is not None:
        gate = gate.clamp(min=None, max=clamp)
        linear = linear.clamp(min=-clamp, max=clamp)
    out = squared_relu(gate) * (linear + offset)
    if probs is not None:
        out = (out * probs.reshape(-1, 1)).to(x.dtype)
    return out


# ---------------------------------------------------------------------------------------------
# padded_swiglu (_swiglu_kernel)
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(SWIGLU_OPTIONS))
@pytest.mark.parametrize("zero_padding", [False, True])
def test_padded_swiglu_options(name, zero_padding):
    """SwiGLU with training's clamp, linear offset and interleaved layout (``_glu``): fp32
    compute and one rounding, as for the default options."""
    options = SWIGLU_OPTIONS[name]
    g = ou.make_generator(device=DEVICE)
    rows, width, n_used = swiglu.PROGRAMS_CAP + 1, 1856, 400
    x = _glu.glu_inputs((rows, 2 * width), g, DEVICE, scale=4.0)  # some |x| > 7: clamp bites
    perm = _glu.permutation_map(rows, "runs", torch.int32, g, DEVICE)
    out = swiglu.run_padded_swiglu(
        x, perm, _common.device_scalar(n_used), zero_padding, **glu_kwargs(**options)
    )
    swiglu.check_rows(out, x, perm, n_used, zero_padding, what=name, **options)


@pytest.mark.parametrize("name", ["clamp", "clamp-not-bf16", "offset", "clamp-offset"])
def test_padded_swiglu_options_special_values(name):
    """The clamp keeps NaN NaN (as torch.clamp does) and bounds infinite inputs."""
    options = SWIGLU_OPTIONS[name]
    for i, x in enumerate(_glu.special_value_inputs(DEVICE)):
        rows = x.shape[0]
        perm = torch.arange(rows, dtype=torch.int32, device=DEVICE)
        out = swiglu.run_padded_swiglu(
            x, perm, _common.device_scalar(rows), **glu_kwargs(**options)
        )
        _glu.assert_swiglu_special_values(out, x, what=f"{name}, grid {i}", **options)


@pytest.mark.parametrize("interleave", [1, 32, 64])
def test_padded_swiglu_interleaving_is_a_relayout(interleave):
    """The interleaved FC1 layout gives bitwise the same output as the same values laid out
    as ``[gate | linear]`` halves."""
    g = ou.make_generator(device=DEVICE)
    rows, width = 300, 1856
    halves = _glu.glu_inputs((rows, 2 * width), g, DEVICE)
    perm = _glu.permutation_map(rows, "random", torch.int32, g, DEVICE)
    n_used = _common.device_scalar(rows)
    expected = swiglu.run_padded_swiglu(halves, perm, n_used, zero_padding=True)
    interleaved = interleave_halves(halves, interleave)
    actual = swiglu.run_padded_swiglu(
        interleaved, perm, n_used, zero_padding=True, **glu_kwargs(interleave=interleave)
    )
    ou.assert_exact(actual, expected, what=f"interleave {interleave} vs halves")


@pytest.mark.parametrize("name", ["none"] + list(SWIGLU_OPTIONS))
def test_padded_swiglu_gated_squared_relu_matches_training(name):
    """``squared_relu_gate``: bitwise training's gated squared ReLU (``relu(gate)^2 * linear``
    with training's per-op bf16 roundings, ``gated_squared_relu_training``), NaN and
    infinities included."""
    options = SWIGLU_OPTIONS.get(name, {})
    g = ou.make_generator(device=DEVICE)
    rows, width, n_used = swiglu.PROGRAMS_CAP + 1, 1856, 500
    x = _glu.glu_inputs((rows, 2 * width), g, DEVICE, scale=4.0)
    x[7, 3] = float("nan")
    x[8, width + 5] = float("nan")
    x[9, 11] = float("inf")
    x[10, width + 12] = float("-inf")
    perm = _glu.permutation_map(rows, "runs", torch.int32, g, DEVICE)
    out = swiglu.run_padded_swiglu(
        x,
        perm,
        _common.device_scalar(n_used),
        zero_padding=True,
        **glu_kwargs(squared_relu_gate=True, **options),
    )
    index = torch.arange(rows, device=out.device)
    live = (index < n_used) & (perm >= 0)
    expected = gated_squared_relu_training(x[live], **options)
    _common.assert_exact_except_nan_payload(out[live], expected, what=f"gated sq-ReLU {name}")
    padding = (index < n_used) & ~live
    ou.assert_exact(out[padding], torch.zeros_like(out[padding]), what="padding rows are +0")
    ou.assert_poisoned(out[index >= n_used], what="rows past n_used")


# ---------------------------------------------------------------------------------------------
# bounded_silu_mul (_silu_mul_bounded_kernel)
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(OPTIONS))
def test_bounded_silu_mul_options(name):
    """SwiGLU with training's clamp, linear offset and interleaved FC1 layout."""
    options = OPTIONS[name]
    g = ou.make_generator(device=DEVICE)
    rows, width, n_rows = silu_mul.PROGRAMS_CAP + 1, 1856, 400
    x = _glu.glu_inputs((rows, 2 * width), g, DEVICE, scale=4.0)
    out = silu_mul.run_bounded_silu_mul(x, _common.device_scalar(n_rows), **glu_kwargs(**options))
    silu_mul.check_rows(out, x, n_rows, what=name, **options)


@pytest.mark.parametrize("name", ["none"] + list(OPTIONS))
def test_bounded_silu_mul_gated_squared_relu_matches_training(name):
    """``squared_relu_gate``: bitwise training's gated squared ReLU, NaN and infinities
    included."""
    options = OPTIONS.get(name, {})
    g = ou.make_generator(device=DEVICE)
    rows, width, n_rows = silu_mul.PROGRAMS_CAP + 1, 1856, 450
    x = _glu.glu_inputs((rows, 2 * width), g, DEVICE, scale=4.0)
    x[7, 3] = float("nan")
    x[8, width + 5] = float("nan")
    x[9, 11] = float("inf")
    x[10, width + 12] = float("-inf")
    out = silu_mul.run_bounded_silu_mul(
        x, _common.device_scalar(n_rows), **glu_kwargs(squared_relu_gate=True, **options)
    )
    expected = gated_squared_relu_training(x[:n_rows], **options)
    _common.assert_exact_except_nan_payload(out[:n_rows], expected, what=f"gated sq-ReLU {name}")
    ou.assert_poisoned(out[n_rows:], what="rows past n_rows")


# ---------------------------------------------------------------------------------------------
# swiglu_with_probs (_swiglu_with_probs_kernel)
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(OPTIONS))
@pytest.mark.parametrize("zero_padding", [False, True])
def test_swiglu_with_probs_options(name, zero_padding):
    """SwiGLU with training's clamp, linear offset and interleaved layout, times the
    probability: fp32 compute, one rounding."""
    options = OPTIONS[name]
    g = ou.make_generator(device=DEVICE)
    x, perm, probs = with_probs.make_inputs(with_probs.PROGRAMS_CAP + 1, 1856, g, "runs", scale=4.0)
    out = with_probs.run_swiglu_with_probs(
        x, perm, _common.device_scalar(400), probs, zero_padding, **glu_kwargs(**options)
    )
    with_probs.check_rows(out, x, perm, probs, 400, zero_padding, what=name, **options)


@_common.needs_cuda
@pytest.mark.parametrize("clamp", [7.0, 7.3])
@pytest.mark.parametrize("interleave", [None, 32])
def test_swiglu_with_probs_clamp_matches_training_bitwise(clamp, interleave):
    """The clamp is bitwise training's fused ``clamped_weighted_swiglu`` (which batch-invariant
    training runs for ``activation_func_clamp_value``), on the de-interleaved halves."""
    from megatron.core.fusions.fused_bias_swiglu import clamped_weighted_swiglu

    g = ou.make_generator(device=DEVICE)
    halves, probs = with_probs.parity_inputs(1856, g)
    halves = (halves.float() * 4).to(torch.bfloat16)
    rows = halves.shape[0]
    perm = torch.arange(rows, dtype=torch.int32, device=DEVICE)
    x = halves if interleave is None else interleave_halves(halves, interleave)
    out = with_probs.run_swiglu_with_probs(
        x,
        perm,
        _common.device_scalar(rows),
        probs,
        **glu_kwargs(clamp=clamp, interleave=interleave),
    )
    expected = clamped_weighted_swiglu(halves, probs[:, None], clamp)
    ou.assert_exact(out, expected, what=f"clamp {clamp} vs training clamped_weighted_swiglu")


@pytest.mark.parametrize("name", ["none"] + list(OPTIONS))
def test_swiglu_with_probs_gated_squared_relu_matches_training(name):
    """``squared_relu_gate``: bitwise training's gated squared ReLU (bf16 ops) times the fp32
    probability rounded to bf16, NaN and infinities included."""
    options = OPTIONS.get(name, {})
    g = ou.make_generator(device=DEVICE)
    x, perm, probs = with_probs.make_inputs(with_probs.PROGRAMS_CAP + 1, 1856, g, "runs", scale=4.0)
    x[7, 3] = float("nan")
    x[9, 11] = float("inf")
    x[10, 1856 + 12] = float("-inf")
    n_used = 450
    out = with_probs.run_swiglu_with_probs(
        x,
        perm,
        _common.device_scalar(n_used),
        probs,
        zero_padding=True,
        **glu_kwargs(squared_relu_gate=True, **options),
    )
    live, padding, beyond = with_probs.row_masks(perm, n_used)
    expected = gated_squared_relu_training(x[live], probs[live], **options)
    _common.assert_exact_except_nan_payload(out[live], expected, what=f"gated sq-ReLU {name}")
    ou.assert_exact(out[padding], torch.zeros_like(out[padding]), what="padding rows are +0")
    ou.assert_poisoned(out[beyond], what="rows past n_used")


# ---------------------------------------------------------------------------------------------
# weighted_silu_mul_bounded (_weighted_silu_mul_bounded_kernel)
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(OPTIONS))
@pytest.mark.parametrize("weighted", [True, False], ids=["weights", "no-weights"])
def test_weighted_silu_mul_bounded_options(name, weighted):
    """SwiGLU with training's clamp, linear offset and interleaved layout, with or without
    weights: fp32 compute, one rounding."""
    options = OPTIONS[name]
    g = ou.make_generator(device=DEVICE)
    rows, half_n = 300, 1856
    y = _glu.glu_inputs((rows, 2 * half_n), g, DEVICE, scale=4.0)
    weights = torch.rand(rows, generator=g, device=DEVICE, dtype=torch.float32)
    weights = weights if weighted else None
    bound = 211 * half_n
    out = weighted_kernel.run_weighted(y, weights, bound, 3, 1024, **glu_kwargs(**options))
    weighted_kernel.check_elements(out, y, weights, bound, what=name, **options)


@_common.needs_cuda
@pytest.mark.parametrize("clamp", [None, 7.0, 7.3])
def test_weighted_silu_mul_bounded_matches_training_bitwise(clamp):
    """Bitwise training's fused weighted SwiGLU (``clamped_weighted_swiglu`` with a clamp),
    with weights and without (training then weights by ones)."""
    from megatron.core.fusions.fused_bias_swiglu import clamped_weighted_swiglu, weighted_swiglu

    g = ou.make_generator(device=DEVICE)
    rows, half_n = 513, 1856
    y = _glu.glu_inputs((rows, 2 * half_n), g, DEVICE, scale=4.0)
    weights = torch.rand(rows, generator=g, device=DEVICE, dtype=torch.float32)
    for w in (weights, None):
        out = weighted_kernel.run_weighted(y, w, rows * half_n, None, **glu_kwargs(clamp=clamp))
        w_train = torch.ones_like(weights) if w is None else w
        if clamp is None:
            expected = weighted_swiglu(y, w_train[:, None])
        else:
            expected = clamped_weighted_swiglu(y, w_train[:, None], clamp)
        ou.assert_exact(out, expected, what=f"clamp {clamp}, weights {w is not None}")


@pytest.mark.parametrize("name", ["none"] + list(OPTIONS))
@pytest.mark.parametrize("weighted", [True, False], ids=["weights", "no-weights"])
def test_weighted_silu_mul_bounded_gated_squared_relu_matches_training(name, weighted):
    """``squared_relu_gate``: bitwise training's gated squared ReLU (bf16 ops), times the fp32
    weight rounded to bf16 when weights are given."""
    options = OPTIONS.get(name, {})
    g = ou.make_generator(device=DEVICE)
    rows, half_n = 300, 1856
    y = _glu.glu_inputs((rows, 2 * half_n), g, DEVICE, scale=4.0)
    y[7, 3] = float("nan")
    y[9, 11] = float("inf")
    weights = torch.rand(rows, generator=g, device=DEVICE, dtype=torch.float32)
    weights = weights if weighted else None
    live = 211
    out = weighted_kernel.run_weighted(
        y, weights, live * half_n, 3, 1024, **glu_kwargs(squared_relu_gate=True, **options)
    )
    expected = gated_squared_relu_training(
        y[:live], None if weights is None else weights[:live], **options
    )
    _common.assert_exact_except_nan_payload(out[:live], expected, what=f"gated sq-ReLU {name}")
    ou.assert_poisoned(out[live:], what="rows past the bound")
