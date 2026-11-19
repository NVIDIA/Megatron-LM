# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""The per-op fused mHC operations of fused_mhc_glm_projection.

The fixed-launch Triton Sinkhorn that per-op ``sinkhorn`` binds: the CPU tests pin the warp
constant, the signatures and the fail-closed checks. The GPU tests check the accuracy against
float64, the bf16 path, the num_warps that each launch applies, a NaN smoke test at the
production shape under torch's deterministic memory fill, and the shared backward kernel's
barrier at n = 8. CI runs every test on several ranks at once, each on its own GPU.

The GLM projection pair of per-op ``proj_rms`` (the ``glm_pair`` tests): the CPU tests pin the
frozen configs, the signatures, the split plans against the float64 emulator's, the refusals,
the precision predicate, the fail-closed kernel lookup, the source rules and the interpreted
TF32 rounding helper. The GPU tests run on GB300 (compute capability (10, 3)) only, the device
the pair is validated on: float64 non-inferiority, emulator agreement, the bitwise epilogue,
the order, rounding and mutation checks, the drop-in contract, determinism, workspaces, memory
and the lowering of every kernel. The last test checks the float64 non-inferiority of
HyperConnectionModule with the pair bound per op. Every one of them runs on every rank."""

import collections
import contextlib
import copy
import inspect
import json
import os
import pickle
import re
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from megatron.core.fusions import fused_mhc_glm_projection as glm
from megatron.core.fusions import fused_mhc_kernels as fmk
from tests.unit_tests.transformer import glm_mhc_emulator as emu
from tests.unit_tests.transformer.glm_mhc_emulator import rel_rms, sinkhorn_f64

_NEEDS_CUDA_TRITON = pytest.mark.skipif(
    not (torch.cuda.is_available() and fmk.is_triton_available()), reason="needs CUDA and Triton"
)

# The production shape (T = 4096 tokens, n = 4) and layouts: HyperConnectionModule passes the
# logits as a view at offset 8 into the [T, 1, 24] mapping projection, and the upstream
# gradient arrives transposed.
T, N, ITERS, EPS = 4096, 4, 20, 1e-6
_SHAPE = (T, 1, N, N)
_LOGITS_LAYOUT = ((24, 16, 4, 1), 8, T * 24)  # strides, storage offset, storage size
_GRAD_LAYOUT = ((16, 4, 1, 4), 0, T * 16)
# rel-RMS against float64, for the output and the logits gradient.
REL_RMS_FLOOR = 1e-5


@pytest.fixture(autouse=True)
def _bind_local_cuda_device():
    """CI torchrun workers are independent replicas, each on its own local GPU."""
    if torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))


@contextlib.contextmanager
def _det_fill():
    """torch's deterministic mode with fill_uninitialized_memory: every torch.empty is NaN.

    Both settings are restored on exit.
    """
    was_deterministic = torch.are_deterministic_algorithms_enabled()
    was_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    was_fill = torch.utils.deterministic.fill_uninitialized_memory
    try:
        torch.utils.deterministic.fill_uninitialized_memory = True
        torch.use_deterministic_algorithms(True)
        yield
    finally:
        torch.use_deterministic_algorithms(was_deterministic, warn_only=was_warn_only)
        torch.utils.deterministic.fill_uninitialized_memory = was_fill


def _production_inputs(logits_seed, grad_seed, scale=1.0):
    """fp32 logits (a leaf) and a bf16-rounded upstream gradient, in the production layouts."""
    logits = torch.randn(_SHAPE, generator=torch.Generator().manual_seed(logits_seed)) * scale
    grad = torch.randn(_SHAPE, generator=torch.Generator().manual_seed(grad_seed))
    grad = grad.bfloat16().float()  # production's gradient is the fp32 upcast of a bf16 one

    def place(values, layout):
        stride, offset, numel = layout
        view = torch.zeros(numel, device="cuda").as_strided(_SHAPE, stride, offset)
        return view.copy_(values.cuda()).detach()

    return place(logits, _LOGITS_LAYOUT).requires_grad_(), place(grad, _GRAD_LAYOUT)


def _float64_reference(logits, grad, iterations=ITERS):
    """The Sinkhorn output and logits gradient in float64 on CPU, in hyper_connection's order."""
    x64 = logits.detach().cpu().double().requires_grad_()
    out64 = sinkhorn_f64(x64, iterations, EPS)
    (grad64,) = torch.autograd.grad(out64, x64, grad.detach().cpu().double())
    return out64.detach(), grad64


def _record_launches(monkeypatch):
    """Record (kernel, num_warps) for every launch of the two shared Sinkhorn JIT functions.

    The patch point is the JIT function's run method, which fused_mhc_kernels' deterministic
    Sinkhorn launch test patches too.
    """
    launches = []
    for name, tuner in (
        ("fwd", fmk._triton_sinkhorn_fwd_kernel),
        ("bwd", fmk._triton_sinkhorn_bwd_kernel),
    ):

        def record(*args, _name=name, _original=tuner.fn.run, **kwargs):
            launches.append((_name, kwargs.get("num_warps")))
            return _original(*args, **kwargs)

        monkeypatch.setattr(tuner.fn, "run", record)
    return launches


# --- CPU ------------------------------------------------------------------------------------


def test_constants_and_signature():
    """The warp constant, the public signature and the helpers' keyword-only num_warps."""
    assert glm.SINKHORN_NUM_WARPS == 1
    parameters = inspect.signature(glm.fixed_launch_sinkhorn).parameters
    assert list(parameters) == ["input_logits", "num_iterations", "eps"]
    assert parameters["eps"].default == 1e-6
    # The helpers resolve num_warps at each launch. A default bound when the function is
    # defined would not follow a changed SINKHORN_NUM_WARPS, while the per-rank line would.
    for helper in (glm._sinkhorn_fwd, glm._sinkhorn_bwd):
        num_warps = inspect.signature(helper).parameters["num_warps"]
        assert num_warps.kind is inspect.Parameter.KEYWORD_ONLY
        assert num_warps.default is None


def test_kernel_lookup_requires_triton(monkeypatch):
    """Without Triton the kernel lookup raises instead of returning missing kernels."""
    monkeypatch.setattr(fmk, "is_triton_available", lambda: False)
    with pytest.raises(RuntimeError, match="needs Triton"):
        glm._sinkhorn_kernels()


def test_rejects_non_cuda_input(monkeypatch):
    """A CPU tensor fails closed instead of running the native Sinkhorn, with or without Triton."""
    logits = torch.zeros(2, 1, N, N, requires_grad=True)
    for available in (True, False):
        monkeypatch.setattr(fmk, "is_triton_available", lambda available=available: available)
        with pytest.raises(RuntimeError, match="needs CUDA tensors"):
            glm.fixed_launch_sinkhorn(logits, ITERS)
        with pytest.raises(RuntimeError, match="needs CUDA tensors"):
            glm._sinkhorn_bwd(logits.detach(), logits.detach(), ITERS)


# --- GPU ------------------------------------------------------------------------------------


@_NEEDS_CUDA_TRITON
@pytest.mark.parametrize("scale", [1.0, 10.0], ids=["x1", "x10"])
def test_fp32_bar_against_float64(scale):
    """The output and the logits gradient are within the rel-RMS floor of float64."""
    logits, grad = _production_inputs(1234, 4321, scale)
    out = glm.fixed_launch_sinkhorn(logits, ITERS, EPS)
    (logits_grad,) = torch.autograd.grad(out, logits, grad)
    out64, grad64 = _float64_reference(logits, grad)
    for name, actual, reference in (
        ("output", out, out64),
        ("logits gradient", logits_grad, grad64),
    ):
        assert torch.isfinite(actual).all(), name
        error = rel_rms(actual.detach().cpu(), reference)
        assert error <= REL_RMS_FLOOR, f"{name}: rel-RMS {error:.3e} > {REL_RMS_FLOOR:g}"


@_NEEDS_CUDA_TRITON
def test_bf16_input_matches_float64_reference():
    """A bf16 input runs the same kernels, with the output and M_init in bf16."""
    shape = (8, 4, N, N)
    logits = torch.randn(shape, generator=torch.Generator().manual_seed(1234)).bfloat16()
    grad = torch.randn(shape, generator=torch.Generator().manual_seed(4321)).bfloat16()
    x = logits.cuda().requires_grad_()
    out = glm.fixed_launch_sinkhorn(x, ITERS, EPS)
    (x_grad,) = torch.autograd.grad(out, x, grad.cuda())
    assert out.dtype == x_grad.dtype == torch.bfloat16
    out64, grad64 = _float64_reference(logits, grad)
    # The bf16 tolerances of fused_mhc_kernels' tests.
    torch.testing.assert_close(out.detach().cpu().double(), out64, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(x_grad.cpu().double(), grad64, atol=5e-2, rtol=5e-2)


@_NEEDS_CUDA_TRITON
@pytest.mark.parametrize("case", ["default", "deterministic", "patched_constant"])
def test_launches_use_fixed_num_warps(monkeypatch, case):
    """Both launches apply SINKHORN_NUM_WARPS, read at each launch, in either torch mode.

    fused_mhc_kernels' own wrapper launches 4 warps under torch's deterministic mode; this
    launcher has no deterministic branch. A patched constant reaches the launch, as it reaches
    the per-rank fused-ops line (test_mhc_fused_ops.py).
    """
    expected = 1
    if case == "patched_constant":
        monkeypatch.setattr(glm, "SINKHORN_NUM_WARPS", 2)
        expected = 2
    logits, grad = _production_inputs(1234, 4321)
    launches = _record_launches(monkeypatch)
    mode = _det_fill() if case == "deterministic" else contextlib.nullcontext()
    with mode:
        out = glm.fixed_launch_sinkhorn(logits, ITERS, EPS)
        (logits_grad,) = torch.autograd.grad(out, logits, grad)
    assert launches == [("fwd", expected), ("bwd", expected)]
    assert torch.isfinite(out).all() and torch.isfinite(logits_grad).all()


@_NEEDS_CUDA_TRITON
def test_det_fill_nan_free_at_production_shape():
    """A NaN smoke test: 100 calls under det-fill stay finite and repeat bit for bit.

    Under det-fill every torch.empty, the backward's three workspaces included, starts as NaN,
    so an output, M_init or gradient element that no program writes stays NaN; the positive
    control shows this for a grid one program short. The calls alternate two input sets, so a
    read of memory the call did not write shows up as a bit change. At one warp and n = 4 no
    race of the reverse pass can be provoked: test_shared_backward_kernel_barrier_n8 covers the
    barrier.
    """
    input_sets = [_production_inputs(1234, 4321), _production_inputs(5678, 8765)]
    first, nonfinite_calls, changed_calls = {}, [], []
    with _det_fill():
        filled = torch.isnan(torch.empty(64, device="cuda")).all()
        assert filled, "R5 precondition: a fresh torch.empty is not all NaN under det-fill"

        # Positive control: the forward JIT function one program short at the production shape.
        fwd_kernel, _ = glm._sinkhorn_kernels()
        out = torch.empty(T, N, N, device="cuda")
        M_init = torch.empty(T, N, N, device="cuda")
        inp = input_sets[0][0].detach().contiguous().view(T, N, N)
        fwd_kernel[(T - 1,)](inp, out, M_init, T, EPS, N, ITERS, num_warps=glm.SINKHORN_NUM_WARPS)
        for name, tensor in (("out", out), ("M_init", M_init)):
            flagged = torch.isnan(tensor[-1]).all() and torch.isfinite(tensor[:-1]).all()
            assert flagged, f"R5 positive control: {name} is not NaN in exactly its last matrix"

        for call in range(100):
            index = call % len(input_sets)
            logits, grad = input_sets[index]
            out = glm.fixed_launch_sinkhorn(logits, ITERS, EPS)
            (logits_grad,) = torch.autograd.grad(out, logits, grad)
            if not (torch.isfinite(out).all() and torch.isfinite(logits_grad).all()):
                nonfinite_calls.append(call)
            bits = [t.detach().contiguous().view(torch.int32) for t in (out, logits_grad)]
            if index not in first:
                first[index] = [b.clone() for b in bits]
            elif not all(torch.equal(b, f) for b, f in zip(bits, first[index])):
                changed_calls.append(call)
    assert not nonfinite_calls, f"non-finite output or gradient in calls {nonfinite_calls}"
    assert not changed_calls, f"bits differ from the input set's first call in {changed_calls}"


@_NEEDS_CUDA_TRITON
@pytest.mark.parametrize("num_warps", [2, 4, 8])
def test_shared_backward_kernel_barrier_n8(monkeypatch, num_warps):
    """The shared backward kernel at n = 8 under det-fill, at warp counts that can race.

    The backward stores the recomputed matrices and sums in scratch and reads them back in the
    reverse pass; at n = 8 with several warps a read could run before another warp's store
    without the barrier. Inputs and tolerances are those of fused_mhc_kernels' deterministic
    Sinkhorn launch test. The launch record shows the requested warps were applied: at one
    warp the kernel passes without the barrier too.
    """
    generator = torch.Generator().manual_seed(193)
    data = (torch.rand((3, 1, 8, 8), generator=generator) * 0.2 - 0.1).cuda()
    grad_out = (torch.rand((3, 1, 8, 8), generator=generator) * 0.2 - 0.1).cuda()
    out64, grad64 = _float64_reference(data, grad_out, iterations=5)
    launches = _record_launches(monkeypatch)
    with _det_fill():
        out, M_init = glm._sinkhorn_fwd(data, 5, EPS, num_warps=num_warps)
        grad = glm._sinkhorn_bwd(grad_out, M_init, 5, EPS, num_warps=num_warps)
        torch.cuda.synchronize()
    assert launches == [("fwd", num_warps), ("bwd", num_warps)]
    for name, actual, reference in (("output", out, out64), ("input gradient", grad, grad64)):
        assert torch.isfinite(actual).all(), name
        torch.testing.assert_close(actual.cpu().double(), reference, atol=2e-6, rtol=2e-5, msg=name)


# =============================================================================================
# The GLM projection pair (GP-1, GP-2, GP-3 forward; GP-4p, GP-4, GP-4r backward)
# =============================================================================================
#
# Written from the M3 plan's registered rules and the float64 emulator, not from the kernels.
# Every pinned number below names the registered round-2 slot (r2_slots.json) it comes from.

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEVICE = "cuda"
# HyperConnectionModule's norm_eps under the GLM flags (layernorm_epsilon 1e-5). The kernels add
# its fp32 value, which is the emulator's EPS_FP32.
_PAIR_EPS = 1e-5
_PRODUCTION = (4096, 16384)
# The 15 M3 shapes (T x K) and the fixture shapes of slot S5.fixtures (R6 and plan 5.3.9).
_M3_SHAPES = tuple((t, k) for t in (4096, 17, 21, 112, 4095) for k in (16384, 4096, 384))
_FIXTURE_SHAPES = (
    (16, 16384),
    (520, 16384),
    (256, 512),
    (16, 128),
    (8, 512),
    (64, 16384),
    (64, 384),
)
# R2's inputs, all pair_data seed 0: (rows, x scale).
_R2_INPUTS = (("randn", 1.0), ("randn", 1e-3), ("randn", 1e3), ("heavy", 1.0), ("zero_rows", 1.0))
_DROP_IN_SHAPES = ((4096, 16384), (4095, 4096), (17, 384))
# The shipped kernel names (plan 5.3.1), by role.
_SHIPPED = {
    "gp1": "_glm_rms_fwd_kernel",
    "gp2": "_glm_proj_fwd_kernel",
    "gp2_strict": "_glm_proj_fwd_strict_kernel",
    "gp3": "_glm_proj_reduce_kernel",
    "gp4p": "_glm_proj_bwd_rowscale_kernel",
    "gp4": "_glm_proj_bwd_kernel",
    "gp4_strict": "_glm_proj_bwd_strict_kernel",
    "gp4r": "_glm_wgrad_reduce_kernel",
    "gp4r_strict": "_glm_wgrad_reduce_split_kernel",
}
_ROLE = {name: role for role, name in _SHIPPED.items()}
# R3's emulator bars (slot S7.p9_bars); the emulator's P9_M3_BARS must hold the same values.
_EMU_BARS = {
    "p": 1.1844862602519642e-07,
    "g_xn": 2.9209568810857357e-08,
    "g_W": 1.117417368749157e-07,
}
# R2: TF32 mode fused <= 1.1 x unfused and <= 1e-3; strict mode fused <= max(1.1 x unfused, 1e-5);
# s <= 1e-5. An unfused TF32-mode error below 1e-5 is IEEE class (R2's precision-class rule).
_TF32_BAR, _STRICT_FLOOR, _S_BAR, _RATIO, _IEEE_CLASS_LEVEL = 1e-3, 1e-5, 1e-5, 1.1, 1e-5
# R4's power minima of the order and RN tests.
_ORDER_MINIMUM, _RN_MINIMUM = 5e-5, 1e-4
# GP-2's chains per K as (count, length) runs (slot S5.gp2_at); GP-3's SPLIT is their number.
_GP2_CHAINS = {
    16384: ((13, 1216), (1, 576)),
    4096: ((12, 320), (1, 256)),
    512: ((8, 64),),
    384: ((6, 64),),
    128: ((2, 64),),
}
# GP-4's wgrad partition (slots S5.gp4_at and S5.fixtures): layouts, pieces per column tile
# (min, max), tokens per piece (min, max), NPROG, NSLOT and NPIECES (None where not recorded).
_GP4_PLANS = {
    (4096, 16384): (8, (3, 4), (192, 1728), 304, 431, 416),
    (4096, 4096): (2, (10, 10), (192, 448), 304, 335, 320),
    (4096, 384): (1, (64, 64), (64, 64), 192, 194, 192),
    (17, 16384): (1, (1, 1), (17, 17), 128, 255, 128),
    (17, 4096): (1, (1, 1), (17, 17), 32, 63, 32),
    (17, 384): (1, (1, 1), (17, 17), 3, 5, 3),
    (21, 16384): (1, (1, 1), (21, 21), 128, 255, 128),
    (21, 4096): (1, (1, 1), (21, 21), 32, 63, 32),
    (21, 384): (1, (1, 1), (21, 21), 3, 5, 3),
    (112, 16384): (1, (2, 2), (48, 64), 256, 383, 256),
    (112, 4096): (1, (2, 2), (48, 64), 64, 95, 64),
    (112, 384): (1, (2, 2), (48, 64), 6, 8, 6),
    (4095, 16384): (8, (3, 4), (192, 1728), 304, 431, 416),
    (4095, 4096): (2, (10, 10), (192, 448), 304, 335, 320),
    (4095, 384): (1, (64, 64), (63, 64), 192, 194, 192),
    (16, 16384): (1, (1, 1), (16, 16), 128, None, None),
    (520, 16384): (4, (3, 3), (8, 256), 304, None, None),
    (256, 512): (1, (4, 4), (64, 64), 16, None, None),
    (16, 128): (1, (1, 1), (16, 16), 1, None, None),
    (8, 512): (1, (1, 1), (8, 8), 4, None, None),
    (64, 16384): (1, (1, 1), (64, 64), 128, None, None),
    (64, 384): (1, (1, 1), (64, 64), 3, None, None),
}
# TF32-mode workspaces in fp32 elements and their words no kernel writes (slots S11.per_shape and
# S5.gp4_at): (forward size, forward words left unwritten, GP-4 size, GP-4 words left unwritten).
_WORKSPACES = {
    (4096, 16384): (2473984, 86016, 1590272, 46080),
    (4096, 4096): (1941504, 79872, 1295360, 46080),
    (4096, 384): (847872, 36864, 862208, 6144),
    (17, 16384): (535552, 2752, 787520, 390144),
    (17, 4096): (141568, 2592, 197696, 95232),
    (17, 384): (17152, 1216, 19520, 6144),
    (21, 16384): (537344, 2752, 787520, 390144),
    (21, 4096): (143360, 2720, 197696, 95232),
    (21, 384): (17920, 1216, 19520, 6144),
    (112, 16384): (578048, 2688, 1184896, 390144),
    (112, 4096): (180992, 2496, 300160, 95232),
    (112, 384): (35328, 1152, 32896, 6144),
    (4095, 16384): (2473728, 86208, 1590272, 46080),
    (4095, 4096): (1941248, 80032, 1295360, 46080),
    (4095, 384): (847872, 37056, 862208, 6144),
}
# R4's mutation rows on the shipped plans (slot S8.covered): the tensors each emulator-side row
# is covered on, at production shape, pair_data seed 0, randn, scale 1.
_S8_COVERED = {
    "project_first": ("p",),
    "rz_for_rn": ("p", "g_xn", "g_W"),
    "bf16_p": ("p",),
    "drop_kernel_split": ("p", "g_W"),
    "ignore_kernel_split": ("p", "g_W"),
    "dgrad_unrounded_w": ("g_xn",),
}
# eps_outside_sqrt on s and drop_radial on the fp32 g_x: covered per input (slot S8.covered).
_S8_EPS_INPUTS = (
    ("x1e-3", "randn", 1e-3, True),
    ("x1", "randn", 1.0, False),
    ("x1e3", "randn", 1e3, False),
    ("zero_rows", "zero_rows", 1.0, True),
)
_S8_RADIAL_INPUTS = (("x1e-3", 1e-3, False), ("x1", 1.0, True), ("x1e3", 1e3, True))
# The Sinkhorn t0_like_others row: covered at the largest of the emulator test's logit scales.
_SINKHORN_SCALES = ((1e-3, False), (1.0, False), (1e3, True))
_MUTATION_ROWS = tuple(_S8_COVERED) + (
    "eps_outside_sqrt",
    "drop_radial",
    "sinkhorn_t0_like_others",
    "kernel_rz_build",
    "kernel_gp3_short_split",
)
# P1's crafted operands (3 values x 2 signs x 3 scales, key outer) and their TF32 readings.
_PROBE_VALUES = {
    "a1_tie_even": 1.0 + 2.0**-11,
    "a2_tie_odd": 1.0 + 2.0**-10 + 2.0**-11,
    "a3_above_half": 1.0 + 2.0**-11 + 2.0**-14,
}
_PROBE_EXPECTED = {
    "rz": {"a1_tie_even": 1.0, "a2_tie_odd": 1.0 + 2.0**-10, "a3_above_half": 1.0},
    "rne": {"a1_tie_even": 1.0, "a2_tie_odd": 1.0 + 2.0**-9, "a3_above_half": 1.0 + 2.0**-10},
    "rna": {
        "a1_tie_even": 1.0 + 2.0**-10,
        "a2_tie_odd": 1.0 + 2.0**-9,
        "a3_above_half": 1.0 + 2.0**-10,
    },
    "none": dict(_PROBE_VALUES),
}
_PROBE_SIGNS, _PROBE_SCALES = (1.0, -1.0), (1.0, 2.0**-20, 2.0**20)
_PROBE_COMBOS = tuple(
    (key, sign, scale) for key in _PROBE_VALUES for sign in _PROBE_SIGNS for scale in _PROBE_SCALES
)
_CRAFTED_SLOTS = ("xn", "W_fwd", "G_p_dgrad", "W_dgrad", "G_p_wgrad", "xn_wgrad")
# The cancelling column of the dgrad slots: g_xn = T(a) * 1 + (-sign * scale) * 1.
_CANCEL_COLUMN = 6
_INT_VIEW = {torch.float32: torch.int32, torch.bfloat16: torch.int16, torch.float64: torch.int64}


def _shape_id(shape):
    return f"{shape[0]}x{shape[1]}"


if glm._GLM_KERNELS_DEFINED:
    import triton
    import triton.language as tl

    from megatron.core.fusions.fused_mhc_glm_projection import (
        _glm_gp4_piece_blocks,
        _glm_gp4_program_plan,
        _glm_gp4_slot,
        _glm_gp4_tile_programs,
        _tf32_round,
    )

    @triton.jit
    def _tf32_round_probe_kernel(x_ptr, y_ptr, n, MODE: tl.constexpr, BLOCK: tl.constexpr):
        """Apply GLM's _tf32_round at MODE to n fp32 values: a test probe, not a GLM kernel."""
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n
        values = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        tl.store(y_ptr + offsets, _tf32_round(values, MODE), mask=mask)

    @triton.jit
    def _gp4_plan_probe_kernel(out_ptr, NB, NU, NPROG, P: tl.constexpr, C: tl.constexpr):
        """GLM's GP-4 / GP-4r plan helpers at every (program q < P, column tile c < C), as nine
        int32 planes [P, C]: u_lo, u_hi, c_lo, c_hi, b_lo, b_hi, slot, p_first, p_last. A test
        probe, not a GLM kernel."""
        zero = tl.zeros([P, C], dtype=tl.int32)
        q = tl.arange(0, P)[:, None] + zero
        c = tl.arange(0, C)[None, :] + zero
        u_lo, u_hi, c_lo, c_hi = _glm_gp4_program_plan(q, NB, NU, NPROG)
        b_lo, b_hi = _glm_gp4_piece_blocks(u_lo, u_hi, c, NB)
        p_first, p_last = _glm_gp4_tile_programs(c, NB, NU, NPROG)
        slot = _glm_gp4_slot(q, c)
        offs = q * C + c
        tl.store(out_ptr + offs, u_lo)
        tl.store(out_ptr + P * C + offs, u_hi)
        tl.store(out_ptr + 2 * P * C + offs, c_lo)
        tl.store(out_ptr + 3 * P * C + offs, c_hi)
        tl.store(out_ptr + 4 * P * C + offs, b_lo)
        tl.store(out_ptr + 5 * P * C + offs, b_hi)
        tl.store(out_ptr + 6 * P * C + offs, slot)
        tl.store(out_ptr + 7 * P * C + offs, p_first)
        tl.store(out_ptr + 8 * P * C + offs, p_last)


@pytest.fixture
def glm_pair_gpu(_bind_local_cuda_device):
    """Skip unless the bound device is a GB300 (compute capability (10, 3)).

    The capability is read after the rank's own device is bound, never at collection.
    """
    if not (torch.cuda.is_available() and fmk.is_triton_available()):
        pytest.skip("needs CUDA and Triton")
    capability = torch.cuda.get_device_capability()
    if capability != (10, 3):
        pytest.skip(f"the GLM pair is validated on compute capability (10, 3), not {capability}")


@pytest.fixture
def tf32_matmuls():
    """Skip unless torch's fp32 matmuls run TF32 (the session pins it by its environment)."""
    precision = torch.backends.cuda.matmul.fp32_precision
    if precision != "tf32":
        pytest.skip(f"a TF32-mode test; cuda.matmul.fp32_precision is {precision!r}")


# --- GLM pair helpers ------------------------------------------------------------------------


def _bits(tensor):
    tensor = tensor.detach().contiguous()
    return tensor.view(_INT_VIEW[tensor.dtype])


def _same_bits(a, b):
    """Bitwise equality (signs of zero and NaN payloads included); None equals only None."""
    if a is None or b is None:
        return a is None and b is None
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(_bits(a), _bits(b))


def _record(tag, **fields):
    """One record line in the log: the values a session reads beside the outcomes."""
    text = " ".join(
        f"{key}={value:.6e}" if isinstance(value, float) else f"{key}={value}"
        for key, value in fields.items()
    )
    print(f"GLM_PAIR_{tag} {text}")


def _pair_inputs(T, K, rows="randn", scale=1.0, seed=0):
    """pair_data's x (bf16 [T, K]), W (fp32 [24, K]) and G_p (fp32 [T, 24]) on the device."""
    x, W, G_p = emu.pair_data(T, K, scale, seed, rows)
    return x.to(_DEVICE), W.to(_DEVICE), G_p.to(_DEVICE)


def _grad_s(T):
    """R12's G_s: randn [T] from seed 7 (the emulator's layout; the launchers take [T, 1])."""
    return torch.randn(T, generator=torch.Generator().manual_seed(7)).to(_DEVICE)


def _prc_draws(T, K):
    """R12's contract inputs: torch.manual_seed(123) and the projection-recompute test's draws."""
    torch.manual_seed(123)
    x = torch.randn(T, K).bfloat16()
    W = torch.randn(glm.PROJ_N, K) * 0.02
    G_p = torch.randn(T, glm.PROJ_N)
    G_s = torch.randn(T, 1)
    return x.to(_DEVICE), W.to(_DEVICE), G_p.to(_DEVICE), G_s.view(-1).to(_DEVICE)


def _fwd(x, W, rounding="rn", mode="tf32", **kwargs):
    return glm._glm_proj_fwd(x, W, _PAIR_EPS, rounding=rounding, mode=mode, **kwargs)


def _bwd(x, W, s, p, G_p, G_s, rounding="rn", mode="tf32", debug=False, **kwargs):
    grad_s = None if G_s is None else G_s.view(-1, 1)
    return glm._glm_proj_bwd(
        x, W, s, p, G_p, grad_s, rounding=rounding, mode=mode, debug=debug, **kwargs
    )


def _used(p, s, G_p, G_s):
    """The outputs a case uses and their gradients: p with G_p, s with G_s as [T, 1]."""
    outputs, grads = [], []
    if G_p is not None:
        outputs.append(p)
        grads.append(G_p)
    if G_s is not None:
        outputs.append(s)
        grads.append(G_s.view(-1, 1))
    return outputs, grads


def _through_function(x, W, G_p, G_s, rounding="rn"):
    """The production path, fused_glm_proj_rms with autograd: p, s, bf16 g_x and g_W (or None)."""
    x_leaf = x.detach().clone().requires_grad_()
    w_leaf = W.detach().clone().requires_grad_()
    p, s = glm.fused_glm_proj_rms(x_leaf, w_leaf, _PAIR_EPS, rounding=rounding)
    torch.autograd.backward(*_used(p, s, G_p, G_s))
    return p.detach(), s.detach(), x_leaf.grad, w_leaf.grad


def _unfused_pair(x, W, G_p, G_s):
    """Today's _proj_rms_impl (eps inside the sqrt, normalize first) on a leaf fp32 copy of x,
    under the same precision state; also on the bf16 x itself, whose bf16 gradient R2 compares."""
    from megatron.core.transformer.hyper_connection import _proj_rms_impl

    x_leaf = x.detach().float().requires_grad_()
    w_leaf = W.detach().clone().requires_grad_()
    p, s = _proj_rms_impl(
        x_leaf, w_leaf, eps=_PAIR_EPS, eps_inside_sqrt=True, normalize_before_projection=True
    )
    torch.autograd.backward(*_used(p, s, G_p, G_s))
    x_bf16 = x.detach().clone().requires_grad_()
    w_bf16 = W.detach().clone().requires_grad_()
    p_bf16, s_bf16 = _proj_rms_impl(
        x_bf16, w_bf16, eps=_PAIR_EPS, eps_inside_sqrt=True, normalize_before_projection=True
    )
    torch.autograd.backward(*_used(p_bf16, s_bf16, G_p, G_s))
    return {
        "p": p.detach(),
        "s": s.detach(),
        "g_x": x_leaf.grad,
        "g_W": w_leaf.grad,
        "g_x_bf16": x_bf16.grad,
    }


def _emulate(x, W, G_p, s, *, G_s=None, p=None, operand_mode="rne", mutation=None):
    """R3's emulator call: ALIGN_K8_X2 on the shipped plans, with the kernel's own s (and p)."""
    T, K = x.shape
    return emu.glm_pair_kernel_model(
        x,
        W,
        G_p,
        G_s,
        eps=emu.EPS_FP32,
        operand_mode=operand_mode,
        model=emu.ALIGN_K8_X2,
        plans=emu.glm_pair_fused_plans(T, K, glm.PROJ_N),
        s=s.detach().reshape(-1),
        p=None if p is None else p.detach(),
        mutation=mutation,
    )


def _tf32_outputs(x, W, G_p, rounding):
    """p and s of the forward, g_xn of the debug build and g_W of the production build."""
    p, s = _fwd(x, W, rounding=rounding)
    _, g_W, _ = _bwd(x, W, s, p, G_p, None, rounding=rounding)
    _, _, extras = _bwd(x, W, s, p, G_p, None, rounding=rounding, debug=True)
    return {"p": p, "s": s, "g_xn": extras["g_xn"], "g_W": g_W}


def _low_bits_set(tensor):
    """R4's RN-test operands: the fp32 bit patterns with the low 13 bits set."""
    return (tensor.contiguous().view(torch.int32) | 0x1FFF).view(torch.float32)


def _assert_tf32_state():
    precision = torch.backends.cuda.matmul.fp32_precision
    legacy = torch.get_float32_matmul_precision()
    assert precision == "tf32" and legacy == "high", (
        f"R2 precondition: TF32 mode needs cuda.matmul.fp32_precision 'tf32' and "
        f"get_float32_matmul_precision() 'high', got {precision!r} and {legacy!r}"
    )
    assert glm._matmul_mode() == ("tf32", None), f"R2 precondition: {glm._matmul_mode()}"


def _epilogue_failures(label, x, s, p, G_p, G_s, extras, g_x):
    """R3's epilogue rule: the emulator's O6-O9 on the debug build's own g_xn, with the kernel's s
    and p, reproduce the debug g_s and g_x32 and the production bf16 g_x bitwise."""
    T, K = x.shape
    s_cpu = s.detach().reshape(-1).cpu()
    if G_p is not None:
        # O6: a sequential sum in index order from +0.0 with raw G_p and the kernel's p, then one
        # correctly rounded division (both on CPU: no contraction, IEEE division), then + G_s.
        G_p_cpu, p_cpu = G_p.cpu(), p.detach().cpu()
        acc = torch.zeros(T)
        for j in range(glm.PROJ_N):
            acc = acc + G_p_cpu[:, j] * p_cpu[:, j]
        g_s = acc / s_cpu
        if G_s is not None:
            g_s = g_s + G_s.cpu()
    else:
        g_s = torch.zeros(T) if G_s is None else G_s.cpu().clone()
    # O7: coef = fl(fl(fl(fl(s * s) * s) * inv_k) * g_s), inv_k = fl32(1 / K).
    coef = s_cpu * s_cpu * s_cpu * torch.tensor(1.0 / K, dtype=torch.float32) * g_s
    xf = x.float()
    g_xn = extras["g_xn"] if G_p is not None else torch.zeros_like(xf)
    # O8: two products and one subtraction, each its own torch kernel (no contraction).
    product = s.detach().reshape(-1, 1) * g_xn
    g_x32 = product - coef.to(x.device).view(-1, 1) * xf
    failures = []
    if not _same_bits(extras["g_s"].reshape(-1).cpu(), g_s):
        failures.append(f"{label}: R3 epilogue: debug g_s differs from O6 bitwise")
    if not _same_bits(extras["g_x32"], g_x32):
        failures.append(f"{label}: R3 epilogue: debug g_x32 differs from O7-O8 bitwise")
    if not _same_bits(g_x, extras["g_x32"].to(torch.bfloat16)):
        failures.append(f"{label}: R3 epilogue: production bf16 g_x differs from RN(g_x32)")
    return failures


def _operator_bar(quantity, unfused_error, strict, norm_only):
    """R2's bar on one quantity, with its rule."""
    if strict or (quantity == "g_x" and norm_only):
        return max(_RATIO * unfused_error, _STRICT_FLOOR), "max(1.1 x unfused, 1e-5)"
    if unfused_error < _IEEE_CLASS_LEVEL:
        return _TF32_BAR, "1e-3 (unfused GEMM IEEE class)"
    return min(_RATIO * unfused_error, _TF32_BAR), "1.1 x unfused and 1e-3"


def _r2_failures(label, x, W, G_p, G_s, mode):
    """R2 at one case: the float64 bars, the two bitwise checks, the reference check, and R3's
    epilogue rule. mode is the mode the precision state gives the Function."""
    strict, norm_only = mode == "ieee", G_p is None
    p, s, g_x, g_W = _through_function(x, W, G_p, G_s)
    g_x_dbg, g_W_dbg, extras = _bwd(x, W, s, p, G_p, G_s, mode=mode, debug=True)
    p_dbg, s_dbg = _fwd(x, W, mode=mode, debug=True)
    unfused = _unfused_pair(x, W, G_p, G_s)
    reference = emu.glm_pair_reference_f64(x, W, G_p, G_s, eps=emu.EPS_FP32)
    failures = []
    outputs = {"p": p, "s": s, "g_x": g_x, "g_W": g_W, **extras}
    for name, tensor in outputs.items():
        if tensor is not None and not torch.isfinite(tensor).all():
            failures.append(f"{label}: {name} is not finite")
    if (g_W is None) != norm_only or (unfused["g_W"] is None) != norm_only:
        failures.append(f"{label}: weight gradient presence differs from the used outputs")
    if not _same_bits(g_x, extras["g_x32"].to(torch.bfloat16)):
        failures.append(f"{label}: R2 bitwise: production bf16 g_x != bf16_RN(debug g_x32)")
    for name, production, debug in (
        ("g_x", g_x, g_x_dbg),
        ("g_W", g_W, g_W_dbg),
        ("p", p, p_dbg),
        ("s", s, s_dbg),
    ):
        if not _same_bits(production, debug):
            failures.append(f"{label}: R2 bitwise: debug {name} != production {name}")
    if not _same_bits(unfused["g_x"].to(torch.bfloat16), unfused["g_x_bf16"]):
        failures.append(f"{label}: R2: the reference is not today's arithmetic (bf16 g_x)")
    failures += _epilogue_failures(label, x, s, p, G_p, G_s, extras, g_x)
    s_error = rel_rms(s.reshape(-1), reference.s)
    _record("R2", case=label, mode=mode, quantity="s", fused=s_error, bar=_S_BAR)
    if not s_error <= _S_BAR:
        failures.append(f"{label}: s rel-RMS {s_error:.3e} > {_S_BAR:g}")
    for quantity, fused, ours, theirs in (
        ("p", p, unfused["p"], reference.p),
        ("g_x", extras["g_x32"], unfused["g_x"], reference.g_x32),
        ("g_W", g_W, unfused["g_W"], reference.g_W),
    ):
        if fused is None:
            continue
        error, unfused_error = rel_rms(fused, theirs), rel_rms(ours, theirs)
        bar, rule = _operator_bar(quantity, unfused_error, strict, norm_only)
        _record(
            "R2",
            case=label,
            mode=mode,
            quantity=quantity,
            fused=error,
            unfused=unfused_error,
            ratio=error / unfused_error if unfused_error else float("inf"),
            bar=bar,
            rule=rule.replace(" ", "_"),
        )
        if not error <= bar:
            failures.append(f"{label}: {quantity} rel-RMS {error:.3e} > {bar:.3e} ({rule})")
    return failures


def _forward_launch_edit(x, W, edits, rounding="rn", mode="tf32", zero_workspace=False):
    """The forward's own launch list (glm._fwd_launches), with launches edited by name, run on a
    fresh workspace. Returns p, s, the workspace, its layout and the launches as run."""
    T, K = x.shape
    layout = glm._fwd_layout(T, K, mode)
    allocate = torch.zeros if zero_workspace else torch.empty
    workspace = allocate(layout.total, dtype=torch.float32, device=x.device)
    p = torch.empty(T, glm.PROJ_N, dtype=torch.float32, device=x.device)
    s = torch.empty(T, 1, dtype=torch.float32, device=x.device)
    launches = glm._fwd_launches(
        glm._glm_kernels(), x, W, _PAIR_EPS, p, s, workspace, rounding=rounding, mode=mode
    )
    launches = [edits.get(launch.name, lambda launch: launch)(launch) for launch in launches]
    scratch = workspace.narrow(0, layout.tma[0], layout.tma[1]) if layout.tma[1] else None
    glm._run(launches, scratch, None)
    return p, s, workspace, layout, launches


def _one_program_short(launch):
    return launch._replace(grid=(launch.grid[0] - 1,) + tuple(launch.grid[1:]))


@contextlib.contextmanager
def _recorded_empty():
    """Record every torch.empty result (the launchers allocate their workspaces with it)."""
    calls, original = [], torch.empty

    def empty(*args, **kwargs):
        tensor = original(*args, **kwargs)
        calls.append(tensor)
        return tensor

    torch.empty = empty
    try:
        yield calls
    finally:
        torch.empty = original


def _all_outputs(x, W, G_p, G_s, mode, calls=None):
    """Forward, production and debug backward with G_p and G_s, and the norm-only backward.

    With calls (from _recorded_empty), also the largest flat fp32 allocation of each call: its
    workspace."""
    outputs, workspaces = {}, {}

    def take(phase):
        if calls is not None:
            flat = [t for t in calls if t.dim() == 1 and t.dtype == torch.float32]
            workspaces[phase] = max(flat, key=lambda t: t.numel()) if flat else None
            calls.clear()

    if calls is not None:
        calls.clear()
    p, s = _fwd(x, W, mode=mode)
    take("forward")
    g_x, g_W, _ = _bwd(x, W, s, p, G_p, G_s, mode=mode)
    take("backward")
    g_x_dbg, g_W_dbg, extras = _bwd(x, W, s, p, G_p, G_s, mode=mode, debug=True)
    take("backward_debug")
    g_x_norm, g_W_norm, _ = _bwd(x, W, s, p, None, G_s, mode=mode)
    take("backward_norm")
    outputs.update(p=p, s=s, g_x=g_x, g_W=g_W, debug_g_x=g_x_dbg, debug_g_W=g_W_dbg)
    outputs.update({f"debug_{name}": tensor for name, tensor in extras.items()})
    outputs["norm_g_x"] = g_x_norm
    if g_W_norm is not None:
        outputs["norm_g_W"] = g_W_norm
    return outputs, workspaces


def _det_fill_failures(T, K, mode):
    """R8 (d) at one shape and mode: under det-fill every output is finite and bitwise equal to a
    run without it, and each workspace is at least its own layout's size."""
    label = f"R8(d) {mode} {T}x{K}"
    x, W, G_p = _pair_inputs(T, K)
    G_s = _grad_s(T)
    plain, _ = _all_outputs(x, W, G_p, G_s, mode)
    failures = []
    with _det_fill():
        if not torch.isnan(torch.empty(64, device=_DEVICE)).all():
            return [f"{label}: R8 precondition: a fresh torch.empty is not all NaN"]
        # Positive control: GP-3 one program short leaves exactly its last rows of p NaN.
        gp3 = {}

        def short(launch):
            gp3["launch"] = launch
            return _one_program_short(launch)

        p_short, *_ = _forward_launch_edit(x, W, {_SHIPPED["gp3"]: short}, mode=mode)
        written = (gp3["launch"].grid[0] - 1) * gp3["launch"].constexprs["BLOCK_M"]
        if not (torch.isnan(p_short[written:]).all() and torch.isfinite(p_short[:written]).all()):
            failures.append(
                f"{label}: R8 positive control: p is not NaN in exactly rows >= {written}"
            )
        with _recorded_empty() as calls:
            filled, workspaces = _all_outputs(x, W, G_p, G_s, mode, calls)
    for name, tensor in filled.items():
        if not torch.isfinite(tensor).all():
            failures.append(f"{label}: {name} has non-finite elements under det-fill")
        if not _same_bits(tensor, plain[name]):
            failures.append(f"{label}: {name} differs from the run without det-fill")
    if mode == "tf32":
        minimum = {"forward": _WORKSPACES[(T, K)][0], "backward": _WORKSPACES[(T, K)][2]}
        unwritten = {"forward": _WORKSPACES[(T, K)][1], "backward": _WORKSPACES[(T, K)][3]}
    else:
        # The strict forward's 16 chain partials. The backward: GP-4p's coef, T(G_p) and T(G_p)^T
        # over all T_pad = 64 * cdiv(T, 64) rows (plan 5.3.4; 64-row multiples keep every section
        # on a 128 B boundary), then the strict GP-4's split-M 2 partials.
        t_pad = 64 * -(-T // 64)
        minimum = {
            "forward": 16 * T * glm.BLOCK_N,
            "backward": t_pad * (1 + 2 * glm.BLOCK_N) + 2 * glm.PROJ_N * K,
        }
        unwritten = {}
    for phase, workspace in workspaces.items():
        nan_count = -1 if workspace is None else int(torch.isnan(workspace).sum().item())
        size = 0 if workspace is None else workspace.numel()
        _record(
            "R8D",
            case=label,
            phase=phase,
            workspace_fp32=size,
            nan_left=nan_count,
            nan_left_registered=unwritten.get(phase, "-"),
        )
        if phase in minimum and size < minimum[phase]:
            failures.append(
                f"{label}: {phase} workspace {size} fp32 < its layout's {minimum[phase]}"
            )
    return failures


# --- GLM pair: lowering parsers (ported from P2's and round 2's) ------------------------------

_PTX_TOKEN = re.compile(r"^\s*(?:@!?%[\w.]+\s+)?([a-z][\w.:]*)")
# Raw cuobjdump ("/*0450*/  OP ...") and Triton's reformatted asm["sass"] ("ctrl<TAB>OP ...").
_SASS_LINE = re.compile(
    r"^(?:\s*/\*[0-9a-fA-F]+\*/\s+|[^\t\n]*\t)(?:@!?U?P[T0-9]+\s+)?([A-Z][A-Z0-9_.]*)([^;]*)"
)
_TTGIR_SHAPE = re.compile(r"(?:memdesc|tensor)<((?:\d+x)+)[a-z]")
_TTGIR_ALIAS = re.compile(r"^(#[\w.]+)\s*=\s*#(?:ttg|triton_gpu)\.(\w+)<(.*)$")
_TTGIR_RESULT = re.compile(r"->\s*tensor<((?:\d+x)+)[a-z]\w*\s*,\s*(#[\w.]+)")


def _ptx_tokens(ptx):
    tokens = collections.Counter()
    for line in ptx.splitlines():
        match = _PTX_TOKEN.match(line)
        if match:
            tokens[match.group(1)] += 1
    return tokens


def _sass_ops(sass):
    ops = []
    for line in sass.splitlines():
        match = _SASS_LINE.match(line)
        if match:
            ops.append((match.group(1), match.group(2)))
    return ops


def _ttgir_shape(dims):
    return tuple(int(d) for d in dims.strip("x").split("x") if d)


def _ttgir_dots(ttgir):
    """Each dot of a TTGIR module as (family, accumulator shape, input precision).

    tcgen05: ttng.tc_gen5_mma, whose third memdesc is the accumulator. A tt.dot left after
    lowering: its result encoding names the family (blocked = FMA, no tensor cores).
    """
    aliases = {}
    for line in ttgir.splitlines():
        match = _TTGIR_ALIAS.match(line.strip())
        if match and not match.group(1).startswith("#loc"):
            aliases[match.group(1)] = match.group(2)
    dots = []
    for line in ttgir.splitlines():
        text = line.strip()
        if "ttng.tc_gen5_mma" in text and "tc_gen5_mma_scaled" not in text:
            shapes = [_ttgir_shape(dims) for dims in _TTGIR_SHAPE.findall(text)]
            dots.append(("tcgen05", shapes[2] if len(shapes) >= 3 else None, None))
        elif "warp_group_dot" in text and "warp_group_dot_wait" not in text:
            dots.append(("wgmma", None, None))
        elif re.search(r"\btt\.dot\b", text) and "tt.dot_scaled" not in text:
            result = _TTGIR_RESULT.search(text)
            precision = re.search(r"inputPrecision\s*=\s*(\w+)", text)
            kind = aliases.get(result.group(2), "?") if result else "?"
            family = {"blocked": "fma", "nvidia_mma": "mma", "mma": "mma"}.get(kind, kind)
            shape = _ttgir_shape(result.group(1)) if result else None
            dots.append((family, shape, precision.group(1) if precision else None))
    return dots


def _packed_contractions(sass_ops, ptx_tokens):
    """Round 2's census: SASS FFMA2 with a non-RZ addend minus the PTX's own fma.*.f32x2."""
    contracted = 0
    for op, operands in sass_ops:
        if op.split(".")[0] != "FFMA2":
            continue
        parts = [part.strip() for part in operands.split(",")]
        # The zero register prints with operand modifiers (RZ.F32), so they are stripped.
        if len(parts) >= 4 and parts[3].lstrip("-|").split(".")[0].rstrip("|") != "RZ":
            contracted += 1
    ptx_fma_x2 = sum(
        n for token, n in ptx_tokens.items() if token.startswith("fma") and "f32x2" in token
    )
    return contracted - ptx_fma_x2


def _plus_rz_contractions(sass_ops):
    """FFMA2 whose addend is +RZ: d = RN(a * b + 0), which round 2's census rule does not count.

    Without fma in the PTX it can only be a packed multiply and an add or subtract from zero,
    contracted. It differs from the two rounded operations only in the sign of an underflowed
    product (0 - RN(c * x) is +0 where RN(-c * x + 0) is -0), which R3's epilogue and the
    norm-only byte test read; recorded here, not gated (a -RZ addend is an exact multiply)."""
    count = 0
    for op, operands in sass_ops:
        if op.split(".")[0] != "FFMA2":
            continue
        parts = [part.strip() for part in operands.split(",")]
        if len(parts) >= 4 and parts[3].split(".")[0].rstrip("|") == "RZ":
            count += 1
    return count


_TTGIR_BOOL = re.compile(r"^(%[\w.#]+) = arith\.constant (true|false)\b")
_TTGIR_ZERO = re.compile(r"^(%[\w.#]+) = arith\.constant dense<0\.000000e\+00>")
_TTGIR_STORE = re.compile(r"\bttng\.tmem_store (%[\w.#]+), (%[\w.#]+)")
_TTGIR_ALLOC_FROM = re.compile(r"^(%[\w.#]+) = ttng\.tmem_alloc (%[\w.#]+)")
_TTGIR_MMA = re.compile(r"\bttng\.tc_gen5_mma (%[\w.#]+), (%[\w.#]+), (%[\w.#]+), (%[\w.#]+),")
_TTGIR_ITER_ARGS = re.compile(r"\biter_args\((.*?)\)\s*->")
_PTX_PRED_MOV = re.compile(r"^\s*mov\.pred\s+(%p\d+),\s*(-?\d+)\s*;")
_PTX_PRED_DEF = re.compile(r"^\s*(?:@!?%[\w.]+\s+)?[a-z][\w.]*\s+(%p\d+)\s*[,|]")
_PTX_TCGEN05 = re.compile(r"^\s*(?:@!?%[\w.]+\s+)?tcgen05\.mma\S*\s+(.*?);")


def _ttgir_mma_chains(ttgir):
    """Each tc_gen5_mma of a TTGIR module as a dict: its accumulator, the shapes of A and D, and
    how it treats the accumulator (useD): "false" (starts from zero), "true" (accumulates),
    "iter" (a loop-carried flag that is false on the loop's first iteration and true after it)
    or "other"; with the enclosing scf.for loops. Also every value stored into an accumulator
    and whether it is a zero constant."""
    bools, zeros, stack, loops = {}, set(), [], {}
    mmas, stores = [], []
    for number, line in enumerate(ttgir.splitlines()):
        text = line.strip()
        match = _TTGIR_BOOL.match(text)
        if match:
            bools[match.group(1)] = match.group(2)
        match = _TTGIR_ZERO.match(text)
        if match:
            zeros.add(match.group(1))
        if text.startswith("}"):
            stack.pop()
        if text.startswith("scf.yield") and stack and stack[-1][0] == "for":
            values = text[len("scf.yield") :].split(":")[0]
            loops[stack[-1][1]]["yield"] = [v.strip() for v in values.split(",") if v.strip()]
        match = _TTGIR_STORE.search(text)
        if match:
            stores.append((match.group(2), match.group(1), tuple(stack)))
        match = _TTGIR_ALLOC_FROM.match(text)
        if match:
            stores.append((match.group(1), match.group(2), tuple(stack)))
        match = _TTGIR_MMA.search(text)
        if match and "tc_gen5_mma_scaled" not in text:
            shapes = [_ttgir_shape(dims) for dims in _TTGIR_SHAPE.findall(text)]
            mmas.append(
                dict(
                    line=number,
                    acc=match.group(3),
                    use_d=match.group(4),
                    a_shape=shapes[0] if shapes else None,
                    d_shape=shapes[2] if len(shapes) >= 3 else None,
                    loops=tuple(entry[1] for entry in stack if entry[0] == "for"),
                )
            )
        if text.endswith("{"):
            kind = "for" if " scf.for " in f" {text} " or text.startswith("scf.for") else "block"
            if kind == "for":
                args = _TTGIR_ITER_ARGS.search(text)
                pairs = [a.split("=") for a in args.group(1).split(",")] if args else []
                loops[number] = {
                    "inits": [(k.strip(), v.strip()) for k, v in pairs if v],
                    "yield": [],
                }
            stack.append((kind, number))
    for mma in mmas:
        use_d = mma["use_d"]
        mma["kind"] = bools.get(use_d, "other")
        for loop in reversed(mma["loops"]):
            names = [name for name, _ in loops[loop]["inits"]]
            if use_d in names:
                index = names.index(use_d)
                init = bools.get(loops[loop]["inits"][index][1])
                yielded = loops[loop]["yield"]
                last = bools.get(yielded[index]) if index < len(yielded) else None
                mma["kind"] = "iter" if (init, last) == ("false", "true") else "other"
                mma["owner"] = loop
                break
    store_list = [
        dict(acc=acc, zero=value in zeros, loops=tuple(e[1] for e in where if e[0] == "for"))
        for acc, value, where in stores
    ]
    return mmas, store_list


def _ptx_mma_groups(ptx):
    """The tcgen05.mma instructions of a PTX text in groups (each group ends at a
    tcgen05.commit), each as (D operand, enable-input-d predicate, that predicate's
    definitions: "0", "-1" or "other")."""
    definitions = collections.defaultdict(set)
    for line in ptx.splitlines():
        match = _PTX_PRED_MOV.match(line)
        if match:
            definitions[match.group(1)].add(match.group(2))
            continue
        match = _PTX_PRED_DEF.match(line)
        if match and "tcgen05" not in line:
            definitions[match.group(1)].add("other")
    groups, current = [], []
    for line in ptx.splitlines():
        if "tcgen05.commit" in line:
            if current:
                groups.append(current)
            current = []
            continue
        match = _PTX_TCGEN05.match(line)
        if match:
            operands = [part.strip() for part in re.split(r",(?![^\[]*\])", match.group(1))]
            predicate = operands[-1]
            current.append((operands[0], predicate, frozenset(definitions.get(predicate, set()))))
    if current:
        groups.append(current)
    return groups


def _mma_chain_failures(ttgir, ptx, chains):
    """R5's K chain: each TF32 dot's accumulator starts from zero at its chain start, read from
    the TTGIR (useD, or a zero store into the accumulator) and cross-read from the PTX (each
    group's first tcgen05.mma has an enable-input-d predicate that is false at least once, the
    others always true), and the k-step is 8 (the dot's K over its PTX group length).

    chains maps each accumulator shape to where its chain starts: "dot" (every MMA starts from
    zero: a dot without an accumulator argument) or the number of enclosing loops a chain lives
    inside (0: one chain per program; 1: one per iteration of the outermost loop)."""
    failures = []
    mmas, stores = _ttgir_mma_chains(ttgir)
    by_acc = collections.defaultdict(list)
    for mma in mmas:
        by_acc[mma["acc"]].append(mma)
    shapes = {}
    for acc, chain in by_acc.items():
        shape = chain[0]["d_shape"]
        shapes[shape] = shapes.get(shape, 0) + 1
        scope = chains.get(shape)
        label = f"accumulator {acc} {shape}"
        if scope is None:
            failures.append(f"{label}: no chain rule for this accumulator shape")
            continue
        acc_stores = [store for store in stores if store["acc"] == acc]
        if any(not store["zero"] for store in acc_stores):
            failures.append(f"{label}: a value other than zero is stored into the accumulator")
        kinds = [mma["kind"] for mma in chain]
        if "other" in kinds:
            failures.append(
                f"{label}: useD {[m['use_d'] for m in chain]} is not readable as a chain"
            )
            continue
        if scope == "dot":
            if set(kinds) != {"false"}:
                failures.append(f"{label}: useD {kinds}, expected false on every MMA (fresh dot)")
            continue
        for mma in chain:
            if mma["kind"] == "iter" and mma["loops"].index(mma["owner"]) < scope:
                failures.append(f"{label}: the chain restarts with a loop outside its scope")
            if mma["kind"] != "true":
                continue
            starts = [
                start["loops"]
                for start in [m for m in chain if m["line"] < mma["line"] and m["kind"] == "false"]
                + [s for s in acc_stores if s["zero"]]
            ]
            if not any(len(s) >= scope and mma["loops"][: len(s)] == s for s in starts):
                failures.append(f"{label}: an accumulating MMA without a zero start in its chain")
        if not any(kind in ("false", "iter") for kind in kinds) and not any(
            store["zero"] for store in acc_stores
        ):
            failures.append(f"{label}: the chain never starts from zero")
    if set(shapes) != set(chains):
        failures.append(f"TTGIR accumulators {shapes}, expected the chains of {sorted(chains)}")
    groups = _ptx_mma_groups(ptx)
    if not groups:
        failures.append("no tcgen05.mma group in the PTX")
    for group in groups:
        for _, predicate, defined in group[1:]:
            if defined != {"-1"}:
                failures.append(f"PTX: a chain restarts inside a dot ({predicate}: {set(defined)})")
    for target in {d for group in groups for d, _, _ in group}:
        if not any(g[0][0] == target and "0" in g[0][2] for g in groups):
            failures.append(f"PTX: the accumulator {target} never starts from zero")
    lengths = sorted({len(group) for group in groups})
    dot_k = sorted({mma["a_shape"][-1] for mma in mmas if mma["a_shape"]})
    if not all(k // 8 in lengths for k in dot_k) or not all(n * 8 in dot_k for n in lengths):
        failures.append(f"PTX groups of {lengths} MMAs for dots of K {dot_k}: the k-step is not 8")
    return failures


def _launched_constexprs(handle):
    """The constexprs a compiled kernel was specialized on, by argument name, or None."""
    source = getattr(handle, "src", None)
    names = getattr(getattr(source, "fn", None), "arg_names", None)
    constants = getattr(source, "constants", None)
    if names is None or constants is None:
        return None
    return {
        names[key[0]]: value
        for key, value in constants.items()
        if isinstance(key, tuple) and len(key) == 1
    }


def _expected_build(role, *, mode, rounding, K, debug, has_gp):
    """R5's expectations of one compiled kernel, with R0's frozen configs (slots S2, S4, S13, S14).

    None means "recorded, not checked"."""
    tf32 = mode == "tf32"
    operand = (1 if rounding == "rn" else 2) if tf32 else 0  # _tf32_round MODE of W and G_p
    xn_mode = 3 if rounding == "rn" else 2  # cvt.rn.tf32.f32 under rn (S13)
    n_chain = sum(count for count, _ in _GP2_CHAINS[K])
    expected = dict(
        maxnreg=None,  # R0: only the GP-4 main kernel's production build caps registers
        mma=None,
        acc_shapes=None,
        chains=None,
        fma_allowed=False,
        fma_dots=0,
        census_gate=True,
        cvt_rn=None,
        cvt_rna=None,
        div_rn=None,
        wait=None,
        launch_dependents=None,
        launch_pdl=None,
    )
    if role == "gp1":
        expected.update(
            num_warps=8,
            num_stages=1,
            launch_pdl=False,
            launch_dependents=True,
            census_gate=False,  # sums of squares of bf16 values (R5)
            constexprs=dict(
                ROWS=8, BLOCK_K=1024, WR_BLOCK=128, WPRE=int(tf32), WROUND=operand, PDL=1
            ),
        )
    elif role == "gp2":
        expected.update(
            num_warps=8,
            num_stages=3,
            maxnreg=None,
            launch_pdl=True,
            mma="tf32",
            acc_shapes={(128, 32)},
            chains={(128, 32): 0},  # one chain per program
            cvt_rn=rounding == "rn",
            wait=True,
            launch_dependents=True,
            constexprs=dict(
                BLOCK_M=128, BLOCK_N=32, BLOCK_K=64, ORDER=0, PDL=1, ROUND=xn_mode, PRECISION="tf32"
            ),
        )
    elif role == "gp2_strict":
        expected.update(
            num_warps=8,
            num_stages=3,
            mma="tf32x3",
            acc_shapes={(128, 32)},
            cvt_rna=True,
            constexprs=dict(BLOCK_M=128, BLOCK_K=64, PRECISION="tf32x3", ROUND=0),
        )
    elif role == "gp3":
        expected.update(
            num_warps=4,
            num_stages=1,
            launch_pdl=True if tf32 else None,
            wait=True,
            constexprs=dict(BLOCK_M=16, PDL=1, SPLIT=n_chain if tf32 else 16),
        )
    elif role == "gp4p":
        expected.update(
            num_warps=4,
            num_stages=1,
            launch_pdl=False,
            wait=False,
            launch_dependents=False,
            div_rn=True if has_gp else None,
            census_gate=False,  # the div.rn expansion can pair FFMAs (R5)
            constexprs=dict(BLOCK_R=16, ROUND=operand),
        )
    elif role == "gp4":
        expected.update(
            num_warps=8,
            num_stages=1,
            maxnreg=None if debug else 128,
            launch_pdl=False,
            wait=False,
            launch_dependents=False,
            mma="tf32" if has_gp else None,
            acc_shapes={(64, 128), (128, 32)},
            # dgrad: a fresh dot per 64-token tile; wgrad: one chain per piece (column tile)
            chains={(64, 128): "dot", (128, 32): 1} if has_gp else None,
            cvt_rn=(True if has_gp else None) if rounding == "rn" else False,
            constexprs=dict(BLOCK_T=64, BLOCK_C=128, ANCHOR=1, GT=1, ROUND=operand, XROUND=xn_mode),
        )
    elif role == "gp4_strict":
        expected.update(
            num_warps=8,
            num_stages=2,
            launch_pdl=False,
            wait=False,
            launch_dependents=False,
            fma_allowed=True,
            fma_dots=2 if has_gp else 0,
            census_gate=False,  # its FMA dots (R5)
            constexprs=dict(BLOCK_T=64, BLOCK_C=128, PRECISION="ieee", ROUND=0),
        )
    elif role == "gp4r":
        expected.update(
            num_warps=4,
            num_stages=1,
            launch_pdl=False,
            wait=False,
            launch_dependents=False,
            constexprs=dict(RBC=64),
        )
    elif role == "gp4r_strict":
        expected.update(
            num_warps=4,
            num_stages=1,
            launch_pdl=False,
            wait=False,
            launch_dependents=False,
            constexprs=dict(SPLIT=2, BLOCK=512),
        )
    else:
        raise ValueError(f"unknown role {role}")
    return expected


def _kernel_failures(label, name, handle, expected):
    """R5's checks of one compiled kernel against its expectations."""
    failures = []

    def fail(text):
        failures.append(f"{label} {name}: {text}")

    metadata = handle.metadata
    for key in ("num_warps", "num_stages", "maxnreg", "launch_pdl"):
        if key not in expected or (key == "launch_pdl" and expected[key] is None):
            continue
        got = getattr(metadata, key, "<missing>")
        if got != expected[key]:
            fail(f"{key} {got!r}, frozen {expected[key]!r}")
    if getattr(metadata, "enable_fp_fusion", None) is not False:
        fail("enable_fp_fusion is not False")
    constexprs = _launched_constexprs(handle)
    if constexprs is None:
        fail("its constexprs cannot be read from the compiled kernel")
    else:
        for key, value in expected["constexprs"].items():
            if constexprs.get(key, "<missing>") != value:
                fail(f"constexpr {key} {constexprs.get(key, '<missing>')!r}, frozen {value!r}")
    ptx, ttgir, sass = handle.asm["ptx"], handle.asm["ttgir"], handle.asm["sass"]
    if isinstance(sass, bytes):
        sass = sass.decode()
    tokens, ops, dots = _ptx_tokens(ptx), _sass_ops(sass), _ttgir_dots(ttgir)
    target = re.search(r"^\s*\.target\s+(\w+)", ptx, re.MULTILINE)
    if target is None or target.group(1) != "sm_103a":
        fail(f".target {target.group(1) if target else None}, expected sm_103a")
    if re.search(r"\btrap\s*;", ptx):
        fail("trap; in the PTX")
    mma_tokens = {
        t: n for t, n in tokens.items() if t.startswith(("tcgen05.mma", "wgmma.", "mma."))
    }
    sass_mma = collections.Counter(op for op, _ in ops if "MMA" in op)
    tcgen05 = [dot for dot in dots if dot[0] == "tcgen05"]
    left = [dot for dot in dots if dot[0] != "tcgen05"]
    if expected["mma"] is None:
        if mma_tokens or sass_mma or tcgen05 or any(dot[0] != "fma" for dot in left):
            fail(f"an MMA instruction: PTX {mma_tokens}, SASS {dict(sass_mma)}, TTGIR {dots}")
    else:
        if set(mma_tokens) != {"tcgen05.mma.cta_group::1.kind::tf32"}:
            fail(f"PTX MMAs {mma_tokens}, expected tcgen05.mma.cta_group::1.kind::tf32 only")
        if not sass_mma.get("UTCHMMA") or set(sass_mma) != {"UTCHMMA"}:
            fail(f"SASS MMAs {dict(sass_mma)}, expected UTCHMMA only")
        if {dot[1] for dot in tcgen05} != expected["acc_shapes"] or left:
            fail(f"TTGIR dots {dots}, expected tc_gen5_mma at {expected['acc_shapes']} only")
        if expected["mma"] == "tf32x3" and len(tcgen05) % 3:
            fail(f"{len(tcgen05)} tc_gen5_mma, not three per dot (tf32x3)")
        if expected["chains"] is not None:  # the TF32 dots that carry a P9 bar (R5)
            for failure in _mma_chain_failures(ttgir, ptx, expected["chains"]):
                fail(f"K chain: {failure}")
    fma_dots = [dot for dot in left if dot[0] == "fma"]
    if len(fma_dots) != expected["fma_dots"]:
        fail(f"{len(fma_dots)} FMA dots in the TTGIR, expected {expected['fma_dots']}")
    if any(dot[2] == "ieee" and dot[0] != "fma" for dot in dots):
        fail(f"an ieee dot lowered to an MMA: {dots}")
    fma = sum(n for token, n in tokens.items() if token.startswith("fma") and "f32" in token)
    if fma and not expected["fma_allowed"]:
        fail(f"{fma} fma f32 instructions in the PTX")
    checks = (
        ("cvt_rn", "cvt.rn.tf32.f32"),
        ("cvt_rna", "cvt.rna.tf32.f32"),
        ("wait", "griddepcontrol.wait"),
        ("launch_dependents", "griddepcontrol.launch_dependents"),
    )
    for key, token in checks:
        if expected[key] is not None and bool(tokens.get(token)) != expected[key]:
            fail(
                f"{token} x{tokens.get(token, 0)}, expected "
                f"{'present' if expected[key] else 'absent'}"
            )
    if expected["div_rn"] and not any(t.startswith("div.rn") and "f32" in t for t in tokens):
        fail("no div.rn.f32")
    census = _packed_contractions(ops, tokens)
    if expected["census_gate"] and census:
        fail(f"{census} packed contractions (SASS FFMA2 with a non-RZ addend)")
    _record(
        "R5",
        build=label.replace(" ", "_"),
        kernel=name,
        ptx_mma=sum(mma_tokens.values()),
        sass_utchmma=sass_mma.get("UTCHMMA", 0),
        acc_shapes=sorted({dot[1] for dot in tcgen05 if dot[1]}),
        fma_f32=fma,
        ffma2=sum(1 for op, _ in ops if op.split(".")[0] == "FFMA2"),
        census=census,
        ffma2_plus_rz=_plus_rz_contractions(ops),
        cvt_rn=tokens.get("cvt.rn.tf32.f32", 0),
        cvt_rna=tokens.get("cvt.rna.tf32.f32", 0),
        n_regs=getattr(handle, "n_regs", "-"),
        n_spills=getattr(handle, "n_spills", "-"),
    )
    return failures


def _lowering_failures(x, W, G_p, G_s, rounding, mode):
    """R5 at one shape: every kernel the forward and the backward builds launch (production and
    debug; with G_p, with G_p and G_s, and with G_s only) through the launchers' _compiled seam."""
    T, K = x.shape
    builds = []
    compiled = {}
    p, s = _fwd(x, W, rounding=rounding, mode=mode, _compiled=compiled)
    builds.append(("forward", compiled, False, True))
    for debug in (False, True):
        for grads, (G_p_case, G_s_case) in (
            ("G_p", (G_p, None)),
            ("G_p+G_s", (G_p, G_s)),
            ("G_s", (None, G_s)),
        ):
            compiled = {}
            _bwd(x, W, s, p, G_p_case, G_s_case, rounding, mode, debug=debug, _compiled=compiled)
            builds.append(
                (f"backward[{grads},debug={int(debug)}]", compiled, debug, G_p_case is not None)
            )
    strict = "_strict" if mode == "ieee" else ""
    failures = []
    for build, compiled, debug, has_gp in builds:
        if build == "forward":
            roles = {"gp1", "gp2" + strict, "gp3"}
        else:
            roles = {"gp4p", "gp4" + strict} | ({"gp4r" + strict} if has_gp else set())
        label = f"{mode}/{rounding} {T}x{K} {build}"
        if set(compiled) != {_SHIPPED[role] for role in roles}:
            failures.append(f"{label}: launched {sorted(compiled)}, expected {sorted(roles)}")
        for name, handle in compiled.items():
            if name not in _ROLE:
                continue
            expected = _expected_build(
                _ROLE[name], mode=mode, rounding=rounding, K=K, debug=debug, has_gp=has_gp
            )
            failures += _kernel_failures(label, name, handle, expected)
    return failures


# --- GLM pair: CPU ---------------------------------------------------------------------------


def test_glm_pair_constants_and_signatures():
    """The frozen configs (R0; slots S2 and S4), the constants and the public signatures."""
    assert glm.GP1_CONFIG == dict(ROWS=8, BLOCK_K=1024, WR_BLOCK=128, num_warps=8, num_stages=1)
    assert glm.GP2_CONFIG == dict(
        BLOCK_M=128, BLOCK_K=64, SPLIT=14, ORDER=0, num_warps=8, num_stages=3
    )
    assert glm.GP2_STRICT_CONFIG == dict(
        BLOCK_M=128, BLOCK_K=64, SPLIT=16, PRECISION="tf32x3", num_warps=8, num_stages=3
    )
    assert glm.GP3_CONFIG == dict(BLOCK_M=16, num_warps=4, num_stages=1)
    assert glm.GP4P_CONFIG == dict(BLOCK_R=16, num_warps=4, num_stages=1)
    assert glm.GP4_CONFIG == dict(
        BLOCK_T=64,
        BLOCK_C=128,
        num_warps=8,
        num_stages=1,
        maxnreg=128,
        maxnreg_dbg=None,
        ANCHOR=1,
        GT=1,
    )
    assert glm.GP4_STRICT_CONFIG == dict(
        BLOCK_T=64, BLOCK_C=128, SPLIT_M=2, num_warps=8, num_stages=2
    )
    assert glm.GP4R_CONFIG == dict(RBC=64, num_warps=4, num_stages=1)
    # The strict GP-4r's config is the chosen GP-4 unit module's split-reduce RED_CFG.
    assert glm.GP4R_STRICT_CONFIG == dict(BLOCK=512, num_warps=4, num_stages=1)
    assert sorted(name for name in vars(glm) if name.endswith("_CONFIG")) == sorted(
        [
            "GP1_CONFIG",
            "GP2_CONFIG",
            "GP2_STRICT_CONFIG",
            "GP3_CONFIG",
            "GP4P_CONFIG",
            "GP4_CONFIG",
            "GP4_STRICT_CONFIG",
            "GP4R_CONFIG",
            "GP4R_STRICT_CONFIG",
        ]
    )
    assert (glm.GP4_NUM_SMS, glm.GP4_OCC) == (152, 2)
    assert (glm.PROJ_N, glm.BLOCK_N) == (24, 32)
    assert glm.GLM_PAIR_DEVICE_CAPABILITIES == ((10, 3),)
    assert glm._ROUND_MODE == {"rn": 1, "rz": 2}
    assert emu.P9_M3_BARS == _EMU_BARS

    keyword_only, empty = inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.empty
    parameters = inspect.signature(glm.fused_glm_proj_rms).parameters
    assert list(parameters) == ["x2d", "weight", "eps", "rounding"]
    assert parameters["rounding"].kind is keyword_only and parameters["rounding"].default == "rn"
    launchers = (
        (glm._glm_proj_fwd, ["x2d", "weight", "eps"], ["_s"]),
        (glm._glm_proj_bwd, ["x2d", "weight", "s", "p", "grad_p", "grad_s"], []),
    )
    for launcher, positional, extra in launchers:
        parameters = inspect.signature(launcher).parameters
        keywords = ["rounding", "mode", "debug", "_compiled", "_xround"] + extra
        assert list(parameters) == positional + keywords, launcher.__name__
        for name in keywords:
            assert parameters[name].kind is keyword_only, (launcher.__name__, name)
        assert parameters["rounding"].default is empty and parameters["mode"].default is empty
        assert parameters["debug"].default is False
        for name in ["_compiled", "_xround"] + extra:
            assert parameters[name].default is None, (launcher.__name__, name)
    assert issubclass(glm.FusedGLMProjRms, torch.autograd.Function)
    forward = inspect.signature(glm.FusedGLMProjRms.forward).parameters
    assert list(forward) == ["ctx", "x2d", "weight", "eps", "rounding"]
    assert list(inspect.signature(glm.GLMProjRmsOp.__call__).parameters) == [
        "self",
        "x2d",
        "weight",
        "eps",
    ]
    assert glm.GLMProjRmsOp.__slots__ == ("rounding",)
    assert list(inspect.signature(glm._matmul_mode).parameters) == []
    assert list(inspect.signature(glm._read_fp32_matmul_precision).parameters) == []
    for function, names in (
        (glm.gp2_split_bounds, ["K"]),
        (glm.gp2_strict_split_bounds, ["K"]),
        (glm.gp4_wgrad_split_bounds, ["T", "K"]),
        (glm.gp4_strict_wgrad_split_bounds, ["T"]),
    ):
        assert list(inspect.signature(function).parameters) == names, function.__name__


def test_glm_pair_kernel_lookup_requires_triton(monkeypatch):
    """The lookup fails closed: RuntimeError, not missing kernels, without Triton or without the
    kernels this process defined."""
    monkeypatch.setattr(fmk, "is_triton_available", lambda: False)
    with pytest.raises(RuntimeError, match="needs Triton"):
        glm._glm_kernels()
    # A patched-True availability (as the fused-ops tests patch it) without defined kernels.
    monkeypatch.setattr(fmk, "is_triton_available", lambda: True)
    monkeypatch.setattr(glm, "_GLM_KERNELS_DEFINED", False)
    with pytest.raises(RuntimeError, match="needs Triton"):
        glm._glm_kernels()


def test_glm_pair_op_is_a_stateless_singleton():
    """One op object per rounding: identity, copies and pickles return it; its label."""
    for rounding in ("rn", "rz"):
        op = glm.glm_proj_rms_op(rounding)
        assert op is glm.glm_proj_rms_op(rounding)
        assert op is not glm.glm_proj_rms_op("rz" if rounding == "rn" else "rn")
        assert isinstance(op, glm.GLMProjRmsOp) and op.rounding == rounding
        assert op.label == f"glm-triton(tf32={rounding})"
        assert copy.copy(op) is op and copy.deepcopy(op) is op
        assert copy.deepcopy({"op": op})["op"] is op
        assert pickle.loads(pickle.dumps(op)) is op
        assert not hasattr(op, "__dict__")
        with pytest.raises(AttributeError):
            op.extra_state = 1
    with pytest.raises(ValueError):
        glm.GLMProjRmsOp("rne")
    with pytest.raises((ValueError, KeyError)):
        glm.glm_proj_rms_op("rne")


def test_glm_pair_refuses_cpu_input(monkeypatch):
    """CPU input is refused first (row 1), without any CUDA query, with or without Triton."""

    def no_query(*args, **kwargs):
        raise AssertionError("supported() queried a CUDA device for CPU input")

    monkeypatch.setattr(torch.cuda, "get_device_capability", no_query)
    monkeypatch.setattr(torch.cuda, "get_device_properties", no_query)
    x = torch.zeros(16, 128, dtype=torch.bfloat16, requires_grad=True)
    W = torch.zeros(glm.PROJ_N, 128, requires_grad=True)
    op = glm.glm_proj_rms_op("rn")
    for available in (True, False):
        monkeypatch.setattr(fmk, "is_triton_available", lambda available=available: available)
        reason = op.refusal(x, W)
        assert reason is not None and "cpu" in reason and "CUDA" in reason, reason
        assert glm._refusal(x, W) == reason
        assert op.supported(x, W) is False
        with pytest.raises(RuntimeError, match="does not support this input"):
            glm.fused_glm_proj_rms(x, W, _PAIR_EPS)
        with pytest.raises(RuntimeError, match="does not support this input"):
            op(x, W, _PAIR_EPS)
        with torch.no_grad():
            assert op.refusal(x, W) == reason


class _OnCuda:
    """A CPU tensor whose device reads cuda:0, so that rows 1-2 of supported() pass on CPU.

    Rows 3-10 read only metadata, which the wrapped tensor answers."""

    def __init__(self, tensor):
        self._tensor = tensor

    @property
    def device(self):
        return torch.device("cuda", 0)

    def __getattr__(self, name):
        return getattr(self._tensor, name)


def _bf16(*shape):
    return torch.zeros(*shape, dtype=torch.bfloat16)


def _fp32(*shape):
    return torch.zeros(*shape, dtype=torch.float32)


def _misaligned(dtype, *shape):
    return torch.zeros(1 + int(torch.Size(shape).numel()), dtype=dtype)[1:].view(*shape)


# Rows 3-10 of plan 5.3.9: (inputs, the reason text the plan gives, as a substring).
_LAYOUT_CASES = {
    "x_fp32": (lambda: (_fp32(16, 128), _fp32(24, 128)), "x2d is torch.float32, expected bfloat16"),
    "weight_bf16": (lambda: (_bf16(16, 128), _bf16(24, 128)), "weight is torch.bfloat16"),
    "x_3d": (lambda: (_bf16(2, 8, 128), _fp32(24, 128)), "shape"),
    "k_mismatch": (lambda: (_bf16(16, 256), _fp32(24, 128)), "shape"),
    "n_23": (lambda: (_bf16(16, 128), _fp32(23, 128)), "N = 23; validated at 24"),
    "k_448": (lambda: (_bf16(16, 448), _fp32(24, 448)), "K = 448 is not a multiple of 128"),
    "dtype_before_k": (lambda: (_fp32(16, 448), _fp32(24, 448)), "x2d is torch.float32"),
    "t_0": (lambda: (_bf16(0, 128), _fp32(24, 128)), "no tokens"),
    "int32_offsets": (
        lambda: (
            torch.empty(1 << 24, 128, dtype=torch.bfloat16, device="meta"),
            torch.empty(24, 128, device="meta"),
        ),
        "T x K exceeds int32 offsets",
    ),
    # Row 8's reason (the kernels index in int32) at the offsets that grow with K or T alone:
    # the strict wgrad partials span 2 x 24 x K, the strict forward partials 16 x T x 32.
    "int32_offsets_k": (
        lambda: (
            torch.empty(1, 349526 * 128, dtype=torch.bfloat16, device="meta"),
            torch.empty(24, 349526 * 128, device="meta"),
        ),
        "int32 offsets",
    ),
    "int32_offsets_t": (
        lambda: (
            torch.empty(1 << 22, 128, dtype=torch.bfloat16, device="meta"),
            torch.empty(24, 128, device="meta"),
        ),
        "int32 offsets",
    ),
    # Row 9: a one-row view reads as contiguous whatever its row stride.
    "x_one_row_strided": (
        lambda: (
            torch.zeros(2 * 131, dtype=torch.bfloat16).as_strided((1, 128), (131, 1)),
            _fp32(24, 128),
        ),
        "x2d is not contiguous",
    ),
    "x_transposed": (lambda: (_bf16(128, 16).t(), _fp32(24, 16 * 8)), "x2d is not contiguous"),
    "weight_transposed": (lambda: (_bf16(16, 128), _fp32(128, 24).t()), "weight is not contiguous"),
    "x_misaligned": (
        lambda: (_misaligned(torch.bfloat16, 16, 128), _fp32(24, 128)),
        "x2d is not 16-byte aligned",
    ),
    "weight_misaligned": (
        lambda: (_bf16(16, 128), _misaligned(torch.float32, 24, 128)),
        "weight is not 16-byte aligned",
    ),
}


# Rows 2, 11 and 12, reached with rows 1-2's device reads patched: (capability, fp32 precision,
# Triton available, the reason text as a substring).
_ROW_CASES = {
    "row2_capability": ((9, 0), "tf32", True, "compute capability (9, 0)"),
    "row11_precision": ((10, 3), "bf16", True, "bf16"),
    "row12_no_triton": ((10, 3), "tf32", False, "Triton is unavailable"),
}


@pytest.mark.parametrize("case", sorted(_LAYOUT_CASES) + sorted(_ROW_CASES) + ["fixture_shapes"])
def test_glm_pair_layout_refusals(monkeypatch, case):
    """Rows 3-10 on CPU tensors through _layout_refusal; every fixture shape is admitted; with
    rows 1-2's device reads patched to pass, _refusal returns _layout_refusal's reason, and rows
    2, 11 and 12 refuse in their turn. Triton is reported available (row 12) unless a case says
    otherwise, so that the CPU cases do not depend on the process's Triton state."""
    monkeypatch.setattr(fmk, "is_triton_available", lambda: True)
    monkeypatch.setattr(glm, "_GLM_KERNELS_DEFINED", True)
    monkeypatch.setattr(glm, "_device_capability", lambda index: (10, 3))
    monkeypatch.setattr(glm, "_read_fp32_matmul_precision", lambda: "tf32")
    monkeypatch.delenv("NVIDIA_TF32_OVERRIDE", raising=False)
    if case in _ROW_CASES:
        capability, precision, available, expected = _ROW_CASES[case]
        x, W = _OnCuda(_bf16(16, 128)), _OnCuda(_fp32(24, 128))
        assert glm._refusal(x, W) is None and glm.glm_proj_rms_op("rn").supported(x, W)
        monkeypatch.setattr(glm, "_device_capability", lambda index: capability)
        monkeypatch.setattr(glm, "_read_fp32_matmul_precision", lambda: precision)
        monkeypatch.setattr(fmk, "is_triton_available", lambda: available)
        reason = glm._refusal(x, W)
        assert reason is not None and expected in reason, reason
        assert glm.glm_proj_rms_op("rn").supported(x, W) is False
        return
    if case == "fixture_shapes":
        shapes = set(_M3_SHAPES) | set(_FIXTURE_SHAPES) | {(17, 16384), (21, 16384)}
        shapes |= {(112, 16384), (4095, 16384)}
        for T, K in sorted(shapes):
            x, W = torch.empty(T, K, dtype=torch.bfloat16), torch.empty(glm.PROJ_N, K)
            assert glm._layout_refusal(x, W) is None, (T, K)
            assert glm._refusal(_OnCuda(x), _OnCuda(W)) is None, (T, K)
        return
    make, expected = _LAYOUT_CASES[case]
    x, W = make()
    reason = glm._layout_refusal(x, W)
    assert reason is not None and expected in reason, reason
    assert glm._refusal(_OnCuda(x), _OnCuda(W)) == reason


# (fp32 precision reading or the exception the accessor raises, NVIDIA_TF32_OVERRIDE, mode, a
# substring of the refusal reason).
_MODE_CASES = {
    "tf32": ("tf32", None, "tf32", None),
    "tf32_nvidia_override_0": ("tf32", "0", "ieee", None),
    "tf32_nvidia_override_1": ("tf32", "1", "tf32", None),
    "ieee": ("ieee", None, "ieee", None),
    "none": ("none", None, "ieee", None),
    "bf16": ("bf16", None, None, "bf16"),
    "accessor_raises": (RuntimeError, None, None, "RuntimeError"),
}


@pytest.mark.parametrize("case", list(_MODE_CASES))
def test_glm_pair_matmul_mode_predicate(monkeypatch, case):
    """The call-time precision predicate (plan 5.3.6), read through its seam only."""
    reading, override, mode, reason_part = _MODE_CASES[case]

    def read():
        if reading is RuntimeError:
            raise RuntimeError("torch's precision accessors are in a mixed state")
        return reading

    reads = []

    def forbidden(name, value):
        def accessor(*args, **kwargs):
            reads.append(name)
            return value

        return accessor

    monkeypatch.setattr(glm, "_read_fp32_matmul_precision", read)
    monkeypatch.setattr(torch._C, "_get_cublas_allow_tf32", forbidden("allow_tf32", True))
    monkeypatch.setattr(
        torch, "get_float32_matmul_precision", forbidden("float32_matmul_precision", "high")
    )
    if override is None:
        monkeypatch.delenv("NVIDIA_TF32_OVERRIDE", raising=False)
    else:
        monkeypatch.setenv("NVIDIA_TF32_OVERRIDE", override)
    got_mode, reason = glm._matmul_mode()
    assert got_mode == mode
    if mode is None:
        assert reason is not None and reason_part in reason, reason
    else:
        assert reason is None
    assert reads == []


def _runs(lengths):
    runs = []
    for length in lengths:
        if runs and runs[-1][1] == length:
            runs[-1] = (runs[-1][0] + 1, length)
        else:
            runs.append((1, length))
    return tuple(runs)


def _emulator_tile_pieces(column_plans, K, block_c=128):
    """The emulator's per-column plans as {column tile: token pieces}."""
    tiles = {}
    for ranges, plan in column_plans.groups:
        for start, stop in ranges:
            for c0 in range(start, stop, block_c):
                tiles[(c0, min(c0 + block_c, stop))] = plan.splits
    return tiles


@pytest.mark.parametrize("shape", _M3_SHAPES + _FIXTURE_SHAPES, ids=_shape_id)
def test_glm_pair_plans_equal_the_emulator(shape):
    """GLM's split functions equal the emulator's independently written plans, and the values
    the round-2 slots registered (S5)."""
    T, K = shape
    plans = emu.glm_pair_fused_plans(T, K, glm.PROJ_N)
    gp2 = glm.gp2_split_bounds(K)
    assert tuple(gp2) == plans.proj.splits == emu.gp2_plan_nonempty(K).splits
    assert _runs([stop - start for start, stop in gp2]) == _GP2_CHAINS[K]
    assert all(stop > start for start, stop in gp2)  # GP-3's SPLIT is the number of chains
    assert plans.dgrad.splits == ((0, glm.PROJ_N),)
    strict = glm.gp2_strict_split_bounds(K)
    assert tuple(strict) == emu.gp2_plan(K).splits and len(strict) == 16
    assert tuple(glm.gp4_strict_wgrad_split_bounds(T)) == emu.gp4_wgrad_plan(T).splits

    tiles = glm.gp4_wgrad_split_bounds(T, K)
    columns = [tuple(c) for c, _ in tiles]
    assert columns == [(c0, min(c0 + 128, K)) for c0 in range(0, K, 128)]
    ours = {tuple(c): tuple(tuple(piece) for piece in pieces) for c, pieces in tiles}
    assert ours == _emulator_tile_pieces(plans.wgrad, K)
    for pieces in ours.values():  # contiguous token pieces in order, covering [0, T)
        assert pieces[0][0] == 0 and pieces[-1][1] == T
        assert all(a[1] == b[0] for a, b in zip(pieces, pieces[1:]))
    layouts, per_tile, tokens, nprog, nslot, npieces = _GP4_PLANS[shape]
    counts = [len(pieces) for pieces in ours.values()]
    lengths = [stop - start for pieces in ours.values() for start, stop in pieces]
    assert len(set(ours.values())) == layouts
    assert (min(counts), max(counts)) == per_tile
    assert (min(lengths), max(lengths)) == tokens
    partition = glm._gp4_partition(T, K)
    assert partition.NPROG == nprog
    if nslot is not None:
        assert partition.NSLOT == nslot and sum(counts) == npieces

    # The launchers' own grids and GP-3's SPLIT (built on meta tensors with placeholder kernels):
    # the S5 chain counts (14, 13, 6 at the M3 K values; 8 at K 512 and 2 at K 128), 16 in
    # strict mode; GP-4 on NPROG programs, GP-4r on two programs per column tile.
    kernels = glm._GLMKernels(*glm._GLMKernels._fields)
    meta = dict(device="meta")
    x = torch.empty(T, K, dtype=torch.bfloat16, **meta)
    W = torch.empty(glm.PROJ_N, K, **meta)
    p, s = torch.empty(T, glm.PROJ_N, **meta), torch.empty(T, 1, **meta)
    n_chain = sum(count for count, _ in _GP2_CHAINS[K])
    for mode, split, gp4_grid, gp4r_grid in (
        ("tf32", n_chain, (nprog,), (2 * (K // 128),)),
        ("ieee", 16, (K // 128, 2), (-(-glm.PROJ_N * K // 512),)),
    ):
        strict = "_strict" if mode == "ieee" else ""
        ws = torch.empty(glm._fwd_layout(T, K, mode).total, **meta)
        fwd = glm._fwd_launches(kernels, x, W, _PAIR_EPS, p, s, ws, rounding="rn", mode=mode)
        launches = {launch.name: launch for launch in fwd}
        assert launches[_SHIPPED["gp3"]].constexprs["SPLIT"] == split, mode
        assert launches[_SHIPPED["gp2" + strict]].grid == (-(-T // 128), split), mode
        ws = torch.empty(glm._bwd_layout(T, K, mode).total, **meta)
        g_x, g_W = torch.empty_like(x), torch.empty(glm.PROJ_N, K, **meta)
        bwd = glm._bwd_launches(
            kernels, x, W, s, p, p, None, g_x, g_W, ws, {}, rounding="rn", mode=mode, debug=False
        )
        launches = {launch.name: launch for launch in bwd}
        assert launches[_SHIPPED["gp4" + strict]].grid == gp4_grid, mode
        assert launches[_SHIPPED["gp4r" + strict]].grid == gp4r_grid, mode


def test_glm_pair_source_has_no_autotune_or_atomics():
    """GLM's code (comments and strings blanked by the coverage tool's helper) has no autotune,
    no atomics, no private Triton allocator and no host TMA descriptors."""
    from tools import check_kernel_determinism_coverage as coverage

    code = coverage.strip_comments_and_strings(Path(glm.__file__).read_text())
    for pattern in (
        r"\bautotune\b",
        r"\btl\.atomic_",
        r"\batomic_(add|cas|xchg|max|min|and|or|xor)\b",
        r"triton\.runtime\._allocation",
        r"\b_allocation\._allocator\b",
        r"\bTensorDescriptor\b",
    ):
        assert not re.search(pattern, code), pattern


_SUBPROCESS_SCRIPT = """
import importlib.util, json, sys
spec = importlib.util.spec_from_file_location("tglm_subprocess", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
print(json.dumps(getattr(module, sys.argv[2])()))
"""


def _run_in_subprocess(function_name, env, cwd, timeout):
    """Run a function of this file in a fresh interpreter and return the JSON it prints last."""
    pythonpath = os.pathsep.join(filter(None, [str(_REPO_ROOT), env.get("PYTHONPATH")]))
    result = subprocess.run(
        [sys.executable, "-c", _SUBPROCESS_SCRIPT, str(Path(__file__).resolve()), function_name],
        env=dict(env, PYTHONPATH=pythonpath),
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    print(result.stdout[-50000:])
    assert result.returncode == 0, result.stderr[-4000:]
    return json.loads(result.stdout.strip().splitlines()[-1])


def _fp32_class(bits):
    if bits & 0x7F800000 != 0x7F800000:
        return "finite"
    if bits & 0x007FFFFF:
        return "nan"
    return "-inf" if bits & 0x80000000 else "+inf"


def _tf32_round_report(device, modes):
    """GLM's _tf32_round on every emulator edge pattern against the integer oracle: MODE 1 is RNE
    and MODE 2 RZ, bit for bit; MODE 3 (cvt.rn.tf32.f32) keeps the oracle's upper 19 bits on
    every finite pattern and the class of every non-finite one."""
    patterns = emu.tf32_edge_patterns()
    signed = [bits - (1 << 32) if bits >= 1 << 31 else bits for bits in patterns]
    x = torch.tensor(signed, dtype=torch.int32).to(device).view(torch.float32)
    report = {}
    for mode in modes:
        y = torch.empty_like(x)
        grid = ((len(patterns) + 1023) // 1024,)
        _tf32_round_probe_kernel[grid](x, y, len(patterns), MODE=mode, BLOCK=1024)
        got = (y.view(torch.int32).cpu().to(torch.int64) & 0xFFFFFFFF).tolist()
        oracle = "rz" if mode == 2 else "rne"
        bad = []
        for bits, out in zip(patterns, got):
            expected = emu.tf32_round_int_reference(bits, oracle)
            if mode != 3:
                ok = out == expected
            elif _fp32_class(bits) == "finite":
                ok = out & 0xFFFFE000 == expected & 0xFFFFE000
            else:
                ok = _fp32_class(out) == _fp32_class(bits)
            if not ok:
                bad.append(f"{bits:#010x}->{out:#010x} (oracle {expected:#010x})")
        report[str(mode)] = {"checked": len(patterns), "mismatches": len(bad), "first": bad[:8]}
    return report


def _interpreted_tf32_round_report():
    assert os.environ.get("TRITON_INTERPRET") == "1"
    return _tf32_round_report("cpu", (1, 2))


@pytest.mark.skipif(
    not (fmk.is_triton_available() and glm._GLM_KERNELS_DEFINED), reason="needs Triton"
)
def test_glm_pair_tf32_round_matches_oracle_interpreted(tmp_path):
    """MODE 1 and 2 of GLM's _tf32_round, interpreted (TRITON_INTERPRET=1 is read when kernels
    are decorated, so in a fresh process), equal the integer oracle on every edge pattern."""
    report = _run_in_subprocess(
        "_interpreted_tf32_round_report", dict(os.environ, TRITON_INTERPRET="1"), tmp_path, 900
    )
    assert set(report) == {"1", "2"}
    for mode, result in report.items():
        assert result["checked"] == len(emu.tf32_edge_patterns()), mode
        assert result["mismatches"] == 0, (mode, result["first"])


def _gp4_device_plan_problems(T, K):
    """The kernels' plan helpers (interpreted) against gp4_wgrad_split_bounds at one shape: GP-4's
    pieces (program q's units, its column tiles and token blocks) are the function's pieces in
    GP-4r's order; every written slot is unique and below NSLOT; GP-4r reads, for each tile,
    exactly the slots of the programs that wrote it."""
    partition = glm._gp4_partition(T, K)
    nb, nc, nu, nprog, nslot = (
        partition.NB,
        partition.NC,
        partition.NU,
        partition.NPROG,
        partition.NSLOT,
    )
    P, C = 512, 128  # >= GP4_NUM_SMS * GP4_OCC programs and >= the column tiles at K 16384
    assert nprog <= P and nc <= C, (T, K)
    out = torch.full((9 * P * C,), -1, dtype=torch.int32)
    _gp4_plan_probe_kernel[(1,)](out, nb, nu, nprog, P=P, C=C)
    u_lo, u_hi, c_lo, c_hi, b_lo, b_hi, slot, p_first, p_last = out.view(9, P, C).tolist()
    problems, pieces, written = [], {c: [] for c in range(nc)}, set()
    for q in range(nprog):
        if not u_lo[q][0] < u_hi[q][0]:
            problems.append(f"program {q} owns no unit")
        for c in range(c_lo[q][0], c_hi[q][0] + 1):
            if not b_lo[q][c] < b_hi[q][c]:
                problems.append(f"program {q} tile {c}: an empty piece")
            if slot[q][c] in written or not 0 <= slot[q][c] < nslot:
                problems.append(
                    f"program {q} tile {c}: slot {slot[q][c]} collides or is out of range"
                )
            written.add(slot[q][c])
            pieces[c].append((q, b_lo[q][c], b_hi[q][c]))
    for c in range(nc):
        readers = list(range(p_first[0][c], p_last[0][c] + 1))
        if readers != [q for q, _, _ in pieces[c]]:
            problems.append(f"tile {c}: GP-4r reads programs {readers}, written by {pieces[c]}")
    tiles = tuple(
        (
            (c * 128, min((c + 1) * 128, K)),
            tuple((lo * 64, min(hi * 64, T)) for _, lo, hi in pieces[c]),
        )
        for c in range(nc)
    )
    if tiles != glm.gp4_wgrad_split_bounds(T, K):
        problems.append("the pieces differ from gp4_wgrad_split_bounds")
    return problems


def _interpreted_gp4_plan_report():
    assert os.environ.get("TRITON_INTERPRET") == "1"
    return {
        _shape_id(shape): _gp4_device_plan_problems(*shape)
        for shape in _M3_SHAPES + _FIXTURE_SHAPES
    }


@pytest.mark.skipif(
    not (fmk.is_triton_available() and glm._GLM_KERNELS_DEFINED), reason="needs Triton"
)
def test_glm_pair_gp4_device_plan_equals_split_function(tmp_path):
    """GP-4 and GP-4r take their persistent partition, pieces and slots from GLM's plan helpers;
    interpreted at the 15 M3 shapes and the fixture shapes, the helpers give exactly
    gp4_wgrad_split_bounds' pieces, unique slots, and GP-4r reads of exactly the written slots
    (plan 5.3.3: the function is the kernels' plan)."""
    report = _run_in_subprocess(
        "_interpreted_gp4_plan_report", dict(os.environ, TRITON_INTERPRET="1"), tmp_path, 900
    )
    assert set(report) == {_shape_id(shape) for shape in _M3_SHAPES + _FIXTURE_SHAPES}
    for shape, problems in report.items():
        assert problems == [], (shape, problems[:8])


# --- GLM pair: GPU ---------------------------------------------------------------------------


@_NEEDS_CUDA_TRITON
@pytest.mark.parametrize("mode", [1, 2, 3], ids=["mode1", "mode2", "mode3"])
def test_glm_pair_tf32_round_compiled_matches_oracle(glm_pair_gpu, mode):
    """The rounding helper compiled on sm_103 (R4): MODE 1 and 2 bit for bit; MODE 3's upper 19
    bits on every finite pattern and the class of every non-finite one."""
    result = _tf32_round_report(_DEVICE, (mode,))[str(mode)]
    assert result["mismatches"] == 0, result["first"]


def _value_class(tensor):
    """0 finite, 1 +inf, 2 -inf, 3 NaN, per element."""
    t = tensor.detach().float()
    classes = torch.zeros(t.shape, dtype=torch.int8, device=t.device)
    classes[t == float("inf")] = 1
    classes[t == float("-inf")] = 2
    classes[torch.isnan(t)] = 3
    return classes


@_NEEDS_CUDA_TRITON
def test_glm_pair_nonfinite_rows_keep_class(glm_pair_gpu):
    """R4: with an inf row and a NaN row in x, every output element of the shipped rn build has
    the class it has in the test-only MODE 1 build (xn by bit operations instead of cvt)."""
    T, K = 64, 16384
    x, W, G_p = _pair_inputs(T, K)
    x[3] = float("inf")
    x[5] = float("nan")

    def pipeline(xround):
        p, s = _fwd(x, W, _xround=xround)
        g_x, g_W, _ = _bwd(x, W, s, p, G_p, None, _xround=xround)
        _, _, extras = _bwd(x, W, s, p, G_p, None, debug=True, _xround=xround)
        return {"p": p, "s": s, "g_x": g_x, "g_W": g_W, **extras}

    shipped, mode1 = pipeline(None), pipeline(1)
    assert set(shipped) == set(mode1)
    for name in shipped:
        assert torch.equal(_value_class(shipped[name]), _value_class(mode1[name])), name
    assert torch.isinf(x[3].float()).all() and _value_class(shipped["p"]).any()


def _r2_cases():
    cases = []
    for T, K in _M3_SHAPES:
        for rows, scale in _R2_INPUTS:
            cases.append(pytest.param(T, K, rows, scale, "grid", id=f"{T}x{K}-{rows}-x{scale:g}"))
    for T, K in _DROP_IN_SHAPES:
        for kind in ("with_gs", "norm_only"):
            cases.append(pytest.param(T, K, "randn", 1.0, kind, id=f"{T}x{K}-{kind}"))
    return cases


@_NEEDS_CUDA_TRITON
@pytest.mark.parametrize("T,K,rows,scale,kind", _r2_cases())
def test_glm_pair_float64_noninferiority(glm_pair_gpu, tf32_matmuls, T, K, rows, scale, kind):
    """R2 in TF32 mode at the 15 M3 shapes x 5 inputs and the drop-in cases: p, the fp32 g_x
    and g_W against float64 (<= 1.1 x unfused and <= 1e-3, with the norm-only and precision-class
    rules), s <= 1e-5, the bitwise checks, and R3's epilogue rule at the same case."""
    _assert_tf32_state()
    x, W, G_p = _pair_inputs(T, K, rows, scale)
    G_s = None if kind == "grid" else _grad_s(T)
    if kind == "norm_only":
        G_p = None
    failures = _r2_failures(f"{T}x{K}-{rows}-x{scale:g}-{kind}", x, W, G_p, G_s, "tf32")
    assert not failures, "\n".join(failures)


@_NEEDS_CUDA_TRITON
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("rows", ["randn", "heavy", "zero_rows"])
def test_glm_pair_matches_emulator(glm_pair_gpu, tf32_matmuls, rows, seed):
    """R3's 9 gate cases at production shape: p with GP-1's s injected, the debug g_xn and the
    production g_W with GP-3's p injected, on the shipped plans, within P9's bars."""
    _assert_tf32_state()
    T, K = _PRODUCTION
    x, W, G_p = _pair_inputs(T, K, rows, 1.0, seed)
    out = _tf32_outputs(x, W, G_p, "rn")
    forward = _emulate(x, W, None, out["s"])
    backward = _emulate(x, W, G_p, out["s"], p=out["p"])
    errors = {
        "p": rel_rms(out["p"], forward.p),
        "g_xn": rel_rms(out["g_xn"], backward.g_xn),
        "g_W": rel_rms(out["g_W"], backward.g_W),
    }
    for name, error in errors.items():
        _record("R3", case=f"{rows}-seed{seed}", quantity=name, error=error, bar=_EMU_BARS[name])
    for name, error in errors.items():
        assert error <= _EMU_BARS[name], f"{name}: rel-RMS {error:.3e} > {_EMU_BARS[name]:.3e}"


@_NEEDS_CUDA_TRITON
@pytest.mark.parametrize("with_gs", [False, True], ids=["no_gs", "with_gs"])
@pytest.mark.parametrize("shape", _M3_SHAPES, ids=_shape_id)
def test_glm_pair_epilogue_is_bit_exact(glm_pair_gpu, shape, with_gs):
    """R3's epilogue rule at every R2 input of a shape: the debug g_s equals O6 on the kernel's
    p, s and raw G_p; the debug g_x32 equals fl(s * g_xn) - fl(coef * xf) on the kernel's own
    g_xn; the stored bf16 g_x equals RN(g_x32); all bitwise."""
    T, K = shape
    failures = []
    for rows, scale in _R2_INPUTS:
        x, W, G_p = _pair_inputs(T, K, rows, scale)
        G_s = _grad_s(T) if with_gs else None
        p, s = _fwd(x, W)
        g_x, _, _ = _bwd(x, W, s, p, G_p, G_s)
        _, _, extras = _bwd(x, W, s, p, G_p, G_s, debug=True)
        label = f"{T}x{K}-{rows}-x{scale:g}-gs{int(with_gs)}"
        failures += _epilogue_failures(label, x, s, p, G_p, G_s, extras, g_x)
    assert not failures, "\n".join(failures)


@_NEEDS_CUDA_TRITON
def test_glm_pair_order_and_rn(glm_pair_gpu, tf32_matmuls):
    """R4's order test, the statistical RN test (W and G_p with the low 13 bits set), the rz
    build's agreement with operand_mode "rz", and the rz build as a kernel-side mutant."""
    _assert_tf32_state()
    T, K = _PRODUCTION
    x, W, G_p = _pair_inputs(T, K)
    failures = []
    p, s = _fwd(x, W)
    follows_kernel = _emulate(x, W, None, s)
    project_first = _emulate(x, W, None, s, mutation="project_first")
    fused_error = rel_rms(p, follows_kernel.p)
    power = rel_rms(project_first.p, follows_kernel.p)
    _record("R4_ORDER", fused=fused_error, bar=_EMU_BARS["p"], power=power, minimum=_ORDER_MINIMUM)
    if not fused_error <= _EMU_BARS["p"]:
        failures.append(f"order: rel(GP-3 p, emu_HF p) {fused_error:.3e} > the p bar")
    if not power >= _ORDER_MINIMUM:
        failures.append(f"order: rel(emu_A p, emu_HF p) {power:.3e} < {_ORDER_MINIMUM:g}")

    W_rn, G_rn = _low_bits_set(W), _low_bits_set(G_p)
    rn, rz = _tf32_outputs(x, W_rn, G_rn, "rn"), _tf32_outputs(x, W_rn, G_rn, "rz")
    emu_rn = _emulate(x, W_rn, G_rn, rn["s"])
    emu_rz = _emulate(x, W_rn, G_rn, rn["s"], mutation="rz_for_rn")
    if _same_bits(rz["s"], rn["s"]):
        emu_rz_own = emu_rz
    else:
        emu_rz_own = _emulate(x, W_rn, G_rn, rz["s"], operand_mode="rz")
    for name in ("p", "g_xn", "g_W"):
        bar = _EMU_BARS[name]
        values = {
            "rn_vs_emu_rn": rel_rms(rn[name], getattr(emu_rn, name)),
            "emu_rz_vs_emu_rn": rel_rms(getattr(emu_rz, name), getattr(emu_rn, name)),
            "rz_vs_emu_rz": rel_rms(rz[name], getattr(emu_rz_own, name)),
            "rz_vs_emu_rn": rel_rms(rz[name], getattr(emu_rn, name)),
        }
        _record("R4_RN", quantity=name, bar=bar, **values)
        if not values["rn_vs_emu_rn"] <= bar:
            failures.append(f"RN: {name} rel(fused rn, emu_RN) {values['rn_vs_emu_rn']:.3e} > bar")
        if not values["emu_rz_vs_emu_rn"] >= _RN_MINIMUM:
            failures.append(f"RN: {name} rel(emu_RZ, emu_RN) below {_RN_MINIMUM:g}")
        if not values["rz_vs_emu_rz"] <= bar:
            failures.append(f"rz: {name} rel(rz build, emu rz) {values['rz_vs_emu_rz']:.3e} > bar")
        if not values["rz_vs_emu_rn"] > bar:
            failures.append(f"kernel-side mutant: {name} rz build within the bar of emu_RN")
    assert not failures, "\n".join(failures)


def _probe_positions(T, K):
    """P2's crafted-operand positions."""
    k0 = K // 2 + 5
    return {
        "rows": [(229 * i + 3) % T for i in range(len(_PROBE_COMBOS))],
        "t0": 77 % T,
        "k0": k0,
        "cols": [(k0 + 3 + 97 * j) % K for j in range(len(_PROBE_COMBOS))],
        "n0": 5,
    }


def _probe_launches(T):
    """Value indices per launch of a slot with one value per row: all 18 when their rows are
    distinct, else two launches of 9 at rows (229 i + 3) mod T for i < 9."""
    n = len(_PROBE_COMBOS)
    if len({(229 * i + 3) % T for i in range(n)}) == n:
        return [list(range(n))]
    return [list(range(9)), list(range(9, n))]


def _probe_value(i):
    key, sign, scale = _PROBE_COMBOS[i]
    return sign * scale * _PROBE_VALUES[key]


def _probe_classes(readings, predict):
    """P1's table per (sign, scale) case: the first class whose predicted readings all match."""
    classes = []
    for sign in _PROBE_SIGNS:
        for scale in _PROBE_SCALES:
            indices = [i for i, combo in enumerate(_PROBE_COMBOS) if combo[1:] == (sign, scale)]
            found = "unclassified"
            for mode in ("rz", "rne", "rna", "none"):
                if all(readings[i] == predict(mode, i) for i in indices):
                    found = mode
                    break
            classes.append(found)
    return classes


def _leaks(tensor, mask=None):
    """P2's unexpected_nonzero_elsewhere: elements outside the crafted positions that are not
    zero (NaN included: an element no kernel wrote under det-fill)."""
    outside = torch.ones(tensor.shape, dtype=torch.bool, device=tensor.device)
    if mask is not None:
        outside[mask] = False
    return int(((tensor.float() != 0) & outside).sum().item())


def _crafted_slot(slot, rounding, T, K):
    """One slot of R4's crafted-operand test: (readings in combo order, predictor, leaks)."""
    pos = _probe_positions(T, K)
    k0, t0, n0, n1 = pos["k0"], pos["t0"], pos["n0"], _CANCEL_COLUMN
    n = len(_PROBE_COMBOS)
    readings, leaks = [None] * n, collections.Counter()
    f32 = dict(dtype=torch.float32, device=_DEVICE)

    def direct(mode, i):  # a reading of T(value) * 1
        key, sign, scale = _PROBE_COMBOS[i]
        return sign * scale * _PROBE_EXPECTED[mode][key]

    if slot in ("xn", "W_fwd"):
        groups = _probe_launches(T) if slot == "xn" else [list(range(n))]
        for group in groups:
            x = torch.zeros(T, K, dtype=torch.bfloat16, device=_DEVICE)
            W = torch.zeros(glm.PROJ_N, K, **f32)
            s_in = torch.ones(T, 1, **f32)
            mask = torch.zeros(T, glm.PROJ_N, dtype=torch.bool, device=_DEVICE)
            if slot == "xn":
                rows = [(229 * j + 3) % T for j in range(len(group))]
                for row, i in zip(rows, group):
                    x[row, k0] = 1.0
                    s_in[row, 0] = _probe_value(i)
                    mask[row, n0] = True
                W[n0, k0] = 1.0
            else:
                x[t0, k0] = 1.0
                for j in group:
                    W[j, k0] = _probe_value(j)
                    mask[t0, j] = True
            with _det_fill():
                p, _ = _fwd(x, W, rounding=rounding, _s=s_in)
            for j, i in enumerate(group):
                readings[i] = p[rows[j], n0].item() if slot == "xn" else p[t0, i].item()
            leaks["p"] += _leaks(p, mask)
        return readings, direct, leaks

    if slot in ("G_p_dgrad", "W_dgrad"):
        groups = _probe_launches(T) if slot == "G_p_dgrad" else [list(range(n))]
        s_rows = {}
        for group in groups:
            x = torch.zeros(T, K, dtype=torch.bfloat16, device=_DEVICE)
            W = torch.zeros(glm.PROJ_N, K, **f32)
            G_p = torch.zeros(T, glm.PROJ_N, **f32)
            mask = torch.zeros(T, K, dtype=torch.bool, device=_DEVICE)
            places = []
            if slot == "G_p_dgrad":
                rows = [(229 * j + 3) % T for j in range(len(group))]
                for row, i in zip(rows, group):
                    _, sign, scale = _PROBE_COMBOS[i]
                    G_p[row, n0] = _probe_value(i)
                    G_p[row, n1] = -sign * scale
                    places.append((i, row, k0))
                W[n0, k0] = W[n1, k0] = 1.0
            else:
                G_p[t0, n0] = G_p[t0, n1] = 1.0
                for i in group:
                    _, sign, scale = _PROBE_COMBOS[i]
                    W[n0, pos["cols"][i]] = _probe_value(i)
                    W[n1, pos["cols"][i]] = -sign * scale
                    places.append((i, t0, pos["cols"][i]))
            for _, row, col in places:
                mask[row, col] = True
            with _det_fill():
                p, s = _fwd(x, W, rounding=rounding)
                g_x, g_W, _ = _bwd(x, W, s, p, G_p, None, rounding=rounding)
            for i, row, col in places:
                readings[i] = g_x[row, col].float().item()
                s_rows[i] = s[row, 0].item()
            leaks["g_x"] += _leaks(g_x, mask)
            leaks["g_W"] += _leaks(g_W)
            leaks["p"] += _leaks(p)
            leaks["s_nonfinite"] += int((~torch.isfinite(s)).sum().item())

        def through_g_x(mode, i):  # bf16_RN(fl32(s * g_xn)) with g_xn = sign * scale * (T(a) - 1)
            key, sign, scale = _PROBE_COMBOS[i]
            g_xn = torch.tensor(sign * scale * (_PROBE_EXPECTED[mode][key] - 1.0))
            product = torch.tensor(s_rows[i], dtype=torch.float32) * g_xn.float()
            return product.to(torch.bfloat16).float().item()

        return readings, through_g_x, leaks

    # The wgrad slots: s and p are given (p = 0, so g_s = 0 and g_x = 0); W = 0.
    x = torch.zeros(T, K, dtype=torch.bfloat16, device=_DEVICE)
    W = torch.zeros(glm.PROJ_N, K, **f32)
    G_p = torch.zeros(T, glm.PROJ_N, **f32)
    s_in = torch.ones(T, 1, **f32)
    p_in = torch.zeros(T, glm.PROJ_N, **f32)
    mask = torch.zeros(glm.PROJ_N, K, dtype=torch.bool, device=_DEVICE)
    places = []
    if slot == "G_p_wgrad":
        x[t0, k0] = 1.0
        for j in range(n):
            G_p[t0, j] = _probe_value(j)
            places.append((j, j, k0))
    else:  # xn_wgrad: token rows[i // 6] has s = the value's key, x = sign * scale (bf16 exact)
        for i, (key, sign, scale) in enumerate(_PROBE_COMBOS):
            row = pos["rows"][i // 6]
            x[row, pos["cols"][i]] = sign * scale
            s_in[row, 0] = _PROBE_VALUES[key]
            G_p[row, n0] = 1.0
            places.append((i, n0, pos["cols"][i]))
    for _, row, col in places:
        mask[row, col] = True
    with _det_fill():
        g_x, g_W, _ = _bwd(x, W, s_in, p_in, G_p, None, rounding=rounding)
    for i, row, col in places:
        readings[i] = g_W[row, col].item()
    leaks["g_W"] += _leaks(g_W, mask)
    leaks["g_x"] += _leaks(g_x)
    return readings, direct, leaks


@_NEEDS_CUDA_TRITON
@pytest.mark.parametrize("shape", [_PRODUCTION, (17, 384)], ids=_shape_id)
@pytest.mark.parametrize("rounding", ["rn", "rz"])
@pytest.mark.parametrize("slot", _CRAFTED_SLOTS)
def test_glm_pair_crafted_operands_classify(glm_pair_gpu, tf32_matmuls, slot, rounding, shape):
    """R4's crafted-operand test on the compiled production tiles: P1's 18 values in one operand
    slot on a zero background, read under det-fill; every reading classifies RNE under rn and RZ
    under rz (the rz readings are the readout's positive control), and nothing leaks."""
    _assert_tf32_state()
    T, K = shape
    readings, predict, leaks = _crafted_slot(slot, rounding, T, K)
    classes = _probe_classes(readings, predict)
    expected = "rne" if rounding == "rn" else "rz"
    _record("R4_CRAFTED", slot=slot, rounding=rounding, shape=_shape_id(shape), classes=classes)
    _record("R4_CRAFTED_LEAKS", slot=slot, rounding=rounding, shape=_shape_id(shape), **leaks)
    assert classes == [expected] * len(classes), [
        (combo, float.hex(float(value))) for combo, value in zip(_PROBE_COMBOS, readings)
    ]
    assert all(count == 0 for count in leaks.values()), dict(leaks)


def _emulator_row_failures(row):
    T, K = _PRODUCTION
    x, W, G_p = _pair_inputs(T, K)
    out = _tf32_outputs(x, W, G_p, "rn")
    plain = _emulate(x, W, G_p, out["s"])
    mutated = _emulate(x, W, G_p, out["s"], mutation=row)
    failures = []
    for name in ("p", "g_xn", "g_W"):
        bar, covered = _EMU_BARS[name], name in _S8_COVERED[row]
        plain_error = rel_rms(out[name], getattr(plain, name))
        mutated_error = rel_rms(out[name], getattr(mutated, name))
        _record(
            "R4_MUTATION",
            row=row,
            quantity=name,
            covered=covered,
            fused_vs_emu=plain_error,
            fused_vs_mutated=mutated_error,
            bar=bar,
        )
        if covered and not plain_error <= bar:
            failures.append(f"{row}/{name}: rel(fused, emu) {plain_error:.3e} > bar")
        if covered and not mutated_error > bar:
            failures.append(f"{row}/{name}: the fused output meets the mutated specification")
    return failures


def _eps_row_failures():
    T, K = _PRODUCTION
    failures = []
    for label, rows, scale, covered in _S8_EPS_INPUTS:
        x, W, _ = _pair_inputs(T, K, rows, scale)
        _, s = _fwd(x, W)
        mean_square = x.double().square().sum(dim=-1) / K
        inside = torch.rsqrt(mean_square + emu.EPS_FP32)
        outside = 1.0 / (mean_square.sqrt() + emu.EPS_FP32)
        plain_error = rel_rms(s.reshape(-1), inside)
        mutated_error = rel_rms(s.reshape(-1), outside)
        _record(
            "R4_MUTATION",
            row="eps_outside_sqrt",
            input=label,
            covered=covered,
            fused_vs_f64=plain_error,
            fused_vs_mutated=mutated_error,
            bar=_S_BAR,
        )
        if not plain_error <= _S_BAR:
            failures.append(f"eps_outside_sqrt/{label}: rel(s, float64 s) {plain_error:.3e}")
        if covered and not mutated_error > _S_BAR:
            failures.append(f"eps_outside_sqrt/{label}: s meets the eps-outside specification")
    return failures


def _radial_row_failures():
    T, K = _PRODUCTION
    failures = []
    for label, scale, covered in _S8_RADIAL_INPUTS:
        x, W, G_p = _pair_inputs(T, K, "randn", scale)
        p, s = _fwd(x, W)
        _, _, extras = _bwd(x, W, s, p, G_p, None, debug=True)
        reference = emu.glm_pair_reference_f64(x, W, G_p, eps=emu.EPS_FP32)
        unfused_error = rel_rms(_unfused_pair(x, W, G_p, None)["g_x"], reference.g_x32)
        without_radial = reference.s[:, None] * reference.g_xn
        plain_error = rel_rms(extras["g_x32"], reference.g_x32)
        mutated_error = rel_rms(extras["g_x32"], without_radial)
        threshold = min(_RATIO * unfused_error, _TF32_BAR)
        plain_bar, _ = _operator_bar("g_x", unfused_error, False, False)
        _record(
            "R4_MUTATION",
            row="drop_radial",
            input=label,
            covered=covered,
            fused_vs_f64=plain_error,
            fused_vs_mutated=mutated_error,
            unfused=unfused_error,
            threshold=threshold,
        )
        if covered and not plain_error <= plain_bar:
            failures.append(f"drop_radial/{label}: rel(g_x32, float64) {plain_error:.3e} > bar")
        if covered and not mutated_error > threshold:
            failures.append(f"drop_radial/{label}: g_x32 meets the dropped-radial specification")
    return failures


def _sinkhorn_row_failures():
    """The Sinkhorn t0_like_others row on the Triton Sinkhorn's logits gradient, against M2's
    bar max(1.1 x native, 1e-5), on the emulator test's Sinkhorn draws (seed 7)."""
    from megatron.core.transformer.hyper_connection import _sinkhorn_iterations

    failures = []
    for scale, covered in _SINKHORN_SCALES:
        generator = torch.Generator().manual_seed(7)
        logits64 = torch.randn(T, N, N, generator=generator, dtype=torch.float64) * scale
        grad64 = torch.randn(T, N, N, generator=generator, dtype=torch.float64)
        logits = logits64.float().view(_SHAPE).to(_DEVICE).requires_grad_()
        grad = grad64.float().view(_SHAPE).to(_DEVICE)
        out = glm.fixed_launch_sinkhorn(logits, ITERS, EPS)
        (triton_grad,) = torch.autograd.grad(out, logits, grad)
        _, reference = _float64_reference(logits, grad)
        leaf = logits.detach().clone().requires_grad_()
        (native_grad,) = torch.autograd.grad(_sinkhorn_iterations(leaf, ITERS, EPS), leaf, grad)
        bar = max(_RATIO * rel_rms(native_grad.cpu(), reference), _STRICT_FLOOR)
        mutated = emu.sinkhorn_vjp(
            logits.detach().cpu().double().view(T, N, N),
            grad.cpu().double().view(T, N, N),
            ITERS,
            EPS,
            mutation="t0_like_others",
        ).view(_SHAPE)
        plain_error = rel_rms(triton_grad.cpu(), reference)
        mutated_error = rel_rms(triton_grad.cpu(), mutated)
        _record(
            "R4_MUTATION",
            row="sinkhorn_t0_like_others",
            scale=f"x{scale:g}",
            covered=covered,
            fused_vs_f64=plain_error,
            fused_vs_mutated=mutated_error,
            bar=bar,
        )
        if covered and not plain_error <= bar:
            failures.append(
                f"sinkhorn_t0_like_others/x{scale:g}: rel(grad, float64) {plain_error:.3e} > bar"
            )
        if covered and not mutated_error > bar:
            failures.append(f"sinkhorn_t0_like_others/x{scale:g}: meets the mutated specification")
    return failures


def _rz_build_row_failures():
    """The rz build as a compiled mutant of the RN rounding, on R4's RN-test inputs."""
    T, K = _PRODUCTION
    x, W, G_p = _pair_inputs(T, K)
    W_rn, G_rn = _low_bits_set(W), _low_bits_set(G_p)
    rn, rz = _tf32_outputs(x, W_rn, G_rn, "rn"), _tf32_outputs(x, W_rn, G_rn, "rz")
    emu_rn = _emulate(x, W_rn, G_rn, rn["s"])
    failures = []
    for name in ("p", "g_xn", "g_W"):
        distance = rel_rms(rz[name], getattr(emu_rn, name))
        _record("R4_MUTATION", row="kernel_rz_build", quantity=name, rz_vs_emu_rn=distance)
        if not distance > _EMU_BARS[name]:
            failures.append(f"kernel_rz_build/{name}: the rz build is within the bar of emu_RN")
    return failures


def _gp3_short_split_row_failures():
    """GP-3 launched test-only with SPLIT = n_chain(K) - 1 (the last chain's partial left out)."""
    T, K = _PRODUCTION
    x, W, _ = _pair_inputs(T, K)
    _, s = _fwd(x, W)
    split = {}

    def short_split(launch):
        split["shipped"] = launch.constexprs["SPLIT"]
        return launch._replace(constexprs={**launch.constexprs, "SPLIT": split["shipped"] - 1})

    p_short, s_short, *_ = _forward_launch_edit(x, W, {_SHIPPED["gp3"]: short_split})
    plain = _emulate(x, W, None, s)
    dropped = _emulate(x, W, None, s, mutation="drop_kernel_split")
    distance = rel_rms(p_short, plain.p)
    _record(
        "R4_MUTATION",
        row="kernel_gp3_short_split",
        split=split["shipped"],
        short_vs_emu=distance,
        short_vs_drop_kernel_split=rel_rms(p_short, dropped.p),
    )
    failures = []
    if split["shipped"] != sum(count for count, _ in _GP2_CHAINS[K]):
        failures.append(f"kernel_gp3_short_split: GP-3's SPLIT is {split['shipped']}")
    if not _same_bits(s_short, s):
        failures.append("kernel_gp3_short_split: GP-1's s differs between calls")
    if not distance > _EMU_BARS["p"]:
        failures.append("kernel_gp3_short_split: the one-chain-short p is within the p bar")
    return failures


@_NEEDS_CUDA_TRITON
@pytest.mark.parametrize("row", _MUTATION_ROWS)
def test_glm_pair_mutations_are_detected(glm_pair_gpu, tf32_matmuls, row):
    """R4: every mutation row S8 covers fails its mutated specification on the GPU outputs (rows
    not covered are recorded, not asserted), and the two compiled kernel-side mutants."""
    _assert_tf32_state()
    if row in _S8_COVERED:
        failures = _emulator_row_failures(row)
    else:
        failures = {
            "eps_outside_sqrt": _eps_row_failures,
            "drop_radial": _radial_row_failures,
            "sinkhorn_t0_like_others": _sinkhorn_row_failures,
            "kernel_rz_build": _rz_build_row_failures,
            "kernel_gp3_short_split": _gp3_short_split_row_failures,
        }[row]()
    assert not failures, "\n".join(failures)


def _spy_backward(monkeypatch):
    """Record the kernels each call of the module-level backward launcher launches."""
    original, calls = glm._glm_proj_bwd, []

    def spy(*args, **kwargs):
        compiled = kwargs.setdefault("_compiled", {})
        result = original(*args, **kwargs)
        calls.append(set(compiled))
        return result

    monkeypatch.setattr(glm, "_glm_proj_bwd", spy)
    return calls


def _process_mode():
    mode, reason = glm._matmul_mode()
    assert mode is not None, reason
    return mode


@_NEEDS_CUDA_TRITON
@pytest.mark.parametrize("used", ["both", "projection", "norm"])
def test_glm_pair_drop_in_contract(glm_pair_gpu, monkeypatch, used):
    """The drop-in contract (R6) at T 16 x K 16384 and T 8 x K 512: G_p None returns no weight
    gradient, exactly as today's path, and launches no GP-4r; G_s enters as the emulator's
    + G_s; each case against the emulator (bitwise epilogue) and float64 (R2's bars)."""
    mode = _process_mode()
    strict = "_strict" if mode == "ieee" else ""
    calls = _spy_backward(monkeypatch)
    failures = []
    for T, K in ((16, 16384), (8, 512)):
        x, W, G_p, G_s = _prc_draws(T, K)
        G_p = G_p if used in ("both", "projection") else None
        G_s = G_s if used in ("both", "norm") else None
        calls.clear()
        _, _, _, g_W = _through_function(x, W, G_p, G_s)
        unfused = _unfused_pair(x, W, G_p, G_s)
        expected = {_SHIPPED["gp4p"], _SHIPPED["gp4" + strict]}
        if G_p is not None:
            expected.add(_SHIPPED["gp4r" + strict])
        assert calls == [expected], (T, K, calls)
        assert (g_W is None) == (unfused["g_W"] is None) == (used == "norm"), (T, K)
        failures += _r2_failures(f"{T}x{K}-{used}", x, W, G_p, G_s, mode)
    assert not failures, "\n".join(failures)


def _norm_only_zero_sign_failures():
    """Norm only (G_p None), at (17, 384) and production, in the process's mode: the bf16 g_x
    bytes equal the emulator's, which computes 0 - RN(coef * xf) (EMU O8 with product +0).

    Two inputs: zero_rows with every 5th column -0.0 (every element whose x is zero gets +0, not
    -(coef * xf)); and randn x with G_s scaled so that coef = s^3 / K * G_s lies near the
    smallest fp32 subnormals, where 0 < coef * xf <= 2^-150 on some elements: RN(coef * xf) is +0
    there and 0 - (+0) is +0, while a contracted RN(-coef * xf + 0) would give -0."""
    failures = []
    for T, K in ((17, 384), _PRODUCTION):
        x_zero, W, _ = _pair_inputs(T, K, "zero_rows")
        x_zero[:, ::5] = -0.0
        x_randn, _, _ = _pair_inputs(T, K)
        cases = (
            ("zeros", x_zero, _grad_s(T)),
            ("underflow", x_randn, _grad_s(T) * (K * 2.0**-147)),
        )
        for case, x, G_s in cases:
            label = f"norm-only {T}x{K}-{case}"
            p, s, g_x, g_W = _through_function(x, W, None, G_s)
            if g_W is not None:
                failures.append(f"{label}: a weight gradient without G_p")
            model = _emulate(x, W, None, s, G_s=G_s, p=p)
            if not _same_bits(g_x, model.g_x):
                differ = int((_bits(g_x) != _bits(model.g_x)).sum())
                failures.append(f"{label}: {differ} bf16 g_x elements differ from the emulator's")
            if case == "zeros":
                zeros = x == 0
                if not (zeros.any() and (_bits(g_x)[zeros] == 0).all()):
                    failures.append(f"{label}: an element with x == 0 is not +0")
                continue
            # The case's power: O7's coef (on CPU) times xf reaches (0, 2^-150] somewhere.
            s_cpu = s.detach().reshape(-1).cpu()
            inv_k = torch.tensor(1.0 / K, dtype=torch.float32)
            coef = s_cpu * s_cpu * s_cpu * inv_k * G_s.cpu()
            product = coef.double()[:, None] * x.cpu().double()
            reached = int(((product > 0) & (product <= 2.0**-150)).sum())
            _record("NORM_ONLY_UNDERFLOW", case=label, elements=reached)
            if reached == 0:
                failures.append(f"{label}: no element has 0 < coef * xf <= 2^-150")
    return failures


@_NEEDS_CUDA_TRITON
def test_glm_pair_norm_only_zero_signs(glm_pair_gpu):
    """Norm only (G_p None): the bf16 g_x bytes equal the emulator's, and every element whose x
    is zero (+0 or -0) gets +0, as 0.0 - coef * xf gives (not -(coef * xf)); also where coef * xf
    underflows to zero (strict mode: in its subprocess)."""
    failures = _norm_only_zero_sign_failures()
    assert not failures, "\n".join(failures)


@_NEEDS_CUDA_TRITON
@pytest.mark.parametrize("shape", _M3_SHAPES, ids=_shape_id)
def test_glm_pair_debug_build_equals_production(glm_pair_gpu, shape):
    """The debug builds equal the production builds on every production output, bitwise."""
    T, K = shape
    x, W, G_p = _pair_inputs(T, K)
    p, s = _fwd(x, W)
    p_dbg, s_dbg = _fwd(x, W, debug=True)
    assert _same_bits(p, p_dbg) and _same_bits(s, s_dbg)
    for G_s in (None, _grad_s(T)):
        g_x, g_W, extras = _bwd(x, W, s, p, G_p, G_s)
        g_x_dbg, g_W_dbg, debug = _bwd(x, W, s, p, G_p, G_s, debug=True)
        assert extras == {}
        assert set(debug) == {"g_xn", "g_x32", "g_s"}
        assert _same_bits(g_x, g_x_dbg) and _same_bits(g_W, g_W_dbg)
        assert _same_bits(g_x, debug["g_x32"].to(torch.bfloat16))


@_NEEDS_CUDA_TRITON
def test_glm_pair_masked_equals_even(glm_pair_gpu):
    """The masked builds of GP-1, GP-2 and GP-3, forced at 4096 x 16384, write the bytes the EVEN
    builds write: p, s, W_r and the chain partials."""
    T, K = _PRODUCTION
    x, W, _ = _pair_inputs(T, K)
    flags = {_SHIPPED["gp1"]: "EVEN", _SHIPPED["gp2"]: "EVEN_M", _SHIPPED["gp3"]: "EVEN"}
    seen = {}

    def masked(launch):
        flag = flags[launch.name]
        seen[launch.name] = launch.constexprs[flag]
        return launch._replace(constexprs={**launch.constexprs, flag: 0})

    even = _forward_launch_edit(x, W, {}, zero_workspace=True)
    forced = _forward_launch_edit(x, W, dict.fromkeys(flags, masked), zero_workspace=True)
    assert seen == dict.fromkeys(flags, 1)
    p, s = _fwd(x, W)
    assert _same_bits(even[0], p) and _same_bits(even[1], s)
    layout = even[3]
    assert _same_bits(forced[0], even[0]) and _same_bits(forced[1], even[1])
    for name in ("wr", "part"):
        offset, length = getattr(layout, name)
        assert _same_bits(forced[2].narrow(0, offset, length), even[2].narrow(0, offset, length))


@_NEEDS_CUDA_TRITON
@pytest.mark.parametrize("shape", _M3_SHAPES, ids=_shape_id)
def test_glm_pair_det_fill_outputs_unchanged(glm_pair_gpu, shape):
    """R8 (d) in TF32 mode: under det-fill (every torch.empty starts NaN) the forward and every
    backward build are finite and bitwise equal to a run without it; each workspace is at least
    its layout (S11); the words left unwritten are recorded. Strict mode: in its subprocess."""
    failures = _det_fill_failures(*shape, "tf32")
    assert not failures, "\n".join(failures)


@_NEEDS_CUDA_TRITON
def test_glm_pair_two_streams(glm_pair_gpu):
    """R8 (e): two calls on two streams, each with its own workspace and TMA scratch, in flight
    without a sync between them, equal the same calls made one after the other."""
    T, K = _PRODUCTION
    inputs = [_pair_inputs(T, K, seed=seed) for seed in (0, 1)]

    def sequential(x, W, G_p):
        p, s = _fwd(x, W)
        g_x, g_W, _ = _bwd(x, W, s, p, G_p, None)
        return p, s, g_x, g_W

    expected = [sequential(*case) for case in inputs]
    current = torch.cuda.current_stream()
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    for stream in streams:
        stream.wait_stream(current)
    forwards, results = [], []
    for stream, (x, W, _) in zip(streams, inputs):
        with torch.cuda.stream(stream):
            forwards.append(_fwd(x, W))
    for stream, (x, W, G_p), (p, s) in zip(streams, inputs, forwards):
        with torch.cuda.stream(stream):
            g_x, g_W, _ = _bwd(x, W, s, p, G_p, None)
        results.append((p, s, g_x, g_W))
    for stream in streams:
        current.wait_stream(stream)
    for case, (got, want) in enumerate(zip(results, expected)):
        for name, a, b in zip(("p", "s", "g_x", "g_W"), got, want):
            assert _same_bits(a, b), (case, name)
    assert not _same_bits(results[0][0], results[1][0])


@_NEEDS_CUDA_TRITON
def test_glm_pair_saved_tensors_and_peak_memory(glm_pair_gpu, monkeypatch):
    """R6 at production shape: the forward saves exactly x2d (the input itself), W, s and p
    through ctx.save_for_backward; forward and backward peak increments stay below an fp32 copy
    of x (the gate); the tighter thresholds are recorded."""
    T, K = _PRODUCTION
    x, W, G_p, _ = _prc_draws(T, K)
    op = glm.glm_proj_rms_op("rn")
    contexts = []
    original = torch.autograd.function.FunctionCtx.save_for_backward

    def counting(ctx, *tensors):
        contexts.append(len(tensors))
        return original(ctx, *tensors)

    monkeypatch.setattr(torch.autograd.function.FunctionCtx, "save_for_backward", counting)
    saved = []

    def pack(tensor):
        saved.append((tensor.dtype, tuple(tensor.shape), tensor.data_ptr()))
        return tensor

    x_leaf, w_leaf = x.clone().requires_grad_(), W.clone().requires_grad_()
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        p, s = op(x_leaf, w_leaf, _PAIR_EPS)
    torch.autograd.backward((p,), (G_p,))
    assert contexts == [4]
    assert sorted((str(dtype), shape) for dtype, shape, _ in saved) == sorted(
        [
            (str(torch.bfloat16), (T, K)),
            (str(torch.float32), (glm.PROJ_N, K)),
            (str(torch.float32), (T, 1)),
            (str(torch.float32), (T, glm.PROJ_N)),
        ]
    )
    assert [ptr for dtype, _, ptr in saved if dtype == torch.bfloat16] == [x_leaf.data_ptr()]

    fp32_copy = T * K * 4
    x_leaf, w_leaf = x.clone().requires_grad_(), W.clone().requires_grad_()
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    p, s = op(x_leaf, w_leaf, _PAIR_EPS)
    torch.cuda.synchronize()
    forward_peak = torch.cuda.max_memory_allocated() - before
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    torch.autograd.backward((p,), (G_p,))
    torch.cuda.synchronize()
    backward_peak = torch.cuda.max_memory_allocated() - before
    _record(
        "R6_PEAK",
        forward=forward_peak,
        backward=backward_peak,
        gate=fp32_copy,
        forward_tight=fp32_copy // 2,
        backward_tight=T * K * 2 + fp32_copy // 2,
    )
    assert forward_peak < fp32_copy and backward_peak < fp32_copy


@_NEEDS_CUDA_TRITON
def test_glm_pair_autocast_insensitive(glm_pair_gpu):
    """R6: the op called under an outer bf16 autocast gives fp32 p and s and the bytes, gradients
    included, of the call outside it."""
    op = glm.glm_proj_rms_op("rn")
    for T, K in (_PRODUCTION, (17, 384)):
        x, W, G_p = _pair_inputs(T, K)
        G_s = _grad_s(T)

        def run(autocast):
            x_leaf, w_leaf = x.clone().requires_grad_(), W.clone().requires_grad_()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
                p, s = op(x_leaf, w_leaf, _PAIR_EPS)
            torch.autograd.backward(*_used(p, s, G_p, G_s))
            return p.detach(), s.detach(), x_leaf.grad, w_leaf.grad

        outside, inside = run(False), run(True)
        assert inside[0].dtype == inside[1].dtype == torch.float32
        for name, a, b in zip(("p", "s", "g_x", "g_W"), inside, outside):
            assert _same_bits(a, b), (T, K, name)


@_NEEDS_CUDA_TRITON
def test_glm_pair_no_host_sync(glm_pair_gpu):
    """R6: supported(), the forward and the backward at production shape synchronize nothing
    (torch's sync debug mode "error"); the kernels are compiled by a call before it."""
    T, K = _PRODUCTION
    x, W, G_p, _ = _prc_draws(T, K)
    op = glm.glm_proj_rms_op("rn")
    leaves = [(x.clone().requires_grad_(), W.clone().requires_grad_()) for _ in range(2)]
    p, _ = op(*leaves[0], _PAIR_EPS)
    torch.autograd.backward((p,), (G_p,))
    torch.cuda.synchronize()
    previous = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        assert op.supported(*leaves[1])
        p, _ = op(*leaves[1], _PAIR_EPS)
        torch.autograd.backward((p,), (G_p,))
    finally:
        torch.cuda.set_sync_debug_mode(previous)
    torch.cuda.synchronize()
    assert _same_bits(leaves[0][0].grad, leaves[1][0].grad)


@_NEEDS_CUDA_TRITON
@pytest.mark.parametrize("shape", _M3_SHAPES + _FIXTURE_SHAPES, ids=_shape_id)
@pytest.mark.parametrize("rounding", ["rn", "rz"])
def test_glm_pair_kernels_lower_as_registered(glm_pair_gpu, rounding, shape):
    """R5 in TF32 mode, read from the compiled handles of the launches themselves: the MMA family,
    instruction shapes and K chain (k-step 8, each chain's accumulator from zero), no fma, no
    trap, sm_103a, enable_fp_fusion off, the rounding instructions, the PDL instructions, the
    packed-contraction census, and the frozen configs and constexprs, for every kernel and build
    the shape reaches: the 15 M3 shapes and the fixture shapes (GP-3's SPLIT 8 and 2, NB = NU =
    NPROG = 1 at 16 x 128)."""
    T, K = shape
    x, W, G_p = _pair_inputs(T, K)
    failures = _lowering_failures(x, W, G_p, _grad_s(T), rounding, "tf32")
    assert not failures, "\n".join(failures)


def _strict_report():
    """The strict-mode checks, run in a process whose precision state is IEEE: R2's strict bars,
    R8 (d)'s strict det-fill cases, the strict lowering, the norm-only byte test, and a rerun
    bitwise."""
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    precision = torch.backends.cuda.matmul.fp32_precision
    legacy = torch.get_float32_matmul_precision()
    if precision not in ("none", "ieee") or legacy != "highest":
        return {
            "failures": [
                f"R2 precondition: strict mode needs cuda.matmul.fp32_precision 'none' or 'ieee' "
                f"and 'highest', got {precision!r} and {legacy!r}"
            ]
        }
    failures = []
    if glm._matmul_mode() != ("ieee", None):
        failures.append(f"R2 precondition: the pair reads {glm._matmul_mode()}")
    for T, K in (_PRODUCTION, (17, 384)):
        for rows, scale in _R2_INPUTS:
            x, W, G_p = _pair_inputs(T, K, rows, scale)
            failures += _r2_failures(f"{T}x{K}-{rows}-x{scale:g}", x, W, G_p, None, "ieee")
        x, W, G_p = _pair_inputs(T, K)
        G_s = _grad_s(T)
        failures += _r2_failures(f"{T}x{K}-with_gs", x, W, G_p, G_s, "ieee")
        failures += _r2_failures(f"{T}x{K}-norm_only", x, W, None, G_s, "ieee")
        failures += _det_fill_failures(T, K, "ieee")
        failures += _lowering_failures(x, W, G_p, G_s, "rn", "ieee")
        if (T, K) == _PRODUCTION:  # the norm-only byte test covers (17, 384) and production
            failures += _norm_only_zero_sign_failures()
        first, _ = _all_outputs(x, W, G_p, G_s, "ieee")
        again, _ = _all_outputs(x, W, G_p, G_s, "ieee")
        for name in first:
            if not _same_bits(first[name], again[name]):
                failures.append(f"strict {T}x{K}: {name} differs on a rerun")
    return {"failures": failures}


@_NEEDS_CUDA_TRITON
def test_glm_pair_strict_mode_in_subprocess(glm_pair_gpu, tmp_path):
    """Strict mode (fp32 matmuls IEEE: TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0, NVIDIA_TF32_OVERRIDE
    unset) in a fresh process, since no test sets the precision API in process: mode "ieee",
    R2's strict bars at 4096 x 16384 and 17 x 384, R8 (d)'s strict cases, the strict lowering
    (the strict GP-4's FMA dots recorded) and a rerun bitwise."""
    env = {key: value for key, value in os.environ.items() if key != "NVIDIA_TF32_OVERRIDE"}
    env["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"
    report = _run_in_subprocess("_strict_report", env, tmp_path, 3600)
    assert report["failures"] == [], "\n".join(report["failures"])


# --- R7: module-level float64 non-inferiority (through C6's binding) ------------------------

# R7's arms (per-op operations under the explicit triton policy), each against the unfused module
# with the same state dict; its cases: T at K = n x hidden = 16384, pair_data x (seed 0) at scales
# 1 and 1e-3. R12: the modules are built after torch.manual_seed(42), and the gradients of h_pre,
# h_post and H_res are randn from seed 11; the GLM_PAIR_R7 lines record the sha256_16 of each
# generated input (x, W and the three gradients).
_R7_ARMS = {"proj_rms": ["proj_rms"], "proj_rms_sinkhorn": ["proj_rms", "sinkhorn"]}
_R7_HIDDEN = 4096
_R7_TOKENS = (4096, 17, 21, 112, 4095)
_R7_SCALES = (1.0, 1e-3)


def _r7_module(fused_ops):
    """R7's HyperConnectionModule: hidden 4096, n 4, the GLM norm (eps 1e-5 inside the sqrt) with
    fp32 pre and fp32 mappings, gating factor 1.0, 20 Sinkhorn iterations. fused_ops are bound
    per op under the explicit triton policy; None builds the unfused module."""
    from megatron.core.transformer.hyper_connection import HyperConnectionModule
    from megatron.core.transformer.transformer_config import TransformerConfig

    fused = {}
    if fused_ops is not None:
        fused = dict(use_fused_mhc=True, mhc_fused_ops=list(fused_ops), mhc_fused_backend="triton")
    config = TransformerConfig(
        num_layers=1,
        hidden_size=_R7_HIDDEN,
        num_attention_heads=32,
        enable_hyper_connections=True,
        num_residual_streams=N,
        mhc_sinkhorn_iterations=ITERS,
        mhc_init_gating_factor=1.0,
        layernorm_epsilon=_PAIR_EPS,
        mhc_norm_eps_inside_sqrt=True,
        mhc_normalize_before_projection=True,
        mhc_keep_pre_in_fp32=True,
        mhc_keep_mappings_in_fp32=True,
        **fused,
    )
    return HyperConnectionModule(config, layer_number=1).to(_DEVICE)


def _r7_groups(module):
    """(group, its columns of the projection, its alpha) for pre, post and res."""
    return (
        ("pre", slice(0, N), module.alpha_pre),
        ("post", slice(N, 2 * N), module.alpha_post),
        ("res", slice(2 * N, None), module.alpha_res),
    )


def _r7_grads(T):
    """G_pre and G_post [T, n] and G_res [T, n, n]: randn from seed 11, in that order."""
    generator = torch.Generator().manual_seed(11)
    return tuple(
        torch.randn(shape, generator=generator).to(_DEVICE) for shape in ((T, N), (T, N), (T, N, N))
    )


def _sha256_16(tensor):
    """R12's input fingerprint (P6's practice): the first 16 hex digits of the sha256 of the
    tensor's contiguous CPU bytes."""
    import hashlib

    data = tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(data).hexdigest()[:16]


def _r7_run(module, x, grads):
    """compute_mappings on x [T, K] (as [T, 1, K]) and its backward for grads.

    A wrapper around _projection_and_get_norm captures proj, the pre-alpha projection that
    _compute_h consumes, and through a hook its gradient. The contribution tensor of group g is
    (grad_proj / alpha_g) * proj over the group's columns, in float64 from the fp32 captures; its
    sum is dalpha_g. Returns R7's quantities and the three dalpha scalars of the module.
    """
    T, K = x.shape
    captured = {}
    projection_and_get_norm = module._projection_and_get_norm

    def capture(x_in):
        proj, r = projection_and_get_norm(x_in)

        def keep(grad):
            captured["grad"] = grad

        captured["proj"] = proj
        proj.register_hook(keep)
        return proj, r

    module.zero_grad(set_to_none=True)
    module._projection_and_get_norm = capture
    try:
        x_leaf = x.detach().view(T, 1, K).clone().requires_grad_()
        outputs = module.compute_mappings(x_leaf)
        torch.autograd.backward(outputs, [g.view(o.shape) for g, o in zip(grads, outputs)])
    finally:
        del module._projection_and_get_norm
    h_pre, h_post, h_res = (output.detach() for output in outputs)
    proj = captured["proj"].detach().reshape(T, -1).double()
    grad_proj = captured["grad"].reshape(T, -1).double()
    quantities = {
        "h_pre": h_pre.reshape(T, N),
        "h_post": h_post.reshape(T, N),
        "comb": h_res.reshape(T, N, N),
        "dbias": module.bias.grad.detach().clone(),
    }
    dalpha = {}
    for group, columns, alpha in _r7_groups(module):
        quantities[f"c_{group}"] = (
            grad_proj[:, columns] / alpha.detach().double() * proj[:, columns]
        )
        dalpha[group] = alpha.grad.item()
    return quantities, dalpha


def _r7_reference(x, module, grads):
    """R7's float64 reference: EMU.glm_mappings_chain_f64 on the module's parameters, with the
    contribution tensors G_h * p over each group's columns."""
    reference = emu.glm_mappings_chain_f64(
        x,
        module.mapping_proj.weight.detach(),
        tuple(alpha.detach() for _, _, alpha in _r7_groups(module)),
        module.bias.detach(),
        grads,
        n=N,
        iters=ITERS,
        eps=emu.EPS_FP32,
    )
    weighted = reference["G_h"] * reference["p"]
    quantities = {
        "h_pre": reference["h_pre"],
        "h_post": reference["h_post"],
        "comb": reference["H_res"],
        "dbias": reference["d_bias"],
    }
    for group, columns, _ in _r7_groups(module):
        quantities[f"c_{group}"] = weighted[:, columns]
    dalpha = {group: reference[f"d_alpha_{group}"].item() for group, _, _ in _r7_groups(module)}
    return quantities, dalpha


@_NEEDS_CUDA_TRITON
@pytest.mark.parametrize("arm", list(_R7_ARMS))
def test_glm_pair_module_float64_noninferiority(glm_pair_gpu, tf32_matmuls, monkeypatch, arm):
    """R7 in TF32 mode, through HyperConnectionModule: the arm's module against the unfused module
    with the same state dict, both against EMU.glm_mappings_chain_f64, at T in {4096, 17, 21, 112,
    4095} x K 16384 and x scales 1 and 1e-3. h_pre, h_post, H_res, dbias and the contribution
    tensors of the three dalpha each stay within 1.1 x the unfused rel-RMS; the dalpha scalars,
    their ratios and |dalpha - sum(contribution)| are recorded, and so are the inputs' sha256_16.
    The spy shows the bound pair ran with no refusal (and, in the sinkhorn arm, the fixed-launch
    Sinkhorn ran), and the pair's backward launcher ran once per backward."""
    from tests.unit_tests.transformer.mhc_fused_op_spy import spy_fused_mhc_ops

    _assert_tf32_state()
    torch.manual_seed(42)
    unfused = _r7_module(None)
    fused = _r7_module(_R7_ARMS[arm])
    fused.load_state_dict(unfused.state_dict())
    backward_calls = _spy_backward(monkeypatch)
    K = N * _R7_HIDDEN
    weight_sha256_16 = _sha256_16(unfused.mapping_proj.weight)
    failures, cases = [], 0
    with spy_fused_mhc_ops(fused) as spy:
        for T in _R7_TOKENS:
            for scale in _R7_SCALES:
                label = f"{arm}-{T}x{K}-x{scale:g}"
                x = emu.pair_data(T, K, scale, seed=0)[0].to(_DEVICE)
                grads = _r7_grads(T)
                fingerprints = dict(x_sha256_16=_sha256_16(x), W_sha256_16=weight_sha256_16)
                for name, grad in zip(("G_pre", "G_post", "G_res"), grads):
                    fingerprints[f"{name}_sha256_16"] = _sha256_16(grad)
                backward_calls.clear()
                ours, ours_dalpha = _r7_run(fused, x, grads)
                cases += 1
                if len(backward_calls) != 1:
                    failures.append(f"{label}: the pair's backward ran {len(backward_calls)} times")
                theirs, theirs_dalpha = _r7_run(unfused, x, grads)
                reference, reference_dalpha = _r7_reference(x, unfused, grads)
                for quantity, expected in reference.items():
                    error = rel_rms(ours[quantity], expected)
                    unfused_error = rel_rms(theirs[quantity], expected)
                    bar = _RATIO * unfused_error
                    _record(
                        "R7",
                        case=label,
                        quantity=quantity,
                        fused=error,
                        unfused=unfused_error,
                        ratio=error / unfused_error if unfused_error else float("inf"),
                        bar=bar,
                        **fingerprints,
                    )
                    if not error <= bar:
                        failures.append(
                            f"{label}: {quantity} rel-RMS {error:.3e} > {bar:.3e} (1.1 x unfused)"
                        )
                for group in ("pre", "post", "res"):
                    fused_error = abs(ours_dalpha[group] - reference_dalpha[group])
                    unfused_error = abs(theirs_dalpha[group] - reference_dalpha[group])
                    _record(
                        "R7_DALPHA",
                        case=label,
                        group=group,
                        fused=ours_dalpha[group],
                        unfused=theirs_dalpha[group],
                        reference=reference_dalpha[group],
                        ratio=fused_error / unfused_error if unfused_error else float("inf"),
                        fused_sum_gap=abs(ours_dalpha[group] - ours[f"c_{group}"].sum().item()),
                        unfused_sum_gap=abs(
                            theirs_dalpha[group] - theirs[f"c_{group}"].sum().item()
                        ),
                    )
    spy.assert_ran("_glm_proj_rms_op", glm.glm_proj_rms_op("rn"), min_calls=cases)
    assert spy.calls("_glm_proj_rms_op") == cases
    spy.assert_no_unsupported()
    if "sinkhorn" in _R7_ARMS[arm]:
        spy.assert_ran("_sinkhorn_op", glm.fixed_launch_sinkhorn, min_calls=cases)
        assert spy.calls("_sinkhorn_op") == cases
    assert not failures, "\n".join(failures)
