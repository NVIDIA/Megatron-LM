# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Fused Sinkhorn optimizer step (``MCORE_SINKHORN_FUSED``) against Emerging-Optimizers' step.

Every comparison runs ``Sinkhorn.step`` from the same state twice, with the switch off (EO's
step, the reference) and on, for several consecutive steps with evolving momentum:

* the momentum buffers must be bit-identical (the same lerp roundings);
* the balanced update is compared exactly: with lr = 1 and the parameter reset to zero before
  each step, both paths leave -update in the parameter. Every update element is the Nesterov
  value times 2K + 1 normalization factors, a chain of correctly rounded divisions in the
  reference and two multiplies by accumulated factors in the fused path, with the norms summed
  in different orders. Nothing cancels, so every element agrees to a few ulps: at most 13 ulps
  (8.5e-7 relative) on GB300 from 37 x 7 to 2M x 256 and 6000 x 5120 with K = 0, 1 and 5.
  ``UPDATE_RTOL`` allows about 5x that, and exact zeros (masked rows) must stay zero. Every
  input keeps its masked rows far below the masking threshold: a row within a few ulps of it
  may be masked by one path only (see ``sinkhorn_fused``);
* parameters of order 1 (lr = 0.02) differ by at most one rounding of p per step.
"""

import contextlib
import importlib
import math

import pytest
import torch
import torch.distributed as dist

from megatron.core.optimizer import emerging_optimizers as eo_mod
from megatron.core.optimizer import sinkhorn_fused
from tests.unit_tests.test_utilities import Utils

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU"),
    pytest.mark.skipif(not sinkhorn_fused.HAVE_TRITON, reason="needs Triton"),
    pytest.mark.skipif(
        not eo_mod.HAVE_EMERGING_OPTIMIZERS, reason="emerging_optimizers is not installed"
    ),
]

UPDATE_RTOL = 4e-6
PARAM_TOL = dict(rtol=1e-6, atol=1e-6)


@pytest.fixture(autouse=True)
def _needs_sinkhorn():
    pytest.importorskip("emerging_optimizers.orthogonalized_optimizers.sinkhorn_utils")


@contextlib.contextmanager
def _switch(on: bool):
    saved = sinkhorn_fused.SINKHORN_FUSED
    sinkhorn_fused.SINKHORN_FUSED = on
    try:
        yield
    finally:
        sinkhorn_fused.SINKHORN_FUSED = saved


@pytest.fixture
def fused_calls(monkeypatch):
    """Shapes of the parameters the fused kernels updated."""
    calls = []
    original = sinkhorn_fused.fused_sinkhorn_step

    def spy(param, *args, **kwargs):
        calls.append(tuple(param.shape))
        return original(param, *args, **kwargs)

    monkeypatch.setattr(sinkhorn_fused, "fused_sinkhorn_step", spy)
    return calls


def _bitwise_equal(a, b):
    if a.dtype == torch.float32:
        return torch.equal(a.view(torch.int32), b.view(torch.int32))
    return torch.equal(a.view(torch.int16), b.view(torch.int16))


def _run(fused, make_params, grads, *, lr, zero_params=False, optimizer_factory=None, **kwargs):
    """Steps of a fresh optimizer; returns per step the parameters and momenta (copies).

    ``grads`` holds one list of gradients (or None) per step, one per parameter.
    """
    with _switch(fused):
        params = make_params()
        factory = optimizer_factory or (lambda ps, lr: eo_mod.Sinkhorn(ps, lr=lr, **kwargs))
        optimizer = factory(params, lr)
        history = []
        for step_grads in grads:
            for param, grad in zip(params, step_grads):
                if zero_params:
                    param.data.zero_()
                param.grad = None if grad is None else grad.clone()
            optimizer.step()
            history.append(
                [
                    (
                        param.detach().clone(),
                        (
                            optimizer.state[param]["momentum_buffer"].clone()
                            if "momentum_buffer" in optimizer.state[param]
                            else None
                        ),
                    )
                    for param in params
                ]
            )
        return history


def _assert_update_close(fused, reference):
    torch.testing.assert_close(fused, reference, rtol=UPDATE_RTOL, atol=0)


def _compare(make_params, grads, *, check_params=True, **kwargs):
    """Fused vs reference: momentum bit-identical, updates and parameters close."""
    reference = _run(False, make_params, grads, lr=1.0, zero_params=True, **kwargs)
    fused = _run(True, make_params, grads, lr=1.0, zero_params=True, **kwargs)
    for ref_step, fused_step in zip(reference, fused):
        for (ref_update, ref_momentum), (fused_update, fused_momentum) in zip(ref_step, fused_step):
            assert (ref_momentum is None) == (fused_momentum is None)
            if ref_momentum is not None:
                assert _bitwise_equal(ref_momentum, fused_momentum)
            _assert_update_close(fused_update, ref_update)
    if check_params:
        reference = _run(False, make_params, grads, lr=0.02, **kwargs)
        fused = _run(True, make_params, grads, lr=0.02, **kwargs)
        for ref_step, fused_step in zip(reference, fused):
            for (ref_param, ref_momentum), (fused_param, fused_momentum) in zip(
                ref_step, fused_step
            ):
                if ref_momentum is not None:
                    assert _bitwise_equal(ref_momentum, fused_momentum)
                torch.testing.assert_close(fused_param, ref_param, **PARAM_TOL)
    return reference, fused


def _matrix(rows, cols, local_rows=None, seed=0):
    def make():
        torch.manual_seed(seed)
        param = torch.nn.Parameter(torch.randn(rows, cols, device="cuda"))
        if local_rows is not None:
            param.row_shard_local_rows = local_rows
        return [param]

    return make


def _grads(rows, cols, steps=4, seed=1, local_rows=None):
    """Gradients with masked rows (zero, and 1e-7 below the 1e-3 threshold) and poisoned
    padding rows, whose values must not enter the statistics."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    out = []
    for _ in range(steps):
        grad = torch.randn(rows, cols, device="cuda", generator=generator)
        if rows > 4:
            grad[0].zero_()
            grad[1].mul_(1e-7)
        if local_rows is not None and local_rows < rows:
            grad[local_rows:] = 1e5
        out.append([grad])
    return out


@pytest.mark.parametrize("steps", [1, 3, 11])
@pytest.mark.parametrize("shape", [(37, 7), (5, 5), (1, 1), (1000, 256), (600, 600), (5200, 5120)])
def test_whole_matrix_matches_reference(shape, steps, fused_calls):
    _compare(_matrix(*shape), _grads(*shape), num_normalization_steps=steps)
    assert fused_calls and set(fused_calls) == {shape}


@pytest.mark.parametrize("steps", [1, 3, 11])
@pytest.mark.parametrize(
    "shape,local_rows", [((1000, 256), 997), ((40, 8), 9), ((4097, 256), 4096), ((310, 300), 300)]
)
def test_padded_row_shard_matches_reference(shape, local_rows, steps, fused_calls):
    """Row-sharded metadata without a group: padding rows keep p and get the momentum."""
    reference, _ = _compare(
        _matrix(*shape, local_rows=local_rows),
        _grads(*shape, local_rows=local_rows),
        num_normalization_steps=steps,
    )
    torch.manual_seed(0)
    initial = torch.randn(*shape, device="cuda")
    assert torch.equal(reference[-1][0][0][local_rows:], initial[local_rows:])
    assert fused_calls


def _with_normalization_steps(steps):
    """A Sinkhorn factory that sets ``num_normalization_steps`` after construction, which
    rejects values below 1 and even ones."""

    def factory(params, lr):
        optimizer = eo_mod.Sinkhorn(params, lr=lr)
        optimizer.param_groups[0]["num_normalization_steps"] = steps
        return optimizer

    return factory


@pytest.mark.parametrize("steps", [0, -1, -4, 2, 4])
def test_padded_row_shard_with_edited_normalization_steps(steps, fused_calls):
    """Row-sharded matrices take Megatron's loop over range((steps - 1) // 2), which accepts
    any integer: below 1 step no column/row pair runs (only the first row step), and an even
    count runs (steps - 1) // 2 pairs. The fused step must not run a pair for a negative K."""
    shape, local_rows = (1000, 256), 997
    _compare(
        _matrix(*shape, local_rows=local_rows),
        _grads(*shape, local_rows=local_rows),
        optimizer_factory=_with_normalization_steps(steps),
    )
    assert fused_calls


def test_wide_shard_with_few_rows(fused_calls):
    """A 300 x 5120 shard (C > rows, as a row shard may be) through add_param_group."""
    shape, local_rows = (304, 5120), 300

    def factory(params, lr):
        optimizer = eo_mod.Sinkhorn([torch.nn.Parameter(torch.ones(1, 1))], lr=lr)
        optimizer.add_param_group({"params": params})
        return optimizer

    grads = [[None, *g] for g in _grads(*shape, local_rows=local_rows)]

    def make():
        return [torch.nn.Parameter(torch.ones(1, 1)), *_matrix(*shape, local_rows=local_rows)()]

    reference = _run(False, make, grads, lr=1.0, zero_params=True, optimizer_factory=factory)
    fused = _run(True, make, grads, lr=1.0, zero_params=True, optimizer_factory=factory)
    for ref_step, fused_step in zip(reference, fused):
        assert _bitwise_equal(ref_step[1][1], fused_step[1][1])
        _assert_update_close(fused_step[1][0], ref_step[1][0])
    assert fused_calls == [shape] * len(grads)


@pytest.mark.parametrize(
    "shape,scale,steps",
    [((1 << 20, 1), None, 21), ((1 << 20, 1), None, 23), ((1 << 20, 4), 1e-9, 25)],
    ids=["constant-21", "constant-23", "randn-25"],
)
def test_tall_matrix_with_many_normalization_steps(shape, scale, steps, fused_calls):
    """Tall matrices with small rows and many normalization steps keep their factors in range.

    Without the rebalancing, the row factors grow by sqrt(R / C) per pair (2^10 at 2^20 x 1,
    2^9 at 2^20 x 4) from about 1 / r = 2^33 (constant 2^-30 gradient) or 2^32 (randn * 1e-9),
    so they overflowed from K = 10 (21 steps) or K = 11 on, and the factor kept at its last
    finite value made those rows' updates smaller by a factor of 2^10 or 2^9 (or more).
    """
    generator = torch.Generator(device="cuda").manual_seed(6)
    grads = []
    for _ in range(2):
        if scale is None:
            grad = torch.full(shape, 2.0**-30, device="cuda")
        else:
            grad = torch.randn(shape, device="cuda", generator=generator) * scale
        grads.append([grad])
    _compare(_matrix(*shape), grads, num_normalization_steps=steps)
    assert fused_calls and set(fused_calls) == {shape}


def _degenerate_grads(kind, rows=512, cols=64):
    generator = torch.Generator(device="cuda").manual_seed(5)
    grads = []
    for step in range(4):
        grad = torch.randn(rows, cols, device="cuda", generator=generator)
        if kind == "masked_rows":
            grad[: rows // 2].mul_(1e-6)
            grad[rows // 2 : rows // 2 + 7].zero_()
        elif kind == "zero_columns":
            grad[:, 3].zero_()
            grad[:, 10:12].zero_()
        elif kind == "tiny_column":
            # Squares of this column underflow; like the reference, 1 / eps rescales it.
            grad[:, 5].mul_(1e-25)
        elif kind == "zero_rows_and_columns":
            grad[:100].zero_()
            grad[:, :4].zero_()
        elif kind == "all_zero":
            grad.zero_()
        elif kind == "tiny":
            # Squares underflow to 0: every row is masked and nothing is updated.
            grad.mul_(1e-30)
        elif kind == "small":
            grad.mul_(1e-18)
        elif kind == "zero_after_first" and step:
            grad.zero_()
        grads.append([grad])
    return grads


@pytest.mark.parametrize(
    "kind",
    [
        "masked_rows",
        "zero_columns",
        "tiny_column",
        "zero_rows_and_columns",
        "all_zero",
        "tiny",
        "small",
        "zero_after_first",
    ],
)
def test_degenerate_inputs_match_reference(kind, fused_calls):
    reference, fused = _compare(_matrix(512, 64), _degenerate_grads(kind))
    for ref_step, fused_step in zip(reference, fused):
        assert torch.isfinite(fused_step[0][0]).all()
    if kind in ("all_zero", "tiny"):
        torch.manual_seed(0)
        assert torch.equal(fused[-1][0][0], torch.randn(512, 64, device="cuda"))
    assert fused_calls


def test_nan_propagates_like_reference():
    grads = _degenerate_grads("masked_rows")
    grads[1][0][7, 3] = float("nan")
    reference = _run(False, _matrix(512, 64), grads, lr=0.02)
    fused = _run(True, _matrix(512, 64), grads, lr=0.02)
    for step, (ref_step, fused_step) in enumerate(zip(reference, fused)):
        (ref_param, ref_momentum), (fused_param, fused_momentum) = ref_step[0], fused_step[0]
        assert _bitwise_equal(ref_momentum, fused_momentum)
        assert torch.equal(ref_param.isnan(), fused_param.isnan())
        assert fused_param.isnan().all() == (step >= 1)
        if step == 0:
            torch.testing.assert_close(fused_param, ref_param, **PARAM_TOL)


def test_overflowing_row_norm_masks_every_row():
    """A finite gradient whose row norm overflows: S = inf masks every row, so the reference's
    update is exactly zero (the Nesterov value is a convex combination of finite values and
    stays finite; infinite inputs give NaN, see test_nan_propagates_like_reference)."""
    grad = torch.randn(64, 16, device="cuda")
    grad[5, 2] = 1e30
    reference = _run(False, _matrix(64, 16), [[grad]], lr=0.02)
    fused = _run(True, _matrix(64, 16), [[grad]], lr=0.02)
    nesterov = grad.lerp(reference[0][0][1], 0.95)
    assert torch.isfinite(nesterov).all() and torch.isinf(nesterov.norm(dim=1)[5])
    assert _bitwise_equal(reference[0][0][1], fused[0][0][1])
    torch.manual_seed(0)
    initial = torch.randn(64, 16, device="cuda")
    assert _bitwise_equal(reference[0][0][0], initial)
    assert _bitwise_equal(fused[0][0][0], initial)


def test_masked_rows_keep_signed_zeros(fused_calls):
    """Masked rows contribute +0 like the reference's masked_fill: -0.0 parameters stay -0.0
    even where the Nesterov value is negative (a plain G * 0 would give -0 and flip them)."""

    def make():
        param = _matrix(256, 32)()[0]
        param.data[:8] = -0.0
        param.data[8:16] = 0.0
        return [param]

    grads = _grads(256, 32, steps=3)
    for (grad,) in grads:
        grad[:16] = -1e-9 * torch.rand(16, 32, device="cuda")
    reference = _run(False, make, grads, lr=0.02)
    fused = _run(True, make, grads, lr=0.02)
    for ref_step, fused_step in zip(reference, fused):
        assert _bitwise_equal(fused_step[0][0][:16], ref_step[0][0][:16])
        assert _bitwise_equal(fused_step[0][0][:16], make()[0].detach()[:16])
    assert fused_calls


def test_momentum_below_half_and_multiple_groups(fused_calls):
    """Momentum < 0.5 takes torch.lerp's other branch; groups keep their own settings."""

    def make():
        torch.manual_seed(3)
        return [torch.nn.Parameter(torch.randn(256, 32, device="cuda")) for _ in range(3)]

    def factory(params, lr):
        return eo_mod.Sinkhorn(
            [
                {"params": params[:1], "momentum": 0.2, "num_normalization_steps": 3},
                {"params": params[1:], "eps": 1e-12, "extra_scale_factor": 0.5},
            ],
            lr=lr,
        )

    generator = torch.Generator(device="cuda").manual_seed(4)
    grads = [
        [torch.randn(256, 32, device="cuda", generator=generator) for _ in range(3)]
        for _ in range(3)
    ]
    _compare(make, grads, optimizer_factory=factory)
    assert len(fused_calls) == 2 * 3 * 3  # two runs of three steps of three parameters


class _Hook:
    def __init__(self):
        self.calls = 0

    def pre_weight_update_inplace(self, p, update):
        self.calls += 1
        return None

    def post_weight_update_inplace(self, p, state):
        p.mul_(0.5)


class _PreUpdateSinkhorn(eo_mod.Sinkhorn):
    def pre_weight_update_fn_inplace(self, p, update):
        update.mul_(0.25)


def _bf16_matrix():
    torch.manual_seed(0)
    return [torch.nn.Parameter(torch.randn(128, 32, device="cuda", dtype=torch.bfloat16))]


def _fp32_matrix():
    torch.manual_seed(0)
    return [torch.nn.Parameter(torch.randn(128, 32, device="cuda"))]


def _cpu_matrix():
    torch.manual_seed(0)
    return [torch.nn.Parameter(torch.randn(128, 32))]


def _with_weight_decay(params, lr):
    optimizer = eo_mod.Sinkhorn(params, lr=lr)
    optimizer.param_groups[0]["weight_decay"] = 0.1
    return optimizer


def _with_hook(params, lr):
    optimizer = eo_mod.Sinkhorn(params, lr=lr)
    optimizer._weight_update_hook = _Hook()
    return optimizer


def _with_tensor_lr(params, lr):
    optimizer = eo_mod.Sinkhorn(params, lr=lr)
    optimizer.param_groups[0]["lr"] = torch.tensor(lr)
    return optimizer


@pytest.mark.parametrize(
    "make,factory",
    [
        (_bf16_matrix, None),
        (_cpu_matrix, None),
        (_fp32_matrix, _with_weight_decay),
        (_fp32_matrix, _with_hook),
        (_fp32_matrix, lambda params, lr: _PreUpdateSinkhorn(params, lr=lr)),
        (_fp32_matrix, _with_tensor_lr),
    ],
    ids=["bf16", "cpu", "weight-decay", "hook", "pre-update-override", "tensor-lr"],
)
def test_ineligible_parameters_run_the_reference(make, factory, fused_calls):
    """With the switch on, ineligible parameters run EO's own step body, bit for bit."""

    def grads_for(param):
        generator = torch.Generator(device=param.device).manual_seed(9)
        return [
            [torch.randn(param.shape, generator=generator, device=param.device).to(param.dtype)]
            for _ in range(3)
        ]

    grads = grads_for(make()[0])
    reference = _run(False, make, grads, lr=0.02, optimizer_factory=factory)
    fused = _run(True, make, grads, lr=0.02, optimizer_factory=factory)
    for ref_step, fused_step in zip(reference, fused):
        assert _bitwise_equal(ref_step[0][0], fused_step[0][0])
        assert _bitwise_equal(ref_step[0][1], fused_step[0][1])
    assert not fused_calls


def test_mixed_eligibility_keeps_parameter_order(fused_calls, monkeypatch):
    """Eligible and ineligible parameters interleave; each runs at its own position."""
    order = []
    original = eo_mod.OrthogonalizedOptimizer.step

    def reference_step(self, *args, **kwargs):
        order.append(("reference", tuple(self.param_groups[0]["params"][0].shape)))
        return original(self, *args, **kwargs)

    def make():
        torch.manual_seed(2)
        return [
            torch.nn.Parameter(torch.randn(96, 16, device="cuda")),
            torch.nn.Parameter(torch.randn(80, 16, device="cuda", dtype=torch.bfloat16)),
            torch.nn.Parameter(torch.randn(64, 16, device="cuda")),
        ]

    generator = torch.Generator(device="cuda").manual_seed(4)
    grads = [
        [
            torch.randn(96, 16, device="cuda", generator=generator),
            torch.randn(80, 16, device="cuda", generator=generator).bfloat16(),
            torch.randn(64, 16, device="cuda", generator=generator),
        ]
        for _ in range(2)
    ]
    reference = _run(False, make, grads, lr=0.02)
    monkeypatch.setattr(eo_mod.OrthogonalizedOptimizer, "step", reference_step)
    original_fused = sinkhorn_fused.fused_sinkhorn_step

    def fused_step(param, *args, **kwargs):
        order.append(("fused", tuple(param.shape)))
        return original_fused(param, *args, **kwargs)

    monkeypatch.setattr(sinkhorn_fused, "fused_sinkhorn_step", fused_step)
    fused = _run(True, make, grads, lr=0.02)
    assert order == [("fused", (96, 16)), ("reference", (80, 16)), ("fused", (64, 16))] * 2
    for ref_step, fused_step in zip(reference, fused):
        torch.testing.assert_close(fused_step[0][0], ref_step[0][0], **PARAM_TOL)
        assert _bitwise_equal(fused_step[1][0], ref_step[1][0])
        torch.testing.assert_close(fused_step[2][0], ref_step[2][0], **PARAM_TOL)


def test_switch_off_is_the_reference_path(monkeypatch):
    """With the switch off, Sinkhorn.step calls EO's step and nothing of the fused path."""

    def fail(*args, **kwargs):
        raise AssertionError("fused path used with MCORE_SINKHORN_FUSED off")

    monkeypatch.setattr(sinkhorn_fused, "fused_sinkhorn_step", fail)
    monkeypatch.setattr(eo_mod.Sinkhorn, "_step_with_fused_updates", fail)
    grads = _grads(1000, 256, steps=2)
    switch_off = _run(False, _matrix(1000, 256), grads, lr=0.02)
    with _switch(False):
        params = _matrix(1000, 256)()
        optimizer = eo_mod.Sinkhorn(params, lr=0.02)
        for (grad,) in grads:
            params[0].grad = grad.clone()
            eo_mod.OrthogonalizedOptimizer.step(optimizer)
    assert _bitwise_equal(switch_off[-1][0][0], params[0].detach())


def test_switch_is_read_from_the_environment(monkeypatch):
    try:
        monkeypatch.delenv("MCORE_SINKHORN_FUSED", raising=False)
        assert importlib.reload(sinkhorn_fused).SINKHORN_FUSED is False
        monkeypatch.setenv("MCORE_SINKHORN_FUSED", "1")
        assert importlib.reload(sinkhorn_fused).SINKHORN_FUSED is True
    finally:
        monkeypatch.undo()
        importlib.reload(sinkhorn_fused)


def test_direct_call_validates_inputs():
    param = torch.zeros(8, 4, device="cuda")
    args = dict(
        lr=0.1,
        momentum=0.9,
        eps=1e-20,
        num_normalization_steps=3,
        zero_row_threshold=1e-3,
        extra_scale_factor=0.18,
        mean_rows=8,
    )
    with pytest.raises(ValueError, match="contiguous FP32 CUDA"):
        sinkhorn_fused.fused_sinkhorn_step(
            param, param.clone().bfloat16(), param.clone(), local_rows=8, **args
        )
    with pytest.raises(ValueError, match="contiguous FP32 CUDA"):
        sinkhorn_fused.fused_sinkhorn_step(param, param, param.clone(), local_rows=8, **args)
    with pytest.raises(ValueError, match="local_rows"):
        sinkhorn_fused.fused_sinkhorn_step(
            param, param.clone(), param.clone(), local_rows=9, **args
        )
    assert not sinkhorn_fused.fused_step_supported(param, param.t(), param.clone())
    assert not sinkhorn_fused.fused_step_supported(
        torch.zeros(4, 2 * sinkhorn_fused._MAX_BLOCK_C, device="cuda"),
        torch.zeros(4, 2 * sinkhorn_fused._MAX_BLOCK_C, device="cuda"),
        torch.zeros(4, 2 * sinkhorn_fused._MAX_BLOCK_C, device="cuda"),
    )


# ---------------------------------------------------------------------------- row-sharded groups

_GROUPS = {}


def _row_group(size):
    """This rank's group of ``size`` consecutive ranks (every rank creates every group)."""
    Utils.initialize_distributed()
    world = dist.get_world_size()
    if world < size or world % size:
        pytest.skip(f"needs a multiple of {size} ranks")
    if size not in _GROUPS:
        groups = [dist.new_group(list(range(i, i + size))) for i in range(0, world, size)]
        _GROUPS[size] = groups[dist.get_rank() // size]
    return _GROUPS[size]


def _shard_layout(size, empty_last, cols=24):
    """Uneven logical rows per group rank (the last one may hold none); at least ``cols``."""
    scale = 1 if cols <= 29 else 16
    rows = [(29 + 13 * q) * scale for q in range(size)]
    if empty_last:
        rows[-1] = 0
    return rows


def _sharded(group, cols, rows_per_rank, pad):
    rank = group.rank()
    local = rows_per_rank[rank]
    start = sum(rows_per_rank[:rank])

    def make():
        torch.manual_seed(100 + rank)
        param = torch.nn.Parameter(torch.randn(local + pad, cols, device="cuda"))
        param.row_shard_group = group
        param.row_shard_local_rows = local
        param.row_shard_global_rows = sum(rows_per_rank)
        param.row_shard_start = start
        return [param]

    return make, local


def _sharded_grads(group, cols, local, pad, steps, inactive):
    generator = torch.Generator(device="cuda").manual_seed(200 + group.rank())
    grads = []
    for step in range(steps):
        grad = torch.randn(local + pad, cols, device="cuda", generator=generator)
        if local > 4:
            grad[0].zero_()
            grad[1].mul_(1e-7)
            grad[2, : cols // 2].mul_(100)
        grad[local:] = 1e5
        absent = (inactive == "one" and step == 1 and group.rank() == 1) or (
            inactive == "all" and step == 2
        )
        grads.append([None if absent else grad])
    return grads


@pytest.mark.parametrize("size", [2, 4])
@pytest.mark.parametrize("pad", [0, 5])
@pytest.mark.parametrize("cols,steps", [(24, 11), (256, 3)])
@pytest.mark.parametrize("empty_last", [False, True])
def test_row_sharded_group_matches_reference(size, pad, cols, steps, empty_last, fused_calls):
    group = _row_group(size)
    rows_per_rank = _shard_layout(size, empty_last, cols)
    make, local = _sharded(group, cols, rows_per_rank, pad)
    grads = _sharded_grads(group, cols, local, pad, 4, inactive="one")
    reference, _ = _compare(make, grads, num_normalization_steps=steps)
    torch.manual_seed(100 + group.rank())
    initial = torch.randn(local + pad, cols, device="cuda")
    assert torch.equal(reference[-1][0][0][local:], initial[local:])
    assert len(fused_calls) == 2 * 4  # two runs of four steps


def test_row_sharded_all_inactive_does_not_update(fused_calls):
    group = _row_group(2)
    make, local = _sharded(group, 24, _shard_layout(2, False), 3)
    grads = _sharded_grads(group, 24, local, 3, 4, inactive="all")
    _, fused = _compare(make, grads)
    (param_2, momentum_2), (param_1, momentum_1) = fused[2][0], fused[1][0]
    assert _bitwise_equal(param_2, param_1) and _bitwise_equal(momentum_2, momentum_1)
    assert len(fused_calls) == 2 * 3  # two runs of the three active steps


def test_row_sharded_mixed_switch_across_ranks():
    """Some ranks fuse, the others run the reference: same collectives, close results."""
    group = _row_group(2)
    make, local = _sharded(group, 24, _shard_layout(2, False), 5)
    grads = _sharded_grads(group, 24, local, 5, 3, inactive="one")
    reference = _run(False, make, grads, lr=0.02)
    mixed = _run(group.rank() == 0, make, grads, lr=0.02)
    for ref_step, mixed_step in zip(reference, mixed):
        assert _bitwise_equal(ref_step[0][1], mixed_step[0][1])
        torch.testing.assert_close(mixed_step[0][0], ref_step[0][0], **PARAM_TOL)


@pytest.mark.parametrize("steps", [11, 4, 0])
def test_row_sharded_collectives_match_reference(steps, monkeypatch):
    """Both paths all-reduce tensors of the same count, order, shape and dtype (a 0-dim sum of
    the row norms, then [1, C] column sums), so a row group may mix them also under
    TORCH_DISTRIBUTED_DEBUG=DETAIL, whose ProcessGroupWrapper compares shapes across ranks.
    Normalization steps set below 1 after construction all-reduce no column sums in either
    path (an extra all-reduce on fused ranks only would pair with another collective)."""
    group = _row_group(2)
    make, local = _sharded(group, 24, _shard_layout(2, False), 3)
    grads = _sharded_grads(group, 24, local, 3, 2, inactive="none")
    original = dist.all_reduce
    calls = {False: [], True: []}
    for fused in (False, True):

        def spy(tensor, *args, recorded=calls[fused], **kwargs):
            recorded.append((tuple(tensor.shape), tensor.dtype))
            return original(tensor, *args, **kwargs)

        monkeypatch.setattr(dist, "all_reduce", spy)
        _run(fused, make, grads, lr=0.02, optimizer_factory=_with_normalization_steps(steps))
    assert calls[True] == calls[False]
    pairs = max(0, (steps - 1) // 2)  # 5, 1 and 0
    assert calls[True].count(((1, 24), torch.float32)) == 2 * pairs  # two steps
    assert calls[True].count(((), torch.float32)) == 2


def test_fused_step_is_deterministic():
    grads = _grads(4097, 256, steps=2)
    first = _run(True, _matrix(4097, 256), grads, lr=0.02)
    second = _run(True, _matrix(4097, 256), grads, lr=0.02)
    for (a,), (b,) in zip(first, second):
        assert _bitwise_equal(a[0], b[0]) and _bitwise_equal(a[1], b[1])


def test_fused_step_has_no_host_sync_or_full_size_temporary(fused_calls):
    """What the fused step exists for: Sinkhorn.step neither synchronizes with the host nor
    allocates a matrix-sized temporary. A padded 65,543 x 256 shard (L = 65,536) needs about
    2.6 MiB (two row-factor buffers of L floats and the column partials of 2048 programs)
    against 64 MiB per FP32 matrix."""
    rows, local_rows, cols = 65543, 65536, 256
    grads = _grads(rows, cols, steps=2, local_rows=local_rows)
    with _switch(True):
        params = _matrix(rows, cols, local_rows=local_rows)()
        optimizer = eo_mod.Sinkhorn(params, lr=0.02)
        params[0].grad = grads[0][0]
        optimizer.step()  # Creates the momentum buffer and compiles the kernels.
        params[0].grad = grads[1][0]
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.set_sync_debug_mode("error")
        try:
            optimizer.step()
        finally:
            torch.cuda.set_sync_debug_mode("default")
        torch.cuda.synchronize()
    extra = torch.cuda.max_memory_allocated() - base
    assert extra < rows * cols * 4 // 8, f"{extra} bytes of temporaries"
    assert fused_calls == [(rows, cols)] * 2


@pytest.mark.parametrize("steps", [4, 0, -1])
def test_eligibility_of_the_whole_matrix_follows_sinkhorn_balance(steps):
    """Settings sinkhorn_balance rejects keep the reference, which raises."""
    params = _fp32_matrix()
    optimizer = eo_mod.Sinkhorn(params, lr=0.02)
    params[0].grad = torch.randn_like(params[0])
    optimizer.param_groups[0]["num_normalization_steps"] = steps
    with _switch(True), pytest.raises(ValueError, match="odd"):
        optimizer.step()
    assert math.isfinite(params[0].sum().item())
