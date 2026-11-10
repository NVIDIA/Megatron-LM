# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Oracle for the batch-invariant RMSNorm: `BatchInvariantRMSNormFn`, the patched TE
`RMSNorm.forward` and the patched `apply_normalization`.

Kernel: `megatron/core/transformer/custom_layers/batch_invariant_kernels.py`
(`BatchInvariantRMSNormFn` and its wrapper `rmsnorm_batch_invariant`; `_te_rmsnorm_forward_patched`
on TE's `RMSNorm` class; `_te_apply_normalization_patched` on TE's fused-module
`apply_normalization`; all installed by `enable_batch_invariant_mode` for the `triton` and
`deepgemm` backends, while `te_native` keeps TE's norms). Inference reaches them through the
inference-optimized linears' norm (`inference_layers._te_rms_norm_kernel` calls
`rmsnorm_batch_invariant` on those backends), TE `RMSNorm` modules and TE `LayerNormLinear` /
`LayerNormMLP` in a batch-invariant model. The row statistic runs `mean_kernel`
(`test_mean_batch_invariant.py`).

Contract (written from TE's RMSNorm, which these replace, and the backends' promise):
- y = x * rsqrt(mean(x**2 over the last dim) + eps) * g, with g = w, or 1 + w when
  `zero_centered_gamma`; computed in fp32 from x and w and rounded once to x's dtype (bf16,
  fp16, fp32); any leading dims; new output; inputs read only. Unsupported dtypes or CPU inputs
  raise.
- The patched TE `RMSNorm.forward` computes that with the module's eps and
  `zero_centered_gamma`. The patched `apply_normalization` returns `(ln_out, None, rsigma)` for
  RMSNorm without bias or output quantizer: ln_out = y in `output_dtype`, written into the
  caller's `ln_out` when one is given, and rsigma = rsqrt(mean(x**2) + eps) in fp32, one per
  row. LayerNorm, a bias or a quantized output are left to TE's own function, unchanged.
- Dispatch (exact): `triton` and `deepgemm` replace TE's RMSNorm forward and `apply_normalization`;
  `te_native` replaces neither; outside batch-invariant mode TE's own functions run.
- Batch invariance: a row's bits do not depend on the number of rows or on the other rows.

Reference: fp64 RMSNorm from the formula. Trusted implementation: torch's fused
`F.rms_norm` on the fp32 input with the fp32 gain, rounded once to x's dtype (the same
precision class: fp32 statistics, one rounding); rsigma's trusted value is torch's fp32
`rsqrt(mean(x * x) + eps)`. Tolerance class: floating point, bounded per row (default factor 2)
for bf16 / fp16 outputs and over the tensor with factor 3 for fp32 outputs (FP32_FACTOR);
exact for dispatch, the caller's buffer, LayerNorm delegation and batch invariance.

Structure (read after the contract): the forward casts x and w to fp32, adds 1 to the gain for
zero-centered gamma, takes `mean_dim(x * x, -1)` (one program per row, 1024-element blocks),
`rsqrt`, multiplies, and rounds; `rmsnorm_batch_invariant` hard-codes `zero_centered_gamma=False`
(the inference-optimized config forbids zero-centered gamma). Covered: hidden 1, 1023, 1024, 1025,
2049, 4096, 7168 (the statistic's block boundaries), 1 to 2274 rows (TE's native kernel changed
bits between 928 and 2274 rows on GB200), bf16 / fp16 / fp32, both gamma modes, two eps values,
2-D and 3-D inputs, the three entry points with the caller's `ln_out`, delegation, production
widths.

Not covered: the backward (training only), and the module-level `rmsnorm*` wrappers, which wrap
nothing on the pinned TE (step 3 R04, S9: TE 2.20 exposes none of the names).

Main vs open PRs: all cases are expected to pass on main. #8869 (op-fuser norms) adds patch sites
and does not change these functions.

Hardware: H100 (1 GPU), with Transformer Engine.
"""

import pytest
import torch
import torch.nn.functional as F

from megatron.core.transformer.custom_layers import batch_invariant_kernels as bik
from tests.unit_tests.inference.oracles import oracle_utils as ou

pytestmark = [ou.requires_sm(ou.HOPPER)]

_DEVICE = "cuda"

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
BLOCK = 1024  # mean_dim's block over the hidden dim


@pytest.fixture(autouse=True)
def _batch_invariant_mode_off():
    """References and trusted outputs use torch's own ops: the mode must be off around them."""
    assert not bik.is_batch_invariant_mode_enabled(), "batch-invariant mode leaked into the test"
    yield
    assert not bik.is_batch_invariant_mode_enabled(), "the test left batch-invariant mode on"


def _te():
    import transformer_engine.pytorch as te

    return te


def bi_mode(backend: str):
    if backend == "deepgemm":
        assert bik.HAVE_DEEPGEMM_BF16, "the deepgemm backend needs DeepGEMM's bf16 bindings"
    return bik.set_batch_invariant_mode(True, backend=backend)


def gain(w, zero_centered):
    return w + 1.0 if zero_centered else w


def rmsnorm_reference(x, w, eps, zero_centered):
    rsigma = torch.rsqrt((x * x).mean(dim=-1, keepdim=True) + eps)
    return x * rsigma * gain(w, zero_centered)


def rsigma_reference(x, eps):
    return torch.rsqrt((x * x).mean(dim=-1) + eps)


def trusted_rmsnorm(x, w, eps, zero_centered):
    """torch's fused rms_norm in fp32 (gain in fp32), rounded once to x's dtype."""
    g = gain(w.float(), zero_centered)
    return F.rms_norm(x.float(), (x.shape[-1],), g, eps).to(x.dtype)


# fp32 outputs: a row's error is dominated by its one rsigma (one mean of x**2, one division,
# one rsqrt per row), so a per-row bound compares two scalars, and on a correct kernel some rows
# measured up to 3.7x torch's error (pdx 7986825: 15 of 928 rows above 2x). The fp32 bound is
# therefore over the whole tensor (outputs are normalized, so every row weighs alike) with
# factor 3: the kernel's mean divides with Triton's `div.full.f32` (up to 2 ulp) where torch
# multiplies by 1/N, and sums in 1024-element blocks. A wrong eps, gain or statistic errs by
# orders of magnitude more. bf16 / fp16 outputs are dominated by the final rounding: per row,
# default factor.
FP32_FACTOR = 3.0


def check_norm(out, x, w, eps, zero_centered, what):
    assert out.shape == x.shape and out.dtype == x.dtype, f"{what}: {out.shape} {out.dtype}"
    fp32 = out.dtype == torch.float32
    ou.assert_close_to_reference(
        out,
        ou.fp64_reference(rmsnorm_reference, x, w, eps, zero_centered),
        trusted_rmsnorm(x, w, eps, zero_centered),
        factor=FP32_FACTOR if fp32 else ou.DEFAULT_FACTOR,
        dim=None if fp32 else -1,
        what=what,
    )


def inputs(rows, hidden, dtype, seed, three_d=False):
    """x with a different scale per row (2**-3 .. 2**4), and a gain around 1 (or 0)."""
    g = ou.make_generator(seed, device=_DEVICE)
    x = torch.randn(rows, hidden, device=_DEVICE, generator=g)
    x = x * torch.pow(2.0, torch.randint(-3, 5, (rows, 1), device=_DEVICE, generator=g))
    w = 1.0 + 0.25 * torch.randn(hidden, device=_DEVICE, generator=g)
    x, w = x.to(dtype), w.to(dtype)
    if three_d and rows % 3 == 0:
        x = x.view(3, rows // 3, hidden)
    return x, w


def _id(case):
    return "-".join(f"{k}{v}" for k, v in case.items())


_CASES = ou.pairwise_cases(
    {
        "dtype": tuple(DTYPES),
        "hidden": (1, BLOCK - 1, BLOCK, BLOCK + 1, 2 * BLOCK + 1, 4096, 7168),
        "rows": (1, 31, 33, 928, 2274),
        "zc": (False, True),
        "eps": (1e-5, 1e-6),
        "three_d": (False, True),
    }
)


@pytest.mark.parametrize("case", _CASES, ids=[_id(c) for c in _CASES])
def test_rmsnorm_fn_matches_fp64(case):
    """`BatchInvariantRMSNormFn` over hidden sizes around the statistic's 1024-element blocks,
    row counts on both sides of multiples of 32, dtypes, gamma modes, eps and 3-D inputs."""
    dtype = DTYPES[case["dtype"]]
    x, w = inputs(case["rows"], case["hidden"], dtype, seed=case["hidden"], three_d=case["three_d"])
    before = (x.clone(), w.clone())
    out = bik.BatchInvariantRMSNormFn.apply(x, w, case["eps"], case["zc"])
    ou.assert_exact(x, before[0], what="x is read-only")
    ou.assert_exact(w, before[1], what="w is read-only")
    check_norm(out, x, w, case["eps"], case["zc"], _id(case))


@pytest.mark.parametrize("dtype", list(DTYPES), ids=list(DTYPES))
def test_inference_wrapper_matches_fp64(dtype):
    """`rmsnorm_batch_invariant` (the inference-optimized linears' norm): plain gain."""
    x, w = inputs(129, 4096, DTYPES[dtype], seed=3)
    check_norm(bik.rmsnorm_batch_invariant(x, w, 1e-5), x, w, 1e-5, False, f"wrapper {dtype}")


def test_unsupported_inputs_raise():
    """CPU tensors and non-float dtypes raise instead of returning a wrong result."""
    w = torch.ones(64, device=_DEVICE)
    with pytest.raises(RuntimeError):
        bik.BatchInvariantRMSNormFn.apply(torch.ones(2, 64), w.cpu(), 1e-5, False)
    with pytest.raises(RuntimeError):
        bik.BatchInvariantRMSNormFn.apply(
            torch.ones(2, 64, device=_DEVICE, dtype=torch.float64), w, 1e-5, False
        )


def te_rmsnorm(hidden, w, eps, zero_centered):
    te = _te()
    module = te.RMSNorm(
        hidden, eps=eps, zero_centered_gamma=zero_centered, params_dtype=w.dtype, device=_DEVICE
    )
    with torch.no_grad():
        module.weight.copy_(w)
    return module.eval()


class _FnSpy:
    """Counts `BatchInvariantRMSNormFn.apply` calls (the batch-invariant forward)."""

    def __init__(self, monkeypatch):
        self.calls = 0
        apply = bik.BatchInvariantRMSNormFn.apply

        def spy(*args):
            self.calls += 1
            return apply(*args)

        monkeypatch.setattr(bik.BatchInvariantRMSNormFn, "apply", spy)


@pytest.mark.parametrize("backend", ["triton", "deepgemm", "te_native"])
@pytest.mark.parametrize("zero_centered", [False, True], ids=["gain", "zero_centered"])
def test_te_rmsnorm_module(backend, zero_centered, monkeypatch):
    """TE's RMSNorm module under each backend: the batch-invariant forward with the module's eps
    and gamma mode on triton/deepgemm (bit-identical to the Fn), TE's own kernel on te_native;
    against fp64 either way."""
    x, w = inputs(130, 4096, torch.bfloat16, seed=4, three_d=False)
    if zero_centered:
        w = (w.float() - 1.0).to(torch.bfloat16)
    module = te_rmsnorm(4096, w, 1e-6, zero_centered)
    spy = _FnSpy(monkeypatch)
    with torch.no_grad(), bi_mode(backend):
        out = module(x)
    patched = backend != "te_native"
    assert spy.calls == int(patched), f"{backend}: batch-invariant forward ran {spy.calls} times"
    check_norm(out, x, w, 1e-6, zero_centered, f"TE RMSNorm under {backend}")
    if patched:
        want = bik.BatchInvariantRMSNormFn.apply(x, w, 1e-6, zero_centered)
        ou.assert_exact(out, want, what=f"TE RMSNorm under {backend} is the batch-invariant Fn")


def _apply_normalization(x, ln_out, w, bias, eps, normalization, zero_centered):
    return bik._te_apply_normalization_patched(
        x, ln_out, w, bias, eps, None, x.dtype, normalization, 0, zero_centered
    )


@pytest.mark.parametrize("zero_centered", [False, True], ids=["gain", "zero_centered"])
@pytest.mark.parametrize("own_buffer", [False, True], ids=["new_out", "caller_out"])
def test_apply_normalization_rmsnorm(zero_centered, own_buffer):
    """The fused-module entry: ln_out (into the caller's buffer when given) and fp32 rsigma per
    row, against fp64."""
    x, w = inputs(257, 2049, torch.bfloat16, seed=5)
    buf = ou.guarded_empty(x.shape, x.dtype) if own_buffer else None
    with torch.no_grad(), bi_mode("triton"):
        ln_out, mu, rsigma = _apply_normalization(
            x, buf.tensor if own_buffer else None, w, None, 1e-5, "RMSNorm", zero_centered
        )
    assert mu is None, "RMSNorm returns no mean"
    if own_buffer:
        ou.check_guards(ln_out=buf)
        assert ln_out.data_ptr() == buf.tensor.data_ptr(), "ln_out is not the caller's buffer"
    check_norm(ln_out, x, w, 1e-5, zero_centered, "apply_normalization ln_out")
    assert rsigma.shape == (257,) and rsigma.dtype == torch.float32, f"rsigma {rsigma.shape}"
    xf = x.float()
    ou.assert_close_to_reference(
        rsigma,
        ou.fp64_reference(rsigma_reference, x, 1e-5),
        torch.rsqrt((xf * xf).mean(dim=-1) + 1e-5),
        what="rsigma",
    )


def test_apply_normalization_delegates_layernorm_to_te():
    """LayerNorm is left to TE's own `apply_normalization`, called once with the same arguments:
    the result is TE's bits."""
    x, w = inputs(65, 1024, torch.bfloat16, seed=6)
    bias = (0.1 * w.float()).to(w.dtype)
    key = "module._common.apply_normalization"
    calls = []
    with torch.no_grad(), bi_mode("triton"):
        original = bik._TE_APPLY_NORM_ORIGS[key]

        def spy(*args):
            calls.append(args[7])
            return original(*args)

        # Restored before the mode is left: disabling the mode restores TE from this dict.
        bik._TE_APPLY_NORM_ORIGS[key] = spy
        try:
            got = _apply_normalization(x, None, w, bias, 1e-5, "LayerNorm", False)
        finally:
            bik._TE_APPLY_NORM_ORIGS[key] = original
        want = original(x, None, w, bias, 1e-5, None, x.dtype, "LayerNorm", 0, False)
    assert calls == ["LayerNorm"], f"TE's function ran for {calls}"
    for i, (a, b) in enumerate(zip(got, want)):
        if a is None or b is None:
            assert a is None and b is None, f"output {i}"
        else:
            ou.assert_exact(a, b, what=f"LayerNorm output {i}")


def test_layernorm_linear_uses_the_batch_invariant_norm():
    """TE `LayerNormLinear` (RMSNorm, zero-centered gamma) under triton returns the batch-invariant
    norm as its layernorm output."""
    te = _te()
    x, w = inputs(96, 1024, torch.bfloat16, seed=7)
    w = (w.float() - 1.0).to(torch.bfloat16)
    module = te.LayerNormLinear(
        1024,
        256,
        bias=False,
        normalization="RMSNorm",
        zero_centered_gamma=True,
        return_layernorm_output=True,
        params_dtype=torch.bfloat16,
        device=_DEVICE,
    ).eval()
    with torch.no_grad():
        module.layer_norm_weight.copy_(w)
    with torch.no_grad(), bi_mode("triton"):
        _, ln_out = module(x)
    check_norm(ln_out, x, w, module.eps, True, "LayerNormLinear layernorm output")
    want = bik.BatchInvariantRMSNormFn.apply(x, w, module.eps, True)
    ou.assert_exact(ln_out, want, what="LayerNormLinear's norm is the batch-invariant one")


_INVARIANCE = [("fn", 4096), ("fn", 1025), ("module", 7168), ("apply_normalization", 2049)]


@pytest.mark.parametrize("entry,hidden", _INVARIANCE, ids=[f"{e}-{h}" for e, h in _INVARIANCE])
def test_rows_do_not_depend_on_the_batch(entry, hidden):
    """Five probe rows placed among 5 ... 2274 rows (start, odd middle offset, end): the same bits
    from every entry point."""
    probe, w = inputs(5, hidden, torch.bfloat16, seed=8)
    filler, _ = inputs(2274, hidden, torch.bfloat16, seed=9)
    module = te_rmsnorm(hidden, w, 1e-5, False) if entry == "module" else None

    def norm(x):
        with torch.no_grad(), bi_mode("triton"):
            if entry == "fn":
                return bik.BatchInvariantRMSNormFn.apply(x, w, 1e-5, False)
            if entry == "module":
                return module(x)
            return _apply_normalization(x, None, w, None, 1e-5, "RMSNorm", False)[0]

    base = norm(probe)
    check_norm(base, probe, w, 1e-5, False, f"{entry} probe rows")
    for rows in (5, 6, 33, 928, 2274):
        for start in sorted({0, min((rows - 5) // 2 | 1, rows - 5), rows - 5}):
            x = filler[:rows].clone()
            x[start : start + 5] = probe
            ou.assert_exact(norm(x)[start : start + 5], base, what=f"rows={rows} start={start}")


@pytest.mark.parametrize("hidden,rows", [(7168, 4096), (8192, 8192)], ids=["dsv3", "llama70b"])
@pytest.mark.parametrize("zero_centered", [False, True], ids=["gain", "zero_centered"])
def test_production_sizes(hidden, rows, zero_centered):
    """Deployed hidden sizes at prefill row counts, bf16."""
    x, w = inputs(rows, hidden, torch.bfloat16, seed=10)
    out = bik.BatchInvariantRMSNormFn.apply(x, w, 1e-5, zero_centered)
    check_norm(out, x, w, 1e-5, zero_centered, f"production {rows}x{hidden}")


@pytest.mark.sanitizer
def test_sanitizer_case():
    """Small case for compute-sanitizer: three rows of 1025 (a full statistic block and a masked
    one-element block), zero-centered gamma."""
    x, w = inputs(3, BLOCK + 1, torch.bfloat16, seed=11)
    check_norm(bik.BatchInvariantRMSNormFn.apply(x, w, 1e-5, True), x, w, 1e-5, True, "sanitizer")


@pytest.mark.parametrize("backend", ["triton", "deepgemm"])
def test_te_rmsnorm_module_output_dtype_follows_te(backend):
    """TE's RMSNorm module returns its activation dtype (the parameter dtype) whatever the
    input's: an fp32 residual stream into a bf16 norm gives bf16. The batch-invariant forward
    that replaces it must return the same dtype and values within the bound of that output.

    Suspected finding O8a-P2 (post-audit batch): `_te_rmsnorm_forward_patched` returns the
    input's dtype (`BatchInvariantRMSNormFn` casts back to `x.dtype`), so under triton/deepgemm
    an fp32 residual stream (a learned fp32 position embedding, `fp32_residual_connection`)
    reaches the next GEMM in fp32. The LM head's `torch.matmul` then fails ("Incompatible dtypes"
    in `matmul_persistent`), see `test_lm_head_fp32_logits.py`."""
    x, w = inputs(65, 1024, torch.float32, seed=9)
    w = w.to(torch.bfloat16)
    module = te_rmsnorm(1024, w, 1e-5, False)
    with torch.no_grad():
        native = module(x)
        with bi_mode(backend):
            out = module(x)
    assert (
        out.dtype == native.dtype
    ), f"{backend}: {out.dtype}, TE's own forward gives {native.dtype}"
    ou.assert_close_to_reference(
        out,
        ou.fp64_reference(rmsnorm_reference, x, w, 1e-5, False),
        trusted_rmsnorm(x, w, 1e-5, False).to(native.dtype),
        dim=-1,
        what=f"TE RMSNorm on fp32 input under {backend}",
    )
