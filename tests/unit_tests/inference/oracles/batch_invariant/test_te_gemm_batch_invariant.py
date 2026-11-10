# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Oracle for `_te_general_gemm_patched` -> `BatchInvariantTEGemmFn`, the batch-invariant TE GEMM.

Kernel: `megatron/core/transformer/custom_layers/batch_invariant_kernels.py`
(`_te_general_gemm_patched`, which `enable_batch_invariant_mode` installs in place of TE's
`general_gemm` at its import sites for the `triton` and `deepgemm` backends; it runs
`BatchInvariantTEGemmFn`, whose forward is `mm_batch_invariant`: `matmul_persistent` on
`triton`, DeepGEMM `bf16_gemm_nn` for bf16 on `deepgemm`). Inference reaches it through every TE
linear of a batch-invariant model on those backends (TE `Linear` / `LayerNormLinear` forwards,
the TN layout) and through Megatron's `te_general_gemm`, which the MoE router's gating uses with
an fp32 output. The GEMMs underneath have their own oracles (`test_matmul_persistent.py`,
`test_deepgemm_dense.py`); this module checks what the replacement promises on top of them.

Contract (written from TE's `general_gemm`, which the patch replaces, as Megatron calls it through
`te_general_gemm(A, B, out_dtype, layout, out, bias, grad, accumulate)`):
- With layout "XY", D = op_Y(B) @ op_X(A), where op_T(M) = M^T and op_N(M) = M: TN is the
  forward x @ W^T, NN the input gradient, NT the weight gradient. B's leading dims are
  flattened (TN/NN) and restored on D. Accumulated in fp32; `bias` (TN) is added to the fp32
  result; the result has `out_dtype` (fp32 output of bf16 operands keeps fp32 precision).
- `out=`: the result is written into `out`, which is returned; with `accumulate=True` it is
  added to what `out` holds. Userbuffers / comm-overlap arguments are refused. Returns
  `(D, bias_grad, gelu_input, extra_output)`.
- Dispatch (exact): under `triton` and `deepgemm`, TE's `general_gemm` (and so TE Linear's
  forward) runs `BatchInvariantTEGemmFn`; under `te_native` and outside the mode TE's own GEMM
  runs.
- Batch invariance: each row of a TN result depends only on that row of B and on A.

Reference: fp64 `op(B) @ op(A) (+ bias)`. Trusted implementation: the same `te_general_gemm`
call outside batch-invariant mode (TE's cuBLASLt GEMM, one rounding, bias in the epilogue).
Tolerance class: floating point, bounded per row (default factor 2). The replacement adds the
bias to the rounded product in the output dtype (a second rounding, at most 1 ulp against
cuBLASLt's 0.5; documented, as for DeepGEMM's addmm in `test_deepgemm_dense.py`), which the
bound admits. Exact for dispatch, the `out` buffer, refusals and batch invariance.

Structure (read after the contract): `_extract_te_gemm_args` reads A, B positionally and the rest
from keywords; the forward transposes and makes operands contiguous per layout (the triton
backend keeps a transposed A as a view), flattens B, runs `mm_batch_invariant`, adds the bias in
the output dtype, casts to `out_dtype`, and `out.copy_(result)`. Covered: TN / NN / NT, both
backends, 1 to 1000 tokens around the 128-row tiles, K at 64 / 1024 / 4096 and N up to 4096,
2-D and 3-D B, bias, `out`, refusals, TE `Linear` dispatch under the three backends, TN row
invariance (triton), a production shape.

Main vs open PRs: every case here passes on main and with the PRs below. Three parts of the
contract fail on main, and each fix PR tests its own (in
`transformer/test_te_layers_batch_invariant.py` and
`inference/test_batch_invariant_deepgemm_stride.py`):
- an fp32 output of bf16 operands is computed in bf16 and cast afterwards (finding O8a-N3):
  #8957 (s4f-router-fp32-logits), `test_bik_router_gating_logits_are_fp32`;
- `accumulate` is ignored and `out` overwritten (step 3 P4-N4): #8865 (t3-p4n4),
  `test_bik_te_general_gemm_honors_accumulate_alpha_beta`;
- under deepgemm, TN rows change with the token count where DeepGEMM forwards to cuBLASLt:
  #8867 (t3-deepgemm), `test_dense_deepgemm_rows_are_invariant_across_batch_size`.

Hardware: H100 (1 GPU), with Transformer Engine and DeepGEMM (bf16 bindings).
"""

import pytest
import torch

from megatron.core.transformer.custom_layers import batch_invariant_kernels as bik
from tests.unit_tests.inference.oracles import oracle_utils as ou

pytestmark = [ou.requires_sm(ou.HOPPER)]

_DEVICE = "cuda"

BACKENDS = ("triton", "deepgemm")


@pytest.fixture(autouse=True)
def _batch_invariant_mode_off():
    """Trusted outputs come from TE's own GEMM: the mode must be off around each test."""
    assert not bik.is_batch_invariant_mode_enabled(), "batch-invariant mode leaked into the test"
    yield
    assert not bik.is_batch_invariant_mode_enabled(), "the test left batch-invariant mode on"


def te_gemm(*args, **kwargs):
    from megatron.core.extensions.transformer_engine import te_general_gemm

    assert te_general_gemm is not None, "Transformer Engine is required"
    return te_general_gemm(*args, **kwargs)


def bi_mode(backend):
    if backend == "deepgemm":
        assert bik.HAVE_DEEPGEMM_BF16, "the deepgemm backend needs DeepGEMM's bf16 bindings"
    return bik.set_batch_invariant_mode(True, backend=backend)


def bf16(shape, seed, scale=1.0):
    g = ou.make_generator(seed, device=_DEVICE)
    return (torch.randn(*shape, device=_DEVICE, generator=g) * scale).to(torch.bfloat16)


def op(t, transposed):
    return t.transpose(0, 1) if transposed else t


def gemm_reference(a, b, layout, bias=None):
    """D = op_Y(B) @ op_X(A) for layout "XY", B's leading dims flattened and restored."""
    a_op = op(a, layout[0] == "T")
    if layout[1] == "T":
        out = op(b, True) @ a_op
    else:
        out = (b.reshape(-1, b.shape[-1]) @ a_op).reshape(*b.shape[:-1], a_op.shape[-1])
    return out if bias is None else out + bias


def operands(layout, tokens, k, n, three_d, seed):
    """(A, B) for D [tokens, n] with reduction length k, in TE's argument shapes."""
    a = bf16((n, k) if layout[0] == "T" else (k, n), seed, scale=k**-0.5)
    if layout[1] == "T":
        b = bf16((k, tokens), seed + 1)
    elif three_d and tokens % 2 == 0:
        b = bf16((tokens // 2, 2, k), seed + 1)
    else:
        b = bf16((tokens, k), seed + 1)
    return a, b


def check(out, a, b, layout, bias, trusted, what):
    assert out.shape == trusted.shape and out.dtype == trusted.dtype, f"{what}: {out.shape}"
    ref = ou.fp64_reference(gemm_reference, a, b, layout, bias)
    if out.numel():
        ou.assert_close_to_reference(
            out.reshape(-1, out.shape[-1]),
            ref.reshape(-1, ref.shape[-1]),
            trusted.reshape(-1, trusted.shape[-1]),
            dim=-1,
            what=what,
        )


def _id(case):
    text = {k: "x".join(map(str, v)) if isinstance(v, tuple) else v for k, v in case.items()}
    return "-".join(f"{k}{v}" for k, v in text.items())


_CASES = ou.pairwise_cases(
    {
        "backend": BACKENDS,
        "layout": ("TN", "NN", "NT"),
        "tokens": (1, 127, 128, 129, 1000),
        "kn": ((64, 128), (1024, 4096), (4096, 1000)),
        "bias": (False, True),
        "three_d": (False, True),
    }
)


@pytest.mark.parametrize("case", _CASES, ids=[_id(c) for c in _CASES])
def test_general_gemm_matches_fp64(case):
    """Every layout on both backends, tokens around the 128-row tiles, K and N small and large,
    a bias on the forward layout, 2-D and 3-D B: per row against fp64, bounded by TE's GEMM."""
    layout, (k, n) = case["layout"], case["kn"]
    a, b = operands(layout, case["tokens"], k, n, case["three_d"], seed=k + case["tokens"])
    bias = bf16((n,), seed=n) if case["bias"] and layout == "TN" else None
    kwargs = dict(out_dtype=torch.bfloat16, layout=layout, bias=bias)
    # TE's GEMM takes B as a matrix: the trusted result of a 3-D B is that of its 2-D view.
    flat = te_gemm(a, b.reshape(-1, b.shape[-1]) if b.dim() == 3 else b, **kwargs)[0]
    trusted = flat.reshape(*b.shape[:-1], flat.shape[-1]) if b.dim() == 3 else flat
    before = (a.clone(), b.clone())
    with torch.no_grad(), bi_mode(case["backend"]):
        out = te_gemm(a, b, **kwargs)[0]
    ou.assert_exact(a, before[0], what="A is read-only")
    ou.assert_exact(b, before[1], what="B is read-only")
    check(out, a, b, layout, bias, trusted, _id(case))


def check_out_buffer(backend, accumulate):
    """`out=`: the result lands in the caller's buffer, which is returned; with `accumulate` it is
    added to what the buffer held."""
    a, b = bf16((256, 512), 3, scale=512**-0.5), bf16((96, 512), 4)
    held = bf16((96, 256), 5)
    kwargs = dict(out_dtype=torch.bfloat16, layout="TN", accumulate=accumulate)
    trusted_out = held.clone()
    trusted = te_gemm(a, b, out=trusted_out, **kwargs)[0]
    buf = ou.guarded_like(held)
    with torch.no_grad(), bi_mode(backend):
        got = te_gemm(a, b, out=buf.tensor, **kwargs)[0]
    ou.check_guards(out=buf)
    assert got.data_ptr() == buf.tensor.data_ptr(), "the result is not the caller's buffer"
    ref = ou.fp64_reference(
        lambda a, b, h: gemm_reference(a, b, "TN") + (h if accumulate else 0), a, b, held
    )
    ou.assert_close_to_reference(got, ref, trusted, dim=-1, what=f"out= under {backend}")


@pytest.mark.parametrize("backend", BACKENDS)
def test_out_buffer(backend):
    """`out=` without accumulation: the caller's buffer holds the result."""
    check_out_buffer(backend, accumulate=False)


def test_refusals():
    """Userbuffers / overlap arguments and integer operands raise instead of running."""
    a, b = bf16((64, 64), 6), bf16((8, 64), 7)
    with bi_mode("triton"):
        for kwargs in ({"ub": object()}, {"extra_output": torch.empty(1)}, {"bulk_overlap": True}):
            with pytest.raises(RuntimeError):
                bik._te_general_gemm_patched(a, b, layout="TN", **kwargs)
        with pytest.raises(RuntimeError):
            bik._te_general_gemm_patched(a.to(torch.int32), b.to(torch.int32), layout="TN")


@pytest.mark.parametrize("backend", ["triton", "deepgemm", "te_native"])
def test_te_linear_dispatch(backend, monkeypatch):
    """TE `Linear`'s forward runs the batch-invariant GEMM exactly under triton and deepgemm (with
    its bits) and TE's own GEMM under te_native; against fp64 either way."""
    import transformer_engine.pytorch as te

    linear = te.Linear(512, 384, bias=True, params_dtype=torch.bfloat16, device=_DEVICE).eval()
    with torch.no_grad():
        linear.weight.copy_(bf16((384, 512), 8, scale=512**-0.5))
        linear.bias.copy_(bf16((384,), 9))
    x = bf16((7, 3, 512), 10)
    calls = []
    apply = bik.BatchInvariantTEGemmFn.apply

    def spy(*args):
        calls.append(args[4])
        return apply(*args)

    monkeypatch.setattr(bik.BatchInvariantTEGemmFn, "apply", spy)
    with torch.no_grad(), bi_mode(backend):
        out = linear(x)
        direct = (
            apply(linear.weight, x, linear.bias, torch.bfloat16, "TN")
            if backend != "te_native"
            else None
        )
    assert calls == ([] if backend == "te_native" else ["TN"]), f"{backend}: GEMMs {calls}"
    if direct is not None:
        ou.assert_exact(out, direct, what=f"TE Linear under {backend} is the batch-invariant GEMM")
    with torch.no_grad():
        trusted = linear(x)
    weight, bias = linear.weight.detach(), linear.bias.detach()
    check(out, weight, x, "TN", bias, trusted, f"TE Linear under {backend}")


@pytest.mark.parametrize("backend", ["triton"])
def test_tn_rows_do_not_depend_on_tokens(backend):
    """Five probe rows of B among 5 ... 8192 tokens (start, odd middle offset, end), K = N = 4096:
    the same TN bits everywhere. Under deepgemm this fails on main, where DeepGEMM forwards to
    cuBLASLt's heuristic algorithm, which on H100 changes between M = 15 and M = 8192 at this
    shape (pdx 7986825); #8867 fixes and tests that."""
    a = bf16((4096, 4096), 11, scale=1 / 64)
    probe, filler = bf16((5, 4096), 12), bf16((8192, 4096), 13)
    with torch.no_grad(), bi_mode(backend):
        base = te_gemm(a, probe, out_dtype=torch.bfloat16, layout="TN")[0]
        for tokens in (5, 6, 15, 129, 1000, 4096, 8192):
            for start in sorted({0, min((tokens - 5) // 2 | 1, tokens - 5), tokens - 5}):
                b = filler[:tokens].clone()
                b[start : start + 5] = probe
                out = te_gemm(a, b, out_dtype=torch.bfloat16, layout="TN")[0]
                ou.assert_exact(
                    out[start : start + 5], base, what=f"T={tokens} start={start} {backend}"
                )


@pytest.mark.parametrize("backend", BACKENDS)
def test_production_size(backend):
    """A prefill-sized forward: 4096 tokens through a 4096 -> 14336 projection."""
    a, b = bf16((14336, 4096), 14, scale=1 / 64), bf16((4096, 4096), 15)
    trusted = te_gemm(a, b, out_dtype=torch.bfloat16, layout="TN")[0]
    with torch.no_grad(), bi_mode(backend):
        out = te_gemm(a, b, out_dtype=torch.bfloat16, layout="TN")[0]
    check(out, a, b, "TN", None, trusted, f"production under {backend}")


@pytest.mark.sanitizer
def test_sanitizer_case():
    """Small case for compute-sanitizer: TN with a bias, 130 tokens (two row tiles), K = 65 and
    N = 129 (partial tiles), triton backend."""
    a, b, bias = bf16((129, 65), 16, scale=0.125), bf16((130, 65), 17), bf16((129,), 18)
    trusted = te_gemm(a, b, out_dtype=torch.bfloat16, layout="TN", bias=bias)[0]
    with torch.no_grad(), bi_mode("triton"):
        out = te_gemm(a, b, out_dtype=torch.bfloat16, layout="TN", bias=bias)[0]
    check(out, a, b, "TN", bias, trusted, "sanitizer")
