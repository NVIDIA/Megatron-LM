# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Oracle for the inference-optimized TP linears, their non-FlashInfer part.

Kernel: `megatron/core/tensor_parallel/inference_layers.py`: `_te_rms_norm_kernel` (TE's
`tex.rmsnorm_fwd`, or `rmsnorm_batch_invariant` on the triton/deepgemm batch-invariant
backends), `_apply_linear` (`torch.matmul(x, w.t())` for bf16 weights) and the NCCL fallbacks
(TE's `gather_along_first_dim` / `reduce_scatter_along_first_dim`) of
`InferenceLayerNormColumnParallelLinear`, `InferenceColumnParallelLinear`,
`InferenceRowParallelLinear` and `InferenceLinear`. Every layer of an inference-optimized model
(`transformer_impl="inference_optimized"`, `InferenceSpecProvider`) runs them; the MoE latent
projections use `InferenceLinear`. The NVLS kernels these layers can select instead belong to the
NVLS family (`test_fused_tp_dispatch.py` checks the selection), MXFP8 weights to the FlashInfer
family.

Contract (written from the layers' math, which their TE training forward also computes):
- In inference (eval) mode, with TP size p and sequence parallelism (required for p > 1):
  - LayerNorm-column: y_r = RMSNorm(gather(x)) @ W_r^T, where gather concatenates the ranks'
    [T/p, B, H] shards along dim 0, RMSNorm(x) = x * rsqrt(mean(x**2) + eps) * g in fp32
    rounded once to the activation dtype, and W_r is this rank's [N/p, H] rows of W;
  - column: y_r = gather(x) @ W_r^T;
  - row: y_r = rows [r T/p, (r+1) T/p) of sum over ranks of x_s @ W_s^T, where x_s [T, B, K/p]
    and W_s [N, K/p] are rank s's slices (a reduce-scatter along dim 0);
  - `InferenceLinear` (duplicated): y = x @ W^T.
  Each GEMM accumulates in fp32 and rounds once; outputs are bf16 [*, B, N or N/p]. Inputs are
  read only.
- `_te_rms_norm_kernel(x, w, eps)` returns RMSNorm(x) with the plain gain (zero-centered gamma
  is forbidden for inference-optimized models) on every path.
- Batch invariance under batch-invariant mode: at TP 1 each output row depends only on its input
  row on the triton/deepgemm backends (their norm and GEMM replacements); on te_native the native
  TE norm is invariant across token counts that are multiples of 64 (the backend's documented
  "64-multiple alignment discipline"; te_native GEMMs are checked in `test_te_native_gemm.py`).

Reference: fp64 from the formulas above, with the full W and x built on every rank. Trusted
implementation: the same module in training mode (TE's own LayerNormLinear / Linear forward with
its sequence-parallel all-gather and reduce-scatter, outside batch-invariant mode). For
`_te_rms_norm_kernel`, torch's fused `F.rms_norm` on fp32 inputs rounded once. Tolerance class:
floating point, bounded per row (default factor 2); exact for batch invariance and read-only
inputs.

Structure (read after the contract): at TP 1 the column layers skip communication; at TP > 1 the
column layers try NVLS (`inference_disable_triton_nvls_kernels`, NVLS eligibility, a symmetric
buffer) and otherwise all-gather with NCCL; the row layer reduce-scatters with NCCL unless NVLS
is allowed, bf16, eligible, buffered and batch-invariant mode is off. `_te_rms_norm_kernel` passes
`sm_margin=16` and `zero_centered_gamma=False` to `tex.rmsnorm_fwd`, or calls
`rmsnorm_batch_invariant` under triton/deepgemm. Covered: TP 1 and 2 (NCCL forced by
`inference_disable_triton_nvls_kernels`, and by batch-invariant mode for the row layer), T from 1
to 1000 tokens (odd counts at TP 1), [T, 1, H] inputs as dynamic batching passes them, the four
layer kinds, the norm on every path, batch invariance on triton/deepgemm (all T) and te_native
(multiples of 64), and Llama-3.1-8B's QKV and FFN widths.

Main vs open PRs: all cases are expected to pass on main, except
`test_rows_do_not_depend_on_tokens_deepgemm`, which fails where DeepGEMM forwards to
cuBLASLt (as `test_deepgemm_dense.py::test_rows_do_not_depend_on_m`) and passes with #8867.
#8561 (d1) and #8704 change the fused residual wiring, not these paths.

Hardware: H100, 1 GPU (TP 1) and 2 GPUs (TP 2), with Transformer Engine.
"""

import pytest
import torch
import torch.nn.functional as F

import megatron.core.tensor_parallel.inference_layers as inference_layers
from megatron.core import parallel_state
from megatron.core.inference.symmetric_memory import SymmetricMemoryManager
from megatron.core.inference.utils import InferenceMode
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.custom_layers import batch_invariant_kernels as bik
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.inference.oracles import oracle_utils as ou
from tests.unit_tests.test_utilities import Utils

pytestmark = [ou.requires_sm(ou.HOPPER)]

_DEVICE = "cuda"
EPS = 1e-5


@pytest.fixture(autouse=True)
def _batch_invariant_mode_off():
    """Trusted outputs come from TE's own forward: the mode must be off around each test."""
    assert not bik.is_batch_invariant_mode_enabled(), "batch-invariant mode leaked into the test"
    yield
    assert not bik.is_batch_invariant_mode_enabled(), "the test left batch-invariant mode on"


@pytest.fixture
def parallel(request):
    tp = request.param
    if Utils.world_size < tp:
        pytest.skip(f"needs {tp} ranks (run with nproc >= {tp}); the manifest selects it so")
    Utils.initialize_model_parallel(tensor_model_parallel_size=tp)
    model_parallel_cuda_manual_seed(123, force_reset_rng=True)
    SymmetricMemoryManager.destroy("tp")
    yield tp
    SymmetricMemoryManager.destroy("tp")
    Utils.destroy_model_parallel()


def make_config(tp, hidden, *, nvls=False):
    return TransformerConfig(
        num_layers=1,
        hidden_size=hidden,
        num_attention_heads=8,
        tensor_model_parallel_size=tp,
        sequence_parallel=tp > 1,
        transformer_impl="inference_optimized",
        normalization="RMSNorm",
        layernorm_epsilon=EPS,
        add_bias_linear=False,
        params_dtype=torch.bfloat16,
        bf16=True,
        use_cpu_initialization=False,
        inference_disable_triton_nvls_kernels=not nvls,
    )


def cpu_bf16(shape, seed, scale=1.0):
    """The same values on every rank (CPU generator)."""
    g = ou.make_generator(seed)
    return (torch.randn(*shape, generator=g) * scale).to(torch.bfloat16).to(_DEVICE)


def build(kind, config, in_size, out_size):
    """A layer of ``kind`` with weights cut from a full W [out, in] (and norm gain) built on
    every rank: returns (layer, W, gain)."""
    tp = parallel_state.get_tensor_model_parallel_world_size()
    rank = parallel_state.get_tensor_model_parallel_rank()
    init = lambda w: torch.nn.init.zeros_(w)
    common = dict(config=config, init_method=init, bias=False, skip_bias_add=False)
    if kind == "ln_column":
        layer = inference_layers.InferenceLayerNormColumnParallelLinear(
            in_size, out_size, gather_output=False, is_expert=False, **common
        )
    elif kind == "column":
        layer = inference_layers.InferenceColumnParallelLinear(
            in_size, out_size, gather_output=False, is_expert=False, **common
        )
    elif kind == "row":
        layer = inference_layers.InferenceRowParallelLinear(
            in_size, out_size, input_is_parallel=True, is_expert=False, **common
        )
    else:
        layer = inference_layers.InferenceLinear(
            in_size,
            out_size,
            parallel_mode="duplicated",
            skip_weight_param_allocation=False,
            **common,
        )
    w = cpu_bf16((out_size, in_size), seed=11, scale=in_size**-0.5)
    gain = (1.0 + 0.25 * cpu_bf16((in_size,), seed=12).float()).to(torch.bfloat16)
    with torch.no_grad():
        if kind in ("ln_column", "column"):
            layer.weight.copy_(w.chunk(tp, dim=0)[rank])
        elif kind == "row":
            layer.weight.copy_(w.chunk(tp, dim=1)[rank])
        else:
            layer.weight.copy_(w)
        if kind == "ln_column":
            layer.layer_norm_weight.copy_(gain)
    return layer.cuda().eval(), w, gain


def rms_norm(x, g, eps):
    return x * torch.rsqrt((x * x).mean(dim=-1, keepdim=True) + eps) * g


def layer_reference(kind, x_full, w_rank, gain, eps):
    """fp64: the full-sequence output of this rank's weights (row: before the reduce-scatter)."""
    if kind == "ln_column":
        x_full = rms_norm(x_full, gain, eps)
    return x_full @ w_rank.t()


def layer_input(kind, x_full, tp, rank):
    """What this rank's layer receives: a sequence shard (column kinds at TP > 1), a feature
    shard (row), or everything (TP 1, duplicated)."""
    if kind == "row":
        return x_full.chunk(tp, dim=-1)[rank].contiguous()
    if kind in ("ln_column", "column") and tp > 1:
        return x_full.chunk(tp, dim=0)[rank].contiguous()
    return x_full


def run_layer(layer, x, backend=None):
    before = x.clone()
    with torch.no_grad(), InferenceMode.active():
        if backend is None:
            out, _ = layer(x)
        else:
            if backend == "deepgemm":
                assert bik.HAVE_DEEPGEMM_BF16, "the deepgemm backend needs DeepGEMM"
            with bik.set_batch_invariant_mode(True, backend=backend):
                out, _ = layer(x)
    ou.assert_exact(x, before, what="the input is read-only")
    return out


def trusted_forward(layer, x):
    """TE's own forward of the same module (training mode), with its own TP communication."""
    layer.train()
    try:
        with torch.no_grad():
            out, _ = layer(x)
    finally:
        layer.eval()
    return out


def check_layer(kind, layer, w, gain, x_full, out, trusted):
    tp = parallel_state.get_tensor_model_parallel_world_size()
    rank = parallel_state.get_tensor_model_parallel_rank()
    if kind == "row":
        w_rank = w
        reference = lambda x, w_, g: layer_reference(kind, x, w_, g, EPS).chunk(tp, dim=0)[rank]
    else:
        w_rank = w.chunk(tp, dim=0)[rank] if kind != "linear" else w
        reference = lambda x, w_, g: layer_reference(kind, x, w_, g, EPS)
    assert out.shape == trusted.shape and out.dtype == trusted.dtype, f"{kind}: {out.shape}"
    ref = ou.fp64_reference(reference, x_full, w_rank, gain)
    ou.assert_close_to_reference(
        out.reshape(-1, out.shape[-1]),
        ref.reshape(-1, ref.shape[-1]),
        trusted.reshape(-1, trusted.shape[-1]),
        dim=-1,
        what=f"{kind} at TP {tp}",
    )


KINDS = ("ln_column", "column", "row", "linear")


@pytest.mark.parametrize("parallel", [1], indirect=True, ids=["tp1"])
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("tokens", [1, 7, 64, 65, 1000])
def test_layers_match_fp64_tp1(kind, tokens, parallel):
    """TP 1: the norm and torch.matmul (no communication), every layer kind, [T, 1, H] inputs."""
    config = make_config(1, 1024)
    layer, w, gain = build(kind, config, 1024, 1536)
    x_full = cpu_bf16((tokens, 1, 1024), seed=tokens)
    out = run_layer(layer, layer_input(kind, x_full, 1, 0))
    trusted = trusted_forward(layer, layer_input(kind, x_full, 1, 0))
    check_layer(kind, layer, w, gain, x_full, out, trusted)


@pytest.mark.parametrize("parallel", [2], indirect=True, ids=["tp2"])
@pytest.mark.parametrize("kind", ["ln_column", "column", "row"])
@pytest.mark.parametrize("path", ["nccl", "bi_triton", "bi_te_native"])
@pytest.mark.parametrize("tokens", [2, 130, 1000])
def test_layers_match_fp64_tp2(kind, path, tokens, parallel):
    """TP 2 on the NCCL fallbacks (NVLS kernels disabled; batch-invariant mode also forces NCCL
    for the row layer): every rank's output against fp64 of the full problem."""
    tp, rank = 2, parallel_state.get_tensor_model_parallel_rank()
    config = make_config(tp, 1024)
    in_size, out_size = (1536, 1024) if kind == "row" else (1024, 1536)
    layer, w, gain = build(kind, config, in_size, out_size)
    x_full = cpu_bf16((tokens, 1, in_size), seed=tokens)
    x = layer_input(kind, x_full, tp, rank)
    backend = {"nccl": None, "bi_triton": "triton", "bi_te_native": "te_native"}[path]
    out = run_layer(layer, x, backend)
    trusted = trusted_forward(layer, x)
    check_layer(kind, layer, w, gain, x_full, out, trusted)


@pytest.mark.parametrize("parallel", [1], indirect=True, ids=["tp1"])
@pytest.mark.parametrize("backend", [None, "te_native", "triton", "deepgemm"])
@pytest.mark.parametrize("hidden", [1024, 4096, 7168])
def test_norm_kernel_matches_fp64(backend, hidden, parallel):
    """`_te_rms_norm_kernel` on every path (TE's kernel with BI off and on te_native, the
    batch-invariant RMSNorm on triton/deepgemm): RMSNorm with the plain gain, rounded once."""
    x = cpu_bf16((129, 1, hidden), seed=hidden) * 3
    g = (1.0 + 0.25 * cpu_bf16((hidden,), seed=5).float()).to(torch.bfloat16)
    with torch.no_grad():
        if backend is None:
            out = inference_layers._te_rms_norm_kernel(x, g, EPS)
        else:
            if backend == "deepgemm":
                assert bik.HAVE_DEEPGEMM_BF16, "the deepgemm backend needs DeepGEMM"
            with bik.set_batch_invariant_mode(True, backend=backend):
                out = inference_layers._te_rms_norm_kernel(x, g, EPS)
    assert out.shape == x.shape and out.dtype == x.dtype, f"{backend}: {out.shape}"
    trusted = F.rms_norm(x.float(), (hidden,), g.float(), EPS).to(x.dtype)
    ou.assert_close_to_reference(
        out.view(-1, hidden),
        ou.fp64_reference(rms_norm, x, g, EPS).view(-1, hidden),
        trusted.view(-1, hidden),
        dim=-1,
        what=f"norm kernel ({backend})",
    )


def _probe_rows_equal(fn, probe, filler, sizes, what):
    base = fn(probe)
    for tokens in sizes:
        rest = tokens - probe.shape[0]
        for start in sorted({0, min(rest // 2 | 1, rest), rest}):
            x = filler[:tokens].clone()
            x[start : start + probe.shape[0]] = probe
            out = fn(x)[start : start + probe.shape[0]]
            ou.assert_exact(out, base, what=f"{what}: T={tokens} start={start}")


def check_rows_do_not_depend_on_tokens(kind, backend):
    """TP 1: four probe tokens among 4 ... 8192 tokens give the same bits. 8192 tokens is where
    cuBLASLt, which DeepGEMM forwards to on main, changes algorithm at K = N = 4096 on H100
    (`test_deepgemm_dense.py`, pdx 7986825); up to 4096 it did not."""
    config = make_config(1, 4096)
    layer, _, _ = build(kind, config, 4096, 4096)
    probe, filler = cpu_bf16((4, 1, 4096), seed=21), cpu_bf16((8192, 1, 4096), seed=22)
    _probe_rows_equal(
        lambda x: run_layer(layer, x, backend),
        probe,
        filler,
        (4, 5, 15, 33, 1000, 4096, 8192),
        f"{kind} {backend}",
    )


@pytest.mark.parametrize("parallel", [1], indirect=True, ids=["tp1"])
@pytest.mark.parametrize("kind", ["ln_column", "row"])
def test_rows_do_not_depend_on_tokens(kind, parallel):
    """TP 1 under the triton backend (norm and GEMM replaced): batch-invariant rows."""
    check_rows_do_not_depend_on_tokens(kind, "triton")


@pytest.mark.parametrize("parallel", [1], indirect=True, ids=["tp1"])
@pytest.mark.parametrize("kind", ["ln_column", "row"])
def test_rows_do_not_depend_on_tokens_deepgemm(kind, parallel):
    """TP 1 under the deepgemm backend. Fails on main where DeepGEMM forwards to cuBLASLt
    (#8867 makes it pass)."""
    check_rows_do_not_depend_on_tokens(kind, "deepgemm")


@pytest.mark.parametrize("parallel", [1], indirect=True, ids=["tp1"])
def test_te_native_norm_rows_do_not_depend_on_aligned_tokens(parallel):
    """te_native keeps TE's RMSNorm, documented as invariant across token counts that are
    multiples of 64: four probe tokens among 64 ... 8192 such tokens give the same bits."""
    g = (1.0 + 0.25 * cpu_bf16((7168,), seed=23).float()).to(torch.bfloat16)
    probe, filler = cpu_bf16((4, 1, 7168), seed=24), cpu_bf16((8192, 1, 7168), seed=25)

    def norm(x):
        with torch.no_grad(), bik.set_batch_invariant_mode(True, backend="te_native"):
            return inference_layers._te_rms_norm_kernel(x, g, EPS)

    _probe_rows_equal(norm, probe, filler, (64, 128, 960, 4096, 8192), "te_native norm")


@pytest.mark.parametrize("parallel", [2], indirect=True, ids=["tp2"])
@pytest.mark.parametrize("kind", ["ln_column", "row"])
def test_production_widths_tp2(kind, parallel):
    """Llama-3.1-8B at TP 2 on the NCCL fallbacks: the fused QKV projection (4096 -> 6144) and the
    FFN down projection (14336 -> 4096), 4096 tokens."""
    tp, rank = 2, parallel_state.get_tensor_model_parallel_rank()
    config = make_config(tp, 4096)
    in_size, out_size = (14336, 4096) if kind == "row" else (4096, 6144)
    layer, w, gain = build(kind, config, in_size, out_size)
    x_full = cpu_bf16((4096, 1, in_size), seed=31)
    x = layer_input(kind, x_full, tp, rank)
    out = run_layer(layer, x)
    check_layer(kind, layer, w, gain, x_full, out, trusted_forward(layer, x))


@pytest.mark.sanitizer
@pytest.mark.parametrize("parallel", [1], indirect=True, ids=["tp1"])
def test_sanitizer_case(parallel):
    """Small case for compute-sanitizer: the LayerNorm-column layer at TP 1, 7 tokens."""
    config = make_config(1, 1024)
    layer, w, gain = build("ln_column", config, 1024, 512)
    x_full = cpu_bf16((7, 1, 1024), seed=41)
    out = run_layer(layer, x_full)
    check_layer("ln_column", layer, w, gain, x_full, out, trusted_forward(layer, x_full))
