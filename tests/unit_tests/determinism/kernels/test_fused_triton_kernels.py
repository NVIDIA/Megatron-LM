# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Bit-exact replay of Megatron's own Triton kernels under ``megatron/core/fusions/``.

* ``fused_pad_routing_map`` / ``fused_indices_to_multihot``: integer routing bookkeeping with
  unique stores -- replay and agree with the torch reference.
* MLA YaRN RoPE (``fused_mla_yarn_rope_apply``): elementwise rotations under a timing-based
  ``triton.autotune``; forward and backward replay in sbhd and thd layouts.
* mHC hyper-connection kernels (``fused_mhc_kernels``): Sinkhorn, h-aggregate and h-post-BDA
  contain real ``tl.sum`` reductions under autotune, on the Triton, native (``torch.compile``)
  and, when available, cuTile backends.
"""

import pytest
import torch

from megatron.core import config as mcore_config
from megatron.core.fusions import fused_mhc_kernels
from megatron.core.fusions.fused_indices_converter import fused_indices_to_multihot
from megatron.core.fusions.fused_pad_routing_map import fused_pad_routing_map
from megatron.core.transformer.moe.moe_utils import pad_routing_map
from tests.unit_tests.determinism.kernels.harness import assert_replays_bit_exact, seeded

try:
    import triton  # noqa: F401

    HAVE_TRITON = True
except ImportError:
    HAVE_TRITON = False

try:
    from megatron.core.fusions.fused_mla_yarn_rope_apply import (
        fused_mla_rope_inplace,
        fused_mla_rope_kv_split,
        fused_mla_rope_out_of_place,
    )
except ImportError:
    fused_mla_rope_inplace = fused_mla_rope_kv_split = fused_mla_rope_out_of_place = None

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and HAVE_TRITON), reason="needs a GPU and Triton"
)


@pytest.fixture
def experimental_enabled():
    """Force the experimental flag on (the fused routing kernels are experimental features)."""
    prev = mcore_config.ENABLE_EXPERIMENTAL
    mcore_config.ENABLE_EXPERIMENTAL = True
    yield
    mcore_config.ENABLE_EXPERIMENTAL = prev


def _topk_routing_map(num_tokens, num_experts, topk):
    logits = torch.randn(num_tokens, num_experts, device="cuda")
    idx = logits.topk(topk, dim=-1).indices
    routing_map = torch.zeros(num_tokens, num_experts, dtype=torch.bool, device="cuda")
    routing_map.scatter_(1, idx, True)
    return routing_map


# --- routing map padding ------------------------------------------------------------------


@pytest.mark.parametrize("num_tokens", [8192, 4097])
@pytest.mark.parametrize("pad_multiple", [32, 256])
def test_fused_pad_routing_map_replays_and_matches_torch(
    experimental_enabled, num_tokens, pad_multiple
):
    seeded()
    routing_map = _topk_routing_map(num_tokens, 64, 8)
    fused, _ = assert_replays_bit_exact(
        lambda m: fused_pad_routing_map(m, pad_multiple),
        (routing_map,),
        backward=False,
        what="fused_pad_routing_map",
    )
    torch_out, _ = assert_replays_bit_exact(
        lambda m: pad_routing_map(m, pad_multiple),
        (routing_map,),
        backward=False,
        what="pad_routing_map",
    )
    assert torch.equal(fused["out"].to(torch_out["out"].dtype), torch_out["out"])


# --- indices <-> multihot ----------------------------------------------------------------


def test_fused_indices_to_multihot_replays_fwd_bwd(experimental_enabled):
    seeded()
    num_tokens, topk, num_local_experts = 16384, 8, 32
    # Unique experts per row, a random subset of slots masked with -1.
    experts = torch.rand(num_tokens, 64, device="cuda").topk(topk, dim=-1).indices
    experts = experts.to(torch.int32)
    experts[experts >= num_local_experts] = -1
    probs = torch.rand(num_tokens, topk, device="cuda") * (experts != -1)
    probs.requires_grad_(True)

    def fn(indices, probs):
        multihot, probs_in_multihot = fused_indices_to_multihot(indices, probs, num_local_experts)
        return multihot, probs_in_multihot

    assert_replays_bit_exact(fn, (experts, probs), replays=3, what="fused_indices_to_multihot")


# --- MLA YaRN RoPE --------------------------------------------------------------------------


def _yarn_cos_sin(emb_dim, seqlen, dtype):
    from megatron.core.models.common.embeddings.yarn_rotary_pos_embedding import YarnRotaryEmbedding

    rope = YarnRotaryEmbedding(emb_dim, original_max_position_embeddings=seqlen)
    freqs, mscale = rope(seqlen, 0)
    return (torch.cos(freqs) * mscale).to(dtype), (torch.sin(freqs) * mscale).to(dtype)


@pytest.mark.skipif(fused_mla_rope_out_of_place is None, reason="fused MLA RoPE unavailable")
@pytest.mark.parametrize("layout", ["sbhd", "thd"])
@pytest.mark.parametrize("variant", ["out_of_place", "inplace"])
def test_fused_mla_rope_q_replays_fwd_bwd(layout, variant):
    seeded()
    heads, nope_dim, emb_dim = 32, 128, 64
    dtype = torch.bfloat16
    if layout == "sbhd":
        seqlen, batch = 2048, 2
        cu_seqlens = None
        t = torch.randn(seqlen, batch, heads, nope_dim + emb_dim, device="cuda", dtype=dtype)
    else:
        bounds = [0, 1000, 1500, 2048, 4096]
        seqlen = max(b - a for a, b in zip(bounds, bounds[1:]))
        cu_seqlens = torch.tensor(bounds, dtype=torch.int32, device="cuda")
        t = torch.randn(bounds[-1], heads, nope_dim + emb_dim, device="cuda", dtype=dtype)
    cos, sin = _yarn_cos_sin(emb_dim, seqlen, dtype)
    t.requires_grad_(True)
    fn = fused_mla_rope_out_of_place if variant == "out_of_place" else fused_mla_rope_inplace

    def run(t):
        return fn(t, cos, sin, nope_dim, emb_dim, cu_seqlens_q=cu_seqlens)

    assert_replays_bit_exact(run, (t,), replays=3, what=f"fused_mla_rope_{variant}[{layout}]")


@pytest.mark.skipif(fused_mla_rope_kv_split is None, reason="fused MLA RoPE unavailable")
@pytest.mark.parametrize("layout", ["sbhd", "thd"])
def test_fused_mla_rope_kv_split_replays_fwd_bwd(layout):
    seeded()
    heads, k_dim, v_dim, emb_dim = 32, 128, 128, 64
    dtype = torch.bfloat16
    if layout == "sbhd":
        seqlen, batch = 2048, 2
        cu_seqlens = None
        kv = torch.randn(seqlen, batch, heads, k_dim + v_dim, device="cuda", dtype=dtype)
        k_pos_emb = torch.randn(seqlen, batch, 1, emb_dim, device="cuda", dtype=dtype)
    else:
        bounds = [0, 1000, 1500, 2048, 4096]
        seqlen = max(b - a for a, b in zip(bounds, bounds[1:]))
        cu_seqlens = torch.tensor(bounds, dtype=torch.int32, device="cuda")
        kv = torch.randn(bounds[-1], heads, k_dim + v_dim, device="cuda", dtype=dtype)
        k_pos_emb = torch.randn(bounds[-1], 1, emb_dim, device="cuda", dtype=dtype)
    cos, sin = _yarn_cos_sin(emb_dim, seqlen, dtype)
    kv.requires_grad_(True)
    k_pos_emb.requires_grad_(True)

    def run(kv, k_pos_emb):
        return fused_mla_rope_kv_split(
            kv, k_pos_emb, cos, sin, emb_dim, k_dim, v_dim, cu_seqlens_kv=cu_seqlens
        )

    assert_replays_bit_exact(
        run, (kv, k_pos_emb), replays=3, what=f"fused_mla_rope_kv_split[{layout}]"
    )


# --- mHC (hyper-connection) kernels -------------------------------------------------------

_MHC_BACKENDS = ["native", "triton"]
if fused_mhc_kernels.is_cutile_available():
    _MHC_BACKENDS.append("cutile")

S, B, N, C = 1024, 4, 4, 4096


def _mhc_rand(*shape, dtype=torch.bfloat16, grad=True):
    t = torch.empty(*shape, device="cuda", dtype=dtype).uniform_(-0.1, 0.1)
    return t.requires_grad_(grad)


@pytest.mark.parametrize("backend", _MHC_BACKENDS)
def test_mhc_sinkhorn_replays_fwd_bwd(backend):
    if backend == "triton" and not fused_mhc_kernels.is_triton_available():
        pytest.skip("Triton mHC backend unavailable")
    seeded()
    logits = _mhc_rand(S, B, N, N)
    assert_replays_bit_exact(
        lambda l: fused_mhc_kernels.fused_sinkhorn(l, 20, 1e-6, backend=backend),
        (logits,),
        replays=3,
        what=f"mhc sinkhorn[{backend}]",
    )


@pytest.mark.parametrize("backend", _MHC_BACKENDS)
def test_mhc_h_aggregate_replays_fwd_bwd(backend):
    if backend == "triton" and not fused_mhc_kernels.is_triton_available():
        pytest.skip("Triton mHC backend unavailable")
    seeded()
    x = _mhc_rand(S, B, N, C)
    h_pre = _mhc_rand(S, B, N)
    assert_replays_bit_exact(
        lambda x, h: fused_mhc_kernels.fused_h_aggregate(x, h, backend=backend),
        (x, h_pre),
        replays=3,
        what=f"mhc h_aggregate[{backend}]",
    )


@pytest.mark.parametrize("backend", _MHC_BACKENDS)
@pytest.mark.parametrize("with_bias", [False, True])
def test_mhc_h_post_bda_replays_fwd_bwd(backend, with_bias):
    if backend == "triton" and not fused_mhc_kernels.is_triton_available():
        pytest.skip("Triton mHC backend unavailable")
    seeded()
    h_res = _mhc_rand(S, B, N, N)
    orig = _mhc_rand(S, B, N, C)
    h_post = _mhc_rand(S, B, N)
    x = _mhc_rand(S, B, C)
    bias = _mhc_rand(C) if with_bias else None

    def run(h_res, orig, h_post, x, bias):
        return fused_mhc_kernels.fused_h_post_bda(h_res, orig, h_post, x, bias, backend=backend)

    assert_replays_bit_exact(
        run, (h_res, orig, h_post, x, bias), replays=3, what=f"mhc h_post_bda[{backend}]"
    )


@pytest.mark.parametrize("backend", [b for b in _MHC_BACKENDS if b != "triton"])
def test_mhc_proj_rms_compute_h_replays_fwd_bwd(backend):
    """Projection + RMS + compute-h: cuTile fused kernel or the native ``torch.compile`` path."""
    seeded()
    n = N
    x = _mhc_rand(S * B, n * C)
    weight = _mhc_rand(n * n + 2 * n, n * C, dtype=torch.float32)
    alpha_pre = _mhc_rand(1, dtype=torch.float32)
    alpha_post = _mhc_rand(1, dtype=torch.float32)
    alpha_res = _mhc_rand(1, dtype=torch.float32)
    bias = _mhc_rand(n * n + 2 * n, dtype=torch.float32)

    def run(x, weight, alpha_pre, alpha_post, alpha_res, bias):
        return fused_mhc_kernels.fused_proj_rms_compute_h(
            x, weight, alpha_pre, alpha_post, alpha_res, bias, n, backend=backend
        )

    assert_replays_bit_exact(
        run,
        (x, weight, alpha_pre, alpha_post, alpha_res, bias),
        replays=3,
        what=f"mhc proj_rms_compute_h[{backend}]",
    )
