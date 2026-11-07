# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Tests of the temporary stand-in for FlashInfer's `num_active_tokens`; delete with the module.

The forwarding tests run on CPU against fake kernels; the emulation and replay tests need a GPU
and Triton; the parity tests run the real kernels and need Blackwell and FlashInfer.
"""

import pytest
import torch

import megatron.core.inference.moe.flashinfer_active_tokens as shim
from megatron.core.inference.moe.flashinfer_mxfp8 import (
    HAVE_FLASHINFER_ROUTED_MXFP8,
    _unwrap_output,
    flashinfer_routed_mxfp8_moe,
    prepare_routed_mxfp8_weights,
)
from megatron.core.inference.quantization.mxfp8_tensor import MXFP8Tensor
from tests.unit_tests.determinism.kernels.harness import assert_replays_bit_exact, seeded

requires_cuda_triton = pytest.mark.skipif(
    not (torch.cuda.is_available() and shim.HAVE_TRITON), reason="needs a GPU and Triton"
)
_BLACKWELL = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 10
requires_blackwell_cutlass = pytest.mark.skipif(
    not (_BLACKWELL and shim.HAVE_TRITON and shim._cutlass_fused_moe is not None),
    reason="needs Blackwell, Triton and FlashInfer",
)
requires_blackwell_routed = pytest.mark.skipif(
    not (_BLACKWELL and shim.HAVE_TRITON and HAVE_FLASHINFER_ROUTED_MXFP8),
    reason="needs Blackwell, Triton and FlashInfer's routed MXFP8 MoE",
)
BF16 = dict(dtype=torch.bfloat16)
RELU2 = 6  # flashinfer ActivationType.Relu2
# The routed kernel's arguments that the fake-kernel tests do not look at.
ROUTED_ARGS = dict(
    routing_bias=None,
    hidden_states=None,
    hidden_states_scale=None,
    gemm1_weights=None,
    gemm1_weights_scale=None,
    gemm2_weights=None,
    gemm2_weights_scale=None,
    num_experts=8,
    n_group=None,
    topk_group=None,
    intermediate_size=16,
    local_expert_offset=2,
    local_num_experts=2,
    routed_scaling_factor=None,
    fp8_quantization_type=1,
    activation_type=RELU2,
)


def _count(n, device="cpu"):
    return torch.tensor([n], dtype=torch.int32, device=device)


def _cutlass(**kwargs):
    x, ids, probs = (
        torch.empty(4, 8, **BF16),
        torch.zeros(4, 2, dtype=torch.int32),
        torch.zeros(4, 2),
    )
    fc1, fc2 = torch.empty(2, 16, 8, **BF16), torch.empty(2, 8, 16, **BF16)
    return shim.cutlass_fused_moe(
        x,
        ids,
        probs,
        fc1,
        fc2,
        torch.float32,
        activation_type=RELU2,
        ep_size=4,
        ep_rank=1,
        **kwargs,
    )


def _routed(**kwargs):
    ids = torch.zeros(4, 2, dtype=torch.int32)
    return shim.trtllm_fp8_block_scale_routed_moe(topk_ids=ids, top_k=2, **ROUTED_ARGS, **kwargs)


KERNELS = {
    "cutlass": ("_cutlass_fused_moe", "CUTLASS_MOE_HAS_ACTIVE_TOKENS", _cutlass),
    "routed": ("_trtllm_fp8_block_scale_routed_moe", "ROUTED_MOE_HAS_ACTIVE_TOKENS", _routed),
}


def test_probe_reads_the_kernel_signature():
    assert shim._accepts_num_active_tokens(lambda x, num_active_tokens=None: x)
    assert not shim._accepts_num_active_tokens(lambda x: x)
    assert not shim._accepts_num_active_tokens(None)


@pytest.mark.parametrize("flashinfer_has_count", [False, True])
@pytest.mark.parametrize("kernel", list(KERNELS))
def test_count_is_forwarded_only_when_flashinfer_has_it(monkeypatch, kernel, flashinfer_has_count):
    """No count: the call is forwarded without the argument. A count: forwarded as-is if accepted."""
    real, flag, call = KERNELS[kernel]
    seen = {}
    monkeypatch.setattr(
        shim, real, lambda *args, **kwargs: seen.update(kwargs) or [kwargs.get("output")]
    )
    monkeypatch.setattr(shim, flag, flashinfer_has_count)

    call()
    assert "num_active_tokens" not in seen
    if flashinfer_has_count:
        out, count = torch.empty(4, 8), _count(3)
        call(output=out, num_active_tokens=count)
        assert seen["output"] is out and seen["num_active_tokens"] is count
        assert "enable_alltoall" not in seen and seen.get("do_finalize", True) is True


@requires_cuda_triton
def test_cutlass_emulation_combines_the_live_rows(monkeypatch):
    """Without the argument: non-filling finalize, live rows copied to fp32, non-local rows zeroed."""
    rows, valid, hidden, topk = 8, 5, 96, 6
    kernel_rows = torch.randn(rows, hidden, device="cuda", **BF16)
    seen = {}
    monkeypatch.setattr(
        shim, "_cutlass_fused_moe", lambda *a, **k: seen.update(k, dtype=a[5]) or [kernel_rows]
    )
    monkeypatch.setattr(shim, "CUTLASS_MOE_HAS_ACTIVE_TOKENS", False)
    # ep_rank 2 with 2 experts per rank owns experts [4, 6): rows 0, 2, 3 route there, rows 1 and 4
    # do not, and row 6 does but lies beyond the count.
    routing = torch.randint(8, 16, (rows, topk), device="cuda", dtype=torch.int32)
    routing[0, 2], routing[2, 5], routing[3, 0], routing[6, 1] = 4, 5, 4, 4
    x, probs = torch.randn(rows, hidden, device="cuda", **BF16), torch.rand(
        rows, topk, device="cuda"
    )
    fc1, fc2 = torch.empty(2, 16, hidden, device="cuda", **BF16), torch.empty(
        2, hidden, 16, device="cuda", **BF16
    )
    out = torch.full((rows, hidden), 7.0, device="cuda")

    result = shim.cutlass_fused_moe(
        x,
        routing,
        probs,
        fc1,
        fc2,
        torch.float32,
        activation_type=RELU2,
        output=out,
        num_active_tokens=_count(valid, "cuda"),
        ep_size=4,
        ep_rank=2,
    )

    assert result[0] is out and seen["enable_alltoall"] is True and seen["dtype"] is torch.bfloat16
    assert "num_active_tokens" not in seen and "output" not in seen
    assert torch.equal(out[[0, 2, 3]], kernel_rows.float()[[0, 2, 3]])
    assert not out[[1, 4]].any() and torch.all(out[valid:] == 7.0)


@requires_cuda_triton
def test_routed_emulation_finalizes_the_live_rows(monkeypatch):
    """Without the argument: unfinalized kernel, live rows combined with the packed bf16 weights."""
    rows, valid, topk, hidden = 16, 10, 2, 160
    torch.manual_seed(3)
    experts = torch.randint(0, 8, (rows, topk), device="cuda", dtype=torch.int32)
    experts[1] = 0  # a live token with no local expert; local experts are [2, 4)
    experts[valid:] = 2  # stale rows that look routable
    probs = torch.rand(rows, topk, device="cuda")
    topk_ids = (experts << 16) | (
        probs.to(torch.bfloat16).view(torch.int16).to(torch.int32) & 0xFFFF
    )
    is_local = ((experts >= 2) & (experts < 4)).flatten()
    expanded_idx = torch.where(is_local, torch.cumsum(is_local.int(), 0) - 1, -1).int()
    gemm2_rows = torch.randn(int(is_local.sum()) + 3, 256, device="cuda", **BF16)
    seen = {}
    monkeypatch.setattr(
        shim,
        "_trtllm_fp8_block_scale_routed_moe",
        lambda **k: seen.update(k) or [gemm2_rows, None, expanded_idx],
    )
    monkeypatch.setattr(shim, "ROUTED_MOE_HAS_ACTIVE_TOKENS", False)
    out = torch.full((rows, hidden), 7.0, device="cuda")

    result = shim.trtllm_fp8_block_scale_routed_moe(
        topk_ids=topk_ids,
        top_k=topk,
        output=out,
        num_active_tokens=_count(valid, "cuda"),
        **ROUTED_ARGS,
    )

    assert result is out and seen["do_finalize"] is False
    assert "num_active_tokens" not in seen and "output" not in seen
    weights = torch.where(is_local.view(rows, topk), probs.to(torch.bfloat16).float(), 0.0)
    gathered = gemm2_rows[expanded_idx.clamp(min=0).view(rows, topk), :hidden].float()
    expected = (weights[..., None] * gathered).sum(1)
    torch.testing.assert_close(out[:valid], expected[:valid], rtol=1e-6, atol=1e-6)
    assert not out[1].any() and torch.all(out[valid:] == 7.0)


@requires_cuda_triton
@pytest.mark.parametrize("kernel", ["finalize_routed_rows", "combine_valid_rows"])
def test_stand_in_kernels_replay(kernel):
    """Both live-row stand-ins replay bit-exactly."""
    seeded()
    rows, valid, hidden, topk = 4096, 2500, 2688, 6
    count = _count(valid, "cuda")
    ids = torch.randint(0, 128, (rows, topk), device="cuda", dtype=torch.int32)
    out = torch.zeros(rows, hidden, device="cuda")
    if kernel == "finalize_routed_rows":
        src = torch.randn(20000, hidden, device="cuda", **BF16)
        idx = torch.randint(-1, 20000, (rows * topk,), device="cuda").int()
        weights = torch.rand(rows, topk, device="cuda").to(torch.bfloat16).view(torch.int16)
        ids = (ids << 16) | (weights.to(torch.int32) & 0xFFFF)
        fn = lambda src: shim.finalize_routed_rows(src, idx, ids, count, out)[:valid]
    else:
        src = torch.randn(rows, hidden, device="cuda", **BF16)
        fn = lambda src: shim.combine_valid_rows(
            src, ids, count, out, local_expert_offset=4, num_local_experts=2
        )[:valid]
    assert_replays_bit_exact(fn, (src,), backward=False, what=kernel)


def _routed_mxfp8_weight(weight):
    """Quantize a bf16 expert stack per expert and repack it into TRT-LLM's routed layout."""
    experts = [MXFP8Tensor.from_bf16(expert, backend="triton") for expert in weight]
    stacked = MXFP8Tensor(
        data=torch.stack([expert.data for expert in experts]).contiguous(),
        scale=torch.stack([expert.scale for expert in experts]).contiguous(),
        backend="triton",
        dtype=torch.bfloat16,
    )
    return prepare_routed_mxfp8_weights(stacked)


@requires_blackwell_cutlass
def test_cutlass_emulation_matches_the_kernel_finalize(monkeypatch):
    """The real kernel's non-filling finalize plus `combine_valid_rows` equals its own filling
    finalize on the live rows; rows beyond the count stay untouched."""
    from flashinfer.fused_moe import ActivationType

    monkeypatch.setattr(shim, "CUTLASS_MOE_HAS_ACTIVE_TOKENS", False)
    seeded()
    rows, valid, hidden, inter, local_experts, topk = 64, 40, 256, 256, 2, 2
    x = torch.randn(rows, hidden, device="cuda", **BF16)
    # ep_rank 1 of 2 owns experts [2, 4); rows 0 and 1 route only to remote experts.
    routing = torch.randint(0, 2 * local_experts, (rows, topk), device="cuda", dtype=torch.int32)
    routing[:2] = 0
    probs = torch.rand(rows, topk, device="cuda")
    fc1 = (torch.randn(local_experts, inter, hidden, device="cuda") / hidden**0.5).to(
        torch.bfloat16
    )
    fc2 = (torch.randn(local_experts, hidden, inter, device="cuda") / inter**0.5).to(torch.bfloat16)
    common = dict(activation_type=ActivationType.Relu2, ep_size=2, ep_rank=1)

    reference = shim._cutlass_fused_moe(
        x, routing, probs, fc1, fc2, torch.bfloat16, quant_scales=None, **common
    )[0]
    out = torch.full((rows, hidden), 7.0, device="cuda")
    shim.cutlass_fused_moe(
        x,
        routing,
        probs,
        fc1,
        fc2,
        torch.float32,
        output=out,
        num_active_tokens=_count(valid, "cuda"),
        **common,
    )

    # The default top-k finalize accumulates atomically, so allow bf16 rounding.
    torch.testing.assert_close(out[:valid], reference[:valid].float(), rtol=2**-7, atol=2**-7)
    assert not reference[:2].any() and torch.all(out[valid:] == 7.0)


@requires_blackwell_routed
def test_routed_emulation_matches_the_kernel_finalize(monkeypatch):
    """The unfinalized real kernel plus `finalize_routed_rows` equals the kernel's own finalize on
    the live rows; rows beyond the count stay untouched."""
    monkeypatch.setattr(shim, "ROUTED_MOE_HAS_ACTIVE_TOKENS", False)
    kernel, calls = shim._trtllm_fp8_block_scale_routed_moe, []
    monkeypatch.setattr(
        shim,
        "_trtllm_fp8_block_scale_routed_moe",
        lambda **kwargs: calls.append(kwargs) or kernel(**kwargs),
    )
    seeded()
    num_experts, local_experts, hidden, inter, topk, rows, valid = 8, 2, 256, 256, 2, 512, 300
    fc1 = _routed_mxfp8_weight(
        (torch.randn(local_experts, inter, hidden, device="cuda") / hidden**0.5).to(torch.bfloat16)
    )
    fc2 = _routed_mxfp8_weight(
        (torch.randn(local_experts, hidden, inter, device="cuda") / inter**0.5).to(torch.bfloat16)
    )
    x = torch.randn(rows, hidden, device="cuda", **BF16)
    # Local experts are [2, 4); rows 0 and 1 route only to remote experts.
    routing = torch.randint(0, num_experts, (rows, topk), device="cuda", dtype=torch.int64)
    routing[:2] = 0
    probs = torch.rand(rows, topk, device="cuda")
    out = torch.full((rows, hidden), 7.0, device="cuda")

    returned = flashinfer_routed_mxfp8_moe(
        x,
        routing,
        probs,
        fc1,
        fc2,
        num_experts=num_experts,
        local_expert_offset=2,
        activation_type=RELU2,
        out=out,
        valid_tokens=_count(valid, "cuda"),
    )
    (unfinalized,) = calls
    assert returned is out and unfinalized["do_finalize"] is False
    reference = _unwrap_output(kernel(**dict(unfinalized, do_finalize=True)))[:, :hidden]

    torch.testing.assert_close(out[:valid], reference[:valid].float(), rtol=2**-7, atol=2**-7)
    assert not reference[:2].any() and torch.all(out[valid:] == 7.0)
