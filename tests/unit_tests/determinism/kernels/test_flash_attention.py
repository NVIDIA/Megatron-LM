# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Replay paged attention dispatch, including the LSE used for attention sinks."""

from types import SimpleNamespace

import pytest
import torch

from megatron.core.transformer.attention import HAVE_FA4, SelfAttention
from tests.unit_tests.determinism.kernels.harness import assert_replays_bit_exact, seeded


@pytest.mark.parametrize("is_decode_only", [False, True])
@pytest.mark.parametrize("has_sink", [False, True])
@pytest.mark.parametrize(
    "flash_attention_version,head_dim,page_size",
    [pytest.param(4, 64, 128, id="fa4"), pytest.param(None, 8, 256, id="auto-hopper-small-head")],
)
def test_fa4_paged_attention_replays_bit_exactly(
    is_decode_only, has_sink, flash_attention_version, head_dim, page_size
):
    if not torch.cuda.is_available() or not HAVE_FA4:
        pytest.skip("requires CUDA and FlashAttention-4")
    if torch.cuda.get_device_capability()[0] not in (9, 10, 11):
        pytest.skip("requires FA4 paged attention on Hopper, Blackwell, or Rubin")
    if flash_attention_version is None and torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("requires Hopper to exercise the small-head automatic fallback")

    seeded()
    attention = object.__new__(SelfAttention)
    torch.nn.Module.__init__(attention)
    attention.config = SimpleNamespace(
        window_size=None, window_attn_skip_freq=None, attn_logit_softcapping=None
    )
    attention.layer_number = 1
    attention.batch_invariant_mode = False
    attention.flash_attention_version = flash_attention_version
    attention.train(False)

    # Few query blocks and a long KV cache exercise FA4's split heuristic; the
    # small-head case exercises Hopper's fallback with a compatible FA2 page size.
    q = torch.randn(2, 1, 4, head_dim, device="cuda", dtype=torch.bfloat16)
    num_pages = 2048 // page_size
    k = torch.randn(num_pages, page_size, 4, head_dim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    cu_seqlens_q = torch.tensor([0, 1, 2], device="cuda", dtype=torch.int32)
    cu_seqlens_k = torch.tensor([0, 1024, 2048], device="cuda", dtype=torch.int32)
    seqlens_k = torch.full((2,), 1024, device="cuda", dtype=torch.int32)
    block_table = torch.arange(num_pages, device="cuda", dtype=torch.int32).reshape(2, -1)
    offset = torch.arange(4, device="cuda", dtype=torch.float32) if has_sink else None

    @torch.inference_mode()
    def forward(query, key, value):
        output = attention.flash_decode_and_prefill(
            q=query,
            k=key,
            v=value,
            max_seqlen_q=1,
            max_seqlen_k=1024,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            seqlens_k=seqlens_k,
            block_table=block_table,
            is_decode_only=is_decode_only,
            softmax_offset=offset,
        )
        assert torch.isfinite(output).all()
        return output

    assert_replays_bit_exact(
        forward, (q, k, v), replays=3, backward=False, what="paged FlashAttention inference"
    )
