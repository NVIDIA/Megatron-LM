from __future__ import annotations

import importlib.util
from unittest.mock import Mock

import pytest
import torch

from megatron.lite.model.deepseek_v4.vllm import kernels as vllm_ds4
from megatron.lite.model.deepseek_v4.vllm.kernels import (
    insert_qkv,
    sparse_attention,
)


def test_sparse_flashmla_cpu_contract(monkeypatch) -> None:
    sparse = Mock(return_value=(torch.tensor(1), torch.tensor(2)))
    monkeypatch.setattr(vllm_ds4, "flash_mla_sparse_fwd", sparse)
    q = torch.zeros(1, 4, 8)
    indices = torch.zeros(1, 1, 4, dtype=torch.int32)
    sparse_attention(q, torch.zeros(8, 1, 8), indices, sm_scale=0.5)
    sparse.assert_called_once()


def _production_ready() -> bool:
    return torch.cuda.is_available() and importlib.util.find_spec("vllm") is not None


@pytest.mark.gpus(1)
@pytest.mark.skipif(
    not _production_ready(),
    reason="requires CUDA plus official vLLM DS4 FP8 cache-insert kernels",
)
def test_official_fp8_ds_mla_kv_quant_insert_is_bitwise() -> None:
    import vllm._C_stable_libtorch  # noqa: F401

    torch.manual_seed(22)
    tokens, heads, head_dim, block_size = 2, 64, 512, 256
    token_bytes = 584
    q = torch.randn(
        tokens, heads, head_dim, dtype=torch.bfloat16, device="cuda"
    )
    kv = torch.randn(tokens, head_dim, dtype=torch.bfloat16, device="cuda")
    slots = torch.tensor([0, 1], dtype=torch.int64, device="cuda")
    positions = torch.tensor([0, 1], dtype=torch.int64, device="cuda")
    cos = torch.ones(8, 64, dtype=torch.float32, device="cuda")
    reference_cache = torch.zeros(
        1, block_size * token_bytes, dtype=torch.uint8, device="cuda"
    )
    candidate_cache = reference_cache.clone()

    reference = (
        torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
            q.clone(),
            kv.clone(),
            reference_cache,
            slots,
            positions,
            cos,
            heads,
            1e-6,
            block_size,
        )
    )
    candidate = insert_qkv(
        q.clone(),
        kv.clone(),
        candidate_cache,
        slots,
        positions,
        cos,
        eps=1e-6,
        block_size=block_size,
        padded_heads=heads,
    )
    torch.testing.assert_close(candidate, reference, rtol=0, atol=0)
    assert torch.equal(candidate_cache, reference_cache)


@pytest.mark.gpus(1)
@pytest.mark.skipif(
    not _production_ready(),
    reason="requires CUDA plus official vLLM and FlashMLA compiled dependencies",
)
def test_official_flashmla_sparse_is_bitwise() -> None:
    from vllm.v1.attention.ops.flashmla import flash_mla_sparse_fwd

    torch.manual_seed(23)
    q = torch.randn(1, 64, 576, dtype=torch.bfloat16, device="cuda")
    kv = torch.randn(8, 1, 576, dtype=torch.bfloat16, device="cuda")
    indices = torch.randint(8, (1, 1, 128), dtype=torch.int32, device="cuda")
    lengths = torch.full((1,), 128, dtype=torch.int32, device="cuda")
    sink = torch.zeros(64, dtype=torch.float32, device="cuda")
    out_ref = torch.empty(1, 64, 512, dtype=torch.bfloat16, device="cuda")
    out_candidate = torch.empty_like(out_ref)

    reference = flash_mla_sparse_fwd(
        q=q,
        kv=kv,
        indices=indices,
        sm_scale=1.0,
        attn_sink=sink,
        topk_length=lengths,
        out=out_ref,
    )
    candidate = sparse_attention(
        q,
        kv,
        indices,
        sm_scale=1.0,
        attn_sink=sink,
        topk_length=lengths,
        out=out_candidate,
    )
    for actual, expected in zip(candidate, reference, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
