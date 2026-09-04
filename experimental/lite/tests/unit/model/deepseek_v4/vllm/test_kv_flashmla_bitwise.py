from __future__ import annotations

import importlib.util
import pytest
import torch

from megatron.lite.model.deepseek_v4.vllm.primitive.attention.module import insert_qkv


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
