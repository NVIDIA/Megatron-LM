from __future__ import annotations

import importlib.util
from unittest.mock import Mock

import pytest
import torch

from megatron.lite.model.deepseek_v4.vllm import kernels as vllm_ds4
from megatron.lite.model.deepseek_v4.vllm.kernels import (
    DS4KVInsertAdapter,
    FlashMLAAdapter,
    FusedQKVRMSNormAdapter,
)


def test_qkv_norm_cpu_contract_calls_official_symbol(monkeypatch) -> None:
    q, kv = torch.randn(2, 12).split((8, 4), dim=-1)
    assert not q.is_contiguous() and not kv.is_contiguous()
    assert q.stride(-1) == kv.stride(-1) == 1
    qw, kw = torch.ones(8), torch.ones(4)
    expected = (q + 1, kv + 1)
    kernel = Mock(return_value=expected)
    monkeypatch.setattr(vllm_ds4, "_symbol", lambda module, name: kernel)
    actual = FusedQKVRMSNormAdapter()(q, kv, qw, kw, 1e-6)
    assert all(got is want for got, want in zip(actual, expected, strict=True))
    kernel.assert_called_once_with(q, kv, qw, kw, 1e-6)


def test_kv_insert_cpu_contract_calls_exact_custom_op(monkeypatch) -> None:
    q = torch.zeros(2, 3, 8, dtype=torch.bfloat16)
    kernel = Mock(return_value=q)
    monkeypatch.setattr(vllm_ds4, "_op", lambda namespace, name: kernel)
    kv = torch.zeros(2, 8, dtype=torch.bfloat16)
    cache = torch.zeros(2, 4 * 584, dtype=torch.uint8)
    result = DS4KVInsertAdapter("fp8_ds_mla")(
        q,
        kv,
        cache,
        torch.tensor([0, 1], dtype=torch.int64),
        torch.tensor([0, 1], dtype=torch.int64),
        torch.zeros(16, 16, dtype=torch.float32),
        eps=1e-6,
        block_size=4,
        padded_heads=3,
    )
    assert result is q
    assert kernel.call_count == 1
    assert kernel.call_args.args[-3:] == (3, 1e-6, 4)


def test_sparse_flashmla_cpu_contract(monkeypatch) -> None:
    sparse = Mock(return_value=(torch.tensor(1), torch.tensor(2)))
    monkeypatch.setattr(vllm_ds4, "_symbol", lambda _module, _name: sparse)
    adapter = FlashMLAAdapter()
    q = torch.zeros(1, 4, 8)
    indices = torch.zeros(1, 1, 4, dtype=torch.int32)
    adapter.sparse(q, torch.zeros(8, 1, 8), indices, sm_scale=0.5)
    sparse.assert_called_once()


def _production_ready() -> bool:
    return (
        torch.cuda.is_available()
        and importlib.util.find_spec("vllm") is not None
        and importlib.util.find_spec("flash_mla") is not None
    )


@pytest.mark.gpus(1)
@pytest.mark.skipif(
    not _production_ready(),
    reason="requires CUDA plus official vLLM and FlashMLA compiled dependencies",
)
def test_official_fused_qkv_norm_is_bitwise_through_adapter() -> None:
    from vllm.models.common.ops import fused_q_kv_rmsnorm

    torch.manual_seed(19)
    q = torch.randn(3, 128, dtype=torch.bfloat16, device="cuda")
    kv = torch.randn(3, 576, dtype=torch.bfloat16, device="cuda")
    qw = torch.randn(128, dtype=torch.bfloat16, device="cuda")
    kw = torch.randn(576, dtype=torch.bfloat16, device="cuda")
    reference = fused_q_kv_rmsnorm(
        q.clone(), kv.clone(), qw.clone(), kw.clone(), 1e-6
    )
    candidate = FusedQKVRMSNormAdapter()(
        q.clone(), kv.clone(), qw.clone(), kw.clone(), 1e-6
    )
    for actual, expected in zip(candidate, reference, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.gpus(1)
@pytest.mark.skipif(
    not _production_ready(),
    reason="requires CUDA plus official vLLM DS4 cache-insert compiled kernels",
)
def test_official_bf16_kv_insert_is_bitwise_through_adapter() -> None:
    import vllm._C_stable_libtorch  # noqa: F401

    torch.manual_seed(21)
    q = torch.randn(2, 64, 512, dtype=torch.bfloat16, device="cuda")
    kv = torch.randn(2, 512, dtype=torch.bfloat16, device="cuda")
    slots = torch.tensor([0, 1], dtype=torch.int64, device="cuda")
    positions = torch.tensor([0, 1], dtype=torch.int64, device="cuda")
    cos = torch.ones(8, 64, dtype=torch.float32, device="cuda")
    reference_q, candidate_q = q.clone(), q.clone()
    reference_cache = torch.zeros(1, 64, 512, dtype=torch.bfloat16, device="cuda")
    candidate_cache = reference_cache.clone()
    torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_bf16_insert(
        reference_q,
        kv.clone(),
        reference_cache,
        slots,
        positions,
        cos,
        1e-6,
        64,
    )
    candidate = DS4KVInsertAdapter("plain_bf16")(
        candidate_q,
        kv.clone(),
        candidate_cache,
        slots,
        positions,
        cos,
        eps=1e-6,
        block_size=64,
    )
    torch.testing.assert_close(candidate, reference_q, rtol=0, atol=0)
    torch.testing.assert_close(candidate_cache, reference_cache, rtol=0, atol=0)


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
    candidate = DS4KVInsertAdapter("fp8_ds_mla")(
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
def test_official_flashmla_sparse_is_bitwise_through_adapter() -> None:
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
    candidate = FlashMLAAdapter().sparse(
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
