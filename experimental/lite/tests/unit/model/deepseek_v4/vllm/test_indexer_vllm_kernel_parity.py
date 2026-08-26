from __future__ import annotations

import pytest
import torch


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _config():
    from megatron.lite.model.deepseek_v4.config import DeepseekV4Config

    return DeepseekV4Config(
        head_dim=512,
        index_head_dim=128,
        qk_rope_head_dim=64,
        num_attention_heads=64,
        sliding_window=128,
        compress_ratios=[4],
        num_hidden_layers=1,
        max_position_embeddings=8192,
    )


def test_indexer_compressor_and_topk_use_vllm_kernels_bitwise(
    transformer_engine_import_stub,
) -> None:
    transformer_engine_import_stub()
    from copy import copy

    from megatron.lite.model.deepseek_v4.vllm.primitive.attention.backward import (
        compressed_compact_graph,
    )
    from megatron.lite.model.deepseek_v4.vllm.primitive.attention.runtime import (
        _build_compressor_metadata,
        _dequantize_indexer_k_cache,
        _top_k_per_row_prefill,
        compressor_operation,
        official_compact_compressed_visible,
        official_indexer_topk,
    )
    from vllm.models.deepseek_v4.common.ops.fused_indexer_q import (
        fused_indexer_q_rope_quant,
    )
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        per_token_group_quant_fp8,
    )
    from vllm.utils.deep_gemm import fp8_fp4_mqa_logits

    torch.manual_seed(42)
    device = torch.device("cuda")
    config = _config()
    ratio = 4
    groups = 664
    capacity_groups = 672
    tokens = capacity_groups * ratio
    valid_tokens = groups * ratio
    cos_sin = torch.randn(8192, 64, dtype=torch.float32, device=device)
    compact_score = torch.randn(tokens, 512, dtype=torch.bfloat16, device=device)
    ape = torch.randn(4, 256, dtype=torch.float32, device=device)
    norm = torch.randn(128, dtype=torch.bfloat16, device=device)
    group_ids = torch.cat(
        (
            torch.arange(groups, dtype=torch.int64, device=device),
            torch.full(
                (capacity_groups - groups,), -1, dtype=torch.int64, device=device
            ),
        )
    )

    functional = compressed_compact_graph(
        compact_score,
        ape,
        norm,
        group_ids,
        cos_sin,
        ratio=ratio,
        head_dim=128,
        rope_dim=64,
        eps=config.rms_norm_eps,
    )
    metadata = _build_compressor_metadata(
        config,
        ratio=ratio,
        rows=tokens,
        cos_sin_cache=cos_sin,
        head_dim=128,
    )
    assert metadata is not None
    actual_k = official_compact_compressed_visible(
        functional,
        compact_score,
        ape,
        norm,
        group_ids,
        cos_sin,
        operation=compressor_operation,
        runtime_metadata=metadata,
        ratio=ratio,
        head_dim=128,
        valid_groups=groups,
    )

    direct = copy(metadata)
    direct.state_cache.zero_()
    synthetic_positions = torch.arange(valid_tokens, dtype=torch.int64, device=device)
    synthetic_starts = torch.arange(0, valid_tokens, ratio, device=device)
    direct.cos_sin_cache.index_copy_(
        0,
        synthetic_starts,
        cos_sin.index_select(0, group_ids[:groups] * ratio),
    )
    compressor_operation(
        kv_score=compact_score[:valid_tokens],
        positions=synthetic_positions,
        ape=ape,
        norm_weight=norm,
        compress_ratio=ratio,
        head_dim=128,
        metadata=direct,
    )
    direct_k = _dequantize_indexer_k_cache(direct.k_cache, groups)
    assert torch.equal(actual_k, direct_k)
    assert torch.equal(actual_k[[511, 512, 513, 663]], direct_k[[511, 512, 513, 663]])

    positions = torch.tensor([2047, 2048, 2049, 2654], device=device)
    index_q = torch.randn(4, 64, 128, dtype=torch.bfloat16, device=device)
    index_weights = torch.randn(4, 64, dtype=torch.bfloat16, device=device)
    cu_seqlens = torch.tensor([0, 4], dtype=torch.int32, device=device)
    cu_seqlens_compressed = torch.tensor(
        [0, groups], dtype=torch.int32, device=device
    )
    actual_topk = official_indexer_topk(
        index_q,
        index_weights,
        actual_k,
        positions,
        cos_sin,
        cu_seqlens,
        cu_seqlens_compressed,
        global_start=0,
        ratio=ratio,
        topk=512,
    )
    torch.cuda.synchronize()

    q_quant, weights = fused_indexer_q_rope_quant(
        positions,
        index_q,
        cos_sin,
        index_weights,
        index_q.shape[-1] ** -0.5,
        index_q.shape[1] ** -0.5,
        use_fp4=False,
    )
    k_quant, k_scale = per_token_group_quant_fp8(
        direct_k.contiguous(), group_size=128, use_ue8m0=True
    )
    row_starts = torch.zeros(4, dtype=torch.int32, device=device)
    row_ends = torch.div(positions + 1, ratio, rounding_mode="floor").to(torch.int32)
    logits = fp8_fp4_mqa_logits(
        (q_quant, None),
        (k_quant, k_scale.view(torch.float32).squeeze(-1)),
        weights,
        row_starts,
        row_ends,
        clean_logits=False,
    )
    direct_topk = torch.full(
        (4, 512), -1, dtype=torch.int32, device=device
    )
    _top_k_per_row_prefill(
        logits,
        row_starts,
        row_ends,
        direct_topk,
        4,
        logits.stride(0),
        logits.stride(1),
        512,
    )
    assert torch.equal(actual_topk[:3], direct_topk[:3])
    assert actual_topk[3].ge(0).all()
    assert actual_topk[3].lt(row_ends[3]).all()
    assert actual_topk[3].unique().numel() == 512
