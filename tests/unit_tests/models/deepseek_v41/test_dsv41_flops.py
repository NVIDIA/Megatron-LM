# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""FLOPs accounting of the DeepSeek-V4.1 attention layers (CPU only)."""

from types import SimpleNamespace

from megatron.training.training import _dsv41_self_attention_flops


def _released_args(seq_length=131072):
    ratios = [0, 0] + [2] * 18 + [1] * 20
    return SimpleNamespace(
        csa_compress_ratios=ratios,
        hidden_size=5120,
        num_attention_heads=64,
        v_head_dim=512,
        q_lora_rank=1280,
        o_groups=8,
        o_lora_rank=1024,
        csa_window_size=128,
        seq_length=seq_length,
        dsa_indexer_topk=512,
        dsa_indexer_n_heads=32,
        dsa_indexer_head_dim=128,
        csa2_kv_source_layers=[2, 8, 14, 20],
        csa2_index_source_layers=[2, 8, 14, 20, 24, 28, 32, 36],
        csa2_candidate_source_layer=20,
        csa2_candidate_topk_blocks=2048,
        csa2_candidate_block_size=8,
    )


def test_released_model_per_token_budget():
    token_linear, core = _dsv41_self_attention_flops(_released_args())
    # MLA projections dominate: 40 x (q 48.5M + kv 2.6M + o 75.5M) ~= 5.1e9 multiply-adds.
    assert 5.0e9 < token_linear < 8.0e9, token_linear
    # Full index sources score every causal compressed entry: layers 2 / 8 / 14 at ratio 2
    # (n_i * d_i / 4 each) and the candidate source 20 at ratio 1 (n_i * d_i / 2); the Reindex
    # layers 24 / 28 / 32 / 36 only score the candidate set (token-linear).
    assert abs(core - (3 * 32 * 128 / 4 + 32 * 128 / 2)) < 1e-6, core
    # Full iteration in FLOPs: x2 (FMA) x3 (fwd + bwd); per token at 128K, attention only.
    seq = 131072
    per_token = 6 * (token_linear + core * seq)
    assert 3.0e10 < per_token < 6.0e10, per_token


def test_window_only_layers_have_no_compressed_attention():
    args = _released_args()
    args.csa_compress_ratios = [0, 0]
    args.csa2_kv_source_layers = []
    args.csa2_index_source_layers = []
    args.csa2_candidate_source_layer = None
    token_linear, core = _dsv41_self_attention_flops(args)
    expected_attn = 2 * 64 * 128 * 512 * 2
    q = 1280 * (5120 + 64 * 512 + 1)
    kv = 5120 * 512 + 512
    o = 64 * 512 * 1024 + 8 * 1024 * 5120
    assert token_linear == 2 * (q + kv + o) + expected_attn
    assert core == 0
