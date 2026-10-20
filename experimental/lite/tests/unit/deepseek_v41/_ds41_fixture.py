# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Small 40L-isomorphic source graph; synthetic release-format weights."""
from megatron.lite.model.deepseek_v41.config import DeepseekV41Config


def tiny_config():
    return DeepseekV41Config(
        vocab_size=128,
        hidden_size=128,
        moe_intermediate_size=64,
        num_hidden_layers=12,
        q_lora_rank=32,
        o_lora_rank=32,
        n_routed_experts=8,
        num_experts_per_tok=2,
        index_topk=16,
        compress_ratios=[0, 0, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1],
        kv_source_layer_ids=(2, 4, 6, 8),
        index_source_layer_ids=(2, 4, 6, 8, 10),
        candidate_source_layer_id=8,
        candidate_topk_blocks=2,
        candidate_block_size=8,
        sliding_window=8,
        engram_layer_ids=(1, 6),
        engram_num_embeddings=(128, 128),
        engram_max_ngram_size=2,
        engram_vocab_size=31,
        engram_n_heads=2,
        engram_head_dim=32,
        engram_pad_token_id=0,
        engram_compressed_vocab_size=128,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        compress_rope_theta=160000.0,
        rotary_scaling_factor=16.0,
        original_max_position_embeddings=32,
    )


# Spawned workers import this fixture before constructing the actual model.
import sys


class _NoRolloutImports:
    def find_spec(self, name, path=None, target=None):
        assert name != "vllm" and not name.startswith("vllm."), name
        return None


assert not any(n == "vllm" or n.startswith("vllm.") for n in sys.modules)
sys.meta_path.insert(0, _NoRolloutImports())
