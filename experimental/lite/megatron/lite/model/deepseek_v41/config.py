# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""DS4.1 release fields; executable ownership excludes archival prediction layers."""
from dataclasses import dataclass, fields

from megatron.lite.model.deepseek_v4.config import DeepseekV4Config


@dataclass
class DeepseekV41Config(DeepseekV4Config):
    cp_size: int = 1
    hidden_size: int = 5120
    moe_intermediate_size: int = 2304
    num_hidden_layers: int = 40
    head_dim: int = 512
    q_lora_rank: int = 1280
    n_routed_experts: int = 384
    num_hash_layers: int = 0
    index_n_heads: int = 32
    num_nextn_predict_layers: int = 3
    rms_norm_eps: float = 1e-20
    rotary_scaling_factor: float = 16.0
    original_max_position_embeddings: int = 65536
    compress_rope_theta: float = 10000.0
    kv_source_layer_ids: tuple[int, ...] = (2, 8, 14, 20)
    index_source_layer_ids: tuple[int, ...] = (2, 8, 14, 20, 24, 28, 32, 36)
    candidate_source_layer_id: int = 20
    candidate_topk_blocks: int = 2048
    candidate_block_size: int = 8
    engram_layer_ids: tuple[int, ...] = (1, 14)
    engram_num_embeddings: tuple[int, ...] = (384006168, 384016682)
    engram_max_ngram_size: int = 4
    engram_vocab_size: int = 16000000
    engram_n_heads: int = 8
    engram_head_dim: int = 256
    engram_pad_token_id: int = 2
    engram_compressed_vocab_size: int = 99092

    @classmethod
    def _from_hf_dict(cls, hf, **overrides):
        text = hf.get("text_config", hf)
        config = super()._from_hf_dict(text, **overrides)
        if not config.compress_ratios:
            config.compress_ratios = list(DEFAULT_RATIOS)
        for item in fields(cls):
            if item.name in {
                "kv_source_layer_ids",
                "index_source_layer_ids",
                "engram_layer_ids",
                "engram_num_embeddings",
            }:
                setattr(config, item.name, tuple(getattr(config, item.name)))
        return config


DEFAULT_RATIOS = (0, 0) + (2,) * 18 + (1,) * 20
