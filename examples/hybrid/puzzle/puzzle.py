# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Train the full Nemotron Labs 3 Puzzle 75B-A9B architecture from Python configs.

Architecture source (no Hugging Face download or remote code is needed at runtime):
https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-3-Puzzle-75B-A9B-BF16/blob/7cd7fa01bab578cb4e8bde6ebba5e869afa1752d/config.json

The 88-layer decoder and its one attention/MoE prediction head are defined below.
The architecture stays a list of configs throughout construction and is supplied
again by this entrypoint on checkpoint resume. Do not also pass a hybrid layer
pattern: the list and pattern APIs are mutually exclusive. Training settings come
from the normal Megatron arguments; this example downloads no model weights.
"""

import time
from argparse import ArgumentParser, Namespace
from functools import partial

from megatron.core.models.hybrid import HybridLayerConfigListEntry, MTPSplit
from megatron.core.ssm.mamba_layer_config import MambaLayerConfig
from megatron.core.transformer.attention_layer_config import AttentionLayerConfig
from megatron.core.transformer.moe.moe_layer_config import MoELayerConfig
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.argument_utils import hybrid_config_from_args
from megatron.training.models.hybrid import HybridModelConfig


def add_puzzle_args(parser: ArgumentParser) -> ArgumentParser:
    """Set architecture defaults before standard argument/config validation.

    Only scalars go on argparse. In particular, the common expert count is needed
    by existing expert-parallel validation; per-layer widths and top-k live in the
    config list, not in arguments. The family marker also selects hybrid output
    initialization before any layer configs are copied.
    """
    parser.set_defaults(
        is_hybrid_model=True,
        num_layers=88,
        hidden_size=4096,
        ffn_hidden_size=21504,
        num_attention_heads=32,
        group_query_attention=True,
        num_query_groups=2,
        kv_channels=128,
        mamba_num_heads=128,
        mamba_head_dim=64,
        mamba_state_dim=96,
        mamba_num_groups=8,
        position_embedding_type="none",
        max_position_embeddings=262144,
        normalization="RMSNorm",
        layernorm_epsilon=1.0e-5,
        untie_embeddings_and_output_weights=True,
        add_bias_linear=False,
        add_qkv_bias=False,
        squared_relu=True,
        bias_gelu_fusion=False,
        init_method_std=0.02,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        num_experts=512,
        moe_ffn_hidden_size=2688,
        moe_router_topk=22,
        moe_router_topk_scaling_factor=5.0,
        moe_router_num_groups=1,
        moe_router_group_topk=1,
        moe_router_score_function="sigmoid",
        moe_router_enable_expert_bias=True,
        moe_router_dtype="fp32",
        moe_latent_size=1024,
        moe_shared_expert_intermediate_size=5376,
        moe_use_norm_before_up_proj=False,
        # Shared expert overlap with latent projections is unsupported in training.
        moe_shared_expert_overlap=False,
        mtp_num_layers=None,
        padded_vocab_size=131072,
        make_vocab_size_divisible_by=128,
        tokenizer_type="GPT2BPETokenizer",
    )
    return parser


def build_puzzle_layer_config_list(config: TransformerConfig) -> list[HybridLayerConfigListEntry]:
    """Create the published layer sequence without modifying the common config.

    Each entry is a fresh config inheriting runtime settings from ``config``.
    HybridModel owns the additional cloning needed for physical module instances.
    ``MTPSplit`` leaves prediction depth inference to the standard builder.
    """
    mamba = partial(MambaLayerConfig.from_config, config)
    attention = partial(AttentionLayerConfig.from_config, config)

    def moe(width: int, topk: int) -> MoELayerConfig:
        layer_config = MoELayerConfig.from_config(config)
        layer_config.moe_ffn_hidden_size = width
        layer_config.moe_router_topk = topk
        return layer_config

    return [
        mamba(),  # Decoder layers 0-8.
        moe(1280, 4),
        mamba(),
        moe(1280, 8),
        mamba(),
        moe(1280, 10),
        mamba(),
        attention(),
        moe(1280, 8),
        mamba(),  # 9-17.
        moe(1280, 8),
        mamba(),
        moe(1280, 8),
        mamba(),
        moe(1280, 12),
        mamba(),
        attention(),
        moe(1280, 8),
        mamba(),  # 18-26.
        moe(1280, 10),
        mamba(),
        moe(1280, 8),
        mamba(),
        moe(2688, 12),
        mamba(),
        attention(),
        moe(1536, 14),
        mamba(),  # 27-37.
        moe(2688, 12),
        mamba(),
        moe(1536, 12),
        mamba(),
        moe(1536, 12),
        mamba(),
        moe(2688, 12),
        mamba(),
        attention(),
        moe(2688, 12),
        mamba(),  # 38-48.
        moe(2688, 12),
        mamba(),
        moe(1536, 10),
        mamba(),
        moe(2688, 12),
        mamba(),
        moe(2688, 12),
        mamba(),
        attention(),
        moe(1792, 12),
        mamba(),  # 49-59.
        moe(1792, 14),
        mamba(),
        moe(1280, 10),
        mamba(),
        moe(1280, 10),
        mamba(),
        moe(1280, 12),
        mamba(),
        attention(),
        moe(1280, 8),
        mamba(),  # 60-70.
        moe(1280, 12),
        mamba(),
        moe(1280, 10),
        mamba(),
        moe(1280, 8),
        mamba(),
        moe(1280, 8),
        mamba(),
        attention(),
        moe(1280, 10),
        mamba(),  # 71-79.
        moe(1280, 10),
        mamba(),
        moe(1280, 10),
        mamba(),
        moe(1280, 12),
        mamba(),
        attention(),
        moe(1280, 12),
        mamba(),  # 80-87.
        moe(1280, 14),
        mamba(),
        moe(1280, 16),
        mamba(),
        moe(1792, 18),
        mamba(),
        moe(2048, 18),
        MTPSplit,
        attention(),
        moe(2688, 22),
    ]


def build_puzzle_model_config(args: Namespace) -> HybridModelConfig:
    """Combine the Python architecture with validated Megatron runtime arguments."""
    model_config = hybrid_config_from_args(args)
    model_config.hybrid_layer_config_list = build_puzzle_layer_config_list(model_config.transformer)
    return model_config


def main() -> None:
    """Run standard HybridModel training, including normal checkpoint resume."""
    program_start = time.time()

    from megatron.core.enums import ModelType
    from megatron.training import inprocess_restart, pretrain, set_startup_timestamps
    from megatron.training.argument_utils import pretrain_cfg_container_from_args
    from megatron.training.arguments import parse_and_validate_args
    from pretrain_hybrid import forward_step, train_valid_test_datasets_provider

    set_startup_timestamps(program_start=program_start, main_entry=time.time())
    train_valid_test_datasets_provider.is_distributed = True
    pretrain, store = inprocess_restart.maybe_wrap_for_inprocess_restart(pretrain)
    args = parse_and_validate_args(extra_args_provider=add_puzzle_args)
    model_config = build_puzzle_model_config(args)
    full_config = pretrain_cfg_container_from_args(args, model_config)
    pretrain(
        full_config,
        train_valid_test_datasets_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        store=store,
    )


if __name__ == "__main__":
    main()
