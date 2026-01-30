# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""
HLModel DSL Example: Nemotron-3 Nano (30B total, 3B active)
Using `examples/post_training/modelopt/conf/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16.sh` as reference.

Pattern Syntax:
---------------
Layer types (references layer_configs dict):
    M1, A1, E1, etc.

Repetition:
    (M1 E1)x5  = Repeat "M1 E1" pattern 5 times

Pipeline stage boundaries:
    |  = Start new pipeline stage (PP size inferred from | count)
"""

from megatron.core.models.hl import (
    HLModel,
    HLModelConfig,
    AttentionLayer,
    MambaLayer,
    MoELayer,
    ParallelismConfig,
)

# =============================================================================
# NEMOTRON-3 NANO CONFIGURATION (30B total, 3B active parameters)
# =============================================================================

nemotron_config = HLModelConfig(
    vocab_size=256000,
    max_sequence_length=8192,

    # Layer pattern: MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME
    # 52 layers total, split across 4 pipeline stages (marked by |)
    layer_pattern="""
        M1 (E1 M1)x2 A1 (E1 M1)x3 A1 |
        (E1 M1)x3 A1 (E1 M1)x3 A1    |
        (E1 M1)x3 A1 (E1 M1)x4 A1    |
        (E1 M1)x4 E1
    """,

    layer_configs={
        "M1": MambaLayer(
            hidden_size=2688,
            num_heads=64,
            head_dim=64,
            state_size=128,
            conv_kernel_size=4,
            parallelism=ParallelismConfig(
                tensor_parallel_size=8,
                sequence_parallel=True,
            ),
        ),

        "A1": AttentionLayer(
            hidden_size=2688,
            num_attention_heads=32,
            num_query_groups=2,
            kv_channels=128,
            use_flash_attention=True,
            parallelism=ParallelismConfig(
                tensor_parallel_size=8,
                sequence_parallel=True,
            ),
        ),

        "E1": MoELayer(
            hidden_size=2688,
            ffn_hidden_size=1856,
            num_experts=128,
            top_k=6,
            router_score_function="sigmoid",
            router_load_balancing_type="seq_aux_loss",
            router_topk_scaling_factor=2.5,
            router_enable_expert_bias=True,
            router_dtype="fp32",
            moe_aux_loss_coeff=1e-4,
            shared_expert_intermediate_size=3712,
            activation="squared_relu",
            token_dispatcher_type="allgather",
            grouped_gemm=True,
            parallelism=ParallelismConfig(
                tensor_parallel_size=8,
                sequence_parallel=True,
                expert_parallel_size=16,
                expert_tensor_parallel_size=8,
            ),
        ),
    },

    # Model settings
    share_embeddings_and_output_weights=False,
    position_embedding_type="none",
    normalization="RMSNorm",
    disable_bias_linear=True,
    init_method_std=0.0173,
    dtype="bf16",
)


def build_model() -> HLModel:
    """Build and return the Nemotron-3 Nano model."""
    return HLModel(nemotron_config)


if __name__ == "__main__":
    model = build_model()
    print(f"Created Nemotron-3 Nano model with config:\n{nemotron_config}")
