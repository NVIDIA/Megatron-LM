# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import math

import torch

from megatron.core.optimizer import OptimizerConfig
from megatron.core.transformer import TransformerConfig
from megatron.training.models import GPTModelConfig
from megatron.training.theoretical_memory_usage import compute_weight_and_optimizer_memory


def _memory_inputs(**overrides):
    values = dict(
        add_bias_linear=False,
        data_parallel_size=16,
        expert_model_parallel_size=2,
        expert_tensor_parallel_size=4,
        ffn_hidden_size=16,
        hidden_size=8,
        kv_channels=2,
        moe_ffn_hidden_size=16,
        moe_latent_size=None,
        moe_layer_freq=[0, 1],
        moe_router_topk=1,
        moe_router_pre_softmax=True,
        moe_shared_expert_gate=False,
        moe_shared_expert_intermediate_size=8,
        mtp_num_layers=None,
        multi_latent_attention=False,
        normalization="RMSNorm",
        num_attention_heads=4,
        num_moe_experts=4,
        num_layers=2,
        padded_vocab_size=32,
        pipeline_model_parallel_size=1,
        tensor_model_parallel_size=2,
        untie_embeddings_and_output_weights=False,
        use_distributed_optimizer=True,
        world_size=32,
    )
    values.update(overrides)
    data_parallel_size = values.pop("data_parallel_size")
    world_size = values.pop("world_size")
    optimizer = OptimizerConfig(use_distributed_optimizer=values.pop("use_distributed_optimizer"))
    vocab_size = values.pop("padded_vocab_size")
    share_embeddings = not values.pop("untie_embeddings_and_output_weights")
    transformer = TransformerConfig(
        **values,
        bf16=True,
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
        sequence_parallel=values["tensor_model_parallel_size"] > 1,
    )
    return dict(
        model_config=GPTModelConfig(
            transformer=transformer,
            vocab_size=vocab_size,
            share_embeddings_and_output_weights=share_embeddings,
        ),
        optimizer=optimizer,
        data_parallel_size=data_parallel_size,
        world_size=world_size,
    )


def test_weight_and_optimizer_memory_accounts_for_expert_parallelism():
    inputs = _memory_inputs(pipeline_model_parallel_size=2, world_size=64)

    # Most-loaded stage has 1 / PP of the transformer block plus the embedding table.
    # Regular TP-sharded params: (dense attention/MLP + MoE attention/shared-expert) / PP
    # plus embedding, all divided by TP.
    tp_sharded_params_on_rank = ((256 + 256 + 256 + 128) / 2 + 256) / 2
    # Replicated params: 1 / PP of dense norms + MoE norms/router + final norm.
    replicated_params_on_rank = (16 + 16 + 32 + 8) / 2
    # Routed experts are sharded by ETP * EP and use EDP for distributed optimizer state.
    expert_sharded_params_on_rank = 1024 / 2 / (4 * 2)

    # DP = 64 // 2(TP) // 2(PP) = 16
    # EDP = 64 // 4(ETP) // 2(EP) // 2(PP) = 4
    expected_memory = (tp_sharded_params_on_rank + replicated_params_on_rank) * (
        6 + 12 / 16
    ) + expert_sharded_params_on_rank * (6 + 12 / 4)

    assert math.isclose(compute_weight_and_optimizer_memory(**inputs), expected_memory)


def test_weight_and_optimizer_memory_decreases_with_tensor_parallelism():
    memories = [
        compute_weight_and_optimizer_memory(
            **_memory_inputs(
                data_parallel_size=16,
                expert_model_parallel_size=1,
                expert_tensor_parallel_size=1,
                tensor_model_parallel_size=tp_size,
                use_distributed_optimizer=False,
                world_size=16 * tp_size,
            )
        )
        for tp_size in (1, 2, 4)
    ]

    assert memories[0] > memories[1] > memories[2]


def test_weight_and_optimizer_memory_decreases_with_expert_parallelism():
    memories = [
        compute_weight_and_optimizer_memory(
            **_memory_inputs(
                expert_model_parallel_size=ep_size, expert_tensor_parallel_size=etp_size
            )
        )
        for ep_size, etp_size in ((1, 1), (2, 1), (2, 2), (4, 2))
    ]

    assert memories[0] > memories[1] > memories[2] > memories[3]


def test_weight_and_optimizer_memory_accounts_for_latent_moe_experts():
    inputs = _memory_inputs(moe_latent_size=4)

    # Latent MoE routes experts through moe_latent_size instead of hidden_size.
    # The hidden<->latent projections are duplicated non-expert params.
    tp_sharded_params_on_rank = ((256 + 256) + (256 + 128) + 256) / 2
    replicated_params_on_rank = 16 + (16 + 64 + 32) + 8
    expert_sharded_params_on_rank = 512 / (4 * 2)

    # DP = 32 // 2(TP) = 16
    # EDP = 32 // 4(ETP) // 2(EP) = 4
    expected_memory = (tp_sharded_params_on_rank + replicated_params_on_rank) * (
        6 + 12 / 16
    ) + expert_sharded_params_on_rank * (6 + 12 / 4)

    assert math.isclose(compute_weight_and_optimizer_memory(**inputs), expected_memory)
