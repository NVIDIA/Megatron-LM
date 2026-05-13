import math
from types import SimpleNamespace

from megatron.training.theoretical_memory_usage import compute_weight_and_optimizer_memory


def _make_args(**overrides):
    args = SimpleNamespace(
        add_bias_linear=False,
        data_parallel_size=16,
        expert_model_parallel_size=2,
        expert_tensor_parallel_size=4,
        ffn_hidden_size=16,
        group_query_attention=False,
        hidden_size=8,
        kv_channels=4,
        moe_ffn_hidden_size=16,
        moe_layer_freq=[0, 1],
        moe_router_topk=1,
        moe_shared_expert_gate=False,
        moe_shared_expert_intermediate_size=8,
        mtp_num_layers=None,
        multi_latent_attention=False,
        normalization="RMSNorm",
        num_attention_heads=2,
        num_experts=4,
        num_layers=2,
        padded_vocab_size=32,
        pipeline_model_parallel_size=1,
        swiglu=False,
        tensor_model_parallel_size=2,
        untie_embeddings_and_output_weights=False,
        use_distributed_optimizer=True,
        world_size=32,
    )
    for name, value in overrides.items():
        setattr(args, name, value)
    return args


def test_weight_and_optimizer_memory_accounts_for_expert_parallelism():
    args = _make_args(
        pipeline_model_parallel_size=2,
        world_size=64,
    )

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
    expected_memory = (
        (tp_sharded_params_on_rank + replicated_params_on_rank) * (6 + 12 / 16)
        + expert_sharded_params_on_rank * (6 + 12 / 4)
    )

    assert math.isclose(compute_weight_and_optimizer_memory(args), expected_memory)


def test_weight_and_optimizer_memory_decreases_with_tensor_parallelism():
    memories = [
        compute_weight_and_optimizer_memory(
            _make_args(
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
            _make_args(
                expert_model_parallel_size=ep_size,
                expert_tensor_parallel_size=etp_size,
            )
        )
        for ep_size, etp_size in ((1, 1), (2, 1), (2, 2), (4, 2))
    ]

    assert memories[0] > memories[1] > memories[2] > memories[3]
