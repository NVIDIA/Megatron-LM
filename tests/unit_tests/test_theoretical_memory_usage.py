# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from types import SimpleNamespace

import pytest

from megatron.training import theoretical_memory_usage as tmu


def _args(**overrides):
    args = SimpleNamespace(
        kv_channels=4,
        num_attention_heads=8,
        hidden_size=16,
        group_query_attention=False,
        num_query_groups=8,
        num_experts=None,
        swiglu=False,
        moe_shared_expert_intermediate_size=None,
        num_layers=2,
        moe_layer_freq=1,
        moe_ffn_hidden_size=32,
        moe_router_topk=1,
        mtp_num_layers=None,
        normalization="LayerNorm",
        multi_latent_attention=False,
        q_lora_rank=None,
        qk_head_dim=4,
        qk_pos_emb_head_dim=2,
        kv_lora_rank=2,
        v_head_dim=4,
        ffn_hidden_size=64,
        padded_vocab_size=128,
        untie_embeddings_and_output_weights=False,
        pipeline_model_parallel_size=1,
        tensor_model_parallel_size=1,
        data_parallel_size=1,
        use_distributed_optimizer=False,
        seq_length=8,
        micro_batch_size=2,
        virtual_pipeline_model_parallel_size=None,
        sequence_parallel=True,
        recompute_granularity="selective",
        hybrid_layer_pattern=None,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_compute_weight_and_optimizer_memory_dense_model():
    args = _args()

    assert tmu.compute_weight_and_optimizer_memory(args) == pytest.approx(187200.0)


def test_compute_weight_and_optimizer_memory_uses_distributed_optimizer_bytes():
    args = _args(use_distributed_optimizer=True, data_parallel_size=2)

    assert tmu.compute_weight_and_optimizer_memory(args) == pytest.approx(124800.0)


def test_compute_weight_and_optimizer_memory_validates_moe_pattern_length():
    args = _args(num_experts=2, moe_layer_freq=[1])

    with pytest.raises(AssertionError, match="Invalid length of moe_layer_pattern"):
        tmu.compute_weight_and_optimizer_memory(args)


def test_compute_activation_memory_with_sequence_parallel_formula():
    args = _args()

    assert tmu.compute_activation_memory(args, num_microbatches=None) == pytest.approx(27008.0)


def test_compute_activation_memory_without_sequence_parallel_formula():
    args = _args()

    assert tmu.compute_activation_memory_without_sp(args, num_microbatches=None) == pytest.approx(
        23520.0
    )


def test_compute_activation_memory_applies_pipeline_discount():
    args = _args(pipeline_model_parallel_size=4)

    full_memory = tmu.compute_activation_memory_without_sp(args, num_microbatches=None)
    discounted_memory = tmu.compute_activation_memory_without_sp(args, num_microbatches=2)

    assert discounted_memory == pytest.approx(full_memory * 0.5)


def test_report_theoretical_memory_dispatches_to_sp_path(monkeypatch, capsys):
    args = _args()

    monkeypatch.setattr(tmu, "compute_weight_and_optimizer_memory", lambda *a, **k: 10)
    monkeypatch.setattr(tmu, "compute_activation_memory", lambda *a, **k: 20)
    monkeypatch.setattr(tmu, "compute_activation_memory_without_sp", lambda *a, **k: 999)
    monkeypatch.setattr(tmu, "print_rank_0", lambda message: print(message))

    result = tmu.report_theoretical_memory(args)

    assert result == pytest.approx(
        (
            10 / tmu.NUM_BYTES_IN_MEGABYTE,
            20 / tmu.NUM_BYTES_IN_MEGABYTE,
            30 / tmu.NUM_BYTES_IN_MEGABYTE,
        )
    )
    assert "compute_activation_memory with SP" in capsys.readouterr().out


def test_report_theoretical_memory_skips_hybrid_model(capsys):
    args = _args(hybrid_layer_pattern="M-M*-")

    assert tmu.report_theoretical_memory(args) is None
    assert "not yet supported for hybrid" in capsys.readouterr().out
