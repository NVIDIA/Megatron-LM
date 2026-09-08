# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Tests for the DSA indexer's training-loop integration.

These cover the parts of DSA that live in megatron/training rather than in the attention layer:
freezing one half of the model, resetting the indexer on load, and the checkpoint and optimizer
bookkeeping that has to survive such a reset. They import from megatron.training.training and
megatron.training.checkpointing, which is why they are separate from the attention-layer tests.
"""

from types import SimpleNamespace

import pytest
import torch

from megatron.core.transformer.experimental_attention_variant.dsa_gqa import (
    SimplifiedDSGQAIndexer,
    SimplifiedDSGQAIndexerSubmodules,
)
from megatron.core.transformer.transformer_config import TransformerConfig


class _DummyTPGroup:
    def size(self):
        return 1

    def rank(self):
        return 0


class _DummyPGCollection:
    def __init__(self):
        self.tp = _DummyTPGroup()
        self.cp = None


def test_simplified_learned_k_is_model_defining_checkpoint_metadata(monkeypatch):
    import megatron.training.checkpointing as checkpointing

    common = dict(
        num_layers=1,
        hidden_size=32,
        num_attention_heads=4,
        add_position_embedding=True,
        experimental_attention_variant="dsa",
        add_bias_linear=False,
        dsa_indexer_mode="simplified",
        dsa_indexer_n_heads=1,
        dsa_indexer_head_dim=8,
        dsa_indexer_topk=4,
        dsa_indexer_use_hadamard=False,
        vocab_file=None,
        data_parallel_random_init=False,
        phase_transition_iterations=None,
        use_dist_ckpt=True,
    )
    runtime_args = SimpleNamespace(**common, dsa_simplified_use_learned_k=True)
    monkeypatch.setattr(checkpointing, "get_args", lambda: runtime_args)
    monkeypatch.setattr(checkpointing, "get_checkpoint_version", lambda: 3.0)

    checkpointing.check_checkpoint_args(
        SimpleNamespace(**common, dsa_simplified_use_learned_k=True)
    )
    with pytest.raises(AssertionError, match="dsa_simplified_use_learned_k"):
        checkpointing.check_checkpoint_args(
            SimpleNamespace(**common, dsa_simplified_use_learned_k=False)
        )

    runtime_args.dsa_simplified_use_learned_k = False
    checkpointing.check_checkpoint_args(SimpleNamespace(**common))

    load_args = SimpleNamespace(
        load="old-dsa-checkpoint",
        experimental_attention_variant="dsa",
        dsa_simplified_use_learned_k=True,
        use_tokenizer_model_from_checkpoint_args=False,
        use_mp_args_from_checkpoint_args=False,
    )
    old_checkpoint_args = SimpleNamespace(
        experimental_attention_variant="dsa", dsa_indexer_mode="simplified"
    )
    monkeypatch.setattr(
        checkpointing,
        "_load_base_checkpoint",
        lambda *args, **kwargs: (
            {"args": old_checkpoint_args, "iteration": 17},
            "checkpoint.pt",
            False,
            None,
        ),
    )
    loaded_args, _ = checkpointing.load_args_from_checkpoint(load_args)
    assert loaded_args.dsa_simplified_use_learned_k is False


def test_simplified_disabled_main_input_norm_is_model_defining_checkpoint_metadata(monkeypatch):
    import megatron.training.checkpointing as checkpointing

    common = dict(
        num_layers=1,
        hidden_size=32,
        num_attention_heads=4,
        add_position_embedding=True,
        experimental_attention_variant="dsa",
        add_bias_linear=False,
        dsa_indexer_mode="simplified",
        dsa_simplified_use_learned_k=False,
        dsa_indexer_n_heads=1,
        dsa_indexer_head_dim=8,
        dsa_indexer_topk=4,
        dsa_indexer_use_hadamard=False,
        vocab_file=None,
        data_parallel_random_init=False,
        phase_transition_iterations=None,
        use_dist_ckpt=True,
    )
    runtime_args = SimpleNamespace(**common, dsa_simplified_indexer_disable_main_input_norm=True)
    monkeypatch.setattr(checkpointing, "get_args", lambda: runtime_args)
    monkeypatch.setattr(checkpointing, "get_checkpoint_version", lambda: 3.0)

    checkpointing.check_checkpoint_args(
        SimpleNamespace(**common, dsa_simplified_indexer_disable_main_input_norm=True)
    )
    with pytest.raises(AssertionError, match="dsa_simplified_indexer_disable_main_input_norm"):
        checkpointing.check_checkpoint_args(
            SimpleNamespace(**common, dsa_simplified_indexer_disable_main_input_norm=False)
        )

    runtime_args.dsa_simplified_indexer_disable_main_input_norm = False
    checkpointing.check_checkpoint_args(SimpleNamespace(**common))

    load_args = SimpleNamespace(
        load="old-dsa-checkpoint",
        experimental_attention_variant="dsa",
        dsa_simplified_indexer_disable_main_input_norm=True,
        use_tokenizer_model_from_checkpoint_args=False,
        use_mp_args_from_checkpoint_args=False,
    )
    old_checkpoint_args = SimpleNamespace(
        experimental_attention_variant="dsa", dsa_indexer_mode="simplified"
    )
    monkeypatch.setattr(
        checkpointing,
        "_load_base_checkpoint",
        lambda *args, **kwargs: (
            {"args": old_checkpoint_args, "iteration": 17},
            "checkpoint.pt",
            False,
            None,
        ),
    )
    loaded_args, _ = checkpointing.load_args_from_checkpoint(load_args)
    assert not loaded_args.dsa_simplified_indexer_disable_main_input_norm

    conversion_args = SimpleNamespace(
        load="gqa-checkpoint",
        experimental_attention_variant="dsa",
        dsa_indexer_mode="simplified",
        dsa_simplified_indexer_disable_main_input_norm=True,
        use_tokenizer_model_from_checkpoint_args=False,
        use_mp_args_from_checkpoint_args=False,
    )
    gqa_checkpoint_args = SimpleNamespace(experimental_attention_variant=None)
    monkeypatch.setattr(
        checkpointing,
        "_load_base_checkpoint",
        lambda *args, **kwargs: (
            {"args": gqa_checkpoint_args, "iteration": 19},
            "checkpoint.pt",
            False,
            None,
        ),
    )
    loaded_args, _ = checkpointing.load_args_from_checkpoint(conversion_args)
    assert loaded_args.dsa_simplified_indexer_disable_main_input_norm


def test_standard_main_input_norm_is_model_defining_checkpoint_metadata(monkeypatch):
    import megatron.training.checkpointing as checkpointing

    common = dict(
        num_layers=1,
        hidden_size=32,
        num_attention_heads=4,
        add_position_embedding=True,
        experimental_attention_variant="dsa",
        add_bias_linear=False,
        dsa_indexer_mode="standard",
        dsa_simplified_use_learned_k=False,
        dsa_indexer_n_heads=2,
        dsa_indexer_head_dim=8,
        dsa_indexer_topk=4,
        dsa_indexer_use_hadamard=True,
        vocab_file=None,
        data_parallel_random_init=False,
        phase_transition_iterations=None,
        use_dist_ckpt=True,
    )
    runtime_args = SimpleNamespace(**common, dsa_standard_indexer_use_main_input_norm=True)
    monkeypatch.setattr(checkpointing, "get_args", lambda: runtime_args)
    monkeypatch.setattr(checkpointing, "get_checkpoint_version", lambda: 3.0)

    checkpointing.check_checkpoint_args(
        SimpleNamespace(**common, dsa_standard_indexer_use_main_input_norm=True)
    )
    with pytest.raises(AssertionError, match="dsa_standard_indexer_use_main_input_norm"):
        checkpointing.check_checkpoint_args(
            SimpleNamespace(**common, dsa_standard_indexer_use_main_input_norm=False)
        )

    runtime_args.dsa_standard_indexer_use_main_input_norm = False
    checkpointing.check_checkpoint_args(SimpleNamespace(**common))

    load_args = SimpleNamespace(
        load="old-dsa-checkpoint",
        experimental_attention_variant="dsa",
        dsa_standard_indexer_use_main_input_norm=True,
        use_tokenizer_model_from_checkpoint_args=False,
        use_mp_args_from_checkpoint_args=False,
    )
    old_checkpoint_args = SimpleNamespace(
        experimental_attention_variant="dsa", dsa_indexer_mode="standard"
    )
    monkeypatch.setattr(
        checkpointing,
        "_load_base_checkpoint",
        lambda *args, **kwargs: (
            {"args": old_checkpoint_args, "iteration": 17},
            "checkpoint.pt",
            False,
            None,
        ),
    )
    loaded_args, _ = checkpointing.load_args_from_checkpoint(load_args)
    assert loaded_args.dsa_standard_indexer_use_main_input_norm is False

    conversion_args = SimpleNamespace(
        load="gqa-checkpoint",
        experimental_attention_variant="dsa",
        dsa_standard_indexer_use_main_input_norm=True,
        use_tokenizer_model_from_checkpoint_args=False,
        use_mp_args_from_checkpoint_args=False,
    )
    gqa_checkpoint_args = SimpleNamespace(experimental_attention_variant=None)
    monkeypatch.setattr(
        checkpointing,
        "_load_base_checkpoint",
        lambda *args, **kwargs: (
            {"args": gqa_checkpoint_args, "iteration": 19},
            "checkpoint.pt",
            False,
            None,
        ),
    )
    loaded_args, _ = checkpointing.load_args_from_checkpoint(conversion_args)
    assert loaded_args.dsa_standard_indexer_use_main_input_norm is True


def test_dsa_trainability_mode_requires_no_load_optim_for_transitions(monkeypatch):
    import megatron.training.checkpointing as checkpointing

    common = dict(
        num_layers=1,
        hidden_size=32,
        num_attention_heads=4,
        add_position_embedding=True,
        experimental_attention_variant="dsa",
        add_bias_linear=False,
        dsa_indexer_mode="standard",
        dsa_simplified_use_learned_k=False,
        dsa_simplified_indexer_disable_main_input_norm=False,
        dsa_standard_indexer_use_main_input_norm=False,
        dsa_indexer_n_heads=2,
        dsa_indexer_head_dim=8,
        dsa_indexer_topk=4,
        dsa_indexer_use_hadamard=True,
        vocab_file=None,
        data_parallel_random_init=False,
        phase_transition_iterations=None,
        use_dist_ckpt=True,
    )
    runtime_args = SimpleNamespace(
        **common,
        dsa_train_main_only=True,
        dsa_train_indexer_only=False,
        no_load_optim=False,
        finetune=False,
    )
    monkeypatch.setattr(checkpointing, "get_args", lambda: runtime_args)
    monkeypatch.setattr(checkpointing, "get_checkpoint_version", lambda: 3.0)

    checkpointing.check_checkpoint_args(
        SimpleNamespace(**common, dsa_train_main_only=True, dsa_train_indexer_only=False)
    )
    with pytest.raises(AssertionError, match="Use --no-load-optim"):
        checkpointing.check_checkpoint_args(SimpleNamespace(**common))

    runtime_args.no_load_optim = True
    checkpointing.check_checkpoint_args(SimpleNamespace(**common))


def test_simplified_main_q_mean_reset_uses_all_query_heads():
    from megatron.training.training import _reset_simplified_dsa_indexers_from_main_q

    hidden_size, num_query_heads, head_dim = 5, 4, 3

    class _FakeIndexer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_q = torch.nn.Linear(hidden_size, head_dim, bias=False)
            self.config = SimpleNamespace(
                dsa_indexer_mode="simplified",
                num_attention_heads=num_query_heads,
                kv_channels=head_dim,
            )
            self.pg_collection = SimpleNamespace(tp=None)

    class _FakeCore(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.indexer = _FakeIndexer()

    class _FakeAttention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_qkv = torch.nn.Linear(
                hidden_size, (num_query_heads + 2) * head_dim, bias=False
            )
            self.core_attention = _FakeCore()

    attention = _FakeAttention()
    with torch.no_grad():
        attention.linear_qkv.weight.copy_(
            torch.arange(attention.linear_qkv.weight.numel(), dtype=torch.float32).reshape_as(
                attention.linear_qkv.weight
            )
        )
    expected = (
        attention.linear_qkv.weight[: num_query_heads * head_dim]
        .reshape(num_query_heads, head_dim, hidden_size)
        .mean(dim=0)
    )

    assert _reset_simplified_dsa_indexers_from_main_q([attention]) == 1
    torch.testing.assert_close(attention.core_attention.indexer.linear_q.weight, expected)


@pytest.mark.parametrize("attention_output_gate", [False, True])
def test_simplified_main_q_reset_also_initializes_learned_k_from_main_k(attention_output_gate):
    from megatron.training.training import _reset_simplified_dsa_indexers_from_main_q

    hidden_size, num_query_heads, head_dim = 5, 4, 3

    class _FakeIndexer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_q = torch.nn.Linear(hidden_size, head_dim, bias=False)
            self.linear_k = torch.nn.Linear(hidden_size, head_dim, bias=False)
            self.config = SimpleNamespace(
                dsa_indexer_mode="simplified",
                num_attention_heads=num_query_heads,
                kv_channels=head_dim,
                attention_output_gate=attention_output_gate,
            )
            self.pg_collection = SimpleNamespace(tp=None)

    class _FakeAttention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_qkv = torch.nn.Linear(
                hidden_size,
                ((2 * num_query_heads if attention_output_gate else num_query_heads) + 2)
                * head_dim,
                bias=False,
            )
            self.core_attention = torch.nn.Module()
            self.core_attention.indexer = _FakeIndexer()

    attention = _FakeAttention()
    with torch.no_grad():
        attention.linear_qkv.weight.copy_(
            torch.arange(attention.linear_qkv.weight.numel(), dtype=torch.float32).reshape_as(
                attention.linear_qkv.weight
            )
        )
    expected_q = (
        attention.linear_qkv.weight[: num_query_heads * head_dim]
        .reshape(num_query_heads, head_dim, hidden_size)
        .mean(dim=0)
    )
    k_start = num_query_heads * head_dim * (2 if attention_output_gate else 1)
    expected_k = attention.linear_qkv.weight[k_start : k_start + head_dim]

    assert _reset_simplified_dsa_indexers_from_main_q([attention]) == 1
    torch.testing.assert_close(attention.core_attention.indexer.linear_q.weight, expected_q)
    torch.testing.assert_close(attention.core_attention.indexer.linear_k.weight, expected_k)


def test_simplified_main_q_mean_rescaled_reset_gathers_fused_qkv_across_tp(monkeypatch):
    import megatron.training.training as training

    hidden_size, num_query_heads, head_dim, tp_size = 5, 4, 3, 2
    full_rows = (num_query_heads + 2) * head_dim
    full_weight = torch.arange(full_rows * hidden_size, dtype=torch.float32).reshape(
        full_rows, hidden_size
    )
    shards = list(full_weight.chunk(tp_size, dim=0))

    class _WeightModule(torch.nn.Module):
        def __init__(self, weight):
            super().__init__()
            self.weight = torch.nn.Parameter(weight.clone())

    class _FakeIndexer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_q = torch.nn.Linear(hidden_size, head_dim, bias=False)
            self.linear_k = torch.nn.Linear(hidden_size, head_dim, bias=False)
            self.config = SimpleNamespace(
                dsa_indexer_mode="simplified",
                num_attention_heads=num_query_heads,
                kv_channels=head_dim,
                attention_output_gate=False,
            )
            self.pg_collection = SimpleNamespace(tp=object())

    class _FakeAttention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_qkv = _WeightModule(shards[0])
            self.core_attention = torch.nn.Module()
            self.core_attention.indexer = _FakeIndexer()

    def _fake_all_gather(outputs, local_weight, group):
        assert torch.equal(local_weight, shards[0])
        for output, shard in zip(outputs, shards):
            output.copy_(shard)

    monkeypatch.setattr(training, "get_pg_size", lambda group: tp_size)
    monkeypatch.setattr(torch.distributed, "all_gather", _fake_all_gather)

    attention = _FakeAttention()
    main_q_heads = full_weight[: num_query_heads * head_dim].reshape(
        num_query_heads, head_dim, hidden_size
    )
    expected = main_q_heads.mean(dim=0)
    expected = expected * torch.sqrt(
        main_q_heads.square().sum(dim=(1, 2)).mean() / expected.square().sum()
    )
    expected_k = full_weight[num_query_heads * head_dim : (num_query_heads + 1) * head_dim]
    assert training._reset_simplified_dsa_indexers_from_main_q([attention], rescale=True) == 1
    torch.testing.assert_close(attention.core_attention.indexer.linear_q.weight, expected)
    torch.testing.assert_close(attention.core_attention.indexer.linear_k.weight, expected_k)


@pytest.mark.parametrize("no_load_optim", [False, True])
@pytest.mark.parametrize("reset_method", ["main-q-mean", "main-q-mean-rescaled"])
def test_simplified_main_q_reset_handles_optimizer_load_modes(
    monkeypatch, no_load_optim, reset_method
):
    import megatron.training.training as training

    hidden_size, num_query_heads, head_dim = 5, 4, 3

    class _FakeIndexer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_q = torch.nn.Linear(hidden_size, head_dim, bias=False)
            self.linear_k = torch.nn.Linear(hidden_size, head_dim, bias=False)
            self.config = SimpleNamespace(
                dsa_indexer_mode="simplified",
                num_attention_heads=num_query_heads,
                kv_channels=head_dim,
                attention_output_gate=False,
            )
            self.pg_collection = SimpleNamespace(tp=None)

    class _FakeAttention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_qkv = torch.nn.Linear(
                hidden_size, (num_query_heads + 2) * head_dim, bias=False
            )
            self.core_attention = torch.nn.Module()
            self.core_attention.indexer = _FakeIndexer()

    class _FakeOptimizer:
        is_stub_optimizer = False

        def __init__(self):
            self.reload_count = 0

        def reload_model_params(self):
            self.reload_count += 1

    attention = _FakeAttention()
    optimizer = _FakeOptimizer()
    clear_calls = []
    group_step_reset_calls = []
    refresh_calls = []
    monkeypatch.setattr(
        training,
        "_clear_dsa_indexer_optimizer_state",
        lambda model, optimizer: clear_calls.append((model, optimizer)) or 1,
    )
    monkeypatch.setattr(
        training, "_apply_dsa_indexer_lr_warmup", lambda args, optimizer, scheduler: 0.0
    )
    monkeypatch.setattr(
        training,
        "_reload_dsa_indexer_optimizer_params",
        lambda model, optimizer: refresh_calls.append((model, optimizer)) or 1,
    )
    monkeypatch.setattr(
        training,
        "_reset_dsa_indexer_optimizer_group_steps",
        lambda optimizer: group_step_reset_calls.append(optimizer) or 1,
    )
    monkeypatch.setattr(training, "_broadcast_dsa_indexer_params", lambda model: None)
    args = SimpleNamespace(
        dsa_indexer_reset_seed=None,
        dsa_indexer_reset_method=reset_method,
        no_load_optim=no_load_optim,
        finetune=False,
        dsa_indexer_activation_start_samples=0,
        consumed_train_samples=0,
    )

    training._reset_dsa_indexer_after_load([attention], optimizer, None, args, explicit_start=True)

    expected_k = attention.linear_qkv.weight[
        num_query_heads * head_dim : (num_query_heads + 1) * head_dim
    ]
    torch.testing.assert_close(attention.core_attention.indexer.linear_k.weight, expected_k)
    assert optimizer.reload_count == 0
    assert len(refresh_calls) == 1
    assert len(clear_calls) == (0 if no_load_optim else 1)
    assert len(group_step_reset_calls) == (0 if no_load_optim else 1)


def test_dsa_reset_on_load_allows_pipeline_stage_without_local_indexer(monkeypatch):
    import megatron.training.training as training

    monkeypatch.setattr(
        training, "_reset_simplified_dsa_indexers_from_main_q", lambda model, rescale: 0
    )
    monkeypatch.setattr(training, "_global_dsa_indexer_reset_count", lambda local_count: 4)
    monkeypatch.setattr(training, "_broadcast_dsa_indexer_params", lambda model: None)
    monkeypatch.setattr(
        training, "_apply_dsa_indexer_lr_warmup", lambda args, optimizer, scheduler: None
    )
    args = SimpleNamespace(
        dsa_indexer_reset_method="main-q-mean",
        no_load_optim=True,
        dsa_indexer_activation_start_samples=0,
        consumed_train_samples=0,
    )

    training._reset_dsa_indexer_after_load([], None, None, args, explicit_start=True)


@pytest.mark.parametrize("fsdp_arg", ["use_torch_fsdp2", "use_megatron_fsdp"])
def test_dsa_reset_on_load_rejects_fsdp(fsdp_arg):
    import megatron.training.training as training

    args = SimpleNamespace(use_torch_fsdp2=False, use_megatron_fsdp=False)
    setattr(args, fsdp_arg, True)

    with pytest.raises(RuntimeError, match="DDP/distributed-optimizer"):
        training._reset_dsa_indexer_after_load([], None, None, args, explicit_start=True)


def test_dsa_reset_on_load_rejects_optimizer_cpu_offload(monkeypatch):
    import megatron.training.training as training

    class _FakeHybridDeviceOptimizer:
        pass

    monkeypatch.setattr(training, "HybridDeviceOptimizer", _FakeHybridDeviceOptimizer)
    optimizer = SimpleNamespace(
        chained_optimizers=[SimpleNamespace(optimizer=_FakeHybridDeviceOptimizer())]
    )
    args = SimpleNamespace(use_torch_fsdp2=False, use_megatron_fsdp=False)

    with pytest.raises(RuntimeError, match="optimizer CPU offload"):
        training._reset_dsa_indexer_after_load([], optimizer, None, args, explicit_start=True)


def test_dsa_train_indexer_only_allows_pipeline_stage_without_local_indexer(monkeypatch):
    import megatron.training.training as training

    model = torch.nn.Linear(3, 2)
    monkeypatch.setattr(training, "_global_dsa_indexer_reset_count", lambda local_count: 5)

    training._freeze_non_dsa_indexer_parameters([model])

    assert all(not param.requires_grad for param in model.parameters())


def test_dsa_train_indexer_only_freezes_exactly_indexer_submodule_parameters(monkeypatch):
    import megatron.training.training as training

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Linear(3, 2)
            self.block = torch.nn.Module()
            self.block.indexer = torch.nn.Linear(3, 2)
            self.indexer_aux = torch.nn.Linear(3, 2)

    model = _Model()
    monkeypatch.setattr(training, "_global_dsa_indexer_reset_count", lambda count: count)

    training._freeze_non_dsa_indexer_parameters([model])

    for name, param in model.named_parameters():
        assert param.requires_grad == name.startswith("block.indexer.")


def test_dsa_train_main_only_allows_pipeline_stage_without_local_indexer(monkeypatch):
    import megatron.training.training as training

    model = torch.nn.Linear(3, 2)
    monkeypatch.setattr(training, "_global_dsa_indexer_reset_count", lambda local_count: 5)

    training._freeze_dsa_indexer_parameters([model])

    assert all(param.requires_grad for param in model.parameters())


def test_dsa_train_main_only_freezes_exactly_indexer_and_preserves_other_freezes(monkeypatch):
    import megatron.training.training as training

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Linear(3, 2)
            self.block = torch.nn.Module()
            self.block.indexer = torch.nn.Linear(3, 2)
            self.indexer_aux = torch.nn.Linear(3, 2)
            self.backbone.bias.requires_grad_(False)

    model = _Model()
    monkeypatch.setattr(training, "_global_dsa_indexer_reset_count", lambda count: count)

    training._freeze_dsa_indexer_parameters([model])

    for name, param in model.named_parameters():
        if name.startswith("block.indexer.") or name == "backbone.bias":
            assert not param.requires_grad
        else:
            assert param.requires_grad


def test_dsa_indexer_optimizer_refresh_preserves_backbone_master_weights(monkeypatch):
    import megatron.training.training as training

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Linear(3, 2, bias=False)
            self.indexer = torch.nn.Linear(3, 2, bias=False)

    model = _Model()
    backbone_master = torch.full_like(model.backbone.weight, 17.0, dtype=torch.float32)
    indexer_master = torch.full_like(model.indexer.weight, -5.0, dtype=torch.float32)
    with torch.no_grad():
        model.backbone.weight.fill_(1.0)
        model.indexer.weight.fill_(3.0)

    optimizer = SimpleNamespace(
        is_stub_optimizer=False,
        config=SimpleNamespace(use_precision_aware_optimizer_no_fp8_or_ds_fp8=False),
    )
    monkeypatch.setattr(
        training,
        "get_model_to_optimizer_param_map",
        lambda optimizer: {
            model.backbone.weight: backbone_master,
            model.indexer.weight: indexer_master,
        },
    )

    assert training._reload_dsa_indexer_optimizer_params([model], optimizer) == 1
    torch.testing.assert_close(indexer_master, model.indexer.weight.float())
    torch.testing.assert_close(backbone_master, torch.full_like(backbone_master, 17.0))


def test_dsa_indexer_optimizer_group_step_reset_preserves_backbone_clock():
    import megatron.training.training as training

    backbone_group = {"params": [object()], "is_dsa_indexer": False, "step": 123}
    indexer_weight_group = {"params": [object()], "is_dsa_indexer": True, "step": 123}
    indexer_bias_step = torch.tensor(123.0)
    indexer_bias_group = {"params": [object()], "is_dsa_indexer": True, "step": indexer_bias_step}
    empty_indexer_group = {"params": [], "is_dsa_indexer": True}
    optimizer = SimpleNamespace(
        is_stub_optimizer=False,
        optimizer=SimpleNamespace(
            param_groups=[
                backbone_group,
                indexer_weight_group,
                indexer_bias_group,
                empty_indexer_group,
            ]
        ),
    )

    assert training._reset_dsa_indexer_optimizer_group_steps(optimizer) == 3
    assert backbone_group["step"] == 123
    assert indexer_weight_group["step"] == 0
    assert indexer_bias_group["step"] is indexer_bias_step
    assert indexer_bias_group["step"].item() == 0.0
    assert empty_indexer_group["step"] == 0


def test_dsa_indexer_optimizer_refresh_copies_only_owned_distributed_shard(monkeypatch):
    import megatron.training.training as training

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.indexer = torch.nn.Linear(4, 2, bias=False)

    model = _Model()
    with torch.no_grad():
        model.indexer.weight.copy_(torch.arange(8, dtype=torch.float32).reshape(2, 4))
    indexer_shard = torch.full((3,), -1.0, dtype=torch.float32)

    class _Optimizer:
        is_stub_optimizer = False
        config = SimpleNamespace(use_precision_aware_optimizer_no_fp8_or_ds_fp8=False)

        @staticmethod
        def _get_model_param_range_map(param):
            assert param is model.indexer.weight
            return {"param": SimpleNamespace(start=2, end=5)}

    optimizer = _Optimizer()
    monkeypatch.setattr(
        training,
        "get_model_to_optimizer_param_map",
        lambda optimizer: {model.indexer.weight: indexer_shard},
    )

    assert training._reload_dsa_indexer_optimizer_params([model], optimizer) == 1
    torch.testing.assert_close(indexer_shard, torch.tensor([2.0, 3.0, 4.0]))


def test_dsa_indexer_random_reset_is_deterministic_and_preserves_global_rng():
    import megatron.training.training as training

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(init_method=lambda tensor: torch.nn.init.normal_(tensor))
            self.backbone = torch.nn.Linear(4, 3)
            self.indexer = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.LayerNorm(3))

    model_a = _Model()
    model_b = _Model()
    backbone_before = {
        name: tensor.detach().clone() for name, tensor in model_a.backbone.state_dict().items()
    }
    rng_before = torch.random.get_rng_state().clone()

    assert training._reset_dsa_indexer_modules([model_a], seed=9876) == 1
    rng_after = torch.random.get_rng_state()
    assert training._reset_dsa_indexer_modules([model_b], seed=9876) == 1

    assert torch.equal(rng_before, rng_after)
    for name, expected in backbone_before.items():
        torch.testing.assert_close(model_a.backbone.state_dict()[name], expected)
    for actual, expected in zip(model_a.indexer.parameters(), model_b.indexer.parameters()):
        torch.testing.assert_close(actual, expected)


def test_dsa_indexer_reset_seed_derivation_is_tp_invariant_and_dp_aware(monkeypatch):
    import megatron.training.training as training

    monkeypatch.setattr(training.mpu, "get_pipeline_model_parallel_rank", lambda: 2)
    monkeypatch.setattr(training.mpu, "get_data_parallel_rank", lambda: 3)

    args = SimpleNamespace(dsa_indexer_reset_seed=None, seed=1234, data_parallel_random_init=False)
    assert training._get_dsa_indexer_reset_seed(args) == 1434

    args.data_parallel_random_init = True
    assert training._get_dsa_indexer_reset_seed(args) == 1464

    args.dsa_indexer_reset_seed = 77
    assert training._get_dsa_indexer_reset_seed(args) == 77


def test_dsa_indexer_optimizer_state_clear_preserves_backbone_state(monkeypatch):
    import megatron.training.training as training

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Linear(3, 2)
            self.indexer = torch.nn.Linear(3, 2)

    model = _Model()
    torch_optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    for param in model.parameters():
        torch_optimizer.state[param] = {
            "step": torch.tensor(7.0),
            "exp_avg": torch.full_like(param, 2.0),
            "exp_avg_sq": torch.full_like(param, 3.0),
        }
    backbone_state = {
        param: {
            name: value.detach().clone() if torch.is_tensor(value) else value
            for name, value in torch_optimizer.state[param].items()
        }
        for param in model.backbone.parameters()
    }
    optimizer = SimpleNamespace(is_stub_optimizer=False, optimizer=torch_optimizer)
    monkeypatch.setattr(
        training,
        "get_model_to_optimizer_param_map",
        lambda _optimizer: {param: param for param in model.parameters()},
    )

    assert training._clear_dsa_indexer_optimizer_state([model], optimizer) == 2
    assert all(param not in torch_optimizer.state for param in model.indexer.parameters())
    for param, expected_state in backbone_state.items():
        assert param in torch_optimizer.state
        for name, expected in expected_state.items():
            actual = torch_optimizer.state[param][name]
            if torch.is_tensor(expected):
                torch.testing.assert_close(actual, expected)
            else:
                assert actual == expected


def test_dsa_indexer_reset_broadcasts_only_indexer_params_across_dp(monkeypatch):
    import megatron.training.training as training

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Linear(3, 2, bias=False)
            self.indexer = torch.nn.Linear(3, 2, bias=False)

    model = _Model()
    dp_group = object()
    calls = []
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(training.mpu, "get_data_parallel_group", lambda: dp_group)
    monkeypatch.setattr(training, "get_pg_size", lambda group: 2)
    monkeypatch.setattr(torch.distributed, "get_global_rank", lambda group, rank: 11)
    monkeypatch.setattr(
        torch.distributed,
        "broadcast",
        lambda tensor, src, group: calls.append((tensor, src, group)),
    )

    training._broadcast_dsa_indexer_params([model])

    assert len(calls) == 1
    assert calls[0][0].data_ptr() == model.indexer.weight.data_ptr()
    assert calls[0][1:] == (11, dp_group)
