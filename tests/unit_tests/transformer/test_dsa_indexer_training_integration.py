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
from megatron.core.models.hybrid.hybrid_layer_allocation import parse_hybrid_pattern
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
        dsa_indexer_n_heads=2,
        dsa_indexer_head_dim=8,
        dsa_indexer_topk=4,
        vocab_file=None,
        data_parallel_random_init=False,
        phase_transition_iterations=None,
        use_dist_ckpt=True,
    )
    runtime_args = SimpleNamespace(
        **common, dsa_train_indexer_only=True, no_load_optim=False, finetune=False
    )
    monkeypatch.setattr(checkpointing, "get_args", lambda: runtime_args)
    monkeypatch.setattr(checkpointing, "get_checkpoint_version", lambda: 3.0)

    checkpointing.check_checkpoint_args(SimpleNamespace(**common, dsa_train_indexer_only=True))
    with pytest.raises(AssertionError, match="Use --no-load-optim"):
        checkpointing.check_checkpoint_args(SimpleNamespace(**common))

    runtime_args.no_load_optim = True
    checkpointing.check_checkpoint_args(SimpleNamespace(**common))


def test_dsa_reset_on_load_initializes_only_a_checkpoint_without_an_indexer():
    """The flag converts a dense checkpoint, so it keys off what the checkpoint holds.

    Keying off args.iteration instead coupled this to --finetune in both directions: it
    refused to convert under a plain --load, and it happily re-initialised an already-trained
    indexer when --finetune zeroed the iteration.
    """
    import megatron.training.training as training

    # A dense checkpoint has no indexer to preserve, so the conversion runs -- and now does
    # so under a plain --load, without requiring --finetune.
    dense = SimpleNamespace(dsa_reset_indexer_on_load=True, checkpoint_has_dsa_indexer=False)
    assert training._should_reset_dsa_indexer_after_load(dense)

    # A checkpoint that already carries an indexer is never re-initialised, whether this is a
    # requeue or a --finetune run that left the flag set in its launch script.
    trained = SimpleNamespace(dsa_reset_indexer_on_load=True, checkpoint_has_dsa_indexer=True)
    assert not training._should_reset_dsa_indexer_after_load(trained)

    # Without the flag the initialization never runs.
    off = SimpleNamespace(dsa_reset_indexer_on_load=False, checkpoint_has_dsa_indexer=False)
    assert not training._should_reset_dsa_indexer_after_load(off)

    # An absent attribute means no checkpoint args were recorded, which is the no-indexer case.
    missing = SimpleNamespace(dsa_reset_indexer_on_load=True)
    assert training._should_reset_dsa_indexer_after_load(missing)


def test_dsa_reset_on_load_allows_pipeline_stage_without_local_indexer(monkeypatch):
    import megatron.training.training as training

    monkeypatch.setattr(training, "_reset_dsa_indexer_modules", lambda model, seed: 0)
    monkeypatch.setattr(training, "_get_dsa_indexer_reset_seed", lambda args: 1234)
    monkeypatch.setattr(training, "_global_dsa_indexer_reset_count", lambda local_count: 4)
    monkeypatch.setattr(training, "_broadcast_dsa_indexer_params", lambda model: None)
    args = SimpleNamespace(no_load_optim=True, consumed_train_samples=0)

    training._reset_dsa_indexer_after_load([], None, None, args)


@pytest.mark.parametrize("fsdp_arg", ["use_torch_fsdp2", "use_megatron_fsdp"])
def test_dsa_reset_on_load_rejects_fsdp(fsdp_arg):
    import megatron.training.training as training

    args = SimpleNamespace(use_torch_fsdp2=False, use_megatron_fsdp=False)
    setattr(args, fsdp_arg, True)

    with pytest.raises(RuntimeError, match="DDP/distributed-optimizer"):
        training._reset_dsa_indexer_after_load([], None, None, args)


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
        training._reset_dsa_indexer_after_load([], optimizer, None, args)


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


def test_dsa_train_indexer_only_freezes_moe_router_expert_bias(monkeypatch):
    """The router's expert_bias is a buffer, so requires_grad does not reach it.

    finalize_model_grads rewrites it every step from observed token counts for load balancing,
    and only frozen_expert_bias stops that. Without this the backbone's routing keeps moving
    while the run is meant to train the indexer alone.
    """
    import megatron.training.training as training

    class _Router(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.frozen_expert_bias = False
            self.register_buffer("expert_bias", torch.zeros(4))

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.block = torch.nn.Module()
            self.block.indexer = torch.nn.Linear(3, 2)
            self.block.mlp = torch.nn.Module()
            self.block.mlp.router = _Router()

    model = _Model()
    monkeypatch.setattr(training, "_global_dsa_indexer_reset_count", lambda count: count)

    training._freeze_non_dsa_indexer_parameters([model])

    assert model.block.mlp.router.frozen_expert_bias is True


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

    args = SimpleNamespace(seed=1234, data_parallel_random_init=False)
    assert training._get_dsa_indexer_reset_seed(args) == 1434

    args.data_parallel_random_init = True
    assert training._get_dsa_indexer_reset_seed(args) == 1464


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


def _indexer_count_args(pattern, num_layers, mtp_num_layers=0, mtp_use_repeated_layer=False):
    return SimpleNamespace(
        num_layers=num_layers,
        mtp_num_layers=mtp_num_layers,
        mtp_use_repeated_layer=mtp_use_repeated_layer,
        hybrid_layer_pattern=pattern,
        csa_compress_ratios=None,
        experimental_attention_variant='dsa',
    )


def test_indexer_layer_count_counts_attention_layers_not_every_layer(monkeypatch):
    """The tracker spans every decoder layer; only the attention layers own an indexer.

    Averaging over the tracker length instead of the indexer count under-reports the loss by the
    ratio between them -- 13x for the 8B pattern, which is four attention layers in fifty-two.
    """
    from megatron.training import training

    monkeypatch.setattr(training, "is_hybrid_model", lambda args: True)
    pattern = "M-" * 24 + "*-" * 2  # 48 mamba/mlp layers, then 2 attention layers
    args = _indexer_count_args(pattern, num_layers=len(pattern))

    tracker_layers, indexer_layers = training._get_indexer_logging_layer_counts(args)

    assert tracker_layers == len(pattern)
    assert indexer_layers == 2
    assert indexer_layers != tracker_layers


def test_indexer_layer_count_includes_mtp_depths(monkeypatch):
    """MTP repeats its pattern once per prediction depth, and each copy owns an indexer."""
    from megatron.training import training

    monkeypatch.setattr(training, "is_hybrid_model", lambda args: True)
    args = _indexer_count_args("M*M*/M*", num_layers=4, mtp_num_layers=2)

    tracker_layers, indexer_layers = training._get_indexer_logging_layer_counts(args)

    # "M*M*/M*" is two attention layers in the main pattern and one per MTP depth.
    depths = parse_hybrid_pattern(args.hybrid_layer_pattern).mtp_num_depths
    assert indexer_layers == 2 + depths
    assert tracker_layers == 4 + 2


def test_indexer_layer_count_is_left_alone_without_dsa(monkeypatch):
    """A non-DSA run keeps the old behaviour: no count, so the tracker length is used."""
    from megatron.training import training

    monkeypatch.setattr(training, "is_hybrid_model", lambda args: True)
    args = _indexer_count_args("M*M*", num_layers=4)
    args.experimental_attention_variant = None

    assert training._get_indexer_logging_layer_counts(args) == (4, None)
