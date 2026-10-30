# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Container validation is read-only; runtime initialization owns derivation."""

import copy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from megatron.core.transformer import TransformerConfig
from megatron.training import global_vars
from megatron.training.config import (
    CheckpointConfig,
    InferenceConfigContainer,
    InferenceSetupConfig,
    RLConfig,
)
from megatron.training.models import GPTModelConfig


@pytest.mark.parametrize("inference", [False, True])
def test_validation_is_read_only_and_finalization_runs_once(monkeypatch, run_config, inference):
    transformer = TransformerConfig(num_layers=2, hidden_size=32, num_attention_heads=4)
    # Native configs can enable graphs after construction, before runtime setup.
    transformer.cuda_graph_impl = "transformer_engine"
    model = GPTModelConfig(transformer=transformer, vocab_size=128, seq_length=16)
    cfg = (
        InferenceConfigContainer(
            model=model, checkpoint=CheckpointConfig(), inference=InferenceSetupConfig()
        )
        if inference
        else run_config
    )
    cfg.model = model
    if not inference:
        cfg.train.micro_batch_size = 1
        cfg.train.global_batch_size = 8
        cfg.rl = RLConfig(perform_rl_step=True, rl_offload_optimizer_during_inference=True)
    before = copy.deepcopy(cfg.to_dict())
    for _ in range(2):
        cfg.validate()
        assert cfg.to_dict() == before
        assert not cfg._finalized

    monkeypatch.setattr(global_vars, "_GLOBAL_ARGS", SimpleNamespace(batch_parallel_size=1))
    finalize = Mock(wraps=cfg._finalize)
    monkeypatch.setattr(cfg, "_finalize", finalize)
    cfg.finalize()
    assert cfg._finalized
    assert cfg.rng.te_rng_tracker
    assert transformer.use_te_rng_tracker
    if not inference:
        assert cfg.ddp.grad_buffer_offload
    after = copy.deepcopy(cfg.to_dict())
    cfg.validate()
    cfg.finalize()
    assert cfg.to_dict() == after
    finalize.assert_called_once_with()
    assert "_finalized" not in after
    assert copy.deepcopy(cfg)._finalized


def test_invalid_configuration_is_not_partially_finalized(monkeypatch, run_config):
    monkeypatch.setattr(global_vars, "_GLOBAL_ARGS", SimpleNamespace(batch_parallel_size=1))
    transformer = TransformerConfig(num_layers=2, hidden_size=32, num_attention_heads=4)
    transformer.cuda_graph_impl = "transformer_engine"
    run_config.model = GPTModelConfig(transformer=transformer, vocab_size=128, seq_length=16)
    run_config.train.micro_batch_size = 1
    run_config.train.global_batch_size = 8
    run_config.rl = RLConfig(perform_rl_step=True, rl_offload_optimizer_during_inference=True)
    run_config.ddp.fp8_param_gather = True
    run_config.ddp.reuse_grad_buf_for_mxfp8_param_ag = True
    before = copy.deepcopy(run_config.to_dict())
    with pytest.raises(ValueError, match="parameters share the gradient-buffer storage"):
        run_config.finalize()
    assert run_config.to_dict() == before
    assert not run_config._finalized
    assert not run_config.rng.te_rng_tracker

    # A rejected configuration remains editable and can be finalized after correction.
    run_config.ddp.reuse_grad_buf_for_mxfp8_param_ag = False
    run_config.finalize()
    assert run_config._finalized
    assert run_config.ddp.grad_buffer_offload
    assert run_config.rng.te_rng_tracker


def test_serialization_starts_a_new_lifecycle(monkeypatch, run_config):
    monkeypatch.setattr(global_vars, "_GLOBAL_ARGS", SimpleNamespace(batch_parallel_size=1))
    run_config.train.micro_batch_size = 1
    run_config.finalize()
    serialized = run_config.to_dict()
    assert "_finalized" not in serialized
    restored = type(run_config).from_dict(serialized)
    assert not restored._finalized
    assert restored.to_dict() == serialized
    restored.finalize()
    assert restored._finalized
    assert restored.to_dict() == serialized


def test_scheduler_uses_recipe_edits_without_mutating_during_validation(monkeypatch, run_config):
    monkeypatch.setattr(global_vars, "_GLOBAL_ARGS", SimpleNamespace(batch_parallel_size=1))
    run_config.train.micro_batch_size = 2
    run_config.train.global_batch_size = 8
    run_config.train.train_iters = 10
    run_config.scheduler.lr_warmup_fraction = 0.1
    run_config.optimizer.weight_decay = 0.25
    before = copy.deepcopy(run_config.to_dict())
    run_config.validate()
    run_config.validate()
    assert run_config.to_dict() == before

    # Later recipe edits, not a previously supplied argument bundle, drive finalization.
    run_config.optimizer.weight_decay = 0.5
    run_config.train.global_batch_size = 16
    run_config.finalize()
    assert run_config.scheduler.start_weight_decay == 0.5
    assert run_config.scheduler.end_weight_decay == 0.5
    assert run_config.scheduler.lr_decay_steps == 160
    assert run_config.scheduler.lr_warmup_steps == 16


def test_invalid_resolved_rl_batch_does_not_modify_recipe(monkeypatch, run_config):
    monkeypatch.setattr(global_vars, "_GLOBAL_ARGS", SimpleNamespace(batch_parallel_size=3))
    run_config.train.micro_batch_size = 1
    run_config.train.global_batch_size = None
    run_config.rl = RLConfig(perform_rl_step=True)
    before = copy.deepcopy(run_config.to_dict())
    run_config.validate()
    with pytest.raises(ValueError, match="divisible by global_batch_size"):
        run_config.finalize()
    assert run_config.to_dict() == before
    assert not run_config._finalized
