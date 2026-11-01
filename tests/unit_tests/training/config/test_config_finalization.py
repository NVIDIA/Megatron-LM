# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Container validation is read-only; runtime initialization owns derivation."""

import copy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from megatron.core.rerun_state_machine import RerunStateMachine
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
    rng_finalize = Mock(wraps=cfg.rng.finalize_model_config)
    monkeypatch.setattr(cfg.rng, "finalize_model_config", rng_finalize)
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
    rng_finalize.assert_called_once_with(model)
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


@pytest.mark.parametrize(
    "section,field,invalid,valid",
    [
        ("rerun_state_machine", "rerun_mode", "report_stats", "report_determinism_stats"),
        ("rerun_state_machine", "error_injection_type", "unknown", "persistent_error"),
    ],
)
def test_resiliency_validation_observes_recipe_edits(run_config, section, field, invalid, valid):
    config = getattr(run_config, section)
    setattr(config, field, invalid)
    before = copy.deepcopy(run_config.to_dict())
    for _ in range(2):
        with pytest.raises(ValueError):
            run_config.validate()
        assert run_config.to_dict() == before
        assert not run_config._finalized
    setattr(config, field, valid)
    before = copy.deepcopy(run_config.to_dict())
    run_config.validate()
    run_config.validate()
    assert run_config.to_dict() == before


@pytest.mark.parametrize("native_model", [False, True])
@pytest.mark.parametrize(
    "fp16,loss_scale,fake",
    [(False, None, False), (True, None, False), (True, 128, False), (False, None, True)],
)
@pytest.mark.parametrize("loss_check,grad_check", [(True, False), (False, True), (True, True)])
def test_rerun_nan_normalization_is_central(
    monkeypatch, run_config, native_model, fp16, loss_scale, fake, loss_check, grad_check
):
    monkeypatch.setattr(global_vars, "_GLOBAL_ARGS", SimpleNamespace(batch_parallel_size=1))
    run_config.train.micro_batch_size = 1
    run_config.optimizer.fp16 = fp16
    run_config.optimizer.loss_scale = loss_scale
    run_config.dist.fake_process_group = fake
    if native_model:
        transformer = TransformerConfig(num_layers=2, hidden_size=32, num_attention_heads=4)
        transformer.fp16 = fp16
        run_config.model = GPTModelConfig(transformer=transformer, vocab_size=128, seq_length=16)
    run_config.rerun_state_machine.check_for_nan_in_loss = loss_check
    run_config.ddp.check_for_nan_in_grad = grad_check
    before = copy.deepcopy(run_config.to_dict())
    run_config.validate()
    assert run_config.to_dict() == before
    run_config.finalize()
    disabled = fake or (fp16 and loss_scale is None)
    assert run_config.rerun_state_machine.check_for_nan_in_loss == (loss_check and not disabled)
    assert run_config.ddp.check_for_nan_in_grad == (grad_check and not disabled)


def test_tracker_iterations_are_merged_once_without_mutating_cli(monkeypatch, run_config, tmp_path):
    monkeypatch.setattr(global_vars, "_GLOBAL_ARGS", SimpleNamespace(batch_parallel_size=1))
    tracker = tmp_path / "rejected.txt"
    tracker.write_text(
        "".join(
            f"ts=now node=test device=0 jobID={job} rank=0 iteration={iteration} "
            "status=RERUN_DISABLED result=nan message='loss'\n"
            for iteration in (3, 5)
            for job in (1, 2)
        )
    )
    run_config.train.micro_batch_size = 1
    cli_skips = [1, 3]
    run_config.train.iterations_to_skip = cli_skips
    run_config.rerun_state_machine.result_rejected_tracker_filename = str(tracker)
    read_tracker = Mock(wraps=RerunStateMachine.get_skipped_iterations_from_tracker_file)
    monkeypatch.setattr(RerunStateMachine, "get_skipped_iterations_from_tracker_file", read_tracker)
    run_config.validate()
    run_config.validate()
    read_tracker.assert_not_called()
    run_config.finalize()
    run_config.finalize()
    read_tracker.assert_called_once_with(str(tracker))
    assert run_config.train.iterations_to_skip == [1, 3, 5]
    assert cli_skips == [1, 3]
    restored = type(run_config).from_dict(run_config.to_dict())
    restored.finalize()
    assert restored.train.iterations_to_skip == [1, 3, 5]
