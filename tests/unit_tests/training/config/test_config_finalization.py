# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Container validation is read-only; runtime initialization owns derivation."""

import copy
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
from megatron.training.models import (
    BertModelConfig,
    GPTModelConfig,
    LLaVAModelConfig,
    MimoModelConfig,
    T5ModelConfig,
)


@pytest.mark.parametrize("inference", [False, True])
@pytest.mark.parametrize(
    "transformer_impl,cuda_graph_impl",
    [
        ("transformer_engine", "transformer_engine"),
        ("transformer_engine", "local"),
        ("local", "local"),
    ],
)
def test_validation_is_read_only_and_finalization_runs_once(
    monkeypatch, run_config, inference, transformer_impl, cuda_graph_impl
):
    transformer = TransformerConfig(
        num_layers=2, hidden_size=32, num_attention_heads=4, transformer_impl=transformer_impl
    )
    # Native configs can enable graphs after construction, before runtime setup.
    transformer.cuda_graph_impl = cuda_graph_impl
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

    monkeypatch.setattr(global_vars, "_GLOBAL_ARGS", None)
    rng_finalize = Mock(wraps=cfg.rng.finalize_model_config)
    monkeypatch.setattr(cfg.rng, "finalize_model_config", rng_finalize)
    cfg.finalize()
    assert cfg._finalized
    assert cfg.rng.te_rng_tracker is (transformer_impl == "transformer_engine")
    assert transformer.use_te_rng_tracker is cfg.rng.te_rng_tracker
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
    monkeypatch.setattr(global_vars, "_GLOBAL_ARGS", None)
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


@pytest.mark.parametrize(
    "model_cls", [BertModelConfig, T5ModelConfig, LLaVAModelConfig, MimoModelConfig]
)
def test_serialization_starts_a_new_lifecycle(monkeypatch, run_config, model_cls):
    monkeypatch.setattr(global_vars, "_GLOBAL_ARGS", None)
    kwargs = {"encoder_num_layers": 2} if model_cls is T5ModelConfig else {}
    run_config.model = model_cls(
        transformer=run_config.model.transformer, seq_length=16, vocab_size=128, **kwargs
    )
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
