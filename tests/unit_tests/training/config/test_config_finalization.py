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


@pytest.mark.parametrize("invalid_policy", ["ddp_offload", "model_pipeline"])
def test_invalid_configuration_is_not_partially_finalized(monkeypatch, run_config, invalid_policy):
    monkeypatch.setattr(global_vars, "_GLOBAL_ARGS", None)
    transformer = TransformerConfig(num_layers=2, hidden_size=32, num_attention_heads=4)
    transformer.cuda_graph_impl = "transformer_engine"
    run_config.model = GPTModelConfig(transformer=transformer, vocab_size=128, seq_length=16)
    run_config.train.micro_batch_size = 1
    run_config.train.global_batch_size = 8
    run_config.rl = RLConfig(perform_rl_step=True, rl_offload_optimizer_during_inference=True)
    if invalid_policy == "ddp_offload":
        run_config.ddp.fp8_param_gather = True
        run_config.ddp.reuse_grad_buf_for_mxfp8_param_ag = True
        error, message = ValueError, "parameters share the gradient-buffer storage"
    else:
        # Native recipes can edit model layout before startup; validate it before RNG derivation.
        transformer.num_layers = 4
        transformer.pipeline_model_parallel_size = 2
        transformer.virtual_pipeline_model_parallel_size = 3
        monkeypatch.setattr("megatron.training.config.container.safe_get_world_size", lambda: 2)
        error, message = AssertionError, "Make sure the number of model chunks"
    before = copy.deepcopy(run_config.to_dict())
    for check in (run_config.validate, run_config.validate, run_config.finalize):
        with pytest.raises(error, match=message):
            check()
        assert run_config.to_dict() == before
        assert not run_config._finalized
        assert not run_config.rng.te_rng_tracker

    if invalid_policy == "model_pipeline":
        with pytest.raises(error, match=message):
            run_config.model.finalize()
        assert run_config.to_dict() == before

    # A rejected configuration remains editable and can be finalized after correction.
    if invalid_policy == "ddp_offload":
        run_config.ddp.reuse_grad_buf_for_mxfp8_param_ag = False
    else:
        transformer.virtual_pipeline_model_parallel_size = 2
    run_config.finalize()
    assert run_config._finalized
    assert run_config.ddp.grad_buffer_offload
    assert run_config.rng.te_rng_tracker


@pytest.mark.parametrize("storage", ["dict", "yaml"])
@pytest.mark.parametrize(
    "model_cls", [BertModelConfig, T5ModelConfig, LLaVAModelConfig, MimoModelConfig]
)
def test_serialization_starts_a_new_lifecycle(
    monkeypatch, run_config, model_cls, storage, tmp_path
):
    monkeypatch.setattr(global_vars, "_GLOBAL_ARGS", None)
    kwargs = {"encoder_num_layers": 2} if model_cls is T5ModelConfig else {}
    run_config.model = model_cls(
        transformer=run_config.model.transformer, seq_length=16, vocab_size=128, **kwargs
    )
    run_config.train.micro_batch_size = 1
    run_config.finalize()
    # Training attaches live objects to the transformer; checkpoint metadata must omit them.
    transformer = run_config.model.transformer
    transformer.timers = object()
    transformer.grad_scale_func = lambda value: value
    transformer.no_sync_func = [lambda: None]
    serialized = run_config.to_dict()
    assert "_finalized" not in serialized
    for name in ("timers", "grad_scale_func", "no_sync_func"):
        assert name not in serialized["model"]["transformer"]
    if storage == "yaml":
        path = tmp_path / "run_config.yaml"
        run_config.to_yaml(path)
        restored = type(run_config).from_yaml(path)
    else:
        restored = type(run_config).from_dict(serialized)
    assert type(restored.model) is model_cls
    assert restored.model.transformer.timers is None
    assert restored.model.transformer.grad_scale_func is None
    assert restored.model.transformer.no_sync_func is None
    assert not restored._finalized
    assert restored.to_dict() == serialized
    restored.finalize()
    assert restored._finalized
    assert restored.to_dict() == serialized
