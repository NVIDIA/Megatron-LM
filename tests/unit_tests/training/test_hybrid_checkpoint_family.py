# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Checkpoint family metadata for Python-defined HybridModel architectures."""

from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.transformer.attention_layer_config import AttentionLayerConfig
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training import checkpointing


def _hybrid_model():
    config = TransformerConfig(num_layers=1, hidden_size=16, num_attention_heads=2)
    model = Mock(spec=HybridModel)
    model.hybrid_layer_config_list = [AttentionLayerConfig.from_config(config)]
    return model


def test_saved_hybrid_family_preserves_checkpoint_policy_and_runtime_args(run_config):
    model = _hybrid_model()
    model.state_dict_for_save_checkpoint.return_value = {"weight": 1}
    args = Namespace(
        ckpt_format="torch_dist",
        save="/old",
        no_save_optim=False,
        padded_vocab_size=16,
        is_hybrid_model=False,
    )
    original_args = vars(args).copy()
    run_config.checkpoint.save = "/configured"
    run_config.checkpoint.save_optim = False
    run_config.tokenizer.padded_vocab_size = 32

    state = checkpointing.generate_state_dict(
        args, [model], None, None, None, iteration=3, ckpt_format="torch", save="/actual"
    )

    saved_args = state["args"]
    assert saved_args.is_hybrid_model is True
    assert saved_args.ckpt_format == "torch"
    assert saved_args.save == "/actual"
    assert saved_args.no_save_optim is True
    assert saved_args.padded_vocab_size == 32
    assert vars(args) == original_args
    assert state["model"] == {"weight": 1}
    assert checkpointing._maybe_setup_gpt_to_hybrid_load(
        args, saved_args, [model], load_optim=True, saved_save_optim=True
    ) == (None, False)


@pytest.mark.parametrize("wrapper", ["module", "language_model"])
def test_wrapped_config_list_hybrid_is_recognized(wrapper):
    model = SimpleNamespace(**{wrapper: _hybrid_model()})
    assert checkpointing._contains_config_list_hybrid_model(model)
    assert checkpointing._validate_checkpoint_model_family(
        Namespace(), Namespace(is_hybrid_model=True), [model]
    ) == (True, True, None)


def test_config_list_cannot_use_pattern_based_gpt_conversion():
    with pytest.raises(RuntimeError, match="config-list-defined hybrid model"):
        checkpointing._maybe_setup_gpt_to_hybrid_load(
            Namespace(),
            Namespace(hybrid_layer_pattern=None),
            [_hybrid_model()],
            load_optim=False,
            saved_save_optim=False,
        )


@pytest.mark.parametrize(
    "checkpoint_model",
    [
        {"transformer": {"is_hybrid_model": True}},
        {"hybrid_layer_config_list": [{"_target_": "AttentionLayerConfig"}]},
    ],
)
def test_saved_model_config_identifies_hybrid_without_pattern(checkpoint_model):
    assert checkpointing._maybe_setup_gpt_to_hybrid_load(
        Namespace(),
        Namespace(is_hybrid_model=False),
        [_hybrid_model()],
        load_optim=False,
        saved_save_optim=False,
        checkpoint_model=checkpoint_model,
    ) == (None, False)
    with pytest.raises(RuntimeError, match="current run builds a non-hybrid model"):
        checkpointing._validate_checkpoint_model_family(
            Namespace(), Namespace(), [torch.nn.Module()], checkpoint_model=checkpoint_model
        )


def test_saved_model_config_marker_takes_precedence_over_stale_args():
    assert checkpointing._validate_checkpoint_model_family(
        Namespace(),
        Namespace(is_hybrid_model=True),
        [torch.nn.Module()],
        checkpoint_model={"transformer": {"is_hybrid_model": False}},
    ) == (False, False, None)


@pytest.mark.parametrize("pattern_field", ["hybrid_layer_pattern", "hybrid_override_pattern"])
def test_legacy_hybrid_patterns_remain_recognized(pattern_field):
    assert checkpointing._validate_checkpoint_model_family(
        Namespace(), Namespace(**{pattern_field: "*"}), [_hybrid_model()]
    ) == (True, True, "*")


@pytest.mark.parametrize(
    "ckpt_format, ckpt_type",
    [
        ("torch_dist", checkpointing.CheckpointType.GLOBAL),
        ("torch_dcp", checkpointing.CheckpointType.TORCH_DCP),
        ("fsdp_dtensor", checkpointing.CheckpointType.FSDP_DTENSOR),
    ],
)
@pytest.mark.parametrize("runtime_is_hybrid", [False, True])
def test_incompatible_family_is_rejected_before_loading_tensors(
    run_config, monkeypatch, ckpt_format, ckpt_type, runtime_is_hybrid
):
    model = _hybrid_model() if runtime_is_hybrid else Mock(spec=torch.nn.Module)
    saved_args = Namespace(is_hybrid_model=not runtime_is_hybrid)
    run_config.checkpoint.load = "/checkpoint"
    run_config.checkpoint.ckpt_format = ckpt_format
    run_config.checkpoint.auto_detect_ckpt_format = False
    read_checkpoint = Mock(return_value=({"args": saved_args}, "/checkpoint", False, ckpt_type))
    monkeypatch.setattr(checkpointing, "get_args", lambda: Namespace())
    monkeypatch.setattr(checkpointing, "unwrap_model", lambda models: models)
    monkeypatch.setattr(checkpointing, "_load_base_checkpoint", read_checkpoint)
    monkeypatch.setattr(checkpointing, "_read_checkpoint_run_config", lambda *args: None)

    with pytest.raises(RuntimeError, match="config-list-defined hybrid model|non-hybrid model"):
        checkpointing.load_checkpoint([model], None, None)

    # Only common metadata was requested; no rank-local tensor load or model update ran.
    read_checkpoint.assert_called_once()
    assert read_checkpoint.call_args.kwargs["rank0"] is True
    model.load_state_dict.assert_not_called()
