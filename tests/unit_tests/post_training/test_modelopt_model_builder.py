# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for model_provider integration with ModelOpt model_builder."""

from argparse import Namespace
from unittest.mock import Mock

import pytest

import model_provider as mp
from megatron.core.transformer.transformer_config import TransformerConfig


def _sentinel_builder(return_value, calls):
    """Create a builder stub that records invocation."""

    def _builder(args, pre_process, post_process, vp_stage, config=None, pg_collection=None):
        calls.append(
            {
                "args": args,
                "pre_process": pre_process,
                "post_process": post_process,
                "vp_stage": vp_stage,
                "config": config,
                "pg_collection": pg_collection,
            }
        )
        return return_value

    return _builder


def test_model_provider_switches_to_modelopt_builder(monkeypatch):
    """Ensure model_provider delegates to ModelOpt builder when enabled."""
    args = Namespace(record_memory_history=False, modelopt_enabled=True)
    modelopt_calls = []
    original_calls = []

    modelopt_result = object()
    original_result = object()

    # Force ModelOpt availability and stub builders.
    monkeypatch.setattr(mp, "has_nvidia_modelopt", True)
    monkeypatch.setattr(mp, "get_args", lambda: args)
    monkeypatch.setattr(
        mp, "modelopt_gpt_hybrid_builder", _sentinel_builder(modelopt_result, modelopt_calls)
    )

    # original_builder should be ignored when ModelOpt is enabled.
    original_builder = _sentinel_builder(original_result, original_calls)

    returned = mp.model_provider(
        original_builder,
        pre_process=False,
        post_process=False,
        vp_stage=1,
        config="cfg",
        pg_collection="pg",
    )

    assert returned is modelopt_result
    assert modelopt_calls == [
        {
            "args": args,
            "pre_process": False,
            "post_process": False,
            "vp_stage": 1,
            "config": "cfg",
            "pg_collection": "pg",
        }
    ]
    assert len(original_calls) == 0


@pytest.mark.parametrize('table_backend', [None, 'local', 'row_a2a'])
def test_declarative_modelopt_hybrid_preserves_engram_scope(monkeypatch, table_backend):
    """The alternative builder must reject Engram rather than silently construct a plain model."""
    module = pytest.importorskip('megatron.post_training.model_builder')
    transformer = TransformerConfig(
        num_layers=2,
        hidden_size=128,
        num_attention_heads=1,
        engram_layer_ids=[0] if table_backend else None,
        engram_hash_table_min_sizes=[17, 19],
        engram_table_backend=table_backend or 'local',
    )
    config = module.ModelOptHybridModelConfig(
        transformer=transformer, vocab_size=64, hybrid_layer_pattern='*-'
    )
    builder = module.ModelOptHybridModelBuilder(config)
    args = Namespace()
    get_args = Mock(return_value=args)
    returned = object()
    delegate = Mock(return_value=returned)
    monkeypatch.setattr(module, 'get_args', get_args)
    monkeypatch.setattr(module, 'modelopt_gpt_hybrid_builder', delegate)
    groups = Mock()

    if table_backend:
        with pytest.raises(ValueError, match='Engram does not support ModelOpt'):
            builder.build_model(groups, pre_process=True, post_process=False)
        get_args.assert_not_called()
        delegate.assert_not_called()
    else:
        assert builder.build_model(groups, pre_process=True, post_process=False) is returned
        delegate.assert_called_once_with(args, True, False, None, pg_collection=groups)


def test_legacy_modelopt_builder_rejects_engram_before_constructing_config(monkeypatch):
    """The legacy model_provider route enforces the same explicit unsupported combination."""
    module = pytest.importorskip('megatron.post_training.model_builder')
    configure = Mock()
    monkeypatch.setattr(module, 'core_transformer_config_from_args', configure)
    with pytest.raises(ValueError, match='Engram does not support ModelOpt'):
        module.modelopt_gpt_hybrid_builder(Namespace(engram_layer_ids=[0]), True, True)
    configure.assert_not_called()
