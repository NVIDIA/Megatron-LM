# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Runtime integration for layers implemented outside Megatron Core."""

from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from megatron.core.models.hybrid import MTPSplit, PipelineSplit
from megatron.core.models.hybrid.hybrid_block import HybridStack
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe.moe_layer_config import MoELayerConfig
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training import checkpointing
from megatron.training.models.hybrid import HybridModelConfig
from megatron.training.training import (
    _hybrid_config_list_moe_logging_metadata,
    num_floating_point_operations,
)
from megatron.training.utils.checkpoint_utils import read_run_config
from tests.unit_tests.models.hybrid.external_layer import ExternalGainConfig, ExternalGainLayer


def _config(config_type=ExternalGainConfig, **kwargs):
    return config_type(num_layers=3, hidden_size=16, num_attention_heads=2, **kwargs)


def _bare(module_type, config, **attributes):
    """Build only the module hierarchy; constructors would allocate CUDA model state."""
    module = module_type.__new__(module_type)
    MegatronModule.__init__(module, config)
    for name, value in attributes.items():
        setattr(module, name, value)
    return module


def _stack(config, layers):
    return _bare(HybridStack, config, layers=torch.nn.ModuleList(layers))


def _model(config, layers, *, post_process=True, pp_group=None):
    return _bare(
        HybridModel,
        config,
        decoder=_stack(config, layers),
        mtp_process=False,
        post_process=post_process,
        vocab_size=32,
        pg_collection=SimpleNamespace(pp=pp_group),
    )


def _flops(*models, model_config=None):
    return num_floating_point_operations(
        model_config or HybridModelConfig(transformer=models[0].config, vocab_size=32),
        batch_size=2,
        sequence_length=128,
        total_real_tokens_in_batch=7,
        seqlen_squared_sum_in_batch=25,
        hybrid_models=list(models),
    )


@pytest.mark.parametrize("layer_flops", [0, 123, 123.5])
def test_training_forwards_packed_sequence_counts_to_external_estimator(monkeypatch, layer_flops):
    config = _config()
    estimate_layer = Mock(return_value=layer_flops)
    layer = _bare(ExternalGainLayer, config)
    monkeypatch.setattr(layer, "estimate_flops", estimate_layer)
    model = _model(config, [layer])
    stale_config = HybridModelConfig(
        transformer=TransformerConfig(num_layers=3, hidden_size=32, num_attention_heads=2),
        vocab_size=None,
    )

    result = _flops(model, model_config=stale_config)

    estimate_layer.assert_called_once_with(
        total_real_tokens_in_batch=7, seqlen_squared_sum_in_batch=25, include_backward=True
    )
    assert result == layer_flops + 21_504


def _mock_pipeline(monkeypatch):
    pp_group = object()
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 2)
    monkeypatch.setattr(torch.distributed, "get_backend", lambda group: "gloo")
    return pp_group


@pytest.mark.parametrize("with_layers,expected_local", [(False, 0), (True, 672)])
def test_virtual_chunks_are_summed_before_reducing_only_the_pipeline_group(
    monkeypatch, with_layers, expected_local
):
    pp_group = _mock_pipeline(monkeypatch)
    reductions = []

    def reduce_peer(value, *, group):
        assert group is pp_group
        reductions.append(value.item())
        value.add_(1000)

    monkeypatch.setattr(torch.distributed, "all_reduce", reduce_peer)
    config = _config()
    chunks = [
        _model(
            config,
            [_bare(ExternalGainLayer, config)] if with_layers else [],
            post_process=False,
            pp_group=pp_group,
        )
        for _ in range(2)
    ]
    assert _flops(*chunks) == expected_local + 1000
    assert reductions == [expected_local]


@pytest.mark.parametrize("local_estimate", [None, float("nan"), -1, 1j])
def test_estimation_failures_propagate_across_pipeline_ranks(monkeypatch, local_estimate):
    pp_group = _mock_pipeline(monkeypatch)
    reductions = []

    def reduce_failure(value, *, group):
        assert group is pp_group
        if local_estimate is not None:
            assert torch.isnan(value).item()
        reductions.append(value.item())
        value.fill_(float("nan"))

    monkeypatch.setattr(torch.distributed, "all_reduce", reduce_failure)
    config = _config()
    model = _model(
        config, [_bare(ExternalGainLayer, config)], post_process=False, pp_group=pp_group
    )
    if local_estimate is not None:
        # Composite overrides must also be validated before entering the collective.
        monkeypatch.setattr(model, "estimate_flops", Mock(return_value=local_estimate))
    with pytest.raises((ValueError, TypeError)):
        _flops(model)
    assert len(reductions) == 1  # A local error must still reach the shared collective.


@pytest.mark.parametrize("repeated_mtp", [False, True])
def test_builtin_moe_metadata_with_external_layers_uses_standard_tracker_conventions(repeated_mtp):
    dense = _config()
    moe = MoELayerConfig.from_config(_config(num_moe_experts=4, moe_z_loss_coeff=0.01))
    configs = [dense, moe, PipelineSplit, moe, MTPSplit, moe, MTPSplit, moe]

    metadata = _hybrid_config_list_moe_logging_metadata(
        configs, mtp_use_repeated_layer=repeated_mtp, moe_num_hash_layers=0
    )

    assert metadata == (["load_balancing_loss", "z_loss"], 5, 3 if repeated_mtp else 4, True, 0)


def test_builtin_hash_metadata_with_external_non_moe_layer():
    dense = _config()
    moe = MoELayerConfig.from_config(_config(num_moe_experts=4))
    assert _hybrid_config_list_moe_logging_metadata([dense, moe, moe], False, 9) == (
        ["load_balancing_loss"],
        3,
        2,
        True,
        2,
    )


def test_external_non_moe_config_does_not_enable_moe_metrics():
    config = _config()
    assert _hybrid_config_list_moe_logging_metadata([config], False, 0) == ([], 1, 0, False, 0)


def test_saved_hybrid_family_preserves_checkpoint_policy_and_runtime_args(run_config):
    config = _config()
    model = Mock(spec=HybridModel)
    model.hybrid_layer_config_list = [config]
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
    run_config.model = HybridModelConfig(
        transformer=config, vocab_size=32, hybrid_layer_config_list=[config]
    )

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
    assert checkpointing._validate_checkpoint_model_family(
        args, Namespace(is_hybrid_model=False), [model], checkpoint_model=run_config.model.as_dict()
    ) == (True, True, None)


def test_external_layer_keeps_config_list_gpt_conversion_restriction():
    model = Mock(spec=HybridModel)
    model.hybrid_layer_config_list = [_config()]
    with pytest.raises(RuntimeError, match="config-list-defined hybrid model"):
        checkpointing._validate_checkpoint_model_family(
            Namespace(), Namespace(hybrid_layer_pattern=None), [model]
        )


def test_external_layer_fields_are_recorded_in_run_config(run_config, tmp_path):
    root = TransformerConfig(num_layers=2, hidden_size=32, num_attention_heads=4)
    custom = ExternalGainConfig.from_config(root)
    custom.gain_init = 0.5
    custom.notes.append("architecture-specific")
    run_config.model = HybridModelConfig(
        transformer=root, hybrid_layer_config_list=[custom, custom], vocab_size=64
    )
    path = tmp_path / "run_config.yaml"
    run_config.to_yaml(path)
    for saved in (run_config.to_dict(), read_run_config(str(path))):
        layers = saved["model"]["hybrid_layer_config_list"]
        assert layers[0]["_target_"] == (
            "tests.unit_tests.models.hybrid.external_layer.ExternalGainConfig"
        )
        assert layers[0]["gain_init"] == 0.5
        assert layers[0]["notes"] == ["external", "architecture-specific"]
        assert layers[0] == layers[1]
