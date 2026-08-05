# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Focused tests for direct HybridModelConfig and PP/VPP builder behavior."""

import sys
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from megatron.core.models.hybrid.hybrid_architecture import HybridLayerSpec, PipelineSplit
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.transformer import TransformerConfig
from megatron.training.argument_utils import core_transformer_config_from_args
from megatron.training.arguments import parse_args, validate_args
from megatron.training.checkpointing import generate_state_dict
from megatron.training.models.hybrid import HybridModelBuilder, HybridModelConfig


def _make_transformer() -> TransformerConfig:
    return TransformerConfig(
        num_layers=4,
        hidden_size=128,
        num_attention_heads=1,
        pipeline_model_parallel_size=2,
        pipeline_dtype=torch.float32,
    )


def _make_layer(config: TransformerConfig) -> HybridLayerSpec:
    return HybridLayerSpec(
        module_spec=hybrid_stack_spec.submodules.mamba_layer, config=deepcopy(config)
    )


def _make_direct_config() -> HybridModelConfig:
    transformer = _make_transformer()
    layer_specs = [
        [_make_layer(transformer)],
        PipelineSplit(),
        [_make_layer(transformer)],
        PipelineSplit(),
        [_make_layer(transformer)],
        PipelineSplit(),
        [_make_layer(transformer)],
    ]
    return HybridModelConfig(transformer=transformer, vocab_size=32000, layer_specs=layer_specs)


class TestDirectHybridModelConfig:
    def test_direct_fields_default_to_none(self):
        config = HybridModelConfig(transformer=_make_transformer())

        assert config.layer_specs is None
        assert config.mtp_layer_specs is None

    def test_direct_fields_preserve_python_architecture_objects(self):
        transformer = _make_transformer()
        layer_specs = [_make_layer(transformer)]
        mtp_layer_specs = [_make_layer(transformer)]

        config = HybridModelConfig(
            transformer=transformer, layer_specs=layer_specs, mtp_layer_specs=mtp_layer_specs
        )

        assert config.layer_specs is layer_specs
        assert config.mtp_layer_specs is mtp_layer_specs

    def test_as_dict_excludes_raw_direct_specs(self):
        transformer = _make_transformer()
        config = HybridModelConfig(
            transformer=transformer,
            vocab_size=32000,
            layer_specs=[_make_layer(transformer)],
            mtp_layer_specs=[_make_layer(transformer)],
        )

        serialized = config.as_dict()

        assert "layer_specs" not in serialized
        assert "mtp_layer_specs" not in serialized
        assert serialized["vocab_size"] == 32000
        assert serialized["transformer"]["num_layers"] == 4

    def test_checkpoint_args_exclude_runtime_resolved_architecture(self):
        args = SimpleNamespace(
            resolved_hybrid_architecture=SimpleNamespace(source="direct"),
            no_save_optim=True,
            no_save_rng=True,
        )

        state_dict = generate_state_dict(
            args, model=[], optimizer=None, opt_param_scheduler=None, rng_state=None
        )

        assert hasattr(args, "resolved_hybrid_architecture")
        assert not hasattr(state_dict["args"], "resolved_hybrid_architecture")

    def test_legacy_checkpoint_keeps_the_original_args_object(self):
        args = SimpleNamespace(
            hybrid_layer_pattern="M*",
            resolved_hybrid_architecture=SimpleNamespace(source="legacy"),
            no_save_optim=True,
            no_save_rng=True,
        )

        state_dict = generate_state_dict(
            args, model=[], optimizer=None, opt_param_scheduler=None, rng_state=None
        )

        assert state_dict["args"] is args
        assert state_dict["args"].resolved_hybrid_architecture.source == "legacy"


class TestDirectHybridModelBuilder:
    def test_prepares_inferred_vpp_before_distributed_initialization(self):
        config = _make_direct_config()
        args = SimpleNamespace(
            pipeline_model_parallel_size=2,
            virtual_pipeline_model_parallel_size=None,
            overlap_p2p_comm=False,
            batch_p2p_comm=True,
            align_param_gather=False,
            _overlap_p2p_comm_before_direct_vpp=True,
            _align_param_gather_before_direct_vpp=True,
        )

        prepared = HybridModelBuilder.prepare_config_for_distributed_init(config, args)

        assert prepared is True
        assert args.virtual_pipeline_model_parallel_size == 2
        assert args.overlap_p2p_comm is True
        assert args.batch_p2p_comm is False
        assert args.align_param_gather is True
        assert config.transformer.virtual_pipeline_model_parallel_size == 2
        assert config.transformer.overlap_p2p_comm is True
        assert config.transformer.batch_p2p_comm is False

    def test_cli_validation_defers_uneven_direct_pp_vpp_topology(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["test_hybrid_direct_builder.py"])
        monkeypatch.setattr("megatron.training.arguments._print_args", lambda *args: None)
        args = parse_args()
        args.world_size = 2
        args.rank = 0
        args.pipeline_model_parallel_size = 2
        args.num_layers = 5
        args.hidden_size = 128
        args.num_attention_heads = 4
        args.max_position_embeddings = 128
        args.seq_length = 128
        args.micro_batch_size = 1
        args.train_iters = 1
        args.lr = 1.0e-4
        args.tokenizer_type = "NullTokenizer"
        args.vocab_size = 1024

        validate_args(args)
        assert args.virtual_pipeline_model_parallel_size is None
        assert args.overlap_p2p_comm is False
        assert args.align_param_gather is False

        transformer = core_transformer_config_from_args(args)
        layer = _make_layer(transformer)
        model_config = HybridModelConfig(
            transformer=transformer,
            vocab_size=1024,
            layer_specs=[
                layer,
                PipelineSplit(),
                [],
                PipelineSplit(),
                [],
                PipelineSplit(),
                [layer] * 4,
            ],
        )

        prepared = HybridModelBuilder.prepare_config_for_distributed_init(model_config, args)

        assert prepared is True
        assert args.virtual_pipeline_model_parallel_size == 2
        assert args.overlap_p2p_comm is True
        assert args.align_param_gather is True
        assert transformer.overlap_p2p_comm is True
        assert transformer.batch_p2p_comm is False

    def test_cli_validation_does_not_add_direct_vpp_state_to_legacy_patterns(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["test_hybrid_direct_builder.py"])
        monkeypatch.setattr("megatron.training.arguments._print_args", lambda *args: None)
        args = parse_args()
        args.world_size = 2
        args.rank = 0
        args.pipeline_model_parallel_size = 2
        args.num_layers = 2
        args.hidden_size = 128
        args.num_attention_heads = 4
        args.max_position_embeddings = 128
        args.seq_length = 128
        args.micro_batch_size = 1
        args.train_iters = 1
        args.lr = 1.0e-4
        args.tokenizer_type = "NullTokenizer"
        args.vocab_size = 1024
        args.hybrid_layer_pattern = "M*"

        validate_args(args)

        assert not hasattr(args, "_overlap_p2p_comm_before_direct_vpp")
        assert not hasattr(args, "_align_param_gather_before_direct_vpp")

    @pytest.mark.parametrize(
        ("runtime_pp_size", "runtime_vp_size", "message"),
        [
            pytest.param(1, None, "pipeline_model_parallel_size must match", id="pp"),
            pytest.param(2, 3, "splits disagree", id="vpp"),
        ],
    )
    def test_pre_init_topology_must_match_runtime(self, runtime_pp_size, runtime_vp_size, message):
        config = _make_direct_config()
        args = SimpleNamespace(
            pipeline_model_parallel_size=runtime_pp_size,
            virtual_pipeline_model_parallel_size=runtime_vp_size,
            overlap_p2p_comm=True,
        )

        with pytest.raises(ValueError, match=message):
            HybridModelBuilder.prepare_config_for_distributed_init(config, args)

    @patch("megatron.training.models.hybrid.compose_hooks")
    @patch("megatron.training.models.hybrid.unimodal_build_distributed_models")
    def test_resolves_before_distributed_construction_and_infers_vpp(
        self, mock_unimodal, mock_compose
    ):
        config = _make_direct_config()
        assert config.transformer.virtual_pipeline_model_parallel_size is None

        builder = HybridModelBuilder(config)
        assert config.transformer.virtual_pipeline_model_parallel_size == 2
        assert [len(segment) for segment in builder._resolved_architecture.segments] == [1, 1, 1, 1]

        observed_vp_sizes = []

        def fake_distributed_builder(*args):
            observed_vp_sizes.append(args[1].virtual_pipeline_model_parallel_size)
            return []

        mock_unimodal.side_effect = fake_distributed_builder
        mock_compose.return_value = Mock(return_value=None)

        builder.build_distributed_models(Mock(), wrap_with_ddp=False)

        assert observed_vp_sizes == [2]

    @pytest.mark.parametrize(
        ("pp_rank", "vp_stage", "expected_pre", "expected_post"),
        [(0, 0, True, False), (1, 0, False, False), (0, 1, False, False), (1, 1, False, True)],
        ids=["first-pp-first-vp", "last-pp-first-vp", "first-pp-last-vp", "last-pp-last-vp"],
    )
    @patch("megatron.training.models.hybrid.HybridModel")
    def test_default_pre_post_process_ownership(
        self, mock_model, pp_rank, vp_stage, expected_pre, expected_post
    ):
        builder = HybridModelBuilder(_make_direct_config())
        pg_collection = Mock()
        pg_collection.pp = Mock()

        with (
            patch("megatron.training.models.hybrid.is_pp_first_stage", return_value=pp_rank == 0),
            patch("megatron.training.models.hybrid.is_pp_last_stage", return_value=pp_rank == 1),
        ):
            builder.build_model(pg_collection, vp_stage=vp_stage)

        kwargs = mock_model.call_args.kwargs
        assert kwargs["pre_process"] is expected_pre
        assert kwargs["post_process"] is expected_post

    @patch("megatron.training.models.hybrid.HybridModel")
    def test_passes_resolved_architecture_to_hybrid_model(self, mock_model):
        builder = HybridModelBuilder(_make_direct_config())
        pg_collection = Mock()

        builder.build_model(pg_collection, pre_process=False, post_process=False, vp_stage=1)

        kwargs = mock_model.call_args.kwargs
        assert builder._resolved_architecture is not None
        assert kwargs["resolved_hybrid_architecture"] is builder._resolved_architecture
        assert kwargs["hybrid_layer_pattern"] is None
        assert kwargs["vp_stage"] == 1

    @patch("megatron.training.models.hybrid.is_pp_first_stage", return_value=True)
    @patch("megatron.training.models.hybrid.is_pp_last_stage", return_value=False)
    @patch("megatron.training.models.hybrid.HybridModel")
    def test_single_chunk_build_defaults_to_first_virtual_stage(
        self, mock_model, _mock_last_stage, _mock_first_stage
    ):
        builder = HybridModelBuilder(_make_direct_config())

        builder.build_model(Mock())

        kwargs = mock_model.call_args.kwargs
        assert kwargs["vp_stage"] == 0
        assert kwargs["pre_process"] is True
        assert kwargs["post_process"] is False
