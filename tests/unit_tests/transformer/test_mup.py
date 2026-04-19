# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for Maximal Update Parameterization (MuP) implementation.

These tests verify that MuP is correctly implemented in Megatron-LM:
1. Config validation and width_mult computation
2. Initialization scaling
3. Attention scaling
4. LR override computation
"""

import logging
import dataclasses
import json
import math
from argparse import ArgumentParser
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

import megatron.training.arguments as training_args_module
import megatron.training.yaml_arguments as yaml_args_module
from megatron.core.optimizer import (
    _get_megatron_optimizer_based_on_param_groups,
    get_mup_config_overrides,
    get_scaling_config_overrides,
    get_standard_config_overrides,
)
from megatron.core.config_logger import log_config_to_disk
from megatron.core.optimizer.optimizer_config import OptimizerConfig
from megatron.core.optimizer_param_scheduler import combine_param_group_overrides
from megatron.core.models.gpt.fine_grained_callables import _apply_mlp_bda_with_scaling
from megatron.core.parameterization import (
    build_resolved_model_policy,
    build_resolved_scaling_context,
    build_resolved_training_policy,
)
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.multi_token_prediction import process_mtp_loss
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules
from megatron.core.utils import init_method_normal, mup_scaled_init_method_normal
from megatron.training.arguments import (
    add_megatron_arguments,
    core_transformer_config_from_args,
    validate_args,
    validate_depth_mup_optimizer_support,
    validate_muon_scalar_optimizer_support,
)
from megatron.training.checkpointing import check_checkpoint_args, load_args_from_checkpoint
from megatron.training.yaml_arguments import (
    core_config_from_args as core_config_from_yaml_args,
    validate_yaml,
)


def _build_transformer_namespace(**overrides):
    values = {}
    for field in dataclasses.fields(TransformerConfig):
        if field.default is not dataclasses.MISSING:
            values[field.name] = field.default
        elif field.default_factory is not dataclasses.MISSING:
            values[field.name] = field.default_factory()
        else:
            assert field.name in overrides, f"Missing required override for {field.name}"
            values[field.name] = overrides.pop(field.name)
    values.update(overrides)
    return SimpleNamespace(**values)


def _prepare_parsed_args_for_core_config(args):
    # parse_args() populates CLI-backed fields only; validate_args() normally derives params_dtype.
    args.params_dtype = torch.float32
    return args


def _build_minimal_validate_yaml_namespace(**overrides):
    values = dict(
        data_path=None,
        world_size=1,
        rank=0,
        micro_batch_size=1,
        global_batch_size=1,
        num_layers_per_virtual_pipeline_stage=None,
        overlap_param_gather=False,
        overlap_grad_reduce=False,
        use_distributed_optimizer=False,
        accumulate_allreduce_grads_in_fp32=False,
        dataloader_type=None,
        lr_decay_samples=None,
        rampup_batch_size=None,
        train_iters=None,
        train_samples=None,
        lr_decay_iters=None,
        lr_warmup_iters=0,
        lr_warmup_fraction=None,
        lr_warmup_samples=0,
        encoder_num_layers=None,
        seq_length=128,
        encoder_seq_length=None,
        max_position_embeddings=128,
        decoder_seq_length=None,
        lr=1e-3,
        min_lr=1e-5,
        save=None,
        save_interval=None,
        fp16_lm_cross_entropy=False,
        account_for_embedding_in_pipeline_split=False,
        overlap_p2p_comm=False,
        model_parallel=SimpleNamespace(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
            tp_comm_overlap=False,
            sequence_parallel=False,
            fp16=False,
            bf16=False,
            params_dtype=torch.float32,
        ),
        language_model=SimpleNamespace(
            num_layers=12,
            hidden_size=1024,
            num_attention_heads=16,
            ffn_hidden_size=None,
            activation_func='gelu',
            kv_channels=None,
            scaling_recipe='none',
            fp32_residual_connection=False,
            moe_grouped_gemm=False,
        ),
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _combined_override_for_param(overrides, param, param_name):
    matches = [
        override
        for param_key, override in overrides.items()
        if param_key.matches(param, param_name)
    ]
    return combine_param_group_overrides(matches)


class TestMuPConfigValidation:
    """Tests for MuP config validation and width_mult computation."""

    def test_scaling_recipe_mup_matches_legacy_alias(self):
        legacy = TransformerConfig(
            hidden_size=1024,
            num_layers=12,
            num_attention_heads=16,
            use_mup=True,
            mup_base_hidden_size=256,
            mup_base_head_dim=64,
        )
        recipe = TransformerConfig(
            hidden_size=1024,
            num_layers=12,
            num_attention_heads=16,
            scaling_recipe='mup',
            scaling_base_hidden_size=256,
            scaling_base_head_dim=64,
        )

        assert recipe.use_mup is True
        assert recipe.scaling_recipe == 'mup'
        assert recipe.mup_width_mult == legacy.mup_width_mult
        assert recipe.mup_base_hidden_size == legacy.mup_base_hidden_size
        assert recipe.softmax_scale == legacy.softmax_scale
        assert recipe.mup_output_mult == legacy.mup_output_mult

    def test_scaling_recipe_depth_mup_preserves_distinct_recipe_identity(self):
        config = TransformerConfig(
            hidden_size=1024,
            num_layers=12,
            num_attention_heads=16,
            scaling_recipe='depth_mup',
            scaling_base_hidden_size=256,
            scaling_base_num_layers=6,
            scaling_base_head_dim=64,
        )
        context = build_resolved_scaling_context(config)

        assert config.scaling_recipe == 'depth_mup'
        assert config.use_mup is False
        assert context.recipe == 'depth_mup'
        assert context.uses_width_mup is True
        assert context.references.base_hidden_size == 256
        assert context.references.base_num_layers == 6
        assert context.references.base_head_dim == 64
        assert context.residual_branch_depth_power == pytest.approx(-1.0)
        assert context.hidden_lr_depth_power == pytest.approx(0.0)
        assert context.block_out_proj_init_depth_power == pytest.approx(0.5)
        assert context.output_mult == pytest.approx(0.25)

    def test_depth_mup_manual_overrides_can_zero_recipe_defaults(self):
        config = TransformerConfig(
            hidden_size=1024,
            num_layers=12,
            num_attention_heads=16,
            scaling_recipe='depth_mup',
            scaling_base_hidden_size=256,
            scaling_base_num_layers=6,
            scaling_residual_branch_depth_power=0.0,
            scaling_block_out_proj_init_depth_power=0.0,
        )
        context = build_resolved_scaling_context(config)

        assert context.residual_branch_depth_power == pytest.approx(0.0)
        assert context.block_out_proj_init_depth_power == pytest.approx(0.0)

    def test_mup_does_not_inherit_depth_mup_recipe_defaults(self):
        config = TransformerConfig(
            hidden_size=1024,
            num_layers=12,
            num_attention_heads=16,
            scaling_recipe='mup',
            scaling_base_hidden_size=256,
            scaling_base_num_layers=6,
        )
        context = build_resolved_scaling_context(config)

        assert context.recipe == 'mup'
        assert context.residual_branch_depth_power == pytest.approx(0.0)
        assert context.hidden_lr_depth_power == pytest.approx(0.0)
        assert context.block_out_proj_init_depth_power == pytest.approx(0.0)

    def test_conflicting_legacy_and_canonical_scaling_args_error(self):
        with pytest.raises(ValueError, match='conflicts with'):
            TransformerConfig(
                hidden_size=1024,
                num_layers=12,
                num_attention_heads=16,
                scaling_recipe='mup',
                scaling_base_hidden_size=256,
                mup_base_hidden_size=512,
            )

    def test_explicit_none_conflicts_with_legacy_use_mup_alias(self):
        with pytest.raises(ValueError, match='conflicts with --use-mup'):
            TransformerConfig(
                hidden_size=1024,
                num_layers=12,
                num_attention_heads=16,
                scaling_recipe='none',
                use_mup=True,
            )

    def test_depth_mup_conflicts_with_legacy_use_mup_alias(self):
        with pytest.raises(ValueError, match='conflicts with --use-mup'):
            TransformerConfig(
                hidden_size=1024,
                num_layers=12,
                num_attention_heads=16,
                scaling_recipe='depth_mup',
                use_mup=True,
            )

    def test_scaling_overrides_require_recipe(self):
        with pytest.raises(ValueError, match="Scaling overrides require a non-'none' scaling recipe"):
            TransformerConfig(
                hidden_size=512,
                num_layers=12,
                num_attention_heads=8,
                scaling_residual_branch_depth_power=-0.5,
            )

    def test_legacy_mup_knobs_require_mup_recipe(self):
        with pytest.raises(ValueError, match="Scaling overrides require a non-'none' scaling recipe"):
            TransformerConfig(
                hidden_size=512,
                num_layers=12,
                num_attention_heads=8,
                mup_base_hidden_size=256,
            )

    def test_scaling_base_head_dim_must_be_positive(self):
        with pytest.raises(AssertionError, match='scaling-base-head-dim'):
            TransformerConfig(
                hidden_size=1024,
                num_layers=12,
                num_attention_heads=16,
                scaling_recipe='mup',
                scaling_base_head_dim=-64,
            )

    def test_depth_mup_rejects_multi_latent_attention(self):
        with pytest.raises(NotImplementedError, match='multi_latent_attention'):
            TransformerConfig(
                hidden_size=1024,
                num_layers=12,
                num_attention_heads=16,
                scaling_recipe='depth_mup',
                multi_latent_attention=True,
            )

    def test_depth_mup_rejects_experimental_attention_variant(self):
        with pytest.raises(NotImplementedError, match='experimental attention variants'):
            TransformerConfig(
                hidden_size=1024,
                num_layers=12,
                num_attention_heads=16,
                scaling_recipe='depth_mup',
                experimental_attention_variant='gated_delta_net',
                linear_attention_freq=1,
            )

    def test_depth_mup_rejects_moe(self):
        with pytest.raises(NotImplementedError, match='MoE depth transfer'):
            TransformerConfig(
                hidden_size=1024,
                num_layers=12,
                num_attention_heads=16,
                scaling_recipe='depth_mup',
                num_moe_experts=4,
            )

    def test_mup_defaults_base_hidden_size(self):
        """use_mup without base_hidden_size defaults to hidden_size (width_mult=1.0)."""
        config = TransformerConfig(
            hidden_size=512,
            num_layers=4,
            num_attention_heads=8,
            use_mup=True,
            # mup_base_hidden_size not set - should default to hidden_size
        )
        assert config.mup_base_hidden_size == 512
        assert config.mup_width_mult == 1.0

    def test_mup_width_mult_calculation(self):
        """width_mult = hidden_size / base_hidden_size."""
        config = TransformerConfig(
            hidden_size=1024,
            num_layers=4,
            num_attention_heads=16,
            use_mup=True,
            mup_base_hidden_size=256,
        )
        assert config.mup_width_mult == 4.0

    def test_mup_width_mult_fractional(self):
        """width_mult can be fractional (smaller than base)."""
        config = TransformerConfig(
            hidden_size=128,
            num_layers=4,
            num_attention_heads=2,
            use_mup=True,
            mup_base_hidden_size=256,
        )
        assert config.mup_width_mult == 0.5

    def test_resolved_scaling_context_matches_transformer_config_fields(self):
        config = TransformerConfig(
            hidden_size=1536,
            num_layers=18,
            num_attention_heads=12,
            scaling_recipe='mup',
            scaling_base_hidden_size=384,
            scaling_base_num_layers=9,
            scaling_base_head_dim=64,
        )
        context = build_resolved_scaling_context(config)

        assert context.recipe == 'mup'
        assert context.width_mult == pytest.approx(config.mup_width_mult)
        assert context.references.base_hidden_size == config.mup_base_hidden_size
        assert context.references.base_num_layers == config.scaling_base_num_layers
        assert context.references.base_head_dim == config.mup_base_head_dim

    def test_parser_accepts_canonical_and_legacy_mup_flags(self):
        parser = ArgumentParser(allow_abbrev=False)
        parser = add_megatron_arguments(parser)

        canonical_args = parser.parse_args(
            [
                '--num-layers', '12',
                '--hidden-size', '1024',
                '--num-attention-heads', '16',
                '--no-rope-fusion',
                '--scaling-recipe', 'mup',
                '--scaling-base-hidden-size', '256',
                '--scaling-base-head-dim', '64',
            ]
        )
        canonical_args = _prepare_parsed_args_for_core_config(canonical_args)
        canonical_config = core_transformer_config_from_args(canonical_args)
        assert canonical_config.use_mup is True
        assert canonical_config.scaling_recipe == 'mup'

        depth_args = parser.parse_args(
            [
                '--num-layers', '12',
                '--hidden-size', '1024',
                '--num-attention-heads', '16',
                '--no-rope-fusion',
                '--scaling-recipe', 'depth_mup',
                '--optimizer', 'adamw',
                '--scaling-base-hidden-size', '256',
                '--scaling-base-num-layers', '6',
                '--scaling-base-head-dim', '64',
            ]
        )
        depth_args = _prepare_parsed_args_for_core_config(depth_args)
        depth_config = core_transformer_config_from_args(depth_args)
        assert depth_config.use_mup is False
        assert depth_config.scaling_recipe == 'depth_mup'
        assert depth_args.optimizer == 'adamw'
        assert canonical_config.mup_width_mult == pytest.approx(4.0)

        legacy_args = parser.parse_args(
            [
                '--num-layers', '12',
                '--hidden-size', '1024',
                '--num-attention-heads', '16',
                '--no-rope-fusion',
                '--use-mup',
                '--mup-base-hidden-size', '256',
                '--mup-base-head-dim', '64',
                '--mup-width-mult', '3.0',
            ]
        )
        legacy_args = _prepare_parsed_args_for_core_config(legacy_args)
        legacy_config = core_transformer_config_from_args(legacy_args)
        assert legacy_args.mup_width_mult == pytest.approx(3.0)
        assert legacy_config.use_mup is True
        assert legacy_config.mup_width_mult == pytest.approx(4.0)

    def test_validate_args_calls_depth_mup_hook(self):
        parser = ArgumentParser(allow_abbrev=False)
        parser = add_megatron_arguments(parser)
        args = parser.parse_args(
            [
                '--num-layers', '12',
                '--hidden-size', '1024',
                '--num-attention-heads', '16',
                '--seq-length', '128',
                '--max-position-embeddings', '128',
                '--no-rope-fusion',
                '--optimizer', 'adamw',
                '--scaling-recipe', 'depth_mup',
            ]
        )

        with patch.object(
            training_args_module,
            'validate_depth_mup_optimizer_support',
            side_effect=RuntimeError('depth-hook'),
        ), patch.object(
            training_args_module, 'validate_muon_scalar_optimizer_support', return_value=None
        ):
            with pytest.raises(RuntimeError, match='depth-hook'):
                validate_args(args)

    def test_validate_args_calls_muon_scalar_optimizer_hook(self):
        parser = ArgumentParser(allow_abbrev=False)
        parser = add_megatron_arguments(parser)
        args = parser.parse_args(
            [
                '--num-layers', '12',
                '--hidden-size', '1024',
                '--num-attention-heads', '16',
                '--seq-length', '128',
                '--max-position-embeddings', '128',
                '--no-rope-fusion',
                '--optimizer', 'muon',
                '--muon-scalar-optimizer', 'lion',
            ]
        )

        with patch.object(
            training_args_module, 'validate_depth_mup_optimizer_support', return_value=None
        ), patch.object(
            training_args_module,
            'validate_muon_scalar_optimizer_support',
            side_effect=RuntimeError('muon-hook'),
        ):
            with pytest.raises(RuntimeError, match='muon-hook'):
                validate_args(args)

    def test_yaml_namespace_without_scaling_fields_still_builds_transformer_config(self):
        args = _build_transformer_namespace(
            hidden_size=1024,
            num_layers=12,
            num_attention_heads=16,
        )
        for field_name in (
            'scaling_recipe',
            'scaling_base_hidden_size',
            'scaling_base_num_layers',
            'scaling_base_head_dim',
            'scaling_residual_branch_depth_power',
            'scaling_hidden_lr_depth_power',
            'scaling_block_out_proj_init_depth_power',
        ):
            delattr(args, field_name)

        kw_args = core_config_from_yaml_args(args, TransformerConfig)

        assert kw_args['scaling_recipe'] is None
        assert kw_args['scaling_base_hidden_size'] is None
        assert kw_args['scaling_base_num_layers'] is None
        assert kw_args['scaling_base_head_dim'] is None

    def test_yaml_namespace_preserves_depth_mup_scaling_surface(self):
        args = _build_transformer_namespace(
            hidden_size=1024,
            num_layers=12,
            num_attention_heads=16,
            scaling_recipe='depth_mup',
            scaling_base_hidden_size=256,
            scaling_base_num_layers=6,
            scaling_base_head_dim=64,
            optimizer='adamw',
        )

        kw_args = core_config_from_yaml_args(args, TransformerConfig)
        config = TransformerConfig(**kw_args)
        context = build_resolved_scaling_context(config)

        assert config.scaling_recipe == 'depth_mup'
        assert context.recipe == 'depth_mup'
        assert context.uses_width_mup is True
        assert context.residual_branch_depth_power == pytest.approx(-1.0)
        assert context.hidden_lr_depth_power == pytest.approx(0.0)
        assert context.block_out_proj_init_depth_power == pytest.approx(0.5)

    def test_depth_mup_optimizer_gate_rejects_non_adam_yaml_namespace(self):
        args = SimpleNamespace(scaling_recipe='depth_mup', optimizer='sgd')

        with pytest.raises(ValueError, match='Adam/AdamW only'):
            validate_depth_mup_optimizer_support(args)

    def test_depth_mup_optimizer_gate_allows_adamw_yaml_namespace(self):
        args = SimpleNamespace(scaling_recipe='depth_mup', optimizer='adamw')

        validate_depth_mup_optimizer_support(args)

    def test_depth_mup_optimizer_gate_rejects_nested_yaml_namespace(self):
        args = SimpleNamespace(
            optimizer='sgd', language_model=SimpleNamespace(scaling_recipe='depth_mup')
        )

        with pytest.raises(ValueError, match='Adam/AdamW only'):
            validate_depth_mup_optimizer_support(args)

    def test_depth_mup_optimizer_gate_tolerates_yaml_namespace_without_scaling_fields(self):
        validate_depth_mup_optimizer_support(SimpleNamespace())

    def test_muon_scalar_optimizer_gate_rejects_invalid_nested_yaml_namespace(self):
        args = SimpleNamespace(
            muon_scalar_optimizer='soap',
            language_model=SimpleNamespace(scaling_recipe='none'),
        )

        with pytest.raises(ValueError, match="muon_scalar_optimizer must be one of"):
            validate_muon_scalar_optimizer_support(args)

    def test_muon_scalar_optimizer_gate_allows_lion_nested_yaml_namespace(self):
        args = SimpleNamespace(muon_scalar_optimizer='lion')

        validate_muon_scalar_optimizer_support(args)

    def test_validate_yaml_calls_depth_mup_hook(self):
        args = _build_minimal_validate_yaml_namespace(
            optimizer='adamw',
            language_model=SimpleNamespace(
                num_layers=12,
                hidden_size=1024,
                num_attention_heads=16,
                ffn_hidden_size=None,
                activation_func='gelu',
                kv_channels=None,
                scaling_recipe='depth_mup',
                fp32_residual_connection=False,
                moe_grouped_gemm=False,
            ),
        )

        with patch.object(
            yaml_args_module,
            'validate_depth_mup_optimizer_support',
            side_effect=RuntimeError('yaml-depth-hook'),
        ), patch.object(
            yaml_args_module, 'validate_muon_scalar_optimizer_support', return_value=None
        ):
            with pytest.raises(RuntimeError, match='yaml-depth-hook'):
                validate_yaml(args)

    def test_validate_yaml_calls_muon_scalar_optimizer_hook(self):
        args = _build_minimal_validate_yaml_namespace(
            optimizer='adam',
            muon_scalar_optimizer='lion',
        )

        with patch.object(
            yaml_args_module, 'validate_depth_mup_optimizer_support', return_value=None
        ), patch.object(
            yaml_args_module,
            'validate_muon_scalar_optimizer_support',
            side_effect=RuntimeError('yaml-muon-hook'),
        ):
            with pytest.raises(RuntimeError, match='yaml-muon-hook'):
                validate_yaml(args)

    def test_config_logger_serializes_canonical_depth_mup_surface(self, tmp_path):
        config = TransformerConfig(
            hidden_size=1024,
            num_layers=12,
            num_attention_heads=16,
            scaling_recipe='depth_mup',
            scaling_base_hidden_size=256,
            scaling_base_num_layers=6,
            config_logger_dir=str(tmp_path),
        )

        log_config_to_disk(
            config, {'config': config}, prefix='depth_mup_config', rank_str='0_0_0_0_0'
        )

        output_path = tmp_path / 'depth_mup_config.rank_0_0_0_0_0.iter0.json'
        with output_path.open() as fp:
            payload = json.load(fp)

        serialized = payload['config']
        assert serialized['scaling_recipe'] == 'depth_mup'
        assert serialized['scaling_base_hidden_size'] == 256
        assert serialized['scaling_base_num_layers'] == 6
        assert serialized['scaling_residual_branch_depth_power'] == pytest.approx(-1.0)
        assert serialized['scaling_hidden_lr_depth_power'] == pytest.approx(0.0)
        assert serialized['scaling_block_out_proj_init_depth_power'] == pytest.approx(0.5)

    def test_checkpoint_args_compare_effective_scaling_context(self):
        runtime_args = _build_transformer_namespace(
            hidden_size=1024,
            num_layers=12,
            num_attention_heads=16,
            scaling_recipe='mup',
            scaling_base_hidden_size=256,
            scaling_base_head_dim=64,
            add_position_embedding=True,
            vocab_file=None,
            data_parallel_random_init=False,
            phase_transition_iterations=None,
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            use_dist_ckpt=False,
        )
        checkpoint_args = _build_transformer_namespace(
            hidden_size=1024,
            num_layers=12,
            num_attention_heads=16,
            use_mup=True,
            mup_base_hidden_size=256,
            mup_base_head_dim=64,
            add_position_embedding=True,
            vocab_file=None,
            data_parallel_random_init=False,
            phase_transition_iterations=None,
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            use_dist_ckpt=False,
        )
        delattr(runtime_args, 'kv_channels')
        delattr(checkpoint_args, 'kv_channels')

        with (
            patch('megatron.training.checkpointing.get_args', return_value=runtime_args),
            patch('megatron.training.checkpointing.get_checkpoint_version', return_value=3.0),
        ):
            check_checkpoint_args(checkpoint_args)

    def test_checkpoint_args_compare_effective_depth_mup_scaling_context(self):
        runtime_args = _build_transformer_namespace(
            hidden_size=1024,
            num_layers=12,
            num_attention_heads=16,
            scaling_recipe='depth_mup',
            scaling_base_hidden_size=256,
            scaling_base_num_layers=6,
            scaling_base_head_dim=64,
            optimizer='adamw',
            add_position_embedding=True,
            vocab_file=None,
            data_parallel_random_init=False,
            phase_transition_iterations=None,
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            use_dist_ckpt=False,
        )
        checkpoint_args = _build_transformer_namespace(
            hidden_size=1024,
            num_layers=12,
            num_attention_heads=16,
            scaling_recipe='depth_mup',
            scaling_base_hidden_size=256,
            scaling_base_num_layers=6,
            scaling_base_head_dim=64,
            optimizer='adamw',
            add_position_embedding=True,
            vocab_file=None,
            data_parallel_random_init=False,
            phase_transition_iterations=None,
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            use_dist_ckpt=False,
        )
        delattr(runtime_args, 'kv_channels')
        delattr(checkpoint_args, 'kv_channels')

        with (
            patch('megatron.training.checkpointing.get_args', return_value=runtime_args),
            patch('megatron.training.checkpointing.get_checkpoint_version', return_value=3.0),
        ):
            check_checkpoint_args(checkpoint_args)

    def test_checkpoint_args_raise_on_scaling_mismatch(self):
        runtime_args = _build_transformer_namespace(
            hidden_size=1024,
            num_layers=12,
            num_attention_heads=16,
            scaling_recipe='mup',
            scaling_base_hidden_size=256,
            add_position_embedding=True,
            vocab_file=None,
            data_parallel_random_init=False,
            phase_transition_iterations=None,
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            use_dist_ckpt=False,
        )
        checkpoint_args = _build_transformer_namespace(
            hidden_size=1024,
            num_layers=12,
            num_attention_heads=16,
            use_mup=True,
            mup_base_hidden_size=512,
            add_position_embedding=True,
            vocab_file=None,
            data_parallel_random_init=False,
            phase_transition_iterations=None,
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            use_dist_ckpt=False,
        )

        with (
            patch('megatron.training.checkpointing.get_args', return_value=runtime_args),
            patch('megatron.training.checkpointing.get_checkpoint_version', return_value=3.0),
        ):
            with pytest.raises(AssertionError, match='Resolved scaling context'):
                check_checkpoint_args(checkpoint_args)

    def test_checkpoint_args_raise_on_legacy_mup_knob_mismatch(self):
        runtime_args = _build_transformer_namespace(
            hidden_size=1024,
            num_layers=12,
            num_attention_heads=16,
            scaling_recipe='mup',
            scaling_base_hidden_size=256,
            mup_embedding_mult=1.2,
            mup_output_mult=0.35,
            mup_attn_scale_power=0.8,
            add_position_embedding=True,
            vocab_file=None,
            data_parallel_random_init=False,
            phase_transition_iterations=None,
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            use_dist_ckpt=False,
        )
        checkpoint_args = _build_transformer_namespace(
            hidden_size=1024,
            num_layers=12,
            num_attention_heads=16,
            use_mup=True,
            mup_base_hidden_size=256,
            mup_embedding_mult=1.1,
            mup_output_mult=0.35,
            mup_attn_scale_power=0.8,
            add_position_embedding=True,
            vocab_file=None,
            data_parallel_random_init=False,
            phase_transition_iterations=None,
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            use_dist_ckpt=False,
        )

        with (
            patch('megatron.training.checkpointing.get_args', return_value=runtime_args),
            patch('megatron.training.checkpointing.get_checkpoint_version', return_value=3.0),
        ):
            with pytest.raises(AssertionError, match='Resolved scaling context'):
                check_checkpoint_args(checkpoint_args)

    def test_use_checkpoint_args_restores_scaling_surface(self):
        args = SimpleNamespace(
            load='dummy',
            iteration=0,
            use_mp_args_from_checkpoint_args=False,
            use_tokenizer_model_from_checkpoint_args=False,
            use_mup=False,
            scaling_recipe=None,
            scaling_base_hidden_size=None,
            scaling_base_num_layers=None,
            scaling_base_head_dim=None,
            scaling_residual_branch_depth_power=None,
            scaling_hidden_lr_depth_power=None,
            scaling_block_out_proj_init_depth_power=None,
            mup_width_mult=1.0,
            mup_base_hidden_size=None,
            mup_embedding_mult=1.0,
            mup_output_mult=1.0,
            mup_base_head_dim=None,
            mup_attn_scale_power=1.0,
        )
        checkpoint_args = SimpleNamespace(
            use_mup=True,
            mup_width_mult=4.0,
            mup_base_hidden_size=256,
            mup_embedding_mult=1.5,
            mup_output_mult=0.3,
            mup_base_head_dim=64,
            mup_attn_scale_power=0.75,
            scaling_recipe='mup',
            scaling_base_hidden_size=256,
            scaling_base_num_layers=12,
            scaling_base_head_dim=64,
            scaling_residual_branch_depth_power=-0.5,
            scaling_hidden_lr_depth_power=-0.25,
            scaling_block_out_proj_init_depth_power=-0.5,
        )
        state_dict = {'args': checkpoint_args, 'checkpoint_version': 3.0, 'iteration': 17}

        with patch(
            'megatron.training.checkpointing._load_base_checkpoint',
            return_value=(state_dict, 'dummy', False, None),
        ):
            load_args_from_checkpoint(args)

        assert args.iteration == 17
        assert args.use_mup is True
        assert args.mup_embedding_mult == pytest.approx(1.5)
        assert args.mup_output_mult == pytest.approx(0.3)
        assert args.mup_attn_scale_power == pytest.approx(0.75)
        assert args.scaling_recipe == 'mup'
        assert args.scaling_base_hidden_size == 256
        assert args.scaling_base_num_layers == 12
        assert args.scaling_base_head_dim == 64
        assert args.scaling_residual_branch_depth_power == pytest.approx(-0.5)
        assert args.scaling_hidden_lr_depth_power == pytest.approx(-0.25)
        assert args.scaling_block_out_proj_init_depth_power == pytest.approx(-0.5)

    def test_use_checkpoint_args_restores_depth_mup_surface(self):
        args = SimpleNamespace(
            load='dummy',
            iteration=0,
            use_mp_args_from_checkpoint_args=False,
            use_tokenizer_model_from_checkpoint_args=False,
            use_mup=False,
            scaling_recipe=None,
            scaling_base_hidden_size=None,
            scaling_base_num_layers=None,
            scaling_base_head_dim=None,
            scaling_residual_branch_depth_power=None,
            scaling_hidden_lr_depth_power=None,
            scaling_block_out_proj_init_depth_power=None,
            mup_width_mult=1.0,
            mup_base_hidden_size=None,
            mup_embedding_mult=1.0,
            mup_output_mult=1.0,
            mup_base_head_dim=None,
            mup_attn_scale_power=1.0,
        )
        checkpoint_args = SimpleNamespace(
            use_mup=False,
            mup_width_mult=4.0,
            mup_base_hidden_size=256,
            mup_embedding_mult=1.0,
            mup_output_mult=0.25,
            mup_base_head_dim=64,
            mup_attn_scale_power=1.0,
            scaling_recipe='depth_mup',
            scaling_base_hidden_size=256,
            scaling_base_num_layers=6,
            scaling_base_head_dim=64,
            scaling_residual_branch_depth_power=-1.0,
            scaling_hidden_lr_depth_power=0.0,
            scaling_block_out_proj_init_depth_power=0.5,
        )
        state_dict = {'args': checkpoint_args, 'checkpoint_version': 3.0, 'iteration': 23}

        with patch(
            'megatron.training.checkpointing._load_base_checkpoint',
            return_value=(state_dict, 'dummy', False, None),
        ):
            load_args_from_checkpoint(args)

        assert args.iteration == 23
        assert args.use_mup is False
        assert args.scaling_recipe == 'depth_mup'
        assert args.scaling_base_hidden_size == 256
        assert args.scaling_base_num_layers == 6
        assert args.scaling_base_head_dim == 64
        assert args.scaling_residual_branch_depth_power == pytest.approx(-1.0)
        assert args.scaling_hidden_lr_depth_power == pytest.approx(0.0)
        assert args.scaling_block_out_proj_init_depth_power == pytest.approx(0.5)

    def test_mup_backward_compatible(self):
        """Default config unchanged when MuP disabled."""
        config = TransformerConfig(hidden_size=512, num_layers=4, num_attention_heads=8)
        assert config.use_mup is False
        assert config.mup_width_mult == 1.0
        assert config.mup_base_hidden_size is None


class TestMuPModelPolicy:
    @pytest.mark.parametrize('recipe', ['mup', 'depth_mup'])
    def test_model_policy_scales_embedding_and_logits(self, recipe):
        config = TransformerConfig(
            hidden_size=1024,
            num_layers=12,
            num_attention_heads=16,
            scaling_recipe=recipe,
            scaling_base_hidden_size=256,
            scaling_base_num_layers=6,
            mup_embedding_mult=1.25,
            mup_output_mult=0.2,
        )
        policy = build_resolved_model_policy(config)
        embeddings = torch.ones(2, 3)
        logits = torch.ones(2, 3)

        assert torch.equal(policy.scale_embedding_activations(embeddings), embeddings * 1.25)
        assert torch.equal(policy.scale_output_logits(logits), logits * 0.2)

    @pytest.mark.parametrize('recipe', ['mup', 'depth_mup'])
    def test_model_policy_uses_embedding_init_for_untied_readout(self, recipe):
        config = TransformerConfig(
            hidden_size=1024,
            num_layers=12,
            num_attention_heads=16,
            scaling_recipe=recipe,
            scaling_base_hidden_size=256,
            scaling_base_num_layers=6,
        )
        policy = build_resolved_model_policy(config)

        assert (
            policy.output_layer_init_method(
                share_embeddings_and_output_weights=False,
                default_init_method=config.init_method,
                embedding_init_method=config.embedding_init_method,
            )
            is config.embedding_init_method
        )
        assert (
            policy.output_layer_init_method(
                share_embeddings_and_output_weights=True,
                default_init_method=config.init_method,
                embedding_init_method=config.embedding_init_method,
            )
            is config.init_method
        )

    def test_dense_block_output_init_depth_scaling(self):
        config = TransformerConfig(
            hidden_size=1024,
            num_layers=12,
            num_attention_heads=16,
            scaling_recipe='mup',
            scaling_base_hidden_size=256,
            scaling_base_num_layers=6,
            scaling_block_out_proj_init_depth_power=-0.5,
        )
        policy = build_resolved_model_policy(config)
        init_fn = policy.dense_block_output_init_method(
            default_init_method=config.output_layer_init_method,
            init_method_std=config.init_method_std,
            num_layers=config.num_layers,
            is_hybrid_model=config.is_hybrid_model,
            output_layer_init_method_is_user_provided=False,
        )
        weights = torch.empty(200_000)
        init_fn(weights)

        expected_std = (
            config.init_method_std
            / (math.sqrt(2 * config.num_layers) * math.sqrt(config.mup_width_mult))
            * (policy.context.depth_mult ** config.scaling_block_out_proj_init_depth_power)
        )
        actual_std = weights.std().item()
        assert abs(actual_std - expected_std) < expected_std * 0.05

    def test_depth_mup_default_block_output_init_rebases_to_base_depth(self):
        config = TransformerConfig(
            hidden_size=1024,
            num_layers=12,
            num_attention_heads=16,
            scaling_recipe='depth_mup',
            scaling_base_hidden_size=256,
            scaling_base_num_layers=6,
        )
        policy = build_resolved_model_policy(config)
        init_fn = policy.dense_block_output_init_method(
            default_init_method=config.output_layer_init_method,
            init_method_std=config.init_method_std,
            num_layers=config.num_layers,
            is_hybrid_model=config.is_hybrid_model,
            output_layer_init_method_is_user_provided=False,
        )
        weights = torch.empty(200_000)
        init_fn(weights)

        expected_std = config.init_method_std / (
            math.sqrt(2 * config.scaling_base_num_layers) * math.sqrt(policy.context.width_mult)
        )
        actual_std = weights.std().item()
        assert abs(actual_std - expected_std) < expected_std * 0.05

    def test_dense_block_output_init_can_be_disabled_per_site(self):
        config = TransformerConfig(
            hidden_size=1024,
            num_layers=12,
            num_attention_heads=16,
            scaling_recipe='mup',
            scaling_base_hidden_size=256,
            scaling_base_num_layers=6,
            scaling_block_out_proj_init_depth_power=-0.5,
        )
        policy = build_resolved_model_policy(config)

        assert (
            policy.dense_block_output_init_method(
                default_init_method=config.output_layer_init_method,
                init_method_std=config.init_method_std,
                num_layers=config.num_layers,
                is_hybrid_model=config.is_hybrid_model,
                output_layer_init_method_is_user_provided=False,
                apply_depth_hook=False,
            )
            is config.output_layer_init_method
        )

    def test_plain_mlp_requires_explicit_block_output_init_scaling_opt_in(self):
        class DummyLinear(torch.nn.Module):
            def __init__(self, init_method):
                super().__init__()
                self.init_method = init_method

            def forward(self, hidden_states):
                return hidden_states, None

            def backward_dw(self):
                return None

        def fc1_builder(input_size, output_size, *, init_method, **kwargs):
            return DummyLinear(init_method)

        def fc2_builder(input_size, output_size, *, init_method, **kwargs):
            return DummyLinear(init_method)

        config = TransformerConfig(
            hidden_size=1024,
            num_layers=12,
            num_attention_heads=16,
            scaling_recipe='mup',
            scaling_base_hidden_size=256,
            scaling_base_num_layers=6,
            scaling_block_out_proj_init_depth_power=-0.5,
        )
        submodules = MLPSubmodules(linear_fc1=fc1_builder, linear_fc2=fc2_builder)

        plain_mlp = MLP(config, submodules, apply_block_output_init_scaling=False)
        scaled_mlp = MLP(config, submodules, apply_block_output_init_scaling=True)

        assert plain_mlp.linear_fc2.init_method is config.output_layer_init_method
        assert scaled_mlp.linear_fc2.init_method is not config.output_layer_init_method

    def test_transformer_layer_residual_branch_scaling_helper(self):
        config = TransformerConfig(
            hidden_size=1024,
            num_layers=12,
            num_attention_heads=16,
            scaling_recipe='mup',
            scaling_base_hidden_size=256,
            scaling_base_num_layers=6,
            scaling_residual_branch_depth_power=-0.5,
        )
        layer = object.__new__(TransformerLayer)
        layer.model_scaling_policy = build_resolved_model_policy(config)

        output = torch.ones(4, 8)
        bias = torch.ones(8)
        scaled_output, scaled_bias = layer._scale_dense_residual_branch_output(
            (output, bias),
            branch_name='self attention',
            using_fused_tp_inference_kernel=False,
        )
        expected_mult = (
            config.num_layers / config.scaling_base_num_layers
        ) ** config.scaling_residual_branch_depth_power
        assert torch.equal(scaled_output, output * expected_mult)
        assert torch.equal(scaled_bias, bias * expected_mult)

        with pytest.raises(NotImplementedError, match='Residual-branch scaling'):
            layer._scale_dense_residual_branch_output(
                (output, bias),
                branch_name='self attention',
                using_fused_tp_inference_kernel=True,
            )

    def test_depth_mup_rejects_inference_even_at_base_depth(self):
        config = TransformerConfig(
            hidden_size=1024,
            num_layers=12,
            num_attention_heads=16,
            scaling_recipe='depth_mup',
            scaling_base_hidden_size=256,
            scaling_base_num_layers=12,
        )
        layer = object.__new__(TransformerLayer)
        layer.model_scaling_policy = build_resolved_model_policy(config)
        layer.training = False

        with pytest.raises(NotImplementedError, match='during inference'):
            layer._scale_dense_residual_branch_output(
                (torch.ones(2, 2), None),
                branch_name='self attention',
                using_fused_tp_inference_kernel=False,
            )

    def test_depth_mup_rejects_fused_tp_inference_specifically(self):
        config = TransformerConfig(
            hidden_size=1024,
            num_layers=12,
            num_attention_heads=16,
            scaling_recipe='depth_mup',
            scaling_base_hidden_size=256,
            scaling_base_num_layers=12,
        )
        layer = object.__new__(TransformerLayer)
        layer.model_scaling_policy = build_resolved_model_policy(config)
        layer.training = False

        with pytest.raises(NotImplementedError, match='fused TP inference'):
            layer._scale_dense_residual_branch_output(
                (torch.ones(2, 2), None),
                branch_name='self attention',
                using_fused_tp_inference_kernel=True,
            )

    def test_transformer_layer_rejects_cross_attention_for_depth_mup(self):
        class DummyCrossAttention(torch.nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()

            def forward(self, *args, **kwargs):
                return torch.ones(1, 1, 1), None

        config = TransformerConfig(
            hidden_size=16,
            num_layers=12,
            num_attention_heads=4,
            scaling_recipe='depth_mup',
            scaling_base_hidden_size=8,
            scaling_base_num_layers=6,
        )
        submodules = TransformerLayerSubmodules(cross_attention=DummyCrossAttention)

        with pytest.raises(NotImplementedError, match='Cross-attention is out of scope for v1'):
            TransformerLayer(config=config, submodules=submodules)

    def test_depth_mup_rejects_bert_model(self):
        from megatron.core.models.bert.bert_model import BertModel

        config = TransformerConfig(
            hidden_size=16,
            num_layers=12,
            num_attention_heads=4,
            scaling_recipe='depth_mup',
            scaling_base_hidden_size=8,
            scaling_base_num_layers=6,
        )

        with pytest.raises(NotImplementedError, match='BertModel is out of scope for v1'):
            BertModel(
                config=config,
                num_tokentypes=0,
                transformer_layer_spec=None,
                vocab_size=128,
                max_sequence_length=16,
            )

    def test_depth_mup_rejects_t5_model(self):
        from megatron.core.models.T5.t5_model import T5Model

        config = TransformerConfig(
            hidden_size=16,
            num_layers=12,
            num_attention_heads=4,
            scaling_recipe='depth_mup',
            scaling_base_hidden_size=8,
            scaling_base_num_layers=6,
        )

        with pytest.raises(NotImplementedError, match='T5Model is out of scope for v1'):
            T5Model(
                config=config,
                encoder_config=config,
                transformer_encoder_layer_spec=None,
                transformer_decoder_layer_spec=None,
                vocab_size=128,
                max_sequence_length=16,
            )

    def test_depth_mup_rejects_mamba_model(self):
        from megatron.core.models.mamba.mamba_model import MambaModel

        config = TransformerConfig(
            hidden_size=16,
            num_layers=12,
            num_attention_heads=4,
            scaling_recipe='depth_mup',
            scaling_base_hidden_size=8,
            scaling_base_num_layers=6,
        )

        with pytest.raises(NotImplementedError, match='MambaModel is out of scope for v1'):
            MambaModel(
                config=config,
                mamba_stack_spec=None,
                vocab_size=128,
                max_sequence_length=16,
            )

    def test_overlap_helper_routes_through_residual_branch_scaler(self):
        class DummyCtx:
            def __enter__(self):
                return None

            def __exit__(self, exc_type, exc, tb):
                return False

        class DummyConfig:
            bias_dropout_fusion = False

        class DummyLayer:
            training = True
            hidden_dropout = 0.1
            is_moe_layer = False
            config = DummyConfig()

            def __init__(self):
                self.scaler_called = False

            def _scale_dense_residual_branch_output(
                self,
                output_with_bias,
                *,
                branch_name,
                using_fused_tp_inference_kernel,
                apply_depth_hook,
            ):
                self.scaler_called = True
                assert branch_name == 'mlp'
                assert using_fused_tp_inference_kernel is False
                assert apply_depth_hook is True
                output, _ = output_with_bias
                return (output * 2.0, None)

            def bias_dropout_add_exec_handler(self):
                return DummyCtx()

            def mlp_bda(self, training, bias_dropout_fusion):
                assert training is True
                assert bias_dropout_fusion is False

                def apply(output_with_bias, residual, hidden_dropout):
                    output, bias = output_with_bias
                    assert bias is None
                    return output + residual + hidden_dropout

                return apply

        layer = DummyLayer()
        output = torch.ones(3, 4)
        residual = torch.ones(3, 4)
        hidden_states = _apply_mlp_bda_with_scaling(layer, output, residual)

        assert layer.scaler_called is True
        assert torch.equal(hidden_states, output * 2.0 + residual + layer.hidden_dropout)

    def test_overlap_helper_disables_depth_hook_for_moe_layers(self):
        class DummyCtx:
            def __enter__(self):
                return None

            def __exit__(self, exc_type, exc, tb):
                return False

        class DummyConfig:
            bias_dropout_fusion = False

        class DummyLayer:
            training = True
            hidden_dropout = 0.1
            is_moe_layer = True
            config = DummyConfig()

            def __init__(self):
                self.scaler_called = False

            def _scale_dense_residual_branch_output(
                self,
                output_with_bias,
                *,
                branch_name,
                using_fused_tp_inference_kernel,
                apply_depth_hook,
            ):
                self.scaler_called = True
                assert branch_name == 'mlp'
                assert using_fused_tp_inference_kernel is False
                assert apply_depth_hook is False
                return output_with_bias

            def bias_dropout_add_exec_handler(self):
                return DummyCtx()

            def mlp_bda(self, training, bias_dropout_fusion):
                assert training is True
                assert bias_dropout_fusion is False

                def apply(output_with_bias, residual, hidden_dropout):
                    output, bias = output_with_bias
                    assert bias is None
                    return output + residual + hidden_dropout

                return apply

        layer = DummyLayer()
        output = torch.ones(3, 4)
        residual = torch.ones(3, 4)
        hidden_states = _apply_mlp_bda_with_scaling(layer, output, residual)

        assert layer.scaler_called is True
        assert torch.equal(hidden_states, output + residual + layer.hidden_dropout)

    def test_mup_base_hidden_size_must_be_positive(self):
        """mup_base_hidden_size must be positive."""
        with pytest.raises(AssertionError) as exc_info:
            TransformerConfig(
                hidden_size=512,
                num_layers=4,
                num_attention_heads=8,
                use_mup=True,
                mup_base_hidden_size=0,
            )
        assert "positive" in str(exc_info.value).lower()


class TestMuPInitMethods:
    """Tests for MuP initialization methods."""

    def test_mup_init_variance_scaling(self):
        """Init std scales as 1/sqrt(width_mult)."""
        base_sigma = 0.02
        width_mult = 4.0

        init_fn = init_method_normal(base_sigma / math.sqrt(width_mult))
        tensor = torch.empty(1000, 1000)
        init_fn(tensor)

        expected_std = base_sigma / math.sqrt(width_mult)  # 0.02/sqrt(4) = 0.01
        actual_std = tensor.std().item()

        # Allow 5% tolerance for statistical variance
        assert (
            abs(actual_std - expected_std) < 0.002
        ), f"Expected std ~{expected_std:.4f}, got {actual_std:.4f}"

    def test_mup_scaled_init_variance(self):
        """MuP scaled init combines depth and width scaling."""
        sigma = 0.02
        num_layers = 8
        width_mult = 4.0

        init_fn = mup_scaled_init_method_normal(sigma, num_layers, width_mult)
        tensor = torch.empty(1000, 1000)
        init_fn(tensor)

        # std = sigma / (sqrt(2*num_layers) * sqrt(width_mult))
        expected_std = sigma / (math.sqrt(2 * num_layers) * math.sqrt(width_mult))
        actual_std = tensor.std().item()

        # Allow 10% tolerance
        assert (
            abs(actual_std - expected_std) < expected_std * 0.1
        ), f"Expected std ~{expected_std:.6f}, got {actual_std:.6f}"


class TestMuPAttentionScaling:
    """Tests for MuP attention scaling."""

    def test_mup_attention_scale_power_1(self):
        """MuP uses 1/d instead of 1/sqrt(d) when power=1.0."""
        config = TransformerConfig(
            hidden_size=512,
            num_layers=4,
            num_attention_heads=8,
            use_mup=True,
            mup_base_hidden_size=128,
            mup_attn_scale_power=1.0,  # MuP default
        )
        kv_channels = config.kv_channels  # 512 / 8 = 64

        expected_scale = 1.0 / kv_channels  # 1/64 = 0.015625
        assert config.softmax_scale == expected_scale

    def test_standard_attention_scale_power_05(self):
        """Standard uses 1/sqrt(d) when power=0.5."""
        config = TransformerConfig(
            hidden_size=512,
            num_layers=4,
            num_attention_heads=8,
            use_mup=True,
            mup_base_hidden_size=128,
            mup_attn_scale_power=0.5,  # Standard default
        )
        kv_channels = config.kv_channels  # 64

        expected_scale = 1.0 / math.sqrt(kv_channels)  # 1/8 = 0.125
        assert abs(config.softmax_scale - expected_scale) < 1e-6

    def test_depth_mup_inherits_width_mup_attention_scaling(self):
        config = TransformerConfig(
            hidden_size=512,
            num_layers=8,
            num_attention_heads=8,
            scaling_recipe='depth_mup',
            scaling_base_hidden_size=128,
            scaling_base_num_layers=4,
            scaling_base_head_dim=64,
        )
        expected_scale = (config.scaling_base_head_dim**0.5) / config.kv_channels
        assert config.softmax_scale == expected_scale

    def test_attention_scale_not_set_when_disabled(self):
        """softmax_scale should not be auto-set when use_mup=False."""
        config = TransformerConfig(
            hidden_size=512, num_layers=4, num_attention_heads=8, use_mup=False  # MuP disabled
        )
        # softmax_scale defaults to None when MuP is disabled
        # (actual scaling is done in the attention layer)
        assert config.softmax_scale is None


class TestMuPWarnings:
    """Tests for MuP configuration warnings."""

    def test_mup_warns_with_custom_init_method(self):
        """Warn when MuP is enabled and init_method is user-provided."""
        with pytest.warns(UserWarning, match="scaling recipe 'mup' is enabled"):
            TransformerConfig(
                hidden_size=512,
                num_layers=4,
                num_attention_heads=8,
                use_mup=True,
                mup_base_hidden_size=128,
                init_method=init_method_normal(0.01),
            )

    def test_mup_warns_with_custom_output_layer_init_method(self):
        """Warn when MuP is enabled and output_layer_init_method is user-provided."""
        with pytest.warns(UserWarning, match="scaling recipe 'mup' is enabled"):
            TransformerConfig(
                hidden_size=512,
                num_layers=4,
                num_attention_heads=8,
                use_mup=True,
                mup_base_hidden_size=128,
                output_layer_init_method=init_method_normal(0.01),
            )

    def test_mup_warns_with_custom_embedding_init_method(self):
        """Warn when MuP is enabled and embedding_init_method is user-provided."""
        with pytest.warns(
            UserWarning, match="scaling recipe 'mup' is enabled, but custom embedding_init_method"
        ):
            TransformerConfig(
                hidden_size=512,
                num_layers=4,
                num_attention_heads=8,
                use_mup=True,
                mup_base_hidden_size=128,
                embedding_init_method=init_method_normal(0.01),
            )


class TestMuPLRScaling:
    """Tests for MuP learning rate and Adam epsilon scaling."""

    def test_mup_lr_override_computation(self):
        """Hidden LR and Adam eps scale as 1/width_mult."""
        optimizer_config = OptimizerConfig(lr=1e-3, min_lr=1e-5)
        width_mult = 4.0

        overrides = get_mup_config_overrides(optimizer_config, width_mult)

        # Should have one override for hidden layers
        assert len(overrides) == 1

        # Get the override values
        for param_key, override in overrides.items():
            expected_max_lr = 1e-3 / width_mult  # 2.5e-4
            expected_min_lr = 1e-5 / width_mult  # 2.5e-6
            expected_eps = optimizer_config.adam_eps / width_mult

            assert abs(override['max_lr'] - expected_max_lr) < 1e-10
            assert abs(override['min_lr'] - expected_min_lr) < 1e-10
            assert abs(override['eps'] - expected_eps) < 1e-15

    def test_mup_lr_no_scaling_at_unity(self):
        """No LR scaling when width_mult=1.0."""
        optimizer_config = OptimizerConfig(lr=1e-3, min_lr=1e-5)
        width_mult = 1.0

        overrides = get_mup_config_overrides(optimizer_config, width_mult)

        # Should return empty dict when no scaling needed
        assert len(overrides) == 0

    def test_mup_lr_override_has_correct_predicate(self):
        """LR override predicate correctly identifies hidden vs vector-like params.

        Per MuP paper and Microsoft's mup library:
        - Embedding layer: base LR (fan_in is vocab, finite dimension)
        - Hidden layers: scaled LR (fan_in is hidden, infinite dimension)
        - Output layer: base LR when tagged as embedding-class (Table 8 symmetry)
        """
        optimizer_config = OptimizerConfig(lr=1e-3)
        width_mult = 4.0

        overrides = get_mup_config_overrides(optimizer_config, width_mult)

        # Check the predicate is set and works correctly
        for param_key, override in overrides.items():
            assert param_key.with_name_predicate is not None
            predicate_fn = param_key.with_name_predicate.fn

            # Create mock parameters
            hidden_param = torch.nn.Parameter(torch.zeros(10, 10))

            # Hidden param with hidden-layer name should match (scaled LR)
            assert predicate_fn(hidden_param, 'decoder.layer.0.weight') is True

            # Output layer should NOT match when tagged as embedding-class.
            output_param = torch.nn.Parameter(torch.zeros(10, 10))
            output_param.is_embedding_parameter = True
            assert predicate_fn(output_param, 'output_layer.weight') is False

            # Embedding layer should NOT match (base LR) when attribute is set.
            embedding_param = torch.nn.Parameter(torch.zeros(10, 10))
            embedding_param.is_embedding_parameter = True
            assert predicate_fn(embedding_param, 'decoder.layer.0.weight') is False

            # 1D vector-like params (biases/LN) should also keep base LR.
            vector_like_param = torch.nn.Parameter(torch.zeros(10))
            assert predicate_fn(vector_like_param, 'decoder.layer.0.bias') is False

            # Shared embedding copies on tied output stages should also use base LR.
            shared_embedding_param = torch.nn.Parameter(torch.zeros(10, 10))
            shared_embedding_param.shared_embedding = True
            assert predicate_fn(shared_embedding_param, 'output_layer.weight') is False

            # Backward-compatible fallback for older modules without the attribute.
            assert predicate_fn(hidden_param, 'embedding.word_embeddings.weight') is False

    def test_mup_with_decoupled_lr_scales_hidden_only_for_lr(self):
        """With decoupled_lr, MuP scales hidden params only; embedding/output stay decoupled."""
        optimizer_config = OptimizerConfig(
            lr=1e-3, min_lr=1e-5, decoupled_lr=2e-4, decoupled_min_lr=2e-6
        )
        width_mult = 4.0

        standard_overrides = get_standard_config_overrides(optimizer_config)
        mup_overrides = get_mup_config_overrides(optimizer_config, width_mult)
        combined_overrides = {**standard_overrides, **mup_overrides}

        hidden_param = torch.nn.Parameter(torch.zeros(10, 10))
        bias_param = torch.nn.Parameter(torch.zeros(10))
        embedding_param = torch.nn.Parameter(torch.zeros(10, 10))
        embedding_param.is_embedding_parameter = True
        embedding_param.is_embedding_or_output_parameter = True
        output_param = torch.nn.Parameter(torch.zeros(10, 10))
        output_param.is_embedding_or_output_parameter = True
        output_param.is_embedding_parameter = True
        shared_output_param = torch.nn.Parameter(torch.zeros(10, 10))
        shared_output_param.is_embedding_or_output_parameter = True
        shared_output_param.shared_embedding = True

        hidden_matches = [
            override
            for param_key, override in combined_overrides.items()
            if param_key.matches(hidden_param, 'decoder.layer.0.weight')
        ]
        bias_matches = [
            override
            for param_key, override in combined_overrides.items()
            if param_key.matches(bias_param, 'decoder.layer.0.bias')
        ]
        embedding_matches = [
            override
            for param_key, override in combined_overrides.items()
            if param_key.matches(embedding_param, 'embedding.word_embeddings.weight')
        ]
        output_matches = [
            override
            for param_key, override in combined_overrides.items()
            if param_key.matches(output_param, 'output_layer.weight')
        ]
        shared_output_matches = [
            override
            for param_key, override in combined_overrides.items()
            if param_key.matches(shared_output_param, 'output_layer.weight')
        ]

        hidden_override = combine_param_group_overrides(hidden_matches)
        bias_override = combine_param_group_overrides(bias_matches)
        embedding_override = combine_param_group_overrides(embedding_matches)
        output_override = combine_param_group_overrides(output_matches)
        shared_output_override = combine_param_group_overrides(shared_output_matches)

        # Hidden params keep MuP scaling.
        assert hidden_override['max_lr'] == pytest.approx(1e-3 / width_mult)
        assert hidden_override['min_lr'] == pytest.approx(1e-5 / width_mult)
        assert hidden_override['eps'] == pytest.approx(optimizer_config.adam_eps / width_mult)

        # Biases are vector-like; MuP should not override LR/eps.
        assert 'max_lr' not in bias_override
        assert 'min_lr' not in bias_override
        assert 'eps' not in bias_override

        # Embeddings keep decoupled LR and unscaled eps.
        assert embedding_override['max_lr'] == pytest.approx(2e-4)
        assert embedding_override['min_lr'] == pytest.approx(2e-6)
        assert 'eps' not in embedding_override

        # Output params keep decoupled LR and unscaled eps (Table 8 symmetry).
        assert output_override['max_lr'] == pytest.approx(2e-4)
        assert output_override['min_lr'] == pytest.approx(2e-6)
        assert 'eps' not in output_override

        # Shared embedding output copies stay on embedding-class (unscaled) eps.
        assert shared_output_override['max_lr'] == pytest.approx(2e-4)
        assert shared_output_override['min_lr'] == pytest.approx(2e-6)
        assert 'eps' not in shared_output_override

    def test_scaling_policy_matches_legacy_optimizer_overrides_for_width_only(self):
        optimizer_config = OptimizerConfig(lr=1e-3, min_lr=1e-5)
        model_config = TransformerConfig(
            hidden_size=1024,
            num_layers=8,
            num_attention_heads=16,
            scaling_recipe='mup',
            scaling_base_hidden_size=256,
        )
        scaling_policy = build_resolved_training_policy(model_config, optimizer_type='adam')

        policy_overrides = get_scaling_config_overrides(optimizer_config, scaling_policy)
        legacy_overrides = get_mup_config_overrides(
            optimizer_config, model_config.mup_width_mult, optimizer_type='adam'
        )

        hidden_param = torch.nn.Parameter(torch.zeros(10, 10))
        bias_param = torch.nn.Parameter(torch.zeros(10))
        embedding_param = torch.nn.Parameter(torch.zeros(10, 10))
        embedding_param.is_embedding_parameter = True
        output_param = torch.nn.Parameter(torch.zeros(10, 10))
        output_param.is_embedding_parameter = True
        output_param.is_embedding_or_output_parameter = True

        sample_params = [
            (hidden_param, 'decoder.layers.0.self_attention.linear_qkv.weight'),
            (bias_param, 'decoder.layers.0.self_attention.linear_qkv.bias'),
            (embedding_param, 'embedding.word_embeddings.weight'),
            (output_param, 'output_layer.weight'),
        ]

        for param, name in sample_params:
            assert _combined_override_for_param(
                policy_overrides, param, name
            ) == _combined_override_for_param(legacy_overrides, param, name)

    def test_scaling_policy_applies_hidden_lr_depth_power_for_adam(self):
        optimizer_config = OptimizerConfig(lr=1e-3, min_lr=1e-5)
        model_config = TransformerConfig(
            hidden_size=1024,
            num_layers=16,
            num_attention_heads=16,
            scaling_recipe='mup',
            scaling_base_hidden_size=256,
            scaling_base_num_layers=4,
            scaling_hidden_lr_depth_power=-0.5,
        )
        scaling_policy = build_resolved_training_policy(model_config, optimizer_type='adam')
        overrides = get_scaling_config_overrides(optimizer_config, scaling_policy)

        hidden_param = torch.nn.Parameter(torch.zeros(10, 10))
        bias_param = torch.nn.Parameter(torch.zeros(10))
        hidden_override = _combined_override_for_param(
            overrides, hidden_param, 'decoder.layers.0.self_attention.linear_qkv.weight'
        )
        bias_override = _combined_override_for_param(
            overrides, bias_param, 'decoder.layers.0.self_attention.linear_qkv.bias'
        )

        expected_lr_mult = (1.0 / model_config.mup_width_mult) * (
            scaling_policy.context.depth_mult**model_config.scaling_hidden_lr_depth_power
        )
        assert hidden_override['max_lr'] == pytest.approx(optimizer_config.lr * expected_lr_mult)
        assert hidden_override['min_lr'] == pytest.approx(
            optimizer_config.min_lr * expected_lr_mult
        )
        assert hidden_override['eps'] == pytest.approx(
            optimizer_config.adam_eps / model_config.mup_width_mult
        )
        assert 'max_lr' not in bias_override
        assert 'min_lr' not in bias_override
        assert 'eps' not in bias_override

    def test_depth_mup_applies_adam_epsilon_depth_power(self):
        optimizer_config = OptimizerConfig(lr=1e-3, min_lr=1e-5)
        model_config = TransformerConfig(
            hidden_size=1024,
            num_layers=16,
            num_attention_heads=16,
            scaling_recipe='depth_mup',
            scaling_base_hidden_size=256,
            scaling_base_num_layers=4,
        )
        scaling_policy = build_resolved_training_policy(model_config, optimizer_type='adam')
        overrides = get_scaling_config_overrides(optimizer_config, scaling_policy)

        hidden_param = torch.nn.Parameter(torch.zeros(10, 10))
        hidden_override = _combined_override_for_param(
            overrides, hidden_param, 'decoder.layers.0.self_attention.linear_qkv.weight'
        )

        expected_lr_mult = 1.0 / scaling_policy.context.width_mult
        expected_eps_mult = (
            1.0 / scaling_policy.context.width_mult
        ) * (1.0 / scaling_policy.context.depth_mult)
        assert scaling_policy.hidden_eps_depth_power == pytest.approx(-1.0)
        assert hidden_override['max_lr'] == pytest.approx(optimizer_config.lr * expected_lr_mult)
        assert hidden_override['min_lr'] == pytest.approx(
            optimizer_config.min_lr * expected_lr_mult
        )
        assert hidden_override['eps'] == pytest.approx(
            optimizer_config.adam_eps * expected_eps_mult
        )

    def test_depth_mup_applies_adamw_epsilon_depth_power(self):
        model_config = TransformerConfig(
            hidden_size=1024,
            num_layers=16,
            num_attention_heads=16,
            scaling_recipe='depth_mup',
            scaling_base_hidden_size=256,
            scaling_base_num_layers=4,
        )
        scaling_policy = build_resolved_training_policy(model_config, optimizer_type='adamw')

        assert scaling_policy.hidden_lr_multiplier == pytest.approx(1.0 / 4.0)
        assert scaling_policy.hidden_eps_depth_power == pytest.approx(-1.0)
        assert scaling_policy.hidden_eps_multiplier == pytest.approx(1.0 / 16.0)

    def test_depth_mup_residual_multiplier_exact_depth_factors(self):
        base_depth_config = TransformerConfig(
            hidden_size=512,
            num_layers=12,
            num_attention_heads=8,
            scaling_recipe='depth_mup',
            scaling_base_hidden_size=512,
            scaling_base_num_layers=12,
        )
        double_depth_config = TransformerConfig(
            hidden_size=512,
            num_layers=24,
            num_attention_heads=8,
            scaling_recipe='depth_mup',
            scaling_base_hidden_size=512,
            scaling_base_num_layers=12,
        )

        assert build_resolved_model_policy(base_depth_config).residual_branch_multiplier == pytest.approx(1.0)
        assert build_resolved_model_policy(double_depth_config).residual_branch_multiplier == pytest.approx(0.5)

    @pytest.mark.parametrize('optimizer_type', ['adam', 'adamw'])
    def test_depth_mup_hidden_eps_depth_factor_isolated_at_double_depth(self, optimizer_type):
        config = TransformerConfig(
            hidden_size=512,
            num_layers=24,
            num_attention_heads=8,
            scaling_recipe='depth_mup',
            scaling_base_hidden_size=512,
            scaling_base_num_layers=12,
        )
        policy = build_resolved_training_policy(config, optimizer_type=optimizer_type)

        assert policy.hidden_eps_multiplier == pytest.approx(0.5)

    def test_depth_mup_rejects_sgd(self):
        model_config = TransformerConfig(
            hidden_size=1024,
            num_layers=16,
            num_attention_heads=16,
            scaling_recipe='depth_mup',
            scaling_base_hidden_size=256,
            scaling_base_num_layers=4,
        )

        with pytest.raises(ValueError, match="supports Adam/AdamW only"):
            build_resolved_training_policy(model_config, optimizer_type='sgd')

    def test_depth_mup_rejects_muon(self):
        model_config = TransformerConfig(
            hidden_size=1024,
            num_layers=16,
            num_attention_heads=16,
            scaling_recipe='depth_mup',
            scaling_base_hidden_size=256,
            scaling_base_num_layers=4,
        )

        with pytest.raises(ValueError, match="supports Adam/AdamW only"):
            build_resolved_training_policy(model_config, optimizer_type='muon')

    def test_scaling_policy_applies_hidden_lr_depth_power_for_sgd(self):
        optimizer_config = OptimizerConfig(lr=1e-3, min_lr=1e-5)
        model_config = TransformerConfig(
            hidden_size=1024,
            num_layers=16,
            num_attention_heads=16,
            scaling_recipe='mup',
            scaling_base_hidden_size=256,
            scaling_base_num_layers=4,
            scaling_hidden_lr_depth_power=-0.5,
        )
        scaling_policy = build_resolved_training_policy(model_config, optimizer_type='sgd')
        overrides = get_scaling_config_overrides(optimizer_config, scaling_policy)

        hidden_param = torch.nn.Parameter(torch.zeros(10, 10))
        bias_param = torch.nn.Parameter(torch.zeros(10))
        hidden_override = _combined_override_for_param(
            overrides, hidden_param, 'decoder.layers.0.mlp.linear_fc1.weight'
        )
        bias_override = _combined_override_for_param(
            overrides, bias_param, 'decoder.layers.0.mlp.linear_fc1.bias'
        )

        expected_hidden_lr_mult = scaling_policy.context.depth_mult ** (
            model_config.scaling_hidden_lr_depth_power
        )
        assert hidden_override['max_lr'] == pytest.approx(
            optimizer_config.lr * expected_hidden_lr_mult
        )
        assert hidden_override['min_lr'] == pytest.approx(
            optimizer_config.min_lr * expected_hidden_lr_mult
        )
        assert bias_override['max_lr'] == pytest.approx(
            optimizer_config.lr * model_config.mup_width_mult
        )
        assert bias_override['min_lr'] == pytest.approx(
            optimizer_config.min_lr * model_config.mup_width_mult
        )


class TestMuPConfigIntegration:
    """Integration tests for MuP config with init methods."""

    def test_mup_output_layer_init(self):
        """Output layer init should also scale with MuP."""
        config = TransformerConfig(
            hidden_size=1024,
            num_layers=8,
            num_attention_heads=16,
            use_mup=True,
            mup_base_hidden_size=256,
        )

        tensor = torch.empty(1000, 1000)
        config.output_layer_init_method(tensor)

        # Expected: sigma / (sqrt(2*L) * sqrt(m)) = 0.02 / (sqrt(16) * sqrt(4))
        expected_std = config.init_method_std / (
            math.sqrt(2 * config.num_layers) * math.sqrt(config.mup_width_mult)
        )
        actual_std = tensor.std().item()

        assert abs(actual_std - expected_std) < expected_std * 0.15


class TestMuPOptimizerTypeHandling:
    """Tests for MuP optimizer-specific override behavior."""

    def test_adamw_uses_standard_optimizer_path(self):
        param = torch.nn.Parameter(torch.ones(1))
        optimizer_config = OptimizerConfig(
            optimizer='adamw',
            lr=1e-3,
            weight_decay=0.1,
            decoupled_weight_decay=True,
        )

        raw_optimizer, _ = _get_megatron_optimizer_based_on_param_groups(
            config=optimizer_config,
            model_chunks=[],
            param_groups=[{'params': [param]}],
            skip_megatron_wrapping=True,
        )

        assert isinstance(raw_optimizer, torch.optim.AdamW)

    def test_sgd_scales_vector_like_lr_only(self):
        """SGD scales vector-like params by width_mult; hidden params keep base LR."""
        optimizer_config = OptimizerConfig(lr=1e-3, min_lr=1e-5)
        width_mult = 4.0

        for sgd_variant in ['sgd', 'SGD', 'Sgd']:
            overrides = get_mup_config_overrides(
                optimizer_config, width_mult, optimizer_type=sgd_variant
            )
            assert len(overrides) == 1

            param_key, override = next(iter(overrides.items()))
            assert override['max_lr'] == pytest.approx(1e-3 * width_mult)
            assert override['min_lr'] == pytest.approx(1e-5 * width_mult)
            assert 'eps' not in override

            hidden_param = torch.nn.Parameter(torch.zeros(10, 10))
            bias_param = torch.nn.Parameter(torch.zeros(10))
            embedding_param = torch.nn.Parameter(torch.zeros(10, 10))
            embedding_param.is_embedding_parameter = True
            output_param = torch.nn.Parameter(torch.zeros(10, 10))
            output_param.is_embedding_parameter = True

            assert param_key.matches(hidden_param, 'decoder.layer.0.weight') is False
            assert param_key.matches(bias_param, 'decoder.layer.0.bias') is True
            assert param_key.matches(embedding_param, 'embedding.word_embeddings.weight') is True
            assert param_key.matches(output_param, 'output_layer.weight') is True

    def test_sgd_with_decoupled_lr_preserves_embedding_output_precedence(self):
        """With decoupled_lr, embedding/output keep decoupled LR under SGD MuP."""
        optimizer_config = OptimizerConfig(
            lr=1e-3, min_lr=1e-5, decoupled_lr=2e-4, decoupled_min_lr=2e-6
        )
        width_mult = 4.0

        standard_overrides = get_standard_config_overrides(optimizer_config)
        mup_overrides = get_mup_config_overrides(optimizer_config, width_mult, optimizer_type='sgd')
        combined_overrides = {**standard_overrides, **mup_overrides}

        hidden_param = torch.nn.Parameter(torch.zeros(10, 10))
        bias_param = torch.nn.Parameter(torch.zeros(10))
        embedding_param = torch.nn.Parameter(torch.zeros(10, 10))
        embedding_param.is_embedding_parameter = True
        embedding_param.is_embedding_or_output_parameter = True
        output_param = torch.nn.Parameter(torch.zeros(10, 10))
        output_param.is_embedding_parameter = True
        output_param.is_embedding_or_output_parameter = True
        shared_output_param = torch.nn.Parameter(torch.zeros(10, 10))
        shared_output_param.shared_embedding = True
        shared_output_param.is_embedding_or_output_parameter = True

        hidden_matches = [
            override
            for param_key, override in combined_overrides.items()
            if param_key.matches(hidden_param, 'decoder.layer.0.weight')
        ]
        bias_matches = [
            override
            for param_key, override in combined_overrides.items()
            if param_key.matches(bias_param, 'decoder.layer.0.bias')
        ]
        embedding_matches = [
            override
            for param_key, override in combined_overrides.items()
            if param_key.matches(embedding_param, 'embedding.word_embeddings.weight')
        ]
        output_matches = [
            override
            for param_key, override in combined_overrides.items()
            if param_key.matches(output_param, 'output_layer.weight')
        ]
        shared_output_matches = [
            override
            for param_key, override in combined_overrides.items()
            if param_key.matches(shared_output_param, 'output_layer.weight')
        ]

        hidden_override = combine_param_group_overrides(hidden_matches)
        bias_override = combine_param_group_overrides(bias_matches)
        embedding_override = combine_param_group_overrides(embedding_matches)
        output_override = combine_param_group_overrides(output_matches)
        shared_output_override = combine_param_group_overrides(shared_output_matches)

        # Hidden params keep base LR under SGD in current uniform-width setup.
        assert 'max_lr' not in hidden_override
        assert 'min_lr' not in hidden_override
        assert 'eps' not in hidden_override

        # Biases are vector-like and scale up by width_mult.
        assert bias_override['max_lr'] == pytest.approx(1e-3 * width_mult)
        assert bias_override['min_lr'] == pytest.approx(1e-5 * width_mult)
        assert 'eps' not in bias_override

        # Embedding/output params keep explicit decoupled LR precedence.
        assert embedding_override['max_lr'] == pytest.approx(2e-4)
        assert embedding_override['min_lr'] == pytest.approx(2e-6)
        assert output_override['max_lr'] == pytest.approx(2e-4)
        assert output_override['min_lr'] == pytest.approx(2e-6)
        assert shared_output_override['max_lr'] == pytest.approx(2e-4)
        assert shared_output_override['min_lr'] == pytest.approx(2e-6)
        assert 'eps' not in embedding_override
        assert 'eps' not in output_override
        assert 'eps' not in shared_output_override

    def test_adam_scales_lr_by_default(self):
        """Adam optimizer should scale LR and eps; default optimizer_type is adam."""
        optimizer_config = OptimizerConfig(lr=1e-3, min_lr=1e-5)
        width_mult = 4.0

        # Explicit adam
        overrides_explicit = get_mup_config_overrides(
            optimizer_config, width_mult, optimizer_type='adam'
        )
        assert len(overrides_explicit) == 1

        # Default (should behave like adam for backward compat)
        overrides_default = get_mup_config_overrides(optimizer_config, width_mult)
        assert len(overrides_default) == 1

        for param_key, override in overrides_explicit.items():
            expected_max_lr = 1e-3 / width_mult  # 2.5e-4
            expected_eps = optimizer_config.adam_eps / width_mult
            assert abs(override['max_lr'] - expected_max_lr) < 1e-10
            assert abs(override['eps'] - expected_eps) < 1e-15

    def test_non_adam_does_not_set_eps_override(self):
        """Non-Adam optimizers should not receive MuP epsilon overrides."""
        optimizer_config = OptimizerConfig(lr=1e-3, min_lr=1e-5)
        width_mult = 4.0

        for non_adam_optimizer in ['muon', 'dist_muon']:
            overrides = get_mup_config_overrides(
                optimizer_config, width_mult, optimizer_type=non_adam_optimizer
            )
            assert len(overrides) == 1
            for _, override in overrides.items():
                assert override['max_lr'] == pytest.approx(1e-3 / width_mult)
                assert override['min_lr'] == pytest.approx(1e-5 / width_mult)
                assert 'eps' not in override

    @pytest.mark.parametrize('optimizer_type', ['muon', 'dist_muon'])
    def test_muon_excludes_muon_managed_matrices_from_mup_overrides(self, optimizer_type):
        """Muon-managed 2D params should use Muon scaling only, not MuP LR overrides."""
        optimizer_config = OptimizerConfig(lr=1e-3, min_lr=1e-5, muon_scale_mode='unit_rms_norm')
        width_mult = 4.0

        overrides = get_mup_config_overrides(
            optimizer_config, width_mult, optimizer_type=optimizer_type
        )

        muon_managed_param = torch.nn.Parameter(torch.zeros(10, 10))
        muon_managed_param.is_embedding_or_output_parameter = False
        output_param = torch.nn.Parameter(torch.zeros(10, 10))
        output_param.is_embedding_or_output_parameter = True
        output_param.is_output_parameter = True
        output_param.is_embedding_parameter = True
        bias_param = torch.nn.Parameter(torch.zeros(10))

        muon_managed_matches = [
            override
            for param_key, override in overrides.items()
            if param_key.matches(
                muon_managed_param, 'decoder.layers.0.self_attention.linear_proj.weight'
            )
        ]
        output_matches = [
            override
            for param_key, override in overrides.items()
            if param_key.matches(output_param, 'output_layer.weight')
        ]
        bias_matches = [
            override
            for param_key, override in overrides.items()
            if param_key.matches(bias_param, 'decoder.layers.0.self_attention.linear_proj.bias')
        ]

        muon_managed_override = combine_param_group_overrides(muon_managed_matches)
        output_override = combine_param_group_overrides(output_matches)
        bias_override = combine_param_group_overrides(bias_matches)

        # Muon-managed matrix params are excluded from Adam-style MuP LR overrides.
        assert 'max_lr' not in muon_managed_override
        assert 'min_lr' not in muon_managed_override
        assert 'eps' not in muon_managed_override

        # Output params remain on the scalar optimizer path at base LR/eps.
        assert 'max_lr' not in output_override
        assert 'min_lr' not in output_override
        assert 'eps' not in output_override

        # Vector-like params stay unscaled.
        assert 'max_lr' not in bias_override
        assert 'min_lr' not in bias_override
        assert 'eps' not in bias_override

    @pytest.mark.parametrize('optimizer_type', ['muon', 'dist_muon'])
    def test_muon_warns_for_spectral_scale_mode(self, optimizer_type):
        """Muon+MuP should warn when scale mode is spectral."""
        optimizer_config = OptimizerConfig(lr=1e-3, min_lr=1e-5, muon_scale_mode='spectral')
        width_mult = 4.0

        with patch('megatron.core.optimizer.log_single_rank') as mock_warn:
            overrides = get_mup_config_overrides(
                optimizer_config, width_mult, optimizer_type=optimizer_type
            )

        assert len(overrides) == 1
        mock_warn.assert_called_once()
        _, level, message = mock_warn.call_args[0]
        assert level == logging.WARNING
        assert "Both MuP and muon_scale_mode=spectral are enabled." in message
        assert "--muon-scale-mode unit_rms_norm" in message

    @pytest.mark.parametrize('optimizer_type', ['muon', 'dist_muon'])
    def test_muon_unit_rms_norm_mode_has_no_warning(self, optimizer_type):
        """Muon+MuP should not warn when scale mode is unit_rms_norm."""
        optimizer_config = OptimizerConfig(lr=1e-3, min_lr=1e-5, muon_scale_mode='unit_rms_norm')
        width_mult = 4.0

        with patch('megatron.core.optimizer.log_single_rank') as mock_warn:
            overrides = get_mup_config_overrides(
                optimizer_config, width_mult, optimizer_type=optimizer_type
            )

        assert len(overrides) == 1
        mock_warn.assert_not_called()

    @pytest.mark.parametrize('optimizer_type', ['muon', 'dist_muon'])
    def test_muon_warns_for_spectral_mode_at_unity_width_mult(self, optimizer_type):
        """Muon+MuP warning should still fire when width_mult==1.0."""
        optimizer_config = OptimizerConfig(lr=1e-3, min_lr=1e-5, muon_scale_mode='spectral')
        width_mult = 1.0

        with patch('megatron.core.optimizer.log_single_rank') as mock_warn:
            overrides = get_mup_config_overrides(
                optimizer_config, width_mult, optimizer_type=optimizer_type
            )

        assert len(overrides) == 0
        mock_warn.assert_called_once()
        _, level, message = mock_warn.call_args[0]
        assert level == logging.WARNING
        assert "Both MuP and muon_scale_mode=spectral are enabled." in message


class TestMuPMTPLossScaling:
    """Tests for MuP scaling integration with MTP loss processing."""

    def test_process_mtp_loss_applies_scale_hook(self):
        config = TransformerConfig(
            hidden_size=8, num_layers=2, num_attention_heads=2, mtp_num_layers=1
        )
        hidden_states = torch.ones(2, 1, 4)
        labels = torch.ones(1, 4, dtype=torch.long)
        loss_mask = torch.ones_like(labels, dtype=torch.float32)
        observed_logits_mean = {'value': None}

        def output_layer(hidden, weight=None, runtime_gather_output=None):
            return hidden.clone(), None

        def scale_logits_fn(logits):
            return logits * 3.0

        def compute_language_model_loss(mtp_labels, mtp_logits):
            observed_logits_mean['value'] = mtp_logits.mean().item()
            return torch.ones_like(mtp_labels, dtype=mtp_logits.dtype)

        process_mtp_loss(
            hidden_states=hidden_states,
            labels=labels,
            loss_mask=loss_mask,
            output_layer=output_layer,
            output_weight=None,
            runtime_gather_output=None,
            is_training=False,
            compute_language_model_loss=compute_language_model_loss,
            config=config,
            cp_group=None,
            packed_seq_params=None,
            scale_logits_fn=scale_logits_fn,
        )

        assert observed_logits_mean['value'] == pytest.approx(3.0)
