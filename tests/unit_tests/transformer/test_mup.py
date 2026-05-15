# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for Maximal Update Parameterization (MuP) implementation.

These tests verify that MuP is correctly implemented in Megatron-LM:
1. Config validation and width_mult computation
2. Initialization scaling
3. Attention scaling
4. LR override computation
"""

import argparse
import dataclasses
import functools
import logging
import math
import warnings
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from megatron.core.optimizer import (
    get_mup_config_overrides,
    get_scaling_config_overrides,
    get_standard_config_overrides,
)
from megatron.core.optimizer.optimizer_config import OptimizerConfig
from megatron.core.optimizer_param_scheduler import combine_param_group_overrides
from megatron.core.parameterization import (
    allow_scaling_policy_eval,
    build_legacy_mup_training_policy,
    build_model_scaling_policy,
    build_scaling_context,
    build_training_scaling_policy,
)
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.multi_token_prediction import process_mtp_loss
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import (
    TransformerLayer,
    TransformerLayerSubmodules,
    _get_mlp_builder_module,
)
from megatron.core.utils import init_method_normal, mup_scaled_init_method_normal
from megatron.training.arguments import add_megatron_arguments, validate_depth_mup_optimizer_support
from megatron.training.yaml_arguments import core_config_from_args


def _combined_override_for_param(overrides, param, param_name):
    matches = [
        override for param_key, override in overrides.items() if param_key.matches(param, param_name)
    ]
    return combine_param_group_overrides(matches)


class TestMuPConfigValidation:
    """Tests for MuP config validation and width_mult computation."""

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
        assert config.scaling_recipe == 'mup'
        assert config.scaling_base_hidden_size == 512

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
        assert config.scaling_recipe == 'mup'
        assert config.scaling_base_hidden_size == 256

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

    def test_mup_backward_compatible(self):
        """Default config unchanged when MuP disabled."""
        config = TransformerConfig(hidden_size=512, num_layers=4, num_attention_heads=8)
        assert config.use_mup is False
        assert config.mup_width_mult == 1.0
        assert config.mup_base_hidden_size is None
        assert config.scaling_recipe == 'none'
        assert config.scaling_base_hidden_size is None

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

    def test_scaling_recipe_mup_sets_legacy_fields(self):
        """Canonical MuP fields populate the legacy fields used by existing call sites."""
        config = TransformerConfig(
            hidden_size=1024,
            num_layers=4,
            num_attention_heads=16,
            scaling_recipe='mup',
            scaling_base_hidden_size=256,
            scaling_base_head_dim=64,
        )

        assert config.use_mup is True
        assert config.mup_base_hidden_size == 256
        assert config.mup_base_head_dim == 64
        assert config.mup_width_mult == pytest.approx(4.0)
        assert config.scaling_base_hidden_size == 256
        assert config.scaling_base_head_dim == 64

    def test_legacy_mup_fields_resolve_to_canonical_recipe(self):
        """Legacy MuP flags remain compatible but are not separate state."""
        config = TransformerConfig(
            hidden_size=1024,
            num_layers=4,
            num_attention_heads=16,
            use_mup=True,
            mup_base_hidden_size=256,
            mup_base_head_dim=64,
        )

        assert config.scaling_recipe == 'mup'
        assert config.scaling_base_hidden_size == 256
        assert config.scaling_base_head_dim == 64
        assert build_scaling_context(config).width_mult == pytest.approx(4.0)

    def test_scaling_recipe_none_rejects_scaling_overrides(self):
        """Scaling fields cannot silently affect standard parameterization."""
        with pytest.raises(ValueError, match="Scaling overrides"):
            TransformerConfig(
                hidden_size=1024,
                num_layers=4,
                num_attention_heads=16,
                scaling_recipe='none',
                scaling_base_hidden_size=256,
            )

    def test_use_mup_conflicts_with_scaling_recipe_none(self):
        """The deprecated MuP boolean cannot override an explicit canonical recipe."""
        with pytest.raises(ValueError, match="conflicts"):
            TransformerConfig(
                hidden_size=1024,
                num_layers=4,
                num_attention_heads=16,
                scaling_recipe='none',
                use_mup=True,
            )

    def test_canonical_and_legacy_base_hidden_must_match(self):
        """Canonical and deprecated base hidden-size fields are aliases."""
        with pytest.raises(ValueError, match="conflicts"):
            TransformerConfig(
                hidden_size=1024,
                num_layers=4,
                num_attention_heads=16,
                scaling_recipe='mup',
                scaling_base_hidden_size=256,
                mup_base_hidden_size=512,
            )

    def test_deprecated_width_mult_must_match_derived_value(self):
        """mup_width_mult is accepted only when it matches the derived width."""
        config = TransformerConfig(
            hidden_size=1024,
            num_layers=4,
            num_attention_heads=16,
            scaling_recipe='mup',
            scaling_base_hidden_size=256,
            mup_width_mult=4.0,
        )
        assert config.mup_width_mult == pytest.approx(4.0)

        with pytest.raises(ValueError, match="must match the derived"):
            TransformerConfig(
                hidden_size=1024,
                num_layers=4,
                num_attention_heads=16,
                scaling_recipe='mup',
                scaling_base_hidden_size=256,
                mup_width_mult=2.0,
            )

    def test_scaling_override_without_recipe_is_rejected(self):
        """Base scaling fields do not implicitly enable MuP."""
        with pytest.raises(ValueError, match="Scaling overrides"):
            TransformerConfig(
                hidden_size=1024,
                num_layers=4,
                num_attention_heads=16,
                scaling_base_hidden_size=256,
            )

    def test_depth_mup_resolves_distinct_recipe_defaults(self):
        """Depth-MuP is width-MuP-family behavior without setting the legacy use_mup bit."""
        config = TransformerConfig(
            hidden_size=1024,
            num_layers=12,
            num_attention_heads=16,
            scaling_recipe='depth_mup',
            scaling_base_hidden_size=256,
            scaling_base_num_layers=6,
            scaling_base_head_dim=64,
        )
        context = build_scaling_context(config)

        assert config.scaling_recipe == 'depth_mup'
        assert config.use_mup is False
        assert context.uses_width_mup is True
        assert context.is_depth_mup is True
        assert context.width_mult == pytest.approx(4.0)
        assert context.depth_mult == pytest.approx(2.0)
        assert context.base_hidden_size == 256
        assert context.base_num_layers == 6
        assert context.base_head_dim == 64
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
        context = build_scaling_context(config)

        assert context.residual_branch_depth_power == pytest.approx(0.0)
        assert context.block_out_proj_init_depth_power == pytest.approx(0.0)

    def test_mup_does_not_inherit_depth_mup_defaults(self):
        config = TransformerConfig(
            hidden_size=1024,
            num_layers=12,
            num_attention_heads=16,
            scaling_recipe='mup',
            scaling_base_hidden_size=256,
            scaling_base_num_layers=6,
        )
        context = build_scaling_context(config)

        assert context.recipe == 'mup'
        assert context.depth_mult == pytest.approx(2.0)
        assert context.residual_branch_depth_power == pytest.approx(0.0)
        assert context.hidden_lr_depth_power == pytest.approx(0.0)
        assert context.block_out_proj_init_depth_power == pytest.approx(0.0)

    def test_depth_mup_conflicts_with_legacy_use_mup_alias(self):
        with pytest.raises(ValueError, match='conflicts with --use-mup'):
            TransformerConfig(
                hidden_size=1024,
                num_layers=12,
                num_attention_heads=16,
                scaling_recipe='depth_mup',
                use_mup=True,
            )

    def test_depth_mup_rejects_unsupported_attention_and_moe_surfaces(self):
        with pytest.raises(NotImplementedError, match='multi_latent_attention'):
            TransformerConfig(
                hidden_size=1024,
                num_layers=12,
                num_attention_heads=16,
                scaling_recipe='depth_mup',
                multi_latent_attention=True,
            )
        with pytest.raises(NotImplementedError, match='experimental attention variants'):
            TransformerConfig(
                hidden_size=1024,
                num_layers=12,
                num_attention_heads=16,
                scaling_recipe='depth_mup',
                experimental_attention_variant='gated_delta_net',
                linear_attention_freq=1,
            )
        with pytest.raises(NotImplementedError, match='MoE depth transfer'):
            TransformerConfig(
                hidden_size=1024,
                num_layers=12,
                num_attention_heads=16,
                scaling_recipe='depth_mup',
                num_moe_experts=4,
            )
        with pytest.raises(NotImplementedError, match='Hybrid/Mamba'):
            TransformerConfig(
                hidden_size=1024,
                num_layers=12,
                num_attention_heads=16,
                scaling_recipe='depth_mup',
                is_hybrid_model=True,
            )
        with pytest.raises(NotImplementedError, match='MTP depth transfer'):
            TransformerConfig(
                hidden_size=1024,
                num_layers=12,
                num_attention_heads=16,
                scaling_recipe='depth_mup',
                mtp_num_layers=1,
            )


class TestScalingRecipeSurfaces:
    """Tests for public config surfaces that feed the scaling context."""

    SCALING_FIELD_NAMES = {
        'scaling_recipe',
        'scaling_base_hidden_size',
        'scaling_base_num_layers',
        'scaling_base_head_dim',
        'scaling_residual_branch_depth_power',
        'scaling_hidden_lr_depth_power',
        'scaling_block_out_proj_init_depth_power',
        'use_mup',
        'mup_width_mult',
        'mup_base_hidden_size',
        'mup_embedding_mult',
        'mup_output_mult',
        'mup_base_head_dim',
        'mup_attn_scale_power',
    }

    def test_cli_parser_accepts_canonical_scaling_args(self):
        """The explicit scaling arg group owns canonical and legacy MuP flags."""
        parser = argparse.ArgumentParser(allow_abbrev=False)
        add_megatron_arguments(parser)

        args, _ = parser.parse_known_args(
            [
                '--scaling-recipe',
                'mup',
                '--scaling-base-hidden-size',
                '256',
                '--mup-base-head-dim',
                '64',
            ]
        )

        assert args.scaling_recipe == 'mup'
        assert args.scaling_base_hidden_size == 256
        assert args.mup_base_head_dim == 64
        assert args.mup_width_mult == 1.0

        depth_args, _ = parser.parse_known_args(
            [
                '--scaling-recipe',
                'depth_mup',
                '--scaling-base-hidden-size',
                '256',
                '--scaling-base-num-layers',
                '6',
                '--scaling-residual-branch-depth-power',
                '-1.0',
            ]
        )

        assert depth_args.scaling_recipe == 'depth_mup'
        assert depth_args.scaling_base_num_layers == 6
        assert depth_args.scaling_residual_branch_depth_power == pytest.approx(-1.0)

    def test_cli_explicit_mup_width_mult_one_is_validated(self):
        """Explicit legacy width multiplier must match the derived value, even at 1.0."""
        parser = argparse.ArgumentParser(allow_abbrev=False)
        add_megatron_arguments(parser)

        args, _ = parser.parse_known_args(
            [
                '--scaling-recipe',
                'mup',
                '--scaling-base-hidden-size',
                '256',
                '--mup-width-mult',
                '1.0',
            ]
        )
        args.hidden_size = 1024

        with pytest.raises(ValueError, match="must match the derived"):
            build_scaling_context(args)

    def test_yaml_core_config_defaults_missing_scaling_fields(self):
        """Existing YAML files may omit the new scaling fields."""
        values = {}
        for field in dataclasses.fields(TransformerConfig):
            if field.name in self.SCALING_FIELD_NAMES:
                continue
            if field.default is not dataclasses.MISSING:
                values[field.name] = field.default
            elif field.default_factory is not dataclasses.MISSING:
                values[field.name] = field.default_factory()
            elif field.type is int:
                values[field.name] = 1
            else:
                values[field.name] = None
        values['hidden_size'] = 512
        values['num_layers'] = 2
        values['num_attention_heads'] = 8

        kwargs = core_config_from_args(SimpleNamespace(**values), TransformerConfig)

        assert kwargs['scaling_recipe'] is None
        assert kwargs['scaling_base_hidden_size'] is None
        assert kwargs['mup_width_mult'] == 1.0

    def test_yaml_default_width_mult_is_not_treated_as_explicit(self):
        """Full legacy YAML files may materialize the old default width multiplier."""
        yaml_args = SimpleNamespace(
            hidden_size=1024,
            scaling_recipe=None,
            scaling_base_hidden_size=None,
            scaling_base_head_dim=None,
            use_mup=True,
            mup_width_mult=1.0,
            mup_base_hidden_size=256,
            mup_embedding_mult=1.0,
            mup_output_mult=1.0,
            mup_base_head_dim=None,
            mup_attn_scale_power=1.0,
        )

        context = build_scaling_context(yaml_args)

        assert context.width_mult == pytest.approx(4.0)

    def test_scaling_context_matches_legacy_checkpoint_and_canonical_args(self):
        """Checkpoint compatibility compares effective scaling, not flag spelling."""
        legacy_checkpoint_args = SimpleNamespace(
            hidden_size=1024,
            use_mup=True,
            mup_base_hidden_size=256,
            mup_width_mult=1.0,
            mup_embedding_mult=1.0,
            mup_output_mult=1.0,
            mup_base_head_dim=64,
            mup_attn_scale_power=1.0,
        )
        canonical_args = SimpleNamespace(
            hidden_size=1024,
            scaling_recipe='mup',
            scaling_base_hidden_size=256,
            scaling_base_head_dim=64,
            use_mup=False,
            mup_width_mult=1.0,
            mup_base_hidden_size=None,
            mup_embedding_mult=1.0,
            mup_output_mult=1.0,
            mup_base_head_dim=None,
            mup_attn_scale_power=1.0,
        )

        assert build_scaling_context(
            legacy_checkpoint_args
        ) == build_scaling_context(canonical_args)

    def test_checkpoint_scaling_sync_populates_canonical_fields(self):
        """Old checkpoints with only legacy MuP fields become canonical before copy."""
        from megatron.training.checkpointing import _sync_checkpoint_scaling_args

        legacy_checkpoint_args = SimpleNamespace(
            hidden_size=1024,
            use_mup=True,
            mup_base_hidden_size=256,
            mup_width_mult=1.0,
            mup_embedding_mult=1.0,
            mup_output_mult=1.0,
            mup_base_head_dim=64,
            mup_attn_scale_power=1.0,
        )

        _sync_checkpoint_scaling_args(legacy_checkpoint_args)

        assert legacy_checkpoint_args.scaling_recipe == 'mup'
        assert legacy_checkpoint_args.scaling_base_hidden_size == 256
        assert legacy_checkpoint_args.scaling_base_head_dim == 64
        assert legacy_checkpoint_args.mup_width_mult == pytest.approx(4.0)

    def test_check_checkpoint_args_compares_effective_scaling_context(self, monkeypatch):
        """The real checkpoint check accepts legacy and canonical spellings if equivalent."""
        from megatron.training import checkpointing

        runtime_args = SimpleNamespace(
            num_layers=2,
            hidden_size=1024,
            num_attention_heads=16,
            add_position_embedding=True,
            vocab_file=None,
            data_parallel_random_init=False,
            phase_transition_iterations=None,
            use_dist_ckpt=False,
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            scaling_recipe='mup',
            scaling_base_hidden_size=256,
            scaling_base_head_dim=64,
            use_mup=False,
            mup_width_mult=1.0,
            mup_base_hidden_size=None,
            mup_embedding_mult=1.0,
            mup_output_mult=1.0,
            mup_base_head_dim=None,
            mup_attn_scale_power=1.0,
        )
        checkpoint_args = SimpleNamespace(
            num_layers=2,
            hidden_size=1024,
            num_attention_heads=16,
            add_position_embedding=True,
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            use_mup=True,
            mup_base_hidden_size=256,
            mup_width_mult=1.0,
            mup_embedding_mult=1.0,
            mup_output_mult=1.0,
            mup_base_head_dim=64,
            mup_attn_scale_power=1.0,
        )
        monkeypatch.setattr(checkpointing, 'get_args', lambda: runtime_args)
        monkeypatch.setattr(checkpointing, 'get_checkpoint_version', lambda: 3.0)

        checkpointing.check_checkpoint_args(checkpoint_args)

    def test_load_checkpoint_args_clears_optional_scaling_fields(self, monkeypatch):
        """use-checkpoint-args must clear stale optional canonical fields."""
        from megatron.training import checkpointing

        checkpoint_args = SimpleNamespace(
            num_layers=2,
            hidden_size=1024,
            num_attention_heads=16,
            use_mup=True,
            mup_base_hidden_size=256,
            mup_width_mult=1.0,
            mup_embedding_mult=1.0,
            mup_output_mult=1.0,
            mup_attn_scale_power=1.0,
        )
        state_dict = {'args': checkpoint_args, 'iteration': 7}
        monkeypatch.setattr(
            checkpointing,
            '_load_base_checkpoint',
            lambda *args, **kwargs: (state_dict, 'model_optim_rng.pt', False, None),
        )
        runtime_args = SimpleNamespace(
            load='dummy-checkpoint',
            iteration=0,
            scaling_recipe='mup',
            scaling_base_hidden_size=256,
            scaling_base_head_dim=64,
            mup_base_head_dim=64,
            use_tokenizer_model_from_checkpoint_args=False,
            use_mp_args_from_checkpoint_args=False,
        )

        checkpointing.load_args_from_checkpoint(runtime_args)

        assert runtime_args.iteration == 7
        assert runtime_args.scaling_recipe == 'mup'
        assert runtime_args.scaling_base_hidden_size == 256
        assert runtime_args.scaling_base_head_dim is None
        assert runtime_args.mup_base_head_dim is None
        assert runtime_args.mup_width_mult == pytest.approx(4.0)

    def test_load_checkpoint_args_restores_depth_mup_scaling_fields(self, monkeypatch):
        """use-checkpoint-args must preserve the canonical depth-MuP surface."""
        from megatron.training import checkpointing

        checkpoint_args = SimpleNamespace(
            num_layers=12,
            hidden_size=1024,
            num_attention_heads=16,
            scaling_recipe='depth_mup',
            scaling_base_hidden_size=256,
            scaling_base_num_layers=6,
            scaling_base_head_dim=64,
            scaling_residual_branch_depth_power=-1.0,
            scaling_hidden_lr_depth_power=0.0,
            scaling_block_out_proj_init_depth_power=0.5,
            use_mup=False,
            mup_width_mult=1.0,
            mup_embedding_mult=1.0,
            mup_output_mult=1.0,
            mup_attn_scale_power=1.0,
        )
        state_dict = {'args': checkpoint_args, 'iteration': 11}
        monkeypatch.setattr(
            checkpointing,
            '_load_base_checkpoint',
            lambda *args, **kwargs: (state_dict, 'model_optim_rng.pt', False, None),
        )
        runtime_args = SimpleNamespace(
            load='dummy-checkpoint',
            iteration=0,
            use_tokenizer_model_from_checkpoint_args=False,
            use_mp_args_from_checkpoint_args=False,
        )

        checkpointing.load_args_from_checkpoint(runtime_args)

        assert runtime_args.iteration == 11
        assert runtime_args.use_mup is False
        assert runtime_args.scaling_recipe == 'depth_mup'
        assert runtime_args.scaling_base_hidden_size == 256
        assert runtime_args.scaling_base_num_layers == 6
        assert runtime_args.scaling_base_head_dim == 64
        assert runtime_args.scaling_residual_branch_depth_power == pytest.approx(-1.0)
        assert runtime_args.scaling_hidden_lr_depth_power == pytest.approx(0.0)
        assert runtime_args.scaling_block_out_proj_init_depth_power == pytest.approx(0.5)

    def test_distributed_resume_preprocessing_tolerates_missing_optional_group_keys(self):
        """Distributed resume sorting must tolerate groups without eps/optimizer."""
        from megatron.training.training import preprocess_common_state_dict

        common_state_dict = {
            'args': SimpleNamespace(
                use_distributed_optimizer=True,
                rank=3,
                local_rank=1,
            ),
            'optimizer': {
                'optimizer': {
                    'param_groups': [
                        {
                            'wd_mult': 1.0,
                            'lr_mult': 1.0,
                            'is_expert_parallel': False,
                            'is_decoupled_lr': False,
                            'max_lr': 1.0e-3,
                            'min_lr': 1.0e-5,
                            'eps': 1.0e-8,
                            'optimizer': 'adam',
                            'params': [1],
                        },
                        {
                            'wd_mult': 1.0,
                            'lr_mult': 1.0,
                            'is_expert_parallel': False,
                            'is_decoupled_lr': False,
                            'max_lr': 1.0e-3,
                            'min_lr': 1.0e-5,
                            'params': [0],
                        },
                    ]
                }
            },
        }

        preprocessed = preprocess_common_state_dict(common_state_dict)

        param_groups = preprocessed['optimizer']['optimizer']['param_groups']
        assert [group['params'] for group in param_groups] == [[0], [1]]
        assert 'rank' not in preprocessed['args']
        assert 'local_rank' not in preprocessed['args']

    def test_load_non_mup_checkpoint_clears_width_mult_provenance(self, monkeypatch):
        """Old no-scaling checkpoints must clear stale CLI scaling state."""
        from megatron.training import checkpointing

        checkpoint_args = SimpleNamespace(hidden_size=1024)
        state_dict = {'args': checkpoint_args, 'iteration': 3}
        monkeypatch.setattr(
            checkpointing,
            '_load_base_checkpoint',
            lambda *args, **kwargs: (state_dict, 'model_optim_rng.pt', False, None),
        )
        runtime_args = SimpleNamespace(
            load='dummy-checkpoint',
            iteration=0,
            scaling_recipe='mup',
            scaling_base_hidden_size=256,
            scaling_base_head_dim=64,
            use_mup=True,
            mup_width_mult=4.0,
            _mup_width_mult_explicit=True,
            mup_base_hidden_size=256,
            mup_embedding_mult=2.0,
            mup_output_mult=0.25,
            mup_base_head_dim=64,
            mup_attn_scale_power=-0.5,
            use_tokenizer_model_from_checkpoint_args=False,
            use_mp_args_from_checkpoint_args=False,
        )

        checkpointing.load_args_from_checkpoint(runtime_args)

        assert runtime_args.iteration == 3
        assert runtime_args.scaling_recipe == 'none'
        assert runtime_args.scaling_base_hidden_size is None
        assert runtime_args.scaling_base_head_dim is None
        assert runtime_args.use_mup is False
        assert runtime_args.mup_width_mult == 1.0
        assert runtime_args._mup_width_mult_explicit is False
        assert runtime_args.mup_base_hidden_size is None
        assert runtime_args.mup_embedding_mult == 1.0
        assert runtime_args.mup_output_mult == 1.0
        assert runtime_args.mup_base_head_dim is None
        assert runtime_args.mup_attn_scale_power == 1.0
        assert build_scaling_context(runtime_args).recipe == 'none'

    def test_checkpoint_derived_width_mult_does_not_warn_as_deprecated_cli(self):
        """Normalized checkpoint state should not look like user-provided --mup-width-mult."""
        from megatron.training.arguments import warn_deprecated_mup_aliases

        checkpoint_derived_args = SimpleNamespace(
            rank=0,
            use_mup=False,
            mup_base_hidden_size=None,
            mup_base_head_dim=None,
            mup_width_mult=4.0,
            _mup_width_mult_explicit=False,
        )

        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            warn_deprecated_mup_aliases(checkpoint_derived_args)

        assert len(caught_warnings) == 0

    def test_false_width_mult_marker_overrides_stale_non_default_value(self):
        """Marker-present False means checkpoint/internal provenance, not explicit user input."""
        checkpoint_derived_args = SimpleNamespace(
            hidden_size=1024,
            scaling_recipe='none',
            scaling_base_hidden_size=None,
            scaling_base_head_dim=None,
            use_mup=False,
            mup_width_mult=4.0,
            _mup_width_mult_explicit=False,
            mup_base_hidden_size=None,
            mup_embedding_mult=1.0,
            mup_output_mult=1.0,
            mup_base_head_dim=None,
            mup_attn_scale_power=1.0,
        )

        assert build_scaling_context(checkpoint_derived_args).recipe == 'none'

    def test_checkpoint_derived_legacy_aliases_do_not_warn_as_deprecated_cli(self):
        """Checkpoint-synced legacy fields should not masquerade as user CLI aliases."""
        from megatron.training.arguments import warn_deprecated_mup_aliases

        checkpoint_derived_args = SimpleNamespace(
            rank=0,
            use_mup=True,
            _use_mup_explicit=False,
            mup_base_hidden_size=256,
            _mup_base_hidden_size_explicit=False,
            mup_base_head_dim=64,
            _mup_base_head_dim_explicit=False,
            mup_width_mult=4.0,
            _mup_width_mult_explicit=False,
        )

        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            warn_deprecated_mup_aliases(checkpoint_derived_args)

        assert len(caught_warnings) == 0

    def test_user_provided_legacy_aliases_warn_as_deprecated_cli(self):
        """Real user-provided legacy aliases should still produce a deprecation warning."""
        from megatron.training.arguments import warn_deprecated_mup_aliases

        user_args = SimpleNamespace(
            rank=0,
            use_mup=True,
            _use_mup_explicit=True,
            mup_base_hidden_size=256,
            _mup_base_hidden_size_explicit=True,
            mup_base_head_dim=64,
            _mup_base_head_dim_explicit=True,
            mup_width_mult=1.0,
            _mup_width_mult_explicit=True,
        )

        with pytest.warns(UserWarning) as caught_warnings:
            warn_deprecated_mup_aliases(user_args)

        warning_text = str(caught_warnings[0].message)
        assert '--use-mup' in warning_text
        assert '--mup-base-hidden-size' in warning_text
        assert '--mup-base-head-dim' in warning_text
        assert '--mup-width-mult' in warning_text


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


class TestMuPLRScaling:
    """Tests for MuP learning rate and Adam epsilon scaling."""

    def test_mup_overrides_route_through_training_scaling_policy(self):
        """The new policy seam preserves the legacy MuP override surface."""
        optimizer_config = OptimizerConfig(lr=1e-3, min_lr=1e-5)
        width_mult = 4.0

        legacy_overrides = get_mup_config_overrides(optimizer_config, width_mult)
        policy_overrides = get_scaling_config_overrides(
            optimizer_config,
            build_legacy_mup_training_policy(mup_width_mult=width_mult, optimizer_type='adam'),
        )

        assert legacy_overrides.keys() == policy_overrides.keys()
        assert list(legacy_overrides.values()) == list(policy_overrides.values())

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


class TestMuPConfigIntegration:
    """Integration tests for MuP config with init methods."""

    def test_model_scaling_policy_matches_legacy_config_fields(self):
        """Model policy exposes the same effective MuP values as TransformerConfig."""
        config = TransformerConfig(
            hidden_size=1024,
            num_layers=8,
            num_attention_heads=16,
            use_mup=True,
            mup_base_hidden_size=256,
            mup_embedding_mult=3.0,
        )
        policy = build_model_scaling_policy(config)

        assert policy.enabled is True
        assert policy.context.width_mult == pytest.approx(config.mup_width_mult)
        assert policy.context.output_mult == pytest.approx(config.mup_output_mult)
        assert policy.context.embedding_mult == pytest.approx(config.mup_embedding_mult)
        assert config.softmax_scale == pytest.approx(
            policy.resolve_attention_softmax_scale(
                softmax_scale=None, kv_channels=config.kv_channels
            )
        )

    def test_model_scaling_policy_tracks_post_init_multiplier_mutations(self):
        """Policy resolution should preserve legacy live reads of mutable config fields."""
        config = TransformerConfig(
            hidden_size=1024,
            num_layers=8,
            num_attention_heads=16,
            use_mup=True,
            mup_base_hidden_size=256,
        )
        logits = torch.ones(2, 4)
        embeddings = torch.ones(2, 4)

        config.mup_output_mult = 0.25
        config.mup_embedding_mult = 3.0
        policy = build_model_scaling_policy(config)

        assert torch.equal(policy.scale_output_logits(logits), logits * 0.25)
        assert torch.equal(policy.scale_embedding_activations(embeddings), embeddings * 3.0)

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

        assert build_model_scaling_policy(base_depth_config).residual_branch_multiplier == pytest.approx(1.0)
        assert build_model_scaling_policy(double_depth_config).residual_branch_multiplier == pytest.approx(0.5)

    def test_depth_mup_default_block_output_init_rebases_to_base_depth(self):
        config = TransformerConfig(
            hidden_size=1024,
            num_layers=12,
            num_attention_heads=16,
            scaling_recipe='depth_mup',
            scaling_base_hidden_size=256,
            scaling_base_num_layers=6,
        )
        policy = build_model_scaling_policy(config)
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
        assert abs(weights.std().item() - expected_std) < expected_std * 0.05

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

    def test_partial_mlp_builder_receives_block_output_init_scaling_opt_in(self):
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
            scaling_recipe='depth_mup',
            scaling_base_hidden_size=256,
            scaling_base_num_layers=6,
        )
        submodules = MLPSubmodules(linear_fc1=fc1_builder, linear_fc2=fc2_builder)
        mlp_builder = functools.partial(MLP.as_mlp_submodule, submodules=submodules)

        assert _get_mlp_builder_module(mlp_builder) is MLP

        additional_mlp_kwargs = {"apply_block_output_init_scaling": True}
        mlp = mlp_builder(
            config=config,
            pg_collection=SimpleNamespace(tp=None),
            is_mtp_layer=False,
            **additional_mlp_kwargs,
        )

        assert mlp.linear_fc2.init_method is not config.output_layer_init_method

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
        layer.model_scaling_policy = build_model_scaling_policy(config)

        output = torch.ones(4, 8)
        bias = torch.ones(8)
        scaled_output, scaled_bias = layer._scale_dense_residual_branch_output(
            (output, bias), branch_name='self attention', using_fused_tp_inference_kernel=False
        )
        expected_mult = (
            config.num_layers / config.scaling_base_num_layers
        ) ** config.scaling_residual_branch_depth_power
        assert torch.equal(scaled_output, output * expected_mult)
        assert torch.equal(scaled_bias, bias * expected_mult)

        with pytest.raises(NotImplementedError, match='Residual-branch scaling'):
            layer._scale_dense_residual_branch_output(
                (output, bias), branch_name='self attention', using_fused_tp_inference_kernel=True
            )

    def test_allow_scaling_policy_eval_allows_unfused_validation_scaling(self):
        config = TransformerConfig(
            hidden_size=1024,
            num_layers=24,
            num_attention_heads=16,
            scaling_recipe='depth_mup',
            scaling_base_hidden_size=256,
            scaling_base_num_layers=12,
        )
        layer = object.__new__(TransformerLayer)
        layer.model_scaling_policy = build_model_scaling_policy(config)
        layer.training = False

        output = torch.ones(2, 2)
        bias = torch.full((2, 2), 3.0)
        with pytest.raises(NotImplementedError, match='during inference'):
            layer._scale_dense_residual_branch_output(
                (output, bias), branch_name='self attention', using_fused_tp_inference_kernel=False
            )

        with allow_scaling_policy_eval(True):
            scaled_output, scaled_bias = layer._scale_dense_residual_branch_output(
                (output, bias), branch_name='self attention', using_fused_tp_inference_kernel=False
            )

        assert torch.equal(scaled_output, output * 0.5)
        assert torch.equal(scaled_bias, bias * 0.5)

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


class TestMuPOptimizerTypeHandling:
    """Tests for MuP optimizer-specific override behavior."""

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

        # Output params remain in the MuP override path (handled by chained Adam optimizer).
        assert output_override['max_lr'] == pytest.approx(1e-3 / width_mult)
        assert output_override['min_lr'] == pytest.approx(1e-5 / width_mult)
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
