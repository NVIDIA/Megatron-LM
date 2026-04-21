# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for megatron.elastification.arguments."""

from argparse import Namespace

import pytest

from megatron.elastification.arguments import (
    convert_per_lists_to_int_lists,
    validate_flextron_per_int_lists,
)


def _make_config(**overrides):
    defaults = dict(
        hidden_size=1920,
        ffn_hidden_size=960,
        num_attention_heads=32,
        mamba_num_heads=64,
        num_moe_experts=128,
        emb_per_list=None,
        mlp_per_list=None,
        head_per_list=None,
        mamba_per_list=None,
        moe_expert_per_list=None,
        emb_int_list=None,
        mlp_int_list=None,
        head_int_list=None,
        mamba_int_list=None,
        moe_expert_int_list=None,
    )
    defaults.update(overrides)
    return Namespace(**defaults)


class TestConvertPerListsToIntLists:
    def test_ratio_one_maps_to_full_dim(self):
        cfg = _make_config(emb_per_list=[1.0, 0.5])
        convert_per_lists_to_int_lists(cfg)
        assert cfg.emb_int_list == [1920, 960]
        # After conversion the per-list is cleared.
        assert cfg.emb_per_list is None

    def test_floor_rounding(self):
        # 0.71429 * 1920 = 1371.4368 -> floor -> 1371
        cfg = _make_config(emb_per_list=[0.71429, 0.51725])
        convert_per_lists_to_int_lists(cfg)
        assert cfg.emb_int_list == [1371, 993]

    def test_all_axes_converted_with_correct_ref_dim(self):
        cfg = _make_config(
            emb_per_list=[1.0],
            mlp_per_list=[0.5],
            head_per_list=[1.0, 0.25],
            mamba_per_list=[0.75],
            moe_expert_per_list=[0.5],
        )
        convert_per_lists_to_int_lists(cfg)
        assert cfg.emb_int_list == [1920]
        assert cfg.mlp_int_list == [480]         # 0.5 * 960
        assert cfg.head_int_list == [32, 8]      # 1.0 / 0.25 of 32
        assert cfg.mamba_int_list == [48]        # 0.75 * 64
        assert cfg.moe_expert_int_list == [64]   # 0.5 * 128

    def test_axis_with_no_per_list_is_untouched(self):
        cfg = _make_config(emb_per_list=[1.0])  # only emb
        convert_per_lists_to_int_lists(cfg)
        # mlp / head / mamba / moe_expert should not gain int_list from nothing.
        assert cfg.mlp_int_list is None
        assert cfg.head_int_list is None
        assert cfg.mamba_int_list is None
        assert cfg.moe_expert_int_list is None


class TestValidateFlextronPerIntLists:
    def _make_args(self, **overrides):
        defaults = dict(
            emb_per_list=None, emb_int_list=None,
            mlp_per_list=None, mlp_int_list=None,
            head_per_list=None, head_int_list=None,
            mamba_per_list=None, mamba_int_list=None,
            moe_expert_per_list=None, moe_expert_int_list=None,
        )
        defaults.update(overrides)
        return Namespace(**defaults)

    def test_unset_axis_defaults_to_full(self):
        args = self._make_args()
        validate_flextron_per_int_lists(args)
        # Each axis defaults to [1.0] on the per-list side.
        assert args.emb_per_list == [1.0]
        assert args.mlp_per_list == [1.0]
        assert args.head_per_list == [1.0]
        assert args.mamba_per_list == [1.0]
        assert args.moe_expert_per_list == [1.0]

    def test_per_list_preserved_when_set(self):
        args = self._make_args(emb_per_list=[1.0, 0.5])
        validate_flextron_per_int_lists(args)
        assert args.emb_per_list == [1.0, 0.5]

    def test_int_list_preserved_when_set(self):
        args = self._make_args(emb_int_list=[1920, 960])
        validate_flextron_per_int_lists(args)
        # int_list was explicitly set: per_list stays None (not defaulted to [1.0]).
        assert args.emb_per_list is None
        assert args.emb_int_list == [1920, 960]

    def test_both_set_raises(self):
        args = self._make_args(emb_per_list=[1.0], emb_int_list=[1920])
        with pytest.raises(AssertionError, match="not both"):
            validate_flextron_per_int_lists(args)

    def test_per_list_out_of_range_raises(self):
        args = self._make_args(emb_per_list=[1.5])
        with pytest.raises(AssertionError, match=r"\[0, 1\]"):
            validate_flextron_per_int_lists(args)

    def test_per_list_negative_raises(self):
        args = self._make_args(emb_per_list=[-0.1])
        with pytest.raises(AssertionError, match=r"\[0, 1\]"):
            validate_flextron_per_int_lists(args)
