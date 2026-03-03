# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""
Unit tests for validations migrated into TransformerConfig.__post_init__.

Part of #3568: moves pure-validation assertions from validate_args() in
arguments.py into TransformerConfig so they fire at config-construction time
and can be exercised without a full training process.

Covered validation groups
-------------------------
1. hidden_size divisibility by num_attention_heads
2. num_moe_experts divisibility by expert_model_parallel_size
"""

import pytest

from megatron.core.transformer.transformer_config import TransformerConfig

# ---------------------------------------------------------------------------
# Minimal valid kwargs shared across tests.  Extend per-test as needed.
# ---------------------------------------------------------------------------
_BASE_KWARGS = dict(
    num_layers=2,
    hidden_size=64,
    num_attention_heads=8,
)


# ===========================================================================
# Group 1 — hidden_size must be divisible by num_attention_heads
# ===========================================================================
class TestHiddenSizeDivisibility:
    """Validates that hidden_size % num_attention_heads == 0 is enforced when
    kv_channels is derived (i.e., not set explicitly by the caller)."""

    def test_valid_hidden_size_constructs_cleanly(self):
        """64 / 8 = 8 — no remainder, config should be created successfully."""
        cfg = TransformerConfig(**_BASE_KWARGS)
        # kv_channels is derived as hidden_size // num_attention_heads
        assert cfg.kv_channels == 8

    def test_hidden_size_not_divisible_raises(self):
        """65 % 8 != 0 — must raise ValueError when kv_channels is not set."""
        with pytest.raises(ValueError, match="hidden_size.*must be divisible by.*num_attention_heads"):
            TransformerConfig(
                num_layers=2,
                hidden_size=65,
                num_attention_heads=8,
            )

    def test_explicit_kv_channels_bypasses_hidden_size_check(self):
        """When kv_channels is set explicitly the hidden_size check is skipped,
        since the caller is taking responsibility for the channel dimension."""
        # hidden_size=65 is not divisible by num_attention_heads=8, but
        # providing an explicit kv_channels should skip that guard.
        cfg = TransformerConfig(
            num_layers=2,
            hidden_size=65,
            num_attention_heads=8,
            kv_channels=8,  # explicit — bypasses the derivation guard
        )
        assert cfg.kv_channels == 8

    def test_large_valid_hidden_size(self):
        """Larger realistic hidden_size that is cleanly divisible."""
        cfg = TransformerConfig(
            num_layers=4,
            hidden_size=1024,
            num_attention_heads=16,
        )
        assert cfg.kv_channels == 64


# ===========================================================================
# Group 2 — num_moe_experts must be divisible by expert_model_parallel_size
# ===========================================================================
class TestMoEExpertParallelDivisibility:
    """Validates that when expert_model_parallel_size > 1, num_moe_experts is
    required and must be evenly divisible by expert_model_parallel_size.

    The 'not-None' guard was already present in TransformerConfig; the
    divisibility guard is newly migrated from validate_args."""

    # --- base valid MoE config -------------------------------------------

    def _moe_kwargs(self, num_moe_experts, expert_model_parallel_size):
        return dict(
            **_BASE_KWARGS,
            num_moe_experts=num_moe_experts,
            expert_model_parallel_size=expert_model_parallel_size,
        )

    def test_valid_divisible_experts_constructs_cleanly(self):
        """4 experts, EP=2 → 4 % 2 == 0, should succeed."""
        cfg = TransformerConfig(**self._moe_kwargs(num_moe_experts=4, expert_model_parallel_size=2))
        assert cfg.num_moe_experts == 4
        assert cfg.expert_model_parallel_size == 2

    def test_experts_not_divisible_raises(self):
        """3 experts, EP=2 → 3 % 2 != 0, must raise ValueError."""
        with pytest.raises(
            ValueError,
            match="num_moe_experts.*must be divisible by.*expert_model_parallel_size",
        ):
            TransformerConfig(**self._moe_kwargs(num_moe_experts=3, expert_model_parallel_size=2))

    def test_experts_not_divisible_ep4_raises(self):
        """7 experts, EP=4 → 7 % 4 != 0, must raise ValueError."""
        with pytest.raises(
            ValueError,
            match="num_moe_experts.*must be divisible by.*expert_model_parallel_size",
        ):
            TransformerConfig(**self._moe_kwargs(num_moe_experts=7, expert_model_parallel_size=4))

    def test_ep1_with_any_non_zero_experts_is_valid(self):
        """expert_model_parallel_size=1 (default) — divisibility never fires
        because the guard only activates when EP > 1."""
        cfg = TransformerConfig(
            **_BASE_KWARGS,
            num_moe_experts=7,
            expert_model_parallel_size=1,  # default, no EP partitioning
        )
        assert cfg.num_moe_experts == 7

    def test_none_experts_with_ep_greater_than_1_raises(self):
        """num_moe_experts=None with EP>1 must raise (pre-existing guard)."""
        with pytest.raises(ValueError, match="num_moe_experts must be non None"):
            TransformerConfig(
                **_BASE_KWARGS,
                num_moe_experts=None,
                expert_model_parallel_size=2,
            )

    def test_experts_exactly_equal_to_ep_is_valid(self):
        """num_moe_experts == expert_model_parallel_size → 1 expert per rank, valid."""
        cfg = TransformerConfig(**self._moe_kwargs(num_moe_experts=4, expert_model_parallel_size=4))
        assert cfg.num_moe_experts == cfg.expert_model_parallel_size

    def test_large_divisible_moe_config(self):
        """Realistic scale: 64 experts, EP=8 → 64 % 8 == 0."""
        cfg = TransformerConfig(**self._moe_kwargs(num_moe_experts=64, expert_model_parallel_size=8))
        assert cfg.num_moe_experts == 64
