# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Tests for DSA indexer scoring plan resolution.

The cases below mirror configurations whose actual branch was measured on GB200 with a
branch-tracing build, so the resolver is pinned to observed behaviour rather than to a
reading of the dispatcher.
"""

import pytest

from megatron.core.transformer.experimental_attention_variant.dsa_scoring_plan import (
    IndexerScoringPlan,
    resolve_indexer_scoring_plan,
)

BASE = dict(
    bounds_available=True,
    varlen_is_plain_causal=False,
    packed_thd=False,
    cp_size=1,
    mask_is_causal=True,
    explicit_key_positions=False,
    single_sequence_pack=False,
    packed_metadata_available=False,
)


def _resolve(**overrides):
    return resolve_indexer_scoring_plan(**{**BASE, **overrides})


class TestIndexerScoringPlan:
    """Plan resolution for each layout the dispatcher can encounter."""

    def test_plain_causal_reaches_the_fused_scorer(self):
        """Non-packed, no CP: bounds are reconstructible so the single kernel applies."""
        decision = _resolve(varlen_is_plain_causal=True)
        assert decision.plan is IndexerScoringPlan.PLAIN_CAUSAL
        assert decision.is_fused

    def test_packed_cp_single_sequence(self):
        """One sequence per pack under CP takes the zigzag-segment kernel."""
        decision = _resolve(
            packed_thd=True, cp_size=16, single_sequence_pack=True, packed_metadata_available=True
        )
        assert decision.plan is IndexerScoringPlan.PACKED_CP_SINGLE
        assert decision.is_fused

    def test_packed_cp_multi_sequence(self):
        """Several sequences per pack take the cu_seqlens kernel."""
        decision = _resolve(
            packed_thd=True, cp_size=16, single_sequence_pack=False, packed_metadata_available=True
        )
        assert decision.plan is IndexerScoringPlan.PACKED_CP_MULTI
        assert decision.is_fused

    def test_packed_cp_multi_without_metadata_is_unfused(self):
        """Without cu_seqlens the packed kernels cannot express the boundaries."""
        decision = _resolve(
            packed_thd=True, cp_size=16, single_sequence_pack=False, packed_metadata_available=False
        )
        assert decision.plan is IndexerScoringPlan.UNFUSED_BOUNDS
        assert "cu_seqlens" in decision.reason

    def test_packed_without_cp_is_unfused_and_says_why(self):
        """Packed sequences at cp_size=1 miss every fused kernel. Measured on GB200."""
        decision = _resolve(packed_thd=True, single_sequence_pack=True, packed_metadata_available=True)
        assert decision.plan is IndexerScoringPlan.UNFUSED_BOUNDS
        assert "cp_size" in decision.reason

    def test_context_parallel_without_packing_is_unfused_and_says_why(self):
        """CP without packing misses every fused kernel. Measured on GB200."""
        decision = _resolve(cp_size=4)
        assert decision.plan is IndexerScoringPlan.UNFUSED_BOUNDS
        assert "packing" in decision.reason

    def test_explicit_key_positions_declines(self):
        """Custom key positions are not expressible as row-wise bounds at all."""
        decision = _resolve(explicit_key_positions=True, varlen_is_plain_causal=True)
        assert decision.plan is IndexerScoringPlan.DECLINE
        assert not decision.is_fused

    def test_missing_bounds_declines(self):
        """A mask the caller could not turn into bounds must not reach fused scoring."""
        decision = _resolve(bounds_available=False)
        assert decision.plan is IndexerScoringPlan.DECLINE

    def test_non_causal_mask_is_unfused(self):
        """No fused scorer covers a non-causal mask."""
        decision = _resolve(mask_is_causal=False)
        assert decision.plan is IndexerScoringPlan.UNFUSED_BOUNDS

    @pytest.mark.parametrize("cp_size", [1, 2, 8, 16])
    @pytest.mark.parametrize("packed_thd", [False, True])
    @pytest.mark.parametrize("single_sequence_pack", [False, True])
    def test_resolution_is_total(self, cp_size, packed_thd, single_sequence_pack):
        """Every configuration resolves to a plan carrying a non-empty reason.

        Totality is the property that keeps a new layout from silently falling through
        to the slow path the way it does today.
        """
        decision = _resolve(
            cp_size=cp_size,
            packed_thd=packed_thd,
            single_sequence_pack=single_sequence_pack,
            packed_metadata_available=packed_thd,
        )
        assert isinstance(decision.plan, IndexerScoringPlan)
        assert decision.reason

    def test_decline_is_not_reported_as_a_slow_fallback(self):
        """A decline means fused scoring was never attempted, not that it was slow."""
        from megatron.core.transformer.experimental_attention_variant.dsa_scoring_plan import (
            report_unfused_scoring_once,
        )

        decision = _resolve(explicit_key_positions=True)
        assert report_unfused_scoring_once(decision, "test-decline") is None

    def test_unfused_plan_is_reported_once_per_reason(self):
        """The warning exists so a throughput cliff is visible; it must not spam."""
        from megatron.core.transformer.experimental_attention_variant.dsa_scoring_plan import (
            report_unfused_scoring_once,
        )

        decision = _resolve(cp_size=4)
        first = report_unfused_scoring_once(decision, "test-once")
        second = report_unfused_scoring_once(decision, "test-once")
        assert first is not None and decision.reason in first
        assert second is None
