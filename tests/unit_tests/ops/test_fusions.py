# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Selecting a fusion that spans several families.

Exercises the in-tree reference fusion, which declares everything a real megakernel declares
-- so the mechanism is covered before any kernel exists, and a vendor can see what their own
file has to do. What the layer spec builders then do with it is covered alongside them.
"""

import pytest

import megatron.core.ops.fusions as fusions
from megatron.core.ops import PRESETS, BackendOptions, build_spec_provider, get_backend
from megatron.core.ops.attention import CORE_ATTENTION
from megatron.core.ops.fusions.attn_moe import ReferenceFusedAttentionMoELayer
from megatron.core.ops.mlp import MLP_MODULE
from megatron.core.ops.moe import GROUPED_MLP_MODULES
from megatron.core.ops.norm import LAYER_NORM

#: What selects the reference fusion, wherever a test needs it.
FUSION = {"fused_moe_layer": "attn_moe_reference"}


class TestSelectingAFusion:
    @pytest.mark.parametrize("preset", PRESETS)
    def test_no_fusion_unless_asked_for(self, preset):
        """No --transformer-impl implies a fusion; the answer is None and the layer is ordinary."""
        assert get_backend(preset).fused_moe_layer() is None

    def test_a_fusion_is_selected_like_any_other_backend(self):
        provider = build_spec_provider(
            BackendOptions(transformer_impl="local", operation_backends=FUSION)
        )
        assert provider.fused_moe_layer() is ReferenceFusedAttentionMoELayer

    def test_a_fusion_leaves_the_handover_point_it_does_not_cover(self):
        """An attention-plus-MoE kernel has nothing to fuse on a dense layer, so that slot
        stays with FusionNone and the ordinary layer is kept."""
        provider = build_spec_provider(
            BackendOptions(transformer_impl="local", operation_backends=FUSION)
        )
        assert provider.fused_dense_layer() is None

    def test_a_backend_for_a_spanned_operation_is_refused(self):
        """The rule a fusion has to follow: say what you swallowed, and it gets checked.

        The fusion performs attention itself, so a backend selected for ``core_attention``
        would never be built. Refusing beats a run that looks configured and is not.
        """
        options = BackendOptions(
            transformer_impl="local", operation_backends={**FUSION, "core_attention": "local"}
        )
        with pytest.raises(ValueError, match="performs 'core_attention' itself"):
            build_spec_provider(options)

    def test_an_unspanned_operation_is_still_selectable(self):
        """SPANS refuses only what the fusion swallowed; everything else stays configurable."""
        provider = build_spec_provider(
            BackendOptions(
                transformer_impl="local", operation_backends={**FUSION, "layer_norm": "torch"}
            )
        )
        assert provider.fused_moe_layer() is ReferenceFusedAttentionMoELayer


class TestASlotIsAHandoverPointNotAKernel:
    """What makes the family scale: SPANS is per kernel, the slot is per handover point.

    A kernel that swallows more than the slot is named for does not need a new slot -- it
    fills the same one and says so in SPANS. So two vendors whose kernels cover different
    footprints still share ``fused_moe_layer``, and ``FusionSlots`` does not grow.
    """

    class WideAttnMoe:
        """The same handover point, a larger footprint: it eats the input norm too."""

        REQUIRES = None
        DETERMINISM = "unknown"
        SPANS = (CORE_ATTENTION, GROUPED_MLP_MODULES, LAYER_NORM)

        def fused_moe_layer(self):
            return ReferenceFusedAttentionMoELayer

    @pytest.fixture()
    def wide(self, monkeypatch):
        monkeypatch.setitem(fusions.BACKENDS, "wide_attn_moe", self.WideAttnMoe)
        return {"fused_moe_layer": "wide_attn_moe"}

    def test_a_wider_kernel_needs_no_new_slot(self, wide):
        provider = build_spec_provider(
            BackendOptions(transformer_impl="local", operation_backends=wide)
        )
        assert provider.fused_moe_layer() is ReferenceFusedAttentionMoELayer

    def test_the_wider_footprint_is_what_gets_checked(self, wide):
        """LAYER_NORM is refused for this kernel and not for the narrower one, same slot."""
        with pytest.raises(ValueError, match="performs 'layer_norm' itself"):
            build_spec_provider(
                BackendOptions(
                    transformer_impl="local", operation_backends={**wide, "layer_norm": "torch"}
                )
            )
        # The narrower kernel, at the very same slot, still allows it.
        build_spec_provider(
            BackendOptions(
                transformer_impl="local", operation_backends={**FUSION, "layer_norm": "torch"}
            )
        )

    def test_one_slot_per_handover_point(self):
        """Pins the slot list, so FusionSlots cannot quietly accumulate a method per kernel.

        Every fusion operation is a place the layer hands over control, and each one needs a
        call site in the spec builders. Adding to this list is meant to be a reviewed decision;
        a wider footprint, different internals, or another vendor are none of them reasons to.
        See the rule in megatron/core/ops/fusions/__init__.py.
        """
        assert [op.method for op in fusions.OPERATIONS] == ["fused_dense_layer", "fused_moe_layer"]


class TestASecondFusionAtAnotherHandoverPoint:
    """What adding an attention-plus-dense-MLP kernel costs: one file, and no new slot.

    Stands in for a future ``fusions/attn_mlp.py``. It fills the dense handover point, so it
    coexists with the attention-plus-MoE kernel rather than competing with it -- the two are
    separate operations, each independently selectable.
    """

    class AttnMlp:
        """Attention fused with the dense MLP."""

        REQUIRES = None
        DETERMINISM = "unknown"
        SPANS = (CORE_ATTENTION, MLP_MODULE)

        def fused_dense_layer(self):
            return ReferenceFusedAttentionMoELayer

    @pytest.fixture()
    def both(self, monkeypatch):
        monkeypatch.setitem(fusions.BACKENDS, "attn_mlp", self.AttnMlp)
        return {**FUSION, "fused_dense_layer": "attn_mlp"}

    def test_two_fusions_can_be_selected_at_once(self, both):
        provider = build_spec_provider(
            BackendOptions(transformer_impl="local", operation_backends=both)
        )
        assert provider.fused_dense_layer() is ReferenceFusedAttentionMoELayer
        assert provider.fused_moe_layer() is ReferenceFusedAttentionMoELayer

    def test_each_fusion_is_checked_against_its_own_span(self, both):
        """The dense kernel eats mlp_module; the MoE one does not. Both spans are enforced."""
        with pytest.raises(ValueError, match="performs 'mlp_module' itself"):
            build_spec_provider(
                BackendOptions(
                    transformer_impl="local", operation_backends={**both, "mlp_module": "megatron"}
                )
            )
