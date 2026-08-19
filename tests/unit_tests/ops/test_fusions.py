# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""A fusion spanning several families, from selection through a forward pass.

Exercises the in-tree reference fusion, which declares everything a real megakernel declares
and then runs the ordinary layer -- so the mechanism is covered before any kernel exists, and
a vendor can see what their own file has to do.
"""

import pytest
import torch

import megatron.core.ops.fusions as fusions
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.ops import PRESETS, BackendOptions, build_spec_provider, get_backend
from megatron.core.ops.attention import CORE_ATTENTION
from megatron.core.ops.fusions.attn_moe import ReferenceFusedAttentionMoELayer
from megatron.core.ops.mlp import MLP_MODULE
from megatron.core.ops.moe import GROUPED_MLP_MODULES
from megatron.core.ops.norm import LAYER_NORM
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer
from megatron.training.initialize import _set_random_seed
from tests.unit_tests.test_utilities import Utils

#: What selects the reference fusion, wherever a test needs it.
FUSION = {"fused_moe_layer": "attn_moe_reference"}


def _config(**overrides):
    """A two-layer model whose first layer is MoE and whose second is dense."""
    settings = dict(
        num_layers=2,
        hidden_size=16,
        num_attention_heads=4,
        num_moe_experts=2,
        moe_layer_freq=[1, 0],
        moe_router_topk=2,
        moe_ffn_hidden_size=32,
        moe_token_dispatcher_type="alltoall",
        moe_aux_loss_coeff=0.01,
        add_bias_linear=False,
        use_cpu_initialization=True,
        # Off so the parity check below compares arithmetic, not two draws from the RNG.
        hidden_dropout=0.0,
        attention_dropout=0.0,
    )
    settings.update(overrides)
    return TransformerConfig(**settings)


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

    def test_both_layer_kinds_are_replaced_in_the_block_spec(self, both):
        Utils.initialize_model_parallel(1, 1)
        try:
            block = get_gpt_decoder_block_spec(
                _config(op_backend_overrides=both), use_transformer_engine=False
            )
            assert [spec.module for spec in block.layer_specs] == [
                ReferenceFusedAttentionMoELayer,
                ReferenceFusedAttentionMoELayer,
            ]
        finally:
            Utils.destroy_model_parallel()


class TestFusionInTheBlockSpec:
    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def _layer_modules(self, **overrides):
        config = _config(**overrides)
        block = get_gpt_decoder_block_spec(config, use_transformer_engine=False)
        return [spec.module for spec in block.layer_specs]

    def test_moe_layers_are_replaced_and_dense_layers_are_not(self):
        assert self._layer_modules(op_backend_overrides=FUSION) == [
            ReferenceFusedAttentionMoELayer,
            TransformerLayer,
        ]

    def test_without_the_fusion_every_layer_is_ordinary(self):
        assert self._layer_modules() == [TransformerLayer, TransformerLayer]

    def test_the_fused_layer_keeps_the_submodules_it_replaced(self):
        """A fusion is handed the ordinary submodules and ignores what it performs itself.

        They are specs, not modules, so the ones its kernel carries out are never built.
        """
        config = _config(op_backend_overrides=FUSION)
        fused = get_gpt_decoder_block_spec(config, use_transformer_engine=False).layer_specs[0]
        plain = get_gpt_decoder_block_spec(_config(), use_transformer_engine=False).layer_specs[0]
        assert fused.submodules.self_attention.module is plain.submodules.self_attention.module
        assert fused.submodules.mlp is not None


class TestFusedLayerRuns:
    """The reference runs the ordinary path, so selecting it must change nothing but the class."""

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def _forward(self, config):
        _set_random_seed(seed_=123, data_parallel_random_init=False)
        block = TransformerBlock(
            config, get_gpt_decoder_block_spec(config, use_transformer_engine=False)
        ).cuda()
        hidden_states = torch.ones((8, 2, config.hidden_size), device="cuda")
        attention_mask = torch.ones((2, 1, 8, 8), dtype=bool, device="cuda")
        output = block(hidden_states=hidden_states, attention_mask=attention_mask)
        return block, output

    def test_the_fused_layer_is_the_one_that_ran(self):
        block, _ = self._forward(_config(op_backend_overrides=FUSION))
        fused = block.layers[0]
        assert isinstance(fused, ReferenceFusedAttentionMoELayer)
        assert fused.fused_steps == 1
        assert not isinstance(block.layers[1], ReferenceFusedAttentionMoELayer)

    def test_selecting_the_reference_fusion_does_not_change_the_result(self):
        _, plain = self._forward(_config())
        _, fused = self._forward(_config(op_backend_overrides=FUSION))
        torch.testing.assert_close(fused, plain, rtol=0, atol=0)
