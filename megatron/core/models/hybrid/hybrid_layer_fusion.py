# Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Construction of fused hybrid-pattern layers.

A hybrid layer pattern may contain bracketed groups like `[*-]` or `[ME]`
that instruct `HybridStack` to fuse two adjacent layers into a single
`TransformerLayer`. This module owns the logic that takes such a
multi-symbol entry and builds the corresponding `TransformerLayer` at
runtime – the sequence mixer (first symbol) becomes `self_attention` and
the channel mixer (second symbol) becomes `mlp`.

The public entry point is `build_fused_layer`, called by
`megatron.core.models.hybrid.hybrid_block.HybridStack` when it
encounters a multi-character entry in its `layer_type_list`.
"""

from typing import TYPE_CHECKING

from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.dist_checkpointing.utils import apply_prefix_mapping
from megatron.core.extensions.transformer_engine import TENorm
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.models.hybrid.hybrid_layer_allocation import Symbols as LayerSymbols
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.mamba_mixer import MambaMixer
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules

if TYPE_CHECKING:
    # Avoid a circular import at runtime: `HybridStackSubmodules` lives in
    # `hybrid_block` which imports from this module.
    from megatron.core.models.hybrid.hybrid_block import HybridStackSubmodules


# Sequence mixers legal as the first element of a fusion group; maps the
# pattern symbol to the attribute on `HybridStackSubmodules` that supplies
# the primitive spec used in the `self_attention` slot of the fused
# `TransformerLayer`.
_FUSION_SEQUENCE_MIXERS: dict[str, str] = {
    LayerSymbols.MAMBA: "mamba_mixer",
    LayerSymbols.GDN: "gdn_mixer",
    LayerSymbols.ATTENTION: "attention_mixer",
    LayerSymbols.DS_ATTENTION: "dsa_mixer",
}

# Channel mixers legal as the second element of a fusion group; maps the
# pattern symbol to the attribute on `HybridStackSubmodules` that supplies
# the primitive spec used in the `mlp` slot of the fused `TransformerLayer`.
_FUSION_CHANNEL_MIXERS: dict[str, str] = {
    LayerSymbols.MLP: "mlp_mixer",
    LayerSymbols.MOE: "moe_mixer",
}


class MambaMixerForTransformerLayer(MambaMixer):
    """`MambaMixer` adapted for use as `TransformerLayer.self_attention`.

    `MambaMixer` isn't quite drop-in for TransformerLayer's self-attention
    slot: its constructor requires a `d_model` kwarg that TransformerLayer
    does not forward, and its `forward` only accepts `hidden_states,
    inference_context, packed_seq_params` while `TransformerLayer.forward`
    hands its `self_attention` a richer set (`attention_mask`,
    `rotary_pos_*`, `attention_bias`, `sequence_len_offset`).

    This subclass bridges both gaps without touching the underlying Mamba
    mechanism:

    - `__init__` defaults `d_model` to `config.hidden_size` when the caller
      didn't supply it; an explicit `d_model` still wins.
    - `forward` accepts every kwarg that `TransformerLayer.forward`
      currently emits (plus a `**_unused_future_kwargs` sink for forward
      compatibility), but only forwards the three that `MambaMixer` uses.

    Stand-alone `MambaLayer` paths continue to use the plain `MambaMixer`
    class – this wrapper only sits in the fusion code path.
    """

    def __init__(self, config, submodules, **kwargs):
        # TransformerLayer does not forward `d_model` when constructing
        # `self_attention`; fall back to the model hidden size.
        kwargs.setdefault("d_model", config.hidden_size)
        super().__init__(config, submodules, **kwargs)

    def forward(
        self,
        hidden_states,
        # kwargs MambaMixer actually uses:
        inference_context=None,
        packed_seq_params=None,
        *,
        inference_params=None,
        # kwargs TransformerLayer.forward hands to self_attention that
        # MambaMixer does not need – absorbed and ignored. Listed
        # explicitly so the contract with TransformerLayer is visible here.
        attention_mask=None,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        rotary_pos_cos_sin=None,
        attention_bias=None,
        sequence_len_offset=None,
        **_unused_future_kwargs,
    ):
        """Dispatch to `MambaMixer.forward`, discarding unused arguments."""
        return super().forward(
            hidden_states,
            inference_context=inference_context,
            packed_seq_params=packed_seq_params,
            inference_params=inference_params,
        )


def build_fused_layer(
    fused_symbols: str,
    submodules: "HybridStackSubmodules",
    config: TransformerConfig,
    layer_number: int,
    pg_collection: ProcessGroupCollection,
    pp_layer_offset: int,
    is_mtp_layer: bool,
    add_layer_offset: bool,
):
    """Build a single fused TransformerLayer from a multi-symbol pattern entry.

    A pattern like `[*-]` is passed in as `fused_symbols="*-"` and becomes
    a standard `TransformerLayer` whose `self_attention` slot is the
    sequence-mixer primitive for the first symbol and whose `mlp` slot is
    the channel-mixer primitive for the second.

    Not every primitive has the same `__init__` signature. `TransformerLayer`
    forwards only a small, fixed set of kwargs to its `self_attention` slot
    (`config, layer_number, pg_collection, pp_layer_offset, cp_comm_type`),
    so we pick appropriate kwargs to pass to the outer `TransformerLayer`
    based on what the inner sequence mixer will actually accept, mirroring
    the existing per-primitive dispatch in `HybridStack`.

    `MambaMixer`'s signature mismatches (needs `d_model`, forward rejects
    `TransformerLayer`'s extra kwargs) are resolved by
    `MambaMixerForTransformerLayer`, a thin `MambaMixer` subclass wired up
    as the `mamba_mixer` entry in `hybrid_stack_spec.submodules`.

    Args:
        fused_symbols: The fused group contents (e.g., "*-", "ME").
        submodules: The `HybridStackSubmodules` carrying the primitive
            mixer specs to draw from.
        config: The shared `TransformerConfig`.
        layer_number: 1-indexed global layer number for this block
            (`pp_layer_offset` already added by the caller).
        pg_collection: Process-group collection.
        pp_layer_offset: Offset to add to layer numbers for pipeline stages.
        is_mtp_layer: Whether this block sits in an MTP stage.
        add_layer_offset: Whether the enclosing `TransformerLayer` should
            add its own pipeline offset to `layer_number`. Callers inside
            `HybridStack` have already included the offset, so they
            pass `False` here – matching the existing stand-alone dispatches.

    Raises:
        ValueError: If the fused group is not exactly two symbols, or if the
            first symbol is not a sequence mixer, or if the second symbol is
            not a channel mixer.
    """
    if len(fused_symbols) != 2:
        raise ValueError(
            f"Hybrid-layer fusion currently supports exactly two fused layers, "
            f"but got {len(fused_symbols)} in group '[{fused_symbols}]'. The "
            f"first must be a sequence mixer "
            f"({sorted(_FUSION_SEQUENCE_MIXERS)}) and the second must be a "
            f"channel mixer ({sorted(_FUSION_CHANNEL_MIXERS)})."
        )

    seq_sym, chan_sym = fused_symbols[0], fused_symbols[1]
    if seq_sym not in _FUSION_SEQUENCE_MIXERS:
        raise ValueError(
            f"Hybrid-layer fusion requires the first fused layer to be a "
            f"sequence mixer (one of {sorted(_FUSION_SEQUENCE_MIXERS)}), but "
            f"got '{seq_sym}' in group '[{fused_symbols}]'."
        )
    if chan_sym not in _FUSION_CHANNEL_MIXERS:
        raise ValueError(
            f"Hybrid-layer fusion requires the second fused layer to be a "
            f"channel mixer (one of {sorted(_FUSION_CHANNEL_MIXERS)}), but got "
            f"'{chan_sym}' in group '[{fused_symbols}]'."
        )

    self_attention = getattr(submodules, _FUSION_SEQUENCE_MIXERS[seq_sym])
    mlp = getattr(submodules, _FUSION_CHANNEL_MIXERS[chan_sym])

    # Norms that are not already fused into a primitive's linear layer need
    # to be supplied externally by the enclosing TransformerLayer. Currently
    # that is only DSA (input layernorm) and MoE (pre-MLP layernorm).
    input_layernorm = TENorm if seq_sym == LayerSymbols.DS_ATTENTION else IdentityOp
    pre_mlp_layernorm = TENorm if chan_sym == LayerSymbols.MOE else IdentityOp

    fused_spec = ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=input_layernorm,
            self_attention=self_attention,
            self_attn_bda=get_bias_dropout_add,
            pre_mlp_layernorm=pre_mlp_layernorm,
            mlp=mlp,
            mlp_bda=get_bias_dropout_add,
        ),
    )

    # Build kwargs that match the inner sequence mixer's signature
    #
    # TransformerLayer forwards these to `self_attention` (minus the ones
    # TransformerLayer consumes itself: `is_mtp_layer`, `add_layer_offset`).
    # The stand-alone dispatches in HybridStack pick which optional kwargs to
    # pass based on the primitive – we do the same here.
    build_kwargs: dict = dict(
        config=config,
        layer_number=layer_number,
        pg_collection=pg_collection,
        is_mtp_layer=is_mtp_layer,
        add_layer_offset=add_layer_offset,
    )
    # GatedDeltaNet.__init__ does not accept `pp_layer_offset`; SelfAttention,
    # MLASelfAttention, and MambaMixer do. Mirror the existing stand-alone
    # dispatches by conditionally including it.
    if seq_sym != LayerSymbols.GDN:
        build_kwargs["pp_layer_offset"] = pp_layer_offset

    return build_module(fused_spec, **build_kwargs)


# Canonical slot name each layer symbol uses in its stand-alone
# sharded-state-dict output – i.e., the attribute path under which the
# primitive's weights live when the block contains just that symbol. For a
# fused `[XY]` block, the same primitives sit under `self_attention` (for the
# sequence mixer X) and `mlp` (for the channel mixer Y); canonicalization
# uses this table to rewrite fused keys back into the stand-alone layout so
# fused and unfused patterns produce the same checkpoint keys. Only Mamba
# needs an intra-block rename (`self_attention.` -> `mixer.`); every other
# sequence mixer is stand-alone-hosted in a TransformerLayer whose
# `self_attention` slot already matches the fused layout.
_CANONICAL_SLOT_FOR_SYMBOL: dict[str, str] = {
    LayerSymbols.MAMBA: "mixer",
    LayerSymbols.GDN: "self_attention",
    LayerSymbols.ATTENTION: "self_attention",
    LayerSymbols.DS_ATTENTION: "self_attention",
    LayerSymbols.MLP: "mlp",
    LayerSymbols.MOE: "mlp",
}


def canonicalize_hybrid_sharded_state_dict(
    sharded_state_dict: ShardedStateDict,
    layer_prefix: str,
    layer_type_list: list[str],
    physical_offset: int = 0,
    sub_layer_offset: int = 0,
) -> None:
    """Rewrite HybridStack layer keys into the canonical (unfused) layout, in place.

    `HybridStack.sharded_state_dict` emits keys indexed by global physical
    block position within the model (a fused `[XY]` group still occupies a
    single physical block). Fused blocks are realized as `TransformerLayer`s
    whose `self_attention` slot holds the sequence mixer and `mlp` slot
    holds the channel mixer, so their keys do not match what a stand-alone
    `X` followed by stand-alone `Y` would produce. This function rewrites
    each fused block's keys into two sub-layer-indexed prefixes that
    do match: `layers.{sub_layer_offset + i}.mixer.*` for mamba sub-layers,
    `layers.{sub_layer_offset + i}.mlp.*` for MLP, etc. Stand-alone blocks
    are simply re-indexed from physical to sub-layer index.

    The resulting keys are fusion-independent – a checkpoint written with
    `[*-]M` and one written with `*-M` end up with the same set of keys, so
    the dist_checkpointing layer can load either into either.

    Args:
        sharded_state_dict: The sharded state dict to rewrite in place. Only
            entries whose keys start with `layer_prefix` are touched.
        layer_prefix: The full prefix up to and including `"layers."`, e.g.
            `"decoder.layers."`. Keys outside this prefix are left alone.
        layer_type_list: The per-physical-block layer-type symbols for this
            pipeline segment (a single `"M"`, `"*"`, etc. for a stand-alone
            block; a two-char string like `"*-"` for a fused group).
        physical_offset: The global physical-block index at which this
            pipeline segment starts (i.e., the value `HybridStack` uses to
            derive each layer's `layer_number`). Defaults to 0, which is
            correct for non-pipeline-parallel runs.
        sub_layer_offset: The global sub-layer index at which this pipeline
            segment starts. Accounts for sub-layers contributed by earlier
            pipeline segments so that fused groups in those segments are
            correctly counted. When the model is not pipeline-parallel (or
            no earlier segment contains a fusion group), this is equal to
            `physical_offset` and may be left at its default.

    Notes:
        - Build up a combined prefix map across all layers and apply it in
          one `apply_prefix_mapping` pass. This keeps the rewrite narrow
          (sibling keys outside `layer_prefix` are untouched) and lets the
          function safely run on the full model state dict without tripping
          on embedding or output-layer entries.
        - Order matters inside the combined prefix map: `apply_prefix_mapping`
          picks the first matching prefix, so the specific sub-prefixes
          (e.g. `"input_layernorm."`) are inserted before the bare block
          prefix fallback.
        - Norms attached to the X sub-layer (e.g. `input_layernorm` for DSA)
          stay with X's sub-layer index; norms attached to Y (e.g.
          `pre_mlp_layernorm` for MoE) attach to Y's sub-layer index.
    """
    prefix_map: dict[str, str] = {}
    sub_layer_cursor = sub_layer_offset

    for local_layer_idx, layer_type in enumerate(layer_type_list):
        physical_prefix = f'{layer_prefix}{physical_offset + local_layer_idx}.'

        if len(layer_type) == 1:
            # Stand-alone block: one physical block == one sub-layer, and the
            # block's attribute layout already matches the canonical
            # stand-alone layout. Only the outer block index needs to move
            # from the local module-list index to the global sub-layer index.
            canonical_prefix = f'{layer_prefix}{sub_layer_cursor}.'
            prefix_map[physical_prefix] = canonical_prefix
            sub_layer_cursor += 1
        else:
            # Fused block `[XY]`: split the single physical block's keys into
            # two sub-layer prefixes so the checkpoint looks exactly as it
            # would for stand-alone `X` followed by stand-alone `Y`.
            x_sym, y_sym = layer_type[0], layer_type[1]
            canonical_x_prefix = f'{layer_prefix}{sub_layer_cursor}.'
            canonical_y_prefix = f'{layer_prefix}{sub_layer_cursor + 1}.'
            slot_for_x = _CANONICAL_SLOT_FOR_SYMBOL[x_sym]
            slot_for_y = _CANONICAL_SLOT_FOR_SYMBOL[y_sym]

            # Specific sub-prefixes before the bare block prefix fallback.
            prefix_map[f'{physical_prefix}input_layernorm.'] = (
                f'{canonical_x_prefix}input_layernorm.'
            )
            prefix_map[f'{physical_prefix}self_attention.'] = f'{canonical_x_prefix}{slot_for_x}.'
            prefix_map[f'{physical_prefix}self_attn_bda.'] = f'{canonical_x_prefix}self_attn_bda.'
            prefix_map[f'{physical_prefix}pre_mlp_layernorm.'] = (
                f'{canonical_y_prefix}pre_mlp_layernorm.'
            )
            prefix_map[f'{physical_prefix}mlp.'] = f'{canonical_y_prefix}{slot_for_y}.'
            prefix_map[f'{physical_prefix}mlp_bda.'] = f'{canonical_y_prefix}mlp_bda.'
            # Fallback for any stray top-level fused-block keys (e.g.,
            # `_extra_state` attached to the TransformerLayer itself);
            # attach them to X's sub-layer index by convention.
            prefix_map[physical_prefix] = canonical_x_prefix
            sub_layer_cursor += 2

    if prefix_map:
        apply_prefix_mapping(sharded_state_dict, prefix_map)
