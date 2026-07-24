# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Load GPT (pure transformer) distributed checkpoints into HybridModel runs.

A GPTModel decoder layer packs self-attention and an MLP into a single
``TransformerLayer``, so a GPT checkpoint with ``L`` layers stores both
sub-modules under ``decoder.layers.<i>.``. HybridModel gives every sub-block
its own layer: in a pattern such as ``M*-M*-`` each GPT layer corresponds to
one attention ('*') position and one MLP ('-' dense or 'E' MoE) position,
while SSM ('M') positions have no GPT counterpart.

Rather than rewriting the checkpoint on disk, the hybrid run's own sharded
state dict is retargeted at load time:

* attention and MLP entries are rewritten to the GPT checkpoint's canonical
  homogeneous-layer format: the layer index is dropped from the storage
  ``key`` (``decoder.layers.<g>.mlp...`` -> ``decoder.layers.mlp...``) and
  the matching GPT layer index becomes a prepended sharding axis, exactly
  mirroring ``TransformerBlock.sharded_state_dict`` with
  ``non_homogeneous_layers=False`` (the format GPTModel training saves);
* ``decoder.final_norm`` is pointed at GPT's ``decoder.final_layernorm``;
* HybridModel's empty ``output_layer._extra_state`` entry stays local because
  GPT checkpoints intentionally omit that backward-compatibility key;
* entries of layers without a GPT counterpart are wrapped in
  ``LocalNonpersistentObject`` so no storage read is attempted and the
  freshly initialized module values are kept (and remain visible to the
  subsequent strict ``load_state_dict``).

The retargeted sharded state dict is then handed to the regular
``dist_checkpointing.load`` machinery, which reads the GPT checkpoint
directly and reshards across any TP/PP/EP/ETP layout change on the way.

The same retargeting also applies to the distributed optimizer's sharded
state dict. In the model-space checkpoint formats (``fully_reshardable`` /
``fully_sharded_model_space``) every optimizer-state ``ShardedTensor`` is built
by copying the corresponding model param's metadata and prefixing its ``key``
with ``optimizer.state.<state>.`` (see
``DistributedOptimizer.sharded_param_state_*``). Those entries therefore carry
the same ``decoder.layers.<i>.`` keys and sharding as the model tensors, so
:func:`retarget_sharded_state_dict_to_gpt_checkpoint` rewrites them onto the GPT
checkpoint identically -- optimizer moments and fp32 master params for
attention/MLP layers load from the GPT run, while fresh layers (e.g. Mamba)
keep their freshly initialized optimizer state via ``LocalNonpersistentObject``.
"""

import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from megatron.core.dist_checkpointing.dict_utils import dict_list_map_inplace
from megatron.core.dist_checkpointing.mapping import (
    LocalNonpersistentObject,
    ShardedBase,
    ShardedObject,
    ShardedStateDict,
    ShardedTensor,
    ShardedTensorFactory,
)
from megatron.core.models.hybrid.hybrid_layer_allocation import (
    Symbols,
    get_layer_maps_from_layer_type_list,
    parse_hybrid_pattern,
)

# Hybrid layer symbols that have a GPT-side source of weights ('*', '-', 'E')
# or that are explicitly initialized from scratch ('M'). Attention layers map
# onto GPT ``self_attention`` sub-modules; dense and MoE MLP layers map onto
# GPT ``mlp`` sub-modules (both models keep MoE tensors under ``mlp.*``).
# GDN ('G') and DS-attention ('D') use different weight layouts than GPT
# attention and are rejected rather than silently mistranslated.
_GPT_SOURCED_SYMBOLS = (Symbols.ATTENTION, Symbols.MLP, Symbols.MOE)
_FRESH_INIT_SYMBOLS = (Symbols.MAMBA,)

_DECODER_LAYER_KEY_RE = re.compile(r'decoder\.layers\.(\d+)\.')

_GPT_FINAL_NORM_KEY_MAP = {'decoder.final_norm.': 'decoder.final_layernorm.'}
_GPT_OMITTED_LOCAL_KEYS = ('output_layer._extra_state',)


@dataclass(frozen=True)
class GPTCompatLayerMaps:
    """Correspondence between hybrid layer indices and GPT layer indices.

    Attributes:
        attention_to_gpt: hybrid global layer index of the i-th attention
            position -> GPT layer index i.
        mlp_to_gpt: hybrid global layer index of the i-th MLP-bearing
            position ('-' or 'E') -> GPT layer index i.
        fresh_init: hybrid global layer indices with no GPT counterpart;
            their modules keep the run's fresh initialization.
        num_gpt_layers: number of layers the source GPT checkpoint must have.
    """

    attention_to_gpt: Mapping[int, int]
    mlp_to_gpt: Mapping[int, int]
    fresh_init: frozenset
    num_gpt_layers: int


def gpt_compatible_layer_maps(hybrid_layer_pattern: str) -> GPTCompatLayerMaps:
    """Derive hybrid->GPT layer index maps from a hybrid layer pattern.

    Args:
        hybrid_layer_pattern: the run's unified hybrid layer pattern
            (pipeline '|' separators allowed).

    Returns:
        GPTCompatLayerMaps for retargeting a sharded state dict.

    Raises:
        ValueError: if the pattern cannot be paired one-to-one with a GPT
            checkpoint layout (MTP present, non-translatable symbols,
            mixed dense/MoE positions, or unbalanced '*' vs MLP counts).
    """
    parsed = parse_hybrid_pattern(hybrid_layer_pattern)
    if parsed.mtp_num_depths > 0:
        raise ValueError(
            f"Hybrid layer pattern {hybrid_layer_pattern!r} contains MTP layers "
            f"('/{parsed.mtp_pattern}'), which have no source weights in a GPT "
            f"checkpoint. Remove the MTP part of the pattern to load a GPT checkpoint."
        )
    main_pattern = (parsed.main_pattern or '').replace(Symbols.PIPE, '')
    if not main_pattern:
        raise ValueError("Hybrid layer pattern is empty; set --hybrid-layer-pattern.")

    layer_type_list = list(main_pattern)
    translatable = set(_GPT_SOURCED_SYMBOLS) | set(_FRESH_INIT_SYMBOLS)
    unknown = sorted(set(layer_type_list) - translatable)
    if unknown:
        raise ValueError(
            f"Hybrid layer pattern {hybrid_layer_pattern!r} contains layer types "
            f"{unknown} that cannot be translated from a GPT checkpoint. "
            f"Supported: {sorted(translatable)} ('M' layers keep their fresh "
            f"initialization)."
        )

    layer_maps = get_layer_maps_from_layer_type_list(layer_type_list)
    dense_map = layer_maps[Symbols.MLP]
    moe_map = layer_maps[Symbols.MOE]
    if dense_map and moe_map:
        raise ValueError(
            f"Hybrid layer pattern {hybrid_layer_pattern!r} mixes dense ('-') and "
            f"MoE ('E') MLP positions. GPT checkpoints have one MLP kind on every "
            f"layer, so the pattern must use only one of '-' or 'E'."
        )
    mlp_map = moe_map if moe_map else dense_map
    attention_map = layer_maps[Symbols.ATTENTION]

    if len(attention_map) != len(mlp_map) or not attention_map:
        raise ValueError(
            f"Hybrid layer pattern {hybrid_layer_pattern!r} has "
            f"{len(attention_map)} attention ('*') and {len(mlp_map)} MLP ('-'/'E') "
            f"positions. Each GPT layer provides exactly one attention and one MLP "
            f"sub-module, so the pattern needs an equal, nonzero number of each."
        )

    return GPTCompatLayerMaps(
        attention_to_gpt=dict(attention_map),
        mlp_to_gpt=dict(mlp_map),
        fresh_init=frozenset(layer_maps[Symbols.MAMBA]),
        num_gpt_layers=len(attention_map),
    )


def _prepend_gpt_layer_axis(entry, gpt_layer_idx: int, num_gpt_layers: int):
    """Add the GPT layer index as the leading sharding axis of an entry.

    Mirrors what ``TransformerBlock.sharded_state_dict`` does for homogeneous
    layers by passing ``sharded_offsets=[(0, layer_idx, num_layers)]`` down to
    ``make_sharded_tensors_for_checkpoint``:

    * ShardedTensor: one more prepended axis of size ``num_gpt_layers``
      at position 0, this shard sitting at ``gpt_layer_idx``;
    * ShardedObject: ``(1,)/(0,)`` placeholder offsets (from
      ``_get_extra_state_offsets`` with no offsets) are replaced by the layer
      axis, otherwise the layer axis is prepended (e.g. before an expert axis);
    * ShardedTensorFactory: the built sub-entries get the same treatment.
    """
    if isinstance(entry, ShardedTensor):
        entry.global_shape = (num_gpt_layers, *entry.global_shape)
        entry.global_offset = (gpt_layer_idx, *entry.global_offset)
        entry.axis_fragmentations = (num_gpt_layers, *entry.axis_fragmentations)
        entry.prepend_axis_num += 1
    elif isinstance(entry, ShardedObject):
        if entry.global_shape == (1,) and entry.global_offset == (0,):
            entry.global_shape = (num_gpt_layers,)
            entry.global_offset = (gpt_layer_idx,)
        else:
            entry.global_shape = (num_gpt_layers, *entry.global_shape)
            entry.global_offset = (gpt_layer_idx, *entry.global_offset)
    elif isinstance(entry, ShardedTensorFactory):
        inner_build_fn = entry.build_fn

        def _build_with_gpt_layer_axis(key, data, replica_id, flattened_range):
            built = inner_build_fn(key, data, replica_id, flattened_range)
            dict_list_map_inplace(
                lambda sub: _prepend_gpt_layer_axis(sub, gpt_layer_idx, num_gpt_layers), built
            )
            return built

        entry.build_fn = _build_with_gpt_layer_axis
    return entry


def retarget_sharded_state_dict_to_gpt_checkpoint(
    sharded_state_dict: ShardedStateDict, layer_maps: GPTCompatLayerMaps
) -> None:
    """Point a hybrid model's sharded state dict at a GPT checkpoint, in place.

    Only the storage lookup metadata (``key`` and sharding axes) of each
    ``ShardedBase`` entry is rewritten into the GPT checkpoint's homogeneous
    layer format; the nested state dict structure (used by the subsequent
    ``load_state_dict``) keeps the hybrid model's own names. Entries of layers
    with no GPT counterpart are replaced by ``LocalNonpersistentObject`` so the
    loaded state dict returns their current (freshly initialized) values.

    The same routine handles the distributed optimizer's sharded state dict: its
    per-parameter entries embed the model key (``optimizer.state.<state>.decoder.
    layers.<i>...``) and mirror the model param's sharding, so they retarget the
    same way, and fresh-layer optimizer state is likewise kept local.

    Args:
        sharded_state_dict: one model chunk's sharded state dict (as produced by
            ``model.sharded_state_dict()``) or the matching optimizer sharded
            state dict.
        layer_maps: maps from :func:`gpt_compatible_layer_maps` derived from
            the same pattern the model was built with.
    """

    def _retarget(entry):
        if not isinstance(entry, ShardedBase):
            return entry

        if entry.key.endswith(_GPT_OMITTED_LOCAL_KEYS):
            return LocalNonpersistentObject(entry.data)

        layer_match = _DECODER_LAYER_KEY_RE.search(entry.key)
        if layer_match is not None:
            hybrid_idx = int(layer_match.group(1))
            if hybrid_idx in layer_maps.fresh_init:
                return LocalNonpersistentObject(entry.data)
            gpt_idx = layer_maps.attention_to_gpt.get(hybrid_idx)
            if gpt_idx is None:
                gpt_idx = layer_maps.mlp_to_gpt.get(hybrid_idx)
            if gpt_idx is None:
                raise ValueError(
                    f"Sharded state dict entry {entry.key!r} refers to hybrid layer "
                    f"{hybrid_idx}, which is not part of the hybrid layer pattern "
                    f"used to derive the GPT layer maps. The pattern and the "
                    f"instantiated model do not match."
                )
            # GPT checkpoints use the homogeneous layer format: no layer index
            # in the key, the layer is a sharding axis instead.
            entry.key = (
                f'{entry.key[:layer_match.start()]}decoder.layers.'
                f'{entry.key[layer_match.end():]}'
            )
            return _prepend_gpt_layer_axis(entry, gpt_idx, layer_maps.num_gpt_layers)

        for hybrid_prefix, gpt_prefix in _GPT_FINAL_NORM_KEY_MAP.items():
            pos = entry.key.find(hybrid_prefix)
            if pos != -1:
                entry.key = f'{entry.key[:pos]}{gpt_prefix}{entry.key[pos + len(hybrid_prefix):]}'
                break
        return entry

    dict_list_map_inplace(_retarget, sharded_state_dict)


def _retarget_explicit_key_to_gpt_checkpoint(
    key: Any,
    layer_maps: GPTCompatLayerMaps,
    checkpoint_keys: Iterable[str] | None = None,
) -> Any | None:
    """Translate one explicit HybridModel state-dict key to its GPT key.

    ``fsdp_dtensor`` checkpoints store explicit parameter names rather than
    homogeneous-layer ``ShardedTensor`` metadata. Returning ``None`` omits a
    fresh-only or GPT-omitted entry from the DCP load plan while leaving its
    existing HybridModel value untouched.
    """
    if not isinstance(key, str):
        return key
    if key.endswith(_GPT_OMITTED_LOCAL_KEYS):
        return None

    layer_match = _DECODER_LAYER_KEY_RE.search(key)
    if layer_match is not None:
        hybrid_idx = int(layer_match.group(1))
        if hybrid_idx in layer_maps.fresh_init:
            return None
        gpt_idx = layer_maps.attention_to_gpt.get(hybrid_idx)
        if gpt_idx is None:
            gpt_idx = layer_maps.mlp_to_gpt.get(hybrid_idx)
        if gpt_idx is None:
            raise ValueError(
                f"FSDP state dict entry {key!r} refers to hybrid layer {hybrid_idx}, "
                "which is not part of the hybrid layer pattern used to derive the "
                "GPT layer maps."
            )
        key = (
            f'{key[:layer_match.start()]}decoder.layers.{gpt_idx}.'
            f'{key[layer_match.end():]}'
        )

    for hybrid_prefix, gpt_prefix in _GPT_FINAL_NORM_KEY_MAP.items():
        pos = key.find(hybrid_prefix)
        if pos != -1:
            key = f'{key[:pos]}{gpt_prefix}{key[pos + len(hybrid_prefix):]}'
            break

    # FSDP optimizer parameter names include the wrapper hierarchy. GPTModel
    # and HybridModel can have different Float16/FSDP wrapper depths, so use
    # checkpoint metadata to recover the exact source-side ``module.`` prefix.
    if checkpoint_keys is not None:
        bare_key = re.sub(r'^(?:module\.)+', '', key)
        key_pattern = re.compile(rf'(?<![A-Za-z0-9_])((?:module\.)*{re.escape(bare_key)})(?:\.|$)')
        matches = {
            match.group(1)
            for checkpoint_key in checkpoint_keys
            if (match := key_pattern.search(checkpoint_key)) is not None
        }
        if len(matches) == 1:
            key = matches.pop()
    return key


def retarget_fsdp_state_dict_to_gpt_checkpoint(
    state_dict: Mapping[Any, Any],
    layer_maps: GPTCompatLayerMaps,
    checkpoint_keys: Iterable[str] | None = None,
    checkpoint_prefix: str = '',
) -> dict[Any, Any]:
    """Return an ``fsdp_dtensor`` model or optimizer state dict under GPT keys.

    FSDP model state is a flat parameter-name mapping. Distributed-optimizer
    state can contain nested ``state`` and ``param_to_group_meta`` mappings
    (and chained-optimizer integer keys), so the translation recursively
    rewrites every parameter-name key while preserving the DTensor leaves.
    """

    checkpoint_key_set = set(checkpoint_keys) if checkpoint_keys is not None else None

    def _retarget(value, path):
        if isinstance(value, Mapping):
            translated = {}
            for key, child in value.items():
                translated_key = _retarget_explicit_key_to_gpt_checkpoint(
                    key, layer_maps, checkpoint_key_set
                )
                if translated_key is not None:
                    child_path = f'{path}.{translated_key}' if path else str(translated_key)
                    translated_child = _retarget(child, child_path)
                    if (
                        checkpoint_key_set is None
                        or isinstance(child, (Mapping, list, tuple))
                        or child_path in checkpoint_key_set
                    ):
                        translated[translated_key] = translated_child
            return translated
        if isinstance(value, list):
            return [_retarget(child, f'{path}.{idx}') for idx, child in enumerate(value)]
        if isinstance(value, tuple):
            return tuple(_retarget(child, f'{path}.{idx}') for idx, child in enumerate(value))
        return value

    return _retarget(state_dict, checkpoint_prefix)
