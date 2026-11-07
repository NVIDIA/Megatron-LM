# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Load distributed checkpoints saved before a layer spec's ``sharded_state_dict_keys_map``.

A layer spec's ``sharded_state_dict_keys_map`` renames a layer's runtime module names to
canonical checkpoint names (e.g. ``input_layernorm.`` ->
``self_attention.linear_qkv.layer_norm_``). Checkpoints saved without that renaming store the
same tensors under the runtime names. Rather than changing the model, the load request is
retargeted at those names, so later saves still use the canonical ones.
"""

import logging
from dataclasses import replace

from megatron.core.dist_checkpointing.dict_utils import dict_list_map_inplace
from megatron.core.dist_checkpointing.mapping import ShardedTensor, ShardedTensorFactory
from megatron.core.dist_checkpointing.serialization import load_tensors_metadata
from megatron.core.utils import log_single_rank

logger = logging.getLogger(__name__)


def remap_sharded_state_dict_for_load(models, state_dict, checkpoint_dir):
    """Resolve layer-spec key aliases in a distributed checkpoint's load request.

    Prefer the requested checkpoint key. If it is absent, try the corresponding runtime
    module name from the layer specs. Only the load template is changed; the models and
    their canonical save mappings stay untouched. Apply this after other load modifiers
    to the complete template, including model-keyed optimizer state.
    """
    aliases = {
        (runtime, canonical)
        for model in models
        for module in model.modules()
        for runtime, canonical in (
            getattr(getattr(module, "submodules_config", None), "sharded_state_dict_keys_map", None)
            or {}
        ).items()
    }
    if not aliases:
        return
    checkpoint_keys = load_tensors_metadata(checkpoint_dir).keys()
    resolved = []

    def remap(value):
        if isinstance(value, ShardedTensorFactory):
            # Factories can expand one virtual key into several stored tensor keys. Resolve
            # those actual keys lazily, preserving the original merge function and tensor.
            def build(key, data, replica_id, flattened_range):
                return dict_list_map_inplace(
                    remap, value.build_fn(key, data, replica_id, flattened_range)
                )

            return replace(value, build_fn=build)
        if not isinstance(value, ShardedTensor) or value.key in checkpoint_keys:
            return value
        candidates = set()
        for runtime, canonical in aliases:
            prefix, matched, suffix = value.key.rpartition(canonical)
            if matched and (not prefix or prefix.endswith(".")):
                candidate = f"{prefix}{runtime}{suffix}"
                if candidate in checkpoint_keys:
                    candidates.add(candidate)
        if len(candidates) > 1:
            raise ValueError(f"Ambiguous checkpoint aliases for {value.key}: {sorted(candidates)}")
        if candidates:
            resolved.append(value.key)
            return replace(value, key=candidates.pop())
        # Leave missing keys to the normal distributed-checkpoint strictness policy.
        return value

    dict_list_map_inplace(remap, state_dict)
    if resolved:
        log_single_rank(
            logger,
            logging.INFO,
            f"Checkpoint {checkpoint_dir} stores {len(resolved)} tensors under their runtime "
            f"module names (e.g. for '{resolved[0]}'); loading them from there. Checkpoints "
            f"saved from this model use the canonical names.",
        )
