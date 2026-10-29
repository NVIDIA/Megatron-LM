# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
def local_tensors(module):
    """Parameters and persistent buffers owned directly by ``module``."""
    tensors = dict(module.named_parameters(recurse=False))
    tensors.update(
        (n, b)
        for n, b in module.named_buffers(recurse=False)
        if n not in module._non_persistent_buffers_set
    )
    return tensors


def _sharded_state(module, prefix, offsets, ps, metadata):
    """Core sharded state for the Row-sharded tree."""
    from megatron.core.dist_checkpointing.mapping import ShardedTensor
    from megatron.core.transformer.module import MegatronModule
    from megatron.core.transformer.utils import make_sharded_tensors_for_checkpoint

    lookup = getattr(module, "lookup", None)
    if getattr(lookup, "group", None) is not None:
        size, rank = lookup.group.size(), lookup.group.rank()
        shard = lookup.boundaries[1]
        if tuple(lookup.boundaries) != tuple(i * shard for i in range(size + 1)):
            raise ValueError("Row-sharded checkpoint requires equal Core row shards")
        axis = len(offsets)  # rows follow any prepended (pipeline) axes
        return {
            prefix
            + name: ShardedTensor.from_rank_offsets(
                prefix + name,
                tensor,
                *offsets,
                (axis, rank, size),
                replica_id=(0, 0, ps.dp_cp_rank // size),
                prepend_axis_num=axis,
            )
            for name, tensor in local_tensors(module).items()
        }
    rows = [m for m in module.modules() if getattr(getattr(m, "lookup", None), "group", None)]
    if isinstance(module, MegatronModule) and not rows:
        return module.sharded_state_dict(prefix, offsets, metadata)
    state = make_sharded_tensors_for_checkpoint(
        local_tensors(module), prefix, {}, offsets, dp_cp_group=ps.dp_cp_group
    )
    for name, child in module.named_children():
        state.update(_sharded_state(child, f"{prefix}{name}.", offsets, ps, metadata))
    return state


def attach_checkpoint_metadata(chunks, ps):
    """Opt chunks into the Core dist-opt checkpoint (model + optimizer state)."""
    from types import MethodType

    for chunk in chunks:
        model = chunk
        while hasattr(model, "module"):
            model = model.module

        def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None, _m=model):
            # Explicit groups: Core's defaults would consult the global mpu state.
            metadata = dict(metadata or {}, dp_cp_group=ps.dp_cp_group)
            state = {}
            for name, child in _m.named_children():
                state.update(
                    _sharded_state(child, f"{prefix}{name}.", sharded_offsets, ps, metadata)
                )
            return state

        chunk.sharded_state_dict = MethodType(sharded_state_dict, chunk)
        chunk._mlite_dist_opt_sharded_state_dict = True
        chunk._mlite_dist_opt_parallel_state = ps
