# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Portions of this file are ported from nvrx_checkpointing
# (https://github.com/ankurv-nvidia/nvrx_checkpointing, megatron/save.py at commit
# 149cfe8b571010c33f39da15c02bfb415606828b), licensed under the Apache License, Version 2.0.

"""Checkpoint saves through the nvrx_checkpointing engine.

The node-local checkpoint daemons store one registered state structure: the first registration
sets up storage for its tensors and serialized objects, and later registrations refresh the values
in place, so every serialized object keeps the slot size reserved for it. A resumed training run
registers the state of its next save right after its checkpoint load, so that save does not wait
for the daemons to set up storage.
"""

import functools
import io
import logging
import os
import random
from pathlib import Path

import torch
from torch.distributed.checkpoint._nested_dict import flatten_state_dict

from megatron.core import mpu
from megatron.core.dist_checkpointing.dict_utils import nested_values
from megatron.core.dist_checkpointing.mapping import (
    ShardedBase,
    ShardedObject,
    ShardedTensor,
    is_main_replica,
)
from megatron.core.dist_checkpointing.quantized_tensors import (
    keep_quantized_tensors,
    packed_or_dequantized,
)
from megatron.core.dist_checkpointing.state_dict_utils import save_preprocess
from megatron.core.dist_checkpointing.utils import _clean_metadata_for_serialization
from megatron.core.rerun_state_machine import get_rerun_state_machine
from megatron.core.utils import unwrap_model
from megatron.training.global_vars import get_args

logger = logging.getLogger(__name__)

_COMMON_STATE_KEY = ShardedObject('common_state', None, (1,), (0,)).unique_key

_root = None
_session_id = None
_snapshot_id = 0
_capacities = {}
_layout = None


def _serialize(value):
    buffer = io.BytesIO()
    torch.save(value, buffer)
    return buffer.getvalue()


def _slot_size(size):
    return (size + 512 + 1023) // 1024 * 1024


def _padded(key, blob):
    capacity = _capacities.setdefault(key, _slot_size(len(blob)))
    assert len(blob) <= capacity, f'{key} grew to {len(blob)} bytes, past its {capacity}-byte slot'
    return blob.ljust(capacity, b'\0')


def _describe(value, path=()):
    """Yields, for every leaf of a prepared state, what the daemons store for it."""
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _describe(child, (*path, key))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _describe(child, (*path, index))
    elif isinstance(value, ShardedTensor):
        yield (
            path,
            value.key,
            value.dtype,
            value.global_shape,
            value.global_offset,
            value.local_shape,
            type(value.data).__name__,
            is_main_replica(value.replica_id),
        )
    elif isinstance(value, ShardedObject):
        yield path, value.unique_key, is_main_replica(value.replica_id)
    elif isinstance(value, torch.Tensor):
        local = value.to_local() if hasattr(value, 'to_local') else value
        yield path, type(value).__name__, value.dtype, tuple(value.shape), tuple(local.shape)
    else:
        yield path, type(value).__name__


def _on_any_rank(flag):
    device = 'cuda' if torch.distributed.get_backend() == 'nccl' else 'cpu'
    flag = torch.tensor([int(flag)], dtype=torch.int32, device=device)
    torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MAX)
    return bool(flag.item())


def _register(
    state_dict,
    validate_access_integrity=False,
    preprocess_common_before_consistancy_check=None,
    content_metadata=None,
):
    """Supplies `state_dict` to the node daemons for the next snapshot. Collective.

    Megatron sharded state dicts are stored in the torch_dist layout, DTensor state dicts
    (Megatron-FSDP, Torch-FSDP2) in the plain torch.distributed.checkpoint layout that their
    loaders read. A state that the registered storage cannot hold is registered anew.
    """
    global _root, _session_id, _layout
    from nvrx_checkpointing import checkpoint
    from nvrx_checkpointing.peers import OrderedPeers, detect_environment_peers

    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    if any(isinstance(value, ShardedBase) for value in nested_values(state_dict)):
        state_dict['content_metadata'] = content_metadata
        # Factories, like the SwiGLU split, slice quantized parameters only now.
        with keep_quantized_tensors():
            state, common = save_preprocess(
                state_dict,
                validate_access_integrity and _root is None,
                preprocess_common_before_consistancy_check,
            )
        # The daemons store sharded values by their own keys; built factories nest lists in lists.
        state = {str(index): value for index, value in enumerate(nested_values(state))}
        for value in state.values():
            if isinstance(value, ShardedTensor):
                value.data = packed_or_dequantized(value.data)
        objects = {_COMMON_STATE_KEY: [common]} if rank == 0 else {}
        sharded_objects = [
            value
            for value in nested_values(state)
            if isinstance(value, ShardedObject) and is_main_replica(value.replica_id)
        ]
        objects.update((value.unique_key, [value.data]) for value in sharded_objects)
    else:
        state, _ = flatten_state_dict(state_dict)
        sharded_objects = []
        objects = {k: v for k, v in state.items() if not isinstance(v, torch.Tensor)}
    blobs = {key: _serialize(value) for key, value in objects.items()}
    layout = list(_describe(state))

    if _root is not None:
        changed = layout != _layout or any(
            len(blob) > _capacities.get(key, 0) for key, blob in blobs.items()
        )
        if _on_any_rank(changed):
            logger.warning(
                'The checkpoint state no longer fits its registration; registering it anew.'
            )
            reset()
            # Preprocessing is idempotent, and a new registration validates the state.
            return _register(
                state_dict,
                validate_access_integrity,
                preprocess_common_before_consistancy_check,
                content_metadata,
            )

    if _root is None:
        # Ranks serialize a replicated object to different sizes, but the daemons store one slot.
        sizes = [None] * world_size
        torch.distributed.all_gather_object(sizes, {key: len(blob) for key, blob in blobs.items()})
        for rank_sizes in sizes:
            for key, size in rank_sizes.items():
                _capacities[key] = max(_capacities.get(key, 0), _slot_size(size))
        session_id = [random.SystemRandom().randrange(1, 1 << 31)]
        torch.distributed.broadcast_object_list(session_id, src=0)
        _session_id = session_id[0]
        peers = detect_environment_peers(rank, world_size)
        assert (
            peers is not None or world_size <= torch.cuda.device_count()
        ), 'Native checkpointing finds the nodes of a multi-node run through SLURM or ft_launcher'
        peers = peers or OrderedPeers(('localhost',), world_size)
        os.environ.setdefault('NATIVE_CKPT_SERVER_AUTO_SHUTDOWN_TIMEOUT_SECONDS', '0')
        # Launched from the background registration, the daemons could wait forever on a pipe
        # that a process forked meanwhile, like a data loader worker, inherited.
        checkpoint.ping()
        _root = checkpoint.model_state.create_root(
            _session_id, rank, world_size, **peers.checkpoint_args()
        )
        _layout = layout

    for value in sharded_objects:
        value._nvrx_serialized_blob = _padded(value.unique_key, blobs.pop(value.unique_key))
    state.update({key: _padded(key, blob) for key, blob in blobs.items()})
    _root.register_state_dict(state)


def save(
    state_dict,
    checkpoint_dir,
    validate_access_integrity=False,
    preprocess_common_before_consistancy_check=None,
    content_metadata=None,
):
    """Snapshots `state_dict` into the node daemons, which write it to `checkpoint_dir`. Collective.

    Returns once the snapshot no longer reads the state tensors, with a function that waits until
    the checkpoint is durable.
    """
    global _snapshot_id
    from nvrx_checkpointing import checkpoint

    _register(
        state_dict,
        validate_access_integrity,
        preprocess_common_before_consistancy_check,
        content_metadata,
    )
    _snapshot_id += 1
    _root.save_async(_snapshot_id, Path(checkpoint_dir))
    _root.wait_for_async_completion()
    return functools.partial(checkpoint.wait_for_ckpt_finalize, _session_id, _snapshot_id)


def register(
    state_dict,
    validate_access_integrity=False,
    preprocess_common_before_consistancy_check=None,
    content_metadata=None,
):
    """Registers the state of a later save ahead of it. Collective.

    The daemons set up storage for a new state structure in the background, so the save does not
    wait for it. The save supplies its state again, as a snapshot reads the state it is given.
    """
    _register(
        state_dict,
        validate_access_integrity,
        preprocess_common_before_consistancy_check,
        content_metadata,
    )
    _root.finalize_registry()


def register_after_load(
    model,
    optimizer,
    opt_param_scheduler,
    iteration,
    num_floating_point_operations_so_far,
    preprocess_common_before_consistancy_check=None,
):
    """Registers the state of the next save right after a checkpoint load, ahead of that save.

    Builds the state the way `save_checkpoint` does for a save through the daemons, which set up
    its storage while training starts. Collective.
    """
    from megatron.training import checkpointing
    from megatron.training.ckpt.glu_checkpointing import prepare_glu_checkpoint_for_save

    args = get_args()
    model = unwrap_model(model)
    pg_collection = getattr(model[0], 'pg_collection', None)
    tp_group = getattr(pg_collection, 'tp', None)
    pp_group = getattr(pg_collection, 'pp', None)
    dp_cp_group = getattr(pg_collection, 'dp_cp_gtp_remat', None)
    if tp_group is None and pp_group is None:
        tp_group = mpu.get_tensor_model_parallel_group()
        pp_group = mpu.get_pipeline_model_parallel_group()
    rng_state = checkpointing.get_rng_state(
        args.ckpt_format,
        tp_group,
        pp_group,
        dp_cp_group=dp_cp_group,
        dp_group=getattr(pg_collection, 'dp', None),
        key_prefix=getattr(model[0], 'rng_state_key_prefix', ''),
    )
    rerun_state = get_rerun_state_machine().state_dict(
        data_iterator=None, ckpt_format=args.ckpt_format
    )
    metadata = checkpointing._build_sharded_state_dict_metadata(args, dp_cp_group=dp_cp_group)
    with keep_quantized_tensors():
        state_dict = checkpointing.generate_state_dict(
            args,
            model,
            optimizer,
            opt_param_scheduler,
            rng_state,
            iteration=iteration,
            optim_sd_kwargs=dict(metadata=metadata),
            model_sd_kwargs=dict(metadata=metadata),
            rerun_state=rerun_state,
        )
    state_dict = prepare_glu_checkpoint_for_save(state_dict, args)
    state_dict['num_floating_point_operations_so_far'] = num_floating_point_operations_so_far
    if args.ckpt_format == 'fsdp_dtensor':
        state_dict = checkpointing.preprocess_fsdp_dtensor_state_dict(args, state_dict, model[0])
    register(
        state_dict,
        args.ckpt_load_validate_sharding_integrity,
        preprocess_common_before_consistancy_check,
        _clean_metadata_for_serialization(metadata),
    )


def reset():
    """Forgets the registered state structure, so the next save registers a new one."""
    global _root, _layout
    _root = None
    _layout = None
    _capacities.clear()


def spin_down():
    """Stops the node daemons started by this process once every rank is done saving. Collective."""
    global _session_id
    if _session_id is None:
        return
    from nvrx_checkpointing import checkpoint

    # Dropping the registered state waits for the engine work still talking to the daemons.
    reset()
    torch.distributed.barrier()
    checkpoint.svc_spindown()
    _session_id = None
    torch.distributed.barrier()
