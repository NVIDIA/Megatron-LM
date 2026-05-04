# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""
Distributed checkpoint I/O helpers for structural model-conversion tools.

Provides format detection, model-free full-tensor loading, and single-rank
saving for Megatron-LM distributed checkpoints (``torch_dist`` and
``fsdp_dtensor`` backends). This lets conversion tools operate on
TP+PP+FSDP-trained checkpoints without needing to instantiate the model.

The key observation is that PyTorch DCP stores each logical parameter with a
``global_shape`` in its metadata, and the TP / PP / FSDP slicing is just an
on-disk layout detail handled by the read planner. Loading into a plain
``torch.empty(global_shape)`` state dict on rank 0 therefore yields fully
gathered tensors regardless of the parallelism the checkpoint was trained with.
"""

import os
from collections import OrderedDict
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import (
    DefaultLoadPlanner,
    DefaultSavePlanner,
    FileSystemReader,
    FileSystemWriter,
)
from torch.distributed.checkpoint.metadata import (
    BytesStorageMetadata,
    TensorStorageMetadata,
)

from megatron.core.dist_checkpointing.core import (
    CheckpointingConfig,
    maybe_load_config,
    save_config,
)
from megatron.core.dist_checkpointing.strategies.common import (
    load_common,
    save_common,
)


FORMAT_TORCH_DIST = 'torch_dist'
FORMAT_FSDP_DTENSOR = 'fsdp_dtensor'
DIST_FORMATS = (FORMAT_TORCH_DIST, FORMAT_FSDP_DTENSOR)

# Prefixes under which model weights may be keyed in a dist checkpoint.
_KNOWN_MODEL_PREFIXES = ('model.module.module.', 'model.module.', 'model.', '')
# Well-known bare-key suffixes we probe for when detecting the prefix.
_PROBE_SUFFIXES = (
    'embedding.word_embeddings.weight',
    'decoder.layers.',
    'decoder.final_norm.',
    'decoder.final_layernorm.',
    'output_layer.weight',
)
# Keys that identify non-model state we drop during architecture conversion.
_NON_MODEL_TOP_LEVEL_PREFIXES = (
    'optimizer.',
    'rng_state',
    'rerun_state_machine_state',
)


def resolve_checkpoint_subdir(load_dir):
    """Return ``(ckpt_dir, iteration)``.

    Megatron writes checkpoints either flat or under ``iter_XXXXXXX/``. This
    picks the right directory and reports the iteration when it can be
    determined.
    """
    if os.path.exists(os.path.join(load_dir, 'metadata.json')):
        return load_dir, None

    latest_iter = os.path.join(load_dir, 'latest_checkpointed_iteration.txt')
    if os.path.exists(latest_iter):
        with open(latest_iter, 'r') as f:
            iteration = int(f.read().strip())
        iter_dir = os.path.join(load_dir, f'iter_{iteration:07d}')
        if os.path.isdir(iter_dir):
            return iter_dir, iteration

    return load_dir, None


def detect_checkpoint_format(load_dir):
    """Return one of ``{'torch_dist', 'fsdp_dtensor'}``.

    Raises ``ValueError`` if the directory looks like the legacy
    ``mp_rank_XX`` layout (no longer supported) or doesn't match any known
    dist-checkpoint metadata.
    """
    ckpt_dir, _ = resolve_checkpoint_subdir(load_dir)
    config = maybe_load_config(ckpt_dir)
    if config is not None:
        return config.sharded_backend

    if os.path.isdir(ckpt_dir) and any(
        name.startswith('mp_rank_') for name in os.listdir(ckpt_dir)
    ):
        raise ValueError(
            f"{load_dir} looks like a legacy mp_rank_XX checkpoint. "
            f"Legacy format is no longer supported — convert to torch_dist first."
        )

    raise ValueError(f"Unrecognized checkpoint format at {load_dir}")


def ensure_single_rank_process_group():
    """Initialize a 1-rank gloo process group if one isn't already up.

    DCP requires a default process group; this lets the conversion tool run
    in a plain ``python`` invocation (no ``torchrun`` needed).
    """
    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available.")
    if dist.is_initialized():
        return
    os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
    os.environ.setdefault('MASTER_PORT', '29500')
    os.environ.setdefault('RANK', '0')
    os.environ.setdefault('WORLD_SIZE', '1')
    os.environ.setdefault('LOCAL_RANK', '0')
    dist.init_process_group(backend='gloo', rank=0, world_size=1)


def detect_model_prefix(keys):
    """Return the prefix under which model weights live in ``keys``.

    Looks for a recognizable suffix (``embedding.word_embeddings.weight``,
    ``decoder.layers.``, etc.) and returns the matching prefix. Falls back
    to ``''`` if nothing obvious is found.
    """
    keys = list(keys)
    for prefix in _KNOWN_MODEL_PREFIXES:
        for suffix in _PROBE_SUFFIXES:
            probe = prefix + suffix
            for key in keys:
                if key.startswith(probe):
                    return prefix
    return ''


def _is_non_model_key(bare_key):
    if bare_key.startswith(_NON_MODEL_TOP_LEVEL_PREFIXES):
        return True
    # _extra_state blobs are TE per-module state; they are tied to a specific
    # TP/parallelism configuration and aren't meaningful after a structural
    # model conversion, so we drop them.
    if '_extra_state' in bare_key:
        return True
    return False


def load_dist_checkpoint_full(load_dir):
    """Load a dist checkpoint and return fully-gathered model weights.

    Returns:
        model_state_dict (OrderedDict[str, torch.Tensor]): bare keys, full
            tensors on CPU. Optimizer state, RNG state, and ``_extra_state``
            blobs are filtered out.
        common_state (dict): contents of ``common.pt`` (e.g. ``args``).
        model_prefix (str): the prefix we stripped (re-apply on save).
        backend (str): ``'torch_dist'`` or ``'fsdp_dtensor'``.
        iteration (int or None): iteration number if discoverable.
    """
    ensure_single_rank_process_group()

    ckpt_dir, iteration = resolve_checkpoint_subdir(load_dir)
    config = maybe_load_config(ckpt_dir)
    if config is None:
        raise ValueError(
            f"{load_dir} is not a distributed checkpoint (no metadata.json)"
        )
    backend = config.sharded_backend

    reader = FileSystemReader(ckpt_dir)
    metadata = reader.read_metadata()

    model_prefix = detect_model_prefix(metadata.state_dict_metadata.keys())

    raw_state_dict = {}
    for key, md in metadata.state_dict_metadata.items():
        if not isinstance(md, TensorStorageMetadata):
            continue
        if model_prefix and not key.startswith(model_prefix):
            continue
        bare_key = key[len(model_prefix):] if model_prefix else key
        if _is_non_model_key(bare_key):
            continue
        raw_state_dict[key] = torch.empty(
            md.size, dtype=md.properties.dtype, device='cpu'
        )

    if not raw_state_dict:
        raise ValueError(
            f"No model tensors found in {ckpt_dir} (detected prefix "
            f"'{model_prefix}', backend '{backend}')."
        )

    dcp.load(raw_state_dict, storage_reader=reader, planner=DefaultLoadPlanner())

    model_state_dict = OrderedDict()
    for key, tensor in raw_state_dict.items():
        bare_key = key[len(model_prefix):] if model_prefix else key
        model_state_dict[bare_key] = tensor

    common_state = {}
    try:
        common_state = load_common(ckpt_dir)
    except Exception:
        pass

    return model_state_dict, common_state, model_prefix, backend, iteration


def save_dist_checkpoint_full(
    model_state_dict,
    common_state,
    save_dir,
    model_prefix='model.',
    backend=FORMAT_TORCH_DIST,
):
    """Save a fully-gathered state dict as a distributed checkpoint.

    The output is written as a single-rank, fully-replicated DCP checkpoint
    plus ``common.pt`` and ``metadata.json``. A downstream Megatron training
    job reads it back through ``dist_checkpointing.load()`` with its own
    sharded_state_dict template — TP+PP+FSDP resharding happens transparently
    on load, since the on-disk tensors carry their full logical shape.
    """
    ensure_single_rank_process_group()

    os.makedirs(save_dir, exist_ok=True)

    raw_state_dict = OrderedDict()
    for bare_key, tensor in model_state_dict.items():
        full_key = f"{model_prefix}{bare_key}" if model_prefix else bare_key
        raw_state_dict[full_key] = tensor.contiguous() if tensor.is_contiguous() else tensor.contiguous()

    writer = FileSystemWriter(save_dir)
    dcp.save(
        state_dict=raw_state_dict,
        storage_writer=writer,
        planner=DefaultSavePlanner(),
    )

    if common_state:
        save_common(common_state, save_dir)

    if dist.get_rank() == 0:
        save_config(CheckpointingConfig(sharded_backend=backend), save_dir)
    dist.barrier()


def write_latest_iteration_marker(save_dir, iteration):
    """Mirror the legacy ``latest_checkpointed_iteration.txt`` convention.

    When ``save_dir`` points at a top-level checkpoint root with an
    ``iter_XXXXXXX/`` subdirectory, the tracker file lets Megatron auto-find
    the latest iteration on load.
    """
    parent = os.path.dirname(save_dir.rstrip('/')) or save_dir
    if os.path.basename(save_dir.rstrip('/')).startswith('iter_'):
        tracker = os.path.join(parent, 'latest_checkpointed_iteration.txt')
        with open(tracker, 'w') as f:
            f.write(str(iteration))
