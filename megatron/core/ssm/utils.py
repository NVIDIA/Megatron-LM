# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from dataclasses import replace
from typing import Optional

import torch

from megatron.core.dist_checkpointing import ShardedTensor
from megatron.core.dist_checkpointing.mapping import ReplicaId, ShardedTensorFactory
from megatron.core.tensor_parallel import gtp_api
from megatron.core.tensor_parallel.gtp_utils import (
    _fused_projection_optimizer_factory,
    _gtp_gather_rows_for_save,
    _gtp_slice_rows_on_load,
)
from megatron.core.transformer.utils import cat_with_oom_fallback


def _split_in_proj_factory(
    orig_sh_ten: ShardedTensor,
    split_sections: list[int],
    split_names: list[str],
    *,
    weight: torch.Tensor,
    tp_group: torch.distributed.ProcessGroup,
    dp_cp_group: torch.distributed.ProcessGroup,
    sharded_offsets: tuple[tuple[int, int, int], ...] = (),
) -> ShardedTensorFactory:
    """Checkpoint an SSM input projection in its logical, TP-local section layout.

    Mamba, GDN, and GDP concatenate different semantic sections along dimension 0.
    GTP row-shard boundaries can cross those sections, so gather before splitting
    and slice back to the physical shard after merging on load. The shared GTP
    helpers handle alignment padding and elect one checkpoint writer per replica.

    ``split_sections`` contains TP-local sizes; ``split_names`` preserves the
    module's checkpoint keys (including GDP's individual householder copies).
    ``orig_sh_ten.data`` is the checkpoint representation, already dequantized
    for native-FP8 weights. ``weight`` supplies the live GTP group and shard shape.
    Ordinary weights and replicated biases use the section factory directly.

    All GTP ranks must call this together when constructing the checkpoint dict.
    Model weights merge to physical GTP shards. A source-parameter companion
    maps optimizer tensors and flat DP fragments to the same semantic keys
    without additional collectives.
    """
    uses_gtp = gtp_api.HAVE_GTP and gtp_api.is_gtp_param(weight)
    if uses_gtp:
        # Read the parameter's logical width independently of the requested
        # sections, so the split factory still rejects wrong totals.
        target_rows = weight._unsharded_shape[0]
        orig_sh_ten = _gtp_gather_rows_for_save(
            orig_sh_ten,
            orig_sh_ten.key,
            weight,
            target_rows,
            tp_group,
            dp_cp_group,
            sharded_offsets,
        )

    factory = _split_tensor_factory(orig_sh_ten, split_sections, split_names, split_dim=0)
    if uses_gtp:
        factory = _gtp_slice_rows_on_load(factory, weight)
    else:
        factory.optimizer_factory = _fused_projection_optimizer_factory(factory, weight)
    return factory


def _split_tensor_factory(
    orig_sh_ten: ShardedTensor, split_sections: list[int], split_names: list[str], split_dim: int
) -> ShardedTensorFactory:
    """Builds a factory that splits a given ShardedTensor into several independent chunks."""
    assert isinstance(orig_sh_ten, ShardedTensor), type(orig_sh_ten)
    orig_sh_ten_no_data = orig_sh_ten.without_data()  # remove `data` reference

    if sum(split_sections) != orig_sh_ten_no_data.local_shape[split_dim]:
        raise ValueError(
            f"Split sections must cover the whole dimension size, "
            f"got {split_sections=} vs dimensions size "
            f"{orig_sh_ten_no_data.local_shape[split_dim]}"
        )

    assert not isinstance(
        split_sections, int
    ), "Splitting into predefined section sizes is supported (`split_sections` must be a list)"
    assert len(split_sections) == len(split_names), (len(split_sections), len(split_names))

    @torch.no_grad()
    def sh_ten_build_fn(
        key: str, t: torch.Tensor, replica_id: ReplicaId, flattened_range: Optional[slice]
    ):
        factory_sh_ten = replace(
            orig_sh_ten_no_data,
            key=key,
            data=t,
            dtype=t.dtype,
            replica_id=replica_id,
            flattened_range=flattened_range,
        )

        chunk_sh_tens = []
        split_start = 0
        for split_size, split_name in zip(split_sections, split_names):
            split_chunks = factory_sh_ten.narrow(split_dim, split_start, split_size)
            for sh_ten in split_chunks:
                sh_ten.key = f"{sh_ten.key}.{split_name}"
            chunk_sh_tens.extend(split_chunks)
            split_start += split_size

        assert split_start == orig_sh_ten_no_data.local_shape[split_dim], (
            split_start,
            orig_sh_ten_no_data.local_shape[split_dim],
        )
        assert sum(sh_ten.data.numel() for sh_ten in chunk_sh_tens) == t.numel(), (
            chunk_sh_tens,
            t.shape,
        )
        return chunk_sh_tens

    return ShardedTensorFactory(
        orig_sh_ten.key,
        orig_sh_ten.data,
        sh_ten_build_fn,
        cat_with_oom_fallback,
        orig_sh_ten.replica_id,
    )
