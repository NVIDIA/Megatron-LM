# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from dataclasses import replace
from typing import Optional

import torch

from megatron.core.context_parallel.groups import get_valid_dynamic_context_parallel_group_sizes
from megatron.core.dist_checkpointing import ShardedTensor
from megatron.core.dist_checkpointing.mapping import ReplicaId, ShardedTensorFactory
from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.utils import cat_with_oom_fallback


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


def _validate_dynamic_cp_head_geometry(
    config: ModelParallelConfig,
    pg_collection: ProcessGroupCollection,
    *,
    head_counts: dict[str, int],
    group_count: int | None = None,
) -> None:
    """Reject headwise shapes unsupported by any scheduler-enabled CP size at startup.

    Chunkwise GDP does not shard heads and must not call this validator. Mamba/GDP
    may repeat groups when CP exceeds their TP-local group count; GDN must shard
    both its key and value heads without repetition.
    """
    if not config.dynamic_context_parallel:
        return
    if getattr(pg_collection, "dp_cp", None) is None:
        raise ValueError("dynamic CP headwise mixers require pg_collection.dp_cp at construction")
    dynamic_cp_groups = getattr(pg_collection, "dynamic_cp", None)
    sizes = (
        sorted(dynamic_cp_groups)
        if dynamic_cp_groups
        else get_valid_dynamic_context_parallel_group_sizes(pg_collection.dp_cp.size())
    )
    for cp_size in sizes:
        if cp_size < config.min_dynamic_context_parallel_size:
            continue
        for name, head_count in head_counts.items():
            if head_count < 1 or head_count % cp_size:
                raise ValueError(
                    f"dynamic CP can select runtime CP size {cp_size}, but TP-local "
                    f"{name}={head_count} must be positive and divisible by that size. "
                    "Reduce the DP x CP size or TP size, or increase the head count. "
                    "GDP can use linear_cp_mode='chunkwise' to keep all heads."
                )
        if group_count is not None and (
            group_count < 1 or (group_count % cp_size != 0 and cp_size % group_count != 0)
        ):
            raise ValueError(
                f"TP-local group count {group_count} cannot be sharded or repeated "
                f"for runtime CP size {cp_size}"
            )
