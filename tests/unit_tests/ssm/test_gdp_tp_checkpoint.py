# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Regression tests for GDP tensor-parallel checkpoint resharding."""

import inspect
from collections import defaultdict

import torch

from megatron.core.dist_checkpointing import ShardedTensor
from megatron.core.ssm.gated_delta_product import (
    GatedDeltaProductMixer,
    _get_in_proj_checkpoint_split_layout,
    _split_tensor_factory,
)


def test_constructor_drops_unused_mamba_compatibility_arguments():
    """GDP dimensions come from TransformerConfig, not ignored constructor overrides."""
    parameters = inspect.signature(GatedDeltaProductMixer.__init__).parameters
    removed_parameters = {
        "expand",
        "dt_init",
        "dt_scale",
        "use_mem_eff_path",
        "d_state",
        "headdim",
        "ngroups",
    }
    assert parameters.keys().isdisjoint(removed_parameters)


def test_householder_components_reshard_tp2_to_tp1_in_semantic_order():
    """Each householder copy must gather across TP ranks before copies are concatenated."""
    num_householder = 3
    local_sections, names = _get_in_proj_checkpoint_split_layout(
        d_inner_local_tp=2,
        group_state_local_tp=1,
        nheads_local_tp=1,
        num_householder=num_householder,
    )

    # Local layout is [z, V0, V1, V2, K0, K1, K2, Q, b0, b1, b2, a].
    rank_data = [
        torch.tensor([0, 1, 10, 11, 20, 21, 30, 31, 40, 50, 60, 70, 80, 90, 100, 110]),
        torch.tensor([2, 3, 12, 13, 22, 23, 32, 33, 41, 51, 61, 71, 81, 91, 101, 111]),
    ]

    checkpoint_chunks = defaultdict(list)
    for tp_rank, local_data in enumerate(rank_data):
        sharded_tensor = ShardedTensor.from_rank_offsets(
            "in_proj.weight", local_data, (0, tp_rank, 2)
        )
        factory = _split_tensor_factory(sharded_tensor, local_sections, names, split_dim=0)
        for chunk in factory.build():
            checkpoint_chunks[chunk.key].append(chunk.data)

    # Simulate DCP assembling every semantic checkpoint key for a TP=1 load.
    assembled_checkpoint = {
        key: torch.cat(chunks, dim=0) for key, chunks in checkpoint_chunks.items()
    }

    global_sections, global_names = _get_in_proj_checkpoint_split_layout(
        d_inner_local_tp=4,
        group_state_local_tp=2,
        nheads_local_tp=2,
        num_householder=num_householder,
    )
    target_tensor = ShardedTensor.from_rank_offsets(
        "in_proj.weight", torch.empty(sum(global_sections), dtype=torch.int64), (0, 0, 1)
    )
    target_factory = _split_tensor_factory(
        target_tensor, global_sections, global_names, split_dim=0
    )
    loaded_chunks = [assembled_checkpoint[chunk.key] for chunk in target_factory.build()]
    reloaded = target_factory.merge_fn(loaded_chunks)

    expected = torch.tensor(
        [
            0,
            1,
            2,
            3,
            10,
            11,
            12,
            13,
            20,
            21,
            22,
            23,
            30,
            31,
            32,
            33,
            40,
            41,
            50,
            51,
            60,
            61,
            70,
            71,
            80,
            81,
            90,
            91,
            100,
            101,
            110,
            111,
        ]
    )
    torch.testing.assert_close(reloaded, expected)
