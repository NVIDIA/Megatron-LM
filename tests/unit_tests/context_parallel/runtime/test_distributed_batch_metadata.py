# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Packed TP x runtime-CP metadata uses the caller's real parallel grid."""

from types import SimpleNamespace

import pytest
import torch

from megatron.core.datasets.data_schedule import get_batch_on_this_rank_for_sequence_packing
from megatron.core.process_groups_config import ProcessGroupCollection
from tests.unit_tests.test_utilities import Utils

pytestmark = pytest.mark.launch_on_gb200


@pytest.mark.parametrize("local_cp_size", [1, 2])
def test_packed_batch_records_matching_real_tp_cp_group(local_cp_size):
    if Utils.world_size < 4 or Utils.world_size % 4:
        pytest.skip("TP2 with runtime CP2 requires a world size divisible by four")
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=2,
        dynamic_context_parallel=True,
        min_dynamic_context_parallel_size=1,
    )
    try:
        groups = ProcessGroupCollection.use_mpu_process_groups()
        data_iterator = None
        if groups.tp.rank() == 0:
            tokens = torch.arange(32, device="cuda", dtype=torch.int64)
            bounds = torch.tensor([0, 16, 32], device="cuda", dtype=torch.int32)
            data_iterator = iter(
                [
                    {
                        "tokens": tokens,
                        "labels": tokens + 1,
                        "position_ids": torch.arange(16, device="cuda").repeat(2),
                        "loss_mask": torch.ones(32, device="cuda"),
                        "cu_seqlens": bounds,
                        "cu_seqlens_padded": bounds,
                        "max_seqlen": torch.tensor([16], device="cuda", dtype=torch.int32),
                        "local_cp_size": torch.tensor(
                            [local_cp_size], device="cuda", dtype=torch.int32
                        ),
                    }
                ]
            )
        *_, packed, _ = get_batch_on_this_rank_for_sequence_packing(
            data_iterator,
            dynamic_cp=True,
            pg_collection=groups,
            config=SimpleNamespace(sequence_parallel=True),
        )
        assert packed.cp_group is groups.dynamic_cp[local_cp_size]
        assert packed.tp_cp_group is groups.dynamic_tp_cp[local_cp_size]
        cp_ranks = set(torch.distributed.get_process_group_ranks(packed.cp_group))
        tp_ranks = set(torch.distributed.get_process_group_ranks(groups.tp))
        combined = set(torch.distributed.get_process_group_ranks(packed.tp_cp_group))
        assert len(combined) == groups.tp.size() * local_cp_size
        assert cp_ranks <= combined and tp_ranks <= combined
    finally:
        Utils.destroy_model_parallel()
