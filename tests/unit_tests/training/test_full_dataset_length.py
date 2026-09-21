# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from megatron.core import parallel_state
from megatron.training.training import _reduce_sum_across_data_parallel_group
from megatron.training.utils import reduce_max_stat_across_model_parallel_group
from tests.unit_tests.test_utilities import Utils


def test_cp_deduplicated_dataloader_length_reaches_every_rank():
    Utils.initialize_model_parallel(tensor_model_parallel_size=2, context_parallel_size=2)

    # Match the CP-deduplicated multimodal loader: TP=0/CP=0 owns one
    # data-parallel shard length, and every other model-parallel rank has None.
    shard_length = (
        10.0
        if parallel_state.get_tensor_model_parallel_rank() == 0
        and parallel_state.get_context_parallel_rank() == 0
        else None
    )
    total_length = _reduce_sum_across_data_parallel_group(shard_length, with_context_parallel=True)
    total_length = reduce_max_stat_across_model_parallel_group(total_length)

    expected_length = 10.0 * parallel_state.get_data_parallel_world_size()
    assert total_length == expected_length

    Utils.destroy_model_parallel()
