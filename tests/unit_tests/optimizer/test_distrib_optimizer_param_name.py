# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""The distributed optimizer names experts by its model's own expert-parallel layout."""

import pytest
import torch

from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.fsdp_dtensor_checkpoint import get_ep_rank_and_size
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.test_fsdp_dtensor_expert_offsets import (
    expert_model,
    forbid_global_expert_parallel_reads,
)

NUM_LOCAL_EXPERTS = 2


class TestParamNameOnACustomExpertGrid:

    def setup_method(self, method):
        if Utils.world_size < 4 or Utils.world_size % 2 != 0:
            pytest.skip("needs a global EP=2 grid that differs from an EP=world grid")
        # The global grid pairs ranks along EP.
        Utils.initialize_model_parallel(expert_model_parallel_size=2)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_param_names_follow_the_model_expert_layout(self):
        world_size, rank = Utils.world_size, torch.distributed.get_rank()
        # The model's own EP group spans every rank.
        ep_group = torch.distributed.new_group(ranks=list(range(world_size)))
        pg_collection = ProcessGroupCollection(ep=ep_group)
        assert get_ep_rank_and_size(pg_collection) == (rank, world_size)

        model = expert_model(NUM_LOCAL_EXPERTS, num_moe_experts=NUM_LOCAL_EXPERTS * world_size)
        model.pg_collection = pg_collection
        optimizer = object.__new__(DistributedOptimizer)
        optimizer.model_chunks = [model]
        local_experts = model.decoder.layers[0].mlp.experts.local_experts

        with forbid_global_expert_parallel_reads():
            names = [optimizer._param_name(expert.weight) for expert in local_experts]

        assert names == [
            f'decoder.layers.0.mlp.experts.local_experts.{NUM_LOCAL_EXPERTS * rank + i}.weight'
            for i in range(NUM_LOCAL_EXPERTS)
        ]
