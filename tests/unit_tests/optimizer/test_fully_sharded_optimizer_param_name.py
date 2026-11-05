# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""The MFSDP v2 optimizer names experts by its model's own expert-parallel layout."""

from types import SimpleNamespace

import pytest
import torch

from megatron.core.optimizer import OptimizerConfig
from megatron.core.optimizer.fully_sharded_optimizer import FullyShardedOptimizer
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.fsdp_dtensor_checkpoint import get_global_unique_param_name
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.test_fsdp_dtensor_expert_offsets import (
    expert_model,
    forbid_global_expert_parallel_reads,
)

NUM_LOCAL_EXPERTS = 2


def build_optimizer(model):
    """Build the MFSDP v2 optimizer for ``model`` behind a wrapper like the MFSDP v2 one.

    The wrapper has no ``pg_collection`` of its own, so the optimizer has to find the
    model's collection through ``module``.
    """
    model_chunk = torch.nn.Module()
    model_chunk.module = model
    model_chunk.config = model.config
    model_chunk.ddp_config = SimpleNamespace()
    return FullyShardedOptimizer(
        torch.optim.SGD(model_chunk.parameters(), lr=1.0),
        OptimizerConfig(lr=1.0, use_distributed_optimizer=False),
        grad_scaler=None,
        init_state_fn=None,
        model_chunks=[model_chunk],
    )


class TestFullyShardedOptimizerExpertNames:

    def setup_method(self, method):
        if Utils.world_size < 4 or Utils.world_size % 2 != 0:
            pytest.skip("needs a global EP=2 grid that differs from an EP=world grid")
        # The global grid pairs ranks along EP.
        Utils.initialize_model_parallel(expert_model_parallel_size=2)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_names_follow_the_model_expert_layout(self):
        world_size, rank = Utils.world_size, torch.distributed.get_rank()
        model = expert_model(NUM_LOCAL_EXPERTS, num_moe_experts=NUM_LOCAL_EXPERTS * world_size)
        # The model's own EP group spans every rank.
        model.pg_collection = ProcessGroupCollection(
            ep=torch.distributed.new_group(ranks=list(range(world_size)))
        )
        local_experts = model.decoder.layers[0].mlp.experts.local_experts

        with forbid_global_expert_parallel_reads():
            optimizer = build_optimizer(model)

        offset = NUM_LOCAL_EXPERTS * rank
        assert [optimizer._param_to_fqn[expert.weight] for expert in local_experts] == [
            f'module.decoder.layers.0.mlp.experts.local_experts.{offset + i}.weight'
            for i in range(NUM_LOCAL_EXPERTS)
        ]

    def test_names_on_the_standard_grid_match_the_global_layout(self):
        model = expert_model(NUM_LOCAL_EXPERTS, num_moe_experts=NUM_LOCAL_EXPERTS * 2)
        model.pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['ep'])

        optimizer = build_optimizer(model)

        for param, name in optimizer._param_to_fqn.items():
            assert name == get_global_unique_param_name(optimizer.model_chunks, param)

    def test_dense_model_needs_no_collection(self):
        model = torch.nn.Linear(2, 2, bias=False)
        model.config = SimpleNamespace(num_moe_experts=None)

        optimizer = build_optimizer(model)

        assert optimizer._param_to_fqn == {model.weight: 'module.weight'}
