# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""MoE experts fill sharded-state metadata from their own collection.

Called without metadata, the experts' ``sharded_state_dict`` filled in the global
``get_data_parallel_group(with_context_parallel=True)``. The experts now fill in their
collection's dense DP x CP group, the group a model passes down, and still take the DP entry
of each replica id from the expert data-parallel group.
"""

import pytest
import torch

from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_submodules,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.moe.experts import SequentialMLP, TEGroupedMLP
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.spec_utils import get_submodules
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import is_te_min_version
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.test_sharded_state_dict_dp_cp_group import (
    forbid_global_dp_cp_group,
    shard_ids,
)

EXPERT_PARALLEL_SIZE = 2
NUM_EXPERTS = 4


@pytest.mark.skipif(
    not is_te_min_version("1.7.0"),
    reason="Transformer Engine under v1.7.0 doesn't support MoE training.",
)
class TestExpertsShardedStateDictDpCpGroup:

    def setup_method(self, method):
        if Utils.world_size % (2 * EXPERT_PARALLEL_SIZE) != 0:
            pytest.skip("needs EP=2 with at least two expert data-parallel replicas")
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, expert_model_parallel_size=EXPERT_PARALLEL_SIZE
        )
        model_parallel_cuda_manual_seed(123)
        self.pg_collection = ProcessGroupCollection.use_mpu_process_groups()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize(
        "moe_grouped_gemm, experts_class",
        [(True, TEGroupedMLP), (False, SequentialMLP)],
        ids=["TEGroupedMLP", "SequentialMLP"],
    )
    def test_experts_without_metadata(self, moe_grouped_gemm, experts_class):
        """Same ids as with the model's group, with no global read, over the expert DP domain."""
        config = TransformerConfig(
            num_layers=1,
            hidden_size=16,
            num_attention_heads=4,
            num_moe_experts=NUM_EXPERTS,
            moe_router_topk=2,
            moe_grouped_gemm=moe_grouped_gemm,
            expert_model_parallel_size=EXPERT_PARALLEL_SIZE,
            add_bias_linear=False,
            bf16=True,
            params_dtype=torch.bfloat16,
        )
        submodules = get_submodules(
            get_gpt_layer_with_transformer_engine_submodules(
                NUM_EXPERTS, moe_grouped_gemm=moe_grouped_gemm
            ).mlp
        )
        experts = MoELayer(config, submodules, pg_collection=self.pg_collection).experts
        assert isinstance(experts, experts_class)

        own_metadata = {'dp_cp_group': self.pg_collection.dp_cp_gtp_remat}
        expected = shard_ids(experts.sharded_state_dict(prefix='experts.', metadata=own_metadata))

        with forbid_global_dp_cp_group():
            sharded_state_dict = experts.sharded_state_dict(prefix='experts.')

        assert shard_ids(sharded_state_dict) == expected
        # Experts are replicated over the expert data-parallel group, not the dense group the
        # metadata carries; on some ranks the two give a different DP entry.
        expert_dp_rank = self.pg_collection.expt_dp.rank()
        assert {replica_id[2] for _, _, replica_id in expected} == {expert_dp_rank}
