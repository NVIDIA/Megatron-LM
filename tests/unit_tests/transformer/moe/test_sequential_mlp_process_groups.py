# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""SequentialMLP builds its experts from the process groups of the collection it is given.

Each local expert is an MLP whose linears resolve their expert GTP-remat group and stamp a
checkpoint replica group on their GTP shards. Both must come from SequentialMLP's collection:
a model on its own grid must not read the global parallel grid.
"""

import contextlib
import sys
import types
from unittest import mock

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.extensions.transformer_engine import TEColumnParallelLinear, TERowParallelLinear
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.gtp_api import HAVE_GTP
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.mlp import MLPSubmodules
from megatron.core.transformer.moe.experts import SequentialMLP
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils
from tools import check_process_group_usage

LINEAR_SPECS = {
    "local": MLPSubmodules(linear_fc1=ColumnParallelLinear, linear_fc2=RowParallelLinear),
    "te": MLPSubmodules(linear_fc1=TEColumnParallelLinear, linear_fc2=TERowParallelLinear),
}


@contextlib.contextmanager
def forbid_global_grid_reads():
    """Make every read of the global parallel grid raise AssertionError inside the block.

    Covers each parallel_state accessor that tools/check_process_group_usage.py counts and the
    registry getter, on parallel_state and in every imported megatron module that bound it by
    name, and the ProcessGroupCollection.use_mpu_process_groups() shim that resolvers fall back
    to.
    """
    modules = [
        module
        for name, module in list(sys.modules.items())
        if name.split(".")[0] == "megatron" and isinstance(module, types.ModuleType)
    ]
    with contextlib.ExitStack() as stack:
        for name, original in list(vars(parallel_state).items()):
            if not (callable(original) and check_process_group_usage._returns_global_groups(name)):
                continue
            stub = mock.Mock(
                side_effect=AssertionError(f"parallel_state.{name}() reads the global grid")
            )
            for module in modules:
                if vars(module).get(name) is original:
                    stack.enter_context(mock.patch.object(module, name, stub))
        shim = mock.Mock(
            side_effect=AssertionError(
                "ProcessGroupCollection.use_mpu_process_groups() reads the global grid"
            )
        )
        stack.enter_context(
            mock.patch.object(ProcessGroupCollection, "use_mpu_process_groups", shim)
        )
        yield


def same_ranks_group(group):
    """Return a distinct communicator over the ranks of `group`.

    Every rank creates every group of the partition `group` belongs to, in one order, so the
    result is never the global group object while holding the same ranks.
    """
    ranks_by_rank = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(
        ranks_by_rank, torch.distributed.get_process_group_ranks(group)
    )
    partition = sorted({tuple(sorted(ranks)) for ranks in ranks_by_rank})
    subgroup, _ = torch.distributed.new_subgroups_by_enumeration(
        [list(ranks) for ranks in partition]
    )
    return subgroup


def make_config(num_moe_experts, **kwargs):
    """A small MoE config whose expert dims divide evenly over two GTP-remat shards."""
    return TransformerConfig(
        num_layers=1,
        hidden_size=128,
        ffn_hidden_size=256,
        moe_ffn_hidden_size=256,
        num_attention_heads=4,
        num_moe_experts=num_moe_experts,
        add_bias_linear=False,
        params_dtype=torch.bfloat16,
        use_cpu_initialization=False,
        **kwargs,
    )


class TestSequentialMLPProcessGroups:

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("linear_impl", ["local", "te"])
    def test_experts_are_built_without_reading_the_global_grid(self, linear_impl):
        """With a full collection, building the experts reads no global group."""
        if Utils.world_size < 2 or Utils.world_size % 2:
            pytest.skip("Needs an even number of ranks, at least 2")
        Utils.initialize_model_parallel(1, 1, expert_model_parallel_size=2)
        model_parallel_cuda_manual_seed(123)
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        config = make_config(num_moe_experts=4, expert_model_parallel_size=2)

        with forbid_global_grid_reads():
            experts = SequentialMLP(
                2, config, LINEAR_SPECS[linear_impl], pg_collection=pg_collection
            )

        assert len(experts.local_experts) == 2
        for expert in experts.local_experts:
            assert expert.tp_group is pg_collection.expt_tp

    @pytest.mark.skipif(not HAVE_GTP, reason="GTP requires Transformer Engine >= 2.19")
    @pytest.mark.parametrize("linear_impl", ["local", "te"])
    def test_experts_use_the_collection_expert_gtp_and_replica_groups(self, linear_impl):
        """Expert GTP shards use the collection's expt_gtp_remat and stamp its expt_dp."""
        from megatron.core.tensor_parallel.generalized_tensor_parallelism import (
            _GTP_PARAMS,
            is_gtp_param,
            reset_gtp_state,
        )

        if Utils.world_size < 2 or Utils.world_size % 2:
            pytest.skip("Needs an even number of ranks, at least 2")
        Utils.initialize_model_parallel(1, 1, expert_gtp_remat_size=2)
        model_parallel_cuda_manual_seed(123)
        # Same ranks as the global groups, different communicators: a module that reads the
        # global grid ends up with a group that is not the collection's.
        expt_gtp_remat = same_ranks_group(parallel_state.get_expert_gtp_weight_remat_group())
        expt_dp = same_ranks_group(
            parallel_state.get_expert_data_parallel_group(with_gtp_remat=False)
        )
        pg_collection = ProcessGroupCollection(
            ep=parallel_state.get_expert_model_parallel_group(),
            expt_tp=parallel_state.get_expert_tensor_parallel_group(),
            expt_dp=expt_dp,
            expt_gtp_remat=expt_gtp_remat,
        )
        metadata = {
            "dp_cp_group": parallel_state.get_data_parallel_group(with_context_parallel=True)
        }
        config = make_config(num_moe_experts=2, expert_tensor_parallel_num_weight_shards=2)

        try:
            with forbid_global_grid_reads():
                experts = SequentialMLP(
                    2, config, LINEAR_SPECS[linear_impl], pg_collection=pg_collection
                )
                sharded_state_dict = experts.sharded_state_dict(metadata=metadata)

            for expert in experts.local_experts:
                for linear in (expert.linear_fc1, expert.linear_fc2):
                    weight = linear.weight
                    assert is_gtp_param(weight), f"{linear_impl} expert weight is not GTP-sharded"
                    assert (
                        weight.group is expt_gtp_remat
                    ), "the expert weight is sharded over the global expert GTP-remat group"
                    assert (
                        getattr(weight, "gtp_replica_group", None) is expt_dp
                    ), "the expert weight does not carry the collection's expt_dp replica group"
            assert sharded_state_dict
            for key, sharded in sharded_state_dict.items():
                assert sharded.replica_id[2] == expt_dp.rank(), key
        finally:
            _GTP_PARAMS.clear()
            reset_gtp_state()
