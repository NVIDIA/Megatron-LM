# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Megatron-FSDP v1 rejects GTP weight rematerialization when it is constructed."""

import contextlib
import re
import sys
import types
from unittest import mock

import pytest
import torch
import torch.distributed as dist

from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.distributed.fsdp.mcore_fsdp_adapter import FullyShardedDataParallel
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer import TransformerConfig
from megatron.core.utils import is_torch_min_version
from tests.unit_tests.test_utilities import Utils
from tools import check_process_group_usage

pytestmark = pytest.mark.skipif(
    not is_torch_min_version("2.4.0"), reason="Megatron-FSDP requires torch >= 2.4.0"
)


def _skip_unless_world_size_divisible_by(size):
    if Utils.world_size % size != 0:
        pytest.skip(f"needs a world size divisible by {size}, got {Utils.world_size}")


@contextlib.contextmanager
def _forbid_global_process_groups():
    """Make every parallel_state accessor that the process-group checker counts raise.

    The accessors are patched on parallel_state and wherever an imported megatron module bound
    them by name, and ProcessGroupCollection.use_mpu_process_groups() raises too.
    """
    accessors = {
        name: value
        for name, value in vars(parallel_state).items()
        if callable(value) and check_process_group_usage._is_deprecated_accessor(name)
    }

    def forbidden(name):
        def read_global_grid(*args, **kwargs):
            raise AssertionError(f"parallel_state.{name}() reads the global parallel grid")

        return read_global_grid

    stubs = {id(value): forbidden(name) for name, value in accessors.items()}
    with contextlib.ExitStack() as patches:
        for module_name, module in list(sys.modules.items()):
            if module_name.split(".")[0] != "megatron" or not isinstance(module, types.ModuleType):
                continue
            for attr, value in list(vars(module).items()):
                if id(value) in stubs:
                    patches.enter_context(mock.patch.object(module, attr, stubs[id(value)]))
        patches.enter_context(
            mock.patch.object(
                ProcessGroupCollection,
                "use_mpu_process_groups",
                side_effect=AssertionError("use_mpu_process_groups() reads the global grid"),
            )
        )
        yield


def _group(ranks_per_group):
    """Return this rank's group from a partition of all ranks.

    Each call creates new communicators, distinct from the global grid's groups. Creating groups
    is collective: call on every rank.
    """
    group, _ = dist.new_subgroups_by_enumeration(ranks_per_group)
    return group


def _collection(gtp_remat_size=1, expt_gtp_remat_size=1):
    """Collection with TP=1, dp_cp over every rank, and GTP groups of the given sizes."""
    world_size = dist.get_world_size()

    def blocks(size):
        return [list(range(start, start + size)) for start in range(0, world_size, size)]

    return ProcessGroupCollection(
        tp=_group(blocks(1)),
        expt_tp=_group(blocks(1)),
        dp_cp=_group(blocks(world_size)),
        gtp_remat=_group(blocks(gtp_remat_size)),
        expt_gtp_remat=_group(blocks(expt_gtp_remat_size)),
    )


def _wrap(config, pg_collection=None):
    """Wrap a small dense model in Megatron-FSDP v1."""
    return FullyShardedDataParallel(
        config=config,
        ddp_config=DistributedDataParallelConfig(
            data_parallel_sharding_strategy="optim_grads_params",
            overlap_grad_reduce=True,
            overlap_param_gather=True,
            bucket_size=10000,
            use_megatron_fsdp=True,
        ),
        module=torch.nn.Sequential(
            torch.nn.Linear(8, 32), torch.nn.ReLU(), torch.nn.Linear(32, 8)
        ).cuda(),
        fsdp_unit_modules=[torch.nn.Linear],
        pg_collection=pg_collection,
    )


class TestMegatronFSDPV1GTPWeightRemat:
    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize(
        "config_kwargs, grid_kwargs, reported",
        [
            (
                {"tensor_parallel_num_weight_shards": 2},
                {"gtp_remat_size": 2},
                "gtp_weight_remat_size=2",
            ),
            (
                {"num_moe_experts": 4, "expert_tensor_parallel_num_weight_shards": 2},
                {"expert_gtp_remat_size": 2},
                "expert_gtp_weight_remat_size=2",
            ),
        ],
        ids=["dense", "expert"],
    )
    def test_config_gtp_rejected_without_collection(self, config_kwargs, grid_kwargs, reported):
        """GTP in the config rejects the model on a global grid that runs GTP."""
        _skip_unless_world_size_divisible_by(2)
        Utils.initialize_model_parallel(**grid_kwargs)
        config = TransformerConfig(num_attention_heads=1, num_layers=1, **config_kwargs)

        with pytest.raises(ValueError, match=re.escape(f"got {reported}")):
            _wrap(config)

    @pytest.mark.parametrize("field", ["gtp_remat", "expt_gtp_remat"])
    def test_collection_gtp_rejected_without_global_reads(self, field):
        """A GTP group of two ranks in the collection rejects the model; the config has no GTP."""
        _skip_unless_world_size_divisible_by(2)
        Utils.initialize_model_parallel()
        pg_collection = _collection(**{f"{field}_size": 2})
        config = TransformerConfig(num_attention_heads=1, num_layers=1)

        with (
            _forbid_global_process_groups(),
            pytest.raises(ValueError, match=re.escape(f"got pg_collection.{field}=2")),
        ):
            _wrap(config, pg_collection)

    def test_single_rank_gtp_groups_accepted(self):
        """GTP groups of one rank each mean GTP is off, so the collection's groups are used."""
        _skip_unless_world_size_divisible_by(2)
        # The global grid runs TP=2, unlike the collection.
        Utils.initialize_model_parallel(tensor_model_parallel_size=2)
        pg_collection = _collection()
        config = TransformerConfig(num_attention_heads=1, num_layers=1)

        with _forbid_global_process_groups():
            fsdp = _wrap(config, pg_collection)

        assert dist.get_process_group_ranks(fsdp.megatron_fsdp_dist_index.get_dp_group()) == list(
            range(dist.get_world_size())
        )
        assert fsdp.tp_group.size() == 1
