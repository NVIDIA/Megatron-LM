# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""The training loop hands its process-group collection to the Megatron Core entry points.

A group that the training loop reads from the global grid instead of the collection is invisible
when the collection holds the global groups themselves. These tests therefore give the
collection distinct communicators over the same ranks, or a layout that differs from the global
grid, so that a global read changes which group is used.
"""

from types import SimpleNamespace

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.pipeline_parallel.schedules import (
    forward_backward_no_pipelining,
    forward_backward_pipelining_with_interleaving,
    forward_backward_pipelining_without_interleaving,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.module import MegatronModule
from megatron.training import global_vars, training
from megatron.training.arguments import parse_args
from megatron.training.models.dist_utils import _ddp_wrap
from tests.unit_tests.dist_checkpointing.utils import init_basic_mock_args
from tests.unit_tests.test_utilities import Utils

pytestmark = pytest.mark.skipif(
    Utils.world_size < 4 or Utils.world_size % 2 != 0, reason="Requires an even world of 4+ ranks"
)


class _TinyModel(MegatronModule):
    def __init__(self, config):
        super().__init__(config)
        self.linear = torch.nn.Linear(16, 16)


def _config():
    return TransformerConfig(num_layers=1, hidden_size=16, num_attention_heads=1, bf16=True)


def _same_ranks(group):
    """A new communicator over the ranks of ``group``."""
    return torch.distributed.new_group(ranks=torch.distributed.get_process_group_ranks(group))


def _ranks(group):
    return torch.distributed.get_process_group_ranks(group)


class TestTrainingPgCollection:

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("create_all_gather_group", [False, True])
    def test_collection_carries_all_gather_groups_when_requested(self, create_all_gather_group):
        Utils.initialize_model_parallel(expert_model_parallel_size=2)
        args = SimpleNamespace(
            create_all_gather_group=create_all_gather_group,
            distributed_timeout_minutes=None,
            expert_model_parallel_size=2,
        )

        pg_collection = training._build_pg_collection_from_parallel_state(args)

        if not create_all_gather_group:
            assert pg_collection.dp_cp_ag is None and pg_collection.expt_dp_ag is None
            return
        # Separate communicators over the data-parallel ranks that Megatron-FSDP gathers over.
        assert pg_collection.dp_cp_ag is not pg_collection.dp_cp
        assert _ranks(pg_collection.dp_cp_ag) == _ranks(pg_collection.dp_cp)
        assert pg_collection.expt_dp_ag is not pg_collection.expt_dp
        assert _ranks(pg_collection.expt_dp_ag) == _ranks(pg_collection.expt_dp)

    def test_all_gather_groups_must_span_the_data_parallel_ranks(self):
        # Pipeline-before-data order: the data-parallel groups are not the default order's.
        Utils.initialize_model_parallel(pipeline_model_parallel_size=2, order="tp-cp-ep-pp-dp")
        args = SimpleNamespace(
            create_all_gather_group=True,
            distributed_timeout_minutes=None,
            expert_model_parallel_size=1,
        )

        with pytest.raises(ValueError, match="all-gather groups whose ranks differ"):
            training._build_pg_collection_from_parallel_state(args)

    @pytest.mark.usefixtures("run_config")
    def test_get_model_wraps_ddp_with_the_given_collection(self, monkeypatch):
        Utils.initialize_model_parallel()
        args = init_basic_mock_args(parse_args(ignore_unknown_args=True), tp=1, pp=1, bf16=True)
        monkeypatch.setattr(global_vars, "_GLOBAL_ARGS", args)
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        pg_collection.dp = _same_ranks(pg_collection.dp)
        pg_collection.dp_cp = _same_ranks(pg_collection.dp_cp)

        def model_provider(pre_process, post_process, config, pg_collection):
            return _TinyModel(config)

        model = training.get_model(model_provider, config=_config(), pg_collection=pg_collection)

        assert model[0].dp_group is pg_collection.dp
        assert model[0].dp_cp_group is pg_collection.dp_cp
        assert model[0].intra_dp_cp_group is pg_collection.dp_cp

    @pytest.mark.skipif(not training.HAVE_FSDP2, reason="Requires Torch FSDP2")
    @pytest.mark.parametrize("has_gtp_remat_fields", [True, False])
    def test_torch_fsdp2_shards_over_the_collection_dp_cp_group(self, has_gtp_remat_fields):
        Utils.initialize_model_parallel()
        dp_cp_group = _same_ranks(
            parallel_state.get_data_parallel_group(with_context_parallel=True)
        )
        if has_gtp_remat_fields:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups()
            pg_collection.dp_cp_gtp_remat = dp_cp_group
        else:
            pg_collection = ProcessGroupCollection(dp_cp=dp_cp_group)
        ddp_config = DistributedDataParallelConfig()

        # get_model's wrapper and the model builders' wrapper.
        wrapped = training.wrap_model_chunks_with_ddp(
            [_TinyModel(_config()).cuda()],
            _config(),
            ddp_config,
            DP=training.torch_FSDP,
            pg_collection=pg_collection,
        )
        built = _ddp_wrap(
            [_TinyModel(_config()).cuda()],
            data_parallel_random_init=False,
            ddp_config=ddp_config,
            overlap_param_gather_with_optimizer_step=False,
            use_torch_fsdp2=True,
            pg_collection=pg_collection,
        )

        assert wrapped[0].process_group is dp_cp_group
        assert built[0].process_group is dp_cp_group

    def test_schedule_follows_the_collection_pipeline_group(self):
        # The global grid has no pipeline parallelism; the collection has two stages.
        Utils.initialize_model_parallel()
        pp_group, _ = torch.distributed.new_subgroups_by_enumeration(
            [[rank, rank + 1] for rank in range(0, Utils.world_size, 2)]
        )
        pg_collection = ProcessGroupCollection(pp=pp_group)

        def schedule(vp_size):
            config = SimpleNamespace(virtual_pipeline_model_parallel_size=vp_size)
            return training._get_forward_backward_func(pg_collection, config)

        assert schedule(None) is forward_backward_pipelining_without_interleaving
        assert schedule(2) is forward_backward_pipelining_with_interleaving
        # Without a collection the global grid decides.
        assert training._get_forward_backward_func(None, None) is forward_backward_no_pipelining
