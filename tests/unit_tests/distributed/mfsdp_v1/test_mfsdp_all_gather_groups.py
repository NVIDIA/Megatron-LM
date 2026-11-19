# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Independent AG communicators must follow the FSDP shard layout."""

from contextlib import contextmanager

import pytest
import torch
import torch.distributed as dist

from megatron.core import parallel_state as ps
from megatron.core.distributed.distributed_data_parallel_config import DistributedDataParallelConfig
from megatron.core.distributed.fsdp.mcore_adapter_v1 import FullyShardedDataParallelV1
from megatron.core.distributed.fsdp.src.megatron_fsdp.fully_shard import fully_shard_optimizer
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.distributed.mfsdp_v1.test_mfsdp_expert_gradient_scaling import (
    ToyExpertParallelModel,
    assert_loss_curve_matches_reference,
)
from tests.unit_tests.distributed.mfsdp_v1.test_mfsdp_nonuniform_sharding import destroy_device_mesh
from tests.unit_tests.test_utilities import Utils


@contextmanager
def initialized_groups(**kwargs):
    Utils.initialize_distributed()
    previous = set(dist.distributed_c10d._world.pg_map)
    pg = Utils.initialize_model_parallel(
        create_gloo_process_groups=False, distributed_timeout_minutes=2, **kwargs
    )
    try:
        yield pg
    except BaseException as exc:
        print(f"AG_CASE_ERROR rank={dist.get_rank()}: {exc!r}", flush=True)
        raise
    finally:
        torch.cuda.synchronize()
        dist.barrier()
        groups = [g for g in dist.distributed_c10d._world.pg_map if g not in previous]
        Utils.destroy_model_parallel()
        for group in reversed(groups):
            if group in dist.distributed_c10d._world.pg_map:
                dist.destroy_process_group(group)
        destroy_device_mesh(None)


def assert_matching_communicators(ag, shard):
    assert ag is not shard
    ranks = dist.get_process_group_ranks(shard)
    assert dist.get_process_group_ranks(ag) == ranks
    # Exercise concurrent AG/RS, including singleton groups, using distinct payloads.
    value = torch.full((1024,), float(dist.get_rank() + 1), device="cuda")
    gathered = torch.empty(len(ranks) * value.numel(), device="cuda")
    reduced = torch.empty_like(value)
    gather = dist.all_gather_into_tensor(gathered, value, group=ag, async_op=True)
    scatter = dist.reduce_scatter_tensor(
        reduced, value.repeat(len(ranks)), group=shard, async_op=True
    )
    gather.wait()
    scatter.wait()
    torch.testing.assert_close(
        gathered, torch.cat([torch.full_like(value, float(r + 1)) for r in ranks])
    )
    torch.testing.assert_close(reduced, torch.full_like(value, float(sum(r + 1 for r in ranks))))


@pytest.mark.parametrize(
    "topology",
    [
        pytest.param({}, id="dense-fsdp"),
        pytest.param(dict(num_distributed_optimizer_instances=2), id="dense-hsdp"),
        pytest.param(dict(num_distributed_optimizer_instances=4), id="dense-singleton"),
        pytest.param(dict(expert_model_parallel_size=2), id="expert-fsdp"),
        pytest.param(
            dict(expert_model_parallel_size=2, num_distributed_optimizer_instances=2),
            id="expert-hsdp",
        ),
        pytest.param(
            dict(
                expert_model_parallel_size=2,
                context_parallel_size=2,
                num_distributed_optimizer_instances=2,
            ),
            id="expert-cp",
        ),
        pytest.param(
            dict(
                expert_model_parallel_size=2,
                tensor_model_parallel_size=2,
                num_distributed_optimizer_instances=2,
            ),
            id="expert-tp",
        ),
        pytest.param(
            dict(
                expert_model_parallel_size=2,
                pipeline_model_parallel_size=2,
                num_distributed_optimizer_instances=2,
            ),
            id="expert-pp",
        ),
        pytest.param(dict(gtp_remat_size=2), id="gtp"),
        pytest.param(dict(context_parallel_size=2, gtp_remat_size=2), id="cp-gtp"),
        pytest.param(dict(context_parallel_size=2, num_distributed_optimizer_instances=2), id="cp"),
        pytest.param(
            dict(tensor_model_parallel_size=2, num_distributed_optimizer_instances=2), id="tp"
        ),
        pytest.param(
            dict(pipeline_model_parallel_size=2, num_distributed_optimizer_instances=2), id="pp"
        ),
    ],
)
def test_groups_match_inner_shards(topology):
    Utils.initialize_distributed()
    expert_dp_size = dist.get_world_size() // (
        topology.get("expert_model_parallel_size", 1)
        * topology.get("tensor_model_parallel_size", 1)
        * topology.get("pipeline_model_parallel_size", 1)
    )
    if expert_dp_size % topology.get("num_distributed_optimizer_instances", 1):
        pytest.skip("Topology requires a larger world size.")
    with initialized_groups(create_all_gather_group=True, **topology) as pg:
        assert_matching_communicators(pg.dp_cp_ag, pg.intra_dp_cp)
        if topology.get("expert_model_parallel_size", 1) > 1:
            assert_matching_communicators(pg.expt_dp_ag, pg.intra_expt_dp)
        else:
            assert pg.expt_dp_ag is None
        # The standalone helper follows the same layout and creates fresh groups.
        dense, expert = ps.create_all_gather_groups(
            for_expert_parallelism=topology.get("expert_model_parallel_size", 1) > 1
        )
        assert dense is not pg.dp_cp_ag
        assert_matching_communicators(dense, pg.intra_dp_cp)
        if expert is not None:
            assert expert is not pg.expt_dp_ag
            assert_matching_communicators(expert, pg.intra_expt_dp)


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("cp", [1, 2])
def test_gtp_hsdp_remains_unsupported(enabled, cp):
    Utils.initialize_distributed()
    with pytest.raises(AssertionError, match="GTP_remat with num_distributed_optimizer_instances"):
        Utils.initialize_model_parallel(
            gtp_remat_size=2,
            context_parallel_size=cp,
            num_distributed_optimizer_instances=2,
            create_all_gather_group=enabled,
            create_gloo_process_groups=False,
        )
    Utils.destroy_model_parallel()


@pytest.mark.parametrize("enabled", [False, True], ids=["ag-off", "ag-on"])
@pytest.mark.parametrize("expert", [False, True], ids=["dense", "expert"])
@pytest.mark.parametrize("instances", [1, 2, 4])
def test_training_matches_reference(enabled, expert, instances):
    Utils.initialize_distributed()
    if dist.get_world_size() // (2 if expert else 1) % instances:
        pytest.skip("Optimizer instances must divide expert DP size.")
    with initialized_groups(
        create_all_gather_group=enabled,
        expert_model_parallel_size=2 if expert else 1,
        num_distributed_optimizer_instances=instances,
    ) as pg:
        assert (pg.dp_cp_ag is not None) == enabled
        assert (pg.expt_dp_ag is not None) == (enabled and expert)
        assert_training_matches_reference(
            pg, expert, enabled, instances, "optim_grads_params", "no_shard", False
        )


@pytest.mark.parametrize("expert", [False, True], ids=["dense", "expert"])
@pytest.mark.parametrize("layout", ["replica", "wrong-members"])
def test_basic_hsdp_rejects_mismatched_custom_ag(expert, layout):
    with initialized_groups(
        create_all_gather_group=True,
        expert_model_parallel_size=2,
        num_distributed_optimizer_instances=2,
    ) as pg:
        if expert:
            pg.expt_dp_ag = pg.expt_dp if layout == "replica" else pg.ep
        else:
            pg.dp_cp_ag = pg.dp_cp if layout == "replica" else pg.expt_dp
        with pytest.raises(ValueError, match="match the inner shard group's ordered ranks"):
            assert_training_matches_reference(
                pg, True, True, 2, "optim_grads_params", "no_shard", False
            )


def test_outer_sharding_ignores_unused_custom_ag():
    with initialized_groups(
        create_all_gather_group=True,
        expert_model_parallel_size=2,
        num_distributed_optimizer_instances=2,
    ) as pg:
        pg.dp_cp_ag, pg.expt_dp_ag = pg.dp_cp, pg.expt_dp
        assert_training_matches_reference(pg, True, True, 2, "optim_grads_params", "optim", False)


def test_basic_hsdp_ignores_unused_expert_ag():
    with initialized_groups(
        create_all_gather_group=True,
        expert_model_parallel_size=2,
        num_distributed_optimizer_instances=2,
    ) as pg:
        # A collection may include expert handles while this model is dense-only.
        pg.expt_dp_ag = pg.ep
        assert_training_matches_reference(
            pg, False, True, 2, "optim_grads_params", "no_shard", False
        )


def assert_training_matches_reference(
    pg, expert, enabled, instances, strategy, outer_strategy, average
):
    torch.manual_seed(1234)
    factory = (
        ToyExpertParallelModel
        if expert
        else lambda: torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Linear(8, 8))
    )
    reference = factory().cuda()
    model = factory().cuda()
    model.load_state_dict(reference.state_dict())
    config = TransformerConfig(
        num_layers=1,
        hidden_size=8,
        num_attention_heads=1,
        use_cpu_initialization=True,
        gradient_accumulation_fusion=False,
        num_moe_experts=2 if expert else None,
    )
    ddp = DistributedDataParallelConfig(
        data_parallel_sharding_strategy=strategy,
        outer_dp_sharding_strategy=outer_strategy,
        num_distributed_optimizer_instances=instances,
        average_in_collective=average,
    )
    wrapped = FullyShardedDataParallelV1(
        config, ddp, model, fsdp_unit_modules=[torch.nn.Linear], pg_collection=pg
    )
    # This isolated loop has one microbatch per step; the LM driver normally
    # controls the last-microbatch synchronization boundary.
    wrapped.module.is_last_microbatch = True
    for group in wrapped.param_and_grad_buffer.parameter_groups:
        # HFSDP uses helper buffers rather than the independent AG communicator.
        if enabled and strategy == "optim_grads_params" and outer_strategy == "no_shard":
            expected = pg.expt_dp_ag if group.is_expert_param and expert else pg.dp_cp_ag
            assert group.model_weight_buffer.data_parallel_group is expected
        if outer_strategy == "no_shard" and strategy == "optim_grads_params":
            assert group.model_weight_buffer.data_size == group.main_weight_buffer.data_size
    optimizer = fully_shard_optimizer(torch.optim.SGD(wrapped.module.parameters(), lr=0.1))
    error = None
    try:
        assert_loss_curve_matches_reference(wrapped.module, optimizer, reference, pg.expt_dp)
    except AssertionError as exc:
        error = str(exc)
    # A local numerical failure must stop every rank before further collectives.
    errors = [None] * dist.get_world_size()
    dist.all_gather_object(errors, error)
    assert not any(errors), ("Losses differ from reference", errors)
    # Check every updated local shard against the independently trained full model.
    # Reading the persistent weight buffers avoids DTensor checkpoint/reconstruction
    # caches retaining a mesh from an earlier, already destroyed test grid.
    error = None
    try:
        reference_params = dict(reference.named_parameters())
        buffers = wrapped.param_and_grad_buffer
        for group in buffers.parameter_groups:
            buffer = group.main_weight_buffer
            for item_id, param in enumerate(group.params):
                expected = reference_params[buffers.param_to_name[param]].detach().flatten()
                if buffer is None:
                    actual = param.detach().flatten()
                else:
                    actual = buffer.get_item(item_id)
                    if buffer.is_data_distributed:
                        start, end = buffer._get_item_slice_in_shard(item_id)
                        expected = expected[start:end]
                torch.testing.assert_close(actual, expected)
    except AssertionError as exc:
        error = str(exc)
    errors = [None] * dist.get_world_size()
    dist.all_gather_object(errors, error)
    assert not any(errors), ("Optimized shards differ from reference", errors)
