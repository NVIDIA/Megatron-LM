# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from math import log2

import pytest
import torch

import megatron.core.parallel_state as ps
from megatron.core.process_groups_config import ProcessGroupCollection
from tests.unit_tests.test_utilities import Utils

rank = Utils.rank
world_size = Utils.world_size
test_parallel_order = ['tp-cp-ep-dp-pp', 'tp-cp-pp-ep-dp']


@pytest.mark.parametrize('order', test_parallel_order)
@pytest.mark.flaky
@pytest.mark.flaky_in_dev
def test_initialize_and_destroy_model_parallel(order):
    with pytest.raises(AssertionError):
        assert ps.initialize_model_parallel(order=order)
    Utils.initialize_distributed()
    with pytest.raises(RuntimeError):
        assert ps.initialize_model_parallel(tensor_model_parallel_size=2 * world_size, order=order)
    with pytest.raises(RuntimeError):
        assert ps.initialize_model_parallel(
            pipeline_model_parallel_size=2 * world_size, order=order
        )
    with pytest.raises(RuntimeError):
        assert ps.initialize_model_parallel(
            pipeline_model_parallel_size=world_size,
            tensor_model_parallel_size=world_size,
            order=order,
        )
    with pytest.raises(RuntimeError):
        assert ps.initialize_model_parallel(virtual_pipeline_model_parallel_size=2, order=order)
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=2, pipeline_model_parallel_size=4, order=order
    )

    assert ps.model_parallel_is_initialized()
    assert ps.get_model_parallel_group() is not None
    assert ps.get_tensor_model_parallel_group() is not None
    assert ps.get_pipeline_model_parallel_group() is not None
    assert ps.get_data_parallel_group() is not None
    assert ps.get_expert_model_parallel_group() is not None
    assert ps.get_expert_tensor_parallel_group() is not None
    assert ps.get_expert_data_parallel_group() is not None
    assert ps.get_expert_tensor_model_pipeline_parallel_group() is not None
    Utils.destroy_model_parallel()
    assert ps._MODEL_PARALLEL_GROUP is None


@pytest.mark.parametrize('order', test_parallel_order)
def test_pipeline_parallel_initializations(order):
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=2, pipeline_model_parallel_size=4, order=order
    )
    assert ps.get_pipeline_model_parallel_first_rank() == rank % 2
    assert ps.get_data_parallel_src_rank() == rank
    assert ps.get_pipeline_model_parallel_next_rank() == ((rank + 2) % world_size)
    assert ps.get_pipeline_model_parallel_prev_rank() == ((rank - 2) % world_size)
    Utils.destroy_model_parallel()


@pytest.mark.parametrize('order', test_parallel_order)
def test_data_parallel_initializations(order):
    Utils.initialize_model_parallel(pipeline_model_parallel_size=world_size, order=order)
    assert ps.get_data_parallel_src_rank() == rank
    assert ps.get_data_parallel_world_size() == 1
    assert ps.get_data_parallel_rank() == 0
    Utils.destroy_model_parallel()


@pytest.mark.parametrize('order', test_parallel_order)
def test_tensor_model_parellel_world_size(order):
    Utils.initialize_model_parallel(tensor_model_parallel_size=world_size, order=order)
    assert ps.get_tensor_model_parallel_world_size() == world_size
    ps.set_tensor_model_parallel_world_size(None)
    assert ps.get_tensor_model_parallel_world_size() == world_size
    Utils.destroy_model_parallel()


@pytest.mark.parametrize('order', test_parallel_order)
def test_expert_tensor_parellel_world_size(order):
    Utils.initialize_model_parallel(expert_tensor_parallel_size=world_size, order=order)
    assert ps.get_expert_tensor_parallel_world_size() == world_size
    ps.set_expert_tensor_parallel_world_size(None)
    assert ps.get_expert_tensor_parallel_world_size() == world_size
    Utils.destroy_model_parallel()


@pytest.mark.parametrize('order', test_parallel_order)
def test_pipeline_model_parallel_world_size(order):
    Utils.initialize_model_parallel(pipeline_model_parallel_size=world_size, order=order)
    assert ps.get_pipeline_model_parallel_world_size() == world_size
    ps.set_pipeline_model_parallel_world_size(None)
    assert ps.get_pipeline_model_parallel_world_size() == world_size
    Utils.destroy_model_parallel()


@pytest.mark.parametrize('order', test_parallel_order)
def test_tensor_model_parallel_rank(order):
    Utils.initialize_model_parallel(tensor_model_parallel_size=world_size, order=order)
    assert ps.get_tensor_model_parallel_rank() == rank
    ps.set_tensor_model_parallel_rank(None)
    assert ps.get_tensor_model_parallel_rank() == rank
    Utils.destroy_model_parallel()


@pytest.mark.parametrize('order', test_parallel_order)
def test_moe_tensor_model_parellel_rank(order):
    Utils.initialize_model_parallel(expert_tensor_parallel_size=world_size, order=order)
    assert ps.get_expert_tensor_parallel_rank() == rank
    ps.set_expert_tensor_parallel_rank(None)
    assert ps.get_expert_tensor_parallel_rank() == rank
    Utils.destroy_model_parallel()


@pytest.mark.parametrize('order', test_parallel_order)
def test_pipeline_model_parallel_rank(order):
    Utils.initialize_model_parallel(pipeline_model_parallel_size=world_size, order=order)
    assert ps.get_pipeline_model_parallel_rank() == rank
    ps.set_pipeline_model_parallel_rank(None)
    assert ps.get_pipeline_model_parallel_rank() == rank
    Utils.destroy_model_parallel()


def test_context_parallel_rank():
    Utils.initialize_model_parallel(context_parallel_size=world_size)
    assert ps.get_context_parallel_rank() == rank
    Utils.destroy_model_parallel()


def test_expert_model_parallel_rank():
    Utils.initialize_model_parallel(expert_model_parallel_size=world_size)
    assert ps.get_expert_model_parallel_rank() == rank
    ps.set_expert_model_parallel_rank(None)
    assert ps.get_expert_model_parallel_rank() == rank
    Utils.destroy_model_parallel()


@pytest.mark.parametrize('order', test_parallel_order)
def test_is_pipeline_first_stage(order):
    Utils.initialize_model_parallel(pipeline_model_parallel_size=world_size, order=order)
    assert ps.is_pipeline_first_stage(ignore_virtual=False) == (rank == 0)
    assert ps.is_pipeline_first_stage() == (rank == 0)
    Utils.destroy_model_parallel()


@pytest.mark.parametrize('order', test_parallel_order)
def test_is_pipeline_last_stage(order):
    Utils.initialize_model_parallel(pipeline_model_parallel_size=world_size, order=order)
    assert ps.is_pipeline_last_stage(ignore_virtual=False) == (rank == world_size - 1)
    assert ps.is_pipeline_last_stage() == (rank == world_size - 1)
    Utils.destroy_model_parallel()


@pytest.mark.parametrize('order', test_parallel_order)
def test_virtual_pipeline_model_parallel_rank(order):
    Utils.initialize_model_parallel(pipeline_model_parallel_size=world_size, order=order)
    ps.set_virtual_pipeline_model_parallel_rank(rank)
    assert ps.get_virtual_pipeline_model_parallel_rank() == rank
    Utils.destroy_model_parallel()


@pytest.mark.parametrize('order', test_parallel_order)
def test_get_tensor_model_parallel_src_rank(order):
    Utils.initialize_model_parallel(tensor_model_parallel_size=world_size, order=order)
    assert ps.get_tensor_model_parallel_src_rank() == ((rank // world_size) * world_size)
    Utils.destroy_model_parallel()


@pytest.mark.internal
@pytest.mark.parametrize(
    'src_tp_pp, ep_size',
    [
        ((1, 8), 1),
        ((2, 4), 1),
        ((4, 2), 1),
        ((8, 1), 1),
        ((4, 1), 2),
        ((1, 1), 8),
        ((1, 1), 2),
        ((2, 1), 4),
    ],
)
def test_different_initialize_order_consistency(src_tp_pp, ep_size):
    Utils.initialize_model_parallel(
        *src_tp_pp, expert_model_parallel_size=ep_size, order='tp-ep-dp-pp'
    )
    tp_rank = ps.get_tensor_model_parallel_rank()
    dp_rank = ps.get_data_parallel_rank()
    pp_rank = ps.get_pipeline_model_parallel_rank()
    ep_rank = ps.get_expert_model_parallel_rank()

    tp_g = torch.distributed.get_process_group_ranks(ps.get_tensor_model_parallel_group())
    dp_g = torch.distributed.get_process_group_ranks(ps.get_data_parallel_group(False))
    pp_g = torch.distributed.get_process_group_ranks(ps.get_pipeline_model_parallel_group())
    dp_no_ep_g = torch.distributed.get_process_group_ranks(ps.get_expert_data_parallel_group())
    cp_g = torch.distributed.get_process_group_ranks(ps.get_context_parallel_group())
    mp_g = torch.distributed.get_process_group_ranks(ps.get_model_parallel_group())
    tp_ep_g = torch.distributed.get_process_group_ranks(
        ps.get_expert_tensor_and_model_parallel_group()
    )
    tp_dp_g = torch.distributed.get_process_group_ranks(
        ps.get_tensor_and_data_parallel_group(False)
    )

    Utils.destroy_model_parallel()

    Utils.initialize_model_parallel(
        *src_tp_pp, expert_model_parallel_size=ep_size, order='tp-pp-ep-dp'
    )
    assert tp_rank == ps.get_tensor_model_parallel_rank()
    assert dp_rank == ps.get_data_parallel_rank()
    assert pp_rank == ps.get_pipeline_model_parallel_rank()
    assert ep_rank == ps.get_expert_model_parallel_rank()

    assert tp_g == torch.distributed.get_process_group_ranks(ps.get_tensor_model_parallel_group())
    assert dp_g == torch.distributed.get_process_group_ranks(ps.get_data_parallel_group(False))
    assert pp_g == torch.distributed.get_process_group_ranks(ps.get_pipeline_model_parallel_group())
    assert dp_no_ep_g == torch.distributed.get_process_group_ranks(
        ps.get_expert_data_parallel_group()
    )
    assert cp_g == torch.distributed.get_process_group_ranks(ps.get_context_parallel_group())
    assert mp_g == torch.distributed.get_process_group_ranks(ps.get_model_parallel_group())
    assert tp_ep_g == torch.distributed.get_process_group_ranks(
        ps.get_expert_tensor_and_model_parallel_group()
    )
    assert tp_dp_g == torch.distributed.get_process_group_ranks(
        ps.get_tensor_and_data_parallel_group(False)
    )

    Utils.destroy_model_parallel()


@pytest.mark.parametrize(
    'src_tp_pp, ep_size',
    [((1, 2), 1), ((1, 4), 1), ((2, 2), 1), ((1, 2), 2), ((1, 4), 2), ((2, 2), 2)],
)
@pytest.mark.flaky
@pytest.mark.flaky_in_dev
def test_different_initialize_order_unconsistency(src_tp_pp, ep_size):
    Utils.initialize_model_parallel(
        *src_tp_pp, expert_model_parallel_size=ep_size, order='tp-ep-dp-pp'
    )

    tp_g = torch.distributed.get_process_group_ranks(ps.get_tensor_model_parallel_group())
    dp_g = torch.distributed.get_process_group_ranks(ps.get_data_parallel_group(False))
    pp_g = torch.distributed.get_process_group_ranks(ps.get_pipeline_model_parallel_group())
    cp_g = torch.distributed.get_process_group_ranks(ps.get_context_parallel_group())
    amax_g = torch.distributed.get_process_group_ranks(ps.get_amax_reduction_group(False))
    mp_g = torch.distributed.get_process_group_ranks(ps.get_model_parallel_group())

    Utils.destroy_model_parallel()

    Utils.initialize_model_parallel(
        *src_tp_pp, expert_model_parallel_size=ep_size, order='tp-pp-ep-dp'
    )
    assert tp_g == torch.distributed.get_process_group_ranks(ps.get_tensor_model_parallel_group())
    assert dp_g != torch.distributed.get_process_group_ranks(ps.get_data_parallel_group(False))
    assert pp_g != torch.distributed.get_process_group_ranks(ps.get_pipeline_model_parallel_group())
    assert cp_g == torch.distributed.get_process_group_ranks(ps.get_context_parallel_group())
    assert amax_g != torch.distributed.get_process_group_ranks(ps.get_amax_reduction_group(False))
    assert mp_g != torch.distributed.get_process_group_ranks(ps.get_model_parallel_group())

    Utils.destroy_model_parallel()


@pytest.mark.internal
@pytest.mark.parametrize(
    'nodes, num_gpu, tp, pp, cp, ep',
    [
        (1, 1, 1, 1, 1, 1),
        (1, 8, 8, 1, 1, 1),
        (1, 8, 2, 2, 1, 1),
        (1, 8, 2, 4, 1, 1),
        (3, 8, 8, 3, 1, 1),
        (4, 8, 2, 4, 1, 1),
        (8, 8, 8, 8, 1, 1),
        (8, 8, 2, 1, 1, 4),
        (8, 8, 2, 2, 2, 4),
        (8, 8, 2, 1, 4, 8),
        (8, 8, 2, 2, 2, 8),
        (16, 8, 4, 8, 1, 1),
        (16, 8, 4, 8, 1, 4),
        (16, 8, 4, 8, 4, 1),
        (16, 8, 8, 8, 1, 1),
        (16, 8, 4, 8, 1, 1),
        (16, 8, 8, 8, 1, 1),
        (32, 8, 4, 8, 1, 1),
        (32, 8, 8, 8, 1, 1),
        (32, 8, 4, 8, 1, 4),
        (32, 8, 8, 8, 4, 1),
        (64, 8, 4, 2, 8, 8),
        (64, 8, 4, 8, 1, 1),
        (64, 8, 8, 8, 1, 1),
        (96, 8, 4, 8, 1, 1),
        (128, 8, 4, 2, 8, 8),
        (128, 8, 4, 8, 1, 1),
        (256, 8, 4, 8, 1, 1),
        (316, 8, 4, 8, 1, 1),
        (384, 8, 4, 8, 1, 1),
        (512, 8, 4, 8, 1, 1),
        (768, 8, 4, 8, 1, 1),
        (1024, 8, 4, 8, 1, 1),
        (1280, 8, 4, 8, 1, 1),
        (1344, 8, 4, 8, 1, 1),
    ],
)
def test_rank_generator_for_tp_dp_pp(nodes, num_gpu, tp, pp, cp, ep):
    def golden_rank_result_from_past_code(
        world_size: int,
        tensor_model_parallel_size: int = 1,
        pipeline_model_parallel_size: int = 1,
        context_parallel_size: int = 1,
        expert_model_parallel_size: int = 1,
    ):
        data_parallel_size: int = world_size // (
            tensor_model_parallel_size * pipeline_model_parallel_size * context_parallel_size
        )
        num_tensor_model_parallel_groups: int = world_size // tensor_model_parallel_size
        num_pipeline_model_parallel_groups: int = world_size // pipeline_model_parallel_size

        dp_groups = []
        dp_groups_with_cp = []

        all_data_parallel_group_ranks_with_cp = []
        for i in range(pipeline_model_parallel_size):
            start_rank = i * num_pipeline_model_parallel_groups
            end_rank = (i + 1) * num_pipeline_model_parallel_groups
            for j in range(context_parallel_size * tensor_model_parallel_size):
                ranks = range(
                    start_rank + j, end_rank, context_parallel_size * tensor_model_parallel_size
                )
                dp_groups.append(list(ranks))
            for j in range(tensor_model_parallel_size):
                ranks_with_cp = range(start_rank + j, end_rank, tensor_model_parallel_size)
                all_data_parallel_group_ranks_with_cp.append(list(ranks_with_cp))
                dp_groups_with_cp.append(list(ranks_with_cp))

        cp_group = []
        for i in range(pipeline_model_parallel_size):
            for j in range(data_parallel_size):
                start_rank = (
                    i * num_pipeline_model_parallel_groups
                    + j * tensor_model_parallel_size * context_parallel_size
                )
                end_rank = (
                    i * num_pipeline_model_parallel_groups
                    + (j + 1) * tensor_model_parallel_size * context_parallel_size
                )
                for k in range(tensor_model_parallel_size):
                    ranks = range(start_rank + k, end_rank, tensor_model_parallel_size)
                    cp_group.append(list(ranks))

        mp_group = []
        for i in range(data_parallel_size * context_parallel_size):
            ranks = [
                data_parallel_group_ranks_with_cp[i]
                for data_parallel_group_ranks_with_cp in all_data_parallel_group_ranks_with_cp
            ]
            mp_group.append(list(ranks))

        tp_group = []
        for i in range(num_tensor_model_parallel_groups):
            ranks = range(i * tensor_model_parallel_size, (i + 1) * tensor_model_parallel_size)
            tp_group.append(list(ranks))

        pp_group = []
        for i in range(num_pipeline_model_parallel_groups):
            ranks = range(i, world_size, num_pipeline_model_parallel_groups)
            pp_group.append(list(ranks))

        tp_dp_group = []
        tp_dp_cp_group = []
        tensor_and_data_group_size_with_cp: int = (
            tensor_model_parallel_size * data_parallel_size * context_parallel_size
        )
        num_tensor_and_data_groups_with_cp: int = world_size // tensor_and_data_group_size_with_cp
        for i in range(num_tensor_and_data_groups_with_cp):
            start_rank = i * tensor_and_data_group_size_with_cp
            end_rank = start_rank + tensor_and_data_group_size_with_cp
            ranks = range(start_rank, end_rank)
            tp_dp_cp_group.append(list(ranks))

            for j in range(context_parallel_size):
                ranks = []
                for k in range(data_parallel_size):
                    start_rank = (
                        i * tensor_and_data_group_size_with_cp
                        + j * tensor_model_parallel_size
                        + k * tensor_model_parallel_size * context_parallel_size
                    )
                    end_rank = start_rank + tensor_model_parallel_size
                    ranks = ranks + list(range(start_rank, end_rank))
                tp_dp_group.append(list(ranks))

        expert_tp_ep_group = []
        expert_dp_group = []

        expert_data_parallel_size = world_size // (
            tensor_model_parallel_size * pipeline_model_parallel_size * expert_model_parallel_size
        )
        all_ranks = torch.arange(world_size).reshape(
            (
                pipeline_model_parallel_size,
                expert_data_parallel_size,
                expert_model_parallel_size,
                tensor_model_parallel_size,
            )
        )
        # (pp, dp, ep, tp) -> (pp*dp, ep*tp)
        tp_ep_rearrange = torch.reshape(
            all_ranks, (-1, expert_model_parallel_size * tensor_model_parallel_size)
        )
        num_tp_ep_groups = tp_ep_rearrange.shape[0]
        for i in range(num_tp_ep_groups):
            expert_tensor_and_model_parallel_ranks = tp_ep_rearrange[i].tolist()
            expert_tp_ep_group.append(expert_tensor_and_model_parallel_ranks)

        # (pp, dp, ep, tp) -> (pp*ep*tp, dp)
        expert_dp_rearrange = torch.permute(all_ranks, (0, 2, 3, 1)).reshape(
            -1, expert_data_parallel_size
        )
        num_expert_dp_groups = world_size // expert_data_parallel_size
        for i in range(num_expert_dp_groups):
            expert_dp_ranks = expert_dp_rearrange[i].tolist()
            expert_dp_group.append(expert_dp_ranks)

        return (
            dp_groups,
            dp_groups_with_cp,
            cp_group,
            mp_group,
            tp_group,
            pp_group,
            tp_dp_group,
            tp_dp_cp_group,
            expert_tp_ep_group,
            expert_dp_group,
        )

    world_size = nodes * num_gpu
    dp = world_size // (tp * pp * cp)
    expert_dp = world_size // (tp * ep * pp)
    assert dp % ep == 0, f"dp size ({dp}) is not divisible by ep {ep} ."
    assert (
        world_size % (tp * pp * cp) == 0
    ), f"world_size ({world_size}) is not divisible by tp {tp} x pp {pp} x cp {cp}."
    (
        dp_groups,
        dp_groups_with_cp,
        cp_group,
        mp_group,
        tp_group,
        pp_group,
        tp_dp_group,
        tp_dp_cp_group,
        expert_tp_ep_group,
        expert_dp_group,
    ) = golden_rank_result_from_past_code(
        world_size=world_size,
        tensor_model_parallel_size=tp,
        pipeline_model_parallel_size=pp,
        context_parallel_size=cp,
        expert_model_parallel_size=ep,
    )
    rank_generator = ps.RankGenerator(tp=tp, ep=1, dp=dp, pp=pp, cp=cp, order="tp-cp-dp-pp")
    expert_rank_generator = ps.RankGenerator(
        tp=tp, ep=ep, dp=expert_dp, pp=pp, cp=1, order="tp-ep-dp-pp"
    )
    assert dp_groups == rank_generator.get_ranks(
        "dp"
    ), f"{dp_groups} != {rank_generator.get_ranks('dp')}"
    assert dp_groups_with_cp == rank_generator.get_ranks(
        'dp-cp'
    ), f"{dp_groups_with_cp} != {rank_generator.get_ranks('dp-cp')}"
    assert cp_group == rank_generator.get_ranks(
        "cp"
    ), f"{cp_group} != {rank_generator.get_ranks('cp')}."
    assert mp_group == rank_generator.get_ranks(
        "tp-pp"
    ), f"{mp_group} != {rank_generator.get_ranks('tp-pp')}"
    assert tp_group == rank_generator.get_ranks(
        "tp"
    ), f"{tp_group} != {rank_generator.get_ranks('tp')}"
    assert pp_group == rank_generator.get_ranks(
        "pp"
    ), f"{pp_group} != {rank_generator.get_ranks('pp')}"
    assert tp_dp_group == rank_generator.get_ranks(
        "tp-dp"
    ), f"{tp_dp_group} != {rank_generator.get_ranks('tp-dp')}"
    assert tp_dp_cp_group == rank_generator.get_ranks(
        "tp-dp-cp"
    ), f"{tp_dp_cp_group} != {rank_generator.get_ranks('tp-dp-cp')}"
    assert expert_tp_ep_group == expert_rank_generator.get_ranks(
        "tp-ep"
    ), f"{expert_tp_ep_group} != {expert_rank_generator.get_ranks('tp-ep')}."
    assert expert_dp_group == expert_rank_generator.get_ranks(
        "dp"
    ), f"{expert_dp_group} != {expert_rank_generator.get_ranks('dp')}."


@pytest.mark.parametrize(
    "world_size, tp_size, cp_size, dp_size",
    [(8, 1, 2, 4), (8, 1, 1, 8)],  # 8 GPUs, 1 TP, 2 CP, 4 DP  # 8 GPUs, 1 TP, 1 CP, 8 DP
)
def test_hybrid_dp_cp_groups(world_size, tp_size, cp_size, dp_size):
    """
    Test that hybrid DPxCP groups are created correctly.
    """
    Utils.destroy_model_parallel()

    # Skip if world size doesn't match
    actual_world_size = torch.cuda.device_count()
    if actual_world_size != world_size:
        pytest.skip(f"Test requires world_size={world_size}, but got {actual_world_size}")
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=tp_size,
        context_parallel_size=cp_size,
        hybrid_context_parallel=True,
    )

    dp_cp_size = ps.get_data_parallel_world_size(with_context_parallel=True)
    group_sizes = [2**i for i in range(int(log2(dp_cp_size)))][1:]
    for group_size in group_sizes:
        group = ps.get_hybrid_data_context_parallel_groups(group_size=group_size)
        assert group.size() == group_size

    Utils.destroy_model_parallel()


def test_separate_all_gather_group():
    """Test AG groups passed via ProcessGroupCollection for AG/RS overlap."""
    # Initialize model parallel
    Utils.initialize_model_parallel(context_parallel_size=world_size)

    # Get ranks for creating AG groups (as users will do in their training scripts)
    dp_cp_ranks = ps.get_data_parallel_global_ranks(with_context_parallel=True)

    # Create separate AG group with same ranks as dp-cp group
    # This is what enables AG/RS overlap - separate communicator, same ranks
    dp_cp_ag_group = torch.distributed.new_group(ranks=dp_cp_ranks, backend='nccl')

    # Create ProcessGroupCollection and populate with standard groups
    pg_collection = ProcessGroupCollection.use_mpu_process_groups()

    # Set AG group (this is what users will do to enable AG/RS overlap)
    pg_collection.dp_cp_ag = dp_cp_ag_group

    # Verify AG group is accessible from ProcessGroupCollection
    assert hasattr(pg_collection, 'dp_cp_ag')
    assert pg_collection.dp_cp_ag is not None
    assert pg_collection.dp_cp_ag == dp_cp_ag_group

    # Verify AG group has same ranks as regular dp-cp group but different communicator
    regular_group = pg_collection.dp_cp
    assert regular_group is not None

    ag_ranks = torch.distributed.get_process_group_ranks(dp_cp_ag_group)
    regular_ranks = torch.distributed.get_process_group_ranks(regular_group)
    assert ag_ranks == regular_ranks, "AG group should have same ranks as dp-cp group"
    assert dp_cp_ag_group != regular_group, "AG group should be a different communicator"

    Utils.destroy_model_parallel()


def test_expert_all_gather_group():
    """Test expert AG groups for MoE models with AG/RS overlap."""
    # Initialize model parallel with expert parallelism
    Utils.initialize_model_parallel(
        expert_model_parallel_size=min(2, world_size),
        context_parallel_size=max(1, world_size // 2) if world_size > 1 else 1,
    )

    # Get ranks for both regular and expert AG groups
    dp_cp_ranks = ps.get_data_parallel_global_ranks(with_context_parallel=True)
    expt_dp_ranks = ps.get_expert_data_parallel_global_ranks()

    # Create AG groups for both regular and expert parameters
    dp_cp_ag_group = torch.distributed.new_group(ranks=dp_cp_ranks, backend='nccl')
    expt_dp_ag_group = torch.distributed.new_group(ranks=expt_dp_ranks, backend='nccl')

    # Create ProcessGroupCollection with AG groups
    pg_collection = ProcessGroupCollection.use_mpu_process_groups()
    pg_collection.dp_cp_ag = dp_cp_ag_group
    pg_collection.expt_dp_ag = expt_dp_ag_group

    # Verify both AG groups are set
    assert pg_collection.dp_cp_ag is not None
    assert pg_collection.expt_dp_ag is not None

    # Verify expert AG group has same ranks as expert dp group
    expt_dp_group = pg_collection.expt_dp
    if expt_dp_group is not None:
        expt_ag_ranks = torch.distributed.get_process_group_ranks(expt_dp_ag_group)
        expt_dp_ranks_actual = torch.distributed.get_process_group_ranks(expt_dp_group)
        assert (
            expt_ag_ranks == expt_dp_ranks_actual
        ), "Expert AG group should have same ranks as expert dp group"
        assert (
            expt_dp_ag_group != expt_dp_group
        ), "Expert AG group should be a different communicator"

    Utils.destroy_model_parallel()


def test_process_group_collection_defaults():
    """Test that ProcessGroupCollection initializes AG groups to None by default."""
    # Initialize model parallel
    Utils.initialize_model_parallel(context_parallel_size=world_size)

    # Create ProcessGroupCollection without setting AG groups
    pg_collection = ProcessGroupCollection.use_mpu_process_groups()

    # AG groups should be None by default (users must create them explicitly)
    assert hasattr(pg_collection, 'dp_cp_ag')
    assert hasattr(pg_collection, 'expt_dp_ag')
    assert pg_collection.dp_cp_ag is None, "dp_cp_ag should default to None"
    assert pg_collection.expt_dp_ag is None, "expt_dp_ag should default to None"

    # Regular groups should still work
    assert pg_collection.dp is not None
    assert pg_collection.dp_cp is not None

    Utils.destroy_model_parallel()
