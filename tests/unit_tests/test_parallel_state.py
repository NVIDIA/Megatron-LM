import os

import pytest
import torch

import megatron.core.parallel_state as ps
from tests.unit_tests.test_utilities import Utils

rank = Utils.rank
world_size = Utils.world_size
test_parallel_order = ['tp-cp-ep-dp-pp', 'tp-cp-pp-ep-dp']


@pytest.mark.parametrize('order', test_parallel_order)
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
    assert ps.is_pipeline_first_stage(ignore_virtual=True) == (rank == 0)
    assert ps.is_pipeline_first_stage() == (rank == 0)
    Utils.destroy_model_parallel()


@pytest.mark.parametrize('order', test_parallel_order)
def test_is_pipeline_last_stage(order):
    Utils.initialize_model_parallel(pipeline_model_parallel_size=world_size, order=order)
    assert ps.is_pipeline_last_stage(ignore_virtual=True) == (rank == world_size - 1)
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


@pytest.mark.parametrize('order', test_parallel_order)
def test_encoder_tensor_pipeline_parallelism(order):
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=5,
        pipeline_model_parallel_size=1,
        encoder_pipeline_model_parallel_size=1,
        encoder_tensor_model_parallel_size=3,
        order=order,
    )
    if rank < 2:
        assert ps.get_tensor_model_parallel_world_size() == 3
        assert isinstance(ps._PIPELINE_GLOBAL_RANKS[0], list)
    elif rank == 2:
        assert ps.get_tensor_model_parallel_world_size() == 3
        assert isinstance(ps._PIPELINE_GLOBAL_RANKS[0], int)
    else:
        assert ps.get_tensor_model_parallel_world_size() == 5
        assert isinstance(ps._PIPELINE_GLOBAL_RANKS[0], int)
    Utils.destroy_model_parallel()


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
    dp_no_ep_g = torch.distributed.get_process_group_ranks(
        ps.get_data_modulo_expert_parallel_group()
    )
    cp_g = torch.distributed.get_process_group_ranks(ps.get_context_parallel_group())
    mp_g = torch.distributed.get_process_group_ranks(ps.get_model_parallel_group())
    tp_ep_g = torch.distributed.get_process_group_ranks(ps.get_tensor_and_expert_parallel_group())
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
        ps.get_data_modulo_expert_parallel_group()
    )
    assert cp_g == torch.distributed.get_process_group_ranks(ps.get_context_parallel_group())
    assert mp_g == torch.distributed.get_process_group_ranks(ps.get_model_parallel_group())
    assert tp_ep_g == torch.distributed.get_process_group_ranks(
        ps.get_tensor_and_expert_parallel_group()
    )
    assert tp_dp_g == torch.distributed.get_process_group_ranks(
        ps.get_tensor_and_data_parallel_group(False)
    )

    Utils.destroy_model_parallel()


@pytest.mark.parametrize(
    'src_tp_pp, ep_size',
    [((1, 2), 1), ((1, 4), 1), ((2, 2), 1), ((1, 2), 2), ((1, 4), 2), ((2, 2), 2)],
)
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

        tp_ep_group = []
        dp_no_ep_group = []
        dp_no_ep_group_with_cp = []

        all_ranks = torch.arange(world_size).reshape(
            (
                pipeline_model_parallel_size,
                data_parallel_size // expert_model_parallel_size,
                expert_model_parallel_size,
                context_parallel_size,
                tensor_model_parallel_size,
            )
        )
        # 'pp edp ep cp tp -> (pp edp cp) (ep tp)'
        tp_ep_rearrange = torch.transpose(all_ranks, 2, 3)
        tp_ep_rearrange = torch.reshape(
            tp_ep_rearrange, (-1, expert_model_parallel_size * tensor_model_parallel_size)
        )
        tp_ep_rearrange = tp_ep_rearrange.tolist()
        tp_ep_rearrange.sort()
        for tensor_and_expert_parallel_ranks in tp_ep_rearrange:
            tensor_and_expert_parallel_ranks = list(tensor_and_expert_parallel_ranks)
            tensor_and_expert_parallel_ranks.sort()
            tp_ep_group.append(tensor_and_expert_parallel_ranks)
        # 'pp edp ep cp tp -> (pp ep cp tp) edp'
        edp_rearrange = torch.transpose(all_ranks, 1, 4)
        edp_rearrange = torch.reshape(
            edp_rearrange, (-1, data_parallel_size // expert_model_parallel_size)
        )
        edp_rearrange = edp_rearrange.tolist()
        edp_rearrange.sort()
        for expert_data_parallel_ranks in edp_rearrange:
            expert_data_parallel_ranks = list(expert_data_parallel_ranks)
            expert_data_parallel_ranks.sort()
            dp_no_ep_group.append(expert_data_parallel_ranks)
        # 'pp edp ep cp tp -> (pp ep tp) (cp edp)'
        edp_cp_rearrange = torch.transpose(all_ranks, 1, 2)
        edp_cp_rearrange = torch.transpose(edp_cp_rearrange, 2, 4)
        edp_cp_rearrange = torch.reshape(
            edp_cp_rearrange,
            (-1, context_parallel_size * data_parallel_size // expert_model_parallel_size),
        )
        edp_cp_rearrange = edp_cp_rearrange.tolist()
        edp_cp_rearrange.sort()
        for expert_data_parallel_ranksj_with_cp in edp_cp_rearrange:
            expert_data_parallel_ranksj_with_cp = list(expert_data_parallel_ranksj_with_cp)
            expert_data_parallel_ranksj_with_cp.sort()
            dp_no_ep_group_with_cp.append(expert_data_parallel_ranksj_with_cp)

        return (
            dp_groups,
            dp_groups_with_cp,
            cp_group,
            mp_group,
            tp_group,
            pp_group,
            tp_dp_group,
            tp_dp_cp_group,
            tp_ep_group,
            dp_no_ep_group,
            dp_no_ep_group_with_cp,
        )

    world_size = nodes * num_gpu
    dp = world_size // (tp * pp * cp)
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
        tp_ep_group,
        dp_no_ep_group,
        dp_no_ep_group_with_cp,
    ) = golden_rank_result_from_past_code(
        world_size=world_size,
        tensor_model_parallel_size=tp,
        pipeline_model_parallel_size=pp,
        context_parallel_size=cp,
        expert_model_parallel_size=ep,
    )
    rank_generator = ps.RankGenerator(tp=tp, ep=ep, dp=dp, pp=pp, cp=cp, order="tp-cp-ep-dp-pp")
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
    assert tp_ep_group == rank_generator.get_ranks(
        "tp-ep", independent_ep=True
    ), f"{tp_ep_group} != {rank_generator.get_ranks('tp-ep', independent_ep=True)}."
    assert dp_no_ep_group == rank_generator.get_ranks(
        "dp", independent_ep=True
    ), f"{dp_no_ep_group} != {rank_generator.get_ranks('dp', independent_ep=True)}."
    assert dp_no_ep_group_with_cp == rank_generator.get_ranks(
        "dp-cp", independent_ep=True
    ), f"{dp_no_ep_group_with_cp} != {rank_generator.get_ranks('dp-cp', independent_ep=True)}."
