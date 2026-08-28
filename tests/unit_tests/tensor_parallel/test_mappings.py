# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.tensor_parallel import mappings
from megatron.core.utils import get_tensor_model_parallel_group_if_none
from tests.unit_tests.test_utilities import Utils


@pytest.mark.internal
def test_CopyToModelParallelRegion():
    Utils.initialize_model_parallel(4, 2)
    input_data = torch.ones((1)).cuda() * Utils.rank

    tp_group = get_tensor_model_parallel_group_if_none(tp_group=None)

    class Ctx:
        group = tp_group

    output_data, _ = mappings._CopyToModelParallelRegion.backward(Ctx(), input_data)
    result = torch.ones(1).cuda()
    result = result * 22 if Utils.rank >= 4 else result * 6
    assert torch.equal(output_data, result)
    assert torch.equal(input_data, mappings.copy_to_tensor_model_parallel_region(input_data))
    assert torch.equal(
        input_data, mappings._CopyToModelParallelRegion.symbolic(None, input_data, tp_group)
    )
    Utils.destroy_model_parallel()


@pytest.mark.internal
def test_ReduceFromModelParallelRegion():
    Utils.initialize_model_parallel(4, 2)
    input_data = torch.ones((1)).cuda() * Utils.rank

    tp_group = get_tensor_model_parallel_group_if_none(tp_group=None)
    output_data = mappings._ReduceFromModelParallelRegion.symbolic(None, input_data, tp_group)

    result = torch.ones(1).cuda()
    result = result * 22 if Utils.rank >= 4 else result * 6
    assert torch.equal(output_data, result)

    input_data = torch.ones((1)).cuda() * Utils.rank
    assert torch.equal(mappings.reduce_from_tensor_model_parallel_region(input_data), result)

    class Ctx:
        group = tp_group

    output_data, _ = mappings._ReduceFromModelParallelRegion.backward(Ctx(), input_data)
    assert torch.equal(input_data, output_data)
    Utils.destroy_model_parallel()


@pytest.mark.internal
def test_ReduceFromMixedDynamicCPSubgroupsViaParent():
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=2,
        dynamic_context_parallel=True,
        use_dynamic_cp_logical_groups=True,
    )
    try:
        parent_group = torch.distributed.group.WORLD
        parent_rank = parent_group.rank()
        if parent_group.size() >= 8:
            subgroup_size = 4 if parent_rank < 4 else 2 if parent_rank < 6 else 1
        else:
            subgroup_size = 2 if parent_rank < 2 else 1
        subgroup = parallel_state.get_dynamic_data_context_parallel_groups(group_size=subgroup_size)

        input_data = torch.tensor([parent_rank + 1.0], device='cuda', requires_grad=True)
        actual = mappings.reduce_from_dynamic_cp_subgroup(input_data, subgroup, parent_group)

        expected = input_data.new_tensor([sum(rank + 1.0 for rank in subgroup.ranks)])
        torch.testing.assert_close(actual, expected)

        actual.sum().backward()
        torch.testing.assert_close(input_data.grad, torch.ones_like(input_data))
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.internal
@pytest.mark.parametrize("subgroup,parent_group", ((None, object()), (object(), None)))
def test_ReduceFromDynamicCPSubgroupRejectsMissingGroups(subgroup, parent_group):
    with pytest.raises(RuntimeError, match="requires subgroup and parent_group"):
        mappings._reduce_dynamic_cp_subgroup_via_parent(torch.ones(1), subgroup, parent_group)


@pytest.mark.internal
def test_ScatterToModelParallelRegion():
    Utils.initialize_model_parallel(4, 2)
    input_data = torch.rand((8, 4)).cuda()

    tp_group = get_tensor_model_parallel_group_if_none(tp_group=None)
    output_data = mappings.scatter_to_tensor_model_parallel_region(input_data)

    req_dim = int(Utils.rank % (Utils.world_size / 2))
    assert torch.equal(output_data, input_data[:, req_dim].reshape((8, 1)))
    output_data = mappings._ScatterToModelParallelRegion.symbolic(None, input_data, tp_group)
    assert torch.equal(output_data, input_data[:, req_dim].reshape((8, 1)))

    input_data = torch.ones(8).cuda() * Utils.rank

    class Ctx:
        group = tp_group

    actual_output_data, _ = mappings._ScatterToModelParallelRegion.backward(Ctx(), input_data)
    expected_output = torch.cat(
        (torch.ones(8) * 0, torch.ones(8) * 1, torch.ones(8) * 2, torch.ones(8) * 3)
    ).cuda()
    if Utils.rank >= 4:
        expected_output = expected_output + 4
    assert torch.equal(actual_output_data, expected_output)
    Utils.destroy_model_parallel()


@pytest.mark.internal
def test_GatherFromModelParallelRegion():
    Utils.initialize_model_parallel(4, 2)
    input_data = torch.rand((8, 4)).cuda()

    tp_group = get_tensor_model_parallel_group_if_none(tp_group=None)
    req_dim = int(Utils.rank % (Utils.world_size / 2))

    class Ctx:
        group = tp_group

    output_data, _ = mappings._GatherFromModelParallelRegion.backward(Ctx(), input_data)
    assert torch.equal(output_data, input_data[:, req_dim].reshape((8, 1)))

    input_data = torch.ones(8).cuda() * Utils.rank
    actual_output_data = mappings.gather_from_tensor_model_parallel_region(input_data)
    expected_output = torch.cat(
        (torch.ones(8) * 0, torch.ones(8) * 1, torch.ones(8) * 2, torch.ones(8) * 3)
    ).cuda()
    if Utils.rank >= 4:
        expected_output = expected_output + 4
    assert torch.equal(actual_output_data, expected_output)
    assert torch.equal(
        mappings._GatherFromModelParallelRegion.symbolic(None, input_data, tp_group),
        expected_output,
    )
    Utils.destroy_model_parallel()


@pytest.mark.internal
def test_ScatterToSequenceParallelRegion():
    Utils.initialize_model_parallel(4, 2)
    input_data = torch.rand((8, 4)).cuda()

    tp_group = get_tensor_model_parallel_group_if_none(tp_group=None)
    req_dim = int(Utils.rank % (Utils.world_size / 2)) * 2
    output_data = mappings._ScatterToSequenceParallelRegion.symbolic(None, input_data, tp_group)
    assert torch.equal(output_data, input_data[req_dim : req_dim + 2, :])
    output_data = mappings.scatter_to_sequence_parallel_region(input_data)
    assert torch.equal(output_data, input_data[req_dim : req_dim + 2, :])

    input_data = torch.ones(4).cuda() * Utils.rank

    class Ctx:
        group = tp_group

    output_data, _ = mappings._ScatterToModelParallelRegion.backward(Ctx(), input_data)
    expected_output = torch.concat(
        (torch.ones(4) * 0, torch.ones(4) * 1, torch.ones(4) * 2, torch.ones(4) * 3)
    ).cuda()
    if Utils.rank >= 4:
        expected_output = expected_output + 4
    assert torch.equal(output_data, expected_output)
    Utils.destroy_model_parallel()


@pytest.mark.internal
def test_GatherFromSequenceParallelRegion():
    Utils.initialize_model_parallel(4, 2)
    input_data = torch.ones(4).cuda() * Utils.rank

    tp_group = get_tensor_model_parallel_group_if_none(tp_group=None)
    output_data = mappings.gather_from_sequence_parallel_region(input_data)
    expected_output = torch.concat(
        (torch.ones(4) * 0, torch.ones(4) * 1, torch.ones(4) * 2, torch.ones(4) * 3)
    ).cuda()
    if Utils.rank >= 4:
        expected_output = expected_output + 4
    assert torch.equal(output_data, expected_output)
    assert torch.equal(
        mappings._GatherFromSequenceParallelRegion.symbolic(None, input_data, tp_group),
        expected_output,
    )
    input_data = torch.vstack(
        (torch.ones(4) * 0, torch.ones(4) * 1, torch.ones(4) * 2, torch.ones(4) * 3)
    ).cuda()

    class Ctx:
        tensor_parallel_output_grad = True
        output_split_sizes = None
        group = tp_group
        use_global_buffer = False

    output_data = mappings._GatherFromSequenceParallelRegion.backward(Ctx(), input_data)
    expected_output = torch.ones((1, 4)).cuda() * 4 * int(Utils.rank % 4)
    assert torch.equal(output_data[0], expected_output)
    Utils.destroy_model_parallel()


@pytest.mark.internal
def test_AsyncGatherFromSequenceParallelRegion():
    Utils.initialize_model_parallel(4, 1)
    group_rank = Utils.rank % 4
    input_data = (torch.ones((4, 2), device="cuda") * group_rank)[:, 0]
    input_data.requires_grad_(True)
    assert not input_data.is_contiguous()
    tp_group = get_tensor_model_parallel_group_if_none(tp_group=None)

    handle = mappings.async_gather_from_sequence_parallel_region(input_data, group=tp_group)
    assert handle._input_buffer is not None
    output_data = handle.wait()
    expected_output = torch.concat(
        (torch.ones(4) * 0, torch.ones(4) * 1, torch.ones(4) * 2, torch.ones(4) * 3)
    ).cuda()
    assert torch.equal(output_data, expected_output)
    assert handle.wait() is output_data
    assert handle.work is None
    assert handle._input_buffer is None

    output_data.sum().backward()
    assert torch.equal(input_data.grad, torch.ones_like(input_data) * 4)

    split_grad_input = (torch.ones(4, device="cuda") * Utils.rank).requires_grad_(True)
    split_grad_output = mappings.async_gather_from_sequence_parallel_region(
        split_grad_input, tensor_parallel_output_grad=False, group=tp_group
    ).wait()
    split_grad_output.sum().backward()
    assert torch.equal(split_grad_input.grad, torch.ones_like(split_grad_input))
    Utils.destroy_model_parallel()


@pytest.mark.internal
def test_AsyncReduceScatterAlongFirstDim():
    Utils.initialize_model_parallel(4, 1)
    group_rank = Utils.rank % 4
    input_data = (torch.ones((4, 16), device="cuda") * group_rank).t()
    assert not input_data.is_contiguous()
    tp_group = get_tensor_model_parallel_group_if_none(tp_group=None)

    handle = mappings.async_reduce_scatter_along_first_dim(input_data, group=tp_group)
    assert handle._input_buffer is not None
    output_data = handle.wait()
    assert torch.equal(output_data, torch.full_like(output_data, 6))
    assert handle.wait() is output_data
    assert handle.work is None
    assert handle._input_buffer is None

    with pytest.raises(AssertionError, match="First dimension.*divisible"):
        mappings.async_reduce_scatter_along_first_dim(torch.ones(15, device="cuda"), group=tp_group)
    Utils.destroy_model_parallel()


@pytest.mark.internal
def test_AsyncSequenceParallelCollectivesGroupSizeOne():
    Utils.initialize_model_parallel(1, 1)
    input_data = torch.arange(8, dtype=torch.float32, device="cuda").reshape(4, 2)[:, 0]
    input_data.requires_grad_(True)
    tp_group = get_tensor_model_parallel_group_if_none(tp_group=None)

    gather_handle = mappings.async_gather_from_sequence_parallel_region(input_data, group=tp_group)
    reduce_scatter_handle = mappings.async_reduce_scatter_along_first_dim(
        input_data, group=tp_group
    )
    assert gather_handle.wait() is input_data
    assert reduce_scatter_handle.wait() is input_data
    assert gather_handle.work is None
    assert reduce_scatter_handle.work is None

    gather_handle.wait().sum().backward()
    assert torch.equal(input_data.grad, torch.ones_like(input_data))
    Utils.destroy_model_parallel()


@pytest.mark.internal
def test_ReduceScatterToSequenceParallelRegion():
    Utils.initialize_model_parallel(4, 2)
    input_data = torch.vstack(
        (torch.ones(4) * 0, torch.ones(4) * 1, torch.ones(4) * 2, torch.ones(4) * 3)
    ).cuda()

    tp_group = get_tensor_model_parallel_group_if_none(tp_group=None)
    output_data = mappings.reduce_scatter_to_sequence_parallel_region(input_data)
    expected_output = torch.ones(4).cuda() * 4 * int(Utils.rank % 4)
    assert torch.equal(output_data[0], expected_output)
    assert torch.equal(
        mappings._ReduceScatterToSequenceParallelRegion.symbolic(None, input_data, tp_group),
        expected_output.reshape((1, 4)),
    )
    input_data = torch.ones(4).cuda() * Utils.rank

    class Ctx:
        input_split_sizes = None
        group = tp_group
        use_global_buffer = False

    output_data = mappings._ReduceScatterToSequenceParallelRegion.backward(Ctx(), input_data)
    expected_output = torch.concat(
        (torch.ones(4) * 0, torch.ones(4) * 1, torch.ones(4) * 2, torch.ones(4) * 3)
    ).cuda()
    if Utils.rank >= 4:
        expected_output = expected_output + 4
    assert torch.equal(output_data[0], expected_output)
    Utils.destroy_model_parallel()
