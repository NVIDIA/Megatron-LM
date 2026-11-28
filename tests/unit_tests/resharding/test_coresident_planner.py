# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Portions adapted from NVIDIA-NeMo/RL under Apache-2.0.
# https://github.com/NVIDIA-NeMo/RL/tree/34dd6b3b4f8f108554ec0c9312d83ba7e639da9d
"""CPU numerical coverage of native local/residual reshard planning."""

from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from megatron.core.resharding.coresident import (
    LocalSourcePlan,
    RefitSlices,
    plan_local_sources,
    remap_metadata,
)
from megatron.core.resharding.planner import build_plan_from_rosters, index_metadata_rosters
from megatron.core.resharding.utils import ParameterMetadata


def meta(name, tensor, rank, group, *, resolved=None, dim=0, stride=1, is_tp=True):
    return ParameterMetadata(
        name=name,
        resolved_name=resolved or name,
        shape=tuple(tensor.shape),
        dtype=tensor.dtype,
        element_size=tensor.element_size(),
        owner_rank=rank,
        tensor_parallel_group_ranks=group,
        data_parallel_group_ranks=[rank],
        is_tp=is_tp,
        partition_dim=dim,
        partition_stride=stride,
    )


def materialize(source):
    if isinstance(source, torch.Tensor):
        return source
    result = torch.empty(source.shape, dtype=source.parts[0][0].dtype)
    for view, destination in source.parts:
        result[destination].copy_(view)
    return result


@pytest.mark.parametrize("source_tp,target_tp", [(1, 2), (1, 4), (2, 4), (4, 2), (4, 1)])
@pytest.mark.parametrize("dim", [0, 1])
def test_native_residual_plans_and_local_views_cover_exact_full_tensor(source_tp, target_tp, dim):
    world = 4
    full = torch.arange(128, dtype=torch.float32).reshape(16, 8)
    sources, targets, src_meta, dst_meta = [], [], [], []
    for rank in range(world):
        source = full.chunk(source_tp, dim=dim)[rank % source_tp].clone()
        target = torch.empty_like(full.chunk(target_tp, dim=dim)[rank % target_tp])
        sources.append(source)
        targets.append(target)
        src_meta.append(
            meta(
                "source.weight",
                source,
                rank,
                list(range(rank // source_tp * source_tp, (rank // source_tp + 1) * source_tp)),
                resolved="weight",
                dim=dim,
            )
        )
        dst_meta.append(
            meta(
                "engine.weight",
                target,
                rank,
                list(range(rank // target_tp * target_tp, (rank // target_tp + 1) * target_tp)),
                resolved="weight",
                dim=dim,
            )
        )
    classified = [
        plan_local_sources({"weight": sources[r]}, src_meta, [dst_meta[r]], learner_rank=r)
        for r in range(world)
    ]
    rosters = [([src_meta[r]], list(classified[r].residual)) for r in range(world)]
    dst_roster, src_roster = index_metadata_rosters(rosters)
    plans = [build_plan_from_rosters(dst_roster, src_roster, r) for r in range(world)]
    # A rank can send, receive, do both, or sit idle. Send and receive IDs
    # must match even when the first TP shard has a local alias.
    sends = {
        (rank, op.peer_rank, op.task_id, op.batch_id)
        for rank, plan in enumerate(plans)
        for op in plan.send_ops
    }
    receives = {
        (op.peer_rank, rank, op.task_id, op.batch_id)
        for rank, plan in enumerate(plans)
        for op in plan.recv_ops
    }
    assert sends == receives
    assert all(plan.total_tasks == len(sends) for plan in plans)
    for rank, plan in enumerate(plans):
        assert bool(plan.recv_ops) == bool(classified[rank].residual)
    for publication in range(2):
        if publication:
            for source in sources:
                source.add_(100)
        for rank, classification in enumerate(classified):
            if classification.local_sources:
                value = classification.local_sources["engine.weight"]
                targets[rank].copy_(materialize(value))
                if isinstance(value, torch.Tensor):
                    assert (
                        value.untyped_storage().data_ptr()
                        == sources[rank].untyped_storage().data_ptr()
                    )
            else:
                for recv in plans[rank].recv_ops:
                    send = next(
                        op for op in plans[recv.peer_rank].send_ops if op.task_id == recv.task_id
                    )
                    assert send.peer_rank == rank and send.param_name == "source.weight"
                    targets[rank][recv.my_slice].copy_(sources[recv.peer_rank][send.my_slice])
            expected = full.chunk(target_tp, dim=dim)[rank % target_tp] + publication * 100
            torch.testing.assert_close(targets[rank], expected, rtol=0, atol=0)
    if source_tp == 1:
        for rank, item in enumerate(classified):
            view = item.local_sources["engine.weight"]
            expected = sources[rank].chunk(target_tp, dim=dim)[rank % target_tp]
            assert isinstance(view, torch.Tensor)
            assert view.storage_offset() == expected.storage_offset()
            assert view.stride() == expected.stride()
    if source_tp > target_tp:
        assert all(item.residual for item in classified)
    if source_tp == 2 and target_tp == 4:
        assert [bool(item.local_sources) for item in classified] == [True, False, False, True]


def test_packed_local_fragments_and_remote_parameter_partition():
    full = torch.arange(48, dtype=torch.float32).reshape(12, 4)
    packed = torch.cat((full[:3], full[6:9]))
    local_meta = meta("source.packed", full, 0, [0], resolved="packed", stride=2)
    remote = torch.arange(4, dtype=torch.float32)
    remote_meta = meta("source.remote", remote, 1, [1], resolved="remote", is_tp=False)
    destination = [
        meta("head.packed", packed, 0, [0, 1], resolved="packed", stride=2),
        meta("head.remote", remote, 0, [0], resolved="remote", is_tp=False),
    ]
    result = plan_local_sources(
        {"packed": full}, [local_meta, remote_meta], destination, learner_rank=0
    )
    assert isinstance(result.local_sources["head.packed"], RefitSlices)
    torch.testing.assert_close(
        materialize(result.local_sources["head.packed"]), packed, rtol=0, atol=0
    )
    assert [item.name for item in result.residual] == ["head.remote"]
    assert result.residual[0] is not destination[1]


def test_cp_replicas_ep_and_pp_global_names_choose_actual_local_owner():
    tensor = torch.arange(8, dtype=torch.bfloat16)
    # Different local layer/expert names represent the same globally resolved parameter.
    canonical = "decoder.layers.7.experts.weight3"
    sources = [
        meta(
            "decoder.layers.0.experts.weight1",
            tensor,
            rank,
            [rank],
            resolved=canonical,
            is_tp=False,
        )
        for rank in (0, 2)
    ]
    for item in sources:
        item.is_ep = True
        item.global_expert_index = 3
        item.num_experts = 4
        item.expert_parallel_group_ranks = [item.owner_rank + 1, item.owner_rank]
        item.data_parallel_group_ranks = [0, 2]
    destination = meta(
        "decoder.layers.3.experts.weight0", tensor, 2, [2], resolved=canonical, is_tp=False
    )
    destination.is_ep, destination.global_expert_index, destination.num_experts = (True, 3, 4)
    destination.expert_parallel_group_ranks = [3, 2]
    local = tensor.clone().add_(10)
    result = plan_local_sources({canonical: local}, sources, [destination], learner_rank=2)
    assert not result.residual
    assert result.local_sources[destination.name].data_ptr() == local.data_ptr()


def test_rank_remap_preserves_order_names_and_input():
    item = meta("local.weight", torch.zeros(2), 1, [0, 1], resolved="global.weight")
    item.expert_parallel_group_ranks = [1, 3]
    item.data_parallel_group_ranks = [1, 2]
    item.pipeline_parallel_group_ranks = [1, 0]
    original = deepcopy(item)
    result = remap_metadata([item], {0: 2, 1: 0, 2: 3, 3: 1}, learner_world_size=4)[0]
    assert item == original and result is not item
    assert result.owner_rank == 0 and result.tensor_parallel_group_ranks == [2, 0]
    assert result.expert_parallel_group_ranks == [0, 1]
    assert result.data_parallel_group_ranks == [0, 3]
    assert result.pipeline_parallel_group_ranks == [0, 2]
    assert (result.name, result.resolved_name) == ("local.weight", "global.weight")


@pytest.mark.parametrize(
    "change",
    [
        {"is_gtp": True},
        {"dtype": torch.int8, "element_size": 1},
        {"element_size": 8},
        {"partition_stride": 0},
        {"partition_sizes": [1]},
        {"tensor_parallel_group_ranks": [0, 0]},
        {"resolved_name": None},
        {"is_ep": True},
        {"shape": (0,)},
    ],
)
def test_malformed_metadata_never_becomes_residual(change):
    value = torch.zeros(4)
    source = meta("weight", value, 0, [0])
    target = replace(source, **change)
    with pytest.raises(ValueError):
        plan_local_sources({"weight": value}, [source], [target], learner_rank=0)


def test_incomplete_global_source_coverage_is_not_fallback():
    value = torch.zeros(2)
    source = meta("weight", value, 0, [0, 1])
    target = meta("weight", torch.zeros(4), 0, [0])
    with pytest.raises(RuntimeError, match="covered 2 of 4"):
        plan_local_sources({"weight": value}, [source], [target], learner_rank=0)


@pytest.mark.parametrize("remote", [False, True])
def test_mixed_dtype_refit_preserves_live_inference_weight(monkeypatch, remote):
    from types import SimpleNamespace

    from megatron.core.resharding import coresident

    monkeypatch.setattr(
        torch.cuda, "current_stream", lambda: SimpleNamespace(synchronize=lambda: None)
    )
    value = torch.arange(8, dtype=torch.float32)
    src_rank = int(remote)
    source = meta("weight", value, src_rank, [src_rank], is_tp=False)
    target = meta("weight", value.to(torch.bfloat16), 0, [0], is_tp=False)
    inputs = {} if remote else {"weight": value}
    plan = plan_local_sources(inputs, [source], [target], learner_rank=0)
    model = torch.nn.Linear(8, 1, bias=False, dtype=torch.bfloat16)
    model.weight = torch.nn.Parameter(torch.zeros(8, dtype=torch.bfloat16))
    ptr = model.weight.data_ptr()
    fingerprint, handles = plan.export_destinations(model, None)
    buffers = coresident.open_refit_handles(handles)
    if remote:
        assert plan.residual[0].dtype == buffers["weight"].dtype == torch.float32
    else:
        assert not buffers
        assert not coresident.bind_shared(inputs, model.named_parameters(), {id(model.weight)})[
            "tensors_bound"
        ]
    for update in (0.03125, 0.501):
        value.add_(update)
        if remote:
            buffers["weight"].copy_(value)
        assert plan.export_destinations(model, fingerprint) == (fingerprint, None)
        plan.refresh(model)
        assert model.weight.data_ptr() == ptr and model.weight.dtype == torch.bfloat16
        assert torch.equal(model.weight, value.to(torch.bfloat16))


def test_source_replicas_must_agree_on_transport_dtype():
    value = torch.ones(4)
    a = meta("weight", value, 0, [0], is_tp=False)
    b = meta("weight", value.to(torch.bfloat16), 1, [1], is_tp=False)
    with pytest.raises(ValueError, match="Source replicas"):
        plan_local_sources({"weight": value}, [a, b], [a], learner_rank=0)


def test_refit_plan_resolves_replaced_engine_storage():
    engine = torch.nn.Linear(2, 1)
    bias_metadata = meta("bias", engine.bias, 0, [0], is_tp=False)
    plan = LocalSourcePlan({"weight": engine.weight}, (bias_metadata,))
    before = plan.current_tensors(engine)
    engine.bias = torch.nn.Parameter(torch.zeros_like(engine.bias))
    after = plan.current_tensors(engine)
    assert before["bias"] is not after["bias"]
    assert after["bias"] is engine.bias
    del engine.bias
    with pytest.raises(RuntimeError, match="tensor names changed"):
        plan.current_tensors(engine)


def test_native_executor_receives_imported_views_without_copy(monkeypatch):
    from types import SimpleNamespace

    from megatron.core.resharding.execution import _execute_reshard_plan_with_tensors
    from megatron.core.resharding.utils import ReshardPlan

    backing = torch.arange(6, dtype=torch.float32)
    views = {"decoder.weight": backing[2:4]}

    def execute(plan, src, dst, transform):
        assert dst["decoder.weight"] is views["decoder.weight"]
        dst["decoder.weight"].add_(3)
        return True

    service = SimpleNamespace(
        set_plan=lambda *args, **kwargs: None,
        execute_plan=execute,
        requires_process_group_barrier=False,
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    _execute_reshard_plan_with_tensors(ReshardPlan([], []), {}, views, service)
    assert backing.tolist() == [0, 1, 5, 6, 4, 5]


def test_residual_batches_agree_on_all_ranks_including_idle_ranks():
    tensor = torch.zeros(4, 4)
    sources = [meta(name, tensor, 0, [0], is_tp=False) for name in ("a", "b")]
    # Only rank 2 needs these two private weights. Ranks 1 and 3 are idle.
    destinations = [
        replace(
            item,
            owner_rank=2,
            name="engine." + item.name,
            tensor_parallel_group_ranks=[2],
            data_parallel_group_ranks=[2],
        )
        for item in sources
    ]
    dst, src = index_metadata_rosters([(sources, []), ([], []), ([], destinations), ([], [])])
    plans = [build_plan_from_rosters(dst, src, rank, execution_batch_bytes=64) for rank in range(4)]

    assert all(plan.num_batches == 2 and plan.total_tasks == 2 for plan in plans)
    # Disjoint, complete meshes also get Core's native whole-tensor specs.
    assert all(plan.tensor_reshard_specs is not None for plan in plans)
    assert all(not plans[rank].recv_ops and not plans[rank].send_ops for rank in (1, 3))
    assert [(op.peer_rank, op.param_name, op.task_id, op.batch_id) for op in plans[0].send_ops] == [
        (2, "a", 0, 0),
        (2, "b", 1, 1),
    ]
    assert [(op.peer_rank, op.param_name, op.task_id, op.batch_id) for op in plans[2].recv_ops] == [
        (0, "engine.a", 0, 0),
        (0, "engine.b", 1, 1),
    ]


@pytest.mark.parametrize("local_owner", [True, False])
@pytest.mark.parametrize(
    "source_name", ["embedding.word_embeddings.weight", "word_embeddings.weight"]
)
def test_tied_output_head_reuses_native_embedding_alias(local_owner, source_name):
    tensor = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    owner = 0 if local_owner else 1
    source = meta(source_name, tensor, owner, [owner], is_tp=False)
    target = meta(
        "decoder.output_layer.weight", tensor, 0, [0], resolved="output_layer.weight", is_tp=False
    )
    result = plan_local_sources(
        {source_name: tensor} if local_owner else {}, [source], [target], learner_rank=0
    )
    if local_owner:
        assert not result.residual
        assert result.local_sources[target.name].data_ptr() == tensor.data_ptr()
    else:
        assert not result.local_sources
        assert result.residual[0].name == target.name
        dst, src = index_metadata_rosters([([], list(result.residual)), ([source], [])])
        plans = [build_plan_from_rosters(dst, src, rank) for rank in range(2)]
        receive = plans[0].recv_ops[0]
        send = plans[1].send_ops[0]
        assert receive.param_name == target.name and send.param_name == source_name
        output = torch.empty_like(tensor)
        output[receive.my_slice].copy_(tensor[send.my_slice])
        torch.testing.assert_close(output, tensor, rtol=0, atol=0)


def test_partial_world_refit_preserves_send_only_sources_and_full_destinations():
    full = torch.arange(8, dtype=torch.float32)
    sources = [full[:4].clone(), full[4:].clone()]
    source_metadata = [
        meta("source.weight", value, rank, [0, 1], resolved="weight")
        for rank, value in enumerate(sources)
    ]
    outputs = [torch.empty_like(full), torch.empty_like(full)]
    engine_metadata = [
        meta("engine.weight", value, rank, [rank], resolved="weight")
        for rank, value in enumerate(outputs)
    ]
    mapped = remap_metadata(engine_metadata, {0: 2, 1: 3}, learner_world_size=4)
    assert [item.owner_rank for item in mapped] == [2, 3]
    assert [item.tensor_parallel_group_ranks for item in mapped] == [[2], [3]]
    rosters = [([source_metadata[0]], []), ([source_metadata[1]], [])]
    rosters += [([], [item]) for item in mapped]
    destinations, source_roster = index_metadata_rosters(rosters)
    plans = [build_plan_from_rosters(destinations, source_roster, rank) for rank in range(4)]
    assert all(plan.send_ops and not plan.recv_ops for plan in plans[:2])
    for publication in range(2):
        if publication:
            for source in sources:
                source.add_(10)
        for rank, output in enumerate(outputs, start=2):
            for recv in plans[rank].recv_ops:
                send = next(
                    op for op in plans[recv.peer_rank].send_ops if op.task_id == recv.task_id
                )
                output[recv.my_slice].copy_(sources[recv.peer_rank][send.my_slice])
            torch.testing.assert_close(output, full + publication * 10, rtol=0, atol=0)


@pytest.mark.parametrize(
    "mapping,world",
    [
        ({}, 4),
        ({False: 0}, 1),
        ({0: 2, 1: 2}, 4),
        ({0: 2, 1: 4}, 4),
        ({0: 2, 2: 3}, 4),
        ({0: 2}, True),
    ],
)
def test_rank_remap_rejects_invalid_placement(mapping, world):
    with pytest.raises(ValueError):
        remap_metadata([], mapping, learner_world_size=world)
