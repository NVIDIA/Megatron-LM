# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU tests for whole-parameter ownership across optimizer instances.

The Gloo test exercises real reduce-scatter, inter-instance all-reduce, optimizer
updates, and instance-local parameter gathers. It uses AdamW to test the generic
LayerWise wrapper without the CUDA-only emerging-optimizers dependency.
"""

import os
from datetime import timedelta
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import megatron.core.optimizer as optimizer_factory
from megatron.core.dist_checkpointing.mapping import ShardedTensor, is_main_replica
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig
from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
from megatron.core.optimizer.layer_wise_optimizer import (
    LayerWiseDistributedOptimizer,
    _all_gather_param_group_metadata,
    _fold_replica_id,
)
from megatron.core.optimizer.optimizer import ChainedOptimizer, FP32Optimizer, MegatronOptimizer
from megatron.core.process_groups_config import ProcessGroupCollection


class _Group:
    def __init__(self, size, rank=0):
        self._size = size
        self._rank = rank

    def size(self):
        return self._size

    def rank(self):
        return self._rank


class _CPUFP32Optimizer(FP32Optimizer):
    """Use the real FP32 optimizer methods with a CPU loss-scale constant."""

    def __init__(self, optimizer, config):
        MegatronOptimizer.__init__(self, optimizer, config)
        self._scale = torch.ones(1)
        self.is_stub_optimizer = False


def _params():
    generator = torch.Generator().manual_seed(7)
    params = [
        torch.nn.Parameter(torch.randn(shape, generator=generator))
        for shape in ((4, 3), (3, 2), (2, 3), (2, 2))
    ]
    for param in params:
        param.is_managed_by_layer_wise_optimizer = True
    return params


@pytest.mark.parametrize("num_instances", [1, 2, 3])
@pytest.mark.parametrize("local_size", [1, 2])
def test_ownership_repeats_within_each_optimizer_instance(num_instances, local_size):
    assignments = []
    for instance in range(num_instances):
        instance_assignments = []
        for local_rank in range(local_size):
            params = _params()
            ddp_config = DistributedDataParallelConfig(
                use_distributed_optimizer=True, num_distributed_optimizer_instances=num_instances
            )
            layout = LayerWiseDistributedOptimizer.compute_full_param_layout(
                params, None, local_size, ddp_config
            )
            local = _Group(local_size, local_rank)
            groups = ProcessGroupCollection(
                tp=None,
                dp_cp=_Group(local_size * num_instances, instance * local_size + local_rank),
                expt_dp=None,
                intra_dp_cp=local,
                intra_expt_dp=None,
                intra_dist_opt=local,
                inter_dist_opt=_Group(num_instances, instance),
            )
            base = torch.optim.AdamW(params, lr=0.01)
            chunk = SimpleNamespace(ddp_config=ddp_config, full_param_layout=layout)
            with (
                mock.patch(
                    "megatron.core.optimizer.layer_wise_optimizer.get_pg_size",
                    side_effect=lambda group: group.size() if group is not None else 1,
                ),
                mock.patch(
                    "megatron.core.optimizer.layer_wise_optimizer.get_pg_rank",
                    side_effect=lambda group: group.rank() if group is not None else 0,
                ),
            ):
                optimizer = LayerWiseDistributedOptimizer(
                    [base], OptimizerConfig(), groups, model_chunks=[chunk]
                )
            assert optimizer.dp_cp is local
            assert optimizer.grad_stats_parallel_group is local
            assert optimizer.distributed_optimizer_instance_id == instance
            owned = {id(param) for group in base.param_groups for param in group["params"]}
            instance_assignments.append({i for i, param in enumerate(params) if id(param) in owned})
        assert set.union(*instance_assignments) == set(range(4))
        assert sum(map(len, instance_assignments)) == 4
        assignments.append(instance_assignments)
    assert all(assignment == assignments[0] for assignment in assignments)


@pytest.mark.parametrize("is_expert", [False, True])
def test_checkpoint_metadata_stays_within_instance(is_expert):
    local = _Group(2)
    groups = ProcessGroupCollection(
        dp_cp=_Group(6), expt_dp=_Group(6), intra_dp_cp=local, intra_expt_dp=local
    )
    with (
        mock.patch("megatron.core.optimizer.layer_wise_optimizer.get_pg_size", return_value=2),
        mock.patch.object(dist, "all_gather_object") as gather,
    ):
        metadata = {"is_expert_parallel": is_expert, "params": True}
        result = _all_gather_param_group_metadata(metadata, groups)
    gather.assert_called_once_with(result, metadata, group=local)
    assert len(result) == 2


@pytest.mark.parametrize("instance", [0, 1, 2])
def test_checkpoint_replica_id_preserves_optimizer_instance(instance):
    assert _fold_replica_id((0, 0, 7), "weight", {}, instance) == (0, 0, instance)
    assert _fold_replica_id((0, 1, 7), "weight", {"weight": (1, 2)}, instance) == (0, 3, instance)


def test_reject_layout_sized_for_full_dp_instead_of_instance():
    params = _params()
    ddp_config = DistributedDataParallelConfig(use_distributed_optimizer=True)
    layout = LayerWiseDistributedOptimizer.compute_full_param_layout(params, None, 4, ddp_config)
    optimizer = object.__new__(LayerWiseDistributedOptimizer)
    optimizer.dp_cp = _Group(2)
    optimizer.expt_dp = None
    with (
        mock.patch("megatron.core.optimizer.layer_wise_optimizer.get_pg_size", return_value=2),
        mock.patch("megatron.core.optimizer.layer_wise_optimizer.get_pg_rank", return_value=0),
        pytest.raises(ValueError, match="intra-instance ownership group"),
    ):
        optimizer.shard_params([torch.optim.AdamW(params)], [layout])


def test_factory_routes_fallback_domains_and_instance_id():
    config = OptimizerConfig(optimizer="muon", use_layer_wise_distributed_optimizer=True)
    matrix, dense_bias, expert_bias = [
        torch.nn.Parameter(torch.ones(shape)) for shape in ((2, 2), (2,), (2,))
    ]
    matrix.is_managed_by_layer_wise_optimizer = True
    dense_bias.is_managed_by_layer_wise_optimizer = False
    expert_bias.is_managed_by_layer_wise_optimizer = False
    dense_buffer = SimpleNamespace(params=[dense_bias])
    expert_buffer = SimpleNamespace(params=[expert_bias])
    chunk = SimpleNamespace(
        ddp_config=DistributedDataParallelConfig(
            use_distributed_optimizer=True, num_distributed_optimizer_instances=3
        ),
        named_parameters=lambda: iter(()),
        buffers=[SimpleNamespace(params=[matrix]), dense_buffer],
        expert_parallel_buffers=[expert_buffer],
    )
    groups = ProcessGroupCollection(mp=None, tp_ep_pp=None)
    process_groups = {
        "inter_dist_opt_group": _Group(3, 2),
        "intra_dp_cp_group": _Group(2),
        "intra_expt_dp_group": _Group(1),
        "intra_dp_cp_group_gloo": None,
        "intra_expt_dp_group_gloo": None,
        "mp_group": _Group(1),
        "expt_tp_pp_with_egtp_remat_group": _Group(2),
        "intra_dist_opt_group": _Group(2),
    }
    param_groups = [
        {"params": [matrix], "is_expert_parallel": False},
        {"params": [dense_bias], "is_expert_parallel": False, "optimizer": "adam"},
        {"params": [expert_bias], "is_expert_parallel": True, "optimizer": "adam"},
    ]
    result = SimpleNamespace(config=config, model_chunks=[])
    with (
        mock.patch.object(optimizer_factory, "HAVE_EMERGING_OPTIMIZERS", True),
        mock.patch.dict(
            optimizer_factory._EMERGING_OPTIMIZERS,
            {"muon": SimpleNamespace(default_param_overrides={})},
        ),
        mock.patch.object(optimizer_factory, "_get_param_groups", return_value=param_groups),
        mock.patch.object(
            optimizer_factory, "_create_emerging_optimizer", return_value=(mock.Mock(), None)
        ),
        mock.patch.object(optimizer_factory, "LayerWiseDistributedOptimizer", return_value=result),
        mock.patch.object(
            optimizer_factory, "_get_megatron_optimizer_based_on_param_groups", return_value=result
        ) as fallback,
        mock.patch.object(optimizer_factory, "get_pg_rank", side_effect=lambda group: group.rank()),
        mock.patch.object(
            ProcessGroupCollection,
            "setup_process_groups_for_optimizer",
            return_value=process_groups,
        ),
    ):
        optimizer_factory._get_megatron_emerging_optimizer(config, [chunk], {}, groups)
    dense, expert = [call.kwargs for call in fallback.call_args_list]
    assert dense["per_model_buffers"] == {0: [dense_buffer]}
    assert expert["per_model_buffers"] == {0: [expert_buffer]}
    assert dense["data_parallel_group"] is process_groups["intra_dp_cp_group"]
    assert expert["data_parallel_group"] is process_groups["intra_expt_dp_group"]
    assert dense["distributed_optimizer_instance_id"] == 2
    assert expert["distributed_optimizer_instance_id"] == 2


def _gloo_worker(rank, world_size, num_instances, init_file):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=60),
    )
    try:
        local_size = world_size // num_instances
        instance_groups = [
            dist.new_group(list(range(i * local_size, (i + 1) * local_size)), backend="gloo")
            for i in range(num_instances)
        ]
        inter_groups = [
            dist.new_group(list(range(i, world_size, local_size)), backend="gloo")
            for i in range(local_size)
        ]
        instance, local_rank = divmod(rank, local_size)
        local, inter = instance_groups[instance], inter_groups[local_rank]
        groups = ProcessGroupCollection(
            tp=None,
            dp_cp=dist.group.WORLD,
            expt_dp=None,
            intra_dp_cp=local,
            intra_expt_dp=None,
            intra_dist_opt=local,
            inter_dist_opt=inter,
        )
        params = _params()
        reference = [torch.nn.Parameter(param.detach().clone()) for param in params]
        config = OptimizerConfig(lr=0.01, clip_grad=0.0)
        ddp_config = DistributedDataParallelConfig(
            use_distributed_optimizer=True, num_distributed_optimizer_instances=num_instances
        )
        layout = LayerWiseDistributedOptimizer.compute_full_param_layout(
            params, None, local_size, ddp_config
        )
        (buffer_layout,) = layout.layouts.values()
        assert len(buffer_layout.bucket_indices) == 1
        start, end = buffer_layout.bucket_indices[0]
        param_buffer = torch.zeros(end - start)
        grad_buffer = torch.zeros_like(param_buffer)
        for param in params:
            begin, finish, _ = buffer_layout.param_index_map[param]
            view = param_buffer[begin:finish].view_as(param)
            view.copy_(param.detach())
            param.data = view
            param.main_grad = grad_buffer[begin:finish].view_as(param)
        shard_size = param_buffer.numel() // local_size
        owned_slice = slice(local_rank * shard_size, (local_rank + 1) * shard_size)
        bucket = SimpleNamespace(params_list=params)
        gathers = []

        def gather(bucket_group, force_sync):
            assert bucket_group.buckets[0] is bucket
            send = param_buffer[owned_slice].clone()
            dist.all_gather_into_tensor(param_buffer, send, group=local)
            gathers.append(True)

        chunk = SimpleNamespace(
            ddp_config=ddp_config,
            full_param_layout=layout,
            bucket_groups=[SimpleNamespace(buckets=[bucket])],
            expert_parallel_bucket_groups=[],
            _start_bucket_group_param_sync=gather,
        )
        base = torch.optim.AdamW(params, lr=0.01, foreach=False)
        wrapped = _CPUFP32Optimizer(base, config)
        optimizer = LayerWiseDistributedOptimizer([wrapped], config, groups, model_chunks=[chunk])
        reference_optimizer = torch.optim.AdamW(reference, lr=0.01, foreach=False)
        for step in range(3):
            grad_buffer.zero_()
            for index, (param, ref) in enumerate(zip(params, reference)):
                # Rank-dependent gradients expose missing/duplicated instance contributions.
                contribution = (rank + 1) * (index + 1) + step * 0.5
                param.main_grad.fill_(contribution / world_size)
                ref.grad = torch.full_like(ref, (world_size + 1) / 2 * (index + 1) + step * 0.5)
            local_gradient = torch.empty(shard_size)
            dist.reduce_scatter_tensor(local_gradient, grad_buffer, group=local)
            dist.all_reduce(local_gradient, group=inter)
            # Poison non-owned gradient regions; no optimizer may consume them.
            grad_buffer.fill_(float("nan"))
            grad_buffer[owned_slice].copy_(local_gradient)
            assert not optimizer.prepare_grads()
            assert optimizer.step_with_ready_grads()
            reference_optimizer.step()
            for actual, expected in zip(params, reference):
                torch.testing.assert_close(actual, expected, rtol=0, atol=1e-7)
            for param, state in base.state.items():
                index = next(i for i, candidate in enumerate(params) if candidate is param)
                for name in ("exp_avg", "exp_avg_sq"):
                    torch.testing.assert_close(
                        state[name], reference_optimizer.state[reference[index]][name]
                    )
        assert len(gathers) == 3
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(("world_size", "num_instances"), [(2, 1), (4, 2), (3, 3)])
def test_gloo_updates_match_full_gradient_reference(tmp_path, world_size, num_instances):
    if int(os.environ.get("WORLD_SIZE", "1")) > 1 or dist.is_initialized():
        pytest.skip("Run standalone CPU/Gloo cases with the CPU runner")
    mp.spawn(
        _gloo_worker,
        args=(world_size, num_instances, str(tmp_path / "store")),
        nprocs=world_size,
        join=True,
    )


@pytest.mark.parametrize("instance", [0, 1, 2])
def test_sharded_state_dict_has_one_writer_across_instances(instance):
    tensor = ShardedTensor.from_rank_offsets(
        "optimizer.momentum", torch.ones(2, 2), replica_id=(0, 0, 9)
    )
    state = {"optimizer": {"state": {"momentum": tensor}, "param_groups": []}}
    optimizer = object.__new__(LayerWiseDistributedOptimizer)
    optimizer.distributed_optimizer_instance_id = instance
    optimizer.chained_optimizers = [object()]
    optimizer.pg_collection = None
    optimizer.model_chunks = []
    with (
        mock.patch.object(ChainedOptimizer, "sharded_state_dict", return_value=state),
        mock.patch(
            "megatron.core.optimizer.layer_wise_optimizer._build_gtp_replica_fold", return_value={}
        ),
    ):
        optimizer.sharded_state_dict({})
    assert tensor.replica_id == (0, 0, instance)
    assert is_main_replica(tensor.replica_id) == (instance == 0)


@pytest.mark.parametrize("expert_rank", [0, 1])
def test_expert_ownership_when_dense_instance_has_one_rank(expert_rank):
    params = _params()
    for param in params:
        param.allreduce = False
    ddp_config = DistributedDataParallelConfig(use_distributed_optimizer=True)
    layout = LayerWiseDistributedOptimizer.compute_full_param_layout(
        params, None, 1, ddp_config, expert_data_parallel_world_size=2
    )
    expert = _Group(2, expert_rank)
    groups = ProcessGroupCollection(
        tp=None,
        dp_cp=None,
        expt_dp=expert,
        intra_dp_cp=None,
        intra_expt_dp=expert,
        intra_dist_opt=expert,
    )
    base = torch.optim.AdamW([{"params": params, "is_expert_parallel": True}])
    with (
        mock.patch(
            "megatron.core.optimizer.layer_wise_optimizer.get_pg_size",
            side_effect=lambda group: group.size() if group is not None else 1,
        ),
        mock.patch(
            "megatron.core.optimizer.layer_wise_optimizer.get_pg_rank",
            side_effect=lambda group: group.rank() if group is not None else 0,
        ),
    ):
        optimizer = LayerWiseDistributedOptimizer(
            [base],
            OptimizerConfig(),
            groups,
            model_chunks=[SimpleNamespace(ddp_config=ddp_config, full_param_layout=layout)],
        )
    assert optimizer.expt_dp is expert
    assert 0 < len(base.param_groups[0]["params"]) < len(params)


@pytest.mark.parametrize("owner", ["dense", "expert"])
def test_scalar_optimizer_gathers_only_its_owned_buffers(owner):
    dense, expert, layerwise = [SimpleNamespace(params_list=[]) for _ in range(3)]
    dense_group, expert_group, layerwise_group = [
        SimpleNamespace(buckets=[bucket]) for bucket in (dense, expert, layerwise)
    ]
    gather = mock.Mock()
    chunk = SimpleNamespace(
        bucket_groups=[dense_group, layerwise_group],
        expert_parallel_bucket_groups=[expert_group],
        _start_bucket_group_param_sync=gather,
    )
    optimizer = object.__new__(DistributedOptimizer)
    optimizer.is_stub_optimizer = False
    optimizer.model_chunks = [chunk]
    optimizer.buffers = [SimpleNamespace(buckets=[dense if owner == "dense" else expert])]
    optimizer.start_param_sync_for_bucket_group_subset()
    gather.assert_called_once_with(
        dense_group if owner == "dense" else expert_group, force_sync=False
    )
