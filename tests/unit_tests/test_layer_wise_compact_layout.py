# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Multi-GPU tests of the compact LayerWise layout next to the padded and legacy layouts.

The compact layout keeps Muon-managed matrices in unpadded, replicated DDP buffers (gradient
all-reduce, one owner rank per matrix, uneven parameter all-gather) while a sibling
DistributedOptimizer shards the remaining parameters. Each test trains a small model for a few
steps with real NCCL collectives and checks the DDP structure, rank-consistent ownership and the
updates against a non-distributed Muon reference.
"""

import os

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging.version import Version

from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
from megatron.core.optimizer.layer_wise_optimizer import LayerWiseDistributedOptimizer
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer import TransformerConfig
from megatron.core.utils import get_pg_rank, get_pg_size
from tests.unit_tests.test_utilities import Utils

pytestmark = [
    pytest.mark.skipif(
        Version(os.getenv('NVIDIA_PYTORCH_VERSION', "24.01")) <= Version("25.05"),
        reason="Skip layer-wise optimizer for LTS test",
    ),
    pytest.mark.skipif(
        int(os.getenv('WORLD_SIZE', '1')) < 2, reason="Multi-rank test requires WORLD_SIZE > 1"
    ),
]


class _DenseAndExpertModel(nn.Module):
    """Dense and expert matrices (Muon) with biases (Adam) and an excluded matrix (Adam)."""

    def __init__(self, with_experts=False, hidden=64):
        super().__init__()
        self.fc1 = nn.Linear(32, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 48)
        self.gate = nn.Linear(48, 48, bias=False)
        self.gate.weight.use_muon = False
        self.out = nn.Linear(48, 16)
        self.with_experts = with_experts
        if with_experts:
            # Named "experts" so the Muon factory tags them for the expert TP group.
            self.experts = nn.ModuleList([nn.Linear(48, 48) for _ in range(3)])
            for param in self.experts.parameters():
                param.allreduce = False

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = x + torch.tanh(self.gate(x))
        if self.with_experts:
            for expert in self.experts:
                x = x + 0.5 * torch.tanh(expert(x))
        return self.out(x)


def _layer_wise(optimizer):
    if isinstance(optimizer, LayerWiseDistributedOptimizer):
        return optimizer
    for sub_optimizer in getattr(optimizer, 'chained_optimizers', [optimizer]):
        if isinstance(sub_optimizer, LayerWiseDistributedOptimizer):
            return sub_optimizer
    raise AssertionError("No LayerWiseDistributedOptimizer in the optimizer chain")


def _distopts(optimizer):
    return [
        sub_optimizer
        for sub_optimizer in getattr(optimizer, 'chained_optimizers', [optimizer])
        if isinstance(sub_optimizer, DistributedOptimizer)
    ]


def _build(
    layout,
    *,
    with_experts=False,
    overlap=False,
    bucket_size=None,
    grad_reduce_in_fp32=True,
    seed=1234,
    clip_grad=1.0,
):
    """Return (ddp_model, optimizer) for a LayerWise layout or the ``"reference"`` setup."""
    from megatron.training.training import wrap_model_chunks_with_ddp

    torch.manual_seed(seed)
    model = _DenseAndExpertModel(with_experts=with_experts).bfloat16().cuda()
    reference = layout == "reference"
    ddp_config = DistributedDataParallelConfig(
        overlap_grad_reduce=overlap,
        overlap_param_gather=overlap and not reference,
        grad_reduce_in_fp32=grad_reduce_in_fp32,
        bucket_size=bucket_size,
        layer_wise_param_layout=None if reference else layout,
    )
    model = wrap_model_chunks_with_ddp(
        [model],
        TransformerConfig(num_attention_heads=1, num_layers=1),
        ddp_config,
        use_layer_wise_distributed_optimizer=not reference,
    )[0]
    optimizer = get_megatron_optimizer(
        OptimizerConfig(
            optimizer='muon',
            lr=0.02,
            weight_decay=0.01,
            bf16=True,
            clip_grad=clip_grad,
            muon_tp_mode='duplicated',
            # Muon + distributed optimizer is the layer-wise distributed Muon.
            use_distributed_optimizer=not reference,
        ),
        [model],
        pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
        use_gloo_process_groups=False,
    )
    return model, optimizer


def _train_step(model, optimizer, step):
    """One step with per-DP-rank data; returns the grad norm after params are published."""
    dp_rank = parallel_state.get_data_parallel_rank(with_context_parallel=True)
    generator = torch.Generator(device='cuda').manual_seed(100 * step + dp_rank)
    x = torch.randn(8, 32, generator=generator, device='cuda', dtype=torch.bfloat16)
    model.zero_grad_buffer()
    loss = model(x).float().pow(2).mean()
    loss.backward()
    model.finish_grad_sync()
    success, grad_norm, _ = optimizer.step()
    assert success
    if model.ddp_config.overlap_param_gather:
        # The next forward pre-hook would gather; publish now to compare parameters.
        model.start_param_sync(force_sync=True)
    return float(grad_norm)


def _assert_replicas_equal(model):
    """Every rank of a parameter's data-parallel group holds the same value."""
    for name, param in model.module.named_parameters():
        expert = not getattr(param, 'allreduce', True)
        group = (
            parallel_state.get_expert_data_parallel_group()
            if expert
            else parallel_state.get_data_parallel_group(with_context_parallel=True)
        )
        replicas = [torch.empty_like(param) for _ in range(get_pg_size(group))]
        torch.distributed.all_gather(replicas, param.data.contiguous(), group=group)
        for replica in replicas[1:]:
            torch.testing.assert_close(replica, replicas[0], rtol=0, atol=0, msg=name)


def _owner_names(model, layer_wise):
    names = {param: name for name, param in model.module.named_parameters()}
    owners = {}
    for params_list in (layer_wise.dp_cp_params_list, layer_wise.expt_dp_params_list):
        for rank, owned in enumerate(params_list or []):
            for param in owned:
                owners.setdefault(names[param], []).append(rank)
    return owners


class TestCompactLayerWiseLayout:

    @pytest.fixture(autouse=True)
    def model_parallel(self):
        Utils.initialize_model_parallel()
        yield
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("overlap", [False, True])
    def test_compact_ddp_structure_and_owners(self, overlap):
        model, optimizer = _build("compact", overlap=overlap, bucket_size=4096 if overlap else None)
        layer_wise = _layer_wise(optimizer)
        assert layer_wise.param_layout == "compact"
        assert not layer_wise.use_buffer_param_sync
        assert layer_wise.layerwise_param_sync_via_bucket_group == overlap
        # The model-level config keeps DistOpt for the sibling optimizer.
        assert model.ddp_config.use_distributed_optimizer
        assert model.ddp_config.layer_wise_param_layout == "compact"

        layer_wise_buffers = 0
        for buffer in model.buffers + model.expert_parallel_buffers:
            if getattr(buffer.params[0], 'is_managed_by_layer_wise_optimizer', False):
                layer_wise_buffers += 1
                assert not buffer.ddp_config.use_distributed_optimizer
                assert buffer.param_data is None
                assert buffer.numel == buffer.numel_unpadded
            else:
                assert buffer.ddp_config.use_distributed_optimizer
                assert buffer.param_data is not None
        assert layer_wise_buffers > 0
        for bucket_group in model.bucket_groups + model.expert_parallel_bucket_groups:
            owners = {
                getattr(param, 'is_managed_by_layer_wise_optimizer', False)
                for bucket in bucket_group.buckets
                for param in bucket.params_list
            }
            assert len(owners) == 1, "A bucket group mixes LayerWise and DistOpt buckets"
            assert bucket_group.ddp_config.use_distributed_optimizer != owners.pop()

        # Adam params (biases, embeddings-like and excluded matrices) are byte-sharded.
        distopts = _distopts(optimizer)
        assert len(distopts) == 1
        distopt_params = {p for buffer in distopts[0].buffers for p in buffer.params}
        assert model.module.gate.weight in distopt_params
        assert model.module.fc1.bias in distopt_params
        assert model.module.fc1.weight not in distopt_params

        # Every Muon matrix has exactly one owner, and every rank agrees on it.
        owners = _owner_names(model, layer_wise)
        assert sorted(owners) == ['fc1.weight', 'fc2.weight', 'fc3.weight', 'out.weight']
        assert all(len(ranks) == 1 for ranks in owners.values())
        gathered = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(gathered, owners)
        assert all(other == owners for other in gathered)

        _train_step(model, optimizer, step=0)
        _assert_replicas_equal(model)

    @pytest.mark.parametrize("layout", ["padded", "compact", "legacy"])
    @pytest.mark.parametrize("overlap", [False, True])
    def test_layouts_match_reference_updates(self, layout, overlap):
        model, optimizer = _build(layout, overlap=overlap, bucket_size=4096 if overlap else None)
        reference, reference_optimizer = _build("reference")
        for param, ref_param in zip(model.module.parameters(), reference.module.parameters()):
            torch.testing.assert_close(param, ref_param, rtol=0, atol=0)

        for step in range(3):
            grad_norm = _train_step(model, optimizer, step)
            reference_grad_norm = _train_step(reference, reference_optimizer, step)
            _assert_replicas_equal(model)
            assert grad_norm == pytest.approx(reference_grad_norm, rel=1e-4, abs=1e-5)
            for (name, param), ref_param in zip(
                model.module.named_parameters(), reference.module.parameters()
            ):
                # Reductions differ in order across layouts; one BF16 ulp of slack.
                torch.testing.assert_close(param, ref_param, rtol=8e-3, atol=1e-4, msg=name)


class TestCompactLayerWiseExperts:

    @pytest.fixture(autouse=True)
    def model_parallel(self):
        world = int(os.getenv('WORLD_SIZE', '1'))
        if world % 2:
            pytest.skip("Needs an even number of ranks for expert parallelism")
        Utils.initialize_model_parallel(expert_model_parallel_size=2)
        yield
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("layout", ["padded", "compact", "legacy"])
    def test_expert_params_shard_over_expert_dp(self, layout):
        model, optimizer = _build(layout, with_experts=True)
        layer_wise = _layer_wise(optimizer)
        expt_dp_size = get_pg_size(parallel_state.get_expert_data_parallel_group())
        if expt_dp_size > 1:
            assert layer_wise.expt_dp_params_list is not None
            expert_owned = {p for owned in layer_wise.expt_dp_params_list for p in owned}
            expected = {expert.weight for expert in model.module.experts}
            if layout == "legacy":
                # The legacy layout also keeps the Adam-managed expert biases in LayerWise.
                expected |= {expert.bias for expert in model.module.experts}
            assert expert_owned == expected

        if layout != "legacy":
            # Expert biases are Adam params on expert-parallel buffers: an expert-DP DistOpt.
            distopts = _distopts(optimizer)
            assert len(distopts) == 2
            expert_distopt = next(
                d
                for d in distopts
                if any(not getattr(p, 'allreduce', True) for b in d.buffers for p in b.params)
            )
            assert get_pg_size(expert_distopt.data_parallel_group) == expt_dp_size
            assert {p for b in expert_distopt.buffers for p in b.params} == {
                expert.bias for expert in model.module.experts
            }

        reference, reference_optimizer = _build("reference", with_experts=True)
        for step in range(3):
            grad_norm = _train_step(model, optimizer, step)
            reference_grad_norm = _train_step(reference, reference_optimizer, step)
            _assert_replicas_equal(model)
            assert grad_norm == pytest.approx(reference_grad_norm, rel=1e-4, abs=1e-5)
        for (name, param), ref_param in zip(
            model.module.named_parameters(), reference.module.parameters()
        ):
            torch.testing.assert_close(param, ref_param, rtol=8e-3, atol=1e-4, msg=name)


class TestSingleRankDenseGroup:
    """TP spans all ranks, so dp_cp has one rank while expert DP spans all of them."""

    @pytest.fixture(autouse=True)
    def model_parallel(self):
        world = int(os.getenv('WORLD_SIZE', '1'))
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=world, expert_tensor_parallel_size=1
        )
        yield
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("layout", ["padded", "compact", "legacy"])
    def test_experts_are_owned_once(self, layout):
        assert get_pg_size(parallel_state.get_data_parallel_group(with_context_parallel=True)) == 1
        expt_dp = parallel_state.get_expert_data_parallel_group()
        assert get_pg_size(expt_dp) > 1
        model, optimizer = _build(layout, with_experts=True)
        layer_wise = _layer_wise(optimizer)
        assert layer_wise.dp_cp_params_list is None
        assert layer_wise.expt_dp_params_list is not None
        owned_here = layer_wise.expt_dp_params_list[get_pg_rank(expt_dp)]
        for expert in model.module.experts:
            owners = [None] * get_pg_size(expt_dp)
            owned = any(expert.weight is p for p in owned_here)
            torch.distributed.all_gather_object(owners, owned, group=expt_dp)
            assert sum(owners) == 1, "Each expert matrix must have exactly one owner"

        reference, reference_optimizer = _build("reference", with_experts=True)
        for step in range(2):
            grad_norm = _train_step(model, optimizer, step)
            reference_grad_norm = _train_step(reference, reference_optimizer, step)
            assert grad_norm == pytest.approx(reference_grad_norm, rel=1e-4, abs=1e-5)
        for param, ref_param in zip(model.module.parameters(), reference.module.parameters()):
            torch.testing.assert_close(param, ref_param, rtol=8e-3, atol=1e-4)


def _assert_training_replays_bit_exact(layout, **build_kwargs):
    """Train the same model twice for three steps; grad norms and parameters must match."""
    from tests.unit_tests.determinism.kernels.harness import bytes_equal

    runs = []
    for _ in range(2):
        model, optimizer = _build(layout, **build_kwargs)
        norms = [_train_step(model, optimizer, step) for step in range(3)]
        runs.append((norms, [p.detach().clone() for p in model.module.parameters()]))
    assert runs[0][0] == runs[1][0], "grad norms differ between replays"
    for (name, _), a, b in zip(model.module.named_parameters(), runs[0][1], runs[1][1]):
        assert bytes_equal(a, b), f"{name} differs between replays"


class TestCompactReplay:
    """Bit-exact replay of the compact buffers: gradient all-reduce next to the sibling
    DistOpt reduce-scatter, rank-consistent owners and uneven parameter gathers."""

    @pytest.fixture(autouse=True)
    def model_parallel(self):
        Utils.initialize_model_parallel()
        yield
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("overlap", [False, True])
    def test_compact_training_replays_bit_exact(self, overlap):
        _assert_training_replays_bit_exact(
            "compact", overlap=overlap, bucket_size=4096 if overlap else None
        )


class TestExpertReplay:
    """Bit-exact replay with expert-parallel buffers: LayerWise owns the expert matrices over
    expert DP and a second sibling DistOpt shards the expert biases over expert DP."""

    @pytest.fixture(autouse=True)
    def model_parallel(self):
        if int(os.getenv('WORLD_SIZE', '1')) % 2:
            pytest.skip("Needs an even number of ranks for expert parallelism")
        Utils.initialize_model_parallel(expert_model_parallel_size=2)
        yield
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("layout", ["padded", "compact"])
    def test_expert_training_replays_bit_exact(self, layout):
        _assert_training_replays_bit_exact(layout, with_experts=True)
