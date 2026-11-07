# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import os
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging.version import Version

from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.optimizer import ChainedOptimizer, OptimizerConfig, get_megatron_optimizer
from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
from megatron.core.optimizer.layer_wise_optimizer import LayerWiseDistributedOptimizer
from megatron.core.optimizer.optimizer import Float16OptimizerWithFloat16Params
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer import TransformerConfig
from megatron.core.utils import get_pg_rank, get_pg_size
from tests.unit_tests.test_utilities import Utils

# Skip all tests in this file for LTS versions
pytestmark = pytest.mark.skipif(
    Version(os.getenv('NVIDIA_PYTORCH_VERSION', "24.01")) <= Version("25.05"),
    reason="Skip layer-wise optimizer for LTS test",
)

# reduce-scatter selects the padded LayerWise layout, all-reduce the compact one.
GRAD_COLLECTIVES = ['reduce-scatter', 'all-reduce']

multi_rank = pytest.mark.skipif(
    int(os.getenv('WORLD_SIZE', '1')) == 1, reason="Multi-rank test requires WORLD_SIZE > 1"
)


class SimpleModel(nn.Module):
    """Simple model for testing LayerWiseDistributedOptimizer.

    Model with 5 layers to ensure more than 8 parameters (10 total: 5 weights + 5 biases).
    """

    def __init__(self, input_size=80, hidden_size=48, output_size=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 32)
        self.fc3 = nn.Linear(32, 24)
        self.fc4 = nn.Linear(24, 16)
        self.fc5 = nn.Linear(16, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x


class TinyModel(nn.Module):
    """Tiny model with only 1 layer (2 parameters: weight and bias)."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc1(x)


class MuonExcludedMatrixModel(nn.Module):
    """Model with a 2D matrix explicitly routed to the scalar optimizer."""

    def __init__(self):
        super().__init__()
        self.scalar = nn.Linear(10, 8)
        self.muon = nn.Linear(8, 5)
        self.scalar.weight.use_muon = False

    def forward(self, x):
        return self.muon(F.relu(self.scalar(x)))


class ExpertModel(nn.Module):
    """Dense and expert matrices (Muon) with biases (Adam); the experts are expert-parallel."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(80, 48)
        # Named "experts" so the Muon factory tags them for the expert TP group.
        self.experts = nn.ModuleList([nn.Linear(48, 48) for _ in range(3)])
        for param in self.experts.parameters():
            param.allreduce = False
        self.out = nn.Linear(48, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        for expert in self.experts:
            x = x + 0.5 * torch.tanh(expert(x))
        return self.out(x)


class _FakeLayerwiseBucket:
    def __init__(self, params):
        self.params_list = params
        self.params = set(params)
        self.layerwise_params_list = None
        self.layerwise_param_flat_sizes = None

    def set_layerwise_params_list(self, layerwise_params_list):
        self.layerwise_params_list = layerwise_params_list
        self.layerwise_param_flat_sizes = [
            sum(param.numel() for param in param_list) for param_list in layerwise_params_list
        ]


def test_set_bucket_layerwise_params_list_single_dp_rank():
    """Single-DP-rank LayerWise buckets should still receive param lists.

    Regression for #5203: shard_params() sets dp_cp_params_list to None when
    dp_cp_size == 1, but set_bucket_layerwise_params_list() still needs to
    initialize bucket.layerwise_params_list for the overlap param-gather path.
    """
    params = [torch.nn.Parameter(torch.empty(4, 4)), torch.nn.Parameter(torch.empty(2, 2))]
    for param in params:
        param.is_managed_by_layer_wise_optimizer = True
    bucket = _FakeLayerwiseBucket(params)

    optimizer = object.__new__(LayerWiseDistributedOptimizer)
    optimizer.pg_collection = ProcessGroupCollection(dp_cp=None, expt_dp=None)
    optimizer.dp_cp_params_list = None
    optimizer.expt_dp_params_list = None

    model_chunk = SimpleNamespace(
        bucket_groups=[SimpleNamespace(buckets=[bucket])], expert_parallel_bucket_groups=[]
    )
    optimizer.set_bucket_layerwise_params_list([model_chunk])

    assert bucket.layerwise_params_list == [params]
    assert bucket.layerwise_param_flat_sizes == [sum(param.numel() for param in params)]


def test_layerwise_step_hook_is_noop_without_chunked_offload():
    """Disabled offload must not inspect child-only manager attributes or rebuild pipeline state."""

    optimizer = object.__new__(LayerWiseDistributedOptimizer)
    optimizer.config = SimpleNamespace(chunked_optimizer_state_offload=False)
    optimizer.chained_optimizers = [object()]

    assert optimizer._managed_optimizer_state_offload_child_indices() == ()
    optimizer._before_child_step(0)


def test_layerwise_fraction_zero_skips_offload_specific_child_inspection(monkeypatch):
    """A zero fraction must bypass Muon schema inspection and offloader construction."""

    class FakeWrappedOptimizer:
        def __init__(self, optimizer, config, grad_scaler, init_state_fn):
            self.optimizer = optimizer
            self.config = config
            self._optimizer_state_offloader = None

    def fake_shard_params(self, optimizers, full_param_layouts=None, **kwargs):
        self.dp_cp_params_list = []
        self.expt_dp_params_list = []

    monkeypatch.setattr(
        "megatron.core.optimizer.layer_wise_optimizer.Float16OptimizerWithFloat16Params",
        FakeWrappedOptimizer,
    )
    monkeypatch.setattr(LayerWiseDistributedOptimizer, "shard_params", fake_shard_params)
    config = OptimizerConfig(
        optimizer="muon",
        use_layer_wise_distributed_optimizer=True,
        bf16=True,
        chunked_optimizer_state_offload=True,
        optimizer_state_offload_fraction=0.0,
        overlap_param_gather=False,
    )

    optimizer = LayerWiseDistributedOptimizer([object()], config)

    assert optimizer._managed_optimizer_state_offload_child_indices() == ()
    LayerWiseDistributedOptimizer._before_child_step(optimizer, 0)


def test_layerwise_offload_rejects_multiple_nonempty_muon_children():
    """The supported LayerWise offload schema has at most one local Muon child."""

    from megatron.core.optimizer.emerging_optimizers import TensorParallelMuon

    class FakeWrappedOptimizer:
        def __init__(self, optimizer):
            self.optimizer = optimizer
            self.fp32_from_float16_groups = [[]]
            self._optimizer_state_offloader = None

    wrapped_optimizers = []
    for _ in range(2):
        raw_optimizer = object.__new__(TensorParallelMuon)
        raw_optimizer.param_groups = [{"params": [object()]}]
        wrapped_optimizers.append(FakeWrappedOptimizer(raw_optimizer))

    # Offload is only supported on the compact layout.
    optimizer = LayerWiseDistributedOptimizer.__new__(LayerWiseDistributedOptimizer)
    optimizer.param_layout = "compact"
    optimizer.dp_cp_params_list = []
    optimizer.expt_dp_params_list = []
    optimizer.overlap_param_gather = False

    with pytest.raises(AssertionError, match="at most one non-empty TensorParallelMuon child"):
        optimizer._enable_chunked_optimizer_state_offload(wrapped_optimizers)


def test_layerwise_param_sync_subset_excludes_distopt_buckets():
    """Forced pre-forward sync must leave sibling DistributedOptimizer buckets untouched."""

    layerwise_param = SimpleNamespace(is_managed_by_layer_wise_optimizer=True)
    distopt_param = SimpleNamespace(is_managed_by_layer_wise_optimizer=False)
    layerwise_group = SimpleNamespace(buckets=[SimpleNamespace(params_list=[layerwise_param])])
    distopt_group = SimpleNamespace(buckets=[SimpleNamespace(params_list=[distopt_param])])
    calls = []
    model_chunk = SimpleNamespace(
        bucket_groups=[layerwise_group, distopt_group],
        expert_parallel_bucket_groups=[],
        _start_bucket_group_param_sync=lambda group, force_sync: calls.append((group, force_sync)),
    )
    optimizer = SimpleNamespace(model_chunks=[model_chunk])

    LayerWiseDistributedOptimizer.start_param_sync_for_bucket_group_subset(
        optimizer, force_sync=True
    )

    assert calls == [(layerwise_group, True)]


def test_outer_chain_only_syncs_children_requiring_master_before_offload():
    calls = []

    def child(name, requires_sync):
        return SimpleNamespace(
            optimizer_state_offload_requires_pre_forward_param_sync=lambda: requires_sync,
            start_param_sync_for_bucket_group_subset=lambda force_sync: calls.append(
                (name, force_sync)
            ),
        )

    optimizer = SimpleNamespace(
        chained_optimizers=[child("layerwise", True), child("distopt", False)]
    )
    ChainedOptimizer.start_param_sync_for_bucket_group_subset(optimizer, force_sync=True)
    assert calls == [("layerwise", True)]


def test_layerwise_prefetches_single_managed_child_state():
    """LayerWise prefetches its single managed child before the chained step."""

    events = []

    def child(name, managed):
        return SimpleNamespace(
            _optimizer_state_offloader=(
                SimpleNamespace(staging_pool=SimpleNamespace(hold=nullcontext)) if managed else None
            ),
            prefetch_optimizer_master_weights_for_step=lambda: events.append(f"master:{name}"),
            prefetch_optimizer_state_for_step=lambda: events.append(f"state:{name}"),
            step_with_ready_grads=lambda: events.append(f"step:{name}") or True,
        )

    children = [child("a", False), child("b", True), child("c", False)]
    optimizer = object.__new__(LayerWiseDistributedOptimizer)
    optimizer.chained_optimizers = children
    optimizer.config = SimpleNamespace(
        chunked_optimizer_state_offload=True,
        optimizer_state_offload_fraction=1.0,
        overlap_param_gather_with_optimizer_step=False,
    )
    optimizer._managed_optimizer_state_offload_indices = (1,)

    optimizer.prefetch_optimizer_state_for_gradient_finalization()
    assert events == ["master:a", "master:b", "master:c", "state:b"]

    events.clear()
    assert optimizer._step()
    assert events == ["state:b", "step:a", "step:b", "step:c"]


def _layer_wise(optimizer):
    """The LayerWiseDistributedOptimizer of an optimizer chain."""
    if isinstance(optimizer, LayerWiseDistributedOptimizer):
        return optimizer
    return next(
        sub_optimizer
        for sub_optimizer in optimizer.chained_optimizers
        if isinstance(sub_optimizer, LayerWiseDistributedOptimizer)
    )


def _create(
    model_class=SimpleModel,
    *,
    grad_collective='reduce-scatter',
    use_layer_wise=True,
    overlap_param_gather=False,
    optimizer_overlap_param_gather=None,
    grad_reduce_in_fp32=False,
    bucket_size=None,
    copy_from=None,
    frozen=False,
    seed=1234,
    chunked_optimizer_state_offload=False,
    optimizer_state_offload_chunk_size_mb=0,
    optimizer_state_offload_fraction=1.0,
):
    """Return ``(ddp_model, optimizer, pg_collection)``.

    ``use_layer_wise=False`` builds the reference: non-distributed Muon on plain DDP.
    ``optimizer_overlap_param_gather`` sets a conflicting OptimizerConfig value, to check that
    the DDP config stays authoritative.
    """
    torch.manual_seed(seed)
    model = model_class().bfloat16().cuda()
    model.requires_grad_(not frozen)
    # overlap_param_gather needs bucketing, which DDP only keeps with overlap_grad_reduce.
    ddp_config = DistributedDataParallelConfig(
        overlap_param_gather=overlap_param_gather,
        overlap_grad_reduce=overlap_param_gather,
        grad_reduce_in_fp32=grad_reduce_in_fp32,
        bucket_size=bucket_size,
    )
    transformer_config = TransformerConfig(num_attention_heads=1, num_layers=1)
    if use_layer_wise:
        from megatron.training.training import wrap_model_chunks_with_ddp

        model = wrap_model_chunks_with_ddp(
            [model],
            transformer_config,
            ddp_config,
            use_layer_wise_distributed_optimizer=True,
            second_order_optimizer_grad_collective=grad_collective,
        )[0]
    else:
        model = DistributedDataParallel(transformer_config, ddp_config, model)
    if copy_from is not None:
        model.module.load_state_dict(copy_from.module.state_dict())
    else:
        model.broadcast_params()

    if optimizer_overlap_param_gather is None:
        optimizer_overlap_param_gather = overlap_param_gather
    pg_collection = ProcessGroupCollection.use_mpu_process_groups()
    optimizer = get_megatron_optimizer(
        OptimizerConfig(
            optimizer='muon',
            lr=0.01,
            weight_decay=0.01,
            bf16=True,
            use_distributed_optimizer=False,
            clip_grad=1.0,
            overlap_param_gather=optimizer_overlap_param_gather,
            muon_tp_mode="duplicated",
            use_layer_wise_distributed_optimizer=use_layer_wise,
            chunked_optimizer_state_offload=chunked_optimizer_state_offload,
            optimizer_state_offload_chunk_size_mb=optimizer_state_offload_chunk_size_mb,
            optimizer_state_offload_fraction=optimizer_state_offload_fraction,
            second_order_optimizer_grad_collective=(
                grad_collective if use_layer_wise else 'reduce-scatter'
            ),
        ),
        [model],
        pg_collection=pg_collection,
        use_gloo_process_groups=False,
    )
    return model, optimizer, pg_collection


def _set_same_main_grads(seed, *models):
    """Give the models the same main_grad, equal on every rank."""
    generator = torch.Generator(device='cuda').manual_seed(seed)
    for params in zip(*(model.parameters() for model in models)):
        grad = torch.randn(params[0].shape, generator=generator, device='cuda')
        for param in params:
            param.main_grad = grad.clone()


def _train_step(model, optimizer, step):
    """One step with per-DP-rank data; returns the grad norm after params are published."""
    in_features = next(model.module.parameters()).shape[1]
    dp_rank = parallel_state.get_data_parallel_rank(with_context_parallel=True)
    generator = torch.Generator(device='cuda').manual_seed(100 * step + dp_rank)
    x = torch.randn(8, in_features, generator=generator, device='cuda', dtype=torch.bfloat16)
    model.zero_grad_buffer()
    model(x).float().pow(2).mean().backward()
    model.finish_grad_sync()
    success, grad_norm, _ = optimizer.step()
    assert success
    if model.ddp_config.overlap_param_gather:
        # The next forward pre-hook would gather; publish now to compare parameters.
        model.start_param_sync(force_sync=True)
    return float(grad_norm)


def _assert_params_equal(model, reference, msg=''):
    for (name, param), ref_param in zip(model.named_parameters(), reference.parameters()):
        torch.testing.assert_close(param, ref_param, rtol=0, atol=0, msg=f"{msg} {name}")


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


def _assert_trains_like_reference(model, optimizer, steps):
    """Train next to the non-distributed reference and compare grad norms and parameters."""
    reference, reference_optimizer, _ = _create(
        type(model.module),
        use_layer_wise=False,
        grad_reduce_in_fp32=model.ddp_config.grad_reduce_in_fp32,
        copy_from=model,
    )
    for step in range(steps):
        grad_norm = _train_step(model, optimizer, step)
        reference_grad_norm = _train_step(reference, reference_optimizer, step)
        assert grad_norm == pytest.approx(reference_grad_norm, rel=1e-4, abs=1e-5)
        _assert_replicas_equal(model)
    for (name, param), ref_param in zip(
        model.module.named_parameters(), reference.module.parameters()
    ):
        # Reductions differ in order across layouts; one BF16 ulp of slack.
        torch.testing.assert_close(param, ref_param, rtol=8e-3, atol=1e-4, msg=name)


@multi_rank
class TestLayerWiseOptimizer:
    """Test class for LayerWiseDistributedOptimizer with common setup code."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Setup and teardown for each test."""
        Utils.initialize_model_parallel()
        yield
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize('grad_collective', GRAD_COLLECTIVES)
    def test_basic(self, grad_collective):
        """Construction, bf16 wrapping and one training step; every rank ends equal."""
        model, optimizer, pg_collection = _create(grad_collective=grad_collective)
        layer_wise = _layer_wise(optimizer)
        assert isinstance(layer_wise.chained_optimizers[0], Float16OptimizerWithFloat16Params)
        assert layer_wise.grad_stats_parallel_group is pg_collection.intra_dist_opt
        for sub_optimizer in layer_wise.chained_optimizers:
            assert sub_optimizer.grad_stats_parallel_group is pg_collection.intra_dist_opt

        before = [param.detach().clone() for param in model.parameters()]
        _train_step(model, optimizer, step=0)
        assert any(not torch.equal(p, b) for p, b in zip(model.parameters(), before))
        _assert_replicas_equal(model)

    def test_compact_muon_chunked_state_and_master_offload(self):
        """Compact-layout Muon updates whole tensors while state and masters live on CPU."""

        model, optimizer, _ = _create(
            grad_collective='all-reduce',
            chunked_optimizer_state_offload=True,
            optimizer_state_offload_chunk_size_mb=1,
        )
        reference_model, reference_optimizer, _ = _create(
            grad_collective='all-reduce', copy_from=model
        )

        layerwise_optimizer = next(
            child
            for child in optimizer.chained_optimizers
            if isinstance(child, LayerWiseDistributedOptimizer)
        )
        managed_children = [
            child
            for child in layerwise_optimizer.chained_optimizers
            if child._optimizer_state_offloader is not None
        ]
        # Whole-parameter compact layout can leave some DP ranks without a local Muon
        # parameter. Those ranks legitimately have no LayerWise state offloader, but must
        # continue through the distributed optimizer step with the ranks that do own one.
        max_managed_children = torch.tensor(len(managed_children), dtype=torch.int, device='cuda')
        torch.distributed.all_reduce(max_managed_children, op=torch.distributed.ReduceOp.MAX)
        assert max_managed_children.item() > 0
        if managed_children:
            assert (
                len(
                    {id(child._optimizer_state_offloader._d2h_stream) for child in managed_children}
                )
                == 1
            )
            assert (
                len(
                    {id(child._optimizer_state_offloader._h2d_stream) for child in managed_children}
                )
                == 1
            )

        optimizer.offload_optimizer_state_for_forward()
        optimizer.zero_grad()

        for child in managed_children:
            manager = child._optimizer_state_offloader
            for param in manager.selected_params:
                if any(
                    id(param) == id(master)
                    for group in child.fp32_from_float16_groups
                    for master in group
                ):
                    assert param.device.type == 'cpu'

        inputs = torch.randn(16, 80, dtype=torch.bfloat16, device='cuda')
        model(inputs).sum().backward()
        optimizer.prefetch_optimizer_state_for_gradient_finalization()
        assert optimizer.step()[0]

        reference_model(inputs).sum().backward()
        assert reference_optimizer.step()[0]

        optimizer.offload_optimizer_state_for_forward()
        for child in managed_children:
            manager = child._optimizer_state_offloader
            for param in manager.selected_params:
                for value in manager.optimizer.state[param].values():
                    if isinstance(value, torch.Tensor) and value.numel() == param.numel():
                        assert value.device.type == 'cpu'
                if any(
                    id(param) == id(master)
                    for group in child.fp32_from_float16_groups
                    for master in group
                ):
                    assert param.device.type == 'cpu'

        for offloaded_param, reference_param in zip(
            model.parameters(), reference_model.parameters()
        ):
            torch.testing.assert_close(offloaded_param, reference_param, rtol=1e-4, atol=1e-5)

    def test_split_layerwise_and_distopt_share_offload_streams(self):
        """Muon children and sibling Adam DistOpt serialize copies on one stream pair."""

        _, optimizer, _ = _create(
            grad_collective='all-reduce',
            chunked_optimizer_state_offload=True,
            optimizer_state_offload_chunk_size_mb=1,
        )
        managers = []

        def collect(current_optimizer):
            manager = getattr(current_optimizer, '_optimizer_state_offloader', None)
            if manager is not None:
                managers.append(manager)
            for child in getattr(current_optimizer, 'chained_optimizers', ()):
                collect(child)

        collect(optimizer)
        max_manager_count = torch.tensor(len(managers), dtype=torch.int, device='cuda')
        torch.distributed.all_reduce(max_manager_count, op=torch.distributed.ReduceOp.MAX)
        assert max_manager_count.item() >= 2
        if len(managers) >= 2:
            assert len({id(manager.transfer_streams[0]) for manager in managers}) == 1
            assert len({id(manager.transfer_streams[1]) for manager in managers}) == 1

    @pytest.mark.parametrize('grad_collective', GRAD_COLLECTIVES)
    def test_frozen_stage_builds_an_optimizer(self, grad_collective):
        """A pipeline stage whose params are all frozen has no routing tags; building the
        layer-wise optimizer must not mistake it for a model wrapped without a layout."""
        _, optimizer, _ = _create(grad_collective=grad_collective, frozen=True)
        assert not any(group['params'] for group in _layer_wise(optimizer).param_groups)

    def test_explicit_muon_exclusion_updates_with_param_layout(self):
        """An excluded 2D matrix is updated and synchronized by the scalar DistOpt."""
        model, optimizer, _ = _create(MuonExcludedMatrixModel)
        initial_weight = model.module.scalar.weight.detach().clone()
        _train_step(model, optimizer, step=0)
        assert not torch.equal(model.module.scalar.weight, initial_weight)
        _assert_replicas_equal(model)

    def test_get_grad_norm(self):
        """The grad norm matches the non-distributed reference."""
        model, optimizer, _ = _create()
        reference, reference_optimizer, _ = _create(use_layer_wise=False)
        _set_same_main_grads(0, model, reference)
        optimizer.prepare_grads()
        reference_optimizer.prepare_grads()
        torch.testing.assert_close(
            optimizer.get_grad_norm(), reference_optimizer.get_grad_norm(), rtol=1e-5, atol=1e-5
        )

    def test_sharded_state_dict(self):
        """Test LayerWiseDistributedOptimizer sharded_state_dict method."""
        model, optimizer, _ = _create()

        for param in model.parameters():
            param.grad = torch.randn_like(param)
        optimizer.step()

        # Test sharded_state_dict, with the sibling DistOpt in its default checkpoint format.
        sharded_state_dict = optimizer.sharded_state_dict(
            model.sharded_state_dict(), metadata={'distrib_optim_sharding_type': 'dp_reshardable'}
        )

        # The chain (muon + adam) has integer top-level keys, each sub-dict an 'optimizer' key.
        assert sharded_state_dict is not None, "Sharded state dict should not be None"
        if isinstance(sharded_state_dict, dict) and all(
            isinstance(k, int) for k in sharded_state_dict.keys()
        ):
            for idx, sub_dict in sharded_state_dict.items():
                assert 'optimizer' in sub_dict, f"Sub-dict {idx} should contain 'optimizer' key"
        else:
            assert (
                'optimizer' in sharded_state_dict
            ), "Sharded state dict should contain 'optimizer' key"

        # Verify that replica_id is set correctly (should be 0 for DP dimension)
        from megatron.core.dist_checkpointing import ShardedTensor
        from megatron.core.dist_checkpointing.dict_utils import nested_values

        for sh_base in nested_values(sharded_state_dict):
            if isinstance(sh_base, ShardedTensor):
                assert (
                    len(sh_base.replica_id) == 3
                ), f'Expected replica_id format (PP, TP, DP), got: {sh_base.replica_id}'
                assert (
                    sh_base.replica_id[2] == 0
                ), f'Expected DP replica_id to be 0 for layer-wise optimizer, got: {sh_base.replica_id[2]}'

    def test_multiple_optimizers(self):
        """LayerWise over two base optimizers built directly, without DDP-wrapped chunks."""
        model = DistributedDataParallel(
            TransformerConfig(num_attention_heads=1, num_layers=1),
            DistributedDataParallelConfig(use_distributed_optimizer=False),
            SimpleModel().bfloat16().cuda(),
        )
        params = list(model.parameters())
        half = len(params) // 2
        optimizer = LayerWiseDistributedOptimizer(
            [torch.optim.Adam(params[:half], lr=0.01), torch.optim.Adam(params[half:], lr=0.01)],
            OptimizerConfig(optimizer='adam', lr=0.01, bf16=True, use_distributed_optimizer=False),
            ProcessGroupCollection.use_mpu_process_groups(),
        )
        assert len(optimizer.chained_optimizers) == 2
        for param in params:
            param.grad = torch.randn_like(param)
        assert optimizer.step()[0]

    @pytest.mark.parametrize('with_pg_collection', [True, False])
    def test_direct_construction_pg_collection(self, with_pg_collection):
        """The supplied TP groups are kept; without a collection the optimizer is local, so
        every rank owns and updates every param."""
        pg_collection = (
            ProcessGroupCollection.use_mpu_process_groups() if with_pg_collection else None
        )
        param = torch.nn.Parameter(torch.ones(4, 4, dtype=torch.bfloat16, device='cuda'))
        optimizer = LayerWiseDistributedOptimizer(
            [torch.optim.SGD([param], lr=0.5)],
            OptimizerConfig(lr=0.5, bf16=True, clip_grad=0.0),
            pg_collection,
        )
        if with_pg_collection:
            assert optimizer.tp_group is pg_collection.tp
            assert optimizer.expert_tp_group is pg_collection.expt_tp
        else:
            assert optimizer.tp_group is None and optimizer.expert_tp_group is None

        param.grad = torch.ones_like(param)
        assert optimizer.step()[0]
        assert torch.equal(param.detach(), torch.full_like(param.detach(), 0.5))

    def test_bf16_wrapper_uses_preserved_high_precision_initializer(self):
        """FP32 masters must not be initialized from an already-quantized model weight."""
        model_param = torch.nn.Parameter(
            torch.full((2, 2), -1.0, dtype=torch.bfloat16, device='cuda')
        )
        preserved_init = torch.full((2, 2), 1.25, dtype=torch.bfloat16)
        init_state = {'value': preserved_init}
        model_param.get_high_precision_init_val = lambda: init_state['value']
        model_param.clear_high_precision_init_val = lambda: init_state.update(value=None)

        base_optimizer = torch.optim.SGD([model_param], lr=0.1)
        wrapped_optimizer = Float16OptimizerWithFloat16Params(
            base_optimizer,
            OptimizerConfig(optimizer='sgd', lr=0.1, bf16=True),
            None,
            lambda opt, config: None,
        )

        main_param = wrapped_optimizer.fp32_from_float16_groups[0][0]
        torch.testing.assert_close(main_param, preserved_init.cuda().float(), rtol=0, atol=0)
        assert init_state['value'] is None
        assert model_param.main_param is main_param

    def test_bf16_error(self):
        """LayerWiseDistributedOptimizer rejects an already Float16-wrapped optimizer."""
        _, muon_optimizer, pg_collection = _create(use_layer_wise=False)
        wrapped_optimizer = muon_optimizer.chained_optimizers[0]
        assert isinstance(wrapped_optimizer, Float16OptimizerWithFloat16Params)
        config = OptimizerConfig(optimizer='muon', lr=0.01, bf16=True)
        with pytest.raises(
            TypeError, match='LayerWiseDistributedOptimizer expects base torch optimizers'
        ):
            LayerWiseDistributedOptimizer([wrapped_optimizer], config, pg_collection)

    @pytest.mark.parametrize('model_class', [SimpleModel, TinyModel])
    @pytest.mark.parametrize('grad_collective', GRAD_COLLECTIVES)
    def test_parameter_updates(self, grad_collective, model_class):
        """Updates match the non-distributed reference; TinyModel leaves ranks without params."""
        model, optimizer, _ = _create(model_class, grad_collective=grad_collective)
        reference, reference_optimizer, _ = _create(
            model_class, use_layer_wise=False, copy_from=model
        )
        before = [param.detach().clone() for param in model.parameters()]
        _set_same_main_grads(0, model, reference)
        optimizer.step()
        reference_optimizer.step()
        assert any(not torch.equal(p, b) for p, b in zip(model.parameters(), before))
        for param, ref_param in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(param, ref_param, rtol=1e-5, atol=1e-5)

    def test_broadcast_vs_allgather(self):
        """The whole-param all-gather matches the broadcast fallback bit for bit."""
        model, optimizer, _ = _create(grad_collective='all-reduce')
        reference, reference_optimizer, _ = _create(grad_collective='all-reduce', copy_from=model)
        reference_layer_wise = _layer_wise(reference_optimizer)
        reference_layer_wise.allgather_params = reference_layer_wise.broadcast_params
        _set_same_main_grads(0, model, reference)
        optimizer.step()
        reference_optimizer.step()
        _assert_params_equal(model, reference)

    @pytest.mark.parametrize('overlap', [False, True])
    @pytest.mark.parametrize('grad_collective', GRAD_COLLECTIVES)
    def test_training_matches_reference(self, grad_collective, overlap):
        """Real gradient collectives: LayerWise buckets next to the sibling DistOpt buckets."""
        # FP32 gradient reduction keeps the collectives' rounding below the tolerance.
        model, optimizer, _ = _create(
            grad_collective=grad_collective,
            overlap_param_gather=overlap,
            grad_reduce_in_fp32=True,
            bucket_size=4096 if overlap else None,
        )
        layer_wise = _layer_wise(optimizer)
        # The model-level config keeps DistOpt for the sibling optimizer.
        assert model.ddp_config.use_distributed_optimizer
        if grad_collective == 'all-reduce':
            assert layer_wise.param_layout == 'compact'
            assert layer_wise.layerwise_param_sync_via_bucket_group == overlap
            for buffer in model.buffers + model.expert_parallel_buffers:
                is_layer_wise = getattr(
                    buffer.params[0], 'is_managed_by_layer_wise_optimizer', False
                )
                # LayerWise buffers are unsharded and unpadded; only DistOpt has a param buffer.
                assert buffer.ddp_config.use_distributed_optimizer != is_layer_wise
                assert (buffer.param_data is None) == is_layer_wise
            for bucket_group in model.bucket_groups + model.expert_parallel_bucket_groups:
                owners = {
                    getattr(param, 'is_managed_by_layer_wise_optimizer', False)
                    for bucket in bucket_group.buckets
                    for param in bucket.params_list
                }
                assert len(owners) == 1, "A bucket group mixes LayerWise and DistOpt buckets"
                if owners.pop():
                    # Without overlap the optimizer gathers the LayerWise params after its step.
                    assert bucket_group.ddp_config.param_sync_via_bucket_group == overlap
        _assert_trains_like_reference(model, optimizer, steps=3)

    def test_all_reduce_training_replays_bit_exact(self):
        """Gradient all-reduce next to the sibling DistOpt reduce-scatter, rank-consistent owners
        and uneven whole-param gathers replay bit for bit."""
        from tests.unit_tests.determinism.kernels.harness import bytes_equal

        runs = []
        for _ in range(2):
            model, optimizer, _ = _create(
                grad_collective='all-reduce',
                overlap_param_gather=True,
                grad_reduce_in_fp32=True,
                bucket_size=4096,
            )
            norms = [_train_step(model, optimizer, step) for step in range(3)]
            runs.append((norms, [param.detach().clone() for param in model.parameters()]))
        assert runs[0][0] == runs[1][0], "grad norms differ between replays"
        for (name, _), a, b in zip(model.named_parameters(), runs[0][1], runs[1][1]):
            assert bytes_equal(a, b), f"{name} differs between replays"

    # ---- Overlap-param-gather tests ----

    @pytest.mark.parametrize(
        ('ddp_overlap_param_gather', 'optimizer_overlap_param_gather'),
        [(False, True), (True, False)],
    )
    @pytest.mark.parametrize('grad_collective', GRAD_COLLECTIVES)
    def test_overlap_param_gather_follows_ddp_config(
        self, grad_collective, ddp_overlap_param_gather, optimizer_overlap_param_gather
    ):
        """DDP's per-model overlap policy must override the global optimizer value."""
        model, optimizer, _ = _create(
            grad_collective=grad_collective,
            overlap_param_gather=ddp_overlap_param_gather,
            optimizer_overlap_param_gather=optimizer_overlap_param_gather,
        )
        layer_wise = _layer_wise(optimizer)
        assert layer_wise.ddp_config is model.ddp_config
        assert layer_wise.overlap_param_gather == ddp_overlap_param_gather

    def test_overlap_param_gather_bucket_lw_params(self):
        """Every all-reduce LayerWise bucket lists each rank's whole params and their sizes."""
        model, _, pg_collection = _create(grad_collective='all-reduce', overlap_param_gather=True)
        dp_size = get_pg_size(pg_collection.dp_cp)
        layer_wise_buckets = [
            bucket
            for bucket_group in model.bucket_groups
            for bucket in bucket_group.buckets
            if getattr(bucket.params_list[0], 'is_managed_by_layer_wise_optimizer', False)
        ]
        assert layer_wise_buckets
        for bucket in layer_wise_buckets:
            assert len(bucket.layerwise_params_list) == dp_size
            assert {p for params in bucket.layerwise_params_list for p in params} == bucket.params
            assert bucket.layerwise_param_flat_sizes == [
                sum(p.numel() for p in params) for params in bucket.layerwise_params_list
            ]

    @pytest.mark.parametrize(
        ('model_class', 'grad_reduce_in_fp32'),
        [(SimpleModel, False), (SimpleModel, True), (TinyModel, False)],
        ids=['simple', 'fp32-grad-reduce', 'tiny'],
    )
    @pytest.mark.parametrize('grad_collective', GRAD_COLLECTIVES)
    def test_overlap_param_gather_matches_sync(
        self, grad_collective, model_class, grad_reduce_in_fp32
    ):
        """Deferred gathers publish exactly what the synchronous gather does, however they
        complete, and leave the grad buffer they staged in zeroed.

        TinyModel leaves ranks without params; with grad_reduce_in_fp32 the grad buffer has a
        different dtype than the params staged in it.
        """
        kwargs = dict(grad_collective=grad_collective, grad_reduce_in_fp32=grad_reduce_in_fp32)
        model, optimizer, _ = _create(model_class, overlap_param_gather=True, **kwargs)
        reference, reference_optimizer, _ = _create(model_class, copy_from=model, **kwargs)
        in_features = next(model.module.parameters()).shape[1]
        x = torch.randn(16, in_features, dtype=torch.bfloat16, device='cuda')

        def hooks_disabled_force_sync():
            # The training loop's first iteration; the gather also runs with autograd enabled.
            model.disable_forward_pre_hook(param_sync=False)
            model.start_param_sync(force_sync=True)
            model.enable_forward_pre_hook()

        def finish():
            model.start_param_sync()
            for bucket_group in model.bucket_groups:
                bucket_group.finish_param_sync(skip_next_bucket_dispatch=True)

        def forward_pre_hooks():
            model.start_param_sync()
            model(x)

        def force_sync_in_flight():
            model.start_param_sync()
            model.start_param_sync(force_sync=True)

        def free_overlap_buffers():
            # Before async checkpoint saves.
            model.start_param_sync()
            model.free_overlap_buffers()

        completions = [
            hooks_disabled_force_sync,
            finish,
            forward_pre_hooks,
            force_sync_in_flight,
            free_overlap_buffers,
        ]
        for step, complete in enumerate(completions):
            _set_same_main_grads(step, model, reference)
            optimizer.step()
            complete()
            reference_optimizer.step()
            for bucket_group in model.bucket_groups:
                assert bucket_group.param_gather_handle is None
                for bucket in bucket_group.buckets:
                    assert bucket.layerwise_gather_list is None
                    assert torch.count_nonzero(bucket.grad_data) == 0, complete.__name__
            _assert_params_equal(model, reference, msg=complete.__name__)

    @pytest.mark.parametrize('grad_collective', GRAD_COLLECTIVES)
    def test_overlap_param_gather_finish_chains_next_bucket(self, grad_collective):
        """finish_param_sync() dispatches next_param_gather_bucket_group."""
        model, optimizer, _ = _create(
            grad_collective=grad_collective, overlap_param_gather=True, bucket_size=2000
        )
        bucket_groups = model.bucket_groups
        if len(bucket_groups) <= 1:
            pytest.skip("Need multiple bucket groups to test chaining")
        reference, reference_optimizer, _ = _create(
            grad_collective=grad_collective, bucket_size=2000, copy_from=model
        )
        _set_same_main_grads(0, model, reference)
        optimizer.step()

        # Dispatch only the last bucket group; finishing it must chain-dispatch the next one.
        last_bucket_group = bucket_groups[-1]
        last_bucket_group.start_param_sync()
        next_bucket_group = last_bucket_group.next_param_gather_bucket_group
        assert next_bucket_group is not None, "Last bucket group should have a next"
        assert not next_bucket_group.param_gather_dispatched
        last_bucket_group.finish_param_sync()
        assert next_bucket_group.param_gather_dispatched
        for bucket_group in reversed(bucket_groups[:-1]):
            bucket_group.finish_param_sync()

        reference_optimizer.step()
        _assert_params_equal(model, reference)


@multi_rank
class TestLayerWiseExperts:
    """LayerWise owns the expert matrices over expert DP; a second sibling DistOpt shards the
    expert biases over expert DP."""

    @pytest.fixture(autouse=True)
    def model_parallel(self):
        if int(os.getenv('WORLD_SIZE', '1')) % 2:
            pytest.skip("Needs an even number of ranks for expert parallelism")
        Utils.initialize_model_parallel(expert_model_parallel_size=2)
        yield
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize('grad_collective', GRAD_COLLECTIVES)
    def test_expert_params_shard_over_expert_dp(self, grad_collective):
        model, optimizer, _ = _create(
            ExpertModel, grad_collective=grad_collective, grad_reduce_in_fp32=True
        )
        expt_dp_size = get_pg_size(parallel_state.get_expert_data_parallel_group())
        if expt_dp_size > 1:
            owned = {p for params in _layer_wise(optimizer).expt_dp_params_list for p in params}
            assert owned == {expert.weight for expert in model.module.experts}

        distopts = [o for o in optimizer.chained_optimizers if isinstance(o, DistributedOptimizer)]
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
        _assert_trains_like_reference(model, optimizer, steps=3)


@multi_rank
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

    @pytest.mark.parametrize('grad_collective', GRAD_COLLECTIVES)
    def test_experts_are_owned_once(self, grad_collective):
        expt_dp = parallel_state.get_expert_data_parallel_group()
        assert get_pg_size(parallel_state.get_data_parallel_group(with_context_parallel=True)) == 1
        assert get_pg_size(expt_dp) > 1
        model, optimizer, _ = _create(
            ExpertModel, grad_collective=grad_collective, grad_reduce_in_fp32=True
        )
        layer_wise = _layer_wise(optimizer)
        assert layer_wise.dp_cp_params_list is None
        owned_here = layer_wise.expt_dp_params_list[get_pg_rank(expt_dp)]
        for expert in model.module.experts:
            owners = [None] * get_pg_size(expt_dp)
            owned = any(expert.weight is p for p in owned_here)
            torch.distributed.all_gather_object(owners, owned, group=expt_dp)
            assert sum(owners) == 1, "Each expert matrix must have exactly one owner"
        _assert_trains_like_reference(model, optimizer, steps=2)
