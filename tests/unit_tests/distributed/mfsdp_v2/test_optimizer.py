# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for Megatron-FSDP optimizer behavior."""

import pytest
import torch
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Partial, Replicate, Shard
from transformer_engine.pytorch.optimizers import FusedAdam

from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import (
    FsdpContext,
    Placements,
    fully_shard,
    fully_shard_context,
    fully_shard_optimizer,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.mixed_precision import MixedPrecisionPolicy
from megatron.core.optimizer import OptimizerConfig
from megatron.core.optimizer.fully_sharded_optimizer import FullyShardedOptimizer
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.distributed.mfsdp_v2.mcore_utils import destroy_model_parallel
from tests.unit_tests.test_utilities import Utils


class TinyModel(nn.Module):
    """Small model with two separately shardable units."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the tiny model."""
        return self.fc2(self.relu(self.fc1(x)))


def _default_placements() -> Placements:
    return Placements(dp_axes=[0], parameter=[Shard(0)], gradient=[Shard(0)], optimizer=[Shard(0)])


@pytest.fixture
def dsa_process_groups(distributed_setup):
    """Initialize Core metadata groups for the mixed-mesh optimizer regression."""
    if distributed_setup.world_size < 4 or distributed_setup.world_size % 2:
        pytest.skip("Requires an even world size of at least four for EP=2 and expert DP>=2.")
    Utils.initialize_model_parallel(1, 1, expert_model_parallel_size=2)
    yield ProcessGroupCollection.use_mpu_process_groups()
    destroy_model_parallel()


@pytest.mark.parametrize("dense_placement", [Shard(0), Replicate()], ids=["sharded", "replicated"])
@pytest.mark.parametrize("expert_placement", [Shard(0), Replicate()], ids=["sharded", "replicated"])
@pytest.mark.parametrize("indexer_has_grad", [True, False], ids=["indexer-grad", "no-indexer-grad"])
@pytest.mark.parametrize("indexer_clip_grad", [None, 0.5], ids=["ordinary-clip", "separate-clip"])
def test_dsa_split_norms_and_adam_step(
    distributed_setup,
    dsa_process_groups,
    dense_placement,
    expert_placement,
    indexer_has_grad,
    indexer_clip_grad,
):
    """Check Adam clipping with dense and expert DTensors on different device meshes."""
    world_size = distributed_setup.world_size
    device = distributed_setup.device
    dense_mesh = init_device_mesh(device.type, (world_size,))
    expert_topology = init_device_mesh(
        device.type, (2, world_size // 2), mesh_dim_names=("ep", "expert_dp")
    )
    expert_mesh = expert_topology["expert_dp"]
    expert_value = 12.0 + expert_topology.get_local_rank("ep")
    model = nn.Module()
    model.config = TransformerConfig(num_layers=1, hidden_size=8, num_attention_heads=1)
    model.ddp_config = DistributedDataParallelConfig(use_distributed_optimizer=False)
    for name, mesh, placement, value in (
        ("indexer", dense_mesh, dense_placement, 3.0),
        ("dense", dense_mesh, dense_placement, 4.0),
        ("expert", expert_mesh, expert_placement, expert_value),
    ):
        parameter = nn.Parameter(
            DTensor.from_local(torch.ones(2, device=device), mesh, [placement])
        )
        if name != "indexer" or indexer_has_grad:
            parameter.grad = DTensor.from_local(
                torch.full((2,), value, device=device), mesh, [placement]
            )
        model.register_parameter(name, parameter)

    adam = torch.optim.Adam(
        [
            {"params": [model.indexer], "is_dsa_indexer": True},
            {"params": [model.dense, model.expert]},
        ],
        lr=0.01,
        foreach=False,
    )
    optimizer = FullyShardedOptimizer(
        adam,
        OptimizerConfig(optimizer="adam", clip_grad=1.0, dsa_indexer_clip_grad=indexer_clip_grad),
        grad_scaler=None,
        init_state_fn=None,
        model_chunks=[model],
    )
    optimizer.grad_stats_parallel_group = dsa_process_groups.dp_cp
    optimizer.tp_group = dsa_process_groups.tp
    optimizer.expert_tp_group = dsa_process_groups.expt_tp

    dense_elements = 2 * (world_size if dense_placement.is_shard() else 1)
    expert_elements = 2 * (world_size // 2 if expert_placement.is_shard() else 1)
    expected_indexer = (9.0 * dense_elements) ** 0.5 if indexer_has_grad else 0.0
    expected_non_indexer = (16.0 * dense_elements + (12.0**2 + 13.0**2) * expert_elements) ** 0.5
    expected_total = (expected_indexer**2 + expected_non_indexer**2) ** 0.5
    original_grads = {
        name: parameter.grad.to_local().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }

    success, total_norm, _ = optimizer.step()
    assert success
    assert float(total_norm) == pytest.approx(expected_total)
    assert optimizer.get_last_dsa_split_grad_norms() == pytest.approx(
        (expected_indexer, expected_non_indexer)
    )
    for name, original_grad in original_grads.items():
        if indexer_clip_grad is None:
            scale = 1.0 / (expected_total + 1.0e-6)
        elif name == "indexer":
            scale = indexer_clip_grad / (expected_indexer + 1.0e-6)
        else:
            scale = 1.0 / (expected_non_indexer + 1.0e-6)
        torch.testing.assert_close(getattr(model, name).grad.to_local(), original_grad * scale)
        assert not torch.equal(getattr(model, name).to_local(), torch.ones(2, device=device))
    if not indexer_has_grad:
        assert model.indexer.grad is None
        torch.testing.assert_close(model.indexer.to_local(), torch.ones(2, device=device))


def test_adam_without_adapter_raises_precision_error(distributed_setup):
    """Raw Adam should fail on mixed-precision FSDP parameters without the adapter."""
    world_size = distributed_setup.world_size
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (world_size,))
    torch.manual_seed(2026)
    model = TinyModel().to(device=device, dtype=torch.bfloat16)
    with fully_shard_context(FsdpContext(device=device)):
        fully_shard(model.fc1, mesh=mesh, placements=_default_placements())
        fully_shard(model.fc2, mesh=mesh, placements=_default_placements())
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    x = torch.randn(6, 8, device=device, dtype=torch.bfloat16)
    optimizer.zero_grad(set_to_none=True)
    loss = model(x).sum()
    loss.backward()

    with pytest.raises(RuntimeError, match="dtype"):
        optimizer.step()


def test_fused_adam_adapter_accepts_mismatched_grads(distributed_setup):
    """TE FusedAdam should handle mixed-precision FSDP grads through the adapter."""
    world_size = distributed_setup.world_size
    device = distributed_setup.device

    mesh = init_device_mesh(device.type, (world_size,))
    torch.manual_seed(2026)
    model = TinyModel().to(device=device, dtype=torch.bfloat16)
    # These are the defaults, but spell them out so the test clearly exercises
    # mismatched parameter and gradient precision.
    mixed_precision_policy = MixedPrecisionPolicy(
        main_params_dtype=torch.float32, main_grads_dtype=torch.bfloat16
    )
    with fully_shard_context(FsdpContext(device=device)):
        fully_shard(
            model.fc1,
            mesh=mesh,
            placements=_default_placements(),
            mixed_precision_policy=mixed_precision_policy,
        )
        fully_shard(
            model.fc2,
            mesh=mesh,
            placements=_default_placements(),
            mixed_precision_policy=mixed_precision_policy,
        )
    optimizer = FusedAdam(model.parameters(), lr=0.01)
    fully_shard_optimizer(optimizer, precision_aware=True)

    x = torch.randn(6, 8, device=device, dtype=torch.bfloat16)
    optimizer.zero_grad(set_to_none=True)
    loss = model(x).sum()
    loss.backward()

    for parameter in model.parameters():
        assert parameter.grad is not None
        assert parameter.dtype != parameter.grad.dtype

    params_before_step = [parameter.detach().clone() for parameter in model.parameters()]
    optimizer.step()

    assert any(
        not torch.equal(parameter_before, parameter.detach())
        for parameter_before, parameter in zip(params_before_step, model.parameters())
    )


@pytest.mark.parametrize(
    "parameter_placements",
    [
        pytest.param([Replicate(), Shard(0)], id="hfsdp"),  # ZeRO-1 / ZeRO-3
        pytest.param([Replicate(), Replicate()], id="hybrid_zero2"),  # ZeRO-1 / ZeRO-2
        pytest.param([Shard(0), Shard(0)], id="fsdp"),  # ZeRO-3 / ZeRO-3
    ],
)
@pytest.mark.parametrize("inner_dp_size", [1, 2])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_next_forward_uses_optimizer_updated_weights(
    distributed_setup, parameter_placements, inner_dp_size, dtype
):
    """The next forward should observe weights updated by the previous optimizer step."""
    world_size = distributed_setup.world_size
    device = distributed_setup.device
    if world_size < 2 or world_size % inner_dp_size:
        pytest.skip("Requires at least two ranks and a world size divisible by inner_dp_size.")

    mesh = init_device_mesh(device.type, (world_size // inner_dp_size, inner_dp_size))
    placements = Placements(
        dp_axes=[0, 1],
        parameter=parameter_placements,
        # Reduce gradients inner-then-outer into the optimizer's two-axis shards.
        gradient=[Partial("avg"), Shard(0)],
        optimizer=[Shard(0), Shard(0)],
    )
    # Uneven rows and a bias exercise padding in both gather stages.
    model = nn.Linear(5, 7, device=device, dtype=dtype)
    nn.init.ones_(model.weight)
    nn.init.zeros_(model.bias)

    with fully_shard_context(FsdpContext(device=device)):
        fully_shard(
            model,
            mesh=mesh,
            placements=placements,
            mixed_precision_policy=MixedPrecisionPolicy(main_params_dtype=torch.float32),
        )
    # SGD's foreach/fused CUDA paths require matching parameter and gradient dtypes.
    # Use the scalar path to exercise FP32 main weights with default BF16 main grads.
    optimizer = torch.optim.SGD(model.parameters(), lr=0.25, foreach=False)
    fully_shard_optimizer(optimizer)
    x = torch.ones(1, 5, device=device, dtype=dtype)

    def train_iteration() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        loss = model(x).sum()
        loss.backward()
        optimizer.step()
        return loss.detach().float()

    # Each step subtracts 0.25 from all five weights and the bias of each row.
    for expected in (35.0, 24.5, 14.0):
        torch.testing.assert_close(train_iteration(), torch.tensor(expected, device=device))


def test_optimizer_post_step_syncs_once_per_parameter_group(distributed_setup, monkeypatch):
    """Optimizer synchronization should run once per group, not once per microbatch."""
    world_size = distributed_setup.world_size
    device = distributed_setup.device
    if world_size < 2:
        pytest.skip("This test requires at least 2 ranks.")

    mesh = init_device_mesh(device.type, (world_size,))
    model = TinyModel().to(device=device, dtype=torch.bfloat16)
    with fully_shard_context(FsdpContext(device=device)):
        fully_shard(model.fc1, mesh=mesh, placements=_default_placements())
        fully_shard(model.fc2, mesh=mesh, placements=_default_placements())
    parameter_groups = (*model.fc1.parameter_groups, *model.fc2.parameter_groups)
    sync_counts = {parameter_group: 0 for parameter_group in parameter_groups}

    def make_count_sync(parameter_group):
        sync_model_weight = parameter_group.sync_model_weight_from_main_weight

        def count_sync():
            sync_counts[parameter_group] += 1
            sync_model_weight()

        return count_sync

    for parameter_group in parameter_groups:
        monkeypatch.setattr(
            parameter_group, "sync_model_weight_from_main_weight", make_count_sync(parameter_group)
        )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    fully_shard_optimizer(optimizer)
    inputs = torch.randn(3, 2, 8, device=device, dtype=torch.bfloat16)

    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        for microbatch_input in inputs:
            (model(microbatch_input).sum() / len(inputs)).backward()

        assert all(sync_count == step for sync_count in sync_counts.values())
        optimizer.step()
        assert all(sync_count == step + 1 for sync_count in sync_counts.values())
