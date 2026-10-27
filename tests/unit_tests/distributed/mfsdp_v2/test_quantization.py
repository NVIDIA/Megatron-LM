# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Quantized training tests for the experimental Megatron-FSDP path."""

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
import transformer_engine.pytorch as te
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Partial, Replicate, Shard
from transformer_engine.common.recipe import MXFP8BlockScaling
from transformer_engine.pytorch.distributed import checkpoint as te_checkpoint
from transformer_engine.pytorch.optimizers import FusedAdam
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Tensor

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import (
    Placements,
    fully_shard,
    fully_shard_context,
    fully_shard_optimizer,
    microbatch,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.dbuffer import DBuffer
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.placement import BlockAtomic
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.quantized_dbuffer import (
    QuantizedDBuffer,
)


def _make_mlp(device):
    kwargs = {"params_dtype": torch.bfloat16, "device": device}
    return nn.Sequential(te.Linear(64, 128, **kwargs), nn.GELU(), te.Linear(128, 32, **kwargs))


class _RecomputedMLP(nn.Module):
    """Two-layer MLP that optionally recomputes each linear layer in backward."""

    def __init__(self, device, use_reentrant: bool | None) -> None:
        super().__init__()
        kwargs = {"params_dtype": torch.bfloat16, "device": device}
        self.fc1 = te.Linear(64, 128, **kwargs)
        self.fc2 = te.Linear(128, 32, **kwargs)
        self.use_reentrant = use_reentrant

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the MLP, checkpointing each linear layer unless use_reentrant is None."""
        if self.use_reentrant is None:
            return self.fc2(F.gelu(self.fc1(x)))
        # TE's checkpoint restores the MXFP8 autocast state during recomputation.
        x = te_checkpoint(self.fc1, x, use_reentrant=self.use_reentrant)
        return te_checkpoint(self.fc2, F.gelu(x), use_reentrant=self.use_reentrant)


_ZERO3_PLACEMENTS = Placements(
    dp_axes=[0], parameter=[Shard(0)], gradient=[Shard(0)], optimizer=[Shard(0)]
)
# Sharded compute weights gather MXFP8 planes per phase; replicated ones need no gather.
_MXFP8_PLACEMENTS = [
    pytest.param(_ZERO3_PLACEMENTS, id="zero3"),
    pytest.param(
        Placements(dp_axes=[0], parameter=[Replicate()], gradient=[Shard(0)], optimizer=[Shard(0)]),
        id="zero2",
    ),
    pytest.param(
        Placements(
            dp_axes=[0], parameter=[Replicate()], gradient=[Partial("avg")], optimizer=[Replicate()]
        ),
        id="no_shard",
    ),
]

# Compute weights replicated across DP-outer and sharded within DP-inner. HSDP also
# replicates optimizer weights across DP-outer; HFSDP shards them, so each step
# all-gathers updated compute weights across DP-outer before the per-phase unshard.
_MXFP8_2D_PLACEMENTS = [
    pytest.param(
        Placements(
            dp_axes=[0, 1],
            parameter=[Replicate(), Shard(0)],
            gradient=[Partial("avg"), Shard(0)],
            optimizer=[Replicate(), Shard(0)],
        ),
        id="hsdp",
    ),
    pytest.param(
        Placements(
            dp_axes=[0, 1],
            parameter=[Replicate(), Shard(0)],
            gradient=[Partial("avg"), Shard(0)],
            optimizer=[Shard(0), Shard(0)],
        ),
        id="hfsdp",
    ),
]

requires_mxfp8 = pytest.mark.skipif(
    torch.cuda.get_device_capability()[0] < 10,
    reason="MXFP8 requires Blackwell-or-newer CUDA hardware.",
)


def _check_mxfp8_mlp_matches_reference(distributed_setup, mesh, placements, num_microbatches):
    """Train an MXFP8 MLP with MFSDP and check its losses track an unsharded reference."""
    device = distributed_setup.device
    recipe = MXFP8BlockScaling()
    with te.quantized_model_init(recipe=recipe, preserve_high_precision_init_val=True):
        torch.manual_seed(2026)
        model = _make_mlp(device)
        torch.manual_seed(2026)
        reference = _make_mlp(device)
    with fully_shard_context(device=device) as context:
        fully_shard(model, mesh=mesh, placements=placements)

    # MLP weights share one quantized group; biases use a regular DBuffer.
    [weight_group] = [
        g for g in model.parameter_groups if isinstance(g.model_weight, QuantizedDBuffer)
    ]
    assert len(weight_group.fsdp_parameters) == 2
    assert weight_group.main_weight.placements == tuple(
        BlockAtomic(32) if isinstance(placement, Shard) else placement
        for placement in placements.optimizer
    )
    [bias_group] = [g for g in model.parameter_groups if g is not weight_group]
    assert isinstance(bias_group.model_weight, DBuffer)

    optimizer = FusedAdam(model.parameters(), lr=0.01)
    fully_shard_optimizer(optimizer)
    # The reference has MXFP8 weights, so Adam needs separate FP32 weights for updates.
    # MFSDP doesn't need master_weights=True because it already passes FP32 master
    # weights to the optimizer.
    reference_optimizer = FusedAdam(reference.parameters(), lr=0.01, master_weights=True)
    # Match MFSDP's preserved initialization instead of dequantizing MXFP8 weights.
    for parameter in reference.parameters():
        # FP32 reconstruction from 16-bit remainders requires BF16 weight bits.
        # MXFP8 quantization loses some of those bits, so store full FP32 masters.
        reference_optimizer.initialize_state(parameter, store_param_remainders=False)
        if isinstance(parameter, MXFP8Tensor):
            reference_optimizer.set_scaled_state(
                parameter,
                "master_param",
                parameter.get_high_precision_init_val().to(device=device, dtype=torch.float32),
            )
            parameter.clear_high_precision_init_val()
    # Different rank inputs make an incorrect gradient reduction observable.
    torch.manual_seed(1234 + distributed_setup.rank)
    inputs = torch.randn(5, num_microbatches, 32, 64, dtype=torch.bfloat16, device=device)
    targets = torch.randn(5, num_microbatches, 32, 32, dtype=torch.bfloat16, device=device)

    def train(model, optimizer):
        losses = []
        for step_inputs, step_targets in zip(inputs, targets):
            optimizer.zero_grad(set_to_none=True)
            for index, (x, target) in enumerate(zip(step_inputs, step_targets)):
                # HSDP/HFSDP reduce gradients across DP-outer only on the last microbatch.
                with microbatch(context, is_last=index == num_microbatches - 1):
                    with te.autocast(recipe=recipe):
                        output = model(x)
                    loss = (output.float() - target.float()).square().mean()
                    (loss / num_microbatches).backward()
                losses.append(loss.detach())

            # The unwrapped reference needs explicit data-parallel averaging.
            if model is reference:
                for parameter in model.parameters():
                    dist.all_reduce(parameter.grad, op=dist.ReduceOp.AVG)

            optimizer.step()
        return losses

    reference_losses = train(reference, reference_optimizer)
    sharded_losses = train(model, optimizer)
    # BF16 reduction order can make the independent optimizer updates differ.
    torch.testing.assert_close(
        torch.stack(sharded_losses), torch.stack(reference_losses), rtol=0, atol=3e-3
    )


@pytest.mark.launch_on_gb200
@requires_mxfp8
@pytest.mark.parametrize("placements", _MXFP8_PLACEMENTS)
def test_mxfp8_mlp_training_matches_reference(distributed_setup, placements):
    """MXFP8 MLP losses track an independently trained unsharded model."""
    if distributed_setup.world_size < 2:
        pytest.skip("Distributed MXFP8 coverage requires at least two ranks.")
    mesh = init_device_mesh(distributed_setup.device.type, (distributed_setup.world_size,))
    _check_mxfp8_mlp_matches_reference(distributed_setup, mesh, placements, num_microbatches=1)


@pytest.mark.launch_on_gb200
@requires_mxfp8
@pytest.mark.parametrize("placements", _MXFP8_2D_PLACEMENTS)
@pytest.mark.parametrize("num_microbatches", [1, 3])
def test_mxfp8_mlp_training_on_2d_mesh_matches_reference(
    distributed_setup, placements, num_microbatches
):
    """MXFP8 MLP losses on a 2-D DP mesh track an independently trained unsharded model."""
    world_size = distributed_setup.world_size
    if world_size < 4 or world_size % 2 != 0:
        pytest.skip("A 2-D DP mesh requires an even number of at least 4 ranks.")
    mesh = init_device_mesh(
        distributed_setup.device.type, (2, world_size // 2), mesh_dim_names=("dp_outer", "dp_inner")
    )
    _check_mxfp8_mlp_matches_reference(distributed_setup, mesh, placements, num_microbatches)


@pytest.mark.launch_on_gb200
@requires_mxfp8
def test_mxfp8_unshard_gathers_rowwise_for_forward_and_columnwise_for_backward(distributed_setup):
    """Forward unshards and prefetches only rowwise planes; backward only columnwise ones.

    fc1 and fc2 are separate FsdpModules, so each phase checks both a module unsharded
    for its own compute and the successor it prefetched.
    """
    device = distributed_setup.device
    if distributed_setup.world_size < 2:
        pytest.skip("Distributed MXFP8 coverage requires at least two ranks.")

    recipe = MXFP8BlockScaling()
    with te.quantized_model_init(recipe=recipe, preserve_high_precision_init_val=True):
        model = _RecomputedMLP(device, use_reentrant=None)
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    with fully_shard_context(device=device):
        fully_shard(model.fc1, mesh=mesh, placements=_ZERO3_PLACEMENTS)
        fully_shard(model.fc2, mesh=mesh, placements=_ZERO3_PLACEMENTS)
        fully_shard(model, mesh=mesh, placements=_ZERO3_PLACEMENTS)

    def allocated_planes(module):
        """Return whether the module's unsharded (rowwise, columnwise) planes are allocated."""
        [group] = [
            g for g in module.parameter_groups if isinstance(g.model_weight, QuantizedDBuffer)
        ]
        unsharded = group._unsharded_model_weight
        return unsharded.is_rowwise_allocated, unsharded.is_columnwise_allocated

    def installed_planes(module):
        """Return whether the module's weight carries (rowwise, columnwise) data."""
        weight = module.weight
        return weight._rowwise_data is not None, weight._columnwise_data is not None

    checked_phases = []

    # Registered after fully_shard(), so these run after the FSDP pre-hooks have
    # unsharded the module and prefetched its successor.
    def check_forward(_module, _args):
        # fc1 runs forward GEMMs and prefetches fc2 for its forward GEMMs.
        for module in (model.fc1, model.fc2):
            assert allocated_planes(module) == (True, False)
            assert installed_planes(module) == (True, False)
        checked_phases.append("forward")

    def check_backward(_module, _grad_output):
        # fc2 runs data-gradient GEMMs and prefetches fc1 for its data-gradient GEMMs.
        for module in (model.fc2, model.fc1):
            assert allocated_planes(module) == (False, True)
            assert installed_planes(module) == (False, True)
        checked_phases.append("backward")

    model.fc1.register_forward_pre_hook(check_forward)
    model.fc2.register_full_backward_pre_hook(check_backward)

    x = torch.randn(32, 64, dtype=torch.bfloat16, device=device)
    with te.autocast(recipe=recipe):
        output = model(x)
    output.float().square().mean().backward()

    assert checked_phases == ["forward", "backward"]
    for module in (model.fc1, model.fc2):
        assert allocated_planes(module) == (False, False)


@pytest.mark.launch_on_gb200
@requires_mxfp8
@pytest.mark.parametrize("placements", _MXFP8_PLACEMENTS)
@pytest.mark.parametrize("use_reentrant", [False, True], ids=["non_reentrant", "reentrant"])
def test_mxfp8_activation_recompute_matches_no_recompute(
    distributed_setup, use_reentrant, placements
):
    """Recomputation regathers rowwise planes in backward without changing numerics.

    With sharded compute weights, backward prefetch gathers only columnwise planes,
    so recomputed forwards must gather rowwise planes on demand. Replicated compute
    weights need no gather and keep both planes installed. Shapes need padded
    scales, which covers reinstalling padded scale copies on the Parameter TE saved
    for backward.
    """
    device = distributed_setup.device
    if distributed_setup.world_size < 2:
        pytest.skip("Distributed MXFP8 coverage requires at least two ranks.")

    recipe = MXFP8BlockScaling()
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))

    def build(use_reentrant):
        with te.quantized_model_init(recipe=recipe, preserve_high_precision_init_val=True):
            torch.manual_seed(2026)
            model = _RecomputedMLP(device, use_reentrant)
        with fully_shard_context(device=device):
            fully_shard(model.fc1, mesh=mesh, placements=placements)
            fully_shard(model.fc2, mesh=mesh, placements=placements)
            fully_shard(model, mesh=mesh, placements=placements)
        optimizer = FusedAdam(model.parameters(), lr=0.01)
        fully_shard_optimizer(optimizer)
        return model, optimizer

    torch.manual_seed(1234 + distributed_setup.rank)
    inputs = torch.randn(3, 32, 64, dtype=torch.bfloat16, device=device)

    def train(model, optimizer):
        """Return per-step losses and sharded gradients, and the final sharded weights."""
        losses, grads = [], []
        for x in inputs:
            optimizer.zero_grad(set_to_none=True)
            # Reentrant checkpointing produces gradients only for inputs requiring grad.
            x = x.clone().requires_grad_()
            with te.autocast(recipe=recipe):
                output = model(x)
            loss = output.float().square().mean()
            loss.backward()
            losses.append(loss.detach())
            grads.append([parameter.grad.to_local().clone() for parameter in model.parameters()])
            optimizer.step()
        weights = [parameter.to_local().clone() for parameter in model.parameters()]
        return losses, grads, weights

    expected_losses, expected_grads, expected_weights = train(*build(None))
    losses, grads, weights = train(*build(use_reentrant))
    torch.testing.assert_close(losses, expected_losses, rtol=0, atol=0)
    torch.testing.assert_close(grads, expected_grads, rtol=0, atol=0)
    torch.testing.assert_close(weights, expected_weights, rtol=0, atol=0)
