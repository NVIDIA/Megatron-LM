# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Transformer Engine fused weight-gradient tests for MFSDP v2.

TE modules built with ``fuse_wgrad_accumulation=True`` write their weight gradients into
``weight.main_grad``, which MFSDP points at its reduce-scatter input. Test buffer release in
``test_memory.py`` and the buffer dtype in ``test_grad_comm_dtype.py``.
"""

import pytest
import torch
import transformer_engine.pytorch as te
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Partial, Replicate, Shard
from torch.utils.checkpoint import checkpoint

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import (
    Placements,
    fully_shard,
    fully_shard_context,
    fully_shard_optimizer,
    microbatch,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.module import FsdpModule


def _default_placements() -> Placements:
    return Placements(dp_axes=[0], parameter=[Shard(0)], gradient=[Shard(0)], optimizer=[Shard(0)])


def _hsdp_placements() -> Placements:
    """HSDP: params/optimizer replicated across DP-outer (axis 0), sharded within
    DP-inner (axis 1). main_grad rests [Partial, Shard(0)] between microbatches and is
    all-reduced to [Replicate, Shard(0)] on the last microbatch."""
    return Placements(
        dp_axes=[0, 1],
        parameter=[Replicate(), Shard(0)],
        gradient=[Partial("avg"), Shard(0)],
        optimizer=[Replicate(), Shard(0)],
    )


def test_fused_wgrad_mask_marks_only_te_fused_gemm_weights(distributed_setup):
    """Only 2-D weights of TE modules built with fuse_wgrad_accumulation=True are fused."""
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    model = nn.Sequential(
        te.LayerNormLinear(16, 32, bias=True, device=device, fuse_wgrad_accumulation=True),
        te.Linear(32, 16, bias=True, device=device, fuse_wgrad_accumulation=False),
    )
    frozen = te.Linear(16, 16, bias=True, device=device, fuse_wgrad_accumulation=True)
    frozen.requires_grad_(False)
    with fully_shard_context(device=device):
        fully_shard(model, mesh=mesh, placements=_default_placements())
        fully_shard(frozen, mesh=mesh, placements=_default_placements())

    fused_by_fqn = {
        fqn: fused
        for group in model.parameter_groups
        for fsdp_parameter, fused in zip(group.fsdp_parameters, group.fused_wgrad_mask)
        for fqn in fsdp_parameter.fqns
    }
    assert fused_by_fqn == {
        "0.layer_norm_weight": False,
        "0.layer_norm_bias": False,
        "0.weight": True,
        "0.bias": False,
        "1.weight": False,
        "1.bias": False,
    }
    assert [group.has_fused_wgrad for group in model.parameter_groups] == [True]
    assert [group.has_fused_wgrad for group in frozen.parameter_groups] == [False]


def _fused_layer_norm_linear_group(distributed_setup):
    """Fully shard one fused te.LayerNormLinear; return it, its group, and unsharded params."""
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    model = te.LayerNormLinear(16, 32, bias=True, device=device, fuse_wgrad_accumulation=True)
    with fully_shard_context(device=device):
        fully_shard(model, mesh=mesh, placements=_default_placements())
    [group] = model.parameter_groups
    unsharded = {
        fsdp_parameter.fqns[0]: fsdp_parameter.unsharded for fsdp_parameter in group.fsdp_parameters
    }
    # Return the model too: the group holds only a weak reference to it.
    return model, group, unsharded


def test_fused_grad_buffer_prepare_and_take(distributed_setup):
    """prepare exposes zeroed fused views; take packs autograd gradients and drops main_grad."""
    _model, group, unsharded = _fused_layer_norm_linear_group(distributed_setup)
    unfused_names = ("layer_norm_weight", "layer_norm_bias", "bias")

    group.prepare_fused_grad_buffer()
    main_grad = unsharded["weight"].main_grad
    assert main_grad.shape == unsharded["weight"].shape
    assert main_grad.dtype == group.dtype
    assert torch.count_nonzero(main_grad) == 0
    assert not any(hasattr(unsharded[name], "main_grad") for name in unfused_names)

    # A second prepare in the same backward, e.g. from another unshard(), keeps the buffer.
    group.prepare_fused_grad_buffer()
    assert unsharded["weight"].main_grad is main_grad

    main_grad.fill_(1.0)  # Stands in for TE's fused wgrad GEMM.
    for name in unfused_names:
        unsharded[name].grad = torch.full_like(unsharded[name], 2.0)
    buffer = group.take_fused_grad_buffer()

    assert not any(hasattr(parameter, "main_grad") for parameter in unsharded.values())
    assert all(parameter.grad is None for parameter in unsharded.values())
    views = {
        fsdp_parameter.fqns[0]: buffer.get_tensor_view(index)
        for index, fsdp_parameter in enumerate(group.fsdp_parameters)
    }
    torch.testing.assert_close(views["weight"], torch.ones_like(views["weight"]))
    for name in unfused_names:
        torch.testing.assert_close(views[name], torch.full_like(views[name], 2.0))
    with pytest.raises(RuntimeError, match="not prepared"):
        group.take_fused_grad_buffer()


def test_fused_grad_buffer_requires_unfused_gradients(distributed_setup):
    """An unfused parameter without a gradient is an error, as on the copy path."""
    _model, group, unsharded = _fused_layer_norm_linear_group(distributed_setup)
    group.prepare_fused_grad_buffer()
    unsharded["layer_norm_weight"].grad = torch.ones_like(unsharded["layer_norm_weight"])
    unsharded["layer_norm_bias"].grad = torch.ones_like(unsharded["layer_norm_bias"])
    with pytest.raises(RuntimeError, match="Missing gradient"):
        group.take_fused_grad_buffer()


class _FusedWgradModel(nn.Module):
    """TE LayerNormLinear followed by TE Linear, optionally recomputing the first layer."""

    def __init__(
        self,
        device: torch.device,
        dtype: torch.dtype,
        fuse_first: bool,
        fuse_second: bool,
        checkpoint_first: bool = False,
        use_reentrant: bool = False,
    ) -> None:
        super().__init__()
        self.first = te.LayerNormLinear(
            16, 32, bias=True, params_dtype=dtype, device=device, fuse_wgrad_accumulation=fuse_first
        )
        self.second = te.Linear(
            32,
            16,
            bias=True,
            params_dtype=dtype,
            device=device,
            fuse_wgrad_accumulation=fuse_second,
        )
        self.checkpoint_first = checkpoint_first
        self.use_reentrant = use_reentrant

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run both layers."""
        if self.checkpoint_first:
            hidden = checkpoint(self.first, x, use_reentrant=self.use_reentrant)
        else:
            hidden = self.first(x)
        return self.second(hidden)


def _build_fused_wgrad_run(mesh, placements, variant, dtype, fuse, device):
    """Build, shard, and wrap an optimizer for one side of the fused-vs-copy comparison."""
    torch.manual_seed(1234)
    model = _FusedWgradModel(
        device,
        dtype,
        fuse_first=fuse,
        fuse_second=fuse and variant != "mixed_unit",
        checkpoint_first=variant.startswith("checkpointed"),
        use_reentrant=variant == "checkpointed_reentrant",
    )
    with fully_shard_context(device=device) as context:
        if variant == "mixed_unit":
            fully_shard(model, mesh=mesh, placements=placements)
        else:
            fully_shard(model.first, mesh=mesh, placements=placements)
            fully_shard(model.second, mesh=mesh, placements=placements)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    fully_shard_optimizer(optimizer)
    return model, context, optimizer


def _train_fused_wgrad_model(model, context, optimizer, microbatches, set_to_none):
    """Run five optimizer steps over ``microbatches`` and return every micro-batch loss."""
    losses = []
    for _ in range(5):
        optimizer.zero_grad(set_to_none=set_to_none)
        for index, (x, target) in enumerate(microbatches):
            with microbatch(context, is_last=index == len(microbatches) - 1):
                loss = torch.nn.functional.mse_loss(model(x).float(), target)
                losses.append(loss.detach())
                (loss / len(microbatches)).backward()
        optimizer.step()
    return torch.stack(losses)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize("set_to_none", [True, False])
@pytest.mark.parametrize("num_microbatches", [1, 3])
@pytest.mark.parametrize(
    "variant", ["per_layer", "mixed_unit", "checkpointed", "checkpointed_reentrant", "hsdp"]
)
def test_fused_wgrad_losses_match_unfused(
    distributed_setup, variant, num_microbatches, set_to_none, dtype
):
    """TE layers writing wgrads into the reduce-scatter buffer train like the copy path."""
    device = distributed_setup.device
    world_size = distributed_setup.world_size
    if variant == "hsdp":
        if world_size < 4 or world_size % 2 != 0:
            pytest.skip("HSDP requires an even number of at least 4 ranks for a 2-D DP mesh.")
        mesh = init_device_mesh(
            device.type, (2, world_size // 2), mesh_dim_names=("dp_outer", "dp_inner")
        )
        placements = _hsdp_placements()
    else:
        mesh = init_device_mesh(device.type, (world_size,))
        placements = _default_placements()
    tolerance = (
        {"rtol": 1e-5, "atol": 1e-5} if dtype == torch.float32 else {"rtol": 1.6e-2, "atol": 1e-5}
    )

    torch.manual_seed(5678 + distributed_setup.rank)
    # Inputs require grad: a reentrant checkpoint produces no parameter gradients otherwise.
    microbatches = [
        (
            torch.randn(2, 16, device=device, dtype=dtype, requires_grad=True),
            torch.randn(2, 16, device=device),
        )
        for _ in range(num_microbatches)
    ]
    results = {}
    for fuse in (False, True):
        model, context, optimizer = _build_fused_wgrad_run(
            mesh, placements, variant, dtype, fuse, device
        )
        losses = _train_fused_wgrad_model(model, context, optimizer, microbatches, set_to_none)
        # MFSDP's row-atomic layout can place a whole parameter on one rank, which
        # full_tensor() rejects. Both runs share one layout, so compare local shards.
        weights = [parameter.to_local().detach().clone() for parameter in model.parameters()]
        results[fuse] = (losses, weights)

    torch.testing.assert_close(results[True][0], results[False][0], **tolerance)
    for fused_weight, copied_weight in zip(results[True][1], results[False][1]):
        torch.testing.assert_close(fused_weight, copied_weight, **tolerance)


def test_fused_wgrad_main_grad_only_exists_during_backward(distributed_setup):
    """main_grad is set on fused weights for their module's backward and removed afterwards."""
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    model = _FusedWgradModel(device, torch.float32, fuse_first=True, fuse_second=True)
    with fully_shard_context(device=device):
        fully_shard(model.first, mesh=mesh, placements=_default_placements())
        fully_shard(model.second, mesh=mesh, placements=_default_placements())
    unsharded = {
        f"{layer_name}.{fqn}": fsdp_parameter.unsharded
        for layer_name, layer in (("first", model.first), ("second", model.second))
        for group in layer.parameter_groups
        for fsdp_parameter in group.fsdp_parameters
        for fqn in fsdp_parameter.fqns
    }

    def with_main_grad(prefix: str) -> set[str]:
        return {
            name
            for name, parameter in unsharded.items()
            if name.startswith(prefix) and hasattr(parameter, "main_grad")
        }

    # Registered after fully_shard(), so these run after MFSDP's own backward pre-hooks.
    seen = {}
    model.second.register_full_backward_pre_hook(
        lambda _module, _grad_output: seen.update(second=with_main_grad(""))
    )
    model.first.register_full_backward_pre_hook(
        lambda _module, _grad_output: seen.update(first=with_main_grad("first."))
    )
    x = torch.randn(4, 16, device=device, requires_grad=True)

    with torch.no_grad():
        model(x)
    assert with_main_grad("") == set()
    loss = model(x).square().mean()
    assert with_main_grad("") == set()
    loss.backward()

    # When the second layer's backward starts, only its own fused weight is exposed; prefetching
    # the first layer must not prepare its buffer.
    assert seen["second"] == {"second.weight"}
    assert seen["first"] == {"first.weight"}
    assert with_main_grad("") == set()


@pytest.mark.parametrize("use_reentrant", [False, True], ids=["non_reentrant", "reentrant"])
def test_fused_wgrad_recompute_prepares_each_buffer_at_its_backward(
    distributed_setup, use_reentrant
):
    """Recomputing a checkpointed region prepares no buffer before its module's backward."""
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    model = _FusedWgradModel(device, torch.float32, fuse_first=True, fuse_second=True)
    with fully_shard_context(device=device):
        fully_shard(model.first, mesh=mesh, placements=_default_placements())
        fully_shard(model.second, mesh=mesh, placements=_default_placements())
    [first_weight] = [
        fsdp_parameter.unsharded
        for group in model.first.parameter_groups
        for fsdp_parameter in group.fsdp_parameters
        if fsdp_parameter.fqns == ("weight",)
    ]
    [second_bias] = [
        fsdp_parameter.unsharded
        for group in model.second.parameter_groups
        for fsdp_parameter in group.fsdp_parameters
        if fsdp_parameter.fqns == ("bias",)
    ]

    # The second layer's bias gradient lands after the region's recompute and before the first
    # layer's backward starts, so the first layer's buffer must not exist yet.
    first_prepared = []
    second_bias.register_post_accumulate_grad_hook(
        lambda _parameter: first_prepared.append(hasattr(first_weight, "main_grad"))
    )
    x = torch.randn(4, 16, device=device, requires_grad=True)
    checkpoint(model, x, use_reentrant=use_reentrant).square().mean().backward()
    assert first_prepared == [False]


def test_fused_wgrad_with_delayed_weight_gradient(distributed_setup):
    """TE writes a delayed fused wgrad in backward_dw() into the buffer prepared before backward."""
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))

    def make_layer(fuse: bool) -> te.Linear:
        torch.manual_seed(1234)
        return te.Linear(
            16,
            16,
            bias=False,
            params_dtype=torch.bfloat16,
            device=device,
            delay_wgrad_compute=True,
            fuse_wgrad_accumulation=fuse,
        )

    fused, copied = make_layer(True), make_layer(False)
    # backward_dw() runs after autograd's end-of-backward callback, so the caller orders the
    # current stream after the reductions with finish_grad_sync().
    with fully_shard_context(device=device, caller_managed_grad_sync=True):
        fully_shard(fused, mesh=mesh, placements=_default_placements())
    with fully_shard_context(device=device, caller_managed_grad_sync=True):
        fully_shard(copied, mesh=mesh, placements=_default_placements())

    x = torch.randn(4, 16, device=device, dtype=torch.bfloat16, requires_grad=True)
    for layer in (fused, copied):
        layer(x).float().square().mean().backward()
        assert layer.weight.grad is None
        assert layer.phase is FsdpModule.Phase.BACKWARD
        layer.backward_dw()
        assert layer.phase is FsdpModule.Phase.RESTING
        layer.context.finish_grad_sync()
    torch.testing.assert_close(fused.weight.grad.full_tensor(), copied.weight.grad.full_tensor())
