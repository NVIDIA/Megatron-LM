# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Transformer Engine fused weight-gradient tests for MFSDP v2.

TE modules built with ``fuse_wgrad_accumulation=True`` write their weight gradients into
``weight.main_grad``, which MFSDP points at its reduce-scatter input. Test buffer release in
``test_memory.py`` and the buffer dtype in ``test_grad_comm_dtype.py``.
"""

import gc
import weakref

import pytest
import torch
import transformer_engine.pytorch as te
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Partial, Replicate, Shard
from torch.utils.checkpoint import checkpoint

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import (
    Placements,
    SchedulePolicy,
    fully_shard,
    fully_shard_context,
    fully_shard_optimizer,
    microbatch,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.module import FsdpModule
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.parameter_group import (
    FsdpParameterGroup,
)


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


def test_fused_grad_buffer_allocate_set_and_take(distributed_setup):
    """allocate zeroes the buffer, set exposes fused views, take packs grads and drops main_grad."""
    _model, group, unsharded = _fused_layer_norm_linear_group(distributed_setup)
    unfused_names = ("layer_norm_weight", "layer_norm_bias", "bias")

    with pytest.raises(RuntimeError, match="not allocated"):
        group.set_fused_main_grads()
    group.allocate_fused_grad_buffer()
    buffer = group._fused_grad_buffer
    # A second allocation before the buffer is taken, e.g. from another unshard(), keeps it.
    group.allocate_fused_grad_buffer()
    assert group._fused_grad_buffer is buffer
    group.set_fused_main_grads()
    main_grad = unsharded["weight"].main_grad
    assert main_grad.shape == unsharded["weight"].shape
    assert main_grad.dtype == group.grad_comm_dtype
    assert torch.count_nonzero(main_grad) == 0
    assert not any(hasattr(unsharded[name], "main_grad") for name in unfused_names)

    main_grad.fill_(1.0)  # Stands in for TE's fused wgrad GEMM.
    for name in unfused_names:
        unsharded[name].grad = torch.full_like(unsharded[name], 2.0)
    assert group.take_fused_grad_buffer() is buffer

    assert not any(hasattr(parameter, "main_grad") for parameter in unsharded.values())
    assert all(parameter.grad is None for parameter in unsharded.values())
    views = {
        fsdp_parameter.fqns[0]: buffer.get_tensor_view(index)
        for index, fsdp_parameter in enumerate(group.fsdp_parameters)
    }
    torch.testing.assert_close(views["weight"], torch.ones_like(views["weight"]))
    for name in unfused_names:
        torch.testing.assert_close(views[name], torch.full_like(views[name], 2.0))
    with pytest.raises(RuntimeError, match="not allocated"):
        group.take_fused_grad_buffer()


def test_fused_grad_buffer_requires_unfused_gradients(distributed_setup):
    """An unfused parameter without a gradient is an error, as on the copy path."""
    _model, group, unsharded = _fused_layer_norm_linear_group(distributed_setup)
    group.allocate_fused_grad_buffer()
    group.set_fused_main_grads()
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

    # When the second layer's backward starts, only its own fused weight is exposed: prefetching
    # allocates the first layer's buffer, but main_grad is set only when that backward starts.
    assert seen["second"] == {"second.weight"}
    assert seen["first"] == {"first.weight"}
    assert with_main_grad("") == set()


@pytest.mark.parametrize("use_reentrant", [False, True], ids=["non_reentrant", "reentrant"])
def test_fused_wgrad_recompute_sets_main_grad_at_each_backward(distributed_setup, use_reentrant):
    """Recomputing a checkpointed region sets no main_grad before its module's backward."""
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
    # layer's backward starts, so the first layer's main_grad must not be set yet.
    first_has_main_grad = []
    second_bias.register_post_accumulate_grad_hook(
        lambda _parameter: first_has_main_grad.append(hasattr(first_weight, "main_grad"))
    )
    x = torch.randn(4, 16, device=device, requires_grad=True)
    checkpoint(model, x, use_reentrant=use_reentrant).square().mean().backward()
    assert first_has_main_grad == [False]


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


# About 0.1 s on an H200: long enough for compute that does not wait to finish first.
_DELAY_CYCLES = 1 << 28


def _delay_reduce_scatter_stream(context):
    """Return a backward pre-hook that stalls the reduce-scatter stream."""

    def hook(_module, _grad_output):
        with torch.cuda.stream(context.reduce_scatter_stream):
            torch.cuda._sleep(_DELAY_CYCLES)

    return hook


def _fused_linear_stack(
    distributed_setup,
    *,
    fuse=True,
    delay_wgrad_compute=False,
    schedule_policy=SchedulePolicy(),
    unify_communication_stream=False,
    caller_managed_grad_sync=False,
):
    """Shard three TE Linears, then their parameterless nn.Sequential root, in one context.

    The root is first in backward order and owns no weights, so it allocates the last layer's
    buffer and its own post-backward only runs after every layer.
    """
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    torch.manual_seed(1234)
    model = nn.Sequential(
        *(
            te.Linear(
                16,
                16,
                bias=not delay_wgrad_compute,
                device=device,
                delay_wgrad_compute=delay_wgrad_compute,
                fuse_wgrad_accumulation=fuse,
            )
            for _ in range(3)
        )
    )
    with fully_shard_context(
        device=device,
        unify_communication_stream=unify_communication_stream,
        caller_managed_grad_sync=caller_managed_grad_sync,
    ) as context:
        for layer in model:
            fully_shard(
                layer, mesh=mesh, placements=_default_placements(), schedule_policy=schedule_policy
            )
        fully_shard(
            model, mesh=mesh, placements=_default_placements(), schedule_policy=schedule_policy
        )
    return model, context


@pytest.mark.parametrize(
    "backward_prefetch_size", [None, 0, 1 << 30], ids=["default", "disabled", "unbounded"]
)
def test_fused_wgrad_buffer_is_allocated_by_the_previous_module(
    distributed_setup, backward_prefetch_size
):
    """Each backward allocates the next module's fused buffer and no further one.

    The parameterless root allocates the last layer's buffer, and each layer allocates its
    predecessor's whatever the all-gather prefetch budget, so at most two are alive.
    """
    model, _context = _fused_linear_stack(
        distributed_setup,
        schedule_policy=SchedulePolicy(backward_prefetch_size=backward_prefetch_size),
    )

    def allocated() -> list[int]:
        return [
            index
            for index, layer in enumerate(model)
            if any(group._fused_grad_buffer is not None for group in layer.parameter_groups)
        ]

    # Registered after fully_shard(), so these run after MFSDP's own backward pre-hooks.
    seen = {}
    for index, layer in enumerate(model):
        layer.register_full_backward_pre_hook(
            lambda _module, _grad_output, index=index: seen.update({index: allocated()})
        )
    x = torch.randn(4, 16, device=distributed_setup.device, requires_grad=True)
    model(x).square().mean().backward()

    assert seen == {2: [1, 2], 1: [0, 1], 0: [0]}
    assert allocated() == []


def test_fused_wgrad_buffer_is_freed_after_its_reduce_scatter(distributed_setup, monkeypatch):
    """Like partial_grad, a fused buffer is dropped once its reduce-scatter is launched."""
    model, _context = _fused_linear_stack(distributed_setup)
    taken = []
    original = FsdpParameterGroup.take_fused_grad_buffer

    def spy(self):
        buffer = original(self)
        taken.append(weakref.ref(buffer))
        return buffer

    monkeypatch.setattr(FsdpParameterGroup, "take_fused_grad_buffer", spy)
    alive_when_first_layer_starts = []

    def record_alive(_module, _grad_output):
        gc.collect()
        alive_when_first_layer_starts.append(sum(ref() is not None for ref in taken))

    model[0].register_full_backward_pre_hook(record_alive)
    x = torch.randn(4, 16, device=distributed_setup.device, requires_grad=True)
    model(x).square().mean().backward()

    # The last two layers have reduced their buffers before the first layer's backward starts.
    assert alive_when_first_layer_starts == [0]
    assert len(taken) == 3


@pytest.mark.parametrize(
    "unify_communication_stream", [False, True], ids=["separate_streams", "unified_stream"]
)
def test_fused_wgrad_waits_for_its_buffer_to_be_zeroed(
    distributed_setup, unify_communication_stream
):
    """TE writes a fused wgrad only after its buffer is zeroed on the reduce-scatter stream.

    The parameterless root allocates the last layer's buffer, and no post-backward runs between
    that allocation and the last layer's backward, so only the wait at the start of that
    backward orders the zeroing first. Stalling the reduce-scatter stream exposes a missing wait.
    """
    grads = {}
    for fuse in (False, True):
        model, context = _fused_linear_stack(
            distributed_setup, fuse=fuse, unify_communication_stream=unify_communication_stream
        )
        # prepend=True: stall before MFSDP's pre-hook allocates the last layer's buffer.
        model.register_full_backward_pre_hook(_delay_reduce_scatter_stream(context), prepend=True)
        torch.manual_seed(5678)
        x = torch.randn(4, 16, device=distributed_setup.device, requires_grad=True)
        model(x).square().mean().backward()
        grads[fuse] = [parameter.grad.to_local().clone() for parameter in model.parameters()]

    for fused_grad, copied_grad in zip(grads[True], grads[False]):
        torch.testing.assert_close(fused_grad, copied_grad, rtol=1e-5, atol=1e-5)


def test_fused_wgrad_delayed_weight_gradients_in_any_order(distributed_setup):
    """Delayed fused wgrads computed in the caller's order still wait for their zeroed buffer."""
    grads = {}
    for fuse in (False, True):
        model, context = _fused_linear_stack(
            distributed_setup, fuse=fuse, delay_wgrad_compute=True, caller_managed_grad_sync=True
        )
        model.register_full_backward_pre_hook(_delay_reduce_scatter_stream(context), prepend=True)
        torch.manual_seed(5678)
        x = torch.randn(4, 16, device=distributed_setup.device, requires_grad=True)
        model(x).square().mean().backward()
        # Forward order, the reverse of backward order.
        for layer in model:
            layer.backward_dw()
        context.finish_grad_sync()
        grads[fuse] = [layer.weight.grad.to_local().clone() for layer in model]

    for fused_grad, copied_grad in zip(grads[True], grads[False]):
        torch.testing.assert_close(fused_grad, copied_grad, rtol=1e-5, atol=1e-5)
