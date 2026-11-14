# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Transformer Engine fused weight-gradient tests for MFSDP v2.

TE modules built with ``fuse_wgrad_accumulation=True`` write their weight gradients into
``weight.main_grad``, which MFSDP points at its reduce-scatter input, or at main_grad itself when
no DP-inner reduction is needed, as on one rank. Test buffer release in ``test_memory.py`` and the
buffer dtype in ``test_gradient_accumulation.py``.
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
    FsdpContext,
    Placements,
    PrefetchOrder,
    RuntimePolicy,
    SchedulePolicy,
    finalize_gradients,
    fully_shard,
    fully_shard_context,
    fully_shard_optimizer,
)
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


def test_fuse_wgrad_accumulation_follows_te_modules(distributed_setup):
    """A group fuses wgrads when a trainable TE module in it has fuse_wgrad_accumulation=True."""
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    mixed = nn.Sequential(
        te.LayerNormLinear(16, 32, device=device, fuse_wgrad_accumulation=True),
        te.Linear(32, 16, device=device, fuse_wgrad_accumulation=False),
    )
    unfused = te.Linear(16, 16, device=device, fuse_wgrad_accumulation=False)
    frozen = te.Linear(16, 16, device=device, fuse_wgrad_accumulation=True)
    frozen.requires_grad_(False)
    with fully_shard_context(FsdpContext(device=device)):
        for model in (mixed, unfused, frozen):
            fully_shard(model, mesh=mesh, placements=_default_placements())

    assert [group.fuse_wgrad_accumulation for group in mixed.parameter_groups] == [True]
    assert [group.fuse_wgrad_accumulation for group in unfused.parameter_groups] == [False]
    assert [group.fuse_wgrad_accumulation for group in frozen.parameter_groups] == [False]


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
    # TE can't scale what it accumulates, so check that grad_scale still applies exactly once.
    with fully_shard_context(FsdpContext(device=device)) as context:
        if variant == "mixed_unit":
            fully_shard(model, mesh=mesh, placements=placements, grad_scale=0.5)
        else:
            fully_shard(model.first, mesh=mesh, placements=placements, grad_scale=0.5)
            fully_shard(model.second, mesh=mesh, placements=placements, grad_scale=0.5)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    fully_shard_optimizer(optimizer)
    return model, context, optimizer


def _train_fused_wgrad_model(model, context, optimizer, microbatches, set_to_none):
    """Run five optimizer steps over ``microbatches`` and return every micro-batch loss."""
    losses = []
    for _ in range(5):
        optimizer.zero_grad(set_to_none=set_to_none)
        for index, (x, target) in enumerate(microbatches):
            with finalize_gradients(context, enabled=index == len(microbatches) - 1):
                loss = torch.nn.functional.mse_loss(model(x).float(), target)
                losses.append(loss.detach())
                (loss / len(microbatches)).backward()
        optimizer.step()
    return torch.stack(losses)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize("set_to_none", [True, False])
@pytest.mark.parametrize("num_microbatches", [1, 3])
@pytest.mark.parametrize(
    "variant",
    [
        "per_layer",
        "mixed_unit",
        "checkpointed",
        "checkpointed_reentrant",
        "hsdp",
        "no_inner_reduction",
    ],
)
def test_fused_wgrad_losses_match_unfused(
    distributed_setup, variant, num_microbatches, set_to_none, dtype
):
    """TE layers writing fused wgrads train like the copy path."""
    device = distributed_setup.device
    world_size = distributed_setup.world_size
    if variant == "hsdp":
        if world_size < 4 or world_size % 2 != 0:
            pytest.skip("HSDP requires an even number of at least 4 ranks for a 2-D DP mesh.")
        mesh = init_device_mesh(
            device.type, (2, world_size // 2), mesh_dim_names=("dp_outer", "dp_inner")
        )
        placements = _hsdp_placements()
    elif variant == "no_inner_reduction":
        # DP-inner size 1, as for experts under expert parallelism.
        mesh = init_device_mesh(
            device.type, (world_size, 1), mesh_dim_names=("dp_outer", "dp_inner")
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
    """main_grad is set for a fused module's backward and removed afterwards."""
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    model = _FusedWgradModel(device, torch.float32, fuse_first=True, fuse_second=True)
    with fully_shard_context(FsdpContext(device=device)):
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

    loss = model(x).square().mean()
    assert with_main_grad("") == set()
    loss.backward()

    # When the second layer's backward starts, only its own parameters have main_grad:
    # prefetching allocates the first layer's buffer, but main_grad is set when its backward starts.
    assert seen["second"] == {"second.weight", "second.bias"}
    assert seen["first"] == {
        "first.weight",
        "first.bias",
        "first.layer_norm_weight",
        "first.layer_norm_bias",
    }
    assert with_main_grad("") == set()


@pytest.mark.parametrize("use_reentrant", [False, True], ids=["non_reentrant", "reentrant"])
def test_fused_wgrad_recompute_sets_main_grad_at_each_backward(distributed_setup, use_reentrant):
    """Recomputing a checkpointed region sets no main_grad before its module's backward."""
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    model = _FusedWgradModel(device, torch.float32, fuse_first=True, fuse_second=True)
    with fully_shard_context(FsdpContext(device=device)):
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
    for layer in (fused, copied):
        with fully_shard_context(
            FsdpContext(device=device, runtime_policy=RuntimePolicy(caller_managed_grad_sync=True))
        ):
            fully_shard(layer, mesh=mesh, placements=_default_placements())

    x = torch.randn(4, 16, device=device, dtype=torch.bfloat16, requires_grad=True)
    for layer in (fused, copied):
        [group] = layer.parameter_groups
        [sharded_weight] = [fsdp_parameter.sharded for fsdp_parameter in group.fsdp_parameters]
        layer(x).float().square().mean().backward()
        # Reduction installs the sharded gradient, which waits for backward_dw().
        assert sharded_weight.grad is None
        layer.backward_dw()
        layer.context.finish_grad_sync()
        assert sharded_weight.grad is not None
    torch.testing.assert_close(fused.weight.grad.full_tensor(), copied.weight.grad.full_tensor())


def test_fused_wgrad_without_inner_reduction_writes_into_main_grad(distributed_setup, monkeypatch):
    """With DP-inner size 1, TE writes into main_grad and no reduce-scatter input is allocated."""
    device = distributed_setup.device
    mesh = init_device_mesh(
        device.type, (distributed_setup.world_size, 1), mesh_dim_names=("dp_outer", "dp_inner")
    )
    layer = te.Linear(16, 16, bias=False, device=device, fuse_wgrad_accumulation=True)
    with fully_shard_context(FsdpContext(device=device)):
        fully_shard(layer, mesh=mesh, placements=_hsdp_placements())
    [group] = layer.parameter_groups
    [weight] = [fsdp_parameter.unsharded for fsdp_parameter in group.fsdp_parameters]

    allocated = []
    original = FsdpParameterGroup.allocate_partial_grad_buffer

    def spy(self):
        allocated.append(self)
        return original(self)

    monkeypatch.setattr(FsdpParameterGroup, "allocate_partial_grad_buffer", spy)
    # Registered after fully_shard(), so this runs after MFSDP's backward pre-hook sets main_grad.
    storages = []
    layer.register_full_backward_pre_hook(
        lambda _module, _grad_output: storages.append(weight.main_grad.untyped_storage().data_ptr())
    )
    layer(torch.randn(4, 16, device=device)).square().sum().backward()

    assert storages == [group.main_grad.local_buffer.untyped_storage().data_ptr()]
    assert allocated == []


def test_fused_wgrad_accumulates_across_uses_and_backward_calls(distributed_setup):
    """A layer used twice, and backward calls without zero_grad() in between, add up.

    On one rank TE accumulates into main_grad itself. The second use's backward must not zero it
    again, and the second call must add to the first call's gradient, which grad_scale has scaled.
    """
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    x = torch.randn(4, 16, device=device)
    grads = []
    for fuse in (False, True):
        torch.manual_seed(1234)
        layer = te.Linear(16, 16, bias=False, device=device, fuse_wgrad_accumulation=fuse)
        # Unit-scale weights keep both uses' contributions well above assert_close's atol.
        with torch.no_grad():
            nn.init.normal_(layer.weight, std=16**-0.5)
        # Registration order has the layer once; execution calls it twice.
        with fully_shard_context(FsdpContext(device=device, prefetch_order=PrefetchOrder.RECORDED)):
            fully_shard(layer, mesh=mesh, placements=_default_placements(), grad_scale=0.5)
        for _ in range(2):
            layer(layer(x)).square().sum().backward()
        grads.append(layer.weight.grad.to_local())
    torch.testing.assert_close(grads[1], grads[0])


# About 0.1 s on an H200: long enough for compute that does not wait to finish first.
_DELAY_CYCLES = 1 << 28


def _delay_inner_reduction_stream(context):
    """Return a backward pre-hook that stalls the inner reduction stream."""

    def hook(_module, _grad_output):
        with torch.cuda.stream(context.inner_reduction_stream):
            torch.cuda._sleep(_DELAY_CYCLES)

    return hook


def _stalled_backward_grads(distributed_setup, model, context, *, delay_wgrad_compute=False):
    """Return the local gradients of a backward run with the inner reduction stream stalled.

    A first backward without the stall loads every kernel: with lazy module loading, a kernel's
    first launch can block the CPU until the stalled stream drains, hiding a missing wait.
    """
    torch.manual_seed(5678)
    x = torch.randn(4, 16, device=distributed_setup.device, requires_grad=True)

    def backward():
        model(x).square().mean().backward()
        if delay_wgrad_compute:
            # Forward order, the reverse of backward order.
            for layer in model:
                layer.backward_dw()
            context.finish_grad_sync()

    backward()
    model.zero_grad()
    # prepend=True: stall before MFSDP's pre-hook allocates the last layer's buffer.
    model.register_full_backward_pre_hook(_delay_inner_reduction_stream(context), prepend=True)
    backward()
    return [parameter.grad.to_local().clone() for parameter in model.parameters()]


def _assert_fused_grads_match(grads):
    """Compare fused against copied gradients, which must be large enough to expose zeros."""
    # A buffer zeroed after TE wrote into it leaves zeros, which atol accepts for tiny grads.
    # A rank's shard of a gradient can be empty, so take each gradient's largest entry over all
    # ranks, in one collective issued before any rank can fail an assertion.
    largest = torch.stack(
        [grad.abs().amax() if grad.numel() else grad.new_zeros(()) for grad in grads[False]]
    )
    torch.distributed.all_reduce(largest, op=torch.distributed.ReduceOp.MAX)
    assert (largest > 1e-3).all(), largest
    for fused_grad, copied_grad in zip(grads[True], grads[False]):
        torch.testing.assert_close(fused_grad, copied_grad, rtol=1e-5, atol=1e-5)


def _fused_linear_stack(
    distributed_setup,
    *,
    fuse=True,
    delay_wgrad_compute=False,
    schedule_policy=SchedulePolicy(),
    overlap_allgather_reduction=True,
    caller_managed_grad_sync=False,
):
    """Shard three TE Linears, then their parameterless nn.Sequential root, in one context.

    The root is first in backward order and owns no weights, so it allocates the last layer's
    buffer and its own post-backward only runs after every layer.
    """
    if distributed_setup.world_size < 2:
        pytest.skip("On one rank no DP-inner reduction is needed, so there is no fused buffer.")
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
    # TE's default initialization leaves these gradients near 1e-6, below assert_close's
    # atol, where a wiped buffer would still match. Unit-scale weights keep them well above it.
    with torch.no_grad():
        for layer in model:
            nn.init.normal_(layer.weight, std=16**-0.5)
    runtime_policy = RuntimePolicy(
        overlap_allgather_reduction=overlap_allgather_reduction,
        caller_managed_grad_sync=caller_managed_grad_sync,
    )
    with fully_shard_context(FsdpContext(device=device, runtime_policy=runtime_policy)) as context:
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
            if any(group.partial_grad is not None for group in layer.parameter_groups)
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
    """A fused buffer is dropped once its reduce-scatter is launched, like the copy path's."""
    model, _context = _fused_linear_stack(distributed_setup)
    allocated = []
    original = FsdpParameterGroup.allocate_partial_grad_buffer

    def spy(self):
        buffer = original(self)
        allocated.append(weakref.ref(buffer))
        return buffer

    monkeypatch.setattr(FsdpParameterGroup, "allocate_partial_grad_buffer", spy)
    alive_when_first_layer_starts = []

    def record_alive(_module, _grad_output):
        gc.collect()
        alive_when_first_layer_starts.append(sum(ref() is not None for ref in allocated))

    model[0].register_full_backward_pre_hook(record_alive)
    x = torch.randn(4, 16, device=distributed_setup.device, requires_grad=True)
    model(x).square().mean().backward()

    # When the first layer's backward starts, the last two layers have reduced theirs.
    assert alive_when_first_layer_starts == [1]
    assert len(allocated) == 3


@pytest.mark.parametrize(
    "overlap_allgather_reduction", [True, False], ids=["separate_streams", "unified_stream"]
)
def test_fused_wgrad_waits_for_its_buffer_to_be_zeroed(
    distributed_setup, overlap_allgather_reduction
):
    """TE writes a fused wgrad only after its buffer is zeroed on the inner reduction stream.

    The parameterless root allocates the last layer's buffer, and no reduction runs between that
    allocation and the last layer's backward, so only the wait at the start of that backward
    orders the zeroing first. Stalling that stream exposes a missing wait.
    """
    grads = {}
    for fuse in (False, True):
        model, context = _fused_linear_stack(
            distributed_setup, fuse=fuse, overlap_allgather_reduction=overlap_allgather_reduction
        )
        grads[fuse] = _stalled_backward_grads(distributed_setup, model, context)
    _assert_fused_grads_match(grads)


def test_fused_wgrad_delayed_weight_gradients_in_any_order(distributed_setup):
    """Delayed fused wgrads computed in the caller's order still wait for their zeroed buffer."""
    grads = {}
    for fuse in (False, True):
        model, context = _fused_linear_stack(
            distributed_setup, fuse=fuse, delay_wgrad_compute=True, caller_managed_grad_sync=True
        )
        grads[fuse] = _stalled_backward_grads(
            distributed_setup, model, context, delay_wgrad_compute=True
        )
    _assert_fused_grads_match(grads)
