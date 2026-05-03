"""
Fully Sharded Data Parallel (FSDP) Implementation

This module provides a PyTorch FSDP2-compatible API layer for Megatron Core.
It wraps modules with FSDP sharding semantics, managing parameter sharding,
unsharding, gradient reduction, and checkpointing.

Key components:
- fully_shard(): Main API to wrap a module with FSDP
- FSDPModule: Mixin class added to wrapped modules
- ParameterGroup: Groups parameters with shared buffers
- DataParallelBuffer: Flat buffer for parameter/gradient storage

Sharding Strategies:
- "no_shard": No sharding (like DDP)
- "optim": Shard optimizer state only
- "optim_grads": Shard gradients and optimizer state
- "optim_grads_params": Full sharding (like ZeRO-3)
"""

import functools
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import Shard
from torch.utils._pytree import tree_flatten, tree_map, tree_unflatten

from .param_group import ParameterGroup
from .utils import ParamGroupIdx, RegisterFSDPBackwardFunction, _replace_module_parameter


def fully_shard(
    module: nn.Module,
    *,
    mesh: Optional[DeviceMesh] = None,
    reshard_after_forward: Optional[bool | int] = None,  # TODO: implement
    shard_placement_fn: Optional[
        Callable[[nn.Parameter], Optional[Shard]]
    ] = None,  # TODO: implement
    mp_policy: Optional["MixedPrecisionPolicy"] = None,  # TODO: implement
    offload_policy: Optional["OffloadPolicy"] = None,  # TODO: implement
    ignored_params: Optional[set[nn.Parameter]] = None,
    # --- Megatron-FSDP specific options ---
    enable_unshard_prefetch: bool = True,
    enable_async_reduce_grad: bool = True,
) -> nn.Module:
    """
    Wrap a module with FSDP sharding semantics.

    This function:
    1. Converts the module class to FSDPModule dynamically (mixin pattern)
    2. Groups parameters by (device, dtype, requires_grad)
    3. Creates ParameterGroup for each group with dedicated buffers
    4. Registers forward/backward hooks for unshard/reshard/reduce
    5. Replaces module parameters with DTensor representations

    Args:
        module: The module to wrap with FSDP.
        mesh: DeviceMesh defining the distributed topology.
        ignored_params: Parameters to ignore during sharding.

    Returns:
        The module wrapped with FSDP (class modified to be FSDPModule).

    Example:
        >>> mesh = init_device_mesh("cuda", (8,))
        >>> model = ToyModel(dim=512, n_layers=3)
        >>> for layer in model.layers:
        ...     fully_shard(layer, mesh=mesh)
        >>> fully_shard(model, mesh=mesh)
    """
    if isinstance(module, FSDPModule):
        raise ValueError(
            "The input module has already been fully sharded. "
            "Please do not call fully_shard on the same module more than once."
        )

    # Convert module class to FSDPModule via mixin pattern.
    # This allows isinstance checks while preserving the original class.
    cls = module.__class__
    new_cls = type(f"FSDP{cls.__name__}", (FSDPModule, cls), {})
    module.__class__ = new_cls

    # Initialize FSDP state and parameter groups
    module._init_named_param_groups(mesh, ignored_params, mp_policy=mp_policy)
    module._init_fsdp_state(
        enable_unshard_prefetch=enable_unshard_prefetch,
        enable_async_reduce_grad=enable_async_reduce_grad,
    )
    module._init_param_main_grad_func()

    # Register hooks for unshard/reshard during forward/backward
    _register_forward_pre_hook(module)
    _register_forward_hook(module)
    _register_backward_pre_hook(module)
    _register_backward_hook(module)

    # Initial reshard to release parameters after setup
    module.reshard()

    return module


class FSDPModule(nn.Module):
    """
    Mixin class for FSDP-wrapped modules.

    This class is dynamically added to wrapped modules and provides
    methods for managing parameter sharding state:
    - unshard(): All-gather parameters before forward
    - reshard(): Release unsharded buffer after forward
    - reduce_grad(): Reduce-scatter gradients after backward
    """

    def _init_named_param_groups(
        self,
        mesh: Optional[DeviceMesh],
        ignored_params: Optional[set],
        mp_policy: Optional["MixedPrecisionPolicy"] = None,
    ):
        """
        Initialize parameter groups and build param name mapping.

        This method:
        1. Collects ignored modules (nested FSDP modules)
        2. Materializes meta modules to actual devices
        3. Groups parameters by (device, dtype, requires_grad)
        4. Builds parameter name to parameter mapping
        """
        ignored_params = ignored_params or set()
        ignored_modules = set()

        # Collect nested FSDP modules as ignored
        for _, child in self.named_modules():
            if child is not self and isinstance(child, FSDPModule):
                ignored_params.update(child.parameters())
                for child_submodule in child.modules():
                    ignored_modules.add(child_submodule)

        # Materialize meta parameters to actual device
        self._materialize_meta_module(ignored_modules)

        # Create parameter groups
        fsdp_param_groups = _get_module_fsdp_param_groups(
            self, mesh, ignored_params=ignored_params, mp_policy=mp_policy
        )
        setattr(self, "_fsdp_param_groups", fsdp_param_groups)

        # Build param name to param mapping for later lookup
        param_to_name = {p: n for n, p in self.named_parameters()}
        self._named_param_groups = []

        for fsdp_param_group in fsdp_param_groups:
            param_names = []
            for param in fsdp_param_group.params:
                param_name = param_to_name[param]
                param_names.append(param_name)
            self._named_param_groups.append((param_names, fsdp_param_group))

    def _init_param_main_grad_func(self):
        """
        Initialize main gradient getter function for each parameter.

        This creates a closure that fetches the gradient from the
        gradient buffer when accessed. It handles both sharded and
        unsharded gradient buffers.
        """

        def main_grad_getter(p):
            """Get main gradient from buffer with proper offset/size."""
            gbuf = p._gbuf
            item_id = p._item_id

            gbuf_data = gbuf.fetch_unsharded_buffer()
            assert gbuf_data is not None
            assert gbuf_data.numel() > 0

            # Get offset and size from buffer index
            offset, size = gbuf.buffer_index._get_item_offset(item_id)
            grad_data = gbuf_data[offset : offset + size].view(p.shape)

            return grad_data

        # Attach getter to each parameter
        for param_group in self._fsdp_param_groups:
            for param in param_group.params:
                setattr(param, "_gbuf", param_group.main_grad_buffer)
                setattr(param, "_item_id", param_group.param_idx[param])
                param.get_main_grad = main_grad_getter.__get__(param)

    def _materialize_meta_module(self, ignored_modules: set):
        """
        Materialize meta parameters to actual device and initialize.

        This is needed for large models that cannot fit in a single GPU.
        Meta parameters are moved to the current device and reset.
        """
        materialization_device = torch.cuda.current_device()
        for m in self.modules():
            if m in ignored_modules:
                continue
            # Skip modules that don't have meta parameters
            if all(not p.is_meta for p in m.parameters(recurse=False)):
                continue

            # Move to device and initialize
            m.to_empty(device=materialization_device, recurse=False)
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()
            if hasattr(m, "_reset_parameters"):
                m._reset_parameters()

    def _init_fsdp_state(self, enable_unshard_prefetch, enable_async_reduce_grad):
        """Initialize FSDP state and mark nested FSDP modules as non-root."""
        forward_order = [child for child in self.modules() if isinstance(child, FSDPModule)]
        root_context = _FSDPRootContext(
            ag_stream=(
                torch.cuda.Stream() if enable_unshard_prefetch else torch.cuda.current_stream()
            ),
            rs_stream=(
                torch.cuda.Stream() if enable_async_reduce_grad else torch.cuda.current_stream()
            ),
            forward_order=forward_order,
            reduce_grad_buckets={id(module): [] for module in forward_order},
            unshard_done_events={id(module): None for module in forward_order},
            enable_unshard_prefetch=enable_unshard_prefetch,
            enable_async_reduce_grad=enable_async_reduce_grad,
        )
        setattr(self, "_fsdp_state", _FSDPState())
        setattr(self, "_fsdp_root_context", root_context)
        for child in self.modules():
            if child is not self and isinstance(child, FSDPModule):
                child._init_fsdp_state(
                    enable_unshard_prefetch=enable_unshard_prefetch,
                    enable_async_reduce_grad=enable_async_reduce_grad,
                )
                child._fsdp_state._is_root = False
                setattr(child, "_fsdp_root_context", root_context)

    def unshard(self, async_op: bool = False, bwd_pass: bool = False):
        """
        Unshard parameters by all-gathering from the sharded buffer.

        This is called pre-forward to make parameters available for
        computation. After unsharding, each param.data points to
        the full (unsharded) tensor.
        """
        ctx = self._fsdp_root_context
        stream = ctx.ag_stream if async_op else torch.cuda.current_stream()

        # Unshard this module and optionally prefetch next modules in the forward/backward pass
        if async_op:
            prefetch_modules = self._get_prefetch_next_modules(bwd_pass=bwd_pass)
        else:
            prefetch_modules = []
        for module in [self] + prefetch_modules:
            if ctx.unshard_done_events[id(module)] is not None:
                continue  # Skip if unshard already issued for this module

            # Unshard parameters for this module
            for param_names, param_group in module._named_param_groups:
                # Optional NaN checking for debugging
                if getattr(module, "_enable_nan_checks", False):
                    for name, dist_param in zip(param_names, param_group.dist_params):
                        assert not torch.isnan(
                            dist_param._local_tensor
                        ).any(), f"NaN detected in dist param for parameter {name}"

                with torch.cuda.stream(stream):
                    param_group.unshard(async_op=async_op)

            # Record event to track when unshard is done for this module
            if async_op:
                event = stream.record_event()
                ctx.unshard_done_events[id(module)] = event

        # Ensure unshard is complete before forward
        if ctx.unshard_done_events[id(self)] is not None:
            ctx.unshard_done_events[id(self)].wait()
            ctx.unshard_done_events[id(self)] = None

        # Replace module parameters with unsharded versions
        for param_names, param_group in self._named_param_groups:
            for name, param in zip(param_names, param_group.params):
                _replace_module_parameter(self, name, param)

            # Optional NaN checking for debugging
            # FIXME: Need cuda synchronization before checking for NaN to ensure data is ready.
            # if getattr(self, "_enable_nan_checks", False):
            #     for name, param in zip(param_names, param_group.params):
            #         assert not torch.isnan(param).any(), f"NaN detected in parameter {name}"

    def _get_prefetch_next_modules(self, bwd_pass: bool = False) -> List["FSDPModule"]:
        """Prefetch the next module in the forward/backward pass."""
        ctx = self._fsdp_root_context
        assert self in ctx.forward_order, "Current module not found in forward module order"

        if bwd_pass:
            module_order = list(reversed(ctx.forward_order))
        else:
            module_order = ctx.forward_order

        i = None
        for i, module in enumerate(module_order):
            if module is self:
                break
        assert i is not None, "Current module index not found in forward module order"
        if i + 1 >= len(module_order):
            return []  # No next module to prefetch

        return [module_order[i + 1]]

    def reshard(self):
        """Reshard parameters by replacing with sharded DTensors."""
        ctx = self._fsdp_root_context
        for param_names, param_group in self._named_param_groups:
            param_group.reshard()
            for name, dist_param in zip(param_names, param_group.dist_params):
                _replace_module_parameter(self, name, dist_param)
        ctx.unshard_done_events[id(self)] = None  # Clear unshard event for this module

    def reduce_grad(self, async_op: bool = False):
        """
        Reduce gradients across data-parallel ranks.

        This is called post-backward to:
        1. Copy gradients to main gradient buffer
        2. Perform all-reduce or reduce-scatter
        3. Install reduced gradients to distributed parameters
        """
        ctx = self._fsdp_root_context
        stream = ctx.rs_stream if async_op else torch.cuda.current_stream()

        # Handle pending reduce events before this module to ensure memory is freed in a timely manner.
        if async_op:
            backward_order = list(reversed(ctx.forward_order))
            for i, module in enumerate(backward_order):
                if i - 2 >= 0:
                    buckets = ctx.reduce_grad_buckets[id(backward_order[i - 2])]
                    while len(buckets) > 0:
                        event, param_group = buckets.pop()
                        event.wait()
                        param_group.release_grad_buffer()
                if module is self:
                    break

        # Perform reduction for this module
        for param_names, param_group in self._named_param_groups:
            if not param_group.requires_grad:
                continue

            # NaN check before reduction
            if getattr(self, "_enable_nan_checks", False):
                for param in param_group.params:
                    if param.grad is not None:
                        assert not torch.isnan(param.grad).any(), "NaN in parameter grad"

            # Copy .grad → main grad buffer on main stream (fast memcpy).
            # When gradient_accumulation_fusion is active for FSDP params, the backward
            # kernel writes directly into main_grad (weight.main_grad = get_main_grad() in
            # layers.py) and sets grad_added_to_main_grad=True.  In that case we must NOT
            # zero main_grad, and there is no .grad to copy.
            for name, param in zip(param_names, param_group.params):
                main_grad = param.get_main_grad()
                if param.grad is None:
                    if not getattr(param, 'grad_added_to_main_grad', False):
                        main_grad.zero_()
                else:
                    main_grad.copy_(param.grad.detach())
                    del param.grad

            if async_op:
                # ---- Overlapped path ----
                # Switch to rs_stream for the reduce-scatter kernel
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    param_group.reduce_grad()
                    event = torch.cuda.Event()
                    event.record()

                    ctx.reduce_grad_buckets[id(self)].append((event, param_group))
            else:
                # ---- Non-overlapped path ----
                # Reduce gradients immediately and release grad buffer
                param_group.reduce_grad()
                param_group.release_grad_buffer()

            # Install reduced gradients to distributed parameters
            for name, param, dist_param, dist_grad in zip(
                param_names, param_group.params, param_group.dist_params, param_group.dist_grads
            ):
                if param.requires_grad and dist_grad is not None:
                    with torch.cuda.stream(stream):
                        dist_grad = dist_grad.to(dist_param.dtype)
                    setattr(dist_param, "grad", dist_grad)

            # NaN check after reduction
            if getattr(self, "_enable_nan_checks", False):
                for name, dist_grad in zip(param_names, param_group.dist_grads):
                    if dist_grad is not None:
                        assert not torch.isnan(
                            dist_grad._local_tensor
                        ).any(), f"NaN in dist grad for parameter {name}"

    @torch.no_grad()
    def _scale_gradients(self, scaling_factor: float):
        """Scale gradients by a factor (e.g., for loss scaling)."""
        for _, child in self.named_modules():
            if not isinstance(child, FSDPModule):
                continue
            for param_group in child._fsdp_param_groups:
                for dist_grad in param_group.dist_grads:
                    dist_grad._local_tensor.mul_(scaling_factor)

    def _zero_grad_buffer(self):
        """Zero the gradient buffer for all parameter groups."""
        for _, child in self.named_modules():
            if not isinstance(child, FSDPModule):
                continue
            for param_group in child._fsdp_param_groups:
                if param_group.main_grad_buffer is not None:
                    param_group.main_grad_buffer.data.zero_()

    def _copy_main_weights_to_model_weights(self):
        """Copy main weight buffer to model weight buffer."""
        for _, child in self.named_modules():
            if not isinstance(child, FSDPModule):
                continue
            for param_group in child._fsdp_param_groups:
                if param_group.main_weight_buffer is None:
                    continue
                param_group.model_weight_buffer.data.copy_(param_group.main_weight_buffer.data)

        # Also zero main grads to avoid stale gradients after weight copy
        self._zero_main_grads()

    def _zero_main_grads(self):
        """Zero the main gradient buffer for all parameter groups."""
        for _, child in self.named_modules():
            if not isinstance(child, FSDPModule):
                continue
            for param_group in child._fsdp_param_groups:
                if param_group.main_grad_buffer is not None:
                    param_group.main_grad_buffer.data.zero_()

    def _set_nan_check(self, enable_nan_checks: bool):
        """Enable or disable NaN checking."""
        for _, child in self.named_modules():
            if not isinstance(child, FSDPModule):
                continue
            setattr(child, "_enable_nan_checks", enable_nan_checks)

        if enable_nan_checks:
            for name, param in self.named_parameters():
                if isinstance(param, DTensor):
                    param_data = param.data._local_tensor
                else:
                    param_data = param.data
                assert not torch.isnan(param_data).any(), f"NaN detected in parameter {name}"
            for child in self.modules():
                if not isinstance(child, FSDPModule):
                    continue
                for param_group in child._fsdp_param_groups:
                    for param in param_group.params:
                        wbuf = param_group.model_weight_buffer
                        param_data = wbuf.get_item(param_group.param_idx[param], only_shard=False)
                        assert not torch.isnan(
                            param_data
                        ).any(), "NaN detected in model weight buffer"


class _FSDPState:
    """
    Internal state for FSDP module tracking.

    Attributes:
        _is_root: Whether this is the root FSDP module (handles final callback).
        _post_backward_callback_queued: Whether callback is queued for execution.
    """

    def __init__(self):
        self._is_root = True
        self._post_backward_callback_queued = False


def _get_module_fsdp_param_groups(
    module: nn.Module,
    mesh: Optional[DeviceMesh] = None,
    ignored_params: Optional[set[nn.Parameter]] = None,
    mp_policy: Optional["MixedPrecisionPolicy"] = None,
) -> List[ParameterGroup]:
    """
    Group module parameters by (device, dtype, requires_grad) and create ParameterGroups.

    Parameters are grouped because they share the same buffer management
    and sharding strategy. Each group gets its own DataParallelBuffer.
    """
    param_groups = {}

    for param in module.parameters():
        if ignored_params is not None and param in ignored_params:
            continue

        # Group by (device, dtype, requires_grad)
        param_attrs = (param.device, param.dtype, param.requires_grad)
        if param_attrs not in param_groups:
            param_groups[param_attrs] = []
        param_groups[param_attrs].append(param)

    # Create ParameterGroup for each group
    fsdp_param_groups = []
    for i, params in enumerate(param_groups.values()):
        fsdp_param_groups.append(
            ParameterGroup(
                params,
                mesh=mesh,
                param_group_id=ParamGroupIdx(id(module), i),
                main_params_dtype=mp_policy.main_params_dtype if mp_policy is not None else None,
                main_grads_dtype=mp_policy.main_grads_dtype if mp_policy is not None else None,
            )
        )

    return fsdp_param_groups


def _register_forward_pre_hook(module: FSDPModule):
    """Register pre-forward hook to unshard parameters."""

    def unshard_param_groups(module, *unused):
        ctx = module._fsdp_root_context
        module.unshard(async_op=ctx.enable_unshard_prefetch, bwd_pass=False)

    module._mfsdp_forward_pre_hook = module.register_forward_pre_hook(
        unshard_param_groups, prepend=True
    )


def _register_forward_hook(module: FSDPModule):
    """Register post-forward hook to reshard parameters."""

    def reshard_param_groups(module, *unused):
        module.reshard()

    module._mfsdp_forward_hook = module.register_forward_hook(reshard_param_groups)


def _register_backward_pre_hook(module: FSDPModule):
    """
    Register backward pre-hook to handle gradient computation.

    This uses a custom backward hook that attaches to output tensors
    to trigger unshard at the right time during backward pass.
    """

    def create_custom_backward_hook(module: FSDPModule, custom_backward_handler: Callable):
        """Create a custom backward hook attached to output tensors."""

        @torch.compiler.disable
        def forward_hook(_module, inputs, output):
            # View-as to avoid output being the same tensor object
            output = tree_map(lambda t: t.view_as(t) if torch.is_tensor(t) else t, output)

            # Collect tensor outputs
            output_list = []
            if isinstance(output, torch.Tensor):
                output_list = [output]
            elif isinstance(output, (tuple, list)):
                output_list = [t for t in output if isinstance(t, torch.Tensor)]

            # Register pre-backward hook on output tensors
            # This triggers when gradients are computed
            torch.autograd.graph.register_multi_grad_hook(
                output_list, lambda grads: custom_backward_handler(_module, grads), mode="any"
            )
            return output

        return module.register_forward_hook(forward_hook)

    def pre_backward_hook(module: FSDPModule, grads):
        """Hook called before backward pass for this module."""
        ctx = module._fsdp_root_context
        setattr(module, "post_backward_issued", False)
        for param_group in module._fsdp_param_groups:
            for param in param_group.params:
                setattr(param, "grad_added_to_main_grad", False)
        if module._fsdp_state._is_root and not module._fsdp_state._post_backward_callback_queued:
            _register_post_backward_final_callback(module._fsdp_state, module)
        module.unshard(async_op=ctx.enable_unshard_prefetch, bwd_pass=True)

    module._mfsdp_backward_pre_hook = create_custom_backward_hook(
        module, custom_backward_handler=pre_backward_hook
    )


def _register_backward_hook(module: FSDPModule):
    """
    Register backward hook using autograd Function.

    This inserts a RegisterFSDPBackwardFunction in the backward pass
    that triggers reshard and reduce_grad after gradients are computed.
    """

    def post_backward(module: FSDPModule):
        """Hook called after backward pass for this module."""
        ctx = module._fsdp_root_context
        module.reshard()
        module.reduce_grad(async_op=ctx.enable_async_reduce_grad)
        module.post_backward_issued = True

    @torch.compiler.disable
    def _register_post_backward_hook(
        post_backward_hook: Callable,
        module: nn.Module,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ):
        """
        Register a post-backward hook by inserting an autograd Function.

        This approach works by registering a pre-forward hook that wraps
        input tensors in an autograd Function. The Function's backward
        calls the post_backward_hook after gradients are computed.
        """
        if not torch.is_grad_enabled():
            return args, kwargs

        # Flatten args and kwargs
        args_list, args_spec = tree_flatten(args)
        kwargs_list, kwargs_spec = tree_flatten(kwargs)
        args_kwargs_list = list(args_list) + list(kwargs_list)

        # Filter to tensors with gradients
        inp_tensor_indices: List[int] = []
        inp_tensors: List[torch.Tensor] = []
        for i, obj in enumerate(args_kwargs_list):
            if torch.is_tensor(obj) and obj.requires_grad:
                inp_tensor_indices.append(i)
                inp_tensors.append(obj)

        if len(inp_tensors) == 0:
            return args, kwargs

        # Wrap inputs in autograd Function
        # The Function's backward will call post_backward_hook
        inp_tensors = RegisterFSDPBackwardFunction.apply(
            functools.partial(post_backward_hook, module), *inp_tensors
        )

        # Restore args and kwargs
        for inp_tensor_idx, inp_tensor in zip(inp_tensor_indices, inp_tensors):
            args_kwargs_list[inp_tensor_idx] = inp_tensor
        args_list = args_kwargs_list[: len(args_list)]
        kwargs_list = args_kwargs_list[len(args_list) :]
        args = tree_unflatten(args_list, args_spec)
        kwargs = tree_unflatten(kwargs_list, kwargs_spec)

        return args, kwargs

    module._mfsdp_backward_hook = module.register_forward_pre_hook(
        functools.partial(_register_post_backward_hook, post_backward), with_kwargs=True
    )


def _register_post_backward_final_callback(state: _FSDPState, module: nn.Module) -> None:
    """
    Register the final callback that runs after all backward passes complete.

    This is only registered by the root FSDP module to avoid duplicate
    callbacks. It reshards all modules and reduces gradients at the end
    of the backward pass.
    """
    assert state._is_root, "Only root FSDP should register post-backward callback"
    if state._post_backward_callback_queued:
        return

    def _post_backward_final_callback(root_state: _FSDPState, root_module: nn.Module):
        """Final callback: reshard all modules and reduce gradients."""
        ctx = root_module._fsdp_root_context
        stream = ctx.rs_stream
        for module in reversed(ctx.forward_order):
            if module.post_backward_issued:
                continue
            module.reshard()
            module.reduce_grad(async_op=ctx.enable_async_reduce_grad)
        for buckets in ctx.reduce_grad_buckets.values():
            while len(buckets) > 0:
                event, param_group = buckets.pop()
                event.wait()
                param_group.release_grad_buffer()
        torch.cuda.current_stream().wait_stream(stream)
        root_state._post_backward_callback_queued = False

    state._post_backward_callback_queued = True
    Variable._execution_engine.queue_callback(
        functools.partial(_post_backward_final_callback, state, module)
    )


@dataclass
class _FSDPRootContext:
    """
    Runtime context shared across all FSDP modules within a single root.

    This object coordinates CUDA streams, execution ordering, and async
    communication overlap (all-gather / reduce-scatter) during forward
    and backward passes.
    """

    # ------------------------------------------------------------------
    # CUDA streams (communication overlap)
    # ------------------------------------------------------------------
    ag_stream: torch.cuda.Stream  # all-gather / unshard stream
    rs_stream: torch.cuda.Stream  # reduce-scatter stream

    # ------------------------------------------------------------------
    # Forward execution ordering
    # ------------------------------------------------------------------
    forward_order: List[FSDPModule] = field(default_factory=list)
    """
    FSDP modules in actual forward execution order.

    This ordering is used to:
    - Schedule prefetching of parameters (unshard)
    - Ensure correct overlap between compute and communication
    """

    # ------------------------------------------------------------------
    # Unshard (all-gather) tracking
    # ------------------------------------------------------------------
    unshard_done_events: Dict[int, torch.cuda.Event] = field(default_factory=dict)
    """
    Maps module_id -> CUDA event signaling completion of parameter unshard.

    Used to enforce correct dependency between all-gather and compute.
    """

    enable_unshard_prefetch: bool = True
    """Whether to prefetch (pipeline) parameter unshard for upcoming modules."""

    # ------------------------------------------------------------------
    # Reduce-scatter (gradient sync) tracking
    # ------------------------------------------------------------------
    reduce_grad_buckets: Dict[int, List[Tuple[torch.cuda.Event, "ParameterGroup"]]] = field(
        default_factory=dict
    )
    """
    Maps module_id -> list of (event, parameter_group) tuples.

    Each entry corresponds to a module and contains a list of:
        (event, parameter_group)

    - event: signals gradient readiness
    - parameter_group: gradients to be reduced

    This structure enables ordered overlap of backward compute and
    gradient synchronization.
    """

    enable_async_reduce_grad: bool = True
    """Whether to overlap gradient reduction with backward computation."""
