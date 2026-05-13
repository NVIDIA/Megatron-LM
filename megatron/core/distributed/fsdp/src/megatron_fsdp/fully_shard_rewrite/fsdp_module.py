"""FSDPModule implementation for the Megatron-FSDP fully_shard rewrite path."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor

from .allocator import BucketAllocator, StorageFreeingBucketAllocator, TracePoolAllocator
from .mixed_precision import FullyShardMixedPrecisionPolicy
from .param_group import ParameterGroup
from .utils import ParamGroupIdx, _replace_module_parameter


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
    forward_order: List["FSDPModule"] = field(default_factory=list)
    """
    FSDP modules in actual forward execution order.

    This ordering is used to:
    - Schedule prefetching of parameters (unshard)
    - Ensure correct overlap between compute and communication
    """

    # ------------------------------------------------------------------
    # Unshard (all-gather) tracking
    # ------------------------------------------------------------------
    unshard_done_events: Dict[int, Optional[torch.cuda.Event]] = field(default_factory=dict)
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

    # ------------------------------------------------------------------
    # Activation recompute / gradient checkpointing support
    # ------------------------------------------------------------------
    backward_phase: bool = False
    """True from the root backward pre-hook until the final callback."""

    backward_module: Optional[int] = None
    """``id(module)`` of the FSDP module whose backward is pending next.
    Derived from ``_reversed_order`` and ``backward_done_modules`` — NOT
    set by any hook directly.  Updated by ``_advance_backward_module()``."""

    backward_done_modules: set = field(default_factory=set)
    """Set of ``id(module)`` for FSDP modules whose backward has completed.
    Populated in ``post_backward``, cleared in the root backward pre-hook."""

    _reversed_order: List["FSDPModule"] = field(default_factory=list)
    """``list(reversed(forward_order))`` — precomputed backward processing order."""

    # ------------------------------------------------------------------
    # Bucket allocators (weight and gradient buffers)
    # ------------------------------------------------------------------
    weight_bucket_allocator: Optional[BucketAllocator] = None
    """
    Bucket allocator for weight (parameter) buffers used during all-gather.

    When set, this allocator manages the lifecycle and reuse of flat
    contiguous weight buffers across FSDP modules, enabling memory-efficient
    double-buffering and custom allocation strategies for unsharded parameters.

    If ``None``, each module allocates its own weight buffer independently.
    """

    grad_bucket_allocator: Optional[BucketAllocator] = None
    """
    Bucket allocator for gradient buffers used during reduce-scatter.

    When set, this allocator manages the lifecycle and reuse of flat
    contiguous gradient accumulation buffers across FSDP modules, enabling
    memory-efficient pipelining of gradient reduction.

    If ``None``, each module allocates its own gradient buffer independently.
    """

    def _advance_backward_module(self) -> None:
        """Set ``backward_module`` to the first module in ``_reversed_order``
        that is NOT in ``backward_done_modules``."""
        for m in self._reversed_order:
            if id(m) not in self.backward_done_modules:
                self.backward_module = id(m)
                return
        self.backward_module = None

    def get_prefetch_next_modules(
        self, module: "FSDPModule", bwd_pass: bool = False
    ) -> List["FSDPModule"]:
        """Return the next FSDP module to prefetch in forward or backward order."""
        module_order = list(reversed(self.forward_order)) if bwd_pass else self.forward_order

        for module_index, candidate_module in enumerate(module_order):
            if candidate_module is module:
                if module_index + 1 >= len(module_order):
                    return []
                return [module_order[module_index + 1]]

        raise AssertionError("Current module not found in forward module order")


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
        mp_policy: FullyShardMixedPrecisionPolicy,
        gradient_scaling_factor: Optional[float] = None,
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
        self._materialize_meta_module(ignored_modules, mesh=mesh)

        # Create parameter groups
        fsdp_param_groups = _get_module_fsdp_param_groups(
            self,
            mp_policy=mp_policy,
            mesh=mesh,
            ignored_params=ignored_params,
            gradient_scaling_factor=gradient_scaling_factor,
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

    def _materialize_meta_module(self, ignored_modules: set, mesh: Optional[DeviceMesh] = None):
        """
        Materialize meta parameters to actual device and initialize.

        This is needed for large models that cannot fit in a single GPU.
        Meta parameters are moved to the current device and reset.
        After materialization, full parameters are broadcast from DP rank 0
        before DTensor wrapping so every rank shards the same initialized value.
        """
        materialization_device = torch.cuda.current_device()
        for name, m in self.named_modules():
            if m in ignored_modules:
                continue
            # Skip modules that don't have meta parameters
            if all(not p.is_meta for p in m.parameters(recurse=False)):
                continue

            m._apply(
                lambda t: torch.empty_like(t, device=materialization_device) if t.is_meta else t,
                recurse=False,
            )
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()
            elif hasattr(m, "_reset_parameters"):
                m._reset_parameters()
            else:
                raise ValueError(f"Module {name} contains meta parameters but cannot reset them")

        if mesh is not None and mesh.size() > 1:
            dp_group = mesh.get_group()
            src_rank = torch.distributed.get_global_rank(dp_group, 0)
            for param in self.parameters():
                if param.is_meta or isinstance(param, DTensor):
                    continue
                torch.distributed.broadcast(param.data, src=src_rank, group=dp_group)

    def _init_fsdp_state(
        self, enable_unshard_prefetch, enable_async_reduce_grad, enable_trace_pool=False
    ):
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
            _reversed_order=list(reversed(forward_order)),
            weight_bucket_allocator=(
                TracePoolAllocator() if enable_trace_pool else StorageFreeingBucketAllocator()
            ),
            grad_bucket_allocator=TracePoolAllocator() if enable_trace_pool else None,
        )
        setattr(self, "_fsdp_state", _FSDPState())
        setattr(self, "_fsdp_root_context", root_context)
        for child in self.modules():
            if child is not self and isinstance(child, FSDPModule):
                child._fsdp_state._is_root = False
                setattr(child, "_fsdp_root_context", root_context)

                # Reset the bucket allocator. Since this requires certain global information,
                # we need to update the bucket allocator for all child FSDP modules each time.
                for param_group in child._fsdp_param_groups:
                    if (
                        param_group.model_weight_buffer is not None
                        and root_context.weight_bucket_allocator is not None
                    ):
                        param_group.model_weight_buffer.allocator = (
                            root_context.weight_bucket_allocator
                        )
                    if (
                        param_group.main_grad_buffer is not None
                        and root_context.grad_bucket_allocator is not None
                    ):
                        param_group.main_grad_buffer.allocator = root_context.grad_bucket_allocator

    def unshard(self, async_op: bool = False, bwd_pass: bool = False):
        """
        Unshard parameters by all-gathering from the sharded buffer.

        This is called pre-forward to make parameters available for
        computation. After unsharding, each param.data points to
        the full (unsharded) tensor.
        """
        torch.cuda.nvtx.range_push("MFSDP unshard")
        ctx = self._fsdp_root_context
        stream = ctx.ag_stream if async_op else torch.cuda.current_stream()

        if async_op:
            # Synchronize ag_stream with current_stream to guarantee that main-stream
            # writes to parameter data are visible before the all-gather kernel reads them
            # on ag_stream. Without this barrier, stale or partially-written parameter
            # shards may be gathered, causing convergence divergence.
            stream.wait_stream(torch.cuda.current_stream())

        # Unshard this module and optionally prefetch next modules in the forward/backward pass
        if async_op:
            prefetch_modules = ctx.get_prefetch_next_modules(self, bwd_pass=bwd_pass)
        else:
            prefetch_modules = []
        for module in [self] + prefetch_modules:
            if ctx.unshard_done_events[id(module)] is not None:
                continue  # Skip if unshard already issued for this module
            if bwd_pass and id(module) in ctx.backward_done_modules:
                continue  # Skip prefetch for modules whose backward is already done

            # Unshard parameters for this module
            for param_names, param_group in module._named_param_groups:
                # Optional NaN checking for debugging
                if getattr(module, "_enable_nan_checks", False):
                    for name, dist_param in zip(param_names, param_group.dist_params):
                        assert not torch.isnan(
                            dist_param._local_tensor
                        ).any(), f"NaN detected in dist param for parameter {name}"

                with torch.cuda.stream(stream):
                    param_group.unshard()

            # Record event to track when unshard is done for this module
            if async_op:
                event = stream.record_event()
                ctx.unshard_done_events[id(module)] = event

        # Ensure unshard is complete before forward.
        # The event is NOT cleared here — it persists as a "currently unsharded"
        # flag and is only cleared by reshard().  This prevents redundant
        # all-gathers during activation recompute and prefetch re-entry.
        if ctx.unshard_done_events[id(self)] is not None:
            ctx.unshard_done_events[id(self)].wait()

        # Replace module parameters with unsharded versions
        for param_names, param_group in self._named_param_groups:
            for name, param in zip(param_names, param_group.params):
                _replace_module_parameter(self, name, param)

            # Optional NaN checking for debugging
            if getattr(self, "_enable_nan_checks", False):
                for name, param in zip(param_names, param_group.params):
                    assert not torch.isnan(param).any(), f"NaN detected in parameter {name}"

        torch.cuda.nvtx.range_pop()

    def reshard(self):
        """Reshard parameters by replacing with sharded DTensors."""
        torch.cuda.nvtx.range_push("MFSDP reshard")
        ctx = self._fsdp_root_context
        for param_names, param_group in self._named_param_groups:
            param_group.reshard()
            for name, dist_param in zip(param_names, param_group.dist_params):
                _replace_module_parameter(self, name, dist_param)
        ctx.unshard_done_events[id(self)] = None  # Clear unshard event for this module
        torch.cuda.nvtx.range_pop()

    def _wait_for_previous_async_reduce_grad(self):
        """Release older async reduce buffers in backward order."""
        ctx = self._fsdp_root_context
        if not ctx.enable_async_reduce_grad:
            return

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

    def reduce_grad(self, async_op: bool = False):
        """
        Reduce gradients across data-parallel ranks.

        This is called post-backward to:
        1. Copy gradients to main gradient buffer
        2. Perform all-reduce or reduce-scatter
        3. Install reduced gradients to distributed parameters
        """
        torch.cuda.nvtx.range_push("MFSDP reduce_grad")
        ctx = self._fsdp_root_context
        stream = ctx.rs_stream if async_op else torch.cuda.current_stream()

        # Handle pending reduce events before this module to release buffers promptly.
        self._wait_for_previous_async_reduce_grad()

        # Perform reduction for this module
        for param_names, param_group in self._named_param_groups:
            if not param_group.requires_grad:
                continue

            # NaN check before reduction
            if getattr(self, "_enable_nan_checks", False):
                for param in param_group.params:
                    if param.grad is not None:
                        assert not torch.isnan(param.grad).any(), "NaN in parameter grad"

            # Copy .grad -> main grad buffer on main stream (fast memcpy).
            # When gradient_accumulation_fusion is active for FSDP params, the backward
            # kernel writes directly into main_grad (weight.main_grad = get_main_grad() in
            # layers.py) and sets grad_added_to_main_grad=True. In that case we must NOT
            # zero or overwrite main_grad; discard the dummy .grad tensor if present.
            for name, param in zip(param_names, param_group.params):
                main_grad = param.get_main_grad()
                if getattr(param, "grad_added_to_main_grad", False):
                    if param.grad is not None:
                        del param.grad
                elif param.grad is None:
                    if hasattr(param, "main_grad") and param.main_grad is not None:
                        if param.main_grad.data_ptr() != main_grad.data_ptr():
                            main_grad.copy_(param.main_grad.detach())
                    else:
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

            if async_op:
                event = stream.record_event()
                ctx.reduce_grad_buckets[id(self)].append((event, param_group))

            # NaN check after reduction
            if getattr(self, "_enable_nan_checks", False):
                for name, dist_grad in zip(param_names, param_group.dist_grads):
                    if dist_grad is not None:
                        assert not torch.isnan(
                            dist_grad._local_tensor
                        ).any(), f"NaN in dist grad for parameter {name}"

        torch.cuda.nvtx.range_pop()

    @torch.no_grad()
    def _scale_gradients(self, scaling_factor: float):
        """Scale gradients by a factor (e.g., for loss scaling)."""
        ctx = self._fsdp_root_context
        torch.cuda.current_stream().wait_stream(ctx.rs_stream)
        for _, child in self.named_modules():
            if not isinstance(child, FSDPModule):
                continue
            for param_group in child._fsdp_param_groups:
                for dist_grad in param_group.dist_grads:
                    if dist_grad is None:
                        continue
                    dist_grad._local_tensor.mul_(scaling_factor)

    def _zero_grad_buffer(self):
        """Zero the gradient buffer for all parameter groups."""
        for _, child in self.named_modules():
            if not isinstance(child, FSDPModule):
                continue
            for param_group in child._fsdp_param_groups:
                if param_group.main_grad_buffer is not None:
                    param_group.main_grad_buffer.data.zero_()
                    param_group.release_grad_buffer()
                for dist_param in param_group.dist_params:
                    if dist_param.grad is not None:
                        del dist_param.grad

    def _copy_main_weights_to_model_weights(self):
        """Copy main weight buffer to model weight buffer."""
        for _, child in self.named_modules():
            if not isinstance(child, FSDPModule):
                continue
            for param_group in child._fsdp_param_groups:
                if param_group.main_weight_buffer is None:
                    continue
                param_group.model_weight_buffer.data.copy_(param_group.main_weight_buffer.data)

    def _compute_per_param_norms(self) -> Dict[str, Dict[str, float]]:
        """
        Compute per-parameter L2 norms for params and grads.

        Returns {param_name: {"param_norm": float, "grad_norm": float}}.
        Local squared norms are all-reduced across the DP group.
        """
        results = {}
        dp_group = None

        for module_name, child in self.named_modules():
            if not isinstance(child, FSDPModule):
                continue
            for param_names, param_group in child._named_param_groups:
                if dp_group is None and param_group.dp_group is not None:
                    dp_group = param_group.dp_group
                for param_name, dist_param, dist_grad in zip(
                    param_names, param_group.dist_params, param_group.dist_grads
                ):
                    full_name = f"{module_name}.{param_name}" if module_name else param_name
                    results[full_name] = {"param_norm": 0.0, "grad_norm": 0.0}
                    if dist_param._local_tensor.numel() > 0:
                        results[full_name]["param_norm"] = (
                            dist_param._local_tensor.float().norm(p=2).item() ** 2
                        )
                    if dist_grad is not None and dist_grad._local_tensor.numel() > 0:
                        results[full_name]["grad_norm"] = (
                            dist_grad._local_tensor.float().norm(p=2).item() ** 2
                        )

        if dp_group is not None:
            for param_name in results:
                for key in ("param_norm", "grad_norm"):
                    value = torch.tensor([results[param_name][key]], device="cuda")
                    torch.distributed.all_reduce(value, group=dp_group)
                    results[param_name][key] = value.sqrt().item()

        return results

    def _log_per_param_norms(self, iteration: int, prefix: str = ""):
        """Log per-parameter param and gradient L2 norms on rank 0."""
        norms = self._compute_per_param_norms()
        if torch.distributed.get_rank() != 0:
            return
        for param_name in sorted(norms.keys()):
            param_norm = norms[param_name]["param_norm"]
            grad_norm = norms[param_name]["grad_norm"]
            print(
                f"{prefix} iter={iteration} param={param_name} "
                f"param_norm={param_norm:.6f} grad_norm={grad_norm:.6f}"
            )

    def _log_parameter_groups(self):
        """Print a compact summary of rewrite-path FSDP parameter groups."""

        def _fmt_dtype(dtype: torch.dtype) -> str:
            short = {
                torch.float32: "fp32",
                torch.float16: "fp16",
                torch.bfloat16: "bf16",
                torch.int64: "i64",
                torch.int32: "i32",
                torch.uint8: "u8",
            }
            return short.get(dtype, str(dtype).removeprefix("torch."))

        def _elem_size(dtype: torch.dtype) -> int:
            return {
                torch.float32: 4,
                torch.float16: 2,
                torch.bfloat16: 2,
                torch.int64: 8,
                torch.int32: 4,
                torch.uint8: 1,
            }.get(dtype, 1)

        def _mb(num_bytes: int | float) -> str:
            return f"{num_bytes / 1_000_000:.2f} MB"

        rank = torch.distributed.get_rank()
        lines = [f"FSDP parameter groups (rank {rank})"]
        group_idx = 0
        total_model_elems = 0
        total_comm = 0
        total_pad = 0

        for module_name, child in self.named_modules():
            if not isinstance(child, FSDPModule):
                continue
            for param_names, param_group in child._named_param_groups:
                numel = sum(param.numel() for param in param_group.params)
                total_model_elems += numel
                dp_size = torch.distributed.get_world_size(param_group.dp_group)

                buffer_entries = []
                group_pad = 0
                group_comm = 0
                for buffer_label, buffer in (
                    ("W", param_group.model_weight_buffer),
                    ("MW", param_group.main_weight_buffer),
                    ("G", param_group.main_grad_buffer),
                ):
                    if buffer is None:
                        continue
                    global_size = buffer.buffer_index.bucket_meta.size
                    elem_size = _elem_size(buffer.dtype)
                    group_pad += max(0, global_size - numel) * elem_size
                    group_comm += global_size * elem_size
                    dist_flag = "D" if buffer.is_distributed else "R"
                    buffer_entries.append(
                        f"{buffer_label}[{_fmt_dtype(buffer.dtype)}:{buffer.data_size}:{dist_flag}]"
                    )
                total_pad += group_pad
                total_comm += group_comm

                lines.append(
                    f"- {module_name} #{group_idx} dp={dp_size} "
                    f"strategy={param_group.sharding_strategy} "
                    f"chunk_factor={param_group.chunk_size_factor}"
                )
                lines.append(
                    f"  {numel:,} elems x {_fmt_dtype(param_group.dtype)} "
                    f"comm={_mb(group_comm)} pad={_mb(group_pad)} "
                    f"{' '.join(buffer_entries)}"
                )
                for param_name, param in zip(param_names, param_group.params):
                    dist_idx = param_group.param_idx.get(param)
                    offset_info = ""
                    if param_group.model_weight_buffer is not None and dist_idx is not None:
                        item_index = (
                            param_group.model_weight_buffer.buffer_index.item_index_map.get(
                                dist_idx
                            )
                        )
                        if item_index is not None:
                            offset_info = f" @{item_index.global_data_index:,}+{item_index.size:,}"
                    lines.append(f"    {param_name:50s} {str(tuple(param.shape)):24s}{offset_info}")
                group_idx += 1

        lines.append(
            f"Summary: {group_idx} groups, {total_model_elems:,} model elems, "
            f"comm={_mb(total_comm)}, pad={_mb(total_pad)}"
        )
        print("\n".join(lines))

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


def _get_module_fsdp_param_groups(
    module: nn.Module,
    mp_policy: FullyShardMixedPrecisionPolicy,
    mesh: Optional[DeviceMesh] = None,
    ignored_params: Optional[set[nn.Parameter]] = None,
    gradient_scaling_factor: Optional[float] = None,
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
                mp_policy=mp_policy,
                gradient_scaling_factor=gradient_scaling_factor,
            )
        )

    return fsdp_param_groups
