# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""MoonEP runtime storage, communication buffers, and autograd integration."""

from __future__ import annotations

import inspect
import os
import weakref
from dataclasses import dataclass

import torch

try:
    from moonep import Buffer as MoonEPBuffer
    from moonep._C import nvl_dist_alloc as _moonep_nvl_dist_alloc
    from moonep._C import nvl_dist_map as _moonep_nvl_dist_map
    from moonep._C import nvl_release_mem_handle as _moonep_nvl_release_mem_handle
    from moonep.buffer import _exchange_ipc_fds as _moonep_exchange_ipc_fds
    from moonep.buffer import create_nvl_dist_tensor as _moonep_create_nvl_dist_tensor
    from moonep.buffer import get_vmm_granularity as _moonep_get_vmm_granularity
    from moonep.grad_reduce import launch_grad_reduce as _moonep_launch_grad_reduce
    from moonep.inter_rank_sync import launch_inter_rank_sync as _moonep_inter_rank_sync
    from moonep.prefetch import launch_prefetch as _moonep_launch_prefetch

    HAVE_MOONEP = True
    _MOONEP_IMPORT_ERROR = None
except ImportError as exc:
    MoonEPBuffer = None
    HAVE_MOONEP = False
    _MOONEP_IMPORT_ERROR = exc


_moonep_buffers = weakref.WeakSet()
_moonep_bridges = weakref.WeakSet()
_moonep_dispatch_buffer_pools = weakref.WeakSet()
_moonep_token_buffer_pools = {}
_moonep_buffer_registry = {}
_moonep_group_signatures = {}
_moonep_dispatch_buffer_pool_registry = {}
_moonep_shared_slot_pools = {}


def is_moonep_available() -> bool:
    """Return whether the optional MoonEP package was imported successfully."""
    return HAVE_MOONEP


def new_moonep_buffer(**kwargs):
    """Create and register a MoonEP buffer for explicit collective teardown."""
    if not HAVE_MOONEP:
        raise ImportError(
            "MoonEP is not installed. Install the optional 'moonep' package before using "
            "moe_flex_dispatcher_backend='moonep'."
        ) from _MOONEP_IMPORT_ERROR
    buffer = MoonEPBuffer(explicitly_destroy=True, **kwargs)
    _moonep_buffers.add(buffer)
    return buffer


def get_or_create_moonep_buffer(key, **kwargs):
    """Return the single MoonEP communication runtime for a process-group signature."""
    group_key = key[0]
    signature = key[1:]
    previous_signature = _moonep_group_signatures.get(group_key)
    if previous_signature is not None and previous_signature != signature:
        raise ValueError(
            "MoE layers sharing an EP process group must use the same MoonEP runtime signature: "
            f"existing={previous_signature}, requested={signature}."
        )
    buffer = _moonep_buffer_registry.get(key)
    if buffer is not None:
        return buffer
    buffer = new_moonep_buffer(**kwargs)
    _moonep_buffer_registry[key] = buffer
    _moonep_group_signatures[group_key] = signature
    return buffer


def moonep_supports_external_hidden_buffers() -> bool:
    """Return whether the installed MoonEP exposes Megatron's external-buffer extension."""
    if not HAVE_MOONEP:
        return False
    dispatch_params = inspect.signature(MoonEPBuffer.dispatch).parameters
    combine_params = inspect.signature(MoonEPBuffer.combine).parameters
    return "hidden_buffer" in dispatch_params and "hidden_buffer" in combine_params


def te_supports_external_grouped_linear_buffers() -> bool:
    """Return whether TE exposes the op-fuser buffer keys required for zero copy."""
    try:
        from transformer_engine.pytorch.ops.basic.grouped_linear import (
            GRAD_INPUT_BUFFER_KEY,
            OUTPUT_BUFFER_KEY,
        )
    except ImportError:
        return False
    return GRAD_INPUT_BUFFER_KEY is not None and OUTPUT_BUFFER_KEY is not None


def _allocate_moonep_token_buffer(ctx):
    """Collectively allocate one symmetric hidden-token buffer pair."""
    rank = int(ctx["rank"])
    world_size = int(ctx["R"])
    num_slots = int(ctx["NvS"])
    padded_slots = int(ctx["NvS_padded"])
    full = _moonep_create_nvl_dist_tensor(
        [padded_slots, int(ctx["H"])], torch.bfloat16, rank, world_size, group=ctx["group"]
    )
    local = full[rank * padded_slots : rank * padded_slots + num_slots]
    return full, local


class MoonEPDispatchBufferPool:
    """Pool per-forward symmetric dispatch outputs until their FC1 backward."""

    def __init__(self, buffer):
        self._ctx = buffer._require_ctx()
        self._free = [(self._ctx["hidden_buf"], self._ctx["hidden_buf_local"])]
        self._allocated = list(self._free)
        self._destroyed = False
        _moonep_dispatch_buffer_pools.add(self)

    def acquire(self):
        """Acquire a buffer, growing collectively to the maximum in-flight depth."""
        if self._destroyed:
            raise RuntimeError("MoonEP dispatch buffer pool has been destroyed.")
        if self._free:
            return self._free.pop()
        pair = _allocate_moonep_token_buffer(self._ctx)
        self._allocated.append(pair)
        return pair

    def release(self, pair) -> None:
        """Recycle a dispatch buffer after dispatch backward consumes FC1 dgrad."""
        if not self._destroyed:
            self._free.append(pair)

    def destroy(self) -> None:
        """Drop all VMM tensor references after MoonEP work has synchronized."""
        if self._destroyed:
            return
        self._free.clear()
        self._allocated.clear()
        self._ctx = None
        self._destroyed = True


def get_moonep_dispatch_buffer_pool(buffer):
    """Return the process-group-wide pool of in-flight dispatch output buffers."""
    key = id(buffer)
    pool = _moonep_dispatch_buffer_pool_registry.get(key)
    if pool is None:
        pool = MoonEPDispatchBufferPool(buffer)
        _moonep_dispatch_buffer_pool_registry[key] = pool
    return pool


class _MoonEPSharedTokenBufferPool:
    """Own the two process-group-wide transient expert boundary buffers."""

    def __init__(self, ctx):
        self._buffers = tuple(_allocate_moonep_token_buffer(ctx) for _ in range(2))

    @property
    def forward(self):
        """FC2-output / combine-backward buffer pair."""
        return self._buffers[0]

    @property
    def backward(self):
        """FC1-dgrad / dispatch-backward buffer pair."""
        return self._buffers[1]

    def destroy(self) -> None:
        """Drop VMM tensor references after MoonEP work has synchronized."""
        self._buffers = ()


def get_moonep_zero_copy_token_buffers(buffer):
    """Return shared symmetric buffers for MoonEP's two transient expert boundaries.

    Dispatch output uses a separate per-forward pool because FC1 autograd saves
    it. FC2 output/combine-backward and FC1 dgrad/dispatch-backward have
    non-overlapping lifetimes across layers, so two process-group-wide buffers
    are sufficient and avoid allocating those two boundaries per layer.
    """
    ctx = buffer._require_ctx()
    group = ctx["group"]
    key = (
        id(group),
        str(ctx["device"]),
        int(ctx["R"]),
        int(ctx["NvS"]),
        int(ctx["NvS_padded"]),
        int(ctx["H"]),
    )
    pool = _moonep_token_buffer_pools.get(key)
    if pool is not None:
        return pool

    pool = _MoonEPSharedTokenBufferPool(ctx)
    _moonep_token_buffer_pools[key] = pool
    return pool


def moonep_finalize() -> None:
    """Destroy all live MoonEP buffers and runtime VMM mappings."""
    for buffer in list(_moonep_buffers):
        buffer.destroy()
    _moonep_buffers.clear()
    for pool in list(_moonep_dispatch_buffer_pools):
        pool.destroy()
    _moonep_dispatch_buffer_pools.clear()
    _moonep_dispatch_buffer_pool_registry.clear()
    for bridge in list(_moonep_bridges):
        bridge.destroy()
    _moonep_bridges.clear()
    for pool in _moonep_token_buffer_pools.values():
        pool.destroy()
    _moonep_token_buffer_pools.clear()
    for pool in _moonep_shared_slot_pools.values():
        pool.destroy()
    _moonep_shared_slot_pools.clear()
    _moonep_buffer_registry.clear()
    _moonep_group_signatures.clear()


def _close_fds(fds) -> None:
    """Close a collection of POSIX file descriptors exactly once."""
    for fd in set(fds):
        os.close(fd)


class _MoonEPSharedSlotPool:
    """Own one process-group-wide physical prefetch or gradient slot chunk."""

    def __init__(self, *, chunk_shape, dtype, group) -> None:
        self.group = group
        self.rank = torch.distributed.get_rank(group=group)
        self.world_size = torch.distributed.get_world_size(group=group)
        self.keepalive, local_fd, handle = _moonep_nvl_dist_alloc(
            shape=list(chunk_shape), dtype=dtype
        )
        _moonep_nvl_release_mem_handle(handle)
        exchanged = _moonep_exchange_ipc_fds(
            local_fd, list(range(self.world_size)), self.rank, self.world_size, group
        )
        os.close(local_fd)
        self.fds = tuple(exchanged[idx] for idx in range(self.world_size))
        self._destroyed = False

    @property
    def local_fd(self):
        """Return this rank's retained slot file descriptor."""
        return self.fds[self.rank]

    def destroy(self) -> None:
        """Release retained file descriptors and the physical slot owner."""
        if self._destroyed:
            return
        _close_fds(self.fds)
        self.fds = ()
        self.keepalive = None
        self._destroyed = True


def _get_shared_slot_pool(*, name, chunk_shape, dtype, group):
    """Get a compatible shared slot allocation for all MoE layers in a process group."""
    key = (id(group), name, tuple(chunk_shape), dtype)
    pool = _moonep_shared_slot_pools.get(key)
    if pool is None:
        pool = _MoonEPSharedSlotPool(chunk_shape=chunk_shape, dtype=dtype, group=group)
        _moonep_shared_slot_pools[key] = pool
    return pool


def _allocate_moonep_mapping(*, name, chunk_shape, dtype, group):
    """Map per-layer E source chunks followed by a process-group shared B slot chunk."""
    rank = torch.distributed.get_rank(group=group)
    world_size = torch.distributed.get_world_size(group=group)
    chunk_bytes = int(torch.tensor([], dtype=dtype).element_size())
    for dim in chunk_shape:
        chunk_bytes *= int(dim)
    granularity = int(_moonep_get_vmm_granularity())
    if chunk_bytes % granularity != 0:
        raise ValueError(
            "MoonEP expert chunks must be VMM aligned: "
            f"shape={tuple(chunk_shape)}, dtype={dtype}, bytes={chunk_bytes}, "
            f"granularity={granularity}."
        )

    expert_keepalive, expert_fd, expert_handle = _moonep_nvl_dist_alloc(
        shape=list(chunk_shape), dtype=dtype
    )
    _moonep_nvl_release_mem_handle(expert_handle)

    expert_fds = _moonep_exchange_ipc_fds(
        expert_fd, list(range(world_size)), rank, world_size, group
    )
    os.close(expert_fd)

    slot_pool = _get_shared_slot_pool(
        name=f"{name}.weight", chunk_shape=chunk_shape, dtype=dtype, group=group
    )

    expert_fd_list = [expert_fds[idx] for idx in range(world_size)]
    full = _moonep_nvl_dist_map(
        chunk_shape=list(chunk_shape),
        dtype=dtype,
        fds=[*expert_fd_list, slot_pool.local_fd],
        local_rank=rank,
        world_size=world_size + 1,
    )

    _close_fds(expert_fd_list)

    # The mappings own their virtual addresses. Keep the local physical
    # allocations alive for exactly as long as the bridge owns the mappings.
    return full, (expert_keepalive, slot_pool)


def _allocate_moonep_grad_mapping(*, name, chunk_shape, group):
    """Allocate a rank-private ``[E+B]`` wgrad view and shared slot view.

    The local owner chunk occupies this rank's global expert range. All
    nonlocal expert ranges alias a private disposable sink chunk, so TE can
    write zero-token group outputs without touching peer-owned gradients. The
    final chunk is the local redundant-slot storage and is also mapped from
    every rank as ``[R, B, ...]`` for MoonEP's owner-side reducer.
    """
    rank = torch.distributed.get_rank(group=group)
    world_size = torch.distributed.get_world_size(group=group)
    dtype = torch.float32

    keepalives = []
    local_fds = []
    for _ in range(2):
        keepalive, fd, handle = _moonep_nvl_dist_alloc(shape=list(chunk_shape), dtype=dtype)
        _moonep_nvl_release_mem_handle(handle)
        keepalives.append(keepalive)
        local_fds.append(fd)
    owner_fd, sink_fd = local_fds
    slot_pool = _get_shared_slot_pool(
        name=f"{name}.grad", chunk_shape=chunk_shape, dtype=dtype, group=group
    )

    grad_fds = [sink_fd] * world_size
    grad_fds[rank] = owner_fd
    grad_fds.append(slot_pool.local_fd)
    full_grad = _moonep_nvl_dist_map(
        chunk_shape=list(chunk_shape),
        dtype=dtype,
        fds=grad_fds,
        local_rank=rank,
        world_size=world_size + 1,
    )

    reduce_view = _moonep_nvl_dist_map(
        chunk_shape=list(chunk_shape),
        dtype=dtype,
        fds=list(slot_pool.fds),
        local_rank=rank,
        world_size=world_size,
    )
    _close_fds([owner_fd, sink_fd])
    return full_grad, reduce_view, (*keepalives, slot_pool)


@dataclass
class _MoonEPProjection:
    """MoonEP runtime storage for one grouped expert projection."""

    linear: torch.nn.Module
    parameter: torch.nn.Parameter
    full_weight: torch.Tensor
    full_grad: torch.Tensor
    reduce_buffers: torch.Tensor
    runtime_parameter: torch.nn.Parameter
    dummy_grad: torch.Tensor
    keepalives: tuple


class MoonEPWeightBridge:
    """Connect Megatron grouped expert parameters to MoonEP VMM runtime weights.

    Registered Megatron parameters remain the optimizer/checkpoint source of
    truth. Their contiguous grouped storage is copied into rank-owned MoonEP
    source chunks before each dispatch. Transformer Engine executes against an
    unregistered ``[E+B]`` GroupedTensor whose FP32 ``main_grad`` points at the
    corresponding MoonEP gradient mapping.
    """

    def __init__(
        self,
        *,
        experts,
        group: torch.distributed.ProcessGroup,
        num_experts: int,
        num_local_experts: int,
        num_sms: int | None,
    ) -> None:
        if not HAVE_MOONEP:
            raise ImportError(
                "MoonEP is not installed. Install the optional 'moonep' package before using "
                "moe_flex_dispatcher_backend='moonep'."
            ) from _MOONEP_IMPORT_ERROR

        from transformer_engine.pytorch.tensor.grouped_tensor import GroupedTensor

        self.group = group
        self.rank = torch.distributed.get_rank(group=group)
        self.world_size = torch.distributed.get_world_size(group=group)
        self.num_experts = int(num_experts)
        self.num_local_experts = int(num_local_experts)
        self.num_slots = self.num_local_experts
        self.num_runtime_experts = self.num_experts + self.num_slots
        self.num_sms = 32 if num_sms is None else int(num_sms)
        self.buffer = None
        self.last_plan = None
        self._experts_ref = weakref.ref(experts)
        self._destroyed = False

        if self.num_experts != self.world_size * self.num_local_experts:
            raise ValueError(
                "MoonEP requires an even expert distribution: "
                f"num_experts={self.num_experts}, world_size={self.world_size}, "
                f"num_local_experts={self.num_local_experts}."
            )

        self.projections = []
        for projection_name, linear in zip(
            ("fc1", "fc2"), (experts.linear_fc1, experts.linear_fc2)
        ):
            parameter = dict(linear.named_parameters(recurse=False)).get("weight")
            if parameter is None:
                raise ValueError(
                    "MoonEP requires Transformer Engine to create one contiguous grouped "
                    "weight parameter. Ensure moe_single_grouped_weight=True and "
                    "NVTE_GROUPED_LINEAR_SINGLE_PARAM is not explicitly disabled."
                )
            rowwise_data = getattr(parameter, "rowwise_data", None)
            if rowwise_data is None or rowwise_data.dtype != torch.bfloat16:
                raise ValueError(
                    "MoonEP requires BF16 moe_single_grouped_weight parameters with contiguous "
                    "rowwise_data."
                )
            member_shape = (int(linear.out_features), int(linear.in_features))
            expected_numel = self.num_local_experts * member_shape[0] * member_shape[1]
            if rowwise_data.numel() != expected_numel or not rowwise_data.is_contiguous():
                raise ValueError(
                    "MoonEP grouped parameter storage has an unexpected layout: "
                    f"expected {self.num_local_experts}x{member_shape}, "
                    f"got numel={rowwise_data.numel()}, contiguous={rowwise_data.is_contiguous()}."
                )
            if member_shape[0] % 128 != 0 or member_shape[1] % 128 != 0:
                raise ValueError(
                    "MoonEP weight prefetch requires both projection dimensions to be multiples "
                    f"of 128, got {member_shape}."
                )

            chunk_shape = (self.num_local_experts, *member_shape)
            full_weight, weight_keepalives = _allocate_moonep_mapping(
                name=projection_name,
                chunk_shape=chunk_shape,
                dtype=torch.bfloat16,
                group=self.group,
            )
            full_grad, reduce_full, grad_keepalives = _allocate_moonep_grad_mapping(
                name=projection_name, chunk_shape=chunk_shape, group=self.group
            )
            full_weight = full_weight.view(self.num_runtime_experts, *member_shape)
            full_grad = full_grad.view(self.num_runtime_experts, *member_shape)
            reduce_buffers = reduce_full.view(self.world_size, self.num_slots, *member_shape)
            full_grad[
                self.rank * self.num_local_experts : (self.rank + 1) * self.num_local_experts
            ].zero_()
            full_grad[self.num_experts :].zero_()

            grouped_weight = GroupedTensor.make_grouped_tensor_from_rowwise_data(
                num_tensors=self.num_runtime_experts,
                tensor_shape=member_shape,
                rowwise_data=full_weight,
                dtype=torch.bfloat16,
            )
            grouped_weight.requires_grad_(True)
            runtime_parameter = torch.nn.Parameter(grouped_weight)
            runtime_parameter.main_grad = full_grad
            runtime_parameter.grad_added_to_main_grad = True
            # Nonlocal rows alias a rank-private sink, so overwrite mode
            # cannot corrupt another rank's owner gradients.
            runtime_parameter.overwrite_main_grad = True

            # This cached zero tensor exists only to run the registered
            # parameter's AccumulateGrad/DDP hook. The real gradient has
            # already been accumulated into parameter.main_grad.
            dummy_grad = torch.zeros_like(rowwise_data).view(parameter.shape)
            self.projections.append(
                _MoonEPProjection(
                    linear=linear,
                    parameter=parameter,
                    full_weight=full_weight,
                    full_grad=full_grad,
                    reduce_buffers=reduce_buffers,
                    runtime_parameter=runtime_parameter,
                    dummy_grad=dummy_grad,
                    keepalives=(*weight_keepalives, *grad_keepalives),
                )
            )
        _moonep_bridges.add(self)

    @property
    def runtime_fc1_weight(self) -> torch.nn.Parameter:
        """Return the ``[E+B]`` FC1 runtime grouped parameter."""
        return self.projections[0].runtime_parameter

    @property
    def runtime_fc2_weight(self) -> torch.nn.Parameter:
        """Return the ``[E+B]`` FC2 runtime grouped parameter."""
        return self.projections[1].runtime_parameter

    @property
    def source_parameters(self):
        """Return Megatron's registered FC1/FC2 grouped parameters."""
        return tuple(projection.parameter for projection in self.projections)

    @property
    def dummy_grads(self):
        """Return cached dummy grads used to trigger registered-parameter hooks."""
        return tuple(projection.dummy_grad for projection in self.projections)

    def attach_buffer(self, buffer) -> None:
        """Attach the layer's MoonEP communication buffer."""
        self.buffer = buffer

    def destroy(self) -> None:
        """Release runtime grouped weights and their VMM mappings."""
        if self._destroyed:
            return
        experts = self._experts_ref()
        if experts is not None:
            experts._fused_ops = None
            experts._moonep_weight_bridge = None
        for projection in self.projections:
            projection.runtime_parameter.main_grad = None
        self.projections.clear()
        self.buffer = None
        self.last_plan = None
        self._destroyed = True

    def prepare_forward(self) -> None:
        """Refresh local source weights and clear this rank's gradient scratch."""
        local_start = self.rank * self.num_local_experts
        local_end = local_start + self.num_local_experts
        for projection in self.projections:
            source = projection.parameter.rowwise_data.view_as(
                projection.full_weight[local_start:local_end]
            )
            projection.full_weight[local_start:local_end].copy_(source)
            projection.full_grad[local_start:local_end].zero_()
            projection.full_grad[self.num_experts :].zero_()
        # Distributed-optimizer parameter all-gathers are run by the original
        # linear pre-forward hooks immediately before this method. Publish every
        # rank's local mirror refresh before any peer starts remote prefetch,
        # entirely on the current CUDA stream.
        _moonep_inter_rank_sync(self.buffer._require_ctx())

    def prefetch(self, plan) -> None:
        """Prefetch the plan's redundant FC1/FC2 experts into local slots."""
        experts_to_copy = plan.experts_to_copy[self.rank].contiguous()
        for projection in self.projections:
            _moonep_launch_prefetch(
                projection.full_weight[: self.num_experts],
                projection.full_weight[self.num_experts :],
                experts_to_copy,
                num_sms=self.num_sms,
            )

    def reduce_grads(self, plan) -> None:
        """Reduce redundant wgrads and hand local results to Megatron DDP."""
        if self.buffer is None:
            raise RuntimeError("MoonEPWeightBridge has no attached communication buffer.")
        ctx = self.buffer._require_ctx()
        local_start = self.rank * self.num_local_experts
        local_end = local_start + self.num_local_experts

        # The TE op obeys PyTorch stream semantics: when its backward returns,
        # its wgrad writes are ordered before subsequent work on the current
        # stream. Align the EP ranks on-device before any reducer remote-reads
        # peer slots. No host wait or device-wide fence is needed.
        _moonep_inter_rank_sync(ctx)

        for projection in self.projections:
            main_grad = getattr(projection.parameter, "main_grad", None)
            if main_grad is None:
                raise RuntimeError(
                    "MoonEP requires gradient-accumulation fusion and an initialized "
                    "parameter.main_grad buffer."
                )
            _moonep_launch_grad_reduce(
                projection.full_grad[: self.num_experts],
                projection.reduce_buffers,
                plan.experts_to_copy,
                rank=self.rank,
                num_sms=self.num_sms,
                meta_buf=ctx["meta_buf"],
                meta_stride=int(ctx["meta_chunk_padded"]),
                barrier_off=int(ctx["BARRIER_OFF"]),
                grid_sync_bar=ctx["grid_sync_bar"],
            )

            # FC1 and FC2 reducers are launched consecutively on one stream,
            # matching MoonEP Buffer.reduce_grad(). Each reducer contains its
            # own GPU-side cross-rank barrier and resets the shared barrier state.
            main_grad.add_(projection.full_grad[local_start:local_end].view_as(main_grad))
            projection.full_grad[local_start:local_end].zero_()
            projection.full_grad[self.num_experts :].zero_()
            projection.parameter.grad_added_to_main_grad = True


class MoonEPDispatch(torch.autograd.Function):
    """Autograd-aware MoonEP dispatch, probability gather, and wgrad reduction."""

    @staticmethod
    def forward(
        ctx,
        hidden_states,
        topk_probs,
        topk_indices,
        tokens_per_expert,
        fc1_parameter,
        fc2_parameter,
        buffer,
        bridge,
        dispatch_buffer_pool,
        dgrad_hidden_buffer,
    ):
        """Dispatch activations and route weights while saving the MoonEP plan."""
        dispatch_hidden_buffer = (
            dispatch_buffer_pool.acquire() if dispatch_buffer_pool is not None else None
        )
        try:
            dispatch_args = (
                hidden_states.contiguous(),
                topk_probs.float().contiguous(),
                topk_indices.to(dtype=torch.int32).contiguous(),
                tokens_per_expert.to(dtype=torch.int32).contiguous(),
            )
            if dispatch_hidden_buffer is None:
                dispatched, dispatched_probs, cu_seqlens, plan = buffer.dispatch(*dispatch_args)
            else:
                dispatched, dispatched_probs, cu_seqlens, plan = buffer.dispatch(
                    *dispatch_args,
                    zero_copy=True,
                    zero_copy_weights=False,
                    hidden_buffer=dispatch_hidden_buffer,
                )
        except Exception:
            if dispatch_buffer_pool is not None:
                dispatch_buffer_pool.release(dispatch_hidden_buffer)
            raise
        bridge.last_plan = plan

        starts = torch.cat(
            [torch.zeros(1, dtype=cu_seqlens.dtype, device=cu_seqlens.device), cu_seqlens[:-1]]
        )
        runtime_tokens_per_expert = (cu_seqlens - starts).to(torch.int64)
        ctx.buffer = buffer
        ctx.bridge = bridge
        ctx.plan = plan
        ctx.dispatch_buffer_pool = dispatch_buffer_pool
        ctx.dispatch_hidden_buffer = dispatch_hidden_buffer
        ctx.dgrad_hidden_buffer = dgrad_hidden_buffer
        ctx.mark_non_differentiable(runtime_tokens_per_expert)
        return dispatched, dispatched_probs, runtime_tokens_per_expert

    @staticmethod
    def backward(ctx, grad_hidden, grad_probs, _grad_tokens_per_expert):
        """Combine activation/probability gradients and reduce duplicated wgrads."""
        try:
            ctx.bridge.reduce_grads(ctx.plan)
            grad_hidden = grad_hidden.contiguous()
            if grad_probs is None:
                grad_probs = torch.zeros_like(grad_hidden[:, 0], dtype=torch.float32)
            use_zero_copy = (
                ctx.dgrad_hidden_buffer is not None
                and grad_hidden.data_ptr() == ctx.dgrad_hidden_buffer[1].data_ptr()
            )
            if ctx.dgrad_hidden_buffer is not None and not use_zero_copy:
                raise RuntimeError(
                    "Transformer Engine did not write FC1 dgrad into MoonEP's caller-provided "
                    "buffer while moe_moonep_zero_copy=True."
                )
            combine_kwargs = {
                "plan": ctx.plan,
                "hidden_nvsh": grad_hidden,
                "route_weights_nvs": grad_probs.float().contiguous(),
            }
            if use_zero_copy:
                combine_kwargs.update(
                    zero_copy=True,
                    zero_copy_weights=False,
                    hidden_buffer=ctx.dgrad_hidden_buffer,
                )
            grad_hidden_states, grad_topk_probs, _ = ctx.buffer.combine(**combine_kwargs)
        finally:
            if ctx.dispatch_buffer_pool is not None:
                ctx.dispatch_buffer_pool.release(ctx.dispatch_hidden_buffer)
        dummy_fc1_grad, dummy_fc2_grad = ctx.bridge.dummy_grads
        return (
            grad_hidden_states,
            grad_topk_probs,
            None,
            None,
            dummy_fc1_grad,
            dummy_fc2_grad,
            None,
            None,
            None,
            None,
        )


class MoonEPCombine(torch.autograd.Function):
    """Autograd-aware MoonEP combine and saved-plan backward redispatch."""

    @staticmethod
    def forward(ctx, expert_output, buffer, plan, bridge, fwd_hidden_buffer):
        """Combine expert outputs using the matching forward dispatch plan."""
        expert_output = expert_output.contiguous()
        use_zero_copy = (
            fwd_hidden_buffer is not None
            and expert_output.data_ptr() == fwd_hidden_buffer[1].data_ptr()
        )
        if fwd_hidden_buffer is not None and not use_zero_copy:
            raise RuntimeError(
                "Transformer Engine did not write FC2 output into MoonEP's caller-provided "
                "buffer while moe_moonep_zero_copy=True."
            )
        combine_kwargs = {"plan": plan, "hidden_nvsh": expert_output}
        if use_zero_copy:
            combine_kwargs.update(zero_copy=True, hidden_buffer=fwd_hidden_buffer)
        combined, _, _ = buffer.combine(**combine_kwargs)
        ctx.buffer = buffer
        ctx.plan = plan
        ctx.bridge = bridge
        ctx.fwd_hidden_buffer = fwd_hidden_buffer
        return combined

    @staticmethod
    def backward(ctx, grad_output):
        """Restore plan weights and redispatch the combined output gradient."""
        # Prefetch slots are shared and may have been overwritten by a later
        # layer/microbatch. Restore this plan before the expert dgrad runs.
        ctx.bridge.prefetch(ctx.plan)
        dispatch_kwargs = {"plan": ctx.plan}
        if ctx.fwd_hidden_buffer is not None:
            dispatch_kwargs.update(zero_copy=True, hidden_buffer=ctx.fwd_hidden_buffer)
        grad_expert_output, _, _, _ = ctx.buffer.dispatch(
            grad_output.contiguous(), **dispatch_kwargs
        )
        if (
            ctx.fwd_hidden_buffer is not None
            and grad_expert_output.data_ptr() != ctx.fwd_hidden_buffer[1].data_ptr()
        ):
            raise RuntimeError("MoonEP combine backward did not preserve the external buffer alias.")
        return grad_expert_output, None, None, None, None


def moonep_dispatch(
    hidden_states,
    topk_probs,
    topk_indices,
    tokens_per_expert,
    buffer,
    bridge,
    dispatch_buffer_pool=None,
    dgrad_hidden_buffer=None,
):
    """Dispatch tokens with MoonEP while preserving activation and router gradients."""
    return MoonEPDispatch.apply(
        hidden_states,
        topk_probs,
        topk_indices,
        tokens_per_expert,
        *bridge.source_parameters,
        buffer,
        bridge,
        dispatch_buffer_pool,
        dgrad_hidden_buffer,
    )


def moonep_combine(expert_output, buffer, plan, bridge, fwd_hidden_buffer=None):
    """Combine MoonEP expert output and install its saved-plan backward."""
    return MoonEPCombine.apply(expert_output, buffer, plan, bridge, fwd_hidden_buffer)
