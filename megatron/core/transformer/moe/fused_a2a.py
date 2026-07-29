# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Portions of this code are from DeepSeek DeepEP project
# Copyright (c) 2025 DeepSeek
# Licensed under the MIT License - https://github.com/deepseek-ai/DeepEP/blob/main/LICENSE

import os
import weakref
from dataclasses import dataclass
from typing import Optional

from megatron.core.utils import internal_api

try:
    from deep_ep import Buffer
    from deep_ep.utils import EventHandle, EventOverlap

    HAVE_DEEP_EP = True
except ImportError:
    HAVE_DEEP_EP = False

import torch

_buffer = None

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
    for bridge in list(_moonep_bridges):
        bridge.destroy()
    _moonep_bridges.clear()
    for pool in _moonep_token_buffer_pools.values():
        pool.destroy()
    _moonep_token_buffer_pools.clear()


def _close_fds(fds) -> None:
    """Close a collection of POSIX file descriptors exactly once."""
    for fd in set(fds):
        os.close(fd)


def _allocate_moonep_mapping(*, chunk_shape, dtype, group, with_reduce_view: bool):
    """Allocate an ``[E+B]`` VMM mapping and an optional all-rank slot view.

    Each rank owns one expert chunk and one equally-sized prefetch/gradient-slot
    chunk. The returned composite maps all expert chunks followed by this
    rank's slot chunk. For gradients, the second returned tensor maps every
    rank's slot chunk as ``[R, B, ...]`` for MoonEP's owner-side reduction.
    """
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
    slot_keepalive, slot_fd, slot_handle = _moonep_nvl_dist_alloc(
        shape=list(chunk_shape), dtype=dtype
    )
    _moonep_nvl_release_mem_handle(expert_handle)
    _moonep_nvl_release_mem_handle(slot_handle)

    expert_fds = _moonep_exchange_ipc_fds(
        expert_fd, list(range(world_size)), rank, world_size, group
    )
    os.close(expert_fd)

    slot_fds = None
    if with_reduce_view:
        slot_fds = _moonep_exchange_ipc_fds(
            slot_fd, list(range(world_size)), rank, world_size, group
        )
        os.close(slot_fd)
        local_slot_fd = slot_fds[rank]
    else:
        local_slot_fd = slot_fd

    expert_fd_list = [expert_fds[idx] for idx in range(world_size)]
    full = _moonep_nvl_dist_map(
        chunk_shape=list(chunk_shape),
        dtype=dtype,
        fds=[*expert_fd_list, local_slot_fd],
        local_rank=rank,
        world_size=world_size + 1,
    )

    reduce_view = None
    fds_to_close = [*expert_fd_list, local_slot_fd]
    if slot_fds is not None:
        slot_fd_list = [slot_fds[idx] for idx in range(world_size)]
        reduce_view = _moonep_nvl_dist_map(
            chunk_shape=list(chunk_shape),
            dtype=dtype,
            fds=slot_fd_list,
            local_rank=rank,
            world_size=world_size,
        )
        fds_to_close.extend(slot_fd_list)
    _close_fds(fds_to_close)

    # The mappings own their virtual addresses. Keep the local physical
    # allocations alive for exactly as long as the bridge owns the mappings.
    return full, reduce_view, (expert_keepalive, slot_keepalive)


def _allocate_moonep_grad_mapping(*, chunk_shape, group):
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
    for _ in range(3):
        keepalive, fd, handle = _moonep_nvl_dist_alloc(shape=list(chunk_shape), dtype=dtype)
        _moonep_nvl_release_mem_handle(handle)
        keepalives.append(keepalive)
        local_fds.append(fd)
    owner_fd, sink_fd, slot_fd = local_fds

    slot_fds = _moonep_exchange_ipc_fds(slot_fd, list(range(world_size)), rank, world_size, group)
    os.close(slot_fd)
    local_slot_fd = slot_fds[rank]

    grad_fds = [sink_fd] * world_size
    grad_fds[rank] = owner_fd
    grad_fds.append(local_slot_fd)
    full_grad = _moonep_nvl_dist_map(
        chunk_shape=list(chunk_shape),
        dtype=dtype,
        fds=grad_fds,
        local_rank=rank,
        world_size=world_size + 1,
    )

    slot_fd_list = [slot_fds[idx] for idx in range(world_size)]
    reduce_view = _moonep_nvl_dist_map(
        chunk_shape=list(chunk_shape),
        dtype=dtype,
        fds=slot_fd_list,
        local_rank=rank,
        world_size=world_size,
    )
    _close_fds([owner_fd, sink_fd, *slot_fd_list])
    return full_grad, reduce_view, tuple(keepalives)


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
        num_sms: Optional[int],
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
        for linear in (experts.linear_fc1, experts.linear_fc2):
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
            full_weight, _, weight_keepalives = _allocate_moonep_mapping(
                chunk_shape=chunk_shape,
                dtype=torch.bfloat16,
                group=self.group,
                with_reduce_view=False,
            )
            full_grad, reduce_full, grad_keepalives = _allocate_moonep_grad_mapping(
                chunk_shape=chunk_shape, group=self.group
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
            dispatched, dispatched_probs, cu_seqlens, plan = buffer.dispatch(
                hidden_states.contiguous(),
                topk_probs.float().contiguous(),
                topk_indices.to(dtype=torch.int32).contiguous(),
                tokens_per_expert.to(dtype=torch.int32).contiguous(),
                zero_copy=dispatch_hidden_buffer is not None,
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
            use_zero_copy = (
                ctx.dgrad_hidden_buffer is not None
                and grad_hidden.data_ptr() == ctx.dgrad_hidden_buffer[1].data_ptr()
            )
            grad_hidden_states, grad_topk_probs, _ = ctx.buffer.combine(
                plan=ctx.plan,
                hidden_nvsh=grad_hidden,
                route_weights_nvs=grad_probs.float().contiguous(),
                zero_copy=use_zero_copy,
                hidden_buffer=ctx.dgrad_hidden_buffer,
            )
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
        combined, _, _ = buffer.combine(
            plan=plan,
            hidden_nvsh=expert_output,
            zero_copy=use_zero_copy,
            hidden_buffer=fwd_hidden_buffer,
        )
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
        grad_expert_output, _, _, _ = ctx.buffer.dispatch(
            grad_output.contiguous(),
            plan=ctx.plan,
            zero_copy=ctx.fwd_hidden_buffer is not None,
            hidden_buffer=ctx.fwd_hidden_buffer,
        )
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


def get_hidden_bytes(x: torch.Tensor) -> int:
    """Calculate the number of hidden bytes for a tensor.

    Args:
        x (torch.Tensor): Input tensor

    Returns:
        int: Number of hidden bytes
    """
    return x.size(1) * max(x.element_size(), 2)


def get_buffer(group: torch.distributed.ProcessGroup, hidden_bytes: int):
    """Get or create a buffer for all-to-all communication.

    Args:
        group (torch.distributed.ProcessGroup): Process group for communication
        hidden_bytes (int): Number of hidden bytes needed

    Returns:
        Buffer: Communication buffer
    """
    global _buffer
    num_nvl_bytes, num_rdma_bytes = 0, 0
    for config in (
        Buffer.get_dispatch_config(group.size()),
        Buffer.get_combine_config(group.size()),
    ):
        # Split long line for PEP8 compliance
        num_nvl_bytes = max(
            config.get_nvl_buffer_size_hint(hidden_bytes, group.size()), num_nvl_bytes
        )
        num_rdma_bytes = max(
            config.get_rdma_buffer_size_hint(hidden_bytes, group.size()), num_rdma_bytes
        )

    # Allocate buffer if not existed or not enough buffer
    # NOTES: the adaptive routing configuration of the network **must be off**
    if (
        _buffer is None
        or _buffer.group != group
        or _buffer.num_nvl_bytes < num_nvl_bytes
        or _buffer.num_rdma_bytes < num_rdma_bytes
    ):
        _buffer = Buffer(group, num_nvl_bytes, num_rdma_bytes)
    return _buffer


class FusedDispatch(torch.autograd.Function):
    """Fused dispatch operation for MoE routing combining computation and communication."""

    @staticmethod
    def forward(
        ctx,
        x,
        token_indices,
        token_probs,
        num_experts,
        group,
        async_finish=False,
        allocate_on_comm_stream=False,
    ):
        """Forward pass of fused dispatch."""
        previous_event = None
        if async_finish:
            previous_event = EventOverlap(EventHandle())
        # Calculate layout before actual dispatch
        buffer = get_buffer(group, get_hidden_bytes(x))
        (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            event,
        ) = buffer.get_dispatch_layout(
            token_indices,
            num_experts,
            previous_event=previous_event,
            async_finish=async_finish,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )

        # Do MoE dispatch
        # NOTES: the CPU will wait for GPU's signal to arrive,
        # so this is not compatible with CUDA graph
        (
            recv_x,
            recv_token_indices,
            recv_token_probs,
            num_recv_tokens_per_expert_list,
            handle,
            after_event_overlap,
        ) = buffer.dispatch(
            x,
            topk_idx=token_indices,
            topk_weights=token_probs,  # DeepEP only supports float32 probs
            num_tokens_per_rank=num_tokens_per_rank,
            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert,
            previous_event=event,  # wait in deepep::intra/inter_dispatch
            async_finish=async_finish,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )

        # Make sure current stream is synchronized
        if async_finish:
            after_event_overlap.current_stream_wait()

        # Save for backward
        ctx.group = group
        ctx.handle = handle
        ctx.async_finish = async_finish
        ctx.allocate_on_comm_stream = allocate_on_comm_stream
        tokens_per_expert = torch.tensor(num_recv_tokens_per_expert_list)

        return (recv_x, recv_token_indices, recv_token_probs, tokens_per_expert, handle)

    @staticmethod
    def backward(
        ctx, grad_output, grad_token_indices, grad_token_probs, grad_tokens_per_expert, grad_handle
    ):
        """Backward pass of fused dispatch."""
        buffer = get_buffer(ctx.group, get_hidden_bytes(grad_output))
        handle = ctx.handle
        previous_event = None
        if ctx.async_finish:
            previous_event = EventOverlap(EventHandle())
        grad_x, grad_token_probs, after_event = buffer.combine(
            grad_output.contiguous(),
            handle,
            topk_weights=grad_token_probs.float(),
            previous_event=previous_event,
            async_finish=ctx.async_finish,
            allocate_on_comm_stream=ctx.allocate_on_comm_stream,
        )
        # Make sure current stream is synchronized
        if ctx.async_finish:
            after_event.current_stream_wait()
        return grad_x, None, grad_token_probs, None, None, None, None


class FusedCombine(torch.autograd.Function):
    """Fused combine operation for MoE output combining computation and communication."""

    @staticmethod
    def forward(ctx, x, group, handle, async_finish=False, allocate_on_comm_stream=False):
        """Forward pass of fused combine."""
        previous_event = None
        if async_finish:
            previous_event = EventOverlap(EventHandle())
        buffer = get_buffer(group, get_hidden_bytes(x))
        combined_x, _, after_event = buffer.combine(
            x,
            handle=handle,
            async_finish=async_finish,
            previous_event=previous_event,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )
        # Make sure current stream is synchronized
        if async_finish:
            after_event.current_stream_wait()

        ctx.handle = handle
        ctx.group = group
        ctx.async_finish = async_finish
        ctx.allocate_on_comm_stream = allocate_on_comm_stream
        return combined_x, None

    @staticmethod
    def backward(ctx, grad_output, previous_event=None):
        """Backward pass of fused combine."""
        previous_event = None
        if ctx.async_finish:
            previous_event = EventOverlap(EventHandle())
        buffer = get_buffer(ctx.group, get_hidden_bytes(grad_output))
        grad_x, _, _, _, _, after_event = buffer.dispatch(
            grad_output.contiguous(),
            handle=ctx.handle,
            previous_event=previous_event,
            async_finish=ctx.async_finish,
            allocate_on_comm_stream=ctx.allocate_on_comm_stream,
        )
        # Make sure current stream is synchronized
        if ctx.async_finish:
            after_event.current_stream_wait()
        return grad_x, None, None, None, None


if HAVE_DEEP_EP:

    def fused_dispatch(
        x,
        token_indices,
        token_probs,
        num_experts,
        group,
        async_finish=False,
        allocate_on_comm_stream=False,
    ):
        """Perform fused dispatch operation if deep_ep is available.

        Args:
            x: Input tensor [num_tokens, hidden_size]
            token_indices: Token routing indices [num_tokens, topk]
            token_probs: Token routing probabilities [num_tokens, topk]
            num_experts: Number of experts
            group: Process group
            previous_event: Previous CUDA event

        Returns:
            Result of FusedDispatch
        """
        return FusedDispatch.apply(
            x.contiguous(),
            token_indices,
            token_probs,
            num_experts,
            group,
            async_finish,
            allocate_on_comm_stream,
        )

    def fused_combine(x, group, handle, async_finish=False, allocate_on_comm_stream=False):
        """Perform fused combine operation if deep_ep is available.

        Args:
            x: Input tensor
            group: Process group
            handle: Communication handle
            previous_event: Previous CUDA event

        Returns:
            Result of FusedCombine
        """
        return FusedCombine.apply(x, group, handle, async_finish, allocate_on_comm_stream)

    def set_deepep_num_sms(num_sms):
        """Sets the number of SMs to use for DeepEP"""
        Buffer.set_num_sms(num_sms)

else:
    fused_dispatch = None
    fused_combine = None
    set_deepep_num_sms = None


try:
    from deep_ep import HybridEPBuffer

    HAVE_HYBRIDEP = True
except ImportError:
    HAVE_HYBRIDEP = False

_hybrid_ep_buffer = None


# HybridEP dispatch/combine kernels use 64-token chunks for their public APIs.
HYBRIDEP_TOKEN_ALIGNMENT = 64


def init_hybrid_ep_buffer(
    group: torch.distributed.ProcessGroup,
    hidden_dim: int,
    num_tokens: int,
    num_local_experts: int,
    num_sms_dispatch_api: Optional[int] = None,
    num_sms_combine_api: Optional[int] = None,
    num_blocks_permute: Optional[int] = None,
    num_blocks_unpermute: Optional[int] = None,
    fp8_dispatch: bool = False,
    num_sms_preprocessing_api: Optional[int] = None,
) -> None:
    """
    Initialize the HybridEP buffer, including buffer allocation and metadata
    initialization.

    If a runtime dispatch/combine requires a larger buffer than the one
    initialized, the buffer will be reallocated at runtime,
    incuring extra run-time overhead.

    Args:
        group (torch.distributed.ProcessGroup):
            Process group for HybridEP all-to-all communication.
        hidden_dim (int):
            Hidden dimension of the input tensor.
        num_tokens (int):
            Maximum token count of the input tensor.
        num_local_experts (int):
            Number of local experts.
        num_sms_dispatch_api (Optional[int]):
            Number of SMs used by the dispatch API.
        num_sms_combine_api (Optional[int]):
            Number of SMs used by the combine API.
        num_blocks_permute (Optional[int]):
            Number of blocks used by the permute part.
        num_blocks_unpermute (Optional[int]):
            Number of blocks used by the unpermute part.
        fp8_dispatch (bool):
            Whether to use FP8 communication during the dispatch phase.
        num_sms_preprocessing_api (Optional[int]):
            Number of SMs used by the preprocessing (metadata scan) kernel.
    """
    assert not fp8_dispatch, "HybridEP dispatcher does not support fp8 dispatch now"
    global _hybrid_ep_buffer
    kwargs = {}
    if num_sms_dispatch_api is not None:
        kwargs["num_sms_dispatch_api"] = num_sms_dispatch_api
    if num_sms_combine_api is not None:
        kwargs["num_sms_combine_api"] = num_sms_combine_api
    if num_blocks_permute is not None:
        kwargs["num_blocks_permute"] = num_blocks_permute
    if num_blocks_unpermute is not None:
        kwargs["num_blocks_unpermute"] = num_blocks_unpermute
    if num_sms_preprocessing_api is not None:
        kwargs["num_sms_preprocessing_api"] = num_sms_preprocessing_api
    _hybrid_ep_buffer = HybridEPBuffer(
        group=group,
        hidden_dim=hidden_dim,
        max_num_of_tokens_per_rank=num_tokens,
        num_local_experts=num_local_experts,
        use_fp8=fp8_dispatch,
        **kwargs,
    )


def reset_hybrid_ep_buffer():
    """
    Reset the HybridEP buffer
    """
    global _hybrid_ep_buffer
    _hybrid_ep_buffer = None


class HybridEPDispatch(torch.autograd.Function):
    """
    Fused dispatch operation for permute + dispatch a2a + permute using the HybridEP backend
    """

    @staticmethod
    def forward(
        ctx,
        x,
        routing_map,
        probs,
        group,
        num_local_experts,
        num_sms_dispatch_api=None,
        num_sms_combine_api=None,
        num_blocks_permute=None,
        num_blocks_unpermute=None,
        fused=False,
        num_permuted_tokens=None,
        pad_multiple=None,
        num_sms_preprocessing_api=108,
    ):
        """
        Forward pass of fused dispatch of the HybridEP backend
        """
        if fused or num_blocks_permute is not None or num_blocks_unpermute is not None:
            import inspect
            import warnings

            sig = inspect.signature(HybridEPBuffer.dispatch_with_permute)
            if "fuse_permute_dispatch" not in sig.parameters:
                warnings.warn(
                    "Current DeepEP version does not support fused permute dispatch or "
                    "num_blocks_permute/num_blocks_unpermute. Falling back to unfused "
                    "HybridEP dispatch.",
                    UserWarning,
                    stacklevel=2,
                )
                fused = False
                num_blocks_permute = None
                num_blocks_unpermute = None

        if _hybrid_ep_buffer is None:
            num_tokens, hidden_dim = x.shape[-2:]
            fp8_dispatch = False  # Currently, we do not support fp8 dispatch
            init_hybrid_ep_buffer(
                group,
                hidden_dim,
                num_tokens,
                num_local_experts,
                num_sms_dispatch_api,
                num_sms_combine_api,
                num_blocks_permute,
                num_blocks_unpermute,
                fp8_dispatch,
                num_sms_preprocessing_api,
            )
        # If we provide the num_permuted_tokens, we do not need to use sync to
        # wait for the data in pinned memory ready
        non_blocking = num_permuted_tokens is not None
        # Process the dispatch
        (
            dispatched_hidden,
            dispatched_probs,
            dispatched_scaling_factor,
            tokens_per_expert,
            handle,
        ) = _hybrid_ep_buffer.dispatch_with_permute(
            hidden=x,
            routing_map=routing_map,
            probs=probs,
            scaling_factor=None,
            num_of_experts_per_rank=num_local_experts,
            pad_multiple=pad_multiple,
            num_permuted_tokens=num_permuted_tokens,
            non_blocking=non_blocking,
            **({"fuse_permute_dispatch": fused} if fused else {}),
        )

        ctx.handle = handle
        ctx.pad_multiple = pad_multiple
        ctx.fused = fused
        return (
            dispatched_hidden,
            dispatched_probs,
            dispatched_scaling_factor,
            tokens_per_expert,
            handle,
        )

    @staticmethod
    def backward(ctx, grad_x, grad_probs, grad_scaling_factor, grad_tokens_per_expert, grad_handle):
        """
        Backward pass of fused dispatch of the HybridEP backend
        """
        handle = ctx.handle
        combined_hidden, combined_probs = _hybrid_ep_buffer.combine_with_unpermute(
            hidden=grad_x,
            probs=grad_probs,
            handle=handle,
            pad_multiple=ctx.pad_multiple,
            **({"fuse_unpermute_combine": ctx.fused} if ctx.fused else {}),
        )
        return (
            combined_hidden,
            None,
            combined_probs,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


@internal_api
class HybridEPCombine(torch.autograd.Function):
    """
    Fused combine operation for permute + combine a2a + permute using the HybridEP backend
    """

    @staticmethod
    def forward(ctx, x, handle, num_permuted_tokens=None, pad_multiple=None, fused=False):
        """
        Forward pass of fused combine of the HybridEP backend
        """
        combined_hidden, _ = _hybrid_ep_buffer.combine_with_unpermute(
            hidden=x,
            handle=handle,
            pad_multiple=pad_multiple,
            **({"fuse_unpermute_combine": fused} if fused else {}),
        )
        ctx.handle = handle
        ctx.pad_multiple = pad_multiple
        ctx.num_permuted_tokens = num_permuted_tokens
        ctx.fused = fused
        return combined_hidden

    @staticmethod
    def backward(ctx, grad_x):
        """
        Backward pass of fused combine of the HybridEP backend
        """
        handle = ctx.handle
        dispatched_hidden, _, _, _, _ = _hybrid_ep_buffer.dispatch_with_permute(
            hidden=grad_x,
            scaling_factor=None,
            handle=handle,
            pad_multiple=ctx.pad_multiple,
            num_permuted_tokens=ctx.num_permuted_tokens,
            **({"fuse_permute_dispatch": ctx.fused} if ctx.fused else {}),
        )
        return dispatched_hidden, None, None, None, None


if HAVE_HYBRIDEP:

    @internal_api
    def hybrid_ep_dispatch(
        x,
        routing_map,
        probs,
        group,
        num_local_experts,
        num_sms_dispatch_api=None,
        num_sms_combine_api=None,
        num_blocks_permute=None,
        num_blocks_unpermute=None,
        fused=False,
        num_permuted_tokens=None,
        pad_multiple=None,
        num_sms_preprocessing_api=108,
    ):
        """
        Perform fused dispatch for "permute + dispatch a2a + permute" using the
        HybridEP backend.

        Args:
            x (torch.Tensor):
                Input hidden states to dispatch.
            routing_map (torch.Tensor):
                Map indicating which expert each token is routed to.
            probs (torch.Tensor):
                Routing probabilities for each token-expert pair.
            group (torch.distributed.ProcessGroup):
                Process group used for communication.
            num_local_experts (int):
                Number of local experts.
            num_sms_dispatch_api (Optional[int]):
                Number of SMs used by the dispatch API.
            num_sms_combine_api (Optional[int]):
                Number of SMs used by the combine API.
            num_blocks_permute (Optional[int]):
                Number of blocks used by the permute part.
            num_blocks_unpermute (Optional[int]):
                Number of blocks used by the unpermute part.
            num_permuted_tokens (int):
                Number of tokens after permute. HybridEP uses this to allocate buffers.
                If not provided, HybridEP obtains the size from a GPU tensor,
                which causes a D2H synchronization.
            pad_multiple (int):
                Alignment multiple required for FP8 GEMM. If not provided, no padding
                is performed.
            num_sms_preprocessing_api (int):
                Number of SMs used by the preprocessing (metadata scan) kernel.
        """
        return HybridEPDispatch.apply(
            x,
            routing_map,
            probs,
            group,
            num_local_experts,
            num_sms_dispatch_api,
            num_sms_combine_api,
            num_blocks_permute,
            num_blocks_unpermute,
            fused,
            num_permuted_tokens,
            pad_multiple,
            num_sms_preprocessing_api,
        )

    @internal_api
    def hybrid_ep_combine(x, handle, num_permuted_tokens, pad_multiple, fused=False):
        """
        Perform fused combine operation for unpermute + combine a2a + unpermute
        using the HybridEP backend

        args:
            x (torch.Tensor):
                Input hidden states to combine
            handle (EventHandle):
                Communication handle from dispatch operation
            num_permuted_tokens (int): The number of tokens before unpermute. HybridEP uses this
                to allocate buffers. If not provided, HybridEP obtains the size from a GPU tensor,
                which causes a D2H synchronization.
            pad_multiple (int):
                The alignment multiple required for FP8 GEMM. If not provided, no padding
                is performed.
        """
        return HybridEPCombine.apply(x, handle, num_permuted_tokens, pad_multiple, fused)

else:
    hybrid_ep_dispatch = None
    hybrid_ep_combine = None


try:
    from transformer_engine.pytorch import ep as te_ep

    HAVE_TE_EP = True
except ImportError:
    HAVE_TE_EP = False


def ensure_nccl_ep_bootstrapped(
    ep_group,
    num_experts,
    max_tokens_per_rank,
    recv_capacity_per_rank,
    hidden_dim,
    num_sms=0,
    zero_copy=False,
):
    """Initialize the process-wide NCCL EP context once. Idempotent.

    Collective on ``ep_group``: TE's ``ep_bootstrap`` issues a barrier and borrows the
    group's NCCL communicator, so every rank must call this with identical arguments
    before the first dispatch. Reuses TransformerEngine's own one-time flag, so repeated
    calls (e.g. once per MoE layer) are no-ops.

    Args:
        ep_group (torch.distributed.ProcessGroup): The expert-parallel process group.
        num_experts (int): Total experts across ``ep_group`` (global, not per-rank).
        max_tokens_per_rank (int): Upper bound on local input tokens per forward. Must be
            even (NCCL EP requires ``num_tokens_per_rank * inner_dim % 4 == 0``).
        recv_capacity_per_rank (int): Per-rank receive-buffer capacity in tokens. Must be
            ``>= max_tokens_per_rank``; runtime overflow hard-traps (no soft drop).
        hidden_dim (int): Token hidden size.
        num_sms (int): SM cap passed to TE as ``max_num_sms`` (0 lets TE/NCCL choose).
    """
    if not HAVE_TE_EP:
        raise RuntimeError(
            "transformer_engine.pytorch.ep is unavailable. The 'ncclep' flex dispatcher backend "
            "requires a TransformerEngine build with NCCL EP support (NVTE_BUILD_WITH_NCCL_EP=1)."
        )
    if te_ep._BOOTSTRAPPED:  # reuse TE's own one-time guard; no parallel state to drift
        return
    te_ep.ep_bootstrap(
        ep_group,
        num_experts=num_experts,
        max_tokens_per_rank=max_tokens_per_rank,
        recv_capacity_per_rank=recv_capacity_per_rank,
        hidden_dim=hidden_dim,
        max_num_sms=num_sms,
        zero_copy=zero_copy,
    )


def nccl_ep_finalize():
    """Tear down the NCCL EP context. Idempotent; safe when never bootstrapped.

    Releases the borrowed NCCL communicator and must run before the process group is
    destroyed.
    """
    if HAVE_TE_EP:
        te_ep.ep_finalize()


if HAVE_TE_EP:

    def alloc_ep_symm_buffer(shape, dtype, ep_group):
        """Allocate one persistent NCCL symm-mem buffer (per-buffer collective rendezvous). mcore's
        zero-copy buffers are all persistent and non-pool; the symm mem-pool is used only by TE for
        the per-call recv buffers it recycles."""
        return te_ep.symm_mem_alloc(shape, dtype, ep_group)

    def new_nccl_ep_buffer(
        top_k,
        max_tokens_per_rank,
        recv_capacity_per_rank,
        hidden_dim,
        num_local_experts,
        alignment=0,
    ):
        """Build a fresh TE EpBuffer for one dispatch/combine pair.

        The buffer owns handle_mem (the routing table dispatch writes and combine reads); a new one
        is built per dispatch and dropped after combine. Payload symm buffers are not owned here —
        they are caller-supplied to dispatch/combine or allocated on the fly by TE.
        """
        return te_ep.EpBuffer(
            top_k=top_k,
            max_tokens_per_rank=max_tokens_per_rank,
            recv_capacity_per_rank=recv_capacity_per_rank,
            hidden_dim=hidden_dim,
            num_local_experts=num_local_experts,
            alignment=alignment,
        )

    def nccl_ep_dispatch(
        buffer, tokens, topk_idx, topk_weights, recv_tokens=None, recv_topk_weights=None
    ):
        """Autograd-aware prepare + dispatch via TransformerEngine NCCL EP.

        Args:
            buffer (te_ep.EpBuffer): The TE EP buffer for this dispatch.
            tokens (torch.Tensor): Local input tokens ``[num_local_tokens, hidden]``
                (leading dims flattened by TE), ``payload_dtype``.
            topk_idx (torch.Tensor): ``int64`` ``[num_local_tokens, top_k]`` global expert
                ids per token.
            topk_weights (torch.Tensor): ``float32`` ``[num_local_tokens, top_k]`` weights.
            recv_tokens, recv_topk_weights (torch.Tensor, optional): caller-owned symm dispatch
                recv buffers (fp8 zero-copy). Left None, TE allocates them (bf16 zero-copy: symm
                mem-pool; normal: plain).

        Returns:
            tuple: ``(recv_tokens, tokens_per_expert, dispatched_probs)``:
              * ``recv_tokens``: packed received tokens ``[recv_capacity_per_rank, hidden]``,
                grouped by local expert (no separate compaction step).
              * ``tokens_per_expert``: ``int32`` ``[num_local_experts]`` device tensor of
                received counts per local expert (feeds grouped GEMM as group sizes;
                alignment-padded, == actual when ``alignment=0``).
              * ``dispatched_probs``: ``float32`` ``[recv_capacity_per_rank]`` per-slot
                weights; apply them in the expert MLP (combine is called unweighted).

            ``tokens_per_expert`` is non-differentiable.
        """
        recv_tokens, dispatched_probs, tokens_per_expert = te_ep.ep_dispatch(
            buffer,
            tokens,
            topk_idx,
            topk_weights,
            recv_tokens=recv_tokens,
            recv_topk_weights=recv_topk_weights,
        )
        return recv_tokens, tokens_per_expert, dispatched_probs

    def nccl_ep_combine(buffer, expert_out, num_local_tokens=None, grad_out=None):
        """Autograd-aware combine via TransformerEngine NCCL EP (no scatter step).

        Args:
            buffer (te_ep.EpBuffer): The TE EP buffer for this combine.
            expert_out (torch.Tensor): Expert outputs ``[recv_capacity_per_rank, hidden]``,
                already weighted.
            num_local_tokens (int): Rows of the result (local token count for this
                forward). When None, TE uses ``buffer.max_tokens_per_rank``.
            grad_out (torch.Tensor, optional): caller-owned symm buffer the backward scatters the
                expert_out grad into (zero-copy). Left None, TE allocates it (bf16: symm mem-pool;
                normal: plain).

        Returns:
            torch.Tensor: ``[num_local_tokens, hidden]`` combined output, in local token
            order.
        """
        return te_ep.ep_combine(
            buffer, expert_out, num_local_tokens=num_local_tokens, grad_out=grad_out
        )

else:
    alloc_ep_symm_buffer = None
    new_nccl_ep_buffer = None
    nccl_ep_dispatch = None
    nccl_ep_combine = None
