# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import logging
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from torch.distributed.tensor import DeviceMesh

from .allocator import BucketAllocator, TemporaryBucketAllocator, _free_storage
from .buffer_index import BufferIndex
from .mixed_precision import MixedPrecisionPolicy
from .utils import ParamGroupIdx

logger = logging.getLogger(__name__)


class DataParallelBuffer:
    """Manages a flat buffer that stores (a shard of) a group of parameters.

    On construction it builds its own BufferIndex describing the layout and
    shard ownership.  External callers interact via init_data / set_item /
    get_item only.
    """

    def __init__(
        self,
        params: List[torch.nn.Parameter],
        param_idx: Dict[torch.nn.Parameter, int],
        dtype: torch.dtype,
        device: torch.device,
        mesh: DeviceMesh,
        param_group_id: ParamGroupIdx,
        mp_policy: MixedPrecisionPolicy,
        *,
        allocator: Optional[BucketAllocator] = None,
        buffer_role: str = "model_weight",
        gradient_scaling_factor: Optional[float] = None,
        chunk_size_factor: int = 1,
        sharding_strategy: str = "no_shard",
        outer_dp_sharding_strategy: str = "no_shard",
    ):
        assert mp_policy is not None, "DataParallelBuffer requires a mixed-precision policy"
        self.params = params
        self.param_idx = param_idx
        self.dtype = dtype
        self.device = device
        self.outer_dp_group = mesh.get_group(mesh_dim=0)
        self.inner_dp_group = mesh.get_group(mesh_dim=1)
        self.allocator = allocator if allocator is not None else TemporaryBucketAllocator()
        self.buffer_role = buffer_role
        self.alloc_key = (param_group_id, buffer_role)
        self.mp_policy = mp_policy

        def get_sharding_from_strategy(strategy: str) -> bool:
            if buffer_role in ("model_weight", "transpose_weight"):
                return strategy == "optim_grads_params"
            if buffer_role == "main_weight":
                return strategy != "no_shard"
            if buffer_role == "main_grad":
                return strategy in ("optim_grads", "optim_grads_params")
            raise ValueError(f"Unsupported data-parallel buffer role: {buffer_role}")

        inner_sharded = get_sharding_from_strategy(sharding_strategy)
        outer_sharded = get_sharding_from_strategy(outer_dp_sharding_strategy)
        self.inner_sharded = inner_sharded
        self.outer_sharded = outer_sharded
        # shard_layout=(outer, inner): 1 means sharded and 0 means replicated.
        self.storage_shard_layout = (int(outer_sharded), int(inner_sharded))
        self.sharding_strategy = sharding_strategy
        self.outer_dp_sharding_strategy = outer_dp_sharding_strategy
        self.gradient_scaling_factor = gradient_scaling_factor

        # Always build layout with logical shapes and shared chunk_size_factor
        # so that all buffers share the same proportional item-offset mapping.
        _logical_shapes = [p.shape for p in params]
        self.buffer_index = BufferIndex(
            param_shapes=_logical_shapes,
            mesh=mesh,
            chunk_size_factor=chunk_size_factor,
            param_group_id=param_group_id,
        )

        # Compact NVFP4 weight buffers: scale all indices proportionally so
        # the buffer holds only the packed data without fragment-binning waste.
        if buffer_role in ("model_weight", "transpose_weight") and any(
            mp_policy.is_nvfp4_param(p) for p in params
        ):
            compact_shapes = mp_policy.get_param_storage_shapes(params)
            self.buffer_index.compact(0.5, compact_shapes)

        self.data_size = self.buffer_index.outer_shard_metas[self.storage_shard_layout].size

        self.data: Optional[torch.Tensor] = None
        self._unsharded_buffer: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def init_data(self, data: torch.Tensor) -> None:
        """Bind an externally allocated tensor as the persistent storage."""
        assert data.dtype == self.dtype, f"dtype mismatch: {data.dtype} vs {self.dtype}"
        assert data.numel() == self.data_size, f"size mismatch: {data.numel()} vs {self.data_size}"
        self.data = data
        self._inner_dirty = False
        self._outer_dirty = False

    # ------------------------------------------------------------------ #
    #  CPU offload
    # ------------------------------------------------------------------ #

    def _is_on_cpu(self) -> bool:
        """True if ``self.data`` is resident on CPU."""
        return self.data is not None and self.data.device.type == "cpu"

    def _ensure_data_on_gpu(self) -> bool:
        """Move ``self.data`` to GPU if currently on CPU.

        Returns True if a move happened (caller must rebuild dist views).
        """
        if not self._is_on_cpu():
            return False
        self.data = self.data.to(self.device, non_blocking=True)
        return True

    def _move_data_to(
        self, target_device: torch.device, pin_memory: bool = False, non_blocking: bool = True
    ) -> None:
        """Move ``self.data`` to *target_device*, optionally using pinned memory.

        Caller must call ``ParameterGroup._rebuild_dist_views()`` afterwards
        because ``dist_params._local_tensor`` views share ``self.data`` Storage.
        """
        if self.data is None or self.data.device == target_device:
            return
        if target_device.type == "cpu" and pin_memory:
            cpu_data = torch.empty(self.data.shape, dtype=self.data.dtype, pin_memory=True)
            cpu_data.copy_(self.data, non_blocking=non_blocking)
            _free_storage(self.data)
            self.data = cpu_data
        else:
            self.data = self.data.to(target_device, non_blocking=non_blocking)

    @torch.no_grad()
    def set_item(
        self,
        item_id: int,
        item_data: torch.Tensor,
        *,
        shard_layout: Optional[Iterable[int]] = None,
    ) -> None:
        """Write a parameter tensor into the corresponding region of the buffer."""
        requested_layout = (
            shard_layout if shard_layout is not None else self.storage_shard_layout
        )
        source_slice, local_slice = self.buffer_index.local_slice_for(
            self.buffer_index._get_item_global_range(item_id),
            requested_layout,
            self.storage_shard_layout,
        )
        if source_slice is None or local_slice is None:
            return
        self.data[local_slice].copy_(item_data.flatten()[source_slice])

    def get_item(
        self, item_id: int, *, shard_layout: Optional[Iterable[int]] = None
    ) -> torch.Tensor:
        """Read a parameter tensor (or its shard) from the buffer."""
        requested_layout = (
            shard_layout if shard_layout is not None else self.storage_shard_layout
        )
        _, local_slice = self.buffer_index.local_slice_for(
            self.buffer_index._get_item_global_range(item_id),
            requested_layout,
            self.storage_shard_layout,
        )
        return self.data[:0] if local_slice is None else self.data[local_slice]

    def is_unsharded(self) -> bool:
        """Return whether this buffer currently has a full unsharded view."""
        if self._outer_dirty or self._inner_dirty:
            return False
        # shard_layout=(outer, inner): (0, 0) means neither dimension is sharded.
        if self.storage_shard_layout != (0, 0):
            return self._unsharded_buffer is not None
        return self.data is not None

    @torch.no_grad()
    def unshard(
        self,
        unshard_dim: Optional[int] = 1,
        bind_params: bool = False,
        stream: Optional[torch.cuda.Stream] = None,
    ) -> torch.Tensor:
        """All-gather selected dimensions and optionally bind params.

        ``unshard_dim`` uses mesh dim ids: ``None`` does not unshard,
        ``0`` unshards outer-DP, and ``1`` unshards inner-DP.
        """
        current_stream = torch.cuda.current_stream()
        stream = stream or current_stream
        if stream != current_stream:
            stream.wait_stream(current_stream)

        # If unshard_dim is set, that dimension becomes replicated in the target.
        # Otherwise, every dimension keeps the current storage state.
        target_shard_layout = (
            0 if unshard_dim == 0 else self.storage_shard_layout[0],
            0 if unshard_dim == 1 else self.storage_shard_layout[1],
        )
        dirty_flags = (self._outer_dirty, self._inner_dirty)
        storage_is_dirty = (
            unshard_dim is not None
            and self.storage_shard_layout[unshard_dim] == 0
            and dirty_flags[unshard_dim]
        )
        # If storage is replicated but dirty, dimension d acts as a sharded source.
        # Otherwise, dimension d keeps the current storage state as the source.
        source_shard_layout = (
            1 if storage_is_dirty and unshard_dim == 0 else self.storage_shard_layout[0],
            1 if storage_is_dirty and unshard_dim == 1 else self.storage_shard_layout[1],
        )
        # Only a source-sharded -> target-replicated transition needs all-gather.
        requires_unshard = (
            unshard_dim is not None
            and source_shard_layout[unshard_dim] == 1
            and target_shard_layout[unshard_dim] == 0
        )

        # Fast path: target is already available from clean local storage.
        if not requires_unshard:
            output_buffer = self.fetch_buffer(target_shard_layout)
            if bind_params and target_shard_layout == (0, 0):
                self._bind_buffer_to_params(output_buffer)
            return output_buffer

        output_shard_layout = (
            0 if unshard_dim == 0 else source_shard_layout[0],
            0 if unshard_dim == 1 else source_shard_layout[1],
        )
        group = self.outer_dp_group if unshard_dim == 0 else self.inner_dp_group

        input_buffer = self.fetch_buffer(source_shard_layout)
        output_buffer = self.fetch_buffer(output_shard_layout)
        if torch.distributed.get_world_size(group) == 1:
            with torch.cuda.stream(stream):
                if output_buffer.data_ptr() != input_buffer.data_ptr():
                    output_buffer.copy_(input_buffer)
        else:
            with torch.cuda.stream(stream):
                torch.distributed.all_gather_into_tensor(
                    output_tensor=output_buffer,
                    input_tensor=input_buffer,
                    group=group,
                )

        setattr(self, "_outer_dirty" if unshard_dim == 0 else "_inner_dirty", False)

        # Parameter binding needs the full compute buffer.
        if bind_params and output_shard_layout == (0, 0):
            self._bind_buffer_to_params(output_buffer)
        return output_buffer

    def _bind_buffer_to_params(self, buffer: torch.Tensor) -> None:
        """Bind the given buffer to the params according to the layout."""
        assert buffer.numel() == self.buffer_index.bucket_meta.size, (
            f"Buffer size {buffer.numel()} does not match expected size "
            f"{self.buffer_index.bucket_meta.size}"
        )
        for p in self.params:
            item_id = self.param_idx[p]
            start, end = self.buffer_index._get_item_global_range(item_id)
            idx_shape = self.buffer_index.item_index_map[item_id].shape
            param_data = buffer[start:end].view(idx_shape)
            self.mp_policy.bind_unsharded_param(p, param_data, self.buffer_role)

    @torch.no_grad()
    def reshard(self, shard_dim: Optional[int] = None) -> None:
        """Release temporary buffers allocated by ``fetch_buffer`` / ``unshard``."""
        if shard_dim is not None:
            # If storage is already replicated on this dim, unshard() returned
            # self.data or a self.data view, so no temporary buffer was allocated.
            if self.storage_shard_layout[shard_dim] == 0:
                return
        self.allocator.free(self.alloc_key)
        self._unsharded_buffer = None

    def get_shard_view(self, shard_layout: Optional[Iterable[int]] = None) -> torch.Tensor:
        """Return a shard view inside the persistent data buffer."""
        assert self.data is not None, "DataParallelBuffer data not initialized"
        requested_layout = (
            shard_layout if shard_layout is not None else self.storage_shard_layout
        )
        _, local_slice = self.buffer_index.local_slice_for(
            (0, self.buffer_index.bucket_meta.size),
            requested_layout,
            self.storage_shard_layout,
        )
        return self.data[:0] if local_slice is None else self.data[local_slice]

    def fetch_buffer(self, shard_layout: Tuple[int, int] = (0, 0)) -> torch.Tensor:
        """Return a buffer for ``shard_layout``, allocating temporary storage if needed.

        1. If ``shard_layout`` matches this buffer's storage layout, return
           ``self.data`` directly.
        2. If ``self.data`` is a known parent layout of the requested shard,
           return a view into ``self.data``. Example: storage ``(0, 1)`` can
           return a ``(1, 1)`` view.
        3. Otherwise allocate/reuse the full ``(0, 0)`` unsharded buffer and
           return either that full buffer or a view from it. Example: storage
           ``(1, 1)`` requesting ``(0, 1)`` must materialize the full buffer
           because one outer shard cannot cover the complete inner-DP shard.

        Memory allocation always occurs on the caller stream for deterministic
        caching-allocator behaviour.
        """
        requested_shard_layout = shard_layout

        # 1. Exact storage match: no view or temporary buffer needed.
        if requested_shard_layout == self.storage_shard_layout:
            assert self.data is not None, "DataParallelBuffer data not initialized"
            return self.data

        # 2. Parent storage layouts can directly expose a child shard view.
        data_contains_requested = all(
            storage_dim == 0 or storage_dim == requested_dim
            for storage_dim, requested_dim in zip(self.storage_shard_layout, requested_shard_layout)
        )
        if data_contains_requested:
            return self.get_shard_view(requested_shard_layout)

        # 3. Otherwise materialize the full buffer and return the requested view
        # from it. This covers HSDP storage (1, 1) -> requested (0, 1).
        if self._unsharded_buffer is None:
            bucket = self.allocator.allocate(
                key=self.alloc_key,
                size=self.buffer_index.bucket_meta.size,
                dtype=self.dtype,
                device=self.device,
            )
            self._unsharded_buffer = bucket.data
        if requested_shard_layout == (0, 0):
            return self._unsharded_buffer
        requested_meta = self.buffer_index._get_shard_meta(requested_shard_layout)
        return self._unsharded_buffer[
            requested_meta.bucket_data_index : requested_meta.bucket_data_index
            + requested_meta.size
        ]

    @torch.no_grad()
    def reduce_grad(
        self,
        *,
        accumulate_reduced_grad: bool = False,
        reduce_dim: Optional[int] = 1,
        reduce_scatter: bool = True,
        grad_comm_dtype: Optional[torch.dtype] = None,
        stream: Optional[torch.cuda.Stream] = None,
    ):
        """Reduce gradients into the optimizer-facing local shard.

        ``reduce_dim`` uses mesh dim ids: ``None`` does not reduce,
        ``0`` reduces outer-DP, and ``1`` reduces inner-DP.
        ``reduce_scatter`` selects RS vs AR; ParameterGroup owns that strategy decision.
        ``accumulate_reduced_grad`` adds the collective result to an existing
        local output gradient instead of replacing it.
        """
        if reduce_dim is None:
            return

        current_stream = torch.cuda.current_stream()
        stream = stream or current_stream
        if stream != current_stream:
            stream.wait_stream(current_stream)

        grad_comm_dtype = grad_comm_dtype or self.mp_policy.grad_comm_dtype or self.dtype
        # Scale exactly once, when reducing fresh full grads over inner-DP.
        # Outer-only reduce consumes an already-scaled inner-DP result.
        if reduce_dim != 1 or self.gradient_scaling_factor in (None, 1.0):
            op = torch.distributed.ReduceOp.SUM
            prescale = False
        elif grad_comm_dtype != torch.bfloat16:
            op = torch.distributed._make_nccl_premul_sum(self.gradient_scaling_factor)
            prescale = False
        else:
            op = torch.distributed.ReduceOp.SUM
            prescale = True

        # Inner reduce consumes fresh full grads: (0, 0) -> (0, 1).
        # Outer reduce consumes the inner-reduced view: (0, 1) -> (1, 1).
        input_shard_layout = (
            0,
            0 if reduce_dim == 1 else self.storage_shard_layout[1],
        )
        # AR keeps the same shard view; RS shards the reduced dimension.
        output_shard_layout = (
            1 if reduce_scatter and reduce_dim == 0 else input_shard_layout[0],
            1 if reduce_scatter and reduce_dim == 1 else input_shard_layout[1],
        )
        input_buffer = self.fetch_buffer(input_shard_layout)
        output_buffer = self.fetch_buffer(output_shard_layout)

        # Pick the process group covering exactly the reduced dimension.
        group = self.outer_dp_group if reduce_dim == 0 else self.inner_dp_group
        if torch.distributed.get_world_size(group) == 1:
            if input_buffer.is_cuda:
                input_buffer.record_stream(stream)
            with torch.cuda.stream(stream):
                # A singleton inner-DP group bypasses both NCCL premul-sum and the
                # BF16 prescale path above, so apply its scaling locally.
                if reduce_dim == 1 and self.gradient_scaling_factor not in (None, 1.0):
                    input_buffer.mul_(self.gradient_scaling_factor)
                if output_buffer.data_ptr() != input_buffer.data_ptr():
                    if accumulate_reduced_grad:
                        output_buffer.add_(input_buffer)
                    else:
                        output_buffer.copy_(input_buffer)
            return

        comm_input = input_buffer
        input_key = None
        if grad_comm_dtype != self.dtype:
            input_key = (self.alloc_key, "grad_reduce_input", reduce_dim)
            input_bucket = self.allocator.allocate(
                key=input_key,
                size=input_buffer.numel(),
                dtype=grad_comm_dtype,
                device=self.device,
            )
            comm_input = input_bucket.data
            with torch.cuda.stream(stream):
                comm_input.copy_(input_buffer)
        if comm_input.is_cuda:
            comm_input.record_stream(stream)
        if prescale:
            with torch.cuda.stream(stream):
                comm_input.mul_(self.gradient_scaling_factor)

        if not reduce_scatter:
            with torch.cuda.stream(stream):
                torch.distributed.all_reduce(comm_input, group=group, op=op)
                if input_key is not None:
                    output_buffer.copy_(comm_input.to(self.dtype))
            if input_key is not None:
                self.allocator.free(input_key)
            return

        input_meta = self.buffer_index._get_shard_meta(input_shard_layout)
        output_meta = self.buffer_index._get_shard_meta(output_shard_layout)
        output_offset = output_meta.global_data_index - input_meta.global_data_index
        # Stage RS output in the input buffer slice; avoids untraced temp keys in TracePool.
        comm_output = comm_input[output_offset : output_offset + output_buffer.numel()]

        with torch.cuda.stream(stream):
            torch.distributed.reduce_scatter_tensor(
                output=comm_output,
                input=comm_input,
                group=group,
                op=op,
            )

            if output_buffer.data_ptr() != comm_output.data_ptr():
                if accumulate_reduced_grad:
                    output_buffer += comm_output
                else:
                    output_buffer.copy_(comm_output)
        if input_key is not None:
            self.allocator.free(input_key)
