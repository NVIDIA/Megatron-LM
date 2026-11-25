# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""BF16 packed expert parameters for GTP's DistributedWeight interface.

Storage is [expert, local output row, input column]. Collectives use rank-major
buffers, so expert and rank axes must be exchanged at each communication boundary.
The packed owner shares GTP's prefetch chains, completion events and gradient
reduction lifecycle; only the collective layout adapters differ.
"""

from contextlib import nullcontext
from copy import copy
from dataclasses import replace

import torch
from transformer_engine.pytorch.tensor.grouped_tensor import GroupedTensor

from megatron.core.utils import ensure_params_ready

_PACKED_CLASSES = {}


class _PackedGTPWeight:
    """Methods mixed into TE's GroupedTensor without replacing its storage dispatch."""

    @property
    def _unsharded_shape_padded(self):
        experts, rows, columns = self.shape
        return (experts, rows * self.group.size(), columns)

    @property
    def _unsharded_shape(self):
        experts, rows, columns = self._unsharded_shape_padded
        return (experts, rows - self.pad_length, columns)

    def _strip_padding(self, tensor):
        # Padding belongs to each expert's row axis, not the leading expert axis.
        return tensor[:, : self._unsharded_shape[1]].contiguous() if self.pad_length else tensor

    def _all_gather_weight(
        self, async_op, fwd, nvtx_label=None, recompute=False, recompute_prev=None
    ):
        """Gather and reorder on the AG stream before publishing the completion event."""
        from megatron.core.tensor_parallel.generalized_tensor_parallelism import (
            GTP_CONFIG,
            GTPShardHandle,
            GTPWeightState,
            get_ag_stream,
            get_global_GTP_cache,
            in_activation_recompute_phase,
        )

        if fwd and not in_activation_recompute_phase():
            ensure_params_ready([self])
        if GTP_CONFIG.check_param_states and not in_activation_recompute_phase():
            self._set_state(
                GTPWeightState.ASYNC_WAIT if async_op else GTPWeightState.DATA_READY_SYNC
            )
        self._cached_dtypes = [self.dtype]
        cache = get_global_GTP_cache()
        attr = (
            "_ag_ticket_recompute" if recompute else "_ag_ticket_fwd" if fwd else "_ag_ticket_bwd"
        )
        if getattr(self, attr) is None:
            if recompute:
                self._ensure_no_shared_buffer_with(
                    predecessor=recompute_prev,
                    predecessor_dtype=self.dtype if recompute_prev is not None else None,
                    dtype=self.dtype,
                    parity_attr="_recompute_buf_parity",
                )
            ticket = cache.reserve(self, self.dtype, fwd=fwd, recompute=recompute)
            setattr(self, attr, ticket)
            if fwd:
                cache.get(ticket)
                cache.release(ticket)
        output = cache.get(getattr(self, attr))
        if async_op:
            stream = get_ag_stream(self.chain_id, self.group)
            if getattr(self, "_ag_outer_sync_event", None) is None:
                self._ag_outer_sync_event = torch.cuda.Event()
            self._ag_outer_sync_event.record()
            stream.wait_event(self._ag_outer_sync_event)
            context = torch.cuda.stream(stream)
        else:
            context = nullcontext()
        experts, local_rows, columns = self.shape
        world_size = self.group.size()
        with context:
            # Allocate rank-major scratch on the stream which writes AND consumes it.
            # The cached expert-major output follows the common GTP ticket lifecycle.
            gathered = torch.empty(
                (world_size * experts, local_rows, columns), device=self.device, dtype=self.dtype
            )
            handle = torch.distributed.all_gather_into_tensor(
                gathered, self.rowwise_data.view(self.shape), group=self.group, async_op=async_op
            )
            if handle is not None:
                handle.wait()
            output.view(experts, world_size, local_rows, columns).copy_(
                gathered.view(world_size, experts, local_rows, columns).transpose(0, 1)
            )
            if async_op:
                event = self._recompute_ag_event if recompute else self.ag_event
                event.record()
        return [output], GTPShardHandle(None, [self]) if async_op else None

    def _materialize(self, tensors):
        """Expose the scheduler's single expert-major buffer through TE's packed API."""
        (packed,) = tensors
        return GroupedTensor.make_grouped_tensor_from_rowwise_data(
            num_tensors=self.shape[0],
            tensor_shape=tuple(packed.shape[1:]),
            rowwise_data=packed.reshape(-1),
            dtype=self.dtype,
        )

    def materialize_group_for_forward(self):
        """Consume this weight and prefetch the next block's packed projection."""
        return self._materialize(self.all_gather_and_prefetch(fwd=True))

    def materialize_group_for_backward(self, nvtx_label=None):
        """Consume this weight and prefetch the preceding block for backward."""
        return self._materialize(self.all_gather_and_prefetch_bwd(nvtx_label=nvtx_label))

    def peek_group_for_forward(self):
        """Expose prefetched data without advancing the forward chain."""
        return self._materialize(self._peek_gathered_weight(fwd=True))

    def peek_group_for_backward(self):
        """Expose prefetched data without advancing the backward chain."""
        return self._materialize(self._peek_gathered_weight(fwd=False))

    def grad_buffer(self):
        """Give TE pooled scratch whose lifetime is retained until reduction completes."""
        from megatron.core.tensor_parallel.generalized_tensor_parallelism import _wgrad_pool_get

        return _wgrad_pool_get(self._unsharded_shape, self.main_grad.dtype, self.device)

    def _prepare_wgrad_reduce_scatter_inputs(self, wgrads):
        """Convert expert-major gradients to one rank-major reduction input."""
        from megatron.core.tensor_parallel.generalized_tensor_parallelism import _wgrad_pool_get

        (wgrad,) = wgrads
        if tuple(wgrad.shape) != self._unsharded_shape:
            raise ValueError(
                f"Packed GTP gradient shape {tuple(wgrad.shape)} != {self._unsharded_shape}"
            )
        experts, local_rows, columns = self.shape
        world_size = self.group.size()
        padded = (
            torch.nn.functional.pad(wgrad, (0, 0, 0, self.pad_length)) if self.pad_length else wgrad
        )
        send = _wgrad_pool_get(
            (world_size * experts, local_rows, columns), self.main_grad.dtype, self.device
        )
        send.view(world_size, experts, local_rows, columns).copy_(
            padded.view(experts, world_size, local_rows, columns).transpose(0, 1)
        )
        # Both buffers stay owned by the common RS lifecycle until its completion
        # event. In particular, do not recycle send while the RS stream still reads it.
        return [send], [wgrad, send]

    def finalize_group_grads(self, wgrads, nvtx_label=None):
        """Use GTP's deferred reduction and DDP readiness cascade for the packed owner."""
        if isinstance(wgrads, (list, tuple)):
            (wgrads,) = wgrads
        return self.wgrad_reduce_scatter(wgrads, nvtx_label=nvtx_label)

    def checkpoint_expert_views(self):
        """Expose local per-expert views with GTP metadata for checkpoint resharding."""
        from megatron.core.tensor_parallel.layers import (
            copy_gtp_attributes,
            copy_tensor_model_parallel_attributes,
        )

        views = list(self.rowwise_data.view(self.shape).unbind(0))
        for view in views:
            copy_gtp_attributes(view, self)
            copy_tensor_model_parallel_attributes(view, self)
            view.gtp_remat_size = self.gtp_remat_size
        return views


def attach_packed_gtp_weight(module, group, pad_length, replica_group):
    """Retain one TE parameter and attach a packed BF16 GTP implementation in place."""
    from megatron.core.tensor_parallel.generalized_tensor_parallelism import (
        _gtp_attach_attrs,
        _gtp_native_fp8_subclass,
        _init_gtp_runtime_attrs,
    )

    param = module.weight
    if (
        not isinstance(param, GroupedTensor)
        or param.dtype != torch.bfloat16
        or param.rowwise_data is None
        or param.rowwise_data.dtype != torch.bfloat16
    ):
        raise NotImplementedError("Packed GTP currently requires BF16 expert-weight storage")
    base_cls = type(param)
    if base_cls not in _PACKED_CLASSES:
        _PACKED_CLASSES[base_cls] = type(
            f"GTPPacked_{base_cls.__name__}",
            (_PackedGTPWeight, _gtp_native_fp8_subclass(base_cls)),
            {},
        )
    param.__class__ = _PACKED_CLASSES[base_cls]
    _init_gtp_runtime_attrs(param)
    param.pad_length = pad_length
    _gtp_attach_attrs(param, group, is_grouped=True, replica_group=replica_group)
    param.weight_list = [param]
    # A single packed owner is the optimizer/DDP and prefetch-chain unit.
    # Packed layout conversion currently uses ordinary NCCL allocations.
    param.needs_nccl_mem = False


def make_packed_checkpoint_factory(param, expert_shards):
    """Keep the packed optimizer identity while saving canonical per-expert tensors.

    Compose the existing expert factories (including SwiGLU's GTP mapping) and
    intersect flattened optimizer fragments with each expert's local storage.
    """
    # Expand the per-expert transformations once using the real local weights. Each
    # leaf describes a contiguous interval of the packed parameter (SwiGLU splits a
    # local expert into ordered row tiles). Reuse those exact offsets for optimizer
    # fragments; older SwiGLU factories do not implement flattened_range themselves.
    from megatron.core.dist_checkpointing.dict_utils import nested_values
    from megatron.core.dist_checkpointing.mapping import (
        LocalNonpersistentObject,
        ShardedTensorFactory,
    )

    templates = []
    for expert_shard in expert_shards:
        leaves = (
            list(nested_values(expert_shard.build()))
            if isinstance(expert_shard, ShardedTensorFactory)
            else [expert_shard]
        )
        offset = 0
        for leaf in leaves:
            if leaf.data.data_ptr() != (
                expert_shard.data.data_ptr() + offset * expert_shard.data.element_size()
            ):
                raise ValueError(
                    "Packed checkpoint factories require ordered contiguous expert tiles"
                )
            template = copy(leaf)
            size = leaf.data.numel()
            template.data = None
            templates.append((template, size))
            offset += size
        if offset != expert_shard.data.numel():
            raise ValueError("Expert checkpoint tiles must cover the complete local expert")
    base_key = expert_shards[0].key
    total_size = param.numel()

    def build(key, data, replica_id, flattened_range):
        raw = data.rowwise_data if isinstance(data, GroupedTensor) else data
        flat = raw.reshape(-1)
        start, stop = (
            (0, total_size)
            if flattened_range is None
            else (flattened_range.start, flattened_range.stop)
        )
        if not key.endswith(base_key):
            raise ValueError(f"Packed checkpoint key {key!r} must end in {base_key!r}")
        key_prefix = key[: -len(base_key)]
        parts = []
        offset = 0
        for template, size in templates:
            lo, hi = max(start, offset), min(stop, offset + size)
            if lo < hi:
                view = flat[lo - start : hi - start]
                if flattened_range is None:
                    child = replace(
                        template,
                        key=key_prefix + template.key,
                        data=view.view(template.local_shape),
                        dtype=view.dtype,
                        replica_id=replica_id,
                    )
                    child.gtp_pad_length = getattr(template, "gtp_pad_length", 0)
                    parts.append(child)
                else:
                    # DCP stores rectangular shards, not flattened ranges. A flat
                    # matrix interval is at most a partial first row, full rows,
                    # and a partial last row. Preserve the canonical global shape.
                    columns = template.local_shape[1]
                    position, end = lo - offset, hi - offset
                    consumed = 0
                    while position < end:
                        row, column = divmod(position, columns)
                        remaining = end - position
                        if column or remaining < columns:
                            rows, width = 1, min(columns - column, remaining)
                        else:
                            rows, width = remaining // columns, columns
                        count = rows * width
                        chunk = view[consumed : consumed + count].view(rows, width)
                        global_offset = list(template.global_offset)
                        global_offset[template.prepend_axis_num] += row
                        global_offset[template.prepend_axis_num + 1] += column
                        parts.append(
                            replace(
                                template,
                                key=key_prefix + template.key,
                                data=chunk,
                                dtype=chunk.dtype,
                                local_shape=tuple(chunk.shape),
                                global_offset=tuple(global_offset),
                                axis_fragmentations=None,
                                replica_id=replica_id,
                                flattened_range=None,
                            )
                        )
                        position += count
                        consumed += count
            offset += size
        return {"parts": parts, "shape": LocalNonpersistentObject(tuple(data.shape))}

    def merge(state):
        return torch.cat([part.reshape(-1) for part in state["parts"]]).view(state["shape"])

    return ShardedTensorFactory(base_key, param, build, merge, expert_shards[0].replica_id)
