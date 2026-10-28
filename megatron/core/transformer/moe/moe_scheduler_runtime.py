# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Upstream MoonEP expert storage and GTP/TE parameter lifecycle.

Shared runtime primitives; scheduling policy and kernel submission live separately.
"""

from __future__ import annotations

import math
import weakref
from dataclasses import dataclass
from enum import Enum
from functools import partial
from typing import Literal, NamedTuple

import torch
import torch.distributed as dist

from megatron.core.utils import ensure_params_ready


class WeightDirection(Enum):
    """Weight orientation used by the push, GTP consume and pointer table."""

    FORWARD = "forward"
    BACKWARD = "backward"


_MXFP8_COMPONENTS = (
    "_rowwise_data",
    "_rowwise_scale_inv",
    "_columnwise_data",
    "_columnwise_scale_inv",
)


def _is_gtp(parameter) -> bool:
    """Whether ``parameter`` is a GTP-sharded expert weight, gathered per forward."""
    return getattr(parameter, "is_gtp_weight_remat", False)


# ---- Virtual-expert slots: arenas and runtime parameters shared by every layer ---------------


@dataclass(frozen=True, slots=True)
class _VirtualExpertConfig:
    """The expert storage layout used to share slots between compatible layers."""

    group_name: str
    device: torch.device
    ep_size: int
    num_local_experts: int
    member_shapes: tuple  # (out_features, in_features) of FC1 and FC2
    mxfp8: bool
    grad_dtype: torch.dtype
    num_sms: int
    gtp: tuple[bool, ...] = (False, False)


class _VirtualExpertStorage:
    """Transport arenas and slots shared by layers with the same expert storage layout.

    MXFP8 main layers and BF16 MTP layers retain separate arenas. A layout's storage is reused
    across forwards; its NCCL registrations are released before the process group is destroyed.
    """

    def __init__(self, group, config: _VirtualExpertConfig, templates) -> None:
        self.config = config
        self.weight_arena = self.grad_arena = self.weight_handle = self.grad_handle = None
        self.slot_weights = ()
        try:
            self._allocate(group, config, templates)
        except Exception:
            self.destroy()
            raise

    def _allocate(self, group, config: _VirtualExpertConfig, templates) -> None:
        import torch.distributed._symmetric_memory as symm_mem

        device, count, mxfp8 = config.device, config.num_local_experts, config.mxfp8
        # What the kernels read.
        self.rank = dist.get_rank(group=group)
        self.world_size = config.ep_size
        self.num_local_experts = count
        self.member_numels = tuple(math.prod(shape) for shape in config.member_shapes)
        self.num_sms = config.num_sms
        # One E8M0 scale byte per 32 MXFP8 weight bytes, unpadded (the config requires 128-aligned
        # FC layers so TE's padded scale layout has this exact size).
        scale_numels = tuple(numel // 32 if mxfp8 else 0 for numel in self.member_numels)
        self._weight_sections = [
            count * n for pair in zip(self.member_numels, scale_numels) for n in pair
        ]
        self._grad_sections = [count * numel for numel in self.member_numels]
        try:
            if symm_mem.get_backend(device) != "NCCL":
                symm_mem.set_backend("NCCL")
            self.weight_arena = symm_mem.empty(
                sum(self._weight_sections),
                dtype=torch.uint8 if mxfp8 else torch.bfloat16,
                device=device,
            )
            self.weight_handle = symm_mem.rendezvous(self.weight_arena, group)
            self.grad_arena = symm_mem.empty(
                sum(self._grad_sections), dtype=config.grad_dtype, device=device
            )
            self.grad_handle = symm_mem.rendezvous(self.grad_arena, group)
        except RuntimeError as exc:
            raise RuntimeError(
                "Virtual-expert weights could not allocate NCCL symmetric memory for the EP group; "
                "the EP group must lie within one NVLink domain."
            ) from exc
        # The reduction reaches every peer's gradient slots through one TMA descriptor whose
        # outermost stride is the distance between consecutive peers' windows, so the allocator's
        # uniform mapping is a hard requirement: verify it once, here.
        bases = self.grad_handle.buffer_ptrs
        stride = bases[1] - bases[0] if len(bases) > 1 else 0
        if any(base != bases[0] + peer * stride for peer, base in enumerate(bases)):
            raise RuntimeError(
                "NCCL symmetric memory did not map the peers' gradient windows at a uniform "
                f"stride: {bases}."
            )
        self.weight_arena.zero_()
        self.grad_arena.zero_()
        self.weight_grid_barrier = torch.zeros(1, dtype=torch.int32, device=device)
        self.grad_grid_barrier = torch.zeros(1, dtype=torch.int32, device=device)
        self.slot_weights = tuple(
            self._slot_parameters(i, template) for i, template in enumerate(templates)
        )

    @staticmethod
    def _wrap_mxfp8(template, shape, views, device) -> tuple[torch.Tensor, ...]:
        """Wrap ``(rowwise, rowwise_scale, columnwise, columnwise_scale)`` views as MXFP8 tensors
        with ``template``'s quantization metadata."""
        from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Tensor

        return tuple(
            MXFP8Tensor(
                shape=shape,
                dtype=template.dtype,
                rowwise_data=rowwise,
                rowwise_scale_inv=rowwise_scale,
                columnwise_data=columnwise,
                columnwise_scale_inv=columnwise_scale,
                fp8_dtype=template._fp8_dtype,
                quantizer=template._quantizer,
                with_gemm_swizzled_scales=template._with_gemm_swizzled_scales,
                requires_grad=False,
                device=device,
            )
            for rowwise, rowwise_scale, columnwise, columnwise_scale in views
        )

    @staticmethod
    def _runtime_parameter(
        weight: torch.Tensor, main_grad: torch.Tensor, *, overwrite: bool = True
    ) -> torch.nn.Parameter:
        """A runtime leaf that writes fused wgrads into ``main_grad`` without DDP hooks."""
        parameter = torch.nn.Parameter(weight)
        parameter.main_grad = main_grad
        parameter.overwrite_main_grad = overwrite
        # These leaves have no DDP hooks. Without grad_added_to_main_grad, TE returns None
        # after writing the fused wgrad, avoiding dummy gradients and AccumulateGrad work
        # whose result would otherwise be immediately discarded.
        return parameter

    def _slot_parameters(self, fc_layer: int, template) -> tuple[torch.nn.Parameter, ...]:
        """One runtime parameter per slot of ``fc_layer`` over its arena sections (MXFP8-wrapped
        with ``template``'s quantization metadata), its ``main_grad`` the matching gradient slot.
        """
        count, shape = self.num_local_experts, self.config.member_shapes[fc_layer]
        weights = self.weight_arena.split(self._weight_sections)
        data = weights[2 * fc_layer].view(count, *shape)
        grads = self.grad_arena.split(self._grad_sections)[fc_layer].view(count, *shape)
        if not self.config.mxfp8:
            return tuple(self._runtime_parameter(w, g) for w, g in zip(data, grads))
        scales = weights[2 * fc_layer + 1].view(count, -1)
        if _is_gtp(template):
            rowwise, columnwise = _GTPTEWeightBridge._gtp_scale_shapes(template, shape)
        else:
            rowwise, columnwise = (
                template._rowwise_scale_inv.shape,
                template._columnwise_scale_inv.shape,
            )
        views = tuple(
            (data[i], scales[i].view(rowwise), data[i], scales[i].view(columnwise))
            for i in range(count)
        )
        # set_() empties the components at teardown but would leave a view's arena _base alive.
        views = tuple(tuple(view.detach() for view in slot) for slot in views)
        weights = self._wrap_mxfp8(template, shape, views, data.device)
        return tuple(self._runtime_parameter(w, g) for w, g in zip(weights, grads))

    def clear_accumulating_grads(self) -> None:
        """Zero non-GTP virtual-expert gradient slots after their reduction is complete.

        TE applies the native weights' gradient accumulation mode to the virtual slots too.
        Allocation zeros the slots for the first backward; _finish_grad_reduce clears them
        after waiting for all reduction readers, leaving them ready for the next use.
        GTP slots use overwrite mode and do not need clearing.
        """
        gtp = self.config.gtp
        if not any(gtp):
            self.grad_arena.zero_()
        elif not all(gtp):
            for sharded, section in zip(gtp, self.grad_arena.split(self._grad_sections)):
                if not sharded:
                    section.zero_()

    def destroy(self) -> None:
        """Release shared slots and NCCL registrations before destroying the process group.

        Empty the slot parameters in place so references held by layers and TE release the arenas.
        """
        if self.weight_arena is None and self.grad_arena is None:
            return
        torch.cuda.synchronize(self.config.device)
        with torch.no_grad():
            for slots in self.slot_weights:
                for slot in slots:
                    slot.main_grad = None
                    if self.config.mxfp8:
                        for name in _MXFP8_COMPONENTS:
                            getattr(slot, name).set_()
                    else:
                        slot.set_()
        self.weight_handle = self.grad_handle = self.weight_arena = self.grad_arena = None
        self.slot_weights = ()


class _PointerTable(NamedTuple):
    """Device addresses for a kernel and their host snapshot for stability checks."""

    tensor: torch.Tensor
    pointers: list[list[int]]
    row_views: tuple[torch.Tensor, ...]


class _GTPTEWeightBridge:
    """TE DistributedWeight protocol bound to BF16/MXFP8 runtime parameters.

    TE materializes weights and acquires wgrad buffers. The layer-input hook finalizes
    GTP only after remote virtual-expert gradients arrive."""

    def _gtp_runtime_shells(self, fc_layer: int):
        """Create native weight shells and empty gradients until GTP supplies their buffers."""
        config, storage = self.config, self.storage
        weights = self.parameters[fc_layer]
        if config.mxfp8:
            # Borrow slot storage until the weight push binds the gathered native weights.
            weights = storage._wrap_mxfp8(
                weights[0],
                config.member_shapes[fc_layer],
                tuple(
                    tuple(getattr(slot, name) for name in _MXFP8_COMPONENTS)
                    for slot in storage.slot_weights[fc_layer]
                ),
                config.device,
            )
        else:
            weights = (torch.empty(0, dtype=torch.bfloat16, device=config.device),) * len(weights)
        return weights, (self.placeholder,) * len(weights)

    @staticmethod
    def _gtp_scale_shapes(parameter, shape):
        """Get the gathered MXFP8 scale shapes from GTP's quantizer."""
        return tuple(
            parameter._gtp_gather_quantizer.get_scale_shape(shape, columnwise=c)
            for c in (False, True)
        )

    def _peek_gtp_weights(self, fc_layer: int, direction: WeightDirection):
        """Peek gathered weights and bind runtime shells without advancing GTP's prefetch chain."""
        leader = self.gtp_leaders[fc_layer]
        sources = _as_tuple(
            leader.peek_group_for_forward()
            if direction == WeightDirection.FORWARD
            else leader.peek_group_for_backward()
        )
        # bind the ready GTP buffers into the virtual expert weights
        for parameter, source in zip(self.native_weights[fc_layer], sources):
            if self.config.mxfp8:
                for name in self._components(direction):
                    setattr(parameter, name, getattr(source, name))
            else:
                parameter.data = source
        return sources

    def _bind_gtp_te_weights(self, fc_layer: int) -> None:
        """Bind TE's DistributedWeight API to the runtime weights for one FC layer."""
        weights = self.runtime_weights[fc_layer]
        num_native = len(self.parameters[fc_layer])
        # Bind unbound methods to a weak proxy: TE-held weights must not retain this layer.
        owner = weakref.proxy(self)
        leader = weights[0]
        leader.is_distributed_weight = True
        # Older TE fused GroupedMLP also checks these grouped-weight attributes.
        leader.is_routed_expert = True
        leader.weight_list = tuple(weakref.proxy(w) for w in weights)
        leader.materialize_group_for_forward = partial(
            type(self).materialize, owner, fc_layer, WeightDirection.FORWARD
        )
        leader.materialize_group_for_backward = partial(
            type(self).materialize, owner, fc_layer, WeightDirection.BACKWARD
        )
        leader.finalize_group_grads = partial(type(self)._on_te_wgrad_ready, owner, fc_layer)
        for index, weight in enumerate(weights):
            weight.grad_buffer = (
                partial(type(self)._gtp_grad_buffer, owner, fc_layer, index)
                if index < num_native
                else partial(getattr, weakref.proxy(weight), "main_grad")
            )

    def materialize(self, fc_layer: int, direction: WeightDirection):
        """TE API: consume GTP natives for forward/backward and append ready virtual slots."""
        leader = self.gtp_leaders[fc_layer]
        initializing = direction == WeightDirection.FORWARD and not leader.prefetch_initialized
        sources = _as_tuple(
            leader.materialize_group_for_forward()
            if direction == WeightDirection.FORWARD
            else leader.materialize_group_for_backward()
        )
        # The push already checked the runtime shells. TE computes with these returned sources,
        # so only verify that GTP handed out the same buffers the push read.
        if self._ptrs(direction, sources) != self._tables[fc_layer][direction].pointers:
            raise RuntimeError(
                f"FC{fc_layer + 1} {direction}: "
                "GTP source storage moved between push and materialization."
            )
        # First consume can replace its forward ticket while discovering the chain. The
        # next push binds the established allocation without constraining GTP's cache setup.
        if initializing:
            self._tables[fc_layer].pop(WeightDirection.FORWARD)
        return (*sources, *self.storage.slot_weights[fc_layer])

    def _gtp_grad_buffer(self, fc_layer: int, index: int) -> torch.Tensor:
        """TE API: ``grad_buffer()``. Acquire persistent native wgrad buffers on demand."""
        if self.native_grads[fc_layer] is None:
            self._bind_gtp_grads(
                fc_layer,
                tuple(
                    w.get_wgrad_tensor(persistent=True) for w in self.gtp_leaders[fc_layer]._weights
                ),
            )
        return self.native_grads[fc_layer][index]

    def _on_te_wgrad_ready(self, fc_layer: int, wgrads):
        """TE API: finalize_group_grads starts FC2's VE reduction; GTP finalizes after it."""
        expected = [w.grad_buffer() for w in self.runtime_weights[fc_layer]]
        wgrads = _as_tuple(wgrads)
        if len(wgrads) != len(expected) or any(
            grad.shape != target.shape for grad, target in zip(wgrads, expected)
        ):
            raise RuntimeError("TE virtual-expert wgrads must match the supplied buffer shapes.")
        # Some TE paths return scratch instead of writing into grad_buffer(). Copy before
        # VE reduction reads these buffers; GTP's own copy fallback runs after that reduction.
        for grad, target in zip(wgrads, expected):
            if grad.data_ptr() != target.data_ptr():
                target.copy_(grad)
        if fc_layer == 1 and self._start_grad_reduce is not None:
            self._start_grad_reduce()(fc_layer)
        return (None,) * len(expected)

    def _bind_gtp_grads(self, fc_layer: int, grads) -> None:
        """Bind GTP's wgrad buffers before the GEMM, or park natives on empty placeholders.

        Check an existing pointer table without creating one; the first reduction creates it.
        """
        if grads is not None:
            if self.native_grads[fc_layer] is not None:
                raise RuntimeError(f"FC{fc_layer + 1}: native wgrads bound twice in one backward.")
            if "grad" in self._tables[fc_layer]:
                self.get_weight_table(fc_layer, "grad", grads)
        targets = (
            grads if grads is not None else (self.placeholder,) * len(self.parameters[fc_layer])
        )
        self.native_grads[fc_layer] = grads
        for parameter, grad in zip(self.native_weights[fc_layer], targets):
            parameter.main_grad = grad

    def _reduce_scatter_gtp_grads(self, fc_layer: int) -> tuple[torch.Tensor, ...]:
        """Finalize GTP after remote VE gradients arrive, then release the native bindings."""
        if self.native_grads[fc_layer] is None:
            raise RuntimeError(f"FC{fc_layer + 1}: no GTP wgrad scratch bound for this backward.")
        reduced = self.gtp_leaders[fc_layer].finalize_group_grads(list(self.native_grads[fc_layer]))
        self._bind_gtp_grads(fc_layer, None)
        return _as_tuple(reduced)


class _VirtualExperts(_GTPTEWeightBridge):
    """One layer's runtime weights, GTP bindings and pointer tables over shared storage."""

    def __init__(self, storage: _VirtualExpertStorage, parameters, start_grad_reduce=None) -> None:
        self.storage = storage
        self.config = config = storage.config
        self.parameters = parameters
        self.source_parameters = tuple(parameter for group in parameters for parameter in group)
        self._start_grad_reduce = (
            weakref.WeakMethod(start_grad_reduce) if start_grad_reduce is not None else None
        )
        self.gtp_leaders = tuple(p[0] if _is_gtp(p[0]) else None for p in parameters)
        self.native_grads = [None] * len(parameters)
        self.placeholder = torch.empty(0, dtype=config.grad_dtype, device=config.device)
        self.runtime_weights = []
        self.native_weights = []
        for i, weights in enumerate(parameters):
            slots = storage.slot_weights[i]
            sharded = self.gtp_leaders[i] is not None
            if sharded:
                weights, grads = self._gtp_runtime_shells(i)
            else:
                grads = tuple(p.main_grad for p in weights)
                self.native_grads[i] = grads
            natives = tuple(
                storage._runtime_parameter(w, g, overwrite=sharded) for w, g in zip(weights, grads)
            )
            self.native_weights.append(natives)
            self.runtime_weights.append((*natives, *slots))
            if sharded:
                self._bind_gtp_te_weights(i)
        self._tables: list[dict[WeightDirection | Literal["grad"], _PointerTable]] = [
            {} for _ in parameters
        ]

    def _components(self, key: WeightDirection | Literal["grad"]) -> tuple[str, ...]:
        if self.config.mxfp8 and key != "grad":
            offset = 2 if key == WeightDirection.BACKWARD else 0
            return _MXFP8_COMPONENTS[offset : offset + 2]
        return ("data",)

    def _ptrs(self, key: WeightDirection | Literal["grad"], sources) -> list[list[int]]:
        return [
            [getattr(source, name).data_ptr() for source in sources]
            for name in self._components(key)
        ]

    def get_weight_table(
        self, fc_layer: int, key: WeightDirection | Literal["grad"], sources
    ) -> torch.Tensor:
        """Create the ``forward``, ``backward`` or ``grad`` table once, then check stability."""
        rows = self._ptrs(key, sources)
        table = self._tables[fc_layer].get(key)
        natives = self.native_weights[fc_layer]
        if (table is not None and rows != table.pointers) or (
            key != "grad" and self._ptrs(key, natives) != rows
        ):
            raise RuntimeError(
                f"FC{fc_layer + 1} {key}: source or runtime storage moved after binding; "
                "storage must be static."
            )
        if table is None:
            tensor = torch.tensor(rows, dtype=torch.int64, device=self.config.device)
            table = _PointerTable(tensor, rows, tuple(tensor.unbind()))
            self._tables[fc_layer][key] = table
        return table.tensor

    def grad_table(self, fc_layer: int) -> torch.Tensor:
        """Get only this FC's targets; the other FC may not have acquired its wgrad buffers yet."""
        if "grad" not in self._tables[fc_layer]:
            self.get_weight_table(fc_layer, "grad", self.native_grads[fc_layer])
        return self._tables[fc_layer]["grad"].row_views[0]

    def weight_tables(self, direction: WeightDirection) -> tuple[torch.Tensor, ...]:
        """Publish DDP parameters or peek GTP gathers before the push, then check their tables.

        GTP consumption and its next prefetch remain at the expert GEMMs.
        """
        tables = []
        for i, sources in enumerate(self.parameters):
            if self.gtp_leaders[i] is not None:
                sources = self._peek_gtp_weights(i, direction)
            elif direction == WeightDirection.FORWARD:
                ensure_params_ready(sources)
            tables.append(self.get_weight_table(i, direction, sources))
        return tuple(tables)

    def prepare_backward(self) -> None:
        """Bind DDP gradient pointers; TE acquires GTP buffers through grad_buffer()."""
        for i, sharded in enumerate(self.config.gtp):
            if not sharded:
                self.get_weight_table(i, "grad", tuple(p.main_grad for p in self.parameters[i]))

    def hand_off_wgrads(self, fc_layer: int) -> tuple[torch.Tensor, ...]:
        """Finish GTP reduction or return dummy grads to trigger DDP's grad-ready hooks.

        Non-GTP weights already accumulated into main_grad; do not add their gradients again."""
        from transformer_engine.pytorch.module.base import get_dummy_wgrad

        if self.gtp_leaders[fc_layer] is not None:
            return self._reduce_scatter_gtp_grads(fc_layer)

        parameters = self.parameters[fc_layer]
        grads = []
        for parameter in parameters:
            parameter.grad_added_to_main_grad = True
            grads.append(
                get_dummy_wgrad(
                    list(parameter.shape),
                    parameter.dtype,
                    zero=getattr(parameter, "zero_out_wgrad", False),
                )
            )
        return tuple(grads)


# ---- Load balancer ---------------------------------------------------------------------------


class _VirtualExpertHook(torch.autograd.Function):
    """Call a method during backward, preserving the tensor gradient.

    Its return values are gradients for ``inputs``. Hold the method weakly: the manager
    retains forward tensors, so a strong reference would keep the autograd graph alive."""

    @staticmethod
    def forward(ctx, tensor, method, args, *inputs):
        """Save the callback for backward and return the tensor unchanged."""
        ctx.method, ctx.args = weakref.WeakMethod(method), args
        return tensor

    @staticmethod
    def backward(ctx, grad):
        """Run the callback to obtain input gradients and preserve the tensor gradient."""
        return (grad, None, None, *(ctx.method()(*ctx.args) or ()))


def _as_tuple(value) -> tuple:
    return tuple(value) if isinstance(value, (list, tuple)) else (value,)
