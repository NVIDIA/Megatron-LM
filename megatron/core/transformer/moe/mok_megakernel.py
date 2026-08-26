# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Minimal Mixture-of-Kittens adapter for MCore MoE training experiments.

This module intentionally keeps the integration narrow: MCore owns routing,
parameters, DDP, the optimizer, and the logical checkpoint format, while MoK
replaces dispatch, routed/shared expert computation, and combine.
"""

from __future__ import annotations

import itertools
from typing import Any, Iterable

import torch
from torch import nn

from megatron.core.fusions.fused_indices_converter import fused_routing_map_to_indices

_MOK_MODULE_INDICES = itertools.count()
_MOK_HIGH_PRECISION_INIT_ATTR = "_mok_high_precision_init_val"
_MOK_MXFP8_COMPAT_WARNING_EMITTED = False


def _warn_mxfp8_compatibility_fallback() -> None:
    """Emit the non-single MXFP8 hybrid-memory tradeoff once on rank zero."""
    global _MOK_MXFP8_COMPAT_WARNING_EMITTED
    if _MOK_MXFP8_COMPAT_WARNING_EMITTED:
        return
    _MOK_MXFP8_COMPAT_WARNING_EMITTED = True
    if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
        return
    print(
        "WARNING: MOK MXFP8 with moe_single_grouped_weight=False uses the "
        "hybrid compatibility fallback: routed weights remain BF16 parameters and "
        "use BF16 parameter buffers, while other eligible TE parameters retain the "
        "configured FP8 parameter gather and grad-buffer reuse behavior. MOK also "
        "materializes cached rowwise/columnwise MXFP8 routed-weight copies, which uses "
        "substantially more GPU memory than the native single-grouped path.",
        flush=True,
    )


def _debug_record(
    stage: str,
    param: torch.Tensor,
    *,
    tensors: dict[str, torch.Tensor | None] | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Record an opt-in lifecycle snapshot without importing debug code normally."""
    from megatron.core.mok_param_lifecycle_debug import enabled, record

    if enabled():
        record(stage, param, tensors=tensors, metadata=metadata)


def _debug_tag(param: nn.Parameter, name: str) -> None:
    from megatron.core.mok_param_lifecycle_debug import enabled, tag_parameter

    if enabled():
        tag_parameter(param, name)


def _copy_parameter_attributes(dst: nn.Parameter, src: torch.Tensor, *, allreduce: bool) -> None:
    """Copy the parameter metadata MCore uses for optimizer/DDP classification."""
    dst.allreduce = allreduce
    for name in (
        "sequence_parallel",
        "tensor_model_parallel",
        "partition_dim",
        "partition_stride",
        "shared",
    ):
        if hasattr(src, name):
            setattr(dst, name, getattr(src, name))


def _dequantize_bf16(tensor: torch.Tensor) -> torch.Tensor:
    """Materialize a logical TE/plain parameter as an ordinary BF16 tensor."""
    tensor = tensor.detach()
    dequantize = getattr(tensor, "dequantize", None)
    if callable(dequantize):
        tensor = dequantize()
    return tensor.to(dtype=torch.bfloat16)


def _materialize_parameter_init(tensor: torch.Tensor) -> tuple[torch.Tensor, bool]:
    """Return the logical initialization value and whether TE preserved it losslessly."""
    get_high_precision_init_val = getattr(tensor, "get_high_precision_init_val", None)
    if callable(get_high_precision_init_val):
        init_val = get_high_precision_init_val().detach()
        tensor.clear_high_precision_init_val()
        return init_val, True
    return _dequantize_bf16(tensor), False


def _attach_high_precision_init(param: nn.Parameter, init_val: torch.Tensor | None) -> None:
    """Preserve a reordered pre-quantization init until MCore creates its FP32 master."""
    if init_val is None:
        return
    if init_val.shape != param.shape:
        raise RuntimeError(
            "MOK high-precision initialization shape mismatch: "
            f"init={tuple(init_val.shape)}, param={tuple(param.shape)}"
        )
    setattr(param, _MOK_HIGH_PRECISION_INIT_ATTR, init_val.detach().contiguous())


def _deinterleave_glu_weight(
    weight: torch.Tensor,
    *,
    intermediate_size: int,
    interleave_size: int,
) -> torch.Tensor:
    """Convert ``[gate block, up block, ...]`` rows to contiguous ``[gate; up]``."""
    if interleave_size <= 0 or intermediate_size % interleave_size != 0:
        raise ValueError(
            "MOK source GLU interleave size must be positive and divide the "
            f"intermediate size, got interleave={interleave_size}, "
            f"intermediate={intermediate_size}"
        )
    if weight.ndim < 2 or weight.shape[-2] != 2 * intermediate_size:
        raise RuntimeError(
            "MOK source routed-FC1 shape mismatch: expected the penultimate "
            f"dimension to be {2 * intermediate_size}, got {tuple(weight.shape)}"
        )

    prefix = weight.shape[:-2]
    hidden_size = weight.shape[-1]
    return (
        weight.reshape(
            *prefix,
            intermediate_size // interleave_size,
            2,
            interleave_size,
            hidden_size,
        )
        .transpose(-4, -3)
        .contiguous()
        .reshape(weight.shape)
    )


@torch.no_grad()
def _convert_native_fc1_init_to_mok_layout(
    param: nn.Parameter,
    *,
    intermediate_size: int,
    interleave_size: int | None,
) -> None:
    """Reorder one native FC1 initialization before DDP/optimizer construction."""
    if interleave_size is None:
        return

    source, has_preserved_init = _materialize_parameter_init(param)
    converted = _deinterleave_glu_weight(
        source,
        intermediate_size=intermediate_size,
        interleave_size=interleave_size,
    )

    from megatron.core.fp8_utils import (
        copy_back_gathered_bf16_into_fp8_param,
        copy_tensor_to_quantized_param,
        is_float8tensor,
        is_grouped_tensor_with_quantized_storage,
    )

    converted_bf16 = converted.to(dtype=torch.bfloat16)
    if is_grouped_tensor_with_quantized_storage(param):
        copy_tensor_to_quantized_param(param, converted_bf16)
    elif is_float8tensor(param):
        copy_back_gathered_bf16_into_fp8_param(param, converted_bf16)
    else:
        param.copy_(converted_bf16)

    if has_preserved_init:
        _attach_high_precision_init(param, converted)


def _indexed_grouped_weight(linear: nn.Module, index: int, num_experts: int) -> torch.Tensor:
    """Return one logical expert weight from either TE grouped layout."""
    if not getattr(linear, "single_grouped_weight", False):
        return getattr(linear, f"weight{index}")

    weight = linear.weight
    split_quantized = getattr(weight, "split_into_quantized_tensors", None)
    if callable(split_quantized):
        return split_quantized()[index]
    if weight.ndim >= 3 and weight.shape[0] == num_experts:
        return weight[index]
    if weight.shape[0] % num_experts != 0:
        raise RuntimeError(
            f"Cannot split grouped weight with shape {tuple(weight.shape)} "
            f"into {num_experts} experts"
        )
    return weight.narrow(
        0, index * (weight.shape[0] // num_experts), weight.shape[0] // num_experts
    )


def _new_bf16_parameter(
    shape: Iterable[int], reference: torch.Tensor, *, allreduce: bool
) -> nn.Parameter:
    data = torch.empty(tuple(shape), dtype=torch.bfloat16, device=reference.device)
    param = nn.Parameter(data)
    _copy_parameter_attributes(param, reference, allreduce=allreduce)
    return param


def _dummy_weight_gradient(param: nn.Parameter) -> torch.Tensor:
    """Return a storage-free gradient sentinel used to trigger MCore DDP hooks.

    MOK has already accumulated the numerical gradient into ``main_grad``. The
    autograd return only needs the parameter's shape and dtype so that MCore's
    post-accumulate hook runs; the hook does not read it when
    ``grad_added_to_main_grad`` is true. A detached parameter view therefore
    avoids allocating a full-sized dummy gradient.
    """
    return param.detach()


def _main_grad_buffer(param: nn.Parameter) -> torch.Tensor:
    """Return and validate the optimizer-visible FP32 or BF16 gradient buffer."""
    main_grad = getattr(param, "main_grad", None)
    if main_grad is None:
        raise RuntimeError(
            "MOK gradient accumulation fusion requires DDP to assign param.main_grad"
        )
    if main_grad.shape != param.shape:
        raise RuntimeError(
            "MOK weight-gradient shape mismatch: "
            f"main_grad={tuple(main_grad.shape)}, param={tuple(param.shape)}"
        )
    if main_grad.dtype not in (torch.float32, torch.bfloat16):
        raise RuntimeError("MOK direct accumulation requires FP32 or BF16 main_grad")
    if not main_grad.is_contiguous():
        raise RuntimeError("MOK direct accumulation requires contiguous main_grad")
    if getattr(param, "zero_out_wgrad", False):
        raise RuntimeError("MOK does not support zero_out_wgrad parameters")
    if main_grad.device != param.device:
        raise RuntimeError("MOK main_grad must be on the parameter device")
    return main_grad


def _finish_weight_gradient(param: nn.Parameter) -> torch.Tensor:
    """Mark an in-kernel accumulation complete and return a DDP hook-only grad."""
    param.grad_added_to_main_grad = True
    return _dummy_weight_gradient(param)


def _storage_view(
    storage: torch.Tensor,
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
    name: str,
) -> torch.Tensor:
    """Return a zero-copy dense view over a TE grouped backing tensor."""
    if not storage.is_cuda or not storage.is_contiguous():
        raise RuntimeError(f"MOK {name} storage must be contiguous CUDA storage")
    if storage.dtype != dtype:
        if storage.dtype == torch.uint8 and dtype == torch.float8_e4m3fn:
            storage = storage.view(dtype)
        else:
            raise RuntimeError(
                f"MOK {name} storage has dtype {storage.dtype}, expected {dtype}"
            )
    expected_numel = 1
    for dim in shape:
        expected_numel *= dim
    if storage.numel() != expected_numel:
        raise RuntimeError(
            f"MOK {name} storage size mismatch: got {storage.numel()}, "
            f"expected {expected_numel} for {shape}"
        )
    return storage.view(shape)


def _grouped_mxfp8_scale_view(
    param: nn.Parameter,
    member_attr: str,
    shape: tuple[int, ...],
    *,
    name: str,
) -> torch.Tensor:
    """Expose all experts' contiguous TE MXFP8 scale storage through member zero."""
    from megatron.core.fp8_utils import get_grouped_quantized_members

    members = get_grouped_quantized_members(param)
    if not members:
        raise RuntimeError(f"MOK {name} grouped parameter has no quantized members")
    first = getattr(members[0], member_attr, None)
    if first is None or first.dtype != torch.uint8 or not first.is_contiguous():
        raise RuntimeError(
            f"MOK {name} requires contiguous uint8 TE member storage {member_attr}"
        )
    expected_numel = 1
    for dim in shape:
        expected_numel *= dim
    storage_numel = first.untyped_storage().nbytes() // first.element_size()
    available_numel = storage_numel - first.storage_offset()
    if available_numel < expected_numel:
        raise RuntimeError(
            f"MOK {name} scale storage is too small: available={available_numel}, "
            f"expected={expected_numel}"
        )
    flat = torch.as_strided(
        first,
        (expected_numel,),
        (1,),
        storage_offset=first.storage_offset(),
    )
    return flat.view(shape)


def _swizzle_mxfp8_scale(
    logical_scale: torch.Tensor,
    *,
    rows: int,
    columns: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Convert TE's logical E8M0 matrix to MOK's tcgen05 scale layout.

    TE stores rowwise scales as ``[E, M, K / 32]``. MOK consumes the
    tcgen05 1x scale-factor layout ``[E * M / 128, K / 128, 32, 16]``.
    Only the scale bytes are copied; the much larger FP8 payload stays in
    native TE storage.
    """
    if rows % 128 != 0 or columns % 128 != 0:
        raise RuntimeError("MOK MXFP8 scale dimensions must be divisible by 128")
    if logical_scale.dtype != torch.uint8 or logical_scale.ndim != 3:
        raise RuntimeError("MOK requires logical uint8 MXFP8 scales shaped [E, M, K/32]")
    num_experts = logical_scale.shape[0]
    expected_shape = (num_experts, rows, columns // 32)
    if tuple(logical_scale.shape) != expected_shape:
        raise RuntimeError(
            f"MOK logical scale shape mismatch: got {tuple(logical_scale.shape)}, "
            f"expected {expected_shape}"
        )
    source = (
        logical_scale.reshape(num_experts, rows // 128, 128, columns // 128, 4)
        .permute(0, 1, 3, 2, 4)
        .reshape(num_experts, rows // 128, columns // 128, 4, 32, 4)
        .transpose(-3, -2)
    )
    output_shape = (num_experts * rows // 128, columns // 128, 32, 16)
    if out is None:
        return source.reshape(output_shape).contiguous()
    if (
        tuple(out.shape) != output_shape
        or out.dtype != torch.uint8
        or out.device != logical_scale.device
        or not out.is_contiguous()
    ):
        raise RuntimeError(
            "MOK MXFP8 scale refresh output mismatch: "
            f"got shape={tuple(out.shape)}, dtype={out.dtype}, device={out.device}, "
            f"contiguous={out.is_contiguous()}; expected shape={output_shape}, "
            f"dtype=torch.uint8, device={logical_scale.device}, contiguous=True"
        )
    # Preserve the destination address referenced by MOK's TMA descriptor.
    # Viewing the final 16-byte lane as [4, 4] lets copy_ consume the strided
    # logical source directly without allocating a temporary CUDA tensor.
    out.view(num_experts, rows // 128, columns // 128, 32, 4, 4).copy_(source)
    return out


def _native_single_grouped_weight_views(
    fc1: nn.Parameter,
    fc2: nn.Parameter,
    *,
    num_experts: int,
    intermediate_size: int,
    hidden_size: int,
    use_mxfp8: bool,
):
    """Build MOK gate/up/down views directly over native TE grouped parameters."""
    e, i, h = num_experts, intermediate_size, hidden_size
    if tuple(fc1.shape) != (e, 2 * i, h) or tuple(fc2.shape) != (e, h, i):
        raise RuntimeError(
            "MOK requires native single-grouped FC1/FC2 shapes "
            f"{(e, 2 * i, h)} and {(e, h, i)}, got "
            f"{tuple(fc1.shape)} and {tuple(fc2.shape)}"
        )

    if not use_mxfp8:
        if fc1.dtype != torch.bfloat16 or fc2.dtype != torch.bfloat16:
            raise RuntimeError("MOK BF16 requires native BF16 grouped parameters")
        if not fc1.is_contiguous() or not fc2.is_contiguous():
            raise RuntimeError("MOK BF16 requires contiguous grouped parameters")
        # A high-precision TE GroupedTensor keeps its authoritative payload in
        # rowwise_data. Passing the wrapper itself through a custom op makes TE
        # materialize each argument independently with torch.stack(), which
        # both copies the weights and loses the gate/up pointer alias MOK uses
        # to recognize a combined [E, 2I, H] FC1 tensor. Expose the backing
        # storage directly so gate and up remain zero-copy aliases.
        fc1_storage = getattr(fc1, "rowwise_data", None)
        fc2_storage = getattr(fc2, "rowwise_data", None)
        if fc1_storage is not None or fc2_storage is not None:
            if fc1_storage is None or fc2_storage is None:
                raise RuntimeError(
                    "MOK BF16 requires both grouped parameters to expose rowwise_data"
                )
            fc1_view = _storage_view(
                fc1_storage, (e, 2 * i, h), dtype=torch.bfloat16, name="FC1 BF16 rowwise"
            )
            fc2_view = _storage_view(
                fc2_storage, (e, h, i), dtype=torch.bfloat16, name="FC2 BF16 rowwise"
            )
            return fc1_view, fc1_view, fc2_view
        return fc1, fc1, fc2

    from megatron.core.fp8_utils import is_grouped_mxfp8tensor

    if not is_grouped_mxfp8tensor(fc1) or not is_grouped_mxfp8tensor(fc2):
        raise RuntimeError("MOK MXFP8 requires native TE grouped MXFP8 parameters")

    fc1_row = _storage_view(
        fc1.rowwise_data, (e, 2 * i, h), dtype=torch.float8_e4m3fn, name="FC1 rowwise"
    )
    fc1_col = _storage_view(
        fc1.columnwise_data,
        (e, 2 * i, h),
        dtype=torch.float8_e4m3fn,
        name="FC1 columnwise",
    )
    fc2_row = _storage_view(
        fc2.rowwise_data, (e, h, i), dtype=torch.float8_e4m3fn, name="FC2 rowwise"
    )
    fc2_col = _storage_view(
        fc2.columnwise_data,
        (e, h, i),
        dtype=torch.float8_e4m3fn,
        name="FC2 columnwise",
    )
    # TE keeps E8M0 scales in logical order. Rowwise member storage is
    # [M, K/32]; columnwise member storage is [M/32, K] for the original
    # matrix, so transpose the latter into logical order for the transposed
    # FP8 payload. These are all zero-copy views.
    fc1_row_sc = _grouped_mxfp8_scale_view(
        fc1, "_rowwise_scale_inv", (e, 2 * i, h // 32), name="FC1 rowwise"
    )
    fc1_col_sc = _grouped_mxfp8_scale_view(
        fc1,
        "_columnwise_scale_inv",
        (e, 2 * i // 32, h),
        name="FC1 columnwise",
    ).transpose(-2, -1)
    fc2_row_sc = _grouped_mxfp8_scale_view(
        fc2, "_rowwise_scale_inv", (e, h, i // 32), name="FC2 rowwise"
    )
    fc2_col_sc = _grouped_mxfp8_scale_view(
        fc2,
        "_columnwise_scale_inv",
        (e, h // 32, i),
        name="FC2 columnwise",
    ).transpose(-2, -1)
    # The final flag tells MOK that ``columnwise_data`` is TE's native
    # columnwise-quantized storage in the original [E, M, K] tensor shape.
    # MOK can then consume it directly for dgrad instead of materializing an
    # explicit [E, K, M] transpose on every backward.
    fc1_views = (fc1_row, fc1_row_sc, fc1_col, fc1_col_sc, True)
    fc2_views = (fc2_row, fc2_row_sc, fc2_col, fc2_col_sc, True)
    return fc1_views, fc1_views, fc2_views



def _mok_mxfp8_backward_weight_views(
    native_weight: tuple[torch.Tensor, ...],
    *,
    rows: int,
    columns: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    """Prepare zero-copy payloads and compact scale layouts for forward/backward."""
    row_data, row_scale, column_data, column_scale, native_columnwise = native_weight
    if native_columnwise is not True:
        raise RuntimeError("MOK MCore integration requires native TE columnwise weights")
    return (
        row_data,
        _swizzle_mxfp8_scale(row_scale, rows=rows, columns=columns),
        column_data,
        _swizzle_mxfp8_scale(column_scale, rows=columns, columns=rows),
        True,
    )


def _parameter_storage_attr(param: nn.Parameter, name: str) -> torch.Tensor | None:
    """Read a TE storage attribute without materializing the logical parameter."""
    candidates = (name,) if name.startswith("_") else (name, f"_{name}")
    for candidate in candidates:
        value = getattr(param, candidate, None)
        if value is not None:
            return value
    data = getattr(param, "data", None)
    if data is not None:
        for candidate in candidates:
            value = getattr(data, candidate, None)
            if value is not None:
                return value
    return None


def _native_split_weight_view(
    params: tuple[nn.Parameter, ...],
    *,
    rows: int,
    columns: int,
    use_mxfp8: bool,
):
    """Expose independent per-expert parameters through MOK descriptor tables."""
    from mok import functional, ops

    if not params:
        raise RuntimeError("MOK split routed weight list must not be empty")
    shape = (rows, columns)
    if not use_mxfp8:
        payloads = []
        for param in params:
            storage = _parameter_storage_attr(param, "rowwise_data")
            payload = param if storage is None else storage
            payloads.append(
                _storage_view(
                    payload,
                    shape,
                    dtype=torch.bfloat16,
                    name="split BF16 rowwise",
                )
            )
        return functional.SplitRoutedWeight(
            data=payloads[0],
            storage_table=ops.make_routed_weight_storage_table_bf16(payloads),
        )

    from megatron.core.fp8_utils import is_mxfp8tensor

    row_payloads = []
    column_payloads = []
    row_scales = []
    column_scales = []
    for param in params:
        if not is_mxfp8tensor(param):
            raise RuntimeError(
                "MOK MXFP8 split weights require every per-expert parameter "
                "to use native TE MXFP8 storage"
            )
        row_payload = _parameter_storage_attr(param, "rowwise_data")
        column_payload = _parameter_storage_attr(param, "columnwise_data")
        row_scale = _parameter_storage_attr(param, "_rowwise_scale_inv")
        column_scale = _parameter_storage_attr(param, "_columnwise_scale_inv")
        if any(value is None for value in (row_payload, column_payload, row_scale, column_scale)):
            missing = [
                name
                for name, value in (
                    ("rowwise_data", row_payload),
                    ("columnwise_data", column_payload),
                    ("_rowwise_scale_inv", row_scale),
                    ("_columnwise_scale_inv", column_scale),
                )
                if value is None
            ]
            data = getattr(param, "data", None)
            quantizer = None
            get_quantizer = getattr(data, "_get_quantizer", None)
            if callable(get_quantizer):
                quantizer = get_quantizer()
            raise RuntimeError(
                "MOK MXFP8 split parameter is missing native storage: "
                f"missing={missing}, param_type={type(param).__name__}, "
                f"data_type={type(data).__name__}, quantizer_type="
                f"{type(quantizer).__name__ if quantizer is not None else None}, "
                f"param_shape={tuple(param.shape)}"
            )
        row_payloads.append(
            _storage_view(
                row_payload,
                shape,
                dtype=torch.float8_e4m3fn,
                name="split MXFP8 rowwise",
            )
        )
        column_payloads.append(
            _storage_view(
                column_payload,
                shape,
                dtype=torch.float8_e4m3fn,
                name="split MXFP8 columnwise",
            )
        )
        if tuple(row_scale.shape) != (rows, columns // 32):
            raise RuntimeError(
                "MOK split MXFP8 rowwise scale shape mismatch: "
                f"got {tuple(row_scale.shape)}, expected {(rows, columns // 32)}"
            )
        if tuple(column_scale.shape) != (rows // 32, columns):
            raise RuntimeError(
                "MOK split MXFP8 columnwise scale shape mismatch: "
                f"got {tuple(column_scale.shape)}, expected {(rows // 32, columns)}"
            )
        row_scales.append(row_scale)
        column_scales.append(column_scale.transpose(-2, -1))

    # Keep expert scales in independent allocations just like the native TE
    # parameters. MOK's descriptor table selects the expert; only each
    # expert's logical E8M0 matrix is converted to tcgen05 scale layout.
    row_scale_tensors = [
        _swizzle_mxfp8_scale(scale.unsqueeze(0), rows=rows, columns=columns)
        for scale in row_scales
    ]
    column_scale_tensors = [
        _swizzle_mxfp8_scale(scale.unsqueeze(0), rows=columns, columns=rows)
        for scale in column_scales
    ]
    row_storage_table = ops.make_routed_weight_storage_table_mxfp8(row_payloads)
    column_storage_table = ops.make_routed_weight_storage_table_mxfp8(column_payloads)
    return functional.SplitRoutedWeight(
        data=row_payloads[0],
        storage_table=row_storage_table,
        scale=row_scale_tensors[0],
        scale_storage_table=ops.make_routed_scale_storage_table(row_scale_tensors),
        scale_tensors=tuple(row_scale_tensors),
        transposed_data=column_payloads[0],
        transposed_scale=column_scale_tensors[0],
        transposed_storage_table=column_storage_table,
        transposed_scale_storage_table=ops.make_routed_scale_storage_table(
            column_scale_tensors
        ),
        transposed_scale_tensors=tuple(column_scale_tensors),
        native_columnwise=True,
    )


def _refresh_native_split_weight_scales(
    prepared,
    params: tuple[nn.Parameter, ...],
    *,
    rows: int,
    columns: int,
) -> None:
    """Refresh split MXFP8 scales without replacing graph-captured descriptors."""
    row_outputs = prepared.scale_tensors
    column_outputs = prepared.transposed_scale_tensors
    if row_outputs is None or column_outputs is None:
        raise RuntimeError("MOK split MXFP8 cache is missing retained scale tensors")
    if len(row_outputs) != len(params) or len(column_outputs) != len(params):
        raise RuntimeError(
            "MOK split MXFP8 scale cache expert count mismatch: "
            f"parameters={len(params)}, row_outputs={len(row_outputs)}, "
            f"column_outputs={len(column_outputs)}"
        )

    for expert, (param, row_out, column_out) in enumerate(
        zip(params, row_outputs, column_outputs, strict=True)
    ):
        row_scale = _parameter_storage_attr(param, "_rowwise_scale_inv")
        column_scale = _parameter_storage_attr(param, "_columnwise_scale_inv")
        if row_scale is None or column_scale is None:
            raise RuntimeError(
                f"MOK split MXFP8 expert {expert} lost its native TE scale storage"
            )
        if tuple(row_scale.shape) != (rows, columns // 32):
            raise RuntimeError(
                "MOK split MXFP8 rowwise scale shape changed after cache creation: "
                f"expert={expert}, got={tuple(row_scale.shape)}, "
                f"expected={(rows, columns // 32)}"
            )
        if tuple(column_scale.shape) != (rows // 32, columns):
            raise RuntimeError(
                "MOK split MXFP8 columnwise scale shape changed after cache creation: "
                f"expert={expert}, got={tuple(column_scale.shape)}, "
                f"expected={(rows // 32, columns)}"
            )
        _swizzle_mxfp8_scale(
            row_scale.unsqueeze(0), rows=rows, columns=columns, out=row_out
        )
        _swizzle_mxfp8_scale(
            column_scale.transpose(-2, -1).unsqueeze(0),
            rows=columns,
            columns=rows,
            out=column_out,
        )


class _MoKAutograd(torch.autograd.Function):
    """Autograd bridge from MCore parameters to MoK's functional API."""

    @staticmethod
    def forward(
        ctx,
        module: "MoKMegakernel",
        x: torch.Tensor,
        router_weights: torch.Tensor,
        top_experts: torch.Tensor,
        *parameters: torch.Tensor,
    ) -> torch.Tensor:
        from mok import functional

        num_routed_parameters = len(module.autograd_routed_parameters)
        if len(parameters) != num_routed_parameters + 2:
            raise RuntimeError(
                "MOK autograd parameter count mismatch: "
                f"got {len(parameters)}, expected {num_routed_parameters + 2}"
            )
        routed_parameters = parameters[:num_routed_parameters]
        shared_fc1, shared_down = parameters[num_routed_parameters:]
        shared_gate, shared_up, shared_down = module.shared_weight_views(
            shared_fc1, shared_down
        )

        workspace = functional.get_workspace(
            module.mok_config,
            module.ep_group,
            device=x.device,
            num_local_tokens=x.shape[0],
            hidden_size=x.shape[1],
            topk=top_experts.shape[1],
        )
        schedule = functional.build_schedule(
            workspace, module.mok_config, top_experts, num_local_experts=module.num_local_experts
        )
        trace_param_lifecycle = module.is_first_microbatch
        routed_debug_param = module.routed_debug_parameter
        if trace_param_lifecycle:
            _debug_record(
                "forward.after_param_sync_before_quantize",
                routed_debug_param,
                tensors={"main_grad": getattr(routed_debug_param, "main_grad", None)},
            )
        prepared_gate, prepared_up, prepared_down = module.quantized_routed_weights()
        if module.use_mxfp8_weights and module.native_single_grouped_weights:
            gate_forward = prepared_gate[:2]
            up_forward = prepared_up[:2]
            down_forward = prepared_down[:2]
        else:
            gate_forward = prepared_gate
            up_forward = prepared_up
            down_forward = prepared_down
        if trace_param_lifecycle and module.use_mxfp8_weights:
            if module.native_single_grouped_weights:
                debug_tensors = {
                    "rowwise_data": prepared_gate[0],
                    "rowwise_scale": prepared_gate[1],
                    "columnwise_data": prepared_gate[2],
                    "columnwise_scale": prepared_gate[3],
                    "actual_forward_data": gate_forward[0],
                    "actual_forward_scale": gate_forward[1],
                }
            else:
                debug_tensors = {
                    "rowwise_data": prepared_gate.data,
                    "rowwise_scale": prepared_gate.scale,
                    "columnwise_data": prepared_gate.transposed_data,
                    "columnwise_scale": prepared_gate.transposed_scale,
                    "weight_descriptor_table": prepared_gate.storage_table,
                }
            _debug_record(
                "forward.mok_quantized_weight_cache",
                routed_debug_param,
                tensors=debug_tensors,
            )
        output, forward_context = functional.forward(
            module.mok_config,
            workspace,
            schedule,
            x,
            router_weights,
            shared_gate,
            shared_up,
            shared_down,
            gate_forward,
            up_forward,
            down_forward,
            swiglu_limit=module.swiglu_limit,
        )

        ctx.module = module
        ctx.workspace = workspace
        ctx.schedule = schedule
        ctx.forward_context = forward_context
        ctx.quantized_weights = (prepared_gate, prepared_up, prepared_down)
        ctx.trace_param_lifecycle = trace_param_lifecycle
        ctx.save_for_backward(
            x,
            router_weights,
            *routed_parameters,
            shared_fc1,
            shared_down,
        )
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        from mok import functional

        x, router_weights, *parameters = ctx.saved_tensors
        num_routed_parameters = len(ctx.module.autograd_routed_parameters)
        shared_fc1, shared_down = parameters[num_routed_parameters:]
        shared_gate, shared_up, shared_down = ctx.module.shared_weight_views(
            shared_fc1, shared_down
        )
        prepared_gate, prepared_up, prepared_down = ctx.quantized_weights
        if ctx.module.use_mxfp8_weights and ctx.module.native_single_grouped_weights:
            backward_gate = prepared_gate
            backward_up = prepared_up
            backward_down = prepared_down[2:]
        else:
            backward_gate = prepared_gate
            backward_up = prepared_up
            backward_down = prepared_down
        direct_wgrad_accumulation = ctx.module.fuse_wgrad_accumulation
        main_grads = None
        main_grad_storage_tables = None
        if direct_wgrad_accumulation:
            main_grads, main_grad_storage_tables = ctx.module.main_grad_arguments()
        if ctx.trace_param_lifecycle:
            routed_debug_param = ctx.module.routed_debug_parameter
            backward_debug_tensors = {
                "main_grad_before": getattr(routed_debug_param, "main_grad", None)
            }
            if isinstance(backward_gate, tuple):
                backward_debug_tensors.update(
                    {
                        "actual_backward_rowwise_data": backward_gate[0],
                        "actual_backward_rowwise_scale": backward_gate[1],
                        "actual_backward_columnwise_data": backward_gate[2],
                        "actual_backward_columnwise_scale": backward_gate[3],
                    }
                )
            elif ctx.module.use_mxfp8_weights:
                backward_debug_tensors.update(
                    {
                        "actual_backward_rowwise_data": backward_gate.data,
                        "actual_backward_rowwise_scale": backward_gate.scale,
                        "actual_backward_columnwise_data": backward_gate.transposed_data,
                        "actual_backward_columnwise_scale": backward_gate.transposed_scale,
                    }
                )
            _debug_record(
                "backward.before_mok_kernel",
                routed_debug_param,
                tensors=backward_debug_tensors,
            )
        (
            d_x,
            d_router_weights,
            d_routed_gate,
            d_routed_up,
            d_routed_down,
            d_shared_gate,
            d_shared_up,
            d_shared_down,
        ) = functional.backward(
            ctx.module.mok_config,
            ctx.workspace,
            ctx.schedule,
            ctx.forward_context,
            grad_output.contiguous(),
            x,
            router_weights,
            shared_gate,
            shared_up,
            shared_down,
            backward_gate,
            backward_up,
            backward_down,
            swiglu_limit=ctx.module.swiglu_limit,
            main_grads=main_grads,
            main_grad_storage_tables=main_grad_storage_tables,
        )

        if ctx.trace_param_lifecycle:
            routed_debug_param = ctx.module.routed_debug_parameter
            _debug_record(
                "backward.after_mok_kernel",
                routed_debug_param,
                tensors={
                    "main_grad_after": getattr(routed_debug_param, "main_grad", None)
                },
            )
        if ctx.module.fuse_wgrad_accumulation:
            routed_parameter_grads = ctx.module.finish_routed_weight_gradients()
            d_shared_fc1 = _finish_weight_gradient(ctx.module.shared_fc1_weight)
            d_shared_down = _finish_weight_gradient(ctx.module.shared_down_weight)
        else:
            # Materialized routed gradients are only supported by the original
            # dense/single-grouped interface.
            routed_parameter_grads = (d_routed_gate, d_routed_down)
            d_shared_fc1 = torch.cat((d_shared_gate, d_shared_up), dim=0)

        ctx.module = None
        ctx.workspace = None
        ctx.schedule = None
        ctx.forward_context = None
        ctx.quantized_weights = None
        return (
            None,
            d_x,
            d_router_weights,
            None,
            *routed_parameter_grads,
            d_shared_fc1,
            d_shared_down,
        )


class MoKMegakernel(nn.Module):
    """Execute MOK using trainable parameters owned by native MCore modules."""

    def __init__(
        self,
        config,
        ep_group,
        routed_experts: nn.Module,
        shared_experts: nn.Module,
        num_local_experts: int,
    ) -> None:
        super().__init__()
        try:
            from mok.functional import MoKConfig
        except ImportError as exc:
            raise ImportError(
                "--use-mok-megakernel requires the latest mixture-of-kittens package "
                "on PYTHONPATH"
            ) from exc

        if not config.gradient_accumulation_fusion:
            raise ValueError(
                "MOK native routed weights require gradient_accumulation_fusion=True"
            )
        if config.moe_mlp_glu_interleave_size is not None:
            raise ValueError("MOK requires non-interleaved native MCore routed FC1 weights")
        if config.moe_shared_expert_glu_interleave_size is not None:
            raise ValueError("MOK requires non-interleaved native shared FC1 weights")
        if config.moe_pad_expert_input_to_capacity:
            raise ValueError(
                "MOK supports at most moe_router_topk logical routes per token; "
                "use MOK internal expert padding instead of moe_pad_expert_input_to_capacity"
            )
        if config.moe_shared_expert_gate:
            raise ValueError("MOK does not support MCore's optional shared-expert output gate")

        self.ep_group = ep_group
        self.num_local_experts = num_local_experts
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.moe_ffn_hidden_size
        shared_intermediate_size = config.moe_shared_expert_intermediate_size
        if shared_intermediate_size != self.intermediate_size:
            raise ValueError(
                "MOK requires routed and shared experts to use the same intermediate "
                f"size, got routed={self.intermediate_size}, "
                f"shared={shared_intermediate_size}"
            )
        self.topk = config.moe_router_topk
        self.swiglu_limit = config.activation_func_clamp_value
        self.use_mxfp8_weights = config.mok_use_mxfp8_weights
        self.source_mlp_glu_interleave_size = config.mok_source_mlp_glu_interleave_size
        self.fuse_wgrad_accumulation = config.gradient_accumulation_fusion
        self.native_single_grouped_weights = bool(config.moe_single_grouped_weight)
        self._debug_module_index = next(_MOK_MODULE_INDICES)
        self.mok_config = MoKConfig(
            fwd_num_comm_sms=config.mok_fwd_num_comm_sms,
            bwd_num_comm_sms=config.mok_bwd_num_comm_sms,
            minibatch_size=config.mok_minibatch_size,
            macrobatch_size=config.mok_macrobatch_size,
            schedule_capacity_multiplier=config.mok_schedule_capacity_multiplier,
            all_gather_top_experts_chunk_bytes=config.mok_all_gather_top_experts_chunk_bytes,
            scale_router_before_fc2=config.mok_scale_router_before_fc2,
        )

        fc1 = routed_experts.linear_fc1
        fc2 = routed_experts.linear_fc2
        actual_single_grouped = bool(getattr(fc1, "single_grouped_weight", False))
        if actual_single_grouped != bool(getattr(fc2, "single_grouped_weight", False)):
            raise ValueError("MOK requires routed FC1 and FC2 to use the same weight layout")
        if actual_single_grouped != self.native_single_grouped_weights:
            raise ValueError(
                "MOK routed weight layout does not match config.moe_single_grouped_weight"
            )

        if self.native_single_grouped_weights:
            _convert_native_fc1_init_to_mok_layout(
                fc1.weight,
                intermediate_size=self.intermediate_size,
                interleave_size=self.source_mlp_glu_interleave_size,
            )
            # Register aliases of the same Parameter objects so DDP's MOK forward
            # pre-hook waits for their overlap-param-gather buckets. named_parameters
            # deduplicates them; no additional payload storage is allocated.
            self.register_parameter("routed_fc1_weight", fc1.weight)
            self.register_parameter("routed_down_weight", fc2.weight)
            _debug_tag(
                self.routed_fc1_weight,
                f"module{self._debug_module_index}.routed_fc1_weight",
            )
            _debug_tag(
                self.routed_down_weight,
                f"module{self._debug_module_index}.routed_down_weight",
            )
        else:
            # Keep each native MCore expert parameter authoritative. MOK selects
            # its per-expert TMA descriptor inside the grouped-GEMM task, so no
            # dense expert-major payload or duplicate optimizer parameter is
            # required. The aliases let MOK participate in parameter-gather hooks;
            # the source expert module remains registered as the canonical owner.
            self._routed_fc1_parameter_names = []
            self._routed_down_parameter_names = []
            for expert_idx in range(self.num_local_experts):
                fc1_param = _indexed_grouped_weight(
                    fc1, expert_idx, self.num_local_experts
                )
                down_param = _indexed_grouped_weight(
                    fc2, expert_idx, self.num_local_experts
                )
                if not isinstance(fc1_param, nn.Parameter) or not isinstance(
                    down_param, nn.Parameter
                ):
                    raise RuntimeError(
                        "MOK non-single integration requires registered per-expert Parameters"
                    )
                _convert_native_fc1_init_to_mok_layout(
                    fc1_param,
                    intermediate_size=self.intermediate_size,
                    interleave_size=self.source_mlp_glu_interleave_size,
                )
                fc1_name = f"routed_fc1_weight{expert_idx}"
                down_name = f"routed_down_weight{expert_idx}"
                self.register_parameter(fc1_name, fc1_param)
                self.register_parameter(down_name, down_param)
                self._routed_fc1_parameter_names.append(fc1_name)
                self._routed_down_parameter_names.append(down_name)
                _debug_tag(
                    fc1_param,
                    f"module{self._debug_module_index}.{fc1_name}",
                )
                _debug_tag(
                    down_param,
                    f"module{self._debug_module_index}.{down_name}",
                )
            self._routed_fc1_parameter_names = tuple(
                self._routed_fc1_parameter_names
            )
            self._routed_down_parameter_names = tuple(
                self._routed_down_parameter_names
            )

        self._adopt_shared_weights(shared_experts)

        # MegatronModule.set_is_first_microbatch discovers this attribute and resets it
        # once per optimizer iteration, matching TE's weight-cache lifecycle.
        self.is_first_microbatch = True
        self._prepared_routed_weight_cache = None
        self._split_main_grad_descriptor_cache = None

    @property
    def routed_fc1_parameters(self) -> tuple[nn.Parameter, ...]:
        if self.native_single_grouped_weights:
            return (self.routed_fc1_weight,)
        return tuple(
            getattr(self, name) for name in self._routed_fc1_parameter_names
        )

    @property
    def routed_down_parameters(self) -> tuple[nn.Parameter, ...]:
        if self.native_single_grouped_weights:
            return (self.routed_down_weight,)
        return tuple(
            getattr(self, name) for name in self._routed_down_parameter_names
        )

    @property
    def autograd_routed_parameters(self) -> tuple[nn.Parameter, ...]:
        return self.routed_fc1_parameters + self.routed_down_parameters

    @property
    def routed_debug_parameter(self) -> nn.Parameter:
        return self.routed_fc1_parameters[0]

    def shared_weight_views(
        self,
        fc1: torch.Tensor | None = None,
        down: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split native combined FC1 into zero-copy gate/up views for MOK."""
        fc1 = self.shared_fc1_weight if fc1 is None else fc1
        down = self.shared_down_weight if down is None else down
        i, h = self.intermediate_size, self.hidden_size
        if tuple(fc1.shape) != (2 * i, h) or tuple(down.shape) != (h, i):
            raise RuntimeError(
                "MOK shared weight shape mismatch: expected "
                f"{(2 * i, h)} and {(h, i)}, got "
                f"{tuple(fc1.shape)} and {tuple(down.shape)}"
            )
        if not fc1.is_contiguous() or not down.is_contiguous():
            raise RuntimeError("MOK shared weights must use contiguous native storage")
        return fc1.narrow(0, 0, i), fc1.narrow(0, i, i), down

    def main_grad_arguments(self):
        """Return MOK logical main grads and optional per-expert descriptors."""
        shared_fc1_grad = _main_grad_buffer(self.shared_fc1_weight)
        shared_gate_grad, shared_up_grad, shared_down_grad = self.shared_weight_views(
            shared_fc1_grad, _main_grad_buffer(self.shared_down_weight)
        )
        fc1_main_grads = tuple(
            _main_grad_buffer(param) for param in self.routed_fc1_parameters
        )
        down_main_grads = tuple(
            _main_grad_buffer(param) for param in self.routed_down_parameters
        )
        main_grads = (
            shared_gate_grad,
            fc1_main_grads[0],
            shared_up_grad,
            fc1_main_grads[0],
            shared_down_grad,
            down_main_grads[0],
        )
        if self.native_single_grouped_weights:
            return main_grads, None

        from mok import ops

        fingerprint = tuple(
            (grad.data_ptr(), grad.dtype, tuple(grad.shape))
            for grad in fc1_main_grads + down_main_grads
        )
        if (
            self._split_main_grad_descriptor_cache is None
            or self._split_main_grad_descriptor_cache[0] != fingerprint
        ):
            fc1_table = ops.make_routed_d_weight_storage_table(
                list(fc1_main_grads)
            )
            down_table = ops.make_routed_d_weight_storage_table(
                list(down_main_grads)
            )
            self._split_main_grad_descriptor_cache = (
                fingerprint,
                (fc1_table, fc1_table, down_table),
            )
        return main_grads, self._split_main_grad_descriptor_cache[1]

    def finish_routed_weight_gradients(self) -> tuple[torch.Tensor, ...]:
        return tuple(
            _finish_weight_gradient(param)
            for param in self.autograd_routed_parameters
        )

    @torch.no_grad()
    def _adopt_shared_weights(self, shared: nn.Module) -> None:
        """Keep native BF16 shared weights and expose zero-copy MOK aliases.

        The routed path may consume MXFP8 parameters directly. The initial MOK
        integration deliberately keeps the shared expert in BF16, so a TE FP8
        parameter created from the global model configuration is converted once
        during construction and replaced in the native shared-expert module.
        """
        fc1_ref = shared.linear_fc1.weight
        down_ref = shared.linear_fc2.weight
        i, h = self.intermediate_size, self.hidden_size
        if tuple(fc1_ref.shape) != (2 * i, h) or tuple(down_ref.shape) != (h, i):
            raise RuntimeError(
                "MOK requires native combined shared FC1/FC2 shapes "
                f"{(2 * i, h)} and {(h, i)}, got "
                f"{tuple(fc1_ref.shape)} and {tuple(down_ref.shape)}"
            )

        from megatron.core.fp8_utils import is_float8tensor

        def native_bf16_parameter(
            param: nn.Parameter, shape: tuple[int, ...], name: str
        ) -> nn.Parameter:
            if not isinstance(param, nn.Parameter):
                raise RuntimeError(f"MOK shared {name} must be an nn.Parameter")
            if (
                not is_float8tensor(param)
                and param.dtype == torch.bfloat16
                and param.is_contiguous()
            ):
                return param

            source, has_preserved_init = _materialize_parameter_init(param)
            source = source.reshape(shape)
            adopted = _new_bf16_parameter(shape, param, allreduce=True)
            adopted.copy_(source.to(torch.bfloat16))
            if has_preserved_init:
                _attach_high_precision_init(adopted, source)
            return adopted

        shared_fc1 = native_bf16_parameter(fc1_ref, (2 * i, h), "FC1")
        shared_down = native_bf16_parameter(down_ref, (h, i), "FC2")
        if shared_fc1 is not fc1_ref:
            shared.linear_fc1.weight = shared_fc1
        if shared_down is not down_ref:
            shared.linear_fc2.weight = shared_down

        # These are aliases, not extra ownership. The native shared_experts
        # module remains registered and emits the canonical checkpoint entries.
        self.register_parameter("shared_fc1_weight", shared_fc1)
        self.register_parameter("shared_down_weight", shared_down)
        _debug_tag(
            self.shared_fc1_weight,
            f"module{self._debug_module_index}.shared_fc1_weight",
        )
        _debug_tag(
            self.shared_down_weight,
            f"module{self._debug_module_index}.shared_down_weight",
        )

    @torch.no_grad()
    def quantized_routed_weights(self):
        """Prepare routed weights without copying their FP8/BF16 payloads."""
        if not self.native_single_grouped_weights:
            if self._prepared_routed_weight_cache is None:
                prepared_fc1 = _native_split_weight_view(
                    self.routed_fc1_parameters,
                    rows=2 * self.intermediate_size,
                    columns=self.hidden_size,
                    use_mxfp8=self.use_mxfp8_weights,
                )
                prepared_down = _native_split_weight_view(
                    self.routed_down_parameters,
                    rows=self.hidden_size,
                    columns=self.intermediate_size,
                    use_mxfp8=self.use_mxfp8_weights,
                )
                self._prepared_routed_weight_cache = (
                    prepared_fc1,
                    prepared_fc1,
                    prepared_down,
                )
            elif self.use_mxfp8_weights and self.is_first_microbatch:
                prepared_fc1, _, prepared_down = self._prepared_routed_weight_cache
                _refresh_native_split_weight_scales(
                    prepared_fc1,
                    self.routed_fc1_parameters,
                    rows=2 * self.intermediate_size,
                    columns=self.hidden_size,
                )
                _refresh_native_split_weight_scales(
                    prepared_down,
                    self.routed_down_parameters,
                    rows=self.hidden_size,
                    columns=self.intermediate_size,
                )
            self.is_first_microbatch = False
            return self._prepared_routed_weight_cache

        if not self.use_mxfp8_weights:
            self.is_first_microbatch = False
            return _native_single_grouped_weight_views(
                self.routed_fc1_weight,
                self.routed_down_weight,
                num_experts=self.num_local_experts,
                intermediate_size=self.intermediate_size,
                hidden_size=self.hidden_size,
                use_mxfp8=False,
            )

        if self._prepared_routed_weight_cache is None or self.is_first_microbatch:
            native_gate, _, native_down = _native_single_grouped_weight_views(
                self.routed_fc1_weight,
                self.routed_down_weight,
                num_experts=self.num_local_experts,
                intermediate_size=self.intermediate_size,
                hidden_size=self.hidden_size,
                use_mxfp8=True,
            )
            prepared_gate = _mok_mxfp8_backward_weight_views(
                native_gate,
                rows=2 * self.intermediate_size,
                columns=self.hidden_size,
            )
            prepared_down = _mok_mxfp8_backward_weight_views(
                native_down,
                rows=self.hidden_size,
                columns=self.intermediate_size,
            )
            # Only the compact scale layouts allocate storage. FP8 row/column
            # payloads remain zero-copy views of the current TE gather buffer.
            self._prepared_routed_weight_cache = (
                prepared_gate,
                prepared_gate,
                prepared_down,
            )

        self.is_first_microbatch = False
        return self._prepared_routed_weight_cache

    def sharded_state_dict(
        self, prefix="", sharded_offsets=(), metadata=None
    ):
        """Emit no aliases; native expert modules own all checkpoint shards."""
        del prefix, sharded_offsets, metadata
        return {}

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        """Emit no aliases in regular state dicts either."""
        del destination, prefix, keep_vars

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """Ignore legacy aliases and invalidate state derived from native weights."""
        del state_dict, prefix, local_metadata, strict
        del missing_keys, unexpected_keys, error_msgs
        self._prepared_routed_weight_cache = None
        self._split_main_grad_descriptor_cache = None
        self.is_first_microbatch = True

    def forward(
        self, hidden_states: torch.Tensor, probs: torch.Tensor, routing_map: torch.Tensor
    ) -> torch.Tensor:
        original_shape = hidden_states.shape
        x = hidden_states.reshape(-1, original_shape[-1]).contiguous()
        probs = probs.reshape(x.shape[0], -1)
        routing_map = routing_map.reshape(x.shape[0], -1)

        # Compact the authoritative route set directly into MOK's fixed [tokens, K]
        # representation. Missing routes are encoded as expert -1 with zero weight.
        router_weights, top_experts = fused_routing_map_to_indices(
            probs, routing_map, self.topk
        )

        output = _MoKAutograd.apply(
            self,
            x,
            router_weights,
            top_experts,
            *self.autograd_routed_parameters,
            self.shared_fc1_weight,
            self.shared_down_weight,
        )
        return output.view(original_shape)
