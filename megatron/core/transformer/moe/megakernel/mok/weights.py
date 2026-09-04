# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""MOK-specific adaptation of MCore-owned expert parameters."""

from __future__ import annotations

import copy
import math
import warnings
from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from megatron.core.transformer.transformer_config import TransformerConfig

_SHARED_EXPERT_BF16_WARNING_EMITTED = False


def prepare_shared_expert_config(config: TransformerConfig) -> TransformerConfig:
    """Construct MOK shared experts as native BF16 MCore modules.

    Routed experts continue to use the original model configuration. The
    shared-only copy expresses the BF16 module configuration; the factory also
    disables any enclosing TE FP8 model-init context during construction.
    """
    if not config.fp8_param:
        return config

    global _SHARED_EXPERT_BF16_WARNING_EMITTED
    if not _SHARED_EXPERT_BF16_WARNING_EMITTED and (
        not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
    ):
        warnings.warn(
            "MOK currently computes shared experts in BF16. Shared-expert modules "
            "will therefore be constructed with FP8 execution and FP8 parameters "
            "disabled while routed "
            "experts retain the model's MXFP8 parameter configuration.",
            stacklevel=3,
        )
        _SHARED_EXPERT_BF16_WARNING_EMITTED = True

    shared_config = copy.copy(config)
    shared_config.fp8 = None
    shared_config.fp8_param = False
    return shared_config


def _storage_view(
    storage: torch.Tensor, shape: tuple[int, ...], *, dtype: torch.dtype, name: str
) -> torch.Tensor:
    """Return a zero-copy dense view over a TE grouped backing tensor."""
    if not storage.is_cuda or not storage.is_contiguous():
        raise RuntimeError(f"MOK {name} storage must be contiguous CUDA storage")
    if storage.dtype != dtype:
        if storage.dtype == torch.uint8 and dtype == torch.float8_e4m3fn:
            storage = storage.view(dtype)
        else:
            raise RuntimeError(f"MOK {name} storage has dtype {storage.dtype}, expected {dtype}")
    expected_numel = 1
    for dim in shape:
        expected_numel *= dim
    if storage.numel() != expected_numel:
        raise RuntimeError(
            f"MOK {name} storage size mismatch: got {storage.numel()}, "
            f"expected {expected_numel} for {shape}"
        )
    return storage.view(shape)


def _single_grouped_mxfp8_scale_view(
    param: nn.Parameter, member_attr: str, shape: tuple[int, ...], *, name: str
) -> torch.Tensor:
    """Expose one single-grouped parameter's expert-major MXFP8 scale storage."""
    from megatron.core.fp8_utils import get_grouped_quantized_members

    # One TE GroupedTensor is one PyTorch Parameter, but TE exposes one quantized
    # member view per expert; their scale views share expert-major backing storage.
    members = get_grouped_quantized_members(param)
    if not members or len(members) != shape[0]:
        raise RuntimeError(
            f"MOK {name} expected {shape[0]} grouped members, got {len(members)}"
        )

    first = getattr(members[0], member_attr)
    expected_numel = math.prod(shape)
    per_expert = expected_numel // shape[0]
    storage_ptr = first.untyped_storage().data_ptr()
    for expert, member in enumerate(members):
        scale = getattr(member, member_attr)
        if (
            tuple(scale.shape) != tuple(shape[1:])
            or not scale.is_contiguous()
            or scale.untyped_storage().data_ptr() != storage_ptr
            or scale.storage_offset() != first.storage_offset() + expert * per_expert
        ):
            raise RuntimeError(f"MOK {name} member {expert} scale is not packed expert-major")

    flat = torch.as_strided(
        first,
        (expected_numel,),
        (1,),
        storage_offset=first.storage_offset(),
    )
    return flat.view(shape)


def _swizzle_mxfp8_scale(
    logical_scale: torch.Tensor, *, rows: int, columns: int, out: torch.Tensor | None = None
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
        # Reshape(E, M_block=M/128, 128, K_block=K/128, 4)
        logical_scale.reshape(num_experts, rows // 128, 128, columns // 128, 4)
        # -> [E, M_block, K_block, 128_rows, 4_scales]
        .permute(0, 1, 3, 2, 4)
        # -> [E, M_block, K_block, 4, 32, 4]
        .reshape(num_experts, rows // 128, columns // 128, 4, 32, 4)
        # -> [E, M_block, K_block, 32, 4, 4]
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


def _native_single_grouped_weight_view(
    weight: nn.Parameter,
    *,
    num_experts: int,
    rows: int,
    columns: int,
    use_mxfp8: bool,
    cached_view=None,
):
    """Build or refresh one MOK view over a native TE single-grouped parameter."""
    shape = (num_experts, rows, columns)
    if tuple(weight.shape) != shape:
        raise RuntimeError(
            f"MOK requires native single-grouped weight shape {shape}, "
            f"got {tuple(weight.shape)}"
        )

    if not use_mxfp8:
        if weight.dtype != torch.bfloat16:
            raise RuntimeError("MOK BF16 requires native BF16 grouped parameters")
        if not weight.is_contiguous():
            raise RuntimeError("MOK BF16 requires contiguous grouped parameters")
        # A high-precision TE GroupedTensor stores its authoritative payload in
        # rowwise_data. Expose that backing storage so a custom op does not
        # materialize the logical wrapper with torch.stack().
        storage = getattr(weight, "rowwise_data", None)
        if storage is not None:
            return _storage_view(
                storage, shape, dtype=torch.bfloat16, name="single-grouped BF16 rowwise"
            )
        return weight

    # Single-grouped MXFP8
    from megatron.core.fp8_utils import is_grouped_mxfp8tensor

    if not is_grouped_mxfp8tensor(weight):
        raise RuntimeError("MOK MXFP8 requires native TE grouped MXFP8 parameters")

    row_data = _storage_view(
        weight.rowwise_data,
        shape,
        dtype=torch.float8_e4m3fn,
        name="single-grouped MXFP8 rowwise",
    )
    column_data = _storage_view(
        weight.columnwise_data,
        shape,
        dtype=torch.float8_e4m3fn,
        name="single-grouped MXFP8 columnwise",
    )
    row_scale = _single_grouped_mxfp8_scale_view(
        weight,
        "_rowwise_scale_inv",
        (num_experts, rows, columns // 32),
        name="single-grouped rowwise",
    )
    column_scale = _single_grouped_mxfp8_scale_view(
        weight,
        "_columnwise_scale_inv",
        (num_experts, rows // 32, columns),
        name="single-grouped columnwise",
    ).transpose(-2, -1)  # Logical transpose [M/32, K] -> [K, M/32] for MOK dgrad.

    # Refresh converted scales in place so graph-captured addresses remain
    # stable; payloads stay zero-copy views of the current TE storage.
    return (
        row_data,
        _swizzle_mxfp8_scale(
            row_scale,
            rows=rows,
            columns=columns,
            out=None if cached_view is None else cached_view[1],
        ),
        column_data,
        _swizzle_mxfp8_scale(
            column_scale,
            rows=columns,
            columns=rows,
            out=None if cached_view is None else cached_view[3],
        ),
        # TE native columnwise payload is in the original [E, M, K] shape.
        True,
    )


def _parameter_storage_attr(param: nn.Parameter, name: str) -> torch.Tensor | None:
    """Read a TE storage attribute without materializing the logical parameter.

    TE releases/wrappers have exposed native storage either directly on the
    Parameter or on param.data, using public or private attribute names.
    """
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
    cached_view=None,
):
    """Build or refresh one MOK view over independent per-expert parameters."""
    from mok import functional, ops

    if not params:
        raise RuntimeError("MOK split routed weight list must not be empty")
    if cached_view is not None:
        if not use_mxfp8:
            raise RuntimeError("MOK BF16 split weights do not require refresh")
        row_outputs = cached_view.scale_tensors
        column_outputs = cached_view.transposed_scale_tensors
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
            # MOK dgrad views [M, K] as [K, M], so its logical scale shape is [K, M/32].
            _swizzle_mxfp8_scale(
                column_scale.transpose(-2, -1).unsqueeze(0),
                rows=columns,
                columns=rows,
                out=column_out,
            )
        return cached_view

    shape = (rows, columns)
    if not use_mxfp8:
        payloads = []
        for param in params:
            storage = _parameter_storage_attr(param, "rowwise_data")
            payload = param if storage is None else storage
            payloads.append(
                _storage_view(payload, shape, dtype=torch.bfloat16, name="split BF16 rowwise")
            )
        return functional.SplitRoutedWeight(
            data=payloads[0],  # Representative tensor for schema/device dispatch.
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
            _storage_view(row_payload, shape, dtype=torch.float8_e4m3fn, name="split MXFP8 rowwise")
        )
        column_payloads.append(
            _storage_view(
                column_payload, shape, dtype=torch.float8_e4m3fn, name="split MXFP8 columnwise"
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
        # MOK dgrad views [M, K] as [K, M], so its logical scale shape is [K, M/32].
        column_scales.append(column_scale.transpose(-2, -1))

    # Keep expert scales in independent allocations just like the native TE
    # parameters. MOK's descriptor table selects the expert; only each
    # expert's logical E8M0 matrix is converted to tcgen05 scale layout.
    # TODO: Evaluate a batched GPU swizzle (e.g. Triton) that consumes all
    # independent expert scale pointers in one launch.
    row_scale_tensors = [
        _swizzle_mxfp8_scale(scale.unsqueeze(0), rows=rows, columns=columns) for scale in row_scales
    ]
    column_scale_tensors = [
        _swizzle_mxfp8_scale(scale.unsqueeze(0), rows=columns, columns=rows)
        for scale in column_scales
    ]
    row_storage_table = ops.make_routed_weight_storage_table_mxfp8(row_payloads)
    column_storage_table = ops.make_routed_weight_storage_table_mxfp8(column_payloads)
    return functional.SplitRoutedWeight(
        data=row_payloads[0],  # Representative rowwise payload tensor.
        storage_table=row_storage_table,
        scale=row_scale_tensors[0],  # Representative rowwise scale tensor.
        scale_storage_table=ops.make_routed_scale_storage_table(row_scale_tensors),
        scale_tensors=tuple(row_scale_tensors),
        transposed_data=column_payloads[0], # Representative columnwise payload tensor.
        transposed_scale=column_scale_tensors[0], # Representative columnwise scale tensor.
        transposed_storage_table=column_storage_table,
        transposed_scale_storage_table=ops.make_routed_scale_storage_table(column_scale_tensors),
        transposed_scale_tensors=tuple(column_scale_tensors),
        native_columnwise=True,
    )
