# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""MOK-specific adaptation of MCore-owned expert parameters."""

from __future__ import annotations

import copy
import warnings
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from megatron.core.transformer.transformer_config import TransformerConfig

_SHARED_EXPERT_BF16_WARNING_EMITTED = False


class _ExpertWeightStorageLayout(Enum):
    """Native MCore expert-parameter ownership layout."""

    SINGLE_GROUPED = auto()
    PER_EXPERT = auto()


@dataclass(frozen=True)
class _NativeExpertWeightSource:
    """One FC direction exposed from native MCore/TE storage.

    SINGLE_GROUPED fields contain one expert-major tensor. PER_EXPERT fields
    contain one tensor per local expert. MXFP8 sources additionally expose
    native TE columnwise payloads and logical E8M0 scales.
    """

    storage_layout: _ExpertWeightStorageLayout
    num_experts: int
    rows: int
    columns: int
    row_data: tuple[torch.Tensor, ...]
    row_scales: tuple[torch.Tensor, ...] | None = None
    column_data: tuple[torch.Tensor, ...] | None = None
    column_scales: tuple[torch.Tensor, ...] | None = None

    @property
    def use_mxfp8(self) -> bool:
        optional_fields = (self.row_scales, self.column_data, self.column_scales)
        if all(field is None for field in optional_fields):
            return False
        if any(field is None for field in optional_fields):
            raise RuntimeError("MOK native MXFP8 source is only partially populated")
        return True

    def validate(self) -> None:
        expected_tensors = (
            1
            if self.storage_layout is _ExpertWeightStorageLayout.SINGLE_GROUPED
            else self.num_experts
        )
        fields = [self.row_data]
        if self.use_mxfp8:
            assert self.row_scales is not None
            assert self.column_data is not None
            assert self.column_scales is not None
            fields.extend((self.row_scales, self.column_data, self.column_scales))
        if any(len(field) != expected_tensors for field in fields):
            raise RuntimeError(
                "MOK native weight source tensor count does not match its storage layout: "
                f"layout={self.storage_layout.name}, expected={expected_tensors}, "
                f"actual={[len(field) for field in fields]}"
            )


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


def _grouped_mxfp8_scale_view(
    param: nn.Parameter, member_attr: str, shape: tuple[int, ...], *, name: str
) -> torch.Tensor:
    """Expose all experts' contiguous TE MXFP8 scale storage through member zero."""
    from megatron.core.fp8_utils import get_grouped_quantized_members

    members = get_grouped_quantized_members(param)
    if not members:
        raise RuntimeError(f"MOK {name} grouped parameter has no quantized members")
    first = getattr(members[0], member_attr, None)
    if first is None or first.dtype != torch.uint8 or not first.is_contiguous():
        raise RuntimeError(f"MOK {name} requires contiguous uint8 TE member storage {member_attr}")
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
    flat = torch.as_strided(first, (expected_numel,), (1,), storage_offset=first.storage_offset())
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


def _extract_single_grouped_weight_source(
    param: nn.Parameter,
    *,
    num_experts: int,
    rows: int,
    columns: int,
    use_mxfp8: bool,
    name: str,
) -> _NativeExpertWeightSource:
    """Expose one native single-grouped parameter without copying its payload."""
    expected_shape = (num_experts, rows, columns)
    if tuple(param.shape) != expected_shape:
        raise RuntimeError(
            f"MOK {name} single-grouped shape mismatch: got {tuple(param.shape)}, "
            f"expected {expected_shape}"
        )

    if not use_mxfp8:
        if param.dtype != torch.bfloat16 or not param.is_contiguous():
            raise RuntimeError(f"MOK {name} BF16 requires a contiguous BF16 parameter")
        # A high-precision TE GroupedTensor keeps its authoritative payload in
        # rowwise_data. Expose that backing storage so the MOK custom op does
        # not materialize the wrapper with torch.stack().
        rowwise_data = getattr(param, "rowwise_data", None)
        row_data = (
            param
            if rowwise_data is None
            else _storage_view(
                rowwise_data,
                expected_shape,
                dtype=torch.bfloat16,
                name=f"{name} BF16 rowwise",
            )
        )
        source = _NativeExpertWeightSource(
            storage_layout=_ExpertWeightStorageLayout.SINGLE_GROUPED,
            num_experts=num_experts,
            rows=rows,
            columns=columns,
            row_data=(row_data,),
        )
        source.validate()
        return source

    from megatron.core.fp8_utils import is_grouped_mxfp8tensor

    if not is_grouped_mxfp8tensor(param):
        raise RuntimeError(
            f"MOK {name} MXFP8 requires a native TE grouped MXFP8 parameter"
        )

    row_data = _storage_view(
        param.rowwise_data,
        expected_shape,
        dtype=torch.float8_e4m3fn,
        name=f"{name} rowwise",
    )
    column_data = _storage_view(
        param.columnwise_data,
        expected_shape,
        dtype=torch.float8_e4m3fn,
        name=f"{name} columnwise",
    )
    row_scale = _grouped_mxfp8_scale_view(
        param,
        "_rowwise_scale_inv",
        (num_experts, rows, columns // 32),
        name=f"{name} rowwise",
    )
    # TE's native columnwise scale is [E, M/32, K]. MOK interprets the
    # associated columnwise payload as the transposed matrix, so expose the
    # logical scale as [E, K, M/32].
    column_scale = _grouped_mxfp8_scale_view(
        param,
        "_columnwise_scale_inv",
        (num_experts, rows // 32, columns),
        name=f"{name} columnwise",
    ).transpose(-2, -1)
    source = _NativeExpertWeightSource(
        storage_layout=_ExpertWeightStorageLayout.SINGLE_GROUPED,
        num_experts=num_experts,
        rows=rows,
        columns=columns,
        row_data=(row_data,),
        row_scales=(row_scale,),
        column_data=(column_data,),
        column_scales=(column_scale,),
    )
    source.validate()
    return source


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


def _extract_per_expert_weight_source(
    params: tuple[nn.Parameter, ...],
    *,
    num_experts: int,
    rows: int,
    columns: int,
    use_mxfp8: bool,
    name: str,
) -> _NativeExpertWeightSource:
    """Expose independent per-expert parameters before building MOK tables."""
    if len(params) != num_experts:
        raise RuntimeError(
            f"MOK {name} per-expert parameter count mismatch: got {len(params)}, "
            f"expected {num_experts}"
        )
    shape = (rows, columns)
    if not use_mxfp8:
        payloads = []
        for param in params:
            storage = _parameter_storage_attr(param, "rowwise_data")
            payload = param if storage is None else storage
            payloads.append(
                _storage_view(
                    payload, shape, dtype=torch.bfloat16, name=f"{name} BF16 rowwise"
                )
            )
        source = _NativeExpertWeightSource(
            storage_layout=_ExpertWeightStorageLayout.PER_EXPERT,
            num_experts=num_experts,
            rows=rows,
            columns=columns,
            row_data=tuple(payloads),
        )
        source.validate()
        return source

    from megatron.core.fp8_utils import is_mxfp8tensor

    row_payloads = []
    column_payloads = []
    row_scales = []
    column_scales = []
    for expert, param in enumerate(params):
        if not is_mxfp8tensor(param):
            raise RuntimeError(
                f"MOK {name} expert {expert} requires native TE MXFP8 storage"
            )
        row_payload = _parameter_storage_attr(param, "rowwise_data")
        column_payload = _parameter_storage_attr(param, "columnwise_data")
        row_scale = _parameter_storage_attr(param, "_rowwise_scale_inv")
        column_scale = _parameter_storage_attr(param, "_columnwise_scale_inv")
        if any(
            value is None
            for value in (row_payload, column_payload, row_scale, column_scale)
        ):
            missing = [
                field_name
                for field_name, value in (
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
                f"MOK {name} expert {expert} is missing native storage: "
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
                name=f"{name} MXFP8 rowwise",
            )
        )
        column_payloads.append(
            _storage_view(
                column_payload,
                shape,
                dtype=torch.float8_e4m3fn,
                name=f"{name} MXFP8 columnwise",
            )
        )
        if tuple(row_scale.shape) != (rows, columns // 32):
            raise RuntimeError(
                f"MOK {name} rowwise scale shape mismatch: got {tuple(row_scale.shape)}, "
                f"expected {(rows, columns // 32)}"
            )
        if tuple(column_scale.shape) != (rows // 32, columns):
            raise RuntimeError(
                f"MOK {name} columnwise scale shape mismatch: "
                f"got {tuple(column_scale.shape)}, expected {(rows // 32, columns)}"
            )
        row_scales.append(row_scale)
        column_scales.append(column_scale.transpose(-2, -1))

    source = _NativeExpertWeightSource(
        storage_layout=_ExpertWeightStorageLayout.PER_EXPERT,
        num_experts=num_experts,
        rows=rows,
        columns=columns,
        row_data=tuple(row_payloads),
        row_scales=tuple(row_scales),
        column_data=tuple(column_payloads),
        column_scales=tuple(column_scales),
    )
    source.validate()
    return source


def _extract_native_expert_weight_source(
    params: tuple[nn.Parameter, ...],
    *,
    storage_layout: _ExpertWeightStorageLayout,
    num_experts: int,
    rows: int,
    columns: int,
    use_mxfp8: bool,
    name: str,
) -> _NativeExpertWeightSource:
    """Dispatch extraction using MCore's explicit single/non-single semantics."""
    if storage_layout is _ExpertWeightStorageLayout.SINGLE_GROUPED:
        if len(params) != 1:
            raise RuntimeError(
                f"MOK {name} SINGLE_GROUPED requires one parameter, got {len(params)}"
            )
        return _extract_single_grouped_weight_source(
            params[0],
            num_experts=num_experts,
            rows=rows,
            columns=columns,
            use_mxfp8=use_mxfp8,
            name=name,
        )
    if storage_layout is _ExpertWeightStorageLayout.PER_EXPERT:
        return _extract_per_expert_weight_source(
            params,
            num_experts=num_experts,
            rows=rows,
            columns=columns,
            use_mxfp8=use_mxfp8,
            name=name,
        )
    raise AssertionError(f"Unhandled MOK weight layout: {storage_layout}")


def _prepare_mok_weight(source: _NativeExpertWeightSource):
    """Build MOK's tensor/descriptor representation from a native source."""
    from mok import functional, ops

    source.validate()
    if source.storage_layout is _ExpertWeightStorageLayout.SINGLE_GROUPED:
        if not source.use_mxfp8:
            return source.row_data[0]
        assert source.row_scales is not None
        assert source.column_data is not None
        assert source.column_scales is not None
        return (
            source.row_data[0],
            _swizzle_mxfp8_scale(
                source.row_scales[0], rows=source.rows, columns=source.columns
            ),
            source.column_data[0],
            _swizzle_mxfp8_scale(
                source.column_scales[0], rows=source.columns, columns=source.rows
            ),
            True,
        )

    if source.storage_layout is not _ExpertWeightStorageLayout.PER_EXPERT:
        raise AssertionError(f"Unhandled MOK weight layout: {source.storage_layout}")

    if not source.use_mxfp8:
        payloads = list(source.row_data)
        return functional.SplitRoutedWeight(
            data=payloads[0],
            storage_table=ops.make_routed_weight_storage_table_bf16(payloads),
        )

    assert source.row_scales is not None
    assert source.column_data is not None
    assert source.column_scales is not None
    row_scale_tensors = [
        _swizzle_mxfp8_scale(
            scale.unsqueeze(0), rows=source.rows, columns=source.columns
        )
        for scale in source.row_scales
    ]
    column_scale_tensors = [
        _swizzle_mxfp8_scale(
            scale.unsqueeze(0), rows=source.columns, columns=source.rows
        )
        for scale in source.column_scales
    ]
    row_payloads = list(source.row_data)
    column_payloads = list(source.column_data)
    return functional.SplitRoutedWeight(
        data=row_payloads[0],
        storage_table=ops.make_routed_weight_storage_table_mxfp8(row_payloads),
        scale=row_scale_tensors[0],
        scale_storage_table=ops.make_routed_scale_storage_table(row_scale_tensors),
        scale_tensors=tuple(row_scale_tensors),
        transposed_data=column_payloads[0],
        transposed_scale=column_scale_tensors[0],
        transposed_storage_table=ops.make_routed_weight_storage_table_mxfp8(
            column_payloads
        ),
        transposed_scale_storage_table=ops.make_routed_scale_storage_table(
            column_scale_tensors
        ),
        transposed_scale_tensors=tuple(column_scale_tensors),
        native_columnwise=True,
    )


def _refresh_prepared_mok_weight(prepared, source: _NativeExpertWeightSource) -> None:
    """Refresh per-expert MXFP8 scales without replacing captured descriptors."""
    source.validate()
    if (
        source.storage_layout is not _ExpertWeightStorageLayout.PER_EXPERT
        or not source.use_mxfp8
    ):
        raise RuntimeError(
            "MOK in-place scale refresh requires PER_EXPERT MXFP8 weights"
        )

    assert source.row_scales is not None
    assert source.column_scales is not None
    row_outputs = prepared.scale_tensors
    column_outputs = prepared.transposed_scale_tensors
    if row_outputs is None or column_outputs is None:
        raise RuntimeError(
            "MOK per-expert MXFP8 cache is missing retained scale tensors"
        )
    if (
        len(row_outputs) != source.num_experts
        or len(column_outputs) != source.num_experts
    ):
        raise RuntimeError(
            "MOK per-expert MXFP8 scale cache expert count mismatch: "
            f"source={source.num_experts}, row_outputs={len(row_outputs)}, "
            f"column_outputs={len(column_outputs)}"
        )

    for row_scale, column_scale, row_out, column_out in zip(
        source.row_scales,
        source.column_scales,
        row_outputs,
        column_outputs,
        strict=True,
    ):
        _swizzle_mxfp8_scale(
            row_scale.unsqueeze(0),
            rows=source.rows,
            columns=source.columns,
            out=row_out,
        )
        _swizzle_mxfp8_scale(
            column_scale.unsqueeze(0),
            rows=source.columns,
            columns=source.rows,
            out=column_out,
        )
