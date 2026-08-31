# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Bit-exact parity for the primitive HF reader's INT4 dequant.

Kimi ships INT4-packed release weights (eight offset-binary INT4 values per
int32 slot + per-group scales).  The MLite load path dequantizes them in
``primitive/ckpt/hf_weights.py`` without any ``transformers`` / external-bridge
dependency. This test pins that dequant to be bit-exact against the reference
algorithm in ``megatron.bridge`` (``conversion/quantization_utils.py``:
``quantize_to_int4`` / ``dequantize_int4``), which the load path was written to
match.  The reference is vendored here (CPU-only, no GPU) so the guard runs in
unit CI and does not import megatron.bridge.
"""

from __future__ import annotations

import torch


# ---------------------------------------------------------------------------
# Reference INT4 quant/dequant, copied verbatim from megatron.bridge
# conversion/quantization_utils.py (quantize_to_int4 / dequantize_int4).  These
# define the "ground truth" packed layout + group-scale semantics that the
# MLite load path must reproduce exactly.
# ---------------------------------------------------------------------------
def _bridge_quantize_to_int4(
    weight: torch.Tensor,
    group_size: int = 32,
    scale_dtype: torch.dtype = torch.bfloat16,
):
    out_features, in_features = weight.shape
    weight_shape = torch.tensor([out_features, in_features], dtype=torch.int32)

    w = weight.float()
    num_groups = (in_features + group_size - 1) // group_size
    w_grouped = w.view(out_features, num_groups, -1)

    group_max = w_grouped.abs().amax(dim=-1)
    scale = (group_max / 7.0).clamp(min=1e-10)

    scale_expanded = scale.unsqueeze(-1).expand_as(w_grouped)
    w_q = (w_grouped / scale_expanded).round().clamp(-8, 7)

    w_q = w_q.view(out_features, -1)[:, :in_features]
    w_q = (w_q + 8).to(torch.uint8)

    assert in_features % 8 == 0
    w_q_grouped = w_q.view(out_features, in_features // 8, 8).to(torch.int32)
    packed = torch.zeros(out_features, in_features // 8, dtype=torch.int32)
    for i in range(8):
        packed |= (w_q_grouped[:, :, i] & 0xF) << (i * 4)

    return packed, scale.to(scale_dtype), weight_shape


def _bridge_dequantize_int4(
    weight_packed: torch.Tensor, weight_scale: torch.Tensor, weight_shape: torch.Tensor
) -> torch.Tensor:
    local_out, local_packed_in = weight_packed.shape
    local_in = local_packed_in * 8

    shifts = torch.arange(8) * 4
    unpacked = ((weight_packed.unsqueeze(-1) >> shifts) & 0xF).float()
    unpacked = unpacked.reshape(local_out, local_in) - 8

    scale = weight_scale.float()
    if scale.ndim == 1:
        scale = scale.view(local_out, scale.numel() // local_out)
    else:
        scale = scale.view(local_out, -1)
    elements_per_group = local_in // scale.shape[1]
    scale_expanded = scale.repeat_interleave(elements_per_group, dim=1)[:, :local_in]

    return (unpacked * scale_expanded).to(torch.bfloat16)


def _reader_for(tensors: dict[str, torch.Tensor]):
    from megatron.lite.primitive.ckpt.hf_weights import SafeTensorReader

    reader = object.__new__(SafeTensorReader)
    reader.device = torch.device("cpu")
    reader._cached_request = None
    reader._cached_tensor = None
    reader.has_tensor = lambda name: name in tensors
    reader._get_raw_tensor = lambda name, device: tensors[name].to(device)
    return reader


def test_kimi_int4_dequant_bit_exact_to_bridge():
    from megatron.lite.primitive.ckpt.hf_weights import (  # isort: skip
        _dequantize_groupwise_int4,
    )

    torch.manual_seed(2026)
    # out_features arbitrary; in_features divisible by both 8 (pack factor) and 32
    # (group_size) so the packed/group layout is exercised with >1 group per row.
    weight = torch.randn(40, 128, dtype=torch.bfloat16)

    packed, scale, shape = _bridge_quantize_to_int4(weight)
    reference = _bridge_dequantize_int4(packed, scale, shape)

    got = _dequantize_groupwise_int4(packed, scale, shape)

    assert got.dtype == torch.bfloat16
    assert got.shape == reference.shape
    assert torch.equal(got, reference), (
        "MLite INT4 dequant diverged from the megatron.bridge reference: "
        f"max|diff|={(got.float() - reference.float()).abs().max().item()}"
    )


def test_reader_passes_through_bf16_when_unquantized():
    """The primitive reader loads plain bf16 unchanged (no scale, no packing).

    MLite export/save emits bf16, so reloading an exported (or already bf16)
    checkpoint must hit the no-dequant path — the third load case alongside
    FP8 (``*_scale_inv``) and INT4 (``*_packed``).
    """
    torch.manual_seed(7)
    weight = torch.randn(16, 24, dtype=torch.bfloat16)
    reader = _reader_for({"layer.weight": weight})

    got = reader.get_tensor("layer.weight")
    assert got.dtype == torch.bfloat16
    assert torch.equal(got, weight)


def test_reader_dispatches_to_int4_when_packed_present():
    """When only ``*_packed`` exists, the primitive reader dequantizes INT4."""

    torch.manual_seed(11)
    weight = torch.randn(8, 64, dtype=torch.bfloat16)
    packed, scale, shape = _bridge_quantize_to_int4(weight)
    reader = _reader_for({"w_packed": packed, "w_scale": scale, "w_shape": shape})

    got = reader.get_tensor("w")
    assert torch.equal(got, _bridge_dequantize_int4(packed, scale, shape))
