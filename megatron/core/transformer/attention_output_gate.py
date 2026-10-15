# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Sigmoid output gating shared by attention implementations."""

import torch


def validate_attention_output_gate_shapes(
    core_attn_out: torch.Tensor, gate: torch.Tensor, granularity: str
) -> None:
    """Check token alignment and the local tensor-parallel gate dimensions."""
    if core_attn_out.ndim == 0 or gate.ndim == 0:
        raise ValueError("Attention output gating requires tensors with a channel dimension.")
    if core_attn_out.shape[:-1] != gate.shape[:-1]:
        raise ValueError(
            "Attention output gate and core attention output must have matching token/batch "
            f"dimensions, got {tuple(gate.shape)} and {tuple(core_attn_out.shape)}."
        )
    if granularity == 'elementwise':
        if core_attn_out.shape != gate.shape:
            raise ValueError(
                "Elementwise attention output gating requires the gate and core attention output "
                f"to have the same shape, got {tuple(gate.shape)} and "
                f"{tuple(core_attn_out.shape)}."
            )
    elif granularity == 'headwise':
        if gate.size(-1) == 0 or core_attn_out.size(-1) % gate.size(-1) != 0:
            raise ValueError(
                "Headwise attention output gating requires the core attention output dimension "
                f"({core_attn_out.size(-1)}) to be divisible by the number of local gates "
                f"({gate.size(-1)})."
            )
    else:
        raise ValueError(f"Unsupported attention output gate granularity: {granularity!r}.")


def apply_attention_output_gate(
    core_attn_out: torch.Tensor, gate: torch.Tensor, granularity: str, *, cast_mode: str
) -> torch.Tensor:
    """Apply an elementwise or per-head sigmoid gate to attention output channels.

    Sigmoid always runs in FP32. ``cast_mode='before'`` casts its scale to the
    input dtype before multiplying (MLA and scalar head gates). ``'after'``
    multiplies with the FP32 scale and casts the product to the input dtype
    (regular attention). ``'none'`` leaves the scale and product in their
    natural dtypes so the caller can apply its own final cast (KDA).
    Callers must choose explicitly to retain their existing rounding and
    gradients. With ``'none'``, the output dtype may differ from the input dtype.

    Inputs have matching token/batch dimensions and flattened local output
    channels. Headwise gates contain one scalar per local head. Gate projection,
    packed-sequence layout restoration, and output normalization belong to the
    caller. This function is eager unless called through a compiled wrapper.
    Headwise gates split the value channels, apply sigmoid and the selected cast
    to the flat gate, then unsqueeze the scale for multiplication and reshape
    the output to its original shape. Elementwise gates require no reshaping.
    """
    validate_attention_output_gate_shapes(core_attn_out, gate, granularity)
    if cast_mode not in ('before', 'after', 'none'):
        raise ValueError(f"Unsupported attention output gate cast mode: {cast_mode!r}.")
    output_shape = core_attn_out.shape
    output_dtype = core_attn_out.dtype
    if granularity == 'headwise':
        core_attn_out = core_attn_out.view(*output_shape[:-1], gate.size(-1), -1)

    scale = torch.sigmoid(gate.float())
    if cast_mode == 'before':
        scale = scale.to(output_dtype)
    if granularity == 'headwise':
        scale = scale.unsqueeze(-1)
    output = core_attn_out * scale
    if cast_mode == 'after':
        output = output.to(output_dtype)
    if granularity == 'headwise':
        return output.reshape(output_shape)
    return output
