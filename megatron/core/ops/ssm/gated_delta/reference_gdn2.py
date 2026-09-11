# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025, Songlin Yang, Jan Kautz, Ali Hatamizadeh.

# Some of this code was adopted from https://github.com/huggingface/transformers
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F

from megatron.core.ops.kernel_metadata import validate_kernel
from megatron.core.ops.ssm.gated_delta.kernel_metadata import FLA_L2NORM


def torch_chunk_gdn2(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    scale: float | None = None,
    chunk_size: int = 64,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    r"""Torch-native chunkwise Gated Delta Rule-2, for deterministic mode.

    Args:
        q: queries of shape ``[B, T, H, K]``.
        k: keys of shape ``[B, T, H, K]``.
        v: values of shape ``[B, T, H, V]``.
        g: channel-wise log-decay of shape ``[B, T, H, K]``.
        b: channel-wise erase gate of shape ``[B, T, H, K]``.
        w: channel-wise write gate of shape ``[B, T, H, V]``.
        scale: attention scale. Defaults to ``1 / sqrt(K)``.
        chunk_size: chunk length of the WY schedule.
        initial_state: optional ``[B, H, K, V]`` initial state.
        output_final_state: whether to also return the final recurrent state.
        use_qk_l2norm_in_kernel: L2-normalize q and k here rather than in the caller.
        cu_seqlens: packed-sequence offsets; unsupported, must be ``None``.
        kwargs: accepted and ignored, so this stays interchangeable with the FLA
            kernel, which takes several options this implementation does not model.

    Returns:
        (tuple[Tensor, Tensor | None]): output of shape ``[B, T, H, V]`` and the
        final state, or ``None`` when ``output_final_state`` is ``False``.
    """
    assert cu_seqlens is None, "cu_seqlens is not supported for torch_chunk_gdn2 for now."

    initial_dtype = q.dtype
    if use_qk_l2norm_in_kernel:
        validate_kernel(FLA_L2NORM)
        from fla.modules.l2norm import l2norm

        q = l2norm(q, dim=-1, eps=1e-6)
        k = l2norm(k, dim=-1, eps=1e-6)

    # b s h d -> b h s d, and compute the whole recurrence in fp32
    query, key, value, g, b, w = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (q, k, v, g, b, w)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    # Zero padding is inert: it leaves the erase/write rows empty and, because the
    # padded log-decay is 0, leaves the chunk's cumulative decay at its last real value.
    query, key, value, g, b, w = [
        F.pad(x, (0, 0, 0, pad_size)) for x in (query, key, value, g, b, w)
    ]
    total_sequence_length = sequence_length + pad_size
    if scale is None:
        scale = 1 / (k_head_dim**0.5)
    query = query * scale

    # reshape to chunks
    query, key, value, g, b, w = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (query, key, value, g, b, w)
    ]

    # Channel-wise cumulative log-decay within each chunk.
    g = g.cumsum(dim=-2)
    decay = g.exp()

    # The pairwise decay exp(G_r - G_j) is carried on the operands, as
    # exp(G_r - c) * exp(c - G_j) for any per-channel c. Centering on half the
    # chunk's total decay halves the exponent range each operand has to represent,
    # which keeps exp() in fp32 range for roughly twice the decay strength.
    center = g[..., -1:, :] * 0.5
    decay_centered = (g - center).exp()
    inv_decay_centered = (center - g).exp()

    erase = decay * b * key  # E = exp(G) * b * k
    erase_centered = decay_centered * b * key
    key_inv_decay = key * inv_decay_centered  # Khat = exp(c - G) * k
    write = w * value  # Z = w * v

    # T = (I + A)^{-1} with A = tril(E @ Khat^T, -1), by forward substitution.
    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0
    )
    attn = -(erase_centered @ key_inv_decay.transpose(-1, -2)).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)

    write = attn @ write  # T @ Z
    k_cumdecay = attn @ erase  # T @ E

    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(write)
    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1
    )
    query_decay = query * decay  # Qtilde = exp(G) * q, exact: multiplies the incoming state
    query_decay_centered = query * decay_centered  # centered: only used pairwise against Khat

    # for each chunk
    for i in range(0, total_sequence_length // chunk_size):
        attn_i = query_decay_centered[:, :, i] @ key_inv_decay[:, :, i].transpose(-1, -2)
        attn_i = attn_i.masked_fill_(mask, 0)
        # U = T @ (Z - E @ S), the chunk's delta residuals against the incoming state
        u_i = write[:, :, i] - k_cumdecay[:, :, i] @ last_recurrent_state
        attn_inter = query_decay[:, :, i] @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn_i @ u_i
        # Carry the state across the chunk: decay it by the chunk total, then add
        # the delta residuals mapped back through the keys. exp(G_C - G) <= 1, so this
        # ratio needs no centering.
        g_chunk = g[:, :, i, -1:]  # G_C, the chunk's total log-decay, [b, h, 1, k]
        key_bar = key[:, :, i] * (g_chunk - g[:, :, i]).exp()
        last_recurrent_state = (
            last_recurrent_state * g_chunk.squeeze(-2).unsqueeze(-1).exp()
            + key_bar.transpose(-1, -2) @ u_i
        )

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.reshape(
        core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1]
    )
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state
