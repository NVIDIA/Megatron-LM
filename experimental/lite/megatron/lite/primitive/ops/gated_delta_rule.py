"""Gated Delta Rule math helpers."""

from __future__ import annotations

import torch
import torch.nn.functional as F

try:
    from fla.modules.l2norm import l2norm as _fla_l2norm  # pyright: ignore[reportMissingImports]

    _HAS_FLA_L2NORM = True
except ImportError:
    _HAS_FLA_L2NORM = False

__all__ = ["l2norm", "torch_chunk_gated_delta_rule"]


def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    if _HAS_FLA_L2NORM and dim == -1 and eps == 1e-6:
        return _fla_l2norm(x)
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


def torch_chunk_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = 64,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Pure PyTorch Gated Delta Rule fallback for correctness smoke tests."""
    original_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query)
        key = l2norm(key)

    query, key, value, beta, g = (
        x.transpose(1, 2).contiguous().float() for x in (query, key, value, beta, g)
    )

    batch, heads, seq_len, key_dim = key.shape
    value_dim = value.shape[-1]
    pad = (chunk_size - seq_len % chunk_size) % chunk_size
    if pad:
        query = F.pad(query, (0, 0, 0, pad))
        key = F.pad(key, (0, 0, 0, pad))
        value = F.pad(value, (0, 0, 0, pad))
        beta = F.pad(beta, (0, pad))
        g = F.pad(g, (0, pad))
    total_len = seq_len + pad
    query = query * (key_dim**-0.5)

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, value, k_beta, v_beta = (
        x.reshape(batch, heads, -1, chunk_size, x.shape[-1])
        for x in (query, key, value, k_beta, v_beta)
    )
    g = g.reshape(batch, heads, -1, chunk_size)

    mask_upper = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
        diagonal=0,
    )
    g_cum = g.cumsum(dim=-1)
    decay = (g_cum.unsqueeze(-1) - g_cum.unsqueeze(-2)).tril().exp().tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay).masked_fill(mask_upper, 0)

    rows = [attn[..., 0, :]]
    for idx in range(1, chunk_size):
        row = attn[..., idx, :idx]
        prev = torch.stack(rows[:idx], dim=-2)[..., :idx]
        acc = row + (row.unsqueeze(-1) * prev).sum(-2)
        rows.append(torch.cat([acc, attn[..., idx, idx:]], dim=-1))
    attn = torch.stack(rows, dim=-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)

    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g_cum.exp().unsqueeze(-1))
    state = (
        torch.zeros(batch, heads, key_dim, value_dim, device=query.device, dtype=query.dtype)
        if initial_state is None
        else initial_state.float()
    )
    strict = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
        diagonal=1,
    )
    out_chunks = []
    for idx in range(total_len // chunk_size):
        q_i, k_i, v_i = query[:, :, idx], key[:, :, idx], value[:, :, idx]
        attn_i = (q_i @ k_i.transpose(-1, -2) * decay[:, :, idx]).masked_fill(strict, 0)
        v_prime = k_cumdecay[:, :, idx] @ state
        v_new = v_i - v_prime
        attn_inter = (q_i * g_cum[:, :, idx, :, None].exp()) @ state
        out_chunks.append(attn_inter + attn_i @ v_new)
        state = (
            state * g_cum[:, :, idx, -1, None, None].exp()
            + (k_i * (g_cum[:, :, idx, -1, None] - g_cum[:, :, idx]).exp().unsqueeze(-1))
            .transpose(-1, -2)
            @ v_new
        )

    out = torch.stack(out_chunks, dim=2).reshape(batch, heads, total_len, value_dim)
    out = out[:, :, :seq_len].transpose(1, 2).contiguous().to(original_dtype)
    return out, state if output_final_state else None
