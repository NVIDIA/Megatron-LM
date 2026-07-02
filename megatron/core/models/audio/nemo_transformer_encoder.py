# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Vendored from NeMo ASR transformer_encoder (custom / experimental module).
# Kept verbatim where possible (comments, MultiHeadAttention with use_cache, helper classes
# `GPTConfig`, `GELU`, `compute_rope_params`, `apply_rope`, `MultiHeadAttentionWithFA`) so this
# file can be diffed against the NeMo source. `flash_attn_func` is imported lazily so this
# module imports without flash-attn installed.

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import GELU as TorchGELU
from torch.nn import Conv1d
from torch.nn.functional import scaled_dot_product_attention
from torch.utils.checkpoint import checkpoint


def _flash_attn_func(*args, **kwargs):
    """Lazy import shim for flash_attn so this module imports without the package."""
    try:
        from flash_attn import flash_attn_func as _impl
    except ImportError as exc:
        raise ImportError(
            "flash_attn is required for MultiHeadAttentionWithFA. "
            "Install flash-attn or use MultiHeadAttentionWithSDPA instead."
        ) from exc
    return _impl(*args, **kwargs)


def _get_te_dot_product_attention():
    """Lazy import of transformer_engine.pytorch.DotProductAttention."""
    try:
        from transformer_engine.pytorch import DotProductAttention as TEDPA
    except ImportError as exc:
        raise ImportError(
            "transformer-engine is required for MultiHeadAttentionWithTE. "
            "Install transformer-engine or set attn_impl='sdpa'."
        ) from exc
    return TEDPA


def _resolve_attn_impl(impl: str) -> str:
    """Resolve 'auto' to 'te' if transformer_engine is importable, else 'sdpa'."""
    if impl != "auto":
        return impl
    try:
        import transformer_engine.pytorch  # noqa: F401

        return "te"
    except ImportError:
        return "sdpa"


@dataclass
class GPTConfig:
    vocab_size: int = 50257
    context_length: int = 1024
    emb_dim: int = 768
    n_heads: int = 12
    n_layers: int = 12
    drop_rate: int = 0.1
    qkv_bias: bool = False
    theta_base: int = 10_000


@dataclass
class TransformerEncoderConfig:
    n_mels: int = 80
    d_model: int = 512
    n_heads: int = 12
    n_layers: int = 12
    drop_rate: float = 0.1
    qkv_bias: bool = False
    causal_mask: bool = False
    theta_base: int = 10_000
    context_length: int = 4096
    qk_norm: bool = False
    # Attention backend: "auto" picks "te" if transformer-engine is installed,
    # otherwise "sdpa". Explicit values: "te" | "sdpa" | "fa".
    attn_impl: str = "auto"
    recompute_layers: bool = False
    # Number of previous encoder positions each token can attend when causal_mask=True.
    # None or a negative value preserves the current unlimited-left causal attention.
    left_context: Optional[int] = None


def _normalize_left_context(left_context: Optional[int]) -> Optional[int]:
    if left_context is None:
        return None
    left_context = int(left_context)
    if left_context < 0:
        return None
    return left_context


def _causal_window_size(
    causal_mask: bool, left_context: Optional[int]
) -> Optional[Tuple[int, int]]:
    left_context = _normalize_left_context(left_context)
    if left_context is None:
        return None
    if not causal_mask:
        raise ValueError("left_context requires causal_mask=True")
    return (left_context, 0)


def _causal_disallow_mask(
    query_len: int, key_len: int, left_context: Optional[int], device
) -> torch.Tensor:
    """Return a bool mask where True means the key is not visible to the query."""
    query_offset = key_len - query_len
    query_positions = torch.arange(query_offset, query_offset + query_len, device=device).unsqueeze(
        1
    )
    key_positions = torch.arange(key_len, device=device).unsqueeze(0)
    disallow = key_positions > query_positions
    left_context = _normalize_left_context(left_context)
    if left_context is not None:
        disallow = disallow | (key_positions < query_positions - left_context)
    return disallow


class FeedForward(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.ffn = nn.Sequential(nn.Linear(dim, 4 * dim), TorchGELU(), nn.Linear(4 * dim, dim))

    def forward(self, x):
        return self.ffn(x)


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return (
            0.5
            * x
            * (
                1
                + torch.tanh(
                    torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))
                )
            )
        )


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))
        self.shift = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        # Cast to fp32 for numerically stable mean/var, then back to original dtype.
        # Original NeMo source used `torch.autocast('cuda', ...)`; we manually upcast so this
        # also works on CPU (e.g. unit tests).
        xf = x.float()
        mean = xf.mean(dim=-1, keepdim=True)
        var = xf.var(dim=-1, keepdim=True, unbiased=False)
        norm = (xf - mean) / torch.sqrt(var + self.eps)
        output = self.scale * norm + self.shift
        return output.to(dtype=x.dtype)


def compute_rope_params(head_dim, theta_base=10_000, context_length=4096, dtype=torch.float32):
    assert head_dim % 2 == 0, "Embedding dimension must be even"

    # Compute the inverse frequencies
    inv_freq = 1.0 / (
        theta_base
        ** (torch.arange(0, head_dim, 2, dtype=dtype)[: (head_dim // 2)].float() / head_dim)
    )

    # Generate position indices
    positions = torch.arange(context_length, dtype=dtype)

    # Compute the angles
    angles = positions.unsqueeze(1) * inv_freq.unsqueeze(
        0
    )  # Shape: (context_length, head_dim // 2)

    # Expand angles to match the head_dim
    angles = torch.cat([angles, angles], dim=1)  # Shape: (context_length, head_dim)

    # Precompute sine and cosine
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    return cos, sin


def apply_rope(x, cos, sin):
    # x: (batch_size, num_heads, seq_len, head_dim)
    batch_size, num_heads, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0, "Head dimension must be even"

    # Split x into first half and second half
    x1 = x[..., : head_dim // 2]  # First half
    x2 = x[..., head_dim // 2 :]  # Second half

    # Adjust sin and cos shapes
    cos = cos[:seq_len, :].unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, seq_len, head_dim)
    sin = sin[:seq_len, :].unsqueeze(0).unsqueeze(0)

    # Apply the rotary transformation
    rotated = torch.cat((-x2, x1), dim=-1)
    x_rotated = (x * cos) + (rotated * sin)

    # It's ok to use lower-precision after applying cos and sin rotation
    return x_rotated.to(dtype=x.dtype)


class MultiHeadAttentionWithFA(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        dropout=0.0,
        qkv_bias=False,
        context_length=1024,
        num_heads=8,
        causal_mask=False,
        left_context=None,
    ):
        super().__init__()
        self.d_out = dim_out
        self.w_query = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.w_key = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.w_value = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.num_heads = num_heads
        self.head_dim = dim_out // num_heads
        self.dropout = dropout
        self.causal_mask = causal_mask
        self.window_size = _causal_window_size(causal_mask, left_context)
        self.out_proj = nn.Linear(self.d_out, self.d_out)

    def forward(self, x):
        B, num_tokens, d_in = x.shape
        H = self.num_heads

        keys = self.w_key(x).view(B, num_tokens, H, self.head_dim)  # Bxnum_tokens x Hx head_dim
        queries = self.w_query(x).view(B, num_tokens, H, self.head_dim)
        values = self.w_value(x).view(B, num_tokens, H, self.head_dim)

        dropout = 0 if self.training == False else self.dropout
        flash_kwargs = {}
        if self.window_size is not None:
            flash_kwargs["window_size"] = self.window_size
        output = _flash_attn_func(
            queries, keys, values, dropout_p=dropout, causal=self.causal_mask, **flash_kwargs
        )

        # Bxnum_tokens x Hx head_dim

        output = output.contiguous().view(B, num_tokens, self.d_out)

        output = self.out_proj(output)

        return output


class MultiHeadAttentionWithTE(nn.Module):
    """Multi-head attention using Transformer Engine's DotProductAttention in THD layout.

    Always runs attention in TE's `thd` (packed) format so flash-attention handles
    variable-length sequences efficiently. Two callsite modes:

    - Unpacked: forward(x, lengths=...) where x is (B, T, C). The module packs valid
      tokens into THD using `lengths`, runs TE attention, then scatters results back
      into a (B, T, C) tensor (padding positions remain zero).
    - Packed: forward(x, packed_seq_params=...) where x is a flat (Ttot, C) (or
      (Ttot, 1, C)) tensor and the caller supplies cu_seqlens / max_seqlens. The
      module returns (Ttot, C) in the same packed layout.

    Linear projection parameter names (`w_query`, `w_key`, `w_value`, `out_proj`)
    match the SDPA/FA variants so .nemo checkpoints load into either backend.
    """

    def __init__(
        self,
        dim_in,
        dim_out,
        dropout=0.0,
        qkv_bias=False,
        num_heads=8,
        causal_mask=False,
        left_context=None,
        qk_norm=False,
        **_,
    ):
        super().__init__()
        if dim_out % num_heads != 0:
            raise ValueError(f"dim_out={dim_out} not divisible by num_heads={num_heads}")
        self.d_out = dim_out
        self.num_heads = num_heads
        self.head_dim = dim_out // num_heads
        self.causal_mask = causal_mask
        self.window_size = _causal_window_size(causal_mask, left_context)
        self.qk_norm = qk_norm
        # In thd layout TE requires a padding-style mask type (cu_seqlens carries segment info).
        self.attn_mask_type = "padding_causal" if causal_mask else "padding"

        self.w_query = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.w_key = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.w_value = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.out_proj = nn.Linear(dim_out, dim_out)
        if self.qk_norm:
            self.q_norm = nn.LayerNorm(self.head_dim)
            self.k_norm = nn.LayerNorm(self.head_dim)

        te_dpa_cls = _get_te_dot_product_attention()
        te_kwargs = dict(
            num_attention_heads=num_heads,
            kv_channels=self.head_dim,
            attention_dropout=dropout,
            qkv_format="thd",
            attn_mask_type=self.attn_mask_type,
            tp_size=1,
            tp_group=None,
            layer_number=1,
        )
        if self.window_size is not None:
            te_kwargs["window_size"] = self.window_size
        self.te_attn = te_dpa_cls(**te_kwargs)

    def forward(self, x, lengths=None, packed_seq_params=None, **_):
        if packed_seq_params is not None:
            x_packed = x.reshape(-1, x.shape[-1])
            cu_q = packed_seq_params.cu_seqlens_q
            cu_kv = packed_seq_params.cu_seqlens_kv
            max_q = packed_seq_params.max_seqlen_q
            max_kv = packed_seq_params.max_seqlen_kv
            valid_mask = None
            bsz_t = None
        else:
            B, T, _C = x.shape
            if lengths is None:
                lengths32 = torch.full((B,), T, dtype=torch.int32, device=x.device)
            else:
                lengths32 = lengths.to(torch.int32)
            valid_mask = torch.arange(T, device=x.device).unsqueeze(0) < lengths32.unsqueeze(1)
            x_packed = x[valid_mask]
            cu_q = torch.nn.functional.pad(lengths32.cumsum(0).to(torch.int32), (1, 0))
            cu_kv = cu_q
            max_q = int(lengths32.max().item())
            max_kv = max_q
            bsz_t = (B, T)

        Ttot = x_packed.shape[0]
        q = self.w_query(x_packed).view(Ttot, self.num_heads, self.head_dim)
        k = self.w_key(x_packed).view(Ttot, self.num_heads, self.head_dim)
        v = self.w_value(x_packed).view(Ttot, self.num_heads, self.head_dim)
        if self.qk_norm and Ttot > 0:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if Ttot == 0:
            # Empty packed batch (e.g. all-text minibatch in a VLM blend). TE's
            # flash-attention backend can't reshape a (0, ...) output, so skip
            # the attention call and produce a properly-shaped zero tensor.
            out = q.new_zeros(0, self.num_heads, self.head_dim)
        else:
            out = self.te_attn(
                q,
                k,
                v,
                None,  # attention_mask -- unused for THD/padding mode
                attn_mask_type=self.attn_mask_type,
                qkv_format="thd",
                cu_seqlens_q=cu_q,
                cu_seqlens_kv=cu_kv,
                max_seqlen_q=max_q,
                max_seqlen_kv=max_kv,
            )
        out = out.reshape(Ttot, self.d_out)
        out = self.out_proj(out)

        if valid_mask is None:
            return out
        B, T = bsz_t
        full = x.new_zeros(B, T, self.d_out)
        full[valid_mask] = out
        return full


class MultiHeadAttentionWithSDPA(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        dropout=0.0,
        qkv_bias=False,
        context_length=1024,
        num_heads=8,
        causal_mask=False,
        left_context=None,
        qk_norm=False,
    ):
        super().__init__()
        self.d_out = dim_out
        self.w_query = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.w_key = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.w_value = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.num_heads = num_heads
        self.head_dim = dim_out // num_heads
        self.dropout = dropout
        self.causal_mask = causal_mask
        self.left_context = _normalize_left_context(left_context)
        if self.left_context is not None and not self.causal_mask:
            raise ValueError("left_context requires causal_mask=True")
        self.out_proj = nn.Linear(self.d_out, self.d_out)
        self.qk_norm = qk_norm
        if self.qk_norm:
            self.q_norm = nn.LayerNorm(self.head_dim)
            self.k_norm = nn.LayerNorm(self.head_dim)

    def forward(self, x, attn_mask=None, use_cache=False):
        B, num_tokens, d_in = x.shape
        H = self.num_heads

        keys = self.w_key(x).view(B, num_tokens, H, self.head_dim)  # Bxnum_tokens x Hx head_dim
        queries = self.w_query(x).view(B, num_tokens, H, self.head_dim)
        values = self.w_value(x).view(B, num_tokens, H, self.head_dim)

        keys = keys.transpose(1, 2)  # BxHxnum_tokens,head_dim
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        if self.qk_norm:
            queries = self.q_norm(queries)
            keys = self.k_norm(keys)

        dropout = 0 if self.training == False else self.dropout
        is_causal = self.causal_mask
        if self.causal_mask and self.left_context is not None:
            causal_mask = ~_causal_disallow_mask(
                num_tokens, num_tokens, self.left_context, x.device
            )
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
            attn_mask = causal_mask if attn_mask is None else attn_mask & causal_mask
            is_causal = False

        output = scaled_dot_product_attention(
            queries, keys, values, attn_mask=attn_mask, is_causal=is_causal, dropout_p=dropout
        )

        # B xH x num_tokens x head_dim

        output = output.transpose(1, 2)  # Bxnum_tokens x Hx head_dim

        output = output.contiguous().view(B, num_tokens, self.d_out)

        output = self.out_proj(output)

        return output


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        dropout=0.0,
        qkv_bias=False,
        context_length=1024,
        num_heads=8,
        causal_mask=False,
        left_context=None,
    ):
        super().__init__()
        self.d_out = dim_out
        self.w_query = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.w_key = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.w_value = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.num_heads = num_heads
        self.head_dim = dim_out // num_heads
        self.dropout = nn.Dropout(dropout)
        self.causal_mask = causal_mask
        self.left_context = _normalize_left_context(left_context)
        if self.left_context is not None and not self.causal_mask:
            raise ValueError("left_context requires causal_mask=True")
        self.out_proj = nn.Linear(self.d_out, self.d_out)

        self.register_buffer(
            "mask",
            (
                _causal_disallow_mask(
                    context_length, context_length, self.left_context, torch.device("cpu")
                )
                if self.causal_mask
                else torch.zeros(context_length, context_length, dtype=torch.bool)
            ),
        )

        self.register_buffer("cache_k", None, persistent=False)
        self.register_buffer("cache_v", None, persistent=False)

    def forward(self, x, use_cache=False):
        B, num_tokens, d_in = x.shape
        H = self.num_heads

        keys = self.w_key(x).view(B, num_tokens, H, self.head_dim)  # Bxnum_tokens x Hx head_dim
        queries = self.w_query(x).view(B, num_tokens, H, self.head_dim)
        values = self.w_value(x).view(B, num_tokens, H, self.head_dim)

        if use_cache:
            if self.cache_k is None:
                self.cache_k = keys
                self.cache_v = values
            else:
                self.cache_k = torch.cat((self.cache_k, keys), dim=1)
                self.cache_v = torch.cat((self.cache_v, values), dim=1)
            keys, values = self.cache_k, self.cache_v

        keys = keys.transpose(1, 2)  # BxHxnum_tokens,head_dim
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # We need to transpose head_dim and num_tokens for keys. alpha = BxNxTqxTk
        attn_scores = torch.matmul(queries, keys.transpose(-1, -2))
        d_k = keys.shape[-1]

        # Masking
        key_tokens = attn_scores.shape[-1]
        if (
            self.causal_mask
            and key_tokens == num_tokens
            and key_tokens <= self.mask.shape[1]
            and num_tokens <= self.mask.shape[0]
        ):
            mask = self.mask[:num_tokens, :key_tokens]
        elif self.causal_mask:
            mask = _causal_disallow_mask(
                num_tokens, key_tokens, self.left_context, attn_scores.device
            )
        else:
            mask = self.mask[:num_tokens, :key_tokens]
        masked = attn_scores.masked_fill(mask.bool(), -torch.inf)

        attn_weights = torch.softmax(masked / d_k**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        output = torch.matmul(attn_weights, values)  # B xH x num_tokens x head_dim

        output = output.transpose(1, 2)  # Bxnum_tokens x Hx head_dim

        output = output.contiguous().view(B, num_tokens, self.d_out)

        output = self.out_proj(output)

        return output

    def reset_cache(self):
        self.cache_k = None
        self.cache_v = None


class TransformerBlock(nn.Module):
    def __init__(self, cfg: TransformerEncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.pre_norm = LayerNorm(self.cfg.d_model)
        self.attn_impl = _resolve_attn_impl(cfg.attn_impl)
        if self.attn_impl == "te":
            self.mha = MultiHeadAttentionWithTE(
                dim_in=self.cfg.d_model,
                dim_out=self.cfg.d_model,
                dropout=self.cfg.drop_rate,
                qkv_bias=self.cfg.qkv_bias,
                num_heads=self.cfg.n_heads,
                causal_mask=self.cfg.causal_mask,
                left_context=self.cfg.left_context,
                qk_norm=self.cfg.qk_norm,
            )
        elif self.attn_impl == "sdpa":
            self.mha = MultiHeadAttentionWithSDPA(
                dim_in=self.cfg.d_model,
                dim_out=self.cfg.d_model,
                dropout=self.cfg.drop_rate,
                qkv_bias=self.cfg.qkv_bias,
                num_heads=self.cfg.n_heads,
                causal_mask=self.cfg.causal_mask,
                left_context=self.cfg.left_context,
                qk_norm=self.cfg.qk_norm,
            )
        elif self.attn_impl == "fa":
            self.mha = MultiHeadAttentionWithFA(
                dim_in=self.cfg.d_model,
                dim_out=self.cfg.d_model,
                dropout=self.cfg.drop_rate,
                qkv_bias=self.cfg.qkv_bias,
                num_heads=self.cfg.n_heads,
                causal_mask=self.cfg.causal_mask,
                left_context=self.cfg.left_context,
            )
        else:
            raise ValueError(
                f"Unknown attn_impl={cfg.attn_impl!r}; expected 'auto', 'te', 'sdpa', or 'fa'."
            )
        self.dropout = nn.Dropout(self.cfg.drop_rate)
        self.post_norm = LayerNorm(self.cfg.d_model)
        self.ffn = FeedForward(self.cfg.d_model)

    def forward(self, x, attn_mask=None, lengths=None, packed_seq_params=None, use_cache=False):
        pre_norm = self.pre_norm(x)

        if self.attn_impl == "te":
            # When the encoder packed once at the boundary, x is (Ttot, D) and
            # packed_seq_params carries cu_seqlens. Otherwise fall back to the
            # unpacked (B, T, D) + lengths path inside MHA.
            if packed_seq_params is not None:
                attn_output = self.mha(pre_norm, packed_seq_params=packed_seq_params)
            else:
                attn_output = self.mha(pre_norm, lengths=lengths)
        elif self.attn_impl == "sdpa":
            attn_output = self.mha(pre_norm, attn_mask=attn_mask, use_cache=use_cache)
        else:  # "fa"
            attn_output = self.mha(pre_norm)

        attn_output = x + self.dropout(attn_output)

        post_norm = self.post_norm(attn_output)
        ffn = self.ffn(post_norm)
        output = attn_output + self.dropout(ffn)
        return output

    def reset_cache(self):
        """Reset KV cache on the attention module if it supports it (e.g. MultiHeadAttention)."""
        reset = getattr(self.mha, "reset_cache", None)
        if callable(reset):
            reset()


class ConvSubsampling(nn.Module):
    def __init__(self, n_mels: int = 80, d_model: int = 512):
        super().__init__()
        self.conv1 = Conv1d(n_mels, d_model, kernel_size=3, padding=1)
        # Decreases the temporal dimension by 2
        self.conv2 = Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1)
        # Decreases the temporal dimension by 2
        self.conv3 = Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1)
        self.gelu = TorchGELU()

    def forward(self, x, length):
        x = self.conv1(x)
        x = self.gelu(x)
        x = self.conv2(x)
        x = self.gelu(x)
        length = length // 2
        x = self.conv3(x)
        x = self.gelu(x)
        length = length // 2
        x = x.transpose(1, 2)  # (B, d_model, T) -> (B, T, d_model)
        return x, length


class DepthwiseConvSubsampling(nn.Module):
    """Depthwise separable conv subsampling: reduces params by replacing standard Conv1d
    with depthwise (groups=channels) + pointwise (1x1) convolutions for the strided layers.
    """

    def __init__(self, n_mels: int = 80, d_model: int = 512):
        super().__init__()
        # Standard conv to project from n_mels to d_model
        self.conv1 = Conv1d(n_mels, d_model, kernel_size=3, padding=1)
        # Depthwise separable conv block 1 (stride=2)
        self.dw_conv2 = Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1, groups=d_model)
        self.pw_conv2 = Conv1d(d_model, d_model, kernel_size=1)
        # Depthwise separable conv block 2 (stride=2)
        self.dw_conv3 = Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1, groups=d_model)
        self.pw_conv3 = Conv1d(d_model, d_model, kernel_size=1)
        self.gelu = TorchGELU()

    def forward(self, x, length):
        x = self.conv1(x)
        x = self.gelu(x)
        x = self.dw_conv2(x)
        x = self.pw_conv2(x)
        x = self.gelu(x)
        length = length // 2
        x = self.dw_conv3(x)
        x = self.pw_conv3(x)
        x = self.gelu(x)
        length = length // 2
        x = x.transpose(1, 2)  # (B, d_model, T) -> (B, T, d_model)
        return x, length


class NGPTStackingSubsampling(torch.nn.Module):
    """Stacking subsampling which simply stacks consecutive frames to reduce the sampling rate
    Args:
        subsampling_factor (int): The subsampling factor
        feat_in (int): size of the input features
        feat_out (int): size of the output features
    """

    def __init__(
        self, subsampling_factor: int, feat_in: int, feat_out: int, use_bias: bool = False
    ):
        super().__init__()
        self.subsampling_factor = subsampling_factor
        self.proj_out = torch.nn.Linear(subsampling_factor * feat_in, feat_out, bias=use_bias)
        self.pad_frame = nn.Parameter(torch.ones(feat_in, dtype=torch.float32))

    def forward(self, x, length):
        """
        Args:
            x (torch.Tensor): (B, C, T)
            length (torch.Tensor): (B,)
        Returns:
            x (torch.Tensor): (B, T', D_model)
            length (torch.Tensor): (B,)
        """
        x = x.transpose(1, 2)  # BxCxT -> BxTxC
        b, t, h = x.size()
        pad_size = (
            self.subsampling_factor - (t % self.subsampling_factor)
        ) % self.subsampling_factor
        length = torch.div(
            length + self.subsampling_factor - 1, self.subsampling_factor, rounding_mode='floor'
        )

        # Pad and fill padding frames (all-zero) with a learnable padding 'embedding'
        x = torch.nn.functional.pad(x, (0, 0, 0, pad_size))
        x[(x == 0).all(dim=-1)] = self.pad_frame

        _, t, _ = x.size()
        x = torch.reshape(x, (b, t // self.subsampling_factor, h * self.subsampling_factor))
        x = self.proj_out(x)

        return x, length


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        n_mels: int = 80,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 17,
        drop_rate: float = 0.1,
        qkv_bias: bool = False,
        causal_mask: bool = False,
        pre_encode: str = "conv",  # "conv" or "stacking"
        nan_debug: bool = True,
        qk_norm: bool = False,
        subsampling_factor: int = 4,
        attn_impl: str = "auto",
        recompute_layers: bool = False,
        left_context: Optional[int] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.nan_debug = nan_debug
        self.recompute_layers = recompute_layers
        self.left_context = _normalize_left_context(left_context)
        if self.left_context is not None and not causal_mask:
            raise ValueError("left_context requires causal_mask=True")
        if pre_encode == "conv":
            self.pre_encode = ConvSubsampling(n_mels, d_model)
        elif pre_encode == "depth_conv":
            self.pre_encode = DepthwiseConvSubsampling(n_mels, d_model)
        elif pre_encode == "stacking":
            self.pre_encode = NGPTStackingSubsampling(
                subsampling_factor=subsampling_factor, feat_in=n_mels, feat_out=d_model
            )
        else:
            raise ValueError(
                f"Invalid pre_encode: {pre_encode}. Choose from: conv, depth_conv, stacking"
            )

        cfg = TransformerEncoderConfig(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            drop_rate=drop_rate,
            qkv_bias=qkv_bias,
            causal_mask=causal_mask,
            qk_norm=qk_norm,
            attn_impl=attn_impl,
            recompute_layers=recompute_layers,
            left_context=self.left_context,
        )
        self.attn_impl = _resolve_attn_impl(cfg.attn_impl)
        self.layers = nn.ModuleList([TransformerBlock(cfg) for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model)
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, audio_signal, length, packed_seq_params=None, return_packed: bool = False):
        """
        Args:
            audio_signal (torch.Tensor): (B, C, T) audio features.
            length (torch.Tensor): (B,) input frame counts.
            packed_seq_params: optional caller-supplied PackedSeqParams. Unused by
                production callers today; reserved for future dataloader-side
                packing. When None and attn_impl=="te", the encoder builds its own
                PackedSeqParams from `length` and runs the block stack on packed
                features for efficiency.
        Returns:
            x (torch.Tensor): (B, D_model, T') with zero-padded positions when
                ``return_packed`` is false, otherwise ``(Ttot, D_model)`` with
                only valid post-subsampling positions.
            length (torch.Tensor): (B,) post-subsampling lengths.
        """
        from megatron.core.packed_seq_params import PackedSeqParams

        x = audio_signal
        x, length = self.pre_encode(x, length)
        if self.nan_debug:
            self._check_nan(x, "pre_encode")
        x = x * (self.d_model**0.5)
        if self.nan_debug:
            self._check_nan(x, "embedding_scale")
        x = self.layer_norm(x)
        if self.nan_debug:
            self._check_nan(x, "layer_norm")

        B, T_prime, _ = x.shape

        if self.attn_impl == "te" and packed_seq_params is None:
            # Pack once: drop padded rows, build cu_seqlens, run blocks on (Ttot, D),
            # scatter back at exit. final_norm and the block ops are per-token so
            # they run unchanged on a 2-D tensor.
            lengths32 = length.to(torch.int32)
            max_seqlen = int(lengths32.max().item())

            if max_seqlen == 0:
                # All-empty batch (e.g. text-only minibatch on this DP rank).
                # Skip the block stack entirely and emit zeros to avoid wasted
                # CPU launches and to keep the TE attention guard from firing
                # layer-by-layer.
                if return_packed:
                    x = x.new_zeros(0, self.d_model)
                else:
                    x = x.new_zeros(B, T_prime, self.d_model)
            else:
                valid_mask = torch.arange(T_prime, device=x.device).unsqueeze(
                    0
                ) < lengths32.unsqueeze(1)
                cu_seqlens = torch.nn.functional.pad(lengths32.cumsum(0).to(torch.int32), (1, 0))
                psp = PackedSeqParams(
                    qkv_format="thd",
                    cu_seqlens_q=cu_seqlens,
                    cu_seqlens_kv=cu_seqlens,
                    max_seqlen_q=max_seqlen,
                    max_seqlen_kv=max_seqlen,
                )
                x_packed = x[valid_mask]  # (Ttot, D)

                for idx, layer in enumerate(self.layers):
                    if self.recompute_layers and self.training and x_packed.requires_grad:
                        x_packed = checkpoint(
                            lambda hidden, layer=layer: layer(hidden, packed_seq_params=psp),
                            x_packed,
                            use_reentrant=False,
                        )
                    else:
                        x_packed = layer(x_packed, packed_seq_params=psp)
                    if self.nan_debug:
                        self._check_nan(x_packed, f"layer_{idx}")

                x_packed = self.final_norm(x_packed)
                if self.nan_debug:
                    self._check_nan(x_packed, "final_norm")

                if return_packed:
                    x = x_packed
                else:
                    x = x.new_zeros(B, T_prime, self.d_model)
                    x[valid_mask] = x_packed
        else:
            # Unpacked path: SDPA / FA backends, or TE with caller-supplied packing.
            max_len = x.shape[1]
            pad_mask = torch.arange(max_len, device=x.device).unsqueeze(0) < length.unsqueeze(1)
            attn_mask = pad_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T)

            for idx, layer in enumerate(self.layers):
                if self.recompute_layers and self.training and x.requires_grad:
                    x = checkpoint(
                        lambda hidden, layer=layer: layer(
                            hidden,
                            attn_mask=attn_mask,
                            lengths=length,
                            packed_seq_params=packed_seq_params,
                        ),
                        x,
                        use_reentrant=False,
                    )
                else:
                    x = layer(
                        x, attn_mask=attn_mask, lengths=length, packed_seq_params=packed_seq_params
                    )
                if self.nan_debug:
                    self._check_nan(x, f"layer_{idx}")
            x = self.final_norm(x)
            if self.nan_debug:
                self._check_nan(x, "final_norm")
            if return_packed:
                lengths32 = length.to(torch.int32)
                valid_mask = torch.arange(x.shape[1], device=x.device).unsqueeze(
                    0
                ) < lengths32.unsqueeze(1)
                x = x[valid_mask]

        if return_packed:
            return x, length

        x = x.transpose(1, 2)  # BxT'xD_model -> BxD_modelxT'
        return x, length

    def _check_nan(self, x, name):
        has_nan = torch.isnan(x).any().item()
        has_inf = torch.isinf(x).any().item()
        if has_nan or has_inf:
            nan_count = torch.isnan(x).sum().item()
            inf_count = torch.isinf(x).sum().item()
            valid = x[~(torch.isnan(x) | torch.isinf(x))]
            abs_max = valid.abs().max().item() if valid.numel() > 0 else float('nan')
            print(
                f"[NaN DEBUG] {name}: NaN={nan_count}, Inf={inf_count}, "
                f"abs_max={abs_max:.6f}, shape={list(x.shape)}",
                flush=True,
            )
            raise RuntimeError(f"[NaN DEBUG] NaN/Inf detected at '{name}'. Stopping training.")

    def reset_cache(self):
        """Reset KV cache on every block (no-op for attention impls without cache)."""
        for layer in self.layers:
            reset = getattr(layer, "reset_cache", None)
            if callable(reset):
                reset()

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self, partial=False):
        for param in self.parameters():
            param.requires_grad = True
