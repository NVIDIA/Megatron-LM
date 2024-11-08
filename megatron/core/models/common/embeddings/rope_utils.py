# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple, Union

if TYPE_CHECKING:
    from megatron.core.transformer.transformer_config import TransformerConfig

import logging

import torch
from torch import Tensor

from megatron.core import parallel_state

logger = logging.getLogger(__name__)

try:
    from apex.transformer.functional import (
        fused_apply_rotary_pos_emb,
        fused_apply_rotary_pos_emb_thd,
    )

    HAVE_APPLY_ROPE_FUSION = True
except ImportError:
    HAVE_APPLY_ROPE_FUSION = False

try:
    import transformer_engine.pytorch.cpp_extensions as tex
    HAVE_TE = True
except ImportError:
    HAVE_TE = False

def get_pos_emb_on_this_cp_rank(pos_emb: Tensor, seq_dim: int) -> Tensor:
    """Get the position embedding on the current context parallel rank.

    Args:
        pos_emb (Tensor): Positional embedding tensor
        seq_dim (int): Sequence dimension
    """
    cp_size = parallel_state.get_context_parallel_world_size()
    cp_rank = parallel_state.get_context_parallel_rank()
    cp_idx = torch.tensor(
        [cp_rank, (2 * cp_size - cp_rank - 1)], device="cpu", pin_memory=True
    ).cuda(non_blocking=True)
    pos_emb = pos_emb.view(
        *pos_emb.shape[:seq_dim], 2 * cp_size, -1, *pos_emb.shape[(seq_dim + 1) :]
    )
    pos_emb = pos_emb.index_select(seq_dim, cp_idx)
    pos_emb = pos_emb.view(*pos_emb.shape[:seq_dim], -1, *pos_emb.shape[(seq_dim + 2) :])
    return pos_emb


def _rotate_half(x: Tensor, rotary_interleaved: bool) -> Tensor:
    """Change sign so the last dimension becomes [-odd, +even]

    Args:
        x (Tensor): Input tensor

    Returns:
        Tensor: Tensor rotated half
    """
    if not rotary_interleaved:
        x1, x2 = torch.chunk(x, 2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    else:
        x1 = x[:, :, :, ::2]
        x2 = x[:, :, :, 1::2]
        x_new = torch.stack((-x2, x1), dim=-1)
        return x_new.view(x_new.shape[0], x_new.shape[1], x_new.shape[2], -1)


def _apply_rotary_pos_emb_bshd(
    t: Tensor,
    freqs: Tensor,
    rotary_interleaved: bool = False,
    multi_latent_attention: bool = False,
    mscale: float = 1.0,
) -> Tensor:
    """Apply rotary positional embedding to input tensor T.

    check https://kexue.fm/archives/8265 for detailed formulas

    Args:
        t (Tensor): Input tensor T is of shape [seq_length, ... , dim]
        freqs (Tensor): Rotary Positional embedding tensor freq is of shape [seq_length, ..., dim]

    Returns:
        Tensor: The input tensor after applying RoPE
    """
    rot_dim = freqs.shape[-1]

    # ideally t_pass is empty so rotary pos embedding is applied to all tensor t
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]

    if multi_latent_attention:
        x1 = t[..., 0::2]
        x2 = t[..., 1::2]
        t = torch.cat((x1, x2), dim=-1)

    # first part is cosine component
    # second part is sine component, need to change signs with _rotate_half method
    cos_ = (torch.cos(freqs) * mscale).to(t.dtype)
    sin_ = (torch.sin(freqs) * mscale).to(t.dtype)

    t = (t * cos_) + (_rotate_half(t, rotary_interleaved) * sin_)
    return torch.cat((t, t_pass), dim=-1)


def _apply_rotary_pos_emb_thd(
    t: Tensor,
    cu_seqlens: Tensor,
    freqs: Tensor,
    rotary_interleaved: bool = False,
    multi_latent_attention: bool = False,
    mscale: float = 1.0,
) -> Tensor:
    """A baseline implementation of applying RoPE for `thd` format.

    Args:
        t (Tensor): Input tensor T is of shape [t, h, d]
        cu_seqlens(Tensor):  Cumulative sum of sequence lengths in a batch for `t`,
        with shape [b + 1] and dtype torch.int32.
        freqs (Tensor): Rotary Positional embedding tensor freq is of shape [max_s, 1, 1, d]

    Returns:
        Tensor: Shape [t, h, d]. The input tensor after applying RoPE.
    """

    seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
    return torch.cat(
        [
            _apply_rotary_pos_emb_bshd(
                x.unsqueeze(1),
                freqs[: x.size(0)],
                rotary_interleaved=rotary_interleaved,
                multi_latent_attention=multi_latent_attention,
                mscale=mscale,
            )
            for x in torch.split(t, seqlens)
        ]
    ).squeeze(1)


def apply_rotary_pos_emb(
    t: Tensor,
    freqs: Tensor,
    config: TransformerConfig,
    cu_seqlens: Optional[Tensor] = None,
    mscale: float = 1.0,
):
    """
    Reroute to the appropriate apply_rotary_pos_emb function depending on
    fused/unfused kernels, or bshd (conventional) / thd (packed seq) format
    """
    if not config.disable_te_fused_rope and HAVE_TE and torch.cuda.is_available() and torch.version.hip:
        return apply_rotary_pos_emb_fused_te(t = t, freqs = freqs, config = config, cu_seqlens = cu_seqlens)
    else:
        if config.apply_rope_fusion and not HAVE_APPLY_ROPE_FUSION:
            # setting apply_rope_fusion in config to False
            # so that subsequent queries to this config also return False
            config.apply_rope_fusion = False
            if not getattr(apply_rotary_pos_emb, "printed_fused_warning", False):
                logger.warning(
                    "Setting apply_rope_fusion to false because its implementation"
                    " is not included in Apex. Try upgrading to the latest version"
                )
                apply_rotary_pos_emb.printed_fused_warning = True

        if getattr(config, "multi_latent_attention", False) and config.rotary_interleaved:
            logger.warning(
                "rotary_interleaved is not supported with multi_latent_attention, setting it to False"
            )
            config.rotary_interleaved = False

        if config.apply_rope_fusion:
            if cu_seqlens is None:
                return fused_apply_rotary_pos_emb(t, freqs, transpose_output_memory=True)
            else:
                return fused_apply_rotary_pos_emb_thd(t, cu_seqlens, freqs)
        else:
            if cu_seqlens is None:
                return _apply_rotary_pos_emb_bshd(
                    t,
                    freqs,
                    rotary_interleaved=config.rotary_interleaved,
                    multi_latent_attention=config.multi_latent_attention,
                    mscale=mscale,
                )
            else:
                return _apply_rotary_pos_emb_thd(
                    t,
                    cu_seqlens,
                    freqs,
                    rotary_interleaved=config.rotary_interleaved,
                    multi_latent_attention=config.multi_latent_attention,
                    mscale=mscale,
                )

class FusedRoPEFunc(torch.autograd.Function):
    """
    Function for FusedRoPE

    This implementation assumes the input tensor to be in `sbhd`, `bshd` or `thd` format and
    the RoPE tensor to be of shape (s, 1, 1, d). It accepts arbitrary memory layouts to avoid
    the expensive `.contiguous()` calls, thus it may not achieve the best memory access pattern.
    """

    @staticmethod
    def forward(
        ctx,
        t: torch.Tensor,
        freqs: torch.Tensor,
        tensor_format: str = "sbhd",
        cu_seqlens: Union[torch.Tensor, None] = None,
    ) -> torch.Tensor:
        if freqs.dtype != torch.float32:
            freqs = freqs.float()
        if tensor_format == "sbhd":
            output = tex.fused_rope_forward(t, freqs, False)
        elif tensor_format == "bshd":
            output = tex.fused_rope_forward(
                t.transpose(0, 1), freqs, True
            ).transpose(0, 1)
        elif tensor_format == "thd":
            output = tex.fused_rope_thd_forward(t, cu_seqlens, freqs)
        else:
            raise ValueError(f"Unsupported tensor_format: {tensor_format}.")
        ctx.save_for_backward(freqs, cu_seqlens)
        ctx.tensor_format = tensor_format
        return output

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> Tuple[Union[torch.Tensor, None], ...]:
        freqs, cu_seqlens = ctx.saved_tensors
        if ctx.tensor_format == "sbhd":
            grad_input = tex.fused_rope_backward(grad_output, freqs, False)
        elif ctx.tensor_format == "bshd":
            grad_input = tex.fused_rope_backward(
                grad_output.transpose(0, 1), freqs, True
            ).transpose(0, 1)
        elif ctx.tensor_format == "thd":
            grad_input = tex.fused_rope_thd_backward(grad_output, cu_seqlens, freqs)
        else:
            raise ValueError(f"Unsupported tensor_format: {ctx.tensor_format}.")

        return grad_input, None, None, None, None


def apply_rotary_pos_emb_fused_te(
    t: torch.Tensor,
    freqs: torch.Tensor,
    tensor_format: str = "sbhd",
    config: TransformerConfig = None,
    fused: bool = True,
    cu_seqlens: Union[torch.Tensor, None] = None,
) -> torch.Tensor:
    """
    Apply rotary positional embedding tensor to the input tensor.

    Parameters
    ----------
    t: torch.Tensor
        Input tensor of shape `[s, b, h, d]`, `[b, s, h, d]` or `[t, h, d]`, on which
        rotary positional embedding will be applied.
    freqs: torch.Tensor
        Rotary positional embedding tensor of shape `[s2, 1, 1, d2]` and dtype 'float',
        with `s2 >= s` and `d2 <= d`.
    fused: bool, default = False
        Whether to use a fused applying RoPE implementation.
    tensor_format: {'sbhd', 'bshd', 'thd'}, default = 'sbhd'
        is `bshd` if `t` is of shape `[bs, seq, ...]`, or `sbhd` if `t` is
        of shape `[seq, bs, ...]`. 'thd' is only supported when `fused` is True.
    cu_seqlens: torch.Tensor, default = None.
        Cumulative sum of sequence lengths in a batch for `t`, with shape [b + 1] and
        dtype torch.int32. Only valid when `tensor_format` is 'thd'.
    """

    if fused:
        assert (
            tensor_format != "thd" or cu_seqlens is not None
        ), "cu_seqlens must not be None when tensor_format is 'thd'."
        return FusedRoPEFunc.apply(t, freqs, tensor_format, cu_seqlens)

    assert tensor_format in ("sbhd", "bshd"), (
        "Only formats `sbhd` or `bshd` are supported for input tensor `t` "
        f"when fused is False, got {tensor_format}."
    )

    max_seq_len = freqs.shape[0]
    cur_seq_len = t.shape[1] if tensor_format == "bshd" else t.shape[0]

    # Only apply the rotary embeddings up to the sequence length of the running
    # input.
    assert cur_seq_len <= max_seq_len, (
        f"Rotary Embeddings only supported up to {max_seq_len} sequence length!"
    )
    freqs = freqs[:cur_seq_len]
    if tensor_format == "bshd":
        freqs = freqs.transpose(0, 1)  # [seq, 1, 1, dim] -> [1, seq, 1, dim]
    # cos/sin first then dtype conversion for better precision
    cos_ = torch.cos(freqs).to(t.dtype)
    sin_ = torch.sin(freqs).to(t.dtype)

    rot_dim = freqs.shape[-1]
    # ideally t_pass is empty so rotary pos embedding is applied to all tensor t
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]

    # first part is cosine component
    # second part is sine component, need to change signs with _rotate_half method
    t = (t * cos_) + (_rotate_half(t) * sin_)
    return torch.cat((t, t_pass), dim=-1)
