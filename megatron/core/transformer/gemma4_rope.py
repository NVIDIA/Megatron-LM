# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import torch


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input (GPT-NeoX style, split at dim/2)."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim: int = 2) -> torch.Tensor:
    """Apply rotary embedding to ``x`` ([B, S, H, hd] by default) with full-width rotate_half.

    Mirrors HF ``apply_rotary_pos_emb`` (modeling_gemma4.py:787-806). cos/sin are
    [B, S, hd]; ``unsqueeze_dim=2`` broadcasts them across the head dimension.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (x * cos) + (rotate_half(x) * sin)


class Gemma4RotaryEmbedding(torch.nn.Module):
    """Per-layer-type rotary embedding for Gemma 4 (modeling_gemma4.py:1087-1174).

    Builds two ``inv_freq`` sets:
      * ``sliding``: base 1e4, head_dim 256, full rotary (128 nonzero freqs).
      * ``full``: base 1e6, head_dim 512, *proportional* RoPE -- inv_freq has
        length 256 where the first 64 entries are ``1/(1e6 ** (arange(0,128,2)/512))``
        and the remaining 192 are exactly 0.0 (NoPE pass-through channels).

    cos/sin span the full head_dim (cat(freqs, freqs)); attention_scaling is 1.0
    for both types. Frequencies are computed in fp32 (autocast disabled) and cast
    to ``x.dtype`` at the end, matching HF exactly.
    """

    def __init__(
        self,
        sliding_head_dim: int = 256,
        full_head_dim: int = 512,
        sliding_base: float = 1e4,
        full_base: float = 1e6,
        partial_rotary_factor: float = 0.25,
    ):
        super().__init__()
        # Keep the inv_freq construction args so forward() can rebuild inv_freq in fp32.
        # The registered buffers below are downcast to bf16 by a model-level
        # ``.bfloat16()``/``.half()`` (e.g. ``model.bfloat16()`` at build time). A bf16
        # inv_freq corrupts the rope frequencies -> cos/sin land ~1e-2 off HF (which
        # keeps inv_freq fp32), which then propagates through q/k RoPE into the whole
        # forward. Rebuilding from these args in forward keeps the tables fp32-faithful.
        self._sliding_inv_freq_args = (sliding_head_dim, sliding_base)
        self._full_inv_freq_args = (full_head_dim, full_base, partial_rotary_factor)
        self.register_buffer(
            "sliding_inv_freq", self._default_inv_freq(sliding_head_dim, sliding_base), persistent=False
        )
        self.register_buffer(
            "full_inv_freq",
            self._proportional_inv_freq(full_head_dim, full_base, partial_rotary_factor),
            persistent=False,
        )

    def _apply(self, fn, *args, **kwargs):
        """Keep the rope ``inv_freq`` buffers in fp32 across module-wide casts.

        A blanket ``model.bfloat16()``/``.half()`` (e.g. at build time) would otherwise
        downcast these buffers to bf16, corrupting the rope frequencies (cos/sin land
        ~1e-2 off vs HF, which keeps inv_freq fp32) and skewing the whole forward. After
        the default ``_apply`` (which handles device moves), re-materialize inv_freq in
        fp32 on the current device — upcasting an already-bf16 buffer cannot recover the
        lost precision, so we rebuild from the construction args.
        """
        out = super()._apply(fn, *args, **kwargs)
        dev = out.sliding_inv_freq.device
        out.sliding_inv_freq = out._default_inv_freq(*out._sliding_inv_freq_args).to(dev)
        out.full_inv_freq = out._proportional_inv_freq(*out._full_inv_freq_args).to(dev)
        return out

    @staticmethod
    def _default_inv_freq(head_dim: int, base: float) -> torch.Tensor:
        return 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.int64).float() / head_dim))

    @staticmethod
    def _proportional_inv_freq(head_dim: int, base: float, partial_rotary_factor: float) -> torch.Tensor:
        rope_angles = int(partial_rotary_factor * head_dim // 2)
        inv_freq_rotated = 1.0 / (
            base ** (torch.arange(0, 2 * rope_angles, 2, dtype=torch.int64).float() / head_dim)
        )
        nope_angles = head_dim // 2 - rope_angles
        if nope_angles > 0:
            return torch.cat((inv_freq_rotated, torch.zeros(nope_angles, dtype=torch.float32)), dim=0)
        return inv_freq_rotated

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor, layer_type: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (cos, sin) of width ``2 * len(inv_freq)`` for the given layer type.

        ``layer_type`` is "sliding" or "full". cos/sin are returned in ``x.dtype``.
        """
        inv_freq = getattr(self, f"{layer_type}_inv_freq")
        inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # force fp32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
