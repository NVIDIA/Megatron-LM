# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import math
from typing import Any

import torch
import torch.nn as nn
import transformer_engine.pytorch as te
# Zero-copy imports of the DSv4 THD-CP helpers that live in Megatron Core. The
# lite CSA module reuses Core's differentiable kernels, CP row-mapping utilities,
# and CuTeDSL layout kernels rather than vendoring them; see the module docstring
# of ``csa_cp_utils`` / ``csa_cp_layout_kernels`` for the exact contracts.
from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region
from megatron.core.transformer.experimental_attention_variant import (
    csa_cp_layout_kernels,
    csa_cp_utils as cp_utils,
)
from megatron.core.transformer.experimental_attention_variant.csa import (
    _unfused_indexer_sparse_attn_from_topk,
    unfused_compressed_sparse_attn,
)
from megatron.core.transformer.experimental_attention_variant.csa_kernels import (
    FusedCSAIndexerSparseAttnFromTopkFunc,
    csa_sparse_attn,
)
from megatron.core.transformer.experimental_attention_variant.dsa import (
    DSAIndexerLossAutoScaler,
    DSAIndexerLossLoggingHelper,
)
from megatron.lite.primitive.modules.attention.dsa import rotate_activation
from megatron.lite.primitive.parallel.state import ParallelState
from megatron.lite.primitive.utils.rotary import (
    _yarn_find_correction_range,
    _yarn_linear_ramp_mask,
)


class GroupedLinear(nn.Module):
    def __init__(self, in_features_per_group: int, out_features: int, n_groups: int):
        super().__init__()
        self.in_features_per_group = in_features_per_group
        self.out_features = out_features
        self.n_groups = n_groups
        self.weight = nn.Parameter(torch.empty(out_features, in_features_per_group))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_per_group = self.out_features // self.n_groups
        weight = self.weight.view(self.n_groups, out_per_group, self.in_features_per_group)
        return torch.einsum("...gd,god->...go", x, weight)


def build_rope_cos_sin(
    position_ids: torch.Tensor,
    rope_head_dim: int,
    rope_theta: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (
        rope_theta
        ** (torch.arange(0, rope_head_dim, 2, device=device, dtype=torch.float32) / rope_head_dim)
    )
    freqs = torch.einsum("bs,d->bsd", position_ids.to(torch.float32), inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos().to(dtype=dtype), emb.sin().to(dtype=dtype)


def build_yarn_rope_cos_sin(
    position_ids: torch.Tensor,
    rope_head_dim: int,
    rope_theta: float,
    *,
    config: Any,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    dim = rope_head_dim
    inv_freq_extra = 1.0 / (
        rope_theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim)
    )
    inv_freq_inter = 1.0 / (
        config.rotary_scaling_factor
        * rope_theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim)
    )
    low, high = _yarn_find_correction_range(
        config.beta_fast,
        config.beta_slow,
        dim,
        rope_theta,
        config.original_max_position_embeddings,
    )
    inv_freq_mask = 1.0 - _yarn_linear_ramp_mask(low, high, dim // 2, device)
    inv_freq = inv_freq_inter * (1 - inv_freq_mask) + inv_freq_extra * inv_freq_mask
    freqs = torch.einsum("bs,d->bsd", position_ids.to(torch.float32), inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos().to(dtype=dtype), emb.sin().to(dtype=dtype)


def build_compressed_rope_cos_sin(
    position_ids: torch.Tensor,
    rope_head_dim: int,
    rope_theta: float,
    *,
    config: Any,
    use_yarn: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    if use_yarn:
        return build_yarn_rope_cos_sin(
            position_ids,
            rope_head_dim,
            rope_theta,
            config=config,
            device=device,
            dtype=dtype,
        )
    return build_rope_cos_sin(position_ids, rope_head_dim, rope_theta, device=device, dtype=dtype)


def apply_partial_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rope_head_dim: int
) -> torch.Tensor:
    if rope_head_dim == 0:
        return x
    rope = x[..., -rope_head_dim:]
    tail = x[..., :-rope_head_dim]
    rope_pairs = rope.unflatten(-1, (-1, 2))
    a, b = rope_pairs[..., 0], rope_pairs[..., 1]
    c = cos[..., : rope_head_dim // 2]
    s = sin[..., : rope_head_dim // 2]
    while c.ndim < a.ndim:
        c = c.unsqueeze(1)
        s = s.unsqueeze(1)
    out_a = a * c - b * s
    out_b = a * s + b * c
    rope_out = torch.stack([out_a, out_b], dim=-1).flatten(-2)
    return torch.cat([tail, rope_out], dim=-1)


class CompressedSequenceCompressor(nn.Module):
    def __init__(self, config: Any, compress_ratio: int, head_dim: int, *, rotate: bool = False):
        super().__init__()
        self.config = config
        self.compress_ratio = compress_ratio
        self.head_dim = head_dim
        self.rope_head_dim = min(config.qk_rope_head_dim, head_dim)
        self.overlap = compress_ratio == 4
        self.coff = 2 if self.overlap else 1
        self.rotate = rotate
        self.wkv = nn.Linear(config.hidden_size, self.coff * head_dim, bias=False)
        self.wgate = nn.Linear(config.hidden_size, self.coff * head_dim, bias=False)
        self.ape = nn.Parameter(
            torch.empty(compress_ratio, self.coff * head_dim, dtype=torch.float32)
        )
        self.norm = te.RMSNorm(head_dim, eps=config.rms_norm_eps)
        nn.init.normal_(self.ape, mean=0.0, std=config.initializer_range)

    def _overlap_transform(self, tensor: torch.Tensor, fill_value: float) -> torch.Tensor:
        bsz, n_blocks, ratio, _, head_dim = tensor.shape
        out = tensor.new_full((bsz, n_blocks, 2 * ratio, head_dim), fill_value)
        out[:, :, ratio:] = tensor[:, :, :, 1]
        out[:, 1:, :ratio] = tensor[:, :-1, :, 0]
        return out

    def forward(
        self, x: torch.Tensor, *, position_ids: torch.Tensor, rope_theta: float
    ) -> torch.Tensor | None:
        bsz, seq_len, _ = x.shape
        ratio = self.compress_ratio
        n_blocks = seq_len // ratio
        if n_blocks == 0:
            return None
        cutoff = n_blocks * ratio
        content = self.wkv(x[:, :cutoff])
        gate = self.wgate(x[:, :cutoff])
        content = content.view(bsz, n_blocks, ratio, self.coff, self.head_dim)
        gate = gate.view_as(content)
        gate = gate + self.ape.view(1, 1, ratio, self.coff, self.head_dim).to(gate.device)
        if self.overlap:
            content = self._overlap_transform(content, 0.0)
            gate = self._overlap_transform(gate, float("-inf"))
        else:
            content = content.squeeze(3)
            gate = gate.squeeze(3)
        weights = torch.softmax(gate.float(), dim=2).to(dtype=content.dtype)
        compressed = self.norm((content * weights).sum(dim=2)).unsqueeze(1)
        compressed_positions = position_ids[:, :cutoff:ratio]
        cos, sin = build_compressed_rope_cos_sin(
            compressed_positions,
            self.rope_head_dim,
            rope_theta,
            config=self.config,
            use_yarn=self.compress_ratio > 1,
            device=x.device,
            dtype=compressed.dtype,
        )
        compressed = apply_partial_rope(compressed, cos, sin, self.rope_head_dim)
        return rotate_activation(compressed) if self.rotate else compressed

    def _overlap_transform_thd(
        self, tensor: torch.Tensor, is_first_in_seg: torch.Tensor, fill_value: float
    ) -> torch.Tensor:
        """Batched overlapping-window transform for the THD (pre-grouped) layout.

        Mirrors Core ``Compressor._overlap_transform_thd``: operates on the flat
        ``(total_comp, ratio, 1, coff * head_dim)`` tensor from all segments at
        once. ``is_first_in_seg`` is a ``(total_comp,)`` bool mask, ``True`` for
        each compressed entry that starts a new segment (no predecessor group).
        Output shape ``(total_comp, 2 * ratio, 1, head_dim)``.
        """
        n, ratio, b_dim, _ = tensor.size()
        d = self.head_dim
        out = tensor.new_full((n, 2 * ratio, b_dim, d), fill_value)
        out[:, ratio:] = tensor[:, :, :, d:]
        prev_data = torch.roll(tensor[:, :, :, :d], shifts=1, dims=0)
        prev_data[is_first_in_seg] = fill_value
        out[:, :ratio] = prev_data
        return out

    def _forward_thd(
        self,
        hidden_compact: torch.Tensor,
        cu_seqlens: torch.Tensor,
        *,
        max_seqlen_q: int,
        compressed_group_ids: torch.Tensor,
    ) -> tuple[torch.Tensor | None, None]:
        """Pre-grouped THD compression for the DSv4 CP path.

        ``hidden_compact`` is ``(compact_group_capacity * ratio, 1, hidden)``,
        already packed into ratio-sized groups by
        ``cp_utils.prepare_cp_compressor_input``; ``compressed_group_ids`` is
        ``(compact_group_capacity,)`` int32 giving each compressed group's
        per-sequence compressed id (its RoPE position is ``id * ratio``).

        Reproduces Core ``Compressor._forward_thd`` pre-grouped semantics using
        lite's compressor params (``wkv``/``wgate``/``ape``/``norm``) and lite's
        RoPE convention (``build_compressed_rope_cos_sin`` + ``apply_partial_rope``)
        so the CP path stays numerically identical to lite's BSHD path. ``cu_seqlens``
        and ``max_seqlen_q`` are accepted to match Core's signature; they are only
        needed by Core's fused-RoPE cache, which lite does not use.

        Returns ``(compressed_thd (total_comp, 1, head_dim), None)``.
        """
        del cu_seqlens, max_seqlen_q  # Only used by Core's fused-RoPE cache path.
        ratio = self.compress_ratio
        total_comp = int(compressed_group_ids.shape[0])
        if total_comp == 0:
            return None, None
        kv = self.wkv(hidden_compact)  # (total_comp * ratio, 1, coff * head_dim)
        gate = self.wgate(hidden_compact)
        kv_grouped = kv.reshape(total_comp, ratio, 1, -1)
        gate_grouped = gate.reshape(total_comp, ratio, 1, -1)
        gate_grouped = gate_grouped + self.ape.view(1, ratio, 1, -1).to(gate_grouped.device)
        if self.overlap:
            is_first = compressed_group_ids[:total_comp] == 0
            kv_grouped = self._overlap_transform_thd(kv_grouped, is_first, 0.0)
            gate_grouped = self._overlap_transform_thd(gate_grouped, is_first, float("-inf"))
        # Non-overlap (coff == 1): kv_grouped/gate_grouped are already
        # (total_comp, ratio, 1, head_dim); no windowing needed.
        weights = torch.softmax(gate_grouped.float(), dim=1).to(kv_grouped.dtype)
        compressed = (kv_grouped * weights).sum(dim=1)  # (total_comp, 1, head_dim)
        compressed = self.norm(compressed)
        positions = compressed_group_ids[:total_comp].clamp_min(0).to(torch.long) * ratio
        cos, sin = build_compressed_rope_cos_sin(
            positions.view(1, total_comp),
            self.rope_head_dim,
            self.config.compress_rope_theta,
            config=self.config,
            use_yarn=ratio > 1,
            device=compressed.device,
            dtype=compressed.dtype,
        )
        # apply_partial_rope wants the sequence axis at dim -2; view as (1, total_comp, d).
        compressed = compressed.transpose(0, 1)  # (1, total_comp, head_dim)
        compressed = apply_partial_rope(compressed, cos, sin, self.rope_head_dim)
        compressed = compressed.transpose(0, 1)  # (total_comp, 1, head_dim)
        if self.rotate:
            compressed = rotate_activation(compressed)
        return compressed, None


def _window_topk_indices(
    batch: int, seq_len: int, window: int, *, device: torch.device
) -> torch.Tensor:
    topk = max(1, min(int(window), seq_len))
    query_pos = torch.arange(seq_len, device=device).view(seq_len, 1)
    offsets = torch.arange(topk, device=device)
    indices = (query_pos - topk + 1).clamp(min=0) + offsets
    indices = torch.where(indices > query_pos, -1, indices)
    return indices.unsqueeze(0).expand(batch, -1, -1).to(torch.int32)


def _load_dsa_kernels():
    from megatron.lite.primitive.kernels import dsa_kernels

    return dsa_kernels


class CompressedSparseAttentionIndexer(nn.Module):
    def __init__(self, config, compress_ratio: int):
        super().__init__()
        self.config = config
        self.index_n_heads = config.index_n_heads
        self.index_head_dim = config.index_head_dim
        self.index_topk = config.index_topk
        self.rope_head_dim = min(config.qk_rope_head_dim, config.index_head_dim)
        self.softmax_scale = self.index_head_dim**-0.5
        self.wq_b = nn.Linear(
            config.q_lora_rank, config.index_n_heads * config.index_head_dim, bias=False
        )
        self.weights_proj = nn.Linear(config.hidden_size, config.index_n_heads, bias=False)
        self.compressor = CompressedSequenceCompressor(
            config, compress_ratio, config.index_head_dim, rotate=True
        )


class CompressedSparseAttention(nn.Module):
    def __init__(
        self,
        config,
        *,
        layer_idx: int,
        ps: ParallelState,
        apply_dsa_kernel_fusion: bool = True,
        dsa_indexer_loss_coeff: float = 0.0,
        dsa_indexer_use_sparse_loss: bool = False,
        calculate_per_token_loss: bool = False,
    ):
        super().__init__()
        self.config = config
        self.ps = ps
        self.layer_idx = layer_idx
        self.attention_backend = "torch"
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.softmax_scale = self.head_dim**-0.5
        self.num_heads_per_group = config.num_attention_heads // config.o_groups
        # Kernel / indexer-loss knobs are implementation config (not part of the
        # HF model config), threaded as constructor arguments like GLM-5's DSA.
        # Fused DSA kernels are the production default; the unfused sparse-attn /
        # indexer-loss path (Core's ``_unfused_indexer_sparse_attn_from_topk``) is
        # a debug fallback selected via ``apply_dsa_kernel_fusion=False``.
        self.apply_dsa_kernel_fusion = apply_dsa_kernel_fusion
        self.dsa_indexer_loss_coeff = dsa_indexer_loss_coeff
        self.dsa_indexer_use_sparse_loss = dsa_indexer_use_sparse_loss
        self.calculate_per_token_loss = calculate_per_token_loss
        # MTP layers use layer_idx == num_hidden_layers (+i), which is past the
        # per-decoder-layer compress_ratios list (length num_hidden_layers); fall
        # back to the last real layer's ratio so the MTP CSA still builds.
        if config.compress_ratios:
            _cr_idx = min(layer_idx, len(config.compress_ratios) - 1)
            self.compress_ratio = config.compress_ratios[_cr_idx]
        else:
            self.compress_ratio = 0
        self.wq_a = nn.Linear(config.hidden_size, config.q_lora_rank, bias=False)
        self.q_norm = te.RMSNorm(config.q_lora_rank, eps=config.rms_norm_eps)
        self.wq_b = nn.Linear(config.q_lora_rank, self.num_heads * self.head_dim, bias=False)
        self.wkv = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.kv_norm = te.RMSNorm(config.head_dim, eps=config.rms_norm_eps)
        self.wo_a = GroupedLinear(
            self.num_heads_per_group * self.head_dim,
            config.o_groups * config.o_lora_rank,
            config.o_groups,
        )
        self.wo_b = nn.Linear(config.o_groups * config.o_lora_rank, config.hidden_size, bias=False)
        self.sinks = nn.Parameter(torch.zeros(self.num_heads))
        self.compressor = (
            CompressedSequenceCompressor(config, self.compress_ratio, self.head_dim)
            if self.compress_ratio > 1
            else None
        )
        self.indexer = (
            CompressedSparseAttentionIndexer(config, self.compress_ratio)
            if self.compress_ratio == 4
            else None
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        packed_seq_params: Any = None,
    ) -> torch.Tensor:
        # THD packed sequences take the primary DSv4 CP path. THD is the only
        # sequence-parallel route now: the BSHD dense fallback has been removed.
        if packed_seq_params is not None:
            if attention_mask is not None:
                raise ValueError(
                    "THD packed CSA expects attention_mask=None; masks are derived "
                    "from cu_seqlens / position_ids."
                )
            return self._forward_thd_packed(x, position_ids, packed_seq_params)
        if self.ps.cp_size > 1 and attention_mask is not None:
            raise ValueError("CP expects attention_mask=None; masks are derived from position_ids.")
        batch, seq_len, _ = x.shape
        attention_rope_theta = (
            self.config.compress_rope_theta if self.compress_ratio > 1 else self.config.rope_theta
        )
        cos, sin = build_compressed_rope_cos_sin(
            position_ids,
            self.rope_head_dim,
            attention_rope_theta,
            config=self.config,
            use_yarn=self.compress_ratio > 1,
            device=x.device,
            dtype=x.dtype,
        )
        q_low = self.q_norm(self.wq_a(x))
        q = self.wq_b(q_low).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        q = q * torch.rsqrt(
            q.float().pow(2).mean(dim=-1, keepdim=True) + self.config.rms_norm_eps
        ).to(dtype=q.dtype)
        kv = self.kv_norm(self.wkv(x)).view(batch, seq_len, 1, self.head_dim).transpose(1, 2)
        q = apply_partial_rope(q, cos, sin, self.rope_head_dim)
        kv = apply_partial_rope(kv, cos, sin, self.rope_head_dim)
        use_sparse_backend = self.attention_backend not in {"local", "eager", "torch"}
        if (
            use_sparse_backend
            and self.compress_ratio == 4
            and self.ps.cp_size == 1
            and self.compressor is not None
            and self.indexer is not None
        ):
            return self._forward_fused_dsa_cp1(
                x,
                q,
                q_low,
                kv,
                position_ids=position_ids,
                cos=cos,
                sin=sin,
                attention_mask=attention_mask,
            )
        if use_sparse_backend and self.ps.cp_size == 1 and attention_mask is None:
            return self._forward_fused_sparse_no_indexer_cp1(
                x,
                q,
                kv,
                position_ids=position_ids,
                cos=cos,
                sin=sin,
            )

        # The BSHD dense-softmax fallback (and its CP all-gather loop) has been
        # removed: DSv4 sequence parallelism now goes exclusively through the THD
        # packed CP path (``packed_seq_params is not None`` above), and the only
        # supported BSHD routes are the CP=1 fused sparse kernels dispatched above.
        raise NotImplementedError(
            "DSv4 CSA BSHD path supports only the CP=1 fused sparse backends; pass "
            "packed_seq_params for the THD context-parallel path. The dense BSHD "
            "fallback was removed."
        )

    def _project_context(
        self, context: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        context = apply_partial_rope(context, cos, -sin, self.rope_head_dim).transpose(1, 2)
        batch, seq_len = context.shape[:2]
        grouped = context.reshape(
            batch, seq_len, self.config.o_groups, self.num_heads_per_group * self.head_dim
        )
        return self.wo_b(self.wo_a(grouped).flatten(2))

    def _forward_fused_sparse_no_indexer_cp1(
        self,
        x: torch.Tensor,
        q: torch.Tensor,
        kv: torch.Tensor,
        *,
        position_ids: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        dsa_kernels = _load_dsa_kernels()
        batch, seq_len, _ = x.shape
        query = q.transpose(1, 2).transpose(0, 1).contiguous()
        kv_full = kv.squeeze(1)
        kv_full = kv_full.transpose(0, 1).contiguous()
        window_idxs = _window_topk_indices(
            batch,
            seq_len,
            self.config.sliding_window,
            device=x.device,
        )

        compressed = None
        if self.compressor is not None and self.compress_ratio > 1:
            compressed = self.compressor(
                x,
                position_ids=position_ids,
                rope_theta=self.config.compress_rope_theta,
            )
            if compressed is not None:
                compressed_kv = compressed.squeeze(1)
                kv_full = torch.cat([kv_full, compressed_kv.transpose(0, 1).contiguous()], dim=0)

        if compressed is not None:
            n_compressed = compressed.size(2)
            comp_idx = torch.arange(n_compressed, device=x.device).view(1, n_compressed)
            valid_per_pos = (
                torch.arange(1, seq_len + 1, device=x.device) // self.compress_ratio
            ).view(seq_len, 1)
            compress_topk_idxs = torch.where(
                comp_idx < valid_per_pos,
                comp_idx + seq_len,
                torch.full_like(comp_idx, -1),
            )
            compress_topk_idxs = (
                compress_topk_idxs.unsqueeze(0).expand(batch, -1, -1).to(torch.int32)
            )
            flat_idxs, _flat_tlen = dsa_kernels.build_flat_topk_idxs(
                window_idxs,
                compress_topk_idxs,
                batch_size=batch,
                seqlen_kv=kv_full.size(0),
            )
        else:
            flat_idxs, _flat_tlen = dsa_kernels.build_flat_topk_idxs(
                window_idxs,
                batch_size=batch,
                seqlen_kv=kv_full.size(0),
            )

        out = dsa_kernels.dsa_sparse_attn(
            query,
            kv_full,
            self.sinks.float(),
            flat_idxs,
            self.head_dim**-0.5,
        )
        context = (
            out.view(seq_len, batch, self.num_heads, self.head_dim).permute(1, 2, 0, 3).contiguous()
        )
        return self._project_context(context, cos, sin)

    def _forward_fused_dsa_cp1(
        self,
        x: torch.Tensor,
        q: torch.Tensor,
        q_low: torch.Tensor,
        kv: torch.Tensor,
        *,
        position_ids: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.ps.cp_size != 1:
            raise NotImplementedError("DeepSeek V4 fused DSA path currently supports CP=1 only.")
        if attention_mask is not None:
            raise NotImplementedError(
                "DeepSeek V4 fused DSA path currently supports causal masking only."
            )
        dsa_kernels = _load_dsa_kernels()
        # The cuDNN SM90 indexer requires seqlen_q <= seqlen_k * ratio, but the
        # compressor floors to seq_len // ratio blocks, so a seq_len that is not a
        # multiple of ratio leaves the last query token(s) without a compressed key
        # block. Right-pad to a multiple of ratio (the causal tail attends only real
        # tokens and is sliced off the output) so seqlen_k * ratio == seqlen_q.
        orig_seq_len = x.shape[1]
        ratio = self.compress_ratio
        pad = (-orig_seq_len) % ratio
        if pad:
            pos_tail = position_ids[:, -1:] + torch.arange(
                1, pad + 1, device=position_ids.device, dtype=position_ids.dtype
            )
            position_ids = torch.cat([position_ids, pos_tail], dim=1)
            x = torch.nn.functional.pad(x, (0, 0, 0, pad))
            q = torch.nn.functional.pad(q, (0, 0, 0, pad))
            q_low = torch.nn.functional.pad(q_low, (0, 0, 0, pad))
            kv = torch.nn.functional.pad(kv, (0, 0, 0, pad))
        batch, seq_len, _ = x.shape
        compressed = self.compressor(
            x,
            position_ids=position_ids,
            rope_theta=self.config.compress_rope_theta,
        )
        index_comp = self.indexer.compressor(
            x,
            position_ids=position_ids,
            rope_theta=self.config.compress_rope_theta,
        )
        if compressed is None or index_comp is None:
            raise RuntimeError("DeepSeek V4 fused DSA requires at least one compressed KV entry.")
        compressed_kv = compressed.squeeze(1)
        index_k = index_comp.squeeze(1).transpose(0, 1).contiguous()
        kv_full = torch.cat([kv.squeeze(1), compressed_kv], dim=1)
        kv_full = kv_full.transpose(0, 1).contiguous()

        idx_cos, idx_sin = build_compressed_rope_cos_sin(
            position_ids,
            self.indexer.rope_head_dim,
            self.config.compress_rope_theta,
            config=self.config,
            use_yarn=self.compress_ratio > 1,
            device=x.device,
            dtype=x.dtype,
        )
        q_indexer = self.indexer.wq_b(q_low).view(
            batch, seq_len, self.indexer.index_n_heads, self.indexer.index_head_dim
        )
        q_indexer = q_indexer.transpose(1, 2)
        q_indexer = apply_partial_rope(q_indexer, idx_cos, idx_sin, self.indexer.rope_head_dim)
        q_indexer = rotate_activation(q_indexer)
        q_indexer = q_indexer.transpose(1, 2).transpose(0, 1).contiguous()
        weights_indexer = (
            (self.indexer.weights_proj(x).to(dtype=x.dtype) * (self.indexer.index_n_heads**-0.5))
            .transpose(0, 1)
            .contiguous()
        )
        indexer_topk = int(self.indexer.index_topk)
        if indexer_topk <= 0:
            raise RuntimeError("DeepSeek V4 fused DSA requires positive indexer_topk.")
        window_idxs = _window_topk_indices(
            batch,
            seq_len,
            self.config.sliding_window,
            device=x.device,
        )
        query = q.transpose(1, 2).transpose(0, 1).contiguous()
        sink = self.sinks.float()

        if self.training and torch.is_grad_enabled():
            out, _indexer_loss = dsa_kernels.fused_indexer_sparse_attn(
                query,
                kv_full,
                sink,
                window_idxs,
                q_indexer,
                index_k,
                weights_indexer,
                indexer_topk,
                self.compress_ratio,
                self.head_dim**-0.5,
                self.indexer.softmax_scale,
                0.0,
                sparse_loss=False,
                kv_offset=seq_len,
                calculate_per_token_loss=False,
            )
        else:
            topk_indices, _topk_length = dsa_kernels.indexer_topk(
                q_indexer,
                index_k,
                weights_indexer,
                indexer_topk,
                self.compress_ratio,
                indexer_softmax_scale=self.indexer.softmax_scale,
            )
            topk_indices = torch.where(
                topk_indices >= 0,
                topk_indices + seq_len,
                topk_indices,
            ).to(torch.int32)
            flat_idxs, flat_tlen = dsa_kernels.build_flat_topk_idxs(
                window_idxs,
                topk_indices,
                batch_size=batch,
                seqlen_kv=kv_full.size(0),
                compact=True,
            )
            out = dsa_kernels.dsa_sparse_attn(
                query,
                kv_full,
                sink,
                flat_idxs,
                self.head_dim**-0.5,
                topk_length=flat_tlen,
            )

        context = (
            out.view(seq_len, batch, self.num_heads, self.head_dim).permute(1, 2, 0, 3).contiguous()
        )
        if pad:
            context = context[:, :, :orig_seq_len, :].contiguous()
        return self._project_context(context, cos, sin)

    # ------------------------------------------------------------------
    # THD packed context-parallel path
    # ------------------------------------------------------------------

    def _thd_cu_seqlens(self, packed_seq_params: Any) -> torch.Tensor:
        """Padded cu_seqlens (falling back to unpadded) for the THD packed layout."""
        return (
            packed_seq_params.cu_seqlens_q_padded
            if packed_seq_params.cu_seqlens_q_padded is not None
            else packed_seq_params.cu_seqlens_q
        )

    def _project_boundary_kv(
        self,
        boundary_hidden: torch.Tensor,
        cu_seqlens: torch.Tensor,
        global_start: int,
        rope_theta: float,
    ) -> torch.Tensor:
        """Project the exchanged left-boundary hidden rows into MQA KV rows.

        Faithful to the DSv4 hybrid-attention boundary-KV path: the boundary rows
        sit immediately left of this rank's block, so they are KV-projected
        (``wkv`` -> ``kv_norm``) and RoPE'd at their own within-sequence positions
        (``global_start - d_window .. global_start - 1``) using lite's RoPE
        convention. Returns ``(d_window, 1, 1, head_dim)`` matching Core's
        ``boundary_kv.squeeze(-2).squeeze(1)`` contract in ``_forward_thd_cp``.
        """
        d_window = boundary_hidden.shape[0]
        bkv = self.kv_norm(self.wkv(boundary_hidden.reshape(d_window, -1)))  # (d_window, head_dim)
        b_pos = cp_utils._thd_cp_position_ids(cu_seqlens, int(global_start) - d_window, d_window)
        cos, sin = build_compressed_rope_cos_sin(
            b_pos.view(1, d_window).long(),
            self.rope_head_dim,
            rope_theta,
            config=self.config,
            use_yarn=self.compress_ratio > 1,
            device=bkv.device,
            dtype=bkv.dtype,
        )
        bkv = bkv.view(1, 1, d_window, self.head_dim)  # (batch=1, head=1, seq=d_window, hd)
        bkv = apply_partial_rope(bkv, cos, sin, self.rope_head_dim)
        return bkv.permute(2, 0, 1, 3).contiguous()  # (d_window, 1, 1, head_dim)

    def _forward_thd_packed(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        packed_seq_params: Any,
    ) -> torch.Tensor:
        """Build THD-packed q/key/x/qr, exchange boundaries, and run CP attention.

        ``x`` is ``[1, total, hidden]`` (batch-first, packed). Produces the
        TE-THD-convention tensors consumed by :meth:`_forward_thd_cp`, then applies
        the output projection (inverse RoPE + ``wo``) via :meth:`_project_context`.
        Returns ``[1, total, hidden]`` for the SBHD shim.
        """
        batch, seq_len, _ = x.shape
        if batch != 1:
            raise RuntimeError(
                f"DSv4 THD packed CSA expects batch-collapsed input [1, total, h]; got batch={batch}."
            )
        cp_group = self.ps.cp_group
        cp_rank = self.ps.cp_rank
        cu_seqlens = self._thd_cu_seqlens(packed_seq_params)
        global_start = cp_rank * seq_len

        attention_rope_theta = (
            self.config.compress_rope_theta if self.compress_ratio > 1 else self.config.rope_theta
        )
        # Positions come from cu_seqlens + this rank's global offset (CP-correct
        # within-sequence positions), not the raw position_ids tensor, so the
        # mapping is identical to the unsharded reference at cp_size == 1.
        del position_ids
        local_pos = cp_utils._thd_cp_position_ids(cu_seqlens, global_start, seq_len).view(1, seq_len)
        cos, sin = build_compressed_rope_cos_sin(
            local_pos.long(),
            self.rope_head_dim,
            attention_rope_theta,
            config=self.config,
            use_yarn=self.compress_ratio > 1,
            device=x.device,
            dtype=x.dtype,
        )
        q_low = self.q_norm(self.wq_a(x))
        q = self.wq_b(q_low).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        q = q * torch.rsqrt(
            q.float().pow(2).mean(dim=-1, keepdim=True) + self.config.rms_norm_eps
        ).to(dtype=q.dtype)
        kv = self.kv_norm(self.wkv(x)).view(batch, seq_len, 1, self.head_dim).transpose(1, 2)
        q = apply_partial_rope(q, cos, sin, self.rope_head_dim)
        kv = apply_partial_rope(kv, cos, sin, self.rope_head_dim)

        # TE-THD convention: query (total, np, hn); key (total, 1, 1, hn); x/qr (total, 1, *).
        query_thd = q.squeeze(0).transpose(0, 1).contiguous()  # (total, np, hn)
        key_thd = kv.permute(2, 0, 1, 3).contiguous()  # (total, 1, 1, hn)
        x_thd = x.transpose(0, 1).contiguous()  # (total, 1, hidden)
        qr_thd = q_low.transpose(0, 1).contiguous()  # (total, 1, q_lora_rank)

        if self.ps.cp_size > 1:
            boundary_hidden = cp_utils.exchange_cp_boundary_hidden(
                x_thd, self.compress_ratio, self.config.sliding_window, cp_group
            )
        else:
            # cp_size == 1 has no left neighbour, so the boundary window is exactly
            # zeros. Core reaches ``_forward_thd_cp`` only at cp>1 and never calls
            # the P2P exchange with an empty op list; the lite path routes cp=1 THD
            # through the same method, so materialize the zero boundary directly
            # (matching ``cp_utils.exchange_cp_boundary_hidden``'s D_window sizing).
            d_comp = (
                8 if self.compress_ratio == 4 else self.compress_ratio if self.compress_ratio > 1 else 0
            )
            d_window = max(int(self.config.sliding_window), d_comp)
            boundary_hidden = x_thd.new_zeros((d_window,) + tuple(x_thd.shape[1:]))
        boundary_kv = self._project_boundary_kv(
            boundary_hidden, cu_seqlens, global_start, attention_rope_theta
        )

        context = self._forward_thd_cp(
            query_thd, key_thd, x_thd, qr_thd, boundary_hidden, boundary_kv, packed_seq_params
        )
        # context: (total, 1, np * hn). Reshape to (1, np, total, hn) for the shared
        # output projection (inverse RoPE + wo), then return (1, total, hidden).
        context = (
            context.squeeze(1)
            .view(seq_len, self.num_heads, self.head_dim)
            .permute(1, 0, 2)
            .unsqueeze(0)
            .contiguous()
        )
        return self._project_context(context, cos, sin)

    def _forward_thd_cp(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        x: torch.Tensor,
        qr: torch.Tensor,
        boundary_hidden: torch.Tensor | None,
        boundary_kv: torch.Tensor | None,
        packed_seq_params: Any,
    ) -> torch.Tensor:
        """THD-packed context-parallel branch (faithful port of Core
        ``CompressedSparseAttention._forward_thd_cp``).

        Builds this rank's local KV context from boundary rows and fixed-capacity
        compressed KV, then runs sparse attention with an optional indexer loss.
        RoPE is applied to the indexer query with lite's convention (deviating from
        Core's ``cp_utils.apply_thd_cp_local_rope_*``) so the CP path stays
        identical to lite's BSHD path; ``apply_dsa_kernel_fusion`` gates the
        fused vs. differentiable-unfused kernels exactly as Core does.
        """
        cp_group = self.ps.cp_group
        cp_size = self.ps.cp_size
        cp_rank = self.ps.cp_rank

        l_local = query.shape[0]
        if l_local != key.shape[0]:
            raise RuntimeError("DSv4 THD CP path currently supports self-attention only.")
        cu_seqlens = self._thd_cu_seqlens(packed_seq_params)
        max_seqlen_q = int(packed_seq_params.max_seqlen_q)

        global_start = cp_rank * l_local
        kv_local = key.squeeze(-2).squeeze(1)  # (l_local, head_dim)
        if boundary_hidden is None or boundary_kv is None:
            raise RuntimeError(
                "DSv4 THD CP path requires boundary_hidden and boundary_kv from the "
                "hidden-only boundary exchange and boundary KV projection path."
            )
        boundary_kv = boundary_kv.squeeze(-2).squeeze(1)  # (d_window, head_dim)
        d_window = boundary_hidden.shape[0]
        compressed_kv_rank_major = kv_local.new_empty((0, kv_local.shape[-1]))
        cu_seqlens_compressed = None

        compressed_topk = seq_to_rank_row = None
        indexer_layout = q_indexer_cp = weights_indexer_cp = None
        k_indexer_rank_major = k_indexer_seq_major = None
        ratio = self.compress_ratio
        indexer = self.indexer
        indexer_loss_coeff = self.dsa_indexer_loss_coeff or 0.0
        training_with_grad = self.training and torch.is_grad_enabled()
        sparse_indexer_loss = self.dsa_indexer_use_sparse_loss
        calculate_per_token_loss = self.calculate_per_token_loss

        if self.compressor is not None and ratio > 1:
            compressed_lens = torch.div(
                cu_seqlens[1:] - cu_seqlens[:-1], ratio, rounding_mode="floor"
            )
            cu_seqlens_compressed = torch.cat(
                (
                    torch.zeros_like(cu_seqlens[:1]),
                    torch.cumsum(compressed_lens, dim=0, dtype=torch.int32),
                )
            )
            hidden_compact, compressed_group_ids, seq_to_rank_row = (
                cp_utils.prepare_cp_compressor_input(
                    x,
                    boundary_hidden,
                    cu_seqlens,
                    cu_seqlens_compressed,
                    global_start,
                    cp_size,
                    ratio,
                )
            )

            if indexer is not None:
                indexer_x, indexer_qr = x.detach(), qr.detach()
                if indexer_x.shape[1] != 1:
                    raise RuntimeError(
                        f"DSv4 THD CP indexer expects bsz=1, got {indexer_x.shape[1]}."
                    )
                q_indexer_cp = indexer.wq_b(indexer_qr.squeeze(1)).view(
                    l_local, indexer.index_n_heads, indexer.index_head_dim
                )
                idx_pos = cp_utils._thd_cp_position_ids(cu_seqlens, global_start, l_local)
                idx_cos, idx_sin = build_compressed_rope_cos_sin(
                    idx_pos.view(1, l_local).long(),
                    indexer.rope_head_dim,
                    self.config.compress_rope_theta,
                    config=self.config,
                    use_yarn=ratio > 1,
                    device=q_indexer_cp.device,
                    dtype=q_indexer_cp.dtype,
                )
                # lite RoPE wants the sequence axis at dim -2: (1, n_heads, l_local, hd).
                q_rope = q_indexer_cp.permute(1, 0, 2).unsqueeze(0)
                q_rope = apply_partial_rope(q_rope, idx_cos, idx_sin, indexer.rope_head_dim)
                q_indexer_cp = q_rope.squeeze(0).permute(1, 0, 2).contiguous()
                q_indexer_cp = rotate_activation(q_indexer_cp)
                weights_indexer_cp = indexer.weights_proj(indexer_x.squeeze(1)) * (
                    indexer.index_n_heads**-0.5
                )

                indexer_compressed_local, _ = indexer.compressor._forward_thd(
                    hidden_compact.detach(),
                    cu_seqlens,
                    max_seqlen_q=max_seqlen_q,
                    compressed_group_ids=compressed_group_ids,
                )
                k_indexer_rank_major = gather_from_sequence_parallel_region(
                    indexer_compressed_local.squeeze(1), group=cp_group
                )
                k_indexer_seq_major = torch.index_select(
                    k_indexer_rank_major, 0, seq_to_rank_row.clamp_min(0)
                )
                compressed_topk, indexer_layout = cp_utils.compute_cp_indexer_topk(
                    q_indexer_cp,
                    weights_indexer_cp,
                    k_indexer_seq_major,
                    cu_seqlens,
                    cu_seqlens_compressed,
                    global_start,
                    ratio,
                    indexer.index_topk,
                    indexer.softmax_scale,
                    max_seqlen_q=max_seqlen_q,
                    use_fused=self.apply_dsa_kernel_fusion,
                )

            compressed_kv_local, _ = self.compressor._forward_thd(
                hidden_compact,
                cu_seqlens,
                max_seqlen_q=max_seqlen_q,
                compressed_group_ids=compressed_group_ids,
            )
            compressed_kv_rank_major = gather_from_sequence_parallel_region(
                compressed_kv_local.squeeze(1), group=cp_group
            )

        kv_full_thd = torch.cat((boundary_kv, kv_local, compressed_kv_rank_major), dim=0)
        use_indexer_loss = (
            training_with_grad and indexer_loss_coeff > 0 and compressed_topk is not None
        )
        compressed_width = (
            compressed_topk.shape[-1]
            if compressed_topk is not None
            else (max_seqlen_q // ratio if ratio > 1 else 0)
        )
        topk_idxs, topk_length, indexer_topk_rank_major = (
            csa_cp_layout_kernels.build_attention_indices(
                cu_seqlens,
                global_start,
                l_local,
                d_window,
                self.config.sliding_window,
                ratio,
                compressed_width,
                compressed_topk,
                cu_seqlens_compressed=cu_seqlens_compressed,
                seq_to_rank_row=seq_to_rank_row,
                for_indexer_loss=use_indexer_loss,
            )
        )

        if use_indexer_loss:
            k_indexer_for_loss = k_indexer_rank_major
            compressed_kv_for_loss = compressed_kv_rank_major
            if not sparse_indexer_loss:
                k_indexer_for_loss = k_indexer_seq_major
                compressed_kv_for_loss = torch.index_select(
                    compressed_kv_rank_major, 0, seq_to_rank_row.clamp_min(0)
                )
            cu_seqlens_q_unpadded = None
            if (
                packed_seq_params.cu_seqlens_q is not None
                and packed_seq_params.cu_seqlens_q_padded is not None
                and packed_seq_params.cu_seqlens_q.data_ptr()
                != packed_seq_params.cu_seqlens_q_padded.data_ptr()
            ):
                cu_seqlens_q_unpadded = packed_seq_params.cu_seqlens_q
            q_padding_mask = None
            if cu_seqlens_q_unpadded is not None:
                global_rows = torch.arange(
                    global_start,
                    global_start + l_local,
                    device=query.device,
                    dtype=cu_seqlens.dtype,
                )
                batch_ids = torch.bucketize(
                    global_rows, cu_seqlens[1:], out_int32=True, right=True
                ).clamp_max(cu_seqlens.shape[0] - 2)
                real_seqlens = cu_seqlens_q_unpadded[1:] - cu_seqlens_q_unpadded[:-1]
                positions = global_rows - cu_seqlens[batch_ids]
                q_padding_mask = positions >= real_seqlens[batch_ids]
            apply_from_topk = (
                FusedCSAIndexerSparseAttnFromTopkFunc.apply
                if self.apply_dsa_kernel_fusion
                else _unfused_indexer_sparse_attn_from_topk
            )
            output, indexer_loss = apply_from_topk(
                query,
                kv_full_thd,
                self.sinks.float(),
                topk_idxs,
                q_indexer_cp,
                k_indexer_for_loss,
                weights_indexer_cp,
                indexer_topk_rank_major,
                compressed_kv_for_loss,
                self.softmax_scale,
                indexer.softmax_scale,
                indexer_loss_coeff,
                1 if calculate_per_token_loss else l_local * cp_size,
                sparse_indexer_loss,
                ratio,
                max_seqlen_q,
                indexer_layout,
                q_padding_mask,
            )
            if indexer_loss_coeff > 0:
                DSAIndexerLossLoggingHelper.save_loss_to_tracker(
                    loss=indexer_loss,
                    layer_number=self.layer_idx,
                    num_layers=self.config.num_hidden_layers
                    + self.config.num_nextn_predict_layers,
                    reduce_group=cp_group,
                )
            output = DSAIndexerLossAutoScaler.apply(output, indexer_loss)
            return output.unsqueeze(1)

        if self.apply_dsa_kernel_fusion:
            output = csa_sparse_attn(
                query,
                kv_full_thd,
                self.sinks.float(),
                topk_idxs,
                self.softmax_scale,
                topk_length=topk_length,
                is_thd=True,
            )
        else:
            output = unfused_compressed_sparse_attn(
                query, kv_full_thd, self.sinks.float(), topk_idxs, self.softmax_scale
            )
        return output.unsqueeze(1)
