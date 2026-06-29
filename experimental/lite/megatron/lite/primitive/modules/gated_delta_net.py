# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Qwen-style Gated DeltaNet primitive."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformer_engine.pytorch as te

from megatron.lite.primitive.kernels.jit import jit_fuser
from megatron.lite.primitive.ops.gated_delta_rule import (
    l2norm,
    torch_chunk_gated_delta_rule,
)
from megatron.lite.primitive.parallel import (
    ColumnParallelLinear,
    ParallelState,
    RowParallelLinear,
)
from megatron.lite.primitive.parallel.cp import (
    contiguous_to_zigzag_chunks,
    zigzag_reconstruct_from_cp_parts,
    zigzag_slice_for_cp,
    zigzag_to_contiguous_chunks,
)
from megatron.lite.primitive.parallel.thd import (
    reconstruct_packed_from_cp_parts,
    split_packed_to_cp_local,
)
from megatron.lite.primitive.utils import ensure_divisible


try:
    from fla.modules.convolution import (
        causal_conv1d as _fla_causal_conv1d,  # pyright: ignore[reportMissingImports]
    )
    from fla.ops.gated_delta_rule import (
        chunk_gated_delta_rule as _fla_chunk_gated_delta_rule,  # pyright: ignore[reportMissingImports]
    )

    _HAS_FLA = True
except ImportError:
    _HAS_FLA = False

try:
    from fla.ops.cp import build_cp_context as _fla_build_cp_context  # pyright: ignore[reportMissingImports]
except ImportError:
    _fla_build_cp_context = None

_CONV_PAD_ALIGNMENT = 4096


class GatedDeltaNet(nn.Module):
    """Native Gated DeltaNet with dense/packed all-gather CP support."""

    def __init__(
        self,
        *,
        hidden_size: int,
        linear_num_key_heads: int,
        linear_key_head_dim: int,
        linear_num_value_heads: int,
        linear_value_head_dim: int,
        linear_conv_kernel_dim: int,
        rms_norm_eps: float,
        ps: ParallelState,
        deterministic: bool = False,
        cp_mode: str = "fla_allgather",
    ):
        super().__init__()
        if cp_mode not in {"fla_allgather", "legacy_full_gather"}:
            raise ValueError(f"Unsupported GatedDeltaNet CP mode: {cp_mode!r}.")
        self.ps = ps
        self.deterministic = bool(deterministic)
        self.cp_mode = cp_mode
        self.num_k_heads = linear_num_key_heads
        self.num_v_heads = linear_num_value_heads
        self.dk = linear_key_head_dim
        self.dv = linear_value_head_dim
        self.v_heads_per_k_head = ensure_divisible(self.num_v_heads, self.num_k_heads)
        self.num_k_heads_local = ensure_divisible(self.num_k_heads, ps.tp_size)
        self.num_v_heads_local = ensure_divisible(self.num_v_heads, ps.tp_size)
        self.qk_dim = self.num_k_heads * self.dk
        self.v_dim = self.num_v_heads * self.dv
        self.qk_dim_local = self.num_k_heads_local * self.dk
        self.v_dim_local = self.num_v_heads_local * self.dv
        self.in_proj_dim = self.qk_dim * 2 + self.v_dim * 2 + self.num_v_heads * 2

        self.in_proj = ColumnParallelLinear(
            hidden_size,
            self.in_proj_dim,
            ps,
            bias=False,
            normalization="RMSNorm",
            eps=rms_norm_eps,
            zero_centered_gamma=True,
        )
        conv_dim_local = self.qk_dim_local * 2 + self.v_dim_local
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim_local,
            out_channels=conv_dim_local,
            kernel_size=linear_conv_kernel_dim,
            groups=conv_dim_local,
            bias=False,
            padding=linear_conv_kernel_dim - 1,
        )
        self.dt_bias = nn.Parameter(
            torch.ones(self.num_v_heads_local, dtype=torch.float32)
        )
        self.A_log = nn.Parameter(
            torch.zeros(self.num_v_heads_local, dtype=torch.float32)
        )
        self.norm = te.RMSNorm(self.dv, eps=rms_norm_eps, zero_centered_gamma=True)
        self.o_proj = RowParallelLinear(self.v_dim, hidden_size, ps, bias=False)
        self._cp_context_cache: dict[
            tuple[int, int, torch.device], tuple[torch.Tensor, object]
        ] = {}

    def forward(
        self, x: torch.Tensor, position_ids: torch.Tensor | None = None, packed_seq_params=None
    ) -> torch.Tensor:
        del position_ids
        is_packed = packed_seq_params is not None
        qkvzba = self.in_proj(x).transpose(0, 1).contiguous()
        cu_seqlens = self._packed_cu_seqlens(packed_seq_params) if is_packed else None
        cp_context = None
        legacy_full_gather = False
        if self.ps.cp_size > 1:
            if self.ps.cp_group is None:
                raise RuntimeError("CP>1 requires ParallelState.cp_group.")
            if self.cp_mode == "legacy_full_gather":
                qkvzba, cu_seqlens = self._legacy_full_gather_qkvzba(qkvzba, cu_seqlens)
                legacy_full_gather = True
            else:
                if not _HAS_FLA or _fla_build_cp_context is None:
                    raise NotImplementedError(
                        "GatedDeltaNet all-gather CP requires FLA kernels."
                    )
                if not is_packed and qkvzba.shape[0] > 1:
                    raise ValueError(
                        "GatedDeltaNet all-gather CP with SBHD inputs currently requires "
                        "micro_batch_size == 1. Use packed THD input or micro_batch_size=1."
                    )
                qkvzba = self._cp_swap_qkvzba(
                    qkvzba,
                    cu_seqlens if is_packed else None,
                    to_contiguous=True,
                )
                cu_seqlens, cp_context = self._build_cp_context(qkvzba, cu_seqlens)
        batch, seq_len = qkvzba.shape[:2]
        query, key, value, gate, beta, alpha = self._split_proj(qkvzba)
        qkv = torch.cat(
            [
                query.reshape(batch, seq_len, -1),
                key.reshape(batch, seq_len, -1),
                value.reshape(batch, seq_len, -1),
            ],
            dim=-1,
        )

        qkv = self._causal_conv1d(
            qkv, seq_len, cu_seqlens=cu_seqlens, cp_context=cp_context
        )
        query, key, value, gate, beta, alpha = self._prepare_qkv(
            qkv, gate, beta, alpha, batch, seq_len
        )
        g, beta = self._compute_g_and_beta(self.A_log, self.dt_bias, alpha, beta)
        out, _ = self._gated_delta_rule(
            query,
            key,
            value,
            g,
            beta,
            initial_state=None,
            output_final_state=False,
            cu_seqlens=cu_seqlens,
            cp_context=cp_context,
        )

        out = self._apply_gated_norm(out, gate)
        out = out.reshape(batch, seq_len, self.v_dim_local)
        if self.ps.cp_size > 1:
            if legacy_full_gather:
                out = self._legacy_slice_output(out, cu_seqlens)
            else:
                out = self._cp_swap_qkvzba(
                    out,
                    cu_seqlens if is_packed else None,
                    to_contiguous=False,
                )
            batch, seq_len = out.shape[:2]
        out = out.transpose(0, 1).contiguous()
        return self.o_proj(out)

    def _all_gather_cp_tensor(self, tensor: torch.Tensor) -> list[torch.Tensor]:
        if self.ps.cp_size <= 1:
            return [tensor]
        try:
            from torch.distributed.nn.functional import all_gather

            return list(all_gather(tensor, group=self.ps.cp_group))
        except Exception:
            parts = [torch.empty_like(tensor) for _ in range(self.ps.cp_size)]
            torch.distributed.all_gather(parts, tensor, group=self.ps.cp_group)
            return parts

    def _legacy_full_gather_qkvzba(
        self,
        qkvzba: torch.Tensor,
        cu_seqlens: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if cu_seqlens is None:
            parts = self._all_gather_cp_tensor(qkvzba)
            return zigzag_reconstruct_from_cp_parts(parts, seq_dim=1), None
        if qkvzba.shape[0] != 1:
            raise ValueError("Packed THD GatedDeltaNet expects a single packed batch row.")
        parts = self._all_gather_cp_tensor(qkvzba[0].contiguous())
        full = reconstruct_packed_from_cp_parts(
            parts,
            cu_seqlens_padded=cu_seqlens,
            cp_size=self.ps.cp_size,
            dim=0,
        )
        return full.unsqueeze(0).contiguous(), cu_seqlens

    def _legacy_slice_output(
        self,
        out: torch.Tensor,
        cu_seqlens: torch.Tensor | None,
    ) -> torch.Tensor:
        if cu_seqlens is None:
            return zigzag_slice_for_cp(out, self.ps.cp_rank, self.ps.cp_size, seq_dim=1)
        if out.shape[0] != 1:
            raise ValueError("Packed THD GatedDeltaNet expects a single packed batch row.")
        local = split_packed_to_cp_local(
            out[0].contiguous(),
            cu_seqlens_padded=cu_seqlens,
            cp_size=self.ps.cp_size,
            cp_rank=self.ps.cp_rank,
            dim=0,
        )
        return local.unsqueeze(0).contiguous()

    def _cp_swap_qkvzba(
        self,
        tensor: torch.Tensor,
        cu_seqlens: torch.Tensor | None,
        *,
        to_contiguous: bool,
    ) -> torch.Tensor:
        if cu_seqlens is None:
            swap = (
                zigzag_to_contiguous_chunks
                if to_contiguous
                else contiguous_to_zigzag_chunks
            )
            return swap(tensor, self.ps.cp_group, seq_dim=1)
        if tensor.shape[0] != 1:
            raise ValueError(
                "Packed THD GatedDeltaNet expects a single packed batch row."
            )
        local_cu_seqlens = cu_seqlens // self.ps.cp_size
        pieces = []
        swap = (
            zigzag_to_contiguous_chunks
            if to_contiguous
            else contiguous_to_zigzag_chunks
        )
        for idx in range(int(local_cu_seqlens.numel()) - 1):
            start = int(local_cu_seqlens[idx].item())
            end = int(local_cu_seqlens[idx + 1].item())
            if end <= start:
                continue
            pieces.append(swap(tensor[:, start:end, :], self.ps.cp_group, seq_dim=1))
        if not pieces:
            return tensor
        return torch.cat(pieces, dim=1).contiguous()

    def _build_cp_context(
        self,
        qkvzba: torch.Tensor,
        cu_seqlens: torch.Tensor | None,
    ) -> tuple[torch.Tensor, object]:
        if _fla_build_cp_context is None:
            raise NotImplementedError(
                "GatedDeltaNet all-gather CP requires FLA cp context."
            )
        if cu_seqlens is not None:
            return (
                cu_seqlens,
                _fla_build_cp_context(
                    cu_seqlens=cu_seqlens,
                    group=self.ps.cp_group,
                    conv1d_kernel_size=self.conv1d.kernel_size[0],
                ),
            )

        batch, local_seq_len = qkvzba.shape[:2]
        global_seq_len = local_seq_len * self.ps.cp_size
        cache_key = (global_seq_len, batch, qkvzba.device)
        cached = self._cp_context_cache.get(cache_key)
        if cached is None:
            dense_cu_seqlens = (
                torch.arange(batch + 1, device=qkvzba.device, dtype=torch.long)
                * global_seq_len
            )
            cached = (
                dense_cu_seqlens,
                _fla_build_cp_context(
                    cu_seqlens=dense_cu_seqlens,
                    group=self.ps.cp_group,
                    conv1d_kernel_size=self.conv1d.kernel_size[0],
                ),
            )
            self._cp_context_cache[cache_key] = cached
        return cached

    def _causal_conv1d(
        self,
        qkv: torch.Tensor,
        seq_len: int,
        *,
        cu_seqlens: torch.Tensor | None,
        cp_context,
    ) -> torch.Tensor:
        if _HAS_FLA and (cp_context is not None or cu_seqlens is not None or not self.deterministic):
            orig_seq_len = qkv.shape[1]
            pad_n = 0 if cp_context is not None else (-orig_seq_len % _CONV_PAD_ALIGNMENT)
            conv_input = qkv
            conv_cu_seqlens = cu_seqlens
            if pad_n > 0:
                conv_input = F.pad(qkv, (0, 0, 0, pad_n))
                if conv_cu_seqlens is not None:
                    conv_cu_seqlens = conv_cu_seqlens.clone()
                    conv_cu_seqlens[-1] += pad_n
            kwargs = {}
            if cp_context is not None:
                kwargs["cp_context"] = cp_context
            qkv, _ = _fla_causal_conv1d(
                x=conv_input,
                weight=self.conv1d.weight.squeeze(1),
                bias=None,
                activation="silu",
                cu_seqlens=conv_cu_seqlens,
                **kwargs,
            )
            if pad_n > 0:
                qkv = qkv[:, :orig_seq_len, :]
            return qkv
        if cu_seqlens is None and cp_context is None:
            return F.silu(
                self.conv1d(qkv.transpose(1, 2))[:, :, :seq_len].transpose(1, 2)
            )
        raise NotImplementedError("GatedDeltaNet packed THD requires FLA causal conv.")

    def _gated_delta_rule(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        *,
        initial_state: torch.Tensor | None,
        output_final_state: bool,
        cu_seqlens: torch.Tensor | None,
        cp_context,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if _HAS_FLA and (cp_context is not None or not self.deterministic):
            kwargs = {}
            if cp_context is not None:
                kwargs["cp_context"] = cp_context
            return _fla_chunk_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=initial_state,
                output_final_state=output_final_state,
                use_qk_l2norm_in_kernel=False,
                cu_seqlens=cu_seqlens,
                **kwargs,
            )
        if cp_context is not None:
            raise NotImplementedError(
                "GatedDeltaNet all-gather CP requires FLA gated delta rule."
            )
        return torch_chunk_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=initial_state,
            output_final_state=output_final_state,
        )

    @staticmethod
    def _packed_cu_seqlens(packed_seq_params) -> torch.Tensor:
        cu_seqlens = (
            packed_seq_params.cu_seqlens_q_padded
            if getattr(packed_seq_params, "cu_seqlens_q_padded", None) is not None
            else packed_seq_params.cu_seqlens_q
        )
        if cu_seqlens is None:
            raise ValueError(
                "packed_seq_params must carry cu_seqlens_q for CP GatedDeltaNet."
            )
        return cu_seqlens

    def _split_proj(self, qkvzba: torch.Tensor):
        q, k, v, z, b, a = qkvzba.split(
            [
                self.qk_dim_local,
                self.qk_dim_local,
                self.v_dim_local,
                self.v_dim_local,
                self.num_v_heads_local,
                self.num_v_heads_local,
            ],
            dim=-1,
        )
        batch, seq_len = qkvzba.shape[:2]
        return (
            q.reshape(batch, seq_len, self.num_k_heads_local, self.dk),
            k.reshape(batch, seq_len, self.num_k_heads_local, self.dk),
            v.reshape(batch, seq_len, self.num_v_heads_local, self.dv),
            z.reshape(batch, seq_len, self.num_v_heads_local, self.dv),
            b.reshape(batch, seq_len, self.num_v_heads_local),
            a.reshape(batch, seq_len, self.num_v_heads_local),
        )

    def _prepare_qkv(
        self, qkv: torch.Tensor, gate, beta, alpha, batch: int, seq_len: int
    ):
        query_key, value = qkv.split([2 * self.qk_dim_local, self.v_dim_local], dim=-1)
        query_key = query_key.reshape(
            batch, seq_len, 2 * self.num_k_heads_local, self.dk
        )
        value = value.reshape(batch, seq_len, self.num_v_heads_local, self.dv)
        query, key = query_key.split(self.num_k_heads_local, dim=2)
        query = self._l2norm(query.contiguous())
        key = self._l2norm(key.contiguous())
        if self.v_heads_per_k_head > 1:
            query = query.repeat_interleave(self.v_heads_per_k_head, dim=2)
            key = key.repeat_interleave(self.v_heads_per_k_head, dim=2)
        return (
            query.contiguous(),
            key.contiguous(),
            value.contiguous(),
            gate.contiguous(),
            beta.contiguous(),
            alpha.contiguous(),
        )

    def _l2norm(self, x: torch.Tensor) -> torch.Tensor:
        return l2norm(x)

    @staticmethod
    def _compute_g_and_beta(A_log, dt_bias, alpha, beta):
        g = -A_log.exp() * F.softplus(alpha.float() + dt_bias)
        return g, beta.sigmoid()

    @jit_fuser
    def _apply_gated_norm(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        x_dtype = x.dtype
        x = x.reshape(-1, x.shape[-1])
        y = self.norm(x)
        gate = gate.reshape(-1, gate.shape[-1])
        y = y * F.silu(gate.float())
        return y.to(x_dtype)


__all__ = ["GatedDeltaNet"]
