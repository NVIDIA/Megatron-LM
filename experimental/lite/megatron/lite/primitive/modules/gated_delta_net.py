# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Qwen-style Gated DeltaNet primitive."""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import transformer_engine.pytorch as te

from megatron.core.jit import jit_fuser
from megatron.lite.primitive.ops.gated_delta_rule import l2norm, torch_chunk_gated_delta_rule
from megatron.lite.primitive.parallel import ColumnParallelLinear, ParallelState, RowParallelLinear
from megatron.lite.primitive.parallel.cp import (
    zigzag_reconstruct_from_cp_parts,
    zigzag_slice_for_cp,
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


class GatedDeltaNet(nn.Module):
    """Native Gated DeltaNet with dense/packed CP reconstruction."""

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
    ):
        super().__init__()
        self.ps = ps
        self.deterministic = bool(deterministic)
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
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads_local, dtype=torch.float32))
        self.A_log = nn.Parameter(torch.zeros(self.num_v_heads_local, dtype=torch.float32))
        self.norm = te.RMSNorm(self.dv, eps=rms_norm_eps, zero_centered_gamma=True)
        self.o_proj = RowParallelLinear(self.v_dim, hidden_size, ps, bias=False)

    def forward(
        self, x: torch.Tensor, position_ids: torch.Tensor | None = None, packed_seq_params=None
    ) -> torch.Tensor:
        del position_ids
        qkvzba = self.in_proj(x).transpose(0, 1).contiguous()
        cp_restore = None
        if self.ps.cp_size > 1:
            qkvzba, cp_restore = self._gather_cp_qkvzba(qkvzba, packed_seq_params)
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

        cu_seqlens = None
        if packed_seq_params is not None:
            cu_seqlens = (
                packed_seq_params.cu_seqlens_q_padded
                if getattr(packed_seq_params, "cu_seqlens_q_padded", None) is not None
                else packed_seq_params.cu_seqlens_q
            )
            if not _HAS_FLA:
                raise NotImplementedError("GatedDeltaNet packed THD requires FLA kernels.")

        qkv = self._causal_conv1d(qkv, seq_len, cu_seqlens=cu_seqlens)
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
        )

        if cp_restore is not None:
            out = self._slice_cp_output(out, cp_restore)
            gate = self._slice_cp_output(gate, cp_restore)
            batch, seq_len = out.shape[:2]
        out = self._apply_gated_norm(out, gate)
        out = out.reshape(batch, seq_len, self.v_dim_local).transpose(0, 1).contiguous()
        return self.o_proj(out)

    def _all_gather_cp(self, tensor: torch.Tensor) -> list[torch.Tensor]:
        if self.ps.cp_group is None:
            raise RuntimeError("CP>1 requires ParallelState.cp_group.")
        try:
            from torch.distributed.nn.functional import all_gather

            return list(all_gather(tensor, group=self.ps.cp_group))
        except Exception:
            parts = [torch.empty_like(tensor) for _ in range(self.ps.cp_size)]
            dist.all_gather(parts, tensor, group=self.ps.cp_group)
            return parts

    def _gather_cp_qkvzba(self, qkvzba: torch.Tensor, packed_seq_params):
        parts = self._all_gather_cp(qkvzba)
        if packed_seq_params is not None:
            cu_seqlens = self._packed_cu_seqlens(packed_seq_params)
            full = reconstruct_packed_from_cp_parts(
                parts, cu_seqlens_padded=cu_seqlens, cp_size=self.ps.cp_size, dim=1
            )
            return full, ("packed", cu_seqlens)
        full = zigzag_reconstruct_from_cp_parts(parts, seq_dim=1)
        return full, ("dense",)

    def _slice_cp_output(self, out: torch.Tensor, cp_restore) -> torch.Tensor:
        kind = cp_restore[0]
        if kind == "packed":
            return split_packed_to_cp_local(
                out,
                cu_seqlens_padded=cp_restore[1],
                cp_size=self.ps.cp_size,
                cp_rank=self.ps.cp_rank,
                dim=1,
            )
        if kind == "dense":
            return zigzag_slice_for_cp(out, self.ps.cp_rank, self.ps.cp_size, seq_dim=1)
        raise RuntimeError(f"Unknown CP restore kind: {kind!r}")

    def _causal_conv1d(
        self, qkv: torch.Tensor, seq_len: int, *, cu_seqlens: torch.Tensor | None
    ) -> torch.Tensor:
        if cu_seqlens is None:
            qkv_t = qkv.transpose(1, 2).contiguous()
            return F.silu(self.conv1d(qkv_t)[:, :, :seq_len].transpose(1, 2))
        if _HAS_FLA:
            qkv, _ = _fla_causal_conv1d(
                x=qkv,
                weight=self.conv1d.weight.squeeze(1),
                bias=None,
                activation="silu",
                cu_seqlens=cu_seqlens,
            )
            return qkv
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
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if _HAS_FLA and not self.deterministic:
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
            raise ValueError("packed_seq_params must carry cu_seqlens_q for CP GatedDeltaNet.")
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

    def _prepare_qkv(self, qkv: torch.Tensor, gate, beta, alpha, batch: int, seq_len: int):
        query_key, value = qkv.split([2 * self.qk_dim_local, self.v_dim_local], dim=-1)
        query_key = query_key.reshape(batch, seq_len, 2 * self.num_k_heads_local, self.dk)
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
