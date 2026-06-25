# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""MoE expert compute: SwiGLU fusions, _AllReduceETP, and Experts."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any

import torch  # pyright: ignore[reportMissingImports]
import torch.distributed as dist  # pyright: ignore[reportMissingImports]
import torch.nn as nn  # pyright: ignore[reportMissingImports]
import transformer_engine.pytorch as te  # pyright: ignore[reportMissingImports]

from megatron.lite.primitive.kernels.swiglu import bias_swiglu_impl, weighted_bias_swiglu_impl
from megatron.lite.primitive.modules.lora import (
    LoraConfig,
    SharedGroupedLinearLoRA,
    normalize_lora_config,
)
from megatron.lite.primitive.parallel import ParallelState
from megatron.lite.primitive.recompute import CheckpointWithoutOutput
from megatron.lite.primitive.utils import ensure_divisible

__all__ = ["Experts", "_AllReduceETP"]


@contextmanager
def _expert_nvtx_range(name: str):
    if os.environ.get("MEGATRON_LITE_EP_EXPERT_NVTX") != "1" or not torch.cuda.is_available():
        yield
        return
    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


def swiglu_with_probs(
    y: torch.Tensor, probs: torch.Tensor | None, swiglu_limit: float = 0.0
) -> torch.Tensor:
    """SwiGLU with optional expert probability scaling."""
    if swiglu_limit > 0:
        gate, up = y.chunk(2, dim=-1)
        up = torch.clamp(up.float(), min=-swiglu_limit, max=swiglu_limit)
        gate = torch.clamp(gate.float(), max=swiglu_limit)
        out = torch.nn.functional.silu(gate) * up
        if probs is not None:
            out = out * probs
        return out.to(dtype=y.dtype)
    if probs is not None:
        return weighted_bias_swiglu_impl(y, bias=None, weights=probs)
    return bias_swiglu_impl(y, bias=None)


class _AllReduceETP(torch.autograd.Function):
    """AllReduce with proper autograd: grad(AllReduce) = AllReduce."""

    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        dist.all_reduce(x, group=group)
        return x

    @staticmethod
    def backward(ctx, grad):
        return grad, None


class Experts(nn.Module):

    def __init__(
        self,
        config: Any,
        ps: ParallelState,
        *,
        fp8: bool = False,
        moe_act_recompute: bool = False,
        lora_config: LoraConfig | dict | None = None,
    ):
        super().__init__()
        self.num_local_experts = ensure_divisible(config.num_experts, ps.ep_size)
        self.fp8 = fp8
        self.moe_act_recompute = moe_act_recompute
        self.etp_group = ps.etp_group if ps.etp_size > 1 else None
        self.swiglu_limit = float(getattr(config, "swiglu_limit", 0.0) or 0.0)

        self.fc1 = te.GroupedLinear(
            self.num_local_experts,
            config.hidden_size,
            config.moe_intermediate_size * 2 // ps.etp_size,
            bias=False,
            params_dtype=torch.bfloat16,
        )
        self.fc2 = te.GroupedLinear(
            self.num_local_experts,
            config.moe_intermediate_size // ps.etp_size,
            config.hidden_size,
            bias=False,
            params_dtype=torch.bfloat16,
        )
        lora = normalize_lora_config(lora_config)
        self.fc1_lora: SharedGroupedLinearLoRA | None = None
        self.fc2_lora: SharedGroupedLinearLoRA | None = None
        if lora.enabled and lora.targets_module("linear_fc1"):
            self.fc1_lora = SharedGroupedLinearLoRA(
                self.num_local_experts,
                config.hidden_size,
                config.moe_intermediate_size * 2 // ps.etp_size,
                lora.rank,
                alpha=lora.alpha,
                dropout=lora.dropout,
            )
        if lora.enabled and lora.targets_module("linear_fc2"):
            self.fc2_lora = SharedGroupedLinearLoRA(
                self.num_local_experts,
                config.moe_intermediate_size // ps.etp_size,
                config.hidden_size,
                lora.rank,
                alpha=lora.alpha,
                dropout=lora.dropout,
            )
        if ps.tp_size > 1 and ps.ep_size == 1 and ps.etp_size == 1:
            tp_group = ps.tp_group
            for module in (self.fc1, self.fc2, self.fc1_lora, self.fc2_lora):
                if module is None:
                    continue
                for param in module.parameters():

                    def _ar(grad, g=tp_group):
                        dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=g)
                        return grad

                    param.register_hook(_ar)

    def forward(
        self,
        x: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        permuted_probs: torch.Tensor | None = None,
        tokens_per_expert_list: list[int] | None = None,
    ) -> torch.Tensor:
        m_splits = (
            tokens_per_expert.tolist()
            if tokens_per_expert_list is None
            else list(tokens_per_expert_list)
        )
        pad_mask = None
        if self.fp8:
            x, permuted_probs, m_splits, pad_mask = self._fp8_pad(x, permuted_probs, m_splits)

        etp_real_len = x.shape[0]
        if self.etp_group is not None:
            max_len = torch.tensor([etp_real_len], device=x.device, dtype=torch.int64)
            dist.all_reduce(max_len, op=dist.ReduceOp.MAX, group=self.etp_group)
            max_len = int(max_len.item())
            if etp_real_len < max_len:
                x = torch.cat(
                    [
                        x,
                        torch.zeros(
                            max_len - etp_real_len, x.shape[1], dtype=x.dtype, device=x.device
                        ),
                    ],
                    dim=0,
                )
                if permuted_probs is not None:
                    permuted_probs = torch.cat(
                        [
                            permuted_probs,
                            torch.zeros(
                                max_len - etp_real_len, dtype=permuted_probs.dtype, device=x.device
                            ),
                        ],
                        dim=0,
                    )
                m_splits = list(m_splits)
                m_splits[-1] += max_len - etp_real_len

        probs = permuted_probs.unsqueeze(-1) if permuted_probs is not None else None
        with _expert_nvtx_range("ep_experts.forward"):
            if self.moe_act_recompute and probs is not None:
                act_ckpt = CheckpointWithoutOutput(preserve_rng_state=True)
                fc1_out = self.fc1(x, m_splits)
                if self.fc1_lora is not None:
                    fc1_out = fc1_out + self.fc1_lora(x, m_splits)
                h = act_ckpt.checkpoint(swiglu_with_probs, fc1_out, probs, self.swiglu_limit)
                out = self.fc2(h, m_splits)
                if self.fc2_lora is not None:
                    out = out + self.fc2_lora(h, m_splits)
                act_ckpt.discard_output_and_register_recompute(out)
            else:
                fc1_out = self.fc1(x, m_splits)
                if self.fc1_lora is not None:
                    fc1_out = fc1_out + self.fc1_lora(x, m_splits)
                h = swiglu_with_probs(fc1_out, probs, self.swiglu_limit)
                out = self.fc2(h, m_splits)
                if self.fc2_lora is not None:
                    out = out + self.fc2_lora(h, m_splits)

        if self.etp_group is not None:
            out = _AllReduceETP.apply(out, self.etp_group)
            out = out[:etp_real_len]

        if pad_mask is not None:
            out = out[pad_mask]
        return out

    @staticmethod
    def _fp8_pad(x, permuted_probs, m_splits):
        padded = [(s + 15) // 16 * 16 for s in m_splits]
        if padded == m_splits:
            return x, permuted_probs, m_splits, None
        device, dtype = x.device, x.dtype
        total_padded = sum(padded)
        x_pad = torch.zeros(total_padded, x.size(1), device=device, dtype=dtype)
        mask = torch.zeros(total_padded, dtype=torch.bool, device=device)
        probs_pad = None
        if permuted_probs is not None:
            probs_pad = torch.zeros(total_padded, device=device, dtype=permuted_probs.dtype)
        src_off, dst_off = 0, 0
        for real, pad in zip(m_splits, padded, strict=True):
            x_pad[dst_off : dst_off + real] = x[src_off : src_off + real]
            mask[dst_off : dst_off + real] = True
            if probs_pad is not None:
                probs_pad[dst_off : dst_off + real] = permuted_probs[src_off : src_off + real]
            src_off += real
            dst_off += pad
        return x_pad, probs_pad, padded, mask
