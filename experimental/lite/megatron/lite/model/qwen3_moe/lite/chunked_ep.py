# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Qwen-specific selection and residual-prefix wiring for ChunkedEP."""

from dataclasses import dataclass

import torch

from megatron.lite.model.qwen3_moe.lite.model import TransformerLayer
from megatron.lite.primitive.modules.chunked_ep_dispatcher import (
    ChunkedDispatcher as TokenDispatcher,
)
from megatron.lite.primitive.modules.chunked_ep_experts import ChunkedExperts as Experts
from megatron.lite.primitive.modules.moe_ep_chunk_overlap import ChunkedMoE, checkpoint_ep_chunk
from megatron.lite.primitive.modules.moe_ep_chunk_overlap_policy import (
    validate_ep_chunk_overlap_config,
)
from megatron.lite.primitive.modules.router import TopKRouter


def validate_chunked_ep_mtp(*, enable_ep_chunk_overlap, mtp_enable):
    if enable_ep_chunk_overlap and mtp_enable:
        raise ValueError("ChunkedEP with MTP is unsupported; disable MTP or ChunkedEP.")


def validate_qwen3_ep_chunk_recompute_composition(
    *, enable_ep_chunk_overlap, ep_chunk_full_recompute, recompute_modules
):
    if ep_chunk_full_recompute and not enable_ep_chunk_overlap:
        raise ValueError("ep_chunk_full_recompute=True requires enable_ep_chunk_overlap=True")
    if (
        enable_ep_chunk_overlap
        and not ep_chunk_full_recompute
        and any(name in {"moe", "full"} for name in recompute_modules)
    ):
        raise ValueError("normal ChunkedEP conflicts with outer MoE recompute")


class _ChunkedTransformerLayer(TransformerLayer):
    def __init__(self, *args, chunked_ep, **kwargs):
        super().__init__(*args, moe_factory=chunked_ep.moe, **kwargs)
        self.full_recompute = chunked_ep.full_recompute

    def forward(self, x, position_ids=None, packed_seq_params=None):
        if not self.full_recompute or not torch.is_grad_enabled():
            return super().forward(x, position_ids, packed_seq_params)
        if torch.is_tensor(position_ids) and position_ids.requires_grad:
            raise ValueError("ChunkedEP position_ids must not require gradients")

        def prefix(value):
            residual = value + self.attn(
                value, position_ids=position_ids, packed_seq_params=packed_seq_params
            )
            return self.mlp_norm(residual), residual

        return checkpoint_ep_chunk(
            prefix,
            x,
            self.moe.chunked_ep,
            (*self.attn.parameters(), *self.mlp_norm.parameters()),
            finish_backward=True,
        )


@dataclass(frozen=True)
class Qwen3ChunkedEP:
    max_input_rows: int
    chunk_count: int = 2
    full_recompute: bool = False

    def validate(self, mtp_enable, recompute_modules):
        validate_chunked_ep_mtp(enable_ep_chunk_overlap=True, mtp_enable=mtp_enable)
        validate_qwen3_ep_chunk_recompute_composition(
            enable_ep_chunk_overlap=True,
            ep_chunk_full_recompute=self.full_recompute,
            recompute_modules=recompute_modules,
        )

    def layer(self, *args, **kwargs):
        return _ChunkedTransformerLayer(*args, chunked_ep=self, **kwargs)

    def moe(self, config, ps, *, use_deepep, router_bias_rate, fp8, moe_act_recompute, lora_config):
        validate_ep_chunk_overlap_config(
            True,
            use_deepep=use_deepep,
            ep_size=ps.ep_size,
            topk=config.num_experts_per_tok,
            max_token_rows_per_rank=self.max_input_rows,
            chunk_count=self.chunk_count,
        )
        return ChunkedMoE(
            router=TopKRouter(
                config, ps, router_bias_rate=router_bias_rate, compute_aux_loss=False
            ),
            experts=Experts(
                config,
                ps,
                fp8=fp8,
                moe_act_recompute=moe_act_recompute,
                delay_wgrad_compute=True,
                lora_config=lora_config,
            ),
            dispatcher_factory=lambda _slot: TokenDispatcher(
                config.num_experts, config.hidden_size, ps, use_deepep=True
            ),
            max_input_rows=self.max_input_rows,
            hidden_size=config.hidden_size,
            expert_intermediate_size=getattr(config, "moe_intermediate_size", None),
            topk=config.num_experts_per_tok,
            ep_size=ps.ep_size,
            ep_group=ps.tp_ep_group,
            chunk_count=self.chunk_count,
            retain_backward=not self.full_recompute,
        )

    def bind(self, layers):
        if not layers:
            return None
        return layers[0].moe.chunked_ep


def release_chunked_ep(chunks):
    """Release model-owned primitive state before model weights leave the device."""
    seen = set()
    for chunk in chunks:
        for module in chunk.modules():
            if isinstance(module, ChunkedMoE) and id(module) not in seen:
                seen.add(id(module))
                device = next(module.parameters()).device
                stream = torch.cuda.current_stream(device) if device.type == "cuda" else None
                module.chunked_ep.release(stream=stream)
