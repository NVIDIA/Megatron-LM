# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""DSpark's parallel draft blocks, Markov correction, and acceptance prediction.

The architecture follows the released V4.1 inference implementation. Training
uses teacher-forced Markov inputs and the CE/L1/confidence objectives described
in deepseek-ai/DeepSpec. All inputs shared with the backbone are detached.
"""

from copy import copy
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from megatron.core.extensions.transformer_engine import TENorm
from megatron.core.models.deepseek_v41.stack import DeepSeekV41Block, ShardedLayerList
from megatron.core.transformer.module import MegatronModule, mark_keep_in_fp32
from megatron.core.transformer.single_pass_mhc import contract_streams


def _rotate(x, positions, dim, base, inverse=False):
    """Adjacent-pair RoPE with per-example absolute draft positions."""
    frequencies = base ** (-torch.arange(0, dim, 2, device=x.device, dtype=torch.float32) / dim)
    angles = positions.float().unsqueeze(-1) * frequencies
    while angles.ndim < x.ndim:
        angles = angles.unsqueeze(-2)
    cos, sin = angles.cos(), angles.sin()
    if inverse:
        sin = -sin
    even, odd = x[..., -dim::2].float(), x[..., -dim + 1 :: 2].float()
    tail = torch.stack((even * cos - odd * sin, odd * cos + even * sin), -1).flatten(-2)
    return torch.cat((x[..., :-dim], tail.to(x.dtype)), -1)


def draft_attention(attention, draft, context, anchors, block_size):
    """Read a causal context window and every position of the same parallel draft block."""
    config = attention.config
    count, batch, _ = draft.shape
    num_anchors = anchors.shape[1]
    heads, dim = config.num_attention_heads, config.v_head_dim
    positions = anchors[..., None] + torch.arange(block_size, device=draft.device)
    positions = positions.flatten(1).transpose(0, 1)
    qr, _ = attention.linear_q_down_proj(draft)
    q, _ = attention.linear_q_up_proj(attention.q_layernorm(qr))
    q = q.view(count, batch, heads, dim)
    q = _rotate(q, positions, config.qk_pos_emb_head_dim, config.rotary_base)
    draft_kv, _ = attention.linear_kv_proj(draft)
    draft_kv = _rotate(
        attention.kv_layernorm(draft_kv), positions, config.qk_pos_emb_head_dim, config.rotary_base
    )
    main_kv, _ = attention.linear_kv_proj(context)
    main_positions = torch.arange(context.shape[0], device=draft.device)[:, None].expand(-1, batch)
    main_kv = _rotate(
        attention.kv_layernorm(main_kv),
        main_positions,
        config.qk_pos_emb_head_dim,
        config.rotary_base,
    )
    window_positions = anchors[..., None] + torch.arange(
        -config.csa_window_size, 0, device=draft.device
    )
    valid = window_positions >= 0
    addresses = window_positions.clamp_min(0).flatten(1)
    window_kv = main_kv.transpose(0, 1).gather(1, addresses[..., None].expand(-1, -1, dim))
    window_kv = window_kv.view(batch, num_anchors, config.csa_window_size, dim)
    draft_kv = draft_kv.transpose(0, 1).reshape(batch, num_anchors, block_size, dim)
    kv = torch.cat((window_kv, draft_kv), dim=2)
    q = q.transpose(0, 1).reshape(batch, num_anchors, block_size, heads, dim)
    scores = torch.einsum("bakhd,bajd->bahkj", q.float(), kv.float()) * dim**-0.5
    mask = F.pad(valid, (0, block_size), value=True)
    scores = scores.masked_fill(~mask[:, :, None, None, :], -torch.inf)
    sink = attention.core_attention.attn_sink.view(1, 1, heads, 1, 1).expand(*scores.shape[:-1], 1)
    probabilities = torch.cat((scores, sink), -1).softmax(-1)[..., :-1]
    output = torch.einsum("bahkj,bajd->bakhd", probabilities, kv.float()).to(draft.dtype)
    output = output.reshape(batch, count, heads, dim).transpose(0, 1)
    output = _rotate(
        output, positions, config.qk_pos_emb_head_dim, config.rotary_base, inverse=True
    )
    groups = config.output_projection_groups
    output = output.reshape(count, batch, groups, -1)
    weights = attention.linear_o_group_proj.reshape(groups, config.output_projection_lora_rank, -1)
    output = torch.einsum("sbgd,grd->sbgr", output, weights).flatten(-2)
    return attention.linear_proj(output)[0]


@dataclass
class DSparkOutput:
    """Teacher-forced draft logits, confidence logits, and next-token supervision."""

    logits: torch.Tensor
    confidence: torch.Tensor
    targets: torch.Tensor

    def loss(self, teacher_logits, *, ce_weight=1.0, l1_weight=1.0, confidence_weight=1.0):
        """Mean CE + L1 distribution distance + acceptance-rate binary cross entropy."""
        teacher = teacher_logits.detach().float().softmax(-1)
        probabilities = self.logits.float().softmax(-1)
        l1 = (probabilities - teacher).abs().sum(-1)
        acceptance = (1 - 0.5 * l1.detach()).clamp(0, 1)
        ce = F.cross_entropy(self.logits.float().flatten(0, -2), self.targets.flatten())
        confidence = F.binary_cross_entropy_with_logits(self.confidence.float(), acceptance)
        return ce_weight * ce + l1_weight * l1.mean() + confidence_weight * confidence


class DSpark(MegatronModule):
    """V4.1's three-stage semi-autoregressive drafter, with no backbone gradient path."""

    def __init__(self, config, vocab_size, pg_collection) -> None:
        super().__init__(config)
        self.tp_group = pg_collection.tp
        options = config.dspark_config
        self.block_size = options.block_size
        self.noise_token_id = options.noise_token_id
        self.target_layer_ids = tuple(options.target_layer_ids)
        if not self.target_layer_ids or any(
            i < 0 or i >= len(config.csa_compress_ratios) for i in self.target_layer_ids
        ):
            raise ValueError("DSpark target layers must identify existing backbone blocks")
        self.main_proj = nn.Linear(
            len(self.target_layer_ids) * config.hidden_size,
            config.hidden_size,
            bias=False,
            dtype=config.params_dtype,
        )
        self.main_norm = TENorm(config, config.hidden_size, eps=config.layernorm_epsilon)
        draft_config = copy(config)
        draft_config.num_layers = options.num_layers
        draft_config.csa_compress_ratios = [0] * draft_config.num_layers
        draft_config.csa2_kv_source_layers = draft_config.csa2_index_source_layers = []
        draft_config.csa2_candidate_source_layer = None
        draft_config.csa2_candidate_block_size = draft_config.csa2_candidate_topk_blocks = 0
        draft_config.num_moe_experts = options.n_routed_experts
        draft_config.moe_router_topk = options.num_experts_per_tok
        draft_config.vision_config = None
        self.layers = ShardedLayerList(
            (
                DeepSeekV41Block(draft_config, i, pg_collection)
                for i in range(draft_config.num_layers)
            ),
            pg_collection.tp,
        )
        self.norm = TENorm(config, config.hidden_size, eps=config.layernorm_epsilon)
        rank = options.markov_rank
        self.markov_embed = nn.Embedding(vocab_size, rank, dtype=config.params_dtype)
        self.markov_head = nn.Linear(rank, vocab_size, bias=False, dtype=config.params_dtype)
        self.confidence_head = nn.Linear(
            config.hidden_size + rank, 1, bias=False, dtype=torch.float32
        )
        mark_keep_in_fp32(self.confidence_head.weight)
        if config.perform_initialization:
            for module in (
                self.main_proj,
                self.markov_embed,
                self.markov_head,
                self.confidence_head,
            ):
                config.init_method(module.weight)

    def forward(self, input_ids, features, anchors, embedding_weight, output_weight):
        """Train on complete draft blocks anchored at explicit positions [batch, anchors]."""
        if anchors.ndim != 2 or anchors.shape[0] != input_ids.shape[0] or anchors.numel() == 0:
            raise ValueError("DSpark anchors must have shape [batch, nonzero anchors]")
        if (anchors < 1).any() or (anchors + self.block_size >= input_ids.shape[1]).any():
            raise ValueError("Each DSpark anchor needs a context token and a complete target block")
        batch, num_anchors = anchors.shape
        offsets = torch.arange(self.block_size, device=input_ids.device)
        previous_positions = anchors[..., None] + offsets
        previous_ids = input_ids.gather(1, previous_positions.flatten(1)).view(
            batch, num_anchors, -1
        )
        targets = input_ids.gather(1, (previous_positions + 1).flatten(1)).view_as(previous_ids)
        noise_ids = torch.full_like(previous_ids, self.noise_token_id)
        noise_ids[..., 0] = previous_ids[..., 0]
        hidden = F.embedding(noise_ids.flatten(1), embedding_weight.detach()).transpose(0, 1)
        n = self.config.mhc_num_residual_streams
        hidden = hidden.unsqueeze(-2).expand(*hidden.shape[:-1], n, -1).flatten(-2)
        context = self.main_norm(self.main_proj(features.detach()))
        previous_mix = None
        for layer in self.layers:
            branch, next_mix, post, residual = layer.attention_mhc(hidden, previous_mix)
            branch = draft_attention(
                layer.attention, layer.attention_norm(branch), context, anchors, self.block_size
            )
            hidden = layer.attention_mhc.combine(branch, hidden, post, residual)
            branch, previous_mix, post, residual = layer.ffn_mhc(hidden, next_mix)
            branch, bias = layer.mlp(layer.ffn_norm(branch))
            if bias is not None:
                branch = branch + bias
            hidden = layer.ffn_mhc.combine(branch, hidden, post, residual)
        hidden = contract_streams(hidden, previous_mix, n)
        base = F.linear(self.norm(hidden), output_weight.detach()).transpose(0, 1)
        base = base.reshape(batch, num_anchors, self.block_size, -1)
        markov = self.markov_embed(previous_ids)
        logits = base + self.markov_head(markov)
        hidden = hidden.transpose(0, 1).reshape(batch, num_anchors, self.block_size, -1)
        confidence = self.confidence_head(torch.cat((hidden.float(), markov.float()), -1)).squeeze(
            -1
        )
        return DSparkOutput(logits, confidence, targets)


def select_verification_length(confidence, verification_cost):
    """Choose draft length maximizing expected accepted tokens per measured verification time.

    ``verification_cost`` supplies positive profiled costs for lengths 1..block_size.
    Confidence values are conditional acceptance probabilities, as in DSpark.
    """
    if verification_cost.shape != confidence.shape[-1:] or (verification_cost <= 0).any():
        raise ValueError("Provide a positive profiled verification cost for every draft length")
    expected = 1 + confidence.clamp(0, 1).cumprod(-1).cumsum(-1)
    return (expected / verification_cost).argmax(-1) + 1
