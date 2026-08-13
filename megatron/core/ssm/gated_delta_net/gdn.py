# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025, Songlin Yang, Jan Kautz, Ali Hatamizadeh.

# Some of this code was adopted from https://github.com/huggingface/transformers
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F

from megatron.core.jit import jit_fuser
from megatron.core.ssm.gated_delta_net.common import _GDNBase, chunk_gated_delta_rule, l2norm


class GatedDeltaNet(_GDNBase):
    # pylint: disable=missing-class-docstring

    variant_name = "GDN"

    def _setup_variant_attrs(self):
        """Set the GDN in_proj sizing, split tables, gate parameter dims, and kernel."""
        # alpha, beta
        self.in_proj_extra_dim = self.num_value_heads * 2

        # Per-section sizes (and names) of the in_proj output, local to this TP rank.
        # Used for the CP head permutation (pre-a2a), for splitting the projection
        # output (post-a2a), and for the sharded checkpoint split of in_proj.weight.
        self.in_proj_split_names = ["query", "key", "value", "z", "beta", "alpha"]
        self.in_proj_split_sections = (
            self.qk_dim_local_tp,  # q
            self.qk_dim_local_tp,  # k
            self.v_dim_local_tp,  # v
            self.v_dim_local_tp,  # gate (z)
            self.num_value_heads // self.tp_size,  # beta
            self.num_value_heads // self.tp_size,  # alpha
        )
        # Per-section sizes of the post-a2a tensor before the headwise-CP head split;
        # ``_forward_compute`` divides these by the runtime headwise cp_size.
        self.feat_dim_split = (
            self.qk_dim_local_tp * 2 + self.v_dim_local_tp,  # qkv
            self.v_dim_local_tp,  # gate (z)
            self.num_value_heads // self.tp_size,  # beta
            self.num_value_heads // self.tp_size,  # alpha
        )

        self.dt_bias_dim = self.num_v_heads_local_tp
        self.a_log_dim = self.num_v_heads_local_tp

        if self.config.deterministic_mode:
            self.gated_delta_rule = torch_chunk_gated_delta_rule
        else:
            self.gated_delta_rule = chunk_gated_delta_rule

    @jit_fuser
    def _compute_gates(
        self,
        A_log_local_cp: torch.Tensor,
        dt_bias_local_cp: torch.Tensor,
        batch: int,
        seq_len: int,
        *gate_feats: tuple[torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the per-head log-decay g and the write strength beta."""
        # ``gate_feats`` arrives in ``in_proj_split_names`` order: beta, then alpha.
        beta, alpha = gate_feats
        g = -A_log_local_cp.exp() * F.softplus(alpha.float() + dt_bias_local_cp)  # In fp32
        beta = beta.sigmoid()
        return g, {"beta": beta.contiguous()}


####################
# Torch native gated delta rule
####################
def torch_chunk_gated_delta_rule(
    q,
    k,
    v,
    g,
    beta,
    scale=None,
    chunk_size=64,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
    cu_seqlens=None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    # pylint: disable=line-too-long
    '''
    Torch-native implementation of chunked gated delta rule for deterministic mode.
    Need this because FLA is not deterministic.

    ``scale`` defaults to ``1 / sqrt(K)``, matching the FLA kernel. Extra keyword
    arguments are accepted and ignored so this stays interchangeable with the FLA
    kernel, which takes several options this implementation does not model.

    Reference: https://github.com/huggingface/transformers/blob/144c8ce2809a2e21914017652700e1ecb450501e/src/transformers/models/qwen3_next/modeling_qwen3_next.py#L470-L547
    '''

    assert (
        cu_seqlens is None
    ), "cu_seqlens is not supported for torch_chunk_gated_delta_rule for now."

    query, key, value = q, k, v
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    if scale is None:
        scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    # reshape to chunks
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0
    )

    # chunk decay
    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)
    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1
    )

    # for each chunk
    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(mask, 0)
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.reshape(
        core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1]
    )
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state
