# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Opt-in chunkwise GDN training while retaining zigzag transformer ownership."""

import torch

from megatron.core import tensor_parallel
from megatron.core.fusions.fused_pre_gated_delta_rule import fused_streamed_pre_gated_delta_rule
from megatron.core.ssm.causal_conv1d import _exchange_initial_states
from megatron.core.ssm.context_parallel.gdn_layout import get_chunkwise_layout
from megatron.core.ssm.gated_delta_net.chunkwise_core import chunkwise_gdn


class _CPGateParameters(torch.autograd.Function):
    """Reduce gate gradients before rounding to the replicated parameter dtype.

    Gate values are promoted exactly as in fused preparation. The ordinary DP+CP
    gradient sum will count the averaged CP gradient once. Parameters, optimizer
    storage and checkpoint representation retain their original dtype.
    """

    @staticmethod
    def forward(ctx, a_log, dt_bias, group):
        ctx.group = group
        ctx.dtypes = (a_log.dtype, dt_bias.dtype)
        return a_log.float(), dt_bias.float()

    @staticmethod
    def backward(ctx, d_a_log, d_dt_bias):
        combined = torch.stack((d_a_log.float(), d_dt_bias.float()))
        torch.distributed.all_reduce(combined, group=ctx.group)
        combined.div_(ctx.group.size())
        return combined[0].to(ctx.dtypes[0]), combined[1].to(ctx.dtypes[1]), None


def forward_chunkwise(module, hidden_states, packed_seq_params, cp_group):
    """Return normalized recurrence output in the original zigzag ownership."""
    if module.tp_size != 1 or module.sp_size != 1 or hidden_states.shape[1] != 1:
        raise ValueError("GDN chunkwise CP currently requires TP1, SP1 and batch size one")
    if (
        module.gdn_backend != "transformer_engine"
        or not module.gdn_pre_gated_delta_rule_fusion
        or module.config.deterministic_mode
    ):
        raise ValueError(
            "GDN chunkwise CP requires TE/cuDNN and non-deterministic fused preparation"
        )
    halo_tokens = module.conv_kernel_dim - 1
    if hidden_states.shape[0] < halo_tokens:
        raise ValueError("A chunkwise CP shard must contain at least the convolution halo")
    transport, metadata, extended_cu, extended_ids = get_chunkwise_layout(
        hidden_states, packed_seq_params, cp_group, module.conv_kernel_dim
    )
    kernels = getattr(module, "_chunkwise_gdn_kernels", None)
    if kernels is None:
        kernels = module.core_attention.get_context_parallel_backend()
        module._chunkwise_gdn_kernels = kernels

    contiguous_hidden = transport.forward(hidden_states)
    halo = _exchange_initial_states(contiguous_hidden.transpose(0, 1), halo_tokens, cp_group)
    extended_hidden = (
        torch.cat((halo.permute(2, 0, 1), contiguous_hidden), dim=0)
        if halo_tokens
        else contiguous_hidden
    )
    projected, _ = module.in_proj(extended_hidden)
    a_log, dt_bias = _CPGateParameters.apply(module.A_log, module.dt_bias, cp_group)
    query, key, value, gate, beta, g = fused_streamed_pre_gated_delta_rule(
        projected,
        module.conv1d.weight,
        module.conv1d.bias,
        a_log,
        dt_bias,
        num_key_heads=module.num_k_heads_local_tp,
        num_value_heads=module.num_v_heads_local_tp,
        key_head_dim=module.key_head_dim,
        value_head_dim=module.value_head_dim,
        use_qk_l2norm=module.use_qk_l2norm,
        cu_seqlens=extended_cu,
        seq_idx=extended_ids,
    )
    query, key, value, gate, beta, g = (
        tensor[:, halo_tokens:] for tensor in (query, key, value, gate, beta, g)
    )
    core = chunkwise_gdn(
        query.squeeze(0),
        key.squeeze(0),
        value.squeeze(0),
        g.squeeze(0),
        beta.squeeze(0),
        metadata,
        cp_group,
        kernels,
    )

    def norm_and_restore(output, output_gate):
        normalized = module._apply_gated_norm(output.unsqueeze(0), output_gate)
        normalized = normalized.reshape(1, hidden_states.shape[0], -1).transpose(0, 1).contiguous()
        return transport.inverse(normalized)

    if module.recompute_norm_out:
        module.norm_out_checkpoint = tensor_parallel.CheckpointWithoutOutput()
        return module.norm_out_checkpoint.checkpoint(norm_and_restore, core, gate)
    return norm_and_restore(core, gate)
