# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Canonical GDP mixer integration shared by training and serving."""

import torch

from megatron.core.ssm.ops.gdp.batch_invariant import GDPCache, append, append_packed, train
from megatron.core.ssm.ops.gdp.batch_invariant_conv import causal_conv
from megatron.core.ssm.packed_seq_helpers import get_cu_seqlens


def validate_config(mixer) -> None:
    """Reject configurations without a shared, supported forward contract."""
    config = mixer.config
    if config.gdp_cutedsl_kernel or config.fp8 or config.fp4:
        raise ValueError('Canonical GDP requires BF16 and gdp_cutedsl_kernel=False in both modes')
    if mixer.pg_collection.cp.size() != 1 or config.sequence_parallel:
        raise NotImplementedError(
            'Canonical GDP currently requires CP=1 and no sequence parallelism'
        )
    if mixer.d_conv != 4:
        raise NotImplementedError('Canonical GDP currently supports convolution width four')
    if mixer.recompute_in_proj or mixer.recompute_qkv or mixer.offload_gdp_qkv:
        raise NotImplementedError(
            'Canonical GDP supports whole-layer checkpointing, not selective GDP offload/recompute'
        )


def _cache(mixer, storage):
    return GDPCache.from_storage(
        storage,
        mixer.nheads_local_cp,
        mixer.d_state,
        mixer.headdim,
        mixer.config.gdp_batch_invariant_block_size,
    )


def _prepare(mixer, zvkqba, conv_state=None, slots=None, cu=None):
    from fla.modules.l2norm import l2_norm

    z, x, ba = mixer._preprocess(zvkqba)
    x = causal_conv(
        x.contiguous(),
        mixer.cp.get_conv1d_weight(),
        mixer.cp.get_conv1d_bias(),
        initial_state=conv_state,
        slots=slots,
        cu_seqlens=cu,
        update_state=conv_state is not None,
    )
    b, t, _ = x.shape
    r, h, groups, d, w = (
        mixer.num_householder,
        mixer.nheads_local_cp,
        mixer.ngroups_local_cp,
        mixer.d_state,
        mixer.headdim,
    )
    value, key, query = torch.split(x, [r * h * w, r * groups * d, groups * d], dim=-1)
    q = l2_norm(query.reshape(b, t, groups, d).contiguous())
    k = l2_norm(key.reshape(b, t, r, groups, d).contiguous())
    if h != groups:
        q = q.repeat_interleave(h // groups, dim=-2)
        k = k.repeat_interleave(h // groups, dim=-2)
    v = value.reshape(b, t, r, h, w).contiguous()
    beta, g = mixer._compute_gating(ba)
    return z, (
        q.contiguous(),
        k.contiguous(),
        v,
        g.float().contiguous(),
        beta.reshape(b, t, r, h).float().contiguous(),
    )


def _packed_train(xs, cu, max_length, block_size):
    """Pad prepared sequences for the batched differentiable recurrence, then unpack."""
    total = xs[0].shape[1]
    torch._assert_async(
        ((cu[1:] - cu[:-1]) <= max_length).all(),
        'Packed GDP max_seqlen must cover padded sequence lengths',
    )
    index = cu[:-1, None].long() + torch.arange(max_length, device=cu.device)[None, :]
    valid = index < cu[1:, None]
    padded = []
    for x in xs:
        value = x[0, index.clamp_max(total - 1)]
        mask = valid.reshape(*valid.shape, *([1] * (value.ndim - 2)))
        padded.append(value.masked_fill(~mask, 0).contiguous())
    output, _ = train(*padded, block_size=block_size)
    rows = torch.arange(total, device=cu.device)
    seq = torch.bucketize(rows, cu[1:], right=True)
    local = rows - cu[seq]
    return output[seq, local].unsqueeze(0)


def chunk_forward(mixer, hidden_states, conv_state=None, ssm_state=None, packed_seq_params=None):
    """Run common projection/preparation/recurrence for training or static prefill."""
    projection, _ = mixer.in_proj(hidden_states)
    cu = None
    if packed_seq_params is not None:
        if hidden_states.shape[1] != 1 or ssm_state is not None:
            raise ValueError('Packed training requires B=1 and no inference state')
        cu = get_cu_seqlens(packed_seq_params).to(dtype=torch.int32).contiguous()
        # The trailing padded rows are a distinct sequence, including an empty
        # sentinel when the original offsets already cover the complete input.
        cu = torch.cat([cu, cu.new_tensor([hidden_states.shape[0]])])
    z, xs = _prepare(mixer, projection, conv_state=conv_state, cu=cu)
    if ssm_state is not None:
        core = append(*xs, _cache(mixer, ssm_state))
    elif cu is not None:
        max_length = packed_seq_params.max_seqlen_q
        if max_length is None:
            max_length = hidden_states.shape[0]
        if packed_seq_params.cu_seqlens_q_padded is not None:
            max_length = hidden_states.shape[0]
        core = _packed_train(xs, cu, max_length, mixer.config.gdp_batch_invariant_block_size)
    else:
        core, _ = train(*xs, block_size=mixer.config.gdp_batch_invariant_block_size)
    return mixer._postprocess(core, z, packed_seq_params=packed_seq_params)


def decode(
    mixer,
    projected,
    conv_state,
    ssm_state,
    slots,
    intermediate_conv_state=None,
    intermediate_ssm_state=None,
):
    """Continue slot-indexed requests through the same preparation and recurrence."""
    if intermediate_conv_state is not None or intermediate_ssm_state is not None:
        raise NotImplementedError('Canonical GDP serving does not yet support speculative buffers')
    z, xs = _prepare(mixer, projected.transpose(0, 1), conv_state=conv_state, slots=slots)
    core = append(*xs, _cache(mixer, ssm_state), slots=slots)
    return mixer._postprocess(core, z).transpose(0, 1)


def packed_prefill(mixer, projected, conv_state, ssm_state, context):
    """Run graph-safe packed prefill, including continuation of partial GDP blocks."""
    if context.mamba_slot_allocator is not None:
        raise NotImplementedError('Canonical GDP serving requires prefix caching disabled')
    metadata = context.mamba_metadata
    slots = metadata.batch_indices_prefill
    cu = metadata.cu_seqlens
    z, xs = _prepare(mixer, projected, conv_state=conv_state, slots=slots, cu=cu)
    core = append_packed(*xs, _cache(mixer, ssm_state), cu, slots)
    return mixer._postprocess(core, z)
