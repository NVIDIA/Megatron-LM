# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Low-level Mamba layer utilities for the two-tower diffusion architecture.

These functions bypass ``MambaLayer.forward()`` to expose explicit SSM
(selective state-space model) and conv1d state I/O.  Standard
``MambaLayer.forward()`` treats states as an internal detail; the two-tower
architecture needs to *read* states from the context tower and *write*
initial states into the denoiser tower at every block boundary.

Functions:

    :func:`forward_mamba_layer_parallel_with_all_states`
        **Context tower.**  Runs the full sequence in one kernel launch and
        extracts conv + SSM states after every ``block_size``-token chunk.
        Requires ``block_size == mixer.chunk_size``.

    :func:`forward_mamba_layer_batched_with_initial_states`
        **Denoiser tower.**  Reshapes the sequence into independent blocks,
        injects per-block initial states from the context cache, and runs
        all blocks in a single batched kernel call.
        Requires ``block_size == mixer.chunk_size``.

    :func:`forward_mamba_layer_with_states`
        **Inference / cache extension.**  Processes a single sequence with
        explicit initial conv + SSM states and returns the final states.

    :func:`forward_mamba_layer_varlen_with_states`
        **Inference prefill.**  Processes packed variable-length sequences
        using varlen conv1d and SSM kernels.  Returns per-request final
        states without cross-sequence leakage.
"""

from typing import TYPE_CHECKING, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

if TYPE_CHECKING:
    from megatron.core.ssm.mamba_layer import MambaLayer


def forward_mamba_layer_parallel_with_all_states(
    layer: 'MambaLayer', hidden_states: Tensor, block_size: int
) -> Tuple[Tensor, Tensor, Tensor]:
    """Run a Mamba layer on the full sequence and extract states at every block boundary.

    Used by the **context tower**.  Calls the Triton
    ``_mamba_chunk_scan_combined_fwd`` kernel directly to obtain intermediate
    chunk states, which are not exposed by the public ``mamba_chunk_scan_combined``
    API.  The intermediate states record the SSM hidden state after each
    ``chunk_size``-token chunk; because ``block_size == chunk_size``, these
    correspond exactly to diffusion-block boundaries.

    Conv states are extracted from the *pre-activation* projection
    (``xBC``) at each block boundary by slicing the last ``d_conv - 1``
    positions.

    Args:
        layer (MambaLayer): A ``MambaLayer`` instance from the context tower.
        hidden_states (Tensor): Input tensor ``(S_local, B, D)``, local to
            this TP rank when sequence parallelism is enabled.
        block_size (int): Tokens per diffusion block.  Must equal
            ``layer.mixer.chunk_size``.

    Returns:
        Tuple[Tensor, Tensor, Tensor]:
            - **output** ``(S_local, B, D)`` — layer output hidden states.
            - **conv_states** ``(B, num_blocks_local, conv_dim, d_conv-1)`` —
              conv1d history at each block boundary.
            - **ssm_states** ``(B, num_blocks_local, nheads, headdim, d_state)``
              — SSM hidden states at each block boundary.
    """
    from einops import rearrange
    from mamba_ssm.ops.triton.ssd_combined import _mamba_chunk_scan_combined_fwd

    try:
        from causal_conv1d import causal_conv1d_fn
    except ImportError:
        causal_conv1d_fn = None

    mixer = layer.mixer
    local_seq_len, batch_size, _ = hidden_states.shape
    local_num_blocks = local_seq_len // block_size

    tp_group = mixer.pg_collection.tp
    tp_size = tp_group.size() if tp_group is not None else 1
    tp_rank = torch.distributed.get_rank(tp_group) if tp_group is not None and tp_size > 1 else 0

    assert local_seq_len % block_size == 0
    assert block_size == mixer.chunk_size

    # Pre-norm
    residual = hidden_states
    if layer.config.fp32_residual_connection:
        residual = residual.to(torch.float32)
    hidden_states = hidden_states.to(dtype=layer.config.params_dtype)
    hidden_states = layer.norm(hidden_states)

    # in_proj (gathers local -> global when SP is enabled)
    zxBCdt, _ = mixer.in_proj(hidden_states)
    global_seq_len = zxBCdt.shape[0]
    zxBCdt = rearrange(zxBCdt, "s b d -> b s d").contiguous()

    d_inner = mixer.cp.d_inner_local_tpcp
    ngroups = mixer.cp.ngroups_local_tpcp
    nheads = mixer.cp.nheads_local_tpcp
    d_state = mixer.d_state
    headdim = mixer.headdim
    conv_dim = d_inner + 2 * ngroups * d_state
    conv_state_width = mixer.d_conv - 1

    z, xBC, dt = torch.split(zxBCdt, [d_inner, conv_dim, nheads], dim=-1)

    # Conv1d on full global sequence
    xBC_t = xBC.transpose(1, 2)
    use_causal_conv1d = (
        causal_conv1d_fn is not None and xBC_t.stride(0) % 8 == 0 and xBC_t.stride(2) % 8 == 0
    )

    if use_causal_conv1d:
        xBC_conv = causal_conv1d_fn(
            xBC_t,
            rearrange(mixer.conv1d.weight, "d 1 w -> d w"),
            mixer.conv1d.bias,
            activation=mixer.activation,
        )
    else:
        xBC_t_cont = xBC_t.contiguous()
        xBC_conv = F.conv1d(
            F.pad(xBC_t_cont, (mixer.d_conv - 1, 0)),
            mixer.conv1d.weight,
            mixer.conv1d.bias,
            groups=conv_dim,
        )
        if mixer.activation == "silu":
            xBC_conv = F.silu(xBC_conv)

    xBC_conv = xBC_conv.transpose(1, 2).contiguous()

    # Extract conv states at LOCAL block boundaries
    global_offset = tp_rank * local_seq_len
    conv_states_list = []
    for block_idx in range(local_num_blocks):
        global_block_end = global_offset + (block_idx + 1) * block_size
        if global_block_end >= conv_state_width:
            conv_state = xBC[
                :, global_block_end - conv_state_width : global_block_end, :
            ].transpose(1, 2)
        else:
            padding = torch.zeros(
                batch_size,
                conv_dim,
                conv_state_width - global_block_end,
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
            conv_state = torch.cat([padding, xBC[:, :global_block_end, :].transpose(1, 2)], dim=2)
        conv_states_list.append(conv_state)
    conv_states = torch.stack(conv_states_list, dim=1)

    # SSM inputs
    x = xBC_conv[:, :, :d_inner]
    B_proj = xBC_conv[:, :, d_inner : d_inner + ngroups * d_state]
    C_proj = xBC_conv[:, :, d_inner + ngroups * d_state :]

    x = rearrange(x, "b s (h p) -> b s h p", p=headdim).contiguous()
    B_proj = rearrange(B_proj, "b s (g n) -> b s g n", n=d_state).contiguous()
    C_proj = rearrange(C_proj, "b s (g n) -> b s g n", n=d_state).contiguous()
    z_proj = rearrange(z, "b s (h p) -> b s h p", p=headdim).contiguous()

    A = -torch.exp(mixer.A_log.float())
    D = rearrange(mixer.D.float(), "(h p) -> h p", p=headdim) if mixer.D_has_hdim else mixer.D

    # Internal API gives us intermediate chunk states
    out, out_x, dt_out, dA_cumsum, internal_states, final_state = _mamba_chunk_scan_combined_fwd(
        x,
        dt.contiguous(),
        A,
        B_proj,
        C_proj,
        block_size,
        D=D,
        z=z_proj if not mixer.rmsnorm else None,
        dt_bias=mixer.dt_bias.float(),
        dt_softplus=True,
    )

    # internal_states[:, 0] is zeros (initial), [:, k] is state AFTER chunk k-1
    all_states = torch.cat([internal_states[:, 1:, :, :, :], final_state.unsqueeze(1)], dim=1)
    state_start = tp_rank * local_num_blocks
    ssm_states = all_states[:, state_start : state_start + local_num_blocks, :, :, :]

    # Output
    y = rearrange(out, "b s h p -> s b (h p)").contiguous()
    if mixer.rmsnorm:
        z_out = rearrange(z_proj, "b s h p -> s b (h p)").contiguous()
        y = mixer.norm(y, z_out)

    y_with_bias = mixer.out_proj(y)

    with layer.bias_dropout_add_exec_handler():
        output = layer.mamba_bda(training=layer.training, fused=layer.config.bias_dropout_fusion)(
            y_with_bias, residual, layer.hidden_dropout
        )

    return output, conv_states, ssm_states


def forward_mamba_layer_batched_with_initial_states(
    layer: 'MambaLayer',
    hidden_states: Tensor,
    block_size: int,
    initial_conv_states: Tensor,
    initial_ssm_states: Tensor,
    bidirectional: bool = False,
    mod_params: Optional[Tuple[Tensor, Tensor, Tensor]] = None,
) -> Tensor:
    """Run a Mamba layer on all denoiser blocks in parallel with per-block initial states.

    Used by the **denoiser tower**.  Reshapes the ``(S, B, D)`` input into
    ``(B * num_blocks, block_size, D)`` independent sub-sequences, injects
    per-block conv and SSM initial states (from the context cache), and runs
    one batched ``mamba_chunk_scan_combined`` call.  Because
    ``block_size == chunk_size``, each sub-sequence is exactly one SSM chunk
    and the blocks do not share recurrent state with each other.

    When *bidirectional* is ``True``, a second backward pass is run on the
    time-reversed sequence with **zero** initial states (no future context
    leaks), and the forward and backward outputs are averaged.  The shared
    ``in_proj`` and ``out_proj`` weights are reused for both directions,
    adding approximately 30 % compute overhead versus a full second tower.

    When *mod_params* is provided (time conditioning), the post-norm hidden
    states are modulated with ``(shift, scale)`` before the ``in_proj``, and
    the layer output is multiplied by ``gate`` before the residual add.

    When sequence parallelism is enabled, the initial states are all-gathered
    across TP ranks before batching.

    Args:
        layer (MambaLayer): A ``MambaLayer`` instance from the denoiser tower.
        hidden_states (Tensor): Denoiser embeddings ``(S_local, B, D)``.
        block_size (int): Tokens per block.  Must equal
            ``layer.mixer.chunk_size``.
        initial_conv_states (Tensor): Conv1d history per block
            ``(B, num_blocks_local, conv_dim, d_conv-1)``.
        initial_ssm_states (Tensor): SSM hidden states per block
            ``(B, num_blocks_local, nheads, headdim, d_state)``.
        bidirectional (bool): Run a second reversed-sequence SSM pass and
            average with the forward output.  Only used by the denoiser;
            the context tower is always unidirectional.
        mod_params (Optional[Tuple[Tensor, Tensor, Tensor]]): Per-layer
            ``(shift, scale, gate)`` from :func:`get_modulation_params`.

    Returns:
        Tensor: Output hidden states ``(S_local, B, D)``.
    """
    from einops import rearrange
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

    try:
        from causal_conv1d import causal_conv1d_fn
    except ImportError:
        causal_conv1d_fn = None

    mixer = layer.mixer
    local_seq_len, batch_size, hidden_dim = hidden_states.shape
    local_num_blocks = local_seq_len // block_size

    assert local_seq_len % block_size == 0
    assert block_size == mixer.chunk_size

    tp_group = mixer.pg_collection.tp
    tp_size = tp_group.size() if tp_group is not None else 1
    sequence_parallel = layer.config.sequence_parallel

    d_inner = mixer.cp.d_inner_local_tpcp
    ngroups = mixer.cp.ngroups_local_tpcp
    nheads = mixer.cp.nheads_local_tpcp
    d_state = mixer.d_state
    headdim = mixer.headdim
    conv_dim = d_inner + 2 * ngroups * d_state

    # Pre-norm
    residual = hidden_states
    if layer.config.fp32_residual_connection:
        residual = residual.to(torch.float32)
    hidden_states = hidden_states.to(dtype=layer.config.params_dtype)
    hidden_states = layer.norm(hidden_states)

    if mod_params is not None:
        from megatron.diffusion.two_tower.time_conditioning import modulate

        shift, scale, gate = mod_params
        hidden_states = modulate(hidden_states, shift, scale)

    # in_proj (gathers local -> global when SP)
    zxBCdt, _ = mixer.in_proj(hidden_states)
    global_seq_len = zxBCdt.shape[0]
    global_num_blocks = global_seq_len // block_size

    # All-gather initial states when sequence-parallel is enabled
    if sequence_parallel and tp_size > 1:
        conv_list = [torch.empty_like(initial_conv_states) for _ in range(tp_size)]
        torch.distributed.all_gather(conv_list, initial_conv_states.contiguous(), group=tp_group)
        initial_conv_global = torch.cat(conv_list, dim=1)

        ssm_list = [torch.empty_like(initial_ssm_states) for _ in range(tp_size)]
        torch.distributed.all_gather(ssm_list, initial_ssm_states.contiguous(), group=tp_group)
        initial_ssm_global = torch.cat(ssm_list, dim=1)
    else:
        initial_conv_global = initial_conv_states
        initial_ssm_global = initial_ssm_states

    zxBCdt = rearrange(zxBCdt, "s b d -> b s d").contiguous()
    z, xBC, dt = torch.split(zxBCdt, [d_inner, conv_dim, nheads], dim=-1)

    # Reshape into batched blocks
    z_batched = rearrange(
        z, "b (n s) d -> (b n) s d", n=global_num_blocks, s=block_size
    ).contiguous()
    xBC_batched = rearrange(
        xBC, "b (n s) d -> (b n) s d", n=global_num_blocks, s=block_size
    ).contiguous()
    dt_batched = rearrange(
        dt, "b (n s) h -> (b n) s h", n=global_num_blocks, s=block_size
    ).contiguous()

    batched_conv_init = rearrange(initial_conv_global, "b n c w -> (b n) c w").contiguous()
    batched_ssm_init = rearrange(initial_ssm_global, "b n h p s -> (b n) h p s").contiguous()

    # Conv1d with batched initial states
    xBC_t = xBC_batched.transpose(1, 2)
    use_causal_conv1d = (
        causal_conv1d_fn is not None and xBC_t.stride(0) % 8 == 0 and xBC_t.stride(2) % 8 == 0
    )

    if use_causal_conv1d:
        conv_init_proper = batched_conv_init.transpose(1, 2).contiguous().transpose(1, 2)
        xBC_conv = causal_conv1d_fn(
            xBC_t,
            rearrange(mixer.conv1d.weight, "d 1 w -> d w"),
            mixer.conv1d.bias,
            activation=mixer.activation,
            initial_states=conv_init_proper,
        )
    else:
        xBC_t_cont = xBC_t.contiguous()
        xBC_with_history = torch.cat([batched_conv_init, xBC_t_cont], dim=2)
        xBC_conv = F.conv1d(
            xBC_with_history, mixer.conv1d.weight, mixer.conv1d.bias, groups=conv_dim
        )
        if mixer.activation == "silu":
            xBC_conv = F.silu(xBC_conv)

    xBC_conv = xBC_conv.transpose(1, 2).contiguous()

    x = xBC_conv[:, :, :d_inner]
    B_proj = xBC_conv[:, :, d_inner : d_inner + ngroups * d_state]
    C_proj = xBC_conv[:, :, d_inner + ngroups * d_state :]

    x = rearrange(x, "b s (h p) -> b s h p", p=headdim).contiguous()
    B_proj = rearrange(B_proj, "b s (g n) -> b s g n", n=d_state).contiguous()
    C_proj = rearrange(C_proj, "b s (g n) -> b s g n", n=d_state).contiguous()
    z_proj = rearrange(z_batched, "b s (h p) -> b s h p", p=headdim).contiguous()

    A = -torch.exp(mixer.A_log.float())
    D = rearrange(mixer.D.float(), "(h p) -> h p", p=headdim) if mixer.D_has_hdim else mixer.D

    y_fwd, _ = mamba_chunk_scan_combined(
        x,
        dt_batched.contiguous(),
        A,
        B_proj,
        C_proj,
        mixer.chunk_size,
        D=D,
        z=z_proj if not mixer.rmsnorm else None,
        dt_bias=mixer.dt_bias.float(),
        dt_softplus=True,
        return_final_states=True,
        initial_states=batched_ssm_init,
    )

    if bidirectional:
        xBC_batched_bwd = xBC_batched.flip(dims=[1])
        dt_batched_bwd = dt_batched.flip(dims=[1])

        xBC_t_bwd = xBC_batched_bwd.transpose(1, 2)
        zero_conv_init = torch.zeros_like(batched_conv_init)

        if use_causal_conv1d:
            conv_init_bwd = zero_conv_init.transpose(1, 2).contiguous().transpose(1, 2)
            xBC_conv_bwd = causal_conv1d_fn(
                xBC_t_bwd,
                rearrange(mixer.conv1d.weight, "d 1 w -> d w"),
                mixer.conv1d.bias,
                activation=mixer.activation,
                initial_states=conv_init_bwd,
            )
        else:
            xBC_t_bwd_cont = xBC_t_bwd.contiguous()
            xBC_with_history_bwd = torch.cat([zero_conv_init, xBC_t_bwd_cont], dim=2)
            xBC_conv_bwd = F.conv1d(
                xBC_with_history_bwd, mixer.conv1d.weight, mixer.conv1d.bias, groups=conv_dim
            )
            if mixer.activation == "silu":
                xBC_conv_bwd = F.silu(xBC_conv_bwd)

        xBC_conv_bwd = xBC_conv_bwd.transpose(1, 2).contiguous()

        x_bwd = xBC_conv_bwd[:, :, :d_inner]
        B_proj_bwd = xBC_conv_bwd[:, :, d_inner : d_inner + ngroups * d_state]
        C_proj_bwd = xBC_conv_bwd[:, :, d_inner + ngroups * d_state :]

        x_bwd = rearrange(x_bwd, "b s (h p) -> b s h p", p=headdim).contiguous()
        B_proj_bwd = rearrange(B_proj_bwd, "b s (g n) -> b s g n", n=d_state).contiguous()
        C_proj_bwd = rearrange(C_proj_bwd, "b s (g n) -> b s g n", n=d_state).contiguous()
        z_proj_bwd = rearrange(
            z_batched.flip(dims=[1]), "b s (h p) -> b s h p", p=headdim
        ).contiguous()

        zero_ssm_init = torch.zeros_like(batched_ssm_init)
        y_bwd, _ = mamba_chunk_scan_combined(
            x_bwd,
            dt_batched_bwd.contiguous(),
            A,
            B_proj_bwd,
            C_proj_bwd,
            mixer.chunk_size,
            D=D,
            z=z_proj_bwd if not mixer.rmsnorm else None,
            dt_bias=mixer.dt_bias.float(),
            dt_softplus=True,
            return_final_states=True,
            initial_states=zero_ssm_init,
        )

        y_bwd = y_bwd.flip(dims=[1])
        y = 0.5 * (y_fwd + y_bwd)
    else:
        y = y_fwd

    # Un-batch: (B*N, block_size, h, p) -> (B, global_seq, h*p)
    y = rearrange(y, "(b n) s h p -> b (n s) (h p)", b=batch_size, n=global_num_blocks).contiguous()

    if mixer.rmsnorm:
        z_full = rearrange(z, "b s (h p) -> b s (h p)", p=headdim)
        y = rearrange(y, "b s d -> s b d")
        z_out = rearrange(z_full, "b s d -> s b d")
        y = mixer.norm(y, z_out)
    else:
        y = rearrange(y, "b s d -> s b d")

    y_with_bias = mixer.out_proj(y)

    if mod_params is not None:
        y_out = y_with_bias[0] if isinstance(y_with_bias, tuple) else y_with_bias
        y_out = gate.unsqueeze(0) * y_out
        y_with_bias = (y_out, y_with_bias[1]) if isinstance(y_with_bias, tuple) else y_out

    with layer.bias_dropout_add_exec_handler():
        output = layer.mamba_bda(training=layer.training, fused=layer.config.bias_dropout_fusion)(
            y_with_bias, residual, layer.hidden_dropout
        )

    return output


def forward_mamba_layer_with_states(
    layer: 'MambaLayer',
    hidden_states: Tensor,
    initial_conv_state: Optional[Tensor] = None,
    initial_ssm_state: Optional[Tensor] = None,
    bidirectional: bool = False,
    mod_params: Optional[Tuple[Tensor, Tensor, Tensor]] = None,
) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
    """Run a Mamba layer with explicit initial states and return the final states.

    Intended for **inference / cache extension** where a single contiguous
    sequence is processed and the final conv + SSM states are retained for
    the next decoding step.  Pass ``None`` for initial states to start from
    zeros.

    When *bidirectional* is ``True``, a second reversed-sequence pass is run
    with zero initial states and averaged with the forward output.  The
    returned final states are from the **forward** pass only (the backward
    pass does not contribute to the recurrent state).

    When *mod_params* is provided (time conditioning), the post-norm hidden
    states are modulated with ``(shift, scale)`` before the ``in_proj``, and
    the layer output is multiplied by ``gate`` before the residual add.

    Args:
        layer (MambaLayer): A ``MambaLayer`` instance.
        hidden_states (Tensor): Input tokens ``(S, B, D)``.
        initial_conv_state (Optional[Tensor]): Conv1d history
            ``(B, conv_dim, d_conv-1)`` or ``None`` for zeros.
        initial_ssm_state (Optional[Tensor]): SSM hidden state
            ``(B, nheads, headdim, d_state)`` or ``None`` for zeros.
        bidirectional (bool): Run a second reversed-sequence SSM pass and
            average with the forward output.
        mod_params (Optional[Tuple[Tensor, Tensor, Tensor]]): Per-layer
            ``(shift, scale, gate)`` from :func:`get_modulation_params`.

    Returns:
        Tuple[Tensor, Tuple[Tensor, Tensor]]:
            - **output** ``(S, B, D)`` — layer output hidden states.
            - **(final_conv_state, final_ssm_state)** — states from the
              forward pass to carry forward to the next call.
    """
    from einops import rearrange
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

    try:
        from causal_conv1d import causal_conv1d_fn
    except ImportError:
        causal_conv1d_fn = None

    mixer = layer.mixer
    seq_len, batch_size, _ = hidden_states.shape

    residual = hidden_states
    if layer.config.fp32_residual_connection:
        residual = residual.to(torch.float32)
    hidden_states = hidden_states.to(dtype=layer.config.params_dtype)
    hidden_states = layer.norm(hidden_states)

    if mod_params is not None:
        from megatron.diffusion.two_tower.time_conditioning import modulate

        shift, scale, gate = mod_params
        hidden_states = modulate(hidden_states, shift, scale)

    zxBCdt, _ = mixer.in_proj(hidden_states)
    zxBCdt = rearrange(zxBCdt, "s b d -> b s d").contiguous()

    d_inner = mixer.cp.d_inner_local_tpcp
    ngroups = mixer.cp.ngroups_local_tpcp
    nheads = mixer.cp.nheads_local_tpcp
    d_state = mixer.d_state
    headdim = mixer.headdim
    conv_dim = d_inner + 2 * ngroups * d_state
    conv_state_width = mixer.d_conv - 1

    z, xBC, dt = torch.split(zxBCdt, [d_inner, conv_dim, nheads], dim=-1)

    if initial_conv_state is None:
        initial_conv_state = torch.zeros(
            batch_size,
            conv_state_width,
            conv_dim,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        ).transpose(1, 2)
    if initial_ssm_state is None:
        initial_ssm_state = torch.zeros(
            batch_size,
            nheads,
            headdim,
            d_state,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

    xBC_t = xBC.transpose(1, 2)
    use_causal_conv1d = (
        causal_conv1d_fn is not None and xBC_t.stride(0) % 8 == 0 and xBC_t.stride(2) % 8 == 0
    )

    if use_causal_conv1d:
        xBC_conv = causal_conv1d_fn(
            xBC_t,
            rearrange(mixer.conv1d.weight, "d 1 w -> d w"),
            mixer.conv1d.bias,
            activation=mixer.activation,
            initial_states=initial_conv_state,
        )
        if seq_len >= conv_state_width:
            final_conv_state = xBC[:, -conv_state_width:, :].transpose(1, 2)
        else:
            init_tokens = initial_conv_state.transpose(1, 2)
            combined = torch.cat([init_tokens, xBC], dim=1)
            final_conv_state = combined[:, -conv_state_width:, :].transpose(1, 2)
    else:
        xBC_t_cont = xBC_t.contiguous()
        if initial_conv_state is not None:
            xBC_with_history = torch.cat([initial_conv_state, xBC_t_cont], dim=2)
            xBC_conv = F.conv1d(
                xBC_with_history, mixer.conv1d.weight, mixer.conv1d.bias, groups=conv_dim
            )
        else:
            xBC_conv = F.conv1d(
                F.pad(xBC_t_cont, (mixer.d_conv - 1, 0)),
                mixer.conv1d.weight,
                mixer.conv1d.bias,
                groups=conv_dim,
            )
        if mixer.activation == "silu":
            xBC_conv = F.silu(xBC_conv)
        if seq_len >= conv_state_width:
            final_conv_state = xBC[:, -conv_state_width:, :].transpose(1, 2).contiguous()
        else:
            if initial_conv_state is not None:
                init_tokens = initial_conv_state.transpose(1, 2)
                combined = torch.cat([init_tokens, xBC], dim=1)
                final_conv_state = combined[:, -conv_state_width:, :].transpose(1, 2).contiguous()
            else:
                final_conv_state = xBC[:, -conv_state_width:, :].transpose(1, 2).contiguous()

    xBC_conv = xBC_conv.transpose(1, 2).contiguous()

    x = xBC_conv[:, :, :d_inner]
    B_proj = xBC_conv[:, :, d_inner : d_inner + ngroups * d_state]
    C_proj = xBC_conv[:, :, d_inner + ngroups * d_state :]

    x = rearrange(x, "b s (h p) -> b s h p", p=headdim).contiguous()
    B_proj = rearrange(B_proj, "b s (g n) -> b s g n", n=d_state).contiguous()
    C_proj = rearrange(C_proj, "b s (g n) -> b s g n", n=d_state).contiguous()
    z_proj = rearrange(z, "b s (h p) -> b s h p", p=headdim).contiguous()

    A = -torch.exp(mixer.A_log.float())
    D = rearrange(mixer.D.float(), "(h p) -> h p", p=headdim) if mixer.D_has_hdim else mixer.D

    y_fwd, final_ssm_state = mamba_chunk_scan_combined(
        x,
        dt.contiguous(),
        A,
        B_proj,
        C_proj,
        mixer.chunk_size,
        D=D,
        z=z_proj if not mixer.rmsnorm else None,
        dt_bias=mixer.dt_bias.float(),
        dt_softplus=True,
        return_final_states=True,
        initial_states=initial_ssm_state,
    )

    if bidirectional:
        xBC_bwd = xBC.flip(dims=[1])
        dt_bwd = dt.flip(dims=[1])

        xBC_t_bwd = xBC_bwd.transpose(1, 2)
        zero_conv = torch.zeros_like(initial_conv_state)

        if use_causal_conv1d:
            xBC_conv_bwd = causal_conv1d_fn(
                xBC_t_bwd,
                rearrange(mixer.conv1d.weight, "d 1 w -> d w"),
                mixer.conv1d.bias,
                activation=mixer.activation,
                initial_states=zero_conv,
            )
        else:
            xBC_t_bwd_cont = xBC_t_bwd.contiguous()
            xBC_with_history_bwd = torch.cat([zero_conv, xBC_t_bwd_cont], dim=2)
            xBC_conv_bwd = F.conv1d(
                xBC_with_history_bwd, mixer.conv1d.weight, mixer.conv1d.bias, groups=conv_dim
            )
            if mixer.activation == "silu":
                xBC_conv_bwd = F.silu(xBC_conv_bwd)

        xBC_conv_bwd = xBC_conv_bwd.transpose(1, 2).contiguous()

        x_bwd = xBC_conv_bwd[:, :, :d_inner]
        B_proj_bwd = xBC_conv_bwd[:, :, d_inner : d_inner + ngroups * d_state]
        C_proj_bwd = xBC_conv_bwd[:, :, d_inner + ngroups * d_state :]

        x_bwd = rearrange(x_bwd, "b s (h p) -> b s h p", p=headdim).contiguous()
        B_proj_bwd = rearrange(B_proj_bwd, "b s (g n) -> b s g n", n=d_state).contiguous()
        C_proj_bwd = rearrange(C_proj_bwd, "b s (g n) -> b s g n", n=d_state).contiguous()
        z_proj_bwd = rearrange(z.flip(dims=[1]), "b s (h p) -> b s h p", p=headdim).contiguous()

        zero_ssm = torch.zeros_like(initial_ssm_state)
        y_bwd, _ = mamba_chunk_scan_combined(
            x_bwd,
            dt_bwd.contiguous(),
            A,
            B_proj_bwd,
            C_proj_bwd,
            mixer.chunk_size,
            D=D,
            z=z_proj_bwd if not mixer.rmsnorm else None,
            dt_bias=mixer.dt_bias.float(),
            dt_softplus=True,
            return_final_states=True,
            initial_states=zero_ssm,
        )

        y_bwd = y_bwd.flip(dims=[1])
        y = 0.5 * (y_fwd + y_bwd)
    else:
        y = y_fwd

    y = rearrange(y, "b s h p -> s b (h p)").contiguous()
    if mixer.rmsnorm:
        z_out = rearrange(z_proj, "b s h p -> s b (h p)").contiguous()
        y = mixer.norm(y, z_out)

    y_with_bias = mixer.out_proj(y)

    if mod_params is not None:
        y_out = y_with_bias[0] if isinstance(y_with_bias, tuple) else y_with_bias
        y_out = gate.unsqueeze(0) * y_out
        y_with_bias = (y_out, y_with_bias[1]) if isinstance(y_with_bias, tuple) else y_out

    with layer.bias_dropout_add_exec_handler():
        output = layer.mamba_bda(training=layer.training, fused=layer.config.bias_dropout_fusion)(
            y_with_bias, residual, layer.hidden_dropout
        )

    return output, (final_conv_state, final_ssm_state)


def _build_varlen_chunk_metadata(
    cu_seqlens: Tensor, chunk_size: int, device: torch.device
) -> Tuple[Tensor, Tensor, Tensor]:
    """Build per-chunk metadata required by ``mamba_chunk_scan_combined_varlen``.

    Args:
        cu_seqlens (Tensor): ``(B+1,)`` int32 cumulative sequence lengths.
        chunk_size (int): SSM chunk size (``mixer.chunk_size``).
        device (torch.device): Target device.

    Returns:
        Tuple[Tensor, Tensor, Tensor]: ``(cu_chunk_seqlens, last_chunk_indices,
        seq_idx)`` tensors on *device*.
    """
    B = cu_seqlens.shape[0] - 1
    seq_lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()

    cu_chunk_seqlens = [0]
    seq_idx = []
    chunks_per_req = []

    for i in range(B):
        remaining = seq_lengths[i]
        n_chunks = 0
        while remaining > 0:
            chunk_len = min(remaining, chunk_size)
            cu_chunk_seqlens.append(cu_chunk_seqlens[-1] + chunk_len)
            seq_idx.append(i)
            remaining -= chunk_len
            n_chunks += 1
        chunks_per_req.append(n_chunks)

    cu_chunk_seqlens = torch.tensor(cu_chunk_seqlens, dtype=torch.int32, device=device)
    seq_idx = torch.tensor(seq_idx, dtype=torch.int32, device=device)

    cum_chunks = 0
    last_chunk_indices_list = []
    for nc in chunks_per_req:
        cum_chunks += nc
        last_chunk_indices_list.append(cum_chunks - 1)
    last_chunk_indices = torch.tensor(last_chunk_indices_list, dtype=torch.int32, device=device)

    return cu_chunk_seqlens, last_chunk_indices, seq_idx


def forward_mamba_layer_varlen_with_states(
    layer: 'MambaLayer', hidden_states: Tensor, cu_seqlens: Tensor
) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
    """Run a Mamba layer on packed variable-length sequences.

    Mirrors :func:`forward_mamba_layer_with_states` but operates on a
    1-D packed tensor using ``causal_conv1d_varlen_fn`` and
    ``mamba_chunk_scan_combined_varlen``.  No cross-sequence state leakage
    occurs — each request is isolated by ``cu_seqlens`` / ``seq_idx``.

    Args:
        layer (MambaLayer): ``MambaLayer`` instance from the context tower.
        hidden_states (Tensor): ``(total_tokens, 1, D)`` packed hidden states
            (SBD layout with ``B=1``).
        cu_seqlens (Tensor): ``(num_requests+1,)`` int32 cumulative sequence
            lengths.

    Returns:
        Tuple[Tensor, Tuple[Tensor, Tensor]]: ``(output, (final_conv_states,
        final_ssm_states))`` where *output* is ``(total_tokens, 1, D)``,
        *final_conv_states* is ``(num_requests, conv_dim, d_conv-1)``, and
        *final_ssm_states* is ``(num_requests, nheads, headdim, d_state)``.
    """
    from einops import rearrange

    from megatron.core.ssm.ops import causal_conv1d_varlen_fn, mamba_chunk_scan_combined_varlen

    mixer = layer.mixer
    total_tokens = hidden_states.shape[0]
    B = cu_seqlens.shape[0] - 1
    device = hidden_states.device

    # Pre-norm
    residual = hidden_states
    if layer.config.fp32_residual_connection:
        residual = residual.to(torch.float32)
    hidden_states = hidden_states.to(dtype=layer.config.params_dtype)
    hidden_states = layer.norm(hidden_states)

    # Project
    zxBCdt, _ = mixer.in_proj(hidden_states)  # (total_tokens, 1, proj_dim)
    zxBCdt = zxBCdt.squeeze(1)  # (total_tokens, proj_dim)

    d_inner = mixer.cp.d_inner_local_tpcp
    ngroups = mixer.cp.ngroups_local_tpcp
    nheads = mixer.cp.nheads_local_tpcp
    d_state = mixer.d_state
    headdim = mixer.headdim
    conv_dim = d_inner + 2 * ngroups * d_state
    conv_state_width = mixer.d_conv - 1

    z, xBC, dt = torch.split(zxBCdt, [d_inner, conv_dim, nheads], dim=-1)

    # Varlen conv1d
    xBC_conv = causal_conv1d_varlen_fn(
        xBC.contiguous(),
        rearrange(mixer.conv1d.weight, "d 1 w -> d w"),
        mixer.conv1d.bias,
        cu_seqlens,
        initial_states=None,
        activation=mixer.activation,
    )

    # Extract per-request final conv states from pre-activation xBC
    seq_lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).long()
    final_conv_states = torch.zeros(B, conv_dim, conv_state_width, device=device, dtype=xBC.dtype)
    for i in range(B):
        end = cu_seqlens[i + 1].item()
        take = min(seq_lengths[i].item(), conv_state_width)
        final_conv_states[i, :, conv_state_width - take :] = xBC[end - take : end].T

    # Split conv output
    x = xBC_conv[:, :d_inner]
    B_proj = xBC_conv[:, d_inner : d_inner + ngroups * d_state]
    C_proj = xBC_conv[:, d_inner + ngroups * d_state :]

    # Reshape for SSM — varlen layout is (total_tokens, ...)
    x = rearrange(x, "t (h p) -> t h p", p=headdim).contiguous()
    B_proj = rearrange(B_proj, "t (g n) -> t g n", n=d_state).contiguous()
    C_proj = rearrange(C_proj, "t (g n) -> t g n", n=d_state).contiguous()

    A = -torch.exp(mixer.A_log.float())
    D = rearrange(mixer.D.float(), "(h p) -> h p", p=headdim) if mixer.D_has_hdim else mixer.D

    # Build chunk metadata
    cu_chunk_seqlens, last_chunk_indices, seq_idx = _build_varlen_chunk_metadata(
        cu_seqlens, mixer.chunk_size, device
    )

    # Varlen SSM scan
    out_buf = torch.empty_like(x)
    z_for_scan = (
        rearrange(z, "t (h p) -> t h p", p=headdim).contiguous() if not mixer.rmsnorm else None
    )

    initial_ssm_states = torch.zeros(B, nheads, headdim, d_state, device=device, dtype=x.dtype)

    final_ssm_states = mamba_chunk_scan_combined_varlen(
        x,
        dt.contiguous(),
        A,
        B_proj,
        C_proj,
        mixer.chunk_size,
        cu_chunk_seqlens,
        last_chunk_indices,
        seq_idx,
        out_buf,
        D=D,
        z=z_for_scan,
        dt_bias=mixer.dt_bias.float(),
        initial_states=initial_ssm_states,
        dt_softplus=True,
    )

    y = out_buf
    if mixer.rmsnorm:
        y = rearrange(y, "t h p -> t (h p)")
        y = mixer.norm(y, z)
    else:
        y = rearrange(y, "t h p -> t (h p)")

    # Output projection — restore SBD layout
    y = y.unsqueeze(1)  # (total_tokens, 1, d_inner)
    y_with_bias = mixer.out_proj(y)

    with layer.bias_dropout_add_exec_handler():
        output = layer.mamba_bda(training=layer.training, fused=layer.config.bias_dropout_fusion)(
            y_with_bias, residual, layer.hidden_dropout
        )

    return output, (final_conv_states.detach(), final_ssm_states.detach())
