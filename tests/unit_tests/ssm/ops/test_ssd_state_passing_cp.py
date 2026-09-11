# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Kernel-level tests for the Mamba2 state-passing context-parallel path.

Each test builds a full-sequence reference on every rank, shards it the way
Megatron's context parallelism would, and checks that the state-passing kernels
reproduce the reference shard for both the forward output and every gradient.

The mixer dimensions are Nemotron-3 Nano's (64 heads of 64, 8 groups, state 128,
chunk 128); the batch size and sequence length are what the parametrization
varies, because those are what the CP sharding and the boundary exchange
actually depend on.
"""

import pytest
import torch

from tests.unit_tests.ssm.mamba_state_passing_cp_utils import (
    MambaModelShape,
    assert_all_close_rms,
    select_balanced_cp_shard,
    select_contiguous_cp_shard,
    select_state_passing_cp_shard,
)
from tests.unit_tests.test_utilities import Utils

try:
    from causal_conv1d import causal_conv1d_fn
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

    from megatron.core.ssm.ops.ssd_state_passing_cp import (
        _causal_conv1d_state_passing_cp_bwd,
        _causal_conv1d_state_passing_cp_fwd,
        _mamba_chunk_scan_combined_state_passing_cp_bwd,
        _mamba_chunk_scan_combined_state_passing_cp_fwd,
        redo_state_passing_cp_load_balancing,
        undo_state_passing_cp_load_balancing,
    )

    HAVE_STATE_PASSING_CP = True
except ImportError:
    HAVE_STATE_PASSING_CP = False

pytestmark = [
    pytest.mark.internal,
    pytest.mark.skipif(
        not HAVE_STATE_PASSING_CP, reason="mamba_ssm and causal_conv1d are required"
    ),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required"),
]

SHAPE = MambaModelShape()

# (batch, sequence_length). Every causal segment must align with the SSD chunk
# size, and ``virtual`` mode halves the segment length, so the sequence length
# must be a multiple of ``4 * cp_size * chunk_size``.
BATCH_AND_SEQUENCE_LENGTHS = ((1, 2048), (3, 4096))


@pytest.fixture
def cp_context(request):
    """Initialize a context-parallel group of the requested size and tear it down."""
    cp_size = request.param
    if Utils.world_size % cp_size != 0:
        pytest.skip(f"world size {Utils.world_size} is not a multiple of cp_size {cp_size}")
    Utils.initialize_model_parallel(context_parallel_size=cp_size)
    from megatron.core import parallel_state

    group = parallel_state.get_context_parallel_group()
    yield group, group.rank(), cp_size, torch.device(torch.cuda.current_device())
    Utils.destroy_model_parallel()


@pytest.mark.parametrize("cp_context", (2, 4), indirect=True)
@pytest.mark.parametrize("backend", ("p2p", "a2a"))
def test_load_balancing_permutation_round_trip(cp_context, backend):
    """The balanced-to-contiguous permutation and its backward must be exact."""
    group, rank, cp_size, device = cp_context
    sequence_length, hidden_size = 2048, SHAPE.hidden_size

    total_chunks = 2 * cp_size
    chunk_length = sequence_length // total_chunks
    full = torch.arange(sequence_length * hidden_size, device=device, dtype=torch.float32).view(
        sequence_length, 1, hidden_size
    )
    grad_full = torch.empty_like(full)
    for chunk_id in range(total_chunks):
        grad_full[chunk_id * chunk_length : (chunk_id + 1) * chunk_length].fill_(chunk_id + 1)

    balanced = select_balanced_cp_shard(full, rank, cp_size).clone().requires_grad_(True)
    contiguous = undo_state_passing_cp_load_balancing(balanced, group, backend=backend)
    restored = redo_state_passing_cp_load_balancing(contiguous, group, backend=backend)
    (contiguous * select_contiguous_cp_shard(grad_full, rank, cp_size)).sum().backward()

    torch.testing.assert_close(
        contiguous, select_contiguous_cp_shard(full, rank, cp_size), rtol=0, atol=0
    )
    torch.testing.assert_close(restored, balanced.detach(), rtol=0, atol=0)
    torch.testing.assert_close(
        balanced.grad, select_balanced_cp_shard(grad_full, rank, cp_size), rtol=0, atol=0
    )


@pytest.mark.parametrize("cp_context", (2, 4), indirect=True)
@pytest.mark.parametrize("virtual", (False, True))
@pytest.mark.parametrize("batch,sequence_length", BATCH_AND_SEQUENCE_LENGTHS)
def test_causal_conv1d_matches_full_sequence(cp_context, virtual, batch, sequence_length):
    """The halo-exchanging causal conv must match a full-sequence causal_conv1d."""
    group, rank, cp_size, device = cp_context
    channels = SHAPE.conv_dim
    dtype = torch.bfloat16

    generator = torch.Generator(device=device).manual_seed(1234)
    x = torch.randn(
        batch, channels, sequence_length, device=device, dtype=dtype, generator=generator
    )
    weight = torch.randn(channels, SHAPE.d_conv, device=device, dtype=dtype, generator=generator)
    bias = torch.randn(channels, device=device, dtype=dtype, generator=generator)
    # Every input must come from the explicit seeded generator: the per-rank RNG
    # that ``randn_like`` would use differs across ranks, which would give each
    # rank a different "full-sequence" reference.
    grad_output = torch.randn(
        batch, channels, sequence_length, device=device, dtype=dtype, generator=generator
    )

    x_reference = x.clone().requires_grad_(True)
    weight_reference = weight.clone().requires_grad_(True)
    bias_reference = bias.clone().requires_grad_(True)
    y_reference = causal_conv1d_fn(x_reference, weight_reference, bias_reference, activation="silu")
    (y_reference * grad_output).sum().backward()

    def local_view(tensor):
        return select_state_passing_cp_shard(
            tensor, rank, cp_size, virtual=virtual, batch_dim=0, sequence_dim=2
        )

    # causal_conv1d requires a channel-last-in-memory layout.
    x_local = local_view(x).transpose(1, 2).contiguous().transpose(1, 2)
    y_local, conv_initial_states = _causal_conv1d_state_passing_cp_fwd(
        x_local,
        weight.clone(),
        bias.clone(),
        activation="silu",
        state_passing_cp_group=group,
        state_passing_cp_virtual=virtual,
    )
    dx_local, dweight_local, dbias_local = _causal_conv1d_state_passing_cp_bwd(
        x_local,
        weight.clone(),
        bias.clone(),
        local_view(grad_output),
        initial_states=conv_initial_states,
        activation="silu",
        state_passing_cp_group=group,
        state_passing_cp_virtual=virtual,
    )

    # Weight and bias are replicated across CP, so their gradients are summed.
    dweight = dweight_local.detach().float()
    dbias = dbias_local.detach().float()
    torch.distributed.all_reduce(dweight, group=group)
    torch.distributed.all_reduce(dbias, group=group)

    assert_all_close_rms(
        {
            "conv output": (y_local, local_view(y_reference)),
            "conv dx": (dx_local, local_view(x_reference.grad)),
            "conv dweight": (dweight, weight_reference.grad),
            "conv dbias": (dbias, bias_reference.grad),
        },
        group,
    )


@pytest.mark.parametrize("cp_context", (2, 4), indirect=True)
@pytest.mark.parametrize("virtual", (False, True))
@pytest.mark.parametrize("batch,sequence_length", BATCH_AND_SEQUENCE_LENGTHS)
def test_chunk_scan_combined_matches_full_sequence(cp_context, virtual, batch, sequence_length):
    """The boundary-scanning SSD kernel must match a full-sequence chunk scan."""
    group, rank, cp_size, device = cp_context
    nheads, headdim, ngroups = SHAPE.nheads, SHAPE.head_dim, SHAPE.ngroups
    dstate, chunk_size = SHAPE.state_dim, SHAPE.chunk_size
    segment_length = sequence_length // (cp_size * (2 if virtual else 1))
    assert (
        segment_length % chunk_size == 0
    ), f"each causal segment ({segment_length}) must align with the SSD chunk size"
    dtype = torch.bfloat16
    kernel_kwargs = dict(dt_softplus=True, dt_limit=(0.0, float("inf")))

    generator = torch.Generator(device=device).manual_seed(1234)
    x = torch.randn(
        batch, sequence_length, nheads, headdim, device=device, dtype=dtype, generator=generator
    )
    dt = (
        torch.rand(batch, sequence_length, nheads, device=device, dtype=dtype, generator=generator)
        * 0.5
        + 0.01
    )
    A_log = torch.log(
        torch.rand(nheads, device=device, dtype=torch.float32, generator=generator) + 0.1
    )
    B = torch.randn(
        batch, sequence_length, ngroups, dstate, device=device, dtype=dtype, generator=generator
    )
    C = torch.randn(
        batch, sequence_length, ngroups, dstate, device=device, dtype=dtype, generator=generator
    )
    # Every input must come from the explicit seeded generator: the per-rank RNG
    # that ``randn_like`` would use differs across ranks, which would give each
    # rank a different "full-sequence" reference.
    z = torch.randn(
        batch, sequence_length, nheads, headdim, device=device, dtype=dtype, generator=generator
    )
    D = torch.randn(nheads, device=device, dtype=torch.float32, generator=generator)
    dt_bias = torch.randn(nheads, device=device, dtype=dtype, generator=generator) * 0.1
    grad_output = torch.randn(
        batch, sequence_length, nheads, headdim, device=device, dtype=dtype, generator=generator
    )

    def local_view(tensor):
        return select_state_passing_cp_shard(
            tensor, rank, cp_size, virtual=virtual, batch_dim=0, sequence_dim=1
        )

    reference_inputs = [tensor.clone().requires_grad_(True) for tensor in (x, dt, B, C, z)]
    A_log_reference = A_log.clone().requires_grad_(True)
    A_reference = -torch.exp(A_log_reference)
    A_reference.retain_grad()
    D_reference = D.clone().requires_grad_(True)
    dt_bias_reference = dt_bias.clone().requires_grad_(True)
    y_reference = mamba_chunk_scan_combined(
        reference_inputs[0],
        reference_inputs[1],
        A_reference,
        reference_inputs[2],
        reference_inputs[3],
        chunk_size,
        D=D_reference,
        z=reference_inputs[4],
        dt_bias=dt_bias_reference,
        **kernel_kwargs,
    )
    (y_reference * grad_output).sum().backward()

    A = -torch.exp(A_log)
    cp_inputs = [local_view(tensor).clone().requires_grad_(True) for tensor in (x, dt, B, C, z)]
    y_cp, out_x, _, _, _, _, initial_states, gathered_decays = (
        _mamba_chunk_scan_combined_state_passing_cp_fwd(
            cp_inputs[0],
            cp_inputs[1],
            A,
            cp_inputs[2],
            cp_inputs[3],
            chunk_size,
            D=D,
            z=cp_inputs[4],
            dt_bias=dt_bias,
            state_passing_cp_group=group,
            state_passing_cp_virtual=virtual,
            **kernel_kwargs,
        )
    )
    cp_grads = _mamba_chunk_scan_combined_state_passing_cp_bwd(
        local_view(grad_output),
        cp_inputs[0],
        cp_inputs[1],
        A,
        cp_inputs[2],
        cp_inputs[3],
        out_x,
        chunk_size,
        D=D,
        z=cp_inputs[4],
        dt_bias=dt_bias,
        state_passing_initial_states=initial_states,
        state_passing_gathered_decays=gathered_decays,
        state_passing_cp_group=group,
        state_passing_cp_virtual=virtual,
        **kernel_kwargs,
    )
    dx, ddt, dA, dB, dC, dD, dz, ddt_bias = cp_grads[:8]

    # A, D, and dt_bias are replicated across CP, so their gradients are summed.
    reduced = []
    for gradient in (dA, dD, ddt_bias):
        value = gradient.detach().float()
        torch.distributed.all_reduce(value, group=group)
        reduced.append(value)
    dA_reduced, dD_reduced, ddt_bias_reduced = reduced

    assert_all_close_rms(
        {
            "ssd output": (y_cp, local_view(y_reference)),
            "ssd dx": (dx, local_view(reference_inputs[0].grad)),
            "ssd ddt": (ddt, local_view(reference_inputs[1].grad)),
            "ssd dB": (dB, local_view(reference_inputs[2].grad)),
            "ssd dC": (dC, local_view(reference_inputs[3].grad)),
            "ssd dz": (dz, local_view(reference_inputs[4].grad)),
            "ssd dA": (dA_reduced, A_reference.grad),
            "ssd dA_log": (dA_reduced * A, A_log_reference.grad),
            "ssd dD": (dD_reduced, D_reference.grad),
            "ssd ddt_bias": (ddt_bias_reduced, dt_bias_reference.grad),
        },
        group,
    )
