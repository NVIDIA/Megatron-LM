# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""The fused pre-GDR kernels on inputs of more than 2**31 elements.

``qkvzba`` is ``[seq_len, batch, in_proj_dim]``; the kernels address it as
``row * in_proj_dim + channel``. With 32-bit offsets the rows past ``2**31 / in_proj_dim``
(row 260617 at the width used here, i.e. a 256K-token sequence) were read from and written to
wrapped addresses: non-finite activations and gradients in the last rows, and gradient writes
outside the buffers. The test compares the last rows of a long sequence with the same rows
computed alone.
"""

import pytest
import torch

NUM_KEY_HEADS = 8
NUM_VALUE_HEADS = 24
HEAD_DIM = 128
CONV_KERNEL = 4
IN_PROJ_DIM = 2 * NUM_KEY_HEADS * HEAD_DIM + 2 * NUM_VALUE_HEADS * HEAD_DIM + 2 * NUM_VALUE_HEADS
FIRST_WRAPPED_ROW = -(-(2**31) // IN_PROJ_DIM)
SEQ_LEN = 261376  # rows 260617 .. 261375 are past 2**31 elements
TAIL = 2048  # rows recomputed alone
CHECK = TAIL - 64  # rows compared; their causal conv window lies inside the tail
REQUIRED_GIB = 48


def _enough_gpu_memory():
    if not torch.cuda.is_available():
        return False
    free, _ = torch.cuda.mem_get_info()
    return free >= REQUIRED_GIB * 2**30


def _tail(tensor, seq_len, rows):
    """The last ``rows`` entries along the sequence dimension of an output."""
    dim = list(tensor.shape).index(seq_len)
    return tensor.narrow(dim, seq_len - rows, rows)


def _tails(qkvzba, weight, a_log, dt_bias):
    """The compared rows of every output of the fused pre-GDR path."""
    from megatron.core.fusions.fused_pre_gated_delta_rule import fused_streamed_pre_gated_delta_rule

    outputs = fused_streamed_pre_gated_delta_rule(
        qkvzba,
        weight,
        None,
        a_log,
        dt_bias,
        num_key_heads=NUM_KEY_HEADS,
        num_value_heads=NUM_VALUE_HEADS,
        key_head_dim=HEAD_DIM,
        value_head_dim=HEAD_DIM,
    )
    return [_tail(out, qkvzba.shape[0], CHECK) for out in outputs]


def _run(qkvzba, weight, a_log, dt_bias, probes):
    tails = _tails(qkvzba, weight, a_log, dt_bias)
    loss = sum((t.float() * p).sum() for t, p in zip(tails, probes))
    grads = torch.autograd.grad(loss, [qkvzba, weight, a_log, dt_bias])
    return [t.detach().clone() for t in tails], grads


@pytest.mark.skipif(
    not _enough_gpu_memory(), reason=f"needs a GPU with {REQUIRED_GIB} GiB free memory"
)
def test_rows_past_two_to_31_elements_match_the_same_rows_alone():
    pytest.importorskip("triton")
    assert SEQ_LEN * IN_PROJ_DIM > 2**31 and SEQ_LEN - CHECK < FIRST_WRAPPED_ROW < SEQ_LEN
    device = torch.device("cuda")
    gen = torch.Generator(device=device).manual_seed(1234)

    def randn(*shape, dtype=torch.bfloat16):
        return torch.randn(*shape, generator=gen, device=device, dtype=torch.float32).to(dtype)

    conv_dim = 2 * NUM_KEY_HEADS * HEAD_DIM + NUM_VALUE_HEADS * HEAD_DIM
    weight = (0.5 * randn(conv_dim, 1, CONV_KERNEL)).requires_grad_()
    a_log = randn(NUM_VALUE_HEADS, dtype=torch.float32).requires_grad_()
    dt_bias = randn(NUM_VALUE_HEADS, dtype=torch.float32).requires_grad_()
    tail_input = randn(TAIL, 1, IN_PROJ_DIM)

    # Long sequence: zeros (cheap to create) followed by the tail.
    full_input = torch.zeros(SEQ_LEN, 1, IN_PROJ_DIM, device=device, dtype=torch.bfloat16)
    full_input[SEQ_LEN - TAIL :] = tail_input
    full_input.requires_grad_()
    tail_input = tail_input.clone().requires_grad_()

    # One fixed probe per output, so that both runs differentiate the same scalar.
    with torch.no_grad():
        shapes = [t.shape for t in _tails(tail_input, weight, a_log, dt_bias)]
    probes = [randn(*shape, dtype=torch.float32) for shape in shapes]

    ref_out, ref_grads = _run(tail_input, weight, a_log, dt_bias, probes)
    out, grads = _run(full_input, weight, a_log, dt_bias, probes)

    names = ("query", "key", "value", "gate", "beta", "g")
    for name, got, ref in zip(names, out, ref_out):
        assert torch.isfinite(got).all(), f"{name}: non-finite rows past 2**31 elements"
        torch.testing.assert_close(got, ref, rtol=1e-3, atol=1e-3, msg=lambda m: f"{name}: {m}")

    d_full, d_ref = grads[0], ref_grads[0]
    torch.testing.assert_close(d_full[SEQ_LEN - TAIL :], d_ref, rtol=2e-3, atol=2e-3)
    # Rows before the tail feed no compared output: their gradient is exactly zero.
    low, high = torch.aminmax(d_full[: SEQ_LEN - TAIL])
    assert float(low) == 0.0 and float(high) == 0.0
    for name, got, ref in zip(("conv1d_weight", "A_log", "dt_bias"), grads[1:], ref_grads[1:]):
        assert torch.isfinite(got).all(), f"d_{name} is not finite"
        torch.testing.assert_close(
            got.float(), ref.float(), rtol=2e-2, atol=2e-2, msg=lambda m: f"d_{name}: {m}"
        )
