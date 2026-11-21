# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""The KDA tail kernels of the fused pre-KDA path on inputs of more than 2**31 elements.

``raw_g`` and ``gate`` are ``[seq_len, batch, num_heads * head_dim]``; the kernels address them as
``row * width + channel``. With 32-bit offsets the rows past ``2**31 / width`` were read from and
written to wrapped addresses. The test compares the last rows of a long sequence with the same
rows computed alone. The kernels are called directly: the convolution of the full path goes
through causal-conv1d, which has a limit of its own (2**32 elements).
"""

import os

import pytest
import torch

NUM_HEADS = 65
HEAD_DIM = 128
CHANNELS = NUM_HEADS * HEAD_DIM
SEQ_LEN = 261376  # raw_g and gate hold more than 2**31 elements
TAIL = 4096  # rows recomputed alone; they start before row 2**31 / CHANNELS
REQUIRED_GIB = 32


def _long(tail):
    """A SEQ_LEN-row tensor that ends with ``tail``: zeros (cheap to create) before it."""
    dim = list(tail.shape).index(TAIL)
    shape = list(tail.shape)
    shape[dim] = SEQ_LEN
    full = torch.zeros(shape, device=tail.device, dtype=tail.dtype)
    full.narrow(dim, SEQ_LEN - TAIL, TAIL).copy_(tail)
    return full


def _last_rows(tensor):
    dim = list(tensor.shape).index(SEQ_LEN)
    return tensor.narrow(dim, SEQ_LEN - TAIL, TAIL)


def test_rows_past_two_to_31_elements_match_the_same_rows_alone():
    pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    # Every torch.distributed.run rank runs this test: use the GPU of this rank.
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")) % torch.cuda.device_count())
    free, _ = torch.cuda.mem_get_info()
    if free < REQUIRED_GIB * 2**30:
        pytest.skip(f"needs a GPU with {REQUIRED_GIB} GiB free memory")
    from megatron.core.fusions.fused_pre_kda import (
        _triton_prepare_kda_tail,
        _triton_prepare_kda_tail_backward,
    )

    assert SEQ_LEN * CHANNELS > 2**31 > (SEQ_LEN - TAIL) * CHANNELS
    device = torch.device("cuda")
    gen = torch.Generator(device=device).manual_seed(1234)

    def randn(*shape, dtype=torch.bfloat16):
        return torch.randn(*shape, generator=gen, device=device, dtype=torch.float32).to(dtype)

    heads = dict(num_heads=NUM_HEADS, head_dim=HEAD_DIM)
    names = ("raw_g", "gate", "beta")

    # Forward: [seq, batch, channels] inputs -> [batch, seq, heads, head_dim] outputs.
    inputs = [randn(TAIL, 1, CHANNELS), randn(TAIL, 1, CHANNELS), randn(TAIL, 1, NUM_HEADS)]
    ref = _triton_prepare_kda_tail(*inputs, **heads)
    out = _triton_prepare_kda_tail(*[_long(t) for t in inputs], **heads)
    for name, got, want in zip(names, out, ref):
        assert torch.equal(_last_rows(got), want), f"{name}: rows past 2**31 elements differ"
    del out

    # Backward: gradients of the outputs -> gradients in the input layouts.
    beta = inputs[2]
    d_outs = [randn(*t.shape, dtype=t.dtype) for t in ref]
    dtypes = dict(raw_g_dtype=torch.bfloat16, gate_dtype=torch.bfloat16)
    ref = _triton_prepare_kda_tail_backward(beta, *d_outs, **dtypes, **heads)
    out = _triton_prepare_kda_tail_backward(
        _long(beta), *[_long(t) for t in d_outs], **dtypes, **heads
    )
    for name, got, want in zip(names, out, ref):
        assert torch.equal(_last_rows(got), want), f"d_{name}: rows past 2**31 elements differ"
        # Rows before the tail have a zero output gradient: their gradient is exactly zero.
        low, high = torch.aminmax(got[: SEQ_LEN - TAIL])
        assert float(low) == 0.0 and float(high) == 0.0, f"d_{name}: writes before the tail"
