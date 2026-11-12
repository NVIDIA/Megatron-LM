# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Bit-exact replay coverage for fused streamwise wide-residual operations."""

import pytest
import torch

from megatron.core.transformer import streamwise_residual_ops
from tests.unit_tests.determinism.kernels.harness import (
    CONTENTION_TOKENS,
    assert_replays_bit_exact,
    seeded,
)
from tests.unit_tests.determinism.kernels.pdl_harness import (
    assert_pdl_launch_bit_exact,
    requires_pdl,
    seeded_randn,
)

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and streamwise_residual_ops.HAVE_STREAMWISE_TRITON),
    reason="needs a GPU and Triton",
)


def _padded_controller_logits(values: tuple[float, ...]) -> torch.Tensor:
    logits = torch.zeros(128, device="cuda", dtype=torch.float32)
    logits[: len(values)] = torch.tensor(values, device="cuda", dtype=torch.float32)
    return logits.requires_grad_()


@pytest.mark.parametrize(
    "with_retention", [False, True], ids=("identity_carry", "learned_retention")
)
@pytest.mark.parametrize("fp32_residual", [False, True], ids=("bf16_residual", "fp32_residual"))
@pytest.mark.parametrize(
    "stream_width,backward_tile",
    [(256, (64, 128)), (1024, (16, 512))],
    ids=("shared_tile", "wide_backward_tile"),
)
def test_fused_streamwise_read_write_replays_fwd_bwd(
    with_retention, fp32_residual, stream_width, backward_tile
):
    """Replay fused outputs and every activation/controller gradient under contention.

    A width of 1024 selects the 16 x 512 backward tile; a width of 256 keeps the backward
    kernels on the forward tile.
    """

    seeded()
    num_streams = 3
    residual_dtype = torch.float32 if fp32_residual else torch.bfloat16
    branch_dtype = torch.bfloat16
    residual = torch.randn(
        CONTENTION_TOKENS,
        num_streams * stream_width,
        device="cuda",
        dtype=residual_dtype,
        requires_grad=True,
    )
    update = torch.randn(
        CONTENTION_TOKENS, stream_width, device="cuda", dtype=branch_dtype, requires_grad=True
    )
    read_logits = _padded_controller_logits((-0.8, -0.7, -0.6))
    write_logits = _padded_controller_logits((-0.01, 0.0, 0.01))
    retention_logits = _padded_controller_logits((4.8, 4.9, 5.0)) if with_retention else None

    block_batch, block_width, _ = streamwise_residual_ops._streamwise_backward_block_config(
        CONTENTION_TOKENS, stream_width
    )
    assert (block_batch, block_width) == backward_tile

    assert streamwise_residual_ops._can_use_streamwise_triton(
        residual,
        read_logits,
        num_streams,
        stream_width,
        output_dtype=branch_dtype if fp32_residual else None,
    )
    assert streamwise_residual_ops._can_use_streamwise_triton(
        residual, write_logits, num_streams, stream_width
    )
    if retention_logits is not None:
        assert streamwise_residual_ops._can_use_streamwise_triton(
            residual, retention_logits, num_streams, stream_width
        )

    def run(residual, update, read_logits, write_logits, retention_logits):
        read = streamwise_residual_ops.streamwise_sigmoid_read(
            residual, read_logits, num_streams, output_dtype=branch_dtype if fp32_residual else None
        )
        written = streamwise_residual_ops.streamwise_sigmoid_writeback(
            residual,
            update + 0.125 * read,
            write_logits,
            num_streams,
            retention_logits=retention_logits,
            retention_max_forget=0.2,
        )
        return read, written

    _, gradients = assert_replays_bit_exact(
        run,
        (residual, update, read_logits, write_logits, retention_logits),
        replays=3,
        grad_outputs={"out[0]": torch.randn_like(update), "out[1]": torch.randn_like(residual)},
        contention=True,
        what=f"streamwise residual fp32={fp32_residual} retention={with_retention}",
    )

    expected_gradients = {f"in[{index}]" for index in range(4 + int(with_retention))}
    assert set(gradients) == expected_gradients
    for index in (2, 3, *([4] if with_retention else [])):
        assert torch.count_nonzero(gradients[f"in[{index}]"][num_streams:]) == 0


@requires_pdl
@pytest.mark.parametrize("op", ["read", "write"])
def test_streamwise_sigmoid_pdl_launch_is_bit_exact(op):
    """Launched with PDL right after a producer that writes the residual late, the fused read
    and write wait for the producer and match their non-PDL launch byte for byte."""
    seeded()
    num_streams, width = 4, 1024
    hidden = torch.empty(186, num_streams * width, device="cuda", dtype=torch.bfloat16)
    read_logits = torch.zeros(128, device="cuda", dtype=torch.float32)
    read_logits[:num_streams] = torch.tensor([-0.7, -0.2, 0.3, 0.9], device="cuda")
    write_logits = torch.zeros(128, device="cuda", dtype=torch.float32)
    write_logits[:num_streams] = torch.tensor([0.4, -0.1, 0.2, -0.6], device="cuda")
    update = seeded_randn(torch.empty(186, width, dtype=torch.bfloat16), 41)

    if op == "read":

        def call():
            return [
                streamwise_residual_ops._streamwise_sigmoid_read_triton(
                    hidden, read_logits, num_streams, torch.bfloat16
                )
            ]

    else:

        def call():
            return [
                streamwise_residual_ops._streamwise_sigmoid_write_triton(
                    hidden, update, write_logits, num_streams, None, 0.0
                )
            ]

    assert_pdl_launch_bit_exact(
        call, hidden, lambda i: seeded_randn(hidden, i), what=f"streamwise sigmoid {op}"
    )
