# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Tests for native full-width stream read and write operations."""

import pytest
import torch

from megatron.core.transformer.streamwise_residual_ops import (
    HAVE_STREAMWISE_TRITON,
    _can_use_streamwise_triton,
    streamwise_read,
    streamwise_sigmoid_read,
    streamwise_sigmoid_writeback,
    streamwise_writeback,
)


def _reference_read(residual: torch.Tensor, factors: torch.Tensor) -> torch.Tensor:
    """Apply the streamwise read equation without using the production helper."""

    num_streams = factors.numel()
    stream_width = residual.shape[-1] // num_streams
    streams = residual.reshape(*residual.shape[:-1], num_streams, stream_width)
    factor_shape = (1,) * (residual.ndim - 1) + (num_streams, 1)
    return (streams * factors.reshape(factor_shape)).sum(dim=-2)


def _reference_writeback(
    residual: torch.Tensor,
    update: torch.Tensor,
    write_factors: torch.Tensor,
    retention_factors: torch.Tensor | None,
) -> torch.Tensor:
    """Apply the streamwise retained write equation directly."""

    num_streams = write_factors.numel()
    stream_width = residual.shape[-1] // num_streams
    streams = residual.reshape(*residual.shape[:-1], num_streams, stream_width)
    factor_shape = (1,) * (residual.ndim - 1) + (num_streams, 1)
    output = (
        streams if retention_factors is None else streams * retention_factors.reshape(factor_shape)
    )
    output = output + update.unsqueeze(-2) * write_factors.reshape(factor_shape)
    return output.reshape_as(residual)


@pytest.mark.parametrize("stream_width", [5, 10, 20])
def test_native_streamwise_forward_matches_equation(stream_width):
    torch.manual_seed(123)
    num_streams = 3
    residual = torch.randn(2, 3, num_streams * stream_width, dtype=torch.float64)
    update = torch.randn(2, 3, stream_width, dtype=torch.float64)
    read_factors = torch.sigmoid(torch.randn(num_streams, dtype=torch.float64))
    write_factors = 2.0 * torch.sigmoid(torch.randn(num_streams, dtype=torch.float64))
    retention = torch.sigmoid(torch.randn(num_streams, dtype=torch.float64))

    native_read = streamwise_read(residual, read_factors)
    native_write = streamwise_writeback(
        residual, update, write_factors, retention_factors=retention
    )

    reference_read = _reference_read(residual, read_factors)
    reference_write = _reference_writeback(residual, update, write_factors, retention)

    assert torch.allclose(native_read, reference_read, atol=1.0e-12, rtol=1.0e-12)
    assert torch.allclose(native_write, reference_write, atol=1.0e-12, rtol=1.0e-12)


@pytest.mark.parametrize("stream_width", [4, 8])
def test_native_streamwise_gradients_match_equation(stream_width):
    torch.manual_seed(456)
    num_streams = 3

    native_inputs = [
        torch.randn(2, num_streams * stream_width, dtype=torch.float64, requires_grad=True),
        torch.randn(num_streams, dtype=torch.float64, requires_grad=True),
        torch.randn(num_streams, dtype=torch.float64, requires_grad=True),
        torch.randn(num_streams, dtype=torch.float64, requires_grad=True),
    ]
    reference_inputs = [value.detach().clone().requires_grad_() for value in native_inputs]

    def run_native(residual, read_logits, write_logits, retention_logits):
        read = streamwise_read(residual, torch.sigmoid(read_logits))
        return streamwise_writeback(
            residual,
            read.square(),
            2.0 * torch.sigmoid(write_logits),
            retention_factors=torch.sigmoid(retention_logits),
        )

    def run_reference(residual, read_logits, write_logits, retention_logits):
        read_factors = torch.sigmoid(read_logits)
        write_factors = 2.0 * torch.sigmoid(write_logits)
        retention = torch.sigmoid(retention_logits)
        read = _reference_read(residual, read_factors)
        return _reference_writeback(residual, read.square(), write_factors, retention)

    native_output = run_native(*native_inputs)
    reference_output = run_reference(*reference_inputs)
    native_grads = torch.autograd.grad(native_output.square().sum(), native_inputs)
    reference_grads = torch.autograd.grad(reference_output.square().sum(), reference_inputs)

    assert torch.allclose(native_output, reference_output, atol=1.0e-12, rtol=1.0e-12)
    for native_grad, reference_grad in zip(native_grads, reference_grads):
        assert torch.allclose(native_grad, reference_grad, atol=1.0e-10, rtol=1.0e-10)


def test_native_streamwise_autograd_gradcheck():
    torch.manual_seed(789)
    residual = torch.randn(2, 12, dtype=torch.float64, requires_grad=True)
    update = torch.randn(2, 4, dtype=torch.float64, requires_grad=True)
    write = torch.randn(3, dtype=torch.float64, requires_grad=True)
    retention = torch.randn(3, dtype=torch.float64, requires_grad=True)

    assert torch.autograd.gradcheck(
        lambda x, u, w, gamma: streamwise_writeback(x, u, w, retention_factors=gamma),
        (residual, update, write, retention),
    )


def test_native_streamwise_rejects_incompatible_shapes():
    with pytest.raises(ValueError, match="non-empty one-dimensional"):
        streamwise_read(torch.randn(2, 12), torch.empty(0))

    with pytest.raises(ValueError, match="not divisible"):
        streamwise_read(torch.randn(2, 10), torch.randn(3))

    with pytest.raises(ValueError, match="branch_update"):
        streamwise_writeback(torch.randn(2, 12), torch.randn(2, 5), torch.randn(3))

    with pytest.raises(ValueError, match="same device"):
        streamwise_writeback(torch.randn(2, 12), torch.empty(2, 4, device="meta"), torch.randn(3))

    with pytest.raises(ValueError, match="same dtype"):
        streamwise_writeback(
            torch.randn(2, 12, dtype=torch.float32),
            torch.randn(2, 4, dtype=torch.float64),
            torch.randn(3),
        )

    with pytest.raises(ValueError, match="same number of streams"):
        streamwise_writeback(
            torch.randn(2, 12), torch.randn(2, 4), torch.randn(3), retention_factors=torch.randn(2)
        )


def _padded_logits(values: torch.Tensor, padded_size: int = 128) -> torch.Tensor:
    logits = torch.zeros(padded_size, device=values.device, dtype=values.dtype)
    logits[: values.numel()] = values
    return logits.requires_grad_()


def test_raw_logit_cpu_fallback_matches_factor_reference_and_padding_gradients():
    torch.manual_seed(2026)
    num_streams = 3
    stream_width = 5
    max_forget = 0.2
    fused_inputs = [
        torch.randn(2, num_streams * stream_width, dtype=torch.float64, requires_grad=True),
        torch.randn(2, stream_width, dtype=torch.float64, requires_grad=True),
        _padded_logits(torch.tensor([-0.7, -0.4, -0.1], dtype=torch.float64)),
        _padded_logits(torch.tensor([-0.2, 0.0, 0.2], dtype=torch.float64)),
        _padded_logits(torch.tensor([4.8, 4.9, 5.0], dtype=torch.float64)),
    ]
    reference_inputs = [value.detach().clone().requires_grad_() for value in fused_inputs]

    def fused(residual, update, read_logits, write_logits, retention_logits):
        read = streamwise_sigmoid_read(residual, read_logits, num_streams)
        return streamwise_sigmoid_writeback(
            residual,
            update + read,
            write_logits,
            num_streams,
            retention_logits=retention_logits,
            retention_max_forget=max_forget,
        )

    def reference(residual, update, read_logits, write_logits, retention_logits):
        read = streamwise_read(residual, torch.sigmoid(read_logits[:num_streams].float()))
        return streamwise_writeback(
            residual,
            update + read,
            2.0 * torch.sigmoid(write_logits[:num_streams].float()),
            retention_factors=(
                1.0 - max_forget * torch.sigmoid(-retention_logits[:num_streams].float())
            ),
        )

    fused_output = fused(*fused_inputs)
    reference_output = reference(*reference_inputs)
    fused_gradients = torch.autograd.grad(fused_output.square().sum(), fused_inputs)
    reference_gradients = torch.autograd.grad(reference_output.square().sum(), reference_inputs)

    assert torch.allclose(fused_output, reference_output, atol=1.0e-12, rtol=1.0e-12)
    for fused_gradient, reference_gradient in zip(fused_gradients, reference_gradients):
        assert torch.allclose(fused_gradient, reference_gradient, atol=1.0e-10, rtol=1.0e-10)
    for gradient in fused_gradients[2:]:
        assert torch.count_nonzero(gradient[num_streams:]) == 0


def test_raw_logit_cpu_fallback_read_output_dtype_matches_explicit_cast():
    """The fallback preserves the same forward and backward cast composition as Triton."""

    torch.manual_seed(1357)
    num_streams = 3
    residual = torch.randn(4, 15, dtype=torch.float32, requires_grad=True)
    read_logits = _padded_logits(torch.tensor([-0.8, -0.7, -0.6], dtype=torch.float32))
    reference_residual = residual.detach().clone().requires_grad_()
    reference_logits = read_logits.detach().clone().requires_grad_()
    grad_output = torch.randn(4, 5, dtype=torch.bfloat16)

    output = streamwise_sigmoid_read(
        residual, read_logits, num_streams, output_dtype=torch.bfloat16
    )
    reference = streamwise_sigmoid_read(reference_residual, reference_logits, num_streams).to(
        torch.bfloat16
    )
    gradients = torch.autograd.grad(output, (residual, read_logits), grad_output)
    reference_gradients = torch.autograd.grad(
        reference, (reference_residual, reference_logits), grad_output
    )

    assert output.dtype == torch.bfloat16
    assert torch.equal(output, reference)
    for gradient, reference_gradient in zip(gradients, reference_gradients):
        assert gradient.dtype == reference_gradient.dtype
        assert torch.equal(gradient, reference_gradient)


def test_raw_logit_cpu_fallback_mixed_write_matches_explicit_cast():
    """The fallback accepts the fused API but materializes the reference cast."""

    torch.manual_seed(2468)
    num_streams = 3
    residual = torch.randn(4, 15, dtype=torch.float32, requires_grad=True)
    update = torch.randn(4, 5, dtype=torch.bfloat16, requires_grad=True)
    write_logits = _padded_logits(torch.tensor([-0.01, 0.0, 0.01], dtype=torch.float32))
    reference_residual = residual.detach().clone().requires_grad_()
    reference_update = update.detach().clone().requires_grad_()
    reference_logits = write_logits.detach().clone().requires_grad_()
    grad_output = torch.randn_like(residual)

    output = streamwise_sigmoid_writeback(residual, update, write_logits, num_streams)
    reference = streamwise_sigmoid_writeback(
        reference_residual, reference_update.float(), reference_logits, num_streams
    )
    gradients = torch.autograd.grad(output, (residual, update, write_logits), grad_output)
    reference_gradients = torch.autograd.grad(
        reference, (reference_residual, reference_update, reference_logits), grad_output
    )

    assert output.dtype == torch.float32
    assert torch.equal(output, reference)
    for gradient, reference_gradient in zip(gradients, reference_gradients):
        assert gradient.dtype == reference_gradient.dtype
        assert torch.equal(gradient, reference_gradient)


def test_raw_logit_api_validates_padded_controllers():
    residual = torch.randn(2, 12)
    update = torch.randn(2, 4)

    with pytest.raises(ValueError, match="num_streams must be positive"):
        streamwise_sigmoid_read(residual, torch.randn(128), 0)
    with pytest.raises(ValueError, match="at least 3"):
        streamwise_sigmoid_read(residual, torch.randn(2), 3)
    with pytest.raises(TypeError, match="floating-point dtype"):
        streamwise_sigmoid_read(residual, torch.ones(128, dtype=torch.int64), 3)
    with pytest.raises(ValueError, match="non-empty hidden dimension"):
        streamwise_sigmoid_read(torch.empty(2, 0), torch.randn(128), 3)
    with pytest.raises(ValueError, match="not divisible"):
        streamwise_sigmoid_read(torch.randn(2, 10), torch.randn(128), 3)
    with pytest.raises(ValueError, match="same device"):
        streamwise_sigmoid_read(residual, torch.empty(128, device="meta"), 3)
    with pytest.raises(ValueError, match="branch_update"):
        streamwise_sigmoid_writeback(residual, torch.randn(2, 5), torch.randn(128), 3)
    with pytest.raises(ValueError, match="same device"):
        streamwise_sigmoid_writeback(
            residual, torch.empty(2, 4, device="meta"), torch.randn(128), 3
        )
    with pytest.raises(ValueError, match="must either match residual_stream dtype"):
        streamwise_sigmoid_writeback(residual, update.double(), torch.randn(128), 3)
    with pytest.raises(ValueError, match="retention_logits and residual_stream"):
        streamwise_sigmoid_writeback(
            residual,
            update,
            torch.randn(128),
            3,
            retention_logits=torch.empty(128, device="meta"),
            retention_max_forget=0.1,
        )
    with pytest.raises(ValueError, match="retention_max_forget"):
        streamwise_sigmoid_writeback(
            residual,
            update,
            torch.randn(128),
            3,
            retention_logits=torch.randn(128),
            retention_max_forget=0.0,
        )
    with pytest.raises(TypeError, match="output_dtype"):
        streamwise_sigmoid_read(residual, torch.randn(128), 3, output_dtype=torch.int32)
    invalid_mixed_pairs = [
        (residual.to(torch.bfloat16), update),
        (residual.to(torch.float16), update.to(torch.bfloat16)),
    ]
    for invalid_residual, invalid_update in invalid_mixed_pairs:
        with pytest.raises(ValueError, match="must either match"):
            streamwise_sigmoid_writeback(invalid_residual, invalid_update, torch.randn(128), 3)


def _relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    denominator = expected.float().norm().clamp_min(1.0e-12)
    return ((actual.float() - expected.float()).norm() / denominator).item()


def _run_fused_cuda_case(
    residual: torch.Tensor,
    update: torch.Tensor,
    read_logits: torch.Tensor,
    write_logits: torch.Tensor,
    retention_logits: torch.Tensor,
    grad_output: torch.Tensor,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    num_streams = 3
    read = streamwise_sigmoid_read(residual, read_logits, num_streams)
    output = streamwise_sigmoid_writeback(
        residual,
        update + 0.125 * read,
        write_logits,
        num_streams,
        retention_logits=retention_logits,
        retention_max_forget=0.2,
    )
    gradients = torch.autograd.grad(
        output, (residual, update, read_logits, write_logits, retention_logits), grad_output
    )
    return output, gradients


def _run_reference_cuda_case(
    residual: torch.Tensor,
    update: torch.Tensor,
    read_logits: torch.Tensor,
    write_logits: torch.Tensor,
    retention_logits: torch.Tensor,
    grad_output: torch.Tensor,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    num_streams = 3
    read = streamwise_read(residual, torch.sigmoid(read_logits[:num_streams].float()))
    output = streamwise_writeback(
        residual,
        update + 0.125 * read,
        2.0 * torch.sigmoid(write_logits[:num_streams].float()),
        retention_factors=(1.0 - 0.2 * torch.sigmoid(-retention_logits[:num_streams].float())),
    )
    gradients = torch.autograd.grad(
        output, (residual, update, read_logits, write_logits, retention_logits), grad_output
    )
    return output, gradients


@pytest.mark.skipif(
    not torch.cuda.is_available() or not HAVE_STREAMWISE_TRITON,
    reason="Direct streamwise Triton kernels require CUDA and Triton.",
)
def test_streamwise_triton_fp32_selector_respects_geometry_and_layout():
    """FP32 should use Triton only when all existing fast-path constraints are satisfied."""

    num_streams = 3
    stream_width = 64
    logits = torch.zeros(128, device="cuda", dtype=torch.float32)
    eligible = torch.empty(256, num_streams * stream_width, device="cuda", dtype=torch.float32)

    assert _can_use_streamwise_triton(eligible, logits, num_streams, stream_width)
    assert not _can_use_streamwise_triton(eligible[:255], logits, num_streams, stream_width)

    narrow = torch.empty(256, num_streams * 32, device="cuda", dtype=torch.float32)
    assert not _can_use_streamwise_triton(narrow, logits, num_streams, 32)

    noncontiguous = torch.empty(
        256, num_streams * stream_width * 2, device="cuda", dtype=torch.float32
    )[:, ::2]
    assert noncontiguous.shape == eligible.shape
    assert not noncontiguous.is_contiguous()
    assert not _can_use_streamwise_triton(noncontiguous, logits, num_streams, stream_width)

    fp64 = eligible.to(dtype=torch.float64)
    assert not _can_use_streamwise_triton(fp64, logits, num_streams, stream_width)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not HAVE_STREAMWISE_TRITON,
    reason="Direct streamwise Triton kernels require CUDA and Triton.",
)
@pytest.mark.parametrize(
    ("input_dtype", "output_dtype"),
    [
        (torch.float32, torch.bfloat16),
        (torch.float32, torch.float16),
        (torch.bfloat16, torch.float32),
        (torch.bfloat16, torch.float16),
        (torch.float16, torch.bfloat16),
    ],
    ids=("fp32_to_bf16", "fp32_to_fp16", "bf16_to_fp32", "bf16_to_fp16", "fp16_to_bf16"),
)
def test_fused_cuda_read_output_dtype_matches_explicit_cast(input_dtype, output_dtype):
    """Fusing the terminal read cast must preserve its forward and backward semantics."""

    torch.manual_seed(9753)
    num_streams = 3
    stream_width = 64
    residual = torch.randn(
        256, num_streams * stream_width, device="cuda", dtype=input_dtype, requires_grad=True
    )
    read_logits = _padded_logits(torch.tensor([-0.8, -0.7, -0.6], device="cuda"))
    reference_residual = residual.detach().clone().requires_grad_()
    reference_logits = read_logits.detach().clone().requires_grad_()
    grad_output = torch.randn(256, stream_width, device="cuda", dtype=output_dtype)

    output = streamwise_sigmoid_read(residual, read_logits, num_streams, output_dtype=output_dtype)
    reference = streamwise_sigmoid_read(reference_residual, reference_logits, num_streams).to(
        output_dtype
    )
    gradients = torch.autograd.grad(output, (residual, read_logits), grad_output)
    reference_gradients = torch.autograd.grad(
        reference, (reference_residual, reference_logits), grad_output
    )

    assert output.dtype == output_dtype
    assert type(output.grad_fn).__name__ == "_StreamwiseSigmoidReadBackward"
    assert torch.equal(output, reference)
    assert torch.equal(gradients[0], reference_gradients[0])
    assert gradients[1].dtype == reference_gradients[1].dtype
    # The fused and explicit-cast graphs specialize the controller reduction on different
    # grad-output pointer dtypes, so controller gradients are numerically rather than bitwise
    # equivalent. Activation gradients remain bitwise equal.
    assert _relative_l2(gradients[1], reference_gradients[1]) <= 1.0e-6
    assert torch.count_nonzero(gradients[1][num_streams:]) == 0


@pytest.mark.skipif(
    not torch.cuda.is_available() or not HAVE_STREAMWISE_TRITON,
    reason="Direct streamwise Triton kernels require CUDA and Triton.",
)
@pytest.mark.parametrize("update_dtype", [torch.bfloat16, torch.float16], ids=("bf16", "fp16"))
@pytest.mark.parametrize("use_retention", [False, True], ids=("identity", "retention"))
def test_fused_cuda_mixed_write_matches_explicit_cast(update_dtype, use_retention):
    """The mixed write fuses only the update's terminal conversion to residual dtype."""

    torch.manual_seed(8642)
    num_streams = 3
    stream_width = 64
    residual = torch.randn(
        256, num_streams * stream_width, device="cuda", dtype=torch.float32, requires_grad=True
    )
    update = torch.randn(256, stream_width, device="cuda", dtype=update_dtype, requires_grad=True)
    write_logits = _padded_logits(torch.tensor([-0.01, 0.0, 0.01], device="cuda"))
    retention_logits = _padded_logits(torch.tensor([4.8, 4.9, 5.0], device="cuda"))
    reference_residual = residual.detach().clone().requires_grad_()
    reference_update = update.detach().clone().requires_grad_()
    reference_write_logits = write_logits.detach().clone().requires_grad_()
    reference_retention_logits = retention_logits.detach().clone().requires_grad_()
    grad_output = torch.randn_like(residual)

    output = streamwise_sigmoid_writeback(
        residual,
        update,
        write_logits,
        num_streams,
        retention_logits=retention_logits if use_retention else None,
        retention_max_forget=0.2 if use_retention else 0.0,
    )
    reference = streamwise_sigmoid_writeback(
        reference_residual,
        reference_update.float(),
        reference_write_logits,
        num_streams,
        retention_logits=reference_retention_logits if use_retention else None,
        retention_max_forget=0.2 if use_retention else 0.0,
    )
    inputs = (residual, update, write_logits)
    reference_inputs = (reference_residual, reference_update, reference_write_logits)
    if use_retention:
        inputs = (*inputs, retention_logits)
        reference_inputs = (*reference_inputs, reference_retention_logits)
    gradients = torch.autograd.grad(output, inputs, grad_output)
    reference_gradients = torch.autograd.grad(reference, reference_inputs, grad_output)

    assert output.dtype == torch.float32
    assert type(output.grad_fn).__name__ == "_StreamwiseSigmoidWritebackBackward"
    torch.testing.assert_close(output, reference, rtol=0.0, atol=0.0)
    for index, (gradient, reference_gradient) in enumerate(zip(gradients, reference_gradients)):
        assert gradient.dtype == reference_gradient.dtype
        if index < 2:
            torch.testing.assert_close(gradient, reference_gradient, rtol=0.0, atol=0.0)
        else:
            assert _relative_l2(gradient, reference_gradient) <= 1.0e-6
    assert gradients[0].dtype == torch.float32
    assert gradients[1].dtype == update_dtype
    for gradient in gradients[2:]:
        assert torch.count_nonzero(gradient[num_streams:]) == 0
    if not use_retention:
        assert torch.equal(gradients[0], grad_output)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not HAVE_STREAMWISE_TRITON,
    reason="Direct streamwise Triton kernels require CUDA and Triton.",
)
def test_fused_cuda_fp16_same_dtype_forward_backward_matches_reference():
    """The generalized pointer-dtype conversions preserve the existing FP16 path."""

    torch.manual_seed(5317)
    num_streams = 3
    stream_width = 64
    source_inputs = [
        torch.randn(
            256, num_streams * stream_width, device="cuda", dtype=torch.float16, requires_grad=True
        ),
        torch.randn(256, stream_width, device="cuda", dtype=torch.float16, requires_grad=True),
        _padded_logits(torch.tensor([-0.8, -0.7, -0.6], device="cuda", dtype=torch.float32)),
        _padded_logits(torch.tensor([-0.01, 0.0, 0.01], device="cuda", dtype=torch.float32)),
        _padded_logits(torch.tensor([4.8, 4.9, 5.0], device="cuda", dtype=torch.float32)),
    ]
    reference_inputs = [value.detach().clone().requires_grad_() for value in source_inputs]
    grad_output = torch.randn_like(source_inputs[0])

    assert _can_use_streamwise_triton(source_inputs[0], source_inputs[2], num_streams, stream_width)
    output, gradients = _run_fused_cuda_case(*source_inputs, grad_output)
    reference, reference_gradients = _run_reference_cuda_case(*reference_inputs, grad_output)

    assert output.dtype == torch.float16
    assert type(output.grad_fn).__name__ == "_StreamwiseSigmoidWritebackBackward"
    assert _relative_l2(output, reference) <= 0.02
    for gradient, reference_gradient in zip(gradients, reference_gradients):
        assert gradient.dtype == reference_gradient.dtype
        assert _relative_l2(gradient, reference_gradient) <= 0.02
    for gradient in gradients[2:]:
        assert torch.count_nonzero(gradient[num_streams:]) == 0


@pytest.mark.skipif(
    not torch.cuda.is_available() or not HAVE_STREAMWISE_TRITON,
    reason="Direct streamwise Triton kernels require CUDA and Triton.",
)
@pytest.mark.parametrize(
    ("batch_tokens", "stream_width"),
    [(24_576, 768), (8_192, 5_120)],
    ids=("1b_geometry", "nt4_geometry"),
)
def test_fused_cuda_production_geometry_gradients_signs_and_determinism(batch_tokens, stream_width):
    torch.manual_seed(1234)
    num_streams = 3
    device = torch.device("cuda")
    residual = torch.randn(
        batch_tokens, num_streams * stream_width, device=device, dtype=torch.bfloat16
    )
    update = torch.randn(batch_tokens, stream_width, device=device, dtype=torch.bfloat16)
    read_logits = _padded_logits(
        torch.tensor([-0.8, -0.7, -0.6], device=device, dtype=torch.float32)
    )
    write_logits = _padded_logits(
        torch.tensor([-0.01, 0.0, 0.01], device=device, dtype=torch.float32)
    )
    retention_logits = _padded_logits(
        torch.tensor([4.8, 4.9, 5.0], device=device, dtype=torch.float32)
    )
    grad_output = torch.randn_like(residual)

    assert _can_use_streamwise_triton(residual, read_logits, num_streams, stream_width)
    fused_leaves = [
        value.detach().clone().requires_grad_()
        for value in (residual, update, read_logits, write_logits, retention_logits)
    ]
    reference_leaves = [value.detach().clone().requires_grad_() for value in fused_leaves]
    fused_output, fused_gradients = _run_fused_cuda_case(*fused_leaves, grad_output)
    reference_output, reference_gradients = _run_reference_cuda_case(*reference_leaves, grad_output)

    assert _relative_l2(fused_output, reference_output) <= 0.02
    for fused_gradient, reference_gradient in zip(fused_gradients, reference_gradients):
        assert _relative_l2(fused_gradient, reference_gradient) <= 0.02
    for fused_gradient, reference_gradient in zip(fused_gradients[2:], reference_gradients[2:]):
        assert torch.equal(
            torch.sign(fused_gradient[:num_streams]), torch.sign(reference_gradient[:num_streams])
        )
        assert torch.count_nonzero(fused_gradient[num_streams:]) == 0

    repeated_leaves = [value.detach().clone().requires_grad_() for value in fused_leaves]
    repeated_output, repeated_gradients = _run_fused_cuda_case(*repeated_leaves, grad_output)
    assert torch.equal(fused_output, repeated_output)
    for first, repeated in zip(fused_gradients[2:], repeated_gradients[2:]):
        assert torch.equal(first, repeated)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not HAVE_STREAMWISE_TRITON,
    reason="Direct streamwise Triton kernels require CUDA and Triton.",
)
@pytest.mark.parametrize(
    ("context_factory", "use_retention"),
    [
        (torch.no_grad, False),
        (torch.no_grad, True),
        (torch.inference_mode, False),
        (torch.inference_mode, True),
    ],
    ids=("no_grad", "no_grad_retention", "inference", "inference_retention"),
)
def test_fused_cuda_forward_only_context_matches_reference(context_factory, use_retention):
    """Forward-only execution must not construct custom-autograd backward state."""

    torch.manual_seed(4321)
    num_streams = 3
    stream_width = 64
    max_forget = 0.2
    residual_source = torch.randn(
        256, num_streams * stream_width, device="cuda", dtype=torch.bfloat16
    )
    update_source = torch.randn(256, stream_width, device="cuda", dtype=torch.bfloat16)
    read_logits = _padded_logits(torch.tensor([-0.8, -0.7, -0.6], device="cuda"))
    write_logits = _padded_logits(torch.tensor([-0.01, 0.0, 0.01], device="cuda"))
    retention_logits = _padded_logits(torch.tensor([4.8, 4.9, 5.0], device="cuda"))

    with context_factory():
        # Cloning inside inference_mode reproduces the inference tensors from the
        # dynamic generation runtime that custom autograd Functions cannot save.
        residual = residual_source.clone()
        update = update_source.clone()
        assert _can_use_streamwise_triton(residual, read_logits, num_streams, stream_width)

        read = streamwise_sigmoid_read(residual, read_logits, num_streams)
        output = streamwise_sigmoid_writeback(
            residual,
            update + 0.125 * read,
            write_logits,
            num_streams,
            retention_logits=retention_logits if use_retention else None,
            retention_max_forget=max_forget if use_retention else 0.0,
        )

        reference_read = streamwise_read(residual, torch.sigmoid(read_logits[:num_streams].float()))
        retention_factors = None
        if use_retention:
            retention_factors = 1.0 - max_forget * torch.sigmoid(
                -retention_logits[:num_streams].float()
            )
        reference_output = streamwise_writeback(
            residual,
            update + 0.125 * reference_read,
            2.0 * torch.sigmoid(write_logits[:num_streams].float()),
            retention_factors=retention_factors,
        )
        read_error = _relative_l2(read, reference_read)
        output_error = _relative_l2(output, reference_output)

    assert not output.requires_grad
    assert read_error <= 0.02
    assert output_error <= 0.02


@pytest.mark.skipif(
    not torch.cuda.is_available() or not HAVE_STREAMWISE_TRITON,
    reason="Direct streamwise Triton kernels require CUDA and Triton.",
)
def test_fused_cuda_profile_has_no_map_construction_or_map_gradient_gemm():
    torch.manual_seed(5678)
    num_streams = 3
    stream_width = 256
    residual = torch.randn(
        2048, num_streams * stream_width, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    update = torch.randn(
        2048, stream_width, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    read_logits = _padded_logits(torch.tensor([-0.8, -0.7, -0.6], device="cuda"))
    write_logits = _padded_logits(torch.tensor([-0.01, 0.0, 0.01], device="cuda"))
    retention_logits = _padded_logits(torch.tensor([4.8, 4.9, 5.0], device="cuda"))
    grad_output = torch.randn_like(residual)

    _run_fused_cuda_case(residual, update, read_logits, write_logits, retention_logits, grad_output)
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    ) as profile:
        _run_fused_cuda_case(
            residual, update, read_logits, write_logits, retention_logits, grad_output
        )
    torch.cuda.synchronize()

    event_names = {event.key for event in profile.key_averages()}
    forbidden = {
        "aten::bmm",
        "aten::matmul",
        "aten::mm",
        "aten::repeat_interleave",
        "aten::sigmoid",
    }
    assert event_names.isdisjoint(forbidden), sorted(event_names & forbidden)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not HAVE_STREAMWISE_TRITON,
    reason="Direct streamwise Triton kernels require CUDA and Triton.",
)
def test_fused_cuda_primitive_mixed_read_write_profile_has_no_cast_kernel():
    """Eligible primitive read/write boundaries must not materialize an aten cast."""

    torch.manual_seed(7531)
    num_streams = 3
    stream_width = 256
    residual_source = torch.randn(
        2048, num_streams * stream_width, device="cuda", dtype=torch.float32
    )
    update_source = torch.randn(2048, stream_width, device="cuda", dtype=torch.bfloat16)
    read_logits_source = _padded_logits(torch.tensor([-0.8, -0.7, -0.6], device="cuda"))
    write_logits_source = _padded_logits(torch.tensor([-0.01, 0.0, 0.01], device="cuda"))
    grad_output = torch.randn_like(residual_source)

    def make_inputs():
        return tuple(
            source.detach().clone().requires_grad_()
            for source in (residual_source, update_source, read_logits_source, write_logits_source)
        )

    def run_case(residual, update, read_logits, write_logits):
        read = streamwise_sigmoid_read(
            residual, read_logits, num_streams, output_dtype=torch.bfloat16
        )
        output = streamwise_sigmoid_writeback(residual, update + read, write_logits, num_streams)
        assert type(read.grad_fn).__name__ == "_StreamwiseSigmoidReadBackward"
        assert type(output.grad_fn).__name__ == "_StreamwiseSigmoidWritebackBackward"
        return torch.autograd.grad(
            output, (residual, update, read_logits, write_logits), grad_output
        )

    run_case(*make_inputs())
    torch.cuda.synchronize()
    profiled_inputs = make_inputs()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    ) as profile:
        run_case(*profiled_inputs)
    torch.cuda.synchronize()

    event_names = {event.key for event in profile.key_averages()}
    assert "aten::_to_copy" not in event_names, sorted(event_names)
