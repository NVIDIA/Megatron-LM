# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Numerical and dispatch tests for the fused MTP TV primitive."""

import os

import pytest
import torch
import torch.nn.functional as F

from megatron.core import parallel_state
from megatron.core.fusions import fused_mtp_tv as fused_mtp_tv_module
from megatron.core.fusions.fused_mtp_tv import (
    fused_mtp_tv_unavailable_reason,
    vocab_parallel_tv_distance,
)
from tests.unit_tests.test_utilities import Utils


@pytest.fixture(scope="module", autouse=True)
def _select_torchrun_local_cuda_device():
    """Keep Triton compilation and CUDA Graph capture on one device per process."""
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank % torch.cuda.device_count())
    yield


def _native_tv_distance(draft_logits, target_logits):
    draft_prob = F.softmax(draft_logits.float(), dim=-1)
    target_prob = F.softmax(target_logits.detach().float(), dim=-1)
    return (1.0 - torch.minimum(draft_prob, target_prob).sum(dim=-1)).clamp(0.0, 1.0)


def _run_tv_and_gradient(function, draft_data, target_logits, grad_output):
    draft_logits = draft_data.detach().clone().requires_grad_(True)
    output = function(draft_logits, target_logits)
    (output * grad_output).sum().backward()
    return output.detach(), draft_logits.grad.detach()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("vocab_size", [257, 154880])
def test_fused_mtp_tv_matches_native_forward_and_backward(dtype, vocab_size):
    """Cover an odd block tail and GLM-5.2's production vocabulary."""
    torch.manual_seed(1234)
    shape = (3, 2, vocab_size)
    draft_data = torch.randn(shape, device="cuda", dtype=dtype)
    target_logits = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    grad_output = torch.randn(shape[:-1], device="cuda", dtype=torch.float32)

    assert fused_mtp_tv_unavailable_reason(draft_data, target_logits) is None
    actual, actual_grad = _run_tv_and_gradient(
        lambda draft, target: vocab_parallel_tv_distance(
            draft, target, logits_are_vocab_sharded=False
        ),
        draft_data,
        target_logits,
        grad_output,
    )
    assert actual.dtype == torch.float32
    reference, reference_grad = _run_tv_and_gradient(
        _native_tv_distance, draft_data, target_logits, grad_output
    )

    rtol = 3e-3 if dtype == torch.bfloat16 else 1e-5
    atol = 3e-3 if dtype == torch.bfloat16 else 1e-6
    torch.testing.assert_close(actual, reference, rtol=rtol, atol=atol)
    torch.testing.assert_close(actual_grad, reference_grad, rtol=rtol, atol=atol)
    assert target_logits.grad is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fused_mtp_tv_clamp_boundaries_and_zero_acceptance():
    """The fused path preserves both clamp endpoints and finite gradients."""
    vocab_size = 257
    identical = torch.randn(2, vocab_size, device="cuda", dtype=torch.float32)
    identical_output, identical_grad = _run_tv_and_gradient(
        lambda draft, target: vocab_parallel_tv_distance(
            draft, target, logits_are_vocab_sharded=False
        ),
        identical,
        identical,
        torch.ones(2, device="cuda"),
    )
    torch.testing.assert_close(
        identical_output, torch.zeros_like(identical_output), atol=1e-6, rtol=0
    )
    torch.testing.assert_close(identical_grad, torch.zeros_like(identical_grad), atol=1e-6, rtol=0)

    draft = torch.full((1, vocab_size), -100.0, device="cuda")
    target = torch.full_like(draft, -100.0)
    draft[:, 0] = 100.0
    target[:, 1] = 100.0
    output, gradient = _run_tv_and_gradient(
        lambda draft_logits, target_logits: vocab_parallel_tv_distance(
            draft_logits, target_logits, logits_are_vocab_sharded=False
        ),
        draft,
        target,
        torch.ones(1, device="cuda"),
    )
    torch.testing.assert_close(output, torch.ones_like(output), atol=1e-6, rtol=0)
    assert torch.isfinite(gradient).all()
    torch.testing.assert_close(gradient, torch.zeros_like(gradient), atol=1e-6, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_noncontiguous_logits_use_reference_fallback(monkeypatch):
    """Unsupported strides retain the PyTorch reference behavior."""
    torch.manual_seed(17)
    draft_data = torch.randn(2, 257, 3, device="cuda").transpose(1, 2)
    target_logits = torch.randn(2, 257, 3, device="cuda").transpose(1, 2)
    grad_output = torch.randn(draft_data.shape[:-1], device="cuda")
    assert not draft_data.is_contiguous()
    assert fused_mtp_tv_unavailable_reason(draft_data, target_logits) == "logits are not contiguous"

    def fail_if_fused(*_args, **_kwargs):
        pytest.fail("The fused kernel must not receive unsupported noncontiguous logits")

    monkeypatch.setattr(fused_mtp_tv_module, "_fused_vocab_parallel_tv_distance", fail_if_fused)
    actual, actual_grad = _run_tv_and_gradient(
        lambda draft, target: vocab_parallel_tv_distance(
            draft, target, logits_are_vocab_sharded=False
        ),
        draft_data,
        target_logits,
        grad_output,
    )
    reference, reference_grad = _run_tv_and_gradient(
        _native_tv_distance, draft_data, target_logits, grad_output
    )
    torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(actual_grad, reference_grad, rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_noncontiguous_target_is_packed_before_fused_dispatch(monkeypatch):
    """A materialized-roll view is packed centrally when the draft supports Triton."""
    torch.manual_seed(19)
    draft_data = torch.randn(2, 3, 257, device="cuda")
    target_logits = torch.randn(2, 257, 3, device="cuda").transpose(1, 2)
    grad_output = torch.randn(2, 3, device="cuda")
    assert draft_data.is_contiguous()
    assert not target_logits.is_contiguous()
    assert fused_mtp_tv_unavailable_reason(draft_data, draft_data) is None

    original_fused = fused_mtp_tv_module._fused_vocab_parallel_tv_distance
    fused_calls = 0

    def record_fused(draft, target, *args, **kwargs):
        nonlocal fused_calls
        fused_calls += 1
        assert target.is_contiguous()
        return original_fused(draft, target, *args, **kwargs)

    monkeypatch.setattr(fused_mtp_tv_module, "_fused_vocab_parallel_tv_distance", record_fused)
    actual, actual_grad = _run_tv_and_gradient(
        lambda draft, target: vocab_parallel_tv_distance(
            draft, target, logits_are_vocab_sharded=False
        ),
        draft_data,
        target_logits,
        grad_output,
    )
    reference, reference_grad = _run_tv_and_gradient(
        _native_tv_distance, draft_data, target_logits, grad_output
    )

    assert fused_calls == 1
    torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(actual_grad, reference_grad, rtol=1e-5, atol=1e-6)


def test_target_rows_use_reference_fallback_with_local_halo_and_invalid_fill(monkeypatch):
    """CPU addressing gathers one target and safely maps invalid rows to zero logits."""
    torch.manual_seed(23)
    draft_data = torch.randn(4, 1, 7, dtype=torch.float32)
    target_logits = torch.randn_like(draft_data, requires_grad=True)
    target_halo = torch.randn(2, 1, 7, dtype=torch.float32, requires_grad=True)
    target_row_indices = torch.tensor([[1], [4], [5], [99]], dtype=torch.int32)
    target_valid_rows = torch.tensor([[True], [True], [False], [True]])
    grad_output = torch.randn(4, 1, dtype=torch.float32)

    def fail_if_fused(*_args, **_kwargs):
        pytest.fail("CPU target-row addressing must use the reference fallback")

    monkeypatch.setattr(fused_mtp_tv_module, "_fused_vocab_parallel_tv_distance", fail_if_fused)

    def mapped_tv(draft, target):
        return vocab_parallel_tv_distance(
            draft,
            target,
            logits_are_vocab_sharded=False,
            target_row_indices=target_row_indices,
            target_valid_rows=target_valid_rows,
            target_halo_logits=target_halo,
        )

    materialized_target = torch.zeros_like(target_logits)
    materialized_target[0].copy_(target_logits[1])
    materialized_target[1].copy_(target_halo[0])
    actual, actual_grad = _run_tv_and_gradient(mapped_tv, draft_data, target_logits, grad_output)
    reference, reference_grad = _run_tv_and_gradient(
        _native_tv_distance, draft_data, materialized_target, grad_output
    )

    torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(actual_grad, reference_grad, rtol=1e-5, atol=1e-6)
    assert target_logits.grad is None
    assert target_halo.grad is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_target_rows_flatten_sequence_and_batch_before_addressing():
    """Address rows in canonical sequence-major order when the batch size exceeds one."""
    torch.manual_seed(29)
    sequence_length = 3
    batch_size = 2
    vocab_size = 257
    draft_data = torch.randn(sequence_length, batch_size, vocab_size, device="cuda")
    target_logits = torch.randn_like(draft_data, requires_grad=True)
    target_halo = torch.randn(2, batch_size, vocab_size, device="cuda", requires_grad=True)
    # Local rows flatten as ``sequence * batch_size + batch``; halo rows
    # continue from ``sequence_length * batch_size`` using the same ordering.
    target_row_indices = torch.tensor([[2, 3], [6, 7], [8, 9]], device="cuda", dtype=torch.int32)
    target_valid_rows = torch.ones(sequence_length, batch_size, device="cuda", dtype=torch.bool)
    grad_output = torch.randn(sequence_length, batch_size, device="cuda")

    def mapped_tv(draft, target):
        return vocab_parallel_tv_distance(
            draft,
            target,
            logits_are_vocab_sharded=False,
            target_row_indices=target_row_indices,
            target_valid_rows=target_valid_rows,
            target_halo_logits=target_halo,
        )

    source_rows = target_logits.detach().reshape(-1, vocab_size)
    halo_rows = target_halo.detach().reshape(-1, vocab_size)
    materialized_target = torch.cat((source_rows, halo_rows), dim=0).index_select(
        0, target_row_indices.reshape(-1).long()
    )
    materialized_target = materialized_target.view_as(draft_data)
    actual, actual_grad = _run_tv_and_gradient(mapped_tv, draft_data, target_logits, grad_output)
    reference, reference_grad = _run_tv_and_gradient(
        _native_tv_distance, draft_data, materialized_target, grad_output
    )

    torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(actual_grad, reference_grad, rtol=1e-5, atol=1e-6)
    assert target_logits.grad is None
    assert target_halo.grad is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_target_rows_noncontiguous_source_uses_reference_fallback(monkeypatch):
    """Addressing remains correct when a noncontiguous source disables fused dispatch."""
    torch.manual_seed(31)
    sequence_length = 3
    batch_size = 2
    vocab_size = 257
    draft_data = torch.randn(sequence_length, batch_size, vocab_size, device="cuda")
    target_logits = (
        torch.randn(sequence_length, vocab_size, batch_size, device="cuda")
        .transpose(1, 2)
        .detach()
        .requires_grad_(True)
    )
    target_halo = torch.randn(1, batch_size, vocab_size, device="cuda", requires_grad=True)
    target_row_indices = torch.tensor([[2, 3], [4, 5], [6, 7]], device="cuda", dtype=torch.int32)
    target_valid_rows = torch.ones(sequence_length, batch_size, device="cuda", dtype=torch.bool)
    grad_output = torch.randn(sequence_length, batch_size, device="cuda")

    assert not target_logits.is_contiguous()
    assert fused_mtp_tv_unavailable_reason(draft_data, target_logits) == "logits are not contiguous"

    def fail_if_fused(*_args, **_kwargs):
        pytest.fail("A noncontiguous addressed source must use the reference fallback")

    monkeypatch.setattr(fused_mtp_tv_module, "_fused_vocab_parallel_tv_distance", fail_if_fused)

    def mapped_tv(draft, target):
        return vocab_parallel_tv_distance(
            draft,
            target,
            logits_are_vocab_sharded=False,
            target_row_indices=target_row_indices,
            target_valid_rows=target_valid_rows,
            target_halo_logits=target_halo,
        )

    source_rows = target_logits.detach().reshape(-1, vocab_size)
    halo_rows = target_halo.detach().reshape(-1, vocab_size)
    materialized_target = torch.cat((source_rows, halo_rows), dim=0).index_select(
        0, target_row_indices.reshape(-1).long()
    )
    materialized_target = materialized_target.view_as(draft_data)
    actual, actual_grad = _run_tv_and_gradient(mapped_tv, draft_data, target_logits, grad_output)
    reference, reference_grad = _run_tv_and_gradient(
        _native_tv_distance, draft_data, materialized_target, grad_output
    )

    torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(actual_grad, reference_grad, rtol=1e-5, atol=1e-6)
    assert target_logits.grad is None
    assert target_halo.grad is None


def test_target_rows_dispatch_source_halo_and_metadata_without_materializing(monkeypatch):
    """Fused dispatch receives the immutable source and compact halo directly."""
    draft = torch.randn(3, 1, 7)
    target = torch.randn_like(draft)
    halo = torch.randn(1, 1, 7)
    indices = torch.tensor([[1], [2], [3]], dtype=torch.int32)
    valid = torch.ones(3, 1, dtype=torch.bool)

    monkeypatch.setattr(fused_mtp_tv_module, "fused_mtp_tv_unavailable_reason", lambda *_args: None)

    def record_fused(draft_arg, target_arg, tp_group, vocab_sharded, **kwargs):
        assert draft_arg is draft
        assert target_arg is target
        assert tp_group is None
        assert vocab_sharded is False
        assert kwargs["target_row_indices"] is indices
        assert kwargs["target_valid_rows"] is valid
        assert kwargs["target_halo_logits"] is halo
        return torch.zeros(draft.shape[:-1], dtype=torch.float32)

    monkeypatch.setattr(fused_mtp_tv_module, "_fused_vocab_parallel_tv_distance", record_fused)
    output = vocab_parallel_tv_distance(
        draft,
        target,
        logits_are_vocab_sharded=False,
        target_row_indices=indices,
        target_valid_rows=valid,
        target_halo_logits=halo,
    )
    assert output.shape == draft.shape[:-1]


@pytest.mark.parametrize(
    ("indices", "valid", "halo", "message"),
    [
        (None, torch.ones(2, 1, dtype=torch.bool), None, "provided together"),
        (None, None, torch.randn(1, 1, 7), "require target row indices"),
        (torch.ones(2, dtype=torch.int32), torch.ones(2, dtype=torch.bool), None, "leading"),
        (
            torch.ones(2, 1, dtype=torch.int32),
            torch.ones(2, dtype=torch.bool),
            None,
            "validity must match",
        ),
        (
            torch.ones(2, 1, dtype=torch.float32),
            torch.ones(2, 1, dtype=torch.bool),
            None,
            "int32 or torch.int64",
        ),
        (
            torch.ones(2, 1, dtype=torch.int32),
            torch.ones(2, 1, dtype=torch.int32),
            None,
            "torch.bool",
        ),
        (
            torch.ones(2, 2, dtype=torch.int32)[:, :1],
            torch.ones(2, 1, dtype=torch.bool),
            None,
            "metadata must be contiguous",
        ),
        (
            torch.ones(2, 1, dtype=torch.int32),
            torch.ones(2, 1, dtype=torch.bool),
            torch.randn(1, 1, 7, dtype=torch.float64),
            "target-logits dtype",
        ),
        (
            torch.ones(2, 1, dtype=torch.int32),
            torch.ones(2, 1, dtype=torch.bool),
            torch.randn(1, 7),
            "target-logits rank",
        ),
        (
            torch.ones(2, 1, dtype=torch.int32),
            torch.ones(2, 1, dtype=torch.bool),
            torch.randn(1, 2, 7),
            "non-sequence target dimensions",
        ),
        (
            torch.ones(2, 1, dtype=torch.int32),
            torch.ones(2, 1, dtype=torch.bool),
            torch.randn(1, 1, 14)[..., ::2],
            "halo logits must be contiguous",
        ),
    ],
)
def test_target_row_addressing_rejects_invalid_metadata(indices, valid, halo, message):
    """Addressed dispatch rejects inconsistent metadata before any fused launch."""
    draft = torch.randn(2, 1, 7)
    target = torch.randn_like(draft)
    with pytest.raises(ValueError, match=message):
        vocab_parallel_tv_distance(
            draft,
            target,
            logits_are_vocab_sharded=False,
            target_row_indices=indices,
            target_valid_rows=valid,
            target_halo_logits=halo,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fused_mtp_tv_saves_only_compact_full_vocab_metadata():
    """Backward state must pack its only vocabulary-sized comparison mask."""
    torch.manual_seed(29)
    draft_logits = torch.randn(3, 2, 4097, device="cuda", requires_grad=True)
    target_logits = torch.randn_like(draft_logits)
    saved_tensors = []

    def pack_hook(tensor):
        saved_tensors.append(tensor)
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack_hook, lambda tensor: tensor):
        output = vocab_parallel_tv_distance(
            draft_logits, target_logits, logits_are_vocab_sharded=False
        )
        output.sum().backward()

    full_vocab_tensors = [
        tensor for tensor in saved_tensors if tensor.numel() == draft_logits.numel()
    ]
    assert len(full_vocab_tensors) == 1
    assert full_vocab_tensors[0].data_ptr() == draft_logits.data_ptr()
    packed_masks = [tensor for tensor in saved_tensors if tensor.dtype == torch.uint8]
    assert [tuple(tensor.shape) for tensor in packed_masks] == [(6, (4097 + 7) // 8)]
    assert not any(
        tensor.dtype in (torch.bool, torch.uint8) and tensor.numel() == draft_logits.numel()
        for tensor in saved_tensors
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fused_mtp_tv_is_deterministic(dtype):
    """Repeated launches of the same implementation are bitwise stable."""
    torch.manual_seed(2026)
    draft_data = torch.randn(4, 1031, device="cuda", dtype=dtype)
    target_logits = torch.randn_like(draft_data)
    grad_output = torch.randn(4, device="cuda")

    def function(draft, target):
        return vocab_parallel_tv_distance(draft, target, logits_are_vocab_sharded=False)

    first_output, first_grad = _run_tv_and_gradient(
        function, draft_data, target_logits, grad_output
    )
    second_output, second_grad = _run_tv_and_gradient(
        function, draft_data, target_logits, grad_output
    )
    assert torch.equal(first_output, second_output)
    assert torch.equal(first_grad, second_grad)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fused_mtp_tv_is_cuda_graph_safe():
    """The fused forward and analytical backward replay inside a CUDA Graph."""
    torch.manual_seed(2031)
    draft_data = torch.randn(4, 1031, device="cuda")
    target_logits = torch.randn_like(draft_data)
    grad_output = torch.randn(4, device="cuda")
    expected_output, expected_grad = _run_tv_and_gradient(
        lambda draft, target: vocab_parallel_tv_distance(
            draft, target, logits_are_vocab_sharded=False
        ),
        draft_data,
        target_logits,
        grad_output,
    )

    static_draft = draft_data.detach().clone().requires_grad_(True)
    static_draft.grad = torch.zeros_like(static_draft)
    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        for _ in range(3):
            static_draft.grad.zero_()
            warmup_output = vocab_parallel_tv_distance(
                static_draft, target_logits, logits_are_vocab_sharded=False
            )
            (warmup_output * grad_output).sum().backward()
    torch.cuda.current_stream().wait_stream(warmup_stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_draft.grad.zero_()
        graph_output = vocab_parallel_tv_distance(
            static_draft, target_logits, logits_are_vocab_sharded=False
        )
        (graph_output * grad_output).sum().backward()
    graph.replay()

    torch.testing.assert_close(graph_output, expected_output, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(static_draft.grad, expected_grad, rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_fused_mtp_tv_target_rows_match_materialized_local_halo_and_invalid(dtype, index_dtype):
    """Direct Triton loads match local, halo, explicitly invalid, and OOB targets."""
    torch.manual_seed(41)
    sequence_length = 5
    vocab_size = 257
    draft_data = torch.randn(sequence_length, 1, vocab_size, device="cuda", dtype=dtype)
    target_logits = torch.randn_like(draft_data, requires_grad=True)
    target_halo = torch.randn(2, 1, vocab_size, device="cuda", dtype=dtype, requires_grad=True)
    target_row_indices = torch.tensor([[1], [4], [5], [6], [99]], device="cuda", dtype=index_dtype)
    target_valid_rows = torch.tensor(
        [[True], [True], [True], [False], [True]], device="cuda", dtype=torch.bool
    )
    grad_output = torch.randn(sequence_length, 1, device="cuda")

    def mapped_tv(draft, target):
        return vocab_parallel_tv_distance(
            draft,
            target,
            tp_group=None,
            logits_are_vocab_sharded=False,
            target_row_indices=target_row_indices,
            target_valid_rows=target_valid_rows,
            target_halo_logits=target_halo,
        )

    materialized_target = torch.zeros_like(target_logits)
    materialized_target[0].copy_(target_logits[1])
    materialized_target[1].copy_(target_logits[4])
    materialized_target[2].copy_(target_halo[0])
    actual, actual_grad = _run_tv_and_gradient(mapped_tv, draft_data, target_logits, grad_output)
    reference, reference_grad = _run_tv_and_gradient(
        _native_tv_distance, draft_data, materialized_target, grad_output
    )

    rtol = 3e-3 if dtype == torch.bfloat16 else 1e-5
    atol = 3e-3 if dtype == torch.bfloat16 else 1e-6
    torch.testing.assert_close(actual, reference, rtol=rtol, atol=atol)
    torch.testing.assert_close(actual_grad, reference_grad, rtol=rtol, atol=atol)
    assert target_logits.grad is None
    assert target_halo.grad is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.skipif(Utils.world_size < 2, reason="TP2 target-row addressing requires two ranks")
def test_fused_mtp_tv_target_rows_match_global_oracle_with_tp2_sharded_vocab():
    """TP2 addressed local, halo, invalid, and OOB rows match a global-vocab oracle."""
    if Utils.world_size % 2 != 0:
        pytest.skip("TP2 target-row addressing requires an even world size")

    Utils.initialize_model_parallel(tensor_model_parallel_size=2)
    try:
        torch.manual_seed(43)
        tp_group = parallel_state.get_tensor_model_parallel_group()
        tp_rank = parallel_state.get_tensor_model_parallel_rank()
        sequence_length = 6
        batch_size = 1
        local_vocab_size = 257
        global_vocab_size = 2 * local_vocab_size

        global_draft_data = torch.randn(
            sequence_length, batch_size, global_vocab_size, device="cuda"
        )
        global_target_logits = torch.randn_like(global_draft_data)
        global_target_halo = torch.randn(2, batch_size, global_vocab_size, device="cuda")
        vocab_start = tp_rank * local_vocab_size
        vocab_end = vocab_start + local_vocab_size
        local_draft_data = global_draft_data[..., vocab_start:vocab_end].contiguous()
        local_target_logits = (
            global_target_logits[..., vocab_start:vocab_end].contiguous().requires_grad_(True)
        )
        local_target_halo = (
            global_target_halo[..., vocab_start:vocab_end].contiguous().requires_grad_(True)
        )

        target_row_indices = torch.tensor(
            [[1], [5], [6], [7], [-1], [99]], device="cuda", dtype=torch.int32
        )
        target_valid_rows = torch.tensor(
            [[True], [True], [True], [True], [False], [True]], device="cuda", dtype=torch.bool
        )
        grad_output = torch.randn(sequence_length, batch_size, device="cuda")

        def mapped_tp_tv(draft, target):
            return vocab_parallel_tv_distance(
                draft,
                target,
                tp_group=tp_group,
                logits_are_vocab_sharded=True,
                target_row_indices=target_row_indices,
                target_valid_rows=target_valid_rows,
                target_halo_logits=local_target_halo,
            )

        materialized_global_target = torch.zeros_like(global_target_logits)
        materialized_global_target[0].copy_(global_target_logits[1])
        materialized_global_target[1].copy_(global_target_logits[5])
        materialized_global_target[2].copy_(global_target_halo[0])
        materialized_global_target[3].copy_(global_target_halo[1])

        actual, actual_grad = _run_tv_and_gradient(
            mapped_tp_tv, local_draft_data, local_target_logits, grad_output
        )
        reference, reference_global_grad = _run_tv_and_gradient(
            _native_tv_distance, global_draft_data, materialized_global_target, grad_output
        )
        reference_local_grad = reference_global_grad[..., vocab_start:vocab_end]

        torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(actual_grad, reference_local_grad, rtol=1e-5, atol=1e-6)
        assert local_target_logits.grad is None
        assert local_target_halo.grad is None
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fused_mtp_tv_target_rows_are_deterministic_and_cuda_graph_safe():
    """Mapped source-plus-halo loads replay deterministically inside a CUDA Graph."""
    torch.manual_seed(2037)
    draft_data = torch.randn(4, 1, 1031, device="cuda")
    target_logits = torch.randn_like(draft_data)
    target_halo = torch.randn(1, 1, 1031, device="cuda")
    target_row_indices = torch.tensor([[1], [2], [3], [4]], device="cuda", dtype=torch.int32)
    target_valid_rows = torch.ones(4, 1, device="cuda", dtype=torch.bool)
    grad_output = torch.randn(4, 1, device="cuda")

    def mapped_tv(draft, target):
        return vocab_parallel_tv_distance(
            draft,
            target,
            tp_group=None,
            logits_are_vocab_sharded=False,
            target_row_indices=target_row_indices,
            target_valid_rows=target_valid_rows,
            target_halo_logits=target_halo,
        )

    expected_output, expected_grad = _run_tv_and_gradient(
        mapped_tv, draft_data, target_logits, grad_output
    )
    repeated_output, repeated_grad = _run_tv_and_gradient(
        mapped_tv, draft_data, target_logits, grad_output
    )
    assert torch.equal(repeated_output, expected_output)
    assert torch.equal(repeated_grad, expected_grad)

    static_draft = draft_data.detach().clone().requires_grad_(True)
    static_draft.grad = torch.zeros_like(static_draft)
    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        for _ in range(3):
            static_draft.grad.zero_()
            warmup_output = mapped_tv(static_draft, target_logits)
            (warmup_output * grad_output).sum().backward()
    torch.cuda.current_stream().wait_stream(warmup_stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_draft.grad.zero_()
        graph_output = mapped_tv(static_draft, target_logits)
        (graph_output * grad_output).sum().backward()
    graph.replay()

    torch.testing.assert_close(graph_output, expected_output, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(static_draft.grad, expected_grad, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("shape", [(0, 257), (3, 0, 257)])
def test_empty_row_logits_use_reference_fallback(shape):
    """Empty leading dimensions never dispatch a zero-program Triton grid."""
    draft = torch.empty(shape)
    target = torch.empty_like(draft)
    assert fused_mtp_tv_unavailable_reason(draft, target) is not None

    if torch.cuda.is_available():
        draft = draft.cuda()
        target = target.cuda()
        assert fused_mtp_tv_unavailable_reason(draft, target) == "logits must have at least one row"
        actual = vocab_parallel_tv_distance(draft, target, logits_are_vocab_sharded=False)
        assert actual.shape == draft.shape[:-1]
        assert actual.numel() == 0
