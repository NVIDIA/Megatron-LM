# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch

from megatron.core.fusions import fused_mtp_acceptance_counts as fused_module
from megatron.core.fusions.fused_mtp_acceptance_counts import fused_mtp_acceptance_counts
from megatron.core.transformer.multi_token_prediction import _compute_mtp_acceptance_counts

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and fused_module.HAVE_TRITON), reason="needs a GPU and Triton"
)


def _torch_counts(preds, mtp_labels, loss_mask):
    """The PyTorch code that ``_compute_mtp_acceptance_counts`` ran before the fused kernel."""
    labels_match = mtp_labels.transpose(0, 1).contiguous()  # [b, s] => [s, b]
    mask_match = loss_mask.transpose(0, 1).contiguous()  # [b, s] => [s, b]
    valid_positions = mask_match.bool()
    correct = ((preds == labels_match) & valid_positions).sum().float()
    total = valid_positions.sum().float()
    return correct, total


def _assert_same(got, ref):
    for g, r in zip(got, ref):
        assert g.dtype == r.dtype and g.shape == r.shape and g.device == r.device
        assert torch.equal(g, r), (g, r)


def _inputs(seq_len, batch, vocab, mask_dtype, seed=0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    preds = torch.randint(0, vocab, (seq_len, batch), device="cuda", generator=gen)
    labels = torch.randint(0, vocab, (batch, seq_len), device="cuda", generator=gen)
    labels[:, ::3] = preds.t()[:, ::3]
    keep = torch.rand(batch, seq_len, device="cuda", generator=gen) > 0.2
    return preds, labels, keep.to(mask_dtype)


@pytest.fixture
def kernel_only(monkeypatch):
    """Fail if the PyTorch fallback runs, so a test checks the kernel itself."""

    def _no_fallback(*args, **kwargs):
        raise AssertionError("took the PyTorch fallback")

    monkeypatch.setattr(fused_module, "mtp_acceptance_counts_reference", _no_fallback)


MASK_DTYPES = [
    torch.float32,
    torch.bfloat16,
    torch.float16,
    torch.bool,
    torch.int64,
    torch.int32,
    torch.uint8,
]


@pytest.mark.parametrize("seq_len, batch", [(4096, 1), (4095, 1), (2048, 2), (1, 1), (1025, 3)])
@pytest.mark.parametrize("mask_dtype", MASK_DTYPES)
@pytest.mark.parametrize("vocab", [2, 129280])
def test_matches_torch(seq_len, batch, mask_dtype, vocab, kernel_only):
    preds, labels, loss_mask = _inputs(seq_len, batch, vocab, mask_dtype)
    ref = _torch_counts(preds, labels, loss_mask)
    _assert_same(fused_mtp_acceptance_counts(preds, labels, loss_mask), ref)


@pytest.mark.parametrize("seq_len, batch", [(8192, 4), (32768, 1)])
def test_largest_kernel_input_matches_torch(seq_len, batch, kernel_only):
    assert seq_len * batch == fused_module._MAX_KERNEL_NUMEL
    preds, labels, loss_mask = _inputs(seq_len, batch, 129280, torch.float32, seed=1)
    ref = _torch_counts(preds, labels, loss_mask)
    _assert_same(fused_mtp_acceptance_counts(preds, labels, loss_mask), ref)


@pytest.mark.parametrize("fill", ["none_valid", "all_valid_all_correct", "special_floats"])
def test_edge_masks(fill, kernel_only):
    preds, labels, loss_mask = _inputs(777, 2, 50, torch.float32)
    if fill == "none_valid":
        loss_mask.zero_()
    elif fill == "all_valid_all_correct":
        labels.copy_(preds.t())
        loss_mask.fill_(1.0)
    else:
        # .bool() treats NaN, inf and negative values as valid and -0.0 as invalid.
        loss_mask[:, 0::5] = float("nan")
        loss_mask[:, 1::5] = -0.0
        loss_mask[:, 2::5] = float("-inf")
        loss_mask[:, 3::5] = -2.5
    ref = _torch_counts(preds, labels, loss_mask)
    _assert_same(fused_mtp_acceptance_counts(preds, labels, loss_mask), ref)


def test_strided_and_mixed_dtype_inputs(kernel_only):
    """Transposed and sliced views and int32 labels are read through their strides."""
    seq_len, batch = 1000, 3
    preds = torch.randint(0, 100, (batch, seq_len), device="cuda").t()  # [s, b], not contiguous
    labels_full = torch.randint(0, 100, (batch, 2 * seq_len), device="cuda", dtype=torch.int32)
    labels = labels_full[:, ::2]
    labels[:, ::4] = preds.t()[:, ::4].to(torch.int32)
    loss_mask = (torch.rand(seq_len, batch, device="cuda") > 0.5).t()  # [b, s] view of [s, b]
    ref = _torch_counts(preds, labels, loss_mask)
    _assert_same(fused_mtp_acceptance_counts(preds, labels, loss_mask), ref)


@pytest.mark.parametrize(
    "make",
    [
        # CPU tensors.
        lambda: tuple(t.cpu() for t in _inputs(64, 2, 10, torch.float32)),
        # Broadcast preds: [s, 1] against [b, s] labels.
        lambda: (_inputs(64, 1, 10, torch.float32)[0],) + _inputs(64, 2, 10, torch.float32)[1:],
        # Float predictions.
        lambda: (lambda p, l, m: (p.float(), l, m))(*_inputs(64, 2, 10, torch.float32)),
        # No positions.
        lambda: _inputs(0, 2, 10, torch.float32),
    ],
)
def test_fallback_matches_torch(make):
    preds, labels, loss_mask = make()
    ref = _torch_counts(preds, labels, loss_mask)
    _assert_same(fused_mtp_acceptance_counts(preds, labels, loss_mask), ref)


def test_too_many_positions_falls_back(monkeypatch):
    called = []
    reference = fused_module.mtp_acceptance_counts_reference
    monkeypatch.setattr(
        fused_module,
        "mtp_acceptance_counts_reference",
        lambda *args: called.append(True) or reference(*args),
    )
    preds, labels, loss_mask = _inputs(fused_module._MAX_KERNEL_NUMEL + 1, 1, 10, torch.float32)
    _assert_same(
        fused_mtp_acceptance_counts(preds, labels, loss_mask),
        _torch_counts(preds, labels, loss_mask),
    )
    assert called


def test_cuda_graph_capture_and_replay(kernel_only):
    preds, labels, loss_mask = _inputs(4096, 1, 129280, torch.float32)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fused_mtp_acceptance_counts(preds, labels, loss_mask)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        correct, total = fused_mtp_acceptance_counts(preds, labels, loss_mask)
    for seed in (1, 2):
        new_preds, new_labels, new_mask = _inputs(4096, 1, 129280, torch.float32, seed=seed)
        preds.copy_(new_preds)
        labels.copy_(new_labels)
        loss_mask.copy_(new_mask)
        graph.replay()
        torch.cuda.synchronize()
        _assert_same((correct, total), _torch_counts(new_preds, new_labels, new_mask))


@pytest.mark.parametrize("given_preds", [True, False])
def test_compute_mtp_acceptance_counts(given_preds):
    seq_len, batch, vocab = 512, 2, 64
    gen = torch.Generator(device="cuda").manual_seed(0)
    logits = torch.randn(seq_len, batch, vocab, device="cuda", generator=gen)
    argmax = logits.argmax(dim=-1)
    labels = torch.randint(0, vocab, (batch, seq_len), device="cuda", generator=gen)
    labels[:, ::2] = argmax.t()[:, ::2]
    loss_mask = (torch.rand(batch, seq_len, device="cuda", generator=gen) > 0.3).float()
    got = _compute_mtp_acceptance_counts(
        logits, labels, loss_mask, preds=argmax if given_preds else None
    )
    _assert_same(got, _torch_counts(argmax, labels, loss_mask))
