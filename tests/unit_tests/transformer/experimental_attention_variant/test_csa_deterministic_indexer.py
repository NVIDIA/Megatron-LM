# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Numerical and replay coverage for deterministic CSA indexer gradients."""

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import inspect
import types
from unittest.mock import patch

import pytest
import torch

from megatron.core.transformer.experimental_attention_variant.csa_utils import (
    fused_sparse_attention as dk,
)


def _indexer_case(batch, dtype, device):
    generator = torch.Generator(device=device).manual_seed(193)
    q = torch.randn(batch, 4, 2, 3, generator=generator, device=device, dtype=dtype)
    w = torch.randn(batch, 4, 2, generator=generator, device=device, dtype=dtype)
    k = torch.randn(batch, 3, 3, generator=generator, device=device, dtype=dtype)
    indices = torch.tensor(
        [[0, 2, -1], [1, 2, -1], [-1, -1, -1], [0, 1, 2]], device=device, dtype=torch.int32
    ).expand(batch, -1, -1)
    # Row 1's target sums to 0.75: its score gradient is predict * 0.75 - target, not
    # predict - target.
    target = torch.tensor(
        [[0.25, 0.75, 0.0], [0.5, 0.25, 0.0], [0.0, 0.0, 0.0], [0.2, 0.3, 0.5]], device=device
    ).expand(batch, -1, -1)
    return q, w, k, indices, target


def _reference_indexer_grads(q, w, k, indices, target, loss_coeff, grad_loss, sm_scale):
    """Differentiate per-sample KL directly; no flattened key addressing or manual dK."""
    ref_k = k.detach().float().requires_grad_()
    ref_w = w.detach().float().requires_grad_()
    predict = torch.zeros_like(target)
    loss = torch.zeros((), device=q.device)
    for batch in range(q.shape[0]):
        for row in range(q.shape[1]):
            valid = indices[batch, row] >= 0
            if not valid.any():
                continue
            selected_keys = ref_k[batch, indices[batch, row, valid].long()]
            dot = q[batch, row].float() @ selected_keys.T
            score = (dot.relu() * ref_w[batch, row, :, None]).sum(0) * sm_scale
            log_predict = torch.log_softmax(score, dim=0)
            predict[batch, row, valid] = log_predict.detach().exp()
            loss = loss - (target[batch, row, valid] * log_predict).sum()
    loss = loss * (loss_coeff * grad_loss / (q.shape[0] * q.shape[1]))
    grad_w, grad_k = torch.autograd.grad(loss, (ref_w, ref_k))
    return predict, grad_w.to(w.dtype), grad_k.to(k.dtype)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("compute_grad_w", [False, True])
@pytest.mark.parametrize("chunk_rows", [None, 3])
@pytest.mark.parametrize("torch_deterministic", [True, False])
def test_deterministic_indexer_grads_match_reference_and_replay(
    batch, dtype, compute_grad_w, chunk_rows, torch_deterministic, monkeypatch
):
    # B=1 is also the synthetic-batch contract used by packed THD and CP callers.
    # Three query rows per chunk forces the B=2 case to cross a sample boundary.
    # The torch flag selects index_add_ accumulation; without it the helper sorts key ids
    # and gathers prefix sums. Both orders use the same batch-offset key ids.
    # Tiny test heads otherwise allocate a large unused padding-id array under the 1 GiB budget.
    monkeypatch.setattr(dk, "_DETERMINISTIC_INDEXER_DK_CHUNK_MAX_BYTES", 4096)
    # The fp32 tolerance below assumes true fp32 GEMMs; an earlier test in the same process may
    # have enabled TF32, which cuBLAS honours for the small bmm/matmul shapes used here.
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
    prior_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    if chunk_rows is not None:
        monkeypatch.setattr(dk, "_deterministic_indexer_dk_chunk_rows", lambda *_: chunk_rows)
    device = torch.device("cuda", torch.cuda.current_device())
    q, w, k, indices, target = _indexer_case(batch, dtype, device)
    loss_coeff, grad_loss, sm_scale = 0.7, 1.3, 0.5
    predict, expected_w, expected_k = _reference_indexer_grads(
        q, w, k, indices, target, loss_coeff, grad_loss, sm_scale
    )
    prior_deterministic = torch.are_deterministic_algorithms_enabled()
    prior_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(torch_deterministic)
        results = [
            dk._deterministic_sparse_indexer_grads_wk(
                q.clone(),
                w.clone(),
                k.clone(),
                target.clone(),
                predict.clone(),
                indices.clone(),
                loss_coeff=loss_coeff,
                grad_loss=torch.tensor(grad_loss, device=device),
                sm_scale=sm_scale,
                compute_grad_w=compute_grad_w,
            )
            for _ in range(2)
        ]
    finally:
        torch.use_deterministic_algorithms(prior_deterministic, warn_only=prior_warn_only)
        torch.set_float32_matmul_precision(prior_precision)
    tolerance = (
        dict(rtol=2e-2, atol=2e-3) if dtype == torch.bfloat16 else dict(rtol=1e-5, atol=2e-6)
    )
    for grad_w, grad_k in results:
        torch.testing.assert_close(grad_k, expected_k, **tolerance)
        if compute_grad_w:
            torch.testing.assert_close(grad_w, expected_w, **tolerance)
        else:
            assert grad_w is None
    for first, second in zip(results[0], results[1]):
        if first is not None:
            assert torch.equal(
                first.contiguous().view(torch.uint8), second.contiguous().view(torch.uint8)
            )


@pytest.mark.launch_on_gb200
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_deterministic_indexer_grads_match_cudnn_kernel():
    """The replacement must follow cuDNN's clipped-log KL, not only well-normalized targets."""
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("compares against the SM100 cuDNN sparse-indexer backward")
    pytest.importorskip("cudnn.deepseek_sparse_attention")
    dk._ensure_dsa_namespace()
    device = torch.device("cuda", torch.cuda.current_device())
    generator = torch.Generator(device=device).manual_seed(11)
    rows, keys, heads, dim, topk = 256, 256, 64, 128, 128
    q = torch.randn(1, rows, heads, dim, generator=generator, device=device, dtype=torch.bfloat16)
    w = torch.randn(1, rows, heads, generator=generator, device=device, dtype=torch.bfloat16)
    k = torch.randn(1, keys, dim, generator=generator, device=device, dtype=torch.bfloat16)
    order = torch.rand(rows, keys, generator=generator, device=device).argsort(dim=-1)
    indices = order[:, :topk].int()
    indices[::7, topk // 2 :] = -1
    valid = indices >= 0
    logits = torch.randn(rows, topk, generator=generator, device=device).masked_fill(
        ~valid, float("-inf")
    )
    predict = torch.softmax(logits, dim=-1)
    # Probabilities below exp(-100), including exact zeros, drop out of cuDNN's KL gradient.
    predict[::5, 0] = 0.0
    predict[1::5, 1] = 1e-45
    target = torch.rand(rows, topk, generator=generator, device=device).masked_fill(~valid, 0.0)
    target = target / target.sum(dim=-1, keepdim=True)
    # Attention-sink and local-window mass leave the selected targets summing below one.
    target[::3] *= 0.5
    grad_loss = torch.ones((), device=device)
    det_w, det_k = dk._deterministic_sparse_indexer_grads_wk(
        q,
        w,
        k,
        target.view(1, rows, topk),
        predict.view(1, rows, topk),
        indices.view(1, rows, topk),
        loss_coeff=0.3,
        grad_loss=grad_loss,
        sm_scale=0.1,
        compute_grad_w=True,
    )
    expected = dk._DSA.indexer_backward_wrapper(
        q,
        w,
        k,
        target.view(1, rows, topk).clone(),
        predict.view(1, rows, topk).clone(),
        indices.view(1, rows, topk),
        sm_scale=0.1,
        loss_coeff=0.3,
        grad_loss=grad_loss,
        block_I=128,
    )
    for actual, reference in ((det_k, expected["d_index_k"]), (det_w, expected["d_weights"])):
        error = (actual.float() - reference.float()).norm() / reference.float().norm()
        assert error < 1e-2, f"relative L2 error {error.item():.3e}"


class _FakeBackwardApi:
    """Stands in for ``cudnn.DSA.SparseAttentionBackward``."""

    workspace_bytes = 4096

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def check_support(self):
        return True

    def scratch_workspace_bytes(self):
        return type(self).workspace_bytes


class _RecordingDSA:
    """Stub DSA namespace whose backward wrapper has the deterministic kernel and records kwargs."""

    SparseAttentionBackward = _FakeBackwardApi

    def __init__(self):
        self.calls = []

    def sparse_attention_backward_wrapper(
        self,
        q,
        kv,
        out,
        dout,
        lse,
        attn_sink,
        topk_idxs,
        softmax_scale,
        topk_length,
        deterministic=False,
        workspace=None,
    ):
        self.calls.append(dict(deterministic=deterministic, workspace=workspace))
        return {
            "dq": torch.zeros_like(q),
            "dkv": torch.zeros_like(kv),
            "d_sink": torch.zeros_like(attn_sink),
        }


def _sparse_attn_sample(device):
    q = torch.zeros(8, 64, 16, device=device, dtype=torch.bfloat16)
    kv = torch.zeros(32, 16, device=device, dtype=torch.bfloat16)
    lse = torch.zeros(8, 64, device=device)
    sink = torch.zeros(64, device=device)
    idx = torch.zeros(8, 4, device=device, dtype=torch.int32)
    return q, kv, q, lse, sink, idx


def _workspace_entry(device):
    return dk._DETERMINISTIC_SPARSE_BWD_WORKSPACES[device.index]


def _reset_workspace_cache(monkeypatch, workspace_bytes=4096):
    monkeypatch.setattr(dk, "_DSA", _RecordingDSA())
    monkeypatch.setattr(dk, "_DETERMINISTIC_SPARSE_BWD_WORKSPACES", {})
    monkeypatch.setattr(dk, "_DETERMINISTIC_SPARSE_BWD_RETIRED_WORKSPACES", [])
    monkeypatch.setattr(_FakeBackwardApi, "workspace_bytes", workspace_bytes)


def _retired_ptrs():
    return [t.data_ptr() for t in dk._DETERMINISTIC_SPARSE_BWD_RETIRED_WORKSPACES]


@pytest.mark.launch_on_gb200
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("cuda_graph", [False, True])
def test_sparse_bwd_kwargs_select_deterministic_kernel_and_bound_workspace(monkeypatch, cuda_graph):
    _reset_workspace_cache(monkeypatch)
    device = torch.device("cuda", torch.cuda.current_device())
    args = _sparse_attn_sample(device)

    # Not requested: the default (atomic) kernel, no extra kwargs.
    assert dk._deterministic_sparse_bwd_kwargs(*args, None, 0.5) == {}
    with patch.object(torch.cuda, "get_device_capability", return_value=(10, 0)):
        kwargs = dk._deterministic_sparse_bwd_kwargs(
            *args, None, 0.5, requested=True, cuda_graph=cuda_graph
        )
        assert kwargs["deterministic"] is True
        workspace = kwargs["workspace"]
        assert workspace.is_cuda and workspace.dtype == torch.uint8 and workspace.numel() == 4096
        entry = _workspace_entry(device)
        assert entry.owner_stream == torch.cuda.current_stream(device)
        assert entry.graph_mode is cuda_graph
        assert entry.graph_owned is cuda_graph
        # Same size again: the retained per-device buffer is reused (same storage).
        again = dk._deterministic_sparse_bwd_kwargs(
            *args, None, 0.5, requested=True, cuda_graph=cuda_graph
        )
        assert again["workspace"].data_ptr() == workspace.data_ptr()
        # Smaller request: a slice of the same buffer, no new allocation.
        _FakeBackwardApi.workspace_bytes = 1024
        smaller = dk._deterministic_sparse_bwd_kwargs(
            *args, None, 0.5, requested=True, cuda_graph=cuda_graph
        )
        assert smaller["workspace"].numel() == 1024
        assert smaller["workspace"].data_ptr() == workspace.data_ptr()
        # Growth preserves graph-owned storage, but releases eager-only storage.
        _FakeBackwardApi.workspace_bytes = 8192
        bigger = dk._deterministic_sparse_bwd_kwargs(
            *args, None, 0.5, requested=True, cuda_graph=cuda_graph
        )
        assert bigger["workspace"].numel() == 8192
        assert len(dk._DETERMINISTIC_SPARSE_BWD_WORKSPACES) == 1
        assert _retired_ptrs() == ([workspace.data_ptr()] if cuda_graph else [])
        assert _workspace_entry(device).graph_mode is cuda_graph
        assert _workspace_entry(device).graph_owned is cuda_graph


@pytest.mark.launch_on_gb200
@pytest.mark.parametrize("initial_graph_mode", [False, True])
@pytest.mark.parametrize("requested_bytes", [4096, 8192])
def test_deterministic_sparse_bwd_workspace_rejects_mixed_graph_modes(
    monkeypatch, initial_graph_mode, requested_bytes
):
    """Reject either configuration change before mutating a reused or growing cache entry."""
    _reset_workspace_cache(monkeypatch)
    device = torch.device("cuda", 0)
    stream = types.SimpleNamespace(cuda_stream=1)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: stream)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    q = types.SimpleNamespace(device=device)
    # CPU storage and mocked stream identities isolate this configuration-only guard.
    buffer = torch.empty(4096, dtype=torch.uint8)
    with patch.object(torch, "empty", return_value=buffer) as allocate:
        dk._get_deterministic_sparse_attention_workspace(
            q, None, None, None, None, None, None, 0.5, cuda_graph=initial_graph_mode
        )
        entry = _workspace_entry(device)
        _FakeBackwardApi.workspace_bytes = requested_bytes
        with pytest.raises(RuntimeError, match="Mixing CUDA-graph and non-graph"):
            dk._get_deterministic_sparse_attention_workspace(
                q, None, None, None, None, None, None, 0.5, cuda_graph=not initial_graph_mode
            )
        assert allocate.call_count == 1
        assert _workspace_entry(device) is entry
        assert entry.buffer is buffer
        assert entry.owner_stream is stream
        assert entry.graph_mode is initial_graph_mode
        assert entry.graph_owned is initial_graph_mode
        assert _retired_ptrs() == []


@pytest.mark.launch_on_gb200
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_sparse_bwd_kwargs_reject_missing_kernel_and_unsupported_devices(monkeypatch):
    """Deterministic mode never degrades silently: an old frontend, a non-SM100 GPU or a shape
    cuDNN rejects all raise. A device without an index resolves to the current device."""
    _reset_workspace_cache(monkeypatch)
    device = torch.device("cuda", torch.cuda.current_device())
    args = _sparse_attn_sample(device)
    # ``torch.device("cuda")`` (no index) is keyed under the current device.
    unindexed = types.SimpleNamespace(device=torch.device("cuda"))
    workspace = dk._get_deterministic_sparse_attention_workspace(
        unindexed, None, None, None, None, None, None, 0.5
    )
    assert workspace.numel() == 4096 and _workspace_entry(device).buffer.numel() == 4096
    with patch.object(torch.cuda, "get_device_capability", return_value=(9, 0)):
        with pytest.raises(RuntimeError, match="SM100"):
            dk._deterministic_sparse_bwd_kwargs(*args, None, 0.5, requested=True)
    with patch.object(torch.cuda, "get_device_capability", return_value=(10, 0)):
        monkeypatch.setattr(_FakeBackwardApi, "check_support", lambda self: False)
        with pytest.raises(RuntimeError, match="cuDNN rejected"):
            dk._deterministic_sparse_bwd_kwargs(*args, None, 0.5, requested=True)
        # A frontend whose wrapper has no inspectable signature / no deterministic kwargs.
        monkeypatch.setattr(dk._DSA, "sparse_attention_backward_wrapper", object())
        with pytest.raises(RuntimeError, match="nvidia-cudnn-frontend"):
            dk._deterministic_sparse_bwd_kwargs(*args, None, 0.5, requested=True)


def test_override_indexer_grads_replaces_only_the_deterministic_parts():
    ig = {
        "d_index_q": torch.zeros(2, 3),
        "d_weights": torch.zeros(2, 2, dtype=torch.bfloat16),
        "d_index_k": torch.zeros(4, dtype=torch.bfloat16),
        "unrelated": object(),
    }
    det_k = torch.arange(4.0).reshape(2, 2)  # fp32 and a different shape, same numel
    out = dk._override_indexer_grads(ig, None, det_k)
    assert out["d_index_k"].dtype == torch.bfloat16 and out["d_index_k"].shape == (4,)
    assert torch.equal(out["d_index_k"].float(), torch.arange(4.0))
    assert out["d_weights"] is ig["d_weights"] and out["d_index_q"] is ig["d_index_q"]
    assert "unrelated" not in out
    out = dk._override_indexer_grads(ig, torch.ones(4), det_k)
    assert out["d_weights"].dtype == torch.bfloat16 and out["d_weights"].shape == (2, 2)
    assert torch.equal(out["d_weights"].float(), torch.ones(2, 2))


def test_deterministic_indexer_grads_are_zero_without_selected_keys():
    q = torch.randn(1, 2, 2, 3)
    w = torch.randn(1, 2, 2)
    k = torch.randn(1, 4, 3)
    empty = torch.zeros(1, 2, 0)
    kwargs = dict(loss_coeff=1.0, grad_loss=torch.ones(()), sm_scale=1.0)
    grad_w, grad_k = dk._deterministic_sparse_indexer_grads_wk(
        q, w, k, empty, empty, empty.to(torch.int32), compute_grad_w=True, **kwargs
    )
    assert torch.equal(grad_k, torch.zeros_like(k)) and torch.equal(grad_w, torch.zeros_like(w))
    grad_w, grad_k = dk._deterministic_sparse_indexer_grads_wk(
        q, w, k, empty, empty, empty.to(torch.int32), compute_grad_w=False, **kwargs
    )
    assert grad_w is None and torch.equal(grad_k, torch.zeros_like(k))


def test_native_deterministic_indexer_grad_w_only_on_sm90():
    assert dk._requires_native_deterministic_indexer_grad_w(torch.zeros(1)) is False
    if torch.cuda.is_available():
        expected = torch.cuda.get_device_capability() == (9, 0)
        assert dk._requires_native_deterministic_indexer_grad_w(torch.zeros(1, device="cuda")) is (
            expected
        )


def _spin_cycles(milliseconds=100):
    """Cycles for ``torch.cuda._sleep`` lasting about ``milliseconds`` on this GPU (measured)."""
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    torch.cuda._sleep(1_000_000)
    end.record()
    end.synchronize()
    return int(1_000_000 / max(start.elapsed_time(end), 1e-3) * milliseconds)


@pytest.mark.launch_on_gb200
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_deterministic_sparse_bwd_workspace_handover_serialises_streams(monkeypatch):
    """One scratch buffer per device: handing it to a second stream makes that stream wait for
    the work the first stream has already enqueued on it (real streams, real kernels)."""
    _reset_workspace_cache(monkeypatch)
    device = torch.device("cuda", torch.cuda.current_device())
    args = _sparse_attn_sample(device)
    first, second = torch.cuda.Stream(device=device), torch.cuda.Stream(device=device)
    probe = torch.zeros(4096, dtype=torch.uint8, device=device)
    torch.cuda.synchronize()
    with patch.object(torch.cuda, "get_device_capability", return_value=(10, 0)):
        with torch.cuda.stream(first):
            ws_first = dk._deterministic_sparse_bwd_kwargs(*args, None, 0.5, requested=True)[
                "workspace"
            ]
            ws_first.zero_()
            # Stands in for the backward kernel still running on ``first`` ...
            torch.cuda._sleep(_spin_cycles())
            ws_first.fill_(7)
        with torch.cuda.stream(second):
            # ... when ``second`` asks for the scratch: the hand-out must order it after ``first``.
            ws_second = dk._deterministic_sparse_bwd_kwargs(*args, None, 0.5, requested=True)[
                "workspace"
            ]
            probe.copy_(ws_second)
    torch.cuda.synchronize()
    assert ws_second.data_ptr() == ws_first.data_ptr()
    assert _workspace_entry(device).owner_stream == second
    assert torch.all(probe == 7), "second stream read the scratch before first finished with it"
    # The owner stream can keep using it without further ordering.
    with patch.object(torch.cuda, "get_device_capability", return_value=(10, 0)):
        with torch.cuda.stream(second):
            dk._deterministic_sparse_bwd_kwargs(*args, None, 0.5, requested=True)
    assert _workspace_entry(device).owner_stream == second


@pytest.mark.launch_on_gb200
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("grow_before_capture", [False, True])
def test_deterministic_sparse_bwd_workspace_survives_graph_replay_after_growth(
    monkeypatch, grow_before_capture
):
    """Graph-mode warmup/capture preserve the mode and captured storage across later growth."""
    _reset_workspace_cache(monkeypatch)
    device = torch.device("cuda", torch.cuda.current_device())
    args = _sparse_attn_sample(device)
    main = torch.cuda.current_stream(device)
    capture_stream = torch.cuda.Stream(device=device)
    out = torch.zeros(16, dtype=torch.uint8, device=device)  # static output of the graph

    def hand_out():
        return dk._deterministic_sparse_bwd_kwargs(
            *args, None, 0.5, requested=True, cuda_graph=True
        )["workspace"]

    with patch.object(torch.cuda, "get_device_capability", return_value=(10, 0)):
        warm = hand_out()
        if grow_before_capture:
            _FakeBackwardApi.workspace_bytes = 8192
            warm = hand_out()
        retired_before_capture = _retired_ptrs()
        # Eager warmup under the graph config, then capture on a side stream, as Megatron does.
        assert _workspace_entry(device).graph_mode
        assert _workspace_entry(device).graph_owned
        capture_stream.wait_stream(main)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(capture_stream):
            with torch.cuda.graph(graph, stream=capture_stream):
                captured = hand_out()
                captured.fill_(7)
                out.copy_(captured[:16])
        main.wait_stream(capture_stream)
        assert captured.data_ptr() == warm.data_ptr()
        # Capture bakes the address in but does not take stream ownership.
        assert _workspace_entry(device).owner_stream == main
        assert _workspace_entry(device).graph_mode
        # A graph-mode eager call needs more scratch: the captured buffer must survive.
        _FakeBackwardApi.workspace_bytes *= 2
        grown = hand_out()
        assert grown.data_ptr() != captured.data_ptr()
        assert _retired_ptrs() == retired_before_capture + [captured.data_ptr()]
        grown.zero_()
        out.zero_()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.all(out == 7)
        assert torch.all(dk._DETERMINISTIC_SPARSE_BWD_RETIRED_WORKSPACES[-1] == 7)
        assert torch.all(grown == 0), "replay wrote into the replacement buffer"


def _real_deterministic_sparse_attention_available():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        return False
    try:
        dk._ensure_flash_mla()
        dk._ensure_dsa_namespace()
    except Exception:  # pylint: disable=broad-except
        return False
    wrapper = getattr(dk._DSA, "sparse_attention_backward_wrapper", None)
    return wrapper is not None and "workspace" in inspect.signature(wrapper).parameters


@pytest.mark.launch_on_gb200
@pytest.mark.skipif(
    not _real_deterministic_sparse_attention_available(),
    reason="FlashMLA and cuDNN's deterministic DSA backward on SM100 required",
)
def test_deterministic_sparse_bwd_on_two_streams_matches_single_stream_bitwise():
    """Two independent sparse-attention backwards issued on two streams (sharing the device's
    deterministic scratch) produce the same gradients as running them one after another."""
    device = torch.device("cuda", torch.cuda.current_device())
    total_sq, total_skv, heads, d, d_v, topk = 256, 512, 64, 576, 512, 128
    gen = torch.Generator(device=device).manual_seed(0)

    def make_problem():
        q = torch.randn(total_sq, heads, d, device=device, dtype=torch.bfloat16, generator=gen)
        kv = torch.randn(total_skv, d, device=device, dtype=torch.bfloat16, generator=gen)
        sink = torch.zeros(heads, device=device)
        idx = torch.stack(
            [
                torch.randperm(total_skv, device=device, generator=gen)[:topk]
                for _ in range(total_sq)
            ]
        ).to(torch.int32)
        grad = torch.randn(
            total_sq, heads * d_v, device=device, dtype=torch.bfloat16, generator=gen
        )
        return q, kv, sink, idx, grad

    problems = [make_problem(), make_problem()]

    def run(streams):
        leaves, outs, grads = [], [], []
        for (q, kv, sink, idx, grad), stream in zip(problems, streams):
            q, kv = q.clone().requires_grad_(), kv.clone().requires_grad_()
            # Input clones are produced on the caller stream; side-stream forward must wait.
            stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(stream):
                outs.append(
                    dk.csa_sparse_attn(q, kv, sink, idx, d**-0.5, is_thd=True, deterministic=True)
                )
            leaves.append((q, kv))
            grads.append(grad)
        # Autograd runs each backward on its forward stream: both kernels are in flight together.
        torch.autograd.backward(outs, grads)
        torch.cuda.synchronize()
        return [(q.grad, kv.grad) for q, kv in leaves]

    main = torch.cuda.current_stream(device)
    reference = run([main, main])
    two_streams = run([torch.cuda.Stream(device=device), torch.cuda.Stream(device=device)])
    for (dq_ref, dkv_ref), (dq, dkv) in zip(reference, two_streams):
        assert torch.equal(dq, dq_ref)
        assert torch.equal(dkv, dkv_ref)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("deterministic", [False, True])
def test_csa_sparse_attn_honours_config_flag_without_torch_switch(monkeypatch, deterministic):
    """``deterministic`` (TransformerConfig.deterministic_mode) alone must select the
    deterministic backward kernel; the torch global switch is off here."""
    recorder = _RecordingDSA()
    monkeypatch.setattr(dk, "_DSA", recorder)
    monkeypatch.setattr(dk, "_DETERMINISTIC_SPARSE_BWD_WORKSPACES", {})
    monkeypatch.setattr(dk, "_DETERMINISTIC_SPARSE_BWD_RETIRED_WORKSPACES", [])
    _FakeBackwardApi.workspace_bytes = 4096
    device = torch.device("cuda", torch.cuda.current_device())

    def fake_fwd(q, kv, topk_idxs, softmax_scale, attn_sink=None, topk_length=None, indexer_topk=0):
        return torch.zeros_like(q), torch.zeros(q.shape[0], q.shape[1], device=device), None

    q = torch.zeros(8, 64, 16, device=device, dtype=torch.bfloat16, requires_grad=True)
    kv = torch.zeros(32, 16, device=device, dtype=torch.bfloat16, requires_grad=True)
    sink = torch.zeros(64, device=device)
    idx = torch.zeros(8, 4, device=device, dtype=torch.int32)
    prior = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(False)
    try:
        with (
            patch.object(dk, "_csa_fwd_flash_mla", fake_fwd),
            patch.object(torch.cuda, "get_device_capability", return_value=(10, 0)),
        ):
            out = dk.csa_sparse_attn(
                q, kv, sink, idx, 0.5, is_thd=True, deterministic=deterministic
            )
            out.backward(torch.ones_like(out))
    finally:
        torch.use_deterministic_algorithms(prior)
    (call,) = recorder.calls
    assert call["deterministic"] is deterministic
    assert (call["workspace"] is not None) is deterministic
