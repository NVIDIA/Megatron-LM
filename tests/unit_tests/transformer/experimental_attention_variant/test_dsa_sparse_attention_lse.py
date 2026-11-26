# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU tests of the sparse-attention LSE output and the indexer-backward ``sm_scale``.

``return_lse=True`` on the fused absorbed sparse attention (``FusedSparseAttentionFunc``, the
cuDNN ``run_fused_absorbed_sparse_attention``, the ``dsa_kernels`` hook and
``dsa._run_sparse_attention``) also returns the LSE of the FlashMLA forward as a
non-differentiable FP32 ``[sq, b, np]`` tensor. ``sm_scale`` on the cuDNN sparse indexer
backward helpers defaults to ``_INDEXER_SOFTMAX_SCALE``. Nothing consumes either yet, so with
the defaults every call must be the call made before. FlashMLA and the cuDNN frontend are
replaced by recording stand-ins; ``test_dsa_sparse_attention_lse_gpu.py`` runs the real
kernels.
"""

from types import SimpleNamespace

import pytest
import torch

from megatron.core import utils as core_utils
from megatron.core.transformer.experimental_attention_variant import dsa as dsa_module
from megatron.core.transformer.experimental_attention_variant import (
    dsa_cudnn_kernels,
    dsa_kernels,
    dsa_tilelang_kernels,
)

# FlashMLA's sparse forward takes a 512-channel latent value.
_VALUE_DIM = 512
_SQ, _BATCH, _HEADS, _SKV = 3, 2, 2, 4
_SOFTMAX_SCALE = 0.25
# (s, b) of the query row of ``_attention_inputs`` that attends no key.
_EMPTY_ROW = (1, 1)


class _RecordingFlashMLA:
    """Stand-in for ``_dsa_fwd_flash_mla`` whose output and LSE encode row and head.

    The LSE is a column slice of a wider buffer, as the real adapter returns it after
    removing FlashMLA's padded heads, so it is not contiguous. Rows without an attended key
    get ``+inf``, FlashMLA's empty-row contract.
    """

    def __init__(self):
        self.calls = []

    def __call__(self, q, kv, topk_idxs, softmax_scale, d_v, attn_sink, topk_length):
        self.calls.append(
            dict(
                q=q.detach().clone(),
                kv=kv.detach().clone(),
                topk_idxs=topk_idxs.detach().clone(),
                softmax_scale=softmax_scale,
                d_v=d_v,
                attn_sink=attn_sink.detach().clone(),
                topk_length=topk_length.detach().clone(),
            )
        )
        rows, heads = q.size(0), q.size(1)
        out = (q[..., :d_v] * 0.5).to(q.dtype)
        wide = torch.zeros((rows, heads + 3), dtype=torch.float32)
        wide[:, :heads] = _encoded_lse(rows, heads)
        wide[topk_length <= 0] = float("inf")
        lse = wide[:, :heads]
        assert not lse.is_contiguous()
        return out, lse


def _encoded_lse(rows, heads):
    """``1000 * flat_row + head + 0.25``: flat row ``s * b + b_idx`` as FlashMLA's rows."""
    flat_rows = torch.arange(rows, dtype=torch.float32).view(rows, 1)
    head_ids = torch.arange(heads, dtype=torch.float32).view(1, heads)
    return 1000.0 * flat_rows + head_ids + 0.25


def _expected_lse():
    """The stand-in's LSE for ``_attention_inputs`` as ``[sq, b, np]``."""
    expected = _encoded_lse(_SQ * _BATCH, _HEADS).view(_SQ, _BATCH, _HEADS).clone()
    expected[_EMPTY_ROW] = float("inf")
    return expected


class _RecordingSparseAttentionBackward:
    """Stand-in for the cuDNN ``sparse_attention_backward_wrapper``."""

    calls: list = []

    @classmethod
    def sparse_attention_backward_wrapper(
        cls, q, kv, out, dO, lse, attn_sink, topk_indices, softmax_scale, topk_length
    ):
        cls.calls.append(
            dict(
                q=q.detach().clone(),
                kv=kv.detach().clone(),
                out=out.detach().clone(),
                dO=dO.detach().clone(),
                lse=lse.detach().clone(),
                attn_sink=attn_sink.detach().clone(),
                topk_indices=topk_indices.detach().clone(),
                softmax_scale=softmax_scale,
                topk_length=topk_length.detach().clone(),
            )
        )
        dq = torch.zeros_like(q)
        dq[..., : dO.size(-1)] = dO * 3
        dkv = kv * dO.float().sum().to(kv.dtype) + 1
        return {"dq": dq, "dkv": dkv}


@pytest.fixture
def fake_flash_mla_and_cudnn(monkeypatch):
    """Replace FlashMLA and the cuDNN DSA namespace with recording stand-ins."""
    flash_mla = _RecordingFlashMLA()
    _RecordingSparseAttentionBackward.calls = []
    monkeypatch.setattr(dsa_cudnn_kernels, "_dsa_fwd_flash_mla", flash_mla)
    monkeypatch.setattr(dsa_cudnn_kernels, "_ensure_dsa_namespace", lambda: None)
    monkeypatch.setattr(dsa_cudnn_kernels, "_cudnn_dsa", _RecordingSparseAttentionBackward)
    return SimpleNamespace(forward=flash_mla, backward=_RecordingSparseAttentionBackward)


def _attention_inputs(dtype=torch.bfloat16):
    generator = torch.Generator().manual_seed(20261111)
    query = torch.randn((_SQ, _BATCH, _HEADS, _VALUE_DIM), generator=generator).to(dtype)
    key = torch.randn((_SKV, _BATCH, 1, _VALUE_DIM), generator=generator).to(dtype)
    # Row 1 of batch 1 attends no key; -1 slots sit between valid ids.
    topk_indices = torch.tensor(
        [[[0, -1, 2], [1, 3, -1], [3, 2, 0]], [[2, -1, -1], [-1, -1, -1], [0, 1, 3]]],
        dtype=torch.int32,
    )
    grad_output = torch.randn((_SQ, _BATCH, _HEADS, _VALUE_DIM), generator=generator).to(dtype)
    return query, key, topk_indices, grad_output


def _assert_same_calls(first, second):
    assert len(first) == len(second)
    for call_a, call_b in zip(first, second):
        assert call_a.keys() == call_b.keys()
        for name, value in call_a.items():
            if isinstance(value, torch.Tensor):
                assert value.dtype == call_b[name].dtype, name
                assert torch.equal(value, call_b[name]), name
            else:
                assert value == call_b[name], name


def _run_cudnn_attention(query, key, topk_indices, grad_output, **kwargs):
    query = query.detach().clone().requires_grad_(True)
    key = key.detach().clone().requires_grad_(True)
    result = dsa_cudnn_kernels.run_fused_absorbed_sparse_attention(
        query=query,
        key=key,
        topk_indices=topk_indices,
        softmax_scale=_SOFTMAX_SCALE,
        v_channels=_VALUE_DIM,
        **kwargs,
    )
    assert result is not None
    output = result[0] if isinstance(result, tuple) else result
    output.backward(grad_output)
    return result, query.grad, key.grad


@pytest.mark.parametrize("with_topk_length", [False, True], ids=["compacted", "topk_length"])
def test_cudnn_sparse_attention_return_lse_returns_saved_lse_and_keeps_calls(
    fake_flash_mla_and_cudnn, with_topk_length
):
    """Default output is a tensor; ``return_lse`` adds the FlashMLA LSE as ``[sq, b, np]``."""
    query, key, topk_indices, grad_output = _attention_inputs()
    kwargs = {}
    if with_topk_length:
        kwargs["topk_length"] = torch.tensor([[2, 2, 3], [1, 0, 3]], dtype=torch.int32)
        topk_indices = torch.tensor(
            [[[0, 2, -1], [1, 3, -1], [3, 2, 0]], [[2, -1, -1], [-1, -1, -1], [0, 1, 3]]],
            dtype=torch.int32,
        )
    fakes = fake_flash_mla_and_cudnn

    default, default_dq, default_dkv = _run_cudnn_attention(
        query, key, topk_indices, grad_output, **kwargs
    )
    default_forward = list(fakes.forward.calls)
    default_backward = list(fakes.backward.calls)
    fakes.forward.calls.clear()
    fakes.backward.calls.clear()
    with_lse, lse_dq, lse_dkv = _run_cudnn_attention(
        query, key, topk_indices, grad_output, return_lse=True, **kwargs
    )

    assert isinstance(default, torch.Tensor)
    assert isinstance(with_lse, tuple) and len(with_lse) == 2
    output, lse = with_lse
    assert torch.equal(output, default)
    assert torch.equal(lse_dq, default_dq)
    assert torch.equal(lse_dkv, default_dkv)
    _assert_same_calls(fakes.forward.calls, default_forward)
    _assert_same_calls(fakes.backward.calls, default_backward)

    assert lse.dtype == torch.float32
    assert lse.shape == (_SQ, _BATCH, _HEADS)
    assert not lse.requires_grad and lse.grad_fn is None
    # Checked after backward: backward did not change the LSE it handed out.
    expected = _expected_lse()
    torch.testing.assert_close(lse, expected, rtol=0, atol=0)
    # Axis 0 is the query row and axis 1 the batch: entry (s, b, h) is flat row s * B + b.
    assert lse[2, 1, 1].item() == 1000.0 * (2 * _BATCH + 1) + 1 + 0.25
    assert torch.isinf(lse[_EMPTY_ROW]).all()
    # The backward read the same LSE.
    torch.testing.assert_close(
        fakes.backward.calls[0]["lse"], expected.view(_SQ * _BATCH, _HEADS), rtol=0, atol=0
    )


def test_cudnn_sparse_attention_lse_is_the_tensor_backward_reads(fake_flash_mla_and_cudnn):
    """Changing the returned LSE in place fails backward instead of changing dQ/dKV."""
    query, key, topk_indices, grad_output = _attention_inputs()
    query.requires_grad_(True)
    output, lse = dsa_cudnn_kernels.run_fused_absorbed_sparse_attention(
        query=query,
        key=key,
        topk_indices=topk_indices,
        softmax_scale=_SOFTMAX_SCALE,
        v_channels=_VALUE_DIM,
        return_lse=True,
    )
    lse.add_(1.0)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        output.backward(grad_output)


def test_cudnn_sparse_attention_lse_output_needs_no_gradient(fake_flash_mla_and_cudnn):
    """A loss built from the output alone backpropagates; the LSE gets no gradient slot."""
    query, key, topk_indices, grad_output = _attention_inputs(dtype=torch.float32)
    query.requires_grad_(True)
    output, lse = dsa_cudnn_kernels.run_fused_absorbed_sparse_attention(
        query=query,
        key=key,
        topk_indices=topk_indices,
        softmax_scale=_SOFTMAX_SCALE,
        v_channels=_VALUE_DIM,
        return_lse=True,
    )
    # The LSE may feed later (gradient-free) work, such as a loss target.
    loss = (output * grad_output).sum() + (lse * 0.0).sum()
    (grad_query,) = torch.autograd.grad(loss, (query,))

    reference_query = query.detach().clone().requires_grad_(True)
    reference = dsa_cudnn_kernels.run_fused_absorbed_sparse_attention(
        query=reference_query,
        key=key,
        topk_indices=topk_indices,
        softmax_scale=_SOFTMAX_SCALE,
        v_channels=_VALUE_DIM,
    )
    (reference_grad,) = torch.autograd.grad((reference * grad_output).sum(), (reference_query,))
    assert torch.equal(grad_query, reference_grad)


def test_fused_sparse_attention_direct_apply_keeps_legacy_arity(fake_flash_mla_and_cudnn):
    """A direct ``apply`` without the trailing ``return_lse`` still runs forward and backward."""
    query, key, topk_indices, grad_output = _attention_inputs()
    query.requires_grad_(True)
    kv_full = key.squeeze(2).contiguous().requires_grad_(True)
    output = dsa_cudnn_kernels.FusedSparseAttentionFunc.apply(
        query, kv_full, topk_indices, _SOFTMAX_SCALE, _VALUE_DIM, None, False, False
    )
    assert isinstance(output, torch.Tensor)
    output.backward(grad_output)
    assert query.grad is not None and kv_full.grad is not None


# ---------------------------------------------------------------------------------------
# dsa_kernels hook
# ---------------------------------------------------------------------------------------


class _Config:
    attention_backend = "auto"
    dsa_kernel_backend = "cudnn"


def _call_hook(**kwargs):
    marker = object()
    return dsa_kernels.run_fused_absorbed_sparse_attention(
        _Config, marker, marker, marker, 1.0, 8, marker, **kwargs
    )


def test_hook_default_call_keeps_old_kwargs_and_return_lse_is_forwarded(monkeypatch):
    calls = []
    output, lse = torch.ones(1), torch.zeros(1)

    def hook(*args, **kwargs):
        calls.append((args, kwargs))
        return (output, lse) if kwargs.get("return_lse") else output

    monkeypatch.setattr(dsa_kernels, "_resolve_fused_hook", lambda _config, _name: hook)

    assert _call_hook(all_topk_rows_nonempty=True) is output
    assert len(calls[0][0]) == 6
    assert calls[0][1] == {"all_topk_rows_nonempty": True, "deterministic": False}

    assert _call_hook(all_topk_rows_nonempty=True, return_lse=True) == (output, lse)
    assert calls[1][1] == {
        "all_topk_rows_nonempty": True,
        "deterministic": False,
        "return_lse": True,
    }


def test_hook_return_lse_reaches_the_cudnn_backend_signature():
    """The cuDNN backend hook declares ``return_lse``, so the hook offers it."""
    accepted = core_utils.filter_kwargs_for_callable(
        dsa_cudnn_kernels.run_fused_absorbed_sparse_attention,
        {"all_topk_rows_nonempty": True, "deterministic": False, "return_lse": True},
    )
    assert accepted["return_lse"] is True


def test_hook_return_lse_raises_for_tilelang_backend(monkeypatch):
    """TileLang's sparse attention returns no LSE: refuse before running it."""

    class TileLangConfig:
        attention_backend = "auto"
        dsa_kernel_backend = "tilelang"

    def fail(*_args, **_kwargs):
        raise AssertionError("the TileLang kernel must not run when it cannot return the LSE")

    monkeypatch.setattr(
        dsa_tilelang_kernels.tilelang_dsa, "run_fused_absorbed_sparse_attention", fail
    )
    monkeypatch.setattr(dsa_kernels, "_load_backend", lambda _config: dsa_tilelang_kernels)
    marker = object()
    with pytest.raises(RuntimeError, match="'tilelang' cannot return the sparse-attention LSE"):
        dsa_kernels.run_fused_absorbed_sparse_attention(
            TileLangConfig, marker, marker, marker, 1.0, 8, None, return_lse=True
        )


def test_hook_return_lse_raises_for_legacy_and_opaque_hooks(monkeypatch):
    calls = []

    def legacy_hook(query, key, topk_indices, softmax_scale, v_channels, topk_length):
        calls.append("legacy")
        return object()

    monkeypatch.setattr(dsa_kernels, "_resolve_fused_hook", lambda _config, _name: legacy_hook)
    with pytest.raises(RuntimeError, match="takes no return_lse argument"):
        _call_hook(return_lse=True)

    def opaque_hook(*_args, **_kwargs):
        calls.append("opaque")
        return object()

    def fail_signature(_fn):
        raise ValueError("opaque callable")

    monkeypatch.setattr(dsa_kernels, "_resolve_fused_hook", lambda _config, _name: opaque_hook)
    monkeypatch.setattr(core_utils.inspect, "signature", fail_signature)
    with pytest.raises(RuntimeError, match="takes no return_lse argument"):
        _call_hook(return_lse=True)
    assert calls == []


@pytest.mark.parametrize(
    "returned", ["tensor", "pair_without_lse"], ids=["bare-output", "lse-none"]
)
def test_hook_return_lse_rejects_a_result_without_lse(monkeypatch, returned):
    output = torch.ones(1)

    def hook(query, key, topk_indices, softmax_scale, v_channels, topk_length, return_lse=False):
        return output if returned == "tensor" else (output, None)

    monkeypatch.setattr(dsa_kernels, "_resolve_fused_hook", lambda _config, _name: hook)
    with pytest.raises(RuntimeError, match="did not return an \\(output, lse\\) pair"):
        _call_hook(return_lse=True)


def test_hook_return_lse_passes_a_decline_through(monkeypatch):
    def hook(query, key, topk_indices, softmax_scale, v_channels, topk_length, return_lse=False):
        return None

    monkeypatch.setattr(dsa_kernels, "_resolve_fused_hook", lambda _config, _name: hook)
    assert _call_hook(return_lse=True) is None


# ---------------------------------------------------------------------------------------
# dsa._run_sparse_attention
# ---------------------------------------------------------------------------------------


def _sparse_attention_config(**overrides):
    values = dict(kv_lora_rank=_VALUE_DIM, attention_backend="auto", dsa_kernel_backend="cudnn")
    values.update(overrides)
    return SimpleNamespace(**values)


def _run_layer_attention(config, query, key, up_v_weight, topk_indices, **kwargs):
    return dsa_module._run_sparse_attention(
        absorbed_mla=True,
        query=query,
        key=key,
        value=None,
        up_v_weight=up_v_weight,
        topk_indices=topk_indices,
        softmax_scale=_SOFTMAX_SCALE,
        config=config,
        mask=None,
        varlen_starts=None,
        varlen_ends=None,
        key_positions=None,
        **kwargs,
    )


def test_run_sparse_attention_return_lse_through_the_cudnn_backend(
    fake_flash_mla_and_cudnn, monkeypatch
):
    """``_run_sparse_attention`` reaches the cuDNN hook; the default call stays unchanged."""
    query, key, topk_indices, grad_output = _attention_inputs()
    up_v_weight = torch.randn((_HEADS, 4, _VALUE_DIM), dtype=torch.bfloat16)
    hook_kwargs = []
    real_hook = dsa_kernels.run_fused_absorbed_sparse_attention

    def recording_hook(*args, **kwargs):
        hook_kwargs.append(dict(kwargs))
        return real_hook(*args, **kwargs)

    monkeypatch.setattr(dsa_kernels, "run_fused_absorbed_sparse_attention", recording_hook)

    def fail_unfused(*_args, **_kwargs):
        raise AssertionError("the fused backend must run this call")

    monkeypatch.setattr(dsa_module, "_unfused_absorbed_dsa_fn", fail_unfused)
    config = _sparse_attention_config()
    grad = torch.randn((_SQ, _BATCH, _HEADS * 4), dtype=torch.bfloat16)

    def run(**kwargs):
        q = query.detach().clone().requires_grad_(True)
        k = key.detach().clone().requires_grad_(True)
        w = up_v_weight.detach().clone().requires_grad_(True)
        result = _run_layer_attention(config, q, k, w, topk_indices, **kwargs)
        output = result[0] if isinstance(result, tuple) else result
        output.backward(grad)
        return result, (q.grad, k.grad, w.grad)

    default, default_grads = run(all_topk_rows_nonempty=False)
    with_lse, lse_grads = run(all_topk_rows_nonempty=False, return_lse=True)

    assert hook_kwargs[0] == {"topk_length": None, "all_topk_rows_nonempty": False}
    assert hook_kwargs[1] == {
        "topk_length": None,
        "all_topk_rows_nonempty": False,
        "return_lse": True,
    }
    assert isinstance(default, torch.Tensor)
    output, lse = with_lse
    assert output.shape == (_SQ, _BATCH, _HEADS * 4)
    assert torch.equal(output, default)
    for grad_a, grad_b in zip(lse_grads, default_grads):
        assert torch.equal(grad_a, grad_b)
    torch.testing.assert_close(lse, _expected_lse(), rtol=0, atol=0)
    assert not lse.requires_grad


@pytest.mark.parametrize(
    "absorbed, overrides",
    [(False, {}), (True, {"attention_backend": "unfused"}), (True, {"dsa_kernel_backend": "none"})],
    ids=["non-absorbed", "unfused-attention-backend", "no-dsa-kernel-backend"],
)
def test_run_sparse_attention_return_lse_raises_without_fused_kernel(
    monkeypatch, absorbed, overrides
):
    """The unfused attention returns no LSE, so ``return_lse`` raises before any attention."""

    def fail(*_args, **_kwargs):
        raise AssertionError("no attention may run when the LSE cannot be returned")

    monkeypatch.setattr(dsa_module, "_unfused_absorbed_dsa_fn", fail)
    monkeypatch.setattr(dsa_module, "unfused_dsa_fn", fail)
    monkeypatch.setattr(dsa_kernels, "run_fused_absorbed_sparse_attention", fail)
    query, key, topk_indices, _grad = _attention_inputs()
    with pytest.raises(RuntimeError, match="unfused DSA attention does not return the LSE"):
        dsa_module._run_sparse_attention(
            absorbed_mla=absorbed,
            query=query,
            key=key,
            value=None if absorbed else key,
            up_v_weight=torch.zeros((_HEADS, 4, _VALUE_DIM), dtype=torch.bfloat16),
            topk_indices=topk_indices,
            softmax_scale=_SOFTMAX_SCALE,
            config=_sparse_attention_config(**overrides),
            mask=None,
            varlen_starts=None,
            varlen_ends=None,
            key_positions=None,
            return_lse=True,
        )


def test_run_sparse_attention_decline_falls_back_only_without_return_lse(monkeypatch):
    """A fused decline keeps the PyTorch fallback by default and raises with ``return_lse``."""
    query, key, topk_indices, _grad = _attention_inputs()
    up_v_weight = torch.zeros((_HEADS, 4, _VALUE_DIM), dtype=torch.bfloat16)
    unfused_calls = []

    def unfused(query, key, topk_indices, softmax_scale, v_channels, **_kwargs):
        unfused_calls.append(True)
        return query.new_zeros((query.size(0), query.size(1), query.size(2), v_channels))

    monkeypatch.setattr(dsa_module, "_unfused_absorbed_dsa_fn", unfused)
    monkeypatch.setattr(
        dsa_kernels, "run_fused_absorbed_sparse_attention", lambda *_args, **_kwargs: None
    )
    config = _sparse_attention_config()

    output = _run_layer_attention(config, query, key, up_v_weight, topk_indices)
    assert isinstance(output, torch.Tensor) and unfused_calls == [True]

    with pytest.raises(RuntimeError, match="declined the fused sparse attention"):
        _run_layer_attention(config, query, key, up_v_weight, topk_indices, return_lse=True)
    assert unfused_calls == [True]


# ---------------------------------------------------------------------------------------
# sm_scale of the sparse indexer backward
# ---------------------------------------------------------------------------------------


class _RecordingIndexerBackward:
    calls: list = []

    @classmethod
    def indexer_backward_wrapper(
        cls,
        q_indexer,
        weights,
        k_indexer,
        attn_score,
        index_score,
        topk_indices,
        sm_scale,
        loss_coeff,
        grad_loss,
        block_I,
        topk_indices_global=False,
    ):
        cls.calls.append(
            dict(
                rows=q_indexer.size(1),
                sm_scale=sm_scale,
                loss_coeff=loss_coeff,
                block_I=block_I,
                topk_indices_global=topk_indices_global,
            )
        )
        return {
            "d_index_q": torch.full_like(q_indexer, sm_scale),
            "d_weights": torch.full_like(weights, sm_scale),
            "d_index_k": torch.full_like(k_indexer, sm_scale),
        }


@pytest.fixture
def recording_indexer_backward(monkeypatch):
    _RecordingIndexerBackward.calls = []
    monkeypatch.setattr(dsa_cudnn_kernels, "_cudnn_dsa", _RecordingIndexerBackward)
    return _RecordingIndexerBackward


def _indexer_backward_inputs(batch=2, sq=4, skv=5, heads=3, dim=4, topk=4):
    """Fixed target and predict over the selected keys, with ``-1`` slots."""
    generator = torch.Generator().manual_seed(20261112)
    q = torch.randn((batch, sq, heads, dim), generator=generator)
    weights = torch.randn((batch, sq, heads), generator=generator)
    k = torch.randn((batch, skv, dim), generator=generator)
    local = torch.tensor(
        [
            [[0, 2, 4, -1], [1, 2, -1, -1], [4, 0, 3, 2], [3, -1, -1, -1]],
            [[4, 2, 0, -1], [1, 0, 4, 3], [3, 2, -1, -1], [0, 1, 2, 4]],
        ],
        dtype=torch.int32,
    )[:batch, :sq, :topk]
    valid = local >= 0
    predict = torch.rand((batch, sq, topk), generator=generator).masked_fill(~valid, 0.0)
    predict = predict / predict.sum(dim=-1, keepdim=True)
    target = torch.rand((batch, sq, topk), generator=generator).masked_fill(~valid, 0.0)
    target = target / target.sum(dim=-1, keepdim=True)
    offsets = torch.arange(batch, dtype=torch.int32).view(batch, 1, 1) * skv
    global_ids = torch.where(valid, local + offsets, torch.full_like(local, -1))
    return SimpleNamespace(
        q=q,
        weights=weights,
        k=k,
        local=local,
        valid=valid,
        target=target,
        predict=predict,
        global_ids=global_ids.contiguous(),
    )


def test_call_sparse_indexer_backward_default_and_explicit_sm_scale(recording_indexer_backward):
    x = _indexer_backward_inputs()
    kwargs = dict(loss_coeff=0.5, grad_loss=torch.ones(()), block_i=128)

    dsa_cudnn_kernels._call_sparse_indexer_backward(
        x.q, x.weights, x.k, x.target, x.predict, x.global_ids, **kwargs
    )
    dsa_cudnn_kernels._call_sparse_indexer_backward(
        x.q, x.weights, x.k, x.target, x.predict, x.global_ids, sm_scale=0.125, **kwargs
    )

    default_call, explicit_call = recording_indexer_backward.calls
    assert dsa_cudnn_kernels._INDEXER_SOFTMAX_SCALE == 1.0
    assert default_call == dict(
        rows=4, sm_scale=1.0, loss_coeff=0.5, block_I=128, topk_indices_global=True
    )
    assert explicit_call == dict(default_call, sm_scale=0.125)


def test_run_sparse_indexer_backward_passes_sm_scale_to_every_chunk(
    recording_indexer_backward, monkeypatch
):
    monkeypatch.setattr(dsa_cudnn_kernels, "_CUDA_GRID_Y_MAX", 2)
    monkeypatch.setattr(dsa_cudnn_kernels, "_SPARSE_INDEXER_BACKWARD_CHUNK_ROWS", 3)
    x = _indexer_backward_inputs(batch=1)
    kwargs = dict(loss_coeff=1.0, grad_loss=torch.ones(()), block_i=128)

    default = dsa_cudnn_kernels._run_sparse_indexer_backward(
        x.q, x.weights, x.k, x.target.clone(), x.predict.clone(), x.global_ids, **kwargs
    )
    scaled = dsa_cudnn_kernels._run_sparse_indexer_backward(
        x.q,
        x.weights,
        x.k,
        x.target.clone(),
        x.predict.clone(),
        x.global_ids,
        sm_scale=0.5,
        **kwargs,
    )

    calls = recording_indexer_backward.calls
    assert [(call["rows"], call["sm_scale"]) for call in calls] == [
        (3, 1.0),
        (1, 1.0),
        (3, 0.5),
        (1, 0.5),
    ]
    for grad in default:
        assert torch.all(grad.float() > 0)
    torch.testing.assert_close(scaled[0], torch.full_like(x.q, 0.5), rtol=0, atol=0)
    torch.testing.assert_close(scaled[1], torch.full_like(x.weights, 0.5), rtol=0, atol=0)
    # dK sums the chunks' stand-in values.
    torch.testing.assert_close(scaled[2], torch.full_like(x.k, 1.0), rtol=0, atol=0)


def _autograd_indexer_grads(x, *, sm_scale, loss_coeff, grad_loss):
    """dW and dK of ``grad_loss * loss_coeff * sum_rows CE(target, softmax(I)) / (B * S_q)``."""
    batch, sq, _heads, dim = x.q.shape
    skv = x.k.size(1)
    weights = x.weights.detach().clone().requires_grad_(True)
    k = x.k.detach().clone().requires_grad_(True)
    safe = x.local.clamp_min(0).long()
    selected = torch.gather(
        k.unsqueeze(1).expand(batch, sq, skv, dim),
        2,
        safe.unsqueeze(-1).expand(batch, sq, safe.size(-1), dim),
    )
    dots = torch.einsum("bqhd,bqtd->bqht", x.q, selected) * sm_scale
    logits = (torch.relu(dots) * weights.unsqueeze(-1)).sum(dim=2)
    log_predict = torch.log_softmax(logits.masked_fill(~x.valid, float("-inf")), dim=-1)
    log_predict = log_predict.masked_fill(~x.valid, 0.0)
    loss = -(x.target * log_predict).sum() * loss_coeff / (batch * sq) * grad_loss
    grad_w, grad_k = torch.autograd.grad(loss, (weights, k))
    predict = log_predict.exp().masked_fill(~x.valid, 0.0)
    return predict.detach(), grad_w, grad_k


def test_deterministic_indexer_grads_follow_sm_scale():
    """Explicit default is bitwise the default; half the scale halves dW and dK exactly."""
    x = _indexer_backward_inputs()
    loss_coeff, grad_loss = 0.7, torch.tensor(1.25)

    def grads(**kwargs):
        return dsa_cudnn_kernels._deterministic_sparse_indexer_grads_wk(
            x.q,
            x.weights,
            x.k,
            x.target.clone(),
            x.predict.clone(),
            x.global_ids,
            loss_coeff=loss_coeff,
            grad_loss=grad_loss,
            compute_grad_w=True,
            **kwargs,
        )

    default_w, default_k = grads()
    explicit_w, explicit_k = grads(sm_scale=dsa_cudnn_kernels._INDEXER_SOFTMAX_SCALE)
    half_w, half_k = grads(sm_scale=0.5 * dsa_cudnn_kernels._INDEXER_SOFTMAX_SCALE)

    assert torch.equal(explicit_w, default_w) and torch.equal(explicit_k, default_k)
    # With the scores held fixed, every term carries one factor of sm_scale; halving it is
    # exact in floating point.
    assert torch.equal(half_w, default_w * 0.5) and torch.equal(half_k, default_k * 0.5)
    assert not torch.equal(half_k, default_k)


@pytest.mark.parametrize("sm_scale", [1.0, 0.5, 1.0 / 64.0])
def test_deterministic_indexer_grads_match_autograd_at_sm_scale(sm_scale):
    """The replacement's dW and dK are the KL gradients of ``sum_h w_h relu(s q_h . k)``."""
    x = _indexer_backward_inputs()
    loss_coeff, grad_loss = 0.7, torch.tensor(1.25)
    predict, reference_w, reference_k = _autograd_indexer_grads(
        x, sm_scale=sm_scale, loss_coeff=loss_coeff, grad_loss=grad_loss
    )

    grad_w, grad_k = dsa_cudnn_kernels._deterministic_sparse_indexer_grads_wk(
        x.q,
        x.weights,
        x.k,
        x.target.clone(),
        predict.contiguous(),
        x.global_ids,
        loss_coeff=loss_coeff,
        grad_loss=grad_loss,
        compute_grad_w=True,
        sm_scale=sm_scale,
    )

    torch.testing.assert_close(grad_w, reference_w, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(grad_k, reference_k, rtol=2e-5, atol=2e-6)


def test_run_sparse_indexer_backward_deterministic_uses_sm_scale(
    recording_indexer_backward, monkeypatch
):
    """Deterministic mode recomputes dK (and SM90 dW) with the requested scale."""
    monkeypatch.setattr(
        dsa_cudnn_kernels, "_requires_native_deterministic_indexer_grad_w", lambda *_: True
    )
    x = _indexer_backward_inputs()
    loss_coeff, grad_loss = 0.7, torch.tensor(1.25)
    predict, reference_w, reference_k = _autograd_indexer_grads(
        x, sm_scale=0.5, loss_coeff=loss_coeff, grad_loss=grad_loss
    )

    grad_q, grad_w, grad_k = dsa_cudnn_kernels._run_sparse_indexer_backward(
        x.q,
        x.weights,
        x.k,
        x.target.clone(),
        predict.contiguous(),
        x.global_ids,
        loss_coeff=loss_coeff,
        grad_loss=grad_loss,
        block_i=128,
        deterministic=True,
        sm_scale=0.5,
    )

    assert [call["sm_scale"] for call in recording_indexer_backward.calls] == [0.5]
    torch.testing.assert_close(grad_q, torch.full_like(x.q, 0.5), rtol=0, atol=0)
    torch.testing.assert_close(grad_w, reference_w, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(grad_k, reference_k, rtol=2e-5, atol=2e-6)
