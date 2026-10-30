# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from types import SimpleNamespace

import pytest
import torch

from megatron.core.ops.attention.dsa import backends
from megatron.core.ops.attention.dsa.reference import torch_kpool_score_postprocess
from megatron.core.transformer.experimental_attention_variant import dsa
from megatron.core.transformer.transformer_config import TransformerConfig

GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA and Triton")


def test_torch_backend_does_not_import_optional_kernels(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("torch selection must not import Triton")

    monkeypatch.setattr(backends, "require", unexpected)
    assert backends.select_kpool_score_postprocess("torch", 2) is torch_kpool_score_postprocess


def test_selected_backend_failure_is_not_silently_downgraded(monkeypatch):
    def unavailable(*args, **kwargs):
        raise ImportError("Triton KPool score postprocessing requires Triton")

    monkeypatch.setattr(backends, "require", unavailable)
    with pytest.raises(ImportError, match="KPool.*Triton"):
        backends.select_kpool_score_postprocess("triton", 32)


@pytest.mark.parametrize("backend,heads", [("auto", 32), ("triton", 16)])
def test_invalid_backend_or_heads_fail_before_import(backend, heads, monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("invalid choices must fail before import")

    monkeypatch.setattr(backends, "require", unexpected)
    with pytest.raises(ValueError):
        backends.select_kpool_score_postprocess(backend, heads)


def test_backend_binds_the_selected_callable_once(monkeypatch):
    requests = []
    implementation = lambda scores, weights, use_relu: scores.sum(2)

    def available(*args, **kwargs):
        requests.append((args, kwargs))
        return SimpleNamespace(triton_kpool_score_postprocess=implementation)

    monkeypatch.setattr(backends, "require", available)
    selected = backends.select_kpool_score_postprocess("triton", 32)
    assert selected is implementation
    selected(torch.ones(1, 1, 32, 2), torch.ones(1, 1, 32), True)
    assert len(requests) == 1


@pytest.mark.parametrize(
    "override,message",
    [
        ({"dsa_indexer_kpool_score_backend": "auto"}, "torch or triton"),
        ({"dsa_indexer_kpool": 1}, "KPool > 1"),
        ({"dsa_indexer_n_heads": 16}, "32 heads"),
        ({"dsa_indexer_weights_proj_output_dtype": "bf16"}, "FP32 weights"),
        ({"dsa_indexer_loss_coeff": 0.1}, "disabled indexer loss"),
    ],
)
def test_direct_indexer_config_validation(override, message):
    values = dict(
        dsa_indexer_kpool=4,
        dsa_indexer_topk=8,
        dsa_indexer_n_heads=32,
        dsa_indexer_kpool_score_backend="triton",
        dsa_indexer_loss_coeff=0.0,
        dsa_indexer_weights_proj_output_dtype="fp32",
    )
    values.update(override)
    with pytest.raises(ValueError, match=message):
        dsa._validate_dsa_kpool_config(SimpleNamespace(**values))


def test_transformer_config_rejects_invalid_score_backend():
    with pytest.raises(ValueError, match="torch or triton"):
        TransformerConfig(
            num_layers=1,
            hidden_size=128,
            num_attention_heads=4,
            experimental_attention_variant="dsa",
            add_bias_linear=False,
            dsa_indexer_kpool_score_backend="auto",
        )


@pytest.mark.parametrize("use_relu", [False, True])
def test_reference_retains_original_arithmetic(use_relu):
    torch.manual_seed(17)
    scores = torch.randn(7, 2, 32, 11)
    weights = torch.randn(7, 2, 32)
    expected = scores.clone()
    if use_relu:
        expected.relu_()
    expected.mul_(weights.unsqueeze(-1))
    expected = expected.sum(2)
    actual = torch_kpool_score_postprocess(scores, weights, use_relu)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@GPU
@pytest.mark.parametrize("keys", [1, 7, 33, 257, 513])
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("use_relu", [False, True])
@pytest.mark.parametrize("batch_major", [False, True])
def test_fused_reference_parity_and_input_preservation(keys, batch, use_relu, batch_major):
    implementation = backends.select_kpool_score_postprocess("triton", 32)
    torch.manual_seed(23)
    scores = torch.randn(5, batch, 32, keys, device="cuda")
    if batch_major:
        scores = scores.transpose(0, 1).contiguous().transpose(0, 1)
    weights = torch.randn(5, batch, 64, device="cuda")[..., ::2]
    before_scores, before_weights = scores.clone(), weights.clone()
    expected = torch_kpool_score_postprocess(scores.clone(), weights, use_relu)
    actual = implementation(scores, weights, use_relu)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(scores, before_scores, rtol=0, atol=0)
    torch.testing.assert_close(weights, before_weights, rtol=0, atol=0)
    assert actual.dtype == torch.float32 and not actual.requires_grad
    assert actual.is_contiguous()


@GPU
@pytest.mark.parametrize("shape", [(0, 2, 32, 7), (3, 0, 32, 7), (3, 2, 32, 0)])
def test_empty_output_does_not_launch(shape):
    implementation = backends.select_kpool_score_postprocess("triton", 32)
    result = implementation(
        torch.empty(shape, device="cuda"), torch.empty(shape[:3], device="cuda"), True
    )
    assert result.shape == (shape[0], shape[1], shape[-1]) and result.numel() == 0


@GPU
def test_broadcast_weights_and_special_values():
    implementation = backends.select_kpool_score_postprocess("triton", 32)
    scores = (
        torch.tensor(
            [0.0, -0.0, float("inf"), -float("inf"), float("nan"), 1e-30, 1e30, -1e30],
            device="cuda",
        )
        .reshape(1, 1, 1, 8)
        .expand(3, 2, 32, 8)
        .clone()
    )
    weights = torch.ones(1, 1, 32, device="cuda").expand(3, 2, 32)
    for use_relu in (False, True):
        actual = implementation(scores, weights, use_relu)
        expected = torch_kpool_score_postprocess(scores.clone(), weights, use_relu)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)


@GPU
@pytest.mark.parametrize("invalid", ["dtype", "shape", "heads", "key_stride", "overlap", "grad"])
def test_fused_rejects_unsupported_inputs(invalid):
    implementation = backends.select_kpool_score_postprocess("triton", 32)
    scores = torch.ones(3, 2, 32, 7, device="cuda")
    weights = torch.ones(3, 2, 32, device="cuda")
    if invalid == "dtype":
        scores = scores.bfloat16()
    elif invalid == "shape":
        weights = weights[:, :1]
    elif invalid == "heads":
        scores, weights = scores[:, :, :16], weights[:, :, :16]
    elif invalid == "key_stride":
        scores = torch.ones(3, 2, 32, 14, device="cuda")[..., ::2]
    elif invalid == "overlap":
        scores = scores[:1].expand(3, 2, 32, 7)
    else:
        weights.requires_grad_(True)
    with pytest.raises(ValueError):
        implementation(scores, weights, True)


@GPU
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("use_relu", [False, True])
@pytest.mark.parametrize("tail", [False, True])
@pytest.mark.parametrize("quantized", [False, True])
def test_actual_kpool_selection_keeps_ordered_indices(
    packed, use_relu, tail, quantized, monkeypatch
):
    if quantized:
        pytest.importorskip("fast_hadamard_transform")
    implementation = backends.select_kpool_score_postprocess("triton", 32)
    torch.manual_seed(29)
    batch = 1 if packed else 2
    queries, keys, dim = 11, 37, 128
    q = torch.randn(queries, batch, 32, dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(keys, batch, dim, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(queries, batch, 32, device="cuda")
    positions = torch.arange(queries, device="cuda") * 3
    kwargs = dict(
        index_topk=8,
        pool_size=4,
        gate_score=torch.randn_like(k),
        ape=torch.randn(4, dim, device="cuda"),
        use_relu=use_relu,
        always_select_tail=tail,
        fp8_indexer=quantized,
        return_index_scores=False,
    )
    if packed:
        cu = torch.tensor([0, 13, 37], device="cuda", dtype=torch.int32)
        kwargs.update(
            cu_seqlens_kv=cu,
            varlen_starts=torch.where(positions < 13, 0, 13),
            varlen_ends=positions + 1,
            key_positions=torch.arange(keys, device="cuda"),
        )
    else:
        mask = torch.full((batch, queries, keys), -float("inf"), device="cuda")
        for i, end in enumerate(positions.tolist()):
            mask[:, i, : end + 1] = 0.0
        mask[1, :, 2] = -float("inf")
        kwargs["mask"] = mask
    monkeypatch.setattr(dsa, "_KPOOL_SCORE_CHUNK_BYTES", batch * 32 * 9 * 4 * 3)
    with torch.no_grad():
        ref_scores, expected = dsa.fused_qk_topk_kpool(q, k, weights, **kwargs)
        scores, actual = dsa.fused_qk_topk_kpool(
            q, k, weights, score_postprocess=implementation, **kwargs
        )
    assert ref_scores is None and scores is None
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@GPU
@pytest.mark.parametrize("near_ties", [False, True])
def test_ordered_topk_with_ties(near_ties):
    implementation = backends.select_kpool_score_postprocess("triton", 32)
    scores = torch.ones(7, 2, 32, 33, device="cuda")
    if near_ties:
        scores[0, 0, 0, 1] = 1.0 + 2**-18
    weights = torch.ones(7, 2, 32, device="cuda")
    expected = torch_kpool_score_postprocess(scores.clone(), weights, True)
    actual = implementation(scores, weights, True)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(actual.topk(8, -1).indices, expected.topk(8, -1).indices)


@GPU
def test_complete_score_path_remains_unfused():
    def unexpected(*args, **kwargs):
        raise AssertionError("complete-score path must not call selection-only postprocessing")

    q = torch.ones(3, 1, 32, 8, device="cuda")
    k = torch.ones(16, 1, 8, device="cuda")
    weights = torch.ones(3, 1, 32, device="cuda")
    scores, _ = dsa.fused_qk_topk_kpool(
        q,
        k,
        weights,
        8,
        4,
        torch.zeros_like(k),
        torch.zeros(4, 8, device="cuda"),
        return_index_scores=True,
        score_postprocess=unexpected,
    )
    assert scores is not None
