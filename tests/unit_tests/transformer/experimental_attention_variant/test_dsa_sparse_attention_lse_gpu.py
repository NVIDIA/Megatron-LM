# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""GPU tests of the sparse-attention LSE output and the indexer-backward ``sm_scale``.

On the real kernels (FlashMLA's sparse forward, the cuDNN frontend's sparse attention
backward and sparse indexer backward):

- the LSE that ``run_fused_absorbed_sparse_attention(return_lse=True)`` returns equals a
  PyTorch FP32 ``logsumexp`` over each row's attended keys, in the natural log, laid out
  ``[sq, b, np]``, with ``+inf`` on a row that attends no key;
- asking for the LSE changes nothing else: the output, the LSE and, with the deterministic
  backward, the query and key gradients are bitwise those of ``return_lse=False``;
- the sparse indexer backward with ``sm_scale`` passed equal to its default is bitwise the
  default call (deterministic mode), and half the scale halves dQ, dW and dK, as the
  gradient of ``sum_h w_h relu(s q_h . k)`` predicts.

Launch from the Megatron-LM root, e.g.::

    pytest -q \
        tests/unit_tests/transformer/experimental_attention_variant/test_dsa_sparse_attention_lse_gpu.py
"""

import os

import pytest
import torch

from megatron.core.transformer.experimental_attention_variant import dsa_cudnn_kernels

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")

# Absorbed MLA rows: 512 latent channels plus 64 RoPE channels. FlashMLA pads the 16 heads to
# its head block in forward (so the LSE it hands back is a column slice), and the deterministic
# backward takes 16 heads directly.
_ATT_HEADS, _ATT_DIM, _LATENT = 16, 576, 512
_ATT_ROWS, _ATT_BATCH, _ATT_KEYS, _ATT_WIDTH = 128, 2, 256, 64
_ATT_SCALE = _ATT_DIM**-0.5
# The indexer backward: 64 heads (the kernel's minimum) x 128, top-k one 128-slot block.
_IDX_HEADS, _IDX_DIM, _IDX_ROWS, _IDX_KEYS, _IDX_TOPK = 64, 128, 128, 256, 128
_TINY = torch.finfo(torch.float32).tiny


@pytest.fixture(autouse=True)
def cudnn_dsa_and_exact_reference():
    """Skip without SM90+ and the cuDNN DSA frontend; run references in true FP32."""
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("the fused DSA kernels need SM90+")
    try:
        dsa_cudnn_kernels._ensure_dsa_namespace()
    except ImportError as exc:
        pytest.skip(f"cuDNN frontend DSA unavailable: {exc}")
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(previous)


def _require_flash_mla():
    try:
        from flash_mla import flash_mla_sparse_fwd  # noqa: F401
    except ImportError:
        pytest.skip("flash_mla is not installed")


# ---------------------------------------------------------------------------------------
# Sparse attention LSE
# ---------------------------------------------------------------------------------------


def _attention_case(seed=20261111):
    """Causal rows at positions 128..255 picking up to 64 distinct earlier keys.

    Random ``-1`` slots sit between valid ids, and row 5 of batch 1 attends no key.
    """
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(seed)
    query = torch.randn(
        (_ATT_ROWS, _ATT_BATCH, _ATT_HEADS, _ATT_DIM), generator=generator, device=device
    ).bfloat16()
    key = torch.randn(
        (_ATT_KEYS, _ATT_BATCH, 1, _ATT_DIM), generator=generator, device=device
    ).bfloat16()
    positions = torch.arange(_ATT_KEYS - _ATT_ROWS, _ATT_KEYS, device=device)
    scores = torch.rand((_ATT_BATCH, _ATT_ROWS, _ATT_KEYS), generator=generator, device=device)
    visible = torch.arange(_ATT_KEYS, device=device).view(1, 1, -1) <= positions.view(1, -1, 1)
    scores = scores.masked_fill(~visible, -1.0)
    ids = scores.topk(_ATT_WIDTH, dim=-1).indices
    dropped = torch.rand(ids.shape, generator=generator, device=device) < 0.25
    ids = ids.masked_fill(dropped, -1)
    ids[1, 5] = -1
    grad_output = torch.randn(
        (_ATT_ROWS, _ATT_BATCH, _ATT_HEADS, _LATENT), generator=generator, device=device
    ).bfloat16()
    return query, key, ids.to(torch.int32).contiguous(), grad_output


def _reference_lse(query, key, ids):
    """FP32 natural-log LSE ``[sq, b, np]`` over each row's ids; ``+inf`` for empty rows."""
    sq, batch, heads, dim = query.shape
    valid = ids >= 0  # [b, sq, W]
    safe = ids.clamp_min(0).long()
    lse = torch.empty((sq, batch, heads), dtype=torch.float32, device=query.device)
    for b in range(batch):
        selected = key[:, b, 0].float().index_select(0, safe[b].reshape(-1))
        selected = selected.view(sq, -1, dim)  # [sq, W, d]
        logits = torch.einsum("shd,swd->shw", query[:, b].float(), selected) * _ATT_SCALE
        logits = logits.masked_fill(~valid[b].unsqueeze(1), float("-inf"))
        lse[:, b] = torch.logsumexp(logits, dim=-1)
    return lse.masked_fill(torch.isneginf(lse), float("inf"))


def _run_attention(query, key, ids, grad_output, *, deterministic, return_lse):
    query = query.detach().clone().requires_grad_(True)
    key = key.detach().clone().requires_grad_(True)
    result = dsa_cudnn_kernels.run_fused_absorbed_sparse_attention(
        query=query,
        key=key,
        topk_indices=ids,
        softmax_scale=_ATT_SCALE,
        v_channels=_LATENT,
        deterministic=deterministic,
        return_lse=return_lse,
    )
    assert result is not None, "the fused sparse attention declined the test case"
    output, lse = result if return_lse else (result, None)
    output.backward(grad_output)
    torch.cuda.synchronize()
    return output.detach(), lse, query.grad, key.grad


def test_returned_lse_matches_reference_logsumexp():
    _require_flash_mla()
    query, key, ids, grad_output = _attention_case()

    _output, lse, _dq, _dkv = _run_attention(
        query, key, ids, grad_output, deterministic=False, return_lse=True
    )
    reference = _reference_lse(query, key, ids)

    assert lse.dtype == torch.float32
    assert lse.shape == (_ATT_ROWS, _ATT_BATCH, _ATT_HEADS)
    assert not lse.requires_grad
    empty = torch.isinf(reference)
    assert empty.sum().item() == _ATT_HEADS  # row 5 of batch 1, every head
    assert torch.equal(torch.isposinf(lse), empty), "rows without a key must carry +inf"
    torch.testing.assert_close(lse[~empty], reference[~empty], rtol=1e-4, atol=1e-3)


@pytest.mark.parametrize("deterministic", [True, False], ids=["deterministic", "atomic"])
def test_return_lse_leaves_output_and_gradients_unchanged(deterministic):
    _require_flash_mla()
    query, key, ids, grad_output = _attention_case(seed=20261112)
    if deterministic and not dsa_cudnn_kernels._supports_deterministic_sparse_attention_backward(
        query
    ):
        pytest.skip("installed cuDNN DSA has no deterministic sparse-attention backward")

    default = _run_attention(
        query, key, ids, grad_output, deterministic=deterministic, return_lse=False
    )
    with_lse = _run_attention(
        query, key, ids, grad_output, deterministic=deterministic, return_lse=True
    )

    # The forward is deterministic in both modes.
    assert torch.equal(with_lse[0], default[0])
    if deterministic:
        assert torch.equal(with_lse[2], default[2])
        assert torch.equal(with_lse[3], default[3])
    else:
        # The atomic backward may reorder its sums between any two launches; compare to the
        # spread of two default launches.
        repeat = _run_attention(query, key, ids, grad_output, deterministic=False, return_lse=False)
        for index in (2, 3):
            spread = (repeat[index].float() - default[index].float()).abs().max()
            difference = (with_lse[index].float() - default[index].float()).abs().max()
            scale = default[index].float().abs().max()
            assert difference <= max(2 * spread, 1e-2 * scale), (index, difference, spread)
    # A second LSE run returns the same LSE bit for bit.
    again = _run_attention(
        query, key, ids, grad_output, deterministic=deterministic, return_lse=True
    )
    assert torch.equal(again[1], with_lse[1])


# ---------------------------------------------------------------------------------------
# Sparse indexer backward sm_scale
# ---------------------------------------------------------------------------------------


def _indexer_case(seed=20261113):
    """Fixed target and predict over 128 slots per row, ``-1`` slots included."""
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn(
        (1, _IDX_ROWS, _IDX_HEADS, _IDX_DIM), generator=generator, device=device
    ).bfloat16()
    weights = torch.randn((1, _IDX_ROWS, _IDX_HEADS), generator=generator, device=device)
    weights = (weights * _IDX_HEADS**-0.5).bfloat16()
    k = torch.randn((1, _IDX_KEYS, _IDX_DIM), generator=generator, device=device).bfloat16()
    ids = torch.rand((_IDX_ROWS, _IDX_KEYS), generator=generator, device=device)
    ids = ids.topk(_IDX_TOPK, dim=-1).indices
    invalid = torch.rand(ids.shape, generator=generator, device=device) < 0.2
    ids = ids.masked_fill(invalid, -1).to(torch.int32)
    valid = ids >= 0
    logits = torch.randn(ids.shape, generator=generator, device=device)
    predict = torch.softmax(logits.masked_fill(~valid, float("-inf")), dim=-1)
    predict = predict.clamp_min(_TINY).masked_fill(~valid, 0.0)
    target = torch.rand(ids.shape, generator=generator, device=device).masked_fill(~valid, 0.0)
    target = target / target.sum(dim=-1, keepdim=True)
    return dict(
        q=q,
        weights=weights,
        k=k,
        ids=ids.unsqueeze(0).contiguous(),
        target=target.unsqueeze(0).contiguous(),
        predict=predict.unsqueeze(0).contiguous(),
    )


def _indexer_backward(case, *, deterministic, **kwargs):
    # The cuDNN score-gradient stage overwrites both score operands, so pass copies.
    grads = dsa_cudnn_kernels._run_sparse_indexer_backward(
        case["q"],
        case["weights"],
        case["k"],
        case["target"].clone(),
        case["predict"].clone(),
        case["ids"],
        loss_coeff=1.0,
        grad_loss=torch.ones((), dtype=torch.float32, device="cuda"),
        block_i=128,
        deterministic=deterministic,
        **kwargs,
    )
    torch.cuda.synchronize()
    return grads


def _reference_indexer_grads(case, sm_scale):
    """FP32 dQ, dW, dK for the fixed slot scores, as the cuDNN kernel defines them.

    The kernel's logit gradient is ``g - p * sum(g)`` with
    ``g = -max(t, tiny) * [p >= tiny] * loss_coeff * grad_loss / (B * S_q)``; the logits are
    ``I = sum_h w_h relu(s * q_h . k)``.
    """
    q = case["q"][0].float()
    w = case["weights"][0].float()
    k = case["k"][0].float()
    ids = case["ids"][0].long()
    target = case["target"][0]
    predict = case["predict"][0]
    valid = ids >= 0
    grad_scale = 1.0 / q.size(0)
    g = -target.clamp_min(_TINY) * (predict >= _TINY).float() * grad_scale
    dlogit = (g - predict * g.sum(-1, keepdim=True)).masked_fill(~valid, 0.0)
    selected = k.index_select(0, ids.clamp_min(0).reshape(-1)).view(q.size(0), -1, k.size(1))
    dots = torch.bmm(q, selected.transpose(1, 2)) * sm_scale  # [rows, H, topk]
    dw = (dots.clamp_min(0) * dlogit.unsqueeze(1)).sum(-1)
    coeff = (dots > 0).float() * w.unsqueeze(-1) * dlogit.unsqueeze(1) * sm_scale
    dq = torch.bmm(coeff, selected)
    dk_selected = torch.bmm(coeff.transpose(1, 2), q)  # [rows, topk, D]
    dk = torch.zeros_like(k)
    dk.index_add_(0, ids[valid], dk_selected[valid])
    return dq, dw, dk


def _relative_error(candidate, reference):
    candidate = candidate.float()
    return ((candidate - reference).norm() / reference.norm()).item()


@pytest.mark.parametrize("deterministic", [True, False], ids=["deterministic", "atomic"])
def test_indexer_backward_sm_scale_default_and_half(deterministic):
    case = _indexer_case()
    default_scale = dsa_cudnn_kernels._INDEXER_SOFTMAX_SCALE

    default = _indexer_backward(case, deterministic=deterministic)
    explicit = _indexer_backward(case, deterministic=deterministic, sm_scale=default_scale)
    half = _indexer_backward(case, deterministic=deterministic, sm_scale=0.5 * default_scale)

    names = ("d_index_q", "d_weights", "d_index_k")
    if deterministic:
        for name, grad_a, grad_b in zip(names, explicit, default):
            assert torch.equal(grad_a, grad_b), name
    else:
        # dK accumulates with atomics: bound by the spread of two default launches.
        repeat = _indexer_backward(case, deterministic=False)
        for name, grad_a, grad_b, grad_r in zip(names, explicit, default, repeat):
            spread = (grad_r.float() - grad_b.float()).abs().max()
            difference = (grad_a.float() - grad_b.float()).abs().max()
            assert difference <= max(2 * spread, 1e-2 * grad_b.float().abs().max()), name

    for name, grad_half, grad_default in zip(names, half, default):
        assert not torch.equal(grad_half, grad_default), name
        # Every gradient carries one factor of sm_scale for fixed slot scores.
        torch.testing.assert_close(
            grad_half.float(), 0.5 * grad_default.float(), rtol=2e-2, atol=1e-6, msg=name
        )
    for sm_scale, grads in ((default_scale, default), (0.5 * default_scale, half)):
        reference = _reference_indexer_grads(case, sm_scale)
        for name, grad, ref in zip(names, grads, reference):
            grad = grad[0]
            assert torch.isfinite(grad.float()).all(), name
            assert _relative_error(grad, ref) <= 2e-2, (name, sm_scale)
