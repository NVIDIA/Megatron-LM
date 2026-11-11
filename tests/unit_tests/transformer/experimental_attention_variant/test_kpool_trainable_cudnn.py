# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU tests of the trainable KPool indexer loss on cuDNN frontend kernels (phase 2).

With TP size 1, no deterministic mode, ``dsa_kernel_backend='cudnn'``, the cuDNN KPool
scorer, HF scoring, an accepted forward and cuDNN frontend >= 1.31, a trainable KPool layer
computes its sparse pool-level KL after the sparse attention: the teacher from FE score
recompute fed the attention's own LSE, the student from the cuDNN selector's pool scores, the
gradients from the FE sparse indexer backward on the pooled keys. Everywhere else the PyTorch
``KPoolSparseIndexerLoss`` (phase 1) runs. Stand-in frontends
(``dsa_cudnn_fe_fakes.Fe131KPoolLossDSA``) model both kernels in plain PyTorch, and a PyTorch
stand-in of the fused sparse attention returns its LSE. Checked here:

- ``run_kpool_pool_topk(return_scores=True)``: the same selection, and each selected pool's
  student score (the HF factor ``(heads * head_dim)**-0.5`` applied only under HF scoring);
  the dispatch hook calls a hook without ``return_scores`` as before;
- ``run_kpool_attention_token_target`` with ``_kpool_pool_target_from_token_target``: the
  oracle teacher, from the selector's own layout padded to 128 without ``topk_length``;
- ``KPoolSparseIndexerLossFromScoresFunc``: loss and gradients of the phase-1 Function, for
  both scorings, every row-count convention, deterministic accumulation and the chunked
  backward; rows and slots outside the loss get exactly zero gradient;
- the cuDNN frontend >= 1.31 gate and the other decline reasons;
- a real ``DSAttention`` (PyTorch stand-ins for Transformer Engine): the dispatch matrix of
  scorer x scoring x layout x TP against the oracle (``kpool_trainable_reference.py``), with
  the one-time log line naming the cuDNN path only where it runs; the frontend gate at the
  layer; and the cuDNN path against the phase-1 path on one layer.
"""

import logging
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.enums import AttnBackend, AttnMaskType
from megatron.core.transformer.experimental_attention_variant import dsa as dsa_module
from megatron.core.transformer.experimental_attention_variant import dsa_cudnn_kernels, dsa_kernels
from megatron.core.transformer.experimental_attention_variant.dsa import (
    DSAIndexerLossAutoScaler,
    DSAIndexerLossLoggingHelper,
)
from megatron.core.transformer.forward_sharing import forward_sharing_lifetime
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import MLATransformerConfig
from megatron.core.utils import init_method_normal
from tests.unit_tests.transformer.experimental_attention_variant import dsa_cudnn_fe_fakes as fakes
from tests.unit_tests.transformer.experimental_attention_variant import kpool_selection_utils as ks
from tests.unit_tests.transformer.experimental_attention_variant import (
    kpool_trainable_reference as ref,
)

POOL = 4
TOPK = 16  # four selected pools per row
SEQ = 32
ALL_ROWS = 1 << 40
_DSA_LOGGER = dsa_module.__name__
_CUDNN_PATH = "runs on the cuDNN frontend kernels"
_PYTORCH_PATH = "runs in PyTorch (KPoolSparseIndexerLoss)"


@pytest.fixture
def frontend(monkeypatch):
    """Install the stand-in frontend at a version; reset every once-per-reason log."""

    def install(version=(1, 31), namespace=fakes.Fe131KPoolLossDSA):
        namespace.reset()
        monkeypatch.setattr(dsa_cudnn_kernels, "_ensure_dsa_namespace", lambda: None)
        monkeypatch.setattr(dsa_cudnn_kernels, "_cudnn_dsa", namespace)
        monkeypatch.setattr(dsa_cudnn_kernels, "_cudnn_frontend_version", lambda: version)
        monkeypatch.setattr(dsa_cudnn_kernels, "_KPOOL_DECLINES_LOGGED", set())
        monkeypatch.setattr(dsa_module, "_KPOOL_SCORER_FALLBACK_REASONS_WARNED", set())
        monkeypatch.setattr(dsa_module, "_KPOOL_LOSS_PATHS_LOGGED", set())
        return namespace

    return install


def _generator(seed):
    return torch.Generator().manual_seed(seed)


def _randn(generator, *shape, scale=1.0, dtype=torch.float32):
    return (torch.randn(shape, generator=generator) * scale).to(dtype)


def _assert_similar(actual, expected, *, label, eps):
    """Cosine similarity and norm ratio within ``eps`` of 1 (an exactly zero reference: zero)."""
    assert actual is not None, f"{label}: no gradient"
    actual = actual.detach().double().flatten()
    expected = expected.detach().double().flatten()
    assert actual.shape == expected.shape, f"{label}: {actual.shape} != {expected.shape}"
    assert torch.isfinite(actual).all(), f"{label} has non-finite values"
    expected_norm = float(expected.norm())
    if expected_norm == 0:
        assert float(actual.abs().max()) == 0, f"{label}: expected zero, got {actual.abs().max()}"
        return
    cosine = float(F.cosine_similarity(actual, expected, dim=0))
    ratio = float(actual.norm()) / expected_norm
    assert cosine > 1 - eps, f"{label}: cosine {cosine:.8f}"
    assert abs(ratio - 1) < eps, f"{label}: norm ratio {ratio:.8f}"


def _selection(seed, *, batch, seq_len=SEQ, cu_seqlens=None):
    """An oracle selection in a selector's slot order (pads interleaved), and its tokens."""
    base = ref.pool_token_base(seq_len, POOL, cu_seqlens)
    pool_ids = ref.random_pool_selection(
        _generator(seed),
        seq_len=seq_len,
        batch=batch,
        pool_size=POOL,
        index_topk=TOPK,
        base=base,
        cu_seqlens=cu_seqlens,
        interleave_pads=True,
    )
    tokens = ref.expand_selection(pool_ids, base, POOL, TOPK, cu_seqlens).to(torch.int32)
    return SimpleNamespace(base=base, pool_ids=pool_ids, token_topk=tokens)


def _attention_lse(query, key, token_topk, softmax_scale):
    """FP32 ``[sq, b, np]`` LSE over each row's attended ids (``+inf`` for an empty row)."""
    batch = query.size(1)
    ids = token_topk.long()
    valid = ids >= 0
    keys = key[:, :, 0].permute(1, 0, 2).float()
    gathered = keys[torch.arange(batch).view(batch, 1, 1), ids.clamp_min(0)]
    logits = torch.einsum("sbhd,bswd->bshw", query.float(), gathered) * softmax_scale
    lse = torch.logsumexp(logits.masked_fill(~valid.unsqueeze(2), float("-inf")), dim=-1)
    lse = torch.where(torch.isfinite(lse), lse, torch.full_like(lse, float("inf")))
    return lse.permute(1, 0, 2).contiguous()


# ---------------------------------------------------------------------------------------
# The student: the cuDNN selector's pool scores
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("hf_scoring", [True, False], ids=["hf", "legacy"])
@pytest.mark.parametrize("seq, cp_size", [(64, 1), (128, 2)])
def test_selector_returns_the_student_score_of_each_selected_pool(
    frontend, hf_scoring, seq, cp_size
):
    """``return_scores`` keeps the selection and returns each selected pool's student score.

    Under HF scoring the kernel drops the uniform factor ``(heads * head_dim)**-0.5``; the
    returned scores carry it again and equal ``_kpool_selected_pool_scores``. Under legacy
    scoring the weights carry every scale and the kernel's scores are the student as they
    are. Unused slots hold ``-inf``.
    """
    frontend()
    heads, dim = 4, 16
    x = ks.kpool_inputs(
        seq=seq, cp_size=cp_size, cp_rank=0, batch=2, heads=heads, dim=dim, device="cpu"
    )
    k_pooled = dsa_module._kpool_compress_keys(
        x.k, x.gate, x.ape, POOL, kpool_key_dtype_compression=True
    )
    weights = x.weights_hf if hf_scoring else x.weights_scaled
    common = dict(
        q=x.q,
        k_pooled=k_pooled,
        weights=weights,
        pool_topk=TOPK // POOL,
        pool_size=POOL,
        query_runs=ks.query_runs(seq, cp_size, 0),
        hf_scoring=hf_scoring,
    )
    ids_only = dsa_cudnn_kernels.run_kpool_pool_topk(**common)
    assert isinstance(ids_only, torch.Tensor), "the frozen call must return the ids alone"
    pool_ids, pool_scores = dsa_cudnn_kernels.run_kpool_pool_topk(**common, return_scores=True)
    assert torch.equal(pool_ids, ids_only), "return_scores changed the selection"
    assert pool_scores.dtype == torch.float32 and pool_scores.shape == pool_ids.shape
    valid = pool_ids >= 0
    assert valid.any() and (~valid).any()
    assert torch.all(pool_scores[~valid] == float("-inf"))
    student = dsa_module._kpool_selected_pool_scores(
        x.q, weights, k_pooled, pool_ids, hf_scoring=hf_scoring
    )
    torch.testing.assert_close(pool_scores[valid], student[valid], rtol=1e-5, atol=1e-6)
    raw = pool_scores * ((heads * dim) ** 0.5 if hf_scoring else 1.0)
    if hf_scoring:
        assert not torch.allclose(raw[valid], student[valid]), "the HF factor was not applied"


def test_dispatch_hook_offers_return_scores_only_when_asked(monkeypatch):
    """The frozen call is unchanged; a hook without ``return_scores`` returns no scores."""
    seen = []
    ids = torch.zeros(1, 2, 3, dtype=torch.int64)

    def legacy_hook(q, k_pooled, weights, pool_topk, pool_size, query_runs, hf_scoring, use_relu):
        seen.append("legacy")
        return ids

    def scoring_hook(
        q,
        k_pooled,
        weights,
        pool_topk,
        pool_size,
        query_runs,
        hf_scoring,
        use_relu,
        return_scores=False,
    ):
        seen.append(return_scores)
        return (ids, torch.ones(ids.shape)) if return_scores else ids

    config = SimpleNamespace(dsa_kernel_backend="cudnn", deterministic_mode=False)
    args = (config, None, None, None, 3, POOL, ((0, 2, 0),), True)
    monkeypatch.setattr(dsa_kernels, "_resolve_fused_hook", lambda _config, _name: legacy_hook)
    assert dsa_kernels.run_kpool_pool_topk(*args) is ids
    assert dsa_kernels.run_kpool_pool_topk(*args, return_scores=True) == (ids, None)
    monkeypatch.setattr(dsa_kernels, "_resolve_fused_hook", lambda _config, _name: scoring_hook)
    assert dsa_kernels.run_kpool_pool_topk(*args) is ids
    result_ids, scores = dsa_kernels.run_kpool_pool_topk(*args, return_scores=True)
    assert result_ids is ids and torch.equal(scores, torch.ones(ids.shape))
    assert seen == ["legacy", "legacy", False, True]


# ---------------------------------------------------------------------------------------
# The teacher: FE score recompute on the selector's token ids
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("packed", [False, True], ids=["plain", "packed"])
def test_score_recompute_teacher_matches_the_oracle(frontend, packed):
    """Pools summed from the kernel's per-slot mass give the oracle teacher and ``p_E``.

    The ids keep the selector's layout (pool slots with interleaved pads, then the tail) and
    reach the kernel padded with ``-1`` to a multiple of 128, without ``topk_length``; the
    kernel normalizes by the local head count, which the pool normalization cancels.
    """
    dsa_ns = frontend()
    cu = (0, 3, 14, SEQ) if packed else None
    batch = 1 if packed else 2
    sel = _selection(5, batch=batch, cu_seqlens=cu)
    generator = _generator(6)
    query = _randn(generator, SEQ, batch, 4, 64, scale=0.5, dtype=torch.bfloat16)
    key = _randn(generator, SEQ, batch, 1, 64, scale=0.5, dtype=torch.bfloat16)
    scale = 0.3
    lse = _attention_lse(query, key, sel.token_topk, scale)

    token_target = dsa_cudnn_kernels.run_kpool_attention_token_target(
        query, key, lse, sel.token_topk, scale
    )
    assert token_target.shape == sel.token_topk.shape and token_target.dtype == torch.float32
    assert torch.all(token_target[sel.token_topk < 0] == 0)
    target, row_has_mass, tail_mass = dsa_module._kpool_pool_target_from_token_target(
        token_target, sel.pool_ids, index_topk=TOPK, pool_size=POOL
    )
    expected = ref.sparse_pool_target(
        query, key, sel.pool_ids, sel.base, POOL, softmax_scale=scale, cu_seqlens=cu
    )
    assert torch.equal(row_has_mass, expected.row_has_mass)
    assert (~row_has_mass).any() and row_has_mass.any()
    torch.testing.assert_close(target, expected.target.float(), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(tail_mass, expected.tail_mass.float(), rtol=1e-5, atol=1e-6)
    assert dsa_ns.score_calls == [dict(width=128, topk_length=None, heads=8)]


def test_score_recompute_leaves_the_attention_lse_untouched(frontend):
    """The kernel reads a private copy: the LSE the attention backward saved stays as it was."""
    frontend()
    sel = _selection(7, batch=1)
    generator = _generator(8)
    query = _randn(generator, SEQ, 1, 8, 64, dtype=torch.bfloat16)
    key = _randn(generator, SEQ, 1, 1, 64, dtype=torch.bfloat16)
    lse = _attention_lse(query, key, sel.token_topk, 0.1)
    before, version = lse.clone(), lse._version
    dsa_cudnn_kernels.run_kpool_attention_token_target(query, key, lse, sel.token_topk, 0.1)
    assert lse._version == version and torch.equal(lse, before)


# ---------------------------------------------------------------------------------------
# The loss and its gradients: KPoolSparseIndexerLossFromScoresFunc
# ---------------------------------------------------------------------------------------


def _loss_case(seed, *, hf_scoring, batch=2, heads=4, dim=16):
    """Indexer inputs, a selection, the selector's pool scores and the oracle teacher."""
    generator = _generator(seed)
    sel = _selection(seed + 1, batch=batch)
    q = _randn(generator, SEQ, batch, heads, dim, dtype=torch.bfloat16)
    k_pooled = _randn(generator, SEQ // POOL, batch, dim, dtype=torch.bfloat16)
    projection = _randn(generator, SEQ, batch, heads, dtype=torch.bfloat16)
    if hf_scoring:
        weights = projection.float() * heads**-0.5
    else:
        weights = projection * (heads**-0.5 * dim**-0.5)
    valid = sel.pool_ids >= 0
    scores = dsa_module._kpool_selected_pool_scores(
        q, weights, k_pooled, sel.pool_ids, hf_scoring=hf_scoring
    ).masked_fill(~valid, float("-inf"))
    query = _randn(generator, SEQ, batch, 4, 32, dtype=torch.bfloat16)
    key = _randn(generator, SEQ, batch, 1, 32, dtype=torch.bfloat16)
    teacher = ref.sparse_pool_target(query, key, sel.pool_ids, sel.base, POOL, softmax_scale=0.2)
    row_mask = teacher.row_has_mass.clone()
    row_mask[:, -3:] = False  # padded rows
    return SimpleNamespace(
        q=q,
        weights=weights,
        k_pooled=k_pooled,
        pool_ids=sel.pool_ids,
        scores=scores,
        target=teacher.target.float(),
        row_mask=row_mask,
    )


def _leaves(case):
    return [t.detach().clone().requires_grad_() for t in (case.q, case.weights, case.k_pooled)]


@pytest.mark.parametrize("chunked", [False, True], ids=["one-launch", "row-chunks"])
@pytest.mark.parametrize("deterministic", [False, True], ids=["atomics", "deterministic"])
@pytest.mark.parametrize("row_count_kind", ["rows", "valid-rows", "per-token"])
@pytest.mark.parametrize("hf_scoring", [True, False], ids=["hf", "legacy"])
def test_cudnn_loss_matches_the_pytorch_loss(
    frontend, monkeypatch, hf_scoring, row_count_kind, deterministic, chunked
):
    """Same student scores, teacher and rows: the phase-1 loss and, to BF16, its gradients.

    The FE backward gets the BF16 head weights rebuilt from the HF weights (``* heads**0.5``)
    with ``sm_scale = (heads * head_dim)**-0.5``, and its ``d_weights`` is scaled back; under
    legacy scoring it gets the weights as they are with ``sm_scale = 1``. Heads are padded to
    64 and the slots to a multiple of 128. ``row-chunks`` forces the backward's row chunking.
    """
    dsa_ns = frontend()
    if chunked:
        monkeypatch.setattr(dsa_cudnn_kernels, "_CUDA_GRID_Y_MAX", 7)
        monkeypatch.setattr(dsa_cudnn_kernels, "_SPARSE_INDEXER_BACKWARD_CHUNK_ROWS", 8)
    # The fake's dK equals the atomic-free one, so values cannot tell the two apart: count
    # the atomic-free replacement's calls instead.
    atomic_free_calls = []
    real_atomic_free = dsa_cudnn_kernels._deterministic_sparse_indexer_grads_wk

    def atomic_free(*args, **kwargs):
        atomic_free_calls.append(kwargs.get("sm_scale"))
        return real_atomic_free(*args, **kwargs)

    monkeypatch.setattr(dsa_cudnn_kernels, "_deterministic_sparse_indexer_grads_wk", atomic_free)
    case = _loss_case(11, hf_scoring=hf_scoring)
    batch = case.q.size(1)
    row_count = {
        "rows": float(batch * SEQ),
        "valid-rows": torch.tensor(float(batch * (SEQ - 3))),
        "per-token": 1.0,
    }[row_count_kind]
    coeff = 0.37
    expected_leaves, leaves = _leaves(case), _leaves(case)
    expected = dsa_module.KPoolSparseIndexerLoss.apply(
        *expected_leaves,
        case.pool_ids,
        case.target,
        case.row_mask,
        row_count,
        coeff,
        hf_scoring,
        True,
        ALL_ROWS,
    )
    loss = dsa_cudnn_kernels.run_kpool_sparse_indexer_loss(
        *leaves,
        case.pool_ids,
        case.scores,
        case.target,
        case.row_mask,
        row_count,
        coeff,
        hf_scoring,
        deterministic=deterministic,
    )
    assert loss.dtype == torch.float32 and loss.ndim == 0
    torch.testing.assert_close(loss, expected, rtol=1e-5, atol=1e-8)
    (expected * 1.7).backward()
    (loss * 1.7).backward()
    for name, leaf, reference in zip(("q", "weights", "k_pooled"), leaves, expected_leaves):
        assert leaf.grad.dtype == leaf.dtype
        _assert_similar(leaf.grad, reference.grad, label=f"d{name}", eps=2e-3)
    heads, dim = case.q.shape[2:]
    assert dsa_ns.backward_calls, "the FE indexer backward did not run"
    expected_scale = (heads * dim) ** -0.5 if hf_scoring else 1.0
    for call in dsa_ns.backward_calls:
        assert call["heads"] == 64 and call["topk"] == 128
        assert call["sm_scale"] == pytest.approx(expected_scale, rel=1e-12)
    # The pooled-key gradient comes from the atomic-free replacement exactly when asked, at
    # the FE backward's score scale.
    if deterministic:
        assert atomic_free_calls, "deterministic: the atomic-free dK replacement did not run"
        assert all(s == pytest.approx(expected_scale, rel=1e-12) for s in atomic_free_calls)
    else:
        assert atomic_free_calls == []
    # Rows outside the loss get exactly zero query and head-weight gradients.
    outside = ~case.row_mask
    assert torch.all(leaves[0].grad.permute(1, 0, 2, 3)[outside] == 0)
    assert torch.all(leaves[1].grad.permute(1, 0, 2)[outside] == 0)


@pytest.mark.parametrize("hf_scoring", [True, False], ids=["hf", "legacy"])
def test_cudnn_loss_keeps_the_gradient_of_pools_whose_probability_underflows(frontend, hf_scoring):
    """A peaked student puts selected pools below FP32 tiny, where the FE score-gradient
    stage drops the KL gradient; the loss floors those probabilities, so the gradients still
    match the phase-1 loss."""
    frontend()
    case = _loss_case(11, hf_scoring=hf_scoring)
    case.q = (case.q.float() * 400).to(torch.bfloat16)
    valid = case.pool_ids >= 0
    case.scores = dsa_module._kpool_selected_pool_scores(
        case.q, case.weights, case.k_pooled, case.pool_ids, hf_scoring=hf_scoring
    ).masked_fill(~valid, float("-inf"))
    counted = valid & case.row_mask.unsqueeze(-1)
    student = torch.softmax(case.scores, dim=-1)
    underflow = counted & (student < torch.finfo(torch.float32).tiny) & (case.target > 0)
    assert underflow.sum() >= 10, "no counted pool underflows: the check below is vacuous"
    row_count = float(case.q.size(1) * SEQ)
    expected_leaves, leaves = _leaves(case), _leaves(case)
    expected = dsa_module.KPoolSparseIndexerLoss.apply(
        *expected_leaves,
        case.pool_ids,
        case.target,
        case.row_mask,
        row_count,
        0.37,
        hf_scoring,
        True,
        ALL_ROWS,
    )
    loss = dsa_cudnn_kernels.run_kpool_sparse_indexer_loss(
        *leaves, case.pool_ids, case.scores, case.target, case.row_mask, row_count, 0.37, hf_scoring
    )
    torch.testing.assert_close(loss, expected, rtol=1e-5, atol=1e-6)
    expected.backward()
    loss.backward()
    for name, leaf, reference in zip(("q", "weights", "k_pooled"), leaves, expected_leaves):
        _assert_similar(leaf.grad, reference.grad, label=f"d{name}", eps=2e-3)


def test_cudnn_loss_gives_zero_gradient_to_pools_no_counted_row_selects(frontend):
    frontend()
    case = _loss_case(21, hf_scoring=True)
    counted = case.row_mask.unsqueeze(-1) & (case.pool_ids >= 0)
    selected = torch.zeros(SEQ // POOL, case.q.size(1), dtype=torch.bool)
    for b in range(case.q.size(1)):
        selected[case.pool_ids[b][counted[b]], b] = True
    assert (~selected).any(), "every pool is selected: the check below is vacuous"
    leaves = _leaves(case)
    loss = dsa_cudnn_kernels.run_kpool_sparse_indexer_loss(
        *leaves,
        case.pool_ids,
        case.scores,
        case.target,
        case.row_mask,
        float(case.q.size(1) * SEQ),
        0.5,
        True,
    )
    loss.backward()
    assert torch.all(leaves[2].grad[~selected] == 0)
    assert leaves[2].grad[selected].abs().sum() > 0


def test_cudnn_loss_without_rows_returns_zero_loss_and_zero_gradients(frontend):
    frontend()
    case = _loss_case(31, hf_scoring=True)
    leaves = _leaves(case)
    loss = dsa_cudnn_kernels.run_kpool_sparse_indexer_loss(
        *leaves,
        case.pool_ids,
        case.scores,
        case.target,
        torch.zeros_like(case.row_mask),
        1.0,
        0.5,
        True,
    )
    loss.backward()
    assert float(loss) == 0
    for leaf in leaves:
        assert torch.all(leaf.grad == 0)


# ---------------------------------------------------------------------------------------
# The gate: cuDNN frontend >= 1.31, and the fused attention whose LSE the teacher reads
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "version, words",
    [
        ((1, 21), "1.21 is older than 1.31"),
        ((1, 30), "1.30 is older than 1.31"),
        (None, "of unknown version"),
        ((1, 31), None),
        ((1, 32), None),
    ],
)
def test_decline_reason_requires_cudnn_frontend_1_31(frontend, version, words):
    frontend(version=version)
    query = torch.zeros(8, 1, 4, 512, dtype=torch.bfloat16)
    key = torch.zeros(8, 1, 1, 512, dtype=torch.bfloat16)
    reason = dsa_cudnn_kernels.kpool_sparse_indexer_loss_decline_reason(query, key, 512)
    if words is None:
        assert reason is None
    else:
        assert words in reason and "#1225" in reason


def test_decline_reason_names_a_layout_the_fused_attention_declines(frontend):
    frontend()
    query = torch.zeros(8, 1, 4, 192, dtype=torch.bfloat16)
    key = torch.zeros(8, 1, 1, 192, dtype=torch.bfloat16)
    reason = dsa_cudnn_kernels.kpool_sparse_indexer_loss_decline_reason(query, key, 128)
    assert "fused sparse attention" in reason and "FlashMLA needs 512" in reason


def test_dispatch_hooks_of_a_backend_without_the_cudnn_loss(monkeypatch):
    """No hook: the decline names the backend; the loss hooks raise instead of declining."""
    monkeypatch.setattr(dsa_kernels, "_resolve_fused_hook", lambda _config, _name: None)
    config = SimpleNamespace(dsa_kernel_backend="tilelang", deterministic_mode=False)
    reason = dsa_kernels.kpool_sparse_indexer_loss_decline_reason(config, None, None)
    assert "'tilelang' has no fused KPool indexer loss" in reason
    with pytest.raises(RuntimeError, match="run_kpool_attention_token_target"):
        dsa_kernels.run_kpool_attention_token_target(config, None, None, None, None, 1.0)
    with pytest.raises(RuntimeError, match="run_kpool_sparse_indexer_loss"):
        dsa_kernels.run_kpool_sparse_indexer_loss(
            config, None, None, None, None, None, None, None, 1.0, 1.0, True
        )


# ---------------------------------------------------------------------------------------
# A real DSAttention layer on CPU
# ---------------------------------------------------------------------------------------

_LAYER_SEQ = 64
_LAYER_TOPK = 16
_LATENT = 512  # FlashMLA's latent value width, so the fused attention accepts the layer
_COEFF = 0.1
_PACKED_CU = (0, 21, 42, _LAYER_SEQ)  # documents off the pool grid
_INDEXER_OUTPUTS = ("q", "k", "weights", "gate_score")
_GATE = "indexer.index_kpool_compress_gate"
_APE = "indexer.index_kpool_compress_ape"


class _FakeGroup:
    def __init__(self, size, rank=0):
        self._size, self._rank = size, rank

    def size(self):
        return self._size

    def rank(self):
        return self._rank


class _Linear(torch.nn.Linear):
    """Stands in for TELinear: parameters in ``params_dtype``, returns ``(output, bias)``."""

    def __init__(self, input_size, output_size, *, config, **_unused):
        super().__init__(input_size, output_size, bias=False, dtype=config.params_dtype)

    def forward(self, x):
        return super().forward(x), None


class _LayerNorm(torch.nn.LayerNorm):
    """Stands in for TENorm as the indexer's key norm."""

    def __init__(self, config, hidden_size, eps):
        super().__init__(hidden_size, eps=eps, dtype=config.params_dtype)


def _fused_sparse_attention(
    query,
    key,
    topk_indices,
    softmax_scale,
    v_channels,
    topk_length=None,
    all_topk_rows_nonempty=False,
    deterministic=False,
    return_lse=False,
):
    """PyTorch stand-in of the cuDNN backend's fused absorbed sparse attention (FlashMLA)."""
    assert topk_length is None
    batch = query.size(1)
    ids = topk_indices.long()
    valid = (ids >= 0) & (ids < key.size(0))
    keys = key[:, :, 0].permute(1, 0, 2)
    gathered = keys[torch.arange(batch).view(batch, 1, 1), ids.clamp(0, key.size(0) - 1)]
    logits = torch.einsum("sbhd,bswd->bshw", query.float(), gathered.float()) * softmax_scale
    logits = logits.masked_fill(~valid.unsqueeze(2), float("-inf"))
    lse = torch.logsumexp(logits, dim=-1, keepdim=True)
    probs = torch.where(torch.isfinite(lse), torch.exp(logits - lse), torch.zeros_like(logits))
    output = torch.einsum("bshw,bswv->sbhv", probs, gathered[..., :v_channels].float())
    output = output.to(query.dtype)
    if not return_lse:
        return output
    lse = lse.squeeze(-1).detach()
    lse = torch.where(torch.isfinite(lse), lse, torch.full_like(lse, float("inf")))
    return output, lse.permute(1, 0, 2).contiguous()


def _layer_config(*, scorer, hf_scoring, per_token=False):
    config = MLATransformerConfig(
        num_layers=2,
        hidden_size=64,
        num_attention_heads=4,
        num_query_groups=4,
        kv_channels=16,
        multi_latent_attention=True,
        experimental_attention_variant="dsa",
        q_lora_rank=16,
        kv_lora_rank=_LATENT,
        qk_head_dim=16,
        qk_pos_emb_head_dim=0,
        v_head_dim=16,
        rope_type="rope",
        rotary_base=10000,
        apply_rope_fusion=False,
        dsa_indexer_n_heads=4,
        dsa_indexer_head_dim=16,
        dsa_indexer_topk=_LAYER_TOPK,
        dsa_indexer_kpool=POOL,
        dsa_indexer_loss_coeff=_COEFF,
        dsa_indexer_use_sparse_loss=True,
        dsa_indexer_rotate_activation=False,
        dsa_indexer_scoring_relu=True,
        dsa_indexer_hf_scoring=hf_scoring,
        dsa_indexer_kpool_key_dtype_compression=True,
        dsa_indexer_weights_proj_output_dtype="bf16",
        dsa_kernel_backend="none",
        attention_backend=AttnBackend.unfused,
        calculate_per_token_loss=per_token,
        add_bias_linear=False,
        normalization="RMSNorm",
        bf16=True,
        params_dtype=torch.bfloat16,
        use_cpu_initialization=True,
        init_method=init_method_normal(0.02),
    )
    # Set after validation, which needs CUDA for the cuDNN backend; the stand-ins run it here.
    config.dsa_kernel_backend = "cudnn"
    config.attention_backend = AttnBackend.auto
    config.dsa_indexer_kpool_scorer_backend = scorer
    return config


def _build_layer(config, *, tp_size=1, seed=5):
    spec = ModuleSpec(
        module=dsa_module.DSAIndexer,
        submodules=dsa_module.DSAIndexerSubmodules(
            linear_wq_b=_Linear, linear_wk=_Linear, k_norm=_LayerNorm, linear_weights_proj=_Linear
        ),
    )
    attention = dsa_module.DSAttention(
        config=config,
        submodules=dsa_module.DSAttentionSubmodules(indexer=spec),
        layer_number=1,
        attn_mask_type=AttnMaskType.causal,
        attention_type="self",
        cp_comm_type="allgather",
        pg_collection=SimpleNamespace(tp=_FakeGroup(tp_size), cp=_FakeGroup(1)),
    )
    generator = _generator(seed)
    with torch.no_grad():
        for parameter in attention.parameters():
            if parameter.ndim >= 2:
                parameter.copy_(_randn(generator, *parameter.shape, scale=0.15))
    return attention.train()


def _layer_inputs(config, *, batch, seed):
    generator = _generator(seed)
    heads, rows = config.num_attention_heads, _LAYER_SEQ

    def randn(*shape, scale=1.0):
        return _randn(generator, *shape, scale=scale, dtype=torch.bfloat16)

    return {
        "query": randn(rows, batch, heads, _LATENT, scale=0.5),
        "key": randn(rows, batch, 1, _LATENT, scale=0.5),
        "x": randn(rows, batch, config.hidden_size, scale=2.0),
        "qr": randn(rows, batch, config.q_lora_rank, scale=2.0),
        "up_v_weight": randn(heads, config.v_head_dim, _LATENT, scale=0.05),
        "grad_output": randn(rows, batch, heads * config.v_head_dim),
    }


def _packed_seq_params(cu_seqlens):
    cu = torch.tensor(cu_seqlens, dtype=torch.int32)
    longest = max(end - start for start, end in zip(cu_seqlens[:-1], cu_seqlens[1:]))
    return PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu,
        cu_seqlens_kv=cu,
        cu_seqlens_q_padded=cu.clone(),
        cu_seqlens_kv_padded=cu.clone(),
        max_seqlen_q=longest,
        max_seqlen_kv=longest,
    )


@pytest.fixture
def cpu_layer(monkeypatch):
    """Record the loss, both loss Functions, the attention's top-k and the student scores."""
    # A lazy tracker on the host (its default storage is on the current CUDA device).
    monkeypatch.setattr(DSAIndexerLossLoggingHelper, "tracker", {"values": torch.zeros(2)})
    monkeypatch.setattr(DSAIndexerLossAutoScaler, "main_loss_backward_scale", torch.ones(()))
    monkeypatch.setattr(
        dsa_cudnn_kernels, "run_fused_absorbed_sparse_attention", _fused_sparse_attention
    )
    seen = SimpleNamespace(losses=[], pytorch=0, cudnn=0, attention=[], student=[])
    real_save = DSAIndexerLossLoggingHelper.save_loss_to_tracker

    def save_loss_to_tracker(*args, **kwargs):
        seen.losses.append(kwargs["loss"].detach().clone())
        return real_save(*args, **kwargs)

    monkeypatch.setattr(
        DSAIndexerLossLoggingHelper, "save_loss_to_tracker", staticmethod(save_loss_to_tracker)
    )
    for owner, name, counter in (
        (dsa_module.KPoolSparseIndexerLoss, "apply", "pytorch"),
        (dsa_cudnn_kernels.KPoolSparseIndexerLossFromScoresFunc, "apply", "cudnn"),
    ):
        real_apply = getattr(owner, name)

        def apply(*args, _real=real_apply, _counter=counter, **kwargs):
            setattr(seen, _counter, getattr(seen, _counter) + 1)
            return _real(*args, **kwargs)

        monkeypatch.setattr(owner, name, apply)
    real_attention = dsa_module._run_sparse_attention

    def run_sparse_attention(**kwargs):
        seen.attention.append(
            SimpleNamespace(
                topk=kwargs["topk_indices"].detach().clone(),
                return_lse=kwargs.get("return_lse", False),
            )
        )
        return real_attention(**kwargs)

    monkeypatch.setattr(dsa_module, "_run_sparse_attention", run_sparse_attention)
    real_loss_hook = dsa_cudnn_kernels.run_kpool_sparse_indexer_loss

    def loss_hook(**kwargs):
        seen.student.append(
            SimpleNamespace(
                scores=kwargs["pool_scores"].detach().clone(),
                pool_ids=kwargs["pool_ids"].detach().clone(),
            )
        )
        return real_loss_hook(**kwargs)

    monkeypatch.setattr(dsa_cudnn_kernels, "run_kpool_sparse_indexer_loss", loss_hook)
    return seen


def _fake_tp(monkeypatch):
    """A TP group of size 2 with this rank holding every head: no collective changes a value.

    The layer then broadcasts and all-reduces as at TP2; the broadcast returns rank 0's own
    selection and the all-reduce sums nothing, so its loss is the oracle's over every head.
    """
    monkeypatch.setattr(dsa_module, "get_pg_size", lambda g=None: g.size() if g is not None else 1)
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        dsa_module.DSAttention,
        "_broadcast_kpool_selection_from_tp_rank0",
        staticmethod(lambda topk_indices, pool_ids, tp_group: (topk_indices, pool_ids)),
    )


def _spy_indexer(attention):
    calls = []
    real = attention.indexer.forward_before_topk

    def keep(store, name, grad):
        store[name] = grad.detach().clone()

    def forward_before_topk(*args, **kwargs):
        result = real(*args, **kwargs)
        record = SimpleNamespace(outputs=tuple(result), grads={})
        for name, tensor in zip(_INDEXER_OUTPUTS, record.outputs):
            if isinstance(tensor, torch.Tensor) and tensor.requires_grad:
                tensor.register_hook(lambda grad, _name=name: keep(record.grads, _name, grad))
        calls.append(record)
        return result

    attention.indexer.forward_before_topk = forward_before_topk
    return calls


def _run_layer(attention, inputs, *, packed_seq_params=None, grad=True):
    attention.zero_grad(set_to_none=True)
    DSAIndexerLossLoggingHelper.clean_loss_in_tracker()
    leaves = {
        name: inputs[name].detach().clone().requires_grad_(grad)
        for name in ("query", "key", "x", "qr", "up_v_weight")
    }
    with forward_sharing_lifetime(packed_seq_params, None, attention.config):
        with nullcontext() if grad else torch.no_grad():
            output = attention(
                query=leaves["query"],
                key=leaves["key"],
                value=None,
                attention_mask=None,
                x=leaves["x"],
                qr=leaves["qr"],
                attn_mask_type=AttnMaskType.causal,
                packed_seq_params=packed_seq_params,
                up_v_weight=leaves["up_v_weight"],
            )
    if grad:
        output.backward(inputs["grad_output"].view_as(output))
    return SimpleNamespace(
        output=output.detach().clone(),
        input_grads={
            name: None if leaf.grad is None else leaf.grad.clone() for name, leaf in leaves.items()
        },
        param_grads={
            name: parameter.grad.detach().clone()
            for name, parameter in attention.named_parameters()
            if parameter.grad is not None
        },
    )


def _pool_ids_from_tokens(token_topk, base):
    """Pool ids ``[b, sq, G]`` of an attended token top-k in the KPool layout."""
    tokens = token_topk[..., :_LAYER_TOPK].long()
    batch, rows, _ = tokens.shape
    first = tokens.reshape(batch, rows, _LAYER_TOPK // POOL, POOL)[..., 0]
    lookup = torch.full((_LAYER_SEQ,), -1, dtype=torch.int64)
    lookup[base[base >= 0]] = torch.nonzero(base >= 0).flatten()
    return torch.where(first >= 0, lookup[first.clamp(0, _LAYER_SEQ - 1)], -1)


def _expected_pytorch_reason(*, scorer, hf_scoring, packed, tp_size):
    """Words of the reason the PyTorch loss runs, or None where the cuDNN path runs."""
    if tp_size > 1:
        return "tensor parallelism (TP size 2)"
    if scorer != "cudnn":
        return "dsa_indexer_kpool_scorer_backend is 'torch', not 'cudnn'"
    if not hf_scoring:
        return "legacy (non-HF) KPool scoring"
    if packed:
        return "the cuDNN KPool scorer declined this forward"
    return None


def _check_against_the_oracle(attention, inputs, run, call, seen, *, cu_seqlens, eps):
    """The logged loss and the indexer gradients against ``kpool_trainable_reference``."""
    config = attention.config
    (loss,) = seen.losses
    attended = seen.attention[-1].topk
    base = ref.pool_token_base(_LAYER_SEQ, POOL, cu_seqlens)
    pool_ids = _pool_ids_from_tokens(attended, base)
    q, k, weights, gate_score = call.outputs
    tensors = dict(
        q=q.detach(),
        weights=weights.detach(),
        k=k.detach(),
        gate_score=gate_score.detach(),
        ape=attention.indexer.index_kpool_compress_ape.detach(),
    )
    batch = q.size(1)
    common = dict(
        query=inputs["query"],
        key=inputs["key"],
        pool_ids=pool_ids,
        pool_size=POOL,
        softmax_scale=attention.softmax_scale,
        hf_scoring=config.dsa_indexer_hf_scoring,
        use_relu=True,
        profile="key_dtype",
        loss_coeff=_COEFF,
        row_count=1.0 if config.calculate_per_token_loss else float(batch * _LAYER_SEQ),
        cu_seqlens=None if cu_seqlens is None else list(cu_seqlens),
    )
    leaves = {
        name: (t.clone() if name in ("k", "gate_score") else t.double()).requires_grad_()
        for name, t in tensors.items()
    }
    oracle = ref.sparse_indexer_loss(**leaves, **common)
    torch.testing.assert_close(loss.double(), oracle.loss.detach().double(), rtol=1e-4, atol=1e-8)
    oracle.loss.backward()
    for name in ("q", "weights", "k", "gate_score"):
        _assert_similar(call.grads.get(name), leaves[name].grad, label=f"d{name}", eps=eps)
    expected_gate = torch.einsum(
        "sbd,sbh->dh", leaves["gate_score"].grad.double(), inputs["x"].double()
    )
    _assert_similar(run.param_grads[_GATE], expected_gate, label="dgate", eps=eps)
    _assert_similar(run.param_grads[_APE], leaves["ape"].grad, label="dape", eps=eps)
    return pool_ids


@pytest.mark.parametrize("tp_size", [1, 2], ids=["tp1", "tp2"])
@pytest.mark.parametrize("layout", ["plain", "packed"])
@pytest.mark.parametrize("hf_scoring", [True, False], ids=["hf", "legacy"])
@pytest.mark.parametrize("scorer", ["torch", "cudnn"])
def test_dispatch_matrix_matches_the_oracle_and_logs_the_path(
    frontend, cpu_layer, monkeypatch, caplog, scorer, hf_scoring, layout, tp_size
):
    """Every combination trains to the oracle; only TP1 + cuDNN scorer + HF + an accepted
    forward runs the cuDNN path, and the one-time log line says which path runs and why.

    The cuDNN path takes the selector's pool scores, which carry the HF factor, and asks the
    attention for its LSE; the PyTorch path asks for neither.
    """
    frontend()
    if tp_size > 1:
        _fake_tp(monkeypatch)
    packed = layout == "packed"
    cu_seqlens = _PACKED_CU if packed else None
    config = _layer_config(scorer=scorer, hf_scoring=hf_scoring)
    attention = _build_layer(config, tp_size=tp_size)
    calls = _spy_indexer(attention)
    inputs = _layer_inputs(config, batch=1 if packed else 2, seed=17)
    expected_reason = _expected_pytorch_reason(
        scorer=scorer, hf_scoring=hf_scoring, packed=packed, tp_size=tp_size
    )
    cudnn = expected_reason is None

    with caplog.at_level(logging.INFO, logger=_DSA_LOGGER):
        run = _run_layer(
            attention, inputs, packed_seq_params=_packed_seq_params(cu_seqlens) if packed else None
        )
    (call,) = calls
    assert (cpu_layer.cudnn, cpu_layer.pytorch) == ((1, 0) if cudnn else (0, 1))
    (attention_call,) = cpu_layer.attention
    assert attention_call.return_lse == cudnn
    messages = [r.getMessage() for r in caplog.records if r.name == _DSA_LOGGER]
    path_lines = [m for m in messages if "trainable KPool indexer loss runs" in m]
    assert len(path_lines) == 1, path_lines
    if cudnn:
        assert _CUDNN_PATH in path_lines[0]
    else:
        assert _PYTORCH_PATH in path_lines[0] and expected_reason in path_lines[0]
    pool_ids = _check_against_the_oracle(
        attention, inputs, run, call, cpu_layer, cu_seqlens=cu_seqlens, eps=2e-3 if cudnn else 1e-3
    )
    if cudnn:
        # The student is the selector's own score of each pool, with the HF factor: the
        # PyTorch student of the same inputs.
        (student,) = cpu_layer.student
        assert torch.equal(student.pool_ids, pool_ids)
        q, k, weights, gate_score = call.outputs
        k_pooled = dsa_module._kpool_compress_keys(
            k.detach(),
            gate_score.detach(),
            attention.indexer.index_kpool_compress_ape.detach(),
            POOL,
            kpool_key_dtype_compression=True,
        )
        expected = dsa_module._kpool_selected_pool_scores(
            q.detach(), weights.detach(), k_pooled, pool_ids, hf_scoring=True
        )
        valid = pool_ids >= 0
        torch.testing.assert_close(student.scores[valid], expected[valid], rtol=1e-5, atol=1e-6)
        assert torch.all(student.scores[~valid] == float("-inf"))
    else:
        assert cpu_layer.student == []

    # A second forward with the same reason logs nothing more.
    caplog.clear()
    cpu_layer.losses.clear()
    with caplog.at_level(logging.INFO, logger=_DSA_LOGGER):
        _run_layer(
            attention,
            inputs,
            packed_seq_params=_packed_seq_params(cu_seqlens) if packed else None,
            grad=True,
        )
    assert not [r for r in caplog.records if "indexer loss runs" in r.getMessage()]


@pytest.mark.parametrize("version", [(1, 21), (1, 30)])
def test_frontend_older_than_1_31_runs_the_pytorch_loss(frontend, cpu_layer, caplog, version):
    """With the frontend version below 1.31 the layer that would run the cuDNN path runs
    phase 1 (its score recompute is wrong on SM100 before FE #1225) and says why."""
    frontend(version=version)
    config = _layer_config(scorer="cudnn", hf_scoring=True)
    attention = _build_layer(config)
    calls = _spy_indexer(attention)
    inputs = _layer_inputs(config, batch=2, seed=19)
    with caplog.at_level(logging.WARNING, logger=_DSA_LOGGER):
        run = _run_layer(attention, inputs)
    assert (cpu_layer.cudnn, cpu_layer.pytorch) == (0, 1)
    assert not cpu_layer.attention[-1].return_lse
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any(_PYTORCH_PATH in m and "older than 1.31" in m for m in warnings), warnings
    _check_against_the_oracle(
        attention, inputs, run, calls[0], cpu_layer, cu_seqlens=None, eps=1e-3
    )


@pytest.mark.parametrize("how", ["deterministic_mode", "torch_deterministic_algorithms"])
def test_deterministic_mode_runs_the_pytorch_loss(frontend, cpu_layer, caplog, how):
    """Deterministic DSA (either switch) keeps phase 1, whose pooled-key gradient is free of
    atomics at a cost set by the selected pools, and says why; it still trains to the oracle."""
    frontend()
    config = _layer_config(scorer="cudnn", hf_scoring=True)
    if how == "deterministic_mode":
        config.deterministic_mode = True
    attention = _build_layer(config)
    calls = _spy_indexer(attention)
    inputs = _layer_inputs(config, batch=2, seed=37)
    previous = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )
    if how == "torch_deterministic_algorithms":
        torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        with caplog.at_level(logging.WARNING, logger=_DSA_LOGGER):
            run = _run_layer(attention, inputs)
    finally:
        torch.use_deterministic_algorithms(previous[0], warn_only=previous[1])
    assert (cpu_layer.cudnn, cpu_layer.pytorch) == (0, 1)
    assert not cpu_layer.attention[-1].return_lse
    assert cpu_layer.student == []
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any(_PYTORCH_PATH in m and "deterministic mode" in m for m in warnings), warnings
    _check_against_the_oracle(
        attention, inputs, run, calls[0], cpu_layer, cu_seqlens=None, eps=1e-3
    )


@pytest.mark.parametrize("per_token", [False, True], ids=["mean", "per-token"])
def test_cudnn_path_matches_the_pytorch_path_on_one_layer(
    frontend, cpu_layer, monkeypatch, per_token
):
    """One layer and one input, the cuDNN path against the phase-1 path forced by the
    frontend gate: the same selection and attention output bitwise, the loss to FP32
    rounding and every indexer gradient (gate and slot bias included) to BF16."""
    config = _layer_config(scorer="cudnn", hf_scoring=True, per_token=per_token)
    attention = _build_layer(config)
    inputs = _layer_inputs(config, batch=2, seed=23)
    runs = {}
    for version, path in (((1, 31), "cudnn"), ((1, 21), "pytorch")):
        frontend(version=version)
        cpu_layer.losses.clear()
        cpu_layer.attention.clear()
        runs[path] = SimpleNamespace(run=_run_layer(attention, inputs))
        runs[path].loss = cpu_layer.losses[-1]
        runs[path].topk = cpu_layer.attention[-1].topk
    assert (cpu_layer.cudnn, cpu_layer.pytorch) == (1, 1)
    cudnn, pytorch = runs["cudnn"], runs["pytorch"]
    assert torch.equal(cudnn.topk, pytorch.topk)
    assert torch.equal(cudnn.run.output, pytorch.run.output)
    for name in ("query", "key", "up_v_weight"):
        assert torch.equal(cudnn.run.input_grads[name], pytorch.run.input_grads[name]), name
    assert cudnn.run.input_grads["x"] is None and cudnn.run.input_grads["qr"] is None
    torch.testing.assert_close(cudnn.loss, pytorch.loss, rtol=1e-5, atol=1e-8)
    assert set(cudnn.run.param_grads) == set(pytorch.run.param_grads)
    for name, grad in cudnn.run.param_grads.items():
        _assert_similar(grad, pytorch.run.param_grads[name], label=name, eps=2e-3)


def test_eval_and_no_grad_select_as_training_without_a_loss_or_log(
    frontend, cpu_layer, monkeypatch, caplog
):
    """Outside training the loss, its scores and its log line are skipped; the selection
    is the training forward's."""
    frontend()
    config = _layer_config(scorer="cudnn", hf_scoring=True)
    attention = _build_layer(config)
    inputs = _layer_inputs(config, batch=2, seed=29)
    _run_layer(attention, inputs)
    trained = cpu_layer.attention[-1].topk
    caplog.clear()
    monkeypatch.setattr(dsa_module, "_KPOOL_LOSS_PATHS_LOGGED", set())
    with caplog.at_level(logging.INFO, logger=_DSA_LOGGER):
        attention.eval()
        _run_layer(attention, inputs, grad=False)
        attention.train()
        _run_layer(attention, inputs, grad=False)
    assert [c.return_lse for c in cpu_layer.attention] == [True, False, False]
    assert len(cpu_layer.student) == 1 and cpu_layer.cudnn == 1
    for record in cpu_layer.attention[1:]:
        assert torch.equal(record.topk, trained)
    assert not [r for r in caplog.records if "indexer loss runs" in r.getMessage()]
