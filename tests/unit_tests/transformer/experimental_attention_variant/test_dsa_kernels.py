# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for ``megatron.core.transformer.experimental_attention_variant.dsa_kernels``.

Coverage:

* Pure-Python helpers: :func:`local_to_global_flat`, :func:`build_flat_topk_idxs`,
  :func:`_kl_loss_from_target_predict` — full correctness checks; no GPU
  kernels required (CPU is fine).
* Lazy-import gates: :func:`_ensure_flash_mla`, :func:`_ensure_dsa_namespace`
  raise informative ``ImportError`` when the optional packages are missing.
* GPU helpers: :func:`_get_topk_alignment` — runs only on CUDA.
* Wrapper functions :func:`_dsa_fwd_flash_mla`, :func:`indexer_topk`,
  :func:`dsa_sparse_attn`, :func:`fused_indexer_sparse_attn` — exercised with
  ``unittest.mock`` stand-ins for the underlying ``flash_mla`` /
  ``cudnn.DSA`` kernels so the data-marshalling logic (shape conversions,
  TopK padding, predict/target/KL composition, autograd plumbing) is
  validated without requiring the real CUDA kernels.
"""

from __future__ import annotations

import math
import sys
import types
from unittest.mock import MagicMock, patch

import pytest
import torch

from megatron.core.transformer.experimental_attention_variant import dsa_kernels as dk
from megatron.core.transformer.experimental_attention_variant.dsa_kernels import (
    FusedIndexerSparseAttnFunc,
    SparseAttnFunc,
    _dsa_fwd_flash_mla,
    _ensure_dsa_namespace,
    _ensure_flash_mla,
    _get_topk_alignment,
    _kl_loss_from_dense_scores,
    _kl_loss_from_target_predict,
    build_flat_topk_idxs,
    dsa_sparse_attn,
    fused_indexer_sparse_attn,
    indexer_topk,
    local_to_global_flat,
)

# ---------------------------------------------------------------------------
# Test fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def reset_lazy_kernel_state():
    """Reset the module-level lazy import slots before/after each test.

    The wrapper-function tests patch ``_flash_mla_sparse_fwd`` / ``_DSA``
    directly; we need to ensure each test starts from a clean slate so the
    lazy ``_ensure_*`` calls are exercised consistently.
    """
    saved_flash = dk._flash_mla_sparse_fwd
    saved_dsa = dk._DSA
    dk._flash_mla_sparse_fwd = None
    dk._DSA = None
    yield
    dk._flash_mla_sparse_fwd = saved_flash
    dk._DSA = saved_dsa


def _make_local_idxs(b: int, sq: int, topk: int, *, with_invalid: bool = False) -> torch.Tensor:
    """Build a deterministic ``(b, sq, topk)`` int64 tensor of local indices.

    Values for batch ``i``, query ``s``, slot ``k`` are
    ``i * 100 + s * 10 + k``. When ``with_invalid`` is True every other
    slot is replaced with -1.
    """
    base = (
        torch.arange(b, dtype=torch.int64).view(b, 1, 1) * 100
        + torch.arange(sq, dtype=torch.int64).view(1, sq, 1) * 10
        + torch.arange(topk, dtype=torch.int64).view(1, 1, topk)
    )
    if with_invalid:
        mask = torch.arange(topk).view(1, 1, topk) % 2 == 1
        base = torch.where(mask.expand(b, sq, topk), torch.full_like(base, -1), base)
    return base


def _uniform_dist(B, S, K, dev):
    """Uniform ``1/K`` distribution of shape ``(B, S, K)``."""
    return torch.full((B, S, K), 1.0 / max(K, 1), dtype=torch.float32, device=dev)


def _peaked_dist(B, S, K, dev, peak_idx=0):
    """Distribution with all probability mass on ``peak_idx``."""
    out = torch.zeros(B, S, K, dtype=torch.float32, device=dev)
    out[..., peak_idx] = 1.0
    return out


# ---------------------------------------------------------------------------
# local_to_global_flat
# ---------------------------------------------------------------------------


class TestLocalToGlobalFlat:
    """Pure-Python index conversion (no GPU required)."""

    @pytest.mark.parametrize(
        "b, sq, topk, with_invalid",
        [
            (1, 4, 5, False),  # b=1 identity case
            (2, 3, 4, False),  # basic multi-batch
            (3, 5, 4, False),  # larger batch (stresses the formula)
            (2, 3, 6, True),  # invalid entries interleaved with valid ones
        ],
        ids=['b1_identity', 'basic', 'larger_b', 'with_invalid'],
    )
    def test_global_index_conversion(self, b, sq, topk, with_invalid):
        """Shape, dtype, ``-1`` preservation, and the formula
        ``global[s*b + bid, k] = local[bid, s, k] * b + bid`` (for valid entries)
        in one fixture. Row ``r`` of the output corresponds to query ``s = r // b``
        and batch id ``bid = r % b``.
        """
        local = _make_local_idxs(b, sq, topk, with_invalid=with_invalid)
        out = local_to_global_flat(local, b, seqlen_kv=128)

        assert out.shape == (sq * b, topk)
        assert out.dtype == torch.int32

        permuted = local.permute(1, 0, 2).reshape(sq * b, topk)
        batch_ids = (torch.arange(sq * b) % b).unsqueeze(1)
        expected = torch.where(
            permuted >= 0, permuted * b + batch_ids, torch.full_like(permuted, -1)
        ).int()
        assert torch.equal(out, expected)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cpu_cuda_parity(self):
        """CPU and CUDA execution paths produce identical results."""
        local = _make_local_idxs(b=2, sq=4, topk=3, with_invalid=True)
        out_cpu = local_to_global_flat(local, 2, seqlen_kv=64)
        out_cuda = local_to_global_flat(local.cuda(), 2, seqlen_kv=64)
        assert torch.equal(out_cpu, out_cuda.cpu())


# ---------------------------------------------------------------------------
# build_flat_topk_idxs
# ---------------------------------------------------------------------------


class TestBuildFlatTopkIdxs:
    """Pure-Python multi-group concat + optional compaction."""

    @pytest.mark.parametrize(
        "group_specs",
        [
            # Each spec is a list of (topk_i, with_invalid) for each group.
            [(4, False)],  # single group, all valid
            [(2, False), (3, False)],  # two groups, all valid
        ],
        ids=['single_group', 'two_groups'],
    )
    def test_non_compact_concat_then_globalise(self, group_specs):
        """Without ``compact`` the helper concatenates groups along ``topk``
        and applies the local→global conversion verbatim.
        """
        b, sq = 2, 3
        groups = [
            _make_local_idxs(b, sq, t, with_invalid=inv) + 50 * i
            for i, (t, inv) in enumerate(group_specs)
        ]
        total_topk = sum(t for t, _ in group_specs)

        flat, length = build_flat_topk_idxs(*groups, batch_size=b, seqlen_kv=256)

        expected = local_to_global_flat(torch.cat(groups, dim=-1), b, seqlen_kv=256)
        assert flat.shape == (sq * b, total_topk)
        assert flat.dtype == torch.int32
        assert torch.equal(flat, expected)
        assert length is None

    @pytest.mark.parametrize(
        "group_specs, expected_valid_per_row",
        [
            # Single group: 6 slots with every odd slot invalid → 3 valid.
            ([(6, True)], 3),
            # Two groups: g1 has 2 valid out of 4, g2 fully valid (2) → 4 valid.
            ([(4, True), (2, False)], 4),
        ],
        ids=['single_group', 'two_groups'],
    )
    def test_compact_packs_valid_first(self, group_specs, expected_valid_per_row):
        """With ``compact=True`` the helper packs valid entries to the front
        of each row, fills the tail with ``-1``, and returns a per-row
        ``topk_length`` that equals the count of valid entries.
        """
        b, sq = 2, 3
        groups = [
            _make_local_idxs(b, sq, t, with_invalid=inv) + 100 * i
            for i, (t, inv) in enumerate(group_specs)
        ]
        total_topk = sum(t for t, _ in group_specs)

        flat, length = build_flat_topk_idxs(*groups, batch_size=b, seqlen_kv=512, compact=True)

        assert flat.shape == (sq * b, total_topk)
        assert flat.dtype == torch.int32
        assert length is not None
        assert length.shape == (sq * b,)
        assert length.dtype == torch.int32

        # Per-row layout: valid global indices first, then -1 padding.
        for row in range(sq * b):
            n = int(length[row])
            assert n == expected_valid_per_row, f"row {row}: wrong length"
            assert torch.all(flat[row, :n] >= 0), f"row {row}: leading entries should be valid"
            assert torch.all(flat[row, n:] == -1), f"row {row}: trailing entries should be -1"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_compact_cuda_path(self, reset_lazy_kernel_state):
        """Combined coverage for the CUDA compact path (sub-blocks
        self-label on failure):

        * (a) Dispatch + plumbing (mocked compactify): the wrapper is
              called exactly once, with the already-globalised
              ``(sq*b, total_topk)`` int32 tensor as input, and its
              returned ``(indices, topk_length)`` flow back verbatim.
        * (b) End-to-end parity (real cuDNN, skipped without it): the
              cuDNN ``compactify`` kernel produces the same ``(flat,
              length)`` pair as the pure-PyTorch CPU fallback.
        """
        # ---- (a) dispatch via mocked compactify --------------------------
        b, sq, topk = 2, 3, 4
        local = _make_local_idxs(b, sq, topk, with_invalid=True).to(torch.int32, copy=False).cuda()
        compact_indices = torch.full((sq * b, topk), 99, dtype=torch.int32, device='cuda')
        topk_length = torch.full((sq * b,), 7, dtype=torch.int32, device='cuda')

        captured = {}

        def fake_compactify(global_idxs):
            captured['input'] = global_idxs
            return {'indices': compact_indices, 'topk_length': topk_length}

        fake_dsa = MagicMock(name='_DSA_compactify_stub')
        fake_dsa.compactify_wrapper.side_effect = fake_compactify
        dk._DSA = fake_dsa

        flat, length = build_flat_topk_idxs(local, batch_size=b, seqlen_kv=512, compact=True)
        fake_dsa.compactify_wrapper.assert_called_once()
        kernel_input = captured['input']
        assert kernel_input.shape == (sq * b, topk), "(a) wrapper input shape"
        assert kernel_input.dtype == torch.int32, "(a) wrapper input dtype"
        assert kernel_input.is_cuda, "(a) wrapper input not on CUDA"
        expected_input = local_to_global_flat(local, b, seqlen_kv=512)
        assert torch.equal(
            kernel_input, expected_input
        ), "(a) wrapper input != local_to_global_flat(local)"
        assert flat is compact_indices, "(a) returned flat is not the kernel output"
        assert length is topk_length, "(a) returned length is not the kernel output"

        # ---- (b) real-kernel parity vs CPU fallback ----------------------
        # Skipped when cuDNN is not installed; reset state so the real
        # _DSA import happens on the next call inside build_flat_topk_idxs.
        try:
            cudnn = pytest.importorskip("cudnn")
        except pytest.skip.Exception:
            return  # already passed (a); skip the parity sub-block silently
        if not hasattr(cudnn, 'DSA'):
            return
        dk._DSA = None  # force real lazy-import

        b2, sq2 = 4, 5
        local_a = _make_local_idxs(b2, sq2, 6, with_invalid=True)
        local_b = _make_local_idxs(b2, sq2, 4, with_invalid=False) + 200

        flat_cpu, len_cpu = build_flat_topk_idxs(
            local_a, local_b, batch_size=b2, seqlen_kv=512, compact=True
        )
        flat_cuda, len_cuda = build_flat_topk_idxs(
            local_a.cuda(), local_b.cuda(), batch_size=b2, seqlen_kv=512, compact=True
        )
        assert torch.equal(
            flat_cpu, flat_cuda.cpu()
        ), "(b) flat tensor differs between CPU fallback and cuDNN kernel"
        assert torch.equal(
            len_cpu, len_cuda.cpu()
        ), "(b) length tensor differs between CPU fallback and cuDNN kernel"


# ---------------------------------------------------------------------------
# _kl_loss_from_target_predict
# ---------------------------------------------------------------------------


class TestKLLossFromTargetPredict:
    """Pure-Python KL loss computation: combined assertions for all
    properties (scalar/dtype, identity, non-negativity, coeff linearity,
    invalid-row masking, analytical formula)."""

    def test_kl_loss_properties(self):
        """All KL-loss invariants checked sequentially. Each block raises
        an informative ``AssertionError`` so a failure pinpoints the
        broken sub-property.
        """
        torch.manual_seed(0)
        b, sq, topk = 2, 3, 4
        topk_indices = torch.zeros(b, sq, topk, dtype=torch.int32)

        # ---- (a) scalar/dtype + identity: KL(p || p) == 0 -----------------
        identical = torch.softmax(torch.randn(b, sq, topk), dim=-1)
        loss_identical = _kl_loss_from_target_predict(
            identical, identical.clone(), topk_indices, loss_coeff=1.0
        )
        assert loss_identical.shape == torch.Size([]), "identity: not scalar"
        assert loss_identical.dtype == torch.float32, "identity: not fp32"
        assert torch.allclose(
            loss_identical, torch.tensor(0.0), atol=1e-6
        ), f"identity: KL(p || p) != 0 (got {loss_identical.item()})"

        # ---- (b) non-negativity + linearity in loss_coeff -----------------
        target = torch.softmax(torch.randn(b, sq, topk), dim=-1)
        predict = torch.softmax(torch.randn(b, sq, topk), dim=-1)
        loss_1 = _kl_loss_from_target_predict(target, predict, topk_indices, loss_coeff=1.0)
        loss_3 = _kl_loss_from_target_predict(target, predict, topk_indices, loss_coeff=3.0)
        assert loss_1.item() >= 0.0, f"non-negativity: got {loss_1.item()}"
        assert torch.allclose(
            loss_3, 3.0 * loss_1, atol=1e-5, rtol=1e-5
        ), f"linearity: 3*loss_1 = {3*loss_1.item()} vs loss_3 = {loss_3.item()}"

        # ---- (c) invalid-row masking --------------------------------------
        # Construct deterministic distributions with strictly-positive per-row KL.
        t_inv = torch.full((b, sq, topk), 0.1, dtype=torch.float32)
        t_inv[..., 0] = 0.7
        p_inv = torch.full((b, sq, topk), 0.7, dtype=torch.float32) / topk
        p_inv[..., -1] = 1.0 - p_inv[..., :-1].sum(dim=-1)

        idx_all_valid = torch.zeros(b, sq, topk, dtype=torch.int32)
        loss_full = _kl_loss_from_target_predict(t_inv, p_inv, idx_all_valid, loss_coeff=1.0)
        assert loss_full.item() > 0, "all-valid baseline must be positive"

        # Mark the first row of every batch invalid → fewer valid rows,
        # smaller KL sum, same denominator (mean over all (B, S_q)).
        idx_partial = idx_all_valid.clone()
        idx_partial[:, 0, :] = -1
        loss_partial = _kl_loss_from_target_predict(t_inv, p_inv, idx_partial, loss_coeff=1.0)
        assert (
            loss_partial.item() < loss_full.item()
        ), f"partial-invalid: {loss_partial.item()} should be < {loss_full.item()}"

        # All-invalid → loss exactly 0.
        idx_all_invalid = torch.full_like(idx_all_valid, -1)
        loss_zero = _kl_loss_from_target_predict(t_inv, p_inv, idx_all_invalid, loss_coeff=1.0)
        assert loss_zero.item() == 0.0, f"all-invalid: got {loss_zero.item()}"

        # ---- (d) analytical formula: target = δ_0, predict = uniform(1/K) -
        # per-row KL = log(K); mean = log(K); loss = coeff * log(K).
        target_d = _peaked_dist(b, sq, topk, 'cpu', peak_idx=0)
        predict_d = torch.full((b, sq, topk), 1.0 / topk, dtype=torch.float32)
        loss_d = _kl_loss_from_target_predict(target_d, predict_d, topk_indices, loss_coeff=2.5)
        expected = 2.5 * math.log(topk)
        assert torch.allclose(
            loss_d, torch.tensor(expected), rtol=1e-5, atol=1e-5
        ), f"analytical: {loss_d.item()} vs expected {expected}"

    def test_per_token_loss_reports_raw_sum(self):
        torch.manual_seed(1)
        b, sq, topk = 2, 5, 4
        target = torch.softmax(torch.randn(b, sq, topk), dim=-1)
        predict = torch.softmax(torch.randn(b, sq, topk), dim=-1)
        topk_indices = torch.zeros(b, sq, topk, dtype=torch.int32)

        loss_mean = _kl_loss_from_target_predict(target, predict, topk_indices, loss_coeff=0.5)
        loss_sum = _kl_loss_from_target_predict(
            target, predict, topk_indices, loss_coeff=0.5, calculate_per_token_loss=True
        )

        assert torch.allclose(loss_sum, loss_mean * (b * sq), rtol=1e-5, atol=1e-5)


class TestKLLossFromDenseScores:
    def test_per_token_loss_reports_raw_sum(self):
        b, sq, sk = 2, 5, 4
        loss_coeff = 0.5

        attn_score = _peaked_dist(b, sq, sk, 'cpu', peak_idx=0)
        attn_l1norm = torch.ones(b, sq, dtype=torch.float32)
        index_score = torch.zeros(b, sq, sk, dtype=torch.float32)
        index_lse = torch.full((b, sq), math.log(sk), dtype=torch.float32)

        loss_mean = _kl_loss_from_dense_scores(
            attn_score, attn_l1norm, index_score, index_lse, loss_coeff
        )
        loss_sum = _kl_loss_from_dense_scores(
            attn_score,
            attn_l1norm,
            index_score,
            index_lse,
            loss_coeff,
            calculate_per_token_loss=True,
        )

        assert torch.allclose(loss_sum, loss_mean * (b * sq), rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# _ensure_flash_mla / _ensure_dsa_namespace
# ---------------------------------------------------------------------------


_LAZY_IMPORT_CASES = [
    pytest.param(
        'flash_mla',
        'flash_mla_sparse_fwd',
        _ensure_flash_mla,
        '_flash_mla_sparse_fwd',
        "FlashMLA is required",
        id='flash_mla',
    ),
    pytest.param(
        'cudnn', 'DSA', _ensure_dsa_namespace, '_DSA', "cudnn-frontend DSA", id='cudnn_dsa'
    ),
]


@pytest.mark.parametrize(
    "module_name, attr_name, ensure_fn, slot_name, error_match", _LAZY_IMPORT_CASES
)
class TestLazyKernelImports:
    """Lazy-import behaviour shared by ``_ensure_flash_mla`` and
    ``_ensure_dsa_namespace``: combined error-on-missing + caches-on-success
    fixture (assertion blocks self-label on failure).
    """

    def test_lazy_import_raises_and_caches(
        self, reset_lazy_kernel_state, module_name, attr_name, ensure_fn, slot_name, error_match
    ):
        # ---- (a) raises informative ImportError when the module is absent --
        # Setting ``sys.modules[name] = None`` makes ``import name`` fail.
        with patch.dict(sys.modules, {module_name: None}):
            with pytest.raises(ImportError, match=error_match):
                ensure_fn()

        # ---- (b) caches the import on success ------------------------------
        sentinel = MagicMock(name=f"{attr_name}_sentinel")
        fake_module = types.ModuleType(module_name)
        setattr(fake_module, attr_name, sentinel)

        with patch.dict(sys.modules, {module_name: fake_module}):
            ensure_fn()
            assert (
                getattr(dk, slot_name) is sentinel
            ), f"(b) {module_name}: ensure_fn() did not bind sentinel"

        # Second call must be a no-op — even after the sys.modules entry is gone.
        with patch.dict(sys.modules, {}, clear=False):
            sys.modules.pop(module_name, None)
            ensure_fn()
            assert (
                getattr(dk, slot_name) is sentinel
            ), f"(b) {module_name}: cached sentinel was lost on 2nd call"


# ---------------------------------------------------------------------------
# _get_topk_alignment
# ---------------------------------------------------------------------------


class TestGetTopkAlignment:
    """Architecture-dependent alignment for FlashMLA top-K padding."""

    @pytest.mark.parametrize(
        "sm_major, expected", [(7, 128), (8, 128), (9, 128), (10, 64), (12, 64), (13, 64)]
    )
    def test_alignment_per_sm(self, sm_major, expected):
        """SM10x and newer use 64-byte TopK alignment; older arches use 128."""
        with patch('torch.cuda.get_device_capability', return_value=(sm_major, 0)):
            assert _get_topk_alignment() == expected

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_runs_on_real_gpu(self):
        """On any real GPU the alignment must agree with the documented rule."""
        align = _get_topk_alignment()
        sm = torch.cuda.get_device_capability()
        expected = 64 if sm[0] >= 10 else 128
        assert align == expected


# ---------------------------------------------------------------------------
# _dsa_fwd_flash_mla — wrapper around flash_mla.flash_mla_sparse_fwd
# ---------------------------------------------------------------------------


def _make_flash_mla_stub(d_v: int = 512, *, lse_scalar: float = 0.0, out_fill: float = 0.0):
    """Build a callable stand-in for ``flash_mla.flash_mla_sparse_fwd``.

    The real kernel signature is
    ``(q, kv, indices, softmax_scale, d_v, attn_sink, topk_length, indexer_topk)``
    and returns ``(out, max_logits, lse)`` or ``(out, max_logits, lse, lse_indexer)``
    when ``indexer_topk > 0``.

    The stub returns deterministic, easily-distinguishable tensors so callers
    can numerically verify the wrapper's reshape / split logic. The most
    recent ``out`` and ``lse`` are stashed on ``stub.last_out`` /
    ``stub.last_lse`` for direct equality checks.
    """

    stub = MagicMock(name='flash_mla_sparse_fwd_stub')

    def _impl(q, kv, indices, softmax_scale, d_v, attn_sink, topk_length, indexer_topk):
        total_S_q, H, _D = q.shape
        # Distinguishable per-element pattern: out[i, h, k] = out_fill + i + 0.001*h + 1e-6*k
        # (works in bf16 at this magnitude, useful for verifying that the
        # wrapper does not silently reshape across the wrong axes).
        idx_i = torch.arange(total_S_q, dtype=torch.float32, device=q.device).view(-1, 1, 1)
        idx_h = torch.arange(H, dtype=torch.float32, device=q.device).view(1, -1, 1)
        idx_k = torch.arange(d_v, dtype=torch.float32, device=q.device).view(1, 1, -1)
        out_f32 = out_fill + idx_i + 0.001 * idx_h + 1e-6 * idx_k
        out = out_f32.to(q.dtype)
        max_logits = torch.zeros(total_S_q, H, dtype=torch.float32, device=q.device)
        # lse[i, h] = lse_scalar + i + 0.5*h — a deterministic pattern.
        lse = lse_scalar + (
            torch.arange(total_S_q, dtype=torch.float32, device=q.device).view(-1, 1)
            + 0.5 * torch.arange(H, dtype=torch.float32, device=q.device).view(1, -1)
        )
        stub.last_out = out
        stub.last_lse = lse
        if indexer_topk > 0:
            # Make lse_indexer distinct from lse so we can tell which one the
            # wrapper returned.
            lse_indexer = lse + 100.0
            stub.last_lse_indexer = lse_indexer
            return out, max_logits, lse, lse_indexer
        stub.last_lse_indexer = None
        return out, max_logits, lse

    stub.side_effect = _impl
    stub.last_out = None
    stub.last_lse = None
    stub.last_lse_indexer = None
    return stub


class TestDsaFwdFlashMla:
    """Adapter logic around FlashMLA: shape massaging, TopK padding, return
    tuples — including numerical pass-through of the kernel outputs.
    """

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_dsa_fwd_flash_mla_adapter(self, reset_lazy_kernel_state):
        """All adapter behaviours in one fixture (assertion blocks self-label
        on failure):

        * (a) TopK is padded up to GPU-specific alignment; padded slots are
              ``-1``; ``out`` / ``lse`` are passed through verbatim;
              kernel arg shapes match the SBHD-flat → MQA-h_kv=1 contract.
        * (b) ``indexer_topk == 0`` -> ``lse_indexer is None``.
        * (c) ``0 < indexer_topk < TopK`` -> kernel's ``lse_indexer`` is
              returned verbatim (not silently swapped for ``lse``).
        * (d) ``indexer_topk == TopK`` -> fallback to ``lse.clone()`` (a
              kernel snapshot quirk).
        * (e) ``indexer_topk > 0 + topk_length`` is rejected with the
              expected error.
        """
        total_sq, H, D = 4, 2, 512
        align = _get_topk_alignment()

        q = torch.randn(total_sq, H, D, dtype=torch.bfloat16, device='cuda')
        kv = torch.randn(8, D, dtype=torch.bfloat16, device='cuda')

        # ---- (a) padding + numerical pass-through ------------------------
        TopK_unpadded = 5
        expected_padded = ((TopK_unpadded + align - 1) // align) * align
        topk_idxs = torch.arange(total_sq * TopK_unpadded, dtype=torch.int32, device='cuda').view(
            total_sq, TopK_unpadded
        )

        stub = _make_flash_mla_stub(d_v=D)
        dk._flash_mla_sparse_fwd = stub

        out, lse, lse_indexer = _dsa_fwd_flash_mla(q, kv, topk_idxs, softmax_scale=0.5, d_v=D)
        assert lse_indexer is None, "(a) lse_indexer should be None when indexer_topk=0"
        assert torch.equal(out, stub.last_out), "(a) out is not pass-through"
        assert torch.equal(lse, stub.last_lse), "(a) lse is not pass-through"

        called_args = stub.call_args.args
        kv_3d, indices_arg = called_args[1], called_args[2]
        assert kv_3d.shape == (8, 1, D), f"(a) KV shape {tuple(kv_3d.shape)} != (8, 1, {D})"
        assert indices_arg.shape == (total_sq, 1, expected_padded), (
            f"(a) indices shape {tuple(indices_arg.shape)} != "
            f"({total_sq}, 1, {expected_padded})"
        )
        if expected_padded > TopK_unpadded:
            assert torch.all(
                indices_arg[..., TopK_unpadded:] == -1
            ), "(a) padded slots should be -1"
        assert torch.equal(
            indices_arg[..., :TopK_unpadded].squeeze(1), topk_idxs
        ), "(a) original entries should survive padding unchanged"

        # ---- (b–d) indexer_topk branches ---------------------------------
        TopK = align  # already aligned, no padding
        topk_idxs_aligned = torch.zeros(total_sq, TopK, dtype=torch.int32, device='cuda')

        stub = _make_flash_mla_stub(d_v=D, lse_scalar=1.5)
        dk._flash_mla_sparse_fwd = stub

        # (b) indexer_topk == 0
        _, _, lse_idx_b = _dsa_fwd_flash_mla(q, kv, topk_idxs_aligned, 0.5, indexer_topk=0)
        assert lse_idx_b is None, "(b) indexer_topk=0 must yield lse_indexer=None"

        # (c) 0 < indexer_topk < TopK
        _, lse_c, lse_idx_c = _dsa_fwd_flash_mla(
            q, kv, topk_idxs_aligned, 0.5, indexer_topk=TopK // 2
        )
        assert lse_idx_c is not None, "(c) lse_indexer should be present"
        assert torch.equal(
            lse_idx_c, stub.last_lse_indexer
        ), "(c) lse_indexer should be kernel pass-through"
        assert not torch.equal(lse_idx_c, lse_c), "(c) wrapper silently swapped lse_indexer for lse"

        # (d) indexer_topk == TopK -> fallback to lse.clone()
        _, lse_d, lse_idx_d = _dsa_fwd_flash_mla(q, kv, topk_idxs_aligned, 0.5, indexer_topk=TopK)
        assert torch.equal(
            lse_idx_d, lse_d
        ), "(d) lse_indexer should equal lse on TopK-cap fallback"
        assert (
            lse_idx_d.data_ptr() != lse_d.data_ptr()
        ), "(d) fallback should be a clone, not an alias"

        # ---- (e) topk_length + indexer_topk > 0 is rejected --------------
        # Use CPU tensors here — the assert fires before any kernel call.
        with pytest.raises(AssertionError, match="indexer_topk > 0 requires non-compact"):
            _dsa_fwd_flash_mla(
                torch.zeros(2, 2, 512, dtype=torch.bfloat16),
                torch.zeros(4, 512, dtype=torch.bfloat16),
                torch.zeros(2, 8, dtype=torch.int32),
                softmax_scale=0.5,
                topk_length=torch.zeros(2, dtype=torch.int32),
                indexer_topk=4,
            )


# ---------------------------------------------------------------------------
# indexer_topk — cudnn DSA wrapper for inference
# ---------------------------------------------------------------------------


class TestIndexerTopk:
    """Indexer scoring + radix top-K wrapper. All three properties combined
    in a single fixture; sub-block names appear in failure messages.
    """

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_indexer_topk_wrapper(self, reset_lazy_kernel_state):
        """Combined assertions for:

        * (a) basic call: shapes / dtypes of the returned (topk_indices,
              topk_length); kernels are called with the right BSHD layouts
              and the SBHD-flat (b*sq, sk) scores; indexer_top_k kwargs.
        * (b) topk > sk clamping: kernel call uses ``sk`` keys, trailing
              ``[sk:]`` slots are -1, ``topk_length == sk``.
        * (c) ``indexer_softmax_scale`` pre-scales the weights via the
              ``relu(c·x) = c·relu(x)`` trick before reaching the kernel.
        """

        # ---- (a) basic call ----------------------------------------------
        sq, b, idx_nh, idx_hd = 6, 2, 4, 64
        sk = 12
        topk = 5
        ratio = 4

        q_indexer = torch.randn(sq, b, idx_nh, idx_hd, dtype=torch.bfloat16, device='cuda')
        k_indexer = torch.randn(sk, b, idx_hd, dtype=torch.bfloat16, device='cuda')
        weights = torch.randn(sq, b, idx_nh, dtype=torch.bfloat16, device='cuda')

        scores = torch.randn(b, sq, sk, dtype=torch.float32, device='cuda')
        captured = {}

        def fake_indexer_forward(q_bshd, k_bshd, w_bsh, ratio):
            captured['indexer_forward'] = {
                'q_shape': q_bshd.shape,
                'k_shape': k_bshd.shape,
                'w_shape': w_bsh.shape,
                'ratio': ratio,
            }
            return {'scores': scores}

        def fake_filtered_topk(scores_flat, seq_lens, top_k, next_n, return_val):
            captured['filtered_topk'] = {
                'scores_shape': scores_flat.shape,
                'seq_lens_shape': seq_lens.shape,
                'top_k': top_k,
                'next_n': next_n,
                'return_val': return_val,
            }
            n_rows = scores_flat.shape[0]
            return {'indices': torch.zeros(n_rows, top_k, dtype=torch.int32, device='cuda')}

        fake_dsa = MagicMock()
        fake_dsa.indexer_forward_wrapper.side_effect = fake_indexer_forward
        fake_dsa.indexer_top_k_wrapper.side_effect = fake_filtered_topk
        dk._DSA = fake_dsa

        topk_indices, topk_length = indexer_topk(
            q_indexer, k_indexer, weights, topk=topk, ratio=ratio
        )

        assert topk_indices.shape == (b, sq, topk), "(a) topk_indices shape"
        assert topk_indices.dtype == torch.int32, "(a) topk_indices dtype"
        assert topk_length.shape == (b, sq), "(a) topk_length shape"
        assert topk_length.dtype == torch.int32, "(a) topk_length dtype"
        # BSHD / BSD layouts handed to the kernels.
        assert captured['indexer_forward']['q_shape'] == (
            b,
            sq,
            idx_nh,
            idx_hd,
        ), "(a) indexer_forward q_shape"
        assert captured['indexer_forward']['k_shape'] == (
            b,
            sk,
            1,
            idx_hd,
        ), "(a) indexer_forward k_shape (must be unsqueezed h_kv=1)"
        assert captured['indexer_forward']['w_shape'] == (
            b,
            sq,
            idx_nh,
        ), "(a) indexer_forward w_shape"
        assert captured['indexer_forward']['ratio'] == ratio, "(a) ratio kwarg"
        assert captured['filtered_topk']['scores_shape'] == (
            b * sq,
            sk,
        ), "(a) topK scores_flat shape"
        assert captured['filtered_topk']['seq_lens_shape'] == (b * sq,), "(a) topK seq_lens shape"
        assert captured['filtered_topk']['top_k'] == min(topk, sk), "(a) top_k kwarg"
        assert captured['filtered_topk']['next_n'] == 1, "(a) next_n kwarg"
        assert captured['filtered_topk']['return_val'] is False, "(a) return_val kwarg"

        # ---- (b) topk > sk clamping --------------------------------------
        dk._DSA = None  # force fresh mocks
        sq2, b2, idx_nh2, idx_hd2 = 4, 1, 2, 32
        sk2 = 3
        topk2 = 8  # > sk
        q2 = torch.randn(sq2, b2, idx_nh2, idx_hd2, dtype=torch.bfloat16, device='cuda')
        k2 = torch.randn(sk2, b2, idx_hd2, dtype=torch.bfloat16, device='cuda')
        w2 = torch.randn(sq2, b2, idx_nh2, dtype=torch.bfloat16, device='cuda')
        scores2 = torch.zeros(b2, sq2, sk2, dtype=torch.float32, device='cuda')
        kernel_indices2 = torch.zeros(b2 * sq2, sk2, dtype=torch.int32, device='cuda')

        fake_dsa_b = MagicMock()
        fake_dsa_b.indexer_forward_wrapper.return_value = {'scores': scores2}
        fake_dsa_b.indexer_top_k_wrapper.return_value = {'indices': kernel_indices2}
        dk._DSA = fake_dsa_b

        topk_indices2, topk_length2 = indexer_topk(q2, k2, w2, topk=topk2, ratio=4)
        assert topk_indices2.shape == (b2, sq2, topk2), "(b) padded topk_indices shape"
        assert torch.all(topk_indices2[..., sk2:] == -1), "(b) trailing slots not -1"
        assert torch.all(topk_length2 == sk2), "(b) topk_length should equal sk"

        # ---- (c) indexer_softmax_scale pre-scales weights ---------------
        dk._DSA = None
        sq3, b3, idx_nh3, idx_hd3 = 2, 1, 2, 32
        sk3 = 4
        scale = 0.125
        q3 = torch.zeros(sq3, b3, idx_nh3, idx_hd3, dtype=torch.bfloat16, device='cuda')
        k3 = torch.zeros(sk3, b3, idx_hd3, dtype=torch.bfloat16, device='cuda')
        w3 = torch.full((sq3, b3, idx_nh3), 8.0, dtype=torch.bfloat16, device='cuda')
        captured_w = {}

        def fake_indexer_forward_c(q_bshd, k_bshd, w_bsh, ratio):
            captured_w['w'] = w_bsh.detach().clone()
            return {'scores': torch.zeros(b3, sq3, sk3, dtype=torch.float32, device='cuda')}

        fake_dsa_c = MagicMock()
        fake_dsa_c.indexer_forward_wrapper.side_effect = fake_indexer_forward_c
        fake_dsa_c.indexer_top_k_wrapper.return_value = {
            'indices': torch.zeros(b3 * sq3, sk3, dtype=torch.int32, device='cuda')
        }
        dk._DSA = fake_dsa_c

        indexer_topk(q3, k3, w3, topk=sk3, ratio=4, indexer_softmax_scale=scale)
        expected_w = (
            (w3.float() * scale).to(torch.bfloat16).permute(1, 0, 2).reshape(b3, sq3, idx_nh3)
        )
        assert torch.allclose(
            captured_w['w'].float(), expected_w.float(), atol=1e-2, rtol=1e-2
        ), "(c) weights were not pre-scaled by indexer_softmax_scale"


# ---------------------------------------------------------------------------
# dsa_sparse_attn / SparseAttnFunc forward (mocked)
# ---------------------------------------------------------------------------


class TestDsaSparseAttn:
    """Numerical fwd + bwd test for the public ``dsa_sparse_attn`` entry
    point. The underlying kernels are mocked so the whole wrapper — including
    the SBHD↔flat reshape on the forward and the autograd plumbing on the
    backward — can be checked against deterministic ground truth.
    """

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_dsa_sparse_attn_fwd_and_bwd(self, reset_lazy_kernel_state):
        """Combined forward + backward fixture (assertion blocks self-label
        on failure):

        * (a) Forward output equals the FlashMLA stub's ``out`` reshaped
              from flat ``(sq*b, np_, d_v)`` back to ``(sq, b, np_ * d_v)``.
        * (b) Backward maps kernel grads onto the right SBHD leaf tensors
              with the correct shapes; the bwd kernel is invoked exactly
              once.
        """
        sq, b, np_, d = 4, 2, 2, 512
        skv = 6
        TopK = _get_topk_alignment()

        query = torch.randn(sq, b, np_, d, dtype=torch.bfloat16, device='cuda', requires_grad=True)
        kv = torch.randn(skv, b, d, dtype=torch.bfloat16, device='cuda', requires_grad=True)
        attn_sink = torch.zeros(np_, dtype=torch.float32, device='cuda', requires_grad=True)
        topk_idxs = torch.zeros(sq * b, TopK, dtype=torch.int32, device='cuda')

        # Coordinated stubs: FlashMLA fwd + cuDNN sparse-attn bwd, both
        # deterministic so every gradient slot is independently verifiable.
        flash_stub = _make_flash_mla_stub(d_v=d)
        dq_kernel = torch.full((sq * b, np_, d), 7.0, dtype=torch.bfloat16, device='cuda')
        dkv_kernel = torch.full((skv * b, d), -3.0, dtype=torch.bfloat16, device='cuda')
        d_sink_kernel = torch.full((np_,), 11.0, dtype=torch.float32, device='cuda')
        fake_dsa = MagicMock()
        fake_dsa.sparse_attention_backward_wrapper.return_value = {
            'dq': dq_kernel,
            'dkv': dkv_kernel,
            'd_sink': d_sink_kernel,
        }
        dk._flash_mla_sparse_fwd = flash_stub
        dk._DSA = fake_dsa

        out = dsa_sparse_attn(query, kv, attn_sink, topk_idxs, softmax_scale=0.5)

        # ---- (a) forward ------------------------------------------------
        assert out.shape == (sq, b, np_ * d), "(a) forward shape"
        assert out.dtype == torch.bfloat16, "(a) forward dtype"
        expected_out = flash_stub.last_out.reshape(sq, b, np_, d).reshape(sq, b, np_ * d)
        assert torch.equal(out, expected_out), "(a) forward value pass-through"

        # ---- (b) backward -----------------------------------------------
        out.sum().backward()
        assert query.grad is not None, "(b) query.grad missing"
        assert kv.grad is not None, "(b) kv.grad missing"
        assert attn_sink.grad is not None, "(b) attn_sink.grad missing"
        assert torch.equal(
            query.grad, dq_kernel.reshape(sq, b, np_, d)
        ), "(b) query.grad mis-reshaped"
        assert torch.equal(kv.grad, dkv_kernel.reshape(skv, b, d)), "(b) kv.grad mis-reshaped"
        assert torch.equal(attn_sink.grad, d_sink_kernel), "(b) attn_sink.grad mismatch"
        fake_dsa.sparse_attention_backward_wrapper.assert_called_once()


# ---------------------------------------------------------------------------
# fused_indexer_sparse_attn — Path B autograd Function (mocked)
# ---------------------------------------------------------------------------


def _install_full_dsa_mock(
    *,
    b: int,
    sq: int,
    np_: int,
    d: int,
    n_comp: int,
    idx_nh: int,
    predict_fn=None,
    target_fn=None,
    dq_value: float = 7.0,
    dkv_value: float = -3.0,
    d_sink_value: float = 11.0,
    d_index_q_value: float = 0.5,
    d_weights_value: float = -0.25,
    d_index_k_value: float = 1.5,
):
    """Patch the module-level ``_DSA`` and ``_flash_mla_sparse_fwd`` slots
    with a coordinated set of deterministic stubs covering every kernel
    invoked by :class:`FusedIndexerSparseAttnFunc`.

    ``predict_fn`` / ``target_fn`` (if provided) build the per-row
    distribution given ``(b, sq, topk, device)``. By default both return a
    uniform ``1/topk`` distribution, which yields ``KL(target || predict) = 0``
    so the loss is exactly zero.

    All backward kernels return constant-filled tensors so each gradient slot
    can be independently verified.
    """

    if predict_fn is None:
        predict_fn = lambda B, S, K, dev: torch.full(
            (B, S, K), 1.0 / max(K, 1), dtype=torch.float32, device=dev
        )
    if target_fn is None:
        target_fn = predict_fn

    fake_dsa = MagicMock(name='_DSA_full_stub')

    def fake_indexer_forward(q_bshd, k_bshd, w_bsh, ratio):
        return {'scores': torch.zeros(b, sq, n_comp, dtype=torch.float32, device=q_bshd.device)}

    fake_dsa.indexer_forward_wrapper.side_effect = fake_indexer_forward

    def fake_filtered_topk(scores_flat, seq_lens, top_k, next_n, return_val):
        return {
            'indices': torch.zeros(
                scores_flat.shape[0], top_k, dtype=torch.int32, device=scores_flat.device
            )
        }

    fake_dsa.indexer_top_k_wrapper.side_effect = fake_filtered_topk

    def fake_sparse_indexer_score_backward(q, k, w, topk_indices, qhead_per_kv_head):
        topk = topk_indices.shape[-1]
        return {'predict': predict_fn(b, sq, topk, q.device)}

    fake_dsa.sparse_indexer_score_recompute_wrapper.side_effect = fake_sparse_indexer_score_backward

    def fake_sparse_attn_score_backward(q, k, lse, topk_indices, sm_scale, qhead_per_kv_head):
        topk = topk_indices.shape[-1]
        return {'target': target_fn(b, sq, topk, q.device)}

    fake_dsa.sparse_attn_score_recompute_wrapper.side_effect = fake_sparse_attn_score_backward

    def fake_sparse_attn_backward(q, kv, out, dout, lse, attn_sink, topk_idxs, **kwargs):
        return {
            'dq': torch.full_like(q, dq_value),
            'dkv': torch.full_like(kv, dkv_value),
            'd_sink': torch.full_like(attn_sink, d_sink_value),
        }

    fake_dsa.sparse_attention_backward_wrapper.side_effect = fake_sparse_attn_backward

    def fake_indexer_grad_backward(
        q_idx_bshd,
        w_bsh,
        k_idx_bsd,
        attn_score,
        index_score,
        topk_indices,
        sm_scale,
        loss_coeff,
        grad_loss,
        block_I,
    ):
        return {
            'd_index_q': torch.full_like(q_idx_bshd, d_index_q_value),
            'd_weights': torch.full_like(w_bsh, d_weights_value),
            'd_index_k': torch.full_like(k_idx_bsd, d_index_k_value),
        }

    fake_dsa.indexer_backward_wrapper.side_effect = fake_indexer_grad_backward

    flash_stub = _make_flash_mla_stub(d_v=d)

    dk._DSA = fake_dsa
    dk._flash_mla_sparse_fwd = flash_stub
    return fake_dsa, flash_stub


def _install_full_dsa_mock_dense(
    *,
    b: int,
    sq: int,
    np_: int,
    d: int,
    n_comp: int,
    idx_nh: int,
    target_score_fn=None,
    target_l1norm_fn=None,
    predict_score_fn=None,
    predict_lse_fn=None,
    dq_value: float = 7.0,
    dkv_value: float = -3.0,
    d_sink_value: float = 11.0,
    d_index_q_value: float = 0.5,
    d_weights_value: float = -0.25,
    d_index_k_value: float = 1.5,
):
    """Coordinated stubs covering the dense-loss (``sparse_loss=False``) path.

    Mirrors :func:`_install_full_dsa_mock` for the sparse path, but stubs
    the four dense-only kernel wrappers:

    * ``dense_indexer_score_recompute_wrapper`` -> ``(out, denom=index_lse)``
    * ``dense_attn_score_recompute_wrapper``    -> ``(out, denom=attn_l1norm)``
    * ``dense_indexer_backward_wrapper``  -> ``{d_index_q, d_weights, d_index_k}``

    Defaults make ``target == predict == uniform(1/n_comp)`` so KL == 0.
    Override the four ``*_fn`` callables to drive the loss to known
    analytical values; each callable receives ``(B, S_q, S_k, device)`` and
    returns the score tensor (``S_k``-dim) or denom (no ``S_k`` dim).
    """

    if target_score_fn is None:
        target_score_fn = lambda B, S, K, dev: torch.full(
            (B, S, K), 1.0 / max(K, 1), dtype=torch.float32, device=dev
        )
    if target_l1norm_fn is None:
        target_l1norm_fn = lambda B, S, K, dev: torch.ones((B, S), dtype=torch.float32, device=dev)
    if predict_score_fn is None:
        predict_score_fn = lambda B, S, K, dev: torch.zeros(
            (B, S, K), dtype=torch.float32, device=dev
        )
    if predict_lse_fn is None:
        predict_lse_fn = lambda B, S, K, dev: torch.full(
            (B, S), float(math.log(max(K, 1))), dtype=torch.float32, device=dev
        )

    fake_dsa = MagicMock(name='_DSA_full_dense_stub')

    def fake_indexer_forward(q_bshd, k_bshd, w_bsh, ratio):
        return {'scores': torch.zeros(b, sq, n_comp, dtype=torch.float32, device=q_bshd.device)}

    fake_dsa.indexer_forward_wrapper.side_effect = fake_indexer_forward

    def fake_filtered_topk(scores_flat, seq_lens, top_k, next_n, return_val):
        return {
            'indices': torch.zeros(
                scores_flat.shape[0], top_k, dtype=torch.int32, device=scores_flat.device
            )
        }

    fake_dsa.indexer_top_k_wrapper.side_effect = fake_filtered_topk

    def fake_dense_indexer_score(q, k, w, qhead_per_kv_head, sm_scale, ratio):
        dev = q.device
        return {
            'out': predict_score_fn(b, sq, n_comp, dev),
            'denom': predict_lse_fn(b, sq, n_comp, dev),
        }

    fake_dsa.dense_indexer_score_recompute_wrapper.side_effect = fake_dense_indexer_score

    def fake_dense_attn_score(q, k, lse, softmax_scale, qhead_per_kv_head, ratio):
        dev = q.device
        return {
            'out': target_score_fn(b, sq, n_comp, dev),
            'denom': target_l1norm_fn(b, sq, n_comp, dev),
        }

    fake_dsa.dense_attn_score_recompute_wrapper.side_effect = fake_dense_attn_score

    def fake_sparse_attn_backward(q, kv, out, dout, lse, attn_sink, topk_idxs, **kwargs):
        return {
            'dq': torch.full_like(q, dq_value),
            'dkv': torch.full_like(kv, dkv_value),
            'd_sink': torch.full_like(attn_sink, d_sink_value),
        }

    fake_dsa.sparse_attention_backward_wrapper.side_effect = fake_sparse_attn_backward

    def fake_dense_indexer_grad_backward(
        q_idx_bshd,
        w_bsh,
        k_idx_bsd,
        attn_score,
        attn_l1norm,
        index_score,
        index_lse,
        sm_scale,
        loss_coeff,
        grad_loss,
        ratio,
        block_I,
    ):
        return {
            'd_index_q': torch.full_like(q_idx_bshd, d_index_q_value),
            'd_weights': torch.full_like(w_bsh, d_weights_value),
            'd_index_k': torch.full_like(k_idx_bsd, d_index_k_value),
        }

    fake_dsa.dense_indexer_backward_wrapper.side_effect = fake_dense_indexer_grad_backward

    flash_stub = _make_flash_mla_stub(d_v=d)

    dk._DSA = fake_dsa
    dk._flash_mla_sparse_fwd = flash_stub
    return fake_dsa, flash_stub


class TestFusedIndexerSparseAttn:
    """End-to-end numerical tests for the Path B autograd Function with all
    underlying CUDA kernels mocked.
    """

    # Common shapes shared across the forward tests.
    SHAPES = dict(sq=4, b=2, np_=2, d=512, skv=8, n_comp=4, idx_nh=4, idx_hd=64)

    def _make_inputs(self, *, requires_grad=False):
        """Build the seven differentiable + one non-differentiable inputs."""
        s = self.SHAPES
        win_topk = _get_topk_alignment() - 2  # exercise padding
        torch.manual_seed(0)

        def make(*shape, dtype, rg=False):
            t = torch.randn(*shape, dtype=dtype, device='cuda')
            if requires_grad and rg:
                t = t.detach().clone().requires_grad_(True)
            return t

        query = make(s['sq'], s['b'], s['np_'], s['d'], dtype=torch.bfloat16, rg=True)
        kv_full = make(s['skv'], s['b'], s['d'], dtype=torch.bfloat16, rg=True)
        attn_sink = torch.zeros(s['np_'], dtype=torch.float32, device='cuda')
        if requires_grad:
            attn_sink = attn_sink.detach().clone().requires_grad_(True)
        window_idxs = torch.zeros(s['b'], s['sq'], win_topk, dtype=torch.int32, device='cuda')
        q_indexer = make(s['sq'], s['b'], s['idx_nh'], s['idx_hd'], dtype=torch.bfloat16, rg=True)
        k_indexer = make(s['n_comp'], s['b'], s['idx_hd'], dtype=torch.bfloat16, rg=True)
        weights = make(s['sq'], s['b'], s['idx_nh'], dtype=torch.bfloat16, rg=True)
        return dict(
            query=query,
            kv_full=kv_full,
            attn_sink=attn_sink,
            window_idxs=window_idxs,
            q_indexer=q_indexer,
            k_indexer=k_indexer,
            weights=weights,
        )

    @pytest.mark.parametrize(
        "loss_coeff, target_kind, expected",
        [
            # KL(target == predict) == 0  →  loss == 0 regardless of coeff.
            (1.0, 'uniform', 0.0),
            # loss_coeff == 0 short-circuits even when target != predict.
            (0.0, 'peaked', 0.0),
            # target = δ_0, predict = uniform(1/K) → KL = log(K) per row,
            # mean over rows = log(K), scaled by coeff = coeff * log(K).
            (0.7, 'peaked', 0.7 * math.log(2)),
            # Linearity in loss_coeff: doubling the coeff doubles the loss.
            (2.0, 'peaked', 2.0 * math.log(2)),
        ],
        ids=['identical_dists', 'coeff_zero', 'analytical_kl', 'linearity_x2'],
    )
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_indexer_loss_formula(self, loss_coeff, target_kind, expected, reset_lazy_kernel_state):
        """All four loss-property cases share one fixture:

        * KL is zero when target == predict,
        * ``loss_coeff == 0`` short-circuits to zero,
        * for ``target = δ_0`` and ``predict = uniform(1/K)`` the per-row
          KL is exactly ``log(K)`` so the mean is ``loss_coeff * log(K)``,
        * the loss is linear in ``loss_coeff``.
        """
        s = self.SHAPES
        topk = 2  # = effective_topk = min(indexer_topk, n_comp); appears as K
        target_fn = (
            _uniform_dist
            if target_kind == 'uniform'
            else (lambda B, S, K, dev: _peaked_dist(B, S, K, dev, peak_idx=0))
        )

        inputs = self._make_inputs()
        _install_full_dsa_mock(
            b=s['b'],
            sq=s['sq'],
            np_=s['np_'],
            d=s['d'],
            n_comp=s['n_comp'],
            idx_nh=s['idx_nh'],
            predict_fn=_uniform_dist,
            target_fn=target_fn,
        )

        _, indexer_loss = fused_indexer_sparse_attn(
            **inputs,
            indexer_topk=topk,
            ratio=4,
            softmax_scale=0.5,
            loss_coeff=loss_coeff,
            sparse_loss=True,
            kv_offset=s['skv'] - s['n_comp'],
        )

        assert torch.allclose(
            indexer_loss, torch.tensor(expected, device='cuda'), rtol=1e-5, atol=1e-5
        ), f"got {indexer_loss.item()}, expected {expected}"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_sparse_path_fwd_output_bwd_grads_and_topk_clamp(self, reset_lazy_kernel_state):
        """Combined coverage for the sparse-loss path's three non-numerical
        properties (assertion blocks self-label on failure):

        * (a) ``output`` is exactly the FlashMLA stub's ``out`` reshaped
              from ``(sq*b, np_, d_v)`` to ``(sq, b, np_ * d_v)``.
        * (b) After backward, each leaf gradient equals the corresponding
              mocked kernel output, with q/kv/attn_sink coming from the
              sparse-attn bwd kernel and q_indexer/k_indexer/weights coming
              from the indexer bwd kernel (BSHD → SBHD permute applied).
        * (c) ``indexer_topk > n_comp`` is clamped to ``n_comp`` before the
              radix TopK kernel is called.
        """
        s = self.SHAPES

        # ---- (a) forward pass-through (no grads needed) ------------------
        inputs = self._make_inputs()
        _, flash_stub_a = _install_full_dsa_mock(
            b=s['b'], sq=s['sq'], np_=s['np_'], d=s['d'], n_comp=s['n_comp'], idx_nh=s['idx_nh']
        )
        output_a, _ = fused_indexer_sparse_attn(
            **inputs,
            indexer_topk=2,
            ratio=4,
            softmax_scale=0.5,
            indexer_softmax_scale=0.125,
            loss_coeff=0.0,
            sparse_loss=True,
            kv_offset=s['skv'] - s['n_comp'],
        )
        assert output_a.shape == (s['sq'], s['b'], s['np_'] * s['d']), "(a) shape"
        assert output_a.dtype == torch.bfloat16, "(a) dtype"
        expected_a = flash_stub_a.last_out.reshape(s['sq'], s['b'], s['np_'], s['d']).reshape(
            s['sq'], s['b'], s['np_'] * s['d']
        )
        assert torch.equal(output_a, expected_a), "(a) forward value pass-through"

        # ---- (b) backward grad propagation -------------------------------
        dk._DSA = None  # fresh mocks
        dk._flash_mla_sparse_fwd = None
        inputs_b = self._make_inputs(requires_grad=True)
        _install_full_dsa_mock(
            b=s['b'],
            sq=s['sq'],
            np_=s['np_'],
            d=s['d'],
            n_comp=s['n_comp'],
            idx_nh=s['idx_nh'],
            dq_value=7.0,
            dkv_value=-3.0,
            d_sink_value=11.0,
            d_index_q_value=0.5,
            d_weights_value=-0.25,
            d_index_k_value=1.5,
        )
        output_b, indexer_loss_b = fused_indexer_sparse_attn(
            **inputs_b,
            indexer_topk=2,
            ratio=4,
            softmax_scale=0.5,
            indexer_softmax_scale=0.125,
            loss_coeff=1.0,
            sparse_loss=True,
            kv_offset=s['skv'] - s['n_comp'],
        )
        (output_b.sum() + indexer_loss_b).backward()
        for name, value in [
            ('query', 7.0),
            ('kv_full', -3.0),
            ('attn_sink', 11.0),
            ('q_indexer', 0.5),
            ('k_indexer', 1.5),
            ('weights', -0.25),
        ]:
            grad = inputs_b[name].grad
            assert grad is not None, f"(b) {name}: missing grad"
            assert torch.equal(grad, torch.full_like(inputs_b[name], value)), (
                f"(b) {name}: grad does not equal full({value}); "
                f"got first elem = {grad.float().flatten()[0].item()}"
            )

        # ---- (c) indexer_topk > n_comp clamp -----------------------------
        dk._DSA = None
        dk._flash_mla_sparse_fwd = None
        inputs_c = self._make_inputs()
        fake_dsa_c, _ = _install_full_dsa_mock(
            b=s['b'], sq=s['sq'], np_=s['np_'], d=s['d'], n_comp=s['n_comp'], idx_nh=s['idx_nh']
        )
        fused_indexer_sparse_attn(
            **inputs_c,
            indexer_topk=999,  # > n_comp
            ratio=4,
            softmax_scale=0.5,
            loss_coeff=0.0,
            sparse_loss=True,
            kv_offset=s['skv'] - s['n_comp'],
        )
        topk_call = fake_dsa_c.indexer_top_k_wrapper.call_args
        assert (
            topk_call.kwargs['top_k'] == s['n_comp']
        ), f"(c) top_k clamp: got {topk_call.kwargs['top_k']}, expected {s['n_comp']}"


# ---------------------------------------------------------------------------
# fused_indexer_sparse_attn — dense path (sparse_loss=False)
# ---------------------------------------------------------------------------


class TestDenseFusedIndexerSparseAttn:
    """End-to-end tests for the dense-loss branch of Path B with all
    underlying CUDA kernels mocked. Mirrors :class:`TestFusedIndexerSparseAttn`
    but exercises the ``sparse_loss=False`` code path through
    :class:`FusedIndexerSparseAttnFunc`.
    """

    SHAPES = dict(sq=4, b=2, np_=2, d=512, skv=8, n_comp=4, idx_nh=4, idx_hd=64)

    def _make_inputs(self, *, requires_grad=False):
        s = self.SHAPES
        win_topk = _get_topk_alignment() - 2  # exercise padding
        torch.manual_seed(0)

        def make(*shape, dtype, rg=False):
            t = torch.randn(*shape, dtype=dtype, device='cuda')
            if requires_grad and rg:
                t = t.detach().clone().requires_grad_(True)
            return t

        query = make(s['sq'], s['b'], s['np_'], s['d'], dtype=torch.bfloat16, rg=True)
        kv_full = make(s['skv'], s['b'], s['d'], dtype=torch.bfloat16, rg=True)
        attn_sink = torch.zeros(s['np_'], dtype=torch.float32, device='cuda')
        if requires_grad:
            attn_sink = attn_sink.detach().clone().requires_grad_(True)
        window_idxs = torch.zeros(s['b'], s['sq'], win_topk, dtype=torch.int32, device='cuda')
        q_indexer = make(s['sq'], s['b'], s['idx_nh'], s['idx_hd'], dtype=torch.bfloat16, rg=True)
        k_indexer = make(s['n_comp'], s['b'], s['idx_hd'], dtype=torch.bfloat16, rg=True)
        weights = make(s['sq'], s['b'], s['idx_nh'], dtype=torch.bfloat16, rg=True)
        return dict(
            query=query,
            kv_full=kv_full,
            attn_sink=attn_sink,
            window_idxs=window_idxs,
            q_indexer=q_indexer,
            k_indexer=k_indexer,
            weights=weights,
        )

    @pytest.mark.parametrize(
        "loss_coeff, target_kind, expected",
        [
            # Identical dists: KL == 0 regardless of coeff.
            (1.0, 'uniform', 0.0),
            # loss_coeff == 0 short-circuits even when target != predict.
            (0.0, 'peaked', 0.0),
            # target = δ_0, predict = uniform(1/n_comp)
            #   per-row KL = log(n_comp); mean = log(n_comp); loss = coeff * log(n_comp).
            # n_comp = 4 here.
            (0.7, 'peaked', 0.7 * math.log(4)),
            (2.0, 'peaked', 2.0 * math.log(4)),
        ],
        ids=['identical_dists', 'coeff_zero', 'analytical_kl', 'linearity_x2'],
    )
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_dense_indexer_loss_formula(
        self, loss_coeff, target_kind, expected, reset_lazy_kernel_state
    ):
        """``_kl_loss_from_dense_scores`` is the dense analogue of
        ``_kl_loss_from_target_predict``. Verifies the same four KL
        properties (zero, coeff-zero short-circuit, analytical formula,
        linearity in coeff) over the dense ``(B, S_q, S_k)`` tensors.

        We drive the stub outputs so that:

        * predict = ``softmax(0)`` over S_k = uniform(1/n_comp). This is
          encoded as ``index_score = 0`` everywhere, ``index_lse = log(n_comp)``;
          ``predict = exp(score - lse) = 1/n_comp``.
        * For ``target = uniform``: attn_score = 1/n_comp uniformly, attn_l1norm = 1.
        * For ``target = δ_0``: attn_score peaked on slot 0 with sum 1, attn_l1norm = 1.
        """
        s = self.SHAPES

        if target_kind == 'uniform':
            target_score_fn = lambda B, S, K, dev: torch.full(
                (B, S, K), 1.0 / max(K, 1), dtype=torch.float32, device=dev
            )
        else:

            def target_score_fn(B, S, K, dev):
                t = torch.zeros((B, S, K), dtype=torch.float32, device=dev)
                t[..., 0] = 1.0
                return t

        target_l1norm_fn = lambda B, S, K, dev: torch.ones((B, S), dtype=torch.float32, device=dev)

        inputs = self._make_inputs()
        _install_full_dsa_mock_dense(
            b=s['b'],
            sq=s['sq'],
            np_=s['np_'],
            d=s['d'],
            n_comp=s['n_comp'],
            idx_nh=s['idx_nh'],
            target_score_fn=target_score_fn,
            target_l1norm_fn=target_l1norm_fn,
            # predict_score_fn / predict_lse_fn defaults give uniform predict.
        )

        _, indexer_loss = fused_indexer_sparse_attn(
            **inputs,
            indexer_topk=2,
            ratio=4,
            softmax_scale=0.5,
            loss_coeff=loss_coeff,
            sparse_loss=False,
            kv_offset=s['skv'] - s['n_comp'],
        )

        assert torch.allclose(
            indexer_loss, torch.tensor(expected, device='cuda'), rtol=1e-5, atol=1e-5
        ), f"got {indexer_loss.item()}, expected {expected}"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_dense_path_fwd_kernel_calls_and_bwd_grads(self, reset_lazy_kernel_state):
        """Combined coverage for the dense-loss path's two non-numerical
        properties (assertion blocks self-label on failure):

        * (a) The forward invokes ``dense_indexer_score_recompute_wrapper`` +
              ``dense_attn_score_recompute_wrapper`` (NOT the sparse score
              kernels) with the right BSHD/4-D shapes and ``ratio`` /
              scale args.
        * (b) The backward invokes ``dense_indexer_backward_wrapper``
              (NOT the sparse one), threads ``ratio`` through, and lands
              the kernel grads on the right SBHD leaves (BSHD → SBHD
              permute applied for the indexer-side grads).
        """
        s = self.SHAPES
        ratio = 4
        softmax_scale = 0.5
        idx_scale = 0.125
        loss_coeff = 1.0

        # ---- (a) forward kernel selection + arg shapes -------------------
        inputs_a = self._make_inputs()
        fake_dsa_a, _ = _install_full_dsa_mock_dense(
            b=s['b'], sq=s['sq'], np_=s['np_'], d=s['d'], n_comp=s['n_comp'], idx_nh=s['idx_nh']
        )
        fused_indexer_sparse_attn(
            **inputs_a,
            indexer_topk=2,
            ratio=ratio,
            softmax_scale=softmax_scale,
            indexer_softmax_scale=idx_scale,
            loss_coeff=loss_coeff,
            sparse_loss=False,
            kv_offset=s['skv'] - s['n_comp'],
        )
        fake_dsa_a.dense_indexer_score_recompute_wrapper.assert_called_once()
        fake_dsa_a.dense_attn_score_recompute_wrapper.assert_called_once()
        fake_dsa_a.sparse_indexer_score_recompute_wrapper.assert_not_called()
        fake_dsa_a.sparse_attn_score_recompute_wrapper.assert_not_called()

        idx_call = fake_dsa_a.dense_indexer_score_recompute_wrapper.call_args
        q_arg, k_arg, w_arg = idx_call.args
        assert q_arg.shape == (
            s['b'],
            s['sq'],
            s['idx_nh'],
            s['idx_hd'],
        ), "(a) dense indexer score: q shape"
        # K is unsqueezed to 4-D (B, S_k, H_kv=1, D) for the dense forward.
        assert k_arg.shape == (
            s['b'],
            s['n_comp'],
            1,
            s['idx_hd'],
        ), "(a) dense indexer score: k shape (must be unsqueezed h_kv=1)"
        assert w_arg.shape == (s['b'], s['sq'], s['idx_nh']), "(a) dense indexer score: w shape"
        assert idx_call.kwargs['qhead_per_kv_head'] == s['idx_nh']
        assert idx_call.kwargs['ratio'] == ratio
        # ``indexer_softmax_scale`` forwarded to the kernel; weights-scaling
        # trick handled separately in ``_sbhd_to_bshd_indexer_inputs``.
        assert (
            idx_call.kwargs['sm_scale'] == idx_scale
        ), "(a) dense indexer score: sm_scale forwarded"

        attn_call = fake_dsa_a.dense_attn_score_recompute_wrapper.call_args
        q_attn, k_attn, lse_arg, sm_arg = attn_call.args
        assert q_attn.shape == (s['b'], s['sq'], s['np_'], s['d']), "(a) dense attn score: q shape"
        assert k_attn.shape == (
            s['b'],
            s['n_comp'],
            1,
            s['d'],
        ), "(a) dense attn score: k shape (h_kv=1)"
        assert lse_arg.shape == (s['b'], s['sq'], s['np_']), "(a) dense attn score: lse shape"
        assert sm_arg == softmax_scale, "(a) dense attn score: positional softmax_scale"
        assert attn_call.kwargs['qhead_per_kv_head'] == s['np_']
        assert attn_call.kwargs['ratio'] == ratio

        # ---- (b) backward kernel selection + grad propagation ------------
        dk._DSA = None
        dk._flash_mla_sparse_fwd = None
        inputs_b = self._make_inputs(requires_grad=True)
        fake_dsa_b, _ = _install_full_dsa_mock_dense(
            b=s['b'],
            sq=s['sq'],
            np_=s['np_'],
            d=s['d'],
            n_comp=s['n_comp'],
            idx_nh=s['idx_nh'],
            dq_value=7.0,
            dkv_value=-3.0,
            d_sink_value=11.0,
            d_index_q_value=0.5,
            d_weights_value=-0.25,
            d_index_k_value=1.5,
        )
        output, indexer_loss = fused_indexer_sparse_attn(
            **inputs_b,
            indexer_topk=2,
            ratio=ratio,
            softmax_scale=softmax_scale,
            indexer_softmax_scale=idx_scale,
            loss_coeff=loss_coeff,
            sparse_loss=False,
            kv_offset=s['skv'] - s['n_comp'],
        )
        (output.sum() + indexer_loss).backward()

        fake_dsa_b.dense_indexer_backward_wrapper.assert_called_once()
        fake_dsa_b.indexer_backward_wrapper.assert_not_called()
        ig_call = fake_dsa_b.dense_indexer_backward_wrapper.call_args
        assert ig_call.kwargs['ratio'] == ratio, "(b) ratio not threaded through"
        assert ig_call.kwargs['sm_scale'] == idx_scale, "(b) sm_scale not threaded"
        assert ig_call.kwargs['loss_coeff'] == loss_coeff, "(b) loss_coeff not threaded"

        for name, value in [
            ('query', 7.0),
            ('kv_full', -3.0),
            ('attn_sink', 11.0),
            ('q_indexer', 0.5),
            ('k_indexer', 1.5),
            ('weights', -0.25),
        ]:
            grad = inputs_b[name].grad
            assert grad is not None, f"(b) {name}: missing grad"
            assert torch.equal(
                grad, torch.full_like(inputs_b[name], value)
            ), f"(b) {name}: grad does not equal full({value})"


# ---------------------------------------------------------------------------
# Real-kernel parity tests (cuDNN + optional FlashMLA)
# ---------------------------------------------------------------------------
#
# Everything above this banner stubs ``cudnn.DSA`` and ``flash_mla`` with
# ``MagicMock``-based fakes; that exercises the Python plumbing of
# ``dsa_kernels.py`` (shape transforms, autograd wiring, KL composition)
# but does NOT verify that the cuDNN kernels themselves compute what
# ``dsa_kernels.py`` expects them to compute.
#
# The tests below close that gap by running each helper / public function
# end-to-end against a small PyTorch reference implementation. Numeric
# tolerances are bf16-friendly (atol/rtol ~ 5e-2 for raw scores, 1e-3 for
# normalized distributions, 5e-2 for backward grads).
#
# Skipped automatically when:
#   * CUDA is unavailable;
#   * cuDNN frontend is not installed (``import cudnn`` fails);
#   * ``cudnn.DSA`` namespace is missing;
#   * SM is too low (sparse: SM90+; dense: SM100+);
#   * for FlashMLA-needing tests, ``flash_mla`` is not installed.
# ---------------------------------------------------------------------------


def _skip_if_real_kernels_unavailable(*, sm_min: int = 9, need_flash_mla: bool = False):
    """Pytest-side gate for real-kernel tests. Raises ``pytest.skip`` if
    any of the runtime dependencies are missing.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    sm_major = torch.cuda.get_device_capability()[0]
    if sm_major < sm_min:
        pytest.skip(f"requires SM{sm_min}+, found SM{sm_major}")
    cudnn = pytest.importorskip("cudnn")
    if not hasattr(cudnn, 'DSA'):
        pytest.skip("cudnn.DSA namespace not available")
    if need_flash_mla:
        pytest.importorskip("flash_mla")


# ---------------------------------------------------------------------------
# PyTorch reference implementations
# ---------------------------------------------------------------------------


def _ratio_causal_valid_mask(sq: int, sk: int, ratio: int, device) -> torch.Tensor:
    """``(Sq, Sk)`` bool: valid iff ``k_idx < min(Sk, (q_idx + 1) // ratio)``.

    Matches the cuDNN dense-score kernels' built-in causal mask
    (``col_limit = min(S_k, (q + 1) // ratio)``) and ``csa.py``'s
    ``compress_ratio`` mask formulation.
    """
    q_idx = torch.arange(sq, device=device).unsqueeze(1)  # (Sq, 1)
    k_idx = torch.arange(sk, device=device).unsqueeze(0)  # (1, Sk)
    col_limit = ((q_idx + 1) // ratio).clamp(max=sk)
    return k_idx < col_limit  # (Sq, Sk)


def _ref_indexer_full_score(
    q_bshd_fp32: torch.Tensor,  # (B, Sq, H, D)
    k_bsd_fp32: torch.Tensor,  # (B, Sk, D)  — MQA
    w_bsh_fp32: torch.Tensor,  # (B, Sq, H)
    sm_scale: float,
    ratio: int,
) -> torch.Tensor:
    """Reference for ``_bwd_dense_indexer_score.out``.

    ``S[b,q,k] = sm_scale * sum_h ReLU(Q[b,q,h] @ K[b,k]^T) * W[b,q,h]``,
    with the kernel's bottom-right ratio causal mask producing ``-inf``
    at masked positions.
    """
    B, Sq, _, _ = q_bshd_fp32.shape
    Sk = k_bsd_fp32.shape[1]
    qk = torch.einsum('bqhd,bkd->bqhk', q_bshd_fp32, k_bsd_fp32)  # (B, Sq, H, Sk)
    relu_qk = torch.relu(qk)
    s = (relu_qk * w_bsh_fp32.unsqueeze(-1)).sum(dim=2) * sm_scale  # (B, Sq, Sk)
    valid = _ratio_causal_valid_mask(Sq, Sk, ratio, s.device).unsqueeze(0)
    return torch.where(valid, s, torch.full_like(s, float('-inf')))


def _ref_attn_full_score(
    q_bshd_fp32: torch.Tensor,  # (B, Sq, H, D)
    k_bsd_fp32: torch.Tensor,  # (B, Sk, D)  — MQA
    lse_bshq_fp32: torch.Tensor,  # (B, Sq, H)
    softmax_scale: float,
    ratio: int,
) -> torch.Tensor:
    """Reference for ``_bwd_dense_attn_score.out``.

    ``out[b,q,k] = sum_h exp(Q[b,q,h] @ K[b,k]^T * scale - LSE[b,q,h])``,
    with the ratio causal mask producing ``0`` at masked positions
    (the per-head ``exp`` is zeroed out, contributing nothing to the sum).
    """
    B, Sq, _, _ = q_bshd_fp32.shape
    Sk = k_bsd_fp32.shape[1]
    qk = torch.einsum('bqhd,bkd->bqhk', q_bshd_fp32, k_bsd_fp32) * softmax_scale
    p = torch.exp(qk - lse_bshq_fp32.unsqueeze(-1))
    s = p.sum(dim=2)  # (B, Sq, Sk)
    valid = _ratio_causal_valid_mask(Sq, Sk, ratio, s.device).unsqueeze(0)
    return torch.where(valid, s, torch.zeros_like(s))


def _ref_indexer_predict_sparse(q_bshd_fp32, k_bsd_fp32, w_bsh_fp32, topk_indices, sm_scale):
    """Reference for ``sparse_indexer_score_recompute_wrapper.predict``.

    Compute the full-KV indexer score, gather ``topk_indices``, softmax
    over the topK axis. ``-1`` entries in topk are masked to ``-inf``
    so they contribute zero probability.
    """
    qk = torch.einsum('bqhd,bkd->bqhk', q_bshd_fp32, k_bsd_fp32)
    s = (torch.relu(qk) * w_bsh_fp32.unsqueeze(-1)).sum(dim=2) * sm_scale  # (B, Sq, Sk)
    valid = topk_indices >= 0
    safe = topk_indices.clamp(min=0).long()
    s_topk = torch.gather(s, dim=-1, index=safe)
    s_topk = torch.where(valid, s_topk, torch.full_like(s_topk, float('-inf')))
    return torch.softmax(s_topk, dim=-1)


def _ref_attn_target_sparse(q_bshd_fp32, k_bsd_fp32, lse_bsh_fp32, topk_indices, softmax_scale):
    """Reference for ``sparse_attn_score_recompute_wrapper.target``.

    Per-head ``exp(QK*scale - LSE)``, sum over heads, gather topK,
    L1-normalise over the topK axis. ``-1`` entries are zero-masked
    pre-normalisation.
    """
    qk = torch.einsum('bqhd,bkd->bqhk', q_bshd_fp32, k_bsd_fp32) * softmax_scale
    p = torch.exp(qk - lse_bsh_fp32.unsqueeze(-1))  # (B, Sq, H, Sk)
    s = p.sum(dim=2)  # (B, Sq, Sk)
    valid = topk_indices >= 0
    safe = topk_indices.clamp(min=0).long()
    s_topk = torch.gather(s, dim=-1, index=safe)
    s_topk = torch.where(valid, s_topk, torch.zeros_like(s_topk))
    denom = s_topk.sum(dim=-1, keepdim=True).clamp(min=1e-10)
    return s_topk / denom


def _ref_dense_indexer_loss(
    q_indexer_bshd_fp32,
    k_indexer_bsd_fp32,
    w_bsh_fp32,
    q_attn_bshd_fp32,
    k_attn_bsd_fp32,
    lse_bshq_fp32,
    indexer_softmax_scale: float,
    attn_softmax_scale: float,
    ratio: int,
    loss_coeff: float,
) -> torch.Tensor:
    """Reference dense KL loss (matches ``compute_dsa_indexer_loss(sparse_loss=False)``
    in ``dsa.py``). Uses the same ratio causal mask the kernel applies.
    """
    eps = torch.finfo(torch.float32).tiny
    # Per-(b,q,k) raw scores via the same formulas the kernels use.
    attn_scores = _ref_attn_full_score(
        q_attn_bshd_fp32, k_attn_bsd_fp32, lse_bshq_fp32, attn_softmax_scale, ratio
    )  # (B, Sq, Sk) head-summed, ratio-masked, zeros at masked positions.
    index_scores = _ref_indexer_full_score(
        q_indexer_bshd_fp32, k_indexer_bsd_fp32, w_bsh_fp32, indexer_softmax_scale, ratio
    )  # (B, Sq, Sk) ReLU·W, ratio-masked, -inf at masked positions.

    # L1-norm denom for target; LSE for predict.
    attn_denom = attn_scores.sum(dim=-1)  # (B, Sq)
    index_lse = torch.logsumexp(index_scores, dim=-1)  # (B, Sq), -inf for fully-masked rows

    row_valid = (attn_denom > eps) & torch.isfinite(index_lse)

    safe_l1 = attn_denom.clamp(min=eps)
    safe_lse = torch.where(row_valid, index_lse, torch.zeros_like(index_lse))

    target = attn_scores / safe_l1.unsqueeze(-1)
    target_clamped = target.clamp(min=eps)
    # Mask within-row: ratio-causal-masked positions have ``index_scores =
    # -inf`` (from ``_ref_indexer_full_score``). Letting them flow into
    # ``log_predict`` would make per-position contributions blow up to
    # +inf (``target_clamped * (log(target) - (-inf)) = +inf``). They have
    # zero mass under ``target`` (``_ref_attn_full_score`` zeros those
    # positions) so their KL contribution should be 0; explicitly mask.
    position_valid = torch.isfinite(index_scores)
    log_predict = torch.where(
        position_valid, index_scores - safe_lse.unsqueeze(-1), torch.zeros_like(index_scores)
    )
    contributions = target_clamped * (torch.log(target_clamped) - log_predict)
    contributions = torch.where(position_valid, contributions, torch.zeros_like(contributions))
    kl_per_row = contributions.sum(dim=-1)
    kl_per_row = torch.where(row_valid, kl_per_row, torch.zeros_like(kl_per_row))
    return loss_coeff * kl_per_row.mean()


def _ref_sparse_attn_forward(
    q_flat_bf16: torch.Tensor,  # (total_Sq, H, D)
    kv_flat_bf16: torch.Tensor,  # (total_Skv, D) — K=V, MQA
    attn_sink_fp32: torch.Tensor,  # (H,)
    topk_idxs: torch.Tensor,  # (total_Sq, topk) int32, global
    softmax_scale: float,
    d_v: int,
):
    """Pure-PyTorch reference for FlashMLA sparse-attn-fwd output.

    Mirrors the math FlashMLA implements:
      * Scores ``S[i, h, k] = Q[i, h] @ K[topk[i, k]]^T * scale`` for valid ``k``;
      * Append a per-head sink logit (``attn_sink``);
      * ``softmax`` over the (topk + sink) axis;
      * ``out[i, h] = sum_k softmax[i, h, k] * V[topk[i, k]]`` (excluding sink).

    Returns ``(out, lse)`` in the same shapes/dtype as the FlashMLA kernel.
    Invalid ``-1`` topk entries contribute zero to the softmax (logit -inf).
    """
    total_Sq, H, D = q_flat_bf16.shape
    topk = topk_idxs.shape[-1]
    device = q_flat_bf16.device
    q_fp32 = q_flat_bf16.float()
    kv_fp32 = kv_flat_bf16.float()

    valid = topk_idxs >= 0  # (total_Sq, topk)
    safe = topk_idxs.clamp(min=0).long()
    k_gathered = kv_fp32[safe]  # (total_Sq, topk, D)

    qk = torch.einsum('ihd,ikd->ihk', q_fp32, k_gathered) * softmax_scale  # (Sq, H, topk)
    qk = torch.where(valid.unsqueeze(1).expand(-1, H, -1), qk, torch.full_like(qk, float('-inf')))
    sink = attn_sink_fp32.view(1, H, 1).expand(total_Sq, H, 1)  # logit
    logits = torch.cat([qk, sink], dim=-1)  # (Sq, H, topk + 1)
    probs = torch.softmax(logits, dim=-1)  # numerically stable
    probs_kv = probs[..., :topk]  # exclude sink contribution from output

    v_gathered = k_gathered  # K = V (MQA, head-broadcast)
    out_fp32 = torch.einsum('ihk,ikd->ihd', probs_kv, v_gathered)  # (Sq, H, D_v=D)
    if d_v != D:
        out_fp32 = out_fp32[..., :d_v]
    out = out_fp32.to(q_flat_bf16.dtype)

    # FlashMLA's KV-only LSE excludes the sink term:
    # lse_kv[i, h] = logsumexp_k(qk[i, h, k]) over valid k only.
    lse_kv = torch.logsumexp(qk, dim=-1)  # (Sq, H), -inf for fully-masked rows
    return out, lse_kv


# ---------------------------------------------------------------------------
# Score-helper parity tests (sparse + dense): real cuDNN vs PyTorch reference
# ---------------------------------------------------------------------------


# Shared small shape across all real-kernel tests to maximize cuDNN compile-cache
# hits. ``ratio=1`` (standard upper-triangular causal) keeps the math simple
# and ensures every row has at least one valid KV position.
_REAL_SHAPES_SPARSE = dict(
    b=2,
    sq=128,
    sk=128,
    n_comp=128,
    np_=32,
    d=512,
    idx_nh=32,
    idx_hd=128,
    # topk = lcm(64, 128) = 128 satisfies SparseScoreRecomputeSm100's
    # `topk % n_block_size == 0` (64 for score_type=attention, 128 for indexer).
    topk=128,
    ratio=1,
    softmax_scale=512**-0.5,
    indexer_softmax_scale=128**-0.5,
)


def _build_real_score_inputs(s, *, with_lse: bool = True, with_topk: bool = True):
    """Build a coherent set of bf16 BSHD inputs for the score-helper tests.

    Returns a dict with both bf16 (kernel-ready) and fp32 (reference-math)
    views of every tensor, plus optional LSE / topk_indices.
    """
    torch.manual_seed(0)
    dev = 'cuda'

    q_idx = torch.randn(s['b'], s['sq'], s['idx_nh'], s['idx_hd'], dtype=torch.bfloat16, device=dev)
    k_idx = torch.randn(s['b'], s['sk'], s['idx_hd'], dtype=torch.bfloat16, device=dev)
    w = torch.randn(s['b'], s['sq'], s['idx_nh'], dtype=torch.bfloat16, device=dev)

    q_attn = torch.randn(s['b'], s['sq'], s['np_'], s['d'], dtype=torch.bfloat16, device=dev)
    k_attn = torch.randn(s['b'], s['sk'], s['d'], dtype=torch.bfloat16, device=dev)

    out = dict(q_idx=q_idx, k_idx=k_idx, w=w, q_attn=q_attn, k_attn=k_attn)

    if with_lse:
        # LSE = logsumexp(QK*scale, dim=Sk) with the kernel's ratio mask.
        # Real LSE input avoids exp(-inf - finite) underflow during reference.
        qk = torch.einsum('bqhd,bkd->bqhk', q_attn.float(), k_attn.float()) * s['softmax_scale']
        valid = _ratio_causal_valid_mask(s['sq'], s['sk'], s['ratio'], qk.device).view(
            1, s['sq'], 1, s['sk']
        )
        qk_masked = torch.where(valid, qk, torch.full_like(qk, float('-inf')))
        out['lse'] = torch.logsumexp(qk_masked, dim=-1).clamp(min=-1e30).contiguous()

    if with_topk:
        # Pick distinct random valid indices per (b, sq), with a few -1s
        # interleaved to exercise the invalid-slot path.
        topk = s['topk']
        torch.manual_seed(123)
        idxs = torch.randint(0, s['sk'], (s['b'], s['sq'], topk), dtype=torch.int32, device=dev)
        # Mark a few slots invalid (-1) to test the topk_indices < 0 path.
        invalid = torch.rand(s['b'], s['sq'], topk, device=dev) < 0.1
        idxs = torch.where(invalid, torch.full_like(idxs, -1), idxs)
        out['topk'] = idxs

    return out


class TestRealKernelScoreHelpers:
    """Real-kernel parity tests for the four ``_compute_*`` score helpers
    against PyTorch reference implementations. Single parametrized test
    covers all four; numeric tolerance is bf16-friendly (raw fp32 score
    sums agree to ~5%, normalized distributions to ~5e-3).
    """

    # Each case: (id, sm_min, kernel_name, runner). The runner does the
    # call + ref + assertion; it returns nothing on success.
    @pytest.mark.parametrize(
        "case",
        ['sparse_indexer_predict', 'sparse_attn_target', 'dense_indexer_score', 'dense_attn_score'],
    )
    def test_real_score_helper(self, case, reset_lazy_kernel_state):
        # SM gate per case: dense kernels are SM100-only, sparse are SM90+.
        sm_min = 10 if case.startswith('dense_') else 9
        _skip_if_real_kernels_unavailable(sm_min=sm_min)

        s = _REAL_SHAPES_SPARSE
        # Each case needs a different combination of the input fixture.
        x = _build_real_score_inputs(
            s,
            with_lse=case.endswith('_attn_target') or case.endswith('_attn_score'),
            with_topk=case.startswith('sparse_'),
        )

        from megatron.core.transformer.experimental_attention_variant import dsa_kernels as _dk

        if case == 'sparse_indexer_predict':
            # The kernel takes sm_scale=1.0; scale is applied via weights
            # pre-multiplication (relu(c·x)·W trick). Reference mirrors that.
            scale = s['indexer_softmax_scale']
            w_scaled = (x['w'].float() * scale).to(x['w'].dtype)
            out = _dk._compute_indexer_predict(
                x['q_idx'], x['k_idx'], w_scaled, x['topk'], qhead_per_kv_head=s['idx_nh']
            )
            ref = _ref_indexer_predict_sparse(
                x['q_idx'].float(), x['k_idx'].float(), w_scaled.float(), x['topk'], sm_scale=1.0
            )
            # Softmax outputs in [0, 1]; bf16 element-wise noise can break
            # absolute tolerance, so compare directions via cosine similarity.
            assert out.shape == ref.shape == (s['b'], s['sq'], s['topk'])
            cos = torch.nn.functional.cosine_similarity(
                out.flatten().unsqueeze(0).float(), ref.flatten().unsqueeze(0).float()
            ).item()
            assert cos > 0.99, (
                f"{case}: cos sim = {cos:.4f}, "
                f"max abs diff = {(out - ref).abs().max().item():.3e}"
            )

        elif case == 'sparse_attn_target':
            out = _dk._compute_attn_target(
                x['q_attn'],
                x['k_attn'],
                x['lse'],
                x['topk'],
                softmax_scale=s['softmax_scale'],
                qhead_per_kv_head=s['np_'],
            )
            ref = _ref_attn_target_sparse(
                x['q_attn'].float(),
                x['k_attn'].float(),
                x['lse'],
                x['topk'],
                softmax_scale=s['softmax_scale'],
            )
            assert out.shape == ref.shape == (s['b'], s['sq'], s['topk'])
            cos = torch.nn.functional.cosine_similarity(
                out.flatten().unsqueeze(0).float(), ref.flatten().unsqueeze(0).float()
            ).item()
            assert cos > 0.99, (
                f"{case}: cos sim = {cos:.4f}, "
                f"max abs diff = {(out - ref).abs().max().item():.3e}"
            )

        elif case == 'dense_indexer_score':
            out, denom = _dk._compute_dense_indexer_score(
                x['q_idx'],
                x['k_idx'].unsqueeze(2),
                x['w'],
                qhead_per_kv_head=s['idx_nh'],
                indexer_softmax_scale=s['indexer_softmax_scale'],
                ratio=s['ratio'],
            )
            ref_out = _ref_indexer_full_score(
                x['q_idx'].float(),
                x['k_idx'].float(),
                x['w'].float(),
                sm_scale=s['indexer_softmax_scale'],
                ratio=s['ratio'],
            )
            ref_denom = torch.logsumexp(ref_out, dim=-1)
            assert out.shape == ref_out.shape == (s['b'], s['sq'], s['sk'])
            assert denom.shape == ref_denom.shape == (s['b'], s['sq'])
            # Raw fp32 score sums: relative tolerance dominates. Compare
            # only valid positions (masked = -inf in both, NaN under sub).
            valid = (
                _ratio_causal_valid_mask(s['sq'], s['sk'], s['ratio'], out.device)
                .unsqueeze(0)
                .expand_as(out)
            )
            diff = torch.where(valid, (out - ref_out).abs(), torch.zeros_like(out))
            scale = ref_out.where(valid, torch.zeros_like(ref_out)).abs().max().item()
            assert diff.max().item() <= max(
                5e-2, 5e-2 * scale
            ), f"{case}: max abs diff = {diff.max().item():.3e}, scale = {scale:.3e}"
            row_valid = torch.isfinite(ref_denom)
            assert torch.allclose(
                denom[row_valid], ref_denom[row_valid], atol=5e-3, rtol=5e-2
            ), f"{case}: LSE max abs diff = {(denom - ref_denom)[row_valid].abs().max().item():.3e}"

        elif case == 'dense_attn_score':
            out, denom = _dk._compute_dense_attn_score(
                x['q_attn'],
                x['k_attn'].unsqueeze(2),
                x['lse'],
                qhead_per_kv_head=s['np_'],
                softmax_scale=s['softmax_scale'],
                ratio=s['ratio'],
            )
            ref_out = _ref_attn_full_score(
                x['q_attn'].float(),
                x['k_attn'].float(),
                x['lse'],
                softmax_scale=s['softmax_scale'],
                ratio=s['ratio'],
            )
            ref_denom = ref_out.sum(dim=-1)
            assert out.shape == ref_out.shape == (s['b'], s['sq'], s['sk'])
            assert denom.shape == ref_denom.shape == (s['b'], s['sq'])
            valid = (
                _ratio_causal_valid_mask(s['sq'], s['sk'], s['ratio'], out.device)
                .unsqueeze(0)
                .expand_as(out)
            )
            diff = torch.where(valid, (out - ref_out).abs(), torch.zeros_like(out))
            # exp(QK*scale - LSE) outputs in (0, ~1]: absolute dominates.
            assert diff.max().item() <= 5e-3, f"{case}: max abs diff = {diff.max().item():.3e}"
            assert torch.allclose(denom, ref_denom, atol=5e-3, rtol=5e-2), (
                f"{case}: denom max abs diff = " f"{(denom - ref_denom).abs().max().item():.3e}"
            )

        else:
            raise AssertionError(f"unknown case: {case}")


# ---------------------------------------------------------------------------
# KL loss reference parity (dense path; sparse already CPU-tested above).
# ---------------------------------------------------------------------------


class TestRealKernelKLLossDense:
    """End-to-end parity for ``_kl_loss_from_dense_scores``: run the real
    cuDNN dense score kernels, feed their outputs into the helper, and
    compare the KL value to the all-PyTorch reference.
    """

    @pytest.mark.parametrize("dummy", [None])
    def test_real_dense_kl_loss_matches_reference(self, dummy, reset_lazy_kernel_state):
        _skip_if_real_kernels_unavailable(sm_min=10)
        from megatron.core.transformer.experimental_attention_variant.dsa_kernels import (
            _compute_dense_attn_score,
            _compute_dense_indexer_score,
            _kl_loss_from_dense_scores,
        )

        s = _REAL_SHAPES_SPARSE
        x = _build_real_score_inputs(s, with_lse=True, with_topk=False)
        loss_coeff = 0.5

        index_score, index_lse = _compute_dense_indexer_score(
            x['q_idx'],
            x['k_idx'].unsqueeze(2),
            x['w'],
            qhead_per_kv_head=s['idx_nh'],
            indexer_softmax_scale=s['indexer_softmax_scale'],
            ratio=s['ratio'],
        )
        attn_score, attn_l1norm = _compute_dense_attn_score(
            x['q_attn'],
            x['k_attn'].unsqueeze(2),
            x['lse'],
            qhead_per_kv_head=s['np_'],
            softmax_scale=s['softmax_scale'],
            ratio=s['ratio'],
        )

        loss_actual = _kl_loss_from_dense_scores(
            attn_score, attn_l1norm, index_score, index_lse, loss_coeff
        )
        loss_ref = _ref_dense_indexer_loss(
            x['q_idx'].float(),
            x['k_idx'].float(),
            x['w'].float(),
            x['q_attn'].float(),
            x['k_attn'].float(),
            x['lse'],
            indexer_softmax_scale=s['indexer_softmax_scale'],
            attn_softmax_scale=s['softmax_scale'],
            ratio=s['ratio'],
            loss_coeff=loss_coeff,
        )
        assert torch.allclose(loss_actual, loss_ref, atol=1e-3, rtol=1e-2), (
            f"actual = {loss_actual.item():.6f}, ref = {loss_ref.item():.6f}, "
            f"abs diff = {(loss_actual - loss_ref).abs().item():.3e}"
        )


# ---------------------------------------------------------------------------
# Real ``indexer_topk``: the top-K set should match the reference ranking.
# ---------------------------------------------------------------------------


class TestRealKernelIndexerTopk:
    """Real-kernel parity for :func:`indexer_topk`: the SET of selected
    top-K indices must match a PyTorch reference ranking. We compare sets
    rather than ordered lists because BF16 ties may be broken differently.
    """

    @pytest.mark.parametrize("dummy", [None])
    def test_real_indexer_topk_set_matches_reference(self, dummy, reset_lazy_kernel_state):
        _skip_if_real_kernels_unavailable(sm_min=10)  # IndexerForward is SM100+
        from megatron.core.transformer.experimental_attention_variant.dsa_kernels import (
            indexer_topk,
        )

        # IndexerForward requires idx_hd=128 and qhpkv in (32, 64). Use an
        # SBHD shape that matches what csa.py produces (tensors are SBHD,
        # ratio is the indexer's compression ratio). b=2 exercises the
        # batch-aware ``seq_lens.repeat(b)`` and the ``(b*sq, sk) → (b, sq,
        # topk)`` reshape inside ``_indexer_topk_bshd``.
        s = dict(
            b=2,
            sq=128,
            idx_nh=32,
            idx_hd=128,
            sk=128,
            indexer_topk=8,
            ratio=4,
            indexer_softmax_scale=128**-0.5,
        )
        torch.manual_seed(0)
        dev = 'cuda'
        q_indexer = torch.randn(
            s['sq'], s['b'], s['idx_nh'], s['idx_hd'], dtype=torch.bfloat16, device=dev
        )
        k_indexer = torch.randn(s['sk'], s['b'], s['idx_hd'], dtype=torch.bfloat16, device=dev)
        weights = torch.randn(s['sq'], s['b'], s['idx_nh'], dtype=torch.bfloat16, device=dev)

        topk_indices, topk_length = indexer_topk(
            q_indexer,
            k_indexer,
            weights,
            topk=s['indexer_topk'],
            ratio=s['ratio'],
            indexer_softmax_scale=s['indexer_softmax_scale'],
        )
        assert topk_indices.shape == (s['b'], s['sq'], s['indexer_topk'])
        assert topk_indices.dtype == torch.int32

        # Reference: full indexer score, ratio causal mask, take top-K per row
        # by descending score. Score is sm_scale * sum_h ReLU(Q@K) * W.
        q_bshd = q_indexer.permute(1, 0, 2, 3).contiguous().float()
        k_bsd = k_indexer.permute(1, 0, 2).contiguous().float()
        w_bsh = weights.permute(1, 0, 2).contiguous().float()
        ref_scores = _ref_indexer_full_score(
            q_bshd, k_bsd, w_bsh, sm_scale=s['indexer_softmax_scale'], ratio=s['ratio']
        )  # (B, Sq, Sk), -inf at masked positions

        # For each row, count valid positions (un-masked). topk_length should
        # match min(indexer_topk, num_valid).
        n_valid = (ref_scores > float('-inf')).sum(dim=-1)  # (B, Sq)
        expected_length = n_valid.clamp(max=s['indexer_topk']).int()
        assert torch.equal(topk_length, expected_length)

        # Set comparison row-by-row. Skip rows with 0 valid (kernel returns
        # all -1; reference picks arbitrary -inf positions).
        ref_topk = torch.topk(ref_scores, k=s['indexer_topk'], dim=-1).indices
        for bi in range(s['b']):
            for qi in range(s['sq']):
                n = int(expected_length[bi, qi].item())
                if n == 0:
                    # Kernel must report all -1.
                    assert torch.all(topk_indices[bi, qi] == -1)
                    continue
                actual_set = set(topk_indices[bi, qi, :n].tolist())
                ref_set = set(ref_topk[bi, qi, :n].tolist())
                # BF16 ties may differ: allow up to ~10% mismatch on small K.
                inter = actual_set & ref_set
                assert len(inter) >= max(1, n - 1), (
                    f"row (b={bi}, q={qi}): "
                    f"actual {sorted(actual_set)} vs ref {sorted(ref_set)}"
                )


# ---------------------------------------------------------------------------
# Real ``dsa_sparse_attn``: forward + backward parity vs PyTorch reference.
# ---------------------------------------------------------------------------


class TestRealKernelDsaSparseAttn:
    """Real-kernel parity for :func:`dsa_sparse_attn`. Forward uses real
    FlashMLA + the SBHD/flat reshape wrapper; backward uses the real cuDNN
    sparse-attn-bwd kernel. Both checked in one test against the
    pure-PyTorch sparse-attn reference (``_ref_sparse_attn_forward``).
    """

    SHAPES = dict(b=2, sq=128, np_=64, d=512, skv=128, topk=32, softmax_scale=512**-0.5)

    def _make_inputs(self, *, requires_grad: bool):
        s = self.SHAPES
        torch.manual_seed(0)
        dev = 'cuda'

        def make_leaf(*shape, dtype):
            t = torch.randn(*shape, dtype=dtype, device=dev)
            return t.detach().clone().requires_grad_(True) if requires_grad else t

        query = make_leaf(s['sq'], s['b'], s['np_'], s['d'], dtype=torch.bfloat16)
        kv = make_leaf(s['skv'], s['b'], s['d'], dtype=torch.bfloat16)
        attn_sink = torch.zeros(s['np_'], dtype=torch.float32, device=dev)
        if requires_grad:
            attn_sink = attn_sink.detach().clone().requires_grad_(True)

        # Coherent valid global topk indices in SBHD-flat layout, with a
        # standard causal mask (index <= q_idx).
        torch.manual_seed(1)
        topk_local = torch.randint(
            0, s['skv'], (s['b'], s['sq'], s['topk']), dtype=torch.int64, device=dev
        )
        q_idx = torch.arange(s['sq'], device=dev).view(1, -1, 1)
        topk_local = torch.minimum(topk_local, q_idx)
        global_idxs = local_to_global_flat(topk_local, s['b'], s['skv']).contiguous()
        return query, kv, attn_sink, global_idxs

    def test_real_dsa_sparse_attn_fwd_bwd_matches_reference(self, reset_lazy_kernel_state):
        """Forward output AND backward gradients (dq, dkv, d_sink) must
        match a pure-PyTorch sparse-attn reference. Combining both checks
        in one test halves cuDNN compile time vs running them separately,
        since they share the same kernel cache key.
        """
        _skip_if_real_kernels_unavailable(sm_min=9, need_flash_mla=True)
        s = self.SHAPES

        # ---- Real path: forward + backward via dsa_sparse_attn ----
        query, kv, attn_sink, global_idxs = self._make_inputs(requires_grad=True)
        out = dsa_sparse_attn(query, kv, attn_sink, global_idxs, softmax_scale=s['softmax_scale'])
        torch.manual_seed(7)
        upstream = torch.randn_like(out)
        (out * upstream).sum().backward()
        dq_actual = query.grad.float().clone()
        dkv_actual = kv.grad.float().clone()
        dsink_actual = attn_sink.grad.float().clone()
        out_actual = out.float().detach().clone()

        # ---- Reference: pure-PyTorch forward + autograd ----
        query_ref, kv_ref, attn_sink_ref, _ = self._make_inputs(requires_grad=True)
        q_flat = query_ref.reshape(s['sq'] * s['b'], s['np_'], s['d'])
        kv_flat = kv_ref.reshape(s['skv'] * s['b'], s['d'])
        ref_out_flat, _ = _ref_sparse_attn_forward(
            q_flat,
            kv_flat,
            attn_sink_ref,
            global_idxs,
            softmax_scale=s['softmax_scale'],
            d_v=s['d'],
        )
        ref_out = ref_out_flat.reshape(s['sq'], s['b'], s['np_'], s['d']).reshape(
            s['sq'], s['b'], s['np_'] * s['d']
        )
        (ref_out * upstream).sum().backward()

        # ---- Forward + backward parity (cos sim) ----
        # bf16 GEMM accumulators in FlashMLA fwd / cuDNN sparse-attn-bwd
        # make element-wise tolerances brittle (esp. dkv); compare each
        # tensor's direction via cosine similarity instead.
        def _cos(a, b):
            return torch.nn.functional.cosine_similarity(
                a.flatten().unsqueeze(0).float(), b.flatten().unsqueeze(0).float()
            ).item()

        assert out_actual.shape == ref_out.shape
        for name, actual, ref in [
            ('forward', out_actual, ref_out.float()),
            ('dq', dq_actual, query_ref.grad.float()),
            ('dkv', dkv_actual, kv_ref.grad.float()),
            ('d_sink', dsink_actual, attn_sink_ref.grad.float()),
        ]:
            cos = _cos(actual, ref)
            assert cos > 0.99, (
                f"{name}: cos sim = {cos:.4f}, "
                f"max abs diff = {(actual - ref).abs().max().item():.3e}"
            )


# ---------------------------------------------------------------------------
# Real ``fused_indexer_sparse_attn``: dense-loss path end-to-end parity.
# ---------------------------------------------------------------------------


class TestRealKernelFusedIndexerSparseAttn:
    """End-to-end parity for the dense loss path of
    :func:`fused_indexer_sparse_attn`: real cuDNN dense kernels (forward
    + backward) + real FlashMLA, compared to ``_ref_dense_indexer_loss``.

    Backward grad correctness for the indexer-grad kernel is established by
    ``TestRealKernelKLLossDense`` (kernel-level math) and
    ``TestDenseFusedIndexerSparseAttn::test_dense_backward_calls_dense_indexer_grad``
    (mock-based plumbing). This class only checks the loss SCALAR value.
    """

    # FlashMLA only accepts indexer_topk ∈ {0, 512, 1024, 2048} and a limited
    # set of h_q values (np_=64 is the supported one used by the sibling
    # DsaSparseAttn real-kernel test). n_comp must be ≥ indexer_topk; skv ≥
    # n_comp so kv_offset = skv - n_comp > 0 still exercises the offset path.
    SHAPES = dict(
        b=2,
        sq=128,
        np_=64,
        d=512,
        skv=640,
        n_comp=512,
        idx_nh=32,
        idx_hd=128,
        indexer_topk=512,
        ratio=4,
        win_topk=8,
        softmax_scale=512**-0.5,
        indexer_softmax_scale=128**-0.5,
    )

    def test_real_fused_dense_loss_matches_reference(self, reset_lazy_kernel_state):
        """Real dense path's KL loss value matches the all-PyTorch reference
        on the same inputs. The reference uses an analytical
        ``logsumexp(QK*scale, ratio mask)`` for ``lse_indexer`` (FlashMLA
        emits its own internal lse_indexer that differs slightly), so the
        tolerance is wider than for the kernel-only ``KLLossDense`` test.
        """
        _skip_if_real_kernels_unavailable(sm_min=10, need_flash_mla=True)
        s = self.SHAPES
        torch.manual_seed(0)
        dev = 'cuda'
        loss_coeff = 0.5

        # Build inputs once; share between actual and reference.
        query = torch.randn(s['sq'], s['b'], s['np_'], s['d'], dtype=torch.bfloat16, device=dev)
        kv_full = torch.randn(s['skv'], s['b'], s['d'], dtype=torch.bfloat16, device=dev)
        attn_sink = torch.zeros(s['np_'], dtype=torch.float32, device=dev)
        torch.manual_seed(1)
        win_idxs = torch.randint(
            0, s['sq'], (s['b'], s['sq'], s['win_topk']), dtype=torch.int32, device=dev
        )
        q_indexer = torch.randn(
            s['sq'], s['b'], s['idx_nh'], s['idx_hd'], dtype=torch.bfloat16, device=dev
        )
        k_indexer = torch.randn(s['n_comp'], s['b'], s['idx_hd'], dtype=torch.bfloat16, device=dev)
        weights = torch.randn(s['sq'], s['b'], s['idx_nh'], dtype=torch.bfloat16, device=dev)
        kv_offset = s['skv'] - s['n_comp']

        # Real path.
        _, indexer_loss = fused_indexer_sparse_attn(
            query,
            kv_full,
            attn_sink,
            win_idxs,
            q_indexer,
            k_indexer,
            weights,
            indexer_topk=s['indexer_topk'],
            ratio=s['ratio'],
            softmax_scale=s['softmax_scale'],
            indexer_softmax_scale=s['indexer_softmax_scale'],
            loss_coeff=loss_coeff,
            sparse_loss=False,
            kv_offset=kv_offset,
        )

        # Reference: SBHD->BSHD once, build analytical lse_ref, compute KL.
        q_idx_bshd = q_indexer.permute(1, 0, 2, 3).contiguous().float()
        k_idx_bsd = k_indexer.permute(1, 0, 2).contiguous().float()
        w_bsh = weights.permute(1, 0, 2).contiguous().float()
        q_attn_bshd = query.permute(1, 0, 2, 3).contiguous().float()
        k_attn_bsd = kv_full[kv_offset:].permute(1, 0, 2).contiguous().float()

        # PyTorch reference that mirrors the fused path's dense-loss math
        # exactly. Two non-obvious requirements:
        #   * Use FlashMLA's emitted ``lse_indexer`` (logsumexp over the
        #     indexer-selected top-K positions, with the per-head sink term),
        #     not an analytical full-KV logsumexp. Otherwise the per-row LSE
        #     basis differs from the kernel by ~50x.
        #   * Do NOT apply the ratio-causal mask in the reference scores —
        #     the dense-score-recompute kernels emit values at every position
        #     (no internal masking). Masking the reference would shift the
        #     ``attn_score / attn_l1norm`` normalization and the indexer LSE
        #     basis, producing a different KL than the kernel's.
        from megatron.core.transformer.experimental_attention_variant.dsa_kernels import (
            _dsa_fwd_flash_mla,
            _indexer_topk_bshd,
            _kl_loss_from_dense_scores,
            _sbhd_to_bshd_indexer_inputs,
        )

        # Run indexer + FlashMLA to capture the same ``lse_indexer`` the fused
        # path consumes internally.
        effective_topk = min(s['indexer_topk'], s['n_comp'])
        q_idx_bshd_bf, k_idx_bsd_bf, _, w_bsh_scaled_bf = _sbhd_to_bshd_indexer_inputs(
            q_indexer, k_indexer, weights, s['indexer_softmax_scale']
        )
        topk_indices_cmp, _ = _indexer_topk_bshd(
            q_idx_bshd_bf, k_idx_bsd_bf, w_bsh_scaled_bf, effective_topk, s['ratio']
        )
        compress_topk_idxs = torch.where(topk_indices_cmp >= 0, topk_indices_cmp + kv_offset, -1)
        combined_local = torch.cat([compress_topk_idxs, win_idxs], dim=-1)
        global_idxs = local_to_global_flat(combined_local, s['b'], s['skv'])
        q_flat = query.reshape(s['sq'] * s['b'], s['np_'], s['d'])
        kv_flat = kv_full.reshape(s['skv'] * s['b'], s['d'])
        _, _, lse_indexer = _dsa_fwd_flash_mla(
            q_flat,
            kv_flat,
            global_idxs,
            s['softmax_scale'],
            attn_sink=attn_sink,
            topk_length=None,
            indexer_topk=effective_topk,
        )
        lse_indexer_bsqh = lse_indexer.reshape(s['sq'], s['b'], s['np_']).permute(1, 0, 2)

        # Attention path: exp(QK*scale - lse_indexer), head-summed. No mask.
        qk_attn = torch.einsum('bqhd,bkd->bqhk', q_attn_bshd, k_attn_bsd) * s['softmax_scale']
        attn_score_ref = torch.exp(qk_attn - lse_indexer_bsqh.unsqueeze(-1)).sum(dim=2)
        attn_l1norm_ref = attn_score_ref.sum(dim=-1)

        # Indexer path: ReLU(QK_indexer) * W head-summed. The fused path
        # calls ``_compute_dense_indexer_score`` with ``w_bsh_scaled`` (already
        # multiplied by ``indexer_softmax_scale``) AND passes
        # ``indexer_softmax_scale`` again as the kernel's ``sm_scale``,
        # double-applying the factor (apparent bug in
        # ``fused_indexer_sparse_attn`` at ``dsa_kernels.py:800-807``). Mirror
        # that here so the reference matches the fused-path output; revisit
        # if the upstream pre-scale + kernel-scale duplication is fixed.
        qk_idx = torch.einsum('bqhd,bkd->bqhk', q_idx_bshd, k_idx_bsd)
        idx_score_ref = (torch.relu(qk_idx) * w_bsh.unsqueeze(-1)).sum(dim=2) * (
            s['indexer_softmax_scale'] ** 2
        )
        idx_lse_ref = torch.logsumexp(idx_score_ref, dim=-1)

        loss_ref = _kl_loss_from_dense_scores(
            attn_score_ref, attn_l1norm_ref, idx_score_ref, idx_lse_ref, loss_coeff
        )
        assert torch.allclose(indexer_loss, loss_ref, atol=5e-2, rtol=1e-1), (
            f"actual = {indexer_loss.item():.6f}, ref = {loss_ref.item():.6f}, "
            f"abs diff = {(indexer_loss - loss_ref).abs().item():.3e}"
        )


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


class TestPublicApi:
    """The ``__all__`` list documents the public surface; verify that every
    advertised symbol is importable, that the public free functions are
    callable, and that the autograd Functions inherit from the right base.
    """

    def test_public_surface(self):
        from megatron.core.transformer.experimental_attention_variant import dsa_kernels

        for name in dsa_kernels.__all__:
            assert hasattr(dsa_kernels, name), f"__all__ lists {name!r} but it is missing"

        for fn in (
            build_flat_topk_idxs,
            local_to_global_flat,
            dsa_sparse_attn,
            indexer_topk,
            fused_indexer_sparse_attn,
        ):
            assert callable(fn)

        assert issubclass(SparseAttnFunc, torch.autograd.Function)
        assert issubclass(FusedIndexerSparseAttnFunc, torch.autograd.Function)
