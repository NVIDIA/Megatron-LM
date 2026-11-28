# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""cuDNN frontend version compatibility of the DSA cuDNN dense call sites.

cuDNN frontend 1.26 (NVIDIA/cudnn-frontend#316) changed the default ``q_causal_offset`` of
the dense DSA kernels from the bottom-right alignment (``seqlen_k * ratio - seqlen_q``) to
zero: FE <= 1.25 aligns bottom-right, FE >= 1.26 top-left. These CPU tests use fake
frontends that model both defaults and check that MCore produces the same top-k on either
one.
"""

import sys
import types

import pytest
import torch

from megatron.core.transformer.experimental_attention_variant import dsa_cudnn_kernels
from tests.unit_tests.transformer.experimental_attention_variant import dsa_cudnn_fe_fakes as fakes


@pytest.fixture
def frontend(monkeypatch):
    """Install a fake frontend namespace and version for one test."""

    def install(namespace, version):
        namespace.reset()
        monkeypatch.setattr(dsa_cudnn_kernels, "_ensure_dsa_namespace", lambda: None)
        monkeypatch.setattr(dsa_cudnn_kernels, "_cudnn_dsa", namespace)
        monkeypatch.setattr(dsa_cudnn_kernels, "_cudnn_frontend_version", lambda: version)
        return namespace

    return install


def _inputs(sq, sk, heads=4, dim=8, seed=0):
    generator = torch.Generator().manual_seed(seed)
    q = torch.randn((1, sq, heads, dim), generator=generator).to(torch.bfloat16)
    k = torch.randn((1, sk, 1, dim), generator=generator).to(torch.bfloat16)
    w = torch.rand((1, sq, heads), generator=generator).to(torch.bfloat16)
    return q, k, w


@pytest.mark.parametrize(
    "version_string, expected",
    [("1.31.0", (1, 31)), ("1.25.0", (1, 25)), ("1.32.0.dev74371060", (1, 32)), ("dev", None)],
)
def test_frontend_version_is_parsed_from_the_cudnn_package(monkeypatch, version_string, expected):
    monkeypatch.setitem(sys.modules, "cudnn", types.SimpleNamespace(__version__=version_string))
    dsa_cudnn_kernels._cudnn_frontend_version.cache_clear()
    try:
        assert dsa_cudnn_kernels._cudnn_frontend_version() == expected
    finally:
        dsa_cudnn_kernels._cudnn_frontend_version.cache_clear()


def test_missing_frontend_has_no_version(monkeypatch):
    monkeypatch.setitem(sys.modules, "cudnn", None)
    dsa_cudnn_kernels._cudnn_frontend_version.cache_clear()
    try:
        assert dsa_cudnn_kernels._cudnn_frontend_version() is None
        assert not dsa_cudnn_kernels._cudnn_frontend_at_least((1, 26))
    finally:
        dsa_cudnn_kernels._cudnn_frontend_version.cache_clear()


@pytest.mark.parametrize("version", [(1, 21), (1, 25)])
def test_bottom_right_offsets_are_omitted_on_legacy_frontends(frontend, version):
    """Frontends <= 1.25 keep their exact call and compile key."""
    dsa = frontend(fakes.BottomRightDefaultDSA, version)
    kwargs = dsa_cudnn_kernels._bottom_right_q_offset_kwargs(
        dsa.indexer_forward_wrapper, batch=2, seqlen_q=4, seqlen_k=12, ratio=1, device="cpu"
    )
    assert kwargs == {}


def test_bottom_right_offsets_are_omitted_when_both_alignments_agree(frontend):
    dsa = frontend(fakes.Fe131DSA, (1, 31))
    kwargs = dsa_cudnn_kernels._bottom_right_q_offset_kwargs(
        dsa.indexer_forward_wrapper, batch=2, seqlen_q=12, seqlen_k=12, ratio=1, device="cpu"
    )
    assert kwargs == {}


@pytest.mark.parametrize("version", [(1, 26), (1, 29), (1, 31)])
def test_bottom_right_offsets_are_explicit_on_top_left_frontends(frontend, version):
    dsa = frontend(fakes.Fe131DSA, version)
    kwargs = dsa_cudnn_kernels._bottom_right_q_offset_kwargs(
        dsa.indexer_forward_wrapper, batch=2, seqlen_q=4, seqlen_k=12, ratio=1, device="cpu"
    )
    torch.testing.assert_close(kwargs["q_causal_offsets"], torch.tensor([8, 8], dtype=torch.int32))


def test_top_left_frontend_without_offsets_fails_closed(frontend):
    def wrapper(q, k, w, ratio=1, sm_scale=1.0):
        raise AssertionError("must not be called")

    frontend(fakes.Fe131DSA, (1, 31))
    with pytest.raises(RuntimeError, match="bottom-right alignment"):
        dsa_cudnn_kernels._bottom_right_q_offset_kwargs(
            wrapper, batch=1, seqlen_q=4, seqlen_k=12, ratio=1, device="cpu"
        )


@pytest.mark.parametrize("cp_rank", [0, 1])
@pytest.mark.parametrize("return_topk_scores", [False, True])
def test_cropped_segment_scorer_matches_across_frontend_defaults(
    frontend, cp_rank, return_topk_scores
):
    """The single-sequence CP scorer crops K so bottom-right masking gives absolute lengths."""
    cp_size, local_sq, topk = 2, 8, 4
    q, k, w = _inputs(local_sq, local_sq * cp_size, seed=cp_rank)

    def run():
        return dsa_cudnn_kernels._indexer_topk_single_packed_cp_segments(
            q, k, w, topk, return_topk_scores, cp_rank
        )

    frontend(fakes.BottomRightDefaultDSA, (1, 21))
    legacy_indices, legacy_scores = run()
    assert all(call["q_causal_offsets"] is None for call in fakes.BottomRightDefaultDSA.calls)

    dsa = frontend(fakes.Fe131DSA, (1, 31))
    new_indices, new_scores = run()
    assert any(call["q_causal_offsets"] is not None for call in dsa.calls)

    torch.testing.assert_close(new_indices, legacy_indices)
    if return_topk_scores:
        torch.testing.assert_close(new_scores, legacy_scores)


def test_cropped_segment_scorer_detects_the_top_left_default(frontend, monkeypatch):
    """Without the explicit offsets FE 1.31 would select keys from the wrong prefix."""
    cp_size, local_sq, topk = 2, 8, 4
    q, k, w = _inputs(local_sq, local_sq * cp_size, seed=3)
    frontend(fakes.BottomRightDefaultDSA, (1, 21))
    legacy_indices, _ = dsa_cudnn_kernels._indexer_topk_single_packed_cp_segments(
        q, k, w, topk, False, 1
    )
    frontend(fakes.Fe131DSA, (1, 31))
    monkeypatch.setattr(dsa_cudnn_kernels, "_bottom_right_q_offset_kwargs", lambda *a, **kw: {})
    broken_indices, _ = dsa_cudnn_kernels._indexer_topk_single_packed_cp_segments(
        q, k, w, topk, False, 1
    )
    assert not torch.equal(broken_indices, legacy_indices)


def test_plain_causal_dense_scores_keep_the_call_unchanged(frontend):
    """Square plain-causal scoring has a zero offset, so no offsets reach either frontend."""
    q, k, w = _inputs(6, 6, seed=5)
    outputs = []
    for namespace, version in ((fakes.BottomRightDefaultDSA, (1, 21)), (fakes.Fe131DSA, (1, 31))):
        dsa = frontend(namespace, version)
        outputs.append(
            dsa_cudnn_kernels._indexer_topk_bshd(q, k[:, :, 0, :], w, topk=3, return_scores=True)
        )
        assert all(call["q_causal_offsets"] is None for call in dsa.calls)
    torch.testing.assert_close(outputs[0][0], outputs[1][0])
    torch.testing.assert_close(outputs[0][2], outputs[1][2])


def _multi_packed_cp_inputs(doc_lengths, cp_size, cp_rank, heads=4, dim=8, seed=0):
    """Local zigzag rows of a multi-sequence pack plus each row's absolute key bounds."""
    generator = torch.Generator().manual_seed(seed)
    cu = torch.tensor([0] + list(doc_lengths)).cumsum(0)
    starts, ends = [], []
    for doc_start, length in zip(cu[:-1].tolist(), doc_lengths):
        half = length // (2 * cp_size)
        for chunk in (cp_rank, 2 * cp_size - 1 - cp_rank):
            for i in range(half):
                starts.append(doc_start)
                ends.append(doc_start + chunk * half + i + 1)
    sq, sk = len(starts), int(cu[-1])
    q = torch.randn((1, sq, heads, dim), generator=generator).to(torch.bfloat16)
    k = torch.randn((1, sk, 1, dim), generator=generator).to(torch.bfloat16)
    w = torch.rand((1, sq, heads), generator=generator).to(torch.bfloat16)
    cu = cu.to(torch.int32)
    return q, k, w, torch.tensor(starts), torch.tensor(ends), cu, max(doc_lengths)


@pytest.mark.parametrize("cp_rank", [0, 1])
def test_multi_packed_cp_thd_scorer_matches_across_frontend_defaults(frontend, cp_rank):
    """The multi-sequence CP scorer's segments need per-segment bottom-right offsets."""
    cp_size, topk = 2, 4
    q, k, w, starts, ends, cu, max_len = _multi_packed_cp_inputs((16, 24), cp_size, cp_rank)

    def run():
        return dsa_cudnn_kernels._indexer_topk_multi_packed_cp_thd(
            q, k, w, topk, True, starts, ends, cu, cu, max_len, max_len, cp_size, cp_rank
        )

    frontend(fakes.BottomRightDefaultDSA, (1, 21))
    legacy_indices, legacy_scores = run()
    assert all(call["q_causal_offsets"] is None for call in fakes.BottomRightDefaultDSA.calls)

    dsa = frontend(fakes.Fe131DSA, (1, 31))
    new_indices, new_scores = run()
    torch.testing.assert_close(new_indices, legacy_indices)
    torch.testing.assert_close(new_scores, legacy_scores)
    offsets = [call["q_causal_offsets"] for call in dsa.calls]
    assert offsets and all(o is not None for o in offsets)
    # Every selected key lies inside its row's absolute causal bounds.
    valid = new_indices[0] >= 0
    rows = torch.arange(new_indices.size(1)).view(-1, 1).expand_as(new_indices[0])
    assert torch.all(new_indices[0][valid] >= starts[rows[valid]])
    assert torch.all(new_indices[0][valid] < ends[rows[valid]])


def test_multi_packed_cp_thd_scorer_detects_the_top_left_default(frontend, monkeypatch):
    """Without the per-segment offsets FE 1.31 would select keys from the wrong prefix."""
    cp_size, topk = 2, 4
    q, k, w, starts, ends, cu, max_len = _multi_packed_cp_inputs((16, 24), cp_size, 1, seed=3)

    def run():
        return dsa_cudnn_kernels._indexer_topk_multi_packed_cp_thd(
            q, k, w, topk, False, starts, ends, cu, cu, max_len, max_len, cp_size, 1
        )[0]

    frontend(fakes.BottomRightDefaultDSA, (1, 21))
    legacy_indices = run()
    frontend(fakes.Fe131DSA, (1, 31))
    monkeypatch.setattr(
        dsa_cudnn_kernels, "_varlen_bottom_right_q_offset_kwargs", lambda *a, **kw: {}
    )
    assert not torch.equal(run(), legacy_indices)


def test_varlen_offsets_are_omitted_on_legacy_frontends(frontend):
    dsa = frontend(fakes.BottomRightDefaultDSA, (1, 25))
    kwargs = dsa_cudnn_kernels._varlen_bottom_right_q_offset_kwargs(
        dsa.indexer_forward_wrapper, torch.tensor([0, 8])
    )
    assert kwargs == {}


def test_varlen_top_left_frontend_without_offsets_fails_closed(frontend):
    def wrapper(q, k, w, ratio=1, sm_scale=1.0, cu_seqlens_q=None, cu_seqlens_k=None):
        raise AssertionError("must not be called")

    frontend(fakes.Fe131DSA, (1, 29))
    with pytest.raises(RuntimeError, match="bottom-right alignment"):
        dsa_cudnn_kernels._varlen_bottom_right_q_offset_kwargs(wrapper, torch.tensor([0, 8]))
