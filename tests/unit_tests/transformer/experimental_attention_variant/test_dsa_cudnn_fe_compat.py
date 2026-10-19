# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""cuDNN frontend version compatibility of the DSA cuDNN dense call sites.

cuDNN frontend 1.29 changed the default ``q_causal_offset`` of the dense DSA kernels from
the bottom-right alignment (``seqlen_k * ratio - seqlen_q``) to zero. These CPU tests use
fake frontends that model both defaults and check that MCore produces the same top-k on
either one.
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
    [("1.31.0", (1, 31)), ("1.21.1", (1, 21)), ("1.32.0.dev74371060", (1, 32)), ("dev", None)],
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
        assert not dsa_cudnn_kernels._cudnn_frontend_at_least((1, 29))
    finally:
        dsa_cudnn_kernels._cudnn_frontend_version.cache_clear()


def test_bottom_right_offsets_are_omitted_on_legacy_frontends(frontend):
    """The production 1.21.1 overlay keeps its exact call and compile key."""
    dsa = frontend(fakes.LegacyOverlayDSA, (1, 21))
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


def test_bottom_right_offsets_are_explicit_on_top_left_frontends(frontend):
    dsa = frontend(fakes.Fe131DSA, (1, 31))
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
            q, k, w, topk, return_topk_scores, cp_size, cp_rank
        )

    frontend(fakes.LegacyOverlayDSA, (1, 21))
    legacy_indices, legacy_scores = run()
    assert all(call["q_causal_offsets"] is None for call in fakes.LegacyOverlayDSA.calls)

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
    frontend(fakes.LegacyOverlayDSA, (1, 21))
    legacy_indices, _ = dsa_cudnn_kernels._indexer_topk_single_packed_cp_segments(
        q, k, w, topk, False, cp_size, 1
    )
    frontend(fakes.Fe131DSA, (1, 31))
    monkeypatch.setattr(dsa_cudnn_kernels, "_bottom_right_q_offset_kwargs", lambda *a, **kw: {})
    broken_indices, _ = dsa_cudnn_kernels._indexer_topk_single_packed_cp_segments(
        q, k, w, topk, False, cp_size, 1
    )
    assert not torch.equal(broken_indices, legacy_indices)


def test_plain_causal_dense_scores_keep_the_call_unchanged(frontend):
    """Square plain-causal scoring has a zero offset, so no offsets reach either frontend."""
    q, k, w = _inputs(6, 6, seed=5)
    outputs = []
    for namespace, version in ((fakes.LegacyOverlayDSA, (1, 21)), (fakes.Fe131DSA, (1, 31))):
        dsa = frontend(namespace, version)
        outputs.append(
            dsa_cudnn_kernels._indexer_topk_bshd(q, k[:, :, 0, :], w, topk=3, return_scores=True)
        )
        assert all(call["q_causal_offsets"] is None for call in dsa.calls)
    torch.testing.assert_close(outputs[0][0], outputs[1][0])
    torch.testing.assert_close(outputs[0][2], outputs[1][2])
