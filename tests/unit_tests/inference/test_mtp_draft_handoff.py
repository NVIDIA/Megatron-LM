# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""MTP draft distributions (q) carried across a disaggregated prefill->decode handoff."""

import json
from types import SimpleNamespace

import msgpack
import pytest
import torch

from megatron.core.inference.disaggregation.inference_state_handoff import (
    InferenceStateHandoffMixin,
)
from megatron.core.inference.disaggregation.mtp_draft_handoff import (
    MTP_DRAFT_Q_KEY,
    MTP_VERIFICATION_KEY,
    decode_mtp_draft_q,
    encode_mtp_draft_q,
)


def _q(num_depths, vocab, *, support=None, seed=0):
    gen = torch.Generator().manual_seed(seed)
    logits = torch.randn(num_depths, vocab, generator=gen) * 3
    if support is not None:
        keep = torch.zeros_like(logits, dtype=torch.bool)
        for depth in range(num_depths):
            keep[depth, torch.randperm(vocab, generator=gen)[:support]] = True
        logits = logits.masked_fill(~keep, float("-inf"))
    q = torch.softmax(logits, dim=-1)
    drafts = [int(torch.multinomial(row, 1, generator=gen)) for row in q]
    return q, drafts


@pytest.mark.parametrize("support,encoding", [(7, "sparse"), (None, "dense")])
def test_round_trip_is_bit_exact_through_msgpack_and_json(support, encoding):
    q, drafts = _q(3, 4096, support=support)
    payload = encode_mtp_draft_q(q, drafts)
    assert payload["encoding"] == encoding
    for wire in (
        lambda p: msgpack.unpackb(msgpack.packb(p, use_bin_type=True), raw=False),
        lambda p: json.loads(json.dumps(p)),
    ):
        decoded = decode_mtp_draft_q(wire(payload), num_depths=3, vocab=4096, drafts=drafts)
        assert decoded.dtype == torch.float32
        assert torch.equal(decoded, q)
    if encoding == "sparse":
        # 3 depths x 7 entries x 8 bytes, base64: far below the 48 KB dense encoding.
        assert len(payload["indices"]) + len(payload["values"]) < 400


@pytest.mark.parametrize(
    "change,match",
    [
        (dict(num_depths=2), "local MTP configuration"),
        (dict(vocab=4095), "local MTP configuration"),
        (dict(drafts="other"), "different drafts"),
    ],
)
def test_decode_rejects_incompatible_payloads(change, match):
    q, drafts = _q(3, 4096, support=5)
    payload = encode_mtp_draft_q(q, drafts)
    kwargs = dict(num_depths=3, vocab=4096, drafts=drafts)
    if change.get("drafts") == "other":
        kwargs["drafts"] = [(drafts[0] + 1) % 4096, *drafts[1:]]
    else:
        kwargs.update(change)
    with pytest.raises(ValueError, match=match):
        decode_mtp_draft_q(payload, **kwargs)


def test_decode_rejects_a_draft_outside_its_q_support():
    q, drafts = _q(2, 64, support=3)
    outside = int((q[0] == 0).nonzero()[0])
    payload = encode_mtp_draft_q(q, [outside, drafts[1]])
    with pytest.raises(ValueError, match="zero probability"):
        decode_mtp_draft_q(payload, num_depths=2, vocab=64, drafts=[outside, drafts[1]])


def _decode_engine(mode, shape=(2, 64), num_speculative_tokens=2):
    return SimpleNamespace(
        context=SimpleNamespace(num_speculative_tokens=num_speculative_tokens),
        controller=SimpleNamespace(
            mtp_verification=mode, mtp_draft_q_shape=lambda: shape if mode == "rejection" else None
        ),
    )


def _import(engine, kv_meta, resume_tokens):
    return InferenceStateHandoffMixin._import_handoff_mtp_draft_q(
        engine, 101, kv_meta, resume_tokens
    )


def test_decode_import_uses_q_only_when_both_sides_verify_by_rejection():
    q, drafts = _q(2, 64, support=4)
    resume = [5, *drafts]
    kv_meta = {MTP_VERIFICATION_KEY: "rejection", MTP_DRAFT_Q_KEY: encode_mtp_draft_q(q, drafts)}
    assert torch.equal(_import(_decode_engine("rejection"), kv_meta, resume), q)
    # Exact-match decode ignores q; a rejection decode without q (greedy, exact-match prefill)
    # falls back to exact match; a mode mismatch is tolerated and falls back too.
    assert _import(_decode_engine("exact_match"), kv_meta, resume) is None
    assert _import(_decode_engine("rejection"), {MTP_VERIFICATION_KEY: "rejection"}, resume) is None
    mismatched = dict(kv_meta, **{MTP_VERIFICATION_KEY: "exact_match"})
    assert _import(_decode_engine("rejection"), mismatched, resume) is None
    # Without MTP the field is never read.
    assert _import(_decode_engine("rejection", num_speculative_tokens=0), kv_meta, [5]) is None


def test_decode_import_rejects_a_q_for_a_different_mtp_depth():
    q, drafts = _q(3, 64, support=4)
    kv_meta = {MTP_VERIFICATION_KEY: "rejection", MTP_DRAFT_Q_KEY: encode_mtp_draft_q(q, drafts)}
    with pytest.raises(RuntimeError, match="Invalid MTP draft distributions"):
        _import(_decode_engine("rejection", shape=(2, 64)), kv_meta, [5, *drafts[:2]])
