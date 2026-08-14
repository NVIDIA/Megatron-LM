# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for the internal GDR backend adapter."""

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from megatron.core.ssm.gated_delta_net.internal_gdn_backend import chunk as backend


def _inputs():
    return {
        "q": torch.empty(1, 2, 3, 4),
        "k": torch.empty(1, 2, 3, 4),
        "v": torch.empty(1, 2, 3, 4),
        "g": torch.empty(1, 2, 3),
        "beta": torch.empty(1, 2, 3),
    }


def test_internal_backend_forwards_fla_compatible_arguments(monkeypatch):
    calls = []
    expected = (torch.empty(1), None)

    def implementation(**kwargs):
        calls.append(kwargs)
        return expected

    monkeypatch.setattr(backend, "_load_internal_chunk_gated_delta_rule", lambda: implementation)

    result = backend.chunk_gated_delta_rule(
        **_inputs(),
        scale=0.5,
        output_final_state=True,
        transpose_state_layout=True,
        custom_option="value",
    )

    assert result is expected
    assert len(calls) == 1
    assert calls[0]["scale"] == 0.5
    assert calls[0]["output_final_state"] is True
    assert calls[0]["state_v_first"] is True
    assert "transpose_state_layout" not in calls[0]
    assert calls[0]["custom_option"] == "value"


@pytest.mark.parametrize("option", ["use_beta_sigmoid_in_kernel", "allow_neg_eigval"])
def test_internal_backend_rejects_unsupported_semantics(option):
    with pytest.raises(ValueError, match="internal GDR backend does not support"):
        backend.chunk_gated_delta_rule(**_inputs(), **{option: True})


def test_internal_backend_rejects_conflicting_state_layout_options():
    with pytest.raises(ValueError, match="conflicts with transpose_state_layout"):
        backend.chunk_gated_delta_rule(
            **_inputs(), state_v_first=True, transpose_state_layout=False
        )


def test_fused_forward_package_exports_wrapper():
    from megatron.core.ssm.gated_delta_net.internal_gdn_backend.kernels import fused_gdr_fwd_cute

    assert fused_gdr_fwd_cute.chunk_gated_delta_rule_prefill_cute.__module__.endswith(".fused_fwd")


def test_vendored_fused_backward_body_matches_local_revision():
    root = Path(__file__).resolve().parents[3]
    kernel = (
        root
        / "megatron/core/ssm/gated_delta_net/internal_gdn_backend/kernels"
        / "fused_gdr_bwd_cute/kernel.py"
    )

    source = kernel.read_text()
    executable_body = source[source.index("from dataclasses import") :]

    assert hashlib.sha256(executable_body.encode()).hexdigest() == (
        "f1e0bab931b218ae368cba18db3e754cac09f1f1b2c0a86ea25da8b872b02768"
    )


def _implementation():
    pytest.importorskip("fla")
    from megatron.core.ssm.gated_delta_net.internal_gdn_backend import implementation

    return implementation


def test_fla_mode_bypasses_cutedsl_capability_checks(monkeypatch):
    implementation = _implementation()
    expected = (torch.empty(1), None)
    monkeypatch.setenv("MCORE_GDN_INTERNAL_BACKEND", "fla")
    monkeypatch.setattr(
        implementation,
        "_cutedsl_support_reason",
        lambda **_kwargs: pytest.fail("CuTe capability checks must not run in FLA mode"),
    )
    monkeypatch.setattr(implementation, "fla_chunk_gated_delta_rule", lambda **_kwargs: expected)

    assert implementation.chunk_gated_delta_rule(**_inputs()) is expected


def test_auto_mode_falls_back_for_unsupported_inputs(monkeypatch):
    implementation = _implementation()
    expected = (torch.empty(1), None)
    monkeypatch.setenv("MCORE_GDN_INTERNAL_BACKEND", "auto")
    monkeypatch.setattr(
        implementation, "_cutedsl_support_reason", lambda **_kwargs: "unsupported test input"
    )
    monkeypatch.setattr(implementation, "fla_chunk_gated_delta_rule", lambda **_kwargs: expected)

    assert implementation.chunk_gated_delta_rule(**_inputs()) is expected


def test_cute_mode_rejects_unsupported_inputs(monkeypatch):
    implementation = _implementation()
    monkeypatch.setenv("MCORE_GDN_INTERNAL_BACKEND", "cute")
    monkeypatch.setattr(
        implementation, "_cutedsl_support_reason", lambda **_kwargs: "unsupported test input"
    )

    with pytest.raises(RuntimeError, match="unsupported test input"):
        implementation.chunk_gated_delta_rule(**_inputs())


@pytest.mark.parametrize("scale, expected_scale", [(None, 0.5), (0.0, 0.0)])
def test_cute_mode_dispatches_to_local_autograd_function(monkeypatch, scale, expected_scale):
    implementation = _implementation()
    inputs = _inputs()
    expected = (torch.empty(1), None)
    calls = []

    def apply(*args):
        calls.append(args)
        return expected

    monkeypatch.setenv("MCORE_GDN_INTERNAL_BACKEND", "cute")
    monkeypatch.setattr(implementation, "_cutedsl_support_reason", lambda **_kwargs: None)
    monkeypatch.setattr(
        implementation, "InternalChunkGatedDeltaRuleFunction", SimpleNamespace(apply=apply)
    )

    assert implementation.chunk_gated_delta_rule(**inputs, scale=scale) is expected
    assert calls[0][5] == expected_scale


@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_cutedsl_backward_routes_supported_batch_to_fused_kernel(monkeypatch, batch_size):
    implementation = _implementation()
    shape = (batch_size, 64, 64, 128)
    scalar_shape = shape[:-1]
    q = torch.empty(shape, dtype=torch.bfloat16)
    inputs = {
        "q": q,
        "k": torch.empty_like(q),
        "v": torch.empty_like(q),
        "g": torch.empty(scalar_shape, dtype=torch.float32),
        "beta": torch.empty(scalar_shape, dtype=torch.float32),
        "A": torch.empty((*scalar_shape, 64), dtype=torch.bfloat16),
        "do": torch.empty_like(q),
    }
    expected = tuple(torch.empty(1) for _ in range(5))
    seen = {}

    def fused(**kwargs):
        seen.update(kwargs)
        return expected

    monkeypatch.setenv("MCORE_GDN_INTERNAL_BACKEND", "auto")
    monkeypatch.setattr(implementation, "_call_fused_gdr_bwd_cute", fused)
    monkeypatch.setattr(
        implementation,
        "_fla_backward",
        lambda **_kwargs: pytest.fail("supported shape must use fused backward"),
    )

    assert implementation._can_use_fused_bwd_forward(
        inputs["q"], inputs["k"], inputs["v"], inputs["g"], inputs["beta"], None, None
    )

    result = implementation._cutedsl_backward(
        **inputs, scale=128**-0.5, dht=None, cu_seqlens=None, chunk_indices=None
    )

    assert result is expected
    assert seen["q"] is q
    assert seen["dht"] is None


@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_fused_backward_adapter_packs_dense_batch(monkeypatch, batch_size):
    implementation = _implementation()
    from megatron.core.ssm.gated_delta_net.internal_gdn_backend.kernels import fused_gdr_bwd_cute

    shape = (batch_size, 64, 64, 128)
    scalar_shape = shape[:-1]
    q = torch.empty(shape, dtype=torch.bfloat16)
    k = torch.empty_like(q)
    v = torch.empty_like(q)
    g = torch.empty(scalar_shape, dtype=torch.bfloat16)
    beta = torch.empty_like(g)
    A = torch.empty((*scalar_shape, 64), dtype=torch.bfloat16)
    do = torch.empty_like(q)
    h = torch.empty((batch_size, 1, 64, 128, 128), dtype=torch.bfloat16)
    total_tokens = batch_size * 64
    packed = q.reshape(1, total_tokens, 64, 128)
    packed_scalar = g.reshape(1, total_tokens, 64)
    fused_outputs = (
        torch.full_like(packed, 1),
        torch.full_like(packed, 2),
        torch.full_like(packed, 3),
        torch.full(packed_scalar.shape, 4.0, dtype=torch.float32),
        torch.full(packed_scalar.shape, 5.0, dtype=torch.float32),
        torch.empty((batch_size, 64, 128, 128), dtype=torch.float32),
    )
    seen = {}

    def fused(**kwargs):
        seen.update(kwargs)
        return fused_outputs

    monkeypatch.setattr(implementation, "_recompute_fused_bwd_h", lambda **_kwargs: h)
    monkeypatch.setattr(fused_gdr_bwd_cute, "fused_gdr_bwd", fused)

    dq, dk, dv, db, dg = implementation._call_fused_gdr_bwd_cute(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        A=A,
        do=do,
        dht=None,
        scale=128**-0.5,
        cu_seqlens=None,
        chunk_indices=None,
    )

    assert seen["q"].shape == (1, total_tokens, 64, 128)
    assert seen["g"].dtype == torch.float32
    assert seen["beta"].dtype == torch.float32
    assert seen["a"].shape == (1, total_tokens, 64, 64)
    assert seen["h"].shape == (1, batch_size, 64, 128, 128)
    assert seen["dht"].shape == (batch_size, 64, 128, 128)
    assert torch.count_nonzero(seen["dht"]) == 0
    assert seen["cu_seqlens"].tolist() == [index * 64 for index in range(batch_size + 1)]
    assert dq.shape == q.shape and torch.all(dq == 1)
    assert dk.shape == k.shape and torch.all(dk == 2)
    assert dv.shape == v.shape and torch.all(dv == 3)
    assert db.shape == beta.shape and torch.all(db == 5)
    assert dg.shape == g.shape and torch.all(dg == 4)
    assert db.dtype == beta.dtype and dg.dtype == g.dtype


def test_fused_backward_accepts_arbitrary_packed_batch():
    implementation = _implementation()
    shape = (1, 256, 64, 128)
    scalar_shape = shape[:-1]
    q = torch.empty(shape, dtype=torch.bfloat16)
    cu_seqlens = torch.tensor([0, 64, 192, 256], dtype=torch.int32)

    reason = implementation._fused_bwd_support_reason(
        q=q,
        k=torch.empty_like(q),
        v=torch.empty_like(q),
        g=torch.empty(scalar_shape, dtype=torch.float32),
        beta=torch.empty(scalar_shape, dtype=torch.float32),
        A=torch.empty((*scalar_shape, 64), dtype=torch.bfloat16),
        do=torch.empty_like(q),
        dht=torch.empty((3, 64, 128, 128), dtype=torch.float32),
        cu_seqlens=cu_seqlens,
    )

    assert reason is None


def test_fused_backward_metadata_supports_arbitrary_batch():
    from megatron.core.ssm.gated_delta_net.internal_gdn_backend.kernels.fused_gdr_bwd_cute import (
        fused_bwd,
    )

    cu_seqlens = torch.tensor([0, 64, 192, 256], dtype=torch.int32)
    metadata = fused_bwd._prepare_varlen_metadata(cu_seqlens, 256, 64)

    assert metadata.num_sequences == 3
    assert metadata.num_chunks == 4
    assert metadata.uniform_sequence_length == 0
    assert metadata.chunk_offsets.tolist() == [0, 1, 3, 4]
