# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for the internal GDR backend adapter."""

import ast
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
        recompute_h=True,
    )

    assert result is expected
    assert len(calls) == 1
    assert calls[0]["scale"] == 0.5
    assert calls[0]["output_final_state"] is True
    assert calls[0]["state_v_first"] is True
    assert "transpose_state_layout" not in calls[0]
    assert calls[0]["custom_option"] == "value"
    assert calls[0]["recompute_h"] is True


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


@pytest.mark.parametrize("recompute_h", [False, True])
@pytest.mark.parametrize("scale, expected_scale", [(None, 0.5), (0.0, 0.0)])
def test_cute_mode_dispatches_to_local_autograd_function(
    monkeypatch, scale, expected_scale, recompute_h
):
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

    result = implementation.chunk_gated_delta_rule(**inputs, scale=scale, recompute_h=recompute_h)
    assert result is expected

    assert calls[0][5] == expected_scale
    assert calls[0][8] is recompute_h


@pytest.mark.parametrize("mode", ["auto", "cute"])
@pytest.mark.parametrize("recompute_h", [False, True])
def test_forward_save_h_policy_is_shared_by_fla_and_cute(monkeypatch, mode, recompute_h):
    implementation = _implementation()
    inputs = _inputs()
    q = inputs["q"]
    saved_h = torch.empty(1)
    seen = []

    def fla_forward(**kwargs):
        seen.append(kwargs["save_fused_bwd_state"])
        h = saved_h if kwargs["save_fused_bwd_state"] else None
        return kwargs["g"], q, torch.empty(1), h, None

    def cute_forward(**kwargs):
        seen.append(kwargs["save_fused_bwd_state"])
        h = saved_h if kwargs["save_fused_bwd_state"] else None
        return kwargs["g"], q, torch.empty(1), h, None, None

    context = SimpleNamespace()
    context.save_for_backward = lambda *args: setattr(context, "saved_tensors", args)
    monkeypatch.setenv("MCORE_GDN_INTERNAL_BACKEND", mode)
    monkeypatch.setattr(implementation, "_can_use_fused_bwd_forward", lambda *_args: True)
    monkeypatch.setattr(implementation, "_fla_forward_for_fused_bwd", fla_forward)
    monkeypatch.setattr(implementation, "_cutedsl_forward", cute_forward)

    forward = implementation.InternalChunkGatedDeltaRuleFunction.forward
    while hasattr(forward, "__wrapped__"):
        forward = forward.__wrapped__
    output, final_state = forward(
        context,
        inputs["q"],
        inputs["k"],
        inputs["v"],
        inputs["g"],
        inputs["beta"],
        0.5,
        None,
        None,
        recompute_h,
    )

    assert output is q and final_state is None
    assert seen == [not recompute_h]
    assert (context.saved_tensors[6] is None) is recompute_h


@pytest.mark.parametrize("save_fused_bwd_state", [False, True])
def test_fla_forward_can_save_fused_backward_h(monkeypatch, save_fused_bwd_state):
    implementation = _implementation()
    shape = (1, 64, 3, 4)
    inputs = {
        "q": torch.empty(shape),
        "k": torch.empty(shape),
        "v": torch.empty(shape),
        "g": torch.empty(shape[:-1]),
        "beta": torch.empty(shape[:-1]),
    }
    raw_h = torch.randn(1, 1, 3, 4, 4, dtype=torch.float32)
    monkeypatch.setattr(implementation, "chunk_local_cumsum", lambda g, **_kwargs: g)
    monkeypatch.setattr(
        implementation,
        "chunk_gated_delta_rule_fwd_intra",
        lambda **kwargs: (kwargs["k"], kwargs["v"], torch.empty(1)),
    )
    monkeypatch.setattr(
        implementation, "chunk_gated_delta_rule_fwd_h", lambda **kwargs: (raw_h, kwargs["u"], None)
    )
    monkeypatch.setattr(implementation, "chunk_fwd_o", lambda **kwargs: kwargs["q"])

    _g, output, _A, saved_h, _chunk_indices = implementation._fla_forward_for_fused_bwd(
        **inputs,
        scale=0.5,
        cu_seqlens=None,
        cu_seqlens_cpu=None,
        save_fused_bwd_state=save_fused_bwd_state,
    )

    assert output is inputs["q"]
    if save_fused_bwd_state:
        assert saved_h.shape == (1, 3, 4, 4)
        assert saved_h.dtype == torch.bfloat16
        assert saved_h.is_contiguous()
        torch.testing.assert_close(saved_h, raw_h.reshape_as(saved_h).to(torch.bfloat16))
    else:
        assert saved_h is None


@pytest.mark.parametrize("save_fused_bwd_state", [False, True])
def test_cutedsl_forward_controls_fused_backward_h(monkeypatch, save_fused_bwd_state):
    implementation = _implementation()
    from megatron.core.ssm.gated_delta_net.internal_gdn_backend.kernels import fused_gdr_fwd_cute

    shape = (1, 64, 3, 4)
    q = torch.empty(shape)
    inputs = {
        "q": q,
        "k": torch.empty_like(q),
        "v": torch.empty_like(q),
        "g": torch.empty(shape[:-1]),
        "beta": torch.empty(shape[:-1]),
    }
    seen = {}

    def launcher(**kwargs):
        seen.update(kwargs)

    monkeypatch.setattr(implementation, "chunk_local_cumsum", lambda g, **_kwargs: g)
    monkeypatch.setattr(fused_gdr_fwd_cute, "chunk_gated_delta_rule_prefill_cute", launcher)

    _g, _output, _A, saved_h, _chunk_indices, _chunk_offsets = implementation._cutedsl_forward(
        **inputs,
        scale=0.5,
        cu_seqlens=None,
        cu_seqlens_cpu=None,
        save_fused_bwd_state=save_fused_bwd_state,
    )

    assert seen["output_h"] is saved_h
    assert seen["output_g"].shape == (64, 3)
    assert seen["output_g"].dtype == torch.float32
    assert seen["g"] is not seen["output_g"]
    assert seen["gate_is_log_decay"] is True
    assert seen.get("gate_is_log_cumsum", False) is False
    assert seen["checkpoint_every_n_tokens"] == (64 if save_fused_bwd_state else 0)
    assert seen["assume_valid_cu_seqlens"] is True
    assert (saved_h is None) is not save_fused_bwd_state


def test_dense_chunk_metadata_cache_reuses_offsets(monkeypatch):
    implementation = _implementation()

    implementation._clear_dense_chunk_metadata_cache_for_test()
    arange_calls = []
    original_arange = torch.arange

    def arange(*args, **kwargs):
        arange_calls.append((args, kwargs))
        return original_arange(*args, **kwargs)

    monkeypatch.setattr(implementation.torch, "arange", arange)

    first = implementation._dense_chunk_metadata(2, 128, torch.device("cpu"))
    second = implementation._dense_chunk_metadata(2, 128, torch.device("cpu"))

    assert second is first
    assert len(arange_calls) == 2
    assert first.cu_seqlens.tolist() == [0, 128, 256]
    assert first.chunk_offsets.tolist() == [0, 2, 4]


def test_dense_chunk_metadata_cache_evicts_least_recently_used(monkeypatch):
    implementation = _implementation()

    implementation._clear_dense_chunk_metadata_cache_for_test()
    monkeypatch.setattr(implementation, "_DENSE_CHUNK_METADATA_CACHE_LIMIT", 2)

    first = implementation._dense_chunk_metadata(1, 64, torch.device("cpu"))
    second = implementation._dense_chunk_metadata(2, 64, torch.device("cpu"))
    assert implementation._dense_chunk_metadata(1, 64, torch.device("cpu")) is first

    third = implementation._dense_chunk_metadata(3, 64, torch.device("cpu"))

    cached_ids = {id(metadata) for metadata in implementation._dense_chunk_metadata_cache.values()}
    assert cached_ids == {id(first), id(third)}
    assert id(second) not in cached_ids


def test_cutedsl_forward_uses_cached_dense_chunk_offsets(monkeypatch):
    implementation = _implementation()
    from megatron.core.ssm.gated_delta_net.internal_gdn_backend.kernels import fused_gdr_fwd_cute

    implementation._clear_dense_chunk_metadata_cache_for_test()
    shape = (2, 64, 3, 4)
    q = torch.empty(shape)
    inputs = {
        "q": q,
        "k": torch.empty_like(q),
        "v": torch.empty_like(q),
        "g": torch.empty(shape[:-1]),
        "beta": torch.empty(shape[:-1]),
    }
    dense_metadata = implementation._dense_chunk_metadata(2, 64, q.device)
    seen = {}

    def launcher(**kwargs):
        seen.update(kwargs)

    monkeypatch.setattr(fused_gdr_fwd_cute, "chunk_gated_delta_rule_prefill_cute", launcher)

    _g, _output, _A, _saved_h, _chunk_indices, chunk_offsets = implementation._cutedsl_forward(
        **inputs,
        scale=0.5,
        cu_seqlens=None,
        cu_seqlens_cpu=None,
        save_fused_bwd_state=True,
    )

    assert seen["cu_seqlens"] is dense_metadata.cu_seqlens
    assert seen["checkpoint_cu_starts"] is dense_metadata.chunk_offsets
    assert chunk_offsets is dense_metadata.chunk_offsets


def test_fused_forward_rejects_conflicting_gate_modes():
    from megatron.core.ssm.gated_delta_net.internal_gdn_backend.kernels.fused_gdr_fwd_cute import (
        chunk_gated_delta_rule_prefill_cute,
    )

    with pytest.raises(ValueError, match="mutually exclusive"):
        chunk_gated_delta_rule_prefill_cute(
            q=torch.empty(1, 1, 128),
            k=torch.empty(1, 1, 128),
            v=torch.empty(1, 1, 128),
            cu_seqlens=torch.tensor([0, 1], dtype=torch.int32),
            gate_is_log_cumsum=True,
            gate_is_log_decay=True,
        )


def test_cutedsl_forward_trusts_validated_varlen_cu_seqlens(monkeypatch):
    implementation = _implementation()
    from megatron.core.ssm.gated_delta_net.internal_gdn_backend.kernels import fused_gdr_fwd_cute

    shape = (1, 128, 3, 4)
    q = torch.empty(shape)
    inputs = {
        "q": q,
        "k": torch.empty_like(q),
        "v": torch.empty_like(q),
        "g": torch.empty(shape[:-1]),
        "beta": torch.empty(shape[:-1]),
    }
    cu_seqlens = torch.tensor([0, 64, 128], dtype=torch.int32)
    seen = {}

    def launcher(**kwargs):
        seen.update(kwargs)

    monkeypatch.setattr(implementation, "prepare_chunk_indices", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(implementation, "chunk_local_cumsum", lambda g, **_kwargs: g)
    monkeypatch.setattr(fused_gdr_fwd_cute, "chunk_gated_delta_rule_prefill_cute", launcher)

    implementation._cutedsl_forward(
        **inputs,
        scale=0.5,
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cu_seqlens,
        save_fused_bwd_state=False,
    )

    assert seen["cu_seqlens"].dtype == torch.int32
    assert seen["assume_valid_cu_seqlens"] is True


def test_cutedsl_forward_reuses_packed_chunk_metadata(monkeypatch):
    implementation = _implementation()
    from megatron.core.ssm.gated_delta_net.internal_gdn_backend.kernels import fused_gdr_fwd_cute

    implementation._clear_packed_chunk_metadata_cache_for_test()
    shape = (1, 128, 3, 4)
    q = torch.empty(shape)
    inputs = {
        "q": q,
        "k": torch.empty_like(q),
        "v": torch.empty_like(q),
        "g": torch.empty(shape[:-1]),
        "beta": torch.empty(shape[:-1]),
    }
    cu_seqlens = torch.tensor([0, 64, 128], dtype=torch.int32)
    chunk_indices = torch.tensor([[0, 0], [1, 0]], dtype=torch.int32)
    prepare_calls = []
    seen = {}

    def prepare(cu_arg, *_args, **_kwargs):
        prepare_calls.append(cu_arg)
        return chunk_indices

    def launcher(**kwargs):
        seen.update(kwargs)

    monkeypatch.setattr(implementation, "prepare_chunk_indices", prepare)
    monkeypatch.setattr(implementation, "chunk_local_cumsum", lambda g, **_kwargs: g)
    monkeypatch.setattr(fused_gdr_fwd_cute, "chunk_gated_delta_rule_prefill_cute", launcher)

    metadata = implementation._packed_chunk_metadata(cu_seqlens, cu_seqlens)
    _g, _output, _A, _saved_h, actual_indices, chunk_offsets = implementation._cutedsl_forward(
        **inputs,
        scale=0.5,
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cu_seqlens,
        packed_metadata=metadata,
        save_fused_bwd_state=True,
    )

    assert len(prepare_calls) == 1
    assert prepare_calls[0] is cu_seqlens
    assert actual_indices is chunk_indices
    assert torch.equal(chunk_offsets, torch.tensor([0, 1, 2], dtype=torch.int32))
    assert seen["checkpoint_cu_starts"] is chunk_offsets
    assert seen["output_g"].shape == (128, 3)
    assert seen["gate_is_log_decay"] is True


def test_packed_chunk_metadata_cache_is_stream_scoped(monkeypatch):
    implementation = _implementation()

    implementation._clear_packed_chunk_metadata_cache_for_test()
    cu_seqlens = torch.tensor([0, 64, 128], dtype=torch.int32)
    stream_keys = iter([(0, 1), (0, 2), (0, 1)])
    prepare_calls = []

    def prepare(cu_arg, *_args, **_kwargs):
        prepare_calls.append(cu_arg)
        return torch.full((2, 2), len(prepare_calls), dtype=torch.int32)

    monkeypatch.setattr(
        implementation, "_current_stream_cache_key", lambda _tensor: next(stream_keys)
    )
    monkeypatch.setattr(implementation, "prepare_chunk_indices", prepare)

    first = implementation._packed_chunk_metadata(cu_seqlens, cu_seqlens)
    second = implementation._packed_chunk_metadata(cu_seqlens, cu_seqlens)
    third = implementation._packed_chunk_metadata(cu_seqlens, cu_seqlens)

    assert len(prepare_calls) == 2
    assert second is not first
    assert third is first


def test_packed_chunk_metadata_cache_tracks_cpu_metadata_owner(monkeypatch):
    implementation = _implementation()

    implementation._clear_packed_chunk_metadata_cache_for_test()
    cu_seqlens = torch.tensor([0, 64, 128], dtype=torch.int32)
    cu_seqlens_cpu_a = torch.tensor([0, 64, 128], dtype=torch.int32)
    cu_seqlens_cpu_b = torch.tensor([0, 64, 128], dtype=torch.int32)
    prepare_calls = []

    def prepare(_cu_arg, *_args, **kwargs):
        prepare_calls.append(kwargs["cu_seqlens_cpu"])
        return torch.full((2, 2), len(prepare_calls), dtype=torch.int32)

    monkeypatch.setattr(implementation, "_current_stream_cache_key", lambda _tensor: None)
    monkeypatch.setattr(implementation, "prepare_chunk_indices", prepare)

    first = implementation._packed_chunk_metadata(cu_seqlens, cu_seqlens_cpu_a)
    second = implementation._packed_chunk_metadata(cu_seqlens, cu_seqlens_cpu_b)
    third = implementation._packed_chunk_metadata(cu_seqlens, cu_seqlens_cpu_b)

    assert prepare_calls == [cu_seqlens_cpu_a, cu_seqlens_cpu_b]
    assert second is not first
    assert third is second


def test_cutedsl_forward_uses_kernel_gate_cumsum_side_output():
    package = Path(__file__).parents[3] / "megatron/core/ssm/gated_delta_net/internal_gdn_backend"
    implementation_source = (package / "implementation.py").read_text()
    wrapper_source = (package / "kernels/fused_gdr_fwd_cute/fused_fwd.py").read_text()
    kernel_source = (package / "kernels/fused_gdr_fwd_cute/kernel.py").read_text()

    assert "flat_raw_g = _reshape_bth_to_th(g.detach())" in implementation_source
    assert "output_g=flat_g" in implementation_source
    assert "gate_is_log_decay=True" in implementation_source
    assert "output_g: Optional[torch.Tensor] = None" in wrapper_source
    assert "tGrGate[i] = tGrGate[i] * 1.4426950408889634" in kernel_source
    assert "cute.copy(tiled_copy_gate_g2r, tGrGate, tGgGateOut" in kernel_source


def test_cutedsl_backward_routes_supported_batch_to_fused_kernel(monkeypatch):
    batch_size = 2
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


def test_fused_backward_zero_dht_cache_is_stream_scoped(monkeypatch):
    implementation = _implementation()

    implementation._fused_bwd_zero_dht_cache.clear()
    stream_keys = iter([(0, 1), (0, 2), (0, 1)])
    monkeypatch.setattr(
        implementation, "_current_stream_cache_key", lambda _device: next(stream_keys)
    )

    first = implementation._fused_bwd_zero_dht(torch.device("cpu"), 1)
    second = implementation._fused_bwd_zero_dht(torch.device("cpu"), 1)
    third = implementation._fused_bwd_zero_dht(torch.device("cpu"), 1)

    assert second is not first
    assert third is first


@pytest.mark.parametrize("use_saved_h", [False, True])
def test_fused_backward_adapter_packs_dense_batch(monkeypatch, use_saved_h):
    batch_size = 3
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
    h = torch.empty((batch_size, 64, 128, 128), dtype=torch.bfloat16)
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

    recompute_calls = []

    def fused(**kwargs):
        seen.update(kwargs)
        return fused_outputs

    def recompute(**_kwargs):
        recompute_calls.append(True)
        return h

    implementation._clear_dense_chunk_metadata_cache_for_test()
    dense_metadata = implementation._dense_chunk_metadata(batch_size, 64, q.device)
    monkeypatch.setattr(implementation, "_recompute_fused_bwd_h", recompute)
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
        h=h if use_saved_h else None,
    )

    assert seen["q"].shape == (1, total_tokens, 64, 128)
    assert recompute_calls == ([] if use_saved_h else [True])
    assert seen["g"].dtype == torch.float32
    assert seen["beta"].dtype == torch.float32
    assert seen["a"].shape == (1, total_tokens, 64, 64)
    assert seen["h"].shape == (1, batch_size, 64, 128, 128)
    assert seen["dht"].shape == (batch_size, 64, 128, 128)
    assert torch.count_nonzero(seen["dht"]) == 0
    assert seen["cu_seqlens"] is dense_metadata.cu_seqlens
    assert seen["chunk_offsets"] is dense_metadata.chunk_offsets
    assert seen["cu_seqlens"].tolist() == [index * 64 for index in range(batch_size + 1)]
    assert dq.shape == q.shape and torch.all(dq == 1)
    assert dk.shape == k.shape and torch.all(dk == 2)
    assert dv.shape == v.shape and torch.all(dv == 3)
    assert db.shape == beta.shape and torch.all(db == 5)
    assert dg.shape == g.shape and torch.all(dg == 4)
    assert db.dtype == beta.dtype and dg.dtype == g.dtype


def test_fused_backward_adapter_forwards_chunk_offsets(monkeypatch):
    implementation = _implementation()
    from megatron.core.ssm.gated_delta_net.internal_gdn_backend.kernels import fused_gdr_bwd_cute

    shape = (1, 128, 64, 128)
    scalar_shape = shape[:-1]
    q = torch.empty(shape, dtype=torch.bfloat16)
    g = torch.empty(scalar_shape, dtype=torch.float32)
    h = torch.empty((2, 64, 128, 128), dtype=torch.bfloat16)
    cu_seqlens = torch.tensor([0, 64, 128], dtype=torch.int32)
    chunk_offsets = torch.tensor([0, 1, 2], dtype=torch.int32)
    fused_outputs = (
        torch.empty_like(q),
        torch.empty_like(q),
        torch.empty_like(q),
        torch.empty_like(g),
        torch.empty_like(g),
        torch.empty((2, 64, 128, 128), dtype=torch.float32),
    )
    seen = {}

    def fused(**kwargs):
        seen.update(kwargs)
        return fused_outputs

    monkeypatch.setattr(fused_gdr_bwd_cute, "fused_gdr_bwd", fused)

    implementation._call_fused_gdr_bwd_cute(
        q=q,
        k=torch.empty_like(q),
        v=torch.empty_like(q),
        g=g,
        beta=torch.empty_like(g),
        A=torch.empty((*scalar_shape, 64), dtype=torch.bfloat16),
        do=torch.empty_like(q),
        dht=None,
        scale=128**-0.5,
        cu_seqlens=cu_seqlens,
        chunk_indices=None,
        chunk_offsets=chunk_offsets,
        h=h,
    )

    assert seen["cu_seqlens"] is cu_seqlens
    assert seen["chunk_offsets"] is chunk_offsets


def test_gdn_pre_gdr_producers_emit_fp32_beta():
    package = Path(__file__).parents[3] / "megatron/core"
    gdn_source = (package / "ssm/gated_delta_net/gdn.py").read_text()
    fused_pre_source = (package / "fusions/fused_pre_gated_delta_rule.py").read_text()

    assert "beta = beta.sigmoid().to(torch.float32)" in gdn_source
    assert (
        "beta = torch.empty(batch, seq_len, num_value_heads, dtype=torch.float32"
        in fused_pre_source
    )


def test_bf16_output_h_initialization_moves_to_kernel():
    package = (
        Path(__file__).parents[3]
        / "megatron/core/ssm/gated_delta_net/internal_gdn_backend/kernels"
        / "fused_gdr_fwd_cute"
    )
    wrapper_source = (package / "fused_fwd.py").read_text()
    kernel_source = (package / "kernel.py").read_text()

    assert "output_h is not None and output_h.dtype != torch.bfloat16" in wrapper_source
    assert "tRG_rState.fill(0.0)" in kernel_source
    assert "+ (chunk_idx * self.b_t) // checkpoint_every_n_tokens" in kernel_source


def test_fused_backward_varlen_metadata_avoids_host_sync_for_device_offsets():
    from megatron.core.ssm.gated_delta_net.internal_gdn_backend.kernels.fused_gdr_bwd_cute import (
        fused_bwd,
    )

    fused_bwd._clear_metadata_cache_for_test()
    cu_seqlens = torch.empty(3, dtype=torch.int32, device="meta")

    metadata = fused_bwd._prepare_varlen_metadata(cu_seqlens, total_tokens=128, chunk_size=64)

    assert metadata.chunk_offsets.device.type == "meta"
    assert metadata.num_sequences == 2
    assert metadata.num_chunks == 2
    assert metadata.uniform_sequence_length == 0


def test_fused_backward_varlen_metadata_cache_is_stream_scoped(monkeypatch):
    from megatron.core.ssm.gated_delta_net.internal_gdn_backend.kernels.fused_gdr_bwd_cute import (
        fused_bwd,
    )

    fused_bwd._clear_metadata_cache_for_test()
    cu_seqlens = torch.tensor([0, 64, 128], dtype=torch.int32)
    stream_keys = iter([(0, 1), (0, 2), (0, 1)])
    monkeypatch.setattr(
        fused_bwd, "_current_stream_cache_key", lambda _tensor: next(stream_keys)
    )

    first = fused_bwd._prepare_varlen_metadata(cu_seqlens, total_tokens=128, chunk_size=64)
    second = fused_bwd._prepare_varlen_metadata(cu_seqlens, total_tokens=128, chunk_size=64)
    third = fused_bwd._prepare_varlen_metadata(cu_seqlens, total_tokens=128, chunk_size=64)

    assert second is not first
    assert third is first


def test_fused_backward_varlen_metadata_reuses_supplied_chunk_offsets():
    from megatron.core.ssm.gated_delta_net.internal_gdn_backend.kernels.fused_gdr_bwd_cute import (
        fused_bwd,
    )

    fused_bwd._clear_metadata_cache_for_test()
    cu_seqlens = torch.empty(3, dtype=torch.int32, device="meta")
    chunk_offsets = torch.empty(3, dtype=torch.int32, device="meta")

    metadata = fused_bwd._prepare_varlen_metadata(
        cu_seqlens, total_tokens=128, chunk_size=64, chunk_offsets=chunk_offsets
    )

    assert metadata.chunk_offsets is chunk_offsets
    assert metadata.num_sequences == 2
    assert metadata.num_chunks == 2
    assert metadata.uniform_sequence_length == 0


def test_fused_backward_support_reason_trusts_device_offsets_only_when_requested():
    implementation = _implementation()
    shape = (1, 128, 64, 128)
    scalar_shape = shape[:-1]
    q = torch.empty(shape, dtype=torch.bfloat16)
    kwargs = {
        "q": q,
        "k": torch.empty_like(q),
        "v": torch.empty_like(q),
        "g": torch.empty(scalar_shape, dtype=torch.float32),
        "beta": torch.empty(scalar_shape, dtype=torch.float32),
        "A": torch.empty((*scalar_shape, 64), dtype=torch.bfloat16),
        "do": torch.empty_like(q),
        "dht": None,
        "cu_seqlens": torch.empty(3, dtype=torch.int32, device="meta"),
    }

    assert (
        implementation._fused_bwd_support_reason(**kwargs)
        == "packed cu_seqlens host metadata is required for validation"
    )
    assert implementation._fused_bwd_support_reason(
        **kwargs, trust_device_cu_seqlens=True
    ) is None


@pytest.mark.parametrize(
    "dtype, batch_size, seqlen, offsets, expected_reason",
    [
        (torch.bfloat16, 2, 64, None, None),
        (torch.float16, 2, 64, None, "fused backward requires bf16 inputs"),
        (torch.bfloat16, 1, 256, [0, 64, 192, 256], None),
    ],
)
def test_fused_backward_shape_and_dtype_contract(
    dtype, batch_size, seqlen, offsets, expected_reason
):
    implementation = _implementation()
    shape = (batch_size, seqlen, 64, 128)
    scalar_shape = shape[:-1]
    q = torch.empty(shape, dtype=dtype)
    cu_seqlens = None if offsets is None else torch.tensor(offsets, dtype=torch.int32)
    num_sequences = batch_size if offsets is None else len(offsets) - 1

    reason = implementation._fused_bwd_support_reason(
        q=q,
        k=torch.empty_like(q),
        v=torch.empty_like(q),
        g=torch.empty(scalar_shape, dtype=torch.float32),
        beta=torch.empty(scalar_shape, dtype=torch.float32),
        A=torch.empty((*scalar_shape, 64), dtype=torch.bfloat16),
        do=torch.empty_like(q),
        dht=torch.empty((num_sequences, 64, 128, 128), dtype=torch.float32),
        cu_seqlens=cu_seqlens,
    )

    assert reason == expected_reason


def test_fused_backward_kernel_uses_named_layout_contracts():
    package = (
        Path(__file__).parents[3]
        / "megatron/core/ssm/gated_delta_net/internal_gdn_backend/kernels"
        / "fused_gdr_bwd_cute"
    )
    kernel_source = (package / "kernel.py").read_text()
    for positional_wiring in (
        "mma_variants[",
        "packed_layouts[",
        "tma_inputs[",
    ):
        assert positional_wiring not in kernel_source

    layouts_tree = ast.parse((package / "layouts.py").read_text())
    assignments = {
        node.targets[0].id: node.value
        for node in layouts_tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
    }
    expected_fields = {
        "MmaVariantBindings": 10,
        "MmaOperationBindings": 19,
        "TmaDescriptorBundle": 9,
    }
    for node in layouts_tree.body:
        if isinstance(node, ast.ClassDef) and node.name in expected_fields:
            fields = [item for item in node.body if isinstance(item, ast.AnnAssign)]
            assert len(fields) == expected_fields.pop(node.name)
    assert not expected_fields

    variant_specs = assignments["MMA_VARIANT_SPECS"].elts
    operation_specs = assignments["MMA_OPERATION_SPECS"].elts
    variant_names = {ast.literal_eval(spec.args[0]) for spec in variant_specs}
    operation_names = {ast.literal_eval(spec.args[0]) for spec in operation_specs}
    canonical_sources = ast.literal_eval(assignments["_CANONICAL_LAYOUT_SOURCES"])
    assert len(variant_names) == len(variant_specs) == 10
    assert len(operation_names) == len(operation_specs) == 19
    for spec in operation_specs:
        _, _, variant, a_view, b_view = map(ast.literal_eval, spec.args)
        assert variant in variant_names
        assert a_view in canonical_sources
        assert b_view in canonical_sources

    storage_tree = ast.parse((package / "storage.py").read_text())
    storage_assignments = {
        node.targets[0].id: node.value
        for node in storage_tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
    }
    shared_storage = next(
        node
        for node in storage_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "SharedStorage"
    )
    fields = [item for item in shared_storage.body if isinstance(item, ast.AnnAssign)]
    assert len(fields) == 79

    allocations = ast.literal_eval(storage_assignments["TMEM_ALLOCATION_COLUMNS"])
    ranges = ast.literal_eval(storage_assignments["TMEM_RANGES"])
    assert allocations == (256, 128, 32, 64)
    assert sum(allocations) == 480
    for index, lhs in enumerate(ranges):
        for rhs in ranges[index + 1 :]:
            columns_overlap = max(lhs[1], rhs[1]) < min(lhs[2], rhs[2])
            phases_overlap = max(lhs[3], rhs[3]) < min(lhs[4], rhs[4])
            assert not (columns_overlap and phases_overlap), (lhs[0], rhs[0])
