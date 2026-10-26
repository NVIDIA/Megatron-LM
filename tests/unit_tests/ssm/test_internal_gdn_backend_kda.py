# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for the internal KDA adapter (FLA Triton kernels + in-tree CP pre-processing)."""

from types import SimpleNamespace

import pytest
import torch

from megatron.core.ssm.gated_delta_net.internal_gdn_backend import chunk as backend

_SHAPE = (1, 64, 8, 128)


def _adapter():
    pytest.importorskip("fla")
    from megatron.core.ssm.gated_delta_net.internal_gdn_backend import kda as adapter

    return adapter


def _inputs(shape=_SHAPE):
    batch, seqlen, heads, _head_dim = shape
    return {
        "q": torch.empty(shape, dtype=torch.bfloat16),
        "k": torch.empty(shape, dtype=torch.bfloat16),
        "v": torch.empty(shape, dtype=torch.bfloat16),
        "g": torch.empty(shape, dtype=torch.bfloat16),
        "beta": torch.empty((batch, seqlen, heads), dtype=torch.float32),
    }


def _gate_params(shape=_SHAPE):
    heads, head_dim = shape[2], shape[3]
    return {
        "A_log": torch.empty(heads, dtype=torch.float32),
        "dt_bias": torch.empty(heads * head_dim, dtype=torch.float32),
    }


def _cp_context(local_tokens=_SHAPE[1]):
    return SimpleNamespace(
        group=object(),
        cu_seqlens=torch.tensor([0, local_tokens], dtype=torch.int32),
        cu_seqlens_cpu=torch.tensor([0, local_tokens], dtype=torch.long),
    )


def _unwrap(function):
    while hasattr(function, "__wrapped__"):
        function = function.__wrapped__
    return function


def test_package_exports_fla_free_chunk_kda():
    from megatron.core.ssm.gated_delta_net.internal_gdn_backend import chunk_kda

    assert chunk_kda is backend.chunk_kda


def test_public_chunk_kda_forwards_fla_compatible_arguments(monkeypatch):
    calls = []
    expected = (torch.empty(1), None)

    def implementation(**kwargs):
        calls.append(kwargs)
        return expected

    monkeypatch.setattr(backend, "_load_internal_chunk_kda", lambda: implementation)

    gate = _gate_params()
    validated = torch.tensor([[0, 0]], dtype=torch.int32)
    result = backend.chunk_kda(
        **_inputs(),
        scale=0.5,
        use_gate_in_kernel=True,
        safe_gate=True,
        lower_bound=-5.0,
        validated_chunk_indices=validated,
        custom_option="value",
        **gate,
    )

    assert result is expected
    assert len(calls) == 1
    assert calls[0]["scale"] == 0.5
    assert calls[0]["use_gate_in_kernel"] is True
    assert calls[0]["safe_gate"] is True
    assert calls[0]["lower_bound"] == -5.0
    assert calls[0]["validated_chunk_indices"] is validated
    assert calls[0]["A_log"] is gate["A_log"]
    assert calls[0]["dt_bias"] is gate["dt_bias"]
    assert calls[0]["custom_option"] == "value"


def test_adapter_rejects_fla_releases_without_kda_recompute_contract(monkeypatch):
    adapter = _adapter()

    def legacy_fwd_intra(q, k, v, gk=None, beta=None, scale=None, cu_seqlens=None):
        del q, k, v, gk, beta, scale, cu_seqlens

    monkeypatch.setattr(adapter, "chunk_kda_fwd_intra", legacy_fwd_intra)

    with pytest.raises(ImportError, match="flash-linear-attention >= 0.5.1"):
        adapter._check_fla_kda_contract()


def test_fla_mode_routes_kda_to_fla(monkeypatch):
    adapter = _adapter()
    expected = (torch.empty(1), None)
    seen = {}

    def fla(*args, **kwargs):
        seen["args"] = args
        seen.update(kwargs)
        return expected

    monkeypatch.setenv("MCORE_GDN_INTERNAL_BACKEND", "fla")
    monkeypatch.setattr(
        adapter,
        "_kda_internal_support_reason",
        lambda **_kwargs: pytest.fail("adapter checks must not run in FLA mode"),
    )
    monkeypatch.setattr(adapter, "fla_chunk_kda", fla)

    inputs = _inputs()
    gate = _gate_params()
    cp_context = _cp_context()
    result = adapter.chunk_kda(
        **inputs,
        use_gate_in_kernel=True,
        safe_gate=True,
        lower_bound=-5.0,
        cp_context=cp_context,
        **gate,
    )

    assert result is expected
    assert seen["args"] == (inputs["q"], inputs["k"], inputs["v"], inputs["g"], inputs["beta"])
    assert seen["cp_context"] is cp_context
    assert seen["A_log"] is gate["A_log"]
    assert seen["dt_bias"] is gate["dt_bias"]
    assert seen["safe_gate"] is True and seen["lower_bound"] == -5.0


@pytest.mark.parametrize("mode", ["auto", "cute"])
def test_kda_without_cp_context_is_fla_verbatim(monkeypatch, mode):
    adapter = _adapter()
    expected = (torch.empty(1), None)
    seen = {}

    def fla(*_args, **kwargs):
        seen.update(kwargs)
        return expected

    monkeypatch.setenv("MCORE_GDN_INTERNAL_BACKEND", mode)
    monkeypatch.setattr(adapter, "fla_chunk_kda", fla)
    monkeypatch.setattr(
        adapter,
        "InternalChunkKDAFunction",
        SimpleNamespace(apply=lambda *_args: pytest.fail("non-CP calls must not use the adapter")),
    )

    cu_seqlens = torch.tensor([0, 64], dtype=torch.int32)
    result = adapter.chunk_kda(
        **_inputs(),
        use_gate_in_kernel=True,
        cu_seqlens=cu_seqlens,
        cp_context=None,
        **_gate_params(),
    )

    assert result is expected
    assert seen["cu_seqlens"] is cu_seqlens
    assert seen["cp_context"] is None


def test_auto_mode_falls_back_to_fla_for_unsupported_kda_options(monkeypatch):
    adapter = _adapter()
    expected = (torch.empty(1), None)
    seen = {}

    def fla(*_args, **kwargs):
        seen.update(kwargs)
        return expected

    monkeypatch.setenv("MCORE_GDN_INTERNAL_BACKEND", "auto")
    monkeypatch.setattr(adapter, "fla_chunk_kda", fla)
    monkeypatch.setattr(
        adapter,
        "InternalChunkKDAFunction",
        SimpleNamespace(apply=lambda *_args: pytest.fail("unsupported options must fall back")),
    )

    cp_context = _cp_context()
    result = adapter.chunk_kda(
        **_inputs(),
        use_gate_in_kernel=True,
        state_v_first=True,
        cp_context=cp_context,
        **_gate_params(),
    )

    assert result is expected
    assert seen["cp_context"] is cp_context
    assert seen["state_v_first"] is True


@pytest.mark.parametrize(
    "option",
    [
        {"output_final_state": True},
        {"initial_state": torch.empty((1, 8, 128, 128), dtype=torch.float32)},
        {"use_qk_l2norm_in_kernel": True},
        {"use_beta_sigmoid_in_kernel": True},
        {"allow_neg_eigval": True},
        {"return_intermediate_states": True},
        {"state_v_first": True},
        {"transpose_state_layout": True},
        {"chunk_size": 32},
        {"custom_option": "value"},
        {"safe_gate": True, "lower_bound": -5.5},
        {"safe_gate": True, "lower_bound": None},
    ],
)
def test_cute_mode_rejects_unsupported_kda_options(monkeypatch, option):
    adapter = _adapter()
    monkeypatch.setenv("MCORE_GDN_INTERNAL_BACKEND", "cute")
    monkeypatch.setattr(
        adapter,
        "fla_chunk_kda",
        lambda *_args, **_kwargs: pytest.fail("cute mode must not fall back"),
    )

    with pytest.raises(RuntimeError, match="Internal KDA CP path is unavailable"):
        adapter.chunk_kda(
            **_inputs(),
            use_gate_in_kernel=True,
            cp_context=_cp_context(),
            **_gate_params(),
            **option,
        )


def test_support_reason_pins_adapter_shape_contract():
    adapter = _adapter()
    base = dict(
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=False,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=False,
        allow_neg_eigval=False,
        safe_gate=True,
        lower_bound=-5.0,
        return_intermediate_states=False,
        state_v_first=False,
        kwargs=_gate_params(),
    )

    assert adapter._kda_internal_support_reason(**_inputs(), **base) is None

    batched = _inputs((2, 64, 8, 128))
    assert "batch size 1" in adapter._kda_internal_support_reason(**batched, **base)

    gva = _inputs()
    gva["v"] = torch.empty((1, 64, 16, 128), dtype=torch.bfloat16)
    gva["g"] = torch.empty((1, 64, 16, 128), dtype=torch.bfloat16)
    gva["beta"] = torch.empty((1, 64, 16), dtype=torch.float32)
    assert "grouped value attention" in adapter._kda_internal_support_reason(**gva, **base)

    scalar_gate = _inputs()
    scalar_gate["g"] = torch.empty((1, 64, 8), dtype=torch.bfloat16)
    assert "g must have shape" in adapter._kda_internal_support_reason(**scalar_gate, **base)

    missing_a_log = dict(base, kwargs={"dt_bias": base["kwargs"]["dt_bias"]})
    assert "A_log is required" in adapter._kda_internal_support_reason(**_inputs(), **missing_a_log)


@pytest.mark.parametrize("scale, expected_scale", [(None, 128**-0.5), (0.0, 0.0)])
def test_auto_mode_dispatches_cp_kda_to_local_autograd_function(monkeypatch, scale, expected_scale):
    adapter = _adapter()
    inputs = _inputs()
    gate = _gate_params()
    expected = (torch.empty(1), None)
    calls = []

    def apply(*args):
        calls.append(args)
        return expected

    monkeypatch.setenv("MCORE_GDN_INTERNAL_BACKEND", "auto")
    monkeypatch.setattr(
        adapter,
        "fla_chunk_kda",
        lambda *_args, **_kwargs: pytest.fail("supported CP calls must not use FLA"),
    )
    monkeypatch.setattr(adapter, "InternalChunkKDAFunction", SimpleNamespace(apply=apply))

    cp_context = _cp_context()
    caller_cu_seqlens = torch.tensor([0, 128], dtype=torch.int32)
    validated = torch.tensor([[0, 0]], dtype=torch.int32)
    result = adapter.chunk_kda(
        **inputs,
        scale=scale,
        use_gate_in_kernel=True,
        safe_gate=True,
        lower_bound=-5.0,
        cu_seqlens=caller_cu_seqlens,
        validated_chunk_indices=validated,
        cp_context=cp_context,
        **gate,
    )

    assert result is expected
    (args,) = calls
    assert len(args) == 16
    assert args[0] is inputs["q"] and args[1] is inputs["k"] and args[2] is inputs["v"]
    assert args[3] is inputs["g"] and args[4] is inputs["beta"]
    assert args[5] is gate["A_log"] and args[6] is gate["dt_bias"]
    assert args[7] == expected_scale
    assert args[8] is True and args[9] == -5.0 and args[10] is True and args[11] is False
    # The context's local offsets replace the caller's global ones (FLA semantics).
    assert args[12] is cp_context.cu_seqlens and args[13] is cp_context.cu_seqlens_cpu
    assert args[14] is validated and args[15] is cp_context


def _forward_stub_tensors(disable_recompute):
    batch, seqlen, heads, head_dim = _SHAPE
    bf16 = dict(dtype=torch.bfloat16)
    fp32 = dict(dtype=torch.float32)
    tensors = {
        "g_cumsum": torch.empty(_SHAPE, **fp32),
        "w": torch.empty(_SHAPE, **bf16),
        "u": torch.empty(_SHAPE, **bf16),
        "qg": torch.empty(_SHAPE, **bf16),
        "kg": torch.empty(_SHAPE, **bf16),
        "Aqk": torch.empty((batch, seqlen, heads, 64), **bf16),
        "Akk": torch.empty((batch, seqlen, heads, 64), **bf16),
        "cp_initial_state": torch.empty((1, heads, head_dim, head_dim), **fp32),
        "compressed_initial_state": torch.empty((1, heads, head_dim, head_dim), **fp32),
        "expanded_initial_state": torch.empty((1, heads, head_dim, head_dim), **fp32),
        "h": torch.empty((batch, 1, heads, head_dim, head_dim), **bf16),
        "v_new": torch.empty(_SHAPE, **bf16),
        "o": torch.empty(_SHAPE, **bf16),
        "chunk_indices": torch.tensor([[0, 0]], dtype=torch.int32),
    }
    tensors["fwd_intra_qg"] = tensors["qg"] if disable_recompute else None
    return tensors


def _backward_stub_tensors():
    batch, seqlen, heads, head_dim = _SHAPE
    scalar = (batch, seqlen, heads)
    return {
        "dAqk": torch.empty((*scalar, 64), dtype=torch.float32),
        "dAv_dv": torch.empty(_SHAPE, dtype=torch.bfloat16),
        "dht_out": torch.empty((1, heads, head_dim, head_dim), dtype=torch.float32),
        "dh": torch.empty((batch, 1, heads, head_dim, head_dim), dtype=torch.bfloat16),
        "dhu_dv": torch.empty(_SHAPE, dtype=torch.bfloat16),
        "dq": torch.empty(_SHAPE, dtype=torch.float32),
        "dk": torch.empty(_SHAPE, dtype=torch.float32),
        "fused_dv": torch.full(_SHAPE, 3.0, dtype=torch.bfloat16),
        "db": torch.empty(scalar, dtype=torch.float32),
        "dg": torch.empty(_SHAPE, dtype=torch.float32),
        "dAkk": torch.empty((*scalar, 64), dtype=torch.float32),
        "intra_dq": torch.full(_SHAPE, 1.0, dtype=torch.float32),
        "intra_dk": torch.full(_SHAPE, 2.0, dtype=torch.float32),
        "intra_db": torch.full(scalar, 5.0, dtype=torch.float32),
        "intra_dg": torch.empty(_SHAPE, dtype=torch.float32),
        "dg_cumsum": torch.empty(_SHAPE, dtype=torch.float32),
        "dg_raw": torch.full(_SHAPE, 4.0, dtype=torch.bfloat16),
        "dA": torch.full((heads,), 6.0, dtype=torch.float32),
        "dbias": torch.full((heads * head_dim,), 7.0, dtype=torch.float32),
    }


def _record(seen, name, result):
    def stub(*args, **kwargs):
        seen.setdefault(name, []).append((args, kwargs))
        return result

    return stub


def _install_kernel_stubs(monkeypatch, adapter, seen, fwd, bwd):
    monkeypatch.setattr(
        adapter,
        "prepare_chunk_indices",
        _record(seen, "prepare_chunk_indices", fwd["chunk_indices"]),
    )
    monkeypatch.setattr(
        adapter, "kda_gate_chunk_cumsum", _record(seen, "kda_gate_chunk_cumsum", fwd["g_cumsum"])
    )
    monkeypatch.setattr(
        adapter,
        "chunk_kda_fwd_intra",
        _record(
            seen,
            "chunk_kda_fwd_intra",
            (fwd["w"], fwd["u"], fwd["fwd_intra_qg"], fwd["kg"], fwd["Aqk"], fwd["Akk"]),
        ),
    )
    monkeypatch.setattr(
        adapter,
        "_cp_forward_preprocess",
        _record(seen, "_cp_forward_preprocess", fwd["cp_initial_state"]),
    )
    monkeypatch.setattr(
        adapter,
        "chunk_gated_delta_rule_fwd_h",
        _record(seen, "chunk_gated_delta_rule_fwd_h", (fwd["h"], fwd["v_new"], None)),
    )
    monkeypatch.setattr(
        adapter, "compress_h0", _record(seen, "compress_h0", fwd["compressed_initial_state"])
    )
    monkeypatch.setattr(
        adapter, "chunk_gla_fwd_o_gk", _record(seen, "chunk_gla_fwd_o_gk", fwd["o"])
    )
    monkeypatch.setattr(
        adapter,
        "recompute_w_u_fwd",
        _record(seen, "recompute_w_u_fwd", (fwd["w"], fwd["u"], fwd["qg"], fwd["kg"])),
    )
    monkeypatch.setattr(
        adapter, "expand_h0", _record(seen, "expand_h0", fwd["expanded_initial_state"])
    )
    monkeypatch.setattr(
        adapter,
        "chunk_kda_bwd_dAv",
        _record(seen, "chunk_kda_bwd_dAv", (bwd["dAqk"], bwd["dAv_dv"])),
    )
    monkeypatch.setattr(
        adapter,
        "_cp_backward_preprocess",
        _record(seen, "_cp_backward_preprocess", (bwd["dht_out"], None)),
    )
    monkeypatch.setattr(
        adapter,
        "chunk_gated_delta_rule_bwd_dhu",
        _record(seen, "chunk_gated_delta_rule_bwd_dhu", (bwd["dh"], None, bwd["dhu_dv"])),
    )
    monkeypatch.setattr(
        adapter,
        "chunk_kda_bwd_wy_dqkg_fused",
        _record(
            seen,
            "chunk_kda_bwd_wy_dqkg_fused",
            (bwd["dq"], bwd["dk"], bwd["fused_dv"], bwd["db"], bwd["dg"], bwd["dAkk"]),
        ),
    )
    monkeypatch.setattr(
        adapter,
        "chunk_kda_bwd_intra",
        _record(
            seen,
            "chunk_kda_bwd_intra",
            (bwd["intra_dq"], bwd["intra_dk"], bwd["intra_db"], bwd["intra_dg"]),
        ),
    )
    monkeypatch.setattr(
        adapter, "chunk_local_cumsum", _record(seen, "chunk_local_cumsum", bwd["dg_cumsum"])
    )
    monkeypatch.setattr(
        adapter,
        "kda_gate_bwd",
        _record(seen, "kda_gate_bwd", (bwd["dg_raw"], bwd["dA"], bwd["dbias"])),
    )


def _only_call(seen, name):
    (call,) = seen[name]
    return call


@pytest.mark.parametrize("disable_recompute", [False, True])
def test_kda_adapter_redirects_only_the_cp_preprocess(monkeypatch, disable_recompute):
    adapter = _adapter()
    monkeypatch.setenv("MCORE_GDN_INTERNAL_BACKEND", "auto")
    inputs = _inputs()
    gate = _gate_params()
    fwd = _forward_stub_tensors(disable_recompute)
    bwd = _backward_stub_tensors()
    cp_context = _cp_context()
    scale = 128**-0.5
    seen = {}
    _install_kernel_stubs(monkeypatch, adapter, seen, fwd, bwd)

    ctx = SimpleNamespace()
    ctx.save_for_backward = lambda *args: setattr(ctx, "saved_tensors", args)
    forward = _unwrap(adapter.InternalChunkKDAFunction.forward)
    output, final_state = forward(
        ctx,
        inputs["q"],
        inputs["k"],
        inputs["v"],
        inputs["g"],
        inputs["beta"],
        gate["A_log"],
        gate["dt_bias"],
        scale,
        True,
        -5.0,
        True,
        disable_recompute,
        cp_context.cu_seqlens,
        cp_context.cu_seqlens_cpu,
        None,
        cp_context,
    )

    assert final_state is None
    assert output.dtype == inputs["q"].dtype and output.shape == fwd["o"].shape

    # Chunk indices are counted on the CPU offsets (no device sync).
    args, kwargs = _only_call(seen, "prepare_chunk_indices")
    assert args[0] is cp_context.cu_seqlens and args[1] == 64
    assert kwargs["cu_seqlens_cpu"] is cp_context.cu_seqlens_cpu

    _args, kwargs = seen["kda_gate_chunk_cumsum"][0]
    assert kwargs["g"] is inputs["g"]
    assert kwargs["A_log"] is gate["A_log"] and kwargs["dt_bias"] is gate["dt_bias"]
    assert kwargs["scale"] == adapter.RCP_LN2 and kwargs["chunk_size"] == 64
    assert kwargs["cu_seqlens"] is cp_context.cu_seqlens
    assert kwargs["chunk_indices"] is fwd["chunk_indices"] and kwargs["lower_bound"] == -5.0

    _args, kwargs = _only_call(seen, "chunk_kda_fwd_intra")
    assert kwargs["q"] is inputs["q"] and kwargs["k"] is inputs["k"] and kwargs["v"] is inputs["v"]
    assert kwargs["gk"] is fwd["g_cumsum"] and kwargs["beta"] is inputs["beta"]
    assert kwargs["scale"] == scale and kwargs["safe_gate"] is True
    assert kwargs["disable_recompute"] is disable_recompute

    # The forward CP pre-process sees the pre-gated kg, the WY tensors, and the fp32
    # log2 cumsum as gk; the scalar g is never passed.
    args, kwargs = _only_call(seen, "_cp_forward_preprocess")
    assert args == ()
    assert set(kwargs) == {"k", "w", "u", "gk", "cu_seqlens", "initial_state", "context"}
    assert kwargs["k"] is fwd["kg"] and kwargs["w"] is fwd["w"] and kwargs["u"] is fwd["u"]
    assert kwargs["gk"] is fwd["g_cumsum"]
    assert kwargs["cu_seqlens"] is cp_context.cu_seqlens
    assert kwargs["initial_state"] is None and kwargs["context"] is cp_context

    _args, kwargs = seen["chunk_gated_delta_rule_fwd_h"][0]
    assert kwargs["k"] is fwd["kg"] and kwargs["w"] is fwd["w"] and kwargs["u"] is fwd["u"]
    assert kwargs["gk"] is fwd["g_cumsum"] and kwargs["initial_state"] is fwd["cp_initial_state"]
    assert (
        kwargs["output_final_state"] is False
        and kwargs["cu_seqlens_cpu"] is cp_context.cu_seqlens_cpu
    )

    args, kwargs = _only_call(seen, "compress_h0")
    assert args == (fwd["cp_initial_state"],) and kwargs == {"context": cp_context}

    _args, kwargs = _only_call(seen, "chunk_gla_fwd_o_gk")
    assert (
        kwargs["q"] is inputs["q"]
        and kwargs["v"] is fwd["v_new"]
        and kwargs["g"] is fwd["g_cumsum"]
    )
    assert kwargs["A"] is fwd["Aqk"] and kwargs["h"] is fwd["h"] and kwargs["scale"] == scale

    saved = ctx.saved_tensors
    assert len(saved) == 19
    assert saved[0] is inputs["q"] and saved[1] is inputs["k"] and saved[2] is inputs["v"]
    assert saved[4] is inputs["g"] and saved[5] is inputs["beta"]
    assert saved[6] is gate["A_log"] and saved[7] is gate["dt_bias"]
    assert saved[8] is fwd["Aqk"] and saved[9] is fwd["Akk"]
    if disable_recompute:
        assert saved[3] is fwd["g_cumsum"]
        assert saved[10:16] == (fwd["w"], fwd["u"], fwd["qg"], fwd["kg"], fwd["v_new"], fwd["h"])
    else:
        # Same saved set as FLA: the gate cumsum and every recompute intermediate drop.
        assert saved[3] is None
        assert saved[10:16] == (None,) * 6
    assert saved[16] is fwd["compressed_initial_state"]
    assert saved[17] is cp_context.cu_seqlens and saved[18] is fwd["chunk_indices"]
    assert ctx.scale == scale and ctx.safe_gate is True and ctx.lower_bound == -5.0
    assert ctx.use_gate_in_kernel is True and ctx.disable_recompute is disable_recompute
    assert ctx.cp_context is cp_context and ctx.strict is False

    do = torch.empty(_SHAPE, dtype=torch.bfloat16)
    backward = _unwrap(adapter.InternalChunkKDAFunction.backward)
    grads = backward(ctx, do, None)

    assert len(grads) == 16
    dq, dk, dv, dg, dbeta, dA, dbias = grads[:7]
    assert dq.dtype == inputs["q"].dtype and torch.all(dq == 1)
    assert dk.dtype == inputs["k"].dtype and torch.all(dk == 2)
    assert dv.dtype == inputs["v"].dtype and torch.all(dv == 3)
    assert dg.dtype == inputs["g"].dtype and torch.all(dg == 4)
    assert dbeta.dtype == inputs["beta"].dtype and torch.all(dbeta == 5)
    assert dA is bwd["dA"] and dbias is bwd["dbias"]
    assert all(entry is None for entry in grads[7:])

    if disable_recompute:
        assert "recompute_w_u_fwd" not in seen
        assert len(seen["kda_gate_chunk_cumsum"]) == 1
        assert len(seen["chunk_gated_delta_rule_fwd_h"]) == 1
    else:
        assert len(seen["kda_gate_chunk_cumsum"]) == 2
        assert seen["kda_gate_chunk_cumsum"][1][1]["g"] is inputs["g"]
        _args, kwargs = _only_call(seen, "recompute_w_u_fwd")
        # q is passed so the backward materialises qg for the CP pre-process.
        assert (
            kwargs["q"] is inputs["q"] and kwargs["k"] is inputs["k"] and kwargs["v"] is inputs["v"]
        )
        assert kwargs["beta"] is inputs["beta"] and kwargs["A"] is fwd["Akk"]
        assert kwargs["gk"] is fwd["g_cumsum"]
        assert len(seen["chunk_gated_delta_rule_fwd_h"]) == 2
        recomputed = seen["chunk_gated_delta_rule_fwd_h"][1][1]
        assert recomputed["initial_state"] is fwd["expanded_initial_state"]
        assert "cu_seqlens_cpu" not in recomputed
    args, kwargs = _only_call(seen, "expand_h0")
    assert args == (fwd["compressed_initial_state"],) and kwargs == {"context": cp_context}

    _args, kwargs = _only_call(seen, "chunk_kda_bwd_dAv")
    assert kwargs["q"] is inputs["q"] and kwargs["k"] is inputs["k"] and kwargs["v"] is fwd["v_new"]
    assert kwargs["do"] is do and kwargs["A"] is fwd["Aqk"] and kwargs["scale"] == scale

    # The backward CP pre-process sees qg (the gk-mode kernel never gates q itself),
    # kg, w, the local dv term, and the fp32 log2 cumsum as gk.
    args, kwargs = _only_call(seen, "_cp_backward_preprocess")
    assert args == ()
    assert set(kwargs) == {
        "q",
        "k",
        "w",
        "do",
        "dv",
        "gk",
        "scale",
        "cu_seqlens",
        "dht",
        "initial_state",
        "context",
    }
    assert kwargs["q"] is fwd["qg"] and kwargs["k"] is fwd["kg"] and kwargs["w"] is fwd["w"]
    assert kwargs["do"] is do and kwargs["dv"] is bwd["dAv_dv"] and kwargs["gk"] is fwd["g_cumsum"]
    assert kwargs["scale"] == scale and kwargs["cu_seqlens"] is cp_context.cu_seqlens
    assert kwargs["dht"] is None and kwargs["initial_state"] is fwd["expanded_initial_state"]
    assert kwargs["context"] is cp_context

    _args, kwargs = _only_call(seen, "chunk_gated_delta_rule_bwd_dhu")
    assert kwargs["q"] is fwd["qg"] and kwargs["k"] is fwd["kg"] and kwargs["w"] is fwd["w"]
    assert kwargs["gk"] is fwd["g_cumsum"]
    # The pre-process returns (dht, None): the local state is dropped, dht is consumed.
    assert kwargs["h0"] is None and kwargs["dht"] is bwd["dht_out"]
    assert kwargs["do"] is do and kwargs["dv"] is bwd["dAv_dv"] and kwargs["scale"] == scale

    _args, kwargs = _only_call(seen, "chunk_kda_bwd_wy_dqkg_fused")
    assert kwargs["q"] is inputs["q"] and kwargs["k"] is inputs["k"] and kwargs["v"] is inputs["v"]
    assert kwargs["v_new"] is fwd["v_new"] and kwargs["g"] is fwd["g_cumsum"]
    assert (
        kwargs["beta"] is inputs["beta"] and kwargs["A"] is fwd["Akk"] and kwargs["h"] is fwd["h"]
    )
    assert kwargs["do"] is do and kwargs["dh"] is bwd["dh"] and kwargs["dv"] is bwd["dhu_dv"]

    _args, kwargs = _only_call(seen, "chunk_kda_bwd_intra")
    assert (
        kwargs["q"] is inputs["q"] and kwargs["k"] is inputs["k"] and kwargs["g"] is fwd["g_cumsum"]
    )
    assert kwargs["dAqk"] is bwd["dAqk"] and kwargs["dAkk"] is bwd["dAkk"]
    assert kwargs["dq"] is bwd["dq"] and kwargs["dk"] is bwd["dk"]
    assert kwargs["db"] is bwd["db"] and kwargs["dg"] is bwd["dg"] and kwargs["safe_gate"] is True

    args, kwargs = _only_call(seen, "chunk_local_cumsum")
    assert args == (bwd["intra_dg"],)
    assert kwargs == {
        "chunk_size": 64,
        "reverse": True,
        "cu_seqlens": cp_context.cu_seqlens,
        "chunk_indices": fwd["chunk_indices"],
    }

    _args, kwargs = _only_call(seen, "kda_gate_bwd")
    assert kwargs["g"] is inputs["g"] and kwargs["dyg"] is bwd["dg_cumsum"]
    assert kwargs["A_log"] is gate["A_log"] and kwargs["dt_bias"] is gate["dt_bias"]
    assert kwargs["lower_bound"] == -5.0


@pytest.mark.parametrize("progress", [False, True], ids=["fallback", "cutedsl"])
def test_cute_mode_requires_the_cutedsl_preprocess_to_run(monkeypatch, progress):
    adapter = _adapter()
    monkeypatch.setenv("MCORE_GDN_INTERNAL_BACKEND", "cute")
    inputs = _inputs()
    gate = _gate_params()
    fwd = _forward_stub_tensors(disable_recompute=False)
    bwd = _backward_stub_tensors()
    cp_context = _cp_context()
    seen = {}
    counters = {"fwd": 0, "bwd": 0, "noop": 0}
    _install_kernel_stubs(monkeypatch, adapter, seen, fwd, bwd)
    monkeypatch.setattr(adapter, "get_cutedsl_fused_launch_count", lambda: counters["fwd"])
    monkeypatch.setattr(adapter, "get_cutedsl_fused_bwd_launch_count", lambda: counters["bwd"])
    monkeypatch.setattr(adapter, "get_cutedsl_cp_noop_count", lambda: counters["noop"])
    monkeypatch.setattr(adapter, "get_cutedsl_fallback_reasons", lambda: {"test reason": 1})

    def forward_preprocess(**_kwargs):
        counters["fwd"] += int(progress)
        return fwd["cp_initial_state"]

    def backward_preprocess(**_kwargs):
        counters["noop"] += int(progress)
        return bwd["dht_out"], None

    monkeypatch.setattr(adapter, "_cp_forward_preprocess", forward_preprocess)
    monkeypatch.setattr(adapter, "_cp_backward_preprocess", backward_preprocess)

    ctx = SimpleNamespace()
    ctx.save_for_backward = lambda *args: setattr(ctx, "saved_tensors", args)
    forward = _unwrap(adapter.InternalChunkKDAFunction.forward)
    forward_args = (
        ctx,
        inputs["q"],
        inputs["k"],
        inputs["v"],
        inputs["g"],
        inputs["beta"],
        gate["A_log"],
        gate["dt_bias"],
        128**-0.5,
        True,
        -5.0,
        True,
        False,
        cp_context.cu_seqlens,
        cp_context.cu_seqlens_cpu,
        None,
        cp_context,
    )
    if not progress:
        with pytest.raises(RuntimeError, match="forward preprocess fell back to FLA Triton"):
            forward(*forward_args)
        return

    forward(*forward_args)
    assert ctx.strict is True

    backward = _unwrap(adapter.InternalChunkKDAFunction.backward)
    grads = backward(ctx, torch.empty(_SHAPE, dtype=torch.bfloat16), None)
    assert len(grads) == 16

    # A backward whose pre-process neither launches nor no-ops is refused as well.
    counters["noop"] = 0
    monkeypatch.setattr(
        adapter, "_cp_backward_preprocess", lambda **_kwargs: (bwd["dht_out"], None)
    )
    with pytest.raises(RuntimeError, match="backward preprocess fell back to FLA Triton"):
        backward(ctx, torch.empty(_SHAPE, dtype=torch.bfloat16), None)
