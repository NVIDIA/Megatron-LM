# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for the internal GDR backend adapter."""

import ast
import threading
import time
from concurrent.futures import ThreadPoolExecutor
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


def test_cutedsl_cp_context_metadata_rebind_invalidates_memos():
    from megatron.core.ssm.gated_delta_net.internal_gdn_backend import prepare_cp_context_metadata

    context = SimpleNamespace(
        _cutedsl_chain_memo={(0, 64): "fused"}, _cutedsl_window_memo={(0, True): (0, 64)}
    )
    prepare_cp_context_metadata(context, global_num_sequences=1)
    first_generation = context._cutedsl_metadata_generation

    assert context._cutedsl_chain_memo == {}
    assert context._cutedsl_window_memo == {}
    assert context.global_num_seqs == 1

    context._cutedsl_chain_memo[(first_generation, 64)] = "fused"
    prepare_cp_context_metadata(context, global_num_sequences=1)
    assert context._cutedsl_metadata_generation == first_generation
    assert context._cutedsl_chain_memo == {(first_generation, 64): "fused"}

    prepare_cp_context_metadata(context, global_num_sequences=2)

    assert context._cutedsl_metadata_generation == first_generation + 1
    assert context.global_num_seqs == 2
    assert context._cutedsl_chain_memo == {}


@pytest.mark.parametrize("captures_gdn", [True, False], ids=["attention", "mlp_only"])
def test_cp_context_metadata_records_cuda_graph_gate(captures_gdn):
    from megatron.core.ssm.gated_delta_net.internal_gdn_backend import prepare_cp_context_metadata
    from megatron.core.transformer.enums import CudaGraphModule

    context = SimpleNamespace(_cutedsl_chain_memo={}, _cutedsl_window_memo={})
    config = SimpleNamespace(
        cuda_graph_impl="transformer_engine",
        cuda_graph_modules=None if captures_gdn else [CudaGraphModule.mlp],
    )

    prepare_cp_context_metadata(context, config=config, global_num_sequences=1)

    assert context._cutedsl_cuda_graph_enabled is captures_gdn


def test_cutedsl_cp_chain_without_host_offsets_only_accepts_single_full_chain():
    from megatron.core.ssm.gated_delta_net.internal_gdn_backend.kernels.fused_gdr_cp_cute import (
        backend as cp_backend,
    )

    cu_seqlens = torch.tensor([0, 256], dtype=torch.int32)
    context = SimpleNamespace(
        cu_seqlens_cpu=torch.tensor([0, 64]), global_num_seqs=1, pre_num_ranks=1, post_num_ranks=2
    )

    assert (
        cp_backend._classify_chain(
            context=context, cu_seqlens=cu_seqlens, T=64, rank=1, world_size=4
        )
        == "fused"
    )

    context.global_num_seqs = 2
    assert (
        cp_backend._classify_chain(
            context=context, cu_seqlens=cu_seqlens, T=64, rank=1, world_size=4
        )
        is None
    )


def test_cutedsl_cp_wrapper_marshals_different_streams(monkeypatch):
    from megatron.core.ssm.gated_delta_net.internal_gdn_backend.kernels.fused_gdr_cp_cute import (
        backend as cp_backend,
    )

    calls = []

    class FakeWrapper:
        def launch_validated(self, value, *, _stream_handle):
            calls.append(("launch", value, _stream_handle))
            return value

    class FakeEvent:
        def record(self, stream):
            calls.append(("record", stream.name))

    class FakeStream:
        def __init__(self, name, handle):
            self.name = name
            self.cuda_stream = handle

        def wait_event(self, event):
            calls.append(("wait", self.name, event))

    class FakeStreamContext:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            calls.append(("enter", self.stream.name))

        def __exit__(self, *_args):
            calls.append(("exit", self.stream.name))

    owner = FakeStream("owner", 11)
    caller = FakeStream("caller", 22)
    # Construction binds owner=11 before the first launch can race in.
    handles = iter([11, 11, 22])
    monkeypatch.setattr(cp_backend, "_raw_stream", lambda _device: next(handles))
    monkeypatch.setattr(cp_backend.torch.cuda, "current_stream", lambda _device: owner)
    monkeypatch.setattr(cp_backend.torch.cuda, "ExternalStream", lambda handle, device: caller)
    monkeypatch.setattr(cp_backend.torch.cuda, "Event", FakeEvent)
    monkeypatch.setattr(cp_backend.torch.cuda, "stream", FakeStreamContext)
    wrapper = cp_backend._SerializedWrapper(FakeWrapper(), torch.device("cuda", 0))
    assert wrapper._owner_handle == 11

    assert wrapper.launch_validated(1) == 1
    assert wrapper.launch_validated(2) == 2

    assert calls[0] == ("launch", 1, 11)
    assert calls[1][0:2] == ("record", "caller")
    assert calls[2][0:2] == ("wait", "owner")
    assert calls[3] == ("enter", "owner")
    assert calls[4] == ("launch", 2, 11)
    assert calls[5] == ("exit", "owner")
    assert calls[6][0:2] == ("record", "owner")
    assert calls[7][0:2] == ("wait", "caller")


def test_cutedsl_cp_wrapper_is_constructed_once_under_concurrent_miss(monkeypatch):
    from megatron.core.ssm.gated_delta_net.internal_gdn_backend.kernels.fused_gdr_cp_cute import (
        backend as cp_backend,
    )

    cache = {}
    group = object()
    barrier = threading.Barrier(2)
    constructed = []
    monkeypatch.setattr(cp_backend, "_rank_consistent_wrapper_init", lambda *_args: True)

    def factory():
        constructed.append(object())
        time.sleep(0.01)
        return constructed[-1]

    def get_wrapper():
        barrier.wait()
        return cp_backend._get_or_create_wrapper(
            cache,
            ("shape",),
            factory,
            group=group,
            device=torch.device("cuda", 0),
            signature=(0, 64, 64, 128, 128),
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        wrappers = list(pool.map(lambda _: get_wrapper(), range(2)))

    assert len(constructed) == 1
    assert wrappers[0] is wrappers[1] is constructed[0]


def test_cutedsl_cp_wrapper_raises_on_cross_rank_initialization_mismatch(monkeypatch):
    from megatron.core.ssm.gated_delta_net.internal_gdn_backend.kernels.fused_gdr_cp_cute import (
        backend as cp_backend,
    )

    monkeypatch.setattr(cp_backend, "_rank_consistent_wrapper_init", lambda *_args: False)
    constructed = []
    with pytest.raises(RuntimeError, match="initialization order differs across ranks"):
        cp_backend._get_or_create_wrapper(
            {},
            ("shape",),
            lambda: constructed.append(object()),
            group=object(),
            device=torch.device("cuda", 0),
            signature=(0, 64, 64, 128, 128),
        )

    assert constructed == []


def test_fused_forward_package_exports_wrapper():
    from megatron.core.ssm.gated_delta_net.internal_gdn_backend.kernels import fused_gdr_fwd_cute

    assert fused_gdr_fwd_cute.chunk_gated_delta_rule_prefill_cute.__module__.endswith(".fused_fwd")


def _implementation():
    pytest.importorskip("fla")
    from megatron.core.ssm.gated_delta_net.internal_gdn_backend import implementation

    return implementation


def test_fla_compat_drops_unsupported_use_exp2_without_rescaling_gate():
    implementation = _implementation()
    gate = torch.tensor([implementation.RCP_LN2], dtype=torch.float32)
    seen = {}

    def primitive(g):
        seen["g"] = g
        return g

    result = implementation._call_fla_compat(primitive, g=gate, use_exp2=True)

    assert result is gate
    assert seen["g"] is gate


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


@pytest.mark.parametrize("mode", ["fla", "auto"])
def test_dense_batch_cp_fla_dispatch_slices_only_the_gdr_operator(monkeypatch, mode):
    implementation = _implementation()
    base = torch.arange(2 * 2 * 3 * 4, dtype=torch.float32).reshape(2, 2, 3, 4)
    inputs = {
        "q": base,
        "k": torch.empty_like(base),
        "v": torch.empty_like(base),
        "g": torch.empty(base.shape[:-1]),
        "beta": torch.empty(base.shape[:-1]),
    }
    local_cu_seqlens = torch.tensor([0, 2], dtype=torch.int32)
    cp_context = SimpleNamespace(group=object(), cu_seqlens=local_cu_seqlens)
    calls = []

    def fla(**kwargs):
        assert kwargs["q"].shape[0] == 1, "FLA CP requires one physical batch slice"
        calls.append(kwargs)
        return kwargs["q"] + len(calls), None

    monkeypatch.setenv("MCORE_GDN_INTERNAL_BACKEND", mode)
    monkeypatch.setattr(implementation, "fla_chunk_gated_delta_rule", fla)
    if mode == "fla":
        monkeypatch.setattr(
            implementation,
            "_cutedsl_support_reason",
            lambda **_kwargs: pytest.fail("explicit FLA mode must bypass CuTe support checks"),
        )
    else:
        monkeypatch.setattr(
            implementation, "_cutedsl_support_reason", lambda **_kwargs: "unsupported test input"
        )

    output, final_state = implementation.chunk_gated_delta_rule(**inputs, cp_context=cp_context)

    assert final_state is None
    assert len(calls) == 2
    assert all(call["cp_context"] is cp_context for call in calls)
    torch.testing.assert_close(output[0], base[0] + 1)
    torch.testing.assert_close(output[1], base[1] + 2)


def test_public_dispatch_forwards_device_cu_seqlens_without_host_metadata(monkeypatch):
    implementation = _implementation()
    expected = (torch.empty(1), None)
    seen = {}

    def support_reason(**kwargs):
        seen.update(kwargs)
        return "unsupported test input"

    monkeypatch.setenv("MCORE_GDN_INTERNAL_BACKEND", "auto")
    monkeypatch.setattr(implementation, "_cutedsl_support_reason", support_reason)
    monkeypatch.setattr(implementation, "fla_chunk_gated_delta_rule", lambda **_kwargs: expected)

    cu_seqlens = torch.tensor([0, 2], dtype=torch.int32)
    assert implementation.chunk_gated_delta_rule(**_inputs(), cu_seqlens=cu_seqlens) is expected
    assert seen["cu_seqlens"] is cu_seqlens


def test_public_dispatch_forwards_validated_chunk_offsets_without_host_metadata(monkeypatch):
    implementation = _implementation()
    expected = (torch.empty(1), None)
    seen = {}

    def support_reason(**kwargs):
        seen.update(kwargs)
        return "unsupported test input"

    monkeypatch.setenv("MCORE_GDN_INTERNAL_BACKEND", "auto")
    monkeypatch.setattr(implementation, "_cutedsl_support_reason", support_reason)
    monkeypatch.setattr(implementation, "fla_chunk_gated_delta_rule", lambda **_kwargs: expected)

    cu_seqlens = torch.tensor([0, 2], dtype=torch.int32)
    chunk_offsets = torch.tensor([0, 1], dtype=torch.int32)
    assert (
        implementation.chunk_gated_delta_rule(
            **_inputs(), cu_seqlens=cu_seqlens, validated_chunk_offsets=chunk_offsets
        )
        is expected
    )
    assert seen["cu_seqlens"] is cu_seqlens


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
@pytest.mark.parametrize("use_cp", [False, True], ids=["non_cp", "cp"])
def test_cute_mode_dispatches_to_local_autograd_function(
    monkeypatch, scale, expected_scale, recompute_h, use_cp
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

    local_cu_seqlens = torch.tensor([0, 2], dtype=torch.int32)
    cp_context = SimpleNamespace(group=object(), cu_seqlens=local_cu_seqlens) if use_cp else None
    result = implementation.chunk_gated_delta_rule(
        **inputs, scale=scale, recompute_h=recompute_h, cp_context=cp_context
    )
    assert result is expected

    assert calls[0][5] == expected_scale
    assert calls[0][8] is None
    assert calls[0][9] is recompute_h
    assert calls[0][10] is cp_context


@pytest.mark.parametrize("use_saved_h", [False, True])
def test_cp_backward_preprocessing_produces_fused_dht_and_state(monkeypatch, use_saved_h):
    implementation = _implementation()
    shape = (1, 64, 2, 4)
    scalar_shape = shape[:-1]
    q = torch.empty(shape, dtype=torch.bfloat16)
    k = torch.empty_like(q)
    v = torch.empty_like(q)
    g = torch.empty(scalar_shape, dtype=torch.float32)
    beta = torch.empty(scalar_shape, dtype=torch.float32)
    A = torch.empty((*scalar_shape, 64), dtype=torch.bfloat16)
    do = torch.empty_like(q)
    compressed_initial_state = torch.empty((1, 2, 4, 4), dtype=torch.float32)
    expanded_initial_state = torch.empty_like(compressed_initial_state)
    recomputed_h = torch.empty((1, 1, 2, 4, 4), dtype=torch.float32)
    saved_h = torch.empty((1, 2, 4, 4), dtype=torch.bfloat16)
    expected_h = saved_h if use_saved_h else recomputed_h
    expected_dht = torch.empty((1, 2, 4, 4), dtype=torch.float32)
    w = torch.empty_like(k)
    u = torch.empty_like(v)
    dv = torch.empty_like(v)
    cp_context = SimpleNamespace(group=object())
    seen = {}

    monkeypatch.setattr(implementation, "recompute_w_u_fwd", lambda **_kwargs: (w, u))
    monkeypatch.setattr(
        implementation, "expand_h0", lambda initial_state, *, context: expanded_initial_state
    )

    def fwd_h(**kwargs):
        seen["fwd_h_initial_state"] = kwargs["initial_state"]
        return recomputed_h, torch.empty_like(v), None

    def cp_preprocess(**kwargs):
        seen["cp_preprocess"] = kwargs
        return expected_dht, None

    monkeypatch.setattr(implementation, "chunk_gated_delta_rule_fwd_h", fwd_h)
    monkeypatch.setattr(implementation, "chunk_bwd_dv_local", lambda **_kwargs: dv)
    monkeypatch.setattr(implementation, "chunk_gated_delta_rule_bwd_dhu_pre_process", cp_preprocess)

    actual_dht, actual_h = implementation._fla_cp_backward_preprocess(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        A=A,
        scale=0.5,
        do=do,
        dht=None,
        cu_seqlens=None,
        chunk_indices=None,
        initial_state=compressed_initial_state,
        cp_context=cp_context,
        h=saved_h if use_saved_h else None,
    )

    assert actual_dht is expected_dht
    assert actual_h is expected_h
    assert ("fwd_h_initial_state" in seen) is not use_saved_h
    if not use_saved_h:
        assert seen["fwd_h_initial_state"] is expanded_initial_state
    assert seen["cp_preprocess"]["w"] is w
    assert seen["cp_preprocess"]["dv"] is dv
    assert seen["cp_preprocess"]["initial_state"] is expanded_initial_state
    assert seen["cp_preprocess"]["context"] is cp_context


@pytest.mark.parametrize("direction", ["forward", "backward"])
def test_cp_preprocess_prefers_local_cutedsl_and_preserves_fla_fallback(monkeypatch, direction):
    implementation = _implementation()
    shape = (1, 64, 2, 4)
    operands = {
        "q": torch.empty(shape, dtype=torch.bfloat16),
        "k": torch.empty(shape, dtype=torch.bfloat16),
        "w": torch.empty(shape, dtype=torch.bfloat16),
        "u": torch.empty(shape, dtype=torch.bfloat16),
        "do": torch.empty(shape, dtype=torch.bfloat16),
        "dv": torch.empty(shape, dtype=torch.bfloat16),
        "g": torch.empty(shape[:-1], dtype=torch.float32),
    }
    cp_context = SimpleNamespace(group=object())
    local_result = torch.empty((1, 2, 4, 4), dtype=torch.float32)
    fallback_result = torch.empty_like(local_result)
    calls = {"local": 0, "fla": 0}

    if direction == "forward":
        local_name = "try_chunk_gated_delta_rule_fwd_h_pre_process_cutedsl"
        fla_name = "chunk_gated_delta_rule_fwd_h_pre_process"
        wrapper = implementation._cp_forward_preprocess
        kwargs = {
            "k": operands["k"],
            "w": operands["w"],
            "u": operands["u"],
            "g": operands["g"],
            "cu_seqlens": None,
            "initial_state": None,
            "context": cp_context,
            "use_exp2": True,
            "transpose_state_layout": False,
        }
        expected_fallback = fallback_result
    else:
        local_name = "try_chunk_gated_delta_rule_bwd_dhu_pre_process_cutedsl"
        fla_name = "chunk_gated_delta_rule_bwd_dhu_pre_process"
        wrapper = implementation._cp_backward_preprocess
        kwargs = {
            "q": operands["q"],
            "k": operands["k"],
            "w": operands["w"],
            "do": operands["do"],
            "dv": operands["dv"],
            "g": operands["g"],
            "scale": 0.5,
            "cu_seqlens": None,
            "dht": None,
            "initial_state": None,
            "context": cp_context,
            "use_exp2": True,
            "transpose_state_layout": False,
        }
        expected_fallback = (fallback_result, None)

    def local(**_kwargs):
        calls["local"] += 1
        return local_result

    def fla(**_kwargs):
        calls["fla"] += 1
        return expected_fallback

    monkeypatch.setattr(implementation, local_name, local)
    monkeypatch.setattr(implementation, fla_name, fla)
    actual_local = wrapper(**kwargs)
    if direction == "forward":
        assert actual_local is local_result
    else:
        assert actual_local[0] is local_result
        assert actual_local[1] is None
    assert calls == {"local": 1, "fla": 0}

    monkeypatch.setattr(implementation, local_name, lambda **_kwargs: None)
    actual_fallback = wrapper(**kwargs)
    if direction == "forward":
        assert actual_fallback is fallback_result
    else:
        assert actual_fallback[0] is fallback_result
        assert actual_fallback[1] is None
    assert calls == {"local": 1, "fla": 1}


def test_cutedsl_cp_backward_feeds_preprocessed_dht_to_fused_kernel(monkeypatch):
    implementation = _implementation()
    shape = (1, 64, 64, 128)
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
    cp_context = SimpleNamespace(group=object())
    initial_state = torch.empty((1, 64, 128, 128), dtype=torch.float32)
    cp_dht = torch.empty_like(initial_state)
    cp_h = torch.empty((1, 64, 128, 128), dtype=torch.bfloat16)
    expected = tuple(torch.empty(1) for _ in range(5))
    seen = {}

    monkeypatch.setattr(implementation, "_fused_bwd_support_reason", lambda **_kwargs: None)
    monkeypatch.setattr(
        implementation, "_fla_cp_backward_preprocess", lambda **_kwargs: (cp_dht, cp_h)
    )

    def fused(**kwargs):
        seen.update(kwargs)
        return expected

    monkeypatch.setattr(implementation, "_call_fused_gdr_bwd_cute", fused)

    result = implementation._cutedsl_backward(
        **inputs,
        scale=128**-0.5,
        dht=None,
        cu_seqlens=None,
        chunk_indices=None,
        initial_state=initial_state,
        cp_context=cp_context,
    )

    assert result is expected
    assert seen["dht"] is cp_dht
    assert seen["h"] is cp_h


def test_cutedsl_cp_auto_fallback_preserves_cp_boundary_inputs(monkeypatch):
    implementation = _implementation()
    shape = (1, 64, 64, 128)
    scalar_shape = shape[:-1]
    q = torch.empty(shape, dtype=torch.float32)
    inputs = {
        "q": q,
        "k": torch.empty_like(q),
        "v": torch.empty_like(q),
        "g": torch.empty(scalar_shape),
        "beta": torch.empty(scalar_shape),
        "A": torch.empty((*scalar_shape, 64)),
        "do": torch.empty_like(q),
    }
    cp_context = SimpleNamespace(group=object())
    initial_state = torch.empty((1, 64, 128, 128), dtype=torch.float32)
    dht = torch.empty_like(initial_state)
    expected = tuple(torch.empty(1) for _ in range(5))
    seen = {}

    monkeypatch.setattr(
        implementation, "_fused_bwd_support_reason", lambda **_kwargs: "unsupported"
    )
    monkeypatch.setattr(implementation, "_backend_mode", lambda: "auto")

    def fla_backward(**kwargs):
        seen.update(kwargs)
        return expected

    monkeypatch.setattr(implementation, "_fla_backward", fla_backward)

    result = implementation._cutedsl_backward(
        **inputs,
        scale=128**-0.5,
        dht=dht,
        cu_seqlens=None,
        chunk_indices=None,
        initial_state=initial_state,
        cp_context=cp_context,
    )

    assert result is expected
    assert seen["dht"] is dht
    assert seen["initial_state"] is initial_state
    assert seen["cp_context"] is cp_context


def test_cp_fla_fallback_uses_exp2_gate_semantics(monkeypatch):
    implementation = _implementation()
    tensor = torch.zeros((1, 1, 1, 1), dtype=torch.float32)
    dht = torch.zeros_like(tensor)
    initial_state = torch.zeros_like(tensor)
    calls = {}

    def call_fla_compat(function, **kwargs):
        calls[function] = kwargs
        if function is implementation.recompute_w_u_fwd:
            return tensor, tensor
        if function is implementation.chunk_gated_delta_rule_fwd_h:
            return tensor, tensor, None
        if function is implementation.chunk_bwd_dv_local:
            return tensor
        if function is implementation.chunk_gated_delta_rule_bwd_dhu_pre_process:
            return dht, initial_state
        if function is implementation.chunk_gated_delta_rule_bwd_dhu:
            return tensor, tensor, tensor
        if function is implementation.chunk_bwd_dqkwg:
            return tensor, tensor.clone(), tensor, tensor.clone()
        if function is implementation.prepare_wy_repr_bwd:
            return tensor, tensor, tensor, tensor
        raise AssertionError(f"unexpected FLA primitive: {function}")

    monkeypatch.setattr(implementation, "_call_fla_compat", call_fla_compat)
    monkeypatch.setattr(implementation, "expand_h0", lambda state, *, context: state)
    monkeypatch.setattr(implementation, "chunk_local_cumsum", lambda value, **_kwargs: value)

    implementation._fla_backward(
        q=tensor,
        k=tensor,
        v=tensor,
        g=tensor,
        beta=tensor,
        A=tensor,
        scale=1.0,
        do=tensor,
        cu_seqlens=None,
        chunk_indices=None,
        dht=dht,
        initial_state=initial_state,
        cp_context=SimpleNamespace(group=object()),
    )

    gate_primitives = (
        implementation.recompute_w_u_fwd,
        implementation.chunk_gated_delta_rule_fwd_h,
        implementation.chunk_bwd_dv_local,
        implementation.chunk_gated_delta_rule_bwd_dhu_pre_process,
        implementation.chunk_gated_delta_rule_bwd_dhu,
        implementation.chunk_bwd_dqkwg,
        implementation.prepare_wy_repr_bwd,
    )
    assert all(calls[function]["use_exp2"] is True for function in gate_primitives)


def test_dense_batch_cp_forward_batches_local_compute_and_slices_boundaries(monkeypatch):
    implementation = _implementation()
    shape = (2, 64, 2, 4)
    q = torch.empty(shape, dtype=torch.bfloat16)
    k = torch.empty_like(q)
    v = torch.empty_like(q)
    g = torch.empty(shape[:-1], dtype=torch.float32)
    beta = torch.empty(shape[:-1], dtype=torch.bfloat16)
    w = torch.empty_like(k)
    u = torch.empty_like(v)
    A = torch.empty((*shape[:-1], 64), dtype=torch.bfloat16)
    h = torch.empty((2, 1, 2, 4, 4), dtype=torch.bfloat16)
    output = torch.empty_like(q)
    cp_context = SimpleNamespace(cu_seqlens=torch.tensor([0, 64], dtype=torch.int32))
    calls = {"intra": [], "cp_boundary": [], "fwd_h": [], "output": []}

    monkeypatch.setattr(implementation, "chunk_local_cumsum", lambda value, **_kwargs: value)

    def intra(**kwargs):
        calls["intra"].append(kwargs["k"].shape)
        return w, u, A

    def cp_boundary(**kwargs):
        calls["cp_boundary"].append(kwargs["k"].shape)
        return torch.empty((1, 2, 4, 4), dtype=torch.float32)

    def fwd_h(**kwargs):
        calls["fwd_h"].append((kwargs["k"].shape, kwargs["initial_state"].shape))
        return h, u, None

    def fwd_o(**kwargs):
        calls["output"].append(kwargs["q"].shape)
        return output

    monkeypatch.setattr(implementation, "chunk_gated_delta_rule_fwd_intra", intra)
    monkeypatch.setattr(implementation, "chunk_gated_delta_rule_fwd_h_pre_process", cp_boundary)
    monkeypatch.setattr(implementation, "chunk_gated_delta_rule_fwd_h", fwd_h)
    monkeypatch.setattr(implementation, "chunk_fwd_o", fwd_o)
    monkeypatch.setattr(implementation, "compress_h0", lambda state, *, context: state)

    actual_g, actual_output, actual_A, saved_h, chunk_indices, initial_state = (
        implementation._fla_forward_for_fused_bwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=0.5,
            cu_seqlens=None,
            cp_context=cp_context,
            save_fused_bwd_state=False,
        )
    )

    assert actual_g is g
    assert actual_output is output
    assert actual_A is A
    assert saved_h is None
    assert chunk_indices is None
    assert initial_state.shape == (2, 2, 4, 4)
    assert calls == {
        "intra": [torch.Size([2, 64, 2, 4])],
        "cp_boundary": [torch.Size([1, 64, 2, 4])] * 2,
        "fwd_h": [(torch.Size([2, 64, 2, 4]), torch.Size([2, 2, 4, 4]))],
        "output": [torch.Size([2, 64, 2, 4])],
    }


def test_single_sequence_cp_forward_uses_dense_local_compute(monkeypatch):
    implementation = _implementation()
    shape = (1, 128, 2, 4)
    q = torch.empty(shape, dtype=torch.bfloat16)
    k = torch.empty_like(q)
    v = torch.empty_like(q)
    g = torch.empty(shape[:-1], dtype=torch.float32)
    beta = torch.empty(shape[:-1], dtype=torch.bfloat16)
    w = torch.empty_like(k)
    u = torch.empty_like(v)
    A = torch.empty((*shape[:-1], 64), dtype=torch.bfloat16)
    h = torch.empty((1, 2, 2, 4, 4), dtype=torch.bfloat16)
    output = torch.empty_like(q)
    boundary_state = torch.empty((1, 2, 4, 4), dtype=torch.float32)
    cp_cu_seqlens = torch.tensor([0, 128], dtype=torch.int32)
    cp_context = SimpleNamespace(group=object(), cu_seqlens=cp_cu_seqlens)
    supplied_chunk_indices = torch.tensor([[0, 0], [0, 1]], dtype=torch.int32)
    seen = {}

    def cumsum(value, **kwargs):
        seen["cumsum"] = kwargs
        return value

    def intra(**kwargs):
        seen["intra"] = kwargs
        return w, u, A

    def cp_boundary(**kwargs):
        seen["cp_boundary"] = kwargs
        return boundary_state

    def fwd_h(**kwargs):
        seen["fwd_h"] = kwargs
        return h, u, None

    def fwd_o(**kwargs):
        seen["output"] = kwargs
        return output

    monkeypatch.setattr(implementation, "chunk_local_cumsum", cumsum)
    monkeypatch.setattr(implementation, "chunk_gated_delta_rule_fwd_intra", intra)
    monkeypatch.setattr(implementation, "chunk_gated_delta_rule_fwd_h_pre_process", cp_boundary)
    monkeypatch.setattr(implementation, "chunk_gated_delta_rule_fwd_h", fwd_h)
    monkeypatch.setattr(implementation, "chunk_fwd_o", fwd_o)
    monkeypatch.setattr(implementation, "compress_h0", lambda state, *, context: state)

    actual_g, actual_output, actual_A, saved_h, chunk_indices, initial_state = (
        implementation._fla_forward_for_fused_bwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=0.5,
            cu_seqlens=cp_cu_seqlens,
            chunk_indices=supplied_chunk_indices,
            cp_context=cp_context,
            save_fused_bwd_state=False,
            dense_local_cp=True,
        )
    )

    assert actual_g is g
    assert actual_output is output
    assert actual_A is A
    assert saved_h is None
    assert chunk_indices is None
    assert initial_state is boundary_state
    assert seen["cumsum"]["cu_seqlens"] is None
    assert seen["cumsum"]["chunk_indices"] is None
    assert seen["intra"]["cu_seqlens"] is None
    assert seen["intra"]["chunk_indices"] is None
    assert seen["cp_boundary"]["cu_seqlens"] is cp_cu_seqlens
    assert seen["fwd_h"]["cu_seqlens"] is None
    assert seen["fwd_h"]["chunk_indices"] is None
    assert seen["output"]["cu_seqlens"] is None
    assert seen["output"]["chunk_indices"] is None


def test_single_sequence_cp_backward_preprocess_uses_dense_local_compute(monkeypatch):
    implementation = _implementation()
    shape = (1, 128, 2, 4)
    q = torch.empty(shape, dtype=torch.bfloat16)
    k = torch.empty_like(q)
    v = torch.empty_like(q)
    g = torch.empty(shape[:-1], dtype=torch.float32)
    beta = torch.empty(shape[:-1], dtype=torch.bfloat16)
    A = torch.empty((*shape[:-1], 64), dtype=torch.bfloat16)
    do = torch.empty_like(q)
    w = torch.empty_like(k)
    u = torch.empty_like(v)
    dv = torch.empty_like(v)
    compressed_initial_state = torch.empty((1, 2, 4, 4), dtype=torch.float32)
    expanded_initial_state = torch.empty_like(compressed_initial_state)
    h = torch.empty((1, 2, 2, 4, 4), dtype=torch.bfloat16)
    expected_dht = torch.empty_like(compressed_initial_state)
    cp_cu_seqlens = torch.tensor([0, 128], dtype=torch.int32)
    cp_context = SimpleNamespace(group=object(), cu_seqlens=cp_cu_seqlens)
    supplied_chunk_indices = torch.tensor([[0, 0], [0, 1]], dtype=torch.int32)
    seen = {}

    def recompute(**kwargs):
        seen["recompute"] = kwargs
        return w, u

    def dv_local(**kwargs):
        seen["dv"] = kwargs
        return dv

    def cp_boundary(**kwargs):
        seen["cp_boundary"] = kwargs
        return expected_dht, None

    monkeypatch.setattr(implementation, "recompute_w_u_fwd", recompute)
    monkeypatch.setattr(
        implementation, "expand_h0", lambda initial_state, *, context: expanded_initial_state
    )
    monkeypatch.setattr(implementation, "chunk_bwd_dv_local", dv_local)
    monkeypatch.setattr(implementation, "chunk_gated_delta_rule_bwd_dhu_pre_process", cp_boundary)

    actual_dht, actual_h = implementation._fla_cp_backward_preprocess(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        A=A,
        scale=0.5,
        do=do,
        dht=None,
        cu_seqlens=cp_cu_seqlens,
        chunk_indices=supplied_chunk_indices,
        initial_state=compressed_initial_state,
        cp_context=cp_context,
        h=h,
    )

    assert actual_dht is expected_dht
    assert actual_h is h
    assert seen["recompute"]["cu_seqlens"] is None
    assert seen["recompute"]["chunk_indices"] is None
    assert seen["dv"]["cu_seqlens"] is None
    assert seen["dv"]["chunk_indices"] is None
    assert seen["cp_boundary"]["cu_seqlens"] is cp_cu_seqlens
    assert seen["cp_boundary"]["initial_state"] is expanded_initial_state


def test_dense_batch_cp_backward_batches_local_compute_and_slices_boundaries(monkeypatch):
    implementation = _implementation()
    shape = (2, 64, 2, 4)
    q = torch.zeros(shape, dtype=torch.bfloat16)
    q[1].fill_(1)
    k = torch.empty_like(q)
    v = torch.empty_like(q)
    g = torch.empty(shape[:-1], dtype=torch.float32)
    beta = torch.empty(shape[:-1], dtype=torch.bfloat16)
    A = torch.empty((*shape[:-1], 64), dtype=torch.bfloat16)
    do = torch.empty_like(q)
    w = torch.empty_like(k)
    u = torch.empty_like(v)
    dv = torch.empty_like(v)
    initial_state = torch.empty((2, 2, 4, 4), dtype=torch.float32)
    h = torch.empty((2, 1, 2, 4, 4), dtype=torch.bfloat16)
    cp_context = SimpleNamespace(cu_seqlens=torch.tensor([0, 64], dtype=torch.int32))
    calls = {"recompute": [], "dv": [], "cp_boundary": []}

    def recompute(**kwargs):
        calls["recompute"].append(kwargs["k"].shape)
        return w, u

    def dv_local(**kwargs):
        calls["dv"].append(kwargs["q"].shape)
        return dv

    def cp_boundary(**kwargs):
        calls["cp_boundary"].append(kwargs["q"].shape)
        batch_value = int(kwargs["q"][0, 0, 0, 0].item())
        dht = torch.full((1, 2, 4, 4), batch_value, dtype=torch.float32)
        return dht, None

    monkeypatch.setattr(implementation, "recompute_w_u_fwd", recompute)
    monkeypatch.setattr(implementation, "expand_h0", lambda state, *, context: state)
    monkeypatch.setattr(implementation, "chunk_bwd_dv_local", dv_local)
    monkeypatch.setattr(implementation, "chunk_gated_delta_rule_bwd_dhu_pre_process", cp_boundary)

    actual_dht, actual_h = implementation._fla_cp_backward_preprocess(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        A=A,
        scale=0.5,
        do=do,
        dht=None,
        cu_seqlens=None,
        chunk_indices=None,
        initial_state=initial_state,
        cp_context=cp_context,
        h=h,
    )

    assert actual_h is h
    assert actual_dht.shape == (2, 2, 4, 4)
    assert torch.equal(actual_dht[:, 0, 0, 0], torch.tensor([0.0, 1.0]))
    assert calls == {
        "recompute": [torch.Size([2, 64, 2, 4])],
        "dv": [torch.Size([2, 64, 2, 4])],
        "cp_boundary": [torch.Size([1, 64, 2, 4])] * 2,
    }


def test_dense_cu_seqlens_reuses_cached_tensor():
    implementation = _implementation()
    first = implementation._dense_cu_seqlens(2, 64, torch.device("cpu"))
    second = implementation._dense_cu_seqlens(2, 64, torch.device("cpu"))

    assert first is second
    assert first.tolist() == [0, 64, 128]


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
        return kwargs["g"], q, torch.empty(1), h, None, None

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
        None,
        recompute_h,
        None,
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

    _g, output, _A, saved_h, _chunk_indices, _initial_state = (
        implementation._fla_forward_for_fused_bwd(
            **inputs, scale=0.5, cu_seqlens=None, save_fused_bwd_state=save_fused_bwd_state
        )
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
        **inputs, scale=0.5, cu_seqlens=None, save_fused_bwd_state=save_fused_bwd_state
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
        **inputs, scale=0.5, cu_seqlens=None, save_fused_bwd_state=True
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
        **inputs, scale=0.5, cu_seqlens=cu_seqlens, save_fused_bwd_state=False
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

    metadata = implementation._packed_chunk_metadata(cu_seqlens)
    _g, _output, _A, _saved_h, actual_indices, chunk_offsets = implementation._cutedsl_forward(
        **inputs,
        scale=0.5,
        cu_seqlens=cu_seqlens,
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


def test_prepare_validated_chunk_metadata_can_skip_chunk_indices(monkeypatch):
    implementation = _implementation()

    implementation._clear_packed_chunk_metadata_cache_for_test()
    cu_seqlens = torch.tensor([0, 128], dtype=torch.int32)

    def fail_prepare(*_args, **_kwargs):
        pytest.fail("offset-only metadata must not prepare chunk_indices")

    monkeypatch.setattr(implementation, "prepare_chunk_indices", fail_prepare)

    chunk_indices, chunk_offsets = implementation.prepare_validated_chunk_metadata(
        cu_seqlens, include_chunk_indices=False
    )

    assert chunk_indices is None
    assert torch.equal(chunk_offsets, torch.tensor([0, 2], dtype=torch.int32))


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

    first = implementation._packed_chunk_metadata(cu_seqlens)
    second = implementation._packed_chunk_metadata(cu_seqlens)
    third = implementation._packed_chunk_metadata(cu_seqlens)

    assert len(prepare_calls) == 2
    assert second is not first
    assert third is first


def test_packed_chunk_metadata_cache_tracks_device_tensor_version(monkeypatch):
    implementation = _implementation()

    implementation._clear_packed_chunk_metadata_cache_for_test()
    cu_seqlens = torch.tensor([0, 64, 128], dtype=torch.int32)
    prepare_calls = []

    def prepare(cu_arg, *_args, **_kwargs):
        prepare_calls.append((cu_arg, cu_arg._version))
        return torch.full((2, 2), len(prepare_calls), dtype=torch.int32)

    monkeypatch.setattr(implementation, "_current_stream_cache_key", lambda _tensor: None)
    monkeypatch.setattr(implementation, "prepare_chunk_indices", prepare)

    first = implementation._packed_chunk_metadata(cu_seqlens)
    second = implementation._packed_chunk_metadata(cu_seqlens)
    cu_seqlens[1] = 128
    third = implementation._packed_chunk_metadata(cu_seqlens)

    assert second is first
    assert third is not first
    assert prepare_calls == [(cu_seqlens, 0), (cu_seqlens, 1)]


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
        inputs["q"], inputs["k"], inputs["v"], inputs["g"], inputs["beta"], None
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


def test_fused_backward_zero_dht_cache_evicts_to_byte_budget(monkeypatch):
    implementation = _implementation()

    implementation._fused_bwd_zero_dht_cache.clear()
    monkeypatch.setattr(implementation, "_FUSED_BWD_ZERO_DHT_CACHE_MAX_BYTES", 5 * 1024 * 1024)
    stream_keys = iter([(0, 1), (0, 2)])
    monkeypatch.setattr(
        implementation, "_current_stream_cache_key", lambda _device: next(stream_keys)
    )

    first = implementation._fused_bwd_zero_dht(torch.device("cpu"), 1)
    second = implementation._fused_bwd_zero_dht(torch.device("cpu"), 1)

    assert second is not first
    assert len(implementation._fused_bwd_zero_dht_cache) == 1
    assert next(iter(implementation._fused_bwd_zero_dht_cache.values())) is second


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
    g = torch.full(scalar_shape, implementation.RCP_LN2, dtype=torch.float32)
    beta = torch.empty_like(g)
    A = torch.empty((*scalar_shape, 64), dtype=torch.bfloat16)
    do = torch.empty_like(q)
    h = torch.empty((batch_size, 64, 128, 128), dtype=torch.bfloat16)
    total_tokens = batch_size * 64
    packed = q.reshape(1, total_tokens, 64, 128)
    packed_scalar = g.reshape(1, total_tokens, 64)
    final_dg = torch.full_like(g, 6.0)
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
    cumsum_calls = []

    def fused(**kwargs):
        seen.update(kwargs)
        return fused_outputs

    def recompute(**_kwargs):
        recompute_calls.append(True)
        return h

    def cumsum(value, **kwargs):
        cumsum_calls.append(kwargs)
        torch.testing.assert_close(value, fused_outputs[3].reshape_as(g))
        assert kwargs["chunk_size"] == 64
        assert kwargs["reverse"] is True
        assert kwargs["cu_seqlens"] is None
        assert kwargs["chunk_indices"] is None
        return final_dg

    implementation._clear_dense_chunk_metadata_cache_for_test()
    dense_metadata = implementation._dense_chunk_metadata(batch_size, 64, q.device)
    monkeypatch.setattr(implementation, "_recompute_fused_bwd_h", recompute)
    monkeypatch.setattr(implementation, "chunk_local_cumsum", cumsum)
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
    torch.testing.assert_close(
        seen["g"], g.reshape(1, total_tokens, 64).float() / implementation.RCP_LN2
    )
    assert cumsum_calls == [
        {"chunk_size": 64, "reverse": True, "cu_seqlens": None, "chunk_indices": None}
    ]
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
    assert dg.shape == g.shape and torch.all(dg == final_dg)
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

    monkeypatch.setattr(implementation, "chunk_local_cumsum", lambda value, **_kwargs: value)
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
    assert seen["trusted_chunk_offsets"] is True


def test_gdn_pre_gdr_producers_emit_fp32_beta():
    package = Path(__file__).parents[3] / "megatron/core"
    gdn_source = (package / "ssm/gated_delta_net/gdn.py").read_text()
    fused_pre_source = (package / "fusions/fused_pre_gated_delta_rule.py").read_text()

    assert "beta = beta.float().sigmoid()" in gdn_source
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
    monkeypatch.setattr(fused_bwd, "_current_stream_cache_key", lambda _tensor: next(stream_keys))

    first = fused_bwd._prepare_varlen_metadata(cu_seqlens, total_tokens=128, chunk_size=64)
    second = fused_bwd._prepare_varlen_metadata(cu_seqlens, total_tokens=128, chunk_size=64)
    third = fused_bwd._prepare_varlen_metadata(cu_seqlens, total_tokens=128, chunk_size=64)

    assert second is not first
    assert third is first


def test_fused_backward_varlen_metadata_reuses_trusted_supplied_chunk_offsets():
    from megatron.core.ssm.gated_delta_net.internal_gdn_backend.kernels.fused_gdr_bwd_cute import (
        fused_bwd,
    )

    fused_bwd._clear_metadata_cache_for_test()
    cu_seqlens = torch.empty(3, dtype=torch.int32, device="meta")
    chunk_offsets = torch.empty(3, dtype=torch.int32, device="meta")

    metadata = fused_bwd._prepare_varlen_metadata(
        cu_seqlens,
        total_tokens=128,
        chunk_size=64,
        chunk_offsets=chunk_offsets,
        trusted_chunk_offsets=True,
    )

    assert metadata.chunk_offsets is chunk_offsets
    assert metadata.num_sequences == 2
    assert metadata.num_chunks == 2
    assert metadata.uniform_sequence_length == 0


def test_fused_backward_varlen_metadata_rejects_untrusted_device_chunk_offsets():
    from megatron.core.ssm.gated_delta_net.internal_gdn_backend.kernels.fused_gdr_bwd_cute import (
        fused_bwd,
    )

    fused_bwd._clear_metadata_cache_for_test()
    cu_seqlens = torch.empty(3, dtype=torch.int32, device="meta")
    chunk_offsets = torch.empty(3, dtype=torch.int32, device="meta")

    with pytest.raises(ValueError, match="trusted_chunk_offsets=True"):
        fused_bwd._prepare_varlen_metadata(
            cu_seqlens, total_tokens=128, chunk_size=64, chunk_offsets=chunk_offsets
        )


def test_fused_backward_varlen_metadata_validates_cpu_supplied_chunk_offsets():
    from megatron.core.ssm.gated_delta_net.internal_gdn_backend.kernels.fused_gdr_bwd_cute import (
        fused_bwd,
    )

    fused_bwd._clear_metadata_cache_for_test()
    cu_seqlens = torch.tensor([0, 64, 128], dtype=torch.int32)
    chunk_offsets = torch.tensor([0, 1, 2], dtype=torch.int32)

    metadata = fused_bwd._prepare_varlen_metadata(
        cu_seqlens, total_tokens=128, chunk_size=64, chunk_offsets=chunk_offsets
    )

    assert metadata.chunk_offsets is chunk_offsets


def test_fused_backward_support_reason_accepts_device_offsets_without_host_metadata():
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

    assert implementation._fused_bwd_support_reason(**kwargs) is None


@pytest.mark.parametrize(
    "dtype, batch_size, seqlen, offsets, expected_reason",
    [
        (torch.bfloat16, 2, 64, None, None),
        (torch.bfloat16, 2, 65, None, None),
        (torch.float16, 2, 64, None, "fused backward requires bf16 inputs"),
        (torch.bfloat16, 1, 256, [0, 64, 192, 256], None),
        (torch.bfloat16, 1, 257, [0, 65, 192, 257], None),
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


def test_fused_backward_metadata_uses_per_sequence_ceil_chunks():
    from megatron.core.ssm.gated_delta_net.internal_gdn_backend.kernels.fused_gdr_bwd_cute import (
        fused_bwd,
    )

    offsets = torch.tensor([0, 65, 128], dtype=torch.int32)
    metadata = fused_bwd._prepare_varlen_metadata(offsets, total_tokens=128, chunk_size=64)

    assert metadata.chunk_offsets.tolist() == [0, 2, 3]
    assert metadata.num_sequences == 2
    assert metadata.num_chunks == 3
    assert metadata.uniform_sequence_length == 0
    assert metadata.has_partial_chunks is True

    uniform_tail_offsets = torch.tensor([0, 65, 130], dtype=torch.int32)
    uniform_tail = fused_bwd._prepare_varlen_metadata(
        uniform_tail_offsets, total_tokens=130, chunk_size=64
    )
    assert uniform_tail.uniform_sequence_length == 0
    assert uniform_tail.has_partial_chunks is True


def test_fused_backward_kernel_uses_named_layout_contracts():
    package = (
        Path(__file__).parents[3]
        / "megatron/core/ssm/gated_delta_net/internal_gdn_backend/kernels"
        / "fused_gdr_bwd_cute"
    )
    kernel_source = (package / "kernel.py").read_text()
    for positional_wiring in ("mma_variants[", "packed_layouts[", "tma_inputs["):
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
