# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU tests for the per-key ``gk`` (KDA) gate mode of the CuTeDSL CP dispatcher.

The dispatcher module imports only torch, so everything here runs without FLA
or the CuTe DSL toolchain.  CPU tensors can only ever fail the dispatcher's
runtime probe (``k.is_cuda``), which runs after every gate, shape, dtype,
contiguity and device predicate -- so reaching that probe is proof that a
``gk`` input passed all of them.  The wrapper modules, which do need CUTLASS,
are stubbed into ``sys.modules`` where the hook flow is exercised.
"""

import inspect
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from megatron.core.ssm.gated_delta_net.internal_gdn_backend.kernels.fused_gdr_cp_cute import (
    backend as cp_backend,
)

_T, _H, _K = 64, 8, 128
_WORLD, _RANK = 4, 1
_RUNTIME_REASON = "runtime unsupported (device/toolchain/backend/world size)"
_PACKAGE = "megatron.core.ssm.gated_delta_net.internal_gdn_backend.kernels.fused_gdr_cp_cute"
_GATE_CODE = {"g": 1, "gk": 2}


def _context(local_offsets=(0, _T), global_offsets=(0, _WORLD * _T), pre=1, post=2):
    """A CP context as `prepare_cp_context_metadata` leaves it, for rank 1 of 4.

    The defaults are one global sequence spanning the whole group; the packed
    variants used below put a boundary inside rank 1's window.
    """
    return SimpleNamespace(
        group=object(),
        cu_seqlens_cpu=torch.tensor(local_offsets, dtype=torch.int32),
        global_num_seqs=len(global_offsets) - 1,
        global_cu_seqlens_cpu=torch.tensor(global_offsets, dtype=torch.long),
        pre_num_ranks=pre,
        post_num_ranks=post,
    )


def _gate(gate, H=_H):
    if gate == "g":
        return torch.empty((1, _T, H), dtype=torch.float32)
    if gate == "gk":
        return torch.empty((1, _T, H, _K), dtype=torch.float32)
    return None


def _operands(direction, gate, H=_H):
    base = torch.empty((1, _T, H, _K), dtype=torch.bfloat16)
    gates = {name: _gate(name, H) if gate == name else None for name in ("g", "gk")}
    if direction == "forward":
        return {
            "k": base,
            "w": torch.empty_like(base),
            "u": torch.empty_like(base),
            **gates,
            "bg": None,
            "v": None,
        }
    return {
        "q": base,
        "k": torch.empty_like(base),
        "w": torch.empty_like(base),
        "do": torch.empty_like(base),
        "dv": torch.empty_like(base),
        **gates,
        "bg": None,
        "dht": None,
    }


def _is_supported(direction, operands, context):
    predicate = cp_backend._is_supported if direction == "forward" else cp_backend._is_supported_bwd
    return predicate(**operands, chunk_size=64, cu_seqlens=context.cu_seqlens_cpu, context=context)


def _reason(direction, reason):
    return reason if direction == "forward" else f"bwd: {reason}"


@pytest.fixture
def dispatcher(monkeypatch):
    monkeypatch.setattr(cp_backend, "_group_topology", lambda _group: (_WORLD, _RANK))
    monkeypatch.setattr(cp_backend, "_backend_mode", lambda: "auto")
    cp_backend.reset_cutedsl_fallback_reasons()
    yield cp_backend
    cp_backend.reset_cutedsl_fallback_reasons()


# ----------------------------------------------------------------- 7.1 item 1


@pytest.mark.parametrize("direction", ["forward", "backward"])
@pytest.mark.parametrize("gate", ["g", "gk"])
def test_dispatcher_admits_both_gate_modes_up_to_the_runtime_probe(dispatcher, direction, gate):
    operands = _operands(direction, gate)

    assert _is_supported(direction, operands, _context()) is None

    assert dispatcher.get_cutedsl_fallback_reasons() == {_reason(direction, _RUNTIME_REASON): 1}


def test_scalar_only_gate_rejection_is_gone():
    assert "gate is not scalar g" not in inspect.getsource(cp_backend)


@pytest.mark.parametrize("direction", ["forward", "backward"])
@pytest.mark.parametrize(
    "mutate, reason",
    [
        (lambda ops: ops.update(g=_gate("g")), "gate must be exactly one of g or gk"),
        (lambda ops: ops.update(gk=None), "gate must be exactly one of g or gk"),
        (lambda ops: ops.update(bg=_gate("gk")), "DPLR bg gate unsupported"),
        (
            lambda ops: ops.update(gk=ops["gk"].to(torch.bfloat16)),
            "gk dtype torch.bfloat16 (only float32)",
        ),
        (
            lambda ops: ops.update(gk=torch.empty((1, _T, _H, 64), dtype=torch.float32)),
            "gk shape mismatch",
        ),
        (
            lambda ops: ops.update(gk=torch.empty((1, _T, _H + 4, _K), dtype=torch.float32)),
            "gk shape mismatch",
        ),
        (lambda ops: ops.update(gk=_gate("g")), "operand rank mismatch"),
        (
            lambda ops: ops.update(
                gk=torch.empty((1, _T, _H, 2 * _K), dtype=torch.float32)[..., ::2]
            ),
            "non-contiguous operand",
        ),
    ],
    ids=["both_gates", "no_gate", "bg", "bf16", "wrong_K", "wrong_HV", "rank3", "strided"],
)
def test_dispatcher_rejects_malformed_gk_inputs(dispatcher, direction, mutate, reason):
    operands = _operands(direction, "gk")
    mutate(operands)

    assert _is_supported(direction, operands, _context()) is None

    assert dispatcher.get_cutedsl_fallback_reasons() == {_reason(direction, reason): 1}


@pytest.mark.parametrize("direction", ["forward", "backward"])
def test_dispatcher_keeps_head_multiple_of_four_in_gk_mode(dispatcher, direction):
    # Only the g TMA box needs HV % 4 == 0, but the dispatcher keeps one head
    # geometry contract for both modes (every KDA production shard satisfies it).
    operands = _operands(direction, "gk", H=6)

    assert _is_supported(direction, operands, _context()) is None

    assert dispatcher.get_cutedsl_fallback_reasons() == {
        _reason(direction, "head geometry H=6,HV=6"): 1
    }


@pytest.mark.parametrize("direction", ["forward", "backward"])
@pytest.mark.parametrize(
    "mutate, reason",
    [
        (lambda ops: None, _RUNTIME_REASON),
        (lambda ops: ops.update(g=ops["g"].to(torch.bfloat16)), _RUNTIME_REASON),
        (lambda ops: ops.update(g=ops["g"].to(torch.float16)), "gate dtype torch.float16"),
        (
            lambda ops: ops.update(g=torch.empty((1, _T, _H + 4), dtype=torch.float32)),
            "operand shape mismatch",
        ),
        (
            lambda ops: ops.update(g=torch.empty((1, _T, _H, 1), dtype=torch.float32)),
            "operand rank mismatch",
        ),
        (
            lambda ops: ops.update(g=torch.empty((1, _T, 2 * _H), dtype=torch.float32)[..., ::2]),
            "non-contiguous operand",
        ),
        (lambda ops: ops.update(bg=_gate("gk")), "DPLR bg gate unsupported"),
    ],
    ids=["fp32", "bf16", "fp16", "wrong_HV", "rank4", "strided", "bg"],
)
def test_dispatcher_keeps_scalar_g_decisions(dispatcher, direction, mutate, reason):
    # Pins the g-mode (GDN) decision table across the gk change.  Only the `bg`
    # row's reason string moved (from the retired "gate is not scalar g ..."),
    # and GDN never supplies `bg`.
    operands = _operands(direction, "g")
    mutate(operands)

    assert _is_supported(direction, operands, _context()) is None

    assert dispatcher.get_cutedsl_fallback_reasons() == {_reason(direction, reason): 1}


# ----------------------------------------------------------------- 7.1 item 2


def test_wrapper_cache_key_and_signature_carry_the_gate_mode():
    group, device = object(), torch.device("cuda", 0)

    g_key = cp_backend._wrapper_cache_key(group, device, _H, _H, _K, _K, "g")
    gk_key = cp_backend._wrapper_cache_key(group, device, _H, _H, _K, _K, "gk")
    assert g_key != gk_key
    assert g_key[:-1] == gk_key[:-1] == (id(group), 0, _H, _H, _K, _K)
    assert (g_key[-1], gk_key[-1]) == ("g", "gk")

    for direction in (0, 1):
        g_signature = cp_backend._wrapper_signature(direction, _H, _H, _K, _K, "g")
        gk_signature = cp_backend._wrapper_signature(direction, _H, _H, _K, _K, "gk")
        assert g_signature != gk_signature
        assert g_signature[:-1] == gk_signature[:-1] == (direction, _H, _H, _K, _K)
        assert (g_signature[-1], gk_signature[-1]) == (_GATE_CODE["g"], _GATE_CODE["gk"])
        # The handshake all-gathers it as int64, so every element must be an int.
        assert all(type(value) is int for value in g_signature + gk_signature)
        assert torch.tensor(g_signature, dtype=torch.int64).shape == (6,)

    assert cp_backend._wrapper_signature(0, _H, _H, _K, _K, "gk") != cp_backend._wrapper_signature(
        1, _H, _H, _K, _K, "gk"
    )
    with pytest.raises(KeyError):
        cp_backend._wrapper_signature(0, _H, _H, _K, _K, "none")


class _FakeWrapper:
    """Stand-in for the CUTLASS host wrappers: records construction and launches."""

    def __init__(self, group, H, HV, K, V, gate_mode="g", device=None):
        self.init = {"group": group, "H": H, "HV": HV, "K": K, "V": V, "gate_mode": gate_mode}
        self.launches = []

    def launch_validated(self, *args, **kwargs):
        self.launches.append((args, kwargs))


def _stub_wrapper_module(monkeypatch, module, class_name):
    stub = ModuleType(f"{_PACKAGE}.{module}")
    created = []

    def factory(*args, **kwargs):
        created.append(_FakeWrapper(*args, **kwargs))
        return created[-1]

    setattr(stub, class_name, factory)
    monkeypatch.setitem(sys.modules, stub.__name__, stub)
    return created


@pytest.fixture
def hook_harness(monkeypatch):
    """Route both hooks through fake wrappers on CPU and record the handshake."""
    signatures = []
    monkeypatch.setattr(cp_backend, "_WRAPPER_CACHE", {})
    monkeypatch.setattr(cp_backend, "_BWD_WRAPPER_CACHE", {})
    monkeypatch.setattr(cp_backend, "_is_supported", lambda **_kwargs: "fused")
    monkeypatch.setattr(cp_backend, "_is_supported_bwd", lambda **_kwargs: "fused")
    monkeypatch.setattr(cp_backend, "_group_topology", lambda _group: (_WORLD, _RANK))
    monkeypatch.setattr(cp_backend, "_SerializedWrapper", lambda wrapper, device: wrapper)
    monkeypatch.setattr(
        cp_backend,
        "_rank_consistent_wrapper_init",
        lambda group, device, signature: signatures.append(signature) or True,
    )
    created = {
        "forward": _stub_wrapper_module(monkeypatch, "fused_ws", "CuteDSLFusedCPPreProcessWS"),
        "backward": _stub_wrapper_module(monkeypatch, "bwd_ws", "CuteDSLFusedCPBwdPreProcessWS"),
    }
    return SimpleNamespace(signatures=signatures, created=created)


def _call_hook(direction, operands, context):
    if direction == "forward":
        return cp_backend.try_chunk_gated_delta_rule_fwd_h_pre_process_cutedsl(
            **operands,
            chunk_size=64,
            state_v_first=False,
            cu_seqlens=context.cu_seqlens_cpu,
            context=context,
        )
    return cp_backend.try_chunk_gated_delta_rule_bwd_dhu_pre_process_cutedsl(
        **operands,
        scale=None,
        chunk_size=64,
        state_v_first=False,
        cu_seqlens=context.cu_seqlens_cpu,
        context=context,
    )


@pytest.mark.parametrize("direction", ["forward", "backward"])
@pytest.mark.parametrize("gate", ["g", "gk"])
def test_hooks_build_and_launch_the_wrapper_of_the_supplied_gate(hook_harness, direction, gate):
    # Rank 1 of 4 holds two local sequences [0, 30) and [30, 64) (global cut at
    # 94): the forward producer window is the LAST one, the backward the FIRST.
    context = _context(local_offsets=(0, 30, 64), global_offsets=(0, 94, 256), pre=1, post=2)
    operands = _operands(direction, gate)
    other_gate = "gk" if gate == "g" else "g"
    counter = (
        cp_backend.get_cutedsl_fused_launch_count
        if direction == "forward"
        else cp_backend.get_cutedsl_fused_bwd_launch_count
    )
    launches_before = counter()

    state = _call_hook(direction, operands, context)

    assert counter() == launches_before + 1
    assert state.shape == (2, _H, _K, _K) and state.dtype == torch.float32
    [wrapper] = hook_harness.created[direction]
    assert wrapper.init["gate_mode"] == gate
    assert wrapper.init["group"] is context.group
    assert hook_harness.signatures == [
        (0 if direction == "forward" else 1, _H, _H, _K, _K, _GATE_CODE[gate])
    ]

    [(args, kwargs)] = wrapper.launches
    assert len(args) == (3 if direction == "forward" else 5)
    assert kwargs[other_gate] is None
    if direction == "forward":
        bos, eos = 30, 64
        assert kwargs["h0_out"].data_ptr() == state[0].data_ptr()
    else:
        bos, eos = 0, 30
        assert kwargs["dht_out"].data_ptr() == state[-1].data_ptr()
        assert kwargs["scale"] == _K**-0.5
    assert (kwargs["emit_h"], kwargs["emit_m"]) == (True, False)
    window = kwargs[gate]
    assert window.shape[1] == eos - bos and window.shape[2:] == operands[gate].shape[2:]
    assert window.data_ptr() == operands[gate][:, bos:eos].data_ptr()
    assert window.is_contiguous()

    # A second call hits the cache: no handshake, no construction, one more launch.
    _call_hook(direction, operands, context)
    assert len(hook_harness.created[direction]) == 1
    assert len(hook_harness.signatures) == 1
    assert len(wrapper.launches) == 2


@pytest.mark.parametrize("direction", ["forward", "backward"])
def test_hooks_cache_g_and_gk_wrappers_separately(hook_harness, direction):
    context = _context()

    _call_hook(direction, _operands(direction, "g"), context)
    _call_hook(direction, _operands(direction, "gk"), context)

    created = hook_harness.created[direction]
    assert [wrapper.init["gate_mode"] for wrapper in created] == ["g", "gk"]
    code = 0 if direction == "forward" else 1
    assert hook_harness.signatures == [(code, _H, _H, _K, _K, 1), (code, _H, _H, _K, _K, 2)]


# -------------------------------------------------- shim contract (spec section 4)


@pytest.mark.parametrize("direction", ["forward", "backward"])
@pytest.mark.parametrize(
    "gates", [{}, {"g": _gate("g"), "gk": _gate("gk")}], ids=["neither", "both"]
)
def test_cp_preprocess_shims_require_exactly_one_gate(direction, gates):
    pytest.importorskip("fla")
    from megatron.core.ssm.gated_delta_net.internal_gdn_backend import implementation

    base = torch.empty((1, _T, _H, _K), dtype=torch.bfloat16)
    if direction == "forward":
        shim = implementation._cp_forward_preprocess
        kwargs = {"k": base, "w": base, "u": base, "cu_seqlens": None, "initial_state": None}
    else:
        shim = implementation._cp_backward_preprocess
        kwargs = {
            "q": base,
            "k": base,
            "w": base,
            "do": base,
            "dv": base,
            "scale": 0.5,
            "cu_seqlens": None,
            "dht": None,
            "initial_state": None,
        }

    with pytest.raises(ValueError, match="exactly one of g / gk"):
        shim(**kwargs, **gates, context=SimpleNamespace(group=object()))
