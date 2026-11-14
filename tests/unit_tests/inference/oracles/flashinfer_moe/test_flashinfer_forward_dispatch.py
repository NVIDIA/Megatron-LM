# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Oracle for the FlashInfer dispatch of ``InferenceGroupedMLP``
(``megatron/core/transformer/moe/experts.py``): ``_resolve_flashinfer_activation_type`` and
``forward`` -> ``_flashinfer_forward`` -> the four FlashInfer MoE call sites.

Contract (from the config semantics and the call sites; written before reading the dispatch):

* Activation (``config.activation_func``, ``gated_linear_unit``,
  ``activation_func_tanh_clamp_scale``) -> FlashInfer ``ActivationType``: ``F.silu`` gated ->
  ``Swiglu``, plain -> ``Silu``; ``F.gelu`` -> ``Gelu``; ``F.relu`` -> ``Relu``; ``squared_relu``
  -> ``Relu2``, or ``ClampedRelu2`` with a clamp scale (RuntimeError when the installed FlashInfer
  lacks ``ClampedRelu2`` or ``cutlass_fused_moe(clamped_relu2_limit=)``); anything else ->
  ValueError. Config validation lets only ``Relu2``, ``ClampedRelu2`` and (NVFP4) ``Swiglu``
  reach a kernel.
* In inference with the FlashInfer backend, ``forward`` requires ``routing_map`` and eval mode
  and returns ``_flashinfer_forward(hidden, routing_map, probs)``, which requires FP32
  probabilities and a clamp only with ``ClampedRelu2``, and dispatches by weight format: MXFP8
  weights in the CUTLASS layout -> ``flashinfer_cutlass_mxfp8_moe`` (EP size and rank, the
  activation, the clamp limit); MXFP8 in the routed layout -> ``flashinfer_routed_mxfp8_moe``
  (``num_experts = E_local * EP``, ``local_expert_offset = rank * E_local``, the configured token
  capacity and the step's ``InferenceMode.use_bounded_mxfp8_rows()``); mixed MXFP8 layouts ->
  TypeError; per-token NVFP4 weights -> ``flashinfer_routed_nvfp4_moe`` (the same EP arithmetic,
  capacity and bound); otherwise BF16 ``cutlass_fused_moe``. With the NVLS dispatcher the MXFP8
  and NVFP4 paths write into its FP32 buffer (``out``); the BF16 path returns its own BF16
  output and leaves the buffer to the dispatcher. It returns ``(output, None)``.
* Each path computes the MoE of the configured activation, over the rank's local experts.

Oracle: the activation table is checked against FlashInfer's enum by name (exact). Each dispatch
path runs the real kernel, with the activation type resolved from the config (never set by
hand), against the fp64 oracle of that kernel's module (``Bf16Problem`` and the MXFP8
ClampedRelu2 inputs of ``test_cutlass_fused_moe.py``, ``Problem`` of
``test_trtllm_fp8_block_scale_routed_moe.py`` and of ``test_cute_dsl_moe_wrapper.py``), whose
activation follows the config's meaning; tolerance class as in those modules (per row, factor
2). The NVLS buffer is a guarded, NaN-poisoned FP32 tensor: rows past the bound must stay
poisoned, and the BF16 path must not touch it. Guards run on any device.

Variants: every path (BF16 Relu2 on H100 and Blackwell; BF16 ClampedRelu2, MXFP8 CUTLASS
ClampedRelu2, routed MXFP8 Relu2, NVFP4 Relu2 and SwiGLU on Blackwell) x {no NVLS, NVLS, NVLS with
bounded rows (routed paths)}, EP (2, 1) and (4, 2), and a production routed case (H 2688, I 1856,
128 experts, EP 16 rank 8, top-6). Not covered: the lazy weight build in ``forward`` (it needs TE
grouped linears; the per-token NVFP4 build is covered by the upstream ``TestNVFP4MoELayer``) and
``refresh_cache``.
"""

import types
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from tests.unit_tests.inference.oracles import oracle_utils as ou
from tests.unit_tests.inference.oracles.flashinfer_moe import spec
from tests.unit_tests.inference.oracles.flashinfer_moe import test_cute_dsl_moe_wrapper as nvfp4
from tests.unit_tests.inference.oracles.flashinfer_moe import test_cutlass_fused_moe as cutlass
from tests.unit_tests.inference.oracles.flashinfer_moe import (
    test_trtllm_fp8_block_scale_routed_moe as routed,
)


def _activation_type():
    spec.require_flashinfer()
    from flashinfer.fused_moe.core import ActivationType

    return ActivationType


def _config(func, *, gated=False, clamp=None):
    return SimpleNamespace(
        activation_func=func,
        gated_linear_unit=gated,
        activation_func_tanh_clamp_scale=clamp,
        deterministic_mode=False,
        batch_invariant_mode=False,
    )


def _squared_relu():
    from megatron.core.activations import squared_relu

    return squared_relu


def _module(config, *, fc1=None, fc2=None, nvfp4_weights=None, mxfp8=False, local, ep, nvls=False):
    """The attributes ``InferenceGroupedMLP.forward`` and ``_flashinfer_forward`` read, with the
    activation type resolved from ``config`` by Megatron itself."""
    from megatron.core.inference.moe import InferenceGroupedGemmBackend
    from megatron.core.transformer.moe.experts import InferenceGroupedMLP

    clamp = config.activation_func_tanh_clamp_scale
    module = SimpleNamespace(
        config=config,
        training=False,
        inference_grouped_gemm_backend=InferenceGroupedGemmBackend.FLASHINFER,
        _concatenated_weights_built=True,
        _uses_mxfp8_weights=mxfp8,
        _fc1_weight=fc1,
        _fc2_weight=fc2,
        _nvfp4_weights=nvfp4_weights,
        _activation_clamp_scale=clamp,
        _flashinfer_clamp_limit=(
            None if clamp is None else torch.full((1,), clamp, dtype=torch.float32, device="cuda")
        ),
        _nvls_dispatcher=nvls,
        _flashinfer_mxfp8_token_capacity=None,
        num_local_experts=local,
        ep_group=SimpleNamespace(size=lambda: ep[0], rank=lambda: ep[1]),
    )
    module._flashinfer_activation_type = InferenceGroupedMLP._resolve_flashinfer_activation_type(
        module
    )
    module._flashinfer_forward = types.MethodType(InferenceGroupedMLP._flashinfer_forward, module)
    return module


def _forward(module, x, ids, probs, monkeypatch, *, rsv=None, bounded=None):
    """``InferenceGroupedMLP.forward`` in inference mode; ``rsv``: the NVLS dispatcher's buffer;
    ``bounded``: the step's bounded-row flag and the configured capacity."""
    from megatron.core.inference.utils import InferenceMode
    from megatron.core.transformer.moe.experts import InferenceGroupedMLP
    from megatron.core.transformer.moe.token_dispatcher_inference import NVLSAllGatherVDispatcher

    monkeypatch.setattr(InferenceMode, "_is_active", True)
    if rsv is not None:
        monkeypatch.setattr(
            NVLSAllGatherVDispatcher, "_get_rsv_tensor", staticmethod(lambda: rsv.tensor)
        )
    if bounded is not None:
        monkeypatch.setattr(InferenceMode, "_use_bounded_mxfp8_rows", True)
        module._flashinfer_mxfp8_token_capacity = bounded
    before = [t.clone() for t in (x, ids, probs)]
    out, bias = InferenceGroupedMLP.forward(module, x, None, probs, routing_map=ids)
    assert bias is None
    for name, t, b in zip(("hidden", "routing", "probs"), (x, ids, probs), before):
        ou.assert_exact(t, b, what=f"{name} is read-only")
    return out


def _rsv(rows, hidden):
    return ou.guarded_empty((rows, hidden), torch.float32)


def _check_rsv(out, rsv, rows):
    """``out`` is the NVLS buffer; its first ``rows`` rows hold a BF16 result, the rest stay
    poisoned. Returns the BF16 head."""
    assert out is rsv.tensor, "the NVLS path must write into the dispatcher's buffer"
    ou.check_guards(rsv=rsv)
    head = out[:rows]
    ou.assert_exact(head, head.bfloat16().float(), what="the buffer holds the BF16 kernel output")
    if rows < out.shape[0]:
        ou.assert_poisoned(out[rows:], what="rows past the bound")
    return head.bfloat16()


# ---------------------------------------------------------------------------------------------
# Activation mapping
# ---------------------------------------------------------------------------------------------

ACTIVATIONS = [
    pytest.param("silu", True, None, "Swiglu", id="silu-gated"),
    pytest.param("silu", False, None, "Silu", id="silu"),
    pytest.param("gelu", False, None, "Gelu", id="gelu"),
    pytest.param("relu", False, None, "Relu", id="relu"),
    pytest.param("squared_relu", False, None, "Relu2", id="squared_relu"),
    pytest.param("squared_relu", False, 2.0, "ClampedRelu2", id="squared_relu-clamped"),
]


def _func(name):
    return {"silu": F.silu, "gelu": F.gelu, "relu": F.relu}.get(name) or _squared_relu()


@pytest.mark.parametrize("func,gated,clamp,expected", ACTIVATIONS)
def test_activation_type_follows_the_config(func, gated, clamp, expected):
    from megatron.core.transformer.moe.experts import InferenceGroupedMLP

    activation_type = _activation_type()
    module = SimpleNamespace(config=_config(_func(func), gated=gated, clamp=clamp))
    got = InferenceGroupedMLP._resolve_flashinfer_activation_type(module)
    assert got.name == expected and got == getattr(activation_type, expected), (got, expected)


def test_unsupported_activation_raises():
    from megatron.core.transformer.moe.experts import InferenceGroupedMLP

    _activation_type()
    module = SimpleNamespace(config=_config(torch.tanh))
    with pytest.raises(ValueError, match="No FlashInfer ActivationType mapping"):
        InferenceGroupedMLP._resolve_flashinfer_activation_type(module)


@pytest.mark.parametrize("missing", ["enum", "limit-argument"])
def test_clamped_relu2_needs_a_flashinfer_that_has_it(missing, monkeypatch):
    import enum

    from megatron.core.transformer.moe import experts

    activation_type = _activation_type()
    if missing == "enum":
        members = {m.name: m.value for m in activation_type if m.name != "ClampedRelu2"}
        monkeypatch.setattr(experts, "ActivationType", enum.IntEnum("ActivationType", members))
    else:
        monkeypatch.setattr(
            experts, "fused_moe", SimpleNamespace(cutlass_fused_moe=lambda *a, **k: None)
        )
    module = SimpleNamespace(config=_config(_squared_relu(), clamp=2.0))
    with pytest.raises(RuntimeError, match="ClampedRelu2"):
        experts.InferenceGroupedMLP._resolve_flashinfer_activation_type(module)


# ---------------------------------------------------------------------------------------------
# Guards (no kernel runs)
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    ["mixed-mxfp8-layouts", "mxfp8-not-flashinfer", "clamp-without-clamped-relu2", "bf16-probs"],
)
def test_flashinfer_forward_guards(bad):
    activation_type = _activation_type()
    from megatron.core.inference.moe.flashinfer_mxfp8 import (
        FlashInferCutlassMXFP8Weight,
        FlashInferRoutedMXFP8Weight,
    )
    from megatron.core.transformer.moe.experts import InferenceGroupedMLP

    t = torch.zeros(1)
    cutlass_w = FlashInferCutlassMXFP8Weight(t, t, t, 128, 128)
    routed_w = FlashInferRoutedMXFP8Weight(t, t, 128, 128, 128, 128)
    module = SimpleNamespace(
        _uses_mxfp8_weights=bad.startswith("mxfp8") or bad.startswith("mixed"),
        _fc1_weight={"mixed-mxfp8-layouts": cutlass_w, "mxfp8-not-flashinfer": t}.get(
            bad, routed_w
        ),
        _fc2_weight=routed_w,
        _nvfp4_weights=None,
        _flashinfer_activation_type=activation_type.Relu2,
        _activation_clamp_scale=1.0 if bad == "clamp-without-clamped-relu2" else None,
        _flashinfer_clamp_limit=None,
        _nvls_dispatcher=False,
        _flashinfer_mxfp8_token_capacity=None,
        num_local_experts=2,
        ep_group=SimpleNamespace(size=lambda: 1, rank=lambda: 0),
    )
    x = torch.zeros(4, 128, dtype=torch.bfloat16)
    ids = torch.zeros(4, 2, dtype=torch.int64)
    probs = torch.zeros(4, 2, dtype=torch.bfloat16 if bad == "bf16-probs" else torch.float32)
    error = TypeError if "mxfp8" in bad else AssertionError
    with pytest.raises(error):
        InferenceGroupedMLP._flashinfer_forward(module, x, ids, probs)


@pytest.mark.parametrize("bad", ["no-routing-map", "training"])
def test_forward_requires_routing_map_and_eval_mode(bad, monkeypatch):
    _activation_type()
    from megatron.core.inference.moe import InferenceGroupedGemmBackend
    from megatron.core.inference.utils import InferenceMode
    from megatron.core.transformer.moe.experts import InferenceGroupedMLP

    monkeypatch.setattr(InferenceMode, "_is_active", True)
    calls = []
    module = SimpleNamespace(
        training=bad == "training",
        inference_grouped_gemm_backend=InferenceGroupedGemmBackend.FLASHINFER,
        _concatenated_weights_built=True,
        _flashinfer_forward=lambda *args: calls.append(args),
    )
    x, probs = torch.zeros(4, 128), torch.zeros(4, 2)
    ids = None if bad == "no-routing-map" else torch.zeros(4, 2, dtype=torch.int64)
    with pytest.raises(AssertionError):
        InferenceGroupedMLP.forward(module, x, None, probs, routing_map=ids)
    assert not calls


# ---------------------------------------------------------------------------------------------
# Real kernels behind forward()
# ---------------------------------------------------------------------------------------------


def _bf16_problem(clamp):
    return cutlass.Bf16Problem(
        tokens=129,
        hidden=256,
        inter=384,
        local=4,
        ep_size=2,
        ep_rank=1,
        top_k=2,
        pattern="random",
        rank1=clamp is None,
    )


def _run_bf16(clamp, nvls, monkeypatch):
    cutlass._cutlass_module_ready()
    problem = _bf16_problem(clamp)
    config = _config(_squared_relu(), clamp=clamp)
    module = _module(config, fc1=problem.w1, fc2=problem.w2, local=4, ep=(2, 1), nvls=nvls)
    rsv = _rsv(129, 256) if nvls else None
    out = _forward(module, problem.x, problem.ids, problem.probs, monkeypatch, rsv=rsv)
    if nvls:  # the BF16 kernel's own output; the dispatcher copies it into its buffer
        assert out.dtype == torch.bfloat16 and out.data_ptr() != rsv.tensor.data_ptr()
        ou.check_guards(rsv=rsv)
        ou.assert_poisoned(rsv.tensor, what="the NVLS buffer (the BF16 path must not write it)")
    activation = cutlass.BF16_RELU2 if clamp is None else spec.clamped_relu2(clamp)
    problem.check(out, activation, what=f"forward BF16 (clamp {clamp}, nvls {nvls})")


@pytest.mark.launch_on_gb200
@ou.requires_sm(ou.HOPPER, ou.BLACKWELL)
@pytest.mark.parametrize("nvls", [False, True], ids=["plain", "nvls"])
def test_bf16_relu2_from_the_config(nvls, monkeypatch):
    """``squared_relu`` resolves to ``Relu2`` and BF16 weights reach ``cutlass_fused_moe``."""
    _run_bf16(None, nvls, monkeypatch)


def _run_cutlass_mxfp8(nvls, monkeypatch):
    cutlass._cutlass_module_ready()
    g = ou.make_generator()
    local, ep, tokens = 4, (2, 1), 65
    w1, w2, x, limit = cutlass._mxfp8_clamped_inputs(local, 256, 384, tokens, g)
    ids, probs = spec.make_routing(tokens, 2, local * ep[0], generator=g, device_="cuda")
    config = _config(_squared_relu(), clamp=limit)
    module = _module(
        config,
        fc1=cutlass._cutlass_mxfp8(w1),
        fc2=cutlass._cutlass_mxfp8(w2),
        mxfp8=True,
        local=local,
        ep=ep,
        nvls=nvls,
    )
    xb = x.to(torch.bfloat16).cuda()
    rsv = _rsv(tokens, 256) if nvls else None
    out = _forward(module, xb, ids, probs, monkeypatch, rsv=rsv)
    if nvls:
        out = _check_rsv(out, rsv, tokens)
    act, offset = spec.clamped_relu2(limit), ep[1] * local
    args = (xb, w1.cuda(), w2.cuda(), ids, probs)
    ref = ou.fp64_reference(spec.moe_reference, *args, activation=act, local_offset=offset)
    with spec.exact_fp32_matmul():
        trusted = spec.moe_reference(
            xb.float(),
            w1.float().cuda(),
            w2.float().cuda(),
            ids,
            probs,
            activation=act,
            local_offset=offset,
            out_dtype=torch.bfloat16,
        )
    assert out.shape == ref.shape and out.dtype == torch.bfloat16
    ou.assert_close_to_reference(out, ref, trusted, dim=-1, what="forward MXFP8 CUTLASS")


def _run_routed(variant, monkeypatch, *, shape=(256, 256), experts=8, local=2, rank=2, top_k=2):
    spec.require_flashinfer()
    problem = routed.Problem(
        tokens=129 if shape == (256, 256) else 256,
        hidden=shape[0],
        inter=shape[1],
        experts=experts,
        local=local,
        offset=rank * local,
        top_k=top_k,
        pattern="random",
    )
    fc1, fc2 = problem.weights()
    tokens = problem.x.shape[0]
    module = _module(
        _config(_squared_relu()),
        fc1=fc1,
        fc2=fc2,
        mxfp8=True,
        local=local,
        ep=(experts // local, rank),
        nvls=variant != "plain",
    )
    rsv = _rsv(tokens, shape[0]) if variant != "plain" else None
    bounded = 64 if variant == "nvls-bounded" else None
    out = _forward(
        module, problem.x, problem.ids, problem.probs, monkeypatch, rsv=rsv, bounded=bounded
    )
    rows = bounded or tokens
    if rsv is not None:
        out = _check_rsv(out, rsv, rows)
    routed._check(problem, out, what=f"forward routed MXFP8 ({variant})", rows=rows)


def _run_nvfp4(gated, variant, monkeypatch):
    spec.require_flashinfer()
    spec.clear_nvfp4_recipe_env(monkeypatch)
    problem = nvfp4.Problem(
        tokens=129, hidden=256, inter=256, experts=8, local=2, offset=4, top_k=2, gated=gated
    )
    config = _config(F.silu, gated=True) if gated else _config(_squared_relu())
    module = _module(
        config, nvfp4_weights=problem.weights(), local=2, ep=(4, 2), nvls=variant != "plain"
    )
    rsv = _rsv(129, 256) if variant != "plain" else None
    bounded = 64 if variant == "nvls-bounded" else None
    out = _forward(
        module, problem.x, problem.ids, problem.probs, monkeypatch, rsv=rsv, bounded=bounded
    )
    rows = bounded or 129
    if rsv is not None:
        out = _check_rsv(out, rsv, rows)
    nvfp4._check(problem, out, what=f"forward NVFP4 (gated {gated}, {variant})", rows=rows)


BLACKWELL_PATHS = [
    ("bf16-clamped-relu2", "plain"),
    ("bf16-clamped-relu2", "nvls"),
    ("mxfp8-cutlass-clamped-relu2", "plain"),
    ("mxfp8-cutlass-clamped-relu2", "nvls"),
    ("mxfp8-routed-relu2", "plain"),
    ("mxfp8-routed-relu2", "nvls"),
    ("mxfp8-routed-relu2", "nvls-bounded"),
    ("nvfp4-relu2", "plain"),
    ("nvfp4-relu2", "nvls"),
    ("nvfp4-relu2", "nvls-bounded"),
    ("nvfp4-swiglu", "plain"),
    ("nvfp4-swiglu", "nvls-bounded"),
]


@pytest.mark.launch_on_gb200
@ou.requires_sm(ou.BLACKWELL)
@pytest.mark.parametrize(
    "path,variant", [pytest.param(*p, id=f"{p[0]}-{p[1]}") for p in BLACKWELL_PATHS]
)
def test_blackwell_paths_from_the_config(path, variant, monkeypatch):
    """Each weight format reaches its kernel with the activation the config resolves to, the EP
    arithmetic, and (NVLS) the dispatcher's buffer and the step's row bound."""
    if path == "bf16-clamped-relu2":
        _run_bf16(8.0, variant == "nvls", monkeypatch)
    elif path == "mxfp8-cutlass-clamped-relu2":
        _run_cutlass_mxfp8(variant == "nvls", monkeypatch)
    elif path == "mxfp8-routed-relu2":
        _run_routed(variant, monkeypatch)
    else:
        _run_nvfp4(path == "nvfp4-swiglu", variant, monkeypatch)


@pytest.mark.launch_on_gb200
@ou.requires_sm(ou.BLACKWELL)
def test_routed_mxfp8_production_ep16(monkeypatch):
    """H 2688, I 1856, 128 experts with 8 local at EP rank 8 of 16, top-6, 256 tokens, NVLS."""
    _run_routed("nvls", monkeypatch, shape=(2688, 1856), experts=128, local=8, rank=8, top_k=6)


@pytest.mark.launch_on_gb200
@ou.requires_sm(ou.BLACKWELL)
@pytest.mark.sanitizer
def test_sanitizer_nvfp4_nvls_bounded_small_case(monkeypatch):
    """The compute-sanitizer case: per-token NVFP4 through ``forward`` into a guarded NVLS
    buffer with bounded rows."""
    _run_nvfp4(False, "nvls-bounded", monkeypatch)
