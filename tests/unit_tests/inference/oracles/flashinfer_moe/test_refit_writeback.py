# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Oracle for refitting FlashInfer expert weights: the routed-MXFP8 write-through of main #8560
(`dda809cad5`) and the NVFP4 requantization on refit of main #7798 (`ebcf9d6722`).

Call sites (`megatron/core/inference/moe/flashinfer_mxfp8.py`, `flashinfer_nvfp4.py`,
`megatron/core/transformer/moe/experts.py`, `megatron/core/resharding/`):
- Routed MXFP8: `InferenceGroupedMLP._build_concatenated_mxfp8_weights` stacks the per-expert
  canonical `MXFP8Tensor`s, repacks them into the TRT-LLM Major-K stacks
  (`prepare_routed_mxfp8_weights`) and calls `attach_routed_mxfp8_writeback` per expert: the
  canonical storage is released to one-element broadcast placeholders, and the expert's
  `quantize_` / `copy_` (what refit's `MXFP8ReshardTransform.finalize_recv` calls) requantizes
  the new value with the canonical Triton quantizer and repacks it into that expert's slot of the
  routed stack, in place. `refresh_flashinfer_mxfp8_weights` then only reports True (refit
  synchronizes before graphs replay).
- CUTLASS MXFP8 (ClampedRelu2): the CUTLASS weights are views of the canonical stack (the int32
  scale view groups the canonical scale bytes); the canonical wrappers are redirected into that
  stack, so their in-place `quantize_` is the refit.
- Per-token NVFP4: refit writes the BF16 expert stacks (the TE parameters alias them);
  `refresh_module_caches` (every refit path's last step) calls `InferenceGroupedMLP.refresh_cache`,
  which requantizes the NVFP4 stacks in place (`prepare_routed_nvfp4_weights(..., out=)`) and
  synchronizes.

Contract (written from #8560's and #7798's descriptions, the refit transform and the formats):
- After a refit with new values W' (BF16, or FP32 through `copy_`), every buffer the inference
  forward reads holds exactly what a fresh build from W' would hold, as the format specifications
  define it, independently of Megatron's quantizers and repack code:
  - routed MXFP8: `routed.data[e]` = the OCP MX (RCEIL) E4M3 codes of W'[e], zero-padded to
    128-aligned rows and columns and Major-K row-shuffled; `routed.scale[e]` = the E8M0 scales,
    zero-padded to `[round_up(M, 128), round_up(K, 128) / 32]`, row-shuffled the same way, in the
    128x4 layout;
  - CUTLASS MXFP8: the codes of W'[e] row-major and the scales in the 128x4 layout, seen through
    both the canonical wrapper and the CUTLASS views;
  - NVFP4: codes, block scales, alpha and the MMA view of the NVFP4 spec quantization of W'
    (`test_nvfp4_quantize_expert_weights.check_quantized_stack`).
- Refit changes no address the forward reads (CUDA graphs captured before it replay the new
  weights), touches only the refit experts, and never reads the released canonical storage.
- The documented placeholder contract holds: after attach the canonical data and scale storage is
  one element; a second attach raises.
- A CUDA graph of the forward captured before the refit, replayed after it, computes the MoE of
  W' (fp64 reference of the MoE math), bitwise the eager forward.

Reference: `spec.mxfp8_quantize_spec`, `spec.swizzle_128x4`, `spec.trtllm_shuffle_source_rows`,
`spec.nvfp4_quantize_spec` (via `check_quantized_stack`) on the new values in fp64; for the MoE
output, `spec.moe_reference` in fp64 on structured, lossless weights (the routed-MXFP8 and NVFP4
`Problem`s of their modules). Tolerance class: bytes bit-exact; MoE output per row within twice
the trusted fp32 implementation (`_check` of those modules); graph replay bitwise. Memory: the
released placeholders are poisoned (NaN E4M3 byte 0x7F, NaN E8M0 byte 0xFF) before the refit and
the forward, and the refit buffers (NVFP4) are poisoned before the requantization.

Structure (read after the contract): the write-through quantizes the full expert
(`MXFP8Tensor.from_bf16(value, backend="triton")`) and repacks it with the same per-expert loop
as the build (`prepare_routed_mxfp8_weights(stacked, out=slot)`, `slot` a one-expert view of the
routed stack); the transform delivers a 1-D-scale weight to `quantize_` once all its slices have
arrived (direct receive for contiguous slices, a staging copy for strided ones). Covered: every
refit entry (`quantize_` BF16, `copy_` FP32, the transform with the full weight, row halves and
column halves), FC1 and FC2, 1 and 3 experts with an all-zero expert, one expert refit alone,
logical sizes needing padding (160 x 96) and production (2688 x 1856; CUTLASS 7168 x 2048), the
CUTLASS aliasing, graph replay after refit for routed MXFP8 and NVFP4 (Relu2 and SwiGLU), the
NVFP4 requantization through `refresh_module_caches` (shell module) and through a real
`InferenceGroupedMLP` built by the inference MoE spec (lazy build in the first forward, refit
through the TE parameters).

Main vs open PRs: every case is expected to pass on main.

Hardware: the MXFP8 byte checks run on any GPU with FlashInfer (H100 and GB200: the Triton
quantizer and tensor indexing); the MoE forwards and NVFP4 quantization need Blackwell.
"""

import types

import pytest
import torch
import torch.nn.functional as F

from tests.unit_tests.inference.oracles import oracle_utils as ou
from tests.unit_tests.inference.oracles.flashinfer_moe import spec
from tests.unit_tests.inference.oracles.flashinfer_moe import test_cute_dsl_moe_wrapper as nvfp4_moe
from tests.unit_tests.inference.oracles.flashinfer_moe import (
    test_nvfp4_quantize_expert_weights as nvfp4_weights,
)
from tests.unit_tests.inference.oracles.flashinfer_moe import (
    test_trtllm_fp8_block_scale_routed_moe as mxfp8_moe,
)

ROUTED_ALIGN = 128
E4M3_NAN, E8M0_NAN = 0x7F, 0xFF
DEVICE = "cuda"
GROUPS = (("linear_fc1", "_fc1_weight"), ("linear_fc2", "_fc2_weight"))


# ---------------------------------------------------------------------------------------------
# A host for InferenceGroupedMLP's MXFP8 build / refresh / forward (no TE grouped linears)
# ---------------------------------------------------------------------------------------------


def _experts_cls():
    from megatron.core.transformer.moe.experts import InferenceGroupedMLP

    class Experts:
        """The production methods on a host holding per-expert canonical MXFP8 weights."""

        _stack_mxfp8_linear_weight = InferenceGroupedMLP._stack_mxfp8_linear_weight
        _build_concatenated_mxfp8_weights = InferenceGroupedMLP._build_concatenated_mxfp8_weights
        refresh_flashinfer_mxfp8_weights = InferenceGroupedMLP.refresh_flashinfer_mxfp8_weights
        _flashinfer_forward = InferenceGroupedMLP._flashinfer_forward

    return Experts


def build_experts(canonical, *, clamp=None, local_offset=0, num_experts=None):
    """`canonical[linear_name]` = per-expert MXFP8Tensors. Builds the FlashInfer stacks with the
    production `_build_concatenated_mxfp8_weights` (routed, or CUTLASS when `clamp` is set)."""
    from flashinfer.fused_moe import ActivationType

    from megatron.core.inference.moe import InferenceGroupedGemmBackend

    host = _experts_cls()()
    local = len(canonical["linear_fc1"])
    host.num_local_experts = local
    host.inference_grouped_gemm_backend = InferenceGroupedGemmBackend.FLASHINFER
    host._activation_clamp_scale = clamp
    host._concatenated_weights_built = False
    for linear_name, weights in canonical.items():
        setattr(host, linear_name, types.SimpleNamespace())
        for i, w in enumerate(weights):
            setattr(getattr(host, linear_name), f"weight{i}", w)
    host._build_concatenated_mxfp8_weights()
    host._concatenated_weights_built = True
    ep = (num_experts or local) // local
    host._nvfp4_weights = None
    host._flashinfer_activation_type = ActivationType.Relu2
    host._flashinfer_clamp_limit = None
    host._nvls_dispatcher = False
    host._flashinfer_mxfp8_token_capacity = None
    host.ep_group = types.SimpleNamespace(size=lambda: ep, rank=lambda: local_offset // local)
    host.config = types.SimpleNamespace(deterministic_mode=False, batch_invariant_mode=False)
    return host


def canonical_experts(values):
    """Per-expert canonical MXFP8Tensors (Triton / cuBLAS layout) holding the MX-exact `values`
    [E, R, C] exactly, built from spec bytes (not by Megatron's quantizer), on CUDA."""
    from megatron.core.inference.quantization.mxfp8_tensor import MXFP8Tensor

    stack = spec.canonical_mxfp8(values)
    return [
        MXFP8Tensor(
            data=stack.data[e].clone().to(DEVICE),
            scale=stack.scale[e].clone().to(DEVICE),
            backend="triton",
            dtype=torch.bfloat16,
        )
        for e in range(values.shape[0])
    ]


def poison_placeholders(weights):
    """Fill each released canonical placeholder (one element) with NaN bytes."""
    for w in weights:
        for t, byte in ((w.data, E4M3_NAN), (w.scale, E8M0_NAN)):
            assert t.untyped_storage().nbytes() == 1, "the canonical storage was not released"
            t.as_strided((1,), (1,)).view(torch.uint8).fill_(byte)


def assert_placeholders_poisoned(weights, what):
    for w in weights:
        for t, byte in ((w.data, E4M3_NAN), (w.scale, E8M0_NAN)):
            assert t.untyped_storage().nbytes() == 1, f"{what}: canonical storage reallocated"
            got = int(t.as_strided((1,), (1,)).view(torch.uint8).item())
            assert got == byte, f"{what}: a write reached the released canonical storage"


# ---------------------------------------------------------------------------------------------
# Spec layouts of new values
# ---------------------------------------------------------------------------------------------


def spec_routed(value):
    """Expected (data bytes, scale bytes) of one routed expert slot for `value` [R, C]."""
    rows, cols = value.shape
    codes, scales = spec.mxfp8_quantize_spec(value.double().cpu())
    pr, pc = spec._ceil_to(rows, ROUTED_ALIGN), spec._ceil_to(cols, ROUTED_ALIGN)
    src = spec.trtllm_shuffle_source_rows(pr)
    data = torch.zeros(pr, pc, dtype=torch.uint8)
    data[:rows, :cols] = codes
    scale = torch.zeros(pr, pc // spec.MX_BLOCK, dtype=torch.uint8)
    scale[:rows, : cols // spec.MX_BLOCK] = scales
    return data[src], spec.swizzle_128x4(scale[src])


def spec_canonical(value):
    """Expected (codes [R, C], 128x4 scale bytes) of a canonical MXFP8 tensor for `value`."""
    codes, scales = spec.mxfp8_quantize_spec(value.double().cpu())
    return codes, spec.swizzle_128x4(scales)


def check_routed(host, values, what):
    """The routed stacks hold `values[linear][e]`: first the logical codes and scales (undoing the
    shuffle and the 128x4 layout, reported apart), then every byte including the padding."""
    for linear_name, buf_name in GROUPS:
        routed = getattr(host, buf_name)
        for e, value in enumerate(values[linear_name]):
            rows, cols = value.shape
            codes, scales = spec.mxfp8_quantize_spec(value.double().cpu())
            data, scale = spec_routed(value)
            pr, pc = data.shape
            inverse = torch.argsort(spec.trtllm_shuffle_source_rows(pr))
            got_data = spec.as_bytes(routed.data[e]).cpu()
            got_scale = routed.scale[e].cpu()
            label = f"{what}: {linear_name} expert {e} ({rows}x{cols})"
            ou.assert_quantized_bit_exact(
                got_data[inverse][:rows, :cols],
                spec.unswizzle_128x4(got_scale, pr, pc // spec.MX_BLOCK)[inverse][
                    :rows, : cols // spec.MX_BLOCK
                ],
                codes,
                scales,
                elems_per_scale=spec.MX_BLOCK,
                what=label,
            )
            ou.assert_exact(got_data, data, what=f"{label} data bytes")
            ou.assert_exact(got_scale, scale, what=f"{label} scale bytes")


def check_cutlass(host, canonical, values, what):
    for linear_name, buf_name in GROUPS:
        cutlass = getattr(host, buf_name)
        for e, value in enumerate(values[linear_name]):
            codes, scale = spec_canonical(value)
            rows, cols = value.shape
            ou.assert_exact(spec.as_bytes(cutlass.data[e]).cpu(), codes, what=f"{what}: data")
            packed = cutlass.scale[e].contiguous().view(torch.uint8).reshape(-1).cpu()
            ou.assert_exact(packed, scale, what=f"{what}: {linear_name} expert {e} int32 scales")
            w = canonical[linear_name][e]
            ou.assert_exact(spec.as_bytes(w.data).cpu(), codes, what=f"{what}: canonical data")
            ou.assert_exact(w.scale.view(torch.uint8).cpu(), scale, what=f"{what}: canonical scale")
            assert w.data.data_ptr() == cutlass.data[e].data_ptr(), "canonical left the stack"


# ---------------------------------------------------------------------------------------------
# Refit entry points
# ---------------------------------------------------------------------------------------------


def refit(canonical, new_values, path, experts=None):
    """Write `new_values[linear][e]` into the canonical weights through one refit entry point."""
    from megatron.core.resharding.transforms import MXFP8ReshardTransform

    buffers = {
        f"{linear_name}.weight{e}": w
        for linear_name, ws in canonical.items()
        for e, w in enumerate(ws)
    }
    transform = MXFP8ReshardTransform(
        convertible_params=set(buffers), persistent_buffers=buffers, backend="triton"
    )
    for linear_name, values in new_values.items():
        for e, value in enumerate(values):
            if experts is not None and e not in experts:
                continue
            name, w = f"{linear_name}.weight{e}", canonical[linear_name][e]
            if path == "quantize_":
                assert w.quantize_(value) is w
            elif path == "copy_fp32":
                assert w.copy_(value.float()) is w
            else:
                rows, cols = value.shape
                if path == "transform":
                    slices = [(slice(None), slice(None))]
                elif path == "transform_rows":
                    slices = [
                        (slice(0, rows // 2), slice(None)),
                        (slice(rows // 2, rows), slice(None)),
                    ]
                else:  # transform_cols: strided receives take the staging copy
                    half = cols // 64 * 32
                    slices = [(slice(None), slice(0, half)), (slice(None), slice(half, cols))]
                for sl in slices:
                    bufs = transform.prepare_recv(name, sl)
                    assert len(bufs) == 1 and bufs[0].dtype == torch.bfloat16
                    bufs[0].copy_(value[sl])
                    transform.finalize_recv(name, sl, bufs)
    assert not transform._pending_1d, "a refit weight was left unquantized"


def random_values(experts, rows, cols, generator, *, zero_expert=None, dtype=torch.bfloat16):
    """Random weights with block magnitudes over 2**-12 .. 2**4 (BF16, or FP32 values that
    BF16 cannot hold), CUDA."""
    out = []
    for e in range(experts):
        if e == zero_expert:
            out.append(torch.zeros(rows, cols, dtype=dtype, device=DEVICE))
            continue
        w = torch.randn(rows, cols // spec.MX_BLOCK, spec.MX_BLOCK, generator=generator)
        exps = torch.randint(-12, 5, (rows, cols // spec.MX_BLOCK, 1), generator=generator)
        out.append((w * torch.exp2(exps.float())).reshape(rows, cols).to(dtype).to(DEVICE))
    return out


# ---------------------------------------------------------------------------------------------
# Routed MXFP8 write-through: bytes against the spec
# ---------------------------------------------------------------------------------------------

SHAPES = [(160, 96), (256, 256), (512, 384)]  # logical (H, I): FC1 [I, H], FC2 [H, I]
PATHS = ["quantize_", "copy_fp32", "transform", "transform_rows", "transform_cols"]
ROUTED_CASES = ou.pairwise_cases(
    {"shape": SHAPES, "experts": [1, 3], "path": PATHS, "zero_expert": [False, True]}
)


def _routed_id(c):
    h, i = c["shape"]
    return f"H{h}-I{i}-E{c['experts']}-{c['path']}{'-zero' if c['zero_expert'] else ''}"


def _initial(experts, hidden, inter, generator):
    """Initial MX-exact weights (spec bytes) for FC1 [I, H] and FC2 [H, I]."""
    fc1 = spec.mx_exact_values(experts * inter, hidden, generator=generator)
    fc2 = spec.mx_exact_values(experts * hidden, inter, generator=generator)
    return {
        "linear_fc1": canonical_experts(fc1.reshape(experts, inter, hidden)),
        "linear_fc2": canonical_experts(fc2.reshape(experts, hidden, inter)),
    }


def _new_values(experts, hidden, inter, generator, *, zero_expert=None, dtype=torch.bfloat16):
    return {
        "linear_fc1": random_values(
            experts, inter, hidden, generator, zero_expert=zero_expert, dtype=dtype
        ),
        "linear_fc2": random_values(
            experts, hidden, inter, generator, zero_expert=zero_expert, dtype=dtype
        ),
    }


def _routed_refit_case(hidden, inter, experts, path, zero_expert, seed):
    spec.require_flashinfer()
    g = ou.make_generator(ou.DEFAULT_SEED + seed)
    canonical = _initial(experts, hidden, inter, g)
    host = build_experts(canonical)
    pointers = {
        b: (getattr(host, b).data.data_ptr(), getattr(host, b).scale.data_ptr()) for _, b in GROUPS
    }
    for ws in canonical.values():
        poison_placeholders(ws)
    dtype = torch.float32 if path == "copy_fp32" else torch.bfloat16
    new = _new_values(
        experts, hidden, inter, g, zero_expert=(experts - 1) if zero_expert else None, dtype=dtype
    )
    refit(canonical, new, path)
    check_routed(host, new, what=f"after {path} refit")
    for _, b in GROUPS:
        routed = getattr(host, b)
        assert (routed.data.data_ptr(), routed.scale.data_ptr()) == pointers[b], "refit moved"
    for linear_name, ws in canonical.items():
        assert_placeholders_poisoned(ws, what=f"{linear_name} after refit")
    assert host.refresh_flashinfer_mxfp8_weights() is True, "refit must synchronize"
    return host, canonical, new


@pytest.mark.launch_on_gb200
@ou.requires_sm(ou.HOPPER, ou.BLACKWELL)
@pytest.mark.parametrize("case", [pytest.param(c, id=_routed_id(c)) for c in ROUTED_CASES])
def test_routed_writeback_matches_spec(case):
    """Every refit entry point rewrites the routed stacks to the spec layout of the spec
    quantization of the new values, in place, without touching the released canonical storage."""
    hidden, inter = case["shape"]
    _routed_refit_case(
        hidden, inter, case["experts"], case["path"], case["zero_expert"], seed=hidden + inter
    )


@pytest.mark.launch_on_gb200
@ou.requires_sm(ou.HOPPER, ou.BLACKWELL)
def test_routed_writeback_production_shape():
    """Nemotron-3 Nano's expert shape (H 2688, I 1856: I padded to 1920), 2 experts, through the
    refit transform."""
    _routed_refit_case(2688, 1856, 2, "transform", False, seed=7)


@pytest.mark.launch_on_gb200
@ou.requires_sm(ou.HOPPER, ou.BLACKWELL)
@pytest.mark.sanitizer
def test_one_expert_refit_leaves_the_others():
    """Refitting expert 1 of 3 rewrites only its slots; experts 0 and 2 keep their bytes. Also
    the placeholder contract: the canonical storage is one element, a second attach raises."""
    spec.require_flashinfer()
    from megatron.core.inference.moe.flashinfer_mxfp8 import attach_routed_mxfp8_writeback

    g = ou.make_generator(ou.DEFAULT_SEED + 3)
    hidden, inter, experts = 160, 96, 3
    fc1 = spec.mx_exact_values(experts * inter, hidden, generator=g).reshape(experts, inter, hidden)
    fc2 = spec.mx_exact_values(experts * hidden, inter, generator=g).reshape(experts, hidden, inter)
    canonical = {"linear_fc1": canonical_experts(fc1), "linear_fc2": canonical_experts(fc2)}
    host = build_experts(canonical)
    for ws in canonical.values():
        poison_placeholders(ws)
    with pytest.raises(RuntimeError, match="already writes through"):
        attach_routed_mxfp8_writeback(canonical["linear_fc1"][0], host._fc1_weight, 0)
    new = _new_values(experts, hidden, inter, g)
    refit(canonical, new, "transform", experts={1})
    expected = {
        "linear_fc1": [fc1[0].to(DEVICE), new["linear_fc1"][1], fc1[2].to(DEVICE)],
        "linear_fc2": [fc2[0].to(DEVICE), new["linear_fc2"][1], fc2[2].to(DEVICE)],
    }
    check_routed(host, expected, what="expert 1 refit alone")


# ---------------------------------------------------------------------------------------------
# CUTLASS MXFP8 (ClampedRelu2): the CUTLASS views alias the canonical stack
# ---------------------------------------------------------------------------------------------


@pytest.mark.launch_on_gb200
@ou.requires_sm(ou.HOPPER, ou.BLACKWELL)
@pytest.mark.parametrize(
    "shape,experts,path",
    [
        ((256, 256), 3, "quantize_"),
        ((512, 384), 1, "transform_rows"),
        ((256, 384), 3, "copy_fp32"),
        ((7168, 2048), 2, "transform"),
    ],
    ids=["H256-I256-E3-quantize_", "H512-I384-E1-rows", "H256-I384-E3-fp32", "H7168-I2048-E2"],
)
def test_cutlass_refit_reaches_the_cutlass_views(shape, experts, path):
    """With a clamp (CUTLASS MXFP8) the refit's in-place quantization is visible through the
    CUTLASS data and packed int32 scales, which stay at their addresses."""
    spec.require_flashinfer()
    hidden, inter = shape
    g = ou.make_generator(ou.DEFAULT_SEED + hidden)
    canonical = _initial(experts, hidden, inter, g)
    host = build_experts(canonical, clamp=16.0)
    pointers = {
        b: (getattr(host, b).data.data_ptr(), getattr(host, b).scale.data_ptr()) for _, b in GROUPS
    }
    dtype = torch.float32 if path == "copy_fp32" else torch.bfloat16
    new = _new_values(
        experts, hidden, inter, g, zero_expert=0 if experts > 1 else None, dtype=dtype
    )
    refit(canonical, new, path)
    check_cutlass(host, canonical, new, what=f"CUTLASS after {path} refit")
    for _, b in GROUPS:
        cutlass = getattr(host, b)
        assert (cutlass.data.data_ptr(), cutlass.scale.data_ptr()) == pointers[b], "refit moved"
    assert host.refresh_flashinfer_mxfp8_weights() is False, "CUTLASS shares the canonical bytes"


# ---------------------------------------------------------------------------------------------
# Routed MXFP8: a graph captured before the refit computes the new MoE
# ---------------------------------------------------------------------------------------------


@pytest.mark.launch_on_gb200
@ou.requires_sm(ou.BLACKWELL)
@pytest.mark.parametrize(
    "shape,path",
    [((256, 256), "transform"), ((160, 96), "quantize_")],
    ids=["H256-I256", "H160-I96"],
)
def test_routed_refit_reaches_graph_replay(shape, path):
    """`_flashinfer_forward` captured in a CUDA graph on the first weights, the canonical
    placeholders poisoned, the second weights refit through the write-through: the replay computes
    the second weights' fp64 MoE (third tokens), bitwise the eager forward."""
    spec.require_flashinfer()
    hidden, inter = shape
    kw = dict(tokens=128, hidden=hidden, inter=inter, experts=8, local=4, offset=4, top_k=2)
    first = mxfp8_moe.Problem(**kw, pattern="random", seed=0)
    second = mxfp8_moe.Problem(**kw, pattern="dropped-routes", seed=1, v0=first.v0)
    third = mxfp8_moe.Problem(**kw, pattern="empty-expert", seed=2, v0=first.v0)
    canonical = {
        "linear_fc1": canonical_experts(first.w1),
        "linear_fc2": canonical_experts(first.w2),
    }
    host = build_experts(canonical, local_offset=4, num_experts=8)
    x, ids, probs = first.x.clone(), first.ids.clone(), first.probs.clone()

    def forward():
        out, bias = host._flashinfer_forward(x, ids, probs)
        assert bias is None
        return out

    mxfp8_moe._check(first, forward(), what="eager before refit")
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_out = forward()
    for ws in canonical.values():
        poison_placeholders(ws)
    new = {
        "linear_fc1": [w.to(torch.bfloat16).to(DEVICE) for w in second.w1],
        "linear_fc2": [w.to(torch.bfloat16).to(DEVICE) for w in second.w2],
    }
    refit(canonical, new, path)
    assert host.refresh_flashinfer_mxfp8_weights() is True
    torch.cuda.synchronize()
    x.copy_(third.x)
    ids.copy_(third.ids)
    probs.copy_(third.probs)
    graph.replay()
    replayed = static_out.clone()
    mxfp8_moe._check(second, replayed, what="graph replay after refit", inputs=third)
    ou.assert_exact(replayed, forward(), what="replay vs eager after refit")
    for linear_name, ws in canonical.items():
        assert_placeholders_poisoned(ws, what=f"{linear_name} after the forward")
    del graph


# ---------------------------------------------------------------------------------------------
# NVFP4: requantization on refit
# ---------------------------------------------------------------------------------------------

NVFP4_FIELDS = (
    "fc1_data",
    "fc1_scale_swizzled",
    "fc1_scale",
    "fc1_alpha",
    "fc2_data",
    "fc2_scale_swizzled",
    "fc2_scale",
    "fc2_alpha",
    "fc2_input_scale",
)


def _poison_nvfp4(weights):
    pointers = {name: getattr(weights, name).data_ptr() for name in NVFP4_FIELDS}
    for name in NVFP4_FIELDS:
        t = getattr(weights, name)
        if name.endswith(("data", "swizzled")):
            t.fill_(0xFF)
        elif name.endswith("alpha"):
            t.fill_(float("nan"))
    return pointers


def _grouped_mlp_shell(fc1, fc2, nvfp4):
    """An InferenceGroupedMLP instance holding only the state its refresh reads."""
    from megatron.core.transformer.moe.experts import InferenceGroupedMLP

    shell = InferenceGroupedMLP.__new__(InferenceGroupedMLP)
    torch.nn.Module.__init__(shell)
    shell._fc1_weight, shell._fc2_weight, shell._nvfp4_weights = fc1, fc2, nvfp4
    return shell


NVFP4_CASES = ou.pairwise_cases(
    {
        "shape": [(256, 256), (255, 257), (1024, 768)],
        "gated": [False, True],
        "experts": [1, 3],
        "math": ["fast", "ieee"],
    }
)


def _nvfp4_id(c):
    h, i = c["shape"]
    return f"H{h}-I{i}-{'swiglu' if c['gated'] else 'relu2'}-E{c['experts']}-{c['math']}"


@pytest.mark.launch_on_gb200
@ou.requires_sm(ou.BLACKWELL)
@pytest.mark.parametrize("case", [pytest.param(c, id=_nvfp4_id(c)) for c in NVFP4_CASES])
def test_nvfp4_refit_requantizes_through_refresh_module_caches(case, monkeypatch):
    """Refit writes new BF16 stacks in place; `refresh_module_caches` (refit's last step) must
    requantize every NVFP4 buffer (poisoned first) to the spec quantization of the new weights, at
    the same addresses, and synchronize."""
    from megatron.core.resharding.execution import refresh_module_caches

    nvfp4_weights._setup(monkeypatch, case["math"])
    hidden, inter = case["shape"]
    g = ou.make_generator(ou.DEFAULT_SEED + hidden + inter)
    kw = dict(hidden=hidden, inter=inter, experts=case["experts"], gated=case["gated"])
    first = nvfp4_weights.Experts(**kw, generator=g)
    second = nvfp4_weights.Experts(
        **kw, generator=g, zero_expert=case["experts"] - 1 if case["experts"] > 1 else None
    )
    fc1, fc2 = first.w1.to(torch.bfloat16), first.w2.to(torch.bfloat16)
    weights = first.prepare()
    pointers = _poison_nvfp4(weights)
    fc1.copy_(second.w1)  # what refit's in-place receive does through the TE parameters
    fc2.copy_(second.w2)
    shell = _grouped_mlp_shell(fc1, fc2, weights)
    syncs = []
    real_sync = torch.cuda.synchronize

    def sync(*args, **kwargs):
        syncs.append(True)
        return real_sync(*args, **kwargs)

    monkeypatch.setattr(torch.cuda, "synchronize", sync)
    refresh_module_caches([torch.nn.Sequential(shell)])
    assert syncs, "the NVFP4 requantization must synchronize before graphs replay"
    assert shell._nvfp4_weights is weights
    assert {name: getattr(weights, name).data_ptr() for name in NVFP4_FIELDS} == pointers
    nvfp4_weights.check_quantized_stack(weights, second, what=f"refit {_nvfp4_id(case)}")


def _moe_layer(hidden, inter, experts, gated):
    from megatron.core.activations import squared_relu
    from megatron.core.models.gpt.moe_module_specs import get_inference_optimized_moe_spec
    from megatron.core.transformer.moe.token_dispatcher_inference import NCCLAllGatherDispatcher
    from megatron.core.transformer.transformer_config import TransformerConfig

    config = TransformerConfig(
        num_layers=1,
        hidden_size=hidden,
        num_attention_heads=4,
        num_moe_experts=experts,
        moe_ffn_hidden_size=inter,
        moe_router_topk=2,
        moe_router_score_function="sigmoid",
        moe_router_enable_expert_bias=True,
        moe_router_dtype="fp32",
        moe_grouped_gemm=True,
        moe_token_dispatcher_type="alltoall",
        activation_func=F.silu if gated else squared_relu,
        gated_linear_unit=gated,
        normalization="RMSNorm",
        add_bias_linear=False,
        bf16=True,
        params_dtype=torch.bfloat16,
        transformer_impl="inference_optimized",
        expert_tensor_parallel_size=1,
        expert_model_parallel_size=1,
        use_cpu_initialization=True,
        inference_grouped_gemm_backend="flashinfer",
        inference_moe_nvfp4_recipe="per_token",
        inference_moe_token_dispatcher_type="nccl",
    )
    NCCLAllGatherDispatcher.allocate_buffers()
    return get_inference_optimized_moe_spec()(config=config).cuda().eval()


@pytest.mark.launch_on_gb200
@ou.requires_sm(ou.BLACKWELL)
@pytest.mark.parametrize("gated", [False, True], ids=["relu2", "swiglu"])
def test_nvfp4_moe_layer_refit_reaches_graph_replay(gated, monkeypatch):
    """A real InferenceGroupedMLP from the inference MoE spec: its first forward builds the BF16
    stacks (aliasing the TE parameters) and the NVFP4 weights; a CUDA graph is captured; refit
    writes the second weights into the TE parameters and calls `refresh_module_caches`; the
    replay computes the second weights' fp64 MoE on the third tokens, bitwise the eager forward."""
    from megatron.core.inference.symmetric_memory import SymmetricMemoryManager
    from megatron.core.inference.utils import InferenceMode
    from megatron.core.resharding.execution import refresh_module_caches
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.moe.token_dispatcher_inference import NVLSAllGatherVDispatcher
    from tests.unit_tests.test_utilities import Utils

    spec.require_flashinfer()
    spec.clear_nvfp4_recipe_env(monkeypatch)
    hidden, inter, experts = 256, 384, 4
    kw = dict(tokens=96, hidden=hidden, inter=inter, experts=experts, local=experts, offset=0)
    kw.update(top_k=2, gated=gated)
    first = nvfp4_moe.Problem(**kw, pattern="random", seed=0)
    second = nvfp4_moe.Problem(**kw, pattern="empty-rows", seed=1, v0=first.v0)
    third = nvfp4_moe.Problem(**kw, pattern="empty-expert", seed=2, v0=first.v0)
    Utils.initialize_model_parallel(1, 1, expert_model_parallel_size=1)
    try:
        model_parallel_cuda_manual_seed(123, force_reset_rng=True)
        layer = _moe_layer(hidden, inter, experts, gated)
        module = layer.experts
        params = {
            linear: [getattr(getattr(module, linear), f"weight{e}") for e in range(experts)]
            for linear, _ in GROUPS
        }
        with torch.no_grad():
            for e in range(experts):
                params["linear_fc1"][e].copy_(first.w1[e].to(torch.bfloat16))
                params["linear_fc2"][e].copy_(first.w2[e].to(torch.bfloat16))
        x, ids, probs = first.x.clone(), first.ids.clone(), first.probs.clone()

        def forward():
            out, bias = module(x, None, probs, ids)
            assert bias is None
            return out

        with torch.no_grad(), InferenceMode.active():
            nvfp4_moe._check(first, forward(), what="first forward (lazy build)")
            weights = module._nvfp4_weights
            assert weights is not None, "the first forward did not build the NVFP4 weights"
            for e in range(experts):
                assert params["linear_fc1"][e].data_ptr() == module._fc1_weight[e].data_ptr()
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    forward()
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                static_out = forward()
            pointers = _poison_nvfp4(weights)
            with torch.no_grad():  # refit's in-place receive into each parameter
                for e in range(experts):
                    params["linear_fc1"][e].data[:, :].copy_(second.w1[e].to(torch.bfloat16))
                    params["linear_fc2"][e].data[:, :].copy_(second.w2[e].to(torch.bfloat16))
            refresh_module_caches(layer)
            assert module._nvfp4_weights is weights
            assert {name: getattr(weights, name).data_ptr() for name in NVFP4_FIELDS} == pointers
            x.copy_(third.x)
            ids.copy_(third.ids)
            probs.copy_(third.probs)
            graph.replay()
            replayed = static_out.clone()
            nvfp4_moe._check(second, replayed, what="graph replay after refit", inputs=third)
            ou.assert_exact(replayed, forward(), what="replay vs eager after refit")
        del graph
    finally:
        torch.cuda.synchronize()
        NVLSAllGatherVDispatcher._delete_buffers()
        SymmetricMemoryManager.destroy()
        Utils.destroy_model_parallel()
