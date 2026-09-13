# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU unit tests for the integer weight-only QAT primitive (phase 1).

Covers the design's three-state separation: master-weight identity, fake-quant
forward + STE backward numerics, disabled=bit-identical, and the packed
deployment round-trip (K-0150 maxdiff=0).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize
from megatron.lite.primitive.quantization.qat import (
    QATSpec,
    WeightFakeQuant,
    _FakeQuantizeSTE,
    apply_qat_to_chunks,
    compute_amax,
    dequantize_weight,
    fake_quantize_weight,
    normalize_qat_spec,
    pack_int4,
    quantize_weight,
    unpack_int4,
)

pytestmark = pytest.mark.mlite


# --------------------------------------------------------------------------- config


def test_spec_defaults_are_inert_and_normalize():
    spec = normalize_qat_spec(None)
    assert spec.enabled is False
    assert (
        normalize_qat_spec(
            {"enabled": True, "format": "int4", "group_size": 8}
        ).num_bits
        == 4
    )
    assert normalize_qat_spec(QATSpec(enabled=True)).num_bits == 8
    with pytest.raises(TypeError, match="QAT config"):
        normalize_qat_spec(object())


def test_spec_rejects_deferred_and_unsupported_formats():
    # NVFP4 stays deferred (needs its own per-16 FP8 block scale + serializer).
    for fmt in ("nvfp4_w4a16", "nvfp4_w4a4"):
        with pytest.raises(ValueError, match="deferred"):
            QATSpec(enabled=True, format=fmt)
    with pytest.raises(ValueError, match="Unknown QAT format"):
        QATSpec(enabled=True, format="int3")
    with pytest.raises(ValueError, match="activation quantization"):
        QATSpec(enabled=True, activation_bits=4)
    with pytest.raises(ValueError, match="learnable_scales"):
        QATSpec(enabled=True, learnable_scales=True)
    # disabled spec never validates format -> stays inert
    assert QATSpec(enabled=False, format="nvfp4_w4a16").enabled is False


def test_float_format_aliases_and_mxfp4_block():
    # "fp8" canonicalises to the exact E4M3 contract key.
    assert QATSpec(enabled=True, format="fp8").format == "fp8_e4m3"
    assert QATSpec(enabled=False, format="fp8").format == "fp8_e4m3"
    assert normalize_qat_spec({"enabled": True, "format": "fp8"}).format == "fp8_e4m3"
    # mxfp4 is a microscaling block format: group_size must be exactly 32.
    for bad in (0, -1, 16, 64):
        with pytest.raises(ValueError, match="microscaling block"):
            QATSpec(enabled=True, format="mxfp4", group_size=bad)
    assert QATSpec(enabled=True, format="mxfp4").group_size == 32
    assert QATSpec(enabled=True, format="mxfp4", group_size=32).num_bits == 4


@pytest.mark.parametrize("group_size", [0, -1, 4])
def test_integer_group_size_keeps_explicit_semantics(group_size):
    assert (
        QATSpec(enabled=True, format="int4", group_size=group_size).group_size
        == group_size
    )


def test_targets_module_skips_ignore_patterns():
    spec = QATSpec(enabled=True)
    assert spec.targets_module("layers.0.mlp.gate_up")
    assert not spec.targets_module("lm_head")
    assert not spec.targets_module("layers.0.mlp.router.gate")


# --------------------------------------------------------------------------- numerics


@pytest.mark.parametrize("num_bits,fmt", [(8, "int8"), (4, "int4")])
@pytest.mark.parametrize("group_size", [0, -1, 4])
def test_fake_quant_matches_manual_qdq(num_bits, fmt, group_size):
    torch.manual_seed(0)
    w = torch.randn(6, 8, dtype=torch.float32)
    spec = QATSpec(enabled=True, format=fmt, group_size=group_size)
    w_hat = fake_quantize_weight(w, spec)

    # manual reference
    qmax = (1 << (num_bits - 1)) - 1
    if group_size == 0:
        scale = w.abs().amax() / qmax
        ref = torch.round(w / scale).clamp(-qmax, qmax) * scale
    elif group_size == -1:
        scale = w.abs().amax(dim=1, keepdim=True) / qmax
        ref = torch.round(w / scale).clamp(-qmax, qmax) * scale
    else:
        v = w.reshape(6, 8 // group_size, group_size)
        scale = v.abs().amax(dim=2, keepdim=True) / qmax
        ref = (torch.round(v / scale).clamp(-qmax, qmax) * scale).reshape(6, 8)
    torch.testing.assert_close(w_hat, ref, rtol=0, atol=0)


def test_fake_quant_error_bounded_by_scale():
    # DQ error must be at most half a quantization step per element.
    w = torch.randn(4, 16, dtype=torch.float32)
    spec = QATSpec(enabled=True, format="int8", group_size=-1)
    w_hat = fake_quantize_weight(w, spec)
    scale = (w.abs().amax(dim=1, keepdim=True) / 127).expand_as(w)
    assert torch.all((w - w_hat).abs() <= scale / 2 + 1e-6)


def test_affine_quant_reconstructs_range():
    w = torch.linspace(-3.0, 5.0, steps=32).reshape(2, 16)
    spec = QATSpec(enabled=True, format="int8", symmetric=False, group_size=0)
    w_hat = fake_quantize_weight(w, spec)
    # affine covers the asymmetric [-3,5] range; error <= one step
    step = (w.max() - w.min()) / 255
    assert torch.all((w - w_hat).abs() <= step + 1e-5)


# --------------------------------------------------------------------------- fp8 / mxfp4


@pytest.mark.parametrize("group_size", [0, -1, 32])
def test_fp8_fake_quant_matches_native_e4m3_cast(group_size):
    torch.manual_seed(10)
    w = torch.randn(8, 64, dtype=torch.float32)
    spec = QATSpec(enabled=True, format="fp8", group_size=group_size)
    w_hat = fake_quantize_weight(w, spec)

    fmax = float(torch.finfo(torch.float8_e4m3fn).max)
    if group_size == 0:
        scale = (w.abs().amax() / fmax).clamp_min(torch.finfo(torch.float32).tiny)
        ref = (w / scale).clamp(-fmax, fmax).to(torch.float8_e4m3fn).float() * scale
    elif group_size == -1:
        scale = (w.abs().amax(dim=1, keepdim=True) / fmax).clamp_min(
            torch.finfo(torch.float32).tiny
        )
        ref = (w / scale).clamp(-fmax, fmax).to(torch.float8_e4m3fn).float() * scale
    else:
        v = w.reshape(8, 64 // group_size, group_size)
        scale = (v.abs().amax(dim=2, keepdim=True) / fmax).clamp_min(
            torch.finfo(torch.float32).tiny
        )
        ref = (
            (v / scale).clamp(-fmax, fmax).to(torch.float8_e4m3fn).float() * scale
        ).reshape(8, 64)
    torch.testing.assert_close(w_hat, ref, rtol=0, atol=0)


def test_fp8_amax_scaling_does_not_saturate():
    # amax/448 scaling maps the peak weight exactly to 448 -> nothing clips.
    torch.manual_seed(11)
    w = torch.randn(4, 128)
    spec = QATSpec(enabled=True, format="fp8", group_size=-1)
    w_hat = fake_quantize_weight(w, spec)
    # E4M3 has 3 mantissa bits: relative error <= 2^-4 of the local binade.
    rel = (w - w_hat).abs() / w.abs().clamp_min(1e-6)
    assert rel.max() < 0.07


def test_mxfp4_matches_e2m1_grid_and_e8m0_scale():
    torch.manual_seed(12)
    w = torch.randn(4, 96, dtype=torch.float32)
    spec = QATSpec(enabled=True, format="mxfp4", group_size=32)
    w_hat = fake_quantize_weight(w, spec)

    # Independent reference: the E8M0 shared scale + E2M1 rounding that the
    # rollout quantizer (ModelOpt MXFP4QTensor) uses -- ceil(log2(amax/6)) and
    # ties-down. This test previously asserted OCP Alg. 1 (floor(log2 amax) - 2)
    # plus ties-to-even, which is a *different* encoding and was what let the
    # training/rollout mismatch ship. Bit-exactness against ModelOpt itself is
    # covered by tests/unit/primitive/test_mxfp4_modelopt_parity_unit.py.
    levels = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    mids = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
    v = w.reshape(4, 96 // 32, 32)
    amax = v.abs().amax(dim=2, keepdim=True)
    exp = torch.ceil(torch.maximum(torch.log2(amax / 6.0), torch.tensor(-127.0)))
    X = torch.exp2(exp)
    idx = torch.bucketize((v / X).abs(), mids, right=False)
    ref = (torch.sign(v / X) * levels[idx] * X).reshape(4, 96)
    torch.testing.assert_close(w_hat, ref, rtol=0, atol=0)
    # every reconstructed value lies on grid*scale (a power of two multiple).
    assert torch.isfinite(w_hat).all()


def test_mxfp4_scale_never_saturates_so_ste_is_identity():
    # The E8M0 scale is chosen as ceil(log2(amax/6)), so amax/X <= 6 always and
    # nothing is ever clipped -- including the block maximum. The old Alg. 1 rule
    # did clip it (amax/X could reach 8), and this test used to assert that the
    # STE zeroed the block top's gradient. The deployed quantizer does not
    # saturate, so training must not drop that gradient either.
    w = torch.zeros(1, 32)
    w[0, 0] = 6.25  # amax=6.25 -> ceil(log2(6.25/6)) = 1 -> X=2 -> 3.125 <= 6
    w[0, 1] = 1.0
    wg = w.clone().requires_grad_(True)
    fake_quantize_weight(
        wg, QATSpec(enabled=True, format="mxfp4", group_size=32)
    ).sum().backward()
    assert wg.grad[0, 0].item() == 1.0  # block top is representable -> passes through
    assert wg.grad[0, 1].item() == 1.0  # in-range passes through


# --------------------------------------------------------------------------- STE


def test_ste_passes_gradient_through_qdq():
    w = torch.randn(5, 8, dtype=torch.float32, requires_grad=True)
    spec = QATSpec(enabled=True, format="int4", group_size=0)
    fake_quantize_weight(w, spec).sum().backward()
    # dynamic max-calibration: no element saturates, so STE is identity (grad=1).
    assert w.grad is not None
    torch.testing.assert_close(w.grad, torch.ones_like(w))


def test_ste_clip_zeroes_saturated_grad_and_passthrough_toggle():
    # Drive saturation explicitly with a too-small scale so codes exceed [-7,7].
    w = torch.tensor([[-2.0, -0.1, 0.1, 2.0]], requires_grad=True)
    scale = torch.tensor(0.1)  # code = w/scale = [-20,-1,1,20] -> saturates at +-7
    out = _FakeQuantizeSTE.apply(w, scale, None, -7, 7, True)
    out.sum().backward()
    assert w.grad.tolist() == [[0.0, 1.0, 1.0, 0.0]]

    w2 = w.detach().clone().requires_grad_(True)
    out2 = _FakeQuantizeSTE.apply(w2, scale, None, -7, 7, False)  # pure pass-through
    out2.sum().backward()
    assert w2.grad.tolist() == [[1.0, 1.0, 1.0, 1.0]]


# --------------------------------------------------------------------------- master identity


def test_parametrization_preserves_master_weight_identity():
    lin = nn.Linear(8, 6, bias=False).to(torch.bfloat16)
    master_before = lin.weight
    spec = QATSpec(enabled=True, format="int4", group_size=-1)
    parametrize.register_parametrization(
        lin, "weight", WeightFakeQuant(spec, lin.weight.shape), unsafe=True
    )
    # master survives untouched as .original, still trainable, still bf16
    original = lin.parametrizations.weight.original
    assert original is master_before
    assert original.requires_grad
    assert original.dtype == torch.bfloat16
    # accessing .weight yields W_hat (quantized), not the master
    assert lin.weight.dtype == torch.bfloat16
    assert not torch.equal(lin.weight, original)
    # gradient flows back to the master through STE
    x = torch.randn(3, 8, dtype=torch.bfloat16)
    lin(x).sum().backward()
    assert original.grad is not None


# --------------------------------------------------------------------------- disabled = bit-identical


def _toy_chunk():
    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.qkv = nn.Linear(8, 12, bias=False)
            self.proj = nn.Linear(12, 8, bias=False)
            self.lm_head = nn.Linear(8, 16, bias=False)

        def forward(self, x):
            return self.lm_head(self.proj(self.qkv(x)))

    return Toy()


def test_disabled_spec_is_bit_identical():
    torch.manual_seed(1)
    chunk = _toy_chunk()
    x = torch.randn(4, 8)
    ref = chunk(x)
    stats = apply_qat_to_chunks([chunk], QATSpec(enabled=False))
    assert stats == {
        "quantized_modules": 0,
        "skipped_ignored": 0,
        "skipped_no_weight": 0,
    }
    assert not parametrize.is_parametrized(chunk.qkv, "weight")
    torch.testing.assert_close(chunk(x), ref, rtol=0, atol=0)


def test_apply_quantizes_targets_and_skips_ignored():
    torch.manual_seed(2)
    chunk = _toy_chunk()
    spec = QATSpec(enabled=True, format="int8", group_size=-1)
    stats = apply_qat_to_chunks([chunk], spec)
    assert stats["quantized_modules"] == 2  # qkv + proj
    assert parametrize.is_parametrized(chunk.qkv, "weight")
    assert parametrize.is_parametrized(chunk.proj, "weight")
    assert not parametrize.is_parametrized(chunk.lm_head, "weight")  # ignored
    # forward changes (weights are now fake-quantized) and grads flow to masters
    x = torch.randn(4, 8)
    chunk(x).sum().backward()
    assert chunk.qkv.parametrizations.weight.original.grad is not None
    assert chunk.lm_head.weight.grad is not None  # untouched, plain param


# --------------------------------------------------------------------------- deployment round-trip


@pytest.mark.parametrize("fmt", ["int8", "int4"])
@pytest.mark.parametrize("group_size", [0, -1, 4])
def test_export_roundtrip_matches_fake_quant(fmt, group_size):
    torch.manual_seed(3)
    w = torch.randn(6, 8, dtype=torch.float32)
    spec = QATSpec(enabled=True, format=fmt, group_size=group_size)
    packed = quantize_weight(w, spec)
    recon = dequantize_weight(packed, spec)
    # deploy dequant must equal the training fake-quant exactly (K-0150 maxdiff=0)
    torch.testing.assert_close(recon, fake_quantize_weight(w, spec), rtol=0, atol=0)


@pytest.mark.parametrize(
    "fmt,group_size",
    [("fp8", 0), ("fp8", -1), ("fp8", 32), ("mxfp4", 32)],
)
def test_float_export_roundtrip_matches_fake_quant(fmt, group_size):
    torch.manual_seed(13)
    w = torch.randn(6, 64, dtype=torch.float32)
    spec = QATSpec(enabled=True, format=fmt, group_size=group_size)
    packed = quantize_weight(w, spec)
    recon = dequantize_weight(packed, spec)
    # deploy dequant must equal the training fake-quant exactly (K-0150 maxdiff=0)
    torch.testing.assert_close(recon, fake_quantize_weight(w, spec), rtol=0, atol=0)


def test_mxfp4_export_layout_is_packed_nibbles_and_e8m0_bytes():
    torch.manual_seed(14)
    w = torch.randn(6, 64, dtype=torch.float32)
    spec = QATSpec(enabled=True, format="mxfp4", group_size=32)
    packed = quantize_weight(w, spec)
    assert packed["format"] == "mxfp4"
    # 64 in-features / 32 block = 2 blocks; each block packs 32 nibbles -> 16 bytes
    assert packed["qweight"].dtype == torch.uint8 and packed["qweight"].shape == (
        6,
        2,
        16,
    )
    assert packed["scale"].dtype == torch.uint8 and packed["scale"].shape == (6, 2, 1)


def test_fp8_export_stores_native_fp8_codes():
    torch.manual_seed(15)
    w = torch.randn(6, 64, dtype=torch.float32)
    spec = QATSpec(enabled=True, format="fp8", group_size=-1)
    packed = quantize_weight(w, spec)
    assert packed["format"] == "fp8_e4m3"
    assert packed["qweight"].dtype == torch.float8_e4m3fn
    assert packed["scale"].dtype == torch.float32


def _toy_chunk_32aligned():
    # in_features divisible by 32 so the mxfp4 block layout applies.
    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.qkv = nn.Linear(64, 96, bias=False)
            self.proj = nn.Linear(96, 64, bias=False)
            self.lm_head = nn.Linear(64, 128, bias=False)

        def forward(self, x):
            return self.lm_head(self.proj(self.qkv(x)))

    return Toy()


@pytest.mark.parametrize("fmt,group_size", [("fp8", -1), ("mxfp4", 32)])
def test_float_apply_preserves_master_and_flows_grad(fmt, group_size):
    torch.manual_seed(16)
    chunk = _toy_chunk_32aligned()
    spec = QATSpec(enabled=True, format=fmt, group_size=group_size)
    stats = apply_qat_to_chunks([chunk], spec)
    assert stats["quantized_modules"] == 2  # qkv + proj, lm_head skipped
    master = chunk.qkv.parametrizations.weight.original
    assert master.requires_grad
    # accessing .weight yields the fake-quantized W_hat, not the master
    assert not torch.equal(chunk.qkv.weight, master)
    x = torch.randn(4, 64)
    chunk(x).sum().backward()
    assert master.grad is not None
    assert chunk.lm_head.weight.grad is not None  # untouched plain param


def test_int4_pack_unpack_roundtrip():
    torch.manual_seed(4)
    w = torch.randn(6, 8, dtype=torch.float32)
    spec = QATSpec(enabled=True, format="int4", group_size=-1)
    codes = quantize_weight(w, spec)["qweight"]
    assert codes.min() >= -7 and codes.max() <= 7
    packed = pack_int4(codes)
    assert packed.dtype == torch.uint8 and packed.shape == (6, 4)
    torch.testing.assert_close(unpack_int4(packed), codes, rtol=0, atol=0)


def test_amax_buffer_shapes():
    w = torch.randn(6, 8)
    assert compute_amax(w, 0).shape == torch.Size([])
    assert compute_amax(w, -1).shape == torch.Size([6, 1])
    assert compute_amax(w, 4).shape == torch.Size([6, 2, 1])


# ------------------------------------------------------------ load-then-apply (QAT-HF-LOAD-MISS)
#
# Regression for the moe BLOCKER QAT-HF-LOAD-MISS: build_model applies QAT
# (parametrize) before the runtime loads HF weights, so the checkpoint loader
# must map the logical ``….weight`` name onto ``….parametrizations.weight.original``.
# Without the mapping the master weight is silently never loaded and training
# runs on random weights. These CPU tests import the real qwen3_5 checkpoint
# loader and prove the master receives the real tensor for all four formats.

from megatron.lite.primitive.ckpt.hf_weights import _resolve_param_name  # noqa: E402
from megatron.lite.primitive.quantization.qat import canonical_state_key  # noqa: E402


def _toy_linear_chunk(in_features: int = 64, out_features: int = 12):
    """Toy chunk whose linears wrap the GEMM as ``.linear`` (like MLite parallel
    linears), so ``_quantizable_weight_owner`` targets ``.linear.weight``."""

    class _Owner(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(in_features, out_features, bias=False)

    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.qkv = _Owner()
            self.proj = _Owner()

    return Toy().to(torch.bfloat16)


def test_canonical_state_key_strips_only_parametrization_original():
    assert canonical_state_key("a.b.linear.weight") == "a.b.linear.weight"
    assert (
        canonical_state_key("a.b.linear.parametrizations.weight.original")
        == "a.b.linear.weight"
    )
    # quantizer buffers keep their real name (must NOT collide with the master)
    assert (
        canonical_state_key("a.b.linear.parametrizations.weight.0.amax")
        == "a.b.linear.parametrizations.weight.0.amax"
    )


def _copy_via_checkpoint_primitive(
    model: nn.Module, loaded: dict[str, torch.Tensor]
) -> None:
    state = model.state_dict()
    targets = dict(model.named_parameters(remove_duplicate=False))
    targets.update(dict(model.named_buffers(remove_duplicate=False)))
    for logical_name, tensor in loaded.items():
        actual = _resolve_param_name(logical_name, state)
        assert actual is not None
        targets[actual].data.copy_(tensor)


@pytest.mark.parametrize(
    "fmt,group_size",
    [("int8", -1), ("int4", -1), ("fp8_e4m3", -1), ("mxfp4", 32)],
)
def test_apply_before_load_still_loads_master_weight(fmt, group_size):
    torch.manual_seed(7)
    chunk = _toy_linear_chunk()
    # 1) apply QAT (parametrize) BEFORE load, exactly as build_model does.
    apply_qat_to_chunks(
        [chunk], QATSpec(enabled=True, format=fmt, group_size=group_size)
    )
    assert parametrize.is_parametrized(chunk.qkv.linear, "weight")

    # 2) load the HF-mapped state (keyed by the logical ``….linear.weight``).
    real_qkv = torch.randn(12, 64, dtype=torch.bfloat16)
    real_proj = torch.randn(12, 64, dtype=torch.bfloat16)
    _copy_via_checkpoint_primitive(
        chunk,
        {"qkv.linear.weight": real_qkv, "proj.linear.weight": real_proj},
    )

    # 3) the BF16 master (.original) must hold the real weights — not random init.
    m_qkv = chunk.qkv.linear.parametrizations.weight.original
    m_proj = chunk.proj.linear.parametrizations.weight.original
    torch.testing.assert_close(m_qkv, real_qkv, rtol=0, atol=0)
    torch.testing.assert_close(m_proj, real_proj, rtol=0, atol=0)

    # 4) the quantized view differs from the master (parametrization is live) and
    #    a forward recomputes amax from the *real* (loaded) weight, not zeros.
    assert not torch.equal(chunk.qkv.linear.weight, m_qkv)
    _ = chunk.qkv.linear.weight  # trigger parametrization forward -> amax update
    amax = chunk.qkv.linear.parametrizations.weight[0].amax
    assert torch.all(amax > 0)


def test_master_left_random_without_mapping_would_fail_naively():
    """Guard: prove the loaded key does NOT substring-match the parametrized key,
    i.e. the naive ``name in key`` path (the original bug) really misses it."""
    torch.manual_seed(8)
    chunk = _toy_linear_chunk()
    apply_qat_to_chunks([chunk], QATSpec(enabled=True, format="int8", group_size=-1))
    state_keys = list(chunk.state_dict().keys())
    loaded_name = "qkv.linear.weight"
    assert loaded_name not in state_keys  # not an exact key anymore
    assert not any(loaded_name in k for k in state_keys)  # not a substring either
    # but canonicalization recovers it
    canon = {canonical_state_key(k): k for k in state_keys}
    assert canon[loaded_name] == "qkv.linear.parametrizations.weight.original"


def test_load_unparametrized_module_unchanged_by_refactor():
    """The primitive resolver must not regress the plain (non-QAT) path."""
    torch.manual_seed(9)
    chunk = _toy_linear_chunk()
    real = torch.randn(12, 64, dtype=torch.bfloat16)
    _copy_via_checkpoint_primitive(chunk, {"qkv.linear.weight": real})
    torch.testing.assert_close(chunk.qkv.linear.weight, real, rtol=0, atol=0)
