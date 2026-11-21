# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Oracle for FlashInfer's ``nvfp4_quantize(padded, encode, sfLayout=layout_128x4)`` at
Megatron's expert-weight call site: ``_quantize_expert`` through ``prepare_routed_nvfp4_weights``
(``megatron/core/inference/moe/flashinfer_nvfp4.py``), which builds the per-token NVFP4 MoE's
weight stacks.

Contract (from the NVFP4 definition, the call site and FlashInfer's documented API; written
before reading the kernel):

* Each expert weight is zero-padded to ``[rows, cols]`` (H and I rounded up to 256; gated FC1 is
  MCore's ``[gate; up]``, stored as FlashInfer's ``[up, gate]`` halves, each padded, interleaved
  in 64-row blocks; ``spec.cute_dsl_fc1_rows``) and quantized with one FP32 encode scale per
  expert, ``g = 448 * 6 / amax`` (``g = 1`` for an all-zero expert).
* Block of 16 along K: scale ``s_b = E4M3(amax_b * g / 6)`` (nearest even, saturating);
  codes ``E2M1(w * g / s_b)`` (nearest even, saturating at 6, the sign of zero kept). A block
  whose nonzero amax rounds to a zero scale decodes to 0 whatever its codes, so its codes are
  not part of the contract; an all-zero block has zero codes.
* Storage: ``fc*_data[e]`` holds the codes two per byte, the even element in the low nibble;
  ``fc*_scale_swizzled`` rows ``[e * rows, (e + 1) * rows)`` hold expert ``e``'s scale bytes in
  the cuBLAS 128x4 layout (rows is a multiple of 256, so the experts' tiles concatenate into the
  stacked layout); ``fc*_alpha[e] = 1 / g = amax / 2688`` in FP32 (1 for an all-zero expert);
  ``fc*_scale`` is the 6-D MMA view of the
  swizzled scales (its own oracle: ``test_convert_sf_to_mma_layout.py``); ``fc2_input_scale``
  is the per-token recipe constant ``1 / 2688``.
* ``out=`` rewrites every byte of an existing set in place (same addresses); changed metadata
  raises ValueError. Inputs are read only.
* Non-finite weights must stay visible: a NaN weight decodes to NaN (its scale, or the expert's
  alpha, is NaN), as Megatron's other expert-weight formats propagate it. On main a NaN weight
  is quantized as a finite value (finding O4b-M2); #9311 (``fix/nvfp4-nan-expert-weight``) fixes
  that and tests it in ``test_flashinfer_nvfp4.py``
  (``test_prepare_routed_nvfp4_weights_keeps_nan_weights_visible``).

Oracle: ``spec.nvfp4_quantize_spec`` in exact fp64 with the exact ``g = 2688 / amax``; tolerance
class bit-exact against the spec, codes and scales reported apart; alpha an FP32 evaluation of
``amax / 2688`` (``spec.assert_fp32_evaluation``, three roundings: Megatron's ``2688 / amax`` with
a Python numerator is ``reciprocal(amax) * 2688`` in PyTorch, then its reciprocal; 1 exactly for
a zero expert). The FP32 ``g`` Megatron passes differs from the exact one by those roundings,
inside the margin below. FlashInfer evaluates
``amax_b * g / 6`` and ``w * g / s_b`` in FP32 with ``rcp.approx.ftz`` by default (IEEE division
with ``FLASHINFER_DISABLE_FP4_QUANT_FAST_MATH=1``, read at every launch), so a value within
about 2**-21 of a rounding boundary may land on either side under any FP32 recipe: the random
weights are redrawn until every block value and every scaled element is at least
``spec.NVFP4_MARGIN = 2**-16`` (relative) from its boundary (``spec.nvfp4_away_from_ties``).
Round-half-to-even is tested at exact ties built with power-of-two scales
(``spec.nvfp4_tie_blocks``: ``g = 2**k``, ``s_b = 2**j``), where FP32 evaluation is exact
provided ``rcp.approx.ftz`` returns the exact reciprocal of a power of two (if only the
fast-math tie case fails, check that first). Exact ties of the block value cannot be tested:
1/6 is not an FP32 value, so no FP32 recipe computes them exactly.

Structure (FlashInfer 0.7.2.post1 ``invokeFP4Quantization``, read after writing the contract):
the TMA kernel when a matrix has at least 1024 rows and ``cols % 512 == 0`` (16-row tiles of
8 x 64 columns, a persistent grid of 2 blocks per SM), else ``quantize_with_block_size``
(``min(cols / 8, 512)`` threads, 16 elements each, so 8192 columns per pass, rows grid-strided);
each compiled with and without fast math. The shapes put each of FC1 and FC2 on each side of
those switches: rows 768 / 1024 / 1280 with ``cols % 512`` 0 and 256, 8448 columns (two passes
of the non-TMA kernel, rows grid-strided), 5120 rows of TMA tiles (more than one persistent
pass), logical sizes 255 / 257 and 1023 / 1025 around the 256 padding, gated and plain FC1, one
and three experts with an all-zero expert, and production (2688, 1856) and (7168, 2048).
Hardware: Blackwell (FlashInfer's FP4 conversion is SM100-only). ``FLASHINFER_NVFP4_4OVER6*``
are unset (Megatron's hard-coded ``448 * 6`` assumes the plain recipe).
"""

import pytest
import torch

from tests.unit_tests.inference.oracles import oracle_utils as ou
from tests.unit_tests.inference.oracles.flashinfer_moe import spec

ALIGNMENT = spec.NVFP4_MOE_ALIGNMENT
DEVICE = "cuda"


def _setup(monkeypatch, mode="fast"):
    spec.require_flashinfer()
    spec.clear_nvfp4_recipe_env(monkeypatch)
    spec.set_fp4_fast_math(monkeypatch, mode)


def _encode_of(t: torch.Tensor) -> torch.Tensor:
    return spec.nvfp4_weight_encode(t.abs().amax())


def _weight(rows, cols, generator, device, *, zero=False):
    """One BF16-exact expert weight ``[rows, cols]`` (fp64 on ``device``) whose quantization
    under the recipe's ``g`` is away from every rounding boundary. The margins are computed on
    the matrix zero-padded to 256 columns, whose 16-blocks are the kernel's."""
    if zero:
        return torch.zeros(rows, cols, dtype=torch.float64, device=device)
    w = spec.nvfp4_spread_values(rows, spec._ceil_to(cols, ALIGNMENT), generator=generator)
    w[:, cols:] = 0
    w = spec.nvfp4_away_from_ties(w.to(device), _encode_of, generator=generator)
    return w[:, :cols].contiguous()


class Experts:
    """BF16 expert stacks ``W1 [E, I or 2I, H]`` (MCore order) and ``W2 [E, H, I]``, fp64 on CUDA."""

    def __init__(self, *, hidden, inter, experts, gated, generator, zero_expert=None):
        dev = torch.device(DEVICE)
        fc1_rows = 2 * inter if gated else inter
        self.hidden, self.inter, self.gated = hidden, inter, gated
        zero = lambda e: e == zero_expert  # noqa: E731
        self.w1 = torch.stack(
            [_weight(fc1_rows, hidden, generator, dev, zero=zero(e)) for e in range(experts)]
        )
        self.w2 = torch.stack(
            [_weight(hidden, inter, generator, dev, zero=zero(e)) for e in range(experts)]
        )

    def prepare(self, out=None):
        from megatron.core.inference.moe.flashinfer_nvfp4 import prepare_routed_nvfp4_weights

        w1, w2 = self.w1.to(torch.bfloat16), self.w2.to(torch.bfloat16)
        before = (w1.clone(), w2.clone())
        weights = prepare_routed_nvfp4_weights(w1, w2, out=out)
        ou.assert_exact(w1, before[0], what="FC1 weights are read-only")
        ou.assert_exact(w2, before[1], what="FC2 weights are read-only")
        return weights

    def padded(self):
        return spec._ceil_to(self.hidden, ALIGNMENT), spec._ceil_to(self.inter, ALIGNMENT)

    def matrices(self, which):
        """The padded matrices FlashInfer quantizes for ``which`` in ``fc1``/``fc2``, per expert."""
        ph, pi = self.padded()
        if which == "fc1":
            return [spec.cute_dsl_fc1_rows(w, self.gated, pi, ph) for w in self.w1]
        out = []
        for w in self.w2:
            m = torch.zeros(ph, pi, dtype=torch.float64, device=w.device)
            m[: self.hidden, : self.inter] = w
            out.append(m)
        return out


def check_quantized_stack(weights, experts: Experts, what: str) -> None:
    """Every expert of FC1 and FC2: codes, swizzled scales, alpha and the 6-D scale view equal the
    spec quantization of the padded matrix with the recipe's FP32 ``g``."""
    ph, pi = experts.padded()
    assert (weights.hidden_size, weights.intermediate_size) == (experts.hidden, experts.inter)
    assert (weights.padded_hidden_size, weights.padded_intermediate_size) == (ph, pi)
    assert weights.gated == experts.gated and weights.num_local_experts == len(experts.w1)
    ou.assert_exact(
        weights.fc2_input_scale.cpu(),
        torch.tensor([1.0 / spec.NVFP4_PER_TOKEN_DIVISOR], dtype=torch.float32),
        what="fc2_input_scale is the per-token recipe constant 1 / 2688",
    )
    for which in ("fc1", "fc2"):
        data, swizzled = getattr(weights, f"{which}_data"), getattr(
            weights, f"{which}_scale_swizzled"
        )
        alpha, mma = getattr(weights, f"{which}_alpha"), getattr(weights, f"{which}_scale")
        logical_scales = []
        for e, m in enumerate(experts.matrices(which)):
            rows, cols = m.shape
            enc = _encode_of(m)
            codes, scales, defined, _, _ = spec.nvfp4_quantize_spec(m, enc)
            assert data.dtype == torch.uint8 and tuple(data[e].shape) == (rows, cols // 2)
            got_codes = spec.unpack_fp4(data[e])
            flat = swizzled[e * rows : (e + 1) * rows].reshape(-1)
            got_scales = spec.unswizzle_128x4(flat, rows, cols // spec.NVFP4_BLOCK)
            ou.assert_quantized_bit_exact(
                torch.where(defined, got_codes, 0),
                got_scales,
                torch.where(defined, codes, 0),
                scales,
                elems_per_scale=spec.NVFP4_BLOCK,
                what=f"{what}: {which} expert {e} ({rows}x{cols})",
            )
            assert alpha.dtype == torch.float32
            spec.assert_fp32_evaluation(
                alpha[e : e + 1],
                1.0 / enc,
                roundings=spec.NVFP4_ALPHA_ROUNDINGS,
                what=f"{what}: {which} expert {e} alpha = amax / 2688",
            )
            logical_scales.append(scales)
        ou.assert_exact(
            mma,
            spec.mma_layout_expected(torch.stack(logical_scales)),
            what=f"{what}: {which}_scale",
        )


SHAPES = [  # logical (H, I); padded FC1 [I (2I gated), H] and FC2 [H, I], see the docstring
    (256, 256),
    (255, 257),
    (1024, 768),
    (1024, 1024),
    (1023, 1025),
    (768, 8448),
    (5120, 512),
]
CASES = ou.pairwise_cases(
    {
        "shape": SHAPES,
        "gated": [False, True],
        "experts": [1, 3],
        "math": ["fast", "ieee"],
        "zero_expert": [False, True],
    }
)


def _case_id(c):
    return (
        f"H{c['shape'][0]}I{c['shape'][1]}-{'swiglu' if c['gated'] else 'relu2'}-E{c['experts']}"
        f"-{c['math']}{'-zero' if c['zero_expert'] else ''}"
    )


def test_spec_quantizer_and_inputs():
    """CPU check of the oracle: NVFP4-exact weights round-trip through the spec quantizer, the
    redrawn weights keep every margin and the amax, tie blocks sit exactly on E2M1 boundaries
    with power-of-two scales and round to even codes, the recipe's ``g`` of ``2688 * 2**-k`` is
    ``2**k``, the nibble order, and the gated FC1 layout."""
    g = ou.make_generator()
    w = spec.nvfp4_exact_values(64, 256, generator=g)
    enc = _encode_of(w)
    codes, scales, defined, _, _ = spec.nvfp4_quantize_spec(w, enc)
    assert bool(defined.all())
    ou.assert_exact(spec.nvfp4_decode(codes, scales) / enc.double(), w, what="exact round trip")
    for _ in range(3):
        x = spec.nvfp4_spread_values(96, 512, generator=g)
        y = spec.nvfp4_away_from_ties(x, _encode_of, generator=g)
        _, _, defined, block_margin, element_margin = spec.nvfp4_quantize_spec(y, _encode_of(y))
        assert bool((block_margin >= spec.NVFP4_MARGIN).all())
        assert bool((element_margin >= spec.NVFP4_MARGIN).all())
        assert y.abs().amax() == x.abs().amax() and not bool(defined.all())
        assert torch.equal(y.to(torch.bfloat16).double(), y)
    for k in (-20, 0, 12, 30):
        row = spec.nvfp4_tie_blocks(k, g)[None]
        enc = _encode_of(row)
        assert float(enc) == 2.0**k, (k, float(enc))
        codes, scales, defined, _, element_margin = spec.nvfp4_quantize_spec(row, enc)
        s = spec.e4m3_values()[scales.long()].repeat_interleave(spec.NVFP4_BLOCK, -1)
        y = row * 2.0**k / s
        ties = torch.isin(y.abs(), torch.tensor(spec.E2M1_BOUNDARIES, dtype=torch.float64))
        assert int(ties.sum()) == 18 * 15 and bool(defined.all())
        assert bool((codes[ties] % 2 == 0).all()), "ties must round to the even code"
        assert set(torch.log2(s[0, 16::16]).tolist()) == set(range(-9, 9))
    packed = spec.pack_fp4(torch.tensor([[0x3, 0xA]], dtype=torch.uint8))
    assert packed.tolist() == [[0xA3]] and spec.unpack_fp4(packed).tolist() == [[0x3, 0xA]]
    w1 = torch.arange(2 * 96 * 8, dtype=torch.float64).reshape(192, 8)
    rows = spec.cute_dsl_fc1_rows(w1, True, 256, 16)
    assert torch.equal(rows[:64, :8], w1[96:160]) and torch.equal(rows[64:128, :8], w1[:64])
    assert torch.equal(rows[128:160, :8], w1[160:]) and torch.equal(rows[192:224, :8], w1[64:96])
    assert not bool(rows[160:192].any()) and not bool(rows[:, 8:].any())


@pytest.mark.launch_on_gb200
@ou.requires_sm(ou.BLACKWELL)
@pytest.mark.parametrize("case", [pytest.param(c, id=_case_id(c)) for c in CASES])
def test_prepare_routed_nvfp4_weights_matches_spec(case, monkeypatch):
    _setup(monkeypatch, case["math"])
    hidden, inter = case["shape"]
    experts = Experts(
        hidden=hidden,
        inter=inter,
        experts=case["experts"],
        gated=case["gated"],
        generator=ou.make_generator(),
        zero_expert=case["experts"] // 2 if case["zero_expert"] else None,
    )
    check_quantized_stack(experts.prepare(), experts, what=_case_id(case))


@pytest.mark.launch_on_gb200
@ou.requires_sm(ou.BLACKWELL)
@pytest.mark.parametrize(
    "hidden,inter,gated",
    [(2688, 1856, False), (2688, 1856, True), (7168, 2048, False)],
    ids=["2688x1856-relu2", "2688x1856-swiglu", "7168x2048-relu2"],
)
def test_production_shapes(hidden, inter, gated, monkeypatch):
    """Nemotron-like (2688, 1856 -> 2048) with both activations and DeepSeek-V3-like (7168,
    2048): FC1 and FC2 run the TMA and the non-TMA kernels, FC2 [7168, 2048] over more than one
    persistent pass."""
    _setup(monkeypatch)
    experts = Experts(
        hidden=hidden, inter=inter, experts=2, gated=gated, generator=ou.make_generator()
    )
    check_quantized_stack(experts.prepare(), experts, what=f"production {hidden}x{inter}")


@pytest.mark.launch_on_gb200
@ou.requires_sm(ou.BLACKWELL)
@pytest.mark.parametrize("mode", ["fast", "ieee"])
def test_round_half_to_even_at_exact_ties(mode, monkeypatch):
    """Weights made of ``spec.nvfp4_tie_blocks`` rows (``g = 2**k`` exactly, block scales
    ``2**-9 .. 2**8``): every element of 18 of the 19 blocks per row is exactly halfway between two
    E2M1 values and must round to the even code, in both FlashInfer variants."""
    _setup(monkeypatch, mode)
    g = ou.make_generator()
    width = spec.nvfp4_tie_blocks(0, g).numel()  # 19 blocks of 16 = 304 (padded to 512)
    w1 = torch.stack([torch.stack([spec.nvfp4_tie_blocks(12, g) for _ in range(width)])])
    w2 = torch.stack([torch.stack([spec.nvfp4_tie_blocks(-3, g) for _ in range(width)])])
    experts = Experts.__new__(Experts)
    experts.hidden = experts.inter = width
    experts.gated = False
    experts.w1, experts.w2 = w1.to(DEVICE), w2.to(DEVICE)
    check_quantized_stack(experts.prepare(), experts, what=f"ties ({mode})")


@pytest.mark.launch_on_gb200
@ou.requires_sm(ou.BLACKWELL)
@pytest.mark.parametrize("mode", ["fast", "ieee"])
def test_signed_zeros_subnormals_and_underflowing_blocks(mode, monkeypatch):
    """Blocks of -0.0 (codes 0x8, scale 0), BF16 subnormal elements inside normal blocks (signed
    zero codes), and nonzero blocks far below the expert's amax whose scale rounds to an E4M3
    subnormal or to zero (scale byte 0; their codes decode to 0 whatever they are)."""
    _setup(monkeypatch, mode)
    g = ou.make_generator()
    experts = Experts(hidden=512, inter=256, experts=2, gated=False, generator=g)
    w1 = experts.w1[0]
    w1[3, 16:32] = -0.0
    w1[5, :] = 0.0
    w1[7, 33] = -(2.0**-130)
    w1[7, 40] = 2.0**-133
    w1[9, 64:80] = w1[9, 64:80] * 2.0**-24
    w1[9, 80:96] = w1[9, 80:96] * 2.0**-14
    experts.w1[0] = spec.nvfp4_away_from_ties(w1, _encode_of, generator=g)
    m = experts.matrices("fc1")[0]
    codes, scales, defined, _, _ = spec.nvfp4_quantize_spec(m, _encode_of(m))
    assert bool((codes[3, 16:32] == 8).all()) and int(scales[3, 1]) == 0
    assert not bool(defined[9, 64:80].all()) and int(scales[9, 4]) == 0
    check_quantized_stack(experts.prepare(), experts, what=f"special values ({mode})")


@pytest.mark.launch_on_gb200
@ou.requires_sm(ou.BLACKWELL)
@pytest.mark.sanitizer
def test_refresh_rewrites_every_byte_in_place(monkeypatch):
    """``prepare_routed_nvfp4_weights(..., out=weights)`` (refit) requantizes into the same
    storage: every output poisoned first (0xFF bytes, NaN alpha) is rewritten to the new spec
    bytes, at the same addresses; changed metadata raises. Small gated case with partial 16-blocks
    and padding (H 257 -> 512, I 96 -> 256), also the compute-sanitizer case."""
    _setup(monkeypatch)
    g = ou.make_generator()
    first = Experts(hidden=257, inter=96, experts=2, gated=True, generator=g)
    second = Experts(hidden=257, inter=96, experts=2, gated=True, generator=g)
    weights = first.prepare()
    check_quantized_stack(weights, first, what="first build")
    tensors = {
        name: getattr(weights, name)
        for name in (
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
    }
    pointers = {name: t.data_ptr() for name, t in tensors.items()}
    for name, t in tensors.items():
        if t.dtype == torch.uint8 and name.endswith(("data", "swizzled")):
            t.fill_(0xFF)
        elif name.endswith("alpha"):
            t.fill_(float("nan"))
    assert second.prepare(out=weights) is weights
    assert {name: getattr(weights, name).data_ptr() for name in tensors} == pointers
    check_quantized_stack(weights, second, what="refreshed in place")
    other = Experts(hidden=257, inter=128, experts=2, gated=True, generator=g)
    from megatron.core.inference.moe.flashinfer_nvfp4 import prepare_routed_nvfp4_weights

    with pytest.raises(ValueError, match="metadata changed"):
        prepare_routed_nvfp4_weights(
            other.w1.to(torch.bfloat16), other.w2.to(torch.bfloat16), out=weights
        )
