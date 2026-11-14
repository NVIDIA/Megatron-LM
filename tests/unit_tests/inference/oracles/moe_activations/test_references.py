# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Self-tests of the references the MoE activation and MXFP8 oracles use (no kernels).

* ``_mx_spec``: against the OCP tables it is written from (E4M3 Table 2, E8M0 Table 7), against
  torch's own ``float8_e4m3fn`` conversion (an independent round-to-nearest-even E4M3
  implementation), against the cuBLAS layout description, and for its round-up scale rule.
* ``_common``'s exact rounding, on every device the oracles compute on: its bf16 rounding agrees
  with torch's fp32 -> bf16 cast, and torch's own fp32 SwiGLU rounded once passes
  ``assert_rounded_once`` (so the window admits a real fp32 evaluation).
"""

import pytest
import torch

from tests.unit_tests.inference.oracles import oracle_utils as ou
from tests.unit_tests.inference.oracles.moe_activations import _common
from tests.unit_tests.inference.oracles.moe_activations import _mx_spec as mx

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() and not _common.INTERPRET else [])


def all_bf16_values():
    """Every finite bf16 value."""
    bits = torch.arange(0, 2**16, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    return bits[torch.isfinite(bits)]


@pytest.mark.parametrize("device", DEVICES)
def test_e4m3_table(device):
    """OCP FP8 Table 2 entries, and decode/encode round trips on every code."""
    table = torch.tensor([0x7E, 0xFE, 0x08, 0x07, 0x01, 0x00, 0x80], dtype=torch.uint8)
    decode = mx.e4m3_decode(table.to(device)).cpu()
    expected = [448.0, -448.0, 2.0**-6, 0.875 * 2.0**-6, 2.0**-9, 0.0, -0.0]
    ou.assert_exact(decode, torch.tensor(expected, dtype=torch.float64), what="Table 2 values")
    assert torch.signbit(decode[-1]), "0x80 is -0"
    nan = mx.e4m3_decode(torch.tensor([0x7F, 0xFF], dtype=torch.uint8, device=device))
    assert bool(torch.isnan(nan).all()), "S.1111.111 is NaN"
    codes = torch.arange(256, dtype=torch.int64).to(torch.uint8)
    finite = codes[(codes.to(torch.int64) & 0x7F) != 0x7F]
    on_device = finite.to(device)
    ou.assert_exact(
        mx.e4m3_encode(mx.e4m3_decode(on_device)).cpu(), finite, what="encode(decode(c))"
    )
    ou.assert_exact(
        mx.e4m3_decode(on_device).cpu(),
        finite.view(torch.float8_e4m3fn).double(),
        what="decode vs torch float8_e4m3fn",
    )


@pytest.mark.parametrize("device", DEVICES)
def test_e4m3_rounding_matches_torch(device):
    """Round-to-nearest-even onto E4M3 agrees with torch's float8_e4m3fn cast on every bf16 value
    up to 448 and on random fp32 values (ties included: bf16 values have few mantissa bits)."""
    g = ou.make_generator()
    bf16 = all_bf16_values().double()
    fp32 = (
        torch.randn(200_000, generator=g)
        * torch.exp2(torch.randint(-12, 9, (200_000,), generator=g).float())
    ).double()
    for name, values in (("bf16", bf16), ("fp32", fp32)):
        values = values[values.abs() <= mx.E4M3_MAX].to(device)
        ours = mx.e4m3_encode(mx.e4m3_round(values))
        torch_codes = values.to(torch.float32).to(torch.float8_e4m3fn).view(torch.uint8)
        ou.assert_exact(ours, torch_codes, what=f"E4M3 RNE of {name} values vs torch on {device}")


def test_e4m3_saturates():
    """SAT conversion (Table 3): values past the largest finite magnitude become ±448."""
    v = torch.tensor([464.0, 470.0, 1e6, -500.0], dtype=torch.float64)
    ou.assert_exact(
        mx.e4m3_round(v), torch.tensor([448.0, 448.0, 448.0, -448.0], dtype=torch.float64)
    )


@pytest.mark.parametrize("device", DEVICES)
def test_bf16_rounding_matches_torch_fp32_cast(device):
    """``rne_bf16`` and ``bf16_neighbours`` give bf16 values, and ``rne_bf16`` agrees with
    torch's (correctly rounded) fp32 -> bf16 cast, on the device the oracles compute on."""
    g = ou.make_generator()
    x = torch.randn(200_000, generator=g) * torch.exp2(
        torch.randint(-130, 120, (200_000,), generator=g).float()
    )
    x = x[torch.isfinite(x)].to(device)
    nearest = _common.rne_bf16(x.double())
    ou.assert_exact(
        nearest.to(torch.float32), x.to(torch.bfloat16).to(torch.float32), what="rne_bf16 vs torch"
    )
    lo, hi = _common.bf16_neighbours(x.double() * (1 + 2.0**-20))
    for name, t in (("nearest", nearest), ("lower", lo), ("upper", hi)):
        ou.assert_exact(t, t.to(torch.bfloat16).double(), what=f"{name} values are bf16 values")
    assert bool((lo <= hi).all()) and bool(((hi - lo) <= _common.format_quantum(hi, 7, -126)).all())


@pytest.mark.parametrize("device", DEVICES)
def test_pow2_is_exact(device):
    """Powers of two over the whole range the references use, bit for bit."""
    k = torch.arange(-1022, 1024, device=device)
    expected = torch.tensor([2.0**e for e in range(-1022, 1024)], dtype=torch.float64)
    ou.assert_exact(_common.pow2(k).cpu(), expected, what="pow2")


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("scale", [1.0, 4.0, 30.0])
def test_torch_fp32_swiglu_is_rounded_once(device, scale):
    """Torch's fp32 SiLU times the linear half, rounded once to bf16 on the device, passes the
    per-element "rounded once from fp32" check (with its subnormal clause: at scale 30 some gates
    are below -87.34, where fp32 SiLU underflows): the window admits a real fp32 evaluation."""
    from tests.unit_tests.inference.oracles.moe_activations import _glu

    g = ou.make_generator()
    x = (torch.randn(4096, 2 * 512, generator=g) * scale).to(torch.bfloat16).to(device)
    x[0, :512] = torch.linspace(-87.0, 100.0, 512, dtype=torch.float64).to(x)
    exact = ou.fp64_reference(_glu.swiglu_exact, x)
    _common.assert_rounded_once(
        _glu.swiglu_trusted(x),
        exact,
        what=f"torch fp32 SwiGLU on {device}",
        may_flush=_glu.silu_may_flush(x, exact),
    )


@pytest.mark.parametrize("device", DEVICES)
def test_e8m0_table(device):
    """Table 7: bias 127, exponents -127..127, 0xFF is NaN."""
    s = mx.e8m0_decode(torch.tensor([0, 1, 127, 254, 255], dtype=torch.uint8, device=device))
    s = s.cpu()
    ou.assert_exact(s[:4], torch.tensor([2.0**-127, 2.0**-126, 1.0, 2.0**127], dtype=torch.float64))
    assert bool(torch.isnan(s[4]))


def test_round_up_scale_rule():
    """The scale is the smallest power of two X with amax / X <= 448, within E8M0's range; it is
    the §6.3 exponent, or one more when amax's significand exceeds 1.75 (where §6.3 saturates)."""
    k = torch.arange(-119, 120, dtype=torch.float64)
    at = 448.0 * torch.exp2(k)
    ou.assert_exact(mx.rceil_scale_exponent(at), k.to(torch.int64), what="amax = 448 * 2**k")
    above = torch.nextafter(at.float(), torch.tensor(float("inf"))).double()
    ou.assert_exact(mx.rceil_scale_exponent(above), k.to(torch.int64) + 1, what="just above")
    tiny = torch.tensor([0.0, 2.0**-149, 2.0**-133, 448.0 * 2.0**-127], dtype=torch.float64)
    ou.assert_exact(mx.rceil_scale_exponent(tiny), torch.full((4,), -127), what="E8M0 floor")
    g = ou.make_generator()
    amax = torch.rand(100_000, generator=g, dtype=torch.float64) * torch.exp2(
        torch.randint(-100, 100, (100_000,), generator=g).double()
    )
    amax = amax.float().double()
    up = mx.rceil_scale_exponent(amax)
    assert bool((amax / torch.exp2(up.double()) <= mx.E4M3_MAX).all())
    assert bool((amax / torch.exp2(up.double() - 1) > mx.E4M3_MAX).all())
    significand, _ = torch.frexp(amax)
    bump = (2 * significand > 1.75).to(torch.int64)
    ou.assert_exact(up, mx.floor_scale_exponent(amax) + bump, what="round-up vs OCP 6.3")


def test_swizzle_layout():
    """The cuBLAS 128 x 4 tile layout: positions named in its description, padding, inverse."""
    rows, cols = 300, 10
    pr, pc = mx.padded_scale_shape(rows, cols)
    assert (pr, pc) == (384, 12)
    ids = (
        (torch.arange(rows * cols, dtype=torch.int64) % 251 + 1).to(torch.uint8).reshape(rows, cols)
    )
    buf = mx.swizzle(ids)
    assert buf.numel() == mx.scale_bytes(rows, cols * mx.BLOCK) == 384 * 12
    tiles_per_row = pc // 4

    def at(r, c):
        tile = (r // 128) * tiles_per_row + c // 4
        return int(buf[tile * 512 + (r % 32) * 16 + ((r % 128) // 32) * 4 + c % 4])

    for r, c in ((0, 0), (32, 0), (96, 3), (1, 0), (0, 4), (128, 0), (299, 9), (161, 7)):
        assert at(r, c) == int(ids[r, c]), (r, c)
    assert int(buf[4]) == int(ids[32, 0]) and int(buf[16]) == int(ids[1, 0])
    back, padding = mx.unswizzle(buf, rows, cols)
    ou.assert_exact(back, ids, what="unswizzle(swizzle(s))")
    assert padding.numel() == pr * pc - rows * cols and int(padding.count_nonzero()) == 0


def test_quantize_decodes_within_half_a_code():
    """Every finite element decodes to the E4M3 grid point nearest to x / X, times X."""
    g = ou.make_generator()
    x = (
        torch.randn(64, 256, generator=g)
        * torch.exp2(torch.randint(-20, 20, (64, 256), generator=g).float())
    ).bfloat16()
    codes, scales = mx.quantize(x)
    decoded = mx.dequantize(codes, scales)
    x64 = x.double().reshape(64, 8, 32)
    scale = mx.e8m0_decode(scales)[..., None]
    q = x64 / scale
    grid = mx.e4m3_round(q)
    ou.assert_exact(decoded.reshape(64, 8, 32) / scale, grid, what="decode = RNE(x / X)")
    assert bool((q.abs() <= mx.E4M3_MAX).all()), "round-up scales never saturate"
