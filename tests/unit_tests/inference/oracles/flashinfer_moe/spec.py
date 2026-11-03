# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Spec references shared by the FlashInfer MoE, GEMM and quantization oracles.

Everything here is written from format specifications and the call sites' documented contracts,
not from FlashInfer's code:

* **OCP Microscaling (MX) v1.0.** A block of 32 elements along the contiguous dimension shares one
  E8M0 scale ``2**(b - 127)`` (``b = 0xFF`` is NaN). MXFP8 elements are FP8 E4M3: bias 7, no
  infinities, ``S.1111.111`` is NaN, largest magnitude 448, subnormals ``m/8 * 2**-6``. A block
  is quantized with the *RCEIL* scale used by NVIDIA's MXFP8 recipe (Transformer Engine,
  Megatron's Triton quantizer, FlashInfer's documented ``mxfp8_quantize``): the smallest power of
  two ``X`` with ``amax <= 448 * X``, clamped to E8M0's range, then each element is rounded to
  nearest even in E4M3. OCP's reference algorithm instead takes ``X = 2**(floor(log2 amax) - 8)``
  and saturates; the two agree unless the block amax mantissa lies in (1.75, 2).
* **NVFP4.** 16 FP4 E2M1 elements (values 0, 0.5, 1, 1.5, 2, 3, 4, 6) share an E4M3 block scale;
  a tensor (weights) or token (per-token activations) carries an FP32 scale. Per token:
  ``s_t = amax_t / (448 * 6)``, block scale ``E4M3(amax_b / (6 * s_t))``, codes
  ``E2M1(x / (scale_b * s_t))``. Weights: ``encode = 448 * 6 / amax``, block scale
  ``E4M3(amax_b * encode / 6)``, codes ``E2M1(w * encode / scale_b)``, decode ``/ encode``.
* **cuBLAS 128x4 block-scale layout** ("1D block scaling factors layout"): the scale matrix is
  padded to 128 rows and 4 columns and stored in 512-byte tiles, tiles row-block-major; inside a
  tile, scale ``(m, k)`` sits at ``(m % 32) * 16 + (m % 128 // 32) * 4 + k % 4``.
* **TRT-LLM Major-K row shuffle** (``shuffleMatrixA`` with ``epilogue_tile_m = 128``): inside each
  block of 32 rows, source row ``4a + b`` (``a < 8``, ``b < 4``) moves to row ``8b + a`` (the
  32 rows are a row-major 8x4 grid written out column-major).

The MoE reference is the per-token math of an MCore MoE layer in fp64:
``out[t] = sum_k p[t, k] * W2[e] @ act(W1[e] @ x[t])`` over the routes ``e = ids[t, k]`` that
are local experts. The structured-input generators below build inputs on which every
quantization step of the MXFP8 and NVFP4 kernels (input, weights, and the FC2-input requantization
inside the kernel) is lossless, so the fp64 reference of the unquantized math is the exact
contract and the only rounding left is the kernel's accumulation and its BF16 output.
"""

import contextlib
import math
import os
from typing import Callable, Optional, Sequence, Tuple

import pytest
import torch

from tests.unit_tests.inference.oracles import oracle_utils as ou

E4M3_MAX = 448.0
E2M1_MAX = 6.0
MX_BLOCK = 32
NVFP4_BLOCK = 16
E8M0_BIAS = 127
NVFP4_PER_TOKEN_DIVISOR = E4M3_MAX * E2M1_MAX  # 2688
TRTLLM_SHUFFLE_BLOCK = 32  # epilogue_tile_m = 128

# ---------------------------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------------------------


def require_flashinfer(module: str = "flashinfer"):
    """Import a FlashInfer module; fail when the run targets GPU hardware, else skip.

    FlashInfer is part of the CI image, so a targeted runner (``MCORE_ORACLE_TARGET_SM`` or a
    GB200 ``-m launch_on_gb200`` run) without it is a broken environment, not a reason to skip.
    """
    try:
        return __import__(module, fromlist=["_"])
    except ImportError as error:
        reason = f"{module} is not importable: {error}"
        if ou.session_target_sm() is not None:
            pytest.fail(reason, pytrace=False)
        pytest.skip(reason)


def device() -> torch.device:
    """The device the layout tests run on: CUDA when present (production), else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@contextlib.contextmanager
def exact_fp32_matmul():
    """Disable TF32 for the trusted fp32 implementations (restored afterwards)."""
    matmul, cudnn = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = matmul, cudnn


NVFP4_RECIPE_ENV_PREFIX = "FLASHINFER_NVFP4_4OVER6"


def clear_nvfp4_recipe_env(monkeypatch) -> None:
    """Unset FlashInfer's ``FLASHINFER_NVFP4_4OVER6*`` recipe switches (they change NVFP4's
    scale choice, which Megatron's hard-coded 448 * 6 global scale does not follow)."""
    for key in list(os.environ):
        if key.startswith(NVFP4_RECIPE_ENV_PREFIX):
            monkeypatch.delenv(key)


# ---------------------------------------------------------------------------------------------
# Element codecs (fp64 values <-> codes), from the format definitions
# ---------------------------------------------------------------------------------------------


def pow2(e: torch.Tensor) -> torch.Tensor:
    """Exact fp64 ``2**e`` for integer-valued ``e`` in [-1022, 1023], built from the exponent bits
    (``torch.pow`` on CUDA is not exact for fp64)."""
    bits = (e.to(torch.int64) + 1023) << 52
    return bits.view(torch.float64)


def e4m3_values(device_: torch.device | str = "cpu") -> torch.Tensor:
    """fp64 value of each of the 256 FP8 E4M3 bytes (NaN for 0x7F and 0xFF)."""
    b = torch.arange(256, dtype=torch.int64, device=device_)
    sign = torch.where(b >= 128, -1.0, 1.0).double()
    exponent = (b >> 3) & 0xF
    mantissa = (b & 0x7).double()
    normal = (1.0 + mantissa / 8.0) * pow2(exponent - 7)
    subnormal = mantissa / 8.0 * 2.0**-6
    values = sign * torch.where(exponent == 0, subnormal, normal)
    return torch.where((exponent == 15) & ((b & 0x7) == 7), torch.nan, values)


E2M1_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def e2m1_values(device_: torch.device | str = "cpu") -> torch.Tensor:
    """fp64 value of each of the 16 FP4 E2M1 codes (bit 3 is the sign)."""
    grid = torch.tensor(E2M1_GRID, dtype=torch.float64, device=device_)
    return torch.cat([grid, -grid])


def _round_to_grid(magnitude: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """Index of the nearest ``grid`` value (ascending, from 0) to each magnitude, ties to the
    even index. In E4M3 and E2M1 the code's last bit is the significand's last bit, so the even
    index is IEEE round-to-nearest-even, also across binade and subnormal boundaries."""
    hi = torch.searchsorted(grid, magnitude.contiguous()).clamp(max=grid.numel() - 1)
    lo = (hi - 1).clamp(min=0)
    d_lo = magnitude - grid[lo]
    d_hi = grid[hi] - magnitude
    pick_hi = (d_hi < d_lo) | ((d_hi == d_lo) & (hi % 2 == 0))
    return torch.where(pick_hi, hi, lo)


def e4m3_encode(values: torch.Tensor) -> torch.Tensor:
    """Bytes of ``values`` rounded to nearest even E4M3, saturating to +-448; the sign of zero is
    kept. NaN maps to 0x7F. Returns uint8."""
    v = values.double()
    grid = e4m3_values(v.device)[:127]  # +0 ... 448, ascending
    idx = _round_to_grid(v.abs().clamp(max=E4M3_MAX).nan_to_num(0.0), grid)
    code = idx + torch.where(torch.signbit(v), 128, 0)
    return torch.where(torch.isnan(v), 0x7F, code).to(torch.uint8)


def e4m3_round(values: torch.Tensor) -> torch.Tensor:
    """``values`` rounded to E4M3 (saturating), as fp64."""
    return e4m3_values(values.device)[e4m3_encode(values).long()]


def e2m1_encode(values: torch.Tensor) -> torch.Tensor:
    """4-bit E2M1 codes (sign in bit 3) of ``values`` rounded to nearest even, saturating to
    +-6. Returns uint8 in [0, 16)."""
    v = values.double()
    grid = torch.tensor(E2M1_GRID, dtype=torch.float64, device=v.device)
    idx = _round_to_grid(v.abs().clamp(max=E2M1_MAX), grid)
    return (idx + torch.where(torch.signbit(v), 8, 0)).to(torch.uint8)


def e8m0_value(scale_bytes: torch.Tensor) -> torch.Tensor:
    """fp64 value ``2**(b - 127)`` of E8M0 bytes (0xFF is NaN)."""
    b = scale_bytes.to(torch.int64)
    value = pow2(b - E8M0_BIAS)
    return torch.where(b == 0xFF, torch.nan, value)


# ---------------------------------------------------------------------------------------------
# MXFP8 quantization (RCEIL) and decoding
# ---------------------------------------------------------------------------------------------


def rceil_exponent(amax: torch.Tensor) -> torch.Tensor:
    """Smallest integer ``e`` with ``amax <= 448 * 2**e``, clamped to E8M0's [-127, 127]; exact.

    ``amax == 0`` gives -127 (the smallest scale). Computed with exact fp64 comparisons, so a
    block whose amax is exactly ``448 * 2**k`` gets ``k``.
    """
    a = amax.double()
    e = torch.ceil(torch.log2(a / E4M3_MAX)).nan_to_num(nan=-127.0, neginf=-127.0, posinf=127.0)
    e = e.clamp(-200.0, 200.0)
    e = torch.where(a > E4M3_MAX * pow2(e), e + 1, e)
    e = torch.where(a <= E4M3_MAX * pow2(e - 1), e - 1, e)
    e = torch.where(a == 0, -127.0, e)
    return e.clamp(-127.0, 127.0)


def mxfp8_quantize_spec(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Spec MXFP8 of a finite ``[M, K]`` tensor (``K % 32 == 0``), blocks along K.

    Returns ``(codes, scales)``: E4M3 bytes ``[M, K]`` (uint8) and E8M0 bytes ``[M, K // 32]``
    (uint8), linear layout.
    """
    m, k = x.shape
    assert k % MX_BLOCK == 0, k
    xb = x.double().reshape(m, k // MX_BLOCK, MX_BLOCK)
    e = rceil_exponent(xb.abs().amax(-1))
    codes = e4m3_encode(xb / pow2(e)[..., None])
    return codes.reshape(m, k), (e + E8M0_BIAS).to(torch.uint8)


def mxfp8_decode(codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """fp64 values of MXFP8 ``codes [..., K]`` (bytes) with linear ``scales [..., K // 32]``."""
    values = e4m3_values(codes.device)[codes.to(torch.int64)]
    return values * e8m0_value(scales).repeat_interleave(MX_BLOCK, dim=-1)


def canonical_mxfp8(values: torch.Tensor):
    """A canonical (Triton / cuBLAS layout) ``MXFP8Tensor`` holding MX-exact ``values [E, R,
    C]`` exactly (CPU): spec codes, and spec scales in the 128x4 layout per expert."""
    from megatron.core.inference.quantization.mxfp8_tensor import MXFP8Tensor

    experts, rows, cols = values.shape
    codes, scales = mxfp8_quantize_spec(values.reshape(experts * rows, cols))
    back = mxfp8_decode(codes, scales).reshape(values.shape)
    ou.assert_exact(back, values.double(), what="structured weights are MXFP8-exact")
    scales = scales.reshape(experts, rows, cols // MX_BLOCK)
    swizzled = torch.stack([swizzle_128x4(s) for s in scales])
    return MXFP8Tensor(
        data=codes.reshape(values.shape).view(torch.float8_e4m3fn),
        scale=swizzled.to(torch.uint8).view(torch.float8_e8m0fnu),
        backend="triton",
    )


def to_cuda(weight):
    """A copy of an ``MXFP8Tensor`` on CUDA."""
    return type(weight)(data=weight.data.cuda(), scale=weight.scale.cuda(), backend=weight.backend)


def as_bytes(t: torch.Tensor) -> torch.Tensor:
    """The tensor's bytes as uint8 (same shape for 1-byte dtypes)."""
    return t.contiguous().view(torch.uint8)


# ---------------------------------------------------------------------------------------------
# Layouts: cuBLAS 128x4 scale swizzle and the TRT-LLM row shuffle
# ---------------------------------------------------------------------------------------------


def _ceil_to(n: int, m: int) -> int:
    return -(-n // m) * m


def swizzle_offsets(rows: int, cols: int, device_: torch.device | str = "cpu") -> torch.Tensor:
    """Byte offset of scale ``(m, k)`` in the 128x4 layout of a ``[rows, cols]`` matrix."""
    tiles_k = _ceil_to(cols, 4) // 4
    m = torch.arange(rows, device=device_)[:, None]
    k = torch.arange(cols, device=device_)[None, :]
    tile = (m // 128) * tiles_k + k // 4
    return tile * 512 + (m % 32) * 16 + (m % 128 // 32) * 4 + k % 4


def swizzle_128x4(scales: torch.Tensor) -> torch.Tensor:
    """1-D 128x4 layout of a ``[rows, cols]`` byte matrix, zero-padded to 128 rows, 4 columns."""
    rows, cols = scales.shape
    out = torch.zeros(
        _ceil_to(rows, 128) * _ceil_to(cols, 4), dtype=scales.dtype, device=scales.device
    )
    out[swizzle_offsets(rows, cols, scales.device).reshape(-1)] = scales.reshape(-1)
    return out


def unswizzle_128x4(flat: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """Inverse of ``swizzle_128x4``: the ``[rows, cols]`` matrix held in a 1-D 128x4 buffer."""
    offsets = swizzle_offsets(rows, cols, flat.device)
    return flat.reshape(-1)[offsets.reshape(-1)].reshape(rows, cols)


def trtllm_shuffle_source_rows(rows: int, device_: torch.device | str = "cpu") -> torch.Tensor:
    """For each destination row of the Major-K shuffle, the source row it takes (``rows % 32 ==
    0``): destination ``8b + a`` of a 32-row block takes source ``4a + b``."""
    assert rows % TRTLLM_SHUFFLE_BLOCK == 0, rows
    d = torch.arange(rows, device=device_)
    inner = d % TRTLLM_SHUFFLE_BLOCK
    return d - inner + 4 * (inner % 8) + inner // 8


# ---------------------------------------------------------------------------------------------
# NVFP4 quantization (used to check that the structured inputs are lossless)
# ---------------------------------------------------------------------------------------------


def nvfp4_per_token_roundtrip(x: torch.Tensor) -> torch.Tensor:
    """Per-token NVFP4 quantize-dequantize of ``[T, K]`` (``K % 16 == 0``) in fp64.

    With ``s = amax / 2688``: block scale ``E4M3(amax_b / (6 s)) = E4M3(448 amax_b / amax)``,
    codes ``E2M1(x / (scale_b s))``, value ``code * scale_b * amax / 2688`` (evaluated in that
    order, so a lossless input comes back bit for bit despite ``s`` not being exact in fp64).
    """
    t, k = x.shape
    xb = x.double().reshape(t, k // NVFP4_BLOCK, NVFP4_BLOCK)
    amax = xb.abs().amax((1, 2))[:, None]  # [T, 1]
    safe_amax = torch.where(amax > 0, amax, torch.ones_like(amax))
    block = e4m3_round(E4M3_MAX * xb.abs().amax(-1) / safe_amax)  # [T, K/16]
    unit = (block * safe_amax / NVFP4_PER_TOKEN_DIVISOR)[..., None]
    safe_unit = torch.where(unit > 0, unit, torch.ones_like(unit))
    codes = e2m1_values(x.device)[e2m1_encode(xb / safe_unit).long()]
    deq = codes * block[..., None] * safe_amax[..., None] / NVFP4_PER_TOKEN_DIVISOR
    return torch.where(unit > 0, deq, torch.zeros_like(deq)).reshape(t, k)


def nvfp4_weight_roundtrip(w: torch.Tensor) -> torch.Tensor:
    """Per-tensor-scaled NVFP4 quantize-dequantize of ``[R, K]`` (``K % 16 == 0``), fp64.

    ``encode = 2688 / amax``; block scale ``E4M3(amax_b * encode / 6) = E4M3(448 amax_b /
    amax)``; codes ``E2M1(w * encode / scale_b)``; value ``code * scale_b * amax / 2688``.
    """
    r, k = w.shape
    wb = w.double().reshape(r, k // NVFP4_BLOCK, NVFP4_BLOCK)
    amax = wb.abs().amax()
    safe_amax = amax if amax > 0 else torch.ones_like(amax)
    block = e4m3_round(E4M3_MAX * wb.abs().amax(-1) / safe_amax)[..., None]
    unit = block * safe_amax / NVFP4_PER_TOKEN_DIVISOR
    safe_unit = torch.where(unit > 0, unit, torch.ones_like(unit))
    codes = e2m1_values(w.device)[e2m1_encode(wb / safe_unit).long()]
    deq = codes * block * safe_amax / NVFP4_PER_TOKEN_DIVISOR
    return torch.where(unit > 0, deq, torch.zeros_like(deq)).reshape(r, k)


def mxfp8_roundtrip(x: torch.Tensor) -> torch.Tensor:
    """Spec MXFP8 quantize-dequantize (fp64)."""
    return mxfp8_decode(*mxfp8_quantize_spec(x))


# ---------------------------------------------------------------------------------------------
# MoE reference
# ---------------------------------------------------------------------------------------------


def relu2(z: torch.Tensor) -> torch.Tensor:
    """Squared ReLU."""
    return torch.relu(z).square()


def relu2_of_bf16(z: torch.Tensor) -> torch.Tensor:
    """Squared ReLU of the FC1 output stored in BF16: GEMM1 writes BF16 and a separate kernel
    applies the activation (FlashInfer's unfused CUTLASS MoE activation; MCore's BF16 path)."""
    return relu2(z.to(torch.bfloat16).to(z.dtype))


def clamped_relu2(limit: float) -> Callable[[torch.Tensor], torch.Tensor]:
    """``(L * tanh(relu(z) / L))**2``: MCore's squared ReLU with ``activation_func_tanh_clamp_scale
    = L``."""

    def act(z: torch.Tensor) -> torch.Tensor:
        return (limit * torch.tanh(torch.relu(z) / limit)).square()

    return act


def swiglu(z: torch.Tensor) -> torch.Tensor:
    """MCore GLU: FC1 output is ``[gate; up]`` along the last dim; ``silu(gate) * up``."""
    gate, up = z.chunk(2, dim=-1)
    return torch.nn.functional.silu(gate) * up


def moe_reference(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    expert_ids: torch.Tensor,
    probs: torch.Tensor,
    *,
    activation: Callable[[torch.Tensor], torch.Tensor],
    local_offset: int,
    act_round: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    out_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Per-token MoE: ``out[t] = sum_k probs[t, k] * w2[e] @ act(w1[e] @ x[t])`` over routes to
    local experts ``e - local_offset in [0, len(w1))``; other routes (including -1) add nothing.

    Computed in the dtype of ``x`` (fp64 for the reference, fp32 for a trusted implementation);
    ``act_round`` rounds the activation output where the contract stores it in lower precision.
    ``out_dtype`` (default: ``x``'s dtype) is the accumulator of the weighted route outputs: with
    BF16, each route's weighted output is rounded to BF16 and added in BF16, as an output-dtype
    finalize (atomics or a sequential sum) does.
    """
    out = torch.zeros(x.shape[0], w2.shape[1], dtype=out_dtype or x.dtype, device=x.device)
    for j in range(w1.shape[0]):
        hit = expert_ids == local_offset + j  # [T, k]
        rows = hit.any(-1).nonzero().reshape(-1)
        if rows.numel() == 0:
            continue
        weight = (probs.to(x.dtype) * hit).sum(-1)[rows]
        a = activation(x[rows] @ w1[j].to(x.dtype).T)
        if act_round is not None:
            a = act_round(a)
        out[rows] += (weight[:, None] * (a @ w2[j].to(x.dtype).T)).to(out.dtype)
    return out


def bf16_tie_distance(values: torch.Tensor) -> torch.Tensor:
    """Distance from each value to the nearest BF16 rounding boundary (the midpoint between two
    BF16 neighbours), relative to the value; ``inf`` for zeros (fp64)."""
    v = values.double()
    r = v.to(torch.bfloat16).double()
    dist = ou.ulp(v, torch.bfloat16) / 2 - (v - r).abs()
    return torch.where(v != 0, dist / v.abs().clamp(min=1e-300), torch.full_like(v, math.inf))


def round_to(dtype: torch.dtype) -> Callable[[torch.Tensor], torch.Tensor]:
    """Rounding to ``dtype`` and back (for ``act_round``)."""

    def rnd(t: torch.Tensor) -> torch.Tensor:
        return t.to(dtype).to(t.dtype)

    return rnd


# ---------------------------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------------------------


def make_routing(
    tokens: int,
    top_k: int,
    num_experts: int,
    *,
    generator: torch.Generator,
    device_: torch.device | str,
    empty_expert: Optional[int] = None,
    empty_rows: Sequence[int] = (),
    dropped_routes: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Random top-k routing: distinct global expert ids per token and FP32 probabilities.

    ``empty_expert`` is never chosen; tokens in ``empty_rows`` route nowhere (all -1, as the
    dispatcher marks rows past the live count); ``dropped_routes`` routes elsewhere are -1.
    """
    choices = [e for e in range(num_experts) if e != empty_expert]
    assert len(choices) >= top_k, (num_experts, top_k)
    pool = torch.tensor(choices, dtype=torch.int64)
    keys = torch.rand(tokens, len(choices), generator=generator)
    ids = pool[keys.argsort(-1)[:, :top_k]]
    probs = torch.rand(tokens, top_k, generator=generator) + 0.05
    probs = probs / probs.sum(-1, keepdim=True)
    if empty_rows:
        ids[list(empty_rows)] = -1
    if dropped_routes and tokens:
        flat = torch.randperm(tokens * top_k, generator=generator)[:dropped_routes]
        ids.view(-1)[flat] = -1
    return ids.to(device_), probs.float().to(device_)


# ---------------------------------------------------------------------------------------------
# Structured inputs on which every quantization step is lossless
# ---------------------------------------------------------------------------------------------


def random_signs(shape: Sequence[int], generator: torch.Generator) -> torch.Tensor:
    """Random +-1.0 (fp64, CPU)."""
    return torch.randint(0, 2, tuple(shape), generator=generator).double() * 2 - 1


def mx_exact_values(
    rows: int, cols: int, *, generator: torch.Generator, exponents: Tuple[int, int] = (-10, -8)
) -> torch.Tensor:
    """Random ``[rows, cols]`` fp64 values that MXFP8 represents exactly (blocks of 32 along
    ``cols``): E4M3 values in ``[2**-3, 448]`` with random signs, times a per-block ``2**s``,
    ``s`` in ``exponents``; one element per block is pinned to a magnitude in [288, 416], so
    every RCEIL quantizer (exact or with an approximate 1/448) picks ``2**s``."""
    assert cols % MX_BLOCK == 0, cols
    grid = e4m3_values()[:127]
    usable = grid[(grid >= 2.0**-3) & (grid <= E4M3_MAX)]
    pins = grid[(grid >= 288.0) & (grid <= 416.0)]
    blocks = cols // MX_BLOCK
    mag = usable[torch.randint(0, usable.numel(), (rows, blocks, MX_BLOCK), generator=generator)]
    pos = torch.randint(0, MX_BLOCK, (rows, blocks, 1), generator=generator)
    pin = pins[torch.randint(0, pins.numel(), (rows, blocks, 1), generator=generator)]
    mag = mag.scatter(-1, pos, pin)
    s = torch.randint(exponents[0], exponents[1] + 1, (rows, blocks, 1), generator=generator)
    values = mag * random_signs(mag.shape, generator) * pow2(s)
    return values.reshape(rows, cols)


def nvfp4_exact_values(
    rows: int,
    cols: int,
    *,
    generator: torch.Generator,
    block_exponents: Tuple[int, int] = (-3, 0),
    row_exponents: Tuple[int, int] = (0, 0),
) -> torch.Tensor:
    """Random ``[rows, cols]`` fp64 values that NVFP4 represents exactly, both per token and per
    tensor (blocks of 16 along ``cols``): E2M1 values times ``2**(f_b + g_r)``; each block holds a
    +-6 so its scale is ``448 * 2**(f_b - max f)`` exactly, and one block per row has the
    largest ``f``. ``g_r`` (per row, from ``row_exponents``) spreads row magnitudes."""
    assert cols % NVFP4_BLOCK == 0, cols
    blocks = cols // NVFP4_BLOCK
    grid = torch.tensor(E2M1_GRID[1:], dtype=torch.float64)  # nonzero magnitudes
    mag = grid[torch.randint(0, grid.numel(), (rows, blocks, NVFP4_BLOCK), generator=generator)]
    pos = torch.randint(0, NVFP4_BLOCK, (rows, blocks, 1), generator=generator)
    mag = mag.scatter(-1, pos, torch.full((rows, blocks, 1), E2M1_MAX, dtype=torch.float64))
    lo, hi = block_exponents
    f = torch.randint(lo, hi + 1, (rows, blocks, 1), generator=generator)
    top = torch.randint(0, blocks, (rows, 1, 1), generator=generator)
    f = f.scatter(1, top, torch.full((rows, 1, 1), hi, dtype=torch.int64))
    g = torch.randint(row_exponents[0], row_exponents[1] + 1, (rows, 1, 1), generator=generator)
    values = mag * random_signs(mag.shape, generator) * pow2(f + g)
    return values.reshape(rows, cols)


def rank1_factors(
    experts: int,
    rows: int,
    cols: int,
    *,
    row_block: int,
    generator: torch.Generator,
    levels: int = 3,
    v_scale_exponents: Tuple[int, int] = (0, 0),
    v0: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Factors of a rank-1 FC1 ``W1[e] = u[e] v[e]^T`` whose activation requantizes losslessly.

    ``u[e, i] = +-2**-(n_b + d_i)``: ``n_b`` in ``[0, levels)`` per block of ``row_block`` rows
    (the FC2-input quantization block along the intermediate dim), ``d_i`` in {0, 1}. So ``u**2``
    takes two adjacent powers of 4 inside a block and powers of 4 across blocks. ``v[e] = +-2**j_e
    * v0`` with ``v0[h]`` in {+-1, +-1/2} (``v0[0] = 1``) shared by all experts (drawn unless
    given). Returns fp64 ``(u [E, rows], v [E, cols])``.
    """
    blocks = -(-rows // row_block)
    n = torch.randint(0, levels, (experts, blocks), generator=generator).repeat_interleave(
        row_block, dim=1
    )[:, :rows]
    d = torch.randint(0, 2, (experts, rows), generator=generator)
    u = random_signs((experts, rows), generator) * pow2(-(n + d))
    if v0 is None:
        v0 = random_signs((cols,), generator) * pow2(
            -torch.randint(0, 2, (cols,), generator=generator)
        )
        v0[0] = 1.0
    j = torch.randint(
        v_scale_exponents[0], v_scale_exponents[1] + 1, (experts, 1), generator=generator
    )
    v = random_signs((experts, 1), generator) * pow2(j) * v0
    return u, v


def few_hot_tokens(
    tokens: int,
    v0: torch.Tensor,
    *,
    generator: torch.Generator,
    exponents: Tuple[int, int] = (-2, 2),
    max_terms_log2: int = 3,
) -> torch.Tensor:
    """``[T, H]`` fp64 tokens with ``2**p`` nonzeros (``p <= max_terms_log2``) in distinct
    32-blocks, each with ``v0[h] * x[h] = 2**k_t``: so ``v0 . x = 2**(k_t + p)`` exactly, every
    nonzero is a power of two (lossless in MXFP8), and FC1's K loop adds several blocks."""
    h = v0.numel()
    blocks = h // MX_BLOCK
    x = torch.zeros(tokens, h, dtype=torch.float64)
    for t in range(tokens):
        p = int(torch.randint(0, max_terms_log2 + 1, (1,), generator=generator))
        terms = min(2**p, 2 ** int(math.log2(max(blocks, 1))))
        chosen = torch.randperm(blocks, generator=generator)[:terms]
        offset = torch.randint(0, MX_BLOCK, (terms,), generator=generator)
        cols = chosen * MX_BLOCK + offset
        k = int(torch.randint(exponents[0], exponents[1] + 1, (1,), generator=generator))
        x[t, cols] = 2.0**k / v0[cols]
    return x
