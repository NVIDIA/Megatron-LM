# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Float64 emulator of the fused GLM mHC projection pair and of the Sinkhorn VJP.

A test oracle for the fused GLM mHC kernels. It imports torch only (no Triton, no megatron), runs on
CPU or CUDA, and can also be loaded by file path.

Exact mode (float64, no intermediate rounding):
    glm_pair_reference_f64  the GLM projection pair (hyper_connection._proj_rms_impl with eps
                            inside the sqrt and normalize_before_projection), forward and
                            backward;
    glm_mappings_chain_f64  its analytic backward through the normalize-before-projection branch
                            of HyperConnectionModule._compute_h and the Sinkhorn: the chain that
                            owns the alpha and bias gradients;
    sinkhorn_f64            hyper_connection._sinkhorn_iterations;
    sinkhorn_vjp            the algorithm of fused_mhc_kernels._triton_sinkhorn_bwd_kernel.

Rounding mode: glm_pair_kernel_model is the fused kernels' arithmetic. GP-1 computes the RMS scale
s (modeled), GP-2 the split-K projection partials, GP-3 their reduction, GP-4 the dgrad, the fp32
epilogue and the split-M wgrad partials, and GP-4r the wgrad reduction. Each MMA contraction is

    C[..., m, n] = sum_k T(A[..., m, k]) * T(Bt[..., n, k]),

with T() a TF32 operand rounding and the products exact in float64. A ChainPlan holds the kernel's
splits: each split runs its own chain from acc = 0 in MMA k-steps counted from the split start, and
the fp32 split partials are added in index order with IEEE round-to-nearest. A ColumnChainPlans
gives each group of output columns its own ChainPlan (the wgrad's pieces per column tile). Two plan
sets exist: glm_pair_kernel_plans, the P2 prototype's (split-K 16, split-M 2), on which this
module's P2 tests and power rows were computed, and glm_pair_fused_plans, the shipped kernels'
(GP-2's non-empty chains, GP-4's per-column wgrad pieces). pair_data draws the tests' inputs. An
AccumModel sets the arithmetic of one chain (acc is fp32):

    exact  the float64 sum of the products, Neumaier-compensated; splits are summed in float64;
    rn     IEEE round-to-nearest per product, acc = fl32(acc + a * b). With rn_block = B a fresh
           accumulator runs over each B products and is added to a running total (the blocked RN
           of r01, an earlier model of the tensor-core accumulation that the tests reproduce);
           rn_block = None is one continuous chain, as acc = tl.dot(a, b, acc) runs;
    rz     per k-step, acc = RZ32(acc + the sum of the k-step's products);
    align  per k-step, every addend (acc included) is truncated toward zero onto the grid
           ulp32(max |addend|) * 2**-align_extra_bits, the truncated addends are summed exactly,
           and the sum is rounded toward zero to fp32;
    rnk    (diagnostic) per k-step, acc = RN32(acc + the sum of the k-step's products).

ALIGN_K8_X2 with "rne" operands is the configuration calibrated on GB300 (sm_103) hardware, with
tcgen05 kind::tf32 MMAs, for p, g_xn and g_W: it reproduces all 18 crafted hardware outputs.

Exactness. A product of two TF32 operands has at most 22 significant bits, so the float64 k-step
sums of rz and rnk are exact while a k-step's addends span at most about 29 binades (a product of
raw fp32 operands has 48 bits). Beyond that, the float64 rounding can move the fp32 result by one
ulp next to an fp32 boundary. rn is the IEEE fp32 add for TF32 operands: their product is an fp32
value, and the double rounding fl32(fl64(acc + a * b)) is innocuous (53 >= 2 * 24 + 2). align is
exact for any operands: the truncated addends are integer multiples of the grid below
2**(24 + align_extra_bits) each.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor

# fl32(1e-5), the eps that hyper_connection's fp32 "+ eps" adds. The float64 1e-5 differs by a
# relative 2.5e-9, which moves s by about 1e-9 at x scale 1e-3 and on all-zero rows: more than the
# 1e-10 exact bars.
EPS_FP32 = 9.999999747378752e-06
# hyper_connection's _MHC_SINKHORN_EPS and the default mhc_sinkhorn_iterations.
SINKHORN_EPS, SINKHORN_ITERS = 1e-6, 20
COMPUTE_H_EPS = 1e-6  # hyper_connection's _MHC_COMPUTE_H_EPS: h_pre = sigmoid(.) + eps
PAIR_N = 24  # n^2 + 2n mapping outputs at n = 4 residual streams: pair_data's W and G_p width
# TransformerConfig.mhc_fused_tf32_rounding -> TF32 operand mode. "rn" is RNE here (ties to even),
# not the RNA (ties away from zero) that some TF32 emulations call "rn".
CONFIG_ROUNDING_TO_MODE = {"rn": "rne", "rz": "rz"}

TF32_MODES = ("rne", "rz", "rna")
OPERAND_MODES = ("none",) + TF32_MODES
# The MMA operand slots of the projection pair, for per-slot operand modes.
OPERAND_SLOTS = ("xn", "W_fwd", "G_p_dgrad", "W_dgrad", "G_p_wgrad", "xn_wgrad")
ACCUM_KINDS = ("exact", "rn", "rz", "align", "rnk")
DEFAULT_MAX_CHUNK_ELEMS = 1 << 25  # float64 elements of one [rows, N, splits, k_step] product block

_TF32_KEEP_MASK = -0x2000  # 0xFFFFE000 as a signed int32
_SIGN_BIT = 0x80000000
# Named TF32 rounding edge cases (uint32 patterns); tf32_edge_patterns adds each with the sign bit
# flipped. Ties of both kept-bit parities, just above and below half, a sticky bit at every position
# below the guard, the carry into the exponent, max finite to inf, inf and NaN payloads (a bare mask
# turns 0x7F800001 into inf), subnormals with the carry to the minimum normal, and zero.
_TF32_NAMED_EDGES = (
    (0x3F801000, 0x3F803000, 0x3F801001, 0x3F800FFF)
    + tuple(0x3F801000 | 1 << j for j in range(12))
    + (0x3FFFF000, 0x7F7FFFFF, 0x7F7FEFFF)
    + (0x7F800000, 0x7F800001, 0x7FC00000, 0xFFC00001)
    + (0x00000FFF, 0x00001000, 0x00003000, 0x007FF000)
    + (0x00000000, 0x80000000)
)
_TF32_RANDOM_PATTERNS, _TF32_LCG_SEED = 1 << 16, 20261101

# The rel-RMS bars within which the fused kernels must match this emulator in their GPU tests, and
# the power thresholds of the mutation checks (10 x the bar), exactly as recorded from a GB300
# (sm_103) measurement: each bar is 3 x the largest rel-RMS between the outputs of the reference
# Triton TF32 kernels and this emulator's ALIGN_K8_X2 model. In the names, P9 is that measurement
# and M3 those GPU tests. The g_W values are established for a tcgen05 wgrad only; an mma.sync
# wgrad needs its g_W bar re-established first.
P9_M3_BARS = {
    "p": 1.1844862602519642e-07,
    "g_xn": 2.9209568810857357e-08,
    "g_W": 1.117417368749157e-07,
}
P9_POWER_THRESHOLDS = {
    "p": 1.1844862602519643e-06,
    "g_xn": 2.920956881085736e-07,
    "g_W": 1.117417368749157e-06,
}


# --------------------------------------------------------------------------------------------
# TF32 operand rounding and its integer oracle.
# --------------------------------------------------------------------------------------------


def tf32_round(x: Tensor, mode: str) -> Tensor:
    """Round fp32 values to TF32 (10 explicit mantissa bits) under mode; inf and NaN pass through.

    The torch twin of the kernels' integer rounding: RNE is b = (b + 0xFFF + ((b >> 13) & 1)) &
    0xFFFFE000, RZ masks the low 13 bits, and RNA adds 0x1000 before the mask.
    """
    if x.dtype != torch.float32:
        raise TypeError(f"tf32_round expects float32, got {x.dtype}")
    bits = x.contiguous().view(torch.int32)
    if mode == "rne":
        rounded = (bits + 0xFFF + ((bits >> 13) & 1)) & _TF32_KEEP_MASK
    elif mode == "rz":
        rounded = bits & _TF32_KEEP_MASK
    elif mode == "rna":
        rounded = (bits + 0x1000) & _TF32_KEEP_MASK
    else:
        raise ValueError(f"unknown TF32 rounding mode {mode!r}; expected one of {TF32_MODES}")
    return torch.where(torch.isfinite(x), rounded.view(torch.float32), x).view(x.shape)


def tf32_round_int_reference(bits: int, mode: str) -> int:
    """TF32 rounding of one uint32 fp32 pattern, in integer arithmetic on the magnitude.

    An exponent of 0xFF (inf, NaN) passes through. Otherwise the result is sign | (kept + up) << 13,
    so the carry into the exponent and the overflow to inf follow from the integer add.
    """
    if not 0 <= bits <= 0xFFFFFFFF:
        raise ValueError(f"not a uint32 pattern: {bits:#x}")
    sign, magnitude = bits & _SIGN_BIT, bits & 0x7FFFFFFF
    if magnitude >> 23 == 0xFF:
        return bits
    kept, low = magnitude >> 13, magnitude & 0x1FFF
    if mode == "rne":
        round_up = low > 0x1000 or (low == 0x1000 and kept & 1 == 1)
    elif mode == "rz":
        round_up = False
    elif mode == "rna":
        round_up = low >= 0x1000
    else:
        raise ValueError(f"unknown TF32 rounding mode {mode!r}; expected one of {TF32_MODES}")
    return sign | (kept + int(round_up)) << 13


def tf32_edge_patterns() -> List[int]:
    """uint32 patterns for TF32 rounding tests: the named edge cases, each with both signs, then
    65,536 patterns of a seeded 32-bit LCG (whose low 13 bits run through every value 8 times)."""
    named = [bits ^ sign for bits in _TF32_NAMED_EDGES for sign in (0, _SIGN_BIT)]
    state, drawn = _TF32_LCG_SEED, []
    for _ in range(_TF32_RANDOM_PATTERNS):
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        drawn.append(state)
    return list(dict.fromkeys(named)) + drawn


# --------------------------------------------------------------------------------------------
# Elementary operations.
# --------------------------------------------------------------------------------------------


def round_operand(x: Tensor, mode: str) -> Tensor:
    """T(x) as float64: the TF32 rounding of fp32 values under mode, or the raw fp32 value ("none").

    Takes fp32, bf16 and fp16 tensors (bf16 and fp16 convert exactly) and float64 tensors holding
    fp32 values. "none" is a diagnostic: TF32 hardware truncates raw fp32 operands.
    """
    if mode not in OPERAND_MODES:
        raise ValueError(f"unknown operand mode {mode!r}; expected one of {OPERAND_MODES}")
    if x.dtype == torch.float64:
        x32 = x.to(torch.float32)
        if not torch.equal(x32.to(torch.float64), x):
            raise ValueError(
                "float64 operand is not exactly fp32-representable; pass the fp32 tensor"
            )
    elif x.dtype in (torch.float32, torch.bfloat16, torch.float16):
        x32 = x.to(torch.float32)
    else:
        raise TypeError(f"unsupported operand dtype {x.dtype}")
    if mode == "none":
        return x32.to(torch.float64)
    return tf32_round(x32, mode).to(torch.float64)


def rz_to_fp32(v: Tensor) -> Tensor:
    """Round float64 values toward zero to fp32; values beyond the fp32 range give max finite."""
    f = v.to(torch.float32)  # round to nearest
    over = f.to(torch.float64).abs() > v.abs()
    return torch.where(over, torch.nextafter(f, torch.zeros_like(f)), f)


def ulp32_grid(maxabs: Tensor, extra_bits: int = 0, with_inverse: bool = False):
    """2**(e - 24 - extra_bits) for maxabs = m * 2**e with m in [0.5, 1), and 1 where maxabs is 0.

    Built from the float64 exponent field, so it is exact; maxabs must be >= 0 and normal, and the
    grid is clamped to the normal float64 range. With with_inverse, also returns the exact
    reciprocal power of two, so callers multiply instead of divide.
    """
    if maxabs.dtype != torch.float64:
        raise TypeError("ulp32_grid expects float64")
    e64 = maxabs.contiguous().view(torch.int64) >> 52  # biased exponent; no sign bit as maxabs >= 0
    g_exp = torch.clamp(e64 - (23 + int(extra_bits)), min=1, max=2045)  # biased exponent of grid
    positive = maxabs > 0
    one = torch.ones_like(maxabs)
    grid = torch.where(positive, (g_exp << 52).view(torch.float64), one)
    if not with_inverse:
        return grid
    inverse = torch.where(positive, ((2046 - g_exp) << 52).view(torch.float64), one)
    return grid, inverse


def ordered_reduce(partials: Tensor, dim: int = 0) -> Tensor:
    """fp32 round-to-nearest sum of the partials along dim, in index order, from tot = 0.

    The split reduction of GP-3 (projection) and GP-4r (wgrad).
    """
    if partials.dtype != torch.float32:
        raise TypeError("ordered_reduce expects float32 partials")
    parts = partials.movedim(dim, 0)
    tot = torch.zeros_like(parts[0])
    for s in range(parts.shape[0]):
        tot = tot + parts[s]
    return tot


def rel_rms(actual: Tensor, reference: Tensor) -> float:
    """||actual - reference|| / ||reference|| in float64; 0 when both are zero.

    The rel-RMS that P9_M3_BARS and P9_POWER_THRESHOLDS are stated in.
    """
    if actual.shape != reference.shape:
        raise ValueError(f"shape mismatch {tuple(actual.shape)} vs {tuple(reference.shape)}")
    a = actual.detach().to(torch.float64)
    r = reference.detach().to(torch.float64)
    denom = r.norm().item()
    num = (a - r).norm().item()
    if denom == 0.0:
        return 0.0 if num == 0.0 else math.inf
    return num / denom


# --------------------------------------------------------------------------------------------
# Accumulation models and chain plans.
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class AccumModel:
    """The arithmetic of one accumulation chain (see the module docstring)."""

    kind: str
    k_step: int = 8  # products per MMA k-step: 8 for tcgen05 kind::tf32
    align_extra_bits: int = 0  # grid bits below ulp32(max |addend|); align only
    rn_block: Optional[int] = None  # r01's blocked RN; rn only, used to reproduce r01

    def __post_init__(self) -> None:
        if self.kind not in ACCUM_KINDS:
            raise ValueError(
                f"unknown accumulation model {self.kind!r}; expected one of {ACCUM_KINDS}"
            )
        if self.k_step < 1:
            raise ValueError(f"k_step must be >= 1, got {self.k_step}")
        if not 0 <= self.align_extra_bits <= 25:
            raise ValueError(f"align_extra_bits must be in [0, 25], got {self.align_extra_bits}")
        if self.align_extra_bits and self.kind != "align":
            raise ValueError(f"align_extra_bits applies to the align model only, not {self.kind!r}")
        if self.rn_block is not None and (self.kind != "rn" or self.rn_block < 1):
            raise ValueError(f"rn_block must be a positive block size of the rn model, got {self}")


EXACT = AccumModel("exact")
IEEE_RN = AccumModel("rn")
RZ_K8 = AccumModel("rz", 8)
ALIGN_K8_X2 = AccumModel("align", 8, 2)  # the member calibrated on GB300 hardware
R01_ALIGN_K8_X0 = AccumModel("align", 8, 0)  # r01's member, without extra bits: a diagnostic only


@dataclass(frozen=True)
class ChainPlan:
    """The splits of one contraction: contiguous [start, stop) ranges in index order.

    The splits cover [0, length]. Each runs its own chain; an empty split (start == stop) adds +0.
    """

    length: int
    splits: Tuple[Tuple[int, int], ...]

    def __post_init__(self) -> None:
        splits = tuple((int(start), int(stop)) for start, stop in self.splits)
        object.__setattr__(self, "splits", splits)
        if self.length < 0 or not splits:
            raise ValueError(f"a plan needs a length >= 0 and at least one split, got {self}")
        position = 0
        for start, stop in splits:
            if start != position or stop < start:
                raise ValueError(f"splits {splits} are not contiguous [start, stop) ranges from 0")
            position = stop
        if position != self.length:
            raise ValueError(f"splits {splits} end at {position}, not at length {self.length}")

    @classmethod
    def equal(cls, length: int, n: int) -> "ChainPlan":
        """n equal contiguous splits; raises unless n divides length."""
        if n < 1 or length % n:
            raise ValueError(f"length {length} is not divisible into {n} equal splits")
        size = length // n
        return cls(length, tuple((i * size, (i + 1) * size) for i in range(n)))


def single_chain(length: int) -> ChainPlan:
    """One chain over the whole contraction, as GP-4's dgrad runs over N."""
    return ChainPlan(length, ((0, length),))


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


def _kernel_split_plan(length: int, n_split: int, block: int) -> ChainPlan:
    # The kernels round the split length up to a block multiple and mask loads past the end, so
    # short contractions end with empty splits.
    per_split = _cdiv(_cdiv(length, n_split), block) * block
    return ChainPlan(
        length,
        tuple(
            (min(i * per_split, length), min((i + 1) * per_split, length)) for i in range(n_split)
        ),
    )


def gp2_plan(K: int, n_split: int = 16, block_k: int = 64) -> ChainPlan:
    """GP-2's split-K plan: cdiv(cdiv(K, n_split), block_k) * block_k columns per split.

    K = 16384 gives 16 x 1,024; K = 384 gives 6 x 64 followed by 10 empty splits.
    """
    return _kernel_split_plan(K, n_split, block_k)


def gp4_wgrad_plan(T: int, n_split: int = 2, block_t: int = 64) -> ChainPlan:
    """GP-4's split-M wgrad plan over tokens: cdiv(cdiv(T, n_split), block_t) * block_t per split.

    T = 4095 gives 2,048 + 2,047; T = 17 gives 17 tokens followed by an empty split.
    """
    return _kernel_split_plan(T, n_split, block_t)


def gp2_plan_nonempty(K: int, n_split: int = 14, block_k: int = 64) -> ChainPlan:
    """The shipped GP-2's split-K plan, in which no chain is empty.

    k = cdiv(cdiv(K, n), block_k) * block_k columns per chain, n = min(n_split, cdiv(K, block_k)),
    and chain i covers [i * k, min((i + 1) * k, K)) for i < cdiv(K, k), so only the last chain can
    be shorter; GP-3 reduces one partial per chain. K = 16384 gives 13 x 1,216 + 576, K = 4096
    gives 12 x 320 + 256 and K = 384 gives 6 x 64. gp2_plan(K, 14) has the same non-empty chains
    followed by empty splits (1 at K = 4096, 8 at K = 384), which could change only the sign of an
    exactly zero p; this plan has none.
    """
    if K < 1 or n_split < 1 or block_k < 1:
        raise ValueError(f"K, n_split and block_k must be >= 1, got {K}, {n_split} and {block_k}")
    per_chain = _cdiv(_cdiv(K, min(n_split, _cdiv(K, block_k))), block_k) * block_k
    return ChainPlan(
        K, tuple((start, min(start + per_chain, K)) for start in range(0, K, per_chain))
    )


@dataclass(frozen=True)
class ColumnChainPlans:
    """Chain plans per group of output columns: one ChainPlan of the contraction for each group.

    groups holds (column ranges, ChainPlan(length, splits)) pairs, one per distinct split layout,
    in the order of their first column. The column ranges are [start, stop) ranges of output
    columns (the rows of bt in _accumulate_columns); together they cover [0, width) exactly once.
    gp4_wgrad_column_plans gives the shipped wgrad's: a group holds the 128-column tiles whose
    token pieces agree.
    """

    length: int
    groups: Tuple[Tuple[Tuple[Tuple[int, int], ...], ChainPlan], ...]

    def __post_init__(self) -> None:
        groups = tuple(
            (tuple((int(start), int(stop)) for start, stop in ranges), plan)
            for ranges, plan in self.groups
        )
        object.__setattr__(self, "groups", groups)
        if self.length < 0 or not groups:
            raise ValueError(f"column plans need a length >= 0 and at least one group, got {self}")
        for ranges, plan in groups:
            if not isinstance(plan, ChainPlan) or plan.length != self.length:
                raise ValueError(
                    f"every group needs a ChainPlan of length {self.length}, got {plan}"
                )
            if not ranges or any(stop <= start for start, stop in ranges):
                raise ValueError(f"column ranges {ranges} must be non-empty [start, stop) ranges")
        position = 0
        covered = sorted(r for ranges, _ in groups for r in ranges)
        for start, stop in covered:
            if start != position:
                raise ValueError(f"column ranges {covered} do not cover [0, n) exactly once")
            position = stop

    @property
    def width(self) -> int:
        """The number of output columns the groups cover."""
        return max(stop for ranges, _ in self.groups for _, stop in ranges)


def gp4_wgrad_column_plans(
    T: int, K: int, block_t: int = 64, block_c: int = 128, nprog_max: int = 304
) -> ColumnChainPlans:
    """The shipped GP-4 wgrad's token pieces per column tile, from its static partition of the work.

    The NU = NB * NC units u = c * NB + b (token block b of block_t tokens, column tile c of
    block_c columns; NB = cdiv(T, block_t), NC = cdiv(K, block_c)) are dealt in this column-major
    order to NPROG = min(nprog_max, NU) programs, program q owning the units from q * NU // NPROG
    to (q + 1) * NU // NPROG. A piece is one program's run inside one column tile: one chain from
    acc = 0 over its tokens. Each tile's pieces are in program order, which is token order, and
    GP-4r adds their partials in that order from +0.0. nprog_max is the kernels' constant 2 x 152;
    it is never read from a device. At T = 4096, K = 16384, NPROG = 304 gives 8 layouts (tile c
    takes layout c mod 8) of 3 or 4 pieces of 192-1,728 tokens. Tiles with the same pieces form one
    group, and adjacent ones share a column range.
    """
    sizes = (T, K, block_t, block_c, nprog_max)
    if min(sizes) < 1:
        raise ValueError(f"T, K, block_t, block_c and nprog_max must be >= 1, got {sizes}")
    n_blocks, n_tiles = _cdiv(T, block_t), _cdiv(K, block_c)
    n_units = n_blocks * n_tiles
    n_prog = min(nprog_max, n_units)
    pieces: List[List[Tuple[int, int]]] = [[] for _ in range(n_tiles)]
    for q in range(n_prog):
        unit, unit_end = q * n_units // n_prog, (q + 1) * n_units // n_prog
        while unit < unit_end:  # the program's run, cut at column tile boundaries
            tile, block = divmod(unit, n_blocks)
            block_end = min(unit_end - tile * n_blocks, n_blocks)
            pieces[tile].append((block * block_t, min(block_end * block_t, T)))
            unit = tile * n_blocks + block_end
    layouts: Dict[Tuple[Tuple[int, int], ...], List[Tuple[int, int]]] = {}
    for tile, tile_pieces in enumerate(pieces):
        ranges = layouts.setdefault(tuple(tile_pieces), [])
        start, stop = tile * block_c, min((tile + 1) * block_c, K)
        if ranges and ranges[-1][1] == start:
            ranges[-1] = (ranges[-1][0], stop)
        else:
            ranges.append((start, stop))
    return ColumnChainPlans(
        T, tuple((tuple(ranges), ChainPlan(T, layout)) for layout, ranges in layouts.items())
    )


def _chain(a_s: Tensor, b_s: Tensor, model: AccumModel) -> Tensor:
    """One row chunk. a_s: [..., Mc, S, L], b_s: [..., N, S, L] (float64). Returns [..., Mc, N]."""
    seg_len = a_s.shape[-1]
    a_full = a_s.unsqueeze(-3)  # [..., Mc, 1, S, L]
    b_full = b_s.unsqueeze(-4)  # [..., 1, N, S, L]
    shape = torch.broadcast_shapes(a_full.shape[:-1], b_full.shape[:-1])  # [..., Mc, N, S]
    device = a_s.device
    kind = model.kind
    if kind == "exact":
        total = torch.zeros(shape, dtype=torch.float64, device=device)
        comp = torch.zeros_like(total)
    else:
        acc = torch.zeros(shape, dtype=torch.float32, device=device)
        block_total = torch.zeros_like(acc) if model.rn_block else None
    for k0 in range(0, seg_len, model.k_step):
        prods = a_full[..., k0 : k0 + model.k_step] * b_full[..., k0 : k0 + model.k_step]  # exact
        if kind == "exact":
            for i in range(prods.shape[-1]):  # Neumaier, one product at a time, in index order
                x = prods[..., i]
                t = total + x
                comp = comp + torch.where(total.abs() >= x.abs(), (total - t) + x, (x - t) + total)
                total = t
        elif kind == "rn":
            for i in range(prods.shape[-1]):
                acc = (acc.to(torch.float64) + prods[..., i]).to(torch.float32)  # fl32(acc + a*b)
                if block_total is not None:
                    pos = k0 + i + 1
                    if pos % model.rn_block == 0 or pos == seg_len:  # r01: tot = fl32(tot + acc)
                        block_total = block_total + acc
                        acc = torch.zeros_like(acc)
        elif kind == "rnk":
            acc = (acc.to(torch.float64) + prods.sum(-1)).to(torch.float32)
        elif kind == "rz":
            acc = rz_to_fp32(acc.to(torch.float64) + prods.sum(-1))
        else:  # align
            acc64 = acc.to(torch.float64)
            maxabs = torch.maximum(prods.abs().amax(-1), acc64.abs())
            grid, inverse = ulp32_grid(maxabs, model.align_extra_bits, with_inverse=True)
            # trunc(v / grid) == trunc(v * inverse): both scalings by a power of two are exact
            q = torch.trunc(prods * inverse.unsqueeze(-1)).sum(-1) + torch.trunc(acc64 * inverse)
            acc = rz_to_fp32(q * grid)
    if kind == "exact":
        return (total + comp).sum(-1)
    if block_total is not None:
        acc = block_total
    return ordered_reduce(acc, dim=-1)


def _accumulate_equal_splits(
    a64: Tensor, bt64: Tensor, model: AccumModel, nsplit: int, max_chunk_elems: int
) -> Tensor:
    """nsplit equal non-empty splits as one tensor dimension, in row chunks of the outputs."""
    seg_len = a64.shape[-1] // nsplit
    m_len, n_len = a64.shape[-2], bt64.shape[-2]
    batch = torch.broadcast_shapes(a64.shape[:-2], bt64.shape[:-2])
    out_dtype = torch.float64 if model.kind == "exact" else torch.float32
    out = torch.empty(tuple(batch) + (m_len, n_len), dtype=out_dtype, device=a64.device)
    per_row = max(1, math.prod(batch) * n_len * nsplit * min(model.k_step, seg_len))
    rows = max(1, min(m_len, max_chunk_elems // per_row))
    b_s = bt64.unflatten(-1, (nsplit, seg_len))
    for m0 in range(0, m_len, rows):
        a_s = a64[..., m0 : m0 + rows, :].unflatten(-1, (nsplit, seg_len))
        out[..., m0 : m0 + rows, :] = _chain(a_s, b_s, model)
    return out


def _accumulate_split_loop(
    a64: Tensor, bt64: Tensor, model: AccumModel, plan: ChainPlan, max_chunk_elems: int
) -> Tensor:
    """One chain per split of plan, in a loop, then the reduction of the split partials."""
    batch = torch.broadcast_shapes(a64.shape[:-2], bt64.shape[:-2])
    shape = tuple(batch) + (a64.shape[-2], bt64.shape[-2])
    out_dtype = torch.float64 if model.kind == "exact" else torch.float32
    partials = []
    for start, stop in plan.splits:
        if stop > start:
            a_split, b_split = a64[..., start:stop], bt64[..., start:stop]
            partials.append(_accumulate_equal_splits(a_split, b_split, model, 1, max_chunk_elems))
        else:
            partials.append(torch.zeros(shape, dtype=out_dtype, device=a64.device))
    if model.kind == "exact":
        return torch.stack(partials, dim=-1).sum(-1)  # float64, as the vectorized form sums
    return ordered_reduce(torch.stack(partials), dim=0)


def accumulate(
    a64: Tensor,
    bt64: Tensor,
    model: AccumModel,
    plan: ChainPlan,
    *,
    max_chunk_elems: int = DEFAULT_MAX_CHUNK_ELEMS,
) -> Tensor:
    """C[..., m, n] = sum_k a64[..., m, k] * bt64[..., n, k] under model, one chain per split.

    a64 [..., M, K] and bt64 [..., N, K] are float64 operands that are already rounded (see
    round_operand); leading dims broadcast. Returns float32, or float64 for the exact model.
    Equal non-empty splits run vectorized over the splits; other plans loop over their splits.
    Zero products after a split's last real k-step leave every model unchanged, so a vectorized
    form may pad there, but nowhere else: padding inside a split moves its k-step boundaries.
    """
    if not isinstance(model, AccumModel) or not isinstance(plan, ChainPlan):
        raise TypeError("accumulate expects an AccumModel and a ChainPlan")
    if a64.dtype != torch.float64 or bt64.dtype != torch.float64:
        raise TypeError("accumulate expects float64 operands (use round_operand)")
    k_len = a64.shape[-1]
    if bt64.shape[-1] != k_len:
        raise ValueError(f"contraction mismatch {k_len} vs {bt64.shape[-1]}")
    if plan.length != k_len:
        raise ValueError(f"the plan covers {plan.length} positions; the contraction has {k_len}")
    lengths = {stop - start for start, stop in plan.splits}
    if len(lengths) == 1 and 0 not in lengths:
        return _accumulate_equal_splits(a64, bt64, model, len(plan.splits), max_chunk_elems)
    return _accumulate_split_loop(a64, bt64, model, plan, max_chunk_elems)


def _accumulate_columns(
    a64: Tensor,
    bt64: Tensor,
    model: AccumModel,
    plans: ColumnChainPlans,
    *,
    drop_last_split: bool = False,
    max_chunk_elems: int = DEFAULT_MAX_CHUNK_ELEMS,
) -> Tensor:
    """The per-column path of _contract (g_W only): accumulate with each group of output columns
    (rows of bt64) on its own ChainPlan.

    Per group: accumulate over the group's columns, one chain per split from acc = 0 and the split
    partials added in index order from +0.0 (ordered_reduce). With drop_last_split, each group's
    last split contributes zero products: a kernel that never reduces the last piece of a column
    tile. Takes and returns what accumulate takes and returns.
    """
    if not isinstance(model, AccumModel) or not isinstance(plans, ColumnChainPlans):
        raise TypeError("_accumulate_columns expects an AccumModel and a ColumnChainPlans")
    if a64.dtype != torch.float64 or bt64.dtype != torch.float64:
        raise TypeError("_accumulate_columns expects float64 operands (use round_operand)")
    n_columns = bt64.shape[-2]
    if plans.width != n_columns:
        raise ValueError(f"the plans cover {plans.width} output columns; bt has {n_columns}")
    batch = torch.broadcast_shapes(a64.shape[:-2], bt64.shape[:-2])
    out_dtype = torch.float64 if model.kind == "exact" else torch.float32
    out = torch.empty(tuple(batch) + (a64.shape[-2], n_columns), dtype=out_dtype, device=a64.device)
    for ranges, plan in plans.groups:
        columns = torch.cat(
            [torch.arange(start, stop, device=bt64.device) for start, stop in ranges]
        )
        a_group = a64
        if drop_last_split:
            start, stop = plan.splits[-1]
            a_group = a64.clone()
            a_group[..., start:stop] = 0.0
        out[..., columns] = accumulate(
            a_group, bt64.index_select(-2, columns), model, plan, max_chunk_elems=max_chunk_elems
        )
    return out


def emulate(a: Tensor, bt: Tensor, model: AccumModel, operand_mode: str, plan: ChainPlan) -> Tensor:
    """Round both operands with operand_mode, then accumulate.

    The operands of the projection pair are (xn, W) for p, (G_p, W^T) for g_xn and (G_p^T, xn^T)
    for g_W, with the contraction along the last dim.
    """
    return accumulate(round_operand(a, operand_mode), round_operand(bt, operand_mode), model, plan)


# --------------------------------------------------------------------------------------------
# The GLM projection pair.
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class GLMPairPlans:
    """The chain plans of the projection pair: proj over K, dgrad over N and wgrad over T.

    wgrad alone may be a ColumnChainPlans, whose column groups are columns of g_W.
    """

    proj: ChainPlan
    dgrad: ChainPlan
    wgrad: Union[ChainPlan, ColumnChainPlans]


def glm_pair_kernel_plans(T: int, K: int, N: int = 24) -> GLMPairPlans:
    """The P2 prototype's plans: GP-2's split-K, GP-4's single dgrad chain and split-M wgrad.

    The shipped kernels' plans are glm_pair_fused_plans.
    """
    return GLMPairPlans(proj=gp2_plan(K), dgrad=single_chain(N), wgrad=gp4_wgrad_plan(T))


def glm_pair_fused_plans(T: int, K: int, N: int = 24) -> GLMPairPlans:
    """The shipped fused kernels' plans: GP-2's non-empty split-K chains (gp2_plan_nonempty), GP-4's
    single dgrad chain over N and its wgrad pieces per column tile (gp4_wgrad_column_plans)."""
    return GLMPairPlans(
        proj=gp2_plan_nonempty(K), dgrad=single_chain(N), wgrad=gp4_wgrad_column_plans(T, K)
    )


@dataclass
class GLMPairResult:
    """Outputs of the projection pair: fp32 in the kernel model, float64 in the reference.

    g_x32 is g_x before its store: the fp32 value that the kernel rounds once to the bf16 g_x. The
    reference has no store, so its g_x32 and g_x are the same float64 tensor. g_xn (the fp32 dgrad
    product, which the g_xn bar of P9_M3_BARS applies to) and g_W are None without G_p.
    """

    s: Tensor
    xn: Tensor
    p: Tensor
    g_xn: Optional[Tensor]
    g_s: Tensor
    g_x32: Tensor
    g_x: Tensor
    g_W: Optional[Tensor]


# dgrad_unrounded_w is the arithmetic of a kernel that skips the TF32 rounding of W in the dgrad:
# Triton's TF32 MMA truncates raw fp32 operands (measured with Triton 3.6 on GB300 (sm_103)
# hardware: 12 of 12 operand slots), so its W_dgrad slot is "rz".
GLM_PAIR_MUTATIONS = (
    "eps_outside_sqrt",
    "project_first",
    "rz_for_rn",
    "bf16_p",
    "drop_radial",
    "drop_kernel_split",
    "ignore_kernel_split",
    "dgrad_unrounded_w",
)
_S_MUTATIONS = ("eps_outside_sqrt",)
_P_MUTATIONS = ("project_first", "rz_for_rn", "bf16_p", "drop_kernel_split", "ignore_kernel_split")


def pair_data(T, K, scale=1.0, seed=0, rows="randn"):
    """x randn -> bf16 [T, K], W xavier-uniform [24, K] (HyperConnectionModule's init), G_p randn.

    rows="heavy" puts 4 channels of magnitude 100-1000 (log-uniform, random sign) in every row, one
    per quarter of K; rows="zero_rows" zeroes every 8th row, where s = rsqrt(eps) ~ 316.
    """
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(T, K, generator=g) * scale
    if rows == "heavy":
        quarter = K // 4
        for row in range(T):
            columns = torch.randint(quarter, (4,), generator=g) + torch.arange(4) * quarter
            magnitudes = torch.exp(
                torch.empty(4).uniform_(math.log(100), math.log(1000), generator=g)
            )
            signs = torch.randint(2, (4,), generator=g) * 2 - 1
            x[row, columns] = magnitudes * signs
    elif rows == "zero_rows":
        x[::8] = 0.0
    x = x.to(torch.bfloat16)
    bound = math.sqrt(6.0 / (K + PAIR_N))
    W = (torch.rand(PAIR_N, K, generator=g) * 2 - 1) * bound
    G_p = torch.randn(T, PAIR_N, generator=g)
    return x, W, G_p


def _check_pair_inputs(
    x: Tensor, W: Tensor, G_p: Optional[Tensor], G_s: Optional[Tensor]
) -> Tuple[int, int, int]:
    if x.dim() != 2 or W.dim() != 2 or W.shape[1] != x.shape[1]:
        raise ValueError(
            f"expected x [T, K] and W [N, K], got {tuple(x.shape)} and {tuple(W.shape)}"
        )
    (T, K), N = x.shape, W.shape[0]
    if G_p is not None and tuple(G_p.shape) != (T, N):
        raise ValueError(f"G_p must be [{T}, {N}], got {tuple(G_p.shape)}")
    if G_s is not None and tuple(G_s.shape) != (T,):
        raise ValueError(f"G_s must be [{T}], got {tuple(G_s.shape)}")
    return T, K, N


def glm_pair_reference_f64(
    x: Tensor,
    W: Tensor,
    G_p: Optional[Tensor] = None,
    G_s: Optional[Tensor] = None,
    *,
    eps: float = EPS_FP32,
) -> GLMPairResult:
    """The GLM projection pair in float64, without intermediate rounding.

    Forward, as hyper_connection._proj_rms_impl with eps inside the sqrt and
    normalize_before_projection:
    s = (sum_k x^2 / K + eps)^(-1/2), xn = x * s and p = xn W^T. Backward for the gradients G_p
    of p and G_s of s (either may be None): g_xn = G_p W, g_s = (G_p . p) / s + G_s,
    g_x = s g_xn - (s^3 / K) g_s x and g_W = G_p^T xn.
    """
    T, K, _ = _check_pair_inputs(x, W, G_p, G_s)
    x64, W64 = x.detach().to(torch.float64), W.detach().to(torch.float64)
    s = torch.rsqrt(x64.square().sum(dim=-1) / K + eps)
    xn = x64 * s[:, None]
    p = xn @ W64.t()
    g_s = torch.zeros_like(s) if G_s is None else G_s.detach().to(torch.float64)
    g_xn = g_W = None
    if G_p is not None:
        G_p64 = G_p.detach().to(torch.float64)
        g_xn = G_p64 @ W64
        g_W = G_p64.t() @ xn
        g_s = (G_p64 * p).sum(dim=-1) / s + g_s
        g_x = s[:, None] * g_xn - ((s**3) / K * g_s)[:, None] * x64
    else:
        g_x = -((s**3) / K * g_s)[:, None] * x64
    return GLMPairResult(s=s, xn=xn, p=p, g_xn=g_xn, g_s=g_s, g_x32=g_x, g_x=g_x, g_W=g_W)


def _fp32_scalar(value: float, device: torch.device) -> Tensor:
    return torch.tensor(value, dtype=torch.float32, device=device)


def _gp1_mean_square(x: Tensor) -> Tensor:
    # ss = sum_k xf^2 in float64 (each bf16 square is exact), rounded once to fp32; then
    # fl32(ss * inv_k) with inv_k = fl32(1 / K), exact only at K = 2^n.
    ss = x.to(torch.float64).square().sum(dim=-1).to(torch.float32)
    return ss * _fp32_scalar(1.0 / x.shape[-1], x.device)


def gp1_s_model(x: Tensor, eps: float = EPS_FP32) -> Tensor:
    """A model of GP-1's fp32 RMS scale: s = fl32(1 / sqrt(fl32(fl32(ss * inv_k) + eps))).

    ss = sum_k xf^2 is the float64 sum of the exact bf16 squares, rounded once to fp32. This is a
    model, not GP-1's bits: GP-1's tl.sum order and tl.math.rsqrt cannot be emulated, and on
    GB300 (sm_103) hardware a prototype GP-1's s differed from ATen's on 25.5% of tokens by at
    most 2 ulps. Kernel tests therefore pass GP-1's s to glm_pair_kernel_model.
    """
    denom = _gp1_mean_square(x) + _fp32_scalar(eps, x.device)
    return torch.rsqrt(denom.to(torch.float64)).to(torch.float32)


def _slot_modes(operand_mode: Union[str, Dict[str, str]]) -> Dict[str, str]:
    if isinstance(operand_mode, str):
        modes = dict.fromkeys(OPERAND_SLOTS, operand_mode)
    else:
        modes = dict(operand_mode)
        if set(modes) != set(OPERAND_SLOTS):
            raise ValueError(
                f"operand modes must name the slots {OPERAND_SLOTS}, got {sorted(modes)}"
            )
    for slot, mode in modes.items():
        if mode not in OPERAND_MODES:
            raise ValueError(f"unknown operand mode {mode!r} for slot {slot}")
    return modes


def _contract(
    a: Tensor,
    bt: Tensor,
    a_mode: str,
    b_mode: str,
    model: AccumModel,
    plan: Union[ChainPlan, ColumnChainPlans],
    drop_last_split: bool = False,
) -> Tensor:
    a64 = round_operand(a.contiguous(), a_mode)
    b64 = round_operand(bt.contiguous(), b_mode)
    if isinstance(plan, ColumnChainPlans):
        # The per-column path (g_W only): each column group's own splits, its last split dropped
        # for every column tile under drop_last_split.
        return _accumulate_columns(a64, b64, model, plan, drop_last_split=drop_last_split)
    if drop_last_split:
        # A kernel that never reduces the last split's partial: zero products make that partial
        # +0, and adding +0 in the ordered reduction changes no value.
        start, stop = plan.splits[-1]
        a64[..., start:stop] = 0.0
    return accumulate(a64, b64, model, plan)


def glm_pair_kernel_model(
    x: Tensor,
    W: Tensor,
    G_p: Optional[Tensor] = None,
    G_s: Optional[Tensor] = None,
    *,
    eps: float = EPS_FP32,
    operand_mode: Union[str, Dict[str, str]] = "rne",
    model: AccumModel = ALIGN_K8_X2,
    plans: Optional[GLMPairPlans] = None,
    s: Optional[Tensor] = None,
    p: Optional[Tensor] = None,
    mutation: Optional[str] = None,
) -> GLMPairResult:
    """The fused GLM projection pair's arithmetic, one IEEE fp32 operation at a time.

    O11  s = gp1_s_model(x), unless GP-1's s is given;
    O1   xn = fl32(xf * s), one multiply (GP-2 and GP-4 alike);
    O2   T() of every MMA operand, under operand_mode: one mode, or a dict naming every slot of
         OPERAND_SLOTS;
    O3   p = the chains of T(xn), T(W) over plans.proj, unless GP-3's p is given;
    O4   the split partials added in index order with fp32 round-to-nearest (GP-3, GP-4r);
    O5   g_xn = one chain of T(G_p), T(W) over plans.dgrad: the fp32 dgrad product;
    O6   g_s = fl32(fl32(sum_j fl32(G_p,j * p_j)) / s), then fl32(g_s + G_s), with raw G_p and
         the kernel's p, summed sequentially in index order (an order chosen here; GP-4 must use
         it to be bit-exact);
    O7   coef = fl(fl(fl(fl(s * s) * s) * inv_k) * g_s), with inv_k = fl32(1 / K);
    O8   g_x32 = fl(fl(s * g_xn) - fl(coef * xf)), without FMA (enable_fp_fusion=False);
    O9   g_x = bf16_RNE(g_x32);
    O10  g_W = the chains of T(G_p)^T, T(xn)^T over plans.wgrad, then O4; a ColumnChainPlans
         wgrad runs each column group on its own splits (_accumulate_columns).

    x is bf16 [T, K], W fp32 [N, K], G_p fp32 [T, N] and G_s fp32 [T]. Without G_p (the norm-only
    case) g_xn and g_W are None and g_x comes from G_s alone. One ulp of s on one row in 16 moves p
    by 6-25 x its bar, so emulator agreement needs GP-1's s and GP-3's p as inputs. plans default to
    glm_pair_kernel_plans (P2's); the shipped kernels' are glm_pair_fused_plans. mutation, one of
    GLM_PAIR_MUTATIONS, changes one step for the power checks. On a ColumnChainPlans wgrad,
    drop_kernel_split zeroes the last piece of every column tile and ignore_kernel_split runs one
    chain over T for every column.
    """
    if x.dtype != torch.bfloat16 or W.dtype != torch.float32:
        raise TypeError(f"expected bf16 x and fp32 W, got {x.dtype} and {W.dtype}")
    T, K, N = _check_pair_inputs(x, W, G_p, G_s)
    for name, grad in (("G_p", G_p), ("G_s", G_s)):
        if grad is not None and grad.dtype != torch.float32:
            raise TypeError(f"{name} must be fp32, got {grad.dtype}")
    if mutation is not None and mutation not in GLM_PAIR_MUTATIONS:
        raise ValueError(f"unknown mutation {mutation!r}; expected one of {GLM_PAIR_MUTATIONS}")
    if s is not None and mutation in _S_MUTATIONS:
        raise ValueError(f"mutation {mutation!r} changes s, which was given")
    if p is not None and mutation in _P_MUTATIONS:
        raise ValueError(f"mutation {mutation!r} changes p, which was given")
    modes = _slot_modes(operand_mode)
    if mutation == "rz_for_rn":
        modes = dict.fromkeys(OPERAND_SLOTS, "rz")
    elif mutation == "dgrad_unrounded_w":
        modes["W_dgrad"] = "rz"
    plans = plans if plans is not None else glm_pair_kernel_plans(T, K, N)
    if isinstance(plans.proj, ColumnChainPlans) or isinstance(plans.dgrad, ColumnChainPlans):
        raise TypeError("only the wgrad plan may be a ColumnChainPlans (per column of g_W)")
    if (plans.proj.length, plans.dgrad.length, plans.wgrad.length) != (K, N, T):
        raise ValueError(f"plans {plans} do not cover K={K}, N={N} and T={T}")
    if isinstance(plans.wgrad, ColumnChainPlans) and plans.wgrad.width != K:
        raise ValueError(f"the wgrad plans cover {plans.wgrad.width} columns of g_W, not K={K}")
    if mutation == "ignore_kernel_split":
        # One chain over T for every column of g_W, whether wgrad is per column or not.
        plans = GLMPairPlans(proj=single_chain(K), dgrad=plans.dgrad, wgrad=single_chain(T))
    drop_split = mutation == "drop_kernel_split"

    xf = x.to(torch.float32)
    if s is None and mutation == "eps_outside_sqrt":
        # 1 / (sqrt(ms) + eps) from the same fp32 mean square, in float64 and rounded once to fp32
        # as gp1_s_model rounds s: only the place of eps changes, not the rounding of s.
        eps64 = _fp32_scalar(eps, x.device).to(torch.float64)
        s = (1.0 / (_gp1_mean_square(x).to(torch.float64).sqrt() + eps64)).to(torch.float32)
    elif s is None:
        s = gp1_s_model(x, eps)
    elif s.dtype != torch.float32 or tuple(s.shape) != (T,):
        raise ValueError(f"s must be fp32 [{T}], got {s.dtype} {tuple(s.shape)}")
    xn = xf * s[:, None]
    if p is None:
        if mutation == "project_first":
            p = s[:, None] * _contract(xf, W, modes["xn"], modes["W_fwd"], model, plans.proj)
        else:
            p = _contract(xn, W, modes["xn"], modes["W_fwd"], model, plans.proj, drop_split)
        if mutation == "bf16_p":
            p = p.to(torch.bfloat16).to(torch.float32)
    elif p.dtype != torch.float32 or tuple(p.shape) != (T, N):
        raise ValueError(f"p must be fp32 [{T}, {N}], got {p.dtype} {tuple(p.shape)}")

    g_xn = g_W = None
    if G_p is not None:
        g_xn = _contract(G_p, W.t(), modes["G_p_dgrad"], modes["W_dgrad"], model, plans.dgrad)
        acc = torch.zeros_like(s)
        for j in range(N):
            acc = acc + G_p[:, j] * p[:, j]
        g_s = acc / s
        if G_s is not None:
            g_s = g_s + G_s
        g_W = _contract(
            G_p.t(), xn.t(), modes["G_p_wgrad"], modes["xn_wgrad"], model, plans.wgrad, drop_split
        )
    else:
        g_s = torch.zeros_like(s) if G_s is None else G_s.clone()
    coef = s * s * s * _fp32_scalar(1.0 / K, x.device) * g_s
    product = s[:, None] * (g_xn if g_xn is not None else torch.zeros_like(xf))
    g_x32 = product if mutation == "drop_radial" else product - coef[:, None] * xf
    return GLMPairResult(
        s=s, xn=xn, p=p, g_xn=g_xn, g_s=g_s, g_x32=g_x32, g_x=g_x32.to(torch.bfloat16), g_W=g_W
    )


def glm_mappings_chain_f64(
    x: Tensor,
    W: Tensor,
    alpha: Sequence,
    bias: Tensor,
    grads: Sequence[Tensor],
    *,
    n: int = 4,
    iters: int = SINKHORN_ITERS,
    eps: float = EPS_FP32,
) -> Dict[str, Tensor]:
    """Analytic float64 backward of p -> h -> (h_pre, h_post, H_res), the
    normalize-before-projection branch of HyperConnectionModule._compute_h followed by the
    Sinkhorn.

    h = p * alpha_ + bias, with alpha_ = (alpha_pre x n, alpha_post x n, alpha_res x n^2); the
    outputs are h_pre = sigmoid(h[:, :n]) + COMPUTE_H_EPS, h_post = 2 sigmoid(h[:, n:2n]) and
    H_res = Sinkhorn(h[:, 2n:] as [n, n]). alpha holds (alpha_pre, alpha_post, alpha_res), bias is
    [n^2 + 2n] and grads holds the gradients (G_pre [T, n], G_post [T, n], G_res [T, n, n]) of the
    three outputs. The backward takes the sigmoid derivatives, the Sinkhorn VJP,
    d_alpha_g = sum G_h * p over each group, d_bias = sum_t G_h and G_p = G_h * alpha_, then the
    pair's backward (G_s = 0: that branch does not use the norm). Returns the three outputs,
    every gradient, and p and G_h (float64 [T, n^2 + 2n]): the factors of the contributions G_h * p,
    whose sums over the column groups are the three d_alpha values.
    """
    T = x.shape[0]
    alpha_pre, alpha_post, alpha_res = (
        torch.as_tensor(a, dtype=torch.float64, device=x.device).reshape(()) for a in alpha
    )
    alpha_ = torch.cat([alpha_pre.expand(n), alpha_post.expand(n), alpha_res.expand(n * n)], dim=-1)
    p = glm_pair_reference_f64(x, W, eps=eps).p
    h = p * alpha_ + bias.detach().to(torch.float64)
    sig_pre, sig_post = h[:, :n].sigmoid(), h[:, n : 2 * n].sigmoid()
    logits = h[:, 2 * n :].reshape(T, n, n)
    G_pre, G_post, G_res = (g.detach().to(torch.float64) for g in grads)
    G_h = torch.cat(
        [
            G_pre * sig_pre * (1 - sig_pre),
            G_post * 2 * sig_post * (1 - sig_post),
            sinkhorn_vjp(logits, G_res.reshape(T, n, n), iters, SINKHORN_EPS).reshape(T, n * n),
        ],
        dim=-1,
    )
    pair = glm_pair_reference_f64(x, W, G_h * alpha_, eps=eps)
    weighted = G_h * p
    return {
        "h_pre": sig_pre + COMPUTE_H_EPS,
        "h_post": sig_post * 2,
        "H_res": sinkhorn_f64(logits, iters, SINKHORN_EPS),
        "g_x": pair.g_x,
        "g_W": pair.g_W,
        "d_alpha_pre": weighted[:, :n].sum(),
        "d_alpha_post": weighted[:, n : 2 * n].sum(),
        "d_alpha_res": weighted[:, 2 * n :].sum(),
        "d_bias": G_h.sum(dim=0),
        "p": p,
        "G_h": G_h,
    }


# --------------------------------------------------------------------------------------------
# Sinkhorn.
# --------------------------------------------------------------------------------------------

SINKHORN_MUTATIONS = ("t0_like_others",)


def sinkhorn_f64(logits: Tensor, iters: int = SINKHORN_ITERS, eps: float = SINKHORN_EPS) -> Tensor:
    """hyper_connection._sinkhorn_iterations in float64 and in its order: a row softmax + eps,
    a column normalization, then iters - 1 row and column normalizations with + eps in the
    denominator."""
    M = logits.to(torch.float64).softmax(dim=-1) + eps
    M = M / (M.sum(dim=-2, keepdim=True) + eps)
    for _ in range(iters - 1):
        M = M / (M.sum(dim=-1, keepdim=True) + eps)
        M = M / (M.sum(dim=-2, keepdim=True) + eps)
    return M


def sinkhorn_vjp(
    logits: Tensor,
    grad: Tensor,
    iters: int = SINKHORN_ITERS,
    eps: float = SINKHORN_EPS,
    *,
    dtype: torch.dtype = torch.float64,
    mutation: Optional[str] = None,
) -> Tensor:
    """The analytic Sinkhorn VJP of fused_mhc_kernels._triton_sinkhorn_bwd_kernel, in dtype.

    It recomputes the forward from M_init = exp(logits - rowmax), keeping each step's M and row and
    column sums, then runs the column and row steps in reverse. The t = 0 row step is M / rs + eps
    (the softmax + eps), so its VJP is grad / rs with row_corr = sum grad * (M - eps). The result is
    multiplied by M_init. In fp32 it is an error predictor only: tl.exp2 and fp32 division lower to
    approximate instructions, so no CPU bit model of the Triton kernel exists. The mutation
    "t0_like_others" treats t = 0 like the other iterations.
    """
    if mutation is not None and mutation not in SINKHORN_MUTATIONS:
        raise ValueError(f"unknown mutation {mutation!r}; expected one of {SINKHORN_MUTATIONS}")
    t0_special = mutation is None
    logits, grad = logits.to(dtype), grad.to(dtype)
    m_init = torch.exp(logits - logits.amax(dim=-1, keepdim=True))
    M = m_init
    saved_m, row_sums, col_sums = [], [], []
    for t in range(iters):
        saved_m.append(M)
        row_sum = M.sum(dim=-1, keepdim=True)
        row_sums.append(row_sum)
        M = M / row_sum + eps if t == 0 and t0_special else M / (row_sum + eps)
        saved_m.append(M)
        col_sum = M.sum(dim=-2, keepdim=True)
        col_sums.append(col_sum)
        M = M / (col_sum + eps)
    # M is now the forward output: the right value for the VJP of the last column step.
    for t in reversed(range(iters)):
        grad = grad / (col_sums[t] + eps)
        grad = grad - (grad * M).sum(dim=-2, keepdim=True)
        M = saved_m[2 * t + 1]
        if t == 0 and t0_special:
            grad = grad / row_sums[t]
            grad = grad - (grad * (M - eps)).sum(dim=-1, keepdim=True)
        else:
            grad = grad / (row_sums[t] + eps)
            grad = grad - (grad * M).sum(dim=-1, keepdim=True)
        M = saved_m[2 * t]
    return grad * m_init
