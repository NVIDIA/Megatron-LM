# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Shared oracle for the FlashInfer sampling call sites.

Megatron samples through ``FlashInferSampling.sample_kernel`` and scores processed log-probs
through ``FlashInferSampling.log_probs_kernel``
(``megatron/core/inference/sampling/flashinfer_sampling.py``). The six sampler modules of this
package share what is here:

* ``reference_distribution``: the fp64 distribution a row must be sampled from, written from the
  sampling semantics below, not from any kernel's algorithm;
* ``stub_context``: the slice of ``DynamicInferenceContext`` the sampler reads (per-row
  parameters on the GPU, the pinned CPU metadata and the real batch-level filter flags);
* ``Batch`` / ``sample_counts``: one launch samples a batch of replicated rows; repeated
  launches are counted per row group;
* ``KernelSpy``: records which FlashInfer entry points Megatron called, so the draws of a test
  are tied to the kernel under test.

Sampling semantics (the reference). For one row with logits ``l`` over ``V`` tokens,
temperature ``T``, ``top_k`` = k and ``top_p`` = p:

1. ``tau = max(T, 1e-6)`` (``MIN_SAMPLING_TEMPERATURE``, in fp32 as Megatron clamps it);
   ``s = l / tau``.
2. Top-k: for ``0 < k < V``, ``S_k = {i : #{j : s_j > s_i} < k}``: the k largest, and every token
   tied with the k-th value. ``k <= 0`` (the disabled sentinel) or ``k >= V`` keeps every token.
3. ``q = softmax(s)`` restricted to ``S_k`` and renormalized (a token with ``l = -inf`` has
   ``q = 0``).
4. Top-p: for ``0 < p < 1``, ``S = {i in S_k : sum over j with q_j > q_i of q_j < p}``: the
   smallest set whose mass reaches p, with every token tied at the cut kept. ``p <= 0`` (the
   disabled sentinel) or ``p >= 1`` keeps ``S_k``.
5. ``pi = q`` restricted to ``S``, renormalized.

Greedy: Megatron rows with ``top_k == 1`` sample the lowest-index argmax of ``l``. That is the
contract of open PR #8875 (``fix/sampling-contract``); on main a ``top_k == 1`` row draws at
random among tied maxima (finding R08-F2), so only rows with tied maxima tell the two apart. The
raw FlashInfer kernels are top-k samplers, so for them ``k = 1`` keeps every tied maximum.

The reference runs on fp64 copies of the exact inputs (bf16 or fp32 logits, the fp32 clamped
temperature, the fp32 top-p). Megatron and FlashInfer compute in fp32, so a token whose
membership in ``S_k`` or ``S`` is decided within fp32 rounding is ambiguous; the reference
rejects such inputs (``ValueError``: a test design error, not a kernel failure) unless the test
builds exact dyadic inputs and asks for ``exact_boundaries``.

Renormalizations (``assert_renormalized``, ``assert_log_probs``): FlashInfer renormalizes a row
by multiplying with the (approximate, ``rcp.approx``) reciprocal of an fp32 sum taken in its own
order, so its output carries a common relative scale error of a few ulp per row that torch's
pairwise sum does not (H100, pdx 7849977: up to 24x torch's error at V >= 150K). The checks
therefore split a row into its scale (the mass, bounded by the fp32 summation error of the
kernel's reduction tree) and its values after that scale (bounded by the harness policy against
torch fp32 on the same GPU normalized the same way); the kept set is exact.
"""

import math
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence

import torch

from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext
from megatron.core.inference.sampling.flashinfer_sampling import FlashInferSampling
from megatron.core.inference.sampling_params import MIN_SAMPLING_TEMPERATURE
from tests.unit_tests.inference.oracles import oracle_utils as ou

try:
    import flashinfer
    import flashinfer.sampling
except ImportError:  # the package conftest fails or skips every test without FlashInfer
    flashinfer = None

DEVICE = "cuda"

# Structure of FlashInfer's sampling kernels (FlashInfer 0.7.2, include/flashinfer/sampling.cuh):
# 1024 threads per row on sm >= 80 (:78-85), each loading VEC = gcd(16 / sizeof(float), V)
# probabilities (:1495, :1550, :1579), so one block iteration covers 4096 tokens when 4 | V,
# 2048 when V = 2 mod 4 and 1024 for odd V.
SAMPLING_BLOCK_THREADS = 1024
SAMPLING_TILE = 4 * SAMPLING_BLOCK_THREADS

# Radix top-k (top_k_mask_logits, top_k_renorm_probs; include/flashinfer/topk.cuh:1700-1723 and
# :2061-2086): a row of V tokens is split over ceil(V / chunk) CTAs with chunk = (opt-in shared
# memory per block - 2080 bytes) / 4, rounded down to the vector width. H100's 227 KiB
# (232448 bytes) gives 57592 fp32 tokens per CTA: one CTA up to 57592, two up to 115184.
H100_SMEM_OPTIN_BYTES = 232448
RADIX_FIXED_SMEM_BYTES = 2080
RADIX_CHUNK_H100 = (H100_SMEM_OPTIN_BYTES - RADIX_FIXED_SMEM_BYTES) // 4

# Vocabularies of deployed models (Llama 2, Llama 3, Qwen 2, Qwen 2 padded, Gemma 3).
PRODUCTION_VOCABS = (32000, 128256, 151936, 152064, 262144)

# V-1/V/V+1 around the block iteration for each vec size (1024, 2048, 4096 and 2 x 4096), one
# token, V = 4094 (vec 2) and the production vocabularies.
SAMPLING_VOCAB_SIZES = ou.boundary_sizes(
    1024, 2048, SAMPLING_TILE, minimum=1, production=PRODUCTION_VOCABS + (4094,)
)

# Around the radix top-k CTA split on H100: the largest 128-multiples served by one and by two
# CTAs below the shared-memory limit (57472, 114944), the first two- and three-CTA sizes (57593,
# 115185), and the production sizes. Sizes whose chunk ends within a few dozen tokens of
# RADIX_CHUNK_H100 do not launch (finding O4a-N1): RADIX_SMEM_LIMIT_VOCABS.
RADIX_VOCAB_SIZES = sorted({57472, 57593, 114944, 115185, *PRODUCTION_VOCABS})

# O4a-N1: FlashInfer sizes a radix CTA's dynamic shared memory (2080 + 4 * chunk bytes) up to the
# full 227 KiB without counting the kernel's static shared memory: for top_k_renorm_probs two cub
# BlockReduce<float, 1024> scratch buffers (2 x 164 bytes) plus a 4-byte flag on multi-CTA rows,
# for the multi-CTA top_k_mask_logits the flag alone. Measured on H100 (pdx 7849977, dfw
# 21223592): the renorm fails at 0-224 bytes of headroom (all sizes below but the edge probes),
# runs at 480 (57472, 114944); the mask fails only at 0. Edge probes for the renorm's 328/332
# bytes: 57509 and 57510 (332, 328 bytes: should run), 57511 (324: should fail), 115016 (336:
# should run), 115018 (328 < 332: should fail).
RADIX_SMEM_LIMIT_VOCABS = (
    57509,
    57510,
    57511,
    57591,
    57592,
    115016,
    115018,
    115072,
    115183,
    115184,
    172672,
    230272,
    287744,
    287872,
)

# 128-multiple vocabularies whose processed log-probs (top_k_renorm_probs) cannot launch on H100
# by the boundary above (O4a-N1); the call-site case takes the first two.
RENORM_UNLAUNCHABLE_PADDED_VOCABS = (115072, 172544, 172672, 230144, 230272, 287616, 287744, 287872)

# Elementwise roundings of a processed log-prob before the shape comparison: FlashInfer's main
# path multiplies by two rcp.approx reciprocals (up to 1.5 ulp each), takes the log and the shape
# is rounded to fp32 again; torch's trusted pipeline divides once. On H100 a row whose kept logits
# are an exact arithmetic progression (the ramp) rounds every value by the same offset, which for
# the trusted implementation can be 0.1 ulp (dfw 21223592: 2.6e-8 at |log p| ~ 2-4 against the
# kernel's 1 ulp), so the floor, not the factor, has to admit the kernel's rounding count.
LOG_PROB_FLOOR_ULPS = 4.0


UNIT_ROUNDOFF = 2.0**-24  # fp32


def renorm_summation_depth(vocab: int) -> int:
    """Depth of the summation tree FlashInfer 0.7.2 sums a row of V probabilities with.

    Each of 1024 threads adds at most ``ceil(V / 1024) + 3`` terms in sequence (strided or
    4-wide loads: ``sampling.cuh`` ternary top-p, ``air_top_p.cuh:436-447``,
    ``topk.cuh:1840-1870``), the block reduction adds 10 levels, and the multi-CTA top-k one
    partial per CTA.
    """
    return math.ceil(vocab / SAMPLING_BLOCK_THREADS) + 3 + 10 + math.ceil(vocab / RADIX_CHUNK_H100)


def renorm_mass_bound(vocab: int, terms: int, renorms: int = 1) -> float:
    """Bound on ``|sum(out) - 1|`` after ``renorms`` fp32 renormalizations of ``terms`` values.

    A sum of n non-negative terms through a tree of depth d is within ``min(n - 1, d) * u`` of
    the exact sum (each term is rounded at most d times; Higham, *Accuracy and Stability of
    Numerical Algorithms*, ch. 4); the reciprocal (``rcp.approx``, at most 1 ulp) adds ``2u``
    and the products ``u``.
    """
    depth = min(max(terms - 1, 0), renorm_summation_depth(vocab))
    return renorms * (depth + 3) * UNIT_ROUNDOFF


def trusted_renorm(probs: torch.Tensor, refs: torch.Tensor) -> torch.Tensor:
    """torch fp32 on the same device: keep the reference's set, divide by torch's sum."""
    kept = torch.where(refs.to(probs.device) > 0, probs, torch.zeros_like(probs))
    return kept / kept.sum(-1, keepdim=True)


def _assert_mass(mass: torch.Tensor, bound: torch.Tensor, what: str) -> None:
    err = (mass.cpu() - 1.0).abs()
    bad = (err > bound).nonzero().reshape(-1).tolist()
    if bad:
        rows = ", ".join(
            f"row {r}: |mass - 1| {float(err[r]):.3e} > {float(bound[r]):.3e}" for r in bad[:8]
        )
        raise AssertionError(f"{what}: row mass off by more than fp32 summation allows: {rows}")


def assert_renormalized(
    out: torch.Tensor, probs: torch.Tensor, refs: torch.Tensor, *, what: str
) -> None:
    """A renormalization kernel's fp32 rows against the fp64 renormalized reference ``refs``.

    1. Kept set exact: ``out > 0`` exactly where ``refs > 0``.
    2. Scale: the row's mass is 1 within ``renorm_mass_bound`` (FlashInfer's summation order and
       reciprocal).
    3. Values after the scale: ``out / sum(out)`` against ``refs`` with
       ``ou.assert_close_to_reference`` per row (factor 2, one ulp), the trusted implementation
       (``trusted_renorm``) normalized the same way; only elementwise roundings remain.
    """
    ou.assert_exact(out > 0, refs.to(out.device) > 0, what=f"{what}: kept set")
    mass = out.double().sum(-1)
    vocab = out.shape[-1]
    terms = (refs > 0).sum(-1).tolist()
    bound = torch.tensor([renorm_mass_bound(vocab, n) for n in terms], dtype=torch.float64)
    _assert_mass(mass, bound, what)
    trusted = trusted_renorm(probs, refs).double()
    trusted = (trusted / trusted.sum(-1, keepdim=True)).float()
    shape = (out.double() / mass[:, None]).float()
    ou.assert_close_to_reference(
        shape, refs, trusted, dim=-1, what=f"{what}: values after the scale"
    )


def assert_log_probs(
    log_probs: torch.Tensor, refs: torch.Tensor, trusted_log: torch.Tensor, *, what: str
) -> None:
    """Processed log-probs ``[B, V]`` (fp32) against the fp64 distributions ``refs``.

    1. Finite exactly on each row's support.
    2. Scale: ``|logsumexp(row)|`` within three renormalizations of the vocabulary (softmax,
       top-k, top-p: ``renorm_mass_bound``) plus the rounding of the logs (``u`` times the
       row's entropy, at most ``u * ln V``).
    3. Values after the scale: ``row - logsumexp(row)`` against ``log(refs)`` per row (factor 2,
       a floor of ``LOG_PROB_FLOOR_ULPS`` ulp), ``trusted_log`` normalized the same way.
    """
    refs_dev = refs.to(log_probs.device)
    ou.assert_exact(
        torch.isfinite(log_probs), refs_dev > 0, what=f"{what}: finite exactly on the support"
    )
    vocab = log_probs.shape[-1]
    lse = torch.logsumexp(log_probs.double(), dim=-1)
    bound = renorm_mass_bound(vocab, vocab, renorms=3) + math.log(vocab) * UNIT_ROUNDOFF
    _assert_mass(lse.exp(), torch.full((log_probs.shape[0],), bound, dtype=torch.float64), what)
    shape = (log_probs.double() - lse[:, None]).float()
    trusted = trusted_log.double()
    trusted = (trusted - torch.logsumexp(trusted, dim=-1, keepdim=True)).float()
    ou.assert_close_to_reference(
        shape,
        torch.log(refs),
        trusted,
        floor_ulps=LOG_PROB_FLOOR_ULPS,
        dim=-1,
        what=f"{what}: values after the scale",
    )


def fp32(value: float) -> float:
    """The fp32 value Megatron's per-row fp32 metadata holds for ``value``."""
    return float(torch.tensor(value, dtype=torch.float32))


def clamped_temperature(temperature: float) -> float:
    """``max(T, MIN_SAMPLING_TEMPERATURE)`` in fp32, as ``FlashInferSampling`` clamps it."""
    return float(torch.tensor(temperature, dtype=torch.float32).clamp(min=MIN_SAMPLING_TEMPERATURE))


def mass_strictly_above(q: torch.Tensor) -> torch.Tensor:
    """For each token, the total mass of tokens with a strictly larger probability (fp64)."""
    values, order = torch.sort(q, descending=True, stable=True)
    before = torch.cat([values.new_zeros(1), torch.cumsum(values, 0)[:-1]])
    _, inverse, counts = torch.unique_consecutive(values, return_inverse=True, return_counts=True)
    starts = torch.cumsum(counts, 0) - counts
    above = torch.empty_like(q)
    above[order] = before[starts][inverse]
    return above


def _check_probability_cut(q: torch.Tensor, kept: torch.Tensor, rel_gap: float, what: str) -> None:
    """Raise if the lowest kept and the highest dropped probability are within ``rel_gap``.

    fp32 rounding can tie two such tokens, and the kernels keep every tie at a cut.
    """
    dropped = (q > 0) & ~kept
    if not bool(dropped.any()) or not bool(kept.any()):
        return
    low_kept = q[kept].min()
    high_dropped = q[dropped].max()
    if bool(high_dropped >= low_kept * (1 - rel_gap)):
        raise ValueError(
            f"{what}: the cut falls within fp32 rounding (lowest kept {float(low_kept)!r}, highest"
            f" dropped {float(high_dropped)!r}); move the cut or use exact_boundaries"
        )


def reference_distribution(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.0,
    *,
    greedy: bool = False,
    exact_boundaries: bool = False,
    boundary_margin: float = 1e-4,
) -> torch.Tensor:
    """The fp64 distribution ``pi`` of one row (module docstring), as a ``[V]`` tensor on CPU.

    Args:
        logits: ``[V]`` logits as the kernel sees them (exact values; upcast here).
        temperature, top_k, top_p: the row's sampling parameters (Megatron's sentinels allowed).
        greedy: apply Megatron's greedy contract (``top_k == 1`` -> lowest-index argmax).
        exact_boundaries: the inputs are exact (dyadic), so a cut exactly on a boundary is
            well defined; skip the ambiguity checks.
        boundary_margin: relative mass and value margin a cut must keep from fp32 rounding.
    """
    l = logits.detach().to("cpu", torch.float64).reshape(-1)
    if greedy and top_k == 1:
        pi = torch.zeros_like(l)
        pi[int(torch.argmax(l))] = 1.0  # torch.argmax returns the first maximal index
        return pi
    q = torch.softmax(l / clamped_temperature(temperature), 0)
    return filter_distribution(
        q, top_k, top_p, exact_boundaries=exact_boundaries, boundary_margin=boundary_margin
    )


def filter_distribution(
    probs: torch.Tensor,
    top_k: int = 0,
    top_p: float = 0.0,
    *,
    exact_boundaries: bool = False,
    boundary_margin: float = 1e-4,
) -> torch.Tensor:
    """Steps 2-5 of the module docstring on a ``[V]`` probability row (fp64 result, on CPU).

    ``probs`` need not be normalized; top-p compares masses with ``p`` times the row's total.
    Works on the exact values given (e.g. dyadic probabilities whose sums are exact), so a cut
    exactly on a cumulative boundary is decided exactly with ``exact_boundaries``.
    """
    q = probs.detach().to("cpu", torch.float64).reshape(-1)
    vocab = q.numel()
    keep = q >= 0
    if 0 < top_k < vocab:
        kth = torch.topk(q, top_k).values[-1]
        keep = q >= kth
        if not exact_boundaries:
            _check_probability_cut(q, keep, boundary_margin, f"top-k {top_k}")
        q = torch.where(keep, q, torch.zeros_like(q))
    q = q / q.sum()
    p = fp32(top_p)
    if 0.0 < p < 1.0:
        above = mass_strictly_above(q)
        kept = keep & (q > 0) & (above < p)
        if not exact_boundaries:
            # fp32 sums of fp32 probabilities carry a relative error far below boundary_margin.
            near = (q > 0) & ((above - p).abs() <= boundary_margin * above.clamp(min=p))
            if bool(near.any()):
                raise ValueError(
                    f"top-p {p!r}: {int(near.sum())} tokens have mass above them within a"
                    f" relative {boundary_margin} of the cut; move p or use exact_boundaries"
                )
            _check_probability_cut(q, kept, boundary_margin, f"top-p {p!r}")
        q = torch.where(kept, q, torch.zeros_like(q))
        q = q / q.sum()
    return q


def mid_cut_top_p(logits: torch.Tensor, temperature: float, top_k: int, last_kept: int) -> float:
    """A top-p that keeps the ``last_kept + 1`` most likely tokens of the row, with full margins.

    Returns p halfway between the mass above the ``last_kept``-th token (0-based, in descending
    order of the top-k-renormalized probabilities) and that mass plus the token's probability,
    so the cut sits as far from rounding as it can.
    """
    q = reference_distribution(logits, temperature, top_k, 0.0)
    values = torch.sort(q, descending=True).values
    before = float(values[:last_kept].sum())
    return before + 0.5 * float(values[last_kept])


def ramp_logits(
    vocab: int,
    positions: Sequence[int],
    *,
    top: float = 4.0,
    step: float = 0.25,
    background: float = -16.0,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """A ``[V]`` row: ``positions[i]`` gets ``top - step * i``, every other token ``background``.

    With the defaults every value is exact in bf16 and fp16, neighboring hot tokens differ by a
    factor ``exp(step / T)`` in probability, and the background token ``exp(-20)`` of the top
    one; at a large vocabulary the background still carries a visible tail mass.
    """
    row = torch.full((vocab,), background, dtype=torch.float64)
    values = top - step * torch.arange(len(positions), dtype=torch.float64)
    if len(set(positions)) != len(positions):
        raise ValueError("ramp positions must be distinct")
    row[torch.tensor(list(positions), dtype=torch.int64)] = values
    out = row.to(dtype)
    if not torch.equal(out.to(torch.float64), row):
        raise ValueError(f"ramp values are not exact in {dtype}")
    return out


def spread_positions(vocab: int, count: int, generator: torch.Generator) -> List[int]:
    """``count`` distinct token ids: the first and last token, the edges of each 4096-token block
    iteration, and the rest at random (seeded), in a random order."""
    fixed = [0, vocab - 1]
    for edge in range(SAMPLING_TILE, vocab, SAMPLING_TILE):
        fixed += [edge - 1, edge]
    fixed = [p for p in dict.fromkeys(fixed) if 0 <= p < vocab][: max(0, count // 2)]
    taken = set(fixed)
    rest = [p for p in torch.randperm(vocab, generator=generator).tolist() if p not in taken]
    chosen = (fixed + rest)[:count]
    order = torch.randperm(len(chosen), generator=generator).tolist()
    return [chosen[i] for i in order]


@dataclass
class RowSpec:
    """One distribution under test: a ``[V]`` logits row and its sampling parameters."""

    logits: torch.Tensor
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 0.0
    name: str = ""
    exact_boundaries: bool = False

    def reference(self, *, greedy: bool = True) -> torch.Tensor:
        """The fp64 distribution of this row (Megatron's greedy contract by default)."""
        return reference_distribution(
            self.logits,
            self.temperature,
            self.top_k,
            self.top_p,
            greedy=greedy,
            exact_boundaries=self.exact_boundaries,
        )


@dataclass
class Batch:
    """Rows of several ``RowSpec``s, each replicated, as one sampler launch sees them."""

    specs: List[RowSpec]
    logits: torch.Tensor  # [n, V] on the device
    temperature: List[float]
    top_k: List[int]
    top_p: List[float]
    groups: List[slice] = field(default_factory=list)

    @property
    def rows(self) -> int:
        return self.logits.shape[0]

    @property
    def vocab(self) -> int:
        return self.logits.shape[1]


def build_batch(
    specs: Sequence[RowSpec], replicas: int, *, dtype: Optional[torch.dtype] = None
) -> Batch:
    """Replicate each spec ``replicas`` times, in order, into one ``[n, V]`` batch on the GPU.

    FlashInfer draws each row from its own Philox subsequence (the row index), so the replicas of
    a spec are independent draws from that spec's distribution.
    """
    vocab = specs[0].logits.numel()
    dtype = dtype or specs[0].logits.dtype
    logits = torch.empty((len(specs) * replicas, vocab), dtype=dtype, device=DEVICE)
    temperature, top_k, top_p, groups = [], [], [], []
    for i, spec in enumerate(specs):
        if spec.logits.numel() != vocab:
            raise ValueError("every spec of a batch needs the same vocabulary")
        logits[i * replicas : (i + 1) * replicas] = spec.logits.to(DEVICE, dtype)
        temperature += [spec.temperature] * replicas
        top_k += [spec.top_k] * replicas
        top_p += [spec.top_p] * replicas
        groups.append(slice(i * replicas, (i + 1) * replicas))
    return Batch(list(specs), logits, temperature, top_k, top_p, groups)


def stub_context(
    temperature: Sequence[float], top_k: Sequence[int], top_p: Sequence[float]
) -> SimpleNamespace:
    """The part of ``DynamicInferenceContext`` the FlashInfer sampler reads, one entry per request.

    Per-request parameters live in pinned-CPU-like metadata (fp32 temperature, int32 top-k, fp32
    top-p, as ``DynamicInferenceRequest.get_metadata_types`` declares them) and in their GPU view;
    the batch-level filter flags come from the real ``active_sampling_filter_flags``.
    """
    metadata = {
        "temperature": torch.tensor(list(temperature), dtype=torch.float32),
        "top_k": torch.tensor(list(top_k), dtype=torch.int32),
        "top_p": torch.tensor(list(top_p), dtype=torch.float32),
    }
    context = SimpleNamespace(
        active_request_metadata=metadata,
        gpu_view=SimpleNamespace(**{key: value.to(DEVICE) for key, value in metadata.items()}),
        total_request_count=len(metadata["temperature"]),
        paused_request_count=0,
        config=SimpleNamespace(num_speculative_tokens=0),
    )
    context.active_sampling_filter_flags = lambda count=None: (
        DynamicInferenceContext.active_sampling_filter_flags(context, count)
    )
    return context


def batch_context(batch: Batch) -> SimpleNamespace:
    """``stub_context`` with one request per row of ``batch``."""
    return stub_context(batch.temperature, batch.top_k, batch.top_p)


def make_rng(seed: int = ou.DEFAULT_SEED) -> torch.Generator:
    """A seeded CUDA generator, as the controller builds ``sampling_rng``."""
    rng = torch.Generator(device=DEVICE)
    rng.manual_seed(seed)
    return rng


def make_sampler(vocab: int, rng: Optional[torch.Generator] = None) -> FlashInferSampling:
    """Megatron's FlashInfer sampler over ``vocab`` tokens with a seeded CUDA generator."""
    return FlashInferSampling(vocab, make_rng() if rng is None else rng)


def sample(sampler: FlashInferSampling, logits: torch.Tensor, n: int, context, **kwargs):
    """One ``sample_kernel`` call with the context's real batch-level dispatch flags.

    Checks the output contract of every call: ``[n]`` int64 token ids in ``[0, V)``.
    """
    no_top_k, no_top_p = context.active_sampling_filter_flags()
    tokens = sampler.sample_kernel(
        logits, n, context, no_top_k=no_top_k, no_top_p=no_top_p, **kwargs
    )
    if tokens.dtype != torch.int64 or tuple(tokens.shape) != (n,):
        raise AssertionError(
            f"sample_kernel returned {tokens.dtype} {tuple(tokens.shape)}, want int64 ({n},)"
        )
    vocab = logits.shape[-1]
    bad = (tokens < 0) | (tokens >= vocab)
    if bool(bad.any()):
        rows = bad.nonzero().reshape(-1)[:8].tolist()
        raise AssertionError(
            f"sample_kernel returned token ids outside [0, {vocab}) at rows {rows}:"
            f" {tokens[rows].tolist()}"
        )
    return tokens


def sample_counts(
    sampler: FlashInferSampling, batch: Batch, context, launches: int, **kwargs
) -> torch.Tensor:
    """Counts ``[G, V]`` (CPU int64) of the tokens drawn by each row group over ``launches``."""
    group_of_row = torch.empty(batch.rows, dtype=torch.int64, device=DEVICE)
    for g, rows in enumerate(batch.groups):
        group_of_row[rows] = g
    counts = torch.zeros(len(batch.groups) * batch.vocab, dtype=torch.int64, device=DEVICE)
    ones = torch.ones(batch.rows, dtype=torch.int64, device=DEVICE)
    for _ in range(launches):
        tokens = sample(sampler, batch.logits, batch.rows, context, **kwargs)
        counts.index_add_(0, group_of_row * batch.vocab + tokens, ones)
    return counts.view(len(batch.groups), batch.vocab).cpu()


def assert_batch_distribution(
    counts: torch.Tensor,
    specs: Sequence[RowSpec],
    *,
    greedy: bool = True,
    alpha: float = ou.DEFAULT_ALPHA,
    what: str = "sampler",
) -> None:
    """Chi-square of each spec's counts against its fp64 reference, plus exact support.

    A sample on a token with reference probability 0 fails exactly; the rest fails with
    probability ``alpha`` (split over the specs) for a correct sampler. A spec whose reference is
    a single token is checked exactly (every draw on it).
    """
    refs = torch.stack([spec.reference(greedy=greedy) for spec in specs])
    names = ", ".join(f"{i}: {spec.name}" for i, spec in enumerate(specs))
    point = (refs > 0).sum(-1) == 1
    if bool(point.any()):
        ou.assert_counts_within_binomial_bounds(
            counts[point], refs[point], alpha=alpha, what=f"{what} [rows {names}]"
        )
    if bool((~point).any()):
        ou.assert_matches_distribution(
            counts[~point], refs[~point], alpha=alpha, what=f"{what} [rows {names}]"
        )


class KernelSpy:
    """Counts Megatron's top-level calls of FlashInfer sampling entry points.

    FlashInfer's joint sampler calls its own top-p sampler and mask by module-global lookup, so
    nested calls would be counted too; the spy counts only calls made while no spied call is in
    progress, i.e. the calls Megatron itself made.
    """

    def __init__(self, monkeypatch, names: Sequence[str]):
        self.calls: Dict[str, int] = {name: 0 for name in names}
        self.args: Dict[str, list] = {name: [] for name in names}
        self._depth = 0
        for name in names:
            original = getattr(flashinfer.sampling, name)
            monkeypatch.setattr(flashinfer.sampling, name, self._wrap(name, original))

    def _wrap(self, name, original):
        def spied(*args, **kwargs):
            if self._depth == 0:
                self.calls[name] += 1
                self.args[name].append((args, kwargs))
            self._depth += 1
            try:
                return original(*args, **kwargs)
            finally:
                self._depth -= 1

        return spied

    def assert_only(self, name: str, times: int) -> None:
        """Megatron called ``name`` exactly ``times`` times and no other spied entry point."""
        expected = {key: (times if key == name else 0) for key in self.calls}
        if self.calls != expected:
            raise AssertionError(f"FlashInfer entry points called {self.calls}, want {expected}")


SAMPLER_KERNELS = (
    "sampling_from_logits",
    "top_p_sampling_from_probs",
    "top_k_sampling_from_probs",
    "top_k_top_p_sampling_from_logits",
)
