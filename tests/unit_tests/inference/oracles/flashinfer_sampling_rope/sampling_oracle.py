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
"""

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

# Around the radix top-k CTA split (one, two, three CTAs per row on H100) and production sizes.
RADIX_VOCAB_SIZES = ou.boundary_sizes(
    RADIX_CHUNK_H100, degenerate=False, production=PRODUCTION_VOCABS
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
