# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Pure-numpy plan generator for the packed-window varlen mock (v2).

Single source of truth for the document-stream -> fixed-window token plan,
shared by the CPU calibration simulator and the ``packed_window`` dataset
mode (design spec: ``docs/mcore/multimodal_varlen_mock_dataset_v2_spec.md``
in the toolkit repository). This module plans token layouts only: no torch,
no pixels.

Generative model: documents are drawn i.i.d. (text length from a weighted
mixture of truncated-lognormal components with document-count weights;
per-doc image density from a Gamma prior; image counts Poisson in the text
length; image sizes from weighted resolution buckets) and concatenated
into a stream. Window ``idx`` is the
stream slice ``[idx*S, (idx+1)*S)``. A single sequential walk places text
and indivisible image atoms (``vision_start`` + V merged tokens) with
explicit SPILL/FILL rules so that no atom ever crosses a window line:

- SPILL: an atom that does not fit before the next window line is deferred
  to just after the line (stable order, never dropped or resized).
- FILL-1: the gap is first filled by pulling the document's remaining text
  forward (advances only the text cursor; overtaken atoms clamp to it).
- FILL-2: if the document has no text left, explicit ``boundary_fill``
  text tokens pad to the line and are counted separately.
"""

import math
import numbers
from dataclasses import dataclass
from typing import Any

import numpy as np

_STREAM_DOC_TEXT_LEN = 10
_STREAM_DOC_COMPONENT = 11
_STREAM_DOC_MODALITY = 12
_STREAM_DOC_LAMBDA = 13
_STREAM_DOC_COUNT = 14
_STREAM_DOC_SIZES = 15
_STREAM_DOC_OFFSETS = 16

# Acklam's rational approximation of the standard normal inverse CDF
# (|relative error| < 1.15e-9) — keeps the kernel numpy/math-pure.
_ICDF_A = (
    -3.969683028665376e01,
    2.209460984245205e02,
    -2.759285104469687e02,
    1.383577518672690e02,
    -3.066479806614716e01,
    2.506628277459239e00,
)
_ICDF_B = (
    -5.447609879822406e01,
    1.615858368580409e02,
    -1.556989798598866e02,
    6.680131188771972e01,
    -1.328068155288572e01,
)
_ICDF_C = (
    -7.784894002430293e-03,
    -3.223964580411365e-01,
    -2.400758277161838e00,
    -2.549732539343734e00,
    4.374664141464968e00,
    2.938163982698783e00,
)
_ICDF_D = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00, 3.754408661907416e00)


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _normal_icdf(p: float) -> float:
    p = min(max(p, 1e-300), 1.0 - 1e-16)
    if p < 0.02425:
        q = math.sqrt(-2.0 * math.log(p))
        a, b, c, d, e, f = _ICDF_C
        return (((((a * q + b) * q + c) * q + d) * q + e) * q + f) / (
            (((_ICDF_D[0] * q + _ICDF_D[1]) * q + _ICDF_D[2]) * q + _ICDF_D[3]) * q + 1.0
        )
    if p > 1.0 - 0.02425:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        a, b, c, d, e, f = _ICDF_C
        return -(((((a * q + b) * q + c) * q + d) * q + e) * q + f) / (
            (((_ICDF_D[0] * q + _ICDF_D[1]) * q + _ICDF_D[2]) * q + _ICDF_D[3]) * q + 1.0
        )
    q = p - 0.5
    r = q * q
    a0, a1, a2, a3, a4, a5 = _ICDF_A
    return (
        (((((a0 * r + a1) * r + a2) * r + a3) * r + a4) * r + a5)
        * q
        / (
            ((((_ICDF_B[0] * r + _ICDF_B[1]) * r + _ICDF_B[2]) * r + _ICDF_B[3]) * r + _ICDF_B[4])
            * r
            + 1.0
        )
    )


class _TruncatedLognormal:
    """Truncated lognormal parameterized by its post-truncation mean.

    Mirrors the mean semantics and far-tail underflow guards of
    ``mock_varlen._SequenceLengthSampler`` without the torch dependency.
    """

    def __init__(self, *, mean: float, sigma: float, minimum: int, maximum: int) -> None:
        if not (0 < minimum <= maximum):
            raise ValueError(f"Invalid truncation window [{minimum}, {maximum}].")
        if not minimum <= mean <= maximum:
            raise ValueError(f"mean={mean} must lie in [{minimum}, {maximum}].")
        if not math.isfinite(sigma) or sigma < 0:
            raise ValueError(f"sigma must be finite and non-negative, got {sigma}.")
        self.minimum = int(minimum)
        self.maximum = int(maximum)
        self.mean = float(mean)
        self.sigma = float(sigma)
        self._constant: int | None = None
        if sigma == 0 or self.minimum == self.maximum:
            # Degenerate component: all mass at the (validated in-range) mean.
            self._constant = int(round(self.mean))
            return
        self._log_min = math.log(self.minimum)
        self._log_max = math.log(self.maximum)
        self.mu = self._solve_mu()
        self._cdf_lo = _normal_cdf((self._log_min - self.mu) / self.sigma)
        self._cdf_hi = _normal_cdf((self._log_max - self.mu) / self.sigma)
        if self._cdf_hi - self._cdf_lo <= 1e-15:
            raise RuntimeError(
                f"Degenerate truncated-lognormal solve (mu={self.mu:.4f}) for "
                f"[{self.minimum}, {self.maximum}] mean={self.mean} sigma={self.sigma}."
            )

    def _truncated_mean(self, mu: float) -> float:
        sigma = self.sigma
        alpha = (self._log_min - mu) / sigma
        beta = (self._log_max - mu) / sigma
        denominator = _normal_cdf(beta) - _normal_cdf(alpha)
        if denominator <= 0.0:
            return float(self.minimum if mu < self._log_min else self.maximum)
        numerator = _normal_cdf(beta - sigma) - _normal_cdf(alpha - sigma)
        if numerator <= 0.0:
            # math.erf saturates ~8 sigmas out; preserve bracket direction.
            return float(self.minimum if alpha - sigma > 0 else self.maximum)
        return math.exp(mu + sigma * sigma / 2.0) * numerator / denominator

    def _solve_mu(self) -> float:
        lo = self._log_min - 40.0
        hi = self._log_max + 40.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if self._truncated_mean(mid) < self.mean:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def sample(self, rng: np.random.Generator) -> int:
        if self._constant is not None:
            return self._constant
        u = self._cdf_lo + (self._cdf_hi - self._cdf_lo) * float(rng.random())
        value = int(round(math.exp(self.mu + self.sigma * _normal_icdf(u))))
        return min(max(value, self.minimum), self.maximum)


def _zero_truncated_poisson(rng: np.random.Generator, lam: float) -> int:
    """Sample Poisson(lam) conditioned on being >= 1 via the inverse CDF."""
    if lam <= 0:
        return 1
    log_p0 = -lam
    p0 = math.exp(log_p0)
    if p0 >= 1.0:
        return 1
    u = p0 + (1.0 - p0) * float(rng.random())
    k = 0
    log_pmf = log_p0
    cdf = p0
    while cdf < u and k < 10_000_000:
        k += 1
        log_pmf += math.log(lam) - math.log(k)
        cdf += math.exp(log_pmf)
    return max(k, 1)


def _require_number(value: Any, *, what: str, minimum: float | None = None) -> float:
    if not isinstance(value, numbers.Real) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f"{what} must be a finite number, got {value!r}.")
    value = float(value)
    if minimum is not None and value < minimum:
        raise ValueError(f"{what} must be >= {minimum}, got {value}.")
    return value


@dataclass(frozen=True)
class AtomPlan:
    """One placed image atom (vision_start + merged tokens)."""

    window: int
    offset: int  # window-local start of the vision_start token
    merged_tokens: int
    raw_patches: int
    bucket_index: int
    doc_id: int
    index_in_doc: int


@dataclass(frozen=True)
class WindowPlan:
    """Layout of one S-token window."""

    segments: tuple[tuple[int, int], ...]  # (doc_id, logical length), stream order
    atoms: tuple[AtomPlan, ...]
    fill_tokens: int


class PackedWindowPlanGenerator:
    """Deterministic document-stream -> window token-plan generator.

    Args:
        seq_length: window size S.
        num_windows: number of windows to plan.
        seed: global RNG namespace seed.
        config: dict with ``doc_length.components`` (list of named
            components, each with a document-count ``weight`` and a
            truncated lognormal parameterized by post-truncation ``mean``,
            ``sigma`` — 0 means a constant-length component — ``min`` and
            ``max``), ``text_only_document_probability``,
            ``image_poisson_rate_per_1k_text_tokens`` (the LATENT Poisson rate
            of interleaved documents; the realized zero-truncated density is
            slightly higher — lambda/(1-e^-lambda) per document, about +8.5%
            at the parity profile — and short documents floor at one image,
            so window-level densities are calibrated and reported by the
            simulator, never read off this knob), and the optional
            ``image_density_gamma_shape`` (default 1.0 = exponential
            mixing; calibrated recipes set it explicitly).
        bucket_merged_tokens / bucket_raw_patches / bucket_weights: image
            size support (merged tokens and raw patch rows per bucket) and
            categorical weights.
    """

    def __init__(
        self,
        *,
        seq_length: int,
        num_windows: int,
        seed: int,
        config: dict[str, Any],
        bucket_merged_tokens: list[int],
        bucket_raw_patches: list[int],
        bucket_weights: list[float],
    ) -> None:
        if seq_length <= 0 or num_windows <= 0:
            raise ValueError("seq_length and num_windows must be positive.")
        if not (len(bucket_merged_tokens) == len(bucket_raw_patches) == len(bucket_weights) > 0):
            raise ValueError("Bucket arrays must be non-empty and of equal length.")
        largest_atom = 1 + max(bucket_merged_tokens)
        if largest_atom > seq_length:
            raise ValueError(
                f"Largest image atom ({largest_atom} tokens) exceeds the window size "
                f"{seq_length}; the generator does not rely on recipe-level assumptions."
            )

        components = (config.get("doc_length") or {}).get("components")
        if not isinstance(components, (list, tuple)) or not components:
            raise ValueError("doc_length.components must be a non-empty list.")
        names: list[str] = []
        samplers: list[_TruncatedLognormal] = []
        weights: list[float] = []
        for index, component in enumerate(components):
            name = component.get("name")
            if not isinstance(name, str) or not name:
                raise ValueError(f"doc_length component {index} needs a non-empty 'name'.")
            if name in names:
                raise ValueError(f"Duplicate doc_length component name {name!r}.")
            names.append(name)
            weight = _require_number(component["weight"], what=f"{name}.weight", minimum=0.0)
            weights.append(weight)
            minimum, maximum = int(component["min"]), int(component["max"])
            if minimum <= 0 or maximum < minimum:
                raise ValueError(
                    f"{name}: min/max must be positive integers with min <= max; "
                    f"got [{minimum}, {maximum}]."
                )
            mean = _require_number(component["mean"], what=f"{name}.mean")
            if not minimum <= mean <= maximum:
                raise ValueError(f"{name}: mean={mean} must lie in [{minimum}, {maximum}].")
            samplers.append(
                _TruncatedLognormal(
                    mean=mean,
                    sigma=_require_number(component["sigma"], what=f"{name}.sigma"),
                    minimum=minimum,
                    maximum=maximum,
                )
            )
        weight_sum = math.fsum(weights)
        if weight_sum <= 0:
            raise ValueError("doc_length component weights must have a positive sum.")
        self.component_names = tuple(names)
        self._component_samplers = tuple(samplers)
        # Document-count mixture: weights are per-document proportions
        # (what corpus descriptions state), normalized internally.
        self._component_cdf = np.cumsum(np.asarray(weights, dtype=np.float64) / weight_sum)

        self.p_text = _require_number(
            config["text_only_document_probability"],
            what="text_only_document_probability",
            minimum=0.0,
        )
        if self.p_text > 1.0:
            raise ValueError("text_only_document_probability must be in [0, 1].")
        self.density_mean = (
            _require_number(
                config["image_poisson_rate_per_1k_text_tokens"],
                what="image_poisson_rate_per_1k_text_tokens",
            )
            / 1000.0
        )
        if self.density_mean <= 0:
            raise ValueError("image_poisson_rate_per_1k_text_tokens must be positive.")
        self.gamma_shape = _require_number(
            config.get("image_density_gamma_shape", 1.0), what="image_density_gamma_shape"
        )
        if self.gamma_shape <= 0:
            raise ValueError("image_density_gamma_shape must be positive.")
        ceiling = config.get("max_boundary_fill_fraction", 0.005)
        self.max_boundary_fill_fraction = (
            None
            if ceiling is None
            else _require_number(ceiling, what="max_boundary_fill_fraction", minimum=0.0)
        )

        self.seq_length = int(seq_length)
        self.num_windows = int(num_windows)
        self.seed = int(seed)
        self.bucket_merged = tuple(int(v) for v in bucket_merged_tokens)
        self.bucket_raw = tuple(int(v) for v in bucket_raw_patches)
        weights = np.asarray(bucket_weights, dtype=np.float64)
        if not np.all(np.isfinite(weights)) or np.any(weights < 0) or weights.sum() <= 0:
            raise ValueError("Bucket weights must be finite and non-negative with a positive sum.")
        self.bucket_probs = weights / weights.sum()

        self.total_fill_tokens = 0
        self.total_spilled_atoms = 0
        self.total_atoms = 0
        self.total_docs = 0
        self.total_image_free_docs = 0
        self.total_text_tokens = 0
        self.doc_text_lengths: list[int] = []
        self._windows: list[WindowPlan] = []
        self._build()

    # ------------------------------------------------------------------
    # Per-document draws
    # ------------------------------------------------------------------

    def _rng(self, doc_id: int, stream: int) -> np.random.Generator:
        return np.random.default_rng(np.random.SeedSequence([self.seed, int(doc_id), int(stream)]))

    def _draw_doc(self, doc_id: int):
        component_rng = self._rng(doc_id, _STREAM_DOC_COMPONENT)
        component = int(np.searchsorted(self._component_cdf, component_rng.random()))
        sampler = self._component_samplers[min(component, len(self._component_samplers) - 1)]
        text_len = sampler.sample(self._rng(doc_id, _STREAM_DOC_TEXT_LEN))

        if self._rng(doc_id, _STREAM_DOC_MODALITY).random() < self.p_text:
            return text_len, (), ()

        lam = self._rng(doc_id, _STREAM_DOC_LAMBDA).gamma(
            self.gamma_shape, self.density_mean / self.gamma_shape
        )
        # Zero-truncated Poisson: an interleaved document always carries at
        # least one image, so text_only_document_probability is the EXACT
        # text-only document probability (not a lower bound) and the two
        # image knobs are directly settable user semantics.
        count = _zero_truncated_poisson(self._rng(doc_id, _STREAM_DOC_COUNT), lam * text_len)
        buckets = self._rng(doc_id, _STREAM_DOC_SIZES).choice(
            len(self.bucket_probs), size=count, p=self.bucket_probs
        )
        offsets = np.sort(
            self._rng(doc_id, _STREAM_DOC_OFFSETS).integers(0, text_len + 1, size=count),
            kind="stable",
        )
        return text_len, tuple(offsets.tolist()), tuple(int(b) for b in buckets)

    # ------------------------------------------------------------------
    # The sequential walk
    # ------------------------------------------------------------------

    def _build(self) -> None:
        S = self.seq_length
        total = self.num_windows * S
        p = 0
        doc_id = 0

        segments: list[list[list[int]]] = [[] for _ in range(self.num_windows)]
        atoms: list[list[AtomPlan]] = [[] for _ in range(self.num_windows)]
        fill: list[int] = [0] * self.num_windows

        def emit_span(doc: int, n: int, is_fill: bool) -> None:
            nonlocal p
            while n > 0:
                window = p // S
                room = S - (p % S)
                chunk = min(n, room)
                if window < self.num_windows:
                    rows = segments[window]
                    if rows and rows[-1][0] == doc:
                        rows[-1][1] += chunk
                    else:
                        rows.append([doc, chunk])
                    if is_fill:
                        fill[window] += chunk
                p += chunk
                n -= chunk

        def emit_atom(doc: int, index_in_doc: int, bucket: int) -> None:
            nonlocal p
            merged = self.bucket_merged[bucket]
            size = 1 + merged
            window = p // S
            offset = p % S
            assert offset + size <= S, "atom must never cross a window line"
            if window < self.num_windows:
                rows = segments[window]
                if rows and rows[-1][0] == doc:
                    rows[-1][1] += size
                else:
                    rows.append([doc, size])
                atoms[window].append(
                    AtomPlan(
                        window=window,
                        offset=offset,
                        merged_tokens=merged,
                        raw_patches=self.bucket_raw[bucket],
                        bucket_index=bucket,
                        doc_id=doc,
                        index_in_doc=index_in_doc,
                    )
                )
            p += size

        while p < total:
            text_len, offsets, buckets = self._draw_doc(doc_id)
            self.total_docs += 1
            self.total_image_free_docs += not offsets
            self.total_text_tokens += text_len
            self.doc_text_lengths.append(text_len)
            self.total_atoms += len(offsets)
            tc = 0  # text emitted so far (FILL-1 advances only this cursor)
            ai = 0
            while (tc < text_len or ai < len(offsets)) and p < total:
                if ai < len(offsets) and offsets[ai] <= tc:
                    size = 1 + self.bucket_merged[buckets[ai]]
                    room = S - (p % S)
                    if size > room:
                        # SPILL: defer the atom past the line; FILL-1 pulls
                        # the doc's remaining text forward, FILL-2 pads with
                        # explicit boundary_fill tokens.
                        self.total_spilled_atoms += 1
                        take = min(room, text_len - tc)
                        if take:
                            emit_span(doc_id, take, is_fill=False)
                            tc += take
                        gap = room - take
                        if gap:
                            emit_span(doc_id, gap, is_fill=True)
                            self.total_fill_tokens += gap
                    emit_atom(doc_id, ai, buckets[ai])
                    ai += 1
                else:
                    boundary = offsets[ai] if ai < len(offsets) else text_len
                    step = min(boundary, text_len) - tc
                    emit_span(doc_id, step, is_fill=False)
                    tc += step
            doc_id += 1

        self._windows = [
            WindowPlan(
                segments=tuple((doc, length) for doc, length in segments[w]),
                atoms=tuple(atoms[w]),
                fill_tokens=fill[w],
            )
            for w in range(self.num_windows)
        ]
        # boundary_fill tokens are ordinary text that participates in loss;
        # the ceiling bounds construction distortion, and exceeding it means
        # the atom sizes are too large for the window size to be meaningful.
        if (
            self.max_boundary_fill_fraction is not None
            and self.boundary_fill_fraction > self.max_boundary_fill_fraction
        ):
            raise RuntimeError(
                f"boundary_fill fraction {self.boundary_fill_fraction:.4f} exceeds the "
                f"acceptance ceiling {self.max_boundary_fill_fraction}; the configured "
                "image atoms are too large relative to seq_length."
            )

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.num_windows

    def window(self, idx: int) -> WindowPlan:
        """Layout plan for window *idx* (invariant: segment lengths sum to S)."""
        return self._windows[int(idx)]

    @property
    def boundary_fill_fraction(self) -> float:
        """Fraction of all planned tokens that are boundary_fill."""
        return self.total_fill_tokens / float(self.num_windows * self.seq_length)

    @property
    def atom_spill_fraction(self) -> float:
        """Fraction of atoms deferred across a window line."""
        return self.total_spilled_atoms / float(self.total_atoms) if self.total_atoms else 0.0
