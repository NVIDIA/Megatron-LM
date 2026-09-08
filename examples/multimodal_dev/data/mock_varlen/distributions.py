# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Generator-agnostic numeric helpers for the varlen mock plan kernels.

Pure numpy/math building blocks shared by plan generators and their
consumers: a truncated lognormal parameterized by its post-truncation
mean, strict scalar validators (bools and silently-coercible strings are
configuration errors), a strictly validated integer categorical, and the
deterministic seed-stream RNG pattern (``np.random.SeedSequence`` keyed
by integer stream ids). No torch, no pixels, no distribution defaults.
"""

import math
import numbers
import statistics
from typing import Any

import numpy as np

# The standard normal inverse CDF comes from the stdlib (Wichura's AS241,
# accurate to ~1e-15). numpy has no ppf and scipy is not a dependency, so
# this is the only ready-made implementation available to a numpy/math-pure
# module; sampled layouts are reproducible for a given interpreter.
_STANDARD_NORMAL = statistics.NormalDist()

# inv_cdf's domain is the OPEN unit interval; the affine map into [lo, hi]
# can land on either endpoint by rounding.
_P_MIN = math.nextafter(0.0, 1.0)
_P_MAX = math.nextafter(1.0, 0.0)
# Representable doubles a sampling interval must span for the direct-CDF
# branch to resolve it; below this the draw quantizes visibly (~700 doubles
# gave ~1k distinct lengths out of 100k). The context-scaled defaults clear
# it by five orders of magnitude, which is what keeps their layouts fixed.
_MIN_RESOLVED_STEPS = 2.0**32


def _resolved_steps(lo: float, hi: float) -> float:
    """Representable doubles the interval [lo, hi] spans.

    An affine draw into the interval can only produce this many distinct
    values, so it bounds the realized support of the whole distribution.
    """
    return (hi - lo) / max(math.ulp(lo), math.ulp(hi))


def _normal_cdf(x: float) -> float:
    # erfc stays nonzero in the left tail down to the denormal range
    # (~ -37 sigma); the 1 + erf form saturates to exactly 0/1 around
    # 8 sigma and destroys tail differences.
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def _normal_cdf_diff(a: float, b: float) -> float:
    """Stable Phi(b) - Phi(a) for a <= b.

    Same-side tails subtract two erfc values instead of two quantities
    that both round to 0 or 1. This avoids the round-to-0/1 saturation,
    but subtracting two NEARBY erfc values can still lose relative
    precision; the solve-independent Simpson quadrature verifier is the
    actual correctness backstop.
    """
    sqrt2 = math.sqrt(2.0)
    if b <= 0.0:
        return 0.5 * (math.erfc(-b / sqrt2) - math.erfc(-a / sqrt2))
    if a >= 0.0:
        return 0.5 * (math.erfc(a / sqrt2) - math.erfc(b / sqrt2))
    return 0.5 * (math.erf(b / sqrt2) - math.erf(a / sqrt2))


class TruncatedLognormal:
    """Truncated lognormal parameterized by its post-truncation mean.

    Construction fails loudly (RuntimeError) when the configured mean is
    outside the implementation's stable numerical range, so every sampler
    that constructs realizes its configured CONTINUOUS post-truncation
    mean (the solver's contract is defined before integer
    discretization). Samples are integer
    token counts: for means of only a few tokens the rounding itself adds
    visible relative bias (a 1.7-token mean realizes ~1.9) — realistic
    document means are far above that regime.
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
            if self.mean != int(self.mean):
                raise ValueError(
                    f"sigma=0 (or min==max) makes the component constant, so mean "
                    f"must be an integer token count; got {self.mean} (silently "
                    "rounding would break realized-mean == configured-mean)."
                )
            self._constant = int(self.mean)
            return
        self._log_min = math.log(self.minimum)
        self._log_max = math.log(self.maximum)
        self.mu = self._solve_mu()
        alpha = (self._log_min - self.mu) / self.sigma
        beta = (self._log_max - self.mu) / self.sigma
        # Gate on the STABLE mass: the direct CDF difference below can lose every
        # significant digit in a tail, so it must not decide samplability.
        mass = _normal_cdf_diff(alpha, beta)
        if not math.isfinite(mass) or mass <= 0.0:
            raise self._out_of_range(f"the truncation-window mass is {mass}")
        # An upper-tail window saturates both CDF values toward 1.0, leaving
        # [cdf_lo, cdf_hi) too coarse to resolve — and letting the drawn p round
        # to exactly 1.0, which inv_cdf rejects. Sample the survival function
        # Q(x) = Phi(-x) there instead; z = -inv_cdf(q) undoes the negation.
        # Branching on MEASURED resolution rather than on the sign of alpha+beta
        # keeps every well-conditioned window (all context-scaled defaults among
        # them) on the original branch, bit for bit.
        self._cdf_lo = _normal_cdf(alpha)
        self._cdf_hi = _normal_cdf(beta)
        self._survival = _resolved_steps(self._cdf_lo, self._cdf_hi) < _MIN_RESOLVED_STEPS
        if self._survival:
            self._p_lo, self._p_hi = _normal_cdf(-beta), _normal_cdf(-alpha)
        else:
            self._p_lo, self._p_hi = self._cdf_lo, self._cdf_hi
        # The SAME floor on the interval actually drawn from: switching branches
        # only helps if the survival form is itself resolvable. Deep tails can
        # leave it spanning a handful of denormals, which no longer crashes but
        # still quantizes the realized support and mean.
        if _resolved_steps(self._p_lo, self._p_hi) < _MIN_RESOLVED_STEPS:
            raise self._out_of_range("the sampling interval resolves too few values")
        # The mean-equals-configured contract must fail loudly instead of
        # silently drifting. The verifier is INDEPENDENT of the solve
        # formula (positive-integrand quadrature vs CDF differences), so
        # a numerically wrong solve cannot agree with its own error.
        achieved = self._quadrature_mean(self.mu)
        if abs(achieved - self.mean) > 5e-3 * self.mean:
            raise self._out_of_range(f"the solve realizes {achieved:.1f}")

    def _out_of_range(self, detail: str) -> RuntimeError:
        """The one stable-range rejection message; *detail* names the gate."""
        return RuntimeError(
            f"Configured post-truncation mean={self.mean} in "
            f"[{self.minimum}, {self.maximum}] with sigma={self.sigma} is outside this "
            f"implementation's stable numerical range: at mu={self.mu:.2f} {detail}. "
            "Move the mean away from the limiting boundary, raise sigma, or widen "
            "the window."
        )

    def _truncated_mean(self, mu: float) -> float:
        sigma = self.sigma
        alpha = (self._log_min - mu) / sigma
        beta = (self._log_max - mu) / sigma
        denominator = _normal_cdf_diff(alpha, beta)
        if denominator <= 0.0:
            return float(self.minimum if mu < self._log_min else self.maximum)
        numerator = _normal_cdf_diff(alpha - sigma, beta - sigma)
        if numerator <= 0.0:
            # erfc underflows ~37 sigmas out; preserve bracket direction.
            return float(self.minimum if alpha - sigma > 0 else self.maximum)
        return math.exp(mu + sigma * sigma / 2.0) * numerator / denominator

    def _quadrature_mean(self, mu: float) -> float:
        """Independent verifier: Simpson in log-space over the window.

        The integrands are strictly positive (no cancellation anywhere),
        so this cross-checks the CDF-difference solve with unrelated
        arithmetic; the max-exponent shift keeps weights representable
        for any mu.
        """
        # Simpson's rule needs an odd number of sample points.
        points = 4097
        ys = np.linspace(self._log_min, self._log_max, points)
        z = (ys - mu) / self.sigma
        log_w = -0.5 * z * z
        w = np.exp(log_w - np.max(log_w))
        simpson = np.ones(points)
        simpson[1:-1:2] = 4.0
        simpson[2:-1:2] = 2.0
        return float(np.sum(simpson * w * np.exp(ys)) / np.sum(simpson * w))

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
        p = self._p_lo + (self._p_hi - self._p_lo) * float(rng.random())
        # rng.random() can return 0.0 and the map can round up to 1.0.
        z = _STANDARD_NORMAL.inv_cdf(min(max(p, _P_MIN), _P_MAX))
        if self._survival:
            z = -z
        value = int(round(math.exp(z * self.sigma + self.mu)))
        return min(max(value, self.minimum), self.maximum)


def require_integer(value: Any, *, what: str) -> int:
    """Strictly integral config field: bools, floats, and strings that
    int(...) would silently coerce are configuration errors."""
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{what} must be an integer, got {value!r}.")
    return int(value)


def require_number(value: Any, *, what: str, minimum: float | None = None) -> float:
    if not isinstance(value, numbers.Real) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f"{what} must be a finite number, got {value!r}.")
    value = float(value)
    if minimum is not None and value < minimum:
        raise ValueError(f"{what} must be >= {minimum}, got {value}.")
    return value


def require_exact_dict(spec: Any, allowed: set[str], *, what: str) -> dict:
    """A config mapping whose key set must be exactly *allowed*.

    Unknown and missing keys are separate errors so the message names the
    actual mistake; ``what`` carries the config path (e.g.
    ``components[3].length``) so callers need no extra context.
    """
    if not isinstance(spec, dict):
        raise ValueError(f"{what} must be a dict with {', '.join(sorted(allowed))}.")
    unknown = set(spec) - allowed
    if unknown:
        raise ValueError(
            f"{what} has unknown key(s) {sorted(unknown)}; allowed: {sorted(allowed)}."
        )
    missing = allowed - set(spec)
    if missing:
        raise ValueError(f"{what} is missing required key(s) {sorted(missing)}.")
    return spec


class Categorical:
    """Strictly validated integer categorical (``counts`` + ``weights``)."""

    def __init__(self, spec: Any, *, what: str) -> None:
        spec = require_exact_dict(spec, {"counts", "weights"}, what=what)
        counts = spec.get("counts")
        weights = spec.get("weights")
        if not isinstance(counts, (list, tuple)) or not counts:
            raise ValueError(f"{what}.counts must be a non-empty list.")
        if not isinstance(weights, (list, tuple)) or len(weights) != len(counts):
            raise ValueError(f"{what}.weights must be a list matching counts in length.")
        self.counts = tuple(
            require_integer(count, what=f"{what}.counts[{index}]")
            for index, count in enumerate(counts)
        )
        for index, count in enumerate(self.counts):
            if count < 0:
                raise ValueError(f"{what}.counts[{index}] must be >= 0, got {count}.")
        if len(set(self.counts)) != len(self.counts):
            # A duplicated count is a likely config typo; the effective
            # weight would silently be the sum of the duplicate rows.
            raise ValueError(f"{what}.counts contains duplicate values: {list(self.counts)}.")
        parsed = [
            require_number(weight, what=f"{what}.weights[{index}]", minimum=0.0)
            for index, weight in enumerate(weights)
        ]
        try:
            total = math.fsum(parsed)
        except OverflowError:
            # Individually finite weights can still overflow the sum; a
            # non-finite normalizer would corrupt every draw.
            total = math.inf
        if not math.isfinite(total) or total <= 0:
            raise ValueError(f"{what}.weights must have a positive finite sum.")
        self.weights = tuple(parsed)


def draw_from_cdf(rng: np.random.Generator, cdf: np.ndarray) -> int:
    """Draw an index from a cumulative distribution.

    side="right" so zero-weight entries (CDF plateaus) are never drawn.
    ``Generator.random()`` is in [0, 1) and can never return 1.0; the
    clamp instead covers cdf[-1] < 1.0 from normalization rounding,
    where u in [cdf[-1], 1) would make searchsorted return len(cdf).
    The weighted-draw idiom shared by the component and image-count
    draws (bucket geometry uses ``rng.choice(p=...)``).
    """
    index = int(np.searchsorted(cdf, float(rng.random()), side="right"))
    return min(index, len(cdf) - 1)


def seed_stream_rng(seed: int, doc_id: int, stream: int) -> np.random.Generator:
    """Deterministic per-document, per-stream RNG namespace.

    ``np.random.SeedSequence([seed, doc_id, stream])`` makes every draw a
    pure function of its keys, independent of access order.
    """
    return np.random.default_rng(np.random.SeedSequence([int(seed), int(doc_id), int(stream)]))
