# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""CPU-only tests for the generator-agnostic varlen plan helpers."""

import numpy as np
import pytest

from examples.multimodal_dev.data.mock_varlen.distributions import (
    Categorical,
    TruncatedLognormal,
    require_integer,
    require_number,
    seed_stream_rng,
)


class TestTruncatedLognormal:
    def test_validation(self):
        with pytest.raises(ValueError, match="Invalid truncation window"):
            TruncatedLognormal(mean=10, sigma=1.0, minimum=0, maximum=128)
        with pytest.raises(ValueError, match="Invalid truncation window"):
            TruncatedLognormal(mean=10, sigma=1.0, minimum=128, maximum=64)
        with pytest.raises(ValueError, match="must lie in"):
            TruncatedLognormal(mean=256, sigma=1.0, minimum=8, maximum=128)
        with pytest.raises(ValueError, match="sigma"):
            TruncatedLognormal(mean=64, sigma=-1.0, minimum=8, maximum=128)
        with pytest.raises(ValueError, match="sigma"):
            TruncatedLognormal(mean=64, sigma=float("inf"), minimum=8, maximum=128)

    @pytest.mark.parametrize(
        ("mean", "sigma"),
        [
            # Genuinely unsamplable: the solve drives the window past what
            # double precision can represent, so these must fail loudly
            # rather than silently under-realize. (4090, 0.3) exhausts the
            # sampling interval's representable interior; (4000, 1.5) is
            # caught by the quadrature verifier.
            (4090, 0.3),
            (4000, 1.5),
        ],
    )
    def test_out_of_numerical_range_means_fail_loudly(self, mean, sigma):
        with pytest.raises(RuntimeError, match="stable numerical range"):
            TruncatedLognormal(mean=mean, sigma=sigma, minimum=32, maximum=4096)

    @pytest.mark.parametrize(
        ("mean", "sigma"), [(4050, 0.3), (33, 0.3)]  # near-max  # near-min twin
    )
    def test_near_boundary_means_are_samplable(self, mean, sigma):
        """Boundary-adjacent windows are numerically fine and must not be rejected.

        Both were refused by the old absolute mass gate, which measured the
        window with the same unstable CDF subtraction it was meant to guard;
        the stable mass plus a representability check accepts them, and they
        realize their configured mean.
        """
        distribution = TruncatedLognormal(mean=mean, sigma=sigma, minimum=32, maximum=4096)
        rng = np.random.default_rng(0)
        draws = np.array([distribution.sample(rng) for _ in range(20000)])
        assert 32 <= draws.min() and draws.max() <= 4096
        # Integer rounding dominates the error at small means (see the class
        # docstring), so this is looser than the solver's own 5e-3 contract.
        assert abs(draws.mean() - mean) <= 0.01 * mean

    def test_upper_tail_window_samples_its_whole_support(self):
        """An upper-tail window must not crash or quantize.

        Direct-CDF sampling stores Phi(alpha) and Phi(beta) separately; when
        the window sits in the far upper tail both round toward 1.0, the drawn
        p can land on exactly 1.0 (inv_cdf raises), and the interval resolves
        into so few doubles that the realized support and mean drift. The
        survival branch keeps full precision there.
        """
        distribution = TruncatedLognormal(mean=1400, sigma=5.0, minimum=512, maximum=131072)
        assert distribution._survival
        rng = np.random.default_rng(0)
        draws = np.array([distribution.sample(rng) for _ in range(100000)])  # no raise
        assert abs(draws.mean() - 1400) <= 5e-2 * 1400
        # The upper tail must stay reachable and the draw must not quantize:
        # over these 100k draws the direct branch capped the window at ~51k of
        # its 131072 support with ~1k distinct lengths, against ~6.7k here.
        # Both distortions silently reshape every packed window.
        assert draws.max() > 100000
        assert len(set(draws.tolist())) > 4000

    def test_deep_tail_window_is_rejected_rather_than_quantized(self):
        """Switching to the survival branch is not enough on its own.

        A window deep enough in the tail leaves even the survival interval
        spanning a handful of denormals: no crash, but the realized support
        collapses to a few lengths and the mean drifts off the configured
        value. The resolution floor applies to whichever interval is drawn.
        """
        with pytest.raises(RuntimeError, match="resolves too few values"):
            TruncatedLognormal(mean=525.056, sigma=1.0, minimum=512, maximum=131072)

    def test_quadrature_verifier_catches_a_lying_solve(self, monkeypatch):
        # The quadrature verifier (positive-integrand Simpson, independent
        # of the CDF-difference solve) is the second gate; reach it by lying
        # to the bisection only. The liar steers _solve_mu to a wrong but
        # in-window mu (log 512): there the truncation-window mass is ~1, so
        # both representability gates pass, but the realized mean (~535) is
        # far from the configured 1536 — only quadrature can catch it.
        # mean=1536 sigma=0.3 in [32, 4096] constructs fine without the lie
        # (see the realize-the-mean sweep above).
        import math

        wrong_mu = math.log(512.0)
        monkeypatch.setattr(
            TruncatedLognormal,
            "_truncated_mean",
            lambda self, mu: self.mean - 1.0 if mu < wrong_mu else self.mean + 1.0,
        )
        with pytest.raises(RuntimeError, match="stable numerical range") as excinfo:
            TruncatedLognormal(mean=1536, sigma=0.3, minimum=32, maximum=4096)
        # The quadrature branch's message, not the mass gate's.
        assert "the solve realizes" in str(excinfo.value)
        assert "mass to" not in str(excinfo.value)

    @pytest.mark.parametrize(
        ("mean", "sigma", "minimum", "maximum"),
        [
            # Representative sweep incl. near-boundary means: everything
            # that CONSTRUCTS must realize its configured mean.
            (3900, 0.3, 32, 4096),
            (40, 0.8, 32, 4096),
            (1536, 0.3, 32, 4096),
            (48, 1.0, 8, 256),
            (91750, 0.8, 2048, 131072),
            (110000, 0.8, 2048, 131072),
        ],
    )
    def test_constructed_solves_realize_the_configured_mean(self, mean, sigma, minimum, maximum):
        sampler = TruncatedLognormal(mean=mean, sigma=sigma, minimum=minimum, maximum=maximum)
        # The independent quadrature verifier already gates construction;
        # cross-check with actual draws to close the loop end to end, and
        # confirm every draw respects the truncation window.
        rng = np.random.default_rng(7)
        values = [sampler.sample(rng) for _ in range(20_000)]
        assert min(values) >= minimum and max(values) <= maximum
        realized = float(np.mean(values))
        assert abs(realized - mean) / mean < 0.02

    def test_sigma_zero_requires_an_integer_mean(self):
        with pytest.raises(ValueError, match="must be an integer"):
            TruncatedLognormal(mean=96.5, sigma=0, minimum=64, maximum=128)

    @pytest.mark.parametrize(
        ("mean", "sigma", "minimum", "maximum"),
        [
            # The two ways into the same degenerate branch: zero spread, and
            # a window with no room to spread in.
            (96, 0, 64, 128),
            (64, 1.0, 64, 64),
        ],
    )
    def test_degenerate_component_is_constant_at_the_mean(self, mean, sigma, minimum, maximum):
        sampler = TruncatedLognormal(mean=mean, sigma=sigma, minimum=minimum, maximum=maximum)
        rng = np.random.default_rng(0)
        assert {sampler.sample(rng) for _ in range(32)} == {mean}


class TestStrictValidators:
    def test_require_integer_rejects_coercibles(self):
        for bad in (True, False, 1.0, "1", None):
            with pytest.raises(ValueError, match="must be an integer"):
                require_integer(bad, what="field")
        assert require_integer(np.int64(5), what="field") == 5

    def test_require_number_rejects_bools_and_non_finite(self):
        for bad in (True, float("nan"), float("inf"), "1.5", None):
            with pytest.raises(ValueError, match="finite number"):
                require_number(bad, what="field")
        with pytest.raises(ValueError, match="must be >="):
            require_number(-0.5, what="field", minimum=0.0)
        assert require_number(2, what="field") == 2.0


class TestRequireExactDict:
    """The one implementation of the is-dict / unknown / missing rules that
    every config mapping in the kernel goes through."""

    def test_three_branches(self):
        from examples.multimodal_dev.data.mock_varlen.distributions import require_exact_dict

        allowed = {"a", "b"}
        with pytest.raises(ValueError, match="must be a dict"):
            require_exact_dict([1], allowed, what="cfg")
        with pytest.raises(ValueError, match=r"unknown key\(s\) \['c'\]"):
            require_exact_dict({"a": 1, "b": 2, "c": 3}, allowed, what="cfg")
        with pytest.raises(ValueError, match=r"missing required key\(s\) \['b'\]"):
            require_exact_dict({"a": 1}, allowed, what="cfg")
        assert require_exact_dict({"a": 1, "b": 2}, allowed, what="cfg") == {"a": 1, "b": 2}


class TestCategorical:
    def test_validation(self):
        # The is-dict / unknown-key branches belong to require_exact_dict;
        # what is specific here is the counts/weights shape and numerics.
        with pytest.raises(ValueError, match="non-empty list"):
            Categorical({"counts": [], "weights": []}, what="c")
        with pytest.raises(ValueError, match="matching counts in length"):
            Categorical({"counts": [1, 2], "weights": [1]}, what="c")
        with pytest.raises(ValueError, match="must be an integer"):
            Categorical({"counts": [True], "weights": [1]}, what="c")
        with pytest.raises(ValueError, match="positive finite sum"):
            Categorical({"counts": [1, 2], "weights": [0, 0]}, what="c")

    def test_zero_weight_entries_are_never_drawn_via_the_shared_cdf_draw(self):
        # draw_from_cdf is THE weighted-draw idiom every layout sampler
        # shares: side="right" means zero-weight entries (CDF plateaus)
        # are never selected, and the clamp covers the u == 1.0 edge.
        from examples.multimodal_dev.data.mock_varlen.distributions import draw_from_cdf

        categorical = Categorical({"counts": [1, 1000, 3], "weights": [1, 0, 1]}, what="c")
        weights = np.asarray(categorical.weights, dtype=np.float64)
        cdf = np.cumsum(weights / weights.sum())
        rng = np.random.default_rng(1234)
        drawn = {categorical.counts[draw_from_cdf(rng, cdf)] for _ in range(512)}
        assert drawn == {1, 3}


class TestSeedStreamRng:
    def test_pure_function_of_its_keys(self):
        assert seed_stream_rng(1234, 5, 20).random() == seed_stream_rng(1234, 5, 20).random()
        assert seed_stream_rng(1234, 5, 20).random() != seed_stream_rng(1234, 5, 21).random()
        assert seed_stream_rng(1234, 5, 20).random() != seed_stream_rng(1234, 6, 20).random()
        assert seed_stream_rng(1234, 5, 20).random() != seed_stream_rng(999, 5, 20).random()

    def test_matches_the_seed_sequence_pattern(self):
        expected = np.random.default_rng(np.random.SeedSequence([1234, 5, 20])).random()
        assert seed_stream_rng(1234, 5, 20).random() == expected
