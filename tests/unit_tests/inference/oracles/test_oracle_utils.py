# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""CPU tests of the oracle harness itself (oracle_utils.py).

Each assertion helper is checked both ways: it passes on correct data and fails, with a
useful message, on the bug class it exists to catch.
"""

import contextlib
import math

import pytest
import torch

from tests.unit_tests.inference.oracles import oracle_utils as ou

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

# ---------------------------------------------------------------------------------------------
# fp64 references
# ---------------------------------------------------------------------------------------------


def test_fp64_reference_sees_the_kernels_rounded_inputs():
    x = torch.tensor([1 / 3, 2 / 3], dtype=torch.float32).to(torch.bfloat16)
    seen = {}

    def reference(t, scale, *, idx):
        seen.update(t=t, idx=idx)
        return t * scale

    out = ou.fp64_reference(reference, x, 2.0, idx=torch.tensor([1, 0]))
    assert seen["t"].dtype == torch.float64 and seen["idx"].dtype == torch.int64
    assert torch.equal(seen["t"], x.double())  # the bf16 values, not 1/3 and 2/3
    assert out.dtype == torch.float64

    with pytest.raises(TypeError, match="must stay fp64"):
        ou.fp64_reference(lambda t: t.float(), x)


def test_dyadic_tensor_sums_are_exact_in_any_order():
    g = ou.make_generator()
    x = ou.dyadic_tensor((4096,), mantissa_bits=4, exponents=(-2, 2), generator=g)
    assert torch.equal(x.double().to(torch.bfloat16), x)
    # 4096 * 2**(4 + 2 + 2) = 2**20 <= 2**24: every fp32 summation order is exact.
    exact = x.double().sum()
    order = torch.randperm(x.numel(), generator=g)
    assert x.float().sum().double() == exact
    assert x.float()[order].cumsum(0)[-1].double() == exact
    with pytest.raises(ValueError):
        ou.dyadic_tensor((2,), mantissa_bits=9, dtype=torch.bfloat16)


def test_seeded_restores_the_global_generator():
    before = torch.random.get_rng_state()
    with ou.seeded(7):
        a = torch.rand(3)
    with ou.seeded(7):
        b = torch.rand(3)
    assert torch.equal(a, b)
    assert torch.equal(torch.random.get_rng_state(), before)


# ---------------------------------------------------------------------------------------------
# Tolerance policy
# ---------------------------------------------------------------------------------------------


def _matmul_case():
    g = ou.make_generator()
    a = torch.randn(64, 256, generator=g).to(torch.bfloat16)
    b = torch.randn(256, 48, generator=g).to(torch.bfloat16)
    reference = ou.fp64_reference(torch.matmul, a, b)
    return a, b, reference


def test_correct_kernel_passes_against_torch_bf16():
    a, b, reference = _matmul_case()
    trusted = torch.matmul(a, b)
    # A different correct implementation: fp32 accumulation in another order, rounded once.
    kernel = sum(a[:, k : k + 64].float() @ b[k : k + 64].float() for k in range(192, -1, -64)).to(
        torch.bfloat16
    )
    report = ou.assert_close_to_reference(kernel, reference, trusted, what="matmul")
    assert report.kernel_max_abs <= report.bound_max_abs
    assert report.kernel_rms <= report.bound_rms


def test_bf16_accumulation_fails_against_torch_bf16():
    a, b, reference = _matmul_case()
    trusted = torch.matmul(a, b)
    kernel = torch.zeros(64, 48, dtype=torch.bfloat16)
    for k in range(256):  # rounds the running sum to bf16 at every step
        kernel = (kernel.float() + a[:, k : k + 1].float() * b[k : k + 1].float()).bfloat16()
    with pytest.raises(AssertionError, match="exceeds 2.0x the trusted") as info:
        ou.assert_close_to_reference(kernel, reference, trusted, what="matmul")
    message = str(info.value)
    assert "max-abs error: kernel" in message and "trusted" in message
    assert "rms error:" in message and "largest kernel error at" in message


@pytest.mark.parametrize("scale, passes", [(1.5, True), (1.9, True), (2.5, False), (4.0, False)])
def test_policy_bounds_kernel_error_by_factor_times_trusted_error(scale, passes):
    g = ou.make_generator()
    reference = torch.randn(4096, generator=g, dtype=torch.float64)
    noise = 1e-3 * torch.randn(4096, generator=g, dtype=torch.float64)
    trusted = (reference + noise).float()
    kernel = (reference + scale * noise).float()
    if passes:
        report = ou.assert_close_to_reference(kernel, reference, trusted)
        assert report.kernel_rms == pytest.approx(scale * report.trusted_rms, rel=1e-3)
    else:
        with pytest.raises(AssertionError) as info:
            ou.assert_close_to_reference(kernel, reference, trusted)
        # Both errors are in the message.
        rms = lambda x: (x.double() - reference).square().mean().sqrt().item()  # noqa: E731
        assert f"rms error:     kernel {rms(kernel):.3e}, trusted {rms(trusted):.3e}" in str(
            info.value
        )
    # The factor is the knob: a wider one admits the 2.5x kernel, not the 4x one.
    with pytest.raises(AssertionError) if scale > 3.5 else contextlib.nullcontext():
        ou.assert_close_to_reference(kernel, reference, trusted, factor=3.5)


def test_floor_covers_exact_trusted_results():
    g = ou.make_generator()
    x = ou.dyadic_tensor((512,), generator=g)  # exact in bf16, so reference == trusted
    reference = x.double()
    trusted = x.clone()
    ou.assert_close_to_reference(x.clone(), reference, trusted)

    one_ulp = x.clone()
    i = int(x.double().abs().argmax())
    one_ulp.view(torch.int16)[i] += 1  # one ulp at the largest element
    ou.assert_close_to_reference(one_ulp, reference, trusted)

    far = x.clone()
    far.view(torch.int16)[i] += 4
    with pytest.raises(AssertionError, match="trusted 0.000e"):
        ou.assert_close_to_reference(far, reference, trusted)

    # The floor covers one element, not a systematic one-ulp error everywhere.
    everywhere = x.clone()
    everywhere.view(torch.int16)[x != 0] += 1
    with pytest.raises(AssertionError, match="rms error"):
        ou.assert_close_to_reference(everywhere, reference, trusted)


@pytest.mark.parametrize("device", DEVICES)
def test_ulp(device):
    x = torch.tensor([1.0, 1.5, 2.0, -3.0, 0.0, 1e-40], dtype=torch.float64, device=device)
    expected = [2**-7, 2**-7, 2**-6, 2**-6, 0.0, 2**-133]
    assert ou.ulp(x, torch.bfloat16).tolist() == expected
    assert ou.ulp(x[:1], torch.float32).item() == 2**-23


@pytest.mark.parametrize("device", DEVICES)
def test_pow2_is_exact(device):
    # Exact on every device, unlike torch.ldexp on CUDA; grids built from it are exact too.
    e = torch.arange(-1022, 1024, device=device)
    assert ou.pow2(e).tolist() == [2.0**k for k in range(-1022, 1024)]
    x = ou.dyadic_tensor((4096,), exponents=(-60, 60), dtype=torch.float32, device=device)
    m, k = torch.frexp(x.double())
    assert torch.equal((m * 2**5).frac(), torch.zeros_like(m))  # at most 4 mantissa bits


def test_non_finite_values_must_match_the_reference():
    reference = torch.tensor([1.0, float("nan"), float("inf"), 2.0], dtype=torch.float64)
    trusted = reference.float()
    ou.assert_close_to_reference(reference.float(), reference, trusted)
    for bad in ([1.0, 0.0, float("inf"), 2.0], [1.0, float("nan"), float("-inf"), 2.0]):
        with pytest.raises(AssertionError, match="NaN/inf"):
            ou.assert_close_to_reference(torch.tensor(bad), reference, trusted)
    with pytest.raises(AssertionError, match=r"\(3,\): kernel nan"):
        ou.assert_close_to_reference(
            torch.tensor([1.0, float("nan"), float("inf"), float("nan")]), reference, trusted
        )


def test_per_slice_bounds_catch_errors_in_small_rows():
    g = ou.make_generator()
    reference = torch.randn(8, 1024, generator=g, dtype=torch.float64)
    reference[3] *= 1e-6
    trusted = reference.to(torch.bfloat16)
    kernel = trusted.clone()
    kernel[3] *= 1.1  # a 10% error, invisible next to the other rows' rounding
    ou.assert_close_to_reference(kernel, reference, trusted)
    with pytest.raises(AssertionError, match=r"in 1 of 8 slices[\s\S]*slice \(3,\)"):
        ou.assert_close_to_reference(kernel, reference, trusted, dim=-1)


def test_argument_errors_are_test_bugs_not_kernel_failures():
    reference = torch.randn(4, dtype=torch.float64)
    x = reference.float()
    with pytest.raises(TypeError, match="float64"):
        ou.assert_close_to_reference(x, reference.float(), x.clone())
    with pytest.raises(TypeError, match="share a floating dtype"):
        ou.assert_close_to_reference(x, reference, reference.to(torch.bfloat16))
    with pytest.raises(ValueError, match="same tensor"):
        ou.assert_close_to_reference(x, reference, x)
    with pytest.raises(AssertionError, match="shape"):
        ou.assert_close_to_reference(x[:3], reference, x.clone())


# ---------------------------------------------------------------------------------------------
# Exact assertions
# ---------------------------------------------------------------------------------------------


def test_assert_exact_reports_first_mismatches():
    expected = torch.arange(100, dtype=torch.int32).reshape(10, 10)
    ou.assert_exact(expected.to(torch.int64), expected)
    actual = expected.clone()
    actual[2, 3] = -1
    actual[7, 0] = 5
    with pytest.raises(AssertionError, match=r"2 of 100 elements differ") as info:
        ou.assert_exact(actual, expected, what="counts")
    assert "(2, 3): got -1, expected 23" in str(info.value)
    assert "(7, 0): got 5, expected 70" in str(info.value)


def test_assert_exact_compares_floats_bitwise():
    nan = torch.tensor([float("nan"), 1.0], dtype=torch.bfloat16)
    ou.assert_exact(nan, nan.clone())
    with pytest.raises(AssertionError, match="0x8000"):
        ou.assert_exact(torch.tensor([-0.0]), torch.tensor([0.0]))
    with pytest.raises(TypeError, match="dtypes differ"):
        ou.assert_exact(torch.zeros(2), torch.zeros(2, dtype=torch.float64))


def test_assert_is_permutation():
    ou.assert_is_permutation(torch.randperm(50, generator=ou.make_generator()))
    with pytest.raises(AssertionError, match=r"values missing, first \[3\].*repeated, first \[1\]"):
        ou.assert_is_permutation(torch.tensor([0, 1, 1, 2]))
    with pytest.raises(AssertionError, match="out of range"):
        ou.assert_is_permutation(torch.tensor([0, 4, 1, 2]))


def test_assert_rows_equal_unordered():
    g = ou.make_generator()
    rows = torch.randn(32, 16, generator=g).to(torch.bfloat16)
    ou.assert_rows_equal_unordered(rows[torch.randperm(32, generator=g)], rows)
    duplicated = rows.clone()
    duplicated[5] = rows[6]
    with pytest.raises(AssertionError, match=r"2 distinct rows .*\n.*1x in actual, 0x expected"):
        ou.assert_rows_equal_unordered(rows, duplicated)
    ou.assert_rows_equal_unordered(torch.empty(0, 4), torch.empty(0, 4))


def test_quantized_codes_and_scales_are_reported_separately():
    g = ou.make_generator()
    codes = torch.randint(0, 0x7E, (4, 64), generator=g, dtype=torch.uint8)
    scales = torch.randint(100, 140, (4, 2), generator=g, dtype=torch.uint8)
    codes8 = codes.view(torch.float8_e4m3fn)
    ou.assert_quantized_bit_exact(
        codes8, scales, codes8.clone(), scales.clone(), elems_per_scale=32
    )

    bad_codes, bad_scales = codes.clone(), scales.clone()
    bad_codes[1, 40] ^= 1  # block (1, 1): scale matches -> a rounding bug
    bad_codes[2, 3] ^= 1  # block (2, 0): its scale differs too
    bad_scales[2, 0] += 1
    with pytest.raises(AssertionError) as info:
        ou.assert_quantized_bit_exact(
            bad_codes.view(torch.float8_e4m3fn),
            bad_scales,
            codes8,
            scales,
            elems_per_scale=32,
            what="mxfp8",
        )
    message = str(info.value)
    assert "scales: 1 of 8 differ" in message and "block (2, 0)" in message
    assert "codes: 2 of 256 differ (1 in blocks whose scale differs, 1 under a matching scale)" in (
        message
    )


# ---------------------------------------------------------------------------------------------
# Structure-derived sizes
# ---------------------------------------------------------------------------------------------


def test_boundary_sizes():
    assert ou.boundary_sizes(128) == [0, 1, 127, 128, 129, 255, 256, 257]
    assert ou.boundary_sizes(16, 64, multiples=(), degenerate=False) == [15, 16, 17, 63, 64, 65]
    assert ou.boundary_sizes(1, multiples=(3,)) == [0, 1, 2, 3, 4]
    assert ou.boundary_sizes(32, production=(4096,), minimum=2, maximum=4096) == [
        31,
        32,
        33,
        63,
        64,
        65,
        4096,
    ]
    ids = [p.id for p in ou.size_params(128, multiples=(), production=(8192,))]
    assert ids == [
        "0[empty]",
        "1[single]",
        "127[128-1]",
        "128[128]",
        "129[128+1]",
        "8192[production]",
    ]


@pytest.mark.parametrize("row_elems", [1, 3, 4096, 7168, 129280])
def test_int32_overflow_rows_straddle_2_31(row_elems):
    rows = ou.int32_overflow_rows(row_elems)
    fits, crosses = rows[0], rows[1]
    assert fits * row_elems - 1 < 2**31 <= crosses * row_elems - 1  # last element's offset
    starts_past = rows[-1]
    assert (starts_past - 1) * row_elems >= 2**31 > (starts_past - 2) * row_elems
    params = ou.int32_overflow_params(row_elems)
    assert [p.values[0] for p in params] == rows
    assert all(any(m.name == "large" for m in p.marks) for p in params)


def test_pairwise_cases_cover_every_pair():
    factors = {
        "dtype": ["bf16", "fp16", "fp32"],
        "n": [0, 1, 127, 128],
        "topk": [1, 2, 8],
        "bi": [0, 1],
    }
    cases = ou.pairwise_cases(factors)
    assert cases == ou.pairwise_cases(factors)  # deterministic
    assert len(cases) < 3 * 4 * 3 * 2
    names = list(factors)
    for i, f in enumerate(names):
        for g in names[i + 1 :]:
            seen = {(c[f], c[g]) for c in cases}
            assert seen == {(a, b) for a in factors[f] for b in factors[g]}
    assert ou.pairwise_cases({"x": [1, 2]}) == [{"x": 1}, {"x": 2}]


# ---------------------------------------------------------------------------------------------
# Memory safety
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.int32, torch.float8_e4m3fn])
def test_guard_band_catches_out_of_bounds_writes(device, dtype):
    buf = ou.guarded_empty((5, 7), dtype, device)
    t = buf.tensor
    assert t.shape == (5, 7) and t.is_contiguous()
    assert (t.data_ptr() - buf._raw.data_ptr()) % 256 == 0
    t.zero_()  # an in-bounds write
    buf.check()

    # A kernel that writes one row too many (e.g. a missing mask on the last tile).
    t.as_strided((6, 7), (7, 1)).zero_()
    with pytest.raises(AssertionError, match=r"past the end, from byte \+0 .*elements \+0 to \+6"):
        buf.check("out")

    before = ou.guarded_like(t)
    before.tensor.as_strided((1,), (1,), before.tensor.storage_offset() - 2).zero_()
    with pytest.raises(AssertionError, match="before the start"):
        before.check()
    with pytest.raises(
        AssertionError, match=r"(?s)out: .*past the end.*\n  inp: .*before the start"
    ):
        ou.check_guards(out=buf, inp=before)


def test_guard_band_reads_are_nan_for_floats():
    buf = ou.guarded_like(torch.ones(3, 4, dtype=torch.bfloat16))
    assert torch.equal(buf.tensor, torch.ones(3, 4, dtype=torch.bfloat16))
    past = buf.tensor.as_strided((14,), (1,))[12:]  # a read one element past the end
    assert torch.isnan(past).all()
    ou.guarded_empty((0, 16), torch.float16, "cpu").check()


def test_poison():
    assert torch.isnan(ou.poisoned_empty((4,), torch.bfloat16, "cpu")).all()
    assert (ou.poisoned_empty((4,), torch.int32, "cpu") == -(2**31)).all()
    assert (ou.poisoned_empty((4,), torch.uint8, "cpu") == 255).all()
    assert torch.isnan(ou.poisoned_empty((4,), torch.float8_e4m3fn, "cpu").float()).all()
    fp4 = ou.poisoned_empty((4,), torch.float4_e2m1fn_x2, "cpu")
    assert (fp4.view(torch.uint8) == 0xFF).all()
    with pytest.raises(TypeError):
        ou.poison_(torch.zeros(2, dtype=torch.bool))

    padding = ou.poisoned_empty((6, 3), torch.float32, "cpu")
    ou.assert_poisoned(padding)
    padding[4, 1] = 0.0  # a write to a row past the live count
    with pytest.raises(AssertionError, match=r"untouched[\s\S]*\(4, 1\): got 0.0"):
        ou.assert_poisoned(padding, what="padding rows")


# ---------------------------------------------------------------------------------------------
# Hardware gating
# ---------------------------------------------------------------------------------------------

GB200 = ("launch_on_gb200",)


@pytest.mark.parametrize(
    "required, device, target, markers, action",
    [
        ((9,), 9, None, (), "run"),
        ((9,), 10, None, (), "skip"),
        ((9,), None, None, (), "skip"),
        ((9,), None, 9, (), "fail"),  # a runner declared H100 but has no GPU
        ((10,), 10, None, GB200, "run"),
        ((10,), 9, None, GB200, "skip"),  # H100 CI: skipped with a reason, never a silent pass
        ((10,), 9, 10, GB200, "fail"),  # selected for GB200, running elsewhere
        ((10,), 10, None, (), "fail"),  # GB200 CI would never launch it
        ((9, 10), 9, None, GB200, "run"),
        ((9, 10), 9, None, (), "fail"),
        ((9,), 10, 10, GB200, "skip"),  # an H100-only case in a GB200 run
    ],
)
def test_hardware_decision(required, device, target, markers, action):
    decision, reason = ou.hardware_decision(
        required,
        device_major=device,
        target_sm=target,
        launch_markers=markers,
        file_has_markers=markers,
    )
    assert decision == action, reason
    assert bool(reason) == (action != "run")
    if action == "skip" and 10 in required:
        assert "launch_on_gb200" in reason


def test_hardware_decision_needs_the_marker_text_in_the_file():
    decision, reason = ou.hardware_decision(
        (10,), device_major=10, target_sm=None, launch_markers=GB200, file_has_markers=()
    )
    assert decision == "fail" and "find_test_cases.py" in reason


def test_session_target_sm():
    assert ou.session_target_sm("launch_on_gb200 and not flaky_in_dev", {}) == ou.BLACKWELL
    assert ou.session_target_sm("not flaky_in_dev", {}) is None
    assert ou.session_target_sm("not launch_on_gb200", {}) is None
    assert ou.session_target_sm("", {ou.TARGET_SM_ENV: "9"}) == ou.HOPPER
    assert ou.large_selected("large") and ou.large_selected("sanitizer or large")
    assert not ou.large_selected("launch_on_gb200 and not flaky_in_dev")


def test_require_sm_skips_or_fails(monkeypatch):
    monkeypatch.setattr(ou, "device_sm_major", lambda: ou.HOPPER)
    monkeypatch.delenv(ou.TARGET_SM_ENV, raising=False)
    ou.require_sm(ou.HOPPER)
    with pytest.raises(pytest.skip.Exception, match="sm_10x"):
        ou.require_sm(ou.BLACKWELL)
    monkeypatch.setenv(ou.TARGET_SM_ENV, "10")
    with pytest.raises(pytest.fail.Exception, match="targets sm_10x but the device is sm_9x"):
        ou.require_sm(ou.BLACKWELL)


def test_use_local_rank_device(monkeypatch):
    chosen = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 8)
    monkeypatch.setattr(torch.cuda, "set_device", chosen.append)
    assert ou.use_local_rank_device({"LOCAL_RANK": "3"}) == 3
    assert ou.use_local_rank_device({"LOCAL_RANK": "11"}) == 3  # more ranks than GPUs: wrap
    assert ou.use_local_rank_device({}) is None  # not under torch.distributed.run
    assert chosen == [3, 3]
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert ou.use_local_rank_device({"LOCAL_RANK": "3"}) is None
    assert chosen == [3, 3]


@pytest.mark.launch_on_gb200
@ou.requires_sm(ou.BLACKWELL)
def test_requires_sm_marker_runs_on_blackwell_only():
    assert torch.cuda.get_device_capability()[0] == ou.BLACKWELL


@ou.requires_sm(ou.HOPPER)
def test_requires_sm_marker_runs_on_hopper_only():
    assert torch.cuda.get_device_capability()[0] == ou.HOPPER


# ---------------------------------------------------------------------------------------------
# Statistical tests for samplers
# ---------------------------------------------------------------------------------------------


def test_chi_square_sf_matches_tables():
    assert ou.chi_square_sf(3.841458820694124, 1) == pytest.approx(0.05, rel=1e-9)
    assert ou.chi_square_sf(18.307038053275146, 10) == pytest.approx(0.05, rel=1e-9)
    assert ou.chi_square_sf(0.0, 4) == 1.0


def _probs(v=50):
    w = torch.arange(1, v + 1, dtype=torch.float64) ** 1.5
    return w / w.sum()


def _counts(probs, n, seed=ou.DEFAULT_SEED):
    samples = torch.multinomial(probs, n, replacement=True, generator=ou.make_generator(seed))
    return torch.bincount(samples, minlength=probs.numel())


def test_correct_sampler_passes():
    probs = _probs()
    counts = _counts(probs, 100_000)
    ou.assert_matches_distribution(counts, probs)
    ou.assert_counts_within_binomial_bounds(counts, probs)
    batch = torch.stack([_counts(probs, 20_000, seed) for seed in range(3)])
    ou.assert_matches_distribution(batch, probs.expand(3, -1))


def test_biased_sampler_fails():
    probs = _probs()
    biased = probs.clone()
    biased[10] *= 1.5
    biased /= biased.sum()  # category 10: 0.50% -> 0.75%
    n = ou.samples_to_detect(float(probs[10]), float(biased[10] - probs[10]))
    # samples_to_detect sizes the per-category binomial test; the 49-dof chi-square spreads its
    # power over all categories and needs a few times more.
    counts = _counts(biased, 4 * n)
    with pytest.raises(AssertionError, match="probability 1.0e-06"):
        ou.assert_matches_distribution(counts, probs)
    with pytest.raises(AssertionError, match=r"\(0, 10\): count"):
        ou.assert_counts_within_binomial_bounds(counts, probs)


def test_samples_outside_the_support_fail_exactly():
    probs = torch.tensor([0.5, 0.5, 0.0])
    counts = torch.tensor([5000, 5000, 1])
    with pytest.raises(AssertionError, match=r"1 samples fall on categories with probability 0"):
        ou.assert_matches_distribution(counts, probs)
    with pytest.raises(AssertionError, match="probability 0"):
        ou.assert_counts_within_binomial_bounds(counts, probs)
    with pytest.raises(ValueError, match="too few samples"):
        ou.assert_matches_distribution(torch.tensor([3, 2]), torch.tensor([0.5, 0.5]))


def test_false_failure_rate_matches_alpha():
    # A correct sampler must fail at about rate alpha: 400 seeds at alpha = 0.05.
    probs = _probs(10)
    failures = 0
    for seed in range(400):
        try:
            ou.assert_matches_distribution(_counts(probs, 2000, seed), probs, alpha=0.05)
        except AssertionError:
            failures += 1
    lo, hi = ou.binomial_bounds(400, 0.05, 1e-6)
    assert lo <= failures <= hi, failures


@pytest.mark.parametrize("n, p, alpha", [(50, 0.3, 0.01), (200, 0.02, 1e-3), (30, 0.9, 0.1)])
def test_binomial_bounds_are_exact(n, p, alpha):
    pmf = [math.comb(n, k) * p**k * (1 - p) ** (n - k) for k in range(n + 1)]
    lo, hi = ou.binomial_bounds(n, p, alpha)
    assert sum(pmf[:lo]) <= alpha / 2 < sum(pmf[: lo + 1])
    assert sum(pmf[hi + 1 :]) <= alpha / 2 < sum(pmf[hi:])
    assert ou.binomial_bounds(10, 0.0) == (0, 0) and ou.binomial_bounds(10, 1.0) == (10, 10)


def test_samples_to_detect():
    # Normal approximation: (1.96 * 0.5 / 0.01) ** 2 = 9604 at power 0.5.
    assert ou.samples_to_detect(0.5, 0.01, alpha=0.05, power=0.5) == pytest.approx(9604, rel=1e-3)
    assert ou.samples_to_detect(0.1, 0.01) > ou.samples_to_detect(0.1, 0.02)
