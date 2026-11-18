# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""The replay harness itself: the byte-exact comparator and the replay assertions.

These are the semantics every kernel test in this directory relies on, so they are pinned
here: ``bytes_equal`` must see signed zeros and NaN payloads, and the replay assertions must
report such differences instead of accepting them as ``torch.equal`` would.
"""

import pytest
import torch

from tests.unit_tests.determinism.kernels.harness import (
    _describe_mismatch,
    assert_replays_bit_exact,
    bytes_equal,
    clone_inputs,
    count_differing_replays,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

QUIET_NAN = 0x7FC00000
NEGATIVE_QUIET_NAN = 0xFFC00000 - (1 << 32)  # same bits as an int32 literal


def _f32_with_bits(pattern: int, n: int = 1) -> torch.Tensor:
    return torch.full((n,), pattern, dtype=torch.int32, device="cuda").view(torch.float32)


def _flip_sign_of_zero_every_other_call():
    """A stateful "kernel" whose output alternates between +0.0 and -0.0."""
    calls = {"n": 0}

    def fn(t):
        calls["n"] += 1
        sign = -1.0 if calls["n"] % 2 == 0 else 1.0
        return torch.zeros_like(t) * sign

    return fn


class TestBytesEqual:
    def test_identical_finite_tensors(self):
        a = torch.randn(64, 32, device="cuda", dtype=torch.bfloat16)
        assert bytes_equal(a, a.clone())

    def test_value_difference(self):
        a = torch.randn(64, device="cuda")
        b = a.clone()
        b[7] += 1e-6
        assert not bytes_equal(a, b)

    def test_signed_zeros_differ(self):
        pos = torch.tensor([0.0], device="cuda")
        neg = torch.tensor([-0.0], device="cuda")
        assert torch.equal(pos, neg)  # what plain equality would have accepted
        assert not bytes_equal(pos, neg)

    def test_same_nan_payload_matches(self):
        assert bytes_equal(_f32_with_bits(QUIET_NAN), _f32_with_bits(QUIET_NAN))

    def test_different_nan_payloads_differ(self):
        assert not bytes_equal(_f32_with_bits(QUIET_NAN), _f32_with_bits(QUIET_NAN + 1))
        assert not bytes_equal(_f32_with_bits(QUIET_NAN), _f32_with_bits(NEGATIVE_QUIET_NAN))

    def test_shape_and_dtype_mismatch(self):
        a = torch.zeros(4, device="cuda")
        assert not bytes_equal(a, torch.zeros(2, 2, device="cuda"))
        assert not bytes_equal(a, torch.zeros(4, device="cuda", dtype=torch.float64))

    def test_layout_is_not_compared(self):
        a = torch.randn(8, 16, device="cuda")
        strided = a.t().contiguous().t()
        assert not strided.is_contiguous()
        assert bytes_equal(a, strided)

    def test_empty_and_scalar_tensors(self):
        assert bytes_equal(torch.empty(0, device="cuda"), torch.empty(0, device="cuda"))
        assert bytes_equal(torch.tensor(2.5, device="cuda"), torch.tensor(2.5, device="cuda"))
        assert not bytes_equal(torch.tensor(0.0, device="cuda"), torch.tensor(-0.0, device="cuda"))

    def test_bool_and_integer_dtypes(self):
        a = torch.randint(0, 100, (33,), device="cuda", dtype=torch.int64)
        assert bytes_equal(a, a.clone())
        assert not bytes_equal(a, a + 1)
        mask = a > 50
        assert bytes_equal(mask, mask.clone())


class TestDescribeMismatch:
    def test_reports_bit_only_differences(self):
        a = torch.tensor([0.0, 0.0, 0.0, 0.0], device="cuda")
        b = torch.tensor([0.0, -0.0, 0.0, 0.0], device="cuda")
        msg = _describe_mismatch("out", a, b)
        assert "1/4 elements differ" in msg
        assert "bit pattern" in msg

    def test_reports_value_differences(self):
        a = torch.zeros(4, device="cuda")
        b = torch.tensor([0.0, 0.0, 1.5, 0.0], device="cuda")
        msg = _describe_mismatch("out", a, b)
        assert "1/4 elements differ" in msg
        assert "1.500e+00" in msg
        assert "bit pattern" not in msg


class TestReplayAssertions:
    def test_pure_function_passes(self):
        x = torch.randn(1024, 256, device="cuda", requires_grad=True)
        assert_replays_bit_exact(lambda t: (t * 2).sum(-1), (x,), replays=3, what="pure")

    def test_sign_of_zero_flip_is_caught(self):
        x = torch.randn(16, device="cuda")
        with pytest.raises(AssertionError, match="bit pattern"):
            assert_replays_bit_exact(
                _flip_sign_of_zero_every_other_call(), (x,), replays=2, backward=False
            )

    def test_nan_payload_change_is_caught(self):
        calls = {"n": 0}

        def fn(t):
            calls["n"] += 1
            return _f32_with_bits(QUIET_NAN + calls["n"] % 2, t.numel())

        x = torch.randn(16, device="cuda")
        with pytest.raises(AssertionError, match="not bit-exact"):
            assert_replays_bit_exact(fn, (x,), replays=2, backward=False)

    def test_count_differing_replays_counts_bit_differences(self):
        x = torch.randn(16, device="cuda")
        # Reference is call 1 (+0.0); calls 2 and 4 return -0.0.
        assert (
            count_differing_replays(
                _flip_sign_of_zero_every_other_call(), (x,), replays=5, backward=False
            )
            == 2
        )

    def test_clone_inputs_preserves_expanded_layout(self):
        base = torch.randn(4, 1, device="cuda").expand(4, 8)
        clone = clone_inputs(base)
        assert clone.stride() == base.stride()
        assert bytes_equal(clone, base)
