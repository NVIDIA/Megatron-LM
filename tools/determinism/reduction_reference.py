# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Explicit error budgets for independently computed reduction references."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ReductionReference:
    """FP64 sums and conditioning data, computed without the observed result.

    ``rounding`` describes materialization before summation. For example, a
    BF16 per-token bias gradient is rounded before autograd reduces it.
    """

    total: torch.Tensor
    sum_absolute_terms: torch.Tensor
    terms: int
    rounding: str
    accumulation_dtype: torch.dtype = torch.float32
    exact_terms: bool = False
    term_dtype: torch.dtype = torch.float32

    @classmethod
    def from_terms(
        cls,
        values: torch.Tensor,
        *,
        dim: int,
        keepdim: bool,
        rounding: str,
        accumulation_dtype: torch.dtype = torch.float32,
        exact_terms: bool = False,
        term_dtype: torch.dtype = torch.float32,
    ) -> ReductionReference:
        """Summarize independent terms in FP64, preserving the reduction shape."""
        if not values.is_floating_point() or not values.numel() or not rounding:
            raise ValueError(
                "Reduction references require nonempty real terms and rounding semantics"
            )
        terms = values.detach().double()
        if not bool(terms.isfinite().all()):
            raise ValueError("Reduction reference terms must be finite")
        return cls(
            terms.sum(dim=dim, keepdim=keepdim),
            terms.abs().sum(dim=dim, keepdim=keepdim),
            values.shape[dim],
            rounding,
            accumulation_dtype,
            exact_terms,
            term_dtype,
        )

    def error_budget(
        self, expected: torch.Tensor, *, rtol: float, atol: float
    ) -> tuple[torch.Tensor, dict]:
        """Return a conservative component bound, not a proof of kernel accuracy.

        The term budget allows one epsilon of the materialized term dtype;
        output-interface atol/rtol are not multiplied across every term. The accumulation
        budget uses gamma(n-1) for the declared accumulation dtype and FP64,
        without assuming a particular parallel reduction tree. Exact input terms
        exclude pointwise term error, as in a collective over materialized inputs.
        Final rounding allows casts of both sums.
        A separate L2 guard in the caller prevents this worst-case bound from
        admitting systematic drift. Intrinsic errors are not proven here.
        """
        if (
            any(not math.isfinite(value) or value < 0 for value in (rtol, atol))
            or type(self.terms) is not int
            or self.terms < 1
            or not self.rounding
            or self.accumulation_dtype
            not in (torch.float16, torch.bfloat16, torch.float32, torch.float64)
            or type(self.exact_terms) is not bool
            or self.term_dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64)
            or self.total.dtype != torch.float64
            or self.sum_absolute_terms.dtype != torch.float64
            or self.total.shape != expected.shape
            or self.sum_absolute_terms.shape != expected.shape
            or self.total.device != expected.device
            or self.sum_absolute_terms.device != expected.device
            or not bool(self.total.isfinite().all())
            or not bool(self.sum_absolute_terms.isfinite().all())
            or not bool((self.sum_absolute_terms >= self.total.abs()).all())
            or not torch.equal(self.total.to(expected.dtype), expected)
        ):
            raise ValueError("Invalid or mismatched independent reduction reference")

        def gamma(dtype: torch.dtype) -> float:
            nu = (self.terms - 1) * torch.finfo(dtype).eps / 2
            if nu >= 1:
                raise ValueError("Reduction is too long for the declared accumulation bound")
            return nu / (1 - nu)

        gamma32, gamma64 = gamma(torch.float32), gamma(torch.float64)
        gamma_accumulation = gamma(self.accumulation_dtype)
        # Inflate the FP64 sum-of-magnitudes for its own rounding uncertainty.
        scale = self.sum_absolute_terms / (1 - gamma64)
        term_error = (
            torch.zeros_like(scale)
            if self.exact_terms
            else torch.finfo(self.term_dtype).eps * scale
        )
        accumulation = gamma_accumulation * (scale + term_error) + gamma64 * scale
        before_cast = term_error + accumulation
        cast_error = torch.finfo(expected.dtype).eps * (self.total.abs() + before_cast)
        budget = before_cast + cast_error
        if not bool(budget.isfinite().all()):
            raise ValueError("Reduction error budget is not finite")
        nonzero = self.total != 0
        condition = (
            float((scale[nonzero] / self.total[nonzero].abs()).max())
            if bool(nonzero.any())
            else None
        )
        return budget, {
            "policy": (
                "fp32_sum_term_epsilon_and_l2:v2"
                if self.accumulation_dtype == torch.float32 and not self.exact_terms
                else "declared_dtype_sum_term_epsilon_and_l2:v2"
            ),
            "term_count": self.terms,
            "accumulation_dtype": str(self.accumulation_dtype),
            "exact_terms": self.exact_terms,
            "reference_dtype": "torch.float64",
            "rounding": self.rounding,
            "term_dtype": str(self.term_dtype),
            "term_rtol": 0 if self.exact_terms else torch.finfo(self.term_dtype).eps,
            "term_atol": 0,
            "gamma_fp32": gamma32,
            "gamma_accumulation": gamma_accumulation,
            "gamma_fp64": gamma64,
            "max_sum_absolute_terms": float(scale.max()),
            "max_condition_number_nonzero": (
                condition if condition is None or math.isfinite(condition) else str(condition)
            ),
            "zero_sum_components": int((~nonzero).sum()),
            "max_allowed_absolute_error": float(budget.max()),
        }
