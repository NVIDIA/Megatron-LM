# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Declare post-step L2 norm constraints on the rows of trainable parameters."""

import math
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace

import torch


class ParamNormRule(ABC):
    """Map each vector's full L2 norm to its scale, independently of other vectors.

    The input holds only the vectors represented on this model-parallel partition.
    Return multipliers with the same shape and device.
    """

    @abstractmethod
    def validation_key(self) -> tuple:
        """Return the multiplier settings that peers must agree on at setup."""

    @abstractmethod
    def multipliers(self, norms: torch.Tensor) -> torch.Tensor:
        """Return one multiplier per complete vector norm."""


class EmbeddingMinNormStats:
    """Interval totals for the input-embedding floor on one reporting optimizer."""

    def __init__(self) -> None:
        self._totals: torch.Tensor | None = None

    def record(self, norms: torch.Tensor, rows_to_scale: torch.Tensor, target_norm: float) -> None:
        """Accumulate rows scaled, rows checked, squared norm delta, and exact-zero rows."""
        if self._totals is None:
            self._totals = norms.new_zeros((4,), dtype=torch.float64)
        self._totals[0].add_(rows_to_scale.sum())
        self._totals[1].add_(norms.numel())
        norm_delta = torch.where(rows_to_scale, target_norm - norms, 0.0)
        self._totals[2].add_(norm_delta.square().sum())
        self._totals[3].add_((norms == 0.0).sum())

    def consume(self) -> torch.Tensor | None:
        """Return this interval's totals and start a fresh interval on the next update."""
        totals = self._totals
        self._totals = None
        return totals


@dataclass(frozen=True)
class EmbeddingMinNorm(ParamNormRule):
    """Rescale nonzero rows below the threshold to twice the threshold."""

    threshold: float
    stats: EmbeddingMinNormStats | None = field(
        default=None, kw_only=True, compare=False, repr=False
    )

    def __post_init__(self) -> None:
        self.validate_threshold(self.threshold)

    @staticmethod
    def validate_threshold(threshold: float) -> None:
        """Require a finite, strictly positive embedding norm threshold."""
        if not math.isfinite(threshold) or threshold <= 0.0:
            raise ValueError("embedding_min_norm must be a finite positive value")

    def validation_key(self) -> tuple:
        """Only the threshold affects which multipliers each rank applies."""
        return (self.threshold,)

    def multipliers(self, norms: torch.Tensor) -> torch.Tensor:
        """Return one multiplier per row, recording pre-projection norms on the reporting copy."""
        rows_to_scale = (norms > 0.0) & (norms < self.threshold)
        target_norm = 2.0 * self.threshold
        if self.stats is not None:
            self.stats.record(norms, rows_to_scale, target_norm)
        return torch.where(
            rows_to_scale, target_norm / norms.clamp_min(torch.finfo(norms.dtype).tiny), 1.0
        )


@dataclass(frozen=True, eq=False)
class ParamNormConstraint:
    """Declare a norm rule for the rows of a model parameter."""

    parameter: torch.nn.Parameter
    rule: ParamNormRule
    name: str = field(default="", kw_only=True)

    @classmethod
    def rows(cls, parameter: torch.nn.Parameter, rule: ParamNormRule) -> "ParamNormConstraint":
        """Apply the rule to each row of the parameter."""
        return cls(parameter, rule)


class ParamNormRecipe(ABC):
    """Select the parameters and row norm rules for a model."""

    @abstractmethod
    def __call__(self, model_chunk: torch.nn.Module) -> Iterable[ParamNormConstraint]:
        """Return this chunk's constraints when the optimizer is constructed."""


class EmbeddingMinNormRecipe(ParamNormRecipe):
    """Apply the embedding floor to trainable input tables and their tied pipeline copies."""

    def __init__(self, threshold: float) -> None:
        self.rule = EmbeddingMinNorm(threshold)

    def __call__(self, model_chunk: torch.nn.Module) -> list[ParamNormConstraint]:
        return [
            ParamNormConstraint.rows(parameter, self.rule)
            for parameter in model_chunk.parameters()
            if parameter.requires_grad and getattr(parameter, "is_word_embedding_parameter", False)
        ]


def collect_param_norm_constraints(
    model_chunks: Sequence[torch.nn.Module], recipe: ParamNormRecipe
) -> list[ParamNormConstraint]:
    """Evaluate the recipe once per chunk and deduplicate declarations for shared parameters."""
    declarations = {}
    for index, chunk in enumerate(model_chunks):
        names = {p: n for n, p in chunk.named_parameters()}
        for constraint in recipe(chunk):
            parameter = constraint.parameter
            if parameter not in names or not parameter.requires_grad:
                raise ValueError(
                    f"A norm recipe must select a trainable parameter of its chunk: "
                    f"{names.get(parameter, '<outside chunk>')}"
                )
            name = names[parameter]
            if len(model_chunks) > 1:
                name = f"chunk{index}.{name}"
            previous = declarations.get(parameter)
            if previous is None:
                declarations[parameter] = replace(constraint, name=name)
            elif type(previous.rule) is not type(constraint.rule) or (
                previous.rule.validation_key() != constraint.rule.validation_key()
            ):
                raise ValueError(f"Conflicting norm constraints for {name}")
    return sorted(declarations.values(), key=lambda c: c.name)
