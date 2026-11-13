# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Parser and typed representation (:class:`InferenceShardSpec`) for the
``--inference-shards`` shard-layout string."""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Union

VALID_INT_KEYS = ("tp", "pp", "ep", "expt_tp", "dp", "cp")
VALID_ROLES = ("prefill", "decode")
VALID_KEYS = (*VALID_INT_KEYS, "role")


@dataclass(frozen=True)
class InferenceShardSpec:
    """One inference shard's parallelism -- the canonical shard-layout type.

    Frozen and self-validating: ``expt_tp`` defaults to ``tp`` (resolved at
    construction), the expert decomposition is checked to tile within the shard,
    and ``role`` (``"prefill"`` / ``"decode"`` for disaggregation, else ``None``)
    is validated. A list of these, produced by :func:`normalize_shard_specs`, is
    what every inference-shard consumer (the PG builder, the disaggregation
    setup) operates on.
    """

    tp: int = 1
    pp: int = 1
    ep: int = 1
    dp: int = 1
    cp: int = 1
    expt_tp: Optional[int] = None
    role: Optional[str] = None

    def __post_init__(self):
        if self.role is not None and self.role not in VALID_ROLES:
            raise ValueError(f"role must be one of {VALID_ROLES} or None; got {self.role!r}")
        # cp is accepted in the spec for forward-compatibility, but inference
        # shards don't context-parallelize (the KV reshard models layer x head,
        # not the sequence dim), so reject anything but cp=1 with a clear error.
        if self.cp != 1:
            raise ValueError(
                f"context parallelism is not supported for inference shards; cp must be 1, "
                f"got cp={self.cp}"
            )
        # Resolve expt_tp's default (tp) on this frozen instance.
        if self.expt_tp is None:
            object.__setattr__(self, "expt_tp", self.tp)
        # Expert decomposition must tile cleanly within the shard.
        if self.world_size % (self.expt_tp * self.ep * self.pp) != 0:
            raise ValueError(
                f"shard {self} has tp*pp*dp={self.world_size} but expt_tp*ep*pp="
                f"{self.expt_tp * self.ep * self.pp} does not divide it; "
                f"choose compatible sizes."
            )

    @property
    def world_size(self) -> int:
        """Number of ranks this shard occupies (``tp * pp * dp``)."""
        return self.tp * self.pp * self.dp

    def to_dict(self) -> dict:
        """Plain-dict form (e.g. for serialization or external consumers)."""
        d = {"tp": self.tp, "pp": self.pp, "ep": self.ep, "dp": self.dp, "expt_tp": self.expt_tp}
        if self.role is not None:
            d["role"] = self.role
        return d


@dataclass(frozen=True)
class InferenceShardAssignment:
    """Resolved shard membership for one global rank.

    Attributes:
        index: Zero-based index of the containing shard.
        spec: Validated specification for the containing shard.
        rank_offset: First global rank belonging to the shard.
        shard_count: Total number of shards in the layout.
    """

    index: int
    spec: InferenceShardSpec
    rank_offset: int
    shard_count: int


def normalize_shard_specs(
    shards: Union[str, Sequence["InferenceShardSpec"], Sequence[dict]], world_size: int
) -> List["InferenceShardSpec"]:
    """Coerce the public shard-spec input (a spec string, a list of
    :class:`InferenceShardSpec`, or a list of raw dicts) into the validated
    list of :class:`InferenceShardSpec` the shard builders consume."""
    if isinstance(shards, str):
        return parse_inference_shards_spec(shards, world_size)
    out: List[InferenceShardSpec] = [
        s if isinstance(s, InferenceShardSpec) else InferenceShardSpec(**dict(s)) for s in shards
    ]
    return _finalize_and_validate(out, world_size)


def resolve_inference_shard(
    shards: Union[str, Sequence["InferenceShardSpec"], Sequence[dict]], world_size: int, rank: int
) -> InferenceShardAssignment:
    """Resolve a global rank to its shard in a validated layout.

    Args:
        shards: Any shard layout accepted by :func:`normalize_shard_specs`.
        world_size: Total number of ranks partitioned by the layout.
        rank: Global rank to resolve.

    Returns:
        The rank's shard assignment, including its index and rank offset.

    Raises:
        ValueError: If ``rank`` is outside ``[0, world_size)``.
    """
    if not 0 <= rank < world_size:
        raise ValueError(f"rank must be in [0, {world_size}); got {rank}")

    specs = normalize_shard_specs(shards, world_size)
    rank_offset = 0
    for index, spec in enumerate(specs):
        if rank < rank_offset + spec.world_size:
            return InferenceShardAssignment(
                index=index, spec=spec, rank_offset=rank_offset, shard_count=len(specs)
            )
        rank_offset += spec.world_size

    # ``normalize_shard_specs`` guarantees that the specs exactly partition
    # ``world_size``, so a valid rank must have been found above.
    raise RuntimeError(f"rank {rank} was not assigned to an inference shard")


def spec_declares_disaggregation(spec_str: str | None) -> bool:
    """Whether a shard spec tags any shard with a ``role=`` (prefill/decode).

    A role tag is what marks the layout as a prefill->decode handoff rather
    than plain multi-shard / data-parallel inference. Cheap and world_size-
    free, so it can be checked at arg-validation time; full parsing +
    validation is :func:`parse_inference_shards_spec`.
    """
    if not spec_str:
        return False
    return any(
        kv.strip().startswith("role=")
        for shard in spec_str.replace("+", ";").split(";")
        for kv in shard.split(",")
    )


def parse_inference_shards_spec(spec_str: str, world_size: int) -> List[InferenceShardSpec]:
    """Parse + validate the ``--inference-shards`` string.

    Args:
        spec_str: Raw CLI value, e.g. ``"tp=2,dp=1+tp=1,dp=2"`` or with
            disaggregation roles ``"tp=2,role=prefill+tp=1,role=decode"``.
        world_size: Total number of ranks. Specs must partition it
            exactly (no idle ranks; see note below).

    Returns:
        List of :class:`InferenceShardSpec`, one per shard. Order matches the
        input (left-to-right corresponds to ascending ``rank_offset``).

    Raises:
        AssertionError: on syntax errors, unknown keys, or a rank-count
            mismatch with ``world_size``. Idle ranks are rejected to keep the
            partition explicit — any world-collective consumer must be able to
            enumerate every rank's shard membership from the parsed list alone.
        ValueError: on an expert-grid mismatch within a shard (raised by
            :class:`InferenceShardSpec`).
    """
    parsed: List[InferenceShardSpec] = []
    # ``+`` is convenient from shell recipes where ``;`` would otherwise
    # be treated as a command terminator. Normalize before splitting.
    shards_raw = spec_str.replace("+", ";")
    for shard_str in shards_raw.split(";"):
        shard_str = shard_str.strip()
        if not shard_str:
            continue
        kwargs: dict = {}
        for kv in shard_str.split(","):
            kv = kv.strip()
            if not kv:
                continue
            if "=" not in kv:
                raise AssertionError(
                    f"Bad --inference-shards spec entry {kv!r}: expected key=value."
                )
            k, v = kv.split("=", 1)
            k = k.strip()
            v = v.strip()
            if k not in VALID_KEYS:
                raise AssertionError(
                    f"Unknown key {k!r} in --inference-shards "
                    f"(allowed: {','.join(VALID_KEYS)})."
                )
            if k == "role":
                role = v.lower()
                assert role in VALID_ROLES, (
                    f"Unknown role {v!r} in --inference-shards "
                    f"(allowed: {','.join(VALID_ROLES)})."
                )
                kwargs[k] = role
            else:
                kwargs[k] = int(v)
        parsed.append(InferenceShardSpec(**kwargs))

    return _finalize_and_validate(parsed, world_size)


def _finalize_and_validate(
    specs: List["InferenceShardSpec"], world_size: int
) -> List["InferenceShardSpec"]:
    """Assert the shards partition the world exactly.

    Shared by the string parser and the object path (:func:`normalize_shard_specs`).
    Per-shard defaults and expert-grid validation live in
    :class:`InferenceShardSpec`; this only enforces the cross-shard total. Idle
    ranks are rejected so any world-collective consumer can enumerate every
    rank's shard membership from the list alone.
    """
    assert specs, "--inference-shards was empty."
    total_ranks = sum(s.world_size for s in specs)
    assert total_ranks == world_size, (
        f"--inference-shards consumes {total_ranks} ranks but world size is "
        f"{world_size}; specs must partition the full world."
    )
    return specs
