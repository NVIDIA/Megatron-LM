# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Portions adapted from DynamicUSP at commit
# 1ff97a4ae0beb5e088fadb9e112265bdd194b24d (src/hot_switch_solver: profiles/usp_calib_v1.py,
# gbs/config.py, objectives/pipeline.py and stage2/evaluator.py; the packaged table is
# profiles/calib_table_v1.json), BSD-3-Clause license.
# See ../DYNAMIC_USP_NOTICE.txt for the upstream copyright and license text.

"""DynamicUSP U8 all-to-all + ring costs for the original synthetic shape.

Default formulas and the synthetic 44-layer shape are preserved from upstream
1ff97a4a. See DYNAMIC_USP_NOTICE.txt for the upstream BSD-3-Clause attribution.
Cost-model selection does not change the training communication configuration.
An optional upstream-schema table replaces coefficients for this fixed profile;
it does not change its shape or make it a calibration of arbitrary architectures.
The optional ``solver`` section of a table (``solver_section``) holds the coefficients of
the scheduler: the Stage 1 sample weight, the Stage 1 time-balance factors and the
per-microbatch overhead of the step time. The step time is that of the upstream objective
at one pipeline stage plus the global constant of the table. For more than one pipeline
stage the model gives the forward and backward time of a group as the upstream objective
splits it (``pipeline_group_times``); the runtime planner replays the pipeline schedule.
The options of the upstream profile are fixed at their defaults: all 44 layers replayed, FA
cache on, communication terms on, four backward ring chunks, no ``c_seg`` term, and a
forward-KV exposed fraction of 0.87 for a ring of four and 0 for a ring of two.
``ring.fwdkv_exposure_frac_by_rho`` and the other table fields that the default formulas do
not use are not read.
The model also gives what Stage 2 reads from the upstream profile (the ``stage2_`` members):
the work of a sample, its load in a group of every CP size and the exact time of a group.
For a run that switches Stage 2 on and can form a ring tier, a CP size above eight, it also
gives the placement guard of the upstream profile for those tiers
(``stage2_guard_threshold``, ``stage2_guard_margin``), with the guard threshold of the
upstream profile, the constant 0.0, and its option ``guard_domain_factor`` fixed at the
default 0.4. Only that guard reads the table fields
``compute.in_train_overhead_factors.OF_fa_fwd`` and ``ring.in_train_bw_factor``: upstream
requires the two in every table; here a table without them loads unless Stage 2 is on and
a ring tier is usable.
"""

import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path

from ..runtime.config import _unique_object
from ..runtime.stage2 import _finite
from .solver_section import _PIPELINE_KEYS, DEFAULT_SAMPLE_WEIGHT, parse_solver_section

# The packaged table was measured on nodes of eight GPUs, with NVLink inside a node and
# InfiniBand between nodes (its "hardware" and "validity_domain.topology" fields): the U8
# all-to-all stays inside a node and the ring hops between nodes. The runtime file names no
# hardware for this profile and the formulas read no rank positions, so this number selects
# nothing; it states which placement the packaged coefficients describe.
_PACKAGED_TABLE_NVLINK_DOMAIN_SIZE = 8
# Schema version of the upstream calibration table that the formulas read.
_TABLE_VERSION = "v1.1"


def _reject_constant(value):
    raise ValueError(f"values must be finite numbers, got {value}")


class USPCalibV1CostModel:
    """Expose the original compute + U8 A2A + ring costs to the CPU planner.

    Forward/backward include the upstream synthetic block's compute and MoE.
    They are not attention-only times and must not receive additional MoE costs.
    """

    name = "usp_calib_v1"
    placement_invariant = True
    layer_count = 44
    supported_cp_sizes = (8, 16, 32)
    # Largest CP group this model prices; the scheduler forms no larger one.
    largest_cp_size = max(supported_cp_sizes)
    # Stage 2: a group of at most this size is of the reference tier, where the work of a
    # sample is priced. A larger group is a ring tier.
    stage2_reference_cp_size = 8
    # Forward and backward weights of the compute factors (gamma) behind the Stage 2 load of
    # a CP size.
    _GAMMA_BLEND_WEIGHTS = (1.0, 2.9)

    def __init__(self, geometry, *, allow_provisional=False, calibration=None) -> None:
        """Load the bundled coefficients, or an upstream-schema table from ``calibration``."""
        if allow_provisional is not True:
            raise ValueError("usp_calib_v1 requires explicit allow_provisional=True")
        self.geometry = geometry
        self.check_domain(geometry, None)
        path = Path(__file__).with_suffix(".json") if calibration is None else Path(calibration)
        try:
            raw = path.read_bytes()
            # NaN, Infinity and a key given twice in one object are rejected (ORIGIN.json).
            table = json.loads(
                raw, object_pairs_hook=_unique_object, parse_constant=_reject_constant
            )
        except (OSError, ValueError, UnicodeError) as error:
            raise ValueError(
                f"Cannot read DynamicUSP calibration {str(path)!r}: {error}"
            ) from error
        digest = hashlib.sha256(raw).hexdigest()
        if (
            calibration is None
            and digest != "d25989552ab9700f07261231fa13d66672d98684435fff57838442ca12c19cd7"
        ):
            raise ValueError("Frozen DynamicUSP calibration hash mismatch")
        self.table = table
        self._validate_table(self.table)
        try:
            self.solver = parse_solver_section(table, self.supported_cp_sizes)
            # Stage 2 switch and keep threshold of the table; off without them, as in
            # DynamicUSP. Its placement guard reads two more fields, on ring tiers only.
            self.stage2_enabled = self.solver.stage2_enabled
            self.stage2_keep_threshold = self.solver.stage2_keep_threshold
            self._load_stage2_guard()
        except ValueError as error:
            raise ValueError(f"{error} (coefficient file {str(path.resolve())!r})") from error
        # Stage 1 weight of a sample: A, B, C of the table, else DynamicUSP's published fit.
        self.sample_weight_is_default = self.solver.sample_weight is None
        self.sample_weight_coefficients = self.solver.sample_weight or DEFAULT_SAMPLE_WEIGHT
        # Stage 1 time-balance factors of the table; None keeps weight balance.
        self.tier_time_factors = self.solver.tier_time_factors
        # Added to the slowest group of a microbatch; it only moves the Stage 2 epsilon.
        self.stage2_global_constant_ms = float(table["global"]["C_global_clean_ms"])
        # The step terms of more than one pipeline stage, as the plan statistics record them:
        # ``default_terms`` names those the table does not give.
        self.pipeline_terms = dict(
            per_mb_overhead_ms=self.solver.per_mb_overhead_ms,
            compute_fwd_share=self.solver.compute_fwd_share,
            comm_fwd_share=self.solver.comm_fwd_share,
            default_terms=list(self.solver.pipeline_default_terms),
        )
        # A table given with ``calibration`` has no field for the topology it was measured
        # on, so the profile does not know it.
        self.nvlink_domain_size = (
            _PACKAGED_TABLE_NVLINK_DOMAIN_SIZE if calibration is None else None
        )
        self.calibration_path = str(path.resolve())
        self.calibration_sha256 = digest
        self.source_hashes = {
            "calib_table_v1.json": digest,
            "usp_calib_v1.py": "a2dbcff0099936e283b2386783861d7191c5319e1cf9e22a8bff30f33207329d",
        }

    @staticmethod
    def _validate_table(table, guard=False):
        """Check coefficients consumed by the original fixed-shape formulas.

        With ``guard`` also the two fields that only the placement guard of Stage 2 reads.
        """

        def field(*keys):
            value = table
            for key in keys:
                if not isinstance(value, dict) or key not in value:
                    raise ValueError("Invalid DynamicUSP calibration field: " + ".".join(keys))
                value = value[key]
            return value

        def number(value, label, *, positive=False):
            if type(value) not in (int, float):
                raise ValueError("Invalid DynamicUSP calibration number: " + label)
            if not _finite(value) or not (value > 0 if positive else value >= 0):
                bound = "positive" if positive else "nonnegative"
                raise ValueError("DynamicUSP calibration requires finite " + bound + " " + label)
            return value

        def coefficient(*keys, positive=False):
            return number(field(*keys), ".".join(keys), positive=positive)

        version = table.get("version") if isinstance(table, dict) else None
        if version != _TABLE_VERSION:
            raise ValueError(
                f'DynamicUSP calibration requires "version": "{_TABLE_VERSION}", got {version!r}'
            )
        for fit in ("U8_pre", "U8_post"):
            for key in ("floor_ms", "t0_ms", "bw_gbps"):
                coefficient("a2a", "fits", fit, key, positive=key == "bw_gbps")
        for fit in ("kv", "dkv"):
            for key in ("t0_ms", "bw_gbps"):
                coefficient("ring", "fits", fit, key, positive=key == "bw_gbps")
        for tier, gamma in (("sp8", "sp8_fa_cache"), ("sp16", "sp16_ring"), ("sp32", "sp32_ring")):
            coefficient("a2a", "in_train_inflation_by_tier", tier, positive=True)
            for direction in ("fwd", "bwd"):
                coefficient("global", "gamma_comp", "pure_compute", gamma, direction, positive=True)
        for direction in ("fwd", "bwd"):
            coefficient("compute", "fa_additive_law", direction, "P_tflops", positive=True)
        coefficient("compute", "moe_intrain_rate_tflops", positive=True)
        coefficient("global", "C_global_clean_ms")

        tokens = field("compute", "moe_rate_lut", "tokens")
        rates = field("compute", "moe_rate_lut", "tflops")
        if not isinstance(tokens, list) or not isinstance(rates, list):
            raise ValueError("DynamicUSP calibration requires MoE tokens/tflops lists")
        if not tokens or len(tokens) != len(rates):
            raise ValueError(
                "DynamicUSP calibration requires nonempty MoE LUT lists of equal length"
            )
        for index, (token, rate) in enumerate(zip(tokens, rates)):
            number(token, f"compute.moe_rate_lut.tokens[{index}]", positive=True)
            number(rate, f"compute.moe_rate_lut.tflops[{index}]", positive=True)
        if any(right <= left for left, right in zip(tokens, tokens[1:])):
            raise ValueError("DynamicUSP calibration requires strictly increasing MoE tokens")

        beta = field("ring", "beta_by_chunk_bytes")
        if not isinstance(beta, dict):
            raise ValueError("DynamicUSP calibration requires ring.beta_by_chunk_bytes mapping")
        chunks = [key for key in beta if key.endswith("MB")]
        if not chunks:
            raise ValueError("DynamicUSP calibration requires at least one ring beta MB entry")
        coordinates = set()
        for key in chunks:
            try:
                chunk_bytes = float(key[:-2]) * 1e6
            except ValueError as error:
                raise ValueError("Invalid DynamicUSP ring beta chunk: " + key) from error
            number(chunk_bytes, "ring.beta_by_chunk_bytes." + key + " bytes", positive=True)
            if chunk_bytes in coordinates:
                raise ValueError("DynamicUSP ring beta chunk keys must be unique: " + key)
            coordinates.add(chunk_bytes)
            number(beta[key], "ring.beta_by_chunk_bytes." + key, positive=True)
        if guard:
            factors = field("compute", "in_train_overhead_factors", "OF_fa_fwd")
            if not isinstance(factors, list) or not factors:
                raise ValueError("DynamicUSP calibration requires a nonempty OF_fa_fwd list")
            for index, factor in enumerate(factors):
                label = f"compute.in_train_overhead_factors.OF_fa_fwd[{index}]"
                number(factor, label, positive=True)
            coefficient("ring", "in_train_bw_factor", positive=True)

    def check_domain(self, geometry, config=None):
        """Validate the parent geometry and CP widths against the upstream profile."""
        if geometry != self.geometry or geometry.world_size not in self.supported_cp_sizes:
            raise ValueError("usp_calib_v1 requires a DPxCP domain of 8, 16 or 32 ranks")
        if not geometry.allowed_cp_sizes or any(
            cp not in self.supported_cp_sizes for cp in geometry.allowed_cp_sizes
        ):
            raise ValueError("usp_calib_v1 only supports CP8/16/32 (minimum CP8)")

    @staticmethod
    def _curve(fit, nbytes):
        return max(fit["floor_ms"], fit["t0_ms"] + nbytes / 1e6 / fit["bw_gbps"])

    def _gamma(self, cp):
        """Forward and backward compute factors of the table for groups of ``cp`` ranks."""
        key = {8: "sp8_fa_cache", 16: "sp16_ring", 32: "sp32_ring"}[cp]
        return self.table["global"]["gamma_comp"]["pure_compute"][key]

    @lru_cache(maxsize=65536)
    def components(self, lengths, cp):
        """Original full-stack group costs, plus a forward/backward decomposition."""
        if type(cp) is not int or cp not in self.geometry.allowed_cp_sizes:
            raise ValueError("CP size is outside the configured DynamicUSP tiers")
        return self._priced(lengths, cp)

    def _priced(self, lengths, cp):
        """``_components`` with an arithmetic error of the formulas reported as ValueError."""
        try:
            return self._components(lengths, cp)
        except ArithmeticError as error:
            # Upstream's formula preflight is not run when a table is loaded: a table whose
            # formulas divide by zero for this group is reported here.
            raise ValueError("Invalid DynamicUSP cost prediction") from error

    def _components(self, lengths, cp):
        """Group costs at any CP size of the profile, also one this run cannot form."""
        if not lengths or len(lengths) > 64 or any(type(n) is not int or n <= 0 for n in lengths):
            raise ValueError("One to 64 positive logical sequence lengths are required")
        total = sum(lengths)
        squares = sum(n * n for n in lengths)
        ring_size = max(1, cp // 8)
        mode = f"sp{cp}"
        compute = self.table["compute"]
        rate = compute["moe_intrain_rate_tflops"] * 1e9
        routed = float(total) / cp
        if routed < 16384:
            lut = compute["moe_rate_lut"]
            xs, ys = lut["tokens"], lut["tflops"]
            tokens, offline_rate = max(routed, xs[0]), ys[-1]
            for index in range(len(xs) - 1):
                if tokens <= xs[index + 1]:
                    fraction = (math.log(tokens) - math.log(xs[index])) / (
                        math.log(xs[index + 1]) - math.log(xs[index])
                    )
                    offline_rate = ys[index] + fraction * (ys[index + 1] - ys[index])
                    break
            rate *= offline_rate / ys[-1]
        # Upstream synthetic shape: 24 Q heads / U8, head_dim=160, 44 layers.
        fa_flops = 44 * 4.0 * 3 * 160 * float(squares) / ring_size
        moe_flops = 44 * (float(total) / cp) * 5.306e8
        law = compute["fa_additive_law"]
        gamma = self._gamma(cp)
        compute_f = gamma["fwd"] * (fa_flops / (law["fwd"]["P_tflops"] * 1e9) + moe_flops / rate)
        compute_b = gamma["bwd"] * (
            2.5 * fa_flops / (law["bwd"]["P_tflops"] * 1e9) + 3.0 * moe_flops / rate
        )
        a2a = self.table["a2a"]
        nbytes = 24 * 160 * 2 * (float(total) / cp)
        invocation = 3 * self._curve(a2a["fits"]["U8_pre"], nbytes) + self._curve(
            a2a["fits"]["U8_post"], nbytes
        )
        inflation = a2a["in_train_inflation_by_tier"][mode]
        # Defaults: all 44 layers replay; FA cache on; four backward chunks.
        a2a_f, a2a_b = 44 * invocation * inflation, 88 * invocation * inflation
        ring_f = ring_b = 0.0
        if ring_size > 1:
            ring = self.table["ring"]
            hop_bytes = 1920 * float(total) / ring_size
            chunk_bytes = hop_bytes / 4
            beta = sorted(
                (float(key[:-2]) * 1e6, float(value))
                for key, value in ring["beta_by_chunk_bytes"].items()
                if key.endswith("MB")
            )
            amplification = min(beta, key=lambda pair: abs(math.log(pair[0] / chunk_bytes)))[1]
            dkv = ring["fits"]["dkv"]
            ring_b = (
                44 * 4 * (amplification * (dkv["t0_ms"] + chunk_bytes / 1e6 / dkv["bw_gbps"]) / 4)
            )
            if ring_size >= 4:
                kv = ring["fits"]["kv"]
                ring_f = 0.87 * (kv["t0_ms"] + hop_bytes / 1e6 / kv["bw_gbps"]) * 44
        result = dict(
            compute_ms=compute_f + compute_b,
            a2a_ms=132 * invocation * inflation,
            ring_ms=ring_b + ring_f,
            forward_ms=compute_f + a2a_f + ring_f,
            backward_ms=compute_b + a2a_b + ring_b,
        )
        if any(not math.isfinite(value) or value < 0 for value in result.values()):
            raise ValueError("Invalid DynamicUSP cost prediction")
        return result

    def forward(self, lengths, cp, *, ranks=None):
        """Return the upstream proxy mean forward time per synthetic layer."""
        return self.components(tuple(lengths), cp)["forward_ms"] / self.layer_count

    def backward(self, lengths, cp, *, ranks=None):
        """Return the upstream proxy mean backward time per synthetic layer."""
        return self.components(tuple(lengths), cp)["backward_ms"] / self.layer_count

    def program_costs(self, lengths, cp, *, ranks=None):
        """Return named forward/backward block costs for the timeline."""
        return dict(block_f=self.forward(lengths, cp), block_b=self.backward(lengths, cp))

    def pipeline_group_times(self, lengths, cp):
        """Whole-model ``(forward_ms, backward_ms)`` of one group for the pipeline replay.

        The split of the upstream step-time objective: ``solver.pipeline.compute_fwd_share``
        of the compute time and ``solver.pipeline.comm_fwd_share`` of the all-to-all and
        ring time are forward time, the rest is backward time. ``forward`` and ``backward``
        split a group with the formulas of the profile instead (one pipeline stage).
        """
        parts = self.components(tuple(lengths), cp)
        compute, communication = parts["compute_ms"], parts["a2a_ms"] + parts["ring_ms"]
        compute_share, comm_share = self.solver.compute_fwd_share, self.solver.comm_fwd_share
        forward = compute * compute_share + communication * comm_share
        backward = compute * (1.0 - compute_share) + communication * (1.0 - comm_share)
        return forward, backward

    def timing(self):
        """Describe the upstream block program and global iteration-tail constant.

        As in the upstream objective at one pipeline stage, only the ranks of one group wait
        for each other, and ``solver.pipeline.per_mb_overhead_ms`` is added per microbatch.
        """
        forward = [dict(sync="cp", cost="block_f")] * self.layer_count
        if self.solver.per_mb_overhead_ms > 0:
            forward = [dict(sync="local", duration_ms=self.solver.per_mb_overhead_ms)] + forward
        return dict(
            iteration_tail_ms=self.table["global"]["C_global_clean_ms"],
            first_forward_dp_cp_sync=False,
            execution_program=[
                dict(forward=forward, backward=[dict(sync="cp", cost="block_b")] * self.layer_count)
            ],
        )

    def derive_tier_time_factors(self, reference_tokens, cp_sizes=None):
        """Derive a ``solver.tier_time_factors`` table from the group times of this model.

        One sample of ``reference_tokens`` tokens is priced alone in a group of every CP size
        (``cp_sizes``; default: the allowed sizes of the run). The factor of a size is that
        group time times the size, relative to the smallest size (the calibration skill).
        """
        sizes = tuple(sorted(self.geometry.allowed_cp_sizes if cp_sizes is None else cp_sizes))
        if not sizes:
            raise ValueError("cp_sizes must not be empty")
        if type(reference_tokens) is not int or reference_tokens <= 0:
            raise ValueError("reference_tokens must be a positive integer")
        per_unit_weight = {}
        for size in sizes:
            parts = self.components((reference_tokens,), size)
            total = parts["compute_ms"] + parts["a2a_ms"] + parts["ring_ms"]
            per_unit_weight[size] = total * size
        base = per_unit_weight[sizes[0]]
        if not base > 0:
            raise ValueError("The model gives a non-positive time for the reference sample")
        return {size: per_unit_weight[size] / base for size in sizes}

    # ---------------------------------------------------------------- Stage 2 hooks
    def _stage2_load_factor(self, cp):
        """Compute time per unit of work in a group of ``cp`` ranks, relative to the reference."""
        forward, backward = self._GAMMA_BLEND_WEIGHTS

        def blend(size):
            gamma = self._gamma(size)
            return (gamma["fwd"] * forward + gamma["bwd"] * backward) / (forward + backward)

        return blend(cp) / blend(self.stage2_reference_cp_size)

    def stage2_sample_loads(self, length, cp_sizes):
        """Work of one sample and the load it adds to a group of every size in ``cp_sizes``.

        The work is the reference size times the compute time, without communication, of the
        sample alone in a reference group: it is priced there also when the run cannot form
        such a group. The load at a size is that work per rank, times the compute factor of
        the size relative to the reference size.
        """
        reference = self.stage2_reference_cp_size
        work = reference * self._priced((length,), reference)["compute_ms"]
        if not math.isfinite(work) or work <= 0.0:
            raise ValueError("Invalid DynamicUSP sample work")
        return work, {cp: self._stage2_load_factor(cp) * work / cp for cp in cp_sizes}

    def stage2_empty_group_load(self, cp):
        """Load of a group of ``cp`` ranks before its first sample."""
        return 0.0

    def stage2_group_time(self, lengths, cp):
        """Exact time of one group for one microbatch: compute, all-to-all and ring.

        For at most 64 sequences and a CP size of the run, as every group time of this model.
        As upstream rejects such a profile result, a compute time that is not positive or a
        negative communication time raises ValueError.
        """
        parts = self.components(tuple(lengths), cp)
        compute, a2a, ring = parts["compute_ms"], parts["a2a_ms"], parts["ring_ms"]
        if compute <= 0.0 or a2a < 0.0 or ring < 0.0:
            raise ValueError("Invalid DynamicUSP cost prediction")
        return compute + a2a + ring

    # ------------------------------------------------------------------------------------
    # Ring tiers only (the CP sizes above ``stage2_reference_cp_size``): the placement guard
    # of Stage 2. Upstream's ``guard_threshold`` and the default of its option
    # ``guard_domain_factor`` are code constants, not keys of the coefficient file.
    _GUARD_THRESHOLD = 0.0
    _GUARD_DOMAIN_FACTOR = 0.4
    # None: this run has no guard (Stage 2 is off, or no CP size of the run is a ring tier).
    stage2_guard_threshold = None
    _stage2_guard_floors = None

    def _load_stage2_guard(self):
        """Give the model its guard when Stage 2 is on and the run can form a ring tier.

        Only then are the two guard fields of the table read and checked. The floor of a ring
        tier, the least mean segment that passes, is ``_GUARD_DOMAIN_FACTOR`` times upstream's
        conservative ring-overlap segment (``_ring_overlap_min_seg_tokens``): the ring size
        times the length of a segment whose forward attention, at the in-train rate, takes as
        long as passing its key-value bytes at the in-train ring bandwidth.
        """
        reference = self.stage2_reference_cp_size
        if not self.stage2_enabled or max(self.geometry.allowed_cp_sizes) <= reference:
            return
        try:
            self._validate_table(self.table, guard=True)
        except ValueError as error:
            # Say why a table that loads with Stage 2 off is refused with it on.
            raise ValueError(f"{error}, which the ring guard of Stage 2 reads") from error
        compute, ring = self.table["compute"], self.table["ring"]
        # Upstream's arithmetic, operation by operation. 1920 key-value bytes per token over
        # 4 x 3 Q heads x head dimension 160 of the synthetic shape is 1.0.
        try:
            rate = float(compute["fa_additive_law"]["fwd"]["P_tflops"]) * 1e12
            rate /= min(compute["in_train_overhead_factors"]["OF_fa_fwd"])
            bandwidth = float(ring["fits"]["kv"]["bw_gbps"]) * 1e9
            bandwidth *= float(ring["in_train_bw_factor"])
            floors = {
                cp: self._GUARD_DOMAIN_FACTOR
                * ((cp // 8) * (1920 / (4.0 * 3 * 160)) * rate / bandwidth)
                for cp in self.supported_cp_sizes
                if cp > reference
            }
        except ArithmeticError:
            floors = {}
        if not floors or not all(map(math.isfinite, floors.values())):
            # Upstream's check of its formulas at load rejects such a table as well.
            raise ValueError("DynamicUSP calibration gives no finite Stage 2 guard floor")
        self._stage2_guard_floors = floors
        self.stage2_guard_threshold = self._GUARD_THRESHOLD

    def stage2_guard_margin(self, lengths, cp):
        """Guard margin in tokens of a ring tier of ``cp`` ranks that holds samples of ``lengths``.

        Upstream's rule: the mean segment of the group, the sum of the squared lengths over
        the sum of the lengths (0 for a group without a sample), minus the floor of the size.
        ``lengths`` are logical lengths, as for every cost quantity of Stage 2, which asks
        for ring tiers only and checks that the margin is a finite number.
        """
        total = sum(lengths)
        mean_segment = float(sum(n * n for n in lengths)) / float(total) if total else 0.0
        return mean_segment - self._stage2_guard_floors[cp]

    def clear(self):
        """Clear memoized upstream component costs before a new global batch."""
        self.components.cache_clear()

    def metadata(self):
        """Record the calibration table in use, the source commit and the solver settings."""
        solver = dict(
            sample_weight=dict(
                zip("ABC", self.sample_weight_coefficients), default=self.sample_weight_is_default
            ),
            tier_time_factors=self.tier_time_factors,
            pipeline={key: self.pipeline_terms[key] for key in _PIPELINE_KEYS},
        )
        if self.stage2_enabled:
            # Recorded only when the table switches Stage 2 on.
            solver["stage2"] = dict(
                enabled=True,
                keep_threshold=self.stage2_keep_threshold,
                reference_cp_size=self.stage2_reference_cp_size,
            )
        return dict(
            name=self.name,
            nvlink_domain_size=self.nvlink_domain_size,
            source_commit="1ff97a4ae0beb5e088fadb9e112265bdd194b24d",
            source_hashes=self.source_hashes,
            calibration_path=self.calibration_path,
            calibration_sha256=self.calibration_sha256,
            solver=solver,
        )
