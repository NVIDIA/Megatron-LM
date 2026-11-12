# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Portions adapted from DynamicUSP at commit
# 1ff97a4ae0beb5e088fadb9e112265bdd194b24d (src/hot_switch_solver/profiles/usp_calib_v1.py; the
# table usp_calib_v1.json is profiles/calib_table_v1.json), BSD-3-Clause license.
# See ../DYNAMIC_USP_NOTICE.txt for the upstream copyright and license text.

"""DynamicUSP U8 all-to-all + ring costs for the original synthetic shape.

Default formulas and the synthetic 44-layer shape are preserved from upstream
1ff97a4a. See DYNAMIC_USP_NOTICE.txt for the upstream BSD-3-Clause attribution.
Cost-model selection does not change the training communication configuration.
An optional upstream-schema table replaces coefficients for this fixed profile;
it does not change its shape or make it a calibration of arbitrary architectures.
Stage 2 and upstream calibration tooling are not imported by this model.
"""

import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path

# The packaged table was measured on nodes of eight GPUs, with NVLink inside a node and
# InfiniBand between nodes (its "hardware" and "validity_domain.topology" fields): the U8
# all-to-all stays inside a node and the ring hops between nodes. The runtime file names no
# hardware for this profile and the formulas read no rank positions, so this number selects
# nothing; it states which placement the packaged coefficients describe.
_PACKAGED_TABLE_NVLINK_DOMAIN_SIZE = 8


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

    def __init__(self, geometry, *, allow_provisional=False, calibration=None) -> None:
        """Load bundled coefficients or an upstream-schema calibration table.

        ``calibration`` accepts a file path. Its coefficients retain the fixed
        44-layer, 24-Q-head, head-dimension-160 U8 all-to-all + ring model.
        """
        if allow_provisional is not True:
            raise ValueError("usp_calib_v1 requires explicit allow_provisional=True")
        self.geometry = geometry
        self.check_domain(geometry, None)
        if calibration is not None and not isinstance(calibration, (str, Path)):
            raise ValueError("usp_calib_v1 calibration must be a table file path")
        path = Path(__file__).with_suffix(".json") if calibration is None else Path(calibration)
        try:
            raw = path.read_bytes()
            table = json.loads(raw)
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
    def _validate_table(table):
        """Check coefficients consumed by the original fixed-shape formulas."""

        def field(*keys):
            value = table
            try:
                for key in keys:
                    if not isinstance(value, dict):
                        raise TypeError
                    value = value[key]
            except (KeyError, TypeError) as error:
                raise ValueError(
                    "Invalid DynamicUSP calibration field: " + ".".join(keys)
                ) from error
            return value

        def number(value, label, *, positive=False):
            if type(value) not in (int, float):
                raise ValueError("Invalid DynamicUSP calibration number: " + label)
            try:
                valid = math.isfinite(value) and (value > 0 if positive else value >= 0)
            except OverflowError:
                valid = False
            if not valid:
                bound = "positive" if positive else "nonnegative"
                raise ValueError("DynamicUSP calibration requires finite " + bound + " " + label)
            return value

        def coefficient(*keys, positive=False):
            return number(field(*keys), ".".join(keys), positive=positive)

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
        if len(tokens) < 2 or len(tokens) != len(rates):
            raise ValueError(
                "DynamicUSP calibration requires equal MoE LUT lengths of at least two"
            )
        for index, (token, rate) in enumerate(zip(tokens, rates)):
            number(token, f"compute.moe_rate_lut.tokens[{index}]", positive=True)
            number(rate, f"compute.moe_rate_lut.tflops[{index}]", positive=True)
        if tokens[-1] != 16384 or any(right <= left for left, right in zip(tokens, tokens[1:])):
            raise ValueError(
                "DynamicUSP calibration requires increasing MoE tokens ending at 16384"
            )

        beta = field("ring", "beta_by_chunk_bytes")
        if not isinstance(beta, dict):
            raise ValueError("DynamicUSP calibration requires ring.beta_by_chunk_bytes mapping")
        chunks = [key for key in beta if key.endswith("MB")]
        if not chunks:
            raise ValueError("DynamicUSP calibration requires at least one ring beta MB entry")
        for key in chunks:
            try:
                chunk_bytes = float(key[:-2]) * 1e6
            except ValueError as error:
                raise ValueError("Invalid DynamicUSP ring beta chunk: " + key) from error
            number(chunk_bytes, "ring.beta_by_chunk_bytes." + key + " bytes", positive=True)
            number(beta[key], "ring.beta_by_chunk_bytes." + key, positive=True)

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

    @lru_cache(maxsize=65536)
    def components(self, lengths, cp):
        """Original full-stack group costs, plus a forward/backward decomposition."""
        if type(cp) is not int or cp not in self.geometry.allowed_cp_sizes:
            raise ValueError("CP size is outside the configured DynamicUSP tiers")
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
        gamma_key = {8: "sp8_fa_cache", 16: "sp16_ring", 32: "sp32_ring"}[cp]
        gamma = self.table["global"]["gamma_comp"]["pure_compute"][gamma_key]
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

    def timing(self):
        """Describe the upstream block program and global iteration-tail constant."""
        return dict(
            iteration_tail_ms=self.table["global"]["C_global_clean_ms"],
            execution_program=[
                dict(
                    layer_count=self.layer_count,
                    forward=[dict(sync="cp", cost="block_f")] * self.layer_count,
                    backward=[dict(sync="cp", cost="block_b")] * self.layer_count,
                )
            ],
        )

    def clear(self):
        """Clear memoized upstream component costs before a new global batch."""
        self.components.cache_clear()

    def metadata(self):
        """Record the original U8 calibration and its synthetic-shape scope."""
        return dict(
            name=self.name,
            nvlink_domain_size=self.nvlink_domain_size,
            source_commit="1ff97a4ae0beb5e088fadb9e112265bdd194b24d",
            source_hashes=self.source_hashes,
            provisional=True,
            modeled_transport="U8 all-to-all + ring",
            execution_transport="determined by the training communication configuration",
            shape=(
                "upstream synthetic 44-layer shape, 24 Q heads, head_dim=160; "
                "not a Qwen calibration"
            ),
            calibration_path=self.calibration_path,
            calibration_sha256=self.calibration_sha256,
        )
