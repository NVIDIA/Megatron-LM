"""Frozen CPU prediction kernel; provenance is recorded in ORIGIN.json."""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import math
from functools import lru_cache

from .qwen35_proxy import (
    FEATURES,
    Qwen35ProxyPredictor,
    execution_timing,
    load_non_attention_profile,
)


class Qwen35CoreCostModel:
    """Three calibrated GDN modules + two new FA cores and measured FA remainders.

    The injected callback returns a ``forward_ms``/``backward_ms`` mapping. This
    module imports no attention predictor or scheduler. A caller must supply a
    paired measured FA remainder; model-minus-model offsets are rejected.
    """

    name = "qwen35_gb200_new_core"
    layer_count = 5
    placement_invariant = True  # Valid only within the checked 32-rank NVL72 domain.

    def __init__(self, core_predictor, *, fa_remainder, non_attention=None):
        if not callable(core_predictor):
            raise TypeError("core_predictor must be callable")
        if fa_remainder.get("status") != "PAIRED_FA_REMAINDER_PASS":
            raise ValueError("a measured paired FA module-minus-core input is required")
        if (
            fa_remainder.get("hardware") != "gb200"
            or fa_remainder.get("tokens_per_rank") != 16384
            or fa_remainder.get("geometry") != [32, 2, 256]
        ):
            raise ValueError("FA remainder scope requires GB200 / local16K / 32,2,256")
        remainder = fa_remainder.get("cost_ms", {})
        if set(remainder) != {"forward", "backward"} or any(
            not math.isfinite(float(v)) or v < 0 for v in remainder.values()
        ):
            raise ValueError("FA remainder needs finite nonnegative forward and backward costs")
        if not fa_remainder.get("source_hashes"):
            raise ValueError("FA remainder must identify measured source records")
        self.core_predictor = core_predictor
        self.fa_remainder = copy.deepcopy(fa_remainder)
        self.non_attention = (
            copy.deepcopy(non_attention)
            if non_attention is not None
            else load_non_attention_profile("gb200-ep4-local16384-7319138")
        )
        if self.non_attention.get("profile") != "gb200-ep4-local16384-7319138":
            raise ValueError("this composition requires the measured GB200 EP4/local16K profile")
        if self.non_attention.get("status") != "PAIRED_MODULE_PROFILE_PASS":
            raise ValueError("a paired measured non-attention profile is required")
        other = self.non_attention.get("cost_ms", {})
        if set(other) != {
            "layer_forward",
            "layer_backward",
            "microbatch_outer",
            "iteration_tail",
        } or any(
            not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in other.values()
        ):
            raise ValueError("non-attention inputs must be fixed finite nonnegative measured times")
        self.gdn = Qwen35ProxyPredictor(
            profile="gb200-parent32-native-local16384-7319138", allow_provisional=True
        )
        self.source_hashes = {
            "gdn_calibration": self.gdn.calibration_sha256,
            "non_attention_input": hashlib.sha256(
                json.dumps(self.non_attention, sort_keys=True).encode()
            ).hexdigest(),
            "fa_remainder_input": hashlib.sha256(
                json.dumps(self.fa_remainder, sort_keys=True).encode()
            ).hexdigest(),
        }
        self._guide_table = {}

    def _gdn_costs(self, lengths, cp):
        stored = [((n + 63) // 64 * 64 + 2 * cp - 1) // (2 * cp) * (2 * cp) for n in lengths]
        if sum(stored) > cp * 16384:
            raise ValueError("pack exceeds physical capacity")
        stored[-1] += cp * 16384 - sum(stored)
        cu = list(itertools.accumulate(stored, initial=0))
        costs = {}
        for phase, key in (("forward_gpu_ms", "gdn_f"), ("backward_gpu_ms", "gdn_b")):
            coefficients = self.gdn.data["gdn"]["plain"][phase]
            features = self.gdn._features(cu, cp, phase, False, 16384)
            costs[key] = sum(coefficients[name] * value for name, value in zip(FEATURES, features))
        return costs

    def prepare_guide(self, cp_sizes):
        """Small predictor-query table for proposal ordering, NOT final scoring.

        Interpolation uses equal/skewed packs at several sequence counts. No measured
        non-attention value or frozen attention coefficient is adjusted here.
        The caller reports this one-time preparation separately from planning.
        """
        for cp in cp_sizes:
            if cp in self._guide_table:
                continue
            capacity = cp * 16384
            levels = {}
            for count in (1, 2, 3, 4, 6, 8, 16, 32, 64):
                packs = []
                for divisor in (64, 4, 2, 4 / 3, 1):
                    length = max(1, int(capacity // (divisor * count)))
                    while (
                        count * (((length + 63) // 64 * 64 + 2 * cp - 1) // (2 * cp) * (2 * cp))
                        > capacity
                    ):
                        length -= 64
                    packs.append((length,) * count)
                if count > 1:
                    short = min(1024, max(1, capacity // (8 * count)))
                    physical_short = ((short + 63) // 64 * 64 + 2 * cp - 1) // (2 * cp) * (2 * cp)
                    available = capacity - (count - 1) * physical_short
                    long = available // 64 * 64
                    while (long + 2 * cp - 1) // (2 * cp) * (2 * cp) > available:
                        long -= 64
                    packs.append((long,) + (short,) * (count - 1))
                rows = {}
                for values in packs:
                    core = self.core_predictor(values, cp, ranks=tuple(range(cp)))
                    square = sum(length**2 for length in values) / capacity**2
                    rows[square] = float(core["forward_ms"]), float(core["backward_ms"])
                levels[count] = tuple((x, *v) for x, v in sorted(rows.items()))
            self._guide_table[cp] = levels

    def guide_program_costs(self, lengths, cp, *, ranks=None):
        """Approximate FA interpolation plus exact cheap GDN for candidate ordering."""
        return dict(self._guide_program(tuple(lengths), cp))

    @lru_cache(maxsize=65536)
    def _guide_program(self, lengths, cp):
        levels = self._guide_table[cp]
        count = len(lengths)
        low = max(n for n in levels if n <= count)
        high = min(n for n in levels if n >= count)
        square = sum(n * n for n in lengths) / (cp * 16384) ** 2

        def interpolate(rows, index):
            for lower, upper in zip(rows, rows[1:]):
                if square <= upper[0]:
                    fraction = max(0.0, min(1.0, (square - lower[0]) / (upper[0] - lower[0])))
                    return lower[index] + fraction * (upper[index] - lower[index])
            return rows[-1][index]

        result = self._gdn_costs(lengths, cp)
        for index, phase, key in ((1, "forward", "fa_f"), (2, "backward", "fa_b")):
            first = interpolate(levels[low], index)
            weight = (count - low) / (high - low) if high != low else 0.0
            second = interpolate(levels[high], index) if high != low else first
            result[key] = first + weight * (second - first) + self.fa_remainder["cost_ms"][phase]
        return result

    @lru_cache(maxsize=65536)
    def _program(self, lengths, cp):
        costs = self._gdn_costs(lengths, cp)
        core = self.core_predictor(lengths, cp, ranks=tuple(range(cp)))
        costs["fa_f"] = float(core["forward_ms"]) + self.fa_remainder["cost_ms"]["forward"]
        costs["fa_b"] = float(core["backward_ms"]) + self.fa_remainder["cost_ms"]["backward"]
        if any(not math.isfinite(v) or v <= 0 for v in costs.values()):
            raise ValueError("invalid predicted mixed-layer cost")
        return costs

    def program_costs(self, lengths, cp, *, ranks=None):
        values = tuple(lengths)
        if not values or len(values) > 64 or any(type(n) is not int or n <= 0 for n in values):
            raise ValueError("one to 64 positive integer logical lengths required")
        if type(cp) is not int or not 1 <= cp <= 32:
            raise ValueError("the proxy covers CP1..32")
        if ranks is not None:
            ranks = tuple(ranks)
            if (
                len(ranks) != cp
                or any(type(r) is not int for r in ranks)
                or ranks != tuple(range(ranks[0], ranks[0] + cp))
                or ranks[0] < 0
                or ranks[-1] >= 32
            ):
                raise ValueError("the GB200 proxy requires contiguous physical ranks inside 0..31")
        return dict(self._program(values, cp))

    def forward(self, lengths, cp, *, ranks=None):
        p = self.program_costs(lengths, cp, ranks=ranks)
        return (3 * p["gdn_f"] + 2 * p["fa_f"]) / 5

    def backward(self, lengths, cp, *, ranks=None):
        p = self.program_costs(lengths, cp, ranks=ranks)
        return (3 * p["gdn_b"] + 2 * p["fa_b"]) / 5

    def timing(self):
        return execution_timing(self.non_attention["cost_ms"], False)

    def clear(self):
        self._program.cache_clear()
        self._guide_program.cache_clear()

    def check_domain(self, geometry, config=None):
        if (
            geometry.world_size,
            geometry.ep_size,
            geometry.tokens_per_rank,
            geometry.dataset_alignment,
        ) != (32, 4, 16384, 64):
            raise ValueError("the composed proxy requires W32/EP4/local16K/dataset64")

    def metadata(self):
        return dict(
            name=self.name,
            source_hashes=self.source_hashes,
            core=getattr(self.core_predictor, "metadata", lambda: None)(),
            proposal_guide="frozen-predictor queries indexed by CP, sequence count and sum of squared lengths; final scores exact",
            non_attention=self.non_attention,
            fa_remainder=self.fa_remainder,
            gdn=self.gdn.metadata(),
            tail_padding="extend_last",
            dataset_alignment=64,
            provisional=True,
            provisional_reasons=[
                "mixed_eager_native_gdn_and_cuda_graph_nccl_fa",
                "gdn_coefficients_transferred_to_extend_last_padding",
                "measured_fixed_non_attention_transferred_to_variable_packs",
                "cp1_fa_remainder_transferred_to_cp2_32",
                "cp1_remainder_does_not_measure_multicp_contiguous_zigzag_conversion",
            ],
            scope="attention modules plus measured fixed non-attention; prediction, not GPU E2E measurement",
        )
