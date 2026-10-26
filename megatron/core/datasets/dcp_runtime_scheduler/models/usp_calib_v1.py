# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Temporary DynamicUSP U8 cost proxy; execution still uses native P2P.

Default formulas and the synthetic 44-layer shape are preserved from upstream
1ff97a4a, not refitted to Qwen or P2P. See DYNAMIC_USP_NOTICE.txt for attribution.
Only the default profile is ported; Stage 2 and calibration tooling are omitted.
"""

import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path


class USPCalibV1CostModel:
    """Expose the original compute + U8 A2A + ring costs to the CPU planner.

    Forward/backward include the upstream synthetic block's compute and MoE.
    They are not attention-only times and must not receive additional MoE costs.
    """

    name = "usp_calib_v1"
    placement_invariant = True
    layer_count = 44
    supported_cp_sizes = (8, 16, 32)

    def __init__(self, geometry, *, allow_provisional=False) -> None:
        if allow_provisional is not True:
            raise ValueError("usp_calib_v1 requires explicit allow_provisional=True")
        self.geometry = geometry
        self.check_domain(geometry, None)
        raw = Path(__file__).with_suffix(".json").read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != "d25989552ab9700f07261231fa13d66672d98684435fff57838442ca12c19cd7":
            raise ValueError("Frozen DynamicUSP calibration hash mismatch")
        self.table = json.loads(raw)
        self.source_hashes = {
            "calib_table_v1.json": digest,
            "usp_calib_v1.py": "a2dbcff0099936e283b2386783861d7191c5319e1cf9e22a8bff30f33207329d",
        }

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
        """Record the original U8 calibration and provisional P2P transfer scope."""
        return dict(
            name=self.name,
            source_commit="1ff97a4ae0beb5e088fadb9e112265bdd194b24d",
            source_hashes=self.source_hashes,
            provisional=True,
            modeled_transport="U8 all-to-all + ring",
            execution_transport="unchanged; native P2P in the temporary recipe",
            shape="upstream synthetic 44-layer shape, not a Qwen calibration",
        )
