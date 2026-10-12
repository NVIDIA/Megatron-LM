"""Frozen CPU prediction kernel; provenance is recorded in ORIGIN.json."""

import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path

DEFAULT_ATTENTION_PROFILE = "gb300-parent32-native-local4096-v1"
FEATURES = (
    "bias",
    "cp_enabled",
    "summary_hops",
    "max_prefix_chain",
    "boundary_summary_chunks_per_64",
    "extra_local_sequences",
)
ATTENTION_PROFILES = {
    DEFAULT_ATTENTION_PROFILE: dict(
        file="gb300_proxy.json",
        hardware="gb300",
        domain_size=72,
        transport="native",
        world_size=32,
        local_tokens=4096,
        phase_split="normalized_2f_plus_b",
        provisional_reasons=(
            "arbitrary pack/partition transfer is not universally validated",
            "effective F/B split is normalized to the measured checkpointed 2F+B total",
        ),
    ),
    "gb200-parent32-native-local16384-7319138": dict(
        file="gb200_local16384_proxy.json",
        hardware="gb200",
        domain_size=72,
        transport="native",
        world_size=32,
        local_tokens=16384,
        phase_split="measured_separately",
        provisional_reasons=(
            "arbitrary pack/partition transfer is not universally validated",
            "many-short-sequence packs at small odd CP are under-predicted (2026-09-23 GPU comparison, "
            "wide-16K: predicted +82 %, measured -16.5 % versus the power-of-two baseline)",
        ),
    ),
}


class Qwen35ProxyPredictor:
    def __init__(self, *, profile=DEFAULT_ATTENTION_PROFILE, allow_provisional=False):
        if profile not in ATTENTION_PROFILES:
            raise ValueError("unsupported Qwen3.5 predictor profile")
        if allow_provisional is not True:
            raise ValueError("arbitrary packs/partitions require explicit allow_provisional=True")
        spec = ATTENTION_PROFILES[profile]
        root = Path(__file__).with_name("qwen35_calibration")
        manifest = json.loads((root / "manifest.json").read_text())["profiles"][profile]
        if manifest["file"] != spec["file"]:
            raise ValueError("Qwen3.5 calibration manifest names a different artifact")
        raw = (root / spec["file"]).read_bytes()
        self.calibration_sha256 = hashlib.sha256(raw).hexdigest()
        if self.calibration_sha256 != manifest["export_sha256"]:
            raise ValueError("Qwen3.5 calibration hash mismatch")
        self.data = json.loads(raw)
        if self.data["profile"] != profile or tuple(self.data["gdn_features"]) != FEATURES:
            raise ValueError("Qwen3.5 calibration feature contract mismatch")
        for key in (
            "hardware",
            "domain_size",
            "transport",
            "world_size",
            "local_tokens",
            "phase_split",
        ):
            if key in self.data and self.data[key] != spec[key]:
                raise ValueError(f"Qwen3.5 calibration {key} disagrees with the profile registry")
        self.normalized = spec["phase_split"] == "normalized_2f_plus_b"
        if self.normalized != (
            "checkpoint" in self.data["gdn"] and "checkpoint" in self.data["fa"]
        ):
            raise ValueError("Qwen3.5 calibration variants do not match the profile's phase split")
        if set(self.data["fa"]["plain"]) != {str(cp) for cp in range(1, 33)}:
            raise ValueError("Qwen3.5 calibration must cover CP1..32")
        self.profile = profile
        self.spec = spec
        self.local_tokens = spec["local_tokens"]

    def validate_config(self, config):
        if (
            config.world_size != self.spec["world_size"]
            or config.pp_size != 1
            or config.vpp_size != 1
            or config.tokens_per_rank != self.local_tokens
            or config.dataset_alignment != 64
            or config.tail_padding != "separate_sequence"
            or any(config.alignments()[cp] != 2 * cp for cp in config.allowed_cp_sizes)
            or config.rank_maps() != (tuple(range(self.spec["world_size"])),)
        ):
            raise ValueError(
                f"frozen predictor requires parent{self.spec['world_size']}/PP1, "
                f"local{self.local_tokens}, ceil64 then 2CP, appended dummy"
            )

    def metadata(self):
        return dict(
            profile=self.profile,
            hardware=self.spec["hardware"],
            domain_size=self.spec["domain_size"],
            transport_backend=self.spec["transport"],
            geometry=dict(self.data["geometry"]),
            calibration_sha256=self.calibration_sha256,
            cost_scope="attention_module",
            local_tokens=self.local_tokens,
            phase_split=self.spec["phase_split"],
            provisional=True,
            provisional_reasons=list(self.spec["provisional_reasons"]),
            padding="dataset ceil64, sequence ceil2CP, computed appended dummy",
        )

    @staticmethod
    def _features(cu, cp, phase, recompute, local):
        difference = [0] * (cp + 1)
        chain = forward_chunks = backward_chunks = 0
        for begin, end in zip(cu, cu[1:]):
            first, last = begin // local, (end - 1) // local
            difference[first] += 1
            difference[last + 1] -= 1
            chain = max(chain, last - first)
            if last > first:
                forward_chunks = max(forward_chunks, (local - begin % local + 63) // 64)
                backward_chunks = max(backward_chunks, ((end - 1) % local + 64) // 64)
                if last - first > 1:
                    forward_chunks = backward_chunks = local // 64
        active = maximum = 0
        for delta in difference[:-1]:
            active += delta
            maximum = max(maximum, active)
        nf = 1 if phase == "forward_gpu_ms" else int(recompute)
        nb = 0 if phase == "forward_gpu_ms" else 1
        if phase == "forward_backward_gpu_ms":
            nf += 1
        return (
            1.0,
            float(cp > 1),
            (nf + nb) * (cp - 1),
            (nf + nb) * chain,
            (nf * forward_chunks + nb * backward_chunks) / 64,
            float(maximum - 1),
        )

    @lru_cache(maxsize=65536)
    def _costs(self, raw, cp):
        local = self.local_tokens
        stored = tuple((n + 63) // 64 * 64 for n in raw)
        cu = [0]
        for n in stored:
            cu.append(cu[-1] + (n + 2 * cp - 1) // (2 * cp) * (2 * cp))
        physical = cu[-1]
        if physical > cp * local:
            raise ValueError(f"pack exceeds frozen local{local} capacity")
        if physical < cp * local:
            cu.append(cp * local)
        square = (sum(n * n for n in raw) + (cp * local - physical) ** 2) / cp / 1e6
        f, b, fb = "forward_gpu_ms", "backward_gpu_ms", "forward_backward_gpu_ms"

        def gdn(variant, phase):
            coefficients = self.data["gdn"][variant][phase]
            features = self._features(cu, cp, phase, variant == "checkpoint", local)
            return sum(coefficients[name] * value for name, value in zip(FEATURES, features))

        def fa(variant, phase):
            coefficients = self.data["fa"][variant][str(cp)][phase]
            return coefficients["bias_ms"] + coefficients["balanced_square_ms"] * square

        def split(pf, pb, total):
            scale = total / (2 * pf + pb)
            return pf * scale, pb * scale

        if self.normalized:
            gf, gb = split(gdn("plain", f), gdn("plain", b), gdn("checkpoint", fb))
            ff, bb = split(fa("plain", f), fa("plain", b), fa("checkpoint", fb))
        else:
            gf, gb = gdn("plain", f), gdn("plain", b)
            ff, bb = fa("plain", f), fa("plain", b)
        values = (gf, gb, ff, bb)
        if any(not math.isfinite(v) or v < 0 for v in values):
            raise ValueError("invalid predicted module cost")
        return values

    def program_costs(self, lengths, cp, *, ranks=None):
        raw = tuple(lengths)
        if not raw or any(type(n) is not int or n <= 0 for n in raw):
            raise ValueError("positive integer sequence lengths required")
        if type(cp) is not int or not 1 <= cp <= 32:
            raise ValueError("frozen predictor covers CP1..32")
        if ranks is not None:
            world = self.spec["world_size"]
            ranks = tuple(ranks)
            if (
                len(ranks) != cp
                or any(type(r) is not int for r in ranks)
                or not 0 <= ranks[0] <= world - cp
                or ranks != tuple(range(ranks[0], ranks[0] + cp))
            ):
                raise ValueError(
                    f"frozen predictor requires a contiguous interval of physical ranks 0..{world-1}"
                )
        return dict(zip(("gdn_f", "gdn_b", "fa_f", "fa_b"), self._costs(raw, cp)))


LAYER_TYPES = ("gdn", "gdn", "gdn", "fa", "fa")
DEFAULT_NON_ATTENTION_PROFILE = "gb300-ep4-local4096-742013"
NON_ATTENTION_PROFILES = {
    DEFAULT_NON_ATTENTION_PROFILE: dict(
        world_size=32,
        ep_size=4,
        tokens_per_rank=4096,
        recompute=True,
        attention_profile="gb300-parent32-native-local4096-v1",
    ),
    "gb200-ep4-local16384-7319138": dict(
        world_size=32,
        ep_size=4,
        tokens_per_rank=16384,
        recompute=False,
        attention_profile="gb200-parent32-native-local16384-7319138",
    ),
}
BASE_KEYS = {"layer_forward", "layer_backward", "microbatch_outer", "iteration_tail"}
OTHER_KEYS = BASE_KEYS | {"layer_recompute_forward"}


def profile_scope(profile):
    if profile not in NON_ATTENTION_PROFILES:
        raise ValueError("unsupported paired non-attention profile")
    return dict(NON_ATTENTION_PROFILES[profile])


def load_non_attention_profile(profile=DEFAULT_NON_ATTENTION_PROFILE):
    scope = profile_scope(profile)
    root = Path(__file__).with_name("profiles")
    name = f"{profile}.json"
    raw = (root / name).read_bytes()
    manifest = json.loads((root / "manifest.json").read_text())
    if hashlib.sha256(raw).hexdigest() != manifest[name]:
        raise ValueError("paired non-attention profile hash mismatch")
    data = json.loads(raw)
    for key, value in scope.items():
        if key in data and data[key] != value:
            raise ValueError(
                f"paired non-attention profile {key} disagrees with the profile registry"
            )
    return data


def execution_timing(other, recompute):
    """The timing block of the 5-layer program: named ``<kind>_f`` / ``<kind>_b`` group costs, EP remainders, the local outer cost
    and the iteration tail - shared by the e2e evaluator and the runtime cost model so both replay the same program.
    """

    def attention(kind, phase):
        return dict(sync="cp", cost=f"{kind}_{phase}")

    def remainder(key):
        return dict(sync="ep", duration_ms=other[key])

    forward, backward = [], []
    for kind in LAYER_TYPES:
        forward.extend([attention(kind, "f"), remainder("layer_forward")])
    for kind in reversed(LAYER_TYPES):
        if recompute:
            backward.extend([attention(kind, "f"), remainder("layer_recompute_forward")])
        backward.extend([remainder("layer_backward"), attention(kind, "b")])
    backward.append(dict(sync="local", duration_ms=other["microbatch_outer"]))
    return dict(
        execution_program=[dict(layer_count=len(LAYER_TYPES), forward=forward, backward=backward)],
        iteration_tail_ms=other["iteration_tail"],
    )
