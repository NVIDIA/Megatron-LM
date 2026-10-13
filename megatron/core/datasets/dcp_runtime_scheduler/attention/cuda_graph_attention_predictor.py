"""Frozen CPU prediction kernel; provenance is recorded in ORIGIN.json."""

from __future__ import annotations

import hashlib
import heapq
import json
import math
from functools import lru_cache
from pathlib import Path

from .cp_round_geometry import chunk_pair

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "cuda_graph_wp_g3_20261006" / "cuda_graph_attention_model.json"
# sha256 of DATA_PATH as written by tools/wp_g3_cuda_graph/build_cuda_graph_model_config.py (the tool prints the new digest)
DATA_SHA256 = "41747868ed35aaa61f1eb33e9a503fba23ba2d54166a3b0411a0d7d598b0e4c6"
SCHEMA = "dcp-scheduler/cuda-graph-attention-model/3"
LEGACY_SCHEMAS = (
    "dcp-scheduler/cuda-graph-attention-model/2",
)  # the v2 shape (hardware -> heads): loads through normalize_payload, identical outputs
STATUS_MARKER = "provisional: WP-G3 A22.4 served; GB200 sealed program closed (A23): ph4 f 24 / 45, h 5 / 24; P90 <= 10 per cell is the binding gate; known limits: CP2 pairs, qwen3_5_397b geometry (+4.8 %), chip spread without a card (+-4 %)"

LAUNCH_MODES = (
    "eager",
    "cuda_graph",
    "eager_structural",
)  # eager_structural: the WP-G6 eager launch regime on the cuda_graph forward core (eager_attention_predictor)
DEFAULT_LAUNCH_MODE = "eager"
CUDA_GRAPH_LAUNCH_MODE = "cuda_graph"
EAGER_STRUCTURAL_LAUNCH_MODE = "eager_structural"
MODEL_KIND = "wp_g3_cuda_graph_f5_s4"
# the heads of the data file: the SERVED A22.4 model and the superseded A21.1 / A16 models (comparison only, selectable)
SERVED_HEAD = "a22_4"
SUPERSEDED_HEADS = ("a21_1", "a16")
SUPERSEDED_HEAD = SUPERSEDED_HEADS[-1]
HEADS = (SERVED_HEAD,) + SUPERSEDED_HEADS
# the (hardware, direction) matrix: the default entry is gb200 / forward (head a22_4); every other entry is pending until its campaign fills it
DIRECTIONS = ("forward", "backward")
DEFAULT_HARDWARE = "gb200"
DEFAULT_DIRECTION = "forward"
ENTRY_STATUSES = ("served", "served_round1", "served_round2", "pending")
# "served_round1": the entry is served by a VENDORED round-1 composer package rather than by this module's S4 composition - gb200 / backward (WP-G3B B7.1,
# head b7_1; sealed f_bwd 13 / 45 cells PASS, partially validated; round 2 read 22 / 45 + 6 / 24 (not installed), round 3 read with no freeze (B16.2)) and gb300 / backward (WP-G5 BG4 / BG5, head g5_bg5:
# the same composition with the GB300 constants, data directory data_gb300/; sealed f_bwd_gb300 14 / 45, h_bwd_gb300 6 / 24; round 2 deferred) - both through the
# package cuda_graph_backward (one data directory per hardware); load_cuda_graph_model dispatches to it
SERVING_STATUSES = ("served", "served_round1", "served_round2")
ROUND1_STATUS = "served_round1"
ROUND2_STATUS = "served_round2"  # gb200 / backward since 2026-10-09: the round-2 head b12 served by the same vendored package (BackwardRound2Model); b7_1 superseded
VENDORED_STATUSES = (ROUND1_STATUS, ROUND2_STATUS)
VENDORED_PACKAGES = {
    ("gb200", "backward"): "cuda_graph_backward",
    ("gb300", "backward"): "cuda_graph_backward",
}
# the per-step exchange bytes class of the ring layer (RINGDESC_BWD_01.md section 1.3): forward = B_kv on every round (as served),
# backward = B_kv (KV only) / 2 B_kv (KV + dKV) x (cp - 2) / B_kv (dKV only)
EXCHANGE_SCHEDULE_FORWARD = "kv_ring_forward"
EXCHANGE_SCHEDULE_BACKWARD = "kv_dkv_ring_backward"
EXCHANGE_SCHEDULES = (EXCHANGE_SCHEDULE_FORWARD, EXCHANGE_SCHEDULE_BACKWARD)
EXCHANGE_SCHEDULE_OF_DIRECTION = {
    "forward": EXCHANGE_SCHEDULE_FORWARD,
    "backward": EXCHANGE_SCHEDULE_BACKWARD,
}
COMPOSITION_S4 = (
    "s4"  # the forward S4 composition (A10 / A20.1 / A22.4) - the only one this predictor composes
)
# every prediction of the mode carries this reason (the sealed verdicts are partial: ph4 24 / 45 and 5 / 24 cells, reports 39 / 40; the program is closed, A23)
PROVISIONAL_REASON = "cuda_graph_model_provisional"
REASON_STORAGE = "cuda_graph_storage_not_calibrated"  # S outside {4096, 8192, 16384} (coefficients are shared across S, so it is served)
REASON_ALIGNMENT = "cuda_graph_alignment_not_calibrated"  # alignment other than the calibrated 128
REASON_CP_STRUCTURE = "cuda_graph_cp_structure_extrapolated"  # cp without a measured W0 cell of its structure (CP32 and every non-power CP)
REASON_CP1_FORM = "cp1_form_acceptance_failed"  # the (geometry, S) causal set failed the A8 in-sample / LOLO gate (served with its coefficients)
REASON_CLOCK = "cuda_graph_clock_not_supplied"  # served head without a clock input: predicted AT f0 (served_f0); the chip spread is the error floor
REASON_PRODUCTION_CLOCK = "cuda_graph_clock_production_default"  # no card: predicted at the hardware's production default clock (A24; the clock DURING the attention window), not the allocation's
CLOCK_LAWS = (
    "alpha",
    "two_term",
)  # A19 (iii): (f0 / f) ** alpha (alpha 1, the frozen GB200 forward law) | A25: s f0 / f + (1 - s), s = the clock-following share at f0
DEFAULT_CLOCK_LAW = "alpha"
REASON_HEAD_SUPERSEDED = "cuda_graph_head_superseded"  # the a21_1 (ph3 ring read) and a16 (g_ring 0.7474) heads: comparison only, not the served model
MASKS = ("causal", "rectangle")
SECTION_MASK = {"diagonal": "causal", "lower": "rectangle", "upper": "rectangle"}
MIB = float(2**20)
TILE = 128
BLOCK_ROWS = 256
HEAD_DIM_HALF_OCCUPANCY = 256


# ----------------------------------------------------------------------------------------------------------
# layout stage (no coefficient): alignment input, extend_last, pack = S x cp
# ----------------------------------------------------------------------------------------------------------
def layout_extend_last(lengths, cp_size, storage_tokens, alignment, dataset_alignment=1):
    """Padded (physical) lengths: every sequence to a multiple of ``alignment``, the LAST sequence extended so the pack
    fills ``storage_tokens x cp_size`` exactly. Raises when the aligned storage exceeds the capacity or (cp > 1) a padded
    length is not divisible by 2 cp (the zigzag needs two equal chunks per rank)."""
    values = [int(v) for v in lengths]
    if not values or min(values) <= 0:
        raise ValueError("lengths must contain positive sequence lengths")
    cp, storage, align = int(cp_size), int(storage_tokens), int(alignment)
    dataset_align = int(dataset_alignment)
    if cp <= 0 or storage <= 0 or align <= 0 or dataset_align <= 0:
        raise ValueError("cp_size, storage_tokens and alignment must be positive")
    dataset_padded = [-(-v // dataset_align) * dataset_align for v in values]
    padded = [-(-v // align) * align for v in dataset_padded]
    capacity = storage * cp
    if sum(padded) > capacity:
        raise ValueError(
            f"aligned storage {sum(padded)} exceeds the pack capacity {capacity} "
            f"(storage {storage} tokens per rank x cp {cp}, alignment {align})"
        )
    padded[-1] += capacity - sum(padded)
    if cp > 1 and any(v % (2 * cp) for v in padded):
        raise ValueError(
            f"padded lengths {padded} are not divisible by 2 x cp = {2 * cp}: alignment {align} does not "
            f"support the zigzag layout at cp {cp}"
        )
    return padded


def cp1_pieces(lengths):
    """cp = 1: one causal call over the pack, pieces (L_i, L_i) of valid tokens."""
    return [(int(v), int(v)) for v in lengths]


@lru_cache(maxsize=256)
def _round_pieces(lengths, padded, cp_size):
    if cp_size <= 1 or len(lengths) != len(padded) or not lengths or min(lengths) <= 0:
        raise ValueError("round geometry requires CP > 1 and matching positive lengths")
    chunks = [
        [chunk_pair(n, p, cp_size, rank) for n, p in zip(lengths, padded)]
        for rank in range(cp_size)
    ]
    first = [tuple(a for a, _ in row) for row in chunks]
    second = [tuple(b for _, b in row) for row in chunks]
    full = [tuple(a + b for a, b in row) for row in chunks]
    physical = sum(padded) // cp_size
    out, piece_cache = [], {}
    for rank in range(cp_size):
        for step in range(cp_size):
            kv_rank = (rank - step) % cp_size
            if step == 0:
                section, q, kv, pq, pk = "diagonal", full[rank], full[kv_rank], physical, physical
            elif step <= rank:
                section, q, kv, pq, pk = (
                    "lower",
                    full[rank],
                    first[kv_rank],
                    physical,
                    physical // 2,
                )
            else:
                section, q, kv, pq, pk = (
                    "upper",
                    second[rank],
                    full[kv_rank],
                    physical // 2,
                    physical,
                )
            # Many ranks share the same token vectors; keep pack order unchanged.
            key = q, kv
            pieces = piece_cache.get(key)
            if pieces is None:
                pieces = piece_cache[key] = tuple((a, b) for a, b in zip(q, kv) if a > 0 and b > 0)
            out.append(
                dict(
                    rank=rank,
                    step=step,
                    kv_rank=kv_rank,
                    section=section,
                    pieces=pieces,
                    q_physical=pq,
                    kv_physical=pk,
                )
            )
    return tuple(out)


def round_pieces(lengths, padded, cp_size):
    """One record per (rank, step) of TE's zigzag P2P forward: section, the active (q_tokens, kv_tokens) pieces and the
    physical Q / KV tokens of the call (``cp_round_geometry.build_round_geometries``)."""
    return _round_pieces(
        tuple(int(v) for v in lengths), tuple(int(v) for v in padded), int(cp_size)
    )


# ----------------------------------------------------------------------------------------------------------
# CP1 form F5 (A8) / F5_sim+jmax (A21.1): wave-makespan of the fused-attention kernel's (q head, 256-row Q block) units
# ----------------------------------------------------------------------------------------------------------
def ceil_div(a, b):
    return -(-int(a) // int(b))


def tile_of(tokens, tile=TILE):
    """Tile count of a piece: ceil(valid tokens / 128), at least 1 (padding between sequences is skipped by the THD kernel)."""
    return max(1, ceil_div(tokens, tile))


def slots_for(head_dim, p_hw, half_occupancy_head_dim=HEAD_DIM_HALF_OCCUPANCY):
    """Resident units per GPU: P_hw below the half-occupancy head dim, P_hw // 2 from it on (one unit occupies two slots: GB200 / GB300 at
    head_dim 256); ``half_occupancy_head_dim=None`` = no halving at any head dim (H100: the d256 kernel launches one CTA per SM like the d128
    one, PREREGISTRATION_H100.md A6 (iv) - P = P_hw = 132 at both head dims)."""
    if half_occupancy_head_dim is None:
        return int(p_hw)
    return int(p_hw) if int(head_dim) < int(half_occupancy_head_dim) else max(1, int(p_hw) // 2)


def q_blocks(tq, tile=TILE, block_rows=BLOCK_ROWS):
    """Q blocks of a piece with tq 128-row q tiles: ceil(tq / (block_rows / tile)) - 256-row blocks on GB200 / GB300 (ceil(tq / 2); an odd tq
    opens a half-filled block), 128-row blocks on H100 (tq blocks; the sm90 fprop grid, A10 / A11).
    """
    return ceil_div(tq, block_rows // tile)


def causal_block_kv(tq, tile=TILE, block_rows=BLOCK_ROWS):
    """kv tiles walked by the Q blocks b = 1..q_blocks(tq) of a causal piece with tq tiles: min(2 b, tq) (cp1_makespan's unit cost rule,
    applied with the head's block_rows - the campaign's reading on every hardware)."""
    tq = int(tq)
    return [min(2 * b, tq) for b in range(1, q_blocks(tq, tile, block_rows) + 1)]


def rect_block_kv(tq, tkv, tile=TILE, block_rows=BLOCK_ROWS):
    """kv tiles walked by every Q block of a no-mask q x kv block: tkv for each of the q_blocks(tq) blocks."""
    return [int(tkv)] * q_blocks(int(tq), tile, block_rows)


def active_pieces(pieces):
    """(q_tokens, kv_tokens) pieces with q > 0 and kv > 0, pack order kept."""
    return [(int(q), int(kv)) for q, kv in pieces if int(q) > 0 and int(kv) > 0]


def units_kv(pieces, mask, q_heads, tile=TILE, block_rows=BLOCK_ROWS):
    """The kernel's units in launch order (pack order x heads x Q blocks ascending): kv tiles per unit."""
    if mask not in MASKS:
        raise ValueError(f"mask must be one of {MASKS}, got {mask!r}")
    out = []
    for q, kv in active_pieces(pieces):
        tq, tkv = tile_of(q, tile), tile_of(kv, tile)
        per_piece = (
            causal_block_kv(tq, tile, block_rows)
            if mask == "causal"
            else rect_block_kv(tq, tkv, tile, block_rows)
        )
        for _head in range(int(q_heads)):
            out.extend(per_piece)
    return out


def makespan(kv_tiles, P, rho):
    """Makespan in tau units: the longest unit if N <= P, else the greedy list schedule (the next unit in launch order to the
    first free slot - the hardware dispatch rule). 0.0 for an empty unit list."""
    from .dispatch import accelerated_chain

    compiled = accelerated_chain(kv_tiles, P, rho)
    if compiled is not None:
        return compiled[0]
    rho = float(rho)
    costs = [rho + float(k) for k in kv_tiles]
    if not costs:
        return 0.0
    P = int(P)
    if P <= 0:
        raise ValueError("P must be positive")
    if len(costs) <= P:
        return max(costs)
    if costs[0] > 0 and all(cost == costs[0] for cost in costs):
        # Uniform rectangular units: preserve repeated-addition float arithmetic.
        finish = 0.0
        for _ in range(ceil_div(len(costs), P)):
            finish += costs[0]
        return finish
    heap = [0.0] * P
    end = 0.0
    for cost in costs:
        finish = heap[0] + cost
        heapq.heapreplace(heap, finish)
        if finish > end:
            end = finish
    return end


def j_max(kv_tiles):
    """kv tiles of the longest unit (the J_max of the ``+ jmax`` form)."""
    return max((int(k) for k in kv_tiles), default=0)


def static_ms(n_active, us_per_slot, thd_max):
    """The fixed static-batch term: every empty cu_seqlens slot of the thd_max-sized static batch costs us_per_slot."""
    return float(us_per_slot) * 1e-3 * (int(thd_max) - int(n_active))


def f5_wall(
    cset,
    pieces,
    S_tokens,
    mask,
    q_heads,
    head_dim,
    *,
    p_hw,
    static_us_per_slot,
    thd_max,
    n_static=None,
    tile=TILE,
    block_rows=BLOCK_ROWS,
    half_occupancy_head_dim=HEAD_DIM_HALF_OCCUPANCY,
):
    """wall_ms of one fused-attention call from a coefficient set (k0_ms, b_S_ms_per_token, tau_ms_per_kvtile, rho_kvtiles,
    optional tau_jmax_ms_per_kvtile = the F5_sim+jmax form, optional P_hw). S_tokens = the b_S argument (S at cp = 1; the
    call's physical Q tokens in a ring round). Returns the components; the arithmetic order is cp1_makespan.predict_ms's
    (k0 + bS + sched + jmax + static) so a set without tau_jmax reproduces the plain F5 wall bit for bit.
    """
    kv = units_kv(pieces, mask, q_heads, tile, block_rows)
    P = slots_for(head_dim, int(cset.get("P_hw", p_hw)), half_occupancy_head_dim)
    span = makespan(kv, P, float(cset["rho_kvtiles"]))
    n_active = len(active_pieces(pieces))
    n_st = n_active if n_static is None else int(n_static)
    k0 = float(cset["k0_ms"])
    bS = float(cset["b_S_ms_per_token"]) * float(S_tokens)
    sched = float(cset["tau_ms_per_kvtile"]) * span
    beyond = len(kv) > P
    jm_tiles = j_max(kv)
    tj = float(cset.get("tau_jmax_ms_per_kvtile") or 0.0)
    jm = tj * jm_tiles if (tj and beyond) else 0.0
    st = static_ms(n_st, static_us_per_slot, thd_max)
    wall = k0 + bS + sched + jm + st
    return {
        "wall_ms": wall,
        "k0_ms": k0,
        "bS_ms": bS,
        "sched_ms": sched,
        "jmax_ms": jm,
        "static_ms": st,
        "makespan_tau": span,
        "N_units": len(kv),
        "P": P,
        "J_max": jm_tiles,
        "n_active": n_active,
        "n_static": n_st,
        "beyond_first_wave": beyond,
        "set": mask,
        "form": str(cset.get("form") or "F5_sim"),
    }


# ----------------------------------------------------------------------------------------------------------
# the model: one data file, hardware -> {clock, topology, layout, directions -> {status, served_head, heads -> {cp1_sets, o_graph,
# ring_s4, ring_kernel_scale, clock_input}}}; the v2 shape (hardware -> heads) is normalized to it on load
# ----------------------------------------------------------------------------------------------------------
class CudaGraphHeadPendingError(ValueError):
    """The (hardware, direction) entry exists but carries no coefficient set yet: its campaign has not filled it. Never a fallback."""


def _legacy_campaign(direction):
    return {
        "name": "WP-G3" if direction == "forward" else "WP-G3B",
        "status_note": "legacy v2 data file: the gb200 forward heads only",
    }


def normalize_payload(payload):
    """The v3 shape of a data-file payload. A v3 payload is returned as is; the v2 shape (``hardware -> {heads, layout, ...}`` with the
    top-level ``served_head`` / ``heads`` metadata) is lifted into ``hardware -> gb200 -> directions -> forward`` with the per-hardware clock
    and topology blocks derived from the served head's clock_input / ring_s4 - the numbers every head carries are untouched, so a model built
    from either shape predicts identically (tests prove it on the parity rows)."""
    schema = payload.get("schema")
    if schema == SCHEMA:
        return payload
    if schema not in LEGACY_SCHEMAS:
        raise ValueError(
            f"cuda_graph attention model: schema {schema!r}, expected {SCHEMA!r} (or a legacy shape of {LEGACY_SCHEMAS})"
        )
    out = {k: v for k, v in payload.items() if k != "hardware"}
    out["schema_loaded"] = schema
    out["schema"] = SCHEMA
    out["default_hardware"] = DEFAULT_HARDWARE
    out["default_direction"] = DEFAULT_DIRECTION
    hardware = {}
    for name, hw in payload["hardware"].items():
        hw = dict(hw)
        heads = hw.pop("heads")
        served = payload["served_head"]
        block = heads[served]
        clock = block.get("clock_input") or {}
        prod = {
            k: float(v)
            for k, v in (hw.get("production_default_clock_mhz") or {}).items()
            if isinstance(v, (int, float))
        }
        ring = block["ring_s4"]
        pair = ring.get("sequential_pair_structure")
        rule = {int(k): v for k, v in ring["structure_of_cp"].items()}
        intra = rule.get(4) or next((v for k, v in sorted(rule.items()) if k > 2), None)
        inter = (
            rule.get(8) or rule.get(16) or next((v for k, v in sorted(rule.items()) if k > 4), None)
        )
        classes = {}
        for cls, w in ring["W0"].items():
            role = "pair" if cls == pair else ("intra_node" if cls == intra else "inter_node")
            classes[cls] = {
                "role": role,
                "label": role,
                "cps": [int(c) for c in w.get("cps", [])],
                "exchange_group_size": 2 if role != "pair" or pair is not None else None,
            }
        hw["clock"] = {
            "f0_mhz": {"forward": (float(clock["f0_mhz"]) if clock else None), "backward": None},
            "law_default": {
                "forward": str(clock.get("law_default") or DEFAULT_CLOCK_LAW) if clock else None,
                "backward": None,
            },
            "two_term_s": {
                "forward": (
                    float(clock["two_term_s"])
                    if clock and clock.get("two_term_s") is not None
                    else None
                ),
                "backward": None,
            },
            "production_default_clock_mhz": {
                "forward": prod.get("forward"),
                "backward": prod.get("backward"),
            },
            "source": "derived from the legacy v2 payload (the served head's clock_input and the hardware's production_default_clock_mhz)",
        }
        hw["topology"] = {
            "gpus_per_node": 4,
            "classes": classes,
            "class_by_role": {"pair": pair, "intra_node": intra, "inter_node": inter},
            "sequential_pair": pair is not None,
            "rule": "cp = 2 -> the pair class; 3 <= cp <= gpus_per_node -> the intra-node class; cp > gpus_per_node -> the inter-node class",
            "source": "derived from the legacy v2 payload (the served head's ring_s4.structure_of_cp / W0 keys)",
        }
        hw["directions"] = {
            "forward": {
                "status": "served",
                "served_head": served,
                "heads": heads,
                "head_info": payload["heads"],
                "campaign": _legacy_campaign("forward"),
                "exchange_schedule": {"kind": EXCHANGE_SCHEDULE_FORWARD},
            },
            "backward": {
                "status": "pending",
                "served_head": None,
                "heads": {},
                "head_info": {},
                "campaign": _legacy_campaign("backward"),
                "exchange_schedule": {"kind": EXCHANGE_SCHEDULE_BACKWARD},
            },
        }
        hardware[name] = hw
    out["hardware"] = hardware
    return out


def _load_payload(path, expected_sha256=None):
    """(normalized payload, digest) of a data file; the digest must equal ``expected_sha256`` (None = the module's DATA_SHA256, the pin of the
    committed file; a test passes the digest of a legacy fixture). The default entry (gb200 / forward) must serve head a22_4 with the three
    known heads."""
    raw = Path(path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    expected_sha256 = DATA_SHA256 if expected_sha256 is None else expected_sha256
    if digest != expected_sha256:
        raise ValueError(
            f"cuda_graph attention model data file hash mismatch: {Path(path).name} is {digest}, pinned {expected_sha256}"
        )
    payload = json.loads(raw)
    if payload.get("schema") not in (SCHEMA,) + LEGACY_SCHEMAS:
        raise ValueError(
            f"cuda_graph attention model: schema {payload.get('schema')!r}, expected {SCHEMA!r}"
        )
    if payload.get("status_marker") != STATUS_MARKER:
        raise ValueError("cuda_graph attention model: the provisional status marker is missing")
    payload = normalize_payload(payload)
    entry = (
        payload["hardware"].get(DEFAULT_HARDWARE, {}).get("directions", {}).get(DEFAULT_DIRECTION)
    )
    if (
        not entry
        or entry.get("status") != "served"
        or entry.get("served_head") != SERVED_HEAD
        or set(entry.get("heads") or ()) != set(HEADS)
    ):
        raise ValueError(
            f"cuda_graph attention model: the {DEFAULT_HARDWARE} / {DEFAULT_DIRECTION} entry must serve head {SERVED_HEAD!r} with the heads {HEADS}"
        )
    for name, hw in payload["hardware"].items():
        for direction, item in hw["directions"].items():
            if direction not in DIRECTIONS or item.get("status") not in ENTRY_STATUSES:
                raise ValueError(
                    f"cuda_graph attention model: entry {name} / {direction} has status {item.get('status')!r} (expected one of {ENTRY_STATUSES})"
                )
    return payload, digest


def _entry(payload, hardware, direction):
    """The (hardware, direction) entry of a normalized payload; ValueError for an unknown hardware / direction (never a fallback)."""
    if direction not in DIRECTIONS:
        raise ValueError(f"direction must be one of {DIRECTIONS}, got {direction!r}")
    hw = payload["hardware"].get(hardware)
    if hw is None:
        raise ValueError(
            f"no cuda_graph attention model for hardware {hardware!r}; the data file covers {sorted(payload['hardware'])} "
            f"(served entries: {', '.join(f'{h} / {d}' for h, d in sorted(served_entries(payload))) or 'none'})"
        )
    entry = hw["directions"].get(direction)
    if entry is None:
        raise ValueError(
            f"the data file has no {direction!r} entry for hardware {hardware!r} (directions: {sorted(hw['directions'])})"
        )
    return hw, entry


def served_entries(payload):
    """The (hardware, direction) pairs whose entry is served."""
    return tuple(
        (h, d)
        for h, hw in sorted(payload["hardware"].items())
        for d, e in sorted(hw["directions"].items())
        if e.get("status") in SERVING_STATUSES
    )


def pending_message(hardware, direction, entry):
    campaign = entry.get("campaign") or {}
    name = campaign.get("name") or "its campaign"
    where = f" ({campaign['dir']})" if campaign.get("dir") else ""
    stage = f"; {campaign['stage']}" if campaign.get("stage") else ""
    return (
        f"the cuda_graph attention model for hardware {hardware!r} direction {direction!r} is pending: the {name} campaign{where} will fill it{stage}. "
        f"Its entry carries the hardware profile constants only (no coefficient set); there is no fallback to the {DEFAULT_HARDWARE} {DEFAULT_DIRECTION} head"
    )


def cuda_graph_hardware(path=DATA_PATH, direction=DEFAULT_DIRECTION, served_only=True):
    """Hardware names whose ``direction`` entry is served - by this module's heads or by a vendored round-1 composer (``served_only=False``: every hardware the data file carries an entry for)."""
    payload = _load_payload(path)[0]
    if direction not in DIRECTIONS:
        raise ValueError(f"direction must be one of {DIRECTIONS}, got {direction!r}")
    return tuple(
        sorted(
            name
            for name, hw in payload["hardware"].items()
            if direction in hw["directions"]
            and (not served_only or hw["directions"][direction]["status"] in SERVING_STATUSES)
        )
    )


def cuda_graph_matrix(path=DATA_PATH):
    """The (hardware, direction) matrix: {(hardware, direction): {status, served_head, heads, campaign}}."""
    payload = _load_payload(path)[0]
    return {
        (h, d): {
            "status": e["status"],
            "served_head": e.get("served_head"),
            "heads": tuple(e.get("heads") or ()),
            "campaign": dict(e.get("campaign") or {}),
        }
        for h, hw in sorted(payload["hardware"].items())
        for d, e in sorted(hw["directions"].items())
    }


def cuda_graph_heads(path=DATA_PATH, hardware=DEFAULT_HARDWARE, direction=DEFAULT_DIRECTION):
    """(served head, every head) of one (hardware, direction) entry of the data file (a pending entry: (None, ()))."""
    payload = _load_payload(path)[0]
    _hw, entry = _entry(payload, hardware, direction)
    return entry.get("served_head"), tuple(entry.get("heads") or ())


def resolve_head(head, heads=HEADS, served=SERVED_HEAD):
    """The data-file head name of an option value: None -> the served head; one of ``heads`` as given (the gb200 forward heads by default)."""
    if head is None:
        return served
    if head not in heads:
        raise ValueError(
            f"cuda_graph head must be one of {tuple(heads)} (None = the served head {served!r}), got {head!r}"
        )
    return head


def exchange_bytes_by_step(kind, cp_size, kv_bytes):
    """The per-step exchange bytes class of the ring layer, one entry per round (cp >= 2). ``kv_ring_forward``: B_kv on every round - the
    served S4 applies kappa W0(B_kv) to every round (identical numbers to the constant-W0 form). ``kv_dkv_ring_backward`` (RINGDESC_BWD_01
    section 1.3; the dKV accumulators travel with the KV ring in one (2, kv) buffer): step 0 KV only = B_kv, steps 1..cp-2 KV + dKV = 2 B_kv,
    step cp-1 dKV only = B_kv (cp exchanges per rank, 2 (cp - 1) B_kv bytes; cp = 2: two exchanges of B_kv).
    """
    cp, b = int(cp_size), int(kv_bytes)
    if cp < 2:
        raise ValueError("the exchange schedule needs cp >= 2")
    if kind == EXCHANGE_SCHEDULE_FORWARD:
        return [b] * cp
    if kind == EXCHANGE_SCHEDULE_BACKWARD:
        return [b] + [2 * b] * (cp - 2) + [b]
    raise ValueError(f"exchange schedule must be one of {EXCHANGE_SCHEDULES}, got {kind!r}")


def topology_class_for_cp(topology, cp_size):
    """(class name, role) of a CP size under a hardware's topology block: a class whose recorded ``cps`` contain cp wins; otherwise the rule
    cp = 2 -> the pair class, 3 <= cp <= gpus_per_node -> the intra-node class, cp > gpus_per_node -> the inter-node class.
    """
    cp = int(cp_size)
    if cp < 2:
        raise ValueError("the ring layer needs cp >= 2")
    classes = topology.get("classes") or {}
    for name, record in classes.items():
        if cp in [int(c) for c in (record.get("cps") or [])]:
            return name, record.get("role")
    by_role = topology.get("class_by_role") or {}
    gpus = int(topology.get("gpus_per_node") or 4)
    role = "pair" if cp == 2 else ("intra_node" if cp <= gpus else "inter_node")
    name = by_role.get(role)
    if name is None:
        raise ValueError(f"the topology block has no {role!r} class for cp {cp}")
    return name, role


def cuda_graph_topology_class(hardware, cp_size, path=DATA_PATH):
    """The communication class of (hardware, cp) from the data file's per-hardware topology block: {class, role, record, rule_applied}."""
    payload = _load_payload(path)[0]
    hw = payload["hardware"].get(hardware)
    if hw is None:
        raise ValueError(
            f"no hardware {hardware!r} in the data file ({sorted(payload['hardware'])})"
        )
    name, role = topology_class_for_cp(hw["topology"], cp_size)
    record = dict((hw["topology"].get("classes") or {}).get(name) or {})
    listed = int(cp_size) in [int(c) for c in (record.get("cps") or [])]
    return {
        "hardware": hardware,
        "cp": int(cp_size),
        "class": name,
        "role": role,
        "record": record,
        "rule_applied": not listed,
        "gpus_per_node": hw["topology"].get("gpus_per_node"),
    }


class CudaGraphAttentionModel:
    """The WP-G3 CUDA-graph model of one (hardware, direction) entry and one head (see the module docstring). A pending entry raises
    ``CudaGraphHeadPendingError`` naming the campaign that will fill it; there is no fallback to another hardware or direction.
    """

    def __init__(self, payload, hardware, head=None, data_sha256=None, direction=DEFAULT_DIRECTION):
        payload = normalize_payload(payload)
        hw, entry = _entry(payload, hardware, direction)
        if entry.get("status") == "pending":
            raise CudaGraphHeadPendingError(pending_message(hardware, direction, entry))
        if entry.get("status") in VENDORED_STATUSES:
            raise ValueError(
                f"the {hardware} / {direction} entry is served by the vendored backward composer package {VENDORED_PACKAGES.get((hardware, direction), (entry.get('composer') or {}).get('package'))!r} "
                f"(status {entry.get('status')!r}: head {entry.get('served_head')!r}); this class composes the forward S4 heads only - use load_cuda_graph_model(), which dispatches"
            )
        if entry.get("status") != "served":
            raise ValueError(
                f"the {hardware} / {direction} entry has status {entry.get('status')!r} (expected one of {ENTRY_STATUSES})"
            )
        heads = tuple(entry["heads"])
        self.served_head = entry["served_head"]
        self.head = resolve_head(head, heads, self.served_head)
        self.payload = payload
        self.hardware = hardware
        self.direction = direction
        self.entry = entry
        self.campaign = dict(entry.get("campaign") or {})
        self.data_sha256 = data_sha256
        self.status = payload["status"]
        self.status_marker = payload["status_marker"]
        self.is_served_head = self.head == self.served_head
        info = entry.get("head_info") or {}
        self.head_info = dict(
            info.get(self.head) or {"status": entry["heads"][self.head].get("status", "served")}
        )
        self.head_status = self.head_info["status"]
        self.hw = hw
        self.block = hw["directions"][direction]["heads"][self.head]
        self.hw_clock = dict(hw.get("clock") or {})
        self.topology = dict(hw.get("topology") or {})
        constants = payload["constants"]
        cp1 = self.block["cp1_sets"]
        # the head's structural constants (its kernel model's): tile, Q-block rows and the half-occupancy head dim (None = no halving; the H100 head:
        # 128-row units, P_hw 132 at both head dims); the file's top-level constants (the GB200 kernel model's 128 / 256 / 256) when a head omits them
        self.tile = int(cp1.get("tile", constants["tile_tokens"]))
        self.block_rows = int(cp1.get("block_rows", constants["block_rows"]))
        if self.block_rows % self.tile:
            raise ValueError(
                f"head {self.head!r} ({hardware} / {direction}): block_rows {self.block_rows} is not a multiple of the tile {self.tile}"
            )
        half = (
            cp1["head_dim_half_occupancy"]
            if "head_dim_half_occupancy" in cp1
            else constants["head_dim_half_occupancy"]
        )
        self.half_occupancy_head_dim = None if half is None else int(half)
        self.cp1_form = cp1["form"]
        self.p_hw = int(cp1["P_hw"])
        self.static_us_per_slot = float(cp1["static_us_per_slot"])
        self.thd_max = int(cp1["thd_max"])
        self.geometries = {name: dict(entry) for name, entry in cp1["geometries"].items()}
        self._profile_by_shape = {
            (int(e["query_heads"]), int(e["kv_heads"]), int(e["head_dim"])): name
            for name, e in self.geometries.items()
        }
        self.o_graph = self.block["o_graph"]
        self.ring = self.block["ring_s4"]
        self.ring_kernel_scale = float(self.block["ring_kernel_scale"]["value"])
        self.g_ring = (
            self.ring_kernel_scale
        )  # the A16 name of the constant (1.0 on the served head)
        clock = self.block.get("clock_input")
        self.clock = (
            dict(clock) if clock else None
        )  # None = the head's CP1 form is not normalized to f0 (no clock input)
        self.f0_mhz = float(clock["f0_mhz"]) if clock else None
        self.alpha = float(clock["alpha"]) if clock else None
        self.two_term_s = (
            float(clock["two_term_s"]) if clock and clock.get("two_term_s") is not None else None
        )  # A25.1: the hardware's clock-following share at f0
        # A24, per direction: the hardware clock block (v3) or the legacy per-hardware map; the head's own direction is the one used without a card
        prod = (
            self.hw_clock.get("production_default_clock_mhz")
            or hw.get("production_default_clock_mhz")
            or {}
        )
        self.production_default_clock_mhz = {
            k: float(v) for k, v in prod.items() if isinstance(v, (int, float))
        }
        self.layout = hw["layout"]
        self.alignment_default = int(self.layout["alignment_default"])
        self.calibrated_storages = tuple(int(v) for v in self.layout["calibrated_storages"])
        self.calibrated_cps = tuple(int(v) for v in self.ring["calibrated_cps"])
        self.sequential_pair_structure = self.ring["sequential_pair_structure"]
        self.structure_of_cp = {int(k): v for k, v in self.ring["structure_of_cp"].items()}
        self.kappa_W0 = float(self.ring["kappa_W0"])
        self.kappa_W0_pair = float(self.ring.get("kappa_W0_sequential_pair", 0.0))
        # the ring layer's per-step exchange bytes class (supplied by the head, else the entry, else the direction's default) and composition
        schedule = (self.ring.get("exchange_schedule") or entry.get("exchange_schedule") or {}).get(
            "kind"
        ) or EXCHANGE_SCHEDULE_OF_DIRECTION[direction]
        if schedule not in EXCHANGE_SCHEDULES:
            raise ValueError(
                f"head {self.head!r} ({hardware} / {direction}) declares the exchange schedule {schedule!r}; known: {EXCHANGE_SCHEDULES}"
            )
        self.exchange_schedule = schedule
        self.composition_kind = str(self.ring.get("composition_kind") or COMPOSITION_S4)
        if self.composition_kind != COMPOSITION_S4:
            raise ValueError(
                f"head {self.head!r} ({hardware} / {direction}) declares the ring composition {self.composition_kind!r}; this predictor composes "
                f"{COMPOSITION_S4!r} only (the S4-bwd composition of PREREGISTRATION_BWD.md section 0 lands with the WP-G3B head)"
            )
        self._kernel_cache = {}

    @classmethod
    def from_path(
        cls, hardware, head=None, path=DATA_PATH, direction=DEFAULT_DIRECTION, expected_sha256=None
    ):
        payload, digest = _load_payload(path, expected_sha256)
        return cls(payload, hardware, head, digest, direction)

    # -- geometry and coefficient sets ------------------------------------------------------------------
    def geometry_profile(self, query_heads, kv_heads, head_dim):
        """Name of the calibrated geometry profile for (query_heads, kv_heads, head_dim); ValueError when none exists."""
        key = (int(query_heads), int(kv_heads), int(head_dim))
        name = self._profile_by_shape.get(key)
        if name is None:
            known = ", ".join(
                f"{n} {q}/{kv}/{d}"
                for (q, kv, d), n in sorted(self._profile_by_shape.items(), key=lambda kv_: kv_[1])
            )
            raise ValueError(
                f"no WP-G3 cuda_graph coefficient set for geometry {key[0]}/{key[1]}/{key[2]} on {self.hardware}; calibrated: {known}"
            )
        return name

    def cp1_set(self, profile, mask):
        if mask not in MASKS:
            raise ValueError(f"mask must be one of {MASKS}, got {mask!r}")
        return self.geometries[profile][mask]

    def geometry_params(self, profile):
        entry = self.geometries[profile]
        return int(entry["query_heads"]), int(entry["kv_heads"]), int(entry["head_dim"])

    def f5(self, profile, mask, pieces, S_tokens, n_static=None):
        """The full CP1-form wall of one fused-attention call at the head's reference state (the cp = 1 wall at f0 on the served
        head; the raw kernel quantity of a ring round before the A11 entry)."""
        tiles = tuple(
            (tile_of(q, self.tile), tile_of(kv, self.tile)) for q, kv in active_pieces(pieces)
        )
        return dict(self._f5_tiles(profile, mask, tiles, int(S_tokens), n_static))

    @lru_cache(maxsize=8192)
    def _f5_tiles(self, profile, mask, tiles, S_tokens, n_static):
        q_heads, _kv, head_dim = self.geometry_params(profile)
        pieces = [(q * self.tile, kv * self.tile) for q, kv in tiles]
        return f5_wall(
            self.cp1_set(profile, mask),
            pieces,
            S_tokens,
            mask,
            q_heads,
            head_dim,
            p_hw=self.p_hw,
            static_us_per_slot=self.static_us_per_slot,
            thd_max=self.thd_max,
            n_static=n_static,
            tile=self.tile,
            block_rows=self.block_rows,
            half_occupancy_head_dim=self.half_occupancy_head_dim,
        )

    def o_graph_ms(self, P_q, profile):
        """O_graph(P_q, g) in ms: the in-graph per-call overhead of a fused-attention call (prep kernels + output memset + gaps),
        o0 + o1 x MiB(P_q x q_heads x head_dim x 2 B); the between-replay launch gap is recorded in the file and NOT subtracted (A15).
        """
        q_heads, _kv, head_dim = self.geometry_params(profile)
        mib = float(P_q) * (q_heads * head_dim) * 2.0 / MIB
        return (float(self.o_graph["o0_us"]) + float(self.o_graph["o1_us_per_MiB"]) * mib) * 1e-3

    # -- the clock input (A19) and the clock law (A25) ---------------------------------------------------
    def clock_scale_factor(self, f_mhz, law=DEFAULT_CLOCK_LAW, two_term_s=None):
        """The factor the kernel terms at f0 are multiplied by at clock f (holdout_common.clock_scale_factor): law 'alpha' = (f0 / f) ** alpha (A19 (iii),
        alpha 1 - the frozen GB200 forward law, validated on the lanes' 1330-1430 MHz cards); law 'two_term' = s x f0 / f + (1 - s) (A25: a compute part that
        follows the SM clock plus a clock-independent part; s = the hardware's clock-following share at f0, A25.1 report 41b; override for comparison). Both
        laws give 1.0 at f = f0; the glue / exchange terms are never clock-scaled under either law.
        """
        if law not in CLOCK_LAWS:
            raise ValueError(f"clock_law must be one of {CLOCK_LAWS}, got {law!r}")
        ratio = float(self.f0_mhz) / float(f_mhz)
        if law == "two_term":
            s = self.two_term_s if two_term_s is None else float(two_term_s)
            if s is None:
                raise ValueError(
                    f"the two-term clock law needs the hardware's clock-following share s (clock_input.two_term_s of head {self.head!r}; none recorded) - pass two_term_s"
                )
            if not (0.0 <= s <= 1.0):
                raise ValueError(
                    "two_term_s must lie in [0, 1] (the clock-following share of the kernel time at f0)"
                )
            return float(s) * ratio + (1.0 - float(s))
        return ratio ** float(self.alpha)

    def kernel_scale(
        self,
        cp_size,
        clock_mhz_by_rank=None,
        f_crit_mhz=None,
        law=DEFAULT_CLOCK_LAW,
        use_production_default=False,
        two_term_s=None,
    ):
        """(scale, record) of the clock input for one prediction: scale = clock_scale_factor(f_crit, law) with f_crit = the given critical clock or the
        minimum of the per-rank clocks (exactly cp values, one at cp = 1); with use_production_default and no clock, f = the hardware's production default
        clock of the forward phase (A24: the clock DURING the attention window on the slowest rank); None scale = no clock supplied (the prediction at f0,
        the default). A head without a clock input (A16) refuses a clock."""
        cp = int(cp_size)
        if law not in CLOCK_LAWS:
            raise ValueError(f"clock_law must be one of {CLOCK_LAWS}, got {law!r}")
        if clock_mhz_by_rank is not None and f_crit_mhz is not None:
            raise ValueError(
                "give either clock_mhz_by_rank (the CP group's per-rank heater-phase clocks) or f_crit_mhz (its minimum), not both"
            )
        if use_production_default and (clock_mhz_by_rank is not None or f_crit_mhz is not None):
            raise ValueError(
                "use_production_default_clock replaces a missing card: do not combine it with clock_mhz_by_rank / f_crit_mhz"
            )
        if clock_mhz_by_rank is None and f_crit_mhz is None and not use_production_default:
            if self.clock is None:
                return None, {
                    "applied": "not_applicable",
                    "note": self.block["ring_kernel_scale"].get("text")
                    or "this head has no clock input",
                }
            return None, {
                "applied": "f0_default",
                "law": law,
                "f0_mhz": self.f0_mhz,
                "alpha": self.alpha,
                "two_term_s": (self.two_term_s if law == "two_term" else None),
                "f_crit_mhz": None,
                "f_crit_rank": None,
                "scale": 1.0,
                "note": (
                    "no clock supplied: the prediction AT f0 (served_f0); supply the allocation's per-rank clocks read DURING the attention window or f_crit (A19 / A24), "
                    "or use_production_default_clock for the hardware's production default (A24)"
                ),
            }
        if self.clock is None:
            raise ValueError(
                f"the cuda_graph head {self.head!r} has no clock input: its CP1 form is not normalized to f0 (A19 belongs to the f0-normalized heads, "
                f"served {self.served_head!r}); drop clock_mhz_by_rank / f_crit_mhz / use_production_default_clock or use the served head"
            )
        rank, by_rank, applied = None, None, "scaled"
        if use_production_default:
            f_crit = self.production_default_clock_mhz.get(self.direction)
            if f_crit is None:
                raise ValueError(
                    f"hardware {self.hardware!r} has no production_default_clock_mhz for the {self.direction} direction in the data file (A24: it is read "
                    "from that hardware's train-step probe; pass f_crit_mhz instead)"
                )
            applied = "production_default"
        elif clock_mhz_by_rank is not None:
            values = [float(v) for v in clock_mhz_by_rank]
            if len(values) != cp:
                raise ValueError(
                    f"clock_mhz_by_rank needs exactly cp = {cp} per-rank clocks (the CP group's ranks in ring order), got {len(values)}"
                )
            if any(not (math.isfinite(v) and v > 0.0) for v in values):
                raise ValueError("clock_mhz_by_rank values must be finite positive MHz")
            rank = min(range(cp), key=lambda i: values[i])
            f_crit = values[rank]
            by_rank = values
        else:
            f_crit = float(f_crit_mhz)
            if not (math.isfinite(f_crit) and f_crit > 0.0):
                raise ValueError("f_crit_mhz must be a finite positive MHz value")
        scale = self.clock_scale_factor(f_crit, law, two_term_s)
        return scale, {
            "applied": applied,
            "law": law,
            "f0_mhz": self.f0_mhz,
            "alpha": (self.alpha if law == "alpha" else None),
            "two_term_s": (
                (self.two_term_s if two_term_s is None else float(two_term_s))
                if law == "two_term"
                else None
            ),
            "f_crit_mhz": f_crit,
            "f_crit_rank": rank,
            "f_by_rank_mhz": by_rank,
            "scale": scale,
            "phase": self.clock.get("phase"),
            "rule": (
                "cp = 1: O_graph + (F5(f0) - O_graph) x scale; cp > 1: every ring-round kernel entry x scale before the kappa W0 stretch; "
                "scale = (f0 / f) ** alpha (law alpha) or s f0 / f + (1 - s) (law two_term); glue / exchange terms unscaled"
            ),
        }

    # -- ring layer -------------------------------------------------------------------------------------
    def structure_for_cp(self, cp_size):
        """(communication structure, extrapolated) of a CP size: the head's recorded map for the measured / recorded CPs, the HARDWARE's
        topology rule otherwise (``topology_class_for_cp``: cp 2 = the pair class, cp <= gpus_per_node = the intra-node class, beyond = the
        inter-node class; on GB200: sequential pair / cp4_concurrent_intra_node / nvl72_domain); extrapolated when no W0 cell of the
        structure was measured at this CP. The W0 table of the head is keyed by the hardware's class names.
        """
        cp = int(cp_size)
        if cp < 2:
            raise ValueError("the ring layer needs cp >= 2")
        structure = self.structure_of_cp.get(cp)
        if structure is None:
            if cp == 2 and self.sequential_pair_structure:
                structure = self.sequential_pair_structure
            elif self.topology:
                structure, _role = topology_class_for_cp(self.topology, cp)
            elif cp <= 4:
                structure = "cp4_concurrent_intra_node"
            else:
                structure = "nvl72_domain"
        if structure not in self.ring["W0"]:
            raise ValueError(
                f"communication structure {structure!r} has no W0 coefficients on head {self.head!r} ({self.hardware} / {self.direction})"
            )
        return structure, cp not in self.calibrated_cps

    def ring_terms(self, cp_size, storage_tokens, profile, kappa_w0=None):
        """Every rank-independent ring term of a (cp, S, geometry) cell, in ms. ``kappa_w0`` overrides the concurrent-structure kappa
        (comparison only; the sequential pair keeps 0)."""
        cp, S = int(cp_size), int(storage_tokens)
        q_heads, kv_heads, head_dim = self.geometry_params(profile)
        structure, extrapolated = self.structure_for_cp(cp)
        w0 = self.ring["W0"][structure]
        payload_bytes = 2 * S * kv_heads * head_dim * 2  # B_kv: one rank's K and V
        # the per-step exchange bytes class (the head's schedule): W0 per step = latency + bytes_i / bandwidth of the hardware's class
        bytes_by_step = exchange_bytes_by_step(self.exchange_schedule, cp, payload_bytes)
        W0_us_by_step = [
            float(w0["latency_us"]) + float(w0["us_per_byte"]) * b for b in bytes_by_step
        ]
        W0_us = W0_us_by_step[0]  # round 0 (= the constant W0 of the forward schedule)
        x = S * (q_heads * head_dim)
        mib = x * 2 / MIB
        chain0 = (
            float(self.ring["chain_0_comm"]["a0_us"])
            + float(self.ring["chain_0_comm"]["a1_us_per_elem"]) * x
        )
        gmid = float(self.ring["g_mid"]["b0_us"]) + float(self.ring["g_mid"]["b1_us_per_elem"]) * x
        c_tail = float(self.ring["c_tail_us_per_MiB"])
        sequential = structure == self.sequential_pair_structure
        kappa = (
            self.kappa_W0_pair
            if sequential
            else (self.kappa_W0 if kappa_w0 is None else float(kappa_w0))
        )
        W0_ms = W0_us * 1e-3
        W0_by_step_ms = [w * 1e-3 for w in W0_us_by_step]
        stretch_by_step_ms = [kappa * w for w in W0_by_step_ms]
        constant = len(set(bytes_by_step)) == 1
        # a constant class keeps the served arithmetic cp x (kappa W0) bit for bit; a per-step class sums the per-step stretches
        stretch_total_ms = cp * stretch_by_step_ms[0] if constant else sum(stretch_by_step_ms)
        return {
            "cp": cp,
            "S": S,
            "structure": structure,
            "structure_extrapolated": extrapolated,
            "sequential_pair": sequential,
            "payload_bytes": payload_bytes,
            "sqd_MiB": mib,
            "W0_ms": W0_ms,
            "chain0_comm_ms": chain0 * 1e-3,
            "gmid_ms": gmid * 1e-3,
            "tail_ms": cp * c_tail * mib * 1e-3,
            "kappa_W0": kappa,
            "stretch_per_round_ms": kappa * W0_ms,
            "exchange_schedule": self.exchange_schedule,
            "exchange_bytes_by_step": bytes_by_step,
            "W0_by_step_ms": W0_by_step_ms,
            "stretch_by_step_ms": stretch_by_step_ms,
            "stretch_total_ms": stretch_total_ms,
            "exchange_bytes_constant": constant,
        }

    def first_exchange(self, terms, k0_ms):
        """slot_1 = rem_0 + P_1 of the round-0 recursion (A10) from the critical rank's k'_0 (ms)."""
        W0 = terms["W0_ms"]
        cp = terms["cp"]
        if terms["sequential_pair"]:
            rem = 0.5 * W0 + max(0.0, 0.5 * W0 - k0_ms)
            rem_rule = "W0/2 + max(0, W0/2 - k'_0)"
        else:
            rem = max(0.0, W0 - k0_ms)
            rem_rule = "max(0, W0 - k'_0)"
        last_round = cp == 2
        P1 = terms["chain0_comm_ms"] if last_round else 2.0 * terms["gmid_ms"]
        return {
            "rem0_ms": rem,
            "P1_ms": P1,
            "slot1_ms": rem + P1,
            "k0_ms": k0_ms,
            "rem0_rule": rem_rule,
            "P1_rule": (
                "chain_0^comm (round 1 is the last round)"
                if last_round
                else "2 g_mid (the whole round-1 preparation chain)"
            ),
        }

    def round_kernel(self, profile, section, pieces, storage_tokens, q_physical):
        """One ring round's kernel entry (ms): F5(round; P_q) - O_graph(P_q) at the head's reference state, before the clock / g scale
        and the stretch."""
        mask = SECTION_MASK[section]
        S_tokens = (
            int(q_physical)
            if q_physical
            else (int(storage_tokens) // 2 if section == "upper" else int(storage_tokens))
        )
        key = (profile, mask, tuple(pieces), S_tokens)
        hit = self._kernel_cache.get(key)
        if hit is None:
            raw = self.f5(profile, mask, list(pieces), S_tokens)
            og = self.o_graph_ms(S_tokens, profile)
            hit = dict(
                raw,
                f5_wall_ms=raw["wall_ms"],
                o_graph_ms=og,
                k_ms=raw["wall_ms"] - og,
                P_q=S_tokens,
            )
            if len(self._kernel_cache) > 200000:
                self._kernel_cache.clear()
            self._kernel_cache[key] = hit
        return hit

    # -- prediction -------------------------------------------------------------------------------------
    def predict_time_ms(
        self,
        lengths,
        cp_size,
        *,
        query_heads,
        kv_heads,
        head_dim,
        storage_tokens,
        sequence_alignment=None,
        dataset_alignment=1,
        clock_law=DEFAULT_CLOCK_LAW,
        use_production_default_clock=False,
    ):
        """Scalar-only execution of the SAME frozen formula; no trace allocation."""
        cp, S = int(cp_size), int(storage_tokens)
        values = tuple(int(n) for n in lengths)
        if len(values) > self.thd_max:
            raise ValueError("pack exceeds CUDA-graph static sequence slots")
        profile = self.geometry_profile(query_heads, kv_heads, head_dim)
        alignment = (
            self.alignment_default if sequence_alignment is None else int(sequence_alignment)
        )
        padded = layout_extend_last(values, cp, S, alignment, dataset_alignment)
        clock_scale, _ = self.kernel_scale(
            cp, law=clock_law, use_production_default=use_production_default_clock
        )
        if cp == 1:
            wall = self.f5(profile, "causal", cp1_pieces(values), S)["wall_ms"]
            og = self.o_graph_ms(S, profile)
            return wall if clock_scale is None else og + (wall - og) * clock_scale
        terms = self.ring_terms(cp, S, profile)
        scale = (
            self.ring_kernel_scale if clock_scale is None else clock_scale * self.ring_kernel_scale
        )
        k = [[0.0] * cp for _ in range(cp)]
        for item in round_pieces(values, padded, cp):
            kr = self.round_kernel(profile, item["section"], item["pieces"], S, item["q_physical"])
            k[item["rank"]][item["step"]] = kr["k_ms"] * scale
        sums = [sum(row) for row in k]
        crit = max(range(cp), key=lambda rank: sums[rank])
        kernel = sums[crit] + terms["stretch_total_ms"]
        glue = terms["chain0_comm_ms"] + (cp - 2) * terms["gmid_ms"] + terms["tail_ms"]
        first_exchange = self.first_exchange(terms, k[crit][0] + terms["stretch_by_step_ms"][0])
        return kernel + glue + first_exchange["slot1_ms"]

    def predict_with_details(
        self,
        lengths,
        cp_size,
        *,
        query_heads,
        kv_heads,
        head_dim,
        storage_tokens,
        sequence_alignment=None,
        clock_mhz_by_rank=None,
        f_crit_mhz=None,
        clock_law=DEFAULT_CLOCK_LAW,
        use_production_default_clock=False,
        two_term_s=None,
        g_ring=None,
        kappa_w0=None,
        dataset_alignment=1,
    ):
        """Forward core-attention wall (ms) of one pack on one CP group, with the decomposition and the provisional reasons.

        clock_mhz_by_rank / f_crit_mhz: the A19 clock input of the served head (the CP group's per-rank clocks read DURING the attention
        window, or their minimum); omitted = the prediction at f0, flagged (the default, unchanged outputs). clock_law: 'alpha' (the frozen
        a22_4 law, alpha 1) or 'two_term' (A25: s f0 / f + (1 - s), s from the data file; recommended for clocks far from f0); it scales the
        kernel part only. use_production_default_clock: without a card, predict at the hardware's production default clock (A24: GB200
        forward 1630 MHz) with the chosen law, flagged cuda_graph_clock_production_default. two_term_s: overrides s (comparison only).
        g_ring: overrides the constant ring kernel scale (1.0 on the served head, g_ring on the A16 head; comparison only). kappa_w0:
        overrides the concurrent-structure kappa (comparison with compose_a21.py's timed-read kappa; the served value is the config's).
        """
        values = [int(v) for v in lengths]
        cp, S = int(cp_size), int(storage_tokens)
        if cp <= 0 or S <= 0:
            raise ValueError("cp_size and storage_tokens must be positive")
        profile = self.geometry_profile(query_heads, kv_heads, head_dim)
        alignment = (
            self.alignment_default if sequence_alignment is None else int(sequence_alignment)
        )
        if len(values) > self.thd_max:
            raise ValueError(
                f"the pack holds {len(values)} sequences but the CUDA-graph THD batch has thd_max = {self.thd_max} cu_seqlens slots"
            )
        padded = layout_extend_last(values, cp, S, alignment, dataset_alignment)
        constant = self.ring_kernel_scale if g_ring is None else float(g_ring)
        if not constant > 0.0:
            raise ValueError("g_ring (the constant ring kernel scale) must be positive")
        clock_scale, clock = self.kernel_scale(
            cp, clock_mhz_by_rank, f_crit_mhz, clock_law, use_production_default_clock, two_term_s
        )
        reasons = [PROVISIONAL_REASON]
        if not self.is_served_head:
            reasons.append(REASON_HEAD_SUPERSEDED)
        if self.clock is not None and clock_scale is None:
            reasons.append(REASON_CLOCK)
        if clock.get("applied") == "production_default":
            reasons.append(REASON_PRODUCTION_CLOCK)
        if S not in self.calibrated_storages:
            reasons.append(REASON_STORAGE)
        if alignment != self.alignment_default:
            reasons.append(REASON_ALIGNMENT)
        if dataset_alignment != 1:
            reasons.append("cuda_graph_dataset_padding_transfer_provisional")
        causal = self.cp1_set(profile, "causal")
        if str(S) in (causal.get("cp1_form_acceptance_failed") or []):
            reasons.append(REASON_CP1_FORM)
        out = {
            "time_ms": None,
            "model_kind": MODEL_KIND,
            "launch_mode": CUDA_GRAPH_LAUNCH_MODE,
            "hardware": self.hardware,
            "direction": self.direction,
            "head": self.head,
            "head_status": self.head_status,
            "served_head": self.served_head,
            "cp1_form": self.cp1_form,
            "geometry_profile": profile,
            "geometry": [int(query_heads), int(kv_heads), int(head_dim)],
            "cp_size": cp,
            "storage_tokens": S,
            "sequence_alignment": alignment,
            "tail_policy": self.layout["tail_policy"],
            "valid_lengths": values,
            "padded_lengths": padded,
            "physical_tokens": S * cp,
            "sequence_count": len(values),
            "static_slots": self.thd_max - len(values),
            "status": self.status,
            "status_marker": self.status_marker,
            "model_sha256": self.data_sha256,
            "clock": clock,
            "provisional": True,
            "provisional_reasons": reasons,
        }
        if cp == 1:
            k = self.f5(profile, "causal", cp1_pieces(values), S)
            wall_f0 = k["wall_ms"]
            og = self.o_graph_ms(S, profile)
            wall = wall_f0 if clock_scale is None else og + (wall_f0 - og) * clock_scale
            out["time_ms"] = wall
            out["cp1"] = {
                key: k[key]
                for key in (
                    "k0_ms",
                    "bS_ms",
                    "sched_ms",
                    "jmax_ms",
                    "static_ms",
                    "makespan_tau",
                    "N_units",
                    "P",
                    "J_max",
                    "n_active",
                    "beyond_first_wave",
                    "form",
                )
            }
            out["cp1"].update(
                {
                    "wall_ms": wall,
                    "wall_f0_ms": wall_f0,
                    "o_graph_ms": og,
                    "kernel_scale": (1.0 if clock_scale is None else clock_scale),
                    "note": (
                        "cp = 1: the full CP1-form wall (per-call fixed cost k0 + b_S S, kernel tau x makespan [+ tau_jmax J_max], static slots); no ring term; "
                        "with a clock: O_graph + (wall(f0) - O_graph) x f0 / f (A19); the ring kernel scale is not applied"
                    ),
                }
            )
            return out
        terms = self.ring_terms(cp, S, profile, kappa_w0)
        if terms["structure_extrapolated"]:
            reasons.append(REASON_CP_STRUCTURE)
        scale = constant if clock_scale is None else clock_scale * constant
        rounds = round_pieces(values, padded, cp)
        k = [[0.0] * cp for _ in range(cp)]
        detail = {}
        for item in rounds:
            kr = self.round_kernel(profile, item["section"], item["pieces"], S, item["q_physical"])
            k[item["rank"]][item["step"]] = kr["k_ms"] * scale
            detail[(item["rank"], item["step"])] = (item, kr)
        sig = terms[
            "stretch_per_round_ms"
        ]  # the round-0 stretch kappa W0(b_0) (= every round's on the forward schedule)
        sig_step = terms[
            "stretch_by_step_ms"
        ]  # per-step kappa W0(b_i) of the head's exchange schedule
        sig_total = terms[
            "stretch_total_ms"
        ]  # cp x sig on a constant class (the served arithmetic), sum_i otherwise
        per_round_max = [max(k[rank][step] for rank in range(cp)) for step in range(cp)]
        per_rank_sum = [sum(k[rank]) for rank in range(cp)]
        crit = max(range(cp), key=lambda rank: per_rank_sum[rank])
        K_lock = sum(per_round_max) + sig_total
        K_crit = per_rank_sum[crit] + sig_total
        glue = terms["chain0_comm_ms"] + (cp - 2) * terms["gmid_ms"] + terms["tail_ms"]
        first_crit = self.first_exchange(terms, k[crit][0] + sig_step[0])
        first_lock = self.first_exchange(terms, per_round_max[0] + sig_step[0])
        wall = K_crit + glue + first_crit["slot1_ms"]
        wall_lock = K_lock + glue + first_lock["slot1_ms"]
        rows = []
        for step in range(cp):
            item, kr = detail[(crit, step)]
            rows.append(
                {
                    "r": step,
                    "section": item["section"],
                    "kv_rank": item["kv_rank"],
                    "n_active": kr["n_active"],
                    "P_q": kr["P_q"],
                    "N_units": kr["N_units"],
                    "beyond_first_wave": kr["beyond_first_wave"],
                    "f5_wall_ms": kr["f5_wall_ms"],
                    "o_graph_ms": kr["o_graph_ms"],
                    "k_entry_ms": kr["k_ms"],
                    "k_scaled_ms": k[crit][step],
                    "k_stretched_ms": k[crit][step] + sig_step[step],
                    "exchange_bytes": terms["exchange_bytes_by_step"][step],
                    "W0_step_ms": terms["W0_by_step_ms"][step],
                    "stretch_step_ms": sig_step[step],
                }
            )
        out["time_ms"] = wall
        out["ring"] = {
            "structure": terms["structure"],
            "structure_extrapolated": terms["structure_extrapolated"],
            "sequential_pair": terms["sequential_pair"],
            "payload_bytes": terms["payload_bytes"],
            "sqd_MiB": terms["sqd_MiB"],
            "W0_ms": terms["W0_ms"],
            "kappa_W0": terms["kappa_W0"],
            "stretch_per_round_ms": sig,
            "exchange_schedule": terms["exchange_schedule"],
            "exchange_bytes_by_step": terms["exchange_bytes_by_step"],
            "W0_by_step_ms": terms["W0_by_step_ms"],
            "stretch_by_step_ms": sig_step,
            "stretch_total_ms": sig_total,
            "composition": self.composition_kind,
            "kernel_scale": scale,
            "ring_kernel_scale_constant": constant,
            "clock_scale": (None if clock_scale is None else clock_scale),
            "g_ring": constant,
            "critical_rank": crit,
            "per_rank_sum_ms": per_rank_sum,
            "kernel_ms": K_crit,
            "kernel_unstretched_ms": per_rank_sum[crit],
            "glue_ms": glue,
            "glue": {
                "chain0_comm_ms": terms["chain0_comm_ms"],
                "gmid_slots": cp - 2,
                "gmid_ms": terms["gmid_ms"],
                "gmid_total_ms": (cp - 2) * terms["gmid_ms"],
                "tail_ms": terms["tail_ms"],
            },
            "slot1": first_crit,
            "wall_ms": wall,
            "wall_lockstep_ms": wall_lock,
            "kernel_lockstep_ms": K_lock,
            "slot1_lockstep": first_lock,
            "accounting": "critical_rank",
            "rounds_critical_rank": rows,
        }
        return out


@lru_cache(maxsize=None)
def load_cuda_graph_model(
    hardware=DEFAULT_HARDWARE, head=None, path=DATA_PATH, direction=DEFAULT_DIRECTION
):
    """The verified (sha256-pinned) cuda_graph model of one (hardware, direction) entry and head (None = the entry's served head); the
    defaults are gb200 / forward / a22_4. A ``served`` entry is this module's S4 model (gb200 / forward a22_4, gb300 / forward g5_f6, h100 / forward g4_a15); a
    ``served_round1`` entry is served by the vendored round-1 composer package (``cuda_graph_backward.BackwardRound1Model``: gb200 / backward
    head b7_1 - the WP-G3B B7.1 served model, partially validated; gb300 / backward head g5_bg5 - the WP-G5 BG4 port of the same composition with
    the GB300 constants, partially validated; every vendored file sha-pinned by the entry, one data directory per hardware); ``CudaGraphHeadPendingError``
    (a ValueError) for a pending entry - it names the campaign that will fill it (WP-G4 h100 / backward); never a fallback to another entry.
    """
    payload, digest = _load_payload(path)
    _hw, entry = _entry(payload, hardware, direction)
    if entry.get("status") in VENDORED_STATUSES:
        if (hardware, direction) not in VENDORED_PACKAGES:
            raise ValueError(
                f"the {hardware} / {direction} entry has status {entry.get('status')!r} but no vendored composer package is registered for it ({sorted(VENDORED_PACKAGES)})"
            )
        from .cuda_graph_backward import BackwardRound1Model, BackwardRound2Model

        resolved = resolve_head(head, tuple(entry["heads"]), entry["served_head"])
        head_round = int(
            ((entry["heads"].get(resolved) or {}).get("composer") or {}).get("round") or 1
        )
        # a served_round2 entry serves its round-2 head b12 through BackwardRound2Model and its superseded round-1 head b7_1 through BackwardRound1Model (comparison; bit-identical to the round-1 install)
        return (BackwardRound2Model if head_round == 2 else BackwardRound1Model)(
            payload, hardware, resolved, digest, direction
        )
    return CudaGraphAttentionModel(payload, hardware, head, digest, direction)


__all__ = [
    "DATA_PATH",
    "DATA_SHA256",
    "SCHEMA",
    "LEGACY_SCHEMAS",
    "STATUS_MARKER",
    "LAUNCH_MODES",
    "DEFAULT_LAUNCH_MODE",
    "CUDA_GRAPH_LAUNCH_MODE",
    "EAGER_STRUCTURAL_LAUNCH_MODE",
    "MODEL_KIND",
    "SERVED_HEAD",
    "SUPERSEDED_HEAD",
    "SUPERSEDED_HEADS",
    "HEADS",
    "DIRECTIONS",
    "DEFAULT_HARDWARE",
    "DEFAULT_DIRECTION",
    "ENTRY_STATUSES",
    "SERVING_STATUSES",
    "ROUND1_STATUS",
    "ROUND2_STATUS",
    "VENDORED_STATUSES",
    "VENDORED_PACKAGES",
    "EXCHANGE_SCHEDULE_FORWARD",
    "EXCHANGE_SCHEDULE_BACKWARD",
    "EXCHANGE_SCHEDULES",
    "EXCHANGE_SCHEDULE_OF_DIRECTION",
    "COMPOSITION_S4",
    "PROVISIONAL_REASON",
    "REASON_STORAGE",
    "REASON_ALIGNMENT",
    "REASON_CP_STRUCTURE",
    "REASON_CP1_FORM",
    "REASON_CLOCK",
    "REASON_PRODUCTION_CLOCK",
    "REASON_HEAD_SUPERSEDED",
    "CLOCK_LAWS",
    "DEFAULT_CLOCK_LAW",
    "MASKS",
    "SECTION_MASK",
    "layout_extend_last",
    "cp1_pieces",
    "round_pieces",
    "tile_of",
    "slots_for",
    "q_blocks",
    "causal_block_kv",
    "rect_block_kv",
    "active_pieces",
    "units_kv",
    "makespan",
    "j_max",
    "static_ms",
    "f5_wall",
    "resolve_head",
    "exchange_bytes_by_step",
    "topology_class_for_cp",
    "cuda_graph_topology_class",
    "normalize_payload",
    "served_entries",
    "pending_message",
    "CudaGraphHeadPendingError",
    "CudaGraphAttentionModel",
    "load_cuda_graph_model",
    "cuda_graph_hardware",
    "cuda_graph_heads",
    "cuda_graph_matrix",
]
