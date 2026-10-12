"""Frozen CPU prediction kernel; provenance is recorded in ORIGIN.json."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from functools import lru_cache
from pathlib import Path

from ..cuda_graph_attention_predictor import (
    CLOCK_LAWS,
    CUDA_GRAPH_LAUNCH_MODE,
    DEFAULT_CLOCK_LAW,
    PROVISIONAL_REASON,
    REASON_ALIGNMENT,
    REASON_CLOCK,
    REASON_CP1_FORM,
    REASON_CP_STRUCTURE,
    REASON_PRODUCTION_CLOCK,
    REASON_STORAGE,
    CudaGraphHeadPendingError,
    _entry,
    active_pieces,
    ceil_div,
    layout_extend_last,
    normalize_payload,
    pending_message,
    resolve_head,
    round_pieces,
    tile_of,
    topology_class_for_cp,
)

PACKAGE_DIR = Path(__file__).resolve().parent
DATA_DIR = PACKAGE_DIR / "data"
MANIFEST_PATH = DATA_DIR / "MANIFEST.json"
MANIFEST_SCHEMA = "dcp-scheduler/cuda-graph-backward-round1-manifest/1"
MANIFEST_SCHEMA_ROUND2 = "dcp-scheduler/cuda-graph-backward-round2-manifest/1"
PACKAGE = "dcp_scheduler.attention.cuda_graph_backward"
HEAD = "b7_1"
ENTRY_STATUS = "served_round1"
HEAD_STATUS = "served_round1_partially_validated"
MODEL_KIND = "wp_g3b_cuda_graph_f5b1_jmax_s4bwd_round1"
COMPOSITION_S4_BWD = "s4_bwd"
CP1_FORM = "F5b-1+jmax"
CLOCK_QUANTITY = "attention_phase_group_median"
CLOCK_LAW_SERVED = "structural"
F0_BWD_MHZ = 1600.0
CALIBRATED_CPS = (
    2,
    4,
    8,
    16,
)  # the ring read's nsys cells (ring_bwd.json inputs.nsys_cells_traced); cp32 is in structure_of_cp by the topology class only
CALIBRATED_STORAGES = (4096, 8192, 16384)
ALIGNMENT_DEFAULT = 128
THD_MAX = 64
STATIC_US_PER_SLOT = 0.155
TILE = 128
MIB = float(2**20)
# the files the composer reads (<data dir>/<name>); every one is pinned in MANIFEST.json and in the data file's entry
DATA_FILES = (
    "cp1_model_f5b1_jmax_bwd_f0.json",
    "ring_bwd_served.json",
    "ograph_bwd_served.json",
    "served_config_f_bwd.json",
    "structural_constants_bwd.json",
    "clock_law_lambda_bwd.json",
    "sealed_verdict_f_bwd.json",
)
# vendored when the campaign scored the item (gb300: the h_bwd_gb300 verdict was read in the same round; gb200 round 1: h_bwd unscored at the install) - pinned like the
# others when present, recorded in the prediction as sealed_cell_h (no reason flag: the per-prediction flag is the f item's cell, the design the prediction's (geometry, S, cp)
# cell belongs to on every hardware)
OPTIONAL_DATA_FILES = ("sealed_verdict_h_bwd.json",)
# the hardware this package serves, one vendored data directory each (tools/wp_g3_cuda_graph/vendor_backward_round1.py --hardware <hw>): the GB200 round-1 model
# (B7.1) and its port to GB300 (WP-G5 PREREGISTRATION_BWD_GB300.md BG1-BG5.1: the SAME composition and reading rules with the GB300 constants - f0_bwd 1800 MHz, the
# GB300 ring / CP1 / helper reads, the structural clock law served as the GB200 prior lambda 0.5757, the 1845 MHz telemetry ceiling). The module constants above
# (F0_BWD_MHZ, STATUS_MARKER, KNOWN_LIMITS, SEALED_SUMMARY) are the gb200 values byte for byte; the gb300 texts follow below.
# ---- ROUND 2 (gb200 only; PREREGISTRATION_BWD.md B10-B13.1, served B12; FREEZE_F_BWD2 4fc5732e... / FREEZE_H_BWD2 0209ac09..., 2026-10-08): the SAME round-1 kernel / ring /
# O_graph / constants files (data/) under the round-2 ATTRIBUTION (compose_bwd_r2_lib.BwdComposerR2): per-geometry per-step shares of the shared constants, static(n) ONCE per
# pack, no tail term, X = cp exchange steps x g0(head_dim) (kappa 0), the attention-phase group-median clock with readings >= 1750 MHz FLAGGED (never corrected). The round-2
# setting and the sealed verdicts are vendored in data_gb200_round2/ (tools/wp_g3_cuda_graph/vendor_backward_round2.py). Installed here 2026-10-09 (the coordinator's order after the
# round-3 read closed with no freeze, B16.2): the gb200 / backward entry has status served_round2, served head b12; b7_1 stays as the superseded round-1 head.
HEAD_ROUND2 = "b12"
ENTRY_STATUS_ROUND2 = "served_round2"
HEAD_STATUS_ROUND2 = "served_round2_partially_validated"
MODEL_KIND_ROUND2 = "wp_g3b_cuda_graph_f5b1_jmax_s4bwd_round2"
COMPOSITION_S4_BWD_R2 = "s4_bwd_r2"
FLAG_MHZ = 1750.0  # B11.1: rows whose group-median attention clock is >= 1750 MHz are FLAGGED in the verdict, never corrected
ROUND2_DATA_FILES = (
    "round2_served_setting.json",
    "served_config_f_bwd2.json",
    "sealed_verdict_f_bwd2.json",
    "sealed_verdict_h_bwd2.json",
)
ROUND2_HARDWARE = {
    "gb200": {
        "head": HEAD_ROUND2,
        "data_dir": "data_gb200_round2",
        "round1_data_dir": "data",
        "f0_mhz": 1600.0,
        "campaign": "WP-G3B",
        "clock_ceiling_mhz": 1800.0,
        "flag_mhz": FLAG_MHZ,
        "superseded_heads": ("b7_1",),
    }
}
REASON_BACKWARD_ROUND2 = "cuda_graph_backward_round2_partially_validated"  # every prediction of the head: f_bwd2 22 / 45, h_bwd2 6 / 24 (B13 / B13.1)
REASON_CLOCK_FLAG_1750 = "cuda_graph_backward_clock_flag_1750"  # B11.1: the applied group-median attention clock >= 1750 MHz (the telemetry-ceiling regime; flagged, not corrected)
REASON_HEAD_SUPERSEDED_BWD = (
    "cuda_graph_head_superseded"  # the round-1 head b7_1 selected on the round-2 entry
)
STATUS_MARKER_ROUND2 = (
    "served_round2_partially_validated: WP-G3B B12 served backward model (FREEZE_F_BWD2 / FREEZE_H_BWD2, 2026-10-08): the round-1 CP1 model F5b-1+jmax and ring block under the "
    "round-2 attribution - per-geometry per-step shares 0.917 / 0.952 / 0.400 (qwen3_30b / qwen3_5_397b / llama3_70b), static(n) once per pack, no tail, X = cp x g0(head_dim) "
    "with g0 d128 28.0 / d256 36.8 us, the attention-phase group-median clock (>= 1750 MHz flagged); sealed f_bwd2 22 / 45 cells PASS, 23 FAIL, 0 inconclusive (B13; pooled "
    "4.91 WAPE / 11.75 P90 / +0.80 % bias on 675 rows); h_bwd2 6 / 24 PASS, 16 FAIL, 2 inconclusive (B13.1; 5.26 / 16.61 / +0.97 on 1176 rows); round 3 read with NO freeze "
    "(B16.2); known limits: the d256 16K cp16 under-prediction (-9..-12 %, the ceiling lane), the pair layer at 4K (cp2 +5..+13 %), the shares without mechanism, the d256 CP1 "
    "P90, the >= 1750 MHz readings flagged not corrected"
)
KNOWN_LIMITS_ROUND2 = (
    "B12 (b): the d256 16K cp16 under-prediction (-9..-12 % on the f-design long single / pair sequences; -11 / -15 % on the two round-2 d256 allocations) - the telemetry-ceiling lane whose "
    "clock the group median mis-reads (B15.1: a clock-INPUT defect; the round-3 instrument read established the probe quantity but no freeze, B16.2)",
    "B12 (b): the pair layer at 4K (cp2 +5..+13 % on the cov cells); B13 (ii): cp2 2 / 9 f cells PASS, the one-node layer (cp1 / cp2 / cp4) 6 / 27 with a positive bias at cp2 / cp4 on qwen3_30b "
    "(+5..+14 %) and a negative one on qwen3_5_397b cp4",
    "B10.2 (c) / B12 (b): the per-geometry per-step shares of the shared constants (0.917 / 0.952 / 0.400) are per-(hardware, geometry) CONSTANTS WITHOUT a mechanism",
    "B12 (b): the d256 CP1 P90 (period-4 wave quantization); B13 (ii): cp1 1 / 9 f cells PASS (P90 10.2-14.7, bias -1.7..-4.8) on all three geometries",
    "B11.1 / B12 (b): attention-phase clock readings >= 1750 MHz (the telemetry ceiling) are FLAGGED (cuda_graph_backward_clock_flag_1750), never corrected; the scorer flagged 452 / 675 f rows and 964 / 1176 h rows",
    "B13.1: h_bwd2 6 / 24 (cp2 0 / 6, cp4 2 / 6, cp8 1 / 6, cp16 3 / 6; qwen3_30b 0 / 8); two cells inconclusive (qwen3_5_397b 4096 / 16384 cp8: admitted rows below the minimum after the band / flag rules); "
    "the x0-class X term read 9 / 24 alongside (not selected)",
    "the composition is served UNIFORMLY at the group-median clock: the round-3 read (B16.2 (ii)(c)) showed that when one rank of a non-ceiling 16-rank ring sits 7-13 % below the group clock the "
    "measured ring wall does NOT lengthen by the critical rank's clock ratio - the dependence on the per-rank clock DISTRIBUTION is a structural question of the CP > 1 layer, open",
)
SEALED_SUMMARY_ROUND2 = {
    "item": "f_bwd2",
    "cells_pass": 22,
    "cells_fail": 23,
    "cells_inconclusive": 0,
    "cells": 45,
    "pooled": {"n": 675, "wape_pct": 4.913, "p90_pct": 11.753, "bias_pct": 0.796},
    "amendment": "B13 (2026-10-08 05:00): the first and only read",
    "h_bwd2": {
        "cells_pass": 6,
        "cells_fail": 16,
        "cells_inconclusive": 2,
        "cells": 24,
        "pooled": {"n": 1176, "wape_pct": 5.261, "p90_pct": 16.611, "bias_pct": 0.97},
        "amendment": "B13.1 (2026-10-08 12:30)",
    },
    "vs_round_1": "f 13 -> 22 / 45 (round 1 as scored; 16 by the numbers), h 5 -> 6 / 24 (ROUND2_CLOSEOUT_BWD.md section 6)",
    "round_3": "READ, NO FREEZE (B16.2, 2026-10-09): five chains on the full calibration record, no chain meets both gates; the served GB200 backward model remains round 2's",
}

ROUND1_HARDWARE = {
    "gb200": {
        "head": "b7_1",
        "data_dir": "data",
        "f0_mhz": 1600.0,
        "campaign": "WP-G3B",
        "clock_ceiling_mhz": 1800.0,
        "gpu_label": "GB200",
        "production_clock_text": "the GB200 production backward DURING-window clock (1598 rounded; A24 / B0.2)",
    },
    "gb300": {
        "head": "g5_bg5",
        "data_dir": "data_gb300",
        "f0_mhz": 1800.0,
        "campaign": "WP-G5",
        "clock_ceiling_mhz": 1845.0,
        "gpu_label": "GB300",
        "production_clock_text": "the GB300 production backward DURING-window clock (fdur_bwd_gb300 1802 rounded; PREREGISTRATION_GB300.md P1.2 / A24 (ii))",
    },
}
DATA_DIRS = {hw: PACKAGE_DIR / spec["data_dir"] for hw, spec in ROUND1_HARDWARE.items()}
HEADS_BY_HARDWARE = {hw: spec["head"] for hw, spec in ROUND1_HARDWARE.items()}
REASON_BACKWARD_ROUND1 = "cuda_graph_backward_round1_partially_validated"  # every prediction of a round-1 head (b7_1 superseded, g5_bg5): f_bwd 13 / 45 and 14 / 45
REASON_SEALED_CELL_FAILED = "cuda_graph_backward_sealed_cell_failed"  # the (geometry, S, cp) cell FAILED the 5 / 10 gate in the once-read f_bwd verdict
REASON_SEALED_CELL_INCONCLUSIVE = "cuda_graph_backward_sealed_cell_inconclusive"
REASON_SEALED_CELL_UNSCORED = "cuda_graph_backward_sealed_cell_unscored"  # outside the 45 sealed cells (cp 32 / non-power cp / another S)
STATUS_MARKER = (
    "served_round1_partially_validated: WP-G3B B7.1 served backward model (FREEZE_F_BWD / FREEZE_H_BWD, 2026-10-07); sealed f_bwd 13 / 45 cells PASS, "
    "27 FAIL, 5 inconclusive (B8.2 re-read; pooled 8.53 WAPE / 15.96 P90 / -2.21 % bias on 675 rows); h_bwd unscored; SUPERSEDED 2026-10-09 by the round-2 head b12 "
    "(served_round2; this head stays selectable, outputs unchanged); known limits: "
    "llama3_70b cp2 / cp4 over-prediction +10..+16 %, qwen3_5_397b d256 cp16 under-prediction -9..-22 % growing with S, d256 CP1 P90 13-15, pair layer "
    "P90 22, short-pack fragment floor, prep accounting +2.35 bias, clock ceiling >= 1800 MHz"
)
KNOWN_LIMITS = (
    "llama3_70b (64 heads, d128): every cp >= 2 cell FAILS, OVER-predicted +10..+16 % at cp2 / cp4 (4K / 8K), +6..+10 % at cp8 / cp16; pooled 7.51 WAPE / 19.91 P90 / +7.24 bias "
    "(B9 outcome: the bprop ENTRY carries ~90 % of the excess - per-step constants not paid on fragment steps, the CP1 tau ~10 % high at cp > 1 for this geometry; many-piece packs worst: ms +24..+36 %)",
    "qwen3_5_397b (d256): cp16 UNDER-predicted -9..-22 % growing with S (16384 cp16 22.13 / 26.83 / -22.13 FAIL; sg / pair families -27..-29 %; 4K / 8K cp16 -9.7 / -9.1 inconclusive); "
    "cp2 and cp8 cells fail; the CP1 cells fail on P90 only (13.5-15.0: the d256 unit known limit, carried by n 5-16 packs); pooled 11.81 / 13.34 / -10.90 (re-read + carried)",
    "the pair layer (cp2): P90 22.27 pooled; the traced exposed exchange + gap is a per-pack CONSTANT 40-60 us vs the served X 26-38 us (B9 (5))",
    "short-pack fragment floor: the per-call floor 60-160 us of a fragment step is only partly inside k0 + static (calibration short packs 6.04 WAPE / +3.41 bias)",
    "prep accounting: the class-direct ring prep is served ON TOP of k0 (+2.35 bias on the 108 calibration walls; a possible partial double count); H2 (traced vs timed glue, median ratio 1.21) fails",
    "clock ceiling: attention-phase readings >= 1800 MHz are a telemetry ceiling (+-4 %), flagged not corrected; without a card the prediction is at f0 = 1600 MHz = the production backward default "
    "(production during-window clocks spread p10 1405 / p90 1732 MHz: roughly -8..+14 % on the clock-following part)",
    "a once-per-pack post-loop tail (convert_dq + reduce_head, 82-388 us) is named by no per-step term: indistinguishable from k0 at CP1, paid per STEP at cp > 1 (B9 (3))",
    "coverage: cp 32 and every non-power-of-two cp take the hardware's topology class without a measured W0 cell (flagged cuda_graph_cp_structure_extrapolated); the meq (n ~4) / mixu (n ~7) "
    "families have no calibration analogue; h_bwd (the sg / eq grid) is unscored",
)
SEALED_SUMMARY = {
    "item": "f_bwd",
    "cells_pass": 13,
    "cells_fail": 27,
    "cells_inconclusive": 5,
    "cells": 45,
    "pooled_served": {"n": 675, "wape_pct": 8.531, "p90_pct": 15.962, "bias_pct": -2.212},
    "by_geometry": {
        "llama3_70b": {"wape_pct": 7.515, "p90_pct": 19.912, "bias_pct": 7.241},
        "qwen3_30b": {"wape_pct": 3.842, "p90_pct": 11.405, "bias_pct": -1.552},
        "qwen3_5_397b": {"wape_pct": 11.81, "p90_pct": 13.339, "bias_pct": -10.901},
    },
    "by_cp": {
        "1": {"wape_pct": 4.772, "p90_pct": 10.81, "bias_pct": -3.205},
        "2": {"wape_pct": 7.589, "p90_pct": 22.266, "bias_pct": 5.528},
        "4": {"wape_pct": 5.727, "p90_pct": 16.279, "bias_pct": 2.55},
        "8": {"wape_pct": 6.566, "p90_pct": 10.602, "bias_pct": 0.539},
        "16": {"wape_pct": 10.651, "p90_pct": 14.789, "bias_pct": -5.84},
    },
    "h_bwd": "not scored (B8.3: the llama3_70b cp16 resume and the 1-node job were still running when round 1 was installed here)",
    "round_2": "B10 open (B9 outcome 2026-10-07 20:40): the ring-step entry re-attributed (per-call vs per-step constants, the once-per-pack tail, tau at cp > 1 per geometry, the d256 "
    "few-piece long-sequence entry, X's constant, a per-piece d256 term); the CP1 functional form unchanged; round-1 served model and FREEZE_F/H_BWD stay the record of round 1",
}
# ---- gb300 (WP-G5 backward, PREREGISTRATION_BWD_GB300.md BG1-BG5.1; FREEZE_F_BWD_GB300 / FREEZE_H_BWD_GB300 installed 2026-10-08 (BG4); verdicts read once (BG5 / BG5.1); round 2 deferred)
STATUS_MARKER_GB300 = (
    "served_round1_partially_validated: WP-G5 BG4 served GB300 backward model (FREEZE_F_BWD_GB300 / FREEZE_H_BWD_GB300, 2026-10-08) = the GB200 B7.1 structure with the GB300 "
    "constants (f0_bwd 1800 MHz; the structural clock law served as the GB200 prior lambda 0.5757); sealed f_bwd_gb300 14 / 45 cells PASS, 31 FAIL, 0 inconclusive (BG5 + the "
    "BG5.1 restricted re-read of the resumed cell; pooled 5.00 WAPE / 17.10 P90 / +0.59 % bias on 675 rows); h_bwd_gb300 6 / 24 PASS, 18 FAIL (4.72 / 16.62 / +2.60 on 1176 "
    "rows); GB300 round 2 DEFERRED by the user (2026-10-08 13:00: GB200 round 3 first); known limits: the pair / one-node layer (cp2 0 / 9 f cells, +6.0 % pooled), P90 pack "
    "spread (every multi-node llama3_70b 16K f cell fails on P90 alone), qwen3_30b over-predicted +4.8 / +5.8 %, qwen3_5_397b under-predicted -3.4 %, d256 causal CP1 cells, "
    "the prior-served clock law (+1.0..+3.3 % extrapolation band), readings >= 1845 MHz flagged"
)
KNOWN_LIMITS_GB300 = (
    "the pair / one-node layer: cp2 0 / 9 f cells and 1 / 6 h cells pass; cp2 pooled 8.18 / 26.09 / +6.00 on f (91 % of the rows over-predicted; qwen3_30b 4K / 8K / 16K +22.6 / +12.2 / +8.0, llama3_70b "
    "+17.5 / +10.1 / +7.6) and 7.88 / 21.87 / +5.06 on h - the GB200 round-2 picture (ROUND2_CLOSEOUT_BWD.md section 5)",
    "the P90 pack spread: pooled WAPE at the 5 % reading (f 5.00, h 4.72) with the bias inside +3 %, but the per-cell gate fails on the P90 (pooled 17.1 / 16.6); every multi-node llama3_70b f cell at 16K fails "
    "on P90 alone (cp8 / cp16 P90 12.15 / 11.04 at |bias| < 2.3)",
    "geometry-signed bias larger than on GB200: qwen3_30b over-predicted (+4.77 % f, +5.75 % h; 2 / 15 f cells, 0 / 8 h cells pass), qwen3_5_397b under-predicted (-3.39 % f; 9 / 15 f, 6 / 8 h cells pass), "
    "llama3_70b +2.56 % f (3 / 15), +4.21 % h (0 / 8)",
    "the d256 causal CP1 cells (qwen3_5_397b): the three causal units fail the CP1 fit gate as on GB200 (BG3.1; LOLO 6.51 / 10.7); sealed qwen3_5_397b 16K cp1 11.17 / 18.97 / -8.10 FAIL",
    "the clock law is PRIOR-served (BG2.2): the GB200 B6.1 structural lambda 0.5757 +- 0.031 carried to GB300 because the GB300 reads are not independent (the sampled-clock ceiling 1845 MHz lies below "
    "every light-arm clock); the GB300 readings (card-clock lambda 0.356 / s 0.688, implied-clock 0.012 / 0.908) are recorded alternatives; the extrapolation from the lane clock to f0_bwd 1800 "
    "differs between them by +1.0 / +1.9 / +3.3 % at 1720 / 1650 / 1540 MHz",
    "the H1 cell qwen3_30b p8192 cp4 (89.3 % of the exchange hidden): X stays x0 + kappa sum W0 (remainder.served false); the pre / prep alternatives move 2-8 cells (prep_excess_over_cp1 22 / 45 on f "
    "with a -4.1 % bias; pre_folded closer on 945 / 1176 h rows) - the first candidates for a GB300 round 2",
    "the d128 bprop kernel reads Q-PARALLEL on GB300 (slope ratio 1.055; GB200 kv-parallel 1.115): by decision (1) ONE form on every hardware the served form stays F5b-1+jmax (LOLO 2.52) - F5b-Q+jmax "
    "(LOLO 3.07) is a frozen alternative",
    "clock ceiling: attention-phase readings >= 1845 MHz (BG1 R6c) are a telemetry ceiling, flagged not corrected (none raised on the sealed rows); without a card the prediction is at f0 = 1800 MHz = the "
    "production backward default (= the DURING-window clock of the GB300 train-step probe, P1.2)",
    "coverage: cp 32 and every non-power-of-two cp take the hardware's topology class without a measured W0 cell (flagged cuda_graph_cp_structure_extrapolated); R1 marginal (one record 0.27 ms) and R11 "
    "pack independence not met (informational, BG1)",
)
SEALED_SUMMARY_GB300 = {
    "item": "f_bwd_gb300",
    "cells_pass": 14,
    "cells_fail": 31,
    "cells_inconclusive": 0,
    "cells": 45,
    "first_read": {
        "cells_pass": 14,
        "cells_fail": 30,
        "cells_inconclusive": 1,
        "note": "BG5: the resumed cell llama3_70b 4096 cp1 filtered (step_failed / job_id_mismatch_log_vs_results) -> inconclusive; "
        "BG5.1 (scorer mechanics, the resume_of record): the one cell re-read, 44 carried -> FAIL; 14 / 31 / 0",
    },
    "pooled_served": {"n": 675, "wape_pct": 4.999, "p90_pct": 17.102, "bias_pct": 0.592},
    "by_geometry": {
        "llama3_70b": {"wape_pct": 4.331, "p90_pct": 17.32, "bias_pct": 2.557},
        "qwen3_30b": {"wape_pct": 5.324, "p90_pct": 20.882, "bias_pct": 4.767},
        "qwen3_5_397b": {"wape_pct": 5.451, "p90_pct": 10.341, "bias_pct": -3.392},
    },
    "by_cp": {
        "1": {"wape_pct": 5.749, "p90_pct": 10.73, "bias_pct": -3.72},
        "2": {"wape_pct": 8.183, "p90_pct": 26.092, "bias_pct": 6.005},
        "4": {"wape_pct": 4.32, "p90_pct": 15.927, "bias_pct": 2.302},
        "8": {"wape_pct": 4.097, "p90_pct": 13.094, "bias_pct": 1.684},
        "16": {"wape_pct": 5.166, "p90_pct": 11.379, "bias_pct": -0.871},
    },
    "h_bwd": {
        "item": "h_bwd_gb300",
        "cells_pass": 6,
        "cells_fail": 18,
        "cells_inconclusive": 0,
        "cells": 24,
        "pooled_served": {"n": 1176, "wape_pct": 4.716, "p90_pct": 16.619, "bias_pct": 2.602},
        "by_cp": {
            "2": {"wape_pct": 7.879, "p90_pct": 21.867, "bias_pct": 5.062},
            "4": {"wape_pct": 4.648, "p90_pct": 15.672, "bias_pct": 3.226},
            "8": {"wape_pct": 4.141, "p90_pct": 15.847, "bias_pct": 3.24},
            "16": {"wape_pct": 4.534, "p90_pct": 12.622, "bias_pct": 1.666},
        },
        "by_geometry": {
            "llama3_70b": {"wape_pct": 5.252, "p90_pct": 14.091, "bias_pct": 4.211},
            "qwen3_30b": {"wape_pct": 6.124, "p90_pct": 19.812, "bias_pct": 5.752},
            "qwen3_5_397b": {"wape_pct": 3.445, "p90_pct": 9.43, "bias_pct": -0.636},
        },
    },
    "round_2": "DEFERRED by the user (2026-10-08 13:00: GB200 round 3 first; STATUS.md); GB300_BWD_CLOSEOUT.md section 5 names the GB300 ring pre / prep terms and the one-node layer as the first "
    "candidates; the round-1 served model and FREEZE_F/H_BWD_GB300 stay the record of the GB300 backward program",
}


# ----------------------------------------------------------------------------------------------------------
# vendored files (pinned twice: MANIFEST.json and the data file's entry)
# ----------------------------------------------------------------------------------------------------------
def sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_manifest(path=MANIFEST_PATH) -> dict:
    manifest = json.loads(Path(path).read_text())
    if manifest.get("schema") not in (MANIFEST_SCHEMA, MANIFEST_SCHEMA_ROUND2):
        raise ValueError(
            f"{path}: manifest schema {manifest.get('schema')!r}, expected {MANIFEST_SCHEMA!r} or {MANIFEST_SCHEMA_ROUND2!r}"
        )
    required = ROUND2_DATA_FILES if manifest.get("schema") == MANIFEST_SCHEMA_ROUND2 else DATA_FILES
    missing = [name for name in required if name not in manifest["vendored"]]
    if missing:
        raise ValueError(f"{path}: the manifest lacks the vendored files {missing}")
    return manifest


def load_vendored(name, expected_sha256=None, data_dir=DATA_DIR) -> tuple:
    """(payload, sha256) of one vendored file; the digest must equal ``expected_sha256`` when given (the data file's pin). ``data_dir`` = the hardware's vendored
    directory (DATA_DIRS; the gb200 ``data/`` by default)."""
    if name not in DATA_FILES + OPTIONAL_DATA_FILES + ROUND2_DATA_FILES:
        raise ValueError(
            f"{name!r} is not a vendored file of {PACKAGE} ({DATA_FILES + OPTIONAL_DATA_FILES + ROUND2_DATA_FILES})"
        )
    path = Path(data_dir) / name
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(
            f"vendored backward round-1 file {name} hash mismatch: {digest}, pinned {expected_sha256}"
        )
    return json.loads(raw), digest


# ----------------------------------------------------------------------------------------------------------
# CP1 backward form F5b-1+jmax (cp1_makespan_bwd.py: kv_units, greedy_chain, predict_ms, CP1BwdModel.kernel) - verbatim arithmetic
# ----------------------------------------------------------------------------------------------------------
def block_tiles(R_bwd) -> int:
    rb = int(R_bwd) // TILE
    if rb < 1 or int(R_bwd) % TILE:
        raise ValueError(f"R_bwd must be a positive multiple of {TILE} rows (got {R_bwd})")
    return rb


def kv_units(pieces, mask, q_heads, R_bwd) -> list:
    """KV-block units in launch order (pack x heads x blocks): the Q tiles each unit walks (causal: from the diagonal on; rectangle: every Q tile)."""
    rb = block_tiles(R_bwd)
    out = []
    for q, kv in active_pieces(pieces):
        tq, tkv = tile_of(q), tile_of(kv)
        if mask == "causal":
            per = [max(1, tq - (b - 1) * rb) for b in range(1, ceil_div(tq, rb) + 1)]
        elif mask == "rectangle":
            per = [tq] * ceil_div(tkv, rb)
        else:
            raise ValueError(mask)
        for _h in range(int(q_heads)):
            out.extend(per)
    return out


def greedy_chain(units, P, rho) -> tuple:
    """The served greedy list schedule with the critical slot's composition: (makespan, n_chain, chain_tiles, N, J_max); makespan == rho n_chain + chain_tiles exactly
    (PB B6: tile = tau x chain_tiles follows the clock; tau x rho x n_chain belongs to the constant part).
    """
    import heapq

    from ..dispatch import accelerated_chain

    compiled = accelerated_chain(units, P, rho)
    if compiled is not None:
        return compiled
    costs = [(float(rho) + float(k), int(k)) for k in units]
    N = len(costs)
    if N == 0:
        return 0.0, 0, 0, 0, 0
    jmax = max(k for _, k in costs)
    if N <= int(P):
        c, k = max(costs, key=lambda t: t[0])
        return c, 1, k, N, jmax
    if costs[0][0] > 0 and all(pair == costs[0] for pair in costs):
        count = ceil_div(N, P)
        end = 0.0
        for _ in range(count):
            end += costs[0][0]
        return end, count, count * costs[0][1], N, jmax
    heap = [(0.0, i) for i in range(int(P))]
    heapq.heapify(heap)
    n_units = [0] * int(P)
    tiles = [0] * int(P)
    end_t, end_i = 0.0, 0
    for c, k in costs:
        e, i = heap[0]
        e += c
        n_units[i] += 1
        tiles[i] += k
        heapq.heapreplace(heap, (e, i))
        if e > end_t:
            end_t, end_i = e, i
    return end_t, n_units[end_i], tiles[end_i], N, jmax


@lru_cache(maxsize=1024)
def kernel_dispatch(units, P, rho):
    span, n_chain, chain_tiles, count, maximum = greedy_chain(units, P, rho)
    return span, n_chain, chain_tiles, count, maximum


@lru_cache(maxsize=4096)
def kernel_dispatch_tiles(tiles, mask, q_heads, rows, slots, rho):
    pieces = [(q * TILE, kv * TILE) for q, kv in tiles]
    return kernel_dispatch(tuple(kv_units(pieces, mask, q_heads, rows)), slots, rho)


def static_ms(n_active, us_per_slot=STATIC_US_PER_SLOT, thd_max=THD_MAX) -> float:
    """The forward's static term 0.155 us x (64 - n) (PB 3.1 as written; the served static spec is forward_slots)."""
    return float(us_per_slot) * 1e-3 * (int(thd_max) - int(n_active))


def table_for(head_dim, table, fallback=None) -> int:
    """A per-head_dim structural constant: the exact head_dim, else the largest tabulated head_dim below it, else the smallest (cp1_makespan_bwd.p_for)."""
    tab = {int(k): int(v) for k, v in (table or {}).items() if v is not None}
    hd = int(head_dim)
    if hd in tab:
        return tab[hd]
    if not tab:
        return int(fallback)
    below = [k for k in tab if k <= hd]
    return tab[max(below)] if below else min(tab.values())


def helper_term_eval(
    block, which, q_tokens, kv_tokens, n, q_heads, kv_heads, traced=False
) -> float:
    """One helper term (H_q | H_kv) from its read block (bwd_common.helper_term_eval): form 'token' o0 + o_q q + o_kv kv + o_n n (the CP1 model, B4 per-head_dim token form);
    form 'head_token' h0 + c x (q x q_heads | kv x kv_heads) + c_n x n (the ring glue helpers, B4.1). traced: the as-traced class fit (the U record only).
    """
    c = block
    if traced and isinstance(c, dict) and c.get("as_traced"):
        c = c["as_traced"]
    form = (
        (c.get("form") if isinstance(c, dict) else None)
        or block.get("form")
        or (
            "head_token"
            if "h0_us" in (c.get("coef", c) if isinstance(c, dict) else {})
            else "token"
        )
    )
    c = c.get("coef", c)
    if form == "head_token":
        heads = q_heads if which == "H_q" else kv_heads
        if heads is None:
            raise ValueError("the head-token helper form needs the geometry (q_heads / kv_heads)")
        tok = float(q_tokens) if which == "H_q" else float(kv_tokens)
        return (
            float(c.get("h0_us", 0.0))
            + float(c.get("c_us_per_head_token", 0.0)) * tok * float(heads)
            + float(c.get("c_n_us_per_sequence", 0.0)) * float(n)
        )
    return (
        float(c.get("o0_us", 0.0))
        + float(c.get("o_q_us_per_token", 0.0)) * float(q_tokens)
        + float(c.get("o_kv_us_per_token", 0.0)) * float(kv_tokens)
        + float(c.get("o_n_us_per_sequence", 0.0)) * float(n)
    )


def helper_terms_us(
    spec_by_hd, head_dim, q_heads, kv_heads, q_tokens, kv_tokens, n, traced=False
) -> tuple:
    """(H_q, H_kv) in us from the spec block of the geometry's head_dim (bwd_common.helper_terms_us); (0, 0) without a spec."""
    if not spec_by_hd:
        return 0.0, 0.0
    blk = spec_by_hd.get(int(head_dim)) or spec_by_hd.get(str(int(head_dim)))
    if not blk:
        return 0.0, 0.0
    return (
        helper_term_eval(blk["H_q"], "H_q", q_tokens, kv_tokens, n, q_heads, kv_heads, traced),
        helper_term_eval(blk["H_kv"], "H_kv", q_tokens, kv_tokens, n, q_heads, kv_heads, traced),
    )


class CP1BackwardForm:
    """The served CP1 backward model file (cp1_model_f5b1_jmax_bwd_f0.json): per geometry a causal and a rectangle coefficient set of the form F5b-1+jmax at f0_bwd;
    ``kernel(g, mask, pieces, S_tokens)`` = cp1_makespan_bwd.CP1BwdModel.kernel (the fitted form + the token-form helper terms = the CP1 wall at f0).
    """

    def __init__(self, fit: dict) -> None:
        m = fit["model"]
        self.fit = fit
        self.form = str(m["form"])
        if self.form != CP1_FORM:
            raise ValueError(
                f"the vendored CP1 backward model has form {self.form!r}; this composer serves {CP1_FORM!r} (KV-block units + the jmax tail)"
            )
        self.f0_mhz = float(m.get("f0_mhz") or F0_BWD_MHZ)
        self.consts = {
            k: (
                {int(a): int(b) for a, b in v.items()}
                if isinstance(v, dict) and k.endswith("_by_head_dim")
                else v
            )
            for k, v in (m.get("structural") or {}).items()
        }
        self.static_us = float(m.get("static_us_per_slot", STATIC_US_PER_SLOT))
        self.static_spec = m.get("static") or {"form": "forward_slots"}
        if self.static_spec.get("form", "forward_slots") != "forward_slots":
            raise ValueError(
                f"the vendored CP1 backward model serves the static form {self.static_spec.get('form')!r}; this composer serves forward_slots (0.155 us x (64 - n))"
            )
        self.thd_max = int(self.static_spec.get("thd_max", THD_MAX))
        ht = m.get("helper_terms")
        self.helper_terms = (
            {int(k): v for k, v in (ht.get("by_head_dim") or {}).items()}
            if ht and ht.get("applied")
            else None
        )
        self.helper_variant = (ht or {}).get("variant")
        self.u_treatment = m.get("u_treatment") or (
            "subtracted_in_fit" if m.get("kernel_excludes_U") else "inside_k0"
        )
        if self.u_treatment != "none":
            raise ValueError(
                f"the vendored CP1 backward model has u_treatment {self.u_treatment!r}; the B7.1 served model is 'none' (no O_graph subtraction)"
            )
        self.clock_law = m.get("clock_law") or {}
        self.geoms = {g: dict(v) for g, v in m["geometries"].items()}
        self.geometry_params = {g: dict(v) for g, v in (m.get("geometry_params") or {}).items()}

    def set_for(self, g, mask) -> dict:
        cs = (self.geoms.get(g) or {}).get(mask)
        if not cs or cs.get("status") != "fitted":
            raise KeyError(f"CP1 backward model {self.form}: no fitted {mask} set for {g!r}")
        return cs

    def params(self, g) -> tuple:
        p = self.geometry_params[g]
        return int(p["q_heads"]), int(p["kv_heads"]), int(p["head_dim"])

    def kernel(self, g, mask, pieces, S_tokens, n_static=None) -> dict:
        """The CP1 wall at f0 of one fused_attn_bwd call: wall_ms = form_ms + helper_ms; with the B6 split tile_ms / const_ms (U excluded) and the helper terms."""
        cs = self.set_for(g, mask)
        q_heads, kv_heads, head_dim = self.params(g)
        act = active_pieces(pieces)
        n_act = len(act)
        n_st = n_act if n_static is None else int(n_static)
        st = static_ms(n_st, self.static_us, self.thd_max)
        r_tab = {
            int(k): int(v)
            for k, v in (self.consts.get("R_bwd_rows_by_head_dim") or {}).items()
            if v is not None
        }
        R = (
            r_tab[int(head_dim)]
            if int(head_dim) in r_tab
            else int(self.consts.get("R_bwd_rows") or 256)
        )  # bwd_common.r_bwd_for: exact head_dim, else the global R_bwd_rows
        P = table_for(head_dim, self.consts.get("P_bwd_by_head_dim"))  # cp1_makespan_bwd.p_for
        rho = float(cs["rho_tiles"])
        tiles = tuple((tile_of(q), tile_of(kv)) for q, kv in act)
        ms, n_chain, chain_tiles, N, jm = kernel_dispatch_tiles(tiles, mask, q_heads, R, P, rho)
        jmax_val = float(jm) if N > P else 0.0
        k0 = float(cs["k0_ms"])
        bS = float(cs["b_S_ms_per_token"]) * float(S_tokens)
        terms = {
            "tau_ms_per_tile": float(cs["tau_ms_per_tile"]) * float(ms),
            "tau_jmax_ms_per_tile": float(cs["tau_jmax_ms_per_tile"]) * jmax_val,
        }
        wall = k0 + bS + sum(terms.values()) + st
        split = {
            "tau_ms_per_tile": (float(chain_tiles), float(rho) * n_chain),
            "tau_jmax_ms_per_tile": (jmax_val, 0.0),
        }
        tile_ms = sum(float(cs[c]) * split[c][0] for c in split if c in cs)
        const_sched_ms = sum(float(cs[c]) * split[c][1] for c in split if c in cs)
        q_tok, kv_tok = int(sum(q for q, _ in act)), int(sum(kv for _, kv in act))
        hq, hkv = helper_terms_us(
            self.helper_terms, head_dim, q_heads, kv_heads, q_tok, kv_tok, n_act
        )
        hq_tr, hkv_tr = helper_terms_us(
            self.helper_terms, head_dim, q_heads, kv_heads, q_tok, kv_tok, n_act, traced=True
        )
        return {
            "form_ms": wall,
            "helper_q_ms": hq * 1e-3,
            "helper_kv_ms": hkv * 1e-3,
            "helper_ms": (hq + hkv) * 1e-3,
            "helper_traced_ms": (hq_tr + hkv_tr) * 1e-3,
            "wall_ms": wall + (hq + hkv) * 1e-3,
            "k0_ms": k0,
            "bS_ms": bS,
            "static_ms": st,
            "terms_ms": terms,
            "sched_ms": sum(terms.values()),
            "rho_tiles": rho,
            "tile_ms": tile_ms,
            "const_ms": k0 + bS + const_sched_ms + st,
            "const_sched_ms": const_sched_ms,
            "N_kv": N,
            "P": P,
            "R_bwd": R,
            "J_max_kv": jm,
            "n_chain": n_chain,
            "chain_tiles": chain_tiles,
            "n_active": n_act,
            "n_static": n_st,
            "beyond_first_wave": N > P,
            "set": mask,
            "form": self.form,
            "q_tokens": q_tok,
            "kv_tokens": kv_tok,
        }


# ----------------------------------------------------------------------------------------------------------
# the structural clock law (bwd_common.law_parts) and O_graph_bwd (recorded only)
# ----------------------------------------------------------------------------------------------------------
def law_parts(tile_ms, const_ms, law, f_mhz, f0_mhz=F0_BWD_MHZ) -> dict:
    """The kernel entry of one backward call under the ring block's law. structural: k(f) = tile f0/f + (1 - lambda) C f0/f + lambda C (the scaled part tile + (1 - lambda) C
    under alpha 1, lambda C unscaled); two_term: (tile + C) [s f0/f + 1 - s]; alpha: (tile + C) (f0/f)^alpha. f None -> factor 1 (the prediction at f0).
    """
    kind = law.get("kind", "structural")
    k0 = tile_ms + const_ms
    if kind == "structural":
        lam = float(law["lambda"])
        scaled, unscaled = tile_ms + (1.0 - lam) * const_ms, lam * const_ms
        factor = (f0_mhz / float(f_mhz)) if f_mhz else 1.0
        ov = {
            "law": "alpha",
            "alpha": 1.0,
            "s": None,
            "f0_mhz": f0_mhz,
            "structural_lambda": lam,
            "note": "PB B6 structural law: the scaled part is tile + (1 - lambda) C under alpha 1; lambda C unscaled",
        }
    elif kind == "two_term":
        s = float(law["s"])
        scaled, unscaled = k0, 0.0
        factor = (s * (f0_mhz / float(f_mhz)) + (1.0 - s)) if f_mhz else 1.0
        ov = {"law": "two_term", "alpha": 1.0, "s": s, "f0_mhz": f0_mhz}
    elif kind == "alpha":
        a = float(law.get("alpha", 1.0))
        scaled, unscaled = k0, 0.0
        factor = ((f0_mhz / float(f_mhz)) ** a) if f_mhz else 1.0
        ov = {"law": "alpha", "alpha": a, "s": None, "f0_mhz": f0_mhz}
    else:
        raise ValueError(f"unknown kernel clock law kind {kind!r}")
    return {
        "k_ms": scaled * factor + unscaled,
        "scaled_f0_ms": scaled,
        "unscaled_ms": unscaled,
        "factor": factor,
        "clock_law_override": ov,
        "kind": kind,
    }


def ograph_bwd_us(
    fit, tokens, q_heads, head_dim, n_active=None, q_tokens=None, kv_tokens=None
) -> float:
    """O_graph_bwd in us (cp1_bwd_struct_read.ograph_bwd_us): the served token form o0 + o_q q + o_kv kv + o_n n, or the storage form o0 + o1 MiB_q. RECORDED only
    (u_treatment none: never subtracted)."""
    sv = fit.get("served", fit)
    c = sv["coef"]
    if "o_q_us_per_token" in c:
        q = float(tokens if q_tokens is None else q_tokens)
        kv = float(tokens if kv_tokens is None else kv_tokens)
        n = float(1 if n_active is None else n_active)
        return (
            float(c["o0_us"])
            + float(c["o_q_us_per_token"]) * q
            + float(c["o_kv_us_per_token"]) * kv
            + float(c.get("o_n_us_per_sequence", 0.0)) * n
        )
    mib = float(tokens) * (int(q_heads) * int(head_dim)) * 2.0 / MIB
    v = float(c["o0_us"]) + (float(c["o1_us_per_MiB"])) * mib
    if c.get("o_n_us_per_slot") is not None and n_active is not None:
        v += float(c["o_n_us_per_slot"]) * (THD_MAX - int(n_active))
    return v


# ----------------------------------------------------------------------------------------------------------
# the backward ring geometry (bwd_common: b_kv_bytes, mib_q / mib_kv, step_bytes, step_sizes, bwd_steps, w0_us) and the class-direct glue (compose_bwd_lib)
# ----------------------------------------------------------------------------------------------------------
def b_kv_bytes(S, kv_heads, head_dim) -> int:
    """B_kv = 2 x S x kv_heads x head_dim x 2 bytes (one rank's K and V in bf16; the exchange payload) - wp_g3_common.payload_bytes."""
    return 2 * int(S) * int(kv_heads) * int(head_dim) * 2


def mib_q(S, q_heads, head_dim) -> float:
    return float(S) * (int(q_heads) * int(head_dim)) * 2.0 / MIB


def mib_kv(S, kv_heads, head_dim) -> float:
    return b_kv_bytes(S, kv_heads, head_dim) / MIB


def step_bytes(cp, b_kv) -> list:
    """RINGDESC_BWD_01 1.3: one exchange per step - KV only (B_kv) at step 0, KV + dKV (2 B_kv) at steps 1 .. cp-2, dKV only (B_kv) at step cp-1; cp = 2 -> [B_kv, B_kv]."""
    cp = int(cp)
    if cp < 2:
        return []
    return [int(b_kv)] + [2 * int(b_kv)] * (cp - 2) + [int(b_kv)]


def bytes_class_of_step(i, cp) -> str:
    return "B_kv" if i in (0, cp - 1) else "2B_kv"


def step_sizes(section, mq, mkv) -> tuple:
    """(MiB_q, MiB_kv) touched by the glue of one step on one rank: diagonal full / full; lower full Q / half KV; upper half Q / full KV."""
    if section == "diagonal":
        return mq, mkv
    if section == "lower":
        return mq, mkv / 2.0
    if section == "upper":
        return mq / 2.0, mkv
    raise ValueError(section)


def bwd_steps(lengths, padded, cp, S, q_heads, kv_heads, head_dim) -> list:
    """Per (rank, backward step i): the forward zigzag round of step j = cp - 1 - i relabelled (RINGDESC_BWD_01 1.2) with the exchange bytes and the glue sizes; sorted by (rank, i)."""
    cp = int(cp)
    mq, mkv = mib_q(S, q_heads, head_dim), mib_kv(S, kv_heads, head_dim)
    if cp == 1:
        pieces = [(int(v), int(v)) for v in lengths]
        return [
            {
                "rank": 0,
                "i": 0,
                "j": 0,
                "section": "diagonal",
                "pieces": pieces,
                "q_physical": int(S),
                "kv_physical": int(S),
                "bytes": 0,
                "bytes_class": None,
                "mib_q": mq,
                "mib_kv": mkv,
            }
        ]
    sb = step_bytes(cp, b_kv_bytes(S, kv_heads, head_dim))
    out = []
    for r in round_pieces(lengths, padded, cp):
        i = cp - 1 - int(r["step"])
        sq, skv = step_sizes(r["section"], mq, mkv)
        out.append(
            {
                "rank": int(r["rank"]),
                "i": i,
                "j": int(r["step"]),
                "section": r["section"],
                "pieces": [(int(q), int(kv)) for q, kv in r["pieces"]],
                "q_physical": int(r["q_physical"]),
                "kv_physical": int(r["kv_physical"]),
                "bytes": sb[i],
                "bytes_class": bytes_class_of_step(i, cp),
                "mib_q": sq,
                "mib_kv": skv,
            }
        )
    out.sort(key=lambda d: (d["rank"], d["i"]))
    return out


def w0_us(w0_map, structure, nbytes) -> float:
    w = w0_map[structure]
    return float(w["latency_us"]) + float(w["us_per_byte"]) * float(nbytes)


def glue_classes_step_us(ring, q_heads, kv_heads, head_dim, st) -> dict:
    """PB B5 (ii)-(iii): the step's glue CLASS BY CLASS at f0 (compose_bwd_lib.glue_classes_step_us) - prep (p0 + p_q MiB_q + p_kv MiB_kv + p_n pieces; prep_mode class_direct served,
    'none' / 'excess_over_cp1' = the frozen alternatives), the B4.1 head-token helpers on the step's ACTIVE tokens (mode active_tokens served; physical_rows = the alternative),
    accumulate (a0 + a MiB_kv), memset (constant), copy (c0 + c MiB of the step's exchange bytes).
    """
    gl = ring["glue"]
    pieces = st.get("pieces") or []
    n_act = int(st.get("n_active", len(pieces)) or 0)
    pr = gl["prep"]
    prep = (
        float(pr.get("p0_us", 0.0))
        + float(pr.get("p_q_us_per_MiB_q", 0.0)) * float(st["mib_q"])
        + float(pr.get("p_kv_us_per_MiB_kv", 0.0)) * float(st["mib_kv"])
        + float(pr.get("p_n_us_per_piece", 0.0)) * n_act
    )
    pmode = gl.get("prep_mode", "class_direct")
    if pmode == "none":
        prep = 0.0
    elif pmode == "excess_over_cp1":
        cpc = gl.get("cp1_prep_coef") or {}
        q_tok = int(st.get("q_active", sum(q for q, _ in pieces)))
        kv_tok = int(st.get("kv_active", sum(kv for _, kv in pieces)))
        cp1_prep = (
            float(cpc.get("o0_us", 0.0))
            + float(cpc.get("o_q_us_per_token", 0.0)) * q_tok
            + float(cpc.get("o_kv_us_per_token", 0.0)) * kv_tok
            + float(cpc.get("o_n_us_per_sequence", 0.0)) * n_act
        )
        prep = max(0.0, prep - cp1_prep)
    elif pmode != "class_direct":
        raise ValueError(f"unknown ring prep mode {pmode!r}")
    hmode = (gl.get("helpers") or {}).get("mode", "physical_rows")
    if hmode == "active_tokens":
        q_tok = int(st.get("q_active", sum(q for q, _ in pieces)))
        kv_tok = int(st.get("kv_active", sum(kv for _, kv in pieces)))
    else:
        q_tok, kv_tok = int(st.get("q_physical") or 0), int(st.get("kv_physical") or 0)
    hq, hkv = helper_terms_us(
        (gl.get("helpers") or {}).get("by_head_dim"),
        head_dim,
        q_heads,
        kv_heads,
        q_tok,
        kv_tok,
        n_act,
    )
    ac = gl.get("accumulate") or {}
    acc = float(ac.get("a0_us", 0.0)) + float(ac.get("a_us_per_MiB_kv", 0.0)) * float(st["mib_kv"])
    cp_ = gl.get("copy") or {}
    copy = (
        float(cp_.get("c0_us", 0.0))
        + float(cp_.get("c_us_per_MiB_bytes", 0.0)) * float(st.get("bytes", 0)) / 2**20
    )
    return {
        "prep": prep,
        "helper_q": hq,
        "helper_kv": hkv,
        "accumulate": acc,
        "memset": float(gl.get("memset_us", 0.0)),
        "copy": copy,
    }


def pre_us(ring, S, q_heads, kv_heads, head_dim) -> float:
    p = ring.get("pre") or {}
    if not p.get("served"):
        return 0.0
    return (
        float(p.get("p0_us", 0.0))
        + float(p.get("p_q_us_per_MiB", 0.0)) * mib_q(S, q_heads, head_dim)
        + float(p.get("p_kv_us_per_MiB", 0.0)) * mib_kv(S, kv_heads, head_dim)
    )


# ----------------------------------------------------------------------------------------------------------
# the composer (compose_bwd_lib.BwdComposer under the served block: structural law, class-direct glue with clock-following helpers, scale 1, no remainder)
# ----------------------------------------------------------------------------------------------------------
class BackwardRound1Composer:
    """compose(g, S, cp, lengths, padded, f_mhz=None, critical_rank=None) -> the composer's record (wall_ms, split, steps, ...). f_mhz = the UNIFORM kernel clock of
    the call (the CP group's attention-phase group median, or the CP1 lane's attention clock); None = at f0.
    """

    def __init__(
        self, cp1: CP1BackwardForm, ograph: dict | None, ring: dict, f0: float = F0_BWD_MHZ
    ) -> None:
        self.cp1, self.ograph, self.ring, self.f0 = cp1, ograph, ring, float(f0)
        self.kernel_scale = float(ring.get("kernel_scale", 1.0))
        law = (ring.get("clock") or {}).get("law")
        if not law or law.get("kind") != "structural":
            raise ValueError(
                "the served ring block must carry the structural kernel clock law (ring.clock.law.kind == 'structural', PB B6)"
            )
        self.law = law
        self.lam = float(law["lambda"])
        gl = ring.get("glue") or {}
        if gl.get("form") != "class_direct":
            raise ValueError(
                f"the served ring block carries the glue form {gl.get('form')!r}; the round-1 composer serves class_direct (PB B5)"
            )
        self.helpers_follow = bool((ring.get("clock") or {}).get("helpers_clock_following", True))
        if bool(gl.get("scaled", False)):
            raise ValueError(
                "the served ring block scales the whole glue (the run-6 convention); the round-1 composer serves clock-following helpers only"
            )
        if (ring.get("remainder") or {}).get("served"):
            raise ValueError(
                "the served ring block carries a remainder term; the round-1 composer serves none (H1 holds)"
            )
        self.structure_of_cp = {int(k): v for k, v in (ring.get("structure_of_cp") or {}).items()}
        self._cache = {}

    def kernel_parts(self, g, mask, pieces, S_q) -> dict:
        key = (g, mask, tuple(pieces), int(S_q))
        v = self._cache.get(key)
        if v is None:
            k = self.cp1.kernel(g, mask, pieces, int(S_q))
            q_heads, _kv, head_dim = self.cp1.params(g)
            act = active_pieces(pieces)
            u = (
                ograph_bwd_us(
                    self.ograph,
                    int(S_q),
                    q_heads,
                    head_dim,
                    len(act),
                    int(sum(q for q, _ in act)),
                    int(sum(kv for _, kv in act)),
                )
                * 1e-3
                if (self.ograph and self.ograph.get("served"))
                else 0.0
            ) - float(k.get("helper_traced_ms", 0.0))
            tile = float(k.get("tile_ms", 0.0))
            const = float(k.get("const_ms", k["form_ms"] - tile))
            v = {
                "tile_ms": tile,
                "const_ms": const,
                "kernel_f0_ms": tile + const,
                "U_ms": u,
                "helper_f0_ms": float(k.get("helper_ms", 0.0)),
                "form_ms": float(k["form_ms"]),
                "kernel": k,
            }
            if len(self._cache) > 200000:
                self._cache.clear()
            self._cache[key] = v
        return v

    def entry_at(self, parts, f_mhz) -> dict:
        return law_parts(
            parts["tile_ms"] * self.kernel_scale,
            parts["const_ms"] * self.kernel_scale,
            self.law,
            f_mhz,
            self.f0,
        )

    @lru_cache(maxsize=8192)
    def _glue_at(self, g, mq, mkv, n, q_active, kv_active, q_physical, kv_physical, nbytes):
        q_heads, kv_heads, head_dim = self.cp1.params(g)
        return glue_classes_step_us(
            self.ring,
            q_heads,
            kv_heads,
            head_dim,
            dict(
                mib_q=mq,
                mib_kv=mkv,
                n_active=n,
                q_active=q_active,
                kv_active=kv_active,
                q_physical=q_physical,
                kv_physical=kv_physical,
                bytes=nbytes,
            ),
        )

    def glue_at(self, g, step):
        pieces = step["pieces"]
        return self._glue_at(
            g,
            step["mib_q"],
            step["mib_kv"],
            len(pieces),
            sum(q for q, _ in pieces),
            sum(kv for _, kv in pieces),
            step["q_physical"],
            step["kv_physical"],
            step["bytes"],
        )

    @lru_cache(maxsize=8192)
    def clock_entry(self, tile_ms, const_ms, frequency):
        return law_parts(tile_ms, const_ms, self.law, frequency, self.f0)

    def structure_for(self, cp, topology=None) -> tuple:
        """(structure, extrapolated): the ring block's structure_of_cp for the recorded CPs, the hardware's topology rule otherwise (cp 3 -> the intra-node class, cp >= 5 outside the map -> nvl72)."""
        cp = int(cp)
        structure = self.structure_of_cp.get(cp)
        if structure is None:
            if topology:
                structure, _role = topology_class_for_cp(topology, cp)
            else:
                structure = (
                    "cp2_sequential_sendrecv"
                    if cp == 2
                    else ("cp4_concurrent_intra_node" if cp <= 4 else "nvl72_domain")
                )
        if structure not in self.ring["W0"]:
            raise ValueError(
                f"communication structure {structure!r} has no W0 coefficients in the served backward ring block"
            )
        return structure, cp not in CALIBRATED_CPS

    def compose(
        self, g, S, cp, lengths, padded=None, f_mhz=None, critical_rank=None, topology=None
    ) -> dict:
        L = [int(v) for v in lengths]
        pad = (
            layout_extend_last(L, cp, S, ALIGNMENT_DEFAULT)
            if padded is None
            else [int(v) for v in padded]
        )
        q_heads, kv_heads, head_dim = self.cp1.params(g)
        f = float(f_mhz) if f_mhz else None
        if cp == 1:
            pieces = [(v, v) for v in L]
            parts = self.kernel_parts(g, "causal", pieces, int(S))
            k = parts["kernel"]
            e = self.entry_at(parts, f)
            hk = float(k.get("helper_ms", 0.0))
            hf = (
                hk * ((self.f0 / f) if (f and self.helpers_follow) else 1.0)
                if self.helpers_follow
                else 0.0
            )
            wall = hf + e["k_ms"] if self.helpers_follow else hk + e["k_ms"]
            wall_f0 = hk + parts["kernel_f0_ms"] * self.kernel_scale
            scaled_f0 = e["scaled_f0_ms"] + (hk if self.helpers_follow else 0.0)
            ring_ms = e["unscaled_ms"] + (0.0 if self.helpers_follow else hk)
            return {
                "geometry": g,
                "S": int(S),
                "cp": 1,
                "padded_lengths": pad,
                "wall_ms": wall,
                "wall_f0_ms": wall_f0,
                "K_f0_ms": parts["kernel_f0_ms"] * self.kernel_scale,
                "K_ms": e["k_ms"],
                "H_f0_ms": hk,
                "H_ms": hf,
                "U_read_ms": parts["U_ms"],
                "u_treatment": "none",
                "G_ms": 0.0,
                "X_ms": 0.0,
                "pre_ms": 0.0,
                "scale": e["factor"] * self.kernel_scale,
                "critical_rank": 0,
                "steps": [],
                "f_used_mhz": f,
                "structure": None,
                "structure_extrapolated": False,
                "law": self.law,
                "split": {
                    "kernel_f0_ms": scaled_f0,
                    "ring_ms": ring_ms,
                    "rem0_f0_ms": 0.0,
                    "accounting": "cp1",
                    "kind": "cp1_bwd",
                    "clock_quantity": CLOCK_QUANTITY,
                    "critical_rank": 0,
                    "scaled_part": "kernel tile + (1 - lambda) C + helpers",
                    "clock_law_override": e["clock_law_override"],
                    "tile_f0_ms": parts["tile_ms"],
                    "const_f0_ms": parts["const_ms"],
                },
                "cp1_detail": {kk: vv for kk, vv in k.items() if kk not in ("terms_ms",)},
            }
        steps = bwd_steps(L, pad, cp, S, q_heads, kv_heads, head_dim)
        st_of, extrapolated = self.structure_for(cp, topology)
        per_rank = {}
        for st in steps:
            mask = "causal" if st["section"] == "diagonal" else "rectangle"
            parts = self.kernel_parts(g, mask, st["pieces"], st["q_physical"])
            e = self.entry_at(parts, f)
            st_rec = {**st, "n_active": len(st["pieces"])}
            cls = glue_classes_step_us(self.ring, q_heads, kv_heads, head_dim, st_rec)
            hel = cls["helper_q"] + cls["helper_kv"]
            rest = cls["prep"] + cls["accumulate"] + cls["memset"] + cls["copy"]
            rf = (self.f0 / f) if f else 1.0
            if self.helpers_follow:
                g_f0, g_at, g_scaled_f0, g_unscaled = hel + rest, hel * rf + rest, hel, rest
            else:
                g_f0, g_at, g_scaled_f0, g_unscaled = hel + rest, hel + rest, 0.0, hel + rest
            w0 = w0_us(self.ring["W0"], st_of, st["bytes"])
            per_rank.setdefault(st["rank"], []).append(
                {
                    **st_rec,
                    "k_f0_ms": parts["kernel_f0_ms"] * self.kernel_scale,
                    "k_ms": e["k_ms"],
                    "k_scaled_f0_ms": e["scaled_f0_ms"],
                    "k_unscaled_ms": e["unscaled_ms"],
                    "tile_f0_ms": parts["tile_ms"] * self.kernel_scale,
                    "const_f0_ms": parts["const_ms"] * self.kernel_scale,
                    "g_f0_us": g_f0,
                    "g_us": g_at,
                    "g_scaled_f0_us": g_scaled_f0,
                    "g_unscaled_us": g_unscaled,
                    "glue_classes_us": cls,
                    "W0_us": w0,
                    "factor_rank": e["factor"],
                    "f_rank_mhz": f,
                }
            )
        sums = {r: sum(s_["k_ms"] + s_["g_us"] * 1e-3 for s_ in ss) for r, ss in per_rank.items()}
        sums_f0 = {
            r: sum(s_["k_f0_ms"] + s_["g_f0_us"] * 1e-3 for s_ in ss) for r, ss in per_rank.items()
        }
        crit_argmax = max(sums, key=sums.get)
        crit = int(critical_rank) if critical_rank is not None else crit_argmax
        ss = per_rank[crit]
        K_f0 = sum(s_["k_f0_ms"] for s_ in ss)
        K = sum(s_["k_ms"] for s_ in ss)
        K_scaled_f0 = sum(s_["k_scaled_f0_ms"] for s_ in ss)
        K_unscaled = sum(s_["k_unscaled_ms"] for s_ in ss)
        G_f0_us = sum(s_["g_f0_us"] for s_ in ss)
        G_us = sum(s_["g_us"] for s_ in ss)
        G_scaled_f0_us = sum(s_["g_scaled_f0_us"] for s_ in ss)
        G_unscaled_us = sum(s_["g_unscaled_us"] for s_ in ss)
        W0_sum = sum(s_["W0_us"] for s_ in ss)
        xr = self.ring["X"]
        X_us = float(xr["x0_us"]) + float(xr["kappa"]) * W0_sum
        rem_us = 0.0
        pre = pre_us(self.ring, S, q_heads, kv_heads, head_dim)
        scaled_f0 = K_scaled_f0 + G_scaled_f0_us * 1e-3
        ring_ms = (K_unscaled * 1e3 + G_unscaled_us + pre + X_us) * 1e-3
        wall = K + G_us * 1e-3 + (pre + X_us) * 1e-3 + rem_us * 1e-3
        e0 = self.entry_at({"tile_ms": 0.0, "const_ms": 0.0}, f)
        return {
            "geometry": g,
            "S": int(S),
            "cp": int(cp),
            "padded_lengths": pad,
            "wall_ms": wall,
            "K_f0_ms": K_f0,
            "K_ms": K,
            "G_ms": G_us * 1e-3,
            "G_f0_ms": G_f0_us * 1e-3,
            "X_ms": X_us * 1e-3,
            "pre_ms": pre * 1e-3,
            "remainder_ms": 0.0,
            "scale": (ss[0]["factor_rank"] * self.kernel_scale),
            "W0_sum_us": W0_sum,
            "structure": st_of,
            "structure_extrapolated": extrapolated,
            "critical_rank": crit,
            "critical_rank_argmax": crit_argmax,
            "per_rank_sum_ms": {str(r): v for r, v in sums.items()},
            "per_rank_sum_f0_ms": {str(r): v for r, v in sums_f0.items()},
            "f_used_mhz": f,
            "law": self.law,
            "glue_scaled": self.helpers_follow,
            "steps": [{k: v for k, v in s_.items() if k != "pieces"} for s_ in ss],
            "split": {
                "kernel_f0_ms": scaled_f0,
                "ring_ms": ring_ms,
                "rem0_f0_ms": 0.0,
                "accounting": "crit",
                "kind": "s4_bwd",
                "clock_quantity": CLOCK_QUANTITY,
                "critical_rank": crit,
                "scaled_part": "kernel tile + (1 - lambda) C + helper classes",
                "clock_law_override": e0["clock_law_override"],
                "glue_f0_ms": G_f0_us * 1e-3,
                "kernel_only_f0_ms": K_f0,
                "kernel_unscaled_ms": K_unscaled,
                "tile_f0_ms": sum(s_["tile_f0_ms"] for s_ in ss),
                "const_f0_ms": sum(s_["const_f0_ms"] for s_ in ss),
            },
        }


# ----------------------------------------------------------------------------------------------------------
# the model: the gb200 / backward entry of the data file (status served_round1) + the vendored files it pins
# ----------------------------------------------------------------------------------------------------------
class BackwardRound1Model:
    """The GB200 backward round-1 model of the data file's ``served_round1`` entry: the same surface as ``CudaGraphAttentionModel`` for the estimator
    (``geometry_profile``, ``predict_with_details``, ``production_default_clock_mhz``, ``head`` / ``head_status`` / ``status_marker`` / ``data_sha256``).
    """

    def __init__(
        self, payload, hardware, head=None, data_sha256=None, direction="backward", data_dir=None
    ) -> None:
        payload = normalize_payload(payload)
        hw, entry = _entry(payload, hardware, direction)
        if entry.get("status") == "pending":
            raise CudaGraphHeadPendingError(pending_message(hardware, direction, entry))
        if (
            entry.get("status") not in (ENTRY_STATUS, ENTRY_STATUS_ROUND2)
            or direction != "backward"
        ):
            raise ValueError(
                f"the {hardware} / {direction} entry has status {entry.get('status')!r}; {PACKAGE} serves the {ENTRY_STATUS!r} / {ENTRY_STATUS_ROUND2!r} backward entries only"
            )
        heads = tuple(entry["heads"])
        self.served_head = entry["served_head"]
        self.head = resolve_head(head, heads, self.served_head)
        self.payload, self.hardware, self.direction, self.entry, self.hw = (
            payload,
            hardware,
            direction,
            entry,
            hw,
        )
        self.campaign = dict(entry.get("campaign") or {})
        self.hardware_spec = dict(ROUND1_HARDWARE.get(hardware) or {})
        self.campaign_name = str(
            self.campaign.get("name") or self.hardware_spec.get("campaign") or "WP-G3B"
        )
        self.data_sha256 = data_sha256
        self.status = entry["status"]
        self.block = entry["heads"][self.head]
        info = (entry.get("head_info") or {}).get(self.head) or {}
        self.head_status = info.get("status") or self.block.get("status") or HEAD_STATUS
        self.is_served_head = self.head == self.served_head
        self.status_marker = str(self.block.get("status_marker") or STATUS_MARKER)
        self.hw_clock = dict(hw.get("clock") or {})
        self.topology = dict(hw.get("topology") or {})
        self.layout = dict(hw.get("layout") or {})
        self.composition_kind = str(self.block.get("composition_kind") or COMPOSITION_S4_BWD)
        if self.composition_kind not in (COMPOSITION_S4_BWD, COMPOSITION_S4_BWD_R2):
            raise ValueError(
                f"head {self.head!r} declares the composition {self.composition_kind!r}; {PACKAGE} composes {COMPOSITION_S4_BWD!r} and {COMPOSITION_S4_BWD_R2!r}"
            )
        self.round = int((self.block.get("composer") or {}).get("round") or 1)
        if self.round != 1 and type(self) is BackwardRound1Model:
            raise ValueError(
                f"head {self.head!r} is a round-{self.round} head; use BackwardRound2Model (load_cuda_graph_model dispatches)"
            )
        pins = self.block.get("data_files") or {}
        missing = [name for name in DATA_FILES if name not in pins]
        if missing:
            raise ValueError(
                f"the {hardware} / backward head {self.head!r} does not pin the vendored files {missing}"
            )
        # the vendored directory: the argument, else the head's composer.data_dir (relative to this package), else the hardware's default (DATA_DIRS)
        if data_dir is None:
            rel = (self.block.get("composer") or {}).get("data_dir") or self.hardware_spec.get(
                "data_dir"
            )
            if rel is None:
                raise ValueError(
                    f"no vendored round-1 data directory is known for hardware {hardware!r} (ROUND1_HARDWARE {sorted(ROUND1_HARDWARE)}; the head carries no composer.data_dir)"
                )
            data_dir = PACKAGE_DIR / rel
        self.data_dir = Path(data_dir)
        self.files, self.file_sha256 = {}, {}
        for name in DATA_FILES + tuple(n for n in OPTIONAL_DATA_FILES if n in pins):
            self.files[name], self.file_sha256[name] = load_vendored(
                name, pins[name]["sha256"], self.data_dir
            )
        self.manifest = load_manifest(self.data_dir / "MANIFEST.json")
        for name in self.files:
            if name not in self.manifest["vendored"]:
                raise ValueError(
                    f"vendored file {name}: pinned by the head but absent from {self.data_dir / 'MANIFEST.json'}"
                )
            if self.manifest["vendored"][name]["sha256"] != self.file_sha256[name]:
                raise ValueError(
                    f"vendored file {name}: MANIFEST.json pins {self.manifest['vendored'][name]['sha256']}, the file is {self.file_sha256[name]}"
                )
        self.cp1 = CP1BackwardForm(self.files["cp1_model_f5b1_jmax_bwd_f0.json"])
        self.ring = self.files["ring_bwd_served.json"]["served"]
        self.ograph = self.files["ograph_bwd_served.json"]
        self.served_config = self.files["served_config_f_bwd.json"]
        self.verdict = self.files["sealed_verdict_f_bwd.json"]
        self.verdict_h = self.files.get("sealed_verdict_h_bwd.json")
        self.f0_mhz = float(self.ring["f0_mhz"])
        clock = self.block.get("clock_input") or {}
        hw_f0 = (self.hw_clock.get("f0_mhz") or {}).get("backward")
        expected_f0 = self.hardware_spec.get("f0_mhz", self.f0_mhz)
        if (
            abs(self.f0_mhz - float(expected_f0)) > 1e-9
            or abs(self.cp1.f0_mhz - self.f0_mhz) > 1e-9
            or (clock and abs(float(clock["f0_mhz"]) - self.f0_mhz) > 1e-9)
            or (hw_f0 is not None and abs(float(hw_f0) - self.f0_mhz) > 1e-9)
        ):
            raise ValueError(
                f"f0_bwd disagrees: ring {self.f0_mhz}, CP1 model {self.cp1.f0_mhz}, head {clock.get('f0_mhz')}, hardware clock block {hw_f0}; expected {expected_f0} on {hardware}"
            )
        # B6 (iv): attention-phase readings at or above the hardware's telemetry ceiling are flagged (gb200 1800 MHz; gb300 1845 MHz, BG1 R6c) - the ring block carries it
        self.clock_ceiling_mhz = float(
            (self.ring.get("clock") or {}).get("ceiling_mhz")
            or self.hardware_spec.get("clock_ceiling_mhz")
            or 1800.0
        )
        self.composer = BackwardRound1Composer(self.cp1, self.ograph, self.ring, self.f0_mhz)
        self.lam = self.composer.lam
        if clock and abs(float(clock.get("lambda", self.lam)) - self.lam) > 1e-12:
            raise ValueError(
                "the head's clock_input.lambda disagrees with the served ring block's structural lambda"
            )
        self.two_term_s_alternative = (
            float(self.ring.get("s_bwd")) if self.ring.get("s_bwd") is not None else None
        )
        prod = (
            self.hw_clock.get("production_default_clock_mhz")
            or hw.get("production_default_clock_mhz")
            or {}
        )
        self.production_default_clock_mhz = {
            k: float(v) for k, v in prod.items() if isinstance(v, (int, float))
        }
        self.alignment_default = int(
            (self.block.get("layout") or {}).get("alignment_default")
            or self.layout.get("alignment_default")
            or ALIGNMENT_DEFAULT
        )
        self.thd_max = int(
            (self.block.get("layout") or {}).get("thd_max") or self.layout.get("thd_max") or THD_MAX
        )
        self.calibrated_storages = tuple(
            int(v)
            for v in (
                (self.block.get("layout") or {}).get("calibrated_storages") or CALIBRATED_STORAGES
            )
        )
        self.calibrated_cps = tuple(
            int(v) for v in (self.block.get("calibrated_cps") or CALIBRATED_CPS)
        )
        self.geometries = {
            g: {"query_heads": q, "kv_heads": kv, "head_dim": d}
            for g, (q, kv, d) in ((g, self.cp1.params(g)) for g in self.cp1.geometry_params)
        }
        self._profile_by_shape = {
            (e["query_heads"], e["kv_heads"], e["head_dim"]): g for g, e in self.geometries.items()
        }
        self.cp1_form = self.cp1.form
        self.ring_kernel_scale = self.composer.kernel_scale
        self.exchange_schedule = (
            (self.block.get("exchange_schedule") or entry.get("exchange_schedule") or {}).get(
                "kind"
            )
        ) or "kv_dkv_ring_backward"
        self.clock = {
            "f0_mhz": self.f0_mhz,
            "law": CLOCK_LAW_SERVED,
            "lambda": self.lam,
            "quantity": CLOCK_QUANTITY,
            "phase": "attention",
            "two_term_s_alternative": self.two_term_s_alternative,
        }

    @classmethod
    def from_path(
        cls, hardware="gb200", head=None, path=None, direction="backward", expected_sha256=None
    ) -> BackwardRound1Model:
        from ..cuda_graph_attention_predictor import DATA_PATH, _load_payload

        payload, digest = _load_payload(DATA_PATH if path is None else path, expected_sha256)
        return cls(payload, hardware, head, digest, direction)

    # -- geometry ---------------------------------------------------------------------------------------
    def geometry_profile(self, query_heads, kv_heads, head_dim) -> str:
        key = (int(query_heads), int(kv_heads), int(head_dim))
        name = self._profile_by_shape.get(key)
        if name is None:
            known = ", ".join(
                f"{n} {q}/{kv}/{d}"
                for (q, kv, d), n in sorted(self._profile_by_shape.items(), key=lambda kv_: kv_[1])
            )
            raise ValueError(
                f"no {self.campaign_name} backward round-1 coefficient set for geometry {key[0]}/{key[1]}/{key[2]} on {self.hardware}; calibrated: {known}"
            )
        return name

    def geometry_params(self, profile) -> tuple:
        return self.cp1.params(profile)

    def cp1_set(self, profile, mask) -> dict:
        return self.cp1.set_for(profile, mask)

    def sealed_cell(self, profile, S, cp) -> dict:
        """The once-read f_bwd verdict of the (geometry, S, cp) cell: {cell, gate PASS | FAIL | inconclusive | unscored, served {wape, p90, bias}, status}."""
        cell = f"{profile}|{int(S)}|cp{int(cp)}"
        rec = (self.verdict.get("cells") or {}).get(cell)
        if rec is None:
            return {
                "cell": cell,
                "gate": "unscored",
                "served": None,
                "status": "outside the 45 sealed f_bwd cells (geometries x S 4096 / 8192 / 16384 x cp 1 / 2 / 4 / 8 / 16)",
            }
        return {
            "cell": cell,
            "gate": rec["gate"],
            "served": rec.get("served"),
            "status": rec.get("status"),
            "anchor_status": rec.get("anchor_status"),
        }

    # -- clock input ------------------------------------------------------------------------------------
    def clock_input(
        self,
        cp_size,
        clock_mhz_by_rank=None,
        f_crit_mhz=None,
        law=DEFAULT_CLOCK_LAW,
        use_production_default=False,
    ) -> dict:
        """The uniform kernel clock of one prediction (quantity attention_phase_group_median): the MEDIAN of the CP group's per-rank clocks (exactly cp values), or f_crit_mhz
        as given, or the hardware's production default backward clock (1600 MHz = f0: factor 1), or None = at f0. The served law is the B6 structural law whatever
        ``law`` names (clock_law selects the forward entry's law; it is recorded here as law_requested).
        """
        cp = int(cp_size)
        if law not in CLOCK_LAWS:
            raise ValueError(f"clock_law must be one of {CLOCK_LAWS}, got {law!r}")
        if clock_mhz_by_rank is not None and f_crit_mhz is not None:
            raise ValueError(
                "give either clock_mhz_by_rank (the CP group's per-rank attention-phase clocks) or f_crit_mhz (one uniform clock), not both"
            )
        if use_production_default and (clock_mhz_by_rank is not None or f_crit_mhz is not None):
            raise ValueError(
                "use_production_default_clock replaces a missing card: do not combine it with clock_mhz_by_rank / f_crit_mhz"
            )
        base = {
            "law": CLOCK_LAW_SERVED,
            "law_requested": law,
            "structural_lambda": self.lam,
            "f0_mhz": self.f0_mhz,
            "quantity": CLOCK_QUANTITY,
            "phase": "attention",
            "two_term_s_alternative": self.two_term_s_alternative,
            "clock_law_override": {
                "law": "alpha",
                "alpha": 1.0,
                "f0_mhz": self.f0_mhz,
                "structural_lambda": self.lam,
            },
            "rule": (
                "wall(f) = kernel_f0_ms x (f0 / f) + ring_ms: kernel_f0_ms = tile + (1 - lambda) C + the helper classes, ring_ms = lambda C + prep / accumulate / memset / copy + pre + X "
                "(SEALED_SCORING_BWD.md; the B6 structural law); f = the MEDIAN over the CP group's ranks of the attention-phase clock, applied uniformly (the critical rank is the "
                "argmax of the per-rank sums at f0); clock_law names the forward entry's law - the backward head serves the structural law under either value"
            ),
        }
        prod_text = (
            self.hardware_spec.get("production_clock_text")
            or f"the {self.hardware} production backward DURING-window clock (A24 (ii))"
        )
        if clock_mhz_by_rank is None and f_crit_mhz is None and not use_production_default:
            return dict(
                base,
                applied="f0_default",
                f_mhz=None,
                f_group_median_mhz=None,
                f_by_rank_mhz=None,
                f_min_mhz=None,
                f_min_rank=None,
                scale=1.0,
                clock_ceiling=False,
                note=(
                    f"no clock supplied: the prediction AT f0_bwd = {self.f0_mhz:.0f} MHz = {prod_text} - the no-card default "
                    "and the production default coincide for this direction; supply the group's per-rank attention-phase clocks or f_crit_mhz for a card"
                ),
            )
        by_rank, f_min, rank_min, applied = None, None, None, "scaled"
        if use_production_default:
            f = self.production_default_clock_mhz.get("backward")
            if f is None:
                raise ValueError(
                    f"hardware {self.hardware!r} has no production_default_clock_mhz for the backward direction in the data file"
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
            f = float(statistics.median(values))
            rank_min = min(range(cp), key=lambda i: values[i])
            f_min = values[rank_min]
            by_rank = values
        else:
            f = float(f_crit_mhz)
            if not (math.isfinite(f) and f > 0.0):
                raise ValueError("f_crit_mhz must be a finite positive MHz value")
        ceiling = self.clock_ceiling_mhz
        return dict(
            base,
            applied=applied,
            f_mhz=f,
            f_group_median_mhz=(f if by_rank is not None else None),
            f_by_rank_mhz=by_rank,
            f_min_mhz=f_min,
            f_min_rank=rank_min,
            scale=self.f0_mhz / f,
            clock_ceiling=bool(
                f >= ceiling or any(v >= ceiling for v in (by_rank or []))
            ),  # B6 (iv): the applied clock or ANY rank's reading at the telemetry ceiling
            clock_ceiling_mhz=ceiling,
            note=(
                f"the attention-phase group median applied uniformly (B6 (iii)); readings >= {ceiling:.0f} MHz are a telemetry ceiling (+-4 %, flagged not corrected)"
                if by_rank is not None
                else (
                    f"the hardware's production default backward clock (A24: the clock DURING the attention window on the slowest rank = f0_bwd {self.f0_mhz:.0f} MHz; {prod_text}; factor 1)"
                    if applied == "production_default"
                    else "one uniform attention-phase clock for the CP group, as given"
                )
            ),
        )

    # -- prediction -------------------------------------------------------------------------------------
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
    ) -> dict:
        """Backward core-attention wall (ms) of one pack on one CP group: the B7.1 served composition with the decomposition and the provisional reasons."""
        values = [int(v) for v in lengths]
        cp, S = int(cp_size), int(storage_tokens)
        if cp <= 0 or S <= 0:
            raise ValueError("cp_size and storage_tokens must be positive")
        if two_term_s is not None or g_ring is not None or kappa_w0 is not None:
            raise ValueError(
                "two_term_s / g_ring / kappa_w0 are comparison overrides of the forward heads; the backward round-1 head serves the structural law, kernel scale 1 and X = x0 + kappa sum W0 as frozen"
            )
        profile = self.geometry_profile(query_heads, kv_heads, head_dim)
        alignment = (
            self.alignment_default if sequence_alignment is None else int(sequence_alignment)
        )
        if len(values) > self.thd_max:
            raise ValueError(
                f"the pack holds {len(values)} sequences but the CUDA-graph THD batch has thd_max = {self.thd_max} cu_seqlens slots"
            )
        padded = layout_extend_last(values, cp, S, alignment, dataset_alignment)
        clock = self.clock_input(
            cp, clock_mhz_by_rank, f_crit_mhz, clock_law, use_production_default_clock
        )
        reasons = [PROVISIONAL_REASON, REASON_BACKWARD_ROUND1]
        if not self.is_served_head:
            reasons.append(
                REASON_HEAD_SUPERSEDED_BWD
            )  # b7_1 selected on the served_round2 entry: comparison only
        if clock["applied"] == "f0_default":
            reasons.append(REASON_CLOCK)
        elif clock["applied"] == "production_default":
            reasons.append(REASON_PRODUCTION_CLOCK)
        if S not in self.calibrated_storages:
            reasons.append(REASON_STORAGE)
        if alignment != self.alignment_default:
            reasons.append(REASON_ALIGNMENT)
        if dataset_alignment != 1:
            reasons.append("cuda_graph_dataset_padding_transfer_provisional")
        causal = self.cp1_set(profile, "causal")
        if cp == 1 and str(S) in (causal.get("cp1_form_acceptance_failed") or []):
            reasons.append(REASON_CP1_FORM)
        cell = self.sealed_cell(profile, S, cp)
        if cell["gate"] == "FAIL":
            reasons.append(REASON_SEALED_CELL_FAILED)
        elif cell["gate"] == "inconclusive":
            reasons.append(REASON_SEALED_CELL_INCONCLUSIVE)
        elif cell["gate"] == "unscored":
            reasons.append(REASON_SEALED_CELL_UNSCORED)
        f = clock["f_mhz"]
        at_f0 = self.composer.compose(profile, S, cp, values, padded, None, None, self.topology)
        record = (
            at_f0
            if f is None
            else self.composer.compose(
                profile, S, cp, values, padded, f, at_f0["critical_rank"], self.topology
            )
        )
        if cp > 1 and record["structure_extrapolated"]:
            reasons.append(REASON_CP_STRUCTURE)
        out = {
            "time_ms": record["wall_ms"],
            "model_kind": MODEL_KIND,
            "launch_mode": CUDA_GRAPH_LAUNCH_MODE,
            "hardware": self.hardware,
            "direction": self.direction,
            "head": self.head,
            "head_status": self.head_status,
            "served_head": self.served_head,
            "cp1_form": self.cp1_form,
            "composition": self.composition_kind,
            "geometry_profile": profile,
            "geometry": [int(query_heads), int(kv_heads), int(head_dim)],
            "cp_size": cp,
            "storage_tokens": S,
            "sequence_alignment": alignment,
            "tail_policy": "extend_last",
            "valid_lengths": values,
            "padded_lengths": padded,
            "physical_tokens": S * cp,
            "sequence_count": len(values),
            "static_slots": self.thd_max - len(values),
            "status": self.status,
            "status_marker": self.status_marker,
            "model_sha256": self.data_sha256,
            "vendored_sha256": dict(self.file_sha256),
            "clock": clock,
            "provisional": True,
            "provisional_reasons": reasons,
            "sealed_cell": cell,
            "wall_f0_ms": at_f0["wall_ms"],
            "split_f0": at_f0["split"],
            "closed_form_ms": at_f0["split"]["kernel_f0_ms"] * clock["scale"]
            + at_f0["split"]["ring_ms"],
        }
        if self.verdict_h is not None:
            out["sealed_cell_h"] = self.sealed_cell_h(profile, S, cp)
        detail = {
            k: v for k, v in record.items() if k not in ("geometry", "S", "cp", "padded_lengths")
        }
        detail["critical_rank_f0"] = at_f0["critical_rank"]
        if cp == 1:
            out["cp1"] = detail
        else:
            out["ring"] = detail
        return out

    def sealed_cell_h(self, profile, S, cp) -> dict:
        """The once-read h item's verdict of the (geometry, S, cp) cell when the campaign scored it (gb300: h_bwd_gb300, the sg / eq length grids at S 4096 / 16384,
        cp 2-16); recorded in the prediction, never a reason flag (the f item's cell carries the flag).
        """
        cell = f"{profile}|{int(S)}|cp{int(cp)}"
        rec = ((self.verdict_h or {}).get("cells") or {}).get(cell)
        if rec is None:
            return {
                "cell": cell,
                "gate": "unscored",
                "served": None,
                "status": "outside the sealed h cells of this hardware's backward program",
            }
        return {
            "cell": cell,
            "gate": rec["gate"],
            "served": rec.get("served"),
            "status": rec.get("status"),
            "anchor_status": rec.get("anchor_status"),
        }


# ----------------------------------------------------------------------------------------------------------
# ROUND 2: compose_bwd_r2_lib.x_form_eval / BwdComposerR2 ported verbatim (arithmetic order kept) onto the round-1 composer
# ----------------------------------------------------------------------------------------------------------
def x_form_eval(form, cls, steps, w0_sum_us, ring_x, head_dim=None) -> tuple:
    """X in us for one pack under an X form dict (compose_bwd_r2_lib.x_form_eval): {form: x0_kappa} | {form: x0_class, x0_us} | {form: per_step_linear | per_step_pooled, g0_us,
    g1_us_per_MiB[, x_first_us], variant} | {form: by_class, by_class: {cls: form}} | {form: per_step_g0_by_head_dim, g0_by_head_dim: {"128": g0, "256": g0}[, g0_pooled_us]}
    (B10.2 (a), SERVED in round 2: X = sum over the cp exchange steps of g0(head_dim); kappa 0)."""
    kind = form.get("form", "x0_kappa")
    if kind == "by_class":
        sub = (form.get("by_class") or {}).get(cls)
        if sub is None:
            raise KeyError(f"X form by_class: no entry for structure class {cls!r}")
        return x_form_eval(sub, cls, steps, w0_sum_us, ring_x, head_dim)
    if kind == "per_step_g0_by_head_dim":
        tab = {
            int(k): float(v) for k, v in (form.get("g0_by_head_dim") or {}).items() if v is not None
        }
        g0 = tab.get(int(head_dim)) if head_dim is not None else None
        if g0 is None:
            if form.get("g0_pooled_us") is None:
                raise KeyError(
                    f"X form per_step_g0_by_head_dim: no g0 for head_dim {head_dim} and no pooled fallback"
                )
            g0 = float(form["g0_pooled_us"])
        return (
            g0 * len(steps),
            f"cp x g0(d{head_dim}) = {len(steps)} x {g0:.1f} us [per_step_g0_by_head_dim]",
        )
    if kind == "x0_kappa":
        return (
            float(ring_x["x0_us"]) + float(ring_x["kappa"]) * w0_sum_us,
            "x0 + kappa sum W0 (round 1)",
        )
    if kind == "x0_class":
        return float(form["x0_us"]), f"x0_class {cls}"
    if kind in ("per_step_linear", "per_step_pooled"):
        g0 = float(form.get("g0_us", 0.0))
        g1 = float(form.get("g1_us_per_MiB", 0.0) or 0.0)
        mibs = [float(s_["bytes"]) / MIB for s_ in sorted(steps, key=lambda s_: s_["i"])]
        if form.get("x_first_us") is not None and (
            form.get("variant", "").endswith("first") or kind == "per_step_pooled"
        ):
            return (
                float(form["x_first_us"]) + sum(g0 + g1 * m for m in mibs[1:]),
                f"x_first + sum_(i>=1) (g0 + g1 MiB_i) [{kind}]",
            )
        return sum(g0 + g1 * m for m in mibs), f"sum_i (g0 + g1 MiB_i) [{kind}]"
    raise ValueError(kind)


class BackwardRound2Composer(BackwardRound1Composer):
    """compose_bwd_r2_lib.BwdComposerR2 on the round-1 composer: step entry k'_i = tile_i f0/f + share(geometry) x C_i^shared x g(f) with C_i^shared = k0 + b_S q_phys + tau rho n_chain
    (static(n) NOT per step); once per pack on the critical rank: static(n_pack) x g(f) + T_tail (NOT served: the served tail block is empty) + pre + X; X per the served X form
    (per_step_g0_by_head_dim: cp x g0(head_dim), kappa 0); helpers / glue / W0 / pre as round 1; cp = 1 = the round-1 CP1 path (+ T_tail = 0). The kernel clock is ONE uniform clock f
    (the CP group's attention-phase group median, D2 (i)); readings >= flag_mhz are reported as flag_1750 (never corrected). With shares = 1, static per step and X = x0 + kappa sum W0
    (attribution round1) the composer reproduces the round-1 composer exactly."""

    def __init__(
        self,
        cp1,
        ograph,
        ring,
        f0=F0_BWD_MHZ,
        *,
        shares=None,
        tail=None,
        x_form=None,
        static_per_call=True,
        flag_mhz=FLAG_MHZ,
        attribution="round2",
    ) -> None:
        super().__init__(cp1, ograph, ring, f0)
        self.attribution = attribution
        if attribution == "round1":
            shares, tail, x_form, static_per_call = {}, {}, {"form": "x0_kappa"}, False
        self.shares = {str(g): float(v) for g, v in (shares or {}).items()}
        self.tail = {} if tail is None else tail
        if self.tail and (
            not isinstance(self.tail, dict)
            or self.tail.get("ring_applied", True)
            or self.tail.get("cp1_applied", True)
        ):
            raise ValueError(
                "the vendored round-2 composer serves NO tail term (the B12 served tail block is empty); a tail block is a frozen alternative, not served"
            )
        self.x_form = x_form or {"form": "x0_kappa"}
        self.static_per_call = bool(static_per_call)
        self.flag_mhz = float(flag_mhz)

    def share(self, g) -> float:
        return float(self.shares.get(g, 1.0))

    def tail_f0_ms(self, g, S) -> float:
        return 0.0  # no tail term served (B10.2 (b): the tail is not a ring term; the round-2 tail block is {})

    def static_pack_ms(self, g, n_seq) -> float:
        return static_ms(
            int(n_seq), self.cp1.static_us, self.cp1.thd_max
        )  # static_spec forward_slots (asserted by CP1BackwardForm): 0.155 us x (64 - n)

    @lru_cache(maxsize=16384)
    def _step_parts(self, g, section, pieces, q_physical, kv_physical, mq, mkv, nbytes):
        mask = "causal" if section == "diagonal" else "rectangle"
        parts = self.kernel_parts(g, mask, pieces, q_physical)
        static = parts["kernel"]["static_ms"]
        const = self.share(g) * ((parts["const_ms"] - static) * self.kernel_scale)
        if not self.static_per_call:
            const += static * self.kernel_scale
        classes = self._glue_at(
            g,
            mq,
            mkv,
            len(pieces),
            sum(q for q, _ in pieces),
            sum(kv for _, kv in pieces),
            q_physical,
            kv_physical,
            nbytes,
        )
        helpers = classes["helper_q"] + classes["helper_kv"]
        rest = classes["prep"] + classes["accumulate"] + classes["memset"] + classes["copy"]
        return parts["tile_ms"] * self.kernel_scale, const, helpers, rest

    def time_ms(self, g, S, cp, lengths, padded, frequency, topology):
        """Scalar projection of compose(), retaining its f0 critical-rank rule."""
        if cp == 1:
            return self.compose(g, S, cp, lengths, padded, frequency, None, topology)["wall_ms"]
        q_heads, kv_heads, head_dim = self.cp1.params(g)
        structure, _ = self.structure_for(cp, topology)
        scale = self.kernel_scale
        per_rank = [[] for _ in range(cp)]
        mq, mkv = mib_q(S, q_heads, head_dim), mib_kv(S, kv_heads, head_dim)
        sizes = step_bytes(cp, b_kv_bytes(S, kv_heads, head_dim))
        rounds = round_pieces(lengths, padded, cp)
        for rank in range(cp):
            # Reverse forward-ring steps to keep the backward launch/sum order.
            for step in reversed(rounds[rank * cp : (rank + 1) * cp]):
                i = cp - 1 - step["step"]
                sq, sk = step_sizes(step["section"], mq, mkv)
                parts = self._step_parts(
                    g,
                    step["section"],
                    step["pieces"],
                    step["q_physical"],
                    step["kv_physical"],
                    sq,
                    sk,
                    sizes[i],
                )
                per_rank[rank].append(parts)
        sums_f0 = [
            sum(tile + const + (helpers + rest) * 1e-3 for tile, const, helpers, rest in row)
            for row in per_rank
        ]
        crit = max(range(cp), key=lambda rank: sums_f0[rank])
        row = per_rank[crit]
        kernel = sum(self.clock_entry(tile, const, frequency)["k_ms"] for tile, const, _, _ in row)
        ratio = self.f0 / frequency if frequency else 1.0
        glue = sum(
            helpers * ratio + rest if self.helpers_follow else helpers + rest
            for _, _, helpers, rest in row
        )
        static = self.static_pack_ms(g, len(lengths)) * scale if self.static_per_call else 0.0
        static_at = self.clock_entry(0.0, static, frequency)["k_ms"]
        tail_at = self.clock_entry(self.tail_f0_ms(g, S), 0.0, frequency)["k_ms"]
        pre = pre_us(self.ring, S, q_heads, kv_heads, head_dim)
        critical_steps = [dict(bytes=nbytes) for nbytes in sizes]
        w0_sum = sum(w0_us(self.ring["W0"], structure, step["bytes"]) for step in critical_steps)
        extra, _ = x_form_eval(
            self.x_form, structure, critical_steps, w0_sum, self.ring["X"], head_dim
        )
        return kernel + glue * 1e-3 + static_at + tail_at + (pre + extra) * 1e-3

    def compose(
        self, g, S, cp, lengths, padded=None, f_mhz=None, critical_rank=None, topology=None
    ) -> dict:
        L = [int(v) for v in lengths]
        f = float(f_mhz) if f_mhz else None
        if cp == 1:
            o = super().compose(g, S, cp, L, padded, f, critical_rank, topology)
            t0 = self.tail_f0_ms(g, S)
            rf = (self.f0 / f) if f else 1.0
            o["tail_ms"] = t0 * rf
            o["tail_f0_ms"] = t0
            o["wall_ms"] += t0 * rf
            o["wall_f0_ms"] += t0
            o["split"]["kernel_f0_ms"] += t0
            o["split"]["kind"] = "cp1_bwd"
            o["attribution"] = self.attribution
            o["flag_1750"] = bool(f and f >= self.flag_mhz)
            o["once_per_pack"] = {"tail_ms": t0 * rf}
            return o
        pad = (
            layout_extend_last(L, cp, S, ALIGNMENT_DEFAULT)
            if padded is None
            else [int(v) for v in padded]
        )
        q_heads, kv_heads, head_dim = self.cp1.params(g)
        steps = bwd_steps(L, pad, cp, S, q_heads, kv_heads, head_dim)
        st_of, extrapolated = self.structure_for(cp, topology)
        share = self.share(g)
        ks = self.kernel_scale
        per_rank = {}
        for st in steps:
            mask = "causal" if st["section"] == "diagonal" else "rectangle"
            parts = self.kernel_parts(g, mask, st["pieces"], st["q_physical"])
            st_static = float(
                parts["kernel"]["static_ms"]
            )  # = kernel(include_static=True) - kernel(include_static=False) of the campaign's model
            const_shared = (parts["const_ms"] - st_static) * ks
            const_used = share * const_shared + (0.0 if self.static_per_call else st_static * ks)
            e = self.clock_entry(parts["tile_ms"] * ks, const_used, f)
            st_rec = {**st, "n_active": len(st["pieces"])}
            cls = self.glue_at(g, st_rec)
            hel = cls["helper_q"] + cls["helper_kv"]
            rest = cls["prep"] + cls["accumulate"] + cls["memset"] + cls["copy"]
            rf = (self.f0 / f) if f else 1.0
            if self.helpers_follow:
                g_f0, g_at, g_sc, g_un = hel + rest, hel * rf + rest, hel, rest
            else:
                g_f0, g_at, g_sc, g_un = hel + rest, hel + rest, 0.0, hel + rest
            w0 = w0_us(self.ring["W0"], st_of, st["bytes"])
            per_rank.setdefault(st["rank"], []).append(
                {
                    **st_rec,
                    "k_f0_ms": parts["tile_ms"] * ks + const_used,
                    "k_ms": e["k_ms"],
                    "k_scaled_f0_ms": e["scaled_f0_ms"],
                    "k_unscaled_ms": e["unscaled_ms"],
                    "tile_f0_ms": parts["tile_ms"] * ks,
                    "const_shared_f0_ms": const_shared,
                    "const_used_f0_ms": const_used,
                    "static_step_f0_ms": st_static * ks,
                    "share": share,
                    "g_f0_us": g_f0,
                    "g_us": g_at,
                    "g_scaled_f0_us": g_sc,
                    "g_unscaled_us": g_un,
                    "glue_classes_us": cls,
                    "W0_us": w0,
                    "factor_rank": e["factor"],
                    "f_rank_mhz": f,
                }
            )
        sums = {r: sum(s_["k_ms"] + s_["g_us"] * 1e-3 for s_ in ss) for r, ss in per_rank.items()}
        sums_f0 = {
            r: sum(s_["k_f0_ms"] + s_["g_f0_us"] * 1e-3 for s_ in ss) for r, ss in per_rank.items()
        }
        crit_argmax = max(sums, key=sums.get)
        crit = int(critical_rank) if critical_rank is not None else crit_argmax
        ss = per_rank[crit]
        f_used = ss[0]["f_rank_mhz"]
        K = sum(s_["k_ms"] for s_ in ss)
        K_f0 = sum(s_["k_f0_ms"] for s_ in ss)
        K_sc = sum(s_["k_scaled_f0_ms"] for s_ in ss)
        K_un = sum(s_["k_unscaled_ms"] for s_ in ss)
        G_us = sum(s_["g_us"] for s_ in ss)
        G_f0_us = sum(s_["g_f0_us"] for s_ in ss)
        G_sc = sum(s_["g_scaled_f0_us"] for s_ in ss)
        G_un = sum(s_["g_unscaled_us"] for s_ in ss)
        W0_sum = sum(s_["W0_us"] for s_ in ss)
        # once per pack (critical rank): static(n_pack) under the law's constant split, the tail tile-like (0 here), pre and X unscaled
        st_pack = self.static_pack_ms(g, len(L)) * ks if self.static_per_call else 0.0
        e_st = law_parts(0.0, st_pack, self.law, f_used, self.f0)
        t_f0 = self.tail_f0_ms(g, S)
        e_t = law_parts(t_f0, 0.0, self.law, f_used, self.f0)
        pre = pre_us(self.ring, S, q_heads, kv_heads, head_dim)
        X_us, x_text = x_form_eval(self.x_form, st_of, ss, W0_sum, self.ring["X"], head_dim)
        wall = K + G_us * 1e-3 + e_st["k_ms"] + e_t["k_ms"] + (pre + X_us) * 1e-3
        scaled_f0 = K_sc + G_sc * 1e-3 + e_st["scaled_f0_ms"] + e_t["scaled_f0_ms"]
        ring_ms = (
            K_un + G_un * 1e-3 + e_st["unscaled_ms"] + e_t["unscaled_ms"] + (pre + X_us) * 1e-3
        )
        flag = bool(f_used and float(f_used) >= self.flag_mhz)
        wall_f0 = K_f0 + G_f0_us * 1e-3 + st_pack + t_f0 + (pre + X_us) * 1e-3
        return {
            "geometry": g,
            "S": int(S),
            "cp": int(cp),
            "padded_lengths": pad,
            "wall_ms": wall,
            "wall_f0_ms": wall_f0,
            "K_f0_ms": K_f0,
            "K_ms": K,
            "G_ms": G_us * 1e-3,
            "G_f0_ms": G_f0_us * 1e-3,
            "X_ms": X_us * 1e-3,
            "pre_ms": pre * 1e-3,
            "remainder_ms": 0.0,
            "static_pack_ms": e_st["k_ms"],
            "static_pack_f0_ms": st_pack,
            "tail_ms": e_t["k_ms"],
            "tail_f0_ms": t_f0,
            "once_per_pack": {
                "static_ms": e_st["k_ms"],
                "tail_ms": e_t["k_ms"],
                "pre_ms": pre * 1e-3,
                "X_ms": X_us * 1e-3,
                "x_form": x_text,
            },
            "scale": ss[0]["factor_rank"] * ks,
            "W0_sum_us": W0_sum,
            "structure": st_of,
            "structure_extrapolated": extrapolated,
            "critical_rank": crit,
            "critical_rank_argmax": crit_argmax,
            "per_rank_sum_ms": {str(r): v for r, v in sums.items()},
            "per_rank_sum_f0_ms": {str(r): v for r, v in sums_f0.items()},
            "f_used_mhz": f_used,
            "flag_1750": flag,
            "law": self.law,
            "attribution": self.attribution,
            "share_used": share,
            "static_per_call": self.static_per_call,
            "glue_scaled": self.helpers_follow,
            "steps": [{k: v for k, v in s_.items() if k != "pieces"} for s_ in ss],
            "split": {
                "kernel_f0_ms": scaled_f0,
                "ring_ms": ring_ms,
                "rem0_f0_ms": 0.0,
                "rem0_rule": None,
                "rem_bwd_rule": None,
                "accounting": "crit",
                "kind": COMPOSITION_S4_BWD_R2,
                "clock_quantity": CLOCK_QUANTITY,
                "critical_rank": crit,
                "scaled_part": "kernel tile + (1 - lambda) x share C_shared + helper classes + (1 - lambda) static(n_pack) + T_tail",
                "clock_law_override": e_st["clock_law_override"],
                "glue_f0_ms": G_f0_us * 1e-3,
                "kernel_only_f0_ms": K_f0,
                "kernel_unscaled_ms": K_un,
                "tile_f0_ms": sum(s_["tile_f0_ms"] for s_ in ss),
                "const_used_f0_ms": sum(s_["const_used_f0_ms"] for s_ in ss),
                "const_shared_f0_ms": sum(s_["const_shared_f0_ms"] for s_ in ss),
                "static_steps_f0_ms_not_paid": (
                    sum(s_["static_step_f0_ms"] for s_ in ss) if self.static_per_call else 0.0
                ),
                "static_pack_f0_ms": st_pack,
                "X_us": X_us,
                "pre_us": pre,
            },
        }


class BackwardRound2Model(BackwardRound1Model):
    """The GB200 backward ROUND-2 head (b12) of the data file's ``served_round2`` entry: the round-1 vendored files (data/) under the round-2 attribution vendored in
    data_gb200_round2/ (round2_served_setting.json: shares / static once / no tail / X = cp x g0(head_dim); the f_bwd2 / h_bwd2 verdicts). The same surface as the round-1 model.
    """

    def __init__(
        self, payload, hardware, head=None, data_sha256=None, direction="backward", data_dir=None
    ) -> None:
        super().__init__(payload, hardware, head, data_sha256, direction, data_dir)
        if self.round != 2:
            raise ValueError(
                f"head {self.head!r} is a round-{self.round} head; BackwardRound2Model serves the round-2 head {HEAD_ROUND2!r} (use BackwardRound1Model for b7_1)"
            )
        spec = ROUND2_HARDWARE.get(hardware)
        if spec is None:
            raise ValueError(
                f"no vendored round-2 backward setting for hardware {hardware!r} ({sorted(ROUND2_HARDWARE)})"
            )
        self.round2_spec = dict(spec)
        pins = self.block.get("data_files_round2") or {}
        missing = [name for name in ROUND2_DATA_FILES if name not in pins]
        if missing:
            raise ValueError(
                f"the {hardware} / backward head {self.head!r} does not pin the vendored round-2 files {missing}"
            )
        rel = (self.block.get("composer") or {}).get("round2_data_dir") or spec["data_dir"]
        self.round2_data_dir = PACKAGE_DIR / rel
        self.round2_files, self.round2_file_sha256 = {}, {}
        for name in ROUND2_DATA_FILES:
            self.round2_files[name], self.round2_file_sha256[name] = load_vendored(
                name, pins[name]["sha256"], self.round2_data_dir
            )
        self.round2_manifest = load_manifest(self.round2_data_dir / "MANIFEST.json")
        for name in self.round2_files:
            if (
                self.round2_manifest["vendored"].get(name, {}).get("sha256")
                != self.round2_file_sha256[name]
            ):
                raise ValueError(
                    f"vendored round-2 file {name}: MANIFEST.json pins {self.round2_manifest['vendored'].get(name, {}).get('sha256')}, the file is {self.round2_file_sha256[name]}"
                )
        # the round-1 files the setting shares must be the ones loaded from data/ (the round-2 manifest re-pins their source shas)
        for name, rec in self.round2_manifest["shared_round1_files"].items():
            vend = rec["vendored_as"].split("/", 1)[1]
            if self.file_sha256.get(vend) != rec["vendored_sha256"]:
                raise ValueError(
                    f"round 2 shares the round-1 file {name} vendored as {rec['vendored_as']} ({rec['vendored_sha256'][:16]}), the loaded one is {self.file_sha256.get(vend, '')[:16]}"
                )
        setting = self.round2_files["round2_served_setting.json"]
        r2 = setting["r2"]
        if (
            setting["head"] != HEAD_ROUND2
            or int(setting["round"]) != 2
            or r2.get("attribution") != "round2"
            or r2.get("clock_quantity") is not None
        ):
            raise ValueError(
                f"round2_served_setting.json is not the B12 served setting: head {setting.get('head')!r}, round {setting.get('round')!r}, r2 {str(r2)[:120]}"
            )
        if (
            (r2.get("tail") or {}) != {}
            or not r2.get("static_per_call", True)
            or (r2.get("x_form") or {}).get("form") != "per_step_g0_by_head_dim"
        ):
            raise ValueError(
                "round2_served_setting.json: the served r2 block must have an empty tail, static once per pack and the per_step_g0_by_head_dim X form"
            )
        if (
            abs(float(setting["clock"]["f0_mhz"]) - self.f0_mhz) > 1e-9
            or abs(float(setting["clock"]["kernel_law"]["lambda"]) - self.lam) > 1e-12
        ):
            raise ValueError(
                "round2_served_setting.json clock block disagrees with the vendored ring block (f0 / lambda)"
            )
        self.shares = {str(g): float(v) for g, v in r2["shares"].items()}
        self.x_form = dict(r2["x_form"])
        self.g0_by_head_dim = {int(k): float(v) for k, v in self.x_form["g0_by_head_dim"].items()}
        self.flag_mhz = float(spec.get("flag_mhz", FLAG_MHZ))
        self.composer = BackwardRound2Composer(
            self.cp1,
            self.ograph,
            self.ring,
            self.f0_mhz,
            shares=self.shares,
            tail={},
            x_form=self.x_form,
            static_per_call=True,
            flag_mhz=self.flag_mhz,
        )
        self.served_config = self.round2_files["served_config_f_bwd2.json"]
        self.verdict = self.round2_files["sealed_verdict_f_bwd2.json"]
        self.verdict_h = self.round2_files["sealed_verdict_h_bwd2.json"]
        self.status_marker = str(self.block.get("status_marker") or STATUS_MARKER_ROUND2)
        self.clock = dict(self.clock, flag_mhz=self.flag_mhz, round=2)

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
        values = tuple(int(n) for n in lengths)
        cp, S = int(cp_size), int(storage_tokens)
        if len(values) > self.thd_max:
            raise ValueError("pack exceeds CUDA-graph static sequence slots")
        profile = self.geometry_profile(query_heads, kv_heads, head_dim)
        alignment = (
            self.alignment_default if sequence_alignment is None else int(sequence_alignment)
        )
        padded = layout_extend_last(values, cp, S, alignment, dataset_alignment)
        clock = self.clock_input(
            cp, law=clock_law, use_production_default=use_production_default_clock
        )
        return self.composer.time_ms(profile, S, cp, values, padded, clock["f_mhz"], self.topology)

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
    ) -> dict:
        out = super().predict_with_details(
            lengths,
            cp_size,
            query_heads=query_heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            storage_tokens=storage_tokens,
            sequence_alignment=sequence_alignment,
            clock_mhz_by_rank=clock_mhz_by_rank,
            f_crit_mhz=f_crit_mhz,
            clock_law=clock_law,
            use_production_default_clock=use_production_default_clock,
            two_term_s=two_term_s,
            g_ring=g_ring,
            kappa_w0=kappa_w0,
            dataset_alignment=dataset_alignment,
        )
        reasons = [
            REASON_BACKWARD_ROUND2 if r == REASON_BACKWARD_ROUND1 else r
            for r in out["provisional_reasons"]
        ]
        clock = out["clock"]
        f = clock.get("f_mhz")
        any_rank = any(float(v) >= self.flag_mhz for v in (clock.get("f_by_rank_mhz") or []))
        flag = bool(f is not None and float(f) >= self.flag_mhz)
        if flag:
            reasons.append(REASON_CLOCK_FLAG_1750)
        out["provisional_reasons"] = reasons
        out["model_kind"] = MODEL_KIND_ROUND2
        out["round"] = 2
        out["clock_flag_1750"] = {
            "flag_mhz": self.flag_mhz,
            "applied_clock_flagged": flag,
            "any_rank_flagged": any_rank,
            "rule": "B11.1: a row whose group-median attention clock is >= 1750 MHz is FLAGGED (the telemetry-ceiling regime), never corrected",
        }
        out["round2"] = {
            "shares": dict(self.shares),
            "share_used": self.shares.get(out["geometry_profile"], 1.0),
            "static_per_call": True,
            "tail": None,
            "x_form": dict(self.x_form),
            "g0_us": self.g0_by_head_dim.get(int(head_dim)),
            "attribution": "round2",
        }
        return out


__all__ = [
    "MANIFEST_SCHEMA_ROUND2",
    "HEAD_ROUND2",
    "ENTRY_STATUS_ROUND2",
    "HEAD_STATUS_ROUND2",
    "MODEL_KIND_ROUND2",
    "COMPOSITION_S4_BWD_R2",
    "FLAG_MHZ",
    "ROUND2_DATA_FILES",
    "ROUND2_HARDWARE",
    "REASON_BACKWARD_ROUND2",
    "REASON_CLOCK_FLAG_1750",
    "REASON_HEAD_SUPERSEDED_BWD",
    "STATUS_MARKER_ROUND2",
    "KNOWN_LIMITS_ROUND2",
    "SEALED_SUMMARY_ROUND2",
    "x_form_eval",
    "BackwardRound2Composer",
    "BackwardRound2Model",
    "PACKAGE_DIR",
    "DATA_DIR",
    "MANIFEST_PATH",
    "MANIFEST_SCHEMA",
    "PACKAGE",
    "HEAD",
    "ENTRY_STATUS",
    "HEAD_STATUS",
    "MODEL_KIND",
    "COMPOSITION_S4_BWD",
    "CP1_FORM",
    "CLOCK_QUANTITY",
    "CLOCK_LAW_SERVED",
    "F0_BWD_MHZ",
    "CALIBRATED_CPS",
    "CALIBRATED_STORAGES",
    "ALIGNMENT_DEFAULT",
    "THD_MAX",
    "DATA_FILES",
    "OPTIONAL_DATA_FILES",
    "ROUND1_HARDWARE",
    "DATA_DIRS",
    "HEADS_BY_HARDWARE",
    "STATUS_MARKER_GB300",
    "KNOWN_LIMITS_GB300",
    "SEALED_SUMMARY_GB300",
    "REASON_BACKWARD_ROUND1",
    "REASON_SEALED_CELL_FAILED",
    "REASON_SEALED_CELL_INCONCLUSIVE",
    "REASON_SEALED_CELL_UNSCORED",
    "STATUS_MARKER",
    "KNOWN_LIMITS",
    "SEALED_SUMMARY",
    "sha256_file",
    "load_manifest",
    "load_vendored",
    "kv_units",
    "greedy_chain",
    "static_ms",
    "helper_terms_us",
    "CP1BackwardForm",
    "law_parts",
    "ograph_bwd_us",
    "b_kv_bytes",
    "mib_q",
    "mib_kv",
    "step_bytes",
    "step_sizes",
    "bwd_steps",
    "w0_us",
    "glue_classes_step_us",
    "pre_us",
    "BackwardRound1Composer",
    "BackwardRound1Model",
]
