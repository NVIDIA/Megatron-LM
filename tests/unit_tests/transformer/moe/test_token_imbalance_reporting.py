# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU regression tests for the per-rank benchmark reporting commands."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

BENCHMARKS = Path(__file__).resolve().parents[4] / "benchmarks/moe/token_imbalance"
WORKLOAD = "W0_balanced"


def write_job(directory: Path, times: list[float], **changes: object) -> None:
    """Write one baseline-driver row per rank as benchmark.py does."""
    directory.mkdir(parents=True, exist_ok=True)
    for rank, ms in enumerate(times):
        record = {
            "workload_id": WORKLOAD,
            "workload_checksum": "schedule-a",
            "base_sha": "base-revision",
            "experiment": "trace_replay",
            "mode": "dispatch_only",
            "ep": len(times),
            "world_size": len(times),
            "rank": rank,
            "tp": 1,
            "etp": 1,
            "topk": 2,
            "hidden_size": 512,
            "num_experts": 8,
            "logical_tokens": 128,
            "steps": 6,
            "median_ms": ms,
            "logical_tokens_per_second": 1000 / ms,
            "peak_allocated_bytes": 20 + rank * 10,
            "status": "OK",
            "correctness": dict.fromkeys(
                (
                    "distinct_topk_ok",
                    "id_range_ok",
                    "assignments_match_T_times_k",
                    "global_histogram_exact",
                    "roundtrip_exact",
                    "counts_len_ok",
                    "local_counts_sum_ok",
                    "combine_factor_is_topk",
                ),
                True,
            ),
            **changes,
        }
        (directory / f"rank{rank}.jsonl").write_text(json.dumps(record) + "\n")


def run_report(script: str, *args: Path | str) -> str:
    return subprocess.check_output([sys.executable, BENCHMARKS / script, *args], text=True)


def write_block(root: Path, kind: str, worst: list[float]) -> list[Path]:
    directories = []
    for pos, (side, ms) in enumerate(zip(kind, worst)):
        tree = "base-main" if side == "A" else "patch"
        directory = root / f"cfg-{kind}" / f"{pos}-{tree}" / "cfg"
        write_job(directory, [0.5, ms])
        directories.append(directory)
    return directories


@pytest.mark.parametrize("kind", ["APPA", "PAAP"])
def test_paired_blocks_use_worst_rank_and_keep_their_column(tmp_path, kind):
    write_block(tmp_path, kind, [10 if side == "A" else 9 for side in kind])
    result = run_report("analyze_paired.py", tmp_path)
    cells = "-10.000% | n/a" if kind == "APPA" else "n/a | -10.000%"
    assert f"| {WORKLOAD} | {cells} | -10.000% | n/a |" in result
    assert "Across 1 block-level effects" in result


def test_block_without_a_workload_keeps_its_column_empty(tmp_path):
    dirs = write_block(tmp_path, "APPA", [10, 9, 9, 10])
    for directory in dirs:
        write_job(directory, [2, 2], workload_id="W7_burst")
    write_block(tmp_path, "PAAP", [9, 10, 10, 9])
    result = run_report("analyze_paired.py", tmp_path)
    assert f"| {WORKLOAD} | n/a | -10.000% |" in result
    assert "| W7_burst | +0.000% | n/a |" in result


def test_paired_two_blocks_use_geometric_means_and_show_leave_one_out(tmp_path):
    write_block(tmp_path, "APPA", [2, 1, 4, 8])  # patch 2 vs base 4
    write_block(tmp_path, "PAAP", [4, 1, 4, 4])  # patch 4 vs base 2
    result = run_report("analyze_paired.py", tmp_path)
    assert f"| {WORKLOAD} | -50.000% | +100.000% | +0.000% | +100.000, -50.000 |" in result
    assert "Across 2 block-level effects" in result


@pytest.mark.parametrize(
    "problem", ["failed", "rank", "workload", "position", "tree", "duplicate", "checksum"]
)
def test_invalid_block_does_not_hide_valid_other_block(tmp_path, problem):
    dirs = write_block(tmp_path, "APPA", [10, 9, 9, 10])
    write_block(tmp_path, "PAAP", [9, 10, 10, 9])
    path = dirs[1] / "rank1.jsonl"
    if problem == "failed":
        record = json.loads(path.read_text())
        record["status"] = "FAILED_CORRECTNESS"
        path.write_text(json.dumps(record))
    elif problem == "rank":
        path.unlink()
    elif problem == "workload":
        write_job(dirs[1], [1, 9], workload_id="W7_burst")
    elif problem == "position":
        dirs[1].parent.rename(dirs[1].parent.with_name("4-patch"))
    elif problem == "tree":
        dirs[1].parent.rename(dirs[1].parent.with_name("1-other-patch"))
    elif problem == "duplicate":
        write_job(dirs[1].parent.parent / "01-patch" / "cfg", [1, 9])
    else:
        write_job(dirs[1], [1, 9], workload_checksum="different")

    result = run_report("analyze_paired.py", tmp_path)
    row = next(line for line in result.splitlines() if line.startswith(f"| {WORKLOAD} |"))
    assert "| INVALID (" in row and "| -10.000% | -10.000% | n/a |" in row
    assert "Across 1 block-level effects" in result


def test_summary_reduces_rank_metrics_and_report_keeps_trees_separate(tmp_path):
    write_job(tmp_path / "base-main/cfg", [1, 10])
    write_job(tmp_path / "patch/cfg", [0.5, 1])
    summary = run_report("summarize.py", "--results-root", tmp_path)
    assert f"| {WORKLOAD} | 10 | 1 | -90.00% | 100.0 | 1000.0 | +900.00% | 30 | OK/OK |" in summary
    assert "1 paired arms: **0.100000**" in summary
    report = run_report("report.py", tmp_path)
    assert "| base-main/cfg | 10.000 |" in report
    assert "| patch/cfg | 1.000 |" in report


def test_report_ands_oracles_within_each_directory_and_handles_missing_rates(tmp_path):
    for tree in ("base-main", "patch"):
        write_job(tmp_path / tree / "cfg", [1, 3], logical_tokens_per_second=None)
    path = tmp_path / "base-main/cfg/rank0.jsonl"
    record = json.loads(path.read_text())
    record["correctness"]["roundtrip_exact"] = False
    path.write_text(json.dumps(record))
    report = run_report("report.py", tmp_path)
    assert f"| base-main/cfg | 1 | **FAIL** | {WORKLOAD} |" in report
    assert "| patch/cfg | 1 | yes | - |" in report
    summary = run_report("summarize.py", "--results-root", tmp_path)
    assert f"| {WORKLOAD} | 3 | 3 | n/a | n/a | n/a | n/a |" in summary


@pytest.mark.parametrize(
    "field,value",
    [
        ("workload_checksum", "other"),
        ("base_sha", "other"),
        ("num_experts", 16),
        ("tp", 2),
        ("etp", 2),
    ],
)
def test_summary_does_not_pair_incompatible_runs(tmp_path, field, value):
    write_job(tmp_path / "base-main/cfg", [10, 10])
    write_job(tmp_path / "patch/cfg", [1, 1], **{field: value})
    result = run_report("summarize.py", "--results-root", tmp_path)
    assert "paired arms:" not in result
    assert "OK/UNPAIRED" in result and "UNPAIRED/OK" in result


@pytest.mark.parametrize(
    "problem", ["status", "oracle", "rank", "checksum", "missing_check", "unknown_sha"]
)
def test_summary_excludes_failed_or_unverified_pairs(tmp_path, problem):
    write_job(tmp_path / "base-main/cfg", [10, 10])
    write_job(tmp_path / "patch/cfg", [1, 1])
    path = tmp_path / "patch/cfg/rank0.jsonl"
    record = json.loads(path.read_text())
    if problem == "status":
        record["status"] = "FAILED_CORRECTNESS"
    elif problem == "oracle":
        record["correctness"]["roundtrip_exact"] = False
    elif problem == "rank":
        record["rank"] = 1
    elif problem == "checksum":
        record["workload_checksum"] = "other"
    elif problem == "missing_check":
        del record["correctness"]["global_histogram_exact"]
    else:
        for tree, ms in [("base-main", 10), ("patch", 1)]:
            write_job(tmp_path / tree / "cfg", [ms, ms], base_sha="UNKNOWN")
        record["base_sha"] = "UNKNOWN"
    path.write_text(json.dumps(record))
    result = run_report("summarize.py", "--results-root", tmp_path)
    assert "paired arms:" not in result
    assert "patch/cfg" in result
    if problem == "status":
        assert "FAILED_CORRECTNESS" in result
    elif problem == "oracle":
        assert "**FAIL**" in result
    else:
        assert "INVALID" in result or "UNPAIRED" in result


def test_repeated_jobs_require_selection(tmp_path):
    for tree in ["base-main", "patch"]:
        for run in ["first", "second"]:
            write_job(tmp_path / tree / run / "cfg", [1])
    with pytest.raises(subprocess.CalledProcessError):
        run_report("summarize.py", "--results-root", tmp_path)
    selected = run_report("summarize.py", "--results-root", tmp_path, "--run-pattern", "first")
    assert "1 paired arms: **1.000000**" in selected


def test_instrumentation_uses_one_worst_rank_time_per_job(tmp_path):
    for config, slow_rank in (("unchanged", 10), ("slower", 12)):
        write_job(tmp_path / "inst-none" / config, [1, 1, 1, 10])
        write_job(tmp_path / "inst-counters" / config, [2, 2, 2, slow_rank])
    report = run_report("analyze_instrument.py", tmp_path)
    rows = [line for line in report.splitlines() if WORKLOAD in line]
    assert len(rows) == 2
    assert any("unchanged" in row and "10.0000 (1) | 10.0000 (1) | +0.000%" in row for row in rows)
    assert any("slower" in row and "10.0000 (1) | 12.0000 (1) | +20.000%" in row for row in rows)


def test_instrumentation_uses_median_of_repeats_and_excludes_invalid_jobs(tmp_path):
    for setting, times in (("none", [1, 1, 3, 3]), ("counters", [2, 2, 2, 2])):
        for repeat, ms in enumerate(times):
            write_job(tmp_path / f"inst-{setting}/cfg/{repeat}", [ms, ms])
    missing = tmp_path / "inst-none/cfg/missing"
    write_job(missing, [0.1, 0.1])
    (missing / "rank1.jsonl").unlink()
    failed = tmp_path / "inst-counters/cfg/failed"
    write_job(failed, [10, 10])
    rank_file = failed / "rank1.jsonl"
    record = json.loads(rank_file.read_text())
    rank_file.write_text(json.dumps({**record, "status": "FAILED_CORRECTNESS"}))
    report = run_report("analyze_instrument.py", tmp_path)
    assert "2.0000 (4) | 2.0000 (4) | +0.000%" in report
    assert "missing or duplicate ranks" in report and "FAILED_CORRECTNESS" in report


def test_instrumentation_reports_measured_slowdown_and_unpaired_workloads(tmp_path):
    for setting, ms in (("none", 1), ("counters", 2)):
        write_job(tmp_path / f"inst-{setting}/cfg", [ms])
        write_job(tmp_path / f"inst-{setting}/mixed", [ms], workload_checksum=setting)
    report = run_report("analyze_instrument.py", tmp_path)
    assert "1.0000 (1) | 2.0000 (1) | +100.000%" in report
    assert report.count("unpaired setting") == 2
    assert "co-tenant" not in report and "NOT measurably slower" not in report


@pytest.mark.parametrize("problem", ["missing", "duplicate"])
def test_report_marks_jobs_without_every_rank_once(tmp_path, problem):
    job = tmp_path / "base-main/cfg"
    write_job(job, [1, 3])
    if problem == "missing":
        (job / "rank1.jsonl").unlink()
    else:
        rank0 = job / "rank0.jsonl"
        rank0.write_text(rank0.read_text() * 2)
    report = run_report("report.py", tmp_path)
    assert f"| base-main/cfg | 1 | **FAIL** | {WORKLOAD} |" in report
    section1 = report.split("## 2.")[0]
    assert "| base-main/cfg |" not in section1
