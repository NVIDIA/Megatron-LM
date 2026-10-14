# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Cross-run contracts using synthetic CPU records, never hardware evidence."""

import copy
import json
import shutil
import subprocess
import sys

import pytest

from tests.unit_tests.determinism_reporting.test_collective_baseline import (
    artifacts,
    consumer,
    write_json,
)
from tests.unit_tests.determinism_reporting.test_collective_performance import (
    rank_results,
    replay_records,
)
from tests.unit_tests.determinism_reporting.test_paired_performance import SCRIPTS, load_module


@pytest.fixture
def calibration(consumer, monkeypatch):
    monkeypatch.setitem(sys.modules, "collective_baseline", consumer)
    return load_module("collective_calibration")


def bundle(
    calibration,
    root,
    *,
    device="first",
    overhead=1.2,
    scale=1.0,
    has_base=False,
    limit=None,
    mutate=None,
    ranks=2,
):
    producer = calibration.collective_baseline
    capture, coverage, timing = artifacts(producer, root / "input", has_base=has_base, limit=limit)
    captures = [
        json.loads((capture / f"rank-{rank}/manifest.json").read_text()) for rank in range(2)
    ]
    assert ranks in (2, 4)
    for rank in range(2, ranks):
        value = copy.deepcopy(captures[rank - 2])
        value['rank'] = rank
        for event in value['events']:
            event['signature']['configuration']['collective']['group_ranks'] = [2, 3]
        captures.append(value)
        shutil.copytree(capture / f'rank-{rank - 2}', capture / f'rank-{rank}')
    report = json.loads(timing.read_text())
    for value in captures:
        value["context"].update(
            python="3.12-fixture",
            versions={"torch": "fixture", "triton": "fixture"},
            cuda="fixture",
            driver=["fixture-driver"],
            world_size=ranks,
        )
        for event in value["events"]:
            event["signature"]["configuration"]["collective"][
                "device_uuid"
            ] = f"GPU-{device}-{value['rank']}"
    report["machine"] = {
        "host": device + "-host",
        "python": "3.12-fixture",
        "versions": {"torch": "fixture"},
        "cuda_visible_devices": ','.join(map(str, range(ranks))),
        "gpus": [
            f"{rank}, GPU-{device}-{rank}, Synthetic H100, fixture-driver" for rank in range(ranks)
        ],
    }
    if mutate is not None:
        mutate(captures, report, capture)
    for rank, value in enumerate(captures):
        value["context_after"] = copy.deepcopy(value["context"])
        write_json(capture / f"rank-{rank}/manifest.json", value)
    evidence = replay_records(captures)
    evidence['ranks_present'] = list(range(ranks))
    write_json(coverage, evidence)
    settings, sources = report["measurement"], report["sources"]
    settings['world_size'] = ranks
    settings["manifest_sha256"] = [
        producer._digest((capture / f"rank-{rank}/manifest.json").read_bytes())
        for rank in range(ranks)
    ]
    sources["head"]["revision"] = captures[0]["context"]["revision"]
    report["capture"].update(context=captures[0]["context"], recipe_id=captures[0]["recipe_id"])
    report["replay_evidence"].update(
        sha256=producer._digest(coverage.read_bytes()),
        run_id=evidence["run_id"],
        events=producer.collective_case.validate_evidence(
            captures, settings["event_indices"], evidence
        ),
    )
    for arm in report["runs"]:
        pair, label, mode = arm["pair"], arm["revision_label"], arm["mode"]
        directory = timing.parent / f"pair-{pair}/{label}-{mode}"
        results = rank_results(
            producer.collective_case, captures, settings, mode, sources[label]["revision"]
        )
        for rank, value in enumerate(results):
            for row in value["rows"].values():
                factor = scale * (1 + pair / 100)
                factor *= (overhead if label == "head" else 1.1) if mode == "det" else 1
                factor *= 1.02 if label == "head" else 1
                row["samples_ms"] = [sample * factor for sample in row["samples_ms"]]
            write_json(directory / f"rank-{rank}.json", value)
        write_json(
            directory / "request.json",
            {
                "head_checkout": sources["head"]["checkout"],
                "capture": report["capture"]["path"],
                "source": sources[label],
                "mode": mode,
                "measurement": settings,
            },
        )
        arm["rank_files"] = {
            f"rank-{rank}.json": producer._digest((directory / f"rank-{rank}.json").read_bytes())
            for rank in range(ranks)
        }
        arm["rows"] = producer.collective_case.aggregate_arm(
            results, captures, settings, mode, sources[label]["revision"]
        )
    report["comparisons"] = producer.collective_case.comparisons(
        report["runs"], settings, has_base, limit, None
    )
    write_json(timing, report)
    result = producer.publish(
        capture, coverage, timing, root / "store", sources["head"]["revision"], str(root)
    )
    return root / "store" / result["baseline_id"], result["baseline_id"]


def test_each_group_keeps_per_run_estimates_without_pooling(calibration, tmp_path):
    first = bundle(calibration, tmp_path / "first", overhead=1.2)
    second = bundle(calibration, tmp_path / "second", device="second", overhead=0.9)
    result = calibration.compare([second, first])
    assert result == calibration.compare([first, second])
    assert result["status"] == "report_only" and result["performance_gate"] == "not_gated"
    assert len(result["workloads"]) == 1 and len(result["groups"]) == 2
    assert {group["context"]["phase"] for group in result["groups"]} == {"forward", "backward"}
    for group in result["groups"]:
        assert group["run_count"] == group["distinct_rank_device_assignments"] == 2
        assert group["repeat_status"] == "repeated_measurements"
        values = group["comparisons"]["head_overhead"]
        assert values["minimum_ratio"] == pytest.approx(0.9)
        assert values["maximum_ratio"] == pytest.approx(1.2)
        assert values["median_ratio"] == pytest.approx(1.05)
        assert values["observed_range_percentage_points"] == pytest.approx(30)
        assert "bootstrap_95_percent_interval" not in values
        assert all(
            len(run["comparisons"]["head_overhead"]["paired_ratios"]) == 3 for run in group["runs"]
        )
        assert all(len(run["arm_medians"]) == 6 for run in group["runs"])
        assert all(
            "bootstrap_95_percent_interval" in run["comparisons"]["head_overhead"]
            for run in group["runs"]
        )
    markdown = calibration.markdown_report(result)
    assert "1.200000" in markdown and "no samples are pooled" in markdown


def test_same_devices_do_not_prove_independent_allocations(calibration, tmp_path):
    first = bundle(calibration, tmp_path / "first")
    second = bundle(calibration, tmp_path / "second", scale=1.037, overhead=0.95)
    groups = calibration.compare([first, second])["groups"]
    assert all(
        group["run_count"] == 2 and group["distinct_rank_device_assignments"] == 1
        for group in groups
    )
    assert all(
        group["repeat_status"] == "single_measurement"
        for group in calibration.compare([first])["groups"]
    )


def test_distinct_groups_are_not_pooled_and_changed_membership_splits(calibration, tmp_path):
    first = bundle(calibration, tmp_path / 'first', ranks=4)
    second = bundle(calibration, tmp_path / 'second', ranks=4, device='second', overhead=0.9)
    groups = calibration.compare([first, second])['groups']
    assert len(groups) == 4 and all(group['run_count'] == 2 for group in groups)
    assert {tuple(group['context']['group_ranks']) for group in groups} == {(0, 1), (2, 3)}

    def world_group(captures, report, capture):
        for rank, value in enumerate(captures):
            for event in value['events']:
                event['signature']['configuration']['collective'].update(
                    group_ranks=[0, 1, 2, 3], group_rank=rank, size=4
                )

    third = bundle(calibration, tmp_path / 'third', ranks=4, device='third', mutate=world_group)
    groups = calibration.compare([first, third])['groups']
    assert len(groups) == 6 and all(group['run_count'] == 1 for group in groups)


def test_changed_other_group_schedule_is_not_hidden(calibration, tmp_path):
    def other_group(captures, report, capture):
        for value in captures[2:]:
            for event in value['events']:
                event['signature']['configuration']['collective']['group_options']['config'][
                    'min_ctas'
                ] += 1

    first = bundle(calibration, tmp_path / 'first', ranks=4)
    second = bundle(calibration, tmp_path / 'second', ranks=4, device='second', mutate=other_group)
    groups = calibration.compare([first, second])['groups']
    assert len(groups) == 8 and all(group['run_count'] == 1 for group in groups)


def test_base_regressions_and_policy_overhead_stay_separate(calibration, tmp_path):
    result = calibration.compare([bundle(calibration, tmp_path, has_base=True, overhead=1.3)])
    for group in result["groups"]:
        values = group["comparisons"]
        assert set(values) == {
            "head_overhead",
            "base_overhead",
            "default_regression",
            "det_regression",
        }
        assert values["head_overhead"]["median_ratio"] == pytest.approx(1.3)
        assert values["base_overhead"]["median_ratio"] == pytest.approx(1.1)
        assert values["default_regression"]["median_ratio"] == pytest.approx(1.02)
        assert values["det_regression"]["median_ratio"] == pytest.approx(1.02 * 1.3 / 1.1)


@pytest.mark.parametrize(
    "field",
    [
        "source",
        "base",
        "driver",
        "versions",
        "environment",
        "tooling",
        "warmup",
        "gpu_inventory",
        "call_id",
        "recipe",
        "layout",
        "input",
        "gradient",
        "group_options",
        "peer_input",
        "fabric_metadata",
    ],
)
def test_complete_workload_and_runtime_changes_split_cohorts(calibration, tmp_path, field):
    def mutate(captures, report, capture):
        if field in ("source", "driver", "versions", "environment"):
            for value in captures:
                context = value["context"]
                if field == "source":
                    context["revision"] = "c" * 40
                elif field == "driver":
                    context["driver"] = ["changed-driver"]
                elif field == "versions":
                    context["versions"]["torch"] = "changed-version"
                else:
                    context["environment"]["TRITON_CACHE_AUTOTUNING"] = "changed-policy"
        elif field == "base":
            report["sources"]["base"]["revision"] = "c" * 40
        elif field == "tooling":
            report["measurement"]["tooling"]["synthetic-fixture.py"] = "1" * 64
        elif field == "warmup":
            report["measurement"]["warmup"] += 1
        elif field == "gpu_inventory":
            report["machine"]["gpus"] = [
                row.replace("Synthetic H100", "Different GPU") for row in report["machine"]["gpus"]
            ]
        elif field == "fabric_metadata":
            report["machine"]["fabric"] = "different recorded fixture fabric"
        elif field in ("call_id", "recipe", "layout", "group_options"):
            for value in captures:
                if field == "recipe":
                    value["recipe_id"] = "different-fixture"
                for event in value["events"]:
                    collective = event["signature"]["configuration"]["collective"]
                    if field == "call_id":
                        event["call_id"] += 1
                    elif field == "layout":
                        collective["input"]["storage_offset"] += 1
                    elif field == "group_options":
                        collective["group_options"]["config"]["min_ctas"] += 1
        else:
            rank = 1 if field == "peer_input" else 0
            role = "gradient" if field == "gradient" else "input"
            original = next(
                event["signature"]["configuration"]["collective"][role]["sha256"]
                for event in captures[rank]["events"]
                if role in event["signature"]["configuration"]["collective"]
            )
            path = capture / f"rank-{rank}/{original}.bin"
            raw = bytearray(path.read_bytes())
            raw[0] ^= 1
            updated = calibration.collective_baseline._digest(bytes(raw))
            path.rename(path.with_name(updated + ".bin"))
            path.with_name(updated + ".bin").write_bytes(raw)
            for event in captures[rank]["events"]:
                collective = event["signature"]["configuration"]["collective"]
                if role in collective:
                    collective[role]["sha256"] = updated

    first = bundle(calibration, tmp_path / "first", has_base=field == "base")
    second = bundle(
        calibration, tmp_path / "second", device="second", has_base=field == "base", mutate=mutate
    )
    groups = calibration.compare([first, second], compare_cache_locations=True)["groups"]
    assert len(groups) == 4 and all(group["run_count"] == 1 for group in groups)


def test_cache_location_opt_in_preserves_paths_and_unset_state(calibration, tmp_path):
    def cache(path):
        def mutate(captures, report, capture):
            for value in captures:
                value["context"]["environment"]["TRITON_CACHE_DIR"] = path

        return mutate

    first = bundle(calibration, tmp_path / "first", mutate=cache("/first/cache"))
    second = bundle(
        calibration, tmp_path / "second", device="second", mutate=cache("/second/cache")
    )
    assert len(calibration.compare([first, second])["groups"]) == 4
    result = calibration.compare([first, second], compare_cache_locations=True)
    assert len(result["groups"]) == 2
    assert {
        path
        for group in result["groups"]
        for run in group["runs"]
        for path in run["cache_locations"]
    } == {"/first/cache", "/second/cache"}
    unset = bundle(calibration, tmp_path / "unset", device="unset", mutate=cache(None))
    assert len(calibration.compare([first, unset], compare_cache_locations=True)["groups"]) == 4


@pytest.mark.parametrize(
    "kind", ["same_id", "relocated", "relabelled", "recipe_label", "partial_arm_reuse"]
)
def test_duplicates_and_shared_timing_arms_do_not_become_new_runs(calibration, tmp_path, kind):
    first = bundle(calibration, tmp_path / "first")
    if kind == "same_id":
        second = first
    elif kind == "relocated":
        relocated = tmp_path / "relocated"
        shutil.copytree(first[0], relocated)
        second = relocated, first[1]
    else:

        def mutate(captures, report, capture):
            if kind == "recipe_label":
                for value in captures:
                    value["recipe_id"] = "relabeled recipe"

        second = bundle(
            calibration, tmp_path / "second", has_base=kind == "partial_arm_reuse", mutate=mutate
        )
    with pytest.raises(ValueError, match="Repeated"):
        calibration.compare([first, second])


@pytest.mark.parametrize(
    "field",
    [
        "missing_runtime",
        "missing_inventory",
        "duplicate_inventory",
        "foreign_uuid",
        "rank_moves",
        "malformed_inventory",
        "empty_cache",
    ],
)
def test_incomplete_or_ambiguous_metadata_cannot_calibrate(calibration, tmp_path, field):
    def mutate(captures, report, capture):
        if field == "missing_runtime":
            for value in captures:
                del value["context"]["driver"]
        elif field == "missing_inventory":
            report["machine"].pop("gpus")
        elif field == "duplicate_inventory":
            report["machine"]["gpus"].append(report["machine"]["gpus"][0])
        elif field == "foreign_uuid":
            report["machine"]["gpus"][0] = report["machine"]["gpus"][0].replace(
                "GPU-first-0", "GPU-other-0"
            )
        elif field == "malformed_inventory":
            report["machine"]["gpus"][0] = 'invalid\nCSV'
        elif field == "empty_cache":
            for value in captures:
                value["context"]["environment"]["TRITON_CACHE_DIR"] = ""
        else:
            # Preserve the forward/backward pair while mapping two ranks to one GPU.
            for event in captures[1]["events"]:
                event["signature"]["configuration"]["collective"]["device_uuid"] = "GPU-first-0"

    selected = bundle(calibration, tmp_path, mutate=mutate)
    with pytest.raises(ValueError):
        calibration.compare([selected])


@pytest.mark.parametrize(
    "fault",
    [
        "wrong_id",
        "changed_sample",
        "missing_blob",
        "manifest",
        "budget",
        "after_verify_capture",
        "after_verify_timing",
        "after_verify_rank",
    ],
)
def test_bundles_are_reverified_before_comparison(calibration, tmp_path, monkeypatch, fault):
    directory, identity = bundle(calibration, tmp_path, limit=1.35 if fault == "budget" else None)
    if fault.startswith("after_verify_"):
        original = calibration.collective_baseline.verify
        target = {
            "after_verify_capture": "capture/rank-0/manifest.json",
            "after_verify_timing": "timing/benchmark.json",
            "after_verify_rank": "timing/pair-0/head-default/rank-0.json",
        }[fault]

        def changed(*args):
            result = original(*args)
            (directory / target).write_text('{}')
            return result

        monkeypatch.setattr(calibration.collective_baseline, "verify", changed)
    elif fault == "wrong_id":
        identity = "0" * 64
    elif fault == "changed_sample":
        (directory / "timing/pair-0/head-default/rank-0.json").write_text('{}')
    elif fault == "missing_blob":
        next((directory / "capture").rglob('*.bin')).unlink()
    elif fault == "manifest":
        (directory / "baseline.json").write_text('{}')
    with pytest.raises(ValueError):
        calibration.compare([(directory, identity)])


def test_cli_is_stdlib_only_preserves_output_and_rejects_mixed_formats(calibration, tmp_path):
    directory, identity = bundle(calibration, tmp_path / "fixture")
    output = tmp_path / "report.json"
    command = [
        sys.executable,
        "-S",
        str(SCRIPTS / "calibration.py"),
        "--collective-baseline",
        str(directory),
        identity,
        "--output",
        str(output),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    original = output.read_bytes()
    assert json.loads(original)["kind"] == "determinism_collective_calibration"
    assert output.with_suffix('.md').is_file()
    assert (
        subprocess.run(command, capture_output=True).returncode == 1
        and output.read_bytes() == original
    )
    assert (
        subprocess.run(
            command + ["--baseline", str(directory), identity], capture_output=True
        ).returncode
        == 2
    )
    with pytest.raises(ValueError, match="pinned"):
        calibration.compare([])
