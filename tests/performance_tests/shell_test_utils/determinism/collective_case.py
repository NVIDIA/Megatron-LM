# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Contracts for same-allocation timings of actual captured TP/SP calls."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import math
import statistics
import sys
from pathlib import Path

from benchmark import DEFAULT_ENV, MODE_ENV, summarize

ADAPTER = "captured_collective_v1"
VERIFIED = "verified_deterministic"


def digest(path: Path) -> str:
    """Hash retained metadata or a tooling source file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def install_head_helpers(root: Path) -> None:
    """Load head test helpers without putting head production code on sys.path.

    Workers start in the selected checkout with its PYTHONPATH. Only the tools
    package comes from the driver checkout; megatron remains untouched.
    Reject existing packages from another checkout rather than mix revisions.
    """
    for name in ("tools",):
        path = root / name / "__init__.py"
        if name in sys.modules:
            loaded_path = sys.modules[name].__file__
            if not loaded_path or Path(loaded_path).resolve() != path.resolve():
                raise ValueError(f"{name} was already imported from a different checkout")
            continue
        spec = importlib.util.spec_from_file_location(
            name, path, submodule_search_locations=[str(path.parent)]
        )
        if spec is None or spec.loader is None:
            raise ValueError(f"Cannot load head helper package: {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)


def policy_environment(parent: dict, context: dict, collective: dict, mode: str) -> dict:
    """Preserve capture settings, changing only the declared algorithm policy."""
    if mode not in ("det", "default"):
        raise ValueError("Unknown timing mode")
    env = dict(parent)
    for key in list(env):
        if key in MODE_ENV or key.startswith(("NCCL_", "TORCH_NCCL_", "TRITON_AUTOTUNE_BLOCK_")):
            env.pop(key)
    selected = {**context["environment"], **collective["nccl_environment"]}
    if mode == "default":
        selected.update(
            {key: DEFAULT_ENV.get(key) for key in MODE_ENV if key != "TRITON_CACHE_AUTOTUNING"}
        )
    for key, value in selected.items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = value
    env["DETERMINISM_PERF_MODE"] = mode
    env["PYTHONHASHSEED"] = "0"
    return env


def mode_signature(signature: dict, mode: str) -> dict:
    """Keep the original capture immutable and declare the default arm's changes."""
    if mode not in ("det", "default") or signature["deterministic_algorithms"] is not True:
        raise ValueError("Timings require a strict deterministic capture")
    result = copy.deepcopy(signature)
    if mode == "default":
        result["deterministic_algorithms"] = False
        result["runtime"]["cudnn_deterministic"] = False
        result["configuration"]["collective"]["nccl_environment"]["NCCL_ALGO"] = None
    return result


def mode_context(context: dict, mode: str, revision: str) -> dict:
    """Change only the measured source and explicit policy fields."""
    result = copy.deepcopy(context)
    result["revision"] = revision
    if mode == "default":
        for key in MODE_ENV:
            if key != "TRITON_CACHE_AUTOTUNING" and key in result["environment"]:
                result["environment"][key] = DEFAULT_ENV.get(key)
    return result


def configure_policy(root: Path, runtime: dict, mode: str) -> None:
    """Apply head startup before CUDA without importing head megatron in base arms."""
    import torch
    import torch.utils.deterministic

    if runtime["autocast"] or runtime["cudnn_benchmark"]:
        raise ValueError("Autocast or cuDNN benchmark captures need a separate timing adapter")
    if mode == "det":
        name = "_collective_benchmark_startup"
        path = root / "megatron" / "determinism" / "__init__.py"
        spec = importlib.util.spec_from_file_location(
            name, path, submodule_search_locations=[str(path.parent)]
        )
        if spec is None or spec.loader is None:
            raise ValueError(f"Cannot load head startup package: {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        module.configure_determinism({"deterministic_mode": True})
    elif mode == "default":
        torch.use_deterministic_algorithms(False, warn_only=False)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = False
    else:
        raise ValueError("Unknown timing mode")
    # Memory fill and numerical settings are held fixed across policy arms.
    torch.utils.deterministic.fill_uninitialized_memory = runtime["fill_uninitialized_memory"]
    torch.set_float32_matmul_precision(runtime["float32_matmul_precision"])
    torch.backends.cuda.matmul.allow_tf32 = runtime["matmul_allow_tf32"]
    torch.backends.cudnn.allow_tf32 = runtime["cudnn_allow_tf32"]
    torch.set_autocast_dtype(
        "cuda", getattr(torch, runtime["autocast_dtype"].removeprefix("torch."))
    )


def validate_evidence(captures: list[dict], indices: list[int], evidence: dict) -> list[dict]:
    """Require one complete current replay/reference/control case for every event."""
    world = len(captures)
    if (
        evidence.get("schema_version") != 1
        or evidence.get("kind") != "determinism_coverage"
        or evidence.get("context") != captures[0]["context"]
        or evidence.get("ranks_present") != list(range(world))
        or not evidence.get("run_id")
    ):
        raise ValueError("Replay evidence has different source/runtime or incomplete ranks")
    selected = []
    for index in indices:
        signatures = [report["events"][index]["signature"] for report in captures]
        candidates = []
        for case in evidence["cases"]:
            observations = case.get("observations", [])
            if (
                len(observations) == world
                and {row["rank"] for row in observations} == set(range(world))
                and all(row["signature"] == signatures[row["rank"]] for row in observations)
            ):
                candidates.append(case)
        if not candidates:
            raise ValueError(f"Event {index} has no complete matching replay case")
        for case in candidates:
            if case.get("status") != VERIFIED or any(
                case.get("check_status", {}).get(kind) != "passed"
                for kind in ("reference", "sensitivity")
            ):
                raise ValueError(f"Event {index} has failing replay or accuracy evidence")
            for observation in case["observations"]:
                gradient_count = int(observation["signature"]["phase"] == "forward_backward")
                protocol = observation.get("protocol", {})
                if (
                    observation.get("status") != VERIFIED
                    or type(protocol.get("replays")) is not int
                    or protocol["replays"] < 2
                    or protocol.get("warn_only") is not False
                    or protocol.get("contention") is not True
                    or type(observation.get("compared_outputs")) is not int
                    or type(observation.get("compared_gradients")) is not int
                    or observation.get("compared_outputs") != 1
                    or observation.get("compared_gradients") != gradient_count
                ):
                    raise ValueError(f"Event {index} lacks complete replay observations")
                for kind in ("reference", "sensitivity"):
                    checks = [
                        row
                        for row in case.get("checks", [])
                        if row.get("kind") == kind
                        and row.get("rank") == observation["rank"]
                        and row.get("signature") == observation["signature"]
                    ]
                    if not checks or any(
                        row.get("status") != "passed"
                        or type(row.get("compared_outputs")) is not int
                        or type(row.get("compared_gradients")) is not int
                        or row.get("compared_outputs") != 1
                        or row.get("compared_gradients") != gradient_count
                        or (
                            kind == "sensitivity"
                            and (
                                type(row.get("detected_perturbations")) is not int
                                or row["detected_perturbations"] != 1 + gradient_count
                            )
                        )
                        for row in checks
                    ):
                        raise ValueError(f"Event {index} lacks matching {kind} checks")
        selected.append(
            {"event_index": index, "case_ids": [case["case_id"] for case in candidates]}
        )
    return selected


def validate_rank(
    result: dict, captures: list[dict], measurement: dict, mode: str, revision: str
) -> None:
    """Reject missing samples, changed rank-local work, policy or allocation."""
    rank = result.get("rank")
    if type(rank) is not int or not 0 <= rank < len(captures):
        raise ValueError("Invalid timing rank")
    capture = captures[rank]
    if (
        result.get("adapter") != ADAPTER
        or result.get("measurement") != measurement
        or result.get("mode") != mode
        or result.get("context") != mode_context(capture["context"], mode, revision)
        or result.get("context_after") != result["context"]
        or result.get("capture_manifest_sha256") != measurement["manifest_sha256"][rank]
        or result.get("tooling") != measurement["tooling"]
        or set(result.get("rows", {})) != {str(index) for index in measurement["event_indices"]}
    ):
        raise ValueError("Timing rank has different measurement/source/environment/tooling")
    for index in measurement["event_indices"]:
        row = result["rows"][str(index)]
        signature = capture["events"][index]["signature"]
        expected = mode_signature(signature, mode)
        samples = row.get("samples_ms")
        if (
            row.get("capture_signature") != signature
            or row.get("actual_signature") != expected
            or row.get("signature_after") != expected
            or row.get("phase")
            != ("backward" if signature["phase"] == "forward_backward" else "forward")
            or not isinstance(samples, list)
            or len(samples) != measurement["steps"]
            or any(
                type(value) not in (int, float) or not math.isfinite(value) or value <= 0
                for value in samples
            )
        ):
            raise ValueError(f"Timing rank {rank}, event {index} has invalid policy/work/samples")


def aggregate_arm(
    results: list[dict], captures: list[dict], measurement: dict, mode: str, revision: str
) -> dict:
    """Take each group's maximum per aligned sample, never pool ranks as trials."""
    if len(results) != len(captures) or {r.get("rank") for r in results} != set(
        range(len(captures))
    ):
        raise ValueError("Timing ranks are incomplete or duplicated")
    for result in results:
        validate_rank(result, captures, measurement, mode, revision)
    by_rank = {result["rank"]: result for result in results}
    rows = {}
    for index in measurement["event_indices"]:
        for capture in captures:
            signature = capture["events"][index]["signature"]
            collective = signature["configuration"]["collective"]
            members = collective["group_ranks"]
            key = f"event-{index}/ranks-" + "-".join(map(str, members))
            if key in rows:
                continue
            samples = [
                max(by_rank[rank]["rows"][str(index)]["samples_ms"][step] for rank in members)
                for step in range(measurement["steps"])
            ]
            rows[key] = {
                "event_index": index,
                "group_ranks": members,
                "case": collective["case"],
                "phase": by_rank[members[0]]["rows"][str(index)]["phase"],
                "samples_ms": samples,
                "median_ms": statistics.median(samples),
            }
    return rows


def comparisons(
    runs: list[dict],
    measurement: dict,
    has_base: bool,
    overhead: float | None,
    regression: float | None,
) -> dict:
    """Reuse the paired-process estimator separately for every captured group/event."""
    if not runs or any(set(run["rows"]) != set(runs[0]["rows"]) for run in runs):
        raise ValueError("Paired collective rows differ or are missing")
    return {
        key: summarize(
            [
                {
                    **{name: run[name] for name in ("pair", "revision_label", "mode", "status")},
                    "median_ms": run["rows"][key]["median_ms"],
                }
                for run in runs
            ],
            measurement["pairs"],
            has_base,
            overhead,
            regression,
        )
        for key in runs[0]["rows"]
    }
