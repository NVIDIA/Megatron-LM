# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Explicit fixed-worker-count redistribution for Energon lane checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True)
class EnergonRescaleProfile:
    """Semantic identity and ordered native sampler partitions for a fixed P."""

    global_batch_size: int
    semantic_sha256: str
    worker_partitions: tuple[str, ...]
    pipeline_config: dict[str, Any]


def _digest(value: Any) -> str:
    serialized = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )
    return hashlib.sha256(serialized).hexdigest()


def _native_config_digest(value: Any) -> str:
    """Match the provider's strict native-config hash serialization."""
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()


def _placement(contract: dict[str, Any]) -> tuple[int, int, int]:
    if not isinstance(contract, dict):
        raise ValueError("Energon checkpoint contract must be a dictionary")
    values = tuple(contract.get(key) for key in ("lane", "language_lanes", "num_workers"))
    if any(type(value) is not int for value in values):
        raise ValueError("Energon placement requires integer lane, language_lanes, and num_workers")
    lane, lanes, workers = values
    if lanes <= 0 or not 0 <= lane < lanes or workers <= 0:
        raise ValueError("DP resume requires valid lanes and positive process workers")
    if contract.get("micro_batch_size") != 1 or type(contract.get("micro_batch_size")) is not int:
        raise ValueError("DP resume requires micro-batch size 1")
    total_workers = lanes * workers
    logical_workers = contract.get("logical_workers")
    if logical_workers is not None and (
        type(logical_workers) is not int or logical_workers != total_workers
    ):
        raise ValueError("DP resume requires one logical partition per physical worker (L=P)")
    return lane, lanes, workers


def _sampler_slice_fingerprint(
    descriptor: Any, *, sampler_path: tuple[Any, ...], worker: int
) -> tuple[str, int]:
    if isinstance(descriptor, list):
        if not descriptor or any(type(offset) is not int or offset < 0 for offset in descriptor):
            raise ValueError("invalid native sampler slice offsets")
        if any(start > stop for start, stop in zip(descriptor, descriptor[1:])):
            raise ValueError("unordered native sampler slice offsets")
        start, stop = descriptor[0], descriptor[-1]
    elif isinstance(descriptor, dict) and set(descriptor) == {"type", "start", "stop", "step"}:
        if (
            descriptor["type"] != "range"
            or any(type(descriptor[key]) is not int for key in ("start", "stop", "step"))
            or descriptor["start"] < 0
            or descriptor["step"] <= 0
        ):
            raise ValueError("invalid native sampler range descriptor")
        offsets = range(descriptor["start"], descriptor["stop"], descriptor["step"])
        if not offsets:
            raise ValueError("empty native sampler range descriptor")
        start, stop = offsets[0], offsets[-1]
    else:
        raise ValueError("unknown native sampler partition descriptor")
    return _digest({"sampler_path": sampler_path, "slice_offsets": descriptor, "worker": worker}), (
        stop - start
    )


def build_rescale_profile(
    contract: dict[str, Any], pipeline: dict[str, Any], global_batch_size: int
) -> EnergonRescaleProfile:
    """Fingerprint semantics and source partitions across rank/worker regrouping.

    Only explicitly recognized native pipeline edges are traversed to remove
    placement. Other configuration, including factory and cooker fields with
    familiar key names, remains intact in the semantic digest.
    """
    lane, lanes, workers = _placement(contract)
    total_workers = lanes * workers
    if type(global_batch_size) is not int or global_batch_size <= 0:
        raise ValueError("DP resume requires a positive integer global batch size")
    if global_batch_size % total_workers:
        raise ValueError("global batch size must be divisible by total physical workers")
    if not isinstance(pipeline, dict):
        raise ValueError("DP resume requires a native loader configuration dictionary")

    worker_partitions: list[list[dict[str, Any]]] = [[] for _ in range(workers)]
    worker_owners = {
        "BatchDataset",
        "BlendDataset",
        "ConcatDataset",
        "DatasetSampler",
        "EpochizeDataset",
        "LimitDataset",
        "RepeatDataset",
        "ShuffleBufferDataset",
        "StreamingPackingDataset",
        "StrideDataset",
    }
    single_child = {
        "SavableDataLoader",
        "BatchDataset",
        "EpochizeDataset",
        "LimitDataset",
        "RepeatDataset",
        "ShuffleBufferDataset",
        "StreamingPackingDataset",
        "StrideDataset",
        "MapDataset",
        "IterMapDataset",
        "FilterDataset",
        "GcDataset",
        "LogDataset",
    }

    def visit(node: Any, path: tuple[Any, ...]) -> dict[str, Any]:
        if not isinstance(node, dict) or not isinstance(node.get("type"), str):
            raise ValueError("DP resume requires explicit native pipeline component configuration")
        kind = node["type"]
        if (not path) != (kind == "SavableDataLoader"):
            raise ValueError("SavableDataLoader must appear exactly at the pipeline root")
        result = dict(node)
        if kind == "SavableDataLoader":
            if type(node.get("num_workers")) is not int or node["num_workers"] != workers:
                raise ValueError("native loader and bridge worker counts disagree")
            result.pop("num_workers")
        if "worker_config" in node:
            if kind not in worker_owners:
                raise ValueError(f"unsupported worker-placement component: {kind}")
            config = node["worker_config"]
            expected = {"rank": lane, "world_size": lanes, "num_workers": workers}
            if not isinstance(config, dict) or any(
                type(config.get(key)) is not int or config[key] != value
                for key, value in expected.items()
            ):
                raise ValueError("native WorkerConfig disagrees with bridge placement")
            logical = config.get("logical_workers")
            if logical is not None and (type(logical) is not int or logical != total_workers):
                raise ValueError("DP resume requires one logical partition per physical worker")
            normalized = {key: value for key, value in config.items() if key not in expected}
            normalized["logical_workers"] = total_workers
            result["worker_config"] = normalized
        if kind == "StrideDataset" and node.get("assignment") is not None:
            raise ValueError("DP resume does not support explicit stride assignments")
        if kind == "DatasetSampler":
            if "worker_config" not in node:
                raise ValueError("native DatasetSampler is missing its WorkerConfig")
            slices = node.get("slice_offsets")
            if not isinstance(slices, list) or len(slices) != workers:
                raise ValueError("native sampler partition count disagrees with workers")
            local_count = 0
            for worker, descriptor in enumerate(slices):
                fingerprint, count = _sampler_slice_fingerprint(
                    descriptor, sampler_path=path, worker=lane * workers + worker
                )
                local_count += count
                worker_partitions[worker].append(
                    {"sampler_path": path, "partition_sha256": fingerprint}
                )
            if type(node.get("len")) is not int or node["len"] != local_count:
                raise ValueError("native sampler length disagrees with its partitions")
            result.pop("len")
            result.pop("slice_offsets")
        if kind in single_child:
            if "dataset" not in node:
                raise ValueError(f"missing native pipeline child for {kind}")
            result["dataset"] = visit(node["dataset"], path + ("dataset",))
        if kind == "ConcatDataset":
            children = node.get("datasets")
            if not isinstance(children, (list, tuple)):
                raise ValueError("invalid native concatenation configuration")
            result["datasets"] = [
                visit(child, path + ("datasets", index)) for index, child in enumerate(children)
            ]
        if kind == "BlendDataset":
            pairs = node.get("dataset_weights")
            if not isinstance(pairs, (list, tuple)):
                raise ValueError("unsupported native weighted pipeline component")
            normalized_pairs = []
            for index, pair in enumerate(pairs):
                if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                    raise ValueError("invalid native blend dataset/weight pair")
                normalized_pairs.append(
                    (visit(pair[0], path + ("dataset_weights", index, 0)), pair[1])
                )
            result["dataset_weights"] = normalized_pairs
        return result

    normalized_pipeline = visit(pipeline, ())
    if any(not partitions for partitions in worker_partitions):
        raise ValueError("DP resume requires explicit native DatasetSampler partitions")
    policy = {
        key: value
        for key, value in contract.items()
        if key not in {"lane", "language_lanes", "num_workers", "native_config_sha256"}
    }
    semantic_sha256 = _digest(
        {
            "policy": policy,
            "pipeline": normalized_pipeline,
            "global_batch_size": global_batch_size,
            "total_workers": total_workers,
        }
    )
    return EnergonRescaleProfile(
        global_batch_size=global_batch_size,
        semantic_sha256=semantic_sha256,
        worker_partitions=tuple(
            _digest({"global_worker": lane * workers + worker, "partitions": partitions})
            for worker, partitions in enumerate(worker_partitions)
        ),
        pipeline_config=deepcopy(pipeline),
    )


def _read_profile(state: dict[str, Any]) -> EnergonRescaleProfile:
    if not isinstance(state, dict) or type(state.get("format")) is not int or state["format"] != 2:
        raise ValueError("DP resume requires a format-2 dataloader checkpoint")
    profile = state.get("rescale")
    expected_keys = {"global_batch_size", "semantic_sha256", "worker_partitions", "pipeline_config"}
    if not isinstance(profile, dict) or set(profile) != expected_keys:
        raise ValueError("missing or malformed Energon rescale profile")
    contract = state.get("contract")
    if not isinstance(contract, dict) or _native_config_digest(
        profile["pipeline_config"]
    ) != contract.get("native_config_sha256"):
        raise ValueError("stored native pipeline config does not match the strict checkpoint hash")
    expected = build_rescale_profile(
        contract, profile["pipeline_config"], profile["global_batch_size"]
    )
    if _digest(profile) != _digest(asdict(expected)):
        raise ValueError("checkpoint semantic or worker partition profile is inconsistent")
    return expected


def _worker_progress(state: Any, workers: int) -> list[int]:
    from megatron.energon.savable_loader import (
        SavableDataLoaderState,
        SavableDatasetCheckpoint,
        SavableDatasetState,
    )

    if not isinstance(state, SavableDataLoaderState) or not isinstance(state.worker_states, list):
        raise ValueError("missing native per-rank worker state")
    if len(state.worker_states) != workers:
        raise ValueError("native worker count disagrees with checkpoint placement")
    if type(state.micro_batch_size) is not int or state.micro_batch_size != 1:
        raise ValueError("DP resume requires native micro-batch size 1")
    if type(state.next_worker_id) is not int or not 0 <= state.next_worker_id < workers:
        raise ValueError("invalid native next-worker cursor")
    progress = []
    for checkpoint in state.worker_states:
        if not isinstance(checkpoint, SavableDatasetCheckpoint):
            raise ValueError("DP resume requires native process-worker checkpoints")
        if type(checkpoint.offset) is not int or checkpoint.offset < 0:
            raise ValueError("invalid native committed worker replay offset")
        sample_index = 0
        if checkpoint.state is not None:
            if not isinstance(checkpoint.state, SavableDatasetState):
                raise ValueError("invalid native dataset checkpoint state")
            sample_index = checkpoint.state.sample_index
            if type(sample_index) is not int or sample_index < 0:
                raise ValueError("invalid native committed worker sample index")
        progress.append(sample_index + checkpoint.offset)
    return progress


def _global_boundary(
    states: list[Any], workers: int, total_workers: int, global_batch_size: int
) -> int:
    progress = []
    for state in states:
        progress.extend(_worker_progress(state, workers))
        if state.next_worker_id != 0:
            raise ValueError("DP resume requires next_worker_id=0 on every source lane")
    if not progress or len(set(progress)) != 1:
        raise ValueError("source workers do not share the same committed progress")
    consumed_samples = progress[0] * total_workers
    if consumed_samples % global_batch_size:
        raise ValueError("checkpoint is not at a complete committed global-batch boundary")
    return consumed_samples // global_batch_size


def redistribute_energon_states(
    states: list[dict[str, Any]], new_language_lanes: int
) -> list[dict[str, Any]]:
    """Regroup opaque native worker checkpoints across a fixed total P."""
    from megatron.energon.savable_loader import SavableDataLoaderState

    if not states:
        raise ValueError("all source language-lane checkpoints are required")
    profiles = [_read_profile(state) for state in states]
    if any("migration" in state for state in states):
        raise ValueError("restore and save a migrated checkpoint before redistributing it again")
    _, source_lanes, workers = _placement(states[0]["contract"])
    placements = [_placement(state.get("contract")) for state in states]
    if len(states) != source_lanes or sorted(lane for lane, _, _ in placements) != list(
        range(source_lanes)
    ):
        raise ValueError("source language lanes must be complete and unique")
    if any((lanes, count) != (source_lanes, workers) for _, lanes, count in placements):
        raise ValueError("source checkpoint placements disagree")
    if any(profile.global_batch_size != profiles[0].global_batch_size for profile in profiles):
        raise ValueError("source global batch sizes disagree")
    if any(profile.semantic_sha256 != profiles[0].semantic_sha256 for profile in profiles):
        raise ValueError("source semantic contracts disagree")
    ordered = sorted(zip(states, profiles), key=lambda item: item[0]["contract"]["lane"])
    ordered_states = [state for state, _ in ordered]
    ordered_profiles = [profile for _, profile in ordered]
    native_states = [state.get("energon") for state in ordered_states]
    total_workers = source_lanes * workers
    completed_batches = _global_boundary(
        native_states, workers, total_workers, ordered_profiles[0].global_batch_size
    )
    if type(new_language_lanes) is not int or new_language_lanes <= 0:
        raise ValueError("new language lane count must be a positive integer")
    if total_workers % new_language_lanes:
        raise ValueError("new language lane count must divide the fixed physical worker count")
    new_workers = total_workers // new_language_lanes
    native_workers = [
        checkpoint for native_state in native_states for checkpoint in native_state.worker_states
    ]
    redistributed = [
        SavableDataLoaderState(
            worker_states=deepcopy(native_workers[start : start + new_workers]),
            next_worker_id=0,
            micro_batch_size=1,
        )
        for start in range(0, total_workers, new_workers)
    ]
    partitions = [
        partition for profile in ordered_profiles for partition in profile.worker_partitions
    ]
    source_contract_digests = [_digest(state["contract"]) for state in ordered_states]
    result = []
    for lane, native_state in enumerate(redistributed):
        origin_index = lane * new_workers // workers
        origin_state = ordered_states[origin_index]
        start = lane * new_workers
        result.append(
            {
                "format": 2,
                "contract": deepcopy(origin_state["contract"]),
                "rescale": asdict(ordered_profiles[origin_index]),
                "energon": native_state,
                "migration": {
                    "target": {
                        "lane": lane,
                        "language_lanes": new_language_lanes,
                        "num_workers": new_workers,
                        "micro_batch_size": 1,
                    },
                    "semantic_sha256": profiles[0].semantic_sha256,
                    "worker_partitions": partitions[start : start + new_workers],
                    "completed_global_batches": completed_batches,
                    "source_contracts": source_contract_digests,
                },
            }
        )
    return result


def validate_rescale_restore(
    state: dict[str, Any], contract: dict[str, Any], profile: EnergonRescaleProfile | None
) -> None:
    """Validate current topology and semantics before restoring native state."""
    source_profile = _read_profile(state)
    if profile is None:
        raise ValueError("enable --energon-dp-resume with positive workers to restore format 2")
    lane, lanes, workers = _placement(contract)
    native_state = state.get("energon")
    _worker_progress(native_state, workers)
    if "migration" not in state:
        if state["contract"] != contract or source_profile != profile:
            raise ValueError("Energon dataloader contract changed")
        return

    migration = state["migration"]
    expected_keys = {
        "target",
        "semantic_sha256",
        "worker_partitions",
        "completed_global_batches",
        "source_contracts",
    }
    if not isinstance(migration, dict) or set(migration) != expected_keys:
        raise ValueError("malformed explicit Energon checkpoint migration")
    if _placement(migration["target"]) != (lane, lanes, workers):
        raise ValueError("migrated checkpoint destination placement disagrees with loader")
    if (
        source_profile.global_batch_size != profile.global_batch_size
        or source_profile.semantic_sha256 != profile.semantic_sha256
        or migration["semantic_sha256"] != profile.semantic_sha256
    ):
        raise ValueError("migrated checkpoint semantic contract changed")
    if (
        not isinstance(migration["worker_partitions"], (list, tuple))
        or tuple(migration["worker_partitions"]) != profile.worker_partitions
    ):
        raise ValueError("migrated checkpoint worker partitions disagree with loader")
    origins = migration["source_contracts"]
    if (
        not isinstance(origins, (list, tuple))
        or len(origins) != state["contract"].get("language_lanes")
        or any(not isinstance(origin, str) or len(origin) != 64 for origin in origins)
        or _digest(state["contract"]) not in origins
    ):
        raise ValueError("migration source contract provenance is inconsistent")
    completed_batches = _global_boundary(
        [native_state], workers, lanes * workers, profile.global_batch_size
    )
    if (
        type(migration["completed_global_batches"]) is not int
        or migration["completed_global_batches"] != completed_batches
    ):
        raise ValueError("migration committed progress is inconsistent")


def redistribute_checkpoint_files(
    input_paths: list[Path], output_directory: Path, new_language_lanes: int
) -> None:
    """Convert explicit dataloader files into a new directory, never in place."""
    if output_directory.exists() or output_directory.is_symlink():
        raise ValueError("output directory must be new; source checkpoints are never overwritten")
    if not input_paths or any(not path.is_file() for path in input_paths):
        raise ValueError("pass individual source dataloader checkpoint files")
    states = []
    for path in input_paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or set(payload) != {"dataloader_state_dict"}:
            raise ValueError("expected a dataloader-only checkpoint wrapper")
        states.append(payload["dataloader_state_dict"])
    targets = redistribute_energon_states(states, new_language_lanes)
    output_directory.mkdir(parents=True, exist_ok=False)
    for lane, state in enumerate(targets):
        path = output_directory / f"train_dataloader_dprank{lane:03d}.pt"
        with path.open("xb") as stream:
            torch.save({"dataloader_state_dict": state}, stream)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Redistribute explicit Energon dataloader checkpoints across a fixed P."
    )
    parser.add_argument(
        "--input-dataloader-checkpoints",
        type=Path,
        nargs="+",
        required=True,
        metavar="FILE.pt",
        help="Individual source language-lane dataloader files from one committed iteration.",
    )
    parser.add_argument(
        "--new-language-lanes",
        type=int,
        required=True,
        metavar="N",
        help="Target number of language sample lanes; physical worker total remains fixed.",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        required=True,
        metavar="DIR",
        help="New output directory. Source checkpoint files are never overwritten.",
    )
    args = parser.parse_args()
    try:
        redistribute_checkpoint_files(
            args.input_dataloader_checkpoints, args.output_directory, args.new_language_lanes
        )
    except (OSError, TypeError, ValueError) as error:
        parser.error(str(error))
    print(f"Wrote fixed-worker-count Energon checkpoints to {args.output_directory}")


if __name__ == "__main__":
    main()
