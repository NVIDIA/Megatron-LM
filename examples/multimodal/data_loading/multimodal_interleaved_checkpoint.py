# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Explicit worker-preserving checkpoint migration for interleaved data."""

import hashlib
import json
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any

from megatron.energon.savable_loader import (
    SavableDataLoaderState,
    SavableDatasetCheckpoint,
    SavableDatasetState,
)


@dataclass(frozen=True)
class InterleavedRescaleProfile:
    """Protected semantics and ordered worker partitions for explicit migration."""

    global_batch_size: int
    semantic_sha256: str
    worker_partitions: tuple[str, ...]
    pipeline_config: dict[str, Any]


def _digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _placement(contract: dict) -> tuple[int, int, int]:
    if not isinstance(contract, dict):
        raise ValueError("checkpoint contract must be a dictionary")
    values = tuple(contract.get(name) for name in ("lane", "lanes", "workers"))
    if any(type(value) is not int for value in values):
        raise ValueError("checkpoint placement must contain integer lane/lanes/workers")
    lane, lanes, workers = values
    if not 0 <= lane < lanes or workers <= 0:
        raise ValueError("DP resizing requires positive process workers, not workers0")
    return lane, lanes, workers


def build_rescale_profile(
    contract: dict, pipeline: dict, global_batch_size: int
) -> InterleavedRescaleProfile:
    """Fingerprint policy separately from explicitly enumerated worker placement.

    Only native pipeline edges are traversed. Factory/cooker arguments, tags and
    other opaque configuration stay intact even if they contain similarly named
    keys. Worker partitions include global worker IDs and ordered sampler paths.
    """
    lane, lanes, workers = _placement(contract)
    total_workers = lanes * workers
    if type(global_batch_size) is not int or global_batch_size <= 0:
        raise ValueError("resizing requires a positive integer global batch size")
    if global_batch_size % total_workers:
        raise ValueError("global batch size must be divisible by the total workers for MBS1")
    worker_partitions = [[] for _ in range(workers)]
    owners = {
        "BatchDataset",
        "BlendDataset",
        "ConcatDataset",
        "DatasetSampler",
        "EpochizeDataset",
        "LimitDataset",
        "RepeatDataset",
        "ShuffleBufferDataset",
        "StreamingPackingDataset",
    }
    single_child = {
        "SavableDataLoader",
        "BatchDataset",
        "EpochizeDataset",
        "LimitDataset",
        "RepeatDataset",
        "ShuffleBufferDataset",
        "StreamingPackingDataset",
        "MapDataset",
        "IterMapDataset",
        "FilterDataset",
        "GcDataset",
        "LogDataset",
    }

    def visit(node, path):
        if not isinstance(node, dict) or not isinstance(node.get("type"), str):
            raise ValueError("rescaling requires explicit native pipeline component configuration")
        result = dict(node)
        kind = node["type"]
        if (not path) != (kind == "SavableDataLoader"):
            raise ValueError("SavableDataLoader must appear exactly at the pipeline root")
        if kind == "SavableDataLoader":
            if type(node.get("num_workers")) is not int or node["num_workers"] != workers:
                raise ValueError("loader and checkpoint worker counts disagree")
            result.pop("num_workers")
        if "worker_config" in node:
            if kind not in owners:
                raise ValueError(f"unsupported worker-placement component for resizing: {kind}")
            config = node["worker_config"]
            expected = {"rank": lane, "world_size": lanes, "num_workers": workers}
            if not isinstance(config, dict) or any(
                type(config.get(key)) is not int or config[key] != value
                for key, value in expected.items()
            ):
                raise ValueError("pipeline WorkerConfig disagrees with checkpoint placement")
            result["worker_config"] = {
                key: value for key, value in config.items() if key not in expected
            }
        if kind == "DatasetSampler":
            slices = node.get("slice_offsets")
            if not isinstance(slices, list) or len(slices) != workers:
                raise ValueError("sampler partition count disagrees with process workers")
            local_count = 0
            for worker, descriptor in enumerate(slices):
                if isinstance(descriptor, list):
                    if not descriptor or any(
                        type(offset) is not int or offset < 0 for offset in descriptor
                    ):
                        raise ValueError("invalid sampler slice offsets")
                    if any(start > stop for start, stop in zip(descriptor, descriptor[1:])):
                        raise ValueError("unordered sampler slice offsets")
                    start, stop = descriptor[0], descriptor[-1]
                elif isinstance(descriptor, dict) and set(descriptor) == {
                    "type",
                    "start",
                    "stop",
                    "step",
                }:
                    if (
                        descriptor["type"] != "range"
                        or any(
                            type(descriptor[key]) is not int for key in ("start", "stop", "step")
                        )
                        or descriptor["start"] < 0
                        or descriptor["step"] <= 0
                    ):
                        raise ValueError("invalid sampler range descriptor")
                    offsets = range(descriptor["start"], descriptor["stop"], descriptor["step"])
                    if not offsets:
                        raise ValueError("empty sampler range descriptor")
                    start, stop = offsets[0], offsets[-1]
                else:
                    raise ValueError("unknown sampler partition descriptor")
                local_count += stop - start
                worker_partitions[worker].append(
                    {"sampler_path": path, "slice_offsets": descriptor}
                )
            if type(node.get("len")) is not int or node["len"] != local_count:
                raise ValueError("sampler length disagrees with its worker partitions")
            result.pop("len")
            result.pop("slice_offsets")
        if kind in single_child:
            if "dataset" not in node:
                raise ValueError(f"missing native pipeline child for {kind}")
            result["dataset"] = visit(node["dataset"], path + ("dataset",))
        if kind == "ConcatDataset":
            if not isinstance(node.get("datasets"), (list, tuple)):
                raise ValueError("invalid native datasets configuration")
            result["datasets"] = [
                visit(child, path + ("datasets", index))
                for index, child in enumerate(node["datasets"])
            ]
        if kind == "BlendDataset":
            if not isinstance(node.get("dataset_weights"), (list, tuple)):
                raise ValueError("unsupported weighted pipeline component")
            pairs = []
            for index, pair in enumerate(node["dataset_weights"]):
                if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                    raise ValueError("invalid blend dataset/weight pair")
                pairs.append((visit(pair[0], path + ("dataset_weights", index, 0)), pair[1]))
            result["dataset_weights"] = pairs
        return result

    semantic_pipeline = visit(pipeline, ())
    if any(not partition for partition in worker_partitions):
        raise ValueError("resizing requires explicit DatasetSampler worker partitions")
    policy = {
        key: value
        for key, value in contract.items()
        if key not in {"lane", "lanes", "workers", "resolved_pipeline_sha256"}
    }
    return InterleavedRescaleProfile(
        global_batch_size=global_batch_size,
        semantic_sha256=_digest(
            {
                "policy": policy,
                "pipeline": semantic_pipeline,
                "global_batch_size": global_batch_size,
                "total_workers": total_workers,
            }
        ),
        worker_partitions=tuple(
            _digest({"global_worker": lane * workers + worker, "partitions": partition})
            for worker, partition in enumerate(worker_partitions)
        ),
        pipeline_config=pipeline,
    )


def _read_profile(state: dict) -> InterleavedRescaleProfile:
    if not isinstance(state, dict) or type(state.get("format")) is not int or state["format"] != 2:
        raise ValueError("DP resizing requires a rescale-capable format2 checkpoint")
    profile = state.get("rescale")
    if not isinstance(profile, dict) or set(profile) != {
        "global_batch_size",
        "semantic_sha256",
        "worker_partitions",
        "pipeline_config",
    }:
        raise ValueError("missing or malformed checkpoint rescale profile")
    contract = state.get("contract")
    if not isinstance(contract, dict) or _digest(profile["pipeline_config"]) != contract.get(
        "resolved_pipeline_sha256"
    ):
        raise ValueError("stored pipeline configuration does not match the checkpoint contract")
    expected = build_rescale_profile(
        contract, profile["pipeline_config"], profile["global_batch_size"]
    )
    if _digest(profile) != _digest(asdict(expected)):
        raise ValueError("checkpoint semantic or worker partition profile is inconsistent")
    return expected


def _worker_progress(state, workers: int) -> list[int]:
    if not isinstance(state, SavableDataLoaderState) or not isinstance(state.worker_states, list):
        raise ValueError("missing native process-worker rank state")
    if len(state.worker_states) != workers:
        raise ValueError("native worker count disagrees with checkpoint placement")
    if type(state.micro_batch_size) is not int or state.micro_batch_size != 1:
        raise ValueError("interleaved resizing requires native microbatch size1")
    if type(state.next_worker_id) is not int or not 0 <= state.next_worker_id < workers:
        raise ValueError("invalid native next-worker cursor")
    result = []
    for checkpoint in state.worker_states:
        if not isinstance(checkpoint, SavableDatasetCheckpoint):
            raise ValueError("resizing does not convert synchronous workers0 checkpoints")
        if type(checkpoint.offset) is not int or checkpoint.offset < 0:
            raise ValueError("invalid committed worker replay offset")
        index = 0
        if checkpoint.state is not None:
            if not isinstance(checkpoint.state, SavableDatasetState):
                raise ValueError("invalid native worker checkpoint state")
            index = checkpoint.state.sample_index
            if type(index) is not int or index < 0:
                raise ValueError("invalid committed worker sample index")
        result.append(index + checkpoint.offset)
    return result


def _global_boundary(states, workers: int, total_workers: int, global_batch_size: int) -> int:
    progress = []
    for state in states:
        progress.extend(_worker_progress(state, workers))
        if state.next_worker_id != 0:
            raise ValueError("DP resizing requires next_worker_id=0 on every rank")
    if len(set(progress)) != 1:
        raise ValueError("source workers do not describe the same committed progress")
    samples = progress[0] * total_workers
    if samples % global_batch_size:
        raise ValueError("checkpoint is not at a complete committed global-batch boundary")
    return samples // global_batch_size


def redistribute_interleaved_states(states: list[dict], new_lanes: int) -> list[dict]:
    """Validate outer contracts and delegate worker redistribution to Energon.

    Migrated checkpoints must be restored and saved normally before another
    migration: only the actual destination loader can establish its full config.
    """
    from megatron.energon import redistribute_loader_states

    if not states:
        raise ValueError("all source language-lane checkpoints are required")
    profiles = [_read_profile(state) for state in states]
    if any("migration" in state for state in states):
        raise ValueError("restore and save a migrated checkpoint before resizing it again")
    _, lanes, workers = _placement(states[0]["contract"])
    placements = [_placement(state["contract"]) for state in states]
    if len(states) != lanes or sorted(lane for lane, _, _ in placements) != list(range(lanes)):
        raise ValueError("source language lanes must be complete and unique")
    if any((size, count) != (lanes, workers) for _, size, count in placements):
        raise ValueError("source checkpoint placements disagree")
    if any(profile.global_batch_size != profiles[0].global_batch_size for profile in profiles):
        raise ValueError("source global batch sizes disagree")
    if any(profile.semantic_sha256 != profiles[0].semantic_sha256 for profile in profiles):
        raise ValueError("source semantic contracts disagree")
    ordered = sorted(zip(states, profiles), key=lambda item: item[0]["contract"]["lane"])
    states, profiles = map(list, zip(*ordered))
    native = [state.get("energon") for state in states]
    completed = _global_boundary(native, workers, lanes * workers, profiles[0].global_batch_size)
    redistributed = redistribute_loader_states(native, new_lanes)
    new_workers = lanes * workers // new_lanes
    partitions = [partition for profile in profiles for partition in profile.worker_partitions]
    origins = [_digest(state["contract"]) for state in states]
    result = []
    for lane, native_state in enumerate(redistributed):
        origin = states[lane * new_workers // workers]
        result.append(
            {
                "format": 2,
                "contract": deepcopy(origin["contract"]),
                "rescale": deepcopy(origin["rescale"]),
                "energon": native_state,
                "migration": {
                    "target": {"lane": lane, "lanes": new_lanes, "workers": new_workers},
                    "semantic_sha256": profiles[0].semantic_sha256,
                    "worker_partitions": partitions[lane * new_workers : (lane + 1) * new_workers],
                    "completed_global_batches": completed,
                    "source_contracts": origins,
                },
            }
        )
    return result


def validate_rescale_restore(
    state: dict, contract: dict, profile: InterleavedRescaleProfile | None
) -> None:
    """Validate actual destination identity before native restore or first yield."""
    source = _read_profile(state)
    if profile is None:
        raise ValueError("configure positive workers and global batch size to restore format2")
    lane, lanes, workers = _placement(contract)
    native = state.get("energon")
    _worker_progress(native, workers)
    if "migration" not in state:
        if state["contract"] != contract or source != profile:
            raise ValueError("interleaved dataloader contract changed")
        return
    migration = state["migration"]
    if not isinstance(migration, dict) or set(migration) != {
        "target",
        "semantic_sha256",
        "worker_partitions",
        "completed_global_batches",
        "source_contracts",
    }:
        raise ValueError("malformed explicit checkpoint migration")
    if _placement(migration["target"]) != (lane, lanes, workers):
        raise ValueError("migrated checkpoint destination placement disagrees with loader")
    if (
        source.global_batch_size != profile.global_batch_size
        or source.semantic_sha256 != profile.semantic_sha256
    ):
        raise ValueError("migrated checkpoint semantic contract changed")
    if migration["semantic_sha256"] != profile.semantic_sha256:
        raise ValueError("migration semantic identity is inconsistent")
    if (
        not isinstance(migration["worker_partitions"], (list, tuple))
        or tuple(migration["worker_partitions"]) != profile.worker_partitions
    ):
        raise ValueError("migrated checkpoint worker partitions disagree with loader")
    origins = migration["source_contracts"]
    if (
        not isinstance(origins, (list, tuple))
        or len(origins) != state["contract"]["lanes"]
        or any(not isinstance(value, str) or len(value) != 64 for value in origins)
        or _digest(state["contract"]) not in origins
    ):
        raise ValueError("migration source provenance is inconsistent")
    completed = _global_boundary([native], workers, lanes * workers, profile.global_batch_size)
    if (
        type(migration["completed_global_batches"]) is not int
        or migration["completed_global_batches"] != completed
    ):
        raise ValueError("migration committed progress is inconsistent")
