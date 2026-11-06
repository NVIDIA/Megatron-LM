# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Fixed-total-worker Energon checkpoint profile and regrouping checks."""

import json
from copy import deepcopy
from hashlib import sha256

import pytest
import torch

from examples.mimo.training.energon_checkpoint import (
    build_rescale_profile,
    redistribute_checkpoint_files,
    redistribute_energon_states,
    validate_rescale_restore,
)
from megatron.energon.rng import SystemRngState
from megatron.energon.savable_loader import (
    SavableDataLoaderState,
    SavableDatasetCheckpoint,
    SavableDatasetState,
)

GLOBAL_BATCH_SIZE = 8
TOTAL_WORKERS = 4


def _pipeline(lane, lanes, workers, *, seed=17, opaque_worker_config=None):
    start_worker = lane * workers
    slices = [
        [(start_worker + worker) * 10, (start_worker + worker + 1) * 10]
        for worker in range(workers)
    ]
    return {
        "type": "SavableDataLoader",
        "num_workers": workers,
        "dataset": {
            "type": "DatasetSampler",
            "worker_config": {
                "rank": lane,
                "world_size": lanes,
                "num_workers": workers,
                "logical_workers": lanes * workers,
                "factory_args": {"worker_config": opaque_worker_config or "opaque"},
            },
            "slice_offsets": slices,
            "len": workers * 10,
            "seed": seed,
        },
        "loader_tag": {"dataset": {"len": 10}, "worker_config": {"rank": 99}},
    }


def _contract(lane, lanes, workers, *, seed=17, identity="task-v1"):
    pipeline = _pipeline(lane, lanes, workers, seed=seed)
    native_hash = sha256(json.dumps(pipeline, sort_keys=True).encode("utf-8")).hexdigest()
    return {
        "application": "tests.fixture:make_config",
        "checkpoint_identity": identity,
        "native_config_sha256": native_hash,
        "energon_version": "test-version",
        "lane": lane,
        "language_lanes": lanes,
        "num_workers": workers,
        "micro_batch_size": 1,
        "seed": 23,
        "logical_workers": lanes * workers,
    }, pipeline


def _worker_state(worker_id, progress=2):
    native = SavableDatasetState(
        rng=SystemRngState(
            torch=torch.tensor([worker_id]), numpy={"worker": worker_id}, random=(worker_id,)
        ),
        dataset_state={"worker_id": worker_id},
        sample_index=progress,
    )
    return SavableDatasetCheckpoint(state=native, offset=0)


def _normal_state(lane, lanes, workers, worker_states, *, seed=17, identity="task-v1"):
    contract, pipeline = _contract(lane, lanes, workers, seed=seed, identity=identity)
    profile = build_rescale_profile(contract, pipeline, GLOBAL_BATCH_SIZE)
    return {
        "format": 2,
        "contract": contract,
        "rescale": {
            "global_batch_size": profile.global_batch_size,
            "semantic_sha256": profile.semantic_sha256,
            "worker_partitions": profile.worker_partitions,
            "pipeline_config": profile.pipeline_config,
        },
        "energon": SavableDataLoaderState(
            worker_states=deepcopy(worker_states), next_worker_id=0, micro_batch_size=1
        ),
    }


def _initial_states(lanes=2, workers=2):
    states = []
    for lane in range(lanes):
        worker_states = [_worker_state(lane * workers + worker) for worker in range(workers)]
        states.append(_normal_state(lane, lanes, workers, worker_states))
    return states


def _ids(state):
    return [
        checkpoint.state.dataset_state["worker_id"] for checkpoint in state["energon"].worker_states
    ]


def test_fixed_worker_dp_regrouping_preserves_global_worker_streams_and_profile():
    states = _initial_states()
    expected_ids = list(range(TOTAL_WORKERS))

    for lanes, workers in ((1, 4), (4, 1), (2, 2)):
        migrated = redistribute_energon_states(list(reversed(states)), lanes)
        assert [_ids(state) for state in migrated] == [
            expected_ids[lane * workers : (lane + 1) * workers] for lane in range(lanes)
        ]
        assert all(
            state["migration"]["target"]
            == {
                "lane": lane,
                "language_lanes": lanes,
                "num_workers": workers,
                "micro_batch_size": 1,
            }
            for lane, state in enumerate(migrated)
        )
        assert all(state["migration"]["completed_global_batches"] == 1 for state in migrated)

        for lane, state in enumerate(migrated):
            contract, pipeline = _contract(lane, lanes, workers)
            profile = build_rescale_profile(contract, pipeline, GLOBAL_BATCH_SIZE)
            validate_rescale_restore(state, contract, profile)

        # A destination's ordinary save establishes its actual current config
        # and makes it eligible for the next explicit migration.
        states = [
            _normal_state(lane, lanes, workers, state["energon"].worker_states)
            for lane, state in enumerate(migrated)
        ]

    assert [
        checkpoint.state.dataset_state["worker_id"]
        for state in states
        for checkpoint in state["energon"].worker_states
    ] == expected_ids
    assert [
        checkpoint.state.sample_index
        for state in states
        for checkpoint in state["energon"].worker_states
    ] == [2] * TOTAL_WORKERS
    assert all(
        state["contract"]["native_config_sha256"]
        == _contract(
            state["contract"]["lane"],
            state["contract"]["language_lanes"],
            state["contract"]["num_workers"],
        )[0]["native_config_sha256"]
        for state in states
    )


def test_profile_changes_on_semantics_and_partitions_but_not_known_placement():
    contract, pipeline = _contract(0, 2, 2)
    baseline = build_rescale_profile(contract, pipeline, GLOBAL_BATCH_SIZE)
    other_lane_contract, other_lane_pipeline = _contract(1, 2, 2)
    other_lane = build_rescale_profile(other_lane_contract, other_lane_pipeline, GLOBAL_BATCH_SIZE)
    assert baseline.semantic_sha256 == other_lane.semantic_sha256
    assert baseline.worker_partitions != other_lane.worker_partitions

    changed = deepcopy(pipeline)
    changed["dataset"]["worker_config"]["factory_args"]["worker_config"] = {"rank": 1}
    assert (
        build_rescale_profile(contract, changed, GLOBAL_BATCH_SIZE).semantic_sha256
        != baseline.semantic_sha256
    )
    changed_contract, changed_pipeline = _contract(0, 2, 2, identity="changed-task")
    assert (
        build_rescale_profile(changed_contract, changed_pipeline, GLOBAL_BATCH_SIZE).semantic_sha256
        != baseline.semantic_sha256
    )


@pytest.mark.parametrize(
    "mutation, message",
    [
        ("missing_lane", "complete and unique"),
        ("duplicate_lane", "complete and unique"),
        ("worker_drift", "committed progress"),
        ("cursor", "next_worker_id=0"),
    ],
)
def test_migration_rejects_incomplete_or_unaligned_source_state(mutation, message):
    states = _initial_states()
    if mutation == "missing_lane":
        states = states[:1]
    elif mutation == "duplicate_lane":
        states[1] = deepcopy(states[0])
    elif mutation == "worker_drift":
        states[1]["energon"].worker_states[0].state.sample_index += 1
    elif mutation == "cursor":
        states[1]["energon"].next_worker_id = 1
    with pytest.raises(ValueError, match=message):
        redistribute_energon_states(states, 1)


def test_migration_rejects_changed_policy_native_hash_and_global_boundary():
    states = _initial_states()
    changed = deepcopy(states)
    changed[1]["contract"]["seed"] += 1
    with pytest.raises(ValueError, match="profile is inconsistent"):
        redistribute_energon_states(changed, 1)

    changed = deepcopy(states)
    changed[0]["rescale"]["pipeline_config"]["dataset"]["seed"] += 1
    with pytest.raises(ValueError, match="strict checkpoint hash"):
        redistribute_energon_states(changed, 1)

    changed = deepcopy(states)
    changed[1]["energon"].worker_states[0].state.sample_index = 3
    with pytest.raises(ValueError, match="committed progress"):
        redistribute_energon_states(changed, 1)

    unaligned = _initial_states()
    for state in unaligned:
        for worker in state["energon"].worker_states:
            worker.state.sample_index = 1
    with pytest.raises(ValueError, match="complete committed global-batch boundary"):
        redistribute_energon_states(unaligned, 1)


def test_destination_requires_exact_migration_placement_semantics_and_partitions():
    source = _initial_states()
    migrated = redistribute_energon_states(source, 1)[0]
    contract, pipeline = _contract(0, 1, 4)
    profile = build_rescale_profile(contract, pipeline, GLOBAL_BATCH_SIZE)

    variants = []
    bad = deepcopy(migrated)
    bad["migration"]["target"]["num_workers"] = 3
    variants.append(bad)
    bad = deepcopy(migrated)
    bad["migration"]["worker_partitions"][0] = "0" * 64
    variants.append(bad)
    bad = deepcopy(migrated)
    bad["migration"]["completed_global_batches"] += 1
    variants.append(bad)
    for state in variants:
        with pytest.raises(ValueError):
            validate_rescale_restore(state, contract, profile)

    changed_contract, changed_pipeline = _contract(0, 1, 4, identity="changed-task")
    changed_profile = build_rescale_profile(changed_contract, changed_pipeline, GLOBAL_BATCH_SIZE)
    with pytest.raises(ValueError, match="semantic"):
        validate_rescale_restore(migrated, changed_contract, changed_profile)


def test_explicit_file_conversion_writes_new_dataloader_only_output(tmp_path):
    source_paths = [tmp_path / f"source{lane}.pt" for lane in range(2)]
    source_bytes = []
    for path, state in zip(source_paths, _initial_states(), strict=True):
        torch.save({"dataloader_state_dict": state}, path)
        source_bytes.append(path.read_bytes())

    output = tmp_path / "migrated"
    redistribute_checkpoint_files(source_paths, output, 1)
    assert [path.read_bytes() for path in source_paths] == source_bytes
    target = output / "train_dataloader_dprank000.pt"
    payload = torch.load(target, map_location="cpu", weights_only=False)
    assert set(payload) == {"dataloader_state_dict"}
    assert payload["dataloader_state_dict"]["migration"]["target"]["language_lanes"] == 1
    with pytest.raises(ValueError, match="must be new"):
        redistribute_checkpoint_files(source_paths, output, 1)
