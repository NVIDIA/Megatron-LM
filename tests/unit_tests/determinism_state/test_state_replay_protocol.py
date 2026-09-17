# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Exercise actual subprocess training, checkpoint load and a broken RNG restore."""

import json
from itertools import product

import pytest

from tools.determinism import run_state_replay as replay
from tools.determinism.run_state_replay import run_protocol, run_stop_points
from tools.determinism.training_state import UnverifiedState, capture_configuration, snapshot_path


@pytest.mark.parametrize("control,first_component", [("rng", "model"), ("scheduler", "scheduler")])
def test_cpu_training_replay_resume_and_omitted_state_control(tmp_path, control, first_component):
    output = tmp_path / "protocol"
    result = run_protocol(
        output, backend="cpu", world_size=1, steps=4, checkpoint_step=2, control=control
    )
    assert result.get("expected_observations"), (output / "report.json").read_text()
    assert result["fresh"]["comparison_status"] == "equal"
    assert result["fresh"]["compared_snapshots"] == 4
    assert result["resume"]["comparison_status"] == "equal"
    assert result["resume"]["compared_snapshots"] == 2
    assert result["control"]["comparison_status"] == "different"
    assert result["control"]["first_difference"]["step"] == 3
    assert result["control"]["first_difference"]["rank"] == 0
    assert result["control"]["first_difference"]["path"][0] == first_component
    assert result["control_injection"] == f"omit_restore_{control}"
    records = [
        json.loads(snapshot_path(output / name, step, 0).read_text())
        for name, step in (("reference", 1), ("repeat", 1), ("resume", 3), ("control", 3))
    ]
    assert len({record["process_id"] for record in records}) == 4
    dirty = records[0]["provenance"]["source_dirty"]
    assert result["status"] == ("not_verified" if dirty else "passed")
    assert json.loads((output / "report.json").read_text()) == result


@pytest.mark.parametrize("control", ["rng", "scheduler"])
def test_separate_cpu_stop_points_preserve_horizon_and_restore_controls(tmp_path, control):
    output = tmp_path / "stops"
    result = run_stop_points(
        output,
        backend="cpu",
        world_size=1,
        steps=5,
        checkpoint_step=2,
        control=control,
        stop_steps=[3, 5],
    )
    processes = set()
    run_ids = set()
    for target, step in zip(result["targets"], (3, 5)):
        assert target["expected_observations"], target
        assert target["fresh"]["compared_snapshots"] == 1
        assert target["resume"]["compared_snapshots"] == 1
        assert target["control"]["comparison_status"] == "different"
        assert target["control"]["first_difference"]["step"] == step
        for name in ("reference", "repeat", "resume", "control"):
            root = output / f"stop-{step:08d}" / name
            paths = list(root.glob("step-*/rank-*.json"))
            assert paths == [snapshot_path(root, step, 0)]
            record = json.loads(paths[0].read_text())
            assert record["provenance"]["recipe"]["capture"] == capture_configuration(5, 2, step)
            assert json.loads((root / "complete-rank-000000.json").read_text())["steps"] == [step]
            processes.add(record["process_id"])
            run_ids.add(record["run_id"])
            command = json.loads((root.parent / f"{name}-command.json").read_text())
            assert command[command.index("--steps") + 1] == "5"
            assert command[command.index("--stop-step") + 1] == str(step)
            # Each process actually reached the target scheduler update; it did not
            # merely relabel a prior snapshot. The control intentionally differs.
            if name != "control":
                last_epoch = next(
                    item for item in record["index"] if item["path"] == ["scheduler", "last_epoch"]
                )
                assert last_epoch["value"] == step
    assert len(processes) == len(run_ids) == 8
    dirty = record["provenance"]["source_dirty"]
    assert result["status"] == ("not_verified" if dirty else "passed")
    assert json.loads((output / "report.json").read_text()) == result


@pytest.mark.parametrize("stop_steps", [[], [3, 3], [2], [6], [True]])
def test_invalid_stop_points_do_not_launch_or_create_output(tmp_path, stop_steps):
    with pytest.raises(ValueError):
        run_stop_points(
            tmp_path / "invalid",
            backend="cpu",
            world_size=1,
            steps=5,
            checkpoint_step=2,
            control="rng",
            stop_steps=stop_steps,
        )
    assert not (tmp_path / "invalid").exists()


@pytest.mark.parametrize(
    "change", ["earlier_snapshot", "earlier_completion", "horizon", "missing_rank"]
)
def test_stop_point_inventory_rejects_earlier_capture_or_incomplete_contract(tmp_path, change):
    capture = capture_configuration(5, 2, 3)
    for rank in range(2):
        path = snapshot_path(tmp_path, 3, rank)
        path.parent.mkdir(exist_ok=True)
        path.write_text("{}")
        path.with_suffix(".bin").write_bytes(b"fixture")
        (tmp_path / f"complete-rank-{rank:06d}.json").write_text(
            json.dumps({"steps": [3], "provenance": {"recipe": {"capture": capture}}})
        )
    replay._verify_stop_point(tmp_path, 2, capture)
    completion = tmp_path / "complete-rank-000001.json"
    if change == "earlier_snapshot":
        path = snapshot_path(tmp_path, 2, 0)
        path.parent.mkdir()
        path.with_suffix(".bin").write_bytes(b"earlier capture")
    elif change == "missing_rank":
        snapshot_path(tmp_path, 3, 1).with_suffix(".bin").unlink()
    else:
        record = json.loads(completion.read_text())
        if change == "earlier_completion":
            record["steps"] = [2, 3]
        else:
            record["provenance"]["recipe"]["capture"]["training_steps"] = 3
        completion.write_text(json.dumps(record))
    with pytest.raises(UnverifiedState):
        replay._verify_stop_point(tmp_path, 2, capture)


@pytest.mark.parametrize("status", ["failed", "not_verified"])
def test_one_unsuccessful_target_cannot_pass_the_aggregate(tmp_path, monkeypatch, status):
    calls = []

    def run(output, **kwargs):
        calls.append(kwargs["stop_step"])
        return {"status": "passed" if kwargs["stop_step"] == 3 else status}

    monkeypatch.setattr(replay, "run_protocol", run)
    result = run_stop_points(
        tmp_path / "stops",
        backend="cpu",
        world_size=1,
        steps=5,
        checkpoint_step=2,
        control="rng",
        stop_steps=[3, 5],
    )
    assert calls == [3, 5]
    assert result["status"] == status


@pytest.mark.parametrize("world_size", [4, 8])
@pytest.mark.parametrize("pipeline_size", [1, 2])
@pytest.mark.parametrize("change", ["duplicate", "loader_owner"])
def test_megatron_layout_requires_every_pipeline_and_data_rank(
    tmp_path, world_size, pipeline_size, change
):
    data_size = world_size // (2 * pipeline_size)
    for rank, (tp, pp, dp) in enumerate(product(range(2), range(pipeline_size), range(data_size))):
        record = {
            "global_rank": rank,
            "world_size": world_size,
            "sizes": {"TP": 2, "PP": pipeline_size, "DP": data_size, "CP": 1},
            "TP": tp,
            "PP": pp,
            "DP": dp,
            "CP": 0,
            "VPP": None,
            "owns_loader": tp == 0,
        }
        (tmp_path / f"complete-rank-{rank:06d}.json").write_text(
            json.dumps({"provenance": {"rank_layout": record}})
        )
    replay._verify_megatron_layout(tmp_path, world_size, pipeline_size)
    # Every global-rank file is present, but the final coordinate was duplicated.
    path = tmp_path / f"complete-rank-{world_size - 1 if change == 'duplicate' else 0:06d}.json"
    record = json.loads(path.read_text())
    if change == "duplicate":
        record["provenance"]["rank_layout"]["DP"] = 0
        record["provenance"]["rank_layout"]["PP"] = 0
    else:
        record["provenance"]["rank_layout"]["owns_loader"] = False
    path.write_text(json.dumps(record))
    with pytest.raises(UnverifiedState, match="coordinates|rank layout"):
        replay._verify_megatron_layout(tmp_path, world_size, pipeline_size)


@pytest.mark.parametrize("backend", ["cpu", "mcore_gpt"])
def test_pp2_is_rejected_by_adapters_that_do_not_use_a_pipeline(tmp_path, backend):
    with pytest.raises(ValueError, match="PP=2"):
        run_stop_points(
            tmp_path / "invalid",
            backend=backend,
            world_size=4,
            pipeline_size=2,
            steps=5,
            checkpoint_step=2,
            control="rng",
            stop_steps=[3, 5],
        )
    assert not (tmp_path / "invalid").exists()
