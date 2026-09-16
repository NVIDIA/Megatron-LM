# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Exercise actual subprocess training, checkpoint load and a broken RNG restore."""

import json

import pytest

from tools.determinism.run_state_replay import run_protocol
from tools.determinism.training_state import snapshot_path


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
