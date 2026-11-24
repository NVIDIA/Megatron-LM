# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""RerunStateMachine reduces banked device verdicts as single-element flags."""

import pytest
import torch

from megatron.core import rerun_state_machine as rsm
from tests.unit_tests.test_utilities import Utils


@pytest.fixture
def machine(tmp_path):
    Utils.initialize_distributed()
    return rsm.RerunStateMachine(
        mode=rsm.RerunMode.VALIDATE_RESULTS,
        result_rejected_tracker_filename=str(tmp_path / "rejected.txt"),
    )


def test_reduce_any_flag_is_true_when_any_rank_set_it(machine):
    flag = torch.tensor(Utils.rank == 0, device="cuda")
    assert machine._reduce_any_flag(flag) is True
    assert flag.item() == (Utils.rank == 0)  # the caller's buffer is untouched


def test_reduce_any_flag_is_false_when_no_rank_set_it(machine):
    assert machine._reduce_any_flag(torch.zeros((), dtype=torch.bool, device="cuda")) is False


def test_reduce_any_flag_rejects_a_multi_element_tensor(machine):
    with pytest.raises(ValueError, match="single-element"):
        machine._reduce_any_flag(torch.zeros(2, dtype=torch.bool, device="cuda"))


def test_multi_element_verdict_is_not_banked(machine):
    """Like the synchronous path, an ambiguous multi-element verdict is an error."""
    machine.state = rsm.RerunState.INITIAL_RUN
    machine.first_iteration_complete = True
    with pytest.raises(RuntimeError, match="ambiguous"):
        machine.validate_result(
            result=torch.ones(2, device="cuda"),
            rejection_func=lambda value: value > 0,
            message="multi-element verdict",
        )


def test_device_verdict_is_banked_then_attributed_at_the_loop_exit(machine):
    machine.state = rsm.RerunState.INITIAL_RUN
    machine.first_iteration_complete = True
    machine._reset_banked_verdicts()
    for value in (0.0, float("nan")):
        machine.validate_result(
            result=torch.tensor(value, device="cuda"),
            rejection_func=torch.isnan,
            message=f"value {value}",
        )
    # Banked, not read: nothing is recorded until the loop exit reduces the flags.
    assert not machine.rerun_requested
    assert len(machine._banked_calls) == 2
    assert machine._reduce_any_flag(machine._banked_rejection_flag()) is True
    machine._identify_banked_rejection()
    assert machine.rerun_requested
    assert machine.failed_validation_call == machine._banked_calls[1][0]
