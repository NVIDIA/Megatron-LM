# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from unittest.mock import Mock, patch

import pytest
import torch

from megatron.core.full_cuda_graph import FullCudaGraphWrapper
from megatron.core.rerun_state_machine import (
    EXIT_CODE_FAILED_ON_RESULT_VALIDATION,
    EXIT_CODE_RESUME_TO_DISAMBIGUATE,
    RerunMode,
    RerunState,
    RerunStateMachine,
)
from tests.unit_tests.test_utilities import Utils


@pytest.mark.parametrize(
    "bad_value,rejection", [(float("nan"), torch.isnan), (float("inf"), torch.isinf)]
)
@pytest.mark.parametrize("mode", [RerunMode.DISABLED, RerunMode.VALIDATE_RESULTS])
@pytest.mark.parametrize("reproduced", [False, True])
def test_full_cuda_graph_validates_current_replay_before_optimizer(
    bad_value, rejection, mode, reproduced
):
    """Replay failures reach rerun decisions without overwriting the retained failed result."""
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    machine = RerunStateMachine(mode=mode)
    source = torch.ones((), device="cuda")

    def forward_backward_func(**_):
        intermediate = source.clone()
        machine.validate_result(intermediate, rejection, "captured local norm", fatal=True)
        # A reference without a capture-time snapshot would lose the invalid value.
        intermediate.zero_()
        return [{"loss": intermediate}]

    wrapper = FullCudaGraphWrapper(forward_backward_func, cuda_graph_warmup_steps=1)
    kwargs = dict(
        model=[torch.nn.Identity()],
        data_iterator=None,
        num_microbatches=1,
        seq_length=1,
        forward_only=False,
    )
    with (
        patch("megatron.core.full_cuda_graph.get_all_rng_states", return_value={}),
        patch("megatron.core.full_cuda_graph.get_rerun_state_machine", return_value=machine),
    ):
        wrapper.reset_cuda_graph()
        try:
            # Complete a healthy warmup step, then capture and replay another step.
            for _ in range(2):
                assert machine.should_run_forward_backward(None)
                wrapper(**kwargs)
                assert not machine.should_run_forward_backward(None)
                assert machine.should_checkpoint_and_exit() == (False, False, 0)

            source.fill_(bad_value)
            assert machine.should_run_forward_backward(None)
            if mode == RerunMode.DISABLED:
                with pytest.raises(RuntimeError, match="captured local norm"):
                    wrapper(**kwargs)
                return

            wrapper(**kwargs)
            assert machine.rerun_requested
            assert rejection(machine.initial_result)
            assert machine.should_run_forward_backward(None)
            assert machine.state == RerunState.RERUNNING_IN_PLACE

            source.fill_(bad_value if reproduced else 2.0)
            wrapper(**kwargs)
            assert rejection(machine.initial_result), "Replay overwrote saved failed result"
            assert not machine.should_run_forward_backward(None)
            if reproduced:
                assert machine.state == RerunState.WILL_RERUN_FROM_CHECKPOINT
                assert machine.should_checkpoint_and_exit() == (
                    True,
                    True,
                    EXIT_CODE_RESUME_TO_DISAMBIGUATE,
                )
            else:
                assert machine.state == RerunState.RERUNNING_IN_PLACE
                assert machine.should_checkpoint_and_exit() == (
                    False,
                    True,
                    EXIT_CODE_FAILED_ON_RESULT_VALIDATION,
                )
        finally:
            wrapper.reset_cuda_graph()
            # Reset must remove callbacks as well as the CUDA graph itself.
            with patch.object(machine, "validate_result") as validate:
                machine.run_deferred_validations("training")
                validate.assert_not_called()
            Utils.destroy_model_parallel()


def test_deferred_validations_are_replaced_and_cleared_per_graph():
    """Recapture replaces one graph's checks, and reset preserves the other graph's checks."""
    machine = RerunStateMachine(mode=RerunMode.DISABLED)
    reject = Mock(return_value=False)
    for key, value in [("training", 1.0), ("validation", 2.0), ("training", 3.0)]:
        with machine.capture_deferred_validations(key):
            machine.validate_result(torch.tensor(value, device="cuda"), reject, key)
    reject.assert_not_called()
    machine.run_deferred_validations("training")
    reject.assert_called_once()
    assert reject.call_args.args[0].item() == 3.0

    wrapper = FullCudaGraphWrapper(lambda **_: None)
    with patch("megatron.core.full_cuda_graph.get_rerun_state_machine", return_value=machine):
        wrapper.reset_cuda_graph("training")
        machine.run_deferred_validations("training")
        reject.assert_called_once()
        machine.run_deferred_validations("validation")
        assert reject.call_count == 2
        assert reject.call_args.args[0].item() == 2.0
        wrapper.reset_cuda_graph()
        machine.run_deferred_validations("validation")
        assert reject.call_count == 2


def test_validation_capture_restores_state_after_error():
    """A failed capture must discard partial checks and restore immediate validation."""
    machine = RerunStateMachine(mode=RerunMode.DISABLED)
    reject = Mock(return_value=False)
    with pytest.raises(RuntimeError, match="cannot be nested"):
        with machine.capture_deferred_validations("training"):
            machine.validate_result(torch.ones((), device="cuda"), reject, "partial capture")
            with machine.capture_deferred_validations("validation"):
                pass
    machine.run_deferred_validations("training")
    reject.assert_not_called()
    with machine.capture_deferred_validations("training"):
        pass
    with pytest.raises(RuntimeError, match="eager NaN"):
        machine.validate_result(torch.tensor(float("nan")), torch.isnan, "eager NaN")


def test_validation_capture_rejects_host_results():
    """Host values cannot reflect updates made by graph replay."""
    machine = RerunStateMachine(mode=RerunMode.DISABLED)
    with machine.capture_deferred_validations("training"):
        with pytest.raises(TypeError, match="CUDA tensor"):
            machine.validate_result(torch.tensor(1.0), torch.isnan, "host result")
