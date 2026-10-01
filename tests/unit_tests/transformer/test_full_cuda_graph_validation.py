# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from unittest.mock import patch

import pytest
import torch

from megatron.core.full_cuda_graph import FullCudaGraphWrapper
from megatron.core.rerun_state_machine import RerunMode, RerunState, RerunStateMachine
from tests.unit_tests.test_utilities import Utils


@pytest.mark.parametrize(
    "bad_value,rejection", [(float("nan"), torch.isnan), (float("inf"), torch.isinf)]
)
@pytest.mark.parametrize("mode", [RerunMode.DISABLED, RerunMode.VALIDATE_RESULTS])
def test_full_cuda_graph_validates_current_replay_before_optimizer(bad_value, rejection, mode):
    """Replay checks survive overwritten intermediates and retain a failed run's value."""
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    machine = RerunStateMachine(mode=mode)
    machine.first_iteration_complete = True
    machine.state = RerunState.INITIAL_RUN
    source = torch.ones((), device="cuda")

    def forward_backward_func(**_):
        intermediate = source.clone()
        machine.validate_result(intermediate, rejection, "captured local norm", fatal=True)
        # A reference without a capture-time snapshot would lose the invalid value.
        intermediate.zero_()
        return [{"loss": intermediate}]

    wrapper = FullCudaGraphWrapper(forward_backward_func, cuda_graph_warmup_steps=1)
    wrapper.reset_cuda_graph()
    kwargs = dict(
        model=[torch.nn.Identity()],
        data_iterator=None,
        num_microbatches=1,
        seq_length=1,
        forward_only=False,
    )
    try:
        with (
            patch("megatron.core.full_cuda_graph.get_all_rng_states", return_value={}),
            patch("megatron.core.full_cuda_graph.get_rerun_state_machine", return_value=machine),
        ):
            wrapper(**kwargs)
            machine.validation_counts.clear()
            wrapper(**kwargs)
            assert len(wrapper.validation_calls["training"]) == 1
            machine.validation_counts.clear()
            source.fill_(bad_value)
            if mode == RerunMode.DISABLED:
                with pytest.raises(RuntimeError, match="captured local norm"):
                    wrapper(**kwargs)
            else:
                wrapper(**kwargs)
                assert machine.rerun_requested
                assert rejection(machine.initial_result)
            source.fill_(2.0)
            wrapper(**kwargs)
            if mode == RerunMode.VALIDATE_RESULTS:
                assert rejection(machine.initial_result), "Replay overwrote saved failed result"
            assert wrapper.validation_calls["training"][0]["result"].item() == 2.0
    finally:
        wrapper.reset_cuda_graph()
        assert not wrapper.validation_calls['training']
        Utils.destroy_model_parallel()


def test_validation_capture_restores_state_after_error():
    """A failed capture must not suppress validation in subsequent eager execution."""
    machine = RerunStateMachine(mode=RerunMode.DISABLED)
    with pytest.raises(RuntimeError, match='cannot be nested'):
        with machine.capture_validation_calls():
            with machine.capture_validation_calls():
                pass
    with machine.capture_validation_calls() as calls:
        assert not calls
    with pytest.raises(RuntimeError, match='eager NaN'):
        machine.validate_result(torch.tensor(float('nan')), torch.isnan, 'eager NaN')


def test_validation_capture_rejects_host_results():
    """Host values cannot reflect updates made by graph replay."""
    machine = RerunStateMachine(mode=RerunMode.DISABLED)
    with machine.capture_validation_calls():
        with pytest.raises(TypeError, match='CUDA tensor'):
            machine.validate_result(torch.tensor(1.0), torch.isnan, 'host result')
