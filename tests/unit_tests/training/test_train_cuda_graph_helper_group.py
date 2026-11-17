# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""train() builds the Transformer Engine CUDA graph helper on the model's process groups."""

from argparse import ArgumentParser
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.training import arguments, training
from megatron.training.state import TrainState


def test_train_passes_the_model_collection_to_the_cuda_graph_helper(monkeypatch, run_config):
    """The helper decides CUDA graph reuse from the graphed model's pipeline group."""
    parser = ArgumentParser()
    arguments.add_megatron_arguments(parser)
    args = parser.parse_args(["--micro-batch-size", "1"])
    run_config.train.train_iters = 10
    run_config.train.micro_batch_size = 1
    run_config.model.transformer.cuda_graph_impl = "transformer_engine"
    # Run the real train() preamble without the services that it reports to.
    monkeypatch.setattr(training, "get_args", lambda: args)
    monkeypatch.setattr(training, "get_train_state", lambda: TrainState())
    monkeypatch.setattr(training, "get_timers", Mock(return_value=Mock()))
    monkeypatch.setattr(training, "get_rerun_state_machine", Mock(return_value=Mock()))
    monkeypatch.setattr(training, "get_energy_monitor", lambda: None)
    monkeypatch.setattr(training, "get_one_logger", lambda: None)
    monkeypatch.setattr(training, "get_num_microbatches", lambda: 1)
    monkeypatch.setattr(training, "get_current_global_batch_size", lambda: 1)
    monkeypatch.setattr(training, "should_disable_forward_pre_hook", lambda *_: False)
    for name in (
        "get_forward_backward_func",
        "one_logger_utils",
        "write_args_to_tensorboard",
        "print_datetime",
    ):
        monkeypatch.setattr(training, name, Mock())

    class HelperBuilt(Exception):
        pass

    helper = Mock(side_effect=HelperBuilt)
    monkeypatch.setattr(training, "TECudaGraphHelper", helper)
    model_pg_collection = ProcessGroupCollection()

    with pytest.raises(HelperBuilt):
        training.train(
            forward_step_func=Mock(),
            model=[Mock(pg_collection=model_pg_collection)],
            optimizer=None,
            opt_param_scheduler=None,
            train_data_iterator=None,
            config=SimpleNamespace(finalize_model_grads_func=Mock()),
            checkpointing_context={},
            next_train_iterator=lambda iteration, current: current,
            run_eval=Mock(),
        )

    assert helper.call_args.kwargs["pg_collection"] is model_pg_collection
