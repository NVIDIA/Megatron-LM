# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from types import SimpleNamespace

from megatron.core.datasets.data_schedule_utils import create_data_iterator
from megatron.core.rerun_state_machine import RerunDataIterator, RerunMode, RerunStateMachine


def test_dynamic_cp_data_iterator_returns_rerun_iterator():
    """The rerun state machine asserts on the RerunDataIterator type at every
    train step (rerun_state_machine._sanitize_data_iterators); the hybrid-CP
    wrap must preserve it. Regression: a raw iter() around
    HybridCPDataLoaderWrapper stripped the wrapping and crashed before the
    first iteration.
    """
    fake_group = SimpleNamespace(size=lambda: 1, rank=lambda: 0)
    config = SimpleNamespace(virtual_pipeline_model_parallel_size=None)
    wrapped = create_data_iterator([], fake_group, config, is_dynamic_cp=True)

    assert isinstance(wrapped, RerunDataIterator)
    # The exact production check: sanitization must accept the iterator.
    sanitized = RerunStateMachine._sanitize_data_iterators(
        SimpleNamespace(mode=RerunMode.VALIDATE_RESULTS), wrapped
    )
    assert sanitized == [wrapped]
