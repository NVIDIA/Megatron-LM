# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from types import SimpleNamespace

import megatron.core.datasets.data_schedule as data_schedule
from megatron.core.rerun_state_machine import RerunDataIterator, RerunMode, RerunStateMachine
from megatron.training.training import wrap_hybrid_cp_data_iterator


def test_wrap_hybrid_cp_data_iterator_returns_rerun_iterator(monkeypatch):
    """The rerun state machine asserts on the RerunDataIterator type at every
    train step (rerun_state_machine._sanitize_data_iterators); the hybrid-CP
    wrap must preserve it. Regression: a raw iter() around
    HybridCPDataLoaderWrapper stripped the wrapping and crashed before the
    first iteration.
    """
    fake_group = SimpleNamespace(size=lambda: 4, rank=lambda: 0)
    monkeypatch.setattr(
        data_schedule.parallel_state,
        "get_data_parallel_group",
        lambda with_context_parallel=False: fake_group,
    )
    monkeypatch.setattr(
        data_schedule.parallel_state, "get_tensor_model_parallel_group", lambda: fake_group
    )

    config = SimpleNamespace(max_seqlen_per_dp_cp_rank=128)
    wrapped = wrap_hybrid_cp_data_iterator(iter([]), config)

    assert isinstance(wrapped, RerunDataIterator)
    # The exact production check: sanitization must accept the iterator.
    sanitized = RerunStateMachine._sanitize_data_iterators(
        SimpleNamespace(mode=RerunMode.VALIDATE_RESULTS), wrapped
    )
    assert sanitized == [wrapped]
