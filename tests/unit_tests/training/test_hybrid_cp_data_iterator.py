# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from types import SimpleNamespace

import megatron.core.datasets.data_schedule as data_schedule
from megatron.core.datasets.data_schedule_utils import create_data_iterator
from megatron.core.rerun_state_machine import RerunDataIterator, RerunMode, RerunStateMachine


def _assert_sanitizable(wrapped):
    assert isinstance(wrapped, RerunDataIterator)
    # The exact production check: sanitization must accept the iterator.
    sanitized = RerunStateMachine._sanitize_data_iterators(
        SimpleNamespace(mode=RerunMode.VALIDATE_RESULTS), wrapped
    )
    assert sanitized == [wrapped]


def test_wrap_dynamic_cp_data_iterator_returns_rerun_iterator(monkeypatch):
    """The rerun state machine asserts on the RerunDataIterator type at every
    train step (rerun_state_machine._sanitize_data_iterators). The dynamic-CP
    scheduler replaces the removed HybridCPDataLoaderWrapper; its iterator and
    packed-step statistics must survive the wrapper unchanged.
    """
    fake_group = SimpleNamespace(size=lambda: 4, rank=lambda: 0)
    groups = SimpleNamespace(
        dp_cp=fake_group,
        dp=fake_group,
        tp=fake_group,
        pp=fake_group,
        dp_cp_gtp_remat=None,
        dp_gtp_remat=None,
        dynamic_cp={1: fake_group, 2: fake_group, 4: fake_group},
    )
    original = iter([])
    expected = RerunDataIterator(iter([]))

    class Scheduler:
        def __init__(self, *args, **kwargs):
            assert kwargs['min_cp_size'] == 1

        def run(self, iterator, num_microbatches, *args):
            assert iterator is original
            assert num_microbatches == 2
            return expected, 3, 256, 16384

    # scheduler_map is keyed by PackingSchedulerEnum after main's ea99fdfce.
    monkeypatch.setitem(
        data_schedule.scheduler_map,
        data_schedule.PackingSchedulerEnum.DEFAULT_DYNAMIC_CP,
        Scheduler,
    )
    monkeypatch.setattr(data_schedule.torch.cuda, 'current_device', lambda: 0)
    config = SimpleNamespace(
        max_seqlen_per_dp_cp_rank=128,
        sequence_packing_scheduler='default_dynamic_cp',
        min_dynamic_context_parallel_size=1,
        virtual_pipeline_model_parallel_size=None,
    )
    wrapped, microbatches, tokens, squared_tokens = data_schedule.wrap_data_iterator(
        original, config, 2, pg_collection=groups
    )
    assert wrapped is expected
    assert (microbatches, tokens, squared_tokens) == (3, 256, 16384)

    _assert_sanitizable(wrapped)


def test_dynamic_cp_data_iterator_returns_rerun_iterator():
    """The rerun state machine asserts on the RerunDataIterator type at every
    train step (rerun_state_machine._sanitize_data_iterators); the dynamic-CP
    wrap must preserve it. Regression: a raw iter() around
    HybridCPDataLoaderWrapper stripped the wrapping and crashed before the
    first iteration.
    """
    fake_group = SimpleNamespace(size=lambda: 1, rank=lambda: 0)
    config = SimpleNamespace(virtual_pipeline_model_parallel_size=None)
    wrapped = create_data_iterator([], fake_group, config, is_dynamic_cp=True)

    _assert_sanitizable(wrapped)
