# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""evaluate ends cleanly when a packed validation iterator runs out between eval steps."""

from types import SimpleNamespace
from unittest import mock

import pytest

from megatron.training import ft_integration
from megatron.training import training as training_mod
from megatron.training.state import TrainState


class _RankMonitorClient:
    """Record FT sections; like the rank monitor, reject closing a section that is not open."""

    def __init__(self):
        self.open_sections = set()
        self.closed_sections = []

    def start_section(self, section):
        assert section not in self.open_sections, f"Tried to open {section} twice"
        self.open_sections.add(section)

    def end_section(self, section):
        if section not in self.open_sections:
            raise RuntimeError(f"Tried to close {section} that is not open")
        self.open_sections.remove(section)
        self.closed_sections.append(section)


@pytest.mark.parametrize("completed_steps", [0, 1, 2])
def test_evaluate_stops_cleanly_when_packed_iterator_runs_out(monkeypatch, completed_steps):
    client = _RankMonitorClient()
    monkeypatch.setattr(ft_integration, "_GLOBAL_RANK_MONITOR_CLIENT", client)
    monkeypatch.setattr(ft_integration, "_NUM_WARMUP_ITERS", 1)
    monkeypatch.setattr(ft_integration, "_curr_eval_iter_idx", 0)
    monkeypatch.setattr(ft_integration, "_is_setup_section_open", False)
    args = SimpleNamespace(
        eval_global_batch_size=1,
        eval_micro_batch_size=1,
        data_parallel_size=1,
        gtp_weight_remat_size=1,
        cuda_graph_impl="none",
        moe_expert_rank_capacity_factor=None,
        seq_length=8,
        decoder_seq_length=None,
        eval_iters=completed_steps + 2,
        empty_unused_memory_level=0,
        consumed_valid_samples=0,
        exit_duration_in_mins=None,
    )
    config = SimpleNamespace(sequence_packing_scheduler="dp_balanced")
    timers = mock.MagicMock()
    # Scheduling raises StopIteration once the finite validation iterator is exhausted.
    schedules = [(iter([]), 1, 0, 0)] * completed_steps + [StopIteration]
    forward_backward = mock.Mock(return_value=[])
    with (
        mock.patch.object(training_mod, "get_args", return_value=args),
        mock.patch.object(training_mod, "get_train_state", return_value=TrainState()),
        mock.patch.object(training_mod, "get_timers", return_value=timers),
        mock.patch.object(training_mod, "get_rerun_state_machine", return_value=mock.Mock()),
        mock.patch.object(training_mod, "get_forward_backward_func", return_value=forward_backward),
        mock.patch.object(
            training_mod,
            "get_attr_wrapped_model",
            return_value=SimpleNamespace(pp=object(), dp_cp=object()),
        ),
        mock.patch.object(training_mod, "is_pp_last_stage", return_value=True),
        mock.patch.object(training_mod, "has_nvidia_modelopt", False),
        mock.patch.object(training_mod, "_otel_managed_span", return_value=mock.MagicMock()),
        mock.patch.object(training_mod, "wrap_data_iterator", side_effect=schedules),
    ):
        _, _, exited = training_mod.evaluate(
            lambda *a, **kw: None, iter([]), [mock.Mock()], None, config
        )

    assert not exited
    assert forward_backward.call_count == completed_steps
    # FT skips the warmup eval step and monitors each later step in a "step" section.
    assert client.closed_sections == ["step"] * max(completed_steps - 1, 0)
    assert not client.open_sections
    assert ft_integration._curr_eval_iter_idx == completed_steps
    assert config.timers is timers
