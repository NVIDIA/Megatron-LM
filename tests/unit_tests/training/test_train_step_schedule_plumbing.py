# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""train_step forwards p2p_communicator and schedule pg_collection to forward_backward_func."""

from types import SimpleNamespace
from unittest import mock

from megatron.training import training as training_mod


class _Rerun:
    """Run the forward/backward body once, then ask train_step to exit before optimizer.step."""

    _ran = False

    def should_run_forward_backward(self, data_iterator):
        run, self._ran = not self._ran, True
        return run

    def should_checkpoint_and_exit(self):
        return False, True, 0  # (checkpoint, exit, code)


def _run(*, packed_num_microbatches=None, events=None, **kwargs):
    args = SimpleNamespace(
        save_params_interval=None,
        save_activations_interval=None,
        save_tokens_per_expert_interval=None,
        save_wgrads_interval=None,
        save_dgrads_interval=None,
        reuse_grad_buf_for_mxfp8_param_ag=False,
        overlap_param_gather=False,
        seq_length=8,
        global_batch_size=1,
        micro_batch_size=1,
        decoder_seq_length=None,
        empty_unused_memory_level=0,
    )
    captured = {}
    model = [SimpleNamespace(force_all_reduce=False, zero_grad_buffer=lambda: None)]

    def wrap_data_iterator(data_iterator, config, unused_num_microbatches):
        assert config.sequence_packing_scheduler is not None
        assert packed_num_microbatches is not None
        if events is not None:
            events.append(("wrap", packed_num_microbatches))
        return data_iterator, packed_num_microbatches, 0, 0

    def validate_topology(config, *, num_microbatches, micro_batch_size, phase):
        if events is not None:
            events.append(("validate", num_microbatches, micro_batch_size, phase))

    def forward_backward(**forward_backward_kwargs):
        captured.update(forward_backward_kwargs)
        if events is not None:
            events.append(("forward_backward", forward_backward_kwargs["num_microbatches"]))
        return []

    with (
        mock.patch.object(training_mod, "get_args", return_value=args),
        mock.patch.object(training_mod, "get_timers", return_value=mock.MagicMock()),
        mock.patch.object(training_mod, "get_rerun_state_machine", return_value=_Rerun()),
        mock.patch.object(training_mod, "get_num_microbatches", return_value=1),
        mock.patch.object(training_mod, "has_nvidia_modelopt", False),
        mock.patch.object(
            training_mod, "wrap_data_iterator", side_effect=wrap_data_iterator
        ),
        mock.patch.object(
            training_mod,
            "validate_te_cuda_graph_topology",
            side_effect=validate_topology,
        ),
    ):
        training_mod.train_step(
            forward_step_func=lambda *a, **k: None,
            data_iterator=iter([]),
            model=model,
            optimizer=SimpleNamespace(zero_grad=lambda: None),
            opt_param_scheduler=None,
            config=SimpleNamespace(
                sequence_packing_scheduler=(
                    "dp_balanced" if packed_num_microbatches is not None else None
                )
            ),
            forward_backward_func=forward_backward,
            iteration=0,
            **kwargs,
        )
    return captured


def test_train_step_forwards_schedule_plumbing():
    p2p, pg = object(), object()
    captured = _run(p2p_communicator=p2p, schedule_pg_collection=pg)
    assert captured["p2p_communicator"] is p2p and captured["pg_collection"] is pg


def test_train_step_defaults_to_none():
    captured = _run()
    assert captured["p2p_communicator"] is None and captured["pg_collection"] is None


def test_train_step_validates_actual_packed_microbatch_count_before_forward_backward():
    events = []

    captured = _run(packed_num_microbatches=7, events=events)

    assert captured["num_microbatches"] == 7
    assert events == [
        ("wrap", 7),
        ("validate", 7, 1, "training-forward-backward"),
        ("forward_backward", 7),
    ]
