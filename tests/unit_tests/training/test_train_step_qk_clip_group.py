# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""train_step reduces the QK-clip logits over the model's own data-parallel group."""

from types import SimpleNamespace
from unittest import mock

import pytest

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.training import training as training_mod


class _StopAfterClip(Exception):
    """Ends train_step once clip_qk has been called."""


class _Rerun:
    """Run the forward/backward body once, then continue to the optimizer step."""

    _ran = False

    def should_run_forward_backward(self, data_iterator):
        run, self._ran = not self._ran, True
        return run

    def should_checkpoint_and_exit(self):
        return False, False, 0  # (checkpoint, exit, code)


def _clip_qk_call(model_pg_collection):
    """Run train_step with QK clipping on; return the arguments clip_qk received."""
    args = SimpleNamespace(
        save_params_interval=None,
        save_activations_interval=None,
        save_tokens_per_expert_interval=None,
        save_wgrads_interval=None,
        save_dgrads_interval=None,
        reuse_grad_buf_for_mxfp8_param_ag=False,
        overlap_param_gather=False,
        seq_length=8,
        micro_batch_size=1,
        decoder_seq_length=None,
        empty_unused_memory_level=0,
        vision_pretraining=False,
        qk_clip=True,
    )
    model = [
        SimpleNamespace(
            force_all_reduce=False, zero_grad_buffer=lambda: None, pg_collection=model_pg_collection
        )
    ]
    optimizer = SimpleNamespace(
        zero_grad=lambda: None, chained_optimizers=[], step=lambda: (True, 1.0, 0)
    )
    clip_qk = mock.Mock(side_effect=_StopAfterClip)
    with (
        mock.patch.object(training_mod, "get_args", return_value=args),
        mock.patch.object(training_mod, "get_timers", return_value=mock.MagicMock()),
        mock.patch.object(training_mod, "get_rerun_state_machine", return_value=_Rerun()),
        mock.patch.object(training_mod, "get_num_microbatches", return_value=1),
        mock.patch.object(training_mod, "has_nvidia_modelopt", False),
        mock.patch.object(training_mod, "_get_samples_seen_in_iteration", return_value=1),
        mock.patch.object(training_mod, "clip_qk", clip_qk),
        pytest.raises(_StopAfterClip),
    ):
        training_mod.train_step(
            forward_step_func=lambda *a, **k: None,
            data_iterator=iter([]),
            model=model,
            optimizer=optimizer,
            opt_param_scheduler=None,
            config=SimpleNamespace(log_max_attention_logit=False, barrier_with_L1_time=False),
            forward_backward_func=lambda **kw: [],
            iteration=0,
        )
    (clipped_model,), kwargs = clip_qk.call_args
    assert clipped_model is model
    return kwargs


def test_clip_qk_reduces_over_the_model_gtp_inclusive_group():
    pg_collection = ProcessGroupCollection(dp_cp=object(), dp_cp_gtp_remat=object())

    kwargs = _clip_qk_call(pg_collection)

    assert kwargs == {"log_max_only": False, "dp_cp_group": pg_collection.dp_cp_gtp_remat}


def test_clip_qk_uses_dp_cp_when_the_collection_has_no_gtp_remat_group():
    pg_collection = ProcessGroupCollection(dp_cp=object())

    kwargs = _clip_qk_call(pg_collection)

    assert kwargs["dp_cp_group"] is pg_collection.dp_cp
