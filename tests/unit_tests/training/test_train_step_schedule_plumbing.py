# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""train_step forwards the carrier to the schedule and derives reductions from the model."""

from types import SimpleNamespace
from unittest import mock

import torch

from megatron.core.process_groups_config import MultiModuleProcessGroupCollection
from megatron.training import training as training_mod


class _Rerun:
    """Run the forward/backward body once, then ask train_step to exit before optimizer.step."""

    _ran = False

    def should_run_forward_backward(self, data_iterator):
        run, self._ran = not self._ran, True
        return run

    def should_checkpoint_and_exit(self):
        return False, True, 0  # (checkpoint, exit, code)


def _run(
    *,
    exit_before_optimizer=True,
    losses_reduced=None,
    is_last_stage=False,
    model_pgc=None,
    **kwargs,
):
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
        qk_clip=False,
        log_max_attention_logit=False,
        barrier_with_L1_time=False,
        log_num_zeros_in_grad=True,
        data_parallel_size=2,
    )
    captured = {}
    # The model carries the per-rank reduction ProcessGroupCollection.
    model = [
        SimpleNamespace(
            force_all_reduce=False, zero_grad_buffer=lambda: None, pg_collection=model_pgc
        )
    ]
    rerun = _Rerun()
    if not exit_before_optimizer:
        rerun.should_checkpoint_and_exit = lambda: (False, False, 0)
    optimizer = SimpleNamespace(zero_grad=lambda: None, step=mock.Mock(return_value=(True, 2.0, 3)))
    scheduler = SimpleNamespace(step=mock.Mock())
    with (
        mock.patch.object(training_mod, "get_args", return_value=args),
        mock.patch.object(training_mod, "get_timers", return_value=mock.MagicMock()),
        mock.patch.object(training_mod, "get_rerun_state_machine", return_value=rerun),
        mock.patch.object(training_mod, "get_num_microbatches", return_value=1),
        mock.patch.object(training_mod, "has_nvidia_modelopt", False),
        mock.patch.object(training_mod, "is_pp_last_stage", return_value=is_last_stage) as is_last,
        mock.patch.object(training_mod.torch.distributed, "all_reduce") as all_reduce,
        mock.patch.object(
            training_mod, "logical_and_across_model_parallel_group", return_value=True
        ) as logical_and,
        mock.patch.object(
            training_mod,
            "reduce_max_stat_across_model_parallel_group",
            side_effect=lambda value, group=None: value,
        ) as reduce_max,
    ):
        result = training_mod.train_step(
            forward_step_func=lambda *a, **k: None,
            data_iterator=iter([]),
            model=model,
            optimizer=optimizer,
            opt_param_scheduler=scheduler,
            config=SimpleNamespace(),
            forward_backward_func=lambda **kw: captured.update(kw) or (losses_reduced or []),
            iteration=0,
            **kwargs,
        )
    return SimpleNamespace(
        captured=captured,
        result=result,
        is_last=is_last,
        logical_and=logical_and,
        reduce_max=reduce_max,
        all_reduce=all_reduce,
    )


def test_train_step_forwards_carrier_and_p2p_to_schedule():
    p2p = object()
    carrier = MultiModuleProcessGroupCollection(
        module_pgs={"encoder": SimpleNamespace()}, language_model_module_name=None
    )
    reductions = SimpleNamespace(mp=object(), pp=object(), dp_cp=object())
    run = _run(p2p_communicator=p2p, pg_collection=carrier, model_pgc=reductions)
    assert run.captured["p2p_communicator"] is p2p
    assert run.captured["pg_collection"] is carrier


def test_train_step_defaults_carrier_to_none():
    reductions = SimpleNamespace(mp=object(), pp=object(), dp_cp=object())
    run = _run(model_pgc=reductions)
    assert run.captured["p2p_communicator"] is None
    assert run.captured["pg_collection"] is None


def test_train_step_reductions_use_model_pg_collection_not_carrier():
    reductions = SimpleNamespace(mp=object(), pp=object(), dp_cp=object())
    carrier = MultiModuleProcessGroupCollection(
        module_pgs={"language": SimpleNamespace()}, language_model_module_name="language"
    )
    run = _run(
        exit_before_optimizer=False,
        pg_collection=carrier,
        model_pgc=reductions,
        p2p_communicator=object(),
    )
    # mp reductions source the model's collection, never the carrier.
    run.logical_and.assert_called_once_with(True, group=reductions.mp)
    assert run.reduce_max.call_count == 2


def test_train_step_falls_back_to_mpu_when_model_has_no_pg_collection():
    mpu_pgc = SimpleNamespace(mp=object(), pp=object(), dp_cp=object())
    with mock.patch.object(
        training_mod.ProcessGroupCollection, "use_mpu_process_groups", return_value=mpu_pgc
    ):
        run = _run(exit_before_optimizer=False, model_pgc=None)
    run.logical_and.assert_called_once_with(True, group=mpu_pgc.mp)


def test_train_step_reduces_terminal_loss_on_model_dp_cp():
    reductions = SimpleNamespace(mp=object(), pp=object(), dp_cp=object())
    run = _run(
        exit_before_optimizer=False,
        losses_reduced=[{"loss": torch.tensor([6.0, 2.0])}],
        is_last_stage=True,
        model_pgc=reductions,
        p2p_communicator=object(),
    )
    run.is_last.assert_called_once_with(reductions.pp)
    run.all_reduce.assert_called_once()
    assert run.all_reduce.call_args.kwargs["group"] is reductions.dp_cp
    assert run.result[0]["loss"].item() == 3.0
