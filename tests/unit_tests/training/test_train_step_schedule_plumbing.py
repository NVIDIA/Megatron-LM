# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""train_step forwards p2p_communicator and schedule pg_collection to forward_backward_func."""

from types import SimpleNamespace
from unittest import mock

import torch

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.training import training as training_mod


class _Rerun:
    """Run the forward/backward body once, then ask train_step to exit before optimizer.step."""

    _ran = False

    def should_run_forward_backward(self, data_iterator):
        run, self._ran = not self._ran, True
        return run

    def should_checkpoint_and_exit(self):
        return False, True, 0  # (checkpoint, exit, code)


def _run(**kwargs):
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
    )
    captured = {}
    model = [SimpleNamespace(force_all_reduce=False, zero_grad_buffer=lambda: None)]
    with (
        mock.patch.object(training_mod, "get_args", return_value=args),
        mock.patch.object(training_mod, "get_timers", return_value=mock.MagicMock()),
        mock.patch.object(training_mod, "get_rerun_state_machine", return_value=_Rerun()),
        mock.patch.object(training_mod, "get_num_microbatches", return_value=1),
        mock.patch.object(training_mod, "has_nvidia_modelopt", False),
    ):
        training_mod.train_step(
            forward_step_func=lambda *a, **k: None,
            data_iterator=iter([]),
            model=model,
            optimizer=SimpleNamespace(zero_grad=lambda: None),
            opt_param_scheduler=None,
            config=SimpleNamespace(),
            forward_backward_func=lambda **kw: captured.update(kw) or [],
            iteration=0,
            **kwargs,
        )
    return captured


def test_train_step_forwards_schedule_plumbing():
    p2p, pg = object(), object()
    captured = _run(p2p_communicator=p2p, pg_collection=pg)
    assert captured["p2p_communicator"] is p2p and captured["pg_collection"] is pg


def test_train_step_defaults_to_none():
    captured = _run()
    assert captured["p2p_communicator"] is None and captured["pg_collection"] is None


def test_train_step_sets_scheduler_global_seqlen_stats_after_forward():
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
    )
    pg_collection = ProcessGroupCollection()
    model = [
        SimpleNamespace(
            force_all_reduce=False,
            zero_grad_buffer=lambda: None,
            pg_collection=pg_collection,
        )
    ]
    config = SimpleNamespace(sequence_packing_scheduler=object())
    events = []
    original_setter = training_mod.set_seqlen_stats_in_iteration

    def forward_backward(**kwargs):
        events.append("forward")
        training_mod.update_seqlen_stats_from_cu_seqlens(
            torch.tensor([0, 7], dtype=torch.int32)
        )
        return []

    def record_setter(total_real_tokens, seqlen_squared_sum):
        events.append(("setter", total_real_tokens, seqlen_squared_sum))
        return original_setter(total_real_tokens, seqlen_squared_sum)

    training_mod._seqlen_stats_in_iteration = None
    training_mod._seqlen_stats_active = False
    training_mod._seqlen_stats_are_global = False
    try:
        with (
            mock.patch.object(training_mod, "get_args", return_value=args),
            mock.patch.object(training_mod, "get_timers", return_value=mock.MagicMock()),
            mock.patch.object(training_mod, "get_rerun_state_machine", return_value=_Rerun()),
            mock.patch.object(training_mod, "get_num_microbatches", return_value=8),
            mock.patch.object(training_mod, "has_nvidia_modelopt", False),
            mock.patch.object(
                training_mod,
                "wrap_data_iterator",
                return_value=(iter([]), 3, 450, 62500),
            ),
            mock.patch.object(
                training_mod, "set_seqlen_stats_in_iteration", side_effect=record_setter
            ),
        ):
            training_mod.train_step(
                forward_step_func=lambda *args, **kwargs: None,
                data_iterator=iter([]),
                model=model,
                optimizer=SimpleNamespace(zero_grad=lambda: None),
                opt_param_scheduler=None,
                config=config,
                forward_backward_func=forward_backward,
                iteration=0,
            )

        assert events == ["forward", ("setter", 450, 62500)]
        with mock.patch.object(torch.distributed, "all_reduce") as all_reduce:
            assert training_mod.consume_seqlen_stats_in_iteration() == (450, 62500)
        all_reduce.assert_not_called()
    finally:
        training_mod._seqlen_stats_in_iteration = None
        training_mod._seqlen_stats_active = False
        training_mod._seqlen_stats_are_global = False


def test_training_log_uses_scheduled_microbatch_count_for_mtp():
    args = SimpleNamespace(
        timing_log_level=0,
        perform_rl_step=False,
        micro_batch_size=1,
        data_parallel_size=1,
        world_size=1,
        seq_length=8,
        freeze_all_layers=False,
        num_experts=None,
        mtp_num_layers=1,
        dsa_indexer_loss_coeff=None,
        log_interval=100,
    )

    with (
        mock.patch.object(training_mod, "get_args", return_value=args),
        mock.patch.object(training_mod, "get_timers", return_value=mock.MagicMock()),
        mock.patch.object(training_mod, "get_tensorboard_writer", return_value=None),
        mock.patch.object(training_mod, "get_wandb_writer", return_value=None),
        mock.patch.object(training_mod, "get_one_logger", return_value=None),
        mock.patch.object(training_mod, "get_energy_monitor", return_value=None),
        mock.patch.object(training_mod, "get_num_microbatches", return_value=8),
        mock.patch.object(
            training_mod, "reduce_max_stat_across_model_parallel_group", return_value=None
        ),
        mock.patch.object(training_mod.one_logger_utils, "track_app_tag"),
        mock.patch.object(
            training_mod.MTPLossLoggingHelper, "track_mtp_metrics"
        ) as track_mtp_metrics,
    ):
        training_mod.training_log(
            loss_dict={},
            total_loss_dict={},
            learning_rate=None,
            iteration=1,
            loss_scale=1.0,
            report_memory_flag=False,
            skipped_iter=0,
            grad_norm=None,
            params_norm=None,
            num_zeros_in_grad=None,
            max_attention_logit=None,
            num_microbatches=3,
        )

    assert track_mtp_metrics.call_args.args[0] == 1 / 3


def test_training_log_uses_full_hybrid_moe_layout_including_mtp():
    args = SimpleNamespace(
        timing_log_level=0,
        perform_rl_step=False,
        micro_batch_size=1,
        data_parallel_size=1,
        world_size=1,
        seq_length=8,
        freeze_all_layers=False,
        num_experts=8,
        moe_router_load_balancing_type=["seq_aux_loss"],
        moe_z_loss_coeff=None,
        moe_per_layer_logging=True,
        moe_layer_freq=1,
        hybrid_layer_pattern="CECE/WE",
        mtp_num_layers=1,
        dsa_indexer_loss_coeff=None,
        log_interval=100,
    )
    tracker = mock.MagicMock()
    tracker.report.return_value = ""

    with (
        mock.patch.object(training_mod, "get_args", return_value=args),
        mock.patch.object(training_mod, "get_timers", return_value=mock.MagicMock()),
        mock.patch.object(training_mod, "get_tensorboard_writer", return_value=None),
        mock.patch.object(training_mod, "get_wandb_writer", return_value=None),
        mock.patch.object(training_mod, "get_one_logger", return_value=None),
        mock.patch.object(training_mod, "get_energy_monitor", return_value=None),
        mock.patch.object(training_mod, "get_num_microbatches", return_value=8),
        mock.patch.object(
            training_mod, "reduce_max_stat_across_model_parallel_group", return_value=None
        ),
        mock.patch.object(training_mod.one_logger_utils, "track_app_tag"),
        mock.patch.object(training_mod, "get_moe_metrics_tracker", return_value=tracker),
        mock.patch.object(training_mod.MTPLossLoggingHelper, "track_mtp_metrics"),
    ):
        training_mod.training_log(
            loss_dict={},
            total_loss_dict={},
            learning_rate=None,
            iteration=1,
            loss_scale=1.0,
            report_memory_flag=False,
            skipped_iter=0,
            grad_norm=None,
            params_norm=None,
            num_zeros_in_grad=None,
            max_attention_logit=None,
        )

    assert tracker.report.call_args.kwargs["num_layers"] == 5
    assert tracker.report.call_args.kwargs["moe_layer_freq"] == [0, 1, 0, 1, 1]
    assert tracker.report.call_args.kwargs["mtp_num_layers"] is None


def test_training_log_preserves_indexer_groups_only_for_te_graphs():
    for cuda_graph_impl, expected_preserve_groups in (
        ("none", False),
        ("local", False),
        ("full_iteration", False),
        ("transformer_engine", True),
    ):
        args = SimpleNamespace(
            timing_log_level=0,
            perform_rl_step=False,
            micro_batch_size=1,
            data_parallel_size=1,
            world_size=1,
            seq_length=8,
            freeze_all_layers=False,
            num_experts=None,
            mtp_num_layers=None,
            dsa_indexer_loss_coeff=1.0,
            num_layers=4,
            csa_compress_ratios=[0, 4, 128, 0],
            cuda_graph_impl=cuda_graph_impl,
            log_interval=100,
        )

        with (
            mock.patch.object(training_mod, "get_args", return_value=args),
            mock.patch.object(training_mod, "get_timers", return_value=mock.MagicMock()),
            mock.patch.object(training_mod, "get_tensorboard_writer", return_value=None),
            mock.patch.object(training_mod, "get_wandb_writer", return_value=None),
            mock.patch.object(training_mod, "get_one_logger", return_value=None),
            mock.patch.object(training_mod, "get_energy_monitor", return_value=None),
            mock.patch.object(training_mod, "get_num_microbatches", return_value=8),
            mock.patch.object(
                training_mod, "reduce_max_stat_across_model_parallel_group", return_value=None
            ),
            mock.patch.object(training_mod.one_logger_utils, "track_app_tag"),
            mock.patch.object(
                training_mod.DSAIndexerLossLoggingHelper, "track_indexer_metrics"
            ) as track_indexer_metrics,
        ):
            training_mod.training_log(
                loss_dict={},
                total_loss_dict={},
                learning_rate=None,
                iteration=1,
                loss_scale=1.0,
                report_memory_flag=False,
                skipped_iter=0,
                grad_norm=None,
                params_norm=None,
                num_zeros_in_grad=None,
                max_attention_logit=None,
                pg_collection=ProcessGroupCollection(),
            )

        assert track_indexer_metrics.call_args.kwargs["loss_scale"] == 1 / 8
        assert (
            track_indexer_metrics.call_args.kwargs["preserve_groups"]
            is expected_preserve_groups
        )


def test_training_log_uses_scheduled_microbatch_count_for_dsa():
    args = SimpleNamespace(
        timing_log_level=0,
        perform_rl_step=False,
        micro_batch_size=1,
        data_parallel_size=1,
        world_size=1,
        seq_length=8,
        freeze_all_layers=False,
        num_experts=None,
        mtp_num_layers=None,
        dsa_indexer_loss_coeff=1.0,
        num_layers=4,
        csa_compress_ratios=[0, 4, 128, 0],
        cuda_graph_impl="none",
        log_interval=100,
    )

    with (
        mock.patch.object(training_mod, "get_args", return_value=args),
        mock.patch.object(training_mod, "get_timers", return_value=mock.MagicMock()),
        mock.patch.object(training_mod, "get_tensorboard_writer", return_value=None),
        mock.patch.object(training_mod, "get_wandb_writer", return_value=None),
        mock.patch.object(training_mod, "get_one_logger", return_value=None),
        mock.patch.object(training_mod, "get_energy_monitor", return_value=None),
        mock.patch.object(training_mod, "get_num_microbatches", return_value=8),
        mock.patch.object(
            training_mod, "reduce_max_stat_across_model_parallel_group", return_value=None
        ),
        mock.patch.object(training_mod.one_logger_utils, "track_app_tag"),
        mock.patch.object(
            training_mod.DSAIndexerLossLoggingHelper, "track_indexer_metrics"
        ) as track_indexer_metrics,
    ):
        training_mod.training_log(
            loss_dict={},
            total_loss_dict={},
            learning_rate=None,
            iteration=1,
            loss_scale=1.0,
            report_memory_flag=False,
            skipped_iter=0,
            grad_norm=None,
            params_norm=None,
            num_zeros_in_grad=None,
            max_attention_logit=None,
            pg_collection=ProcessGroupCollection(),
            num_microbatches=3,
        )

    assert track_indexer_metrics.call_args.kwargs["loss_scale"] == 1 / 3
