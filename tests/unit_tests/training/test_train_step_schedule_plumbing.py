# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Training-loop plumbing regression tests."""

import inspect
from types import SimpleNamespace
from unittest import mock

from megatron.core.enums import ModelType
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
        global_batch_size=1,
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


def test_train_step_drives_chunked_optimizer_state_offload_lifecycle():
    """MXFP8 staging must offload state first, then masters, through one API."""
    events = []

    class FakeOptimizer:
        config = SimpleNamespace(chunked_optimizer_state_offload=True)

        def offload_optimizer_state_for_forward(self, offload_master=True):
            events.append(f"offload:{offload_master}")

        def prefetch_optimizer_state_for_gradient_finalization(self):
            events.append("prefetch")

        def optimizer_state_offload_requires_pre_forward_param_sync(self):
            return False

        def zero_grad(self):
            events.append("zero_grad")

    args = SimpleNamespace(
        save_params_interval=None,
        save_activations_interval=None,
        save_tokens_per_expert_interval=None,
        save_wgrads_interval=None,
        save_dgrads_interval=None,
        reuse_grad_buf_for_mxfp8_param_ag=True,
        overlap_param_gather=True,
        seq_length=8,
        global_batch_size=1,
        micro_batch_size=1,
        decoder_seq_length=None,
        empty_unused_memory_level=0,
    )
    optimizer = FakeOptimizer()
    model = [
        SimpleNamespace(
            force_all_reduce=False,
            zero_grad_buffer=lambda: events.append("zero_grad_buffer"),
            remove_forward_pre_hook_handles=[],
        )
    ]
    config = SimpleNamespace(
        sequence_packing_scheduler=None,
        finalize_model_grads_func=lambda *args, **kwargs: events.append("finalize"),
    )

    def forward_backward(**kwargs):
        events.append("forward_backward")
        config.finalize_model_grads_func()
        return []

    with (
        mock.patch.object(training_mod, "get_args", return_value=args),
        mock.patch.object(training_mod, "get_timers", return_value=mock.MagicMock()),
        mock.patch.object(training_mod, "get_rerun_state_machine", return_value=_Rerun()),
        mock.patch.object(training_mod, "get_num_microbatches", return_value=1),
        mock.patch.object(training_mod, "has_nvidia_modelopt", False),
        mock.patch.object(training_mod, "get_moe_router_tracer", return_value=None),
    ):
        training_mod.train_step(
            forward_step_func=lambda *args, **kwargs: None,
            data_iterator=iter([]),
            model=model,
            optimizer=optimizer,
            opt_param_scheduler=None,
            config=config,
            forward_backward_func=forward_backward,
            iteration=0,
        )

    assert events == [
        "offload:False",
        "zero_grad_buffer",
        "zero_grad",
        "offload:True",
        "forward_backward",
        "prefetch",
        "finalize",
    ]


def test_train_step_rebinds_finalize_prefetch_without_nesting_optimizers():
    """A new optimizer must replace the prefetch binding while preserving the base hook."""

    events = []

    class FakeOptimizer:
        config = SimpleNamespace(chunked_optimizer_state_offload=True)

        def __init__(self, name):
            self.name = name

        def offload_optimizer_state_for_forward(self, offload_master=True):
            pass

        def prefetch_optimizer_state_for_gradient_finalization(self):
            events.append(f"prefetch:{self.name}")

        def optimizer_state_offload_requires_pre_forward_param_sync(self):
            return False

        def zero_grad(self):
            pass

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
    model = [SimpleNamespace(force_all_reduce=False, zero_grad_buffer=lambda: None)]

    def base_finalize_model_grads(*args, **kwargs):
        events.append("finalize")

    config = SimpleNamespace(
        sequence_packing_scheduler=None, finalize_model_grads_func=base_finalize_model_grads
    )

    def forward_backward(**kwargs):
        config.finalize_model_grads_func()
        return []

    optimizers = [FakeOptimizer("a"), FakeOptimizer("b")]
    with (
        mock.patch.object(training_mod, "get_args", return_value=args),
        mock.patch.object(training_mod, "get_timers", return_value=mock.MagicMock()),
        mock.patch.object(training_mod, "get_rerun_state_machine", side_effect=_Rerun),
        mock.patch.object(training_mod, "get_num_microbatches", return_value=1),
        mock.patch.object(training_mod, "has_nvidia_modelopt", False),
        mock.patch.object(training_mod, "get_moe_router_tracer", return_value=None),
    ):
        for optimizer in optimizers:
            training_mod.train_step(
                forward_step_func=lambda *args, **kwargs: None,
                data_iterator=iter([]),
                model=model,
                optimizer=optimizer,
                opt_param_scheduler=None,
                config=config,
                forward_backward_func=forward_backward,
                iteration=0,
            )

    assert events == ["prefetch:a", "finalize", "prefetch:b", "finalize"]
    wrapper = config.finalize_model_grads_func
    assert wrapper._chunked_optimizer_state_offload_wrapped_optimizer is optimizers[-1]
    assert (
        wrapper._chunked_optimizer_state_offload_base_finalize_model_grads_func
        is base_finalize_model_grads
    )


def test_mxfp8_staging_delegates_master_restore_to_distributed_optimizer():
    """The second MXFP8 staging pass delegates master restoration to its DistOpt entry."""

    events = []

    class FakeDistributedOptimizer:
        def ensure_master_weights_for_param_sync(self):
            raise AssertionError("train_step must not restore DistOpt masters separately")

        def _copy_main_params_to_param_buffer(self):
            events.append("distopt_stage")

    class FakeOptimizer:
        config = SimpleNamespace(chunked_optimizer_state_offload=True)

        def __init__(self):
            self.chained_optimizers = [SimpleNamespace(), FakeDistributedOptimizer()]

        def offload_optimizer_state_for_forward(self, offload_master=True):
            events.append(f"offload:{offload_master}")

        def prefetch_optimizer_state_for_gradient_finalization(self):
            events.append("prefetch")

        def optimizer_state_offload_requires_pre_forward_param_sync(self):
            return False

        def ensure_master_weights_for_param_sync(self):
            raise AssertionError("must not restore the whole optimizer chain")

        def zero_grad(self):
            events.append("zero_grad")

    args = SimpleNamespace(
        save_params_interval=None,
        save_activations_interval=None,
        save_tokens_per_expert_interval=None,
        save_wgrads_interval=None,
        save_dgrads_interval=None,
        reuse_grad_buf_for_mxfp8_param_ag=True,
        overlap_param_gather=True,
        seq_length=8,
        global_batch_size=1,
        micro_batch_size=1,
        decoder_seq_length=None,
        empty_unused_memory_level=0,
    )
    model = [
        SimpleNamespace(
            force_all_reduce=False,
            zero_grad_buffer=lambda: events.append("zero_grad_buffer"),
            remove_forward_pre_hook_handles=[object()],
        )
    ]
    config = SimpleNamespace(
        sequence_packing_scheduler=None,
        finalize_model_grads_func=lambda *args, **kwargs: events.append("finalize"),
    )

    def forward_backward(**kwargs):
        events.append("forward_backward")
        config.finalize_model_grads_func()
        return []

    with (
        mock.patch.object(training_mod, "DistributedOptimizer", FakeDistributedOptimizer),
        mock.patch.object(training_mod, "get_args", return_value=args),
        mock.patch.object(training_mod, "get_timers", return_value=mock.MagicMock()),
        mock.patch.object(training_mod, "get_rerun_state_machine", return_value=_Rerun()),
        mock.patch.object(training_mod, "get_num_microbatches", return_value=1),
        mock.patch.object(training_mod, "has_nvidia_modelopt", False),
        mock.patch.object(training_mod, "get_moe_router_tracer", return_value=None),
    ):
        training_mod.train_step(
            forward_step_func=lambda *args, **kwargs: None,
            data_iterator=iter([]),
            model=model,
            optimizer=FakeOptimizer(),
            opt_param_scheduler=None,
            config=config,
            forward_backward_func=forward_backward,
            iteration=0,
        )

    assert events == [
        "offload:False",
        "zero_grad_buffer",
        "zero_grad",
        "distopt_stage",
        "offload:True",
        "forward_backward",
        "prefetch",
        "finalize",
    ]


def test_train_step_pre_forward_sync_only_uses_layerwise_bucket_subset():
    """Master-dependent FP8 sync must not dispatch sibling DistOpt bucket groups."""
    events = []

    class FakeOptimizer:
        config = SimpleNamespace(chunked_optimizer_state_offload=True)

        def optimizer_state_offload_requires_pre_forward_param_sync(self):
            return True

        def ensure_master_weights_for_pre_forward_param_sync(self):
            events.append("ensure_master")

        def ensure_master_weights_for_param_sync(self):
            raise AssertionError("train_step must use the pre-forward subset restore API")

        def start_param_sync_for_bucket_group_subset(self, force_sync=False):
            assert force_sync
            events.append("layerwise_param_sync")

        def offload_optimizer_state_for_forward(self, offload_master=True):
            events.append("offload")

        def prefetch_optimizer_state_for_gradient_finalization(self):
            events.append("prefetch")

        def zero_grad(self):
            events.append("zero_grad")

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
    whole_model_sync = mock.Mock(side_effect=AssertionError("must not sync all DDP buckets"))
    model = [
        SimpleNamespace(
            force_all_reduce=False,
            zero_grad_buffer=lambda: events.append("zero_grad_buffer"),
            start_param_sync=whole_model_sync,
        )
    ]
    config = SimpleNamespace(
        sequence_packing_scheduler=None,
        finalize_model_grads_func=lambda *args, **kwargs: events.append("finalize"),
    )

    def forward_backward(**kwargs):
        events.append("forward_backward")
        config.finalize_model_grads_func()
        return []

    with (
        mock.patch.object(training_mod, "get_args", return_value=args),
        mock.patch.object(training_mod, "get_timers", return_value=mock.MagicMock()),
        mock.patch.object(training_mod, "get_rerun_state_machine", return_value=_Rerun()),
        mock.patch.object(training_mod, "get_num_microbatches", return_value=1),
        mock.patch.object(training_mod, "has_nvidia_modelopt", False),
        mock.patch.object(training_mod, "get_moe_router_tracer", return_value=None),
    ):
        training_mod.train_step(
            forward_step_func=lambda *args, **kwargs: None,
            data_iterator=iter([]),
            model=model,
            optimizer=FakeOptimizer(),
            opt_param_scheduler=None,
            config=config,
            forward_backward_func=forward_backward,
            iteration=0,
        )

    whole_model_sync.assert_not_called()
    assert events == [
        "zero_grad_buffer",
        "zero_grad",
        "ensure_master",
        "layerwise_param_sync",
        "offload",
        "forward_backward",
        "prefetch",
        "finalize",
    ]


def test_train_step_wraps_sequence_packing_after_rerun_check():
    """The rerun machine must see the original iterator before dynamic-CP packs it."""
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
    original_iterator = object()
    # Non-TP0 ranks legitimately receive no local packed iterator. None must not be
    # mistaken for "not wrapped yet" when the rerun state machine repeats the step.
    packed_iterator = None
    config = SimpleNamespace(sequence_packing_scheduler="default_dynamic_cp")
    captured = {}
    forwarded_iterators = []
    model = [SimpleNamespace(force_all_reduce=False, zero_grad_buffer=lambda: None)]
    rerun = mock.MagicMock()
    rerun.should_run_forward_backward.side_effect = [True, True, False]
    rerun.should_checkpoint_and_exit.return_value = (False, True, 0)

    def forward_backward(**kwargs):
        captured.update(kwargs)
        forwarded_iterators.append(kwargs["data_iterator"])
        return []

    with (
        mock.patch.object(training_mod, "get_args", return_value=args),
        mock.patch.object(training_mod, "get_timers", return_value=mock.MagicMock()),
        mock.patch.object(training_mod, "get_rerun_state_machine", return_value=rerun),
        mock.patch.object(training_mod, "get_num_microbatches", return_value=1),
        mock.patch.object(training_mod, "has_nvidia_modelopt", False),
        mock.patch.object(
            training_mod, "wrap_data_iterator", return_value=(packed_iterator, 3, 12.0, 34.0)
        ) as wrap_data_iterator,
    ):
        result = training_mod.train_step(
            forward_step_func=lambda *a, **k: None,
            data_iterator=original_iterator,
            model=model,
            optimizer=SimpleNamespace(zero_grad=lambda: None),
            opt_param_scheduler=None,
            config=config,
            forward_backward_func=forward_backward,
            iteration=0,
        )

    assert rerun.should_run_forward_backward.call_args_list[0].args[0] is original_iterator
    assert rerun.should_run_forward_backward.call_args_list[1].args[0] is packed_iterator
    assert rerun.should_run_forward_backward.call_args_list[2].args[0] is packed_iterator
    wrap_data_iterator.assert_called_once_with(original_iterator, config, 1)
    assert forwarded_iterators == [packed_iterator, packed_iterator]
    assert captured["num_microbatches"] == 3
    assert result[-3:] == (3, 12.0, 34.0)


def test_config_container_forwards_layer_wise_optimizer_to_model_builder():
    """The config-container path must preserve Muon's layer-wise DDP routing flag."""
    args = SimpleNamespace(
        skip_train=True,
        perform_rl_step=False,
        no_load_optim=True,
        logits_save_dir=None,
        logits_load_dir=None,
        moe_use_upcycling=False,
        load=None,
        pretrained_checkpoint=None,
        data_parallel_size=1,
        micro_batch_size=1,
        fp16=False,
        ckpt_convert_format=None,
    )
    builder = mock.Mock()
    wrapped_model = mock.Mock()
    unwrapped_model = SimpleNamespace()
    builder.build_distributed_models.return_value = [wrapped_model]
    builder_cls = mock.Mock(return_value=builder)
    model_config = mock.Mock()
    model_config.get_builder_cls.return_value = builder_cls
    cfg = SimpleNamespace(
        model=model_config,
        profiling=mock.Mock(),
        ddp=mock.Mock(),
        optimizer=SimpleNamespace(
            overlap_param_gather_with_optimizer_step=False,
            use_layer_wise_distributed_optimizer=True,
        ),
        dist=SimpleNamespace(use_megatron_fsdp=False, use_torch_fsdp2=False),
        rng=SimpleNamespace(data_parallel_random_init=False),
    )
    pg_collection = mock.Mock()

    with (
        mock.patch.object(training_mod, "get_args", return_value=args),
        mock.patch.object(training_mod, "get_timers", return_value=mock.Mock()),
        mock.patch.object(training_mod, "get_one_logger", return_value=None),
        mock.patch.object(training_mod, "unwrap_model", return_value=[unwrapped_model]),
        mock.patch.object(training_mod, "get_num_microbatches", return_value=1),
        mock.patch.object(training_mod, "get_current_global_batch_size", return_value=1),
        mock.patch.object(training_mod.mpu, "model_parallel_is_initialized", return_value=False),
        mock.patch("megatron.training.utils.start_memory_history_recording"),
    ):
        model, optimizer, scheduler = training_mod.setup_model_and_optimizer(
            ModelType.encoder_or_decoder, cfg_container=cfg, pg_collection=pg_collection
        )

    assert model == [wrapped_model]
    assert optimizer is None
    assert scheduler is None
    builder_cls.assert_called_once_with(model_config)
    builder.build_distributed_models.assert_called_once_with(
        pg_collection=pg_collection,
        ddp_config=cfg.ddp,
        overlap_param_gather_with_optimizer_step=False,
        use_megatron_fsdp=False,
        use_torch_fsdp2=False,
        wrap_with_ddp=False,
        data_parallel_random_init=False,
        use_layer_wise_distributed_optimizer=True,
    )


def test_layerwise_wrapper_uses_ddp_config_as_single_layout_source():
    """Compact and padded LayerWise layouts must both run through layout computation."""

    class FakeDDP:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    param = SimpleNamespace(requires_grad=True)
    chunk = SimpleNamespace(parameters=lambda: [param])
    layout = object()
    dp_cp_group = object()
    expert_dp_group = object()
    pg_collection = SimpleNamespace(dp_cp=dp_cp_group, expt_dp=expert_dp_group)
    ddp_config = SimpleNamespace(
        bucket_size=17, use_distributed_optimizer=False, use_layer_wise_param_layout=False
    )

    # The layout choice lives on ddp_config. A second wrapper argument can disagree with
    # it and caused the compact (False) path to skip layout/tag setup during the sync.
    assert (
        "use_layer_wise_param_layout"
        not in inspect.signature(training_mod.wrap_model_chunks_with_ddp).parameters
    )

    with (
        mock.patch.object(training_mod, "DDP", FakeDDP),
        mock.patch.object(training_mod, "get_pg_size", return_value=8),
        mock.patch.object(training_mod, "tag_params_for_buffer_routing") as tag_params,
        mock.patch.object(
            training_mod.LayerWiseDistributedOptimizer,
            "compute_full_param_layout",
            return_value=layout,
        ) as compute_layout,
    ):
        wrapped = training_mod.wrap_model_chunks_with_ddp(
            [chunk],
            config=object(),
            ddp_config=ddp_config,
            use_layer_wise_distributed_optimizer=True,
            DP=FakeDDP,
            pg_collection=pg_collection,
        )

    assert ddp_config.use_distributed_optimizer is True
    tag_params.assert_called_once_with([chunk])
    compute_layout.assert_called_once_with(
        [param], 17, 8, ddp_config, expert_data_parallel_world_size=8
    )
    assert wrapped[0].kwargs["full_param_layout"] is layout


def test_dynamic_cp_cuda_graph_upper_bound_uses_dp_cp_and_sp_padding():
    args = SimpleNamespace(
        seq_length=1000,
        use_varlen_dataset=False,
        sft=False,
        context_parallel_size=4,
        dynamic_context_parallel=True,
        data_parallel_size=8,
        tensor_model_parallel_size=2,
        sequence_parallel=True,
    )

    # ceil(1000 / (DP=8 * CP=4 * 2 * SP=2)) * 128
    assert training_mod._get_thd_sequence_length_upper_bound(args) == 1024
