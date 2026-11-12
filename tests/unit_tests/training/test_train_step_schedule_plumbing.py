# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Training-loop plumbing regression tests."""

import inspect
from contextlib import contextmanager
from types import SimpleNamespace
from unittest import mock

import pytest

from megatron.core.enums import ModelType
from megatron.core.process_groups_config import MultiModuleProcessGroupCollection
from megatron.training import training as training_mod
from megatron.training.state import TrainState


@pytest.fixture(autouse=True)
def training_config(run_config):
    run_config.train.micro_batch_size = 1
    run_config.train.global_batch_size = 1


class _Rerun:
    """Run the forward/backward body once, then ask train_step to exit before optimizer.step."""

    _ran = False

    def should_run_forward_backward(self, data_iterator):
        run, self._ran = not self._ran, True
        return run

    def should_checkpoint_and_exit(self):
        return False, True, 0  # (checkpoint, exit, code)


class _RerunThroughOptimizer(_Rerun):
    """Run one forward/backward body and continue through the optimizer step."""

    def should_checkpoint_and_exit(self):
        return False, False, 0


def _run(
    *, args_overrides=None, model=None, optimizer=None, config=None, return_result=False, **kwargs
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
        global_batch_size=1,
        micro_batch_size=1,
        decoder_seq_length=None,
        empty_unused_memory_level=0,
    )
    for name, value in (args_overrides or {}).items():
        setattr(args, name, value)
    captured = {}
    model = model or [SimpleNamespace(force_all_reduce=False, zero_grad_buffer=lambda: None)]
    optimizer = optimizer or SimpleNamespace(zero_grad=lambda: None, chained_optimizers=[])
    with (
        mock.patch.object(training_mod, "get_args", return_value=args),
        mock.patch.object(training_mod, "get_timers", return_value=mock.MagicMock()),
        mock.patch.object(training_mod, "get_rerun_state_machine", return_value=_Rerun()),
        mock.patch.object(training_mod, "get_num_microbatches", return_value=1),
        mock.patch.object(training_mod, "has_nvidia_modelopt", False),
    ):
        result = training_mod.train_step(
            forward_step_func=lambda *a, **k: None,
            data_iterator=iter([]),
            model=model,
            optimizer=optimizer,
            opt_param_scheduler=None,
            config=config or SimpleNamespace(),
            forward_backward_func=lambda **kw: captured.update(kw) or [],
            iteration=0,
            **kwargs,
        )
    # Dev's 12-field result with main's scheduled microbatch count appended.
    assert len(result) == 13
    assert result[-2] is None  # early exit before optimizer/sample accounting
    captured['returned_num_microbatches'] = result[-1]
    return result if return_result else captured


def test_train_step_early_exit_keeps_normal_result_field_order():
    # Dev's 12-field train_step result, with main's scheduled microbatch count
    # (ea99fdfce) appended as the 13th field.
    result = _run(return_result=True)
    assert len(result) == 13
    assert result[:5] == ({}, True, False, True, 0)
    assert result[5:8] == (None, None, None)
    assert result[8] == 1  # num_microbatches, not an exit-path placeholder
    assert result[11] is None  # samples_seen is not computed before early exit
    assert result[12] == 1  # main's appended scheduled microbatch count


def test_train_step_returns_scheduled_microbatch_count():
    packed_iterator = iter([1, 2, 3])
    with (
        mock.patch.object(
            training_mod, 'wrap_data_iterator', return_value=(packed_iterator, 3, 60, 1200)
        ),
        mock.patch.object(training_mod, 'get_attr_wrapped_model', return_value=object()),
        mock.patch.object(training_mod, 'set_seqlen_stats_in_iteration'),
    ):
        captured = _run(config=SimpleNamespace(sequence_packing_scheduler='default_dynamic_cp'))
    assert captured['num_microbatches'] == captured['returned_num_microbatches'] == 3
    assert captured['data_iterator'] is packed_iterator


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
        ddp_config = SimpleNamespace(
            reuse_grad_buf_for_mxfp8_param_ag=True, overlap_param_gather=True
        )

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


@pytest.mark.parametrize("multi_module_schedule_carrier", [False, True])
def test_train_step_wraps_sequence_packing_after_rerun_check(multi_module_schedule_carrier):
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
    # wrap_data_iterator takes pg_collection from the model (main ea99fdfce), so the
    # mock has to carry one for the plumbing assertion below.
    model_pg_collection = object()
    # The schedule carrier can be a multi-module collection, which has no dp_cp group of
    # its own.
    schedule_carrier = (
        mock.MagicMock(spec=MultiModuleProcessGroupCollection)
        if multi_module_schedule_carrier
        else None
    )
    model = [
        SimpleNamespace(
            force_all_reduce=False, zero_grad_buffer=lambda: None, pg_collection=model_pg_collection
        )
    ]
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
            pg_collection=schedule_carrier,
        )

    assert rerun.should_run_forward_backward.call_args_list[0].args[0] is original_iterator
    assert rerun.should_run_forward_backward.call_args_list[1].args[0] is packed_iterator
    assert rerun.should_run_forward_backward.call_args_list[2].args[0] is packed_iterator
    wrap_data_iterator.assert_called_once_with(
        original_iterator, config, 1, pg_collection=model_pg_collection, prefetch_next=False
    )
    assert forwarded_iterators == [packed_iterator, packed_iterator]
    assert captured["num_microbatches"] == 3
    assert len(result) == 13
    assert result[8:] == (3, 12.0, 34.0, None, 3)


def _schedule_lookahead(iteration, **loader):
    """``prefetch_next`` as one train_step passes it to wrap_data_iterator."""
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
        **loader,
    )
    model = [
        SimpleNamespace(
            force_all_reduce=False, zero_grad_buffer=lambda: None, pg_collection=object()
        )
    ]
    with (
        mock.patch.object(training_mod, "get_args", return_value=args),
        mock.patch.object(training_mod, "get_timers", return_value=mock.MagicMock()),
        mock.patch.object(training_mod, "get_rerun_state_machine", return_value=_Rerun()),
        mock.patch.object(training_mod, "get_num_microbatches", return_value=1),
        mock.patch.object(training_mod, "has_nvidia_modelopt", False),
        mock.patch.object(
            training_mod, "wrap_data_iterator", return_value=(None, 1, 8.0, 64.0)
        ) as wrap_data_iterator,
    ):
        training_mod.train_step(
            forward_step_func=lambda *a, **k: None,
            data_iterator=object(),
            model=model,
            optimizer=SimpleNamespace(zero_grad=lambda: None),
            opt_param_scheduler=None,
            config=SimpleNamespace(sequence_packing_scheduler="default_dynamic_cp"),
            forward_backward_func=lambda **kwargs: [],
            iteration=iteration,
        )
    wrap_data_iterator.assert_called_once()
    return wrap_data_iterator.call_args.kwargs["prefetch_next"]


@pytest.mark.parametrize(
    "train_iters,iteration,dataloader_type,num_workers,use_varlen_dataset,expected",
    [
        # Another step follows and the data comes from a loader that can be read ahead.
        (10, 0, "single", 2, False, True),
        (10, 8, "cyclic", 2, False, True),
        (10, 3, "single", 0, True, True),  # The varlen dataset is read in the main process.
        # Nothing follows the last iteration.
        (10, 9, "single", 2, False, False),
        (1, 0, "single", 2, True, False),
        # A run without an iteration count, or a step without an iteration number.
        (None, 0, "single", 2, False, False),
        (10, None, "single", 2, False, False),
        # In-process loading of a dataset other than the varlen one.
        (10, 0, "single", 0, False, False),
        # A loader that the training loop does not own.
        (10, 0, "external", 2, True, False),
    ],
)
def test_train_step_plans_the_next_schedule_only_when_the_next_batch_can_be_read_ahead(
    run_config, train_iters, iteration, dataloader_type, num_workers, use_varlen_dataset, expected
):
    run_config.train.train_iters = train_iters
    assert (
        _schedule_lookahead(
            iteration,
            dataloader_type=dataloader_type,
            num_workers=num_workers,
            use_varlen_dataset=use_varlen_dataset,
        )
        is expected
    )


def test_pretrain_calls_the_runtime_dcp_startup_hook_right_after_the_model_is_built():
    """Call-site pin: executing pretrain() needs a whole job, so its statements are read."""
    import ast
    import textwrap

    source = textwrap.dedent(inspect.getsource(training_mod.pretrain))
    statements = [ast.unparse(statement) for statement in ast.parse(source).body[0].body]
    built = statements.index("model_cfg = get_model_config(model[0])")
    # A top-level statement of pretrain(): no condition can skip it on any rank.
    assert statements[built + 1] == "_validate_dynamic_cp_runtime_config(model, model_cfg)"
    # Before the data iterators exist, so a rejected file costs no data loading.
    data = [
        index
        for index, statement in enumerate(statements)
        if "build_train_valid_test_data_iterators" in statement
    ]
    assert data and built + 1 < min(data)


def _runtime_spec(**entries):
    """A validated runtime specification, as ``validate_dynamic_cp_runtime`` returns it."""
    spec = {
        "type": "dynamic_usp",
        "model": "qwen3_h100_p2p",
        "hardware": "h100",
        "nvlink_domain_size": 8,
        "geometry": {"allowed_cp_sizes": (1, 2, 4, 8), "max_group_seqs": 64},
        "async": True,
        "config_path": "/path/to/runtime.json",
    }
    spec.update(entries)
    return spec


def _start_up_hook(spec):
    """Run the start-up hook of pretrain() on ``spec``; return the rank-0 print mock."""
    config = SimpleNamespace(dynamic_cp_runtime_config="/path/to/runtime.json")
    with (
        mock.patch.object(training_mod, "validate_dynamic_cp_runtime", return_value=spec),
        mock.patch.object(training_mod, "print_rank_0") as print_rank_0,
    ):
        try:
            training_mod._validate_dynamic_cp_runtime_config(
                [SimpleNamespace(pg_collection="model collection")], config
            )
        except ValueError:
            # A rejected run says nothing about a selected scheduler.
            print_rank_0.assert_not_called()
            raise
    return print_rank_0


def test_startup_hook_prints_one_line_about_the_selected_scheduler_on_rank_zero():
    print_rank_0 = _start_up_hook(_runtime_spec())
    print_rank_0.assert_called_once_with(
        "Runtime DCP scheduler: type=dynamic_usp, cost model=qwen3_h100_p2p, "
        "hardware=h100 (NVLink domain: 8 consecutive ranks), "
        "CP group sizes=[1, 2, 4, 8], at most 64 sequences per CP group, async=on, "
        "runtime file=/path/to/runtime.json"
    )


def test_startup_hook_prints_nothing_when_validation_fails():
    config = SimpleNamespace(dynamic_cp_runtime_config="/path/to/runtime.json")
    with (
        mock.patch.object(
            training_mod, "validate_dynamic_cp_runtime", side_effect=ValueError("bad file")
        ),
        mock.patch.object(training_mod, "print_rank_0") as print_rank_0,
    ):
        with pytest.raises(ValueError, match="bad file"):
            training_mod._validate_dynamic_cp_runtime_config(
                [SimpleNamespace(pg_collection="model collection")], config
            )
    print_rank_0.assert_not_called()


_BATCH_RULE = (
    "The runtime DCP scheduler's 'dynamic_usp' cost model (qwen3_h100_p2p) was measured with "
    "256 samples per global batch and prices no other batch size: {}. Every global batch that "
    "training or evaluation schedules must hold 256 samples"
)
_EVAL_FAULT = (
    "--eval-global-batch-size (by default --global-batch-size) is {} and evaluation is "
    "enabled (--eval-iters {})"
)


def test_startup_hook_accepts_the_global_batch_size_the_cost_model_requires(run_config):
    run_config.train.global_batch_size = 256
    # Evaluation is enabled by default and then uses the training global batch size.
    assert run_config.validation.eval_iters == 100
    assert run_config.validation.eval_global_batch_size is None
    _start_up_hook(_runtime_spec(required_global_batch_size=256)).assert_called_once()
    run_config.validation.eval_global_batch_size = 256
    _start_up_hook(_runtime_spec(required_global_batch_size=256)).assert_called_once()


def test_startup_hook_consults_no_batch_size_for_a_cost_model_without_the_rule(run_config):
    run_config.train.global_batch_size = 24
    run_config.validation.eval_global_batch_size = 8
    with mock.patch.object(training_mod, "get_run_config", side_effect=AssertionError):
        _start_up_hook(_runtime_spec()).assert_called_once()


@pytest.mark.parametrize(
    "train,evaluation,eval_iters,faults",
    [
        (128, 256, 100, "--global-batch-size is 128"),
        (128, None, 0, "--global-batch-size is 128"),
        (256, 128, 100, _EVAL_FAULT.format(128, 100)),
        (256, 128, [10, None], _EVAL_FAULT.format(128, "[10, None]")),
        (128, None, 100, "--global-batch-size is 128; " + _EVAL_FAULT.format(128, 100)),
        (64, 128, 10, "--global-batch-size is 64; " + _EVAL_FAULT.format(128, 10)),
    ],
)
def test_startup_hook_names_the_batch_size_argument_the_cost_model_does_not_price(
    run_config, train, evaluation, eval_iters, faults
):
    run_config.train.global_batch_size = train
    run_config.validation.eval_global_batch_size = evaluation
    run_config.validation.eval_iters = eval_iters
    with pytest.raises(ValueError) as caught:
        _start_up_hook(_runtime_spec(required_global_batch_size=256))
    assert str(caught.value) == _BATCH_RULE.format(faults)


@pytest.mark.parametrize("eval_iters", [0, None, [0, None]])
def test_startup_hook_ignores_the_evaluation_batch_size_when_no_evaluation_runs(
    run_config, eval_iters
):
    run_config.train.global_batch_size = 256
    run_config.validation.eval_global_batch_size = 128
    run_config.validation.eval_iters = eval_iters
    _start_up_hook(_runtime_spec(required_global_batch_size=256)).assert_called_once()
    # Full validation evaluates whatever --eval-iters says.
    run_config.validation.full_validation = True
    with pytest.raises(ValueError) as caught:
        _start_up_hook(_runtime_spec(required_global_batch_size=256))
    assert str(caught.value) == _BATCH_RULE.format(
        "--eval-global-batch-size (by default --global-batch-size) is 128 and evaluation is "
        f"enabled (--eval-iters {eval_iters}, --full-validation)"
    )


def test_startup_hook_checks_only_evaluation_when_training_is_skipped(run_config):
    run_config.validation.skip_train = True
    run_config.train.global_batch_size = 128
    run_config.validation.eval_global_batch_size = 256
    _start_up_hook(_runtime_spec(required_global_batch_size=256)).assert_called_once()
    run_config.validation.eval_global_batch_size = 64
    with pytest.raises(ValueError) as caught:
        _start_up_hook(_runtime_spec(required_global_batch_size=256))
    assert str(caught.value) == _BATCH_RULE.format(_EVAL_FAULT.format(64, 100))


@pytest.fixture
def microbatches(run_config):
    """The real batch-size calculator of a run with micro batch 1 on 32 data-parallel ranks."""
    from megatron.core import num_microbatches_calculator as calculator

    def start(global_batch_size=None, schedule=None, decrease=False, consumed_samples=0):
        run_config.train.global_batch_size = global_batch_size
        run_config.train.step_batch_size_schedule = schedule
        run_config.train.decrease_batch_size_if_needed = decrease
        run_config.validation.eval_iters = 0
        calculator.unset_num_microbatches_calculator()
        calculator.init_num_microbatches_calculator(
            rank=1,
            global_batch_size=global_batch_size,
            micro_batch_size=1,
            data_parallel_size=32,
            decrease_batch_size_if_needed=decrease,
            step_batch_size_schedule=schedule,
        )
        calculator.update_num_microbatches(consumed_samples, consistency_check=False)

    yield start
    calculator.unset_num_microbatches_calculator()


def test_startup_hook_reads_a_batch_size_schedule_from_the_calculator(microbatches):
    spec = _runtime_spec(required_global_batch_size=256)
    # One size throughout. Without --global-batch-size the configured value is not the
    # running one (it resolves to micro batch times data-parallel size), so it is not read.
    microbatches(global_batch_size=32, schedule="0:256")
    _start_up_hook(spec).assert_called_once()
    # A ramp-up starts below the measured size and would stop the first step.
    microbatches(global_batch_size=32, schedule="0:128 1000:256")
    with pytest.raises(ValueError) as caught:
        _start_up_hook(spec)
    assert str(caught.value) == _BATCH_RULE.format(
        "--step-batch-size-schedule '0:128 1000:256' trains with 128 samples per global batch "
        "now and up to 256 later"
    )
    # Resumed past the last threshold, the same schedule only runs the measured size.
    microbatches(global_batch_size=32, schedule="0:128 1000:256", consumed_samples=1024)
    _start_up_hook(spec).assert_called_once()
    # A later, larger size is known at start-up as well.
    microbatches(global_batch_size=32, schedule="0:256 1000:512")
    with pytest.raises(ValueError, match="trains with 256 samples per global batch now and up to"):
        _start_up_hook(spec)


def test_startup_hook_reads_a_rounded_down_batch_size_from_the_calculator(microbatches):
    spec = _runtime_spec(required_global_batch_size=256)
    # 270 is rounded down to 256, a multiple of micro batch times data-parallel size.
    microbatches(global_batch_size=270, decrease=True)
    _start_up_hook(spec).assert_called_once()
    microbatches(global_batch_size=256, decrease=True)
    _start_up_hook(spec).assert_called_once()
    microbatches(global_batch_size=300, decrease=True)
    with pytest.raises(ValueError) as caught:
        _start_up_hook(spec)
    assert str(caught.value) == _BATCH_RULE.format(
        "--global-batch-size 300 with --decrease-batch-size-if-needed trains with 288 samples "
        "per global batch"
    )


def test_evaluate_schedules_with_the_models_own_process_groups(run_config):
    """Evaluation passes the model's groups to wrap_data_iterator, as train_step does."""
    import torch

    run_config.validation.eval_global_batch_size = 4
    run_config.validation.eval_micro_batch_size = 1
    run_config.validation.eval_iters = 1
    args = SimpleNamespace(
        eval_global_batch_size=4,
        eval_micro_batch_size=1,
        data_parallel_size=2,
        gtp_weight_remat_size=1,
        cuda_graph_impl="none",
        moe_expert_rank_capacity_factor=None,
        seq_length=8,
        decoder_seq_length=None,
        eval_iters=1,
        empty_unused_memory_level=0,
        consumed_valid_samples=0,
        exit_duration_in_mins=None,
        sft=False,
    )
    config = SimpleNamespace(sequence_packing_scheduler="default_dynamic_cp")
    groups = SimpleNamespace(pp=object(), dp_cp=object(), dp_cp_gtp_remat=None)
    data_iterator = iter([])
    tensor = torch.tensor

    def cpu_tensor(*args, **kwargs):
        kwargs["device"] = "cpu"
        return tensor(*args, **kwargs)

    losses = [[{"lm loss": tensor([2.0, 1.0])}]]
    with (
        mock.patch.object(training_mod, "get_args", return_value=args),
        mock.patch.object(training_mod, "get_train_state", return_value=TrainState()),
        mock.patch.object(training_mod, "get_timers", return_value=mock.MagicMock()),
        mock.patch.object(training_mod, "get_rerun_state_machine", return_value=mock.Mock()),
        mock.patch.object(
            training_mod, "get_forward_backward_func", return_value=mock.Mock(side_effect=losses)
        ),
        mock.patch.object(training_mod, "get_attr_wrapped_model", return_value=groups),
        mock.patch.object(training_mod, "is_pp_last_stage", return_value=True),
        mock.patch.object(training_mod, "has_nvidia_modelopt", False),
        mock.patch.object(training_mod, "ft_integration"),
        mock.patch.object(training_mod, "_otel_managed_span", return_value=mock.MagicMock()),
        mock.patch.object(
            training_mod, "wrap_data_iterator", return_value=(iter([]), 1, 0, 0)
        ) as wrap_data_iterator,
        mock.patch.object(torch, "tensor", side_effect=cpu_tensor),
        mock.patch.object(torch.distributed, "all_reduce"),
    ):
        training_mod.evaluate(lambda *a, **kw: None, data_iterator, [mock.Mock()], None, config)
    # 4 samples / (micro batch 1 * DP 2) microbatches; no global-group fallback inside wrap.
    wrap_data_iterator.assert_called_once_with(data_iterator, config, 2, pg_collection=groups)


@pytest.mark.parametrize("model_pg_collection", ["model collection", None])
def test_runtime_dcp_config_is_validated_with_the_models_own_process_groups(model_pg_collection):
    """Start-up validation never uses the schedule carrier, which may be multi-module."""
    config = SimpleNamespace(dynamic_cp_runtime_config="/path/to/runtime.json")
    model = [SimpleNamespace(pg_collection=model_pg_collection)]
    with (
        mock.patch.object(
            training_mod, "validate_dynamic_cp_runtime", return_value=_runtime_spec()
        ) as validate,
        mock.patch.object(
            training_mod.ProcessGroupCollection,
            "use_mpu_process_groups",
            return_value="global collection",
        ) as use_global_groups,
    ):
        training_mod._validate_dynamic_cp_runtime_config(model, config)
    validate.assert_called_once_with(config, model_pg_collection or "global collection")
    assert use_global_groups.call_count == (model_pg_collection is None)


def test_models_without_a_runtime_dcp_config_skip_the_startup_validation():
    with mock.patch.object(training_mod, "validate_dynamic_cp_runtime") as validate:
        # No process groups are looked up either: this model object has none.
        training_mod._validate_dynamic_cp_runtime_config(
            [SimpleNamespace()], SimpleNamespace(dynamic_cp_runtime_config=None)
        )
        training_mod._validate_dynamic_cp_runtime_config([SimpleNamespace()], SimpleNamespace())
    validate.assert_not_called()


@pytest.mark.parametrize(
    ("dsa_indexer_loss_coeff", "expect_tracker_init"), [(0.1, True), (None, False)]
)
@pytest.mark.parametrize("ddp_layout", [False, True])
def test_config_container_forwards_layer_wise_optimizer_to_model_builder(
    dsa_indexer_loss_coeff, expect_tracker_init, ddp_layout
):
    """The config-container path must preserve Muon's layer-wise DDP routing flag."""
    args = SimpleNamespace(
        skip_train=True,
        freeze_all_layers=False,
        freeze_base_model_for_mtp=False,
        perform_rl_step=False,
        no_load_optim=True,
        logits_save_dir=None,
        logits_load_dir=None,
        moe_use_upcycling=False,
        load=None,
        pretrained_checkpoint=None,
        data_parallel_size=1,
        gtp_weight_remat_size=1,
        micro_batch_size=1,
        fp16=False,
        ckpt_convert_format=None,
        dsa_indexer_loss_coeff=dsa_indexer_loss_coeff,
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
        ddp=SimpleNamespace(use_layer_wise_param_layout=ddp_layout, use_megatron_fsdp=False),
        optimizer=SimpleNamespace(
            overlap_param_gather_with_optimizer_step=False,
            use_layer_wise_distributed_optimizer=True,
            use_layer_wise_param_layout=False,
        ),
        dist=SimpleNamespace(use_megatron_fsdp=False, use_torch_fsdp2=False),
        rng=SimpleNamespace(data_parallel_random_init=False),
        rl=SimpleNamespace(perform_rl_step=False),
        validation=SimpleNamespace(skip_train=True),
    )
    pg_collection = mock.Mock()

    with (
        mock.patch.object(training_mod, "get_args", return_value=args),
        mock.patch("megatron.post_training.checkpointing.get_args", return_value=args),
        mock.patch.object(training_mod, "get_timers", return_value=mock.Mock()),
        mock.patch.object(training_mod, "get_one_logger", return_value=None),
        mock.patch.object(training_mod, "get_train_state", return_value=TrainState()),
        mock.patch.object(training_mod, "unwrap_model", return_value=[unwrapped_model]),
        mock.patch.object(training_mod, "get_num_microbatches", return_value=1),
        mock.patch.object(training_mod, "get_current_global_batch_size", return_value=1),
        mock.patch.object(training_mod.mpu, "model_parallel_is_initialized", return_value=False),
        mock.patch.object(training_mod, "initialize_dsa_metric_tracker") as initialize_tracker,
        mock.patch("megatron.training.utils.start_memory_history_recording"),
    ):
        model, optimizer, scheduler = training_mod.setup_model_and_optimizer(
            ModelType.encoder_or_decoder, cfg_container=cfg, pg_collection=pg_collection
        )

    assert model == [wrapped_model]
    assert optimizer is None
    assert scheduler is None
    if expect_tracker_init:
        initialize_tracker.assert_called_once_with([unwrapped_model], pg_collection)
    else:
        initialize_tracker.assert_not_called()
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


def test_train_step_uses_optimizer_ddp_config_for_mxfp8_staging():
    class _DistributedOptimizer:
        def __init__(self, overlap_param_gather):
            self.ddp_config = SimpleNamespace(
                reuse_grad_buf_for_mxfp8_param_ag=True, overlap_param_gather=overlap_param_gather
            )
            self._copy_main_params_to_param_buffer = mock.Mock()

    overlapped = _DistributedOptimizer(overlap_param_gather=True)
    nonoverlapped = _DistributedOptimizer(overlap_param_gather=False)
    optimizer = SimpleNamespace(
        zero_grad=lambda: None, chained_optimizers=[overlapped, nonoverlapped]
    )
    model = [
        SimpleNamespace(
            force_all_reduce=False,
            zero_grad_buffer=lambda: None,
            remove_forward_pre_hook_handles={object(): object()},
        )
    ]

    with mock.patch.object(training_mod, "DistributedOptimizer", _DistributedOptimizer):
        # Global args intentionally disagree; the optimizer DDP config is authoritative.
        _run(
            args_overrides={
                "reuse_grad_buf_for_mxfp8_param_ag": False,
                "overlap_param_gather": False,
            },
            model=model,
            optimizer=optimizer,
        )

    overlapped._copy_main_params_to_param_buffer.assert_called_once_with()
    nonoverlapped._copy_main_params_to_param_buffer.assert_not_called()


def test_train_step_supports_bare_distributed_optimizer_for_mxfp8_staging():
    class _DistributedOptimizer:
        def __init__(self):
            self.ddp_config = SimpleNamespace(
                reuse_grad_buf_for_mxfp8_param_ag=True, overlap_param_gather=True
            )
            self._copy_main_params_to_param_buffer = mock.Mock()

        def zero_grad(self):
            pass

    optimizer = _DistributedOptimizer()
    model = [
        SimpleNamespace(
            force_all_reduce=False,
            zero_grad_buffer=lambda: None,
            remove_forward_pre_hook_handles={object(): object()},
        )
    ]

    with mock.patch.object(training_mod, "DistributedOptimizer", _DistributedOptimizer):
        _run(model=model, optimizer=optimizer)

    optimizer._copy_main_params_to_param_buffer.assert_called_once_with()


def test_model_process_groups_fall_back_when_attribute_is_missing_or_none():
    fallback = object()
    with mock.patch.object(
        training_mod.ProcessGroupCollection, "use_mpu_process_groups", return_value=fallback
    ) as use_mpu_process_groups:
        assert training_mod._get_model_pg_collection(SimpleNamespace()) is fallback
        assert (
            training_mod._get_model_pg_collection(SimpleNamespace(pg_collection=None)) is fallback
        )

    assert use_mpu_process_groups.call_count == 2


def test_model_process_groups_are_found_through_model_wrappers():
    pg_collection = object()
    wrapped_model = SimpleNamespace(module=SimpleNamespace(pg_collection=pg_collection))

    assert training_mod._get_model_pg_collection(wrapped_model) is pg_collection


def test_train_step_invokes_tensor_metric_observer_before_optimizer():
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
        log_num_zeros_in_grad=False,
        data_parallel_size=1,
        gtp_weight_remat_size=1,
    )
    events = []
    pg_collection = SimpleNamespace(mp=object(), pp=object(), dp_cp=object(), dp_cp_gtp_remat=None)
    model = [
        SimpleNamespace(
            force_all_reduce=False, zero_grad_buffer=lambda: None, pg_collection=pg_collection
        )
    ]
    optimizer = SimpleNamespace(
        zero_grad=lambda: None, step=lambda: events.append("optimizer") or (True, 1.0, 0)
    )
    scheduler = mock.MagicMock()

    def observer(**kwargs):
        events.append("observer")
        assert model[0].force_all_reduce is False
        assert kwargs == {
            "model": model,
            "optimizer": optimizer,
            "iteration": 7,
            "pg_collection": pg_collection,
        }

    with (
        mock.patch.object(training_mod, "get_args", return_value=args),
        mock.patch.object(training_mod, "get_timers", return_value=mock.MagicMock()),
        mock.patch.object(
            training_mod, "get_rerun_state_machine", return_value=_RerunThroughOptimizer()
        ),
        mock.patch.object(training_mod, "get_num_microbatches", return_value=1),
        mock.patch.object(training_mod, "has_nvidia_modelopt", False),
        mock.patch.object(training_mod, "is_pp_last_stage", return_value=False),
        mock.patch.object(training_mod, "_get_samples_seen_in_iteration", return_value=1),
        mock.patch.object(
            training_mod,
            "logical_and_across_model_parallel_group",
            side_effect=lambda value, group: value,
        ),
        mock.patch.object(
            training_mod,
            "reduce_max_stat_across_model_parallel_group",
            side_effect=lambda value, group: value,
        ),
    ):
        training_mod.train_step(
            forward_step_func=lambda *a, **k: None,
            data_iterator=iter([]),
            model=model,
            optimizer=optimizer,
            opt_param_scheduler=scheduler,
            config=SimpleNamespace(barrier_with_L1_time=False, log_max_attention_logit=False),
            forward_backward_func=lambda **kw: [],
            iteration=7,
            tensor_metric_observer=observer,
        )

    assert events == ["observer", "optimizer"]
    assert model[0].force_all_reduce is False
    scheduler.step.assert_called_once_with(increment=1)


def test_train_step_scopes_forward_tensor_observation_around_schedule():
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
        log_num_zeros_in_grad=False,
        data_parallel_size=1,
        gtp_weight_remat_size=1,
    )
    events = []
    pg_collection = SimpleNamespace(mp=object(), pp=object(), dp_cp=object(), dp_cp_gtp_remat=None)
    model = [
        SimpleNamespace(
            force_all_reduce=False, zero_grad_buffer=lambda: None, pg_collection=pg_collection
        )
    ]
    optimizer = SimpleNamespace(
        zero_grad=lambda: None, step=lambda: events.append("optimizer") or (True, 1.0, 0)
    )

    class Observer:
        @contextmanager
        def observe_forward_backward(self, **kwargs):
            assert kwargs == {"model": model, "iteration": 7, "pg_collection": pg_collection}
            events.append("observe-enter")
            yield
            events.append("observe-exit")

        def __call__(self, **kwargs):
            events.append("commit")

    with (
        mock.patch.object(training_mod, "get_args", return_value=args),
        mock.patch.object(training_mod, "get_timers", return_value=mock.MagicMock()),
        mock.patch.object(
            training_mod, "get_rerun_state_machine", return_value=_RerunThroughOptimizer()
        ),
        mock.patch.object(training_mod, "get_num_microbatches", return_value=1),
        mock.patch.object(training_mod, "has_nvidia_modelopt", False),
        mock.patch.object(training_mod, "is_pp_last_stage", return_value=False),
        mock.patch.object(training_mod, "_get_samples_seen_in_iteration", return_value=1),
        mock.patch.object(
            training_mod,
            "logical_and_across_model_parallel_group",
            side_effect=lambda value, group: value,
        ),
        mock.patch.object(
            training_mod,
            "reduce_max_stat_across_model_parallel_group",
            side_effect=lambda value, group: value,
        ),
    ):
        training_mod.train_step(
            forward_step_func=lambda *a, **k: None,
            data_iterator=iter([]),
            model=model,
            optimizer=optimizer,
            opt_param_scheduler=mock.MagicMock(),
            config=SimpleNamespace(barrier_with_L1_time=False, log_max_attention_logit=False),
            forward_backward_func=lambda **kw: events.append("forward-backward") or [],
            iteration=7,
            tensor_metric_observer=Observer(),
        )

    assert events == ["observe-enter", "forward-backward", "observe-exit", "commit", "optimizer"]
