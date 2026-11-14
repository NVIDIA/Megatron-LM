# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""The training loop's statistics, stage checks and reductions use its process-group collection.

Every test builds a collection whose layout differs from the global grid (TP=PP=CP=1, DP=world)
and makes each read of the global parallel state raise, so a global read either fails the test
or selects ranks that change the result.
"""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.process_groups_config import (
    MultiModuleProcessGroupCollection,
    ProcessGroupCollection,
)
from megatron.training import gpu_sniff_test, training
from megatron.training.config import ValidationConfig
from megatron.training.logging import packed_sequence_stats
from megatron.training.state import TrainState
from tests.unit_tests.test_utilities import Utils

_needs_4_ranks = pytest.mark.skipif(
    Utils.world_size < 4 or Utils.world_size % 4 != 0,
    reason="Requires a world size that is a multiple of 4",
)

_ACCESSOR_SUFFIXES = ("_group", "_groups", "_gloo", "_rank", "_ranks", "_world_size")


def _forbid_global_reads(monkeypatch, *modules):
    """Make every read of the global parallel state raise, also through names imported by name."""

    def _raise(*args, **kwargs):
        raise AssertionError("read the global parallel state")

    names = [
        name
        for name in dir(parallel_state)
        if name.startswith("get_") and name.endswith(_ACCESSOR_SUFFIXES)
    ]
    names += ["is_pipeline_first_stage", "is_pipeline_last_stage", "model_parallel_is_initialized"]
    for module in (parallel_state, *modules):
        for name in names:
            if hasattr(module, name):
                monkeypatch.setattr(module, name, _raise)
    monkeypatch.setattr(ProcessGroupCollection, "use_mpu_process_groups", classmethod(_raise))


# The groups of the running test, which the default_grid fixture destroys.
_created_groups = []


def _subgroup(rank_lists):
    """This rank's group from a partition of the world; every rank creates every group."""
    group, _ = torch.distributed.new_subgroups_by_enumeration(rank_lists)
    _created_groups.append(group)
    return group


def _pairs():
    """Neighboring ranks: [[0, 1], [2, 3], ...]."""
    return [[rank, rank + 1] for rank in range(0, Utils.world_size, 2)]


def _strided_pairs():
    """Ranks two apart within each block of four: [[0, 2], [1, 3], [4, 6], [5, 7], ...]."""
    return [
        [start + offset, start + offset + 2]
        for start in range(0, Utils.world_size, 4)
        for offset in (0, 1)
    ]


def _singletons():
    return [[rank] for rank in range(Utils.world_size)]


class _RunOnce:
    """A rerun state machine that runs the forward/backward body once."""

    def __init__(self):
        self._ran = False

    def should_run_forward_backward(self, data_iterator):
        run, self._ran = not self._ran, True
        return run


class TestTrainingLoopPgCollection:

    @pytest.fixture(autouse=True)
    def default_grid(self):
        # Set up before, and torn down after, the monkeypatched accessors.
        Utils.initialize_model_parallel()
        yield
        # Every rank finishes its collectives before any communicator goes away.
        torch.cuda.synchronize()
        torch.distributed.barrier()
        while _created_groups:
            torch.distributed.destroy_process_group(_created_groups.pop())
        Utils.destroy_model_parallel()

    @_needs_4_ranks
    def test_seqlen_stats_deduplicate_by_the_collection_grid(self, monkeypatch):
        # The collection has TP=2 and PP=2, so four ranks hold the same packed sequences.
        pg_collection = ProcessGroupCollection(
            tp=_subgroup(_pairs()), cp=_subgroup(_singletons()), pp=_subgroup(_strided_pairs())
        )
        monkeypatch.setattr(training, "_seqlen_stats_in_iteration", None)
        monkeypatch.setattr(training, "_seqlen_stats_active", False)
        monkeypatch.setattr(training, "_seqlen_stats_are_global", False)
        _forbid_global_reads(monkeypatch, training)

        cu_seqlens = torch.tensor([0, 100, 300], dtype=torch.int32, device="cuda")
        training.update_seqlen_stats_from_cu_seqlens(cu_seqlens)
        total_real_tokens, seqlen_squared_sum = training.consume_seqlen_stats_in_iteration(
            pg_collection=pg_collection
        )

        assert total_real_tokens == 300 * Utils.world_size / 4
        assert seqlen_squared_sum == (100**2 + 200**2) * Utils.world_size / 4

    @_needs_4_ranks
    def test_samples_seen_sum_over_the_collection_sample_group(self, monkeypatch, run_config):
        pg_collection = ProcessGroupCollection(
            pp=_subgroup(_singletons()), dp_gtp_remat=_subgroup(_pairs())
        )
        _forbid_global_reads(monkeypatch, training)

        losses_reduced = [{"_samples_seen": torch.tensor(3, device="cuda")}]
        samples_seen = training._get_samples_seen_in_iteration(
            losses_reduced, SimpleNamespace(), pg_collection
        )

        # Summed over the two ranks of the collection's group, not over the global DP group.
        assert samples_seen == 6

    def test_samples_seen_require_the_sample_group(self, monkeypatch, run_config):
        pg_collection = ProcessGroupCollection(pp=None)
        _forbid_global_reads(monkeypatch, training)

        losses_reduced = [{"_samples_seen": torch.tensor(3, device="cuda")}]
        with pytest.raises(ValueError, match="dp_gtp_remat"):
            training._get_samples_seen_in_iteration(
                losses_reduced, SimpleNamespace(), pg_collection
            )

    @_needs_4_ranks
    def test_data_parallel_world_size_follows_the_collection(self, monkeypatch):
        group = _subgroup(_pairs())
        _forbid_global_reads(monkeypatch, training)

        # The model settings of the run config; the sequence-shard count is derived there.
        transformer = SimpleNamespace(gtp_weight_remat_size=2, gtp_remat_num_sequence_shards=1)
        cfg = SimpleNamespace(model=SimpleNamespace(transformer=transformer))
        assert (
            training._get_data_parallel_world_size(ProcessGroupCollection(dp_gtp_remat=group), cfg)
            == 2
        )
        # Without the sample group, the weight replicas times the GTP weight-remat size.
        assert training._get_data_parallel_world_size(ProcessGroupCollection(dp=group), cfg) == 4
        transformer.gtp_remat_num_sequence_shards = 2
        with pytest.raises(ValueError, match="dp_gtp_remat"):
            training._get_data_parallel_world_size(ProcessGroupCollection(dp=group), cfg)

    @_needs_4_ranks
    def test_dummy_train_step_reads_batches_on_the_collection(self, monkeypatch, run_config):
        tp_group = _subgroup(_pairs())
        pp_group = _subgroup(_strided_pairs())
        cp_group = _subgroup(_singletons())
        pg_collection = ProcessGroupCollection(
            tp=tp_group, pp=pp_group, cp=cp_group, dynamic_cp={1: cp_group}
        )
        args = SimpleNamespace(
            sft=False,
            dataloader_inter_document_masking=False,
            create_attention_mask_in_dataloader=False,
            seq_length=8,
        )
        run_config.model.transformer.dynamic_context_parallel = False
        run_config.model.transformer.context_parallel_size = 1
        run_config.model.transformer.pipeline_model_parallel_size = 2
        run_config.train.micro_batch_size = 1
        tp_calls, cp_calls = [], []
        monkeypatch.setattr(training, "get_args", lambda: args)
        monkeypatch.setattr(training, "get_num_microbatches", lambda: 1)
        monkeypatch.setattr(training, "get_rerun_state_machine", _RunOnce)
        monkeypatch.setattr(
            training,
            "get_batch_on_this_tp_rank",
            lambda batch, **kwargs: tp_calls.append(kwargs) or batch,
        )
        monkeypatch.setattr(
            training,
            "get_batch_on_this_cp_rank",
            lambda batch, **kwargs: cp_calls.append(kwargs) or batch,
        )
        _forbid_global_reads(monkeypatch, training)

        data_iterator = iter([{}])
        training.dummy_train_step(data_iterator, pg_collection=pg_collection)

        (tp_kwargs,) = tp_calls
        (cp_kwargs,) = cp_calls
        assert tp_kwargs["broadcast_group"] is tp_group
        assert tp_kwargs["broadcast_src_rank"] == (
            torch.distributed.get_process_group_ranks(tp_group)[0]
        )
        assert tp_kwargs["tp_rank"] == tp_group.rank()
        assert tp_kwargs["is_pipeline_first_stage"] == (pp_group.rank() == 0)
        assert tp_kwargs["is_pipeline_last_stage"] == (pp_group.rank() == 1)
        assert cp_kwargs["cp_group"] is cp_group
        assert cp_kwargs["dynamic_cp_group_func"](group_size=1) is cp_group
        # Only tensor-parallel rank 0 of the collection reads from the data iterator.
        assert (next(data_iterator, None) is None) == (tp_group.rank() == 0)

    @_needs_4_ranks
    @pytest.mark.parametrize("deduplicate_across_cp", [False, True])
    def test_full_dataset_length_reduces_over_the_collection(
        self, monkeypatch, run_config, deduplicate_across_cp
    ):
        sample_group, cp_sample_group = _subgroup(_pairs()), _subgroup(_strided_pairs())
        pg_collection = ProcessGroupCollection(
            dp_gtp_remat=sample_group, dp_cp_gtp_remat=cp_sample_group, mp=_subgroup(_singletons())
        )
        rank = torch.distributed.get_rank()
        provider = Mock(
            return_value=(SimpleNamespace(_dataloader=range(rank + 1)), None, None),
            supports_train_full_dataset=True,
        )
        args = SimpleNamespace(deduplicate_dataloader_across_context_parallel=deduplicate_across_cp)
        _forbid_global_reads(monkeypatch, training)

        sample_count = training._get_train_full_dataset_sample_count(
            args, provider, pg_collection=pg_collection
        )

        # A dataloader deduplicated across context parallelism is summed over those ranks too.
        summed_group = cp_sample_group if deduplicate_across_cp else sample_group
        assert sample_count == sum(
            peer + 1 for peer in torch.distributed.get_process_group_ranks(summed_group)
        )

    @_needs_4_ranks
    @pytest.mark.parametrize("model_carries_collection", [True, False])
    def test_logits_hooks_attach_on_the_model_last_stage(
        self, monkeypatch, run_config, model_carries_collection
    ):
        pp_group, dp_group = _subgroup(_strided_pairs()), _subgroup(_pairs())
        pg_collection = ProcessGroupCollection(pp=pp_group, dp=dp_group)
        # The model's own collection decides; a model without one runs on the run's collection.
        if model_carries_collection:
            model = [SimpleNamespace(pg_collection=pg_collection)]
            run_pg_collection = ProcessGroupCollection()
        else:
            model, run_pg_collection = [SimpleNamespace()], pg_collection
        args = SimpleNamespace(
            logits_save_dir="unused",
            logits_save_top_k=1,
            logits_save_top_p=None,
            logits_save_top_p_min_k=None,
            logits_save_dtype=None,
            logits_load_dir="unused",
            moe_use_upcycling=False,
            load=None,
            pretrained_checkpoint=None,
            ckpt_convert_format=None,
        )
        attached, saver_pg_collections = [], []
        captured, reported = [], []

        class _LogitsSaverHooks:
            def __init__(self, **kwargs):
                saver_pg_collections.append(kwargs["pg_collection"])

            def attach_hooks(self, model_chunk):
                attached.append(model_chunk)

        class _StudentLogitsCapture:
            def attach_hooks(self, model_chunk):
                captured.append(model_chunk)

        run_config.validation.skip_train = True
        monkeypatch.setattr("megatron.training.distillation.LogitsSaverHooks", _LogitsSaverHooks)
        monkeypatch.setattr(
            "megatron.training.distillation.StudentLogitsCapture", _StudentLogitsCapture
        )
        monkeypatch.setattr(training, "get_args", lambda: args)
        monkeypatch.setattr(training, "get_timers", Mock())
        monkeypatch.setattr(training, "get_one_logger", lambda: None)
        monkeypatch.setattr(training, "get_train_state", TrainState)
        monkeypatch.setattr(training, "has_nvidia_modelopt", False)
        monkeypatch.setattr(training, "is_gtp_remat_active", lambda args: False)
        monkeypatch.setattr(training, "get_model", Mock(return_value=model))
        monkeypatch.setattr(training, "get_num_microbatches", lambda: 1)
        monkeypatch.setattr(training, "get_current_global_batch_size", lambda: 1)
        monkeypatch.setattr(
            training, "report_memory", lambda name, process_group: reported.append(process_group)
        )
        _forbid_global_reads(monkeypatch, training)

        training.setup_model_and_optimizer(
            None, lambda: None, cfg_container=run_config, pg_collection=run_pg_collection
        )

        on_last_stage = pp_group.rank() == 1
        assert attached == (model if on_last_stage else [])
        assert captured == (model if on_last_stage else [])
        # The saver runs on the groups that decided its stage.
        assert [id(groups) for groups in saver_pg_collections] == (
            [id(pg_collection)] if on_last_stage else []
        )
        # The setup memory reports gate their print on the same collection's ranks.
        assert reported and all(group is dp_group for group in reported)

    @_needs_4_ranks
    def test_full_validation_lengths_come_from_the_collection(self, monkeypatch, run_config):
        pg_collection = ProcessGroupCollection(tp=_subgroup(_pairs()))
        run_config.validation = ValidationConfig(
            eval_iters=[3, 4], full_validation=True, multiple_validation_sets=True
        )
        evaluated = []

        def evaluate(*args, eval_iters, **kwargs):
            evaluated.append(eval_iters)
            return {}, None, False

        sent = []
        broadcast = torch.distributed.broadcast

        def record_and_broadcast(tensor, *args, **kwargs):
            sent.append(tensor.tolist())
            return broadcast(tensor, *args, **kwargs)

        monkeypatch.setattr(training, "evaluate", evaluate)
        monkeypatch.setattr(training, "get_args", SimpleNamespace)
        monkeypatch.setattr(training, "get_train_state", TrainState)
        monkeypatch.setattr(training, "get_wandb_writer", lambda: None)
        monkeypatch.setattr(torch.distributed, "broadcast", record_and_broadcast)
        _forbid_global_reads(monkeypatch, training)

        training.evaluate_and_print_results(
            "prefix",
            None,
            [None, None],
            [SimpleNamespace(pg_collection=pg_collection)],
            0,
            None,
            None,
            write_to_tensorboard=False,
        )

        # World rank 0 broadcasts its lengths, so the result alone does not show which ranks
        # filled them in: only tensor-parallel rank 0 of the collection does.
        assert sent == [[3, 4] if pg_collection.tp.rank() == 0 else [0, 0]]
        assert evaluated == [3, 4]
        assert run_config.validation.eval_iters == [3, 4]

    def test_model_collection_precedes_the_run_collection(self, monkeypatch):
        model_pg_collection, run_pg_collection = ProcessGroupCollection(), ProcessGroupCollection()
        _forbid_global_reads(monkeypatch, training)

        assert (
            training._get_model_pg_collection(
                SimpleNamespace(pg_collection=model_pg_collection), run_pg_collection
            )
            is model_pg_collection
        )
        # A model without a collection runs on the run's single-grid collection.
        assert (
            training._get_model_pg_collection(
                SimpleNamespace(pg_collection=None), run_pg_collection
            )
            is run_pg_collection
        )

    def test_train_step_observes_tensor_metrics_on_the_run_collection(
        self, monkeypatch, run_config
    ):
        # The model carries no collection, so every read in train_step uses the run's.
        pg_collection = ProcessGroupCollection(mp=object(), pp=object(), dp_cp=object())
        model = [SimpleNamespace(force_all_reduce=False, zero_grad_buffer=lambda: None)]
        optimizer = SimpleNamespace(zero_grad=lambda: None, step=lambda: (True, 1.0, 0))
        args = SimpleNamespace(
            save_params_interval=None,
            save_activations_interval=None,
            save_tokens_per_expert_interval=None,
            save_wgrads_interval=None,
            save_dgrads_interval=None,
            seq_length=8,
            decoder_seq_length=None,
            vision_pretraining=False,
            qk_clip=False,
        )
        observed, counted = [], []

        class _Observer:
            def observe_forward_backward(self, pg_collection, **kwargs):
                observed.append(pg_collection)
                return nullcontext()

            def __call__(self, pg_collection, **kwargs):
                observed.append(pg_collection)

        class _RunOnceThroughOptimizer(_RunOnce):
            def should_checkpoint_and_exit(self):
                return False, False, 0

        monkeypatch.setattr(training, "get_args", lambda: args)
        monkeypatch.setattr(training, "get_timers", lambda: MagicMock())
        monkeypatch.setattr(training, "get_num_microbatches", lambda: 1)
        monkeypatch.setattr(training, "get_rerun_state_machine", _RunOnceThroughOptimizer)
        monkeypatch.setattr(training, "has_nvidia_modelopt", False)
        monkeypatch.setattr(training, "is_pp_last_stage", lambda group: False)
        monkeypatch.setattr(
            training,
            "_get_samples_seen_in_iteration",
            lambda losses_reduced, args, pg_collection: counted.append(pg_collection) or 1,
        )
        monkeypatch.setattr(
            training, "logical_and_across_model_parallel_group", lambda value, group: value
        )
        monkeypatch.setattr(
            training, "reduce_max_stat_across_model_parallel_group", lambda value, group: value
        )
        _forbid_global_reads(monkeypatch, training)

        training.train_step(
            forward_step_func=None,
            data_iterator=iter([]),
            model=model,
            optimizer=optimizer,
            opt_param_scheduler=MagicMock(),
            config=SimpleNamespace(barrier_with_L1_time=False, log_max_attention_logit=False),
            forward_backward_func=lambda **kwargs: [],
            iteration=0,
            pg_collection=pg_collection,
            tensor_metric_observer=_Observer(),
        )

        assert observed == [pg_collection, pg_collection]
        assert counted == [pg_collection]

    def test_sniff_test_runs_on_the_collection(self, monkeypatch, run_config):
        pg_collection = ProcessGroupCollection()
        run_config.train.gpu_sniff_test_interval = 1
        sniffed = []
        monkeypatch.setattr(
            gpu_sniff_test,
            "run_gpu_sniff_test",
            lambda tag, pg_collection=None: sniffed.append(pg_collection),
        )
        monkeypatch.setattr(
            training,
            "get_args",
            lambda: SimpleNamespace(log_straggler=False, adlr_autoresume=False),
        )
        monkeypatch.setattr(training, "get_timers", lambda: MagicMock())
        monkeypatch.setattr(training, "print_datetime", lambda *args, **kwargs: None)
        monkeypatch.setattr(training, "_otel_managed_span", lambda *args, **kwargs: nullcontext())
        _forbid_global_reads(monkeypatch, training)

        training._run_gpu_sniff_test("start", pg_collection=pg_collection)
        training.post_training_step_callbacks(
            [], None, None, 1, None, 0, pg_collection=pg_collection
        )

        assert sniffed == [pg_collection, pg_collection]

    def test_sniff_test_rejects_a_multi_module_collection(self, monkeypatch):
        module_pg_collection = ProcessGroupCollection()
        multi_module_pg_collection = MultiModuleProcessGroupCollection(
            module_pgs={"language": module_pg_collection}, language_model_module_name="language"
        )
        _forbid_global_reads(monkeypatch, training)

        with pytest.raises(ValueError, match="gpu-sniff-test-interval"):
            training._check_gpu_sniff_test_pg_collection(multi_module_pg_collection, 1)
        # Without the sniff test, or on a single grid, the collection is accepted.
        training._check_gpu_sniff_test_pg_collection(multi_module_pg_collection, None)
        training._check_gpu_sniff_test_pg_collection(module_pg_collection, 1)
        training._check_gpu_sniff_test_pg_collection(None, 1)

    def test_train_rejects_the_sniff_test_for_a_multi_module_collection(
        self, monkeypatch, run_config
    ):
        # An encoder rank of heterogeneous MiMo: one module and no language model.
        multi_module_pg_collection = MultiModuleProcessGroupCollection(
            module_pgs={"encoder": ProcessGroupCollection()}
        )
        run_config.train.gpu_sniff_test_interval = 1
        sniffed = []
        monkeypatch.setattr(
            gpu_sniff_test,
            "run_gpu_sniff_test",
            lambda tag, pg_collection=None: sniffed.append(pg_collection),
        )
        monkeypatch.setattr(training, "get_args", SimpleNamespace)
        monkeypatch.setattr(training, "get_timers", lambda: MagicMock())
        monkeypatch.setattr(training, "get_train_state", TrainState)
        _forbid_global_reads(monkeypatch, training)

        with pytest.raises(ValueError, match="gpu-sniff-test-interval"):
            training.train(
                None,
                [],
                None,
                None,
                None,
                SimpleNamespace(),
                None,
                next_train_iterator=lambda iteration, iterator: iterator,
                run_eval=lambda **kwargs: None,
                pg_collection=multi_module_pg_collection,
            )

        assert sniffed == []

    @_needs_4_ranks
    def test_packed_sequence_stats_count_the_collection_last_stage(self, monkeypatch):
        pg_collection = ProcessGroupCollection(
            pp=_subgroup(_strided_pairs()), tp=_subgroup(_singletons()), cp=_subgroup(_singletons())
        )
        packed_sequence_stats._reset_packed_sequence_stats_in_iteration()
        _forbid_global_reads(monkeypatch, packed_sequence_stats)

        packed_sequence_stats.update_packed_sequence_stats(
            torch.tensor([[100, 150, 0]]),
            torch.tensor([[1.0, 1.0, 0.0]]),
            pg_collection=pg_collection,
        )
        stats = packed_sequence_stats.consume_packed_sequence_stats_in_iteration()

        # Only the last pipeline stage of the collection, half of the ranks, contributes.
        assert stats["packed_sequence/original_samples"] == 2.0 * Utils.world_size / 2
        assert stats["packed_sequence/total_tokens"] == 250.0 * Utils.world_size / 2


def test_sample_sum_requires_the_collection_sample_group(monkeypatch):
    _forbid_global_reads(monkeypatch, training)

    with pytest.raises(ValueError, match="dp_gtp_remat"):
        training._reduce_sum_across_data_parallel_group(1.0, pg_collection=ProcessGroupCollection())


@pytest.mark.parametrize("num_microbatches", [None, 0])
def test_setup_rejects_a_batch_too_small_for_its_ranks(monkeypatch, run_config, num_microbatches):
    args = SimpleNamespace(logits_save_dir=None, logits_load_dir=None, moe_use_upcycling=False)
    run_config.validation.skip_train = True
    run_config.train.micro_batch_size = 1
    monkeypatch.setattr(training, "get_args", lambda: args)
    monkeypatch.setattr(training, "get_timers", Mock())
    monkeypatch.setattr(training, "get_one_logger", lambda: None)
    monkeypatch.setattr(training, "get_train_state", TrainState)
    monkeypatch.setattr(training, "has_nvidia_modelopt", False)
    monkeypatch.setattr(training, "is_gtp_remat_active", lambda args: False)
    monkeypatch.setattr(training, "get_model", Mock(return_value=[SimpleNamespace()]))
    monkeypatch.setattr(training, "report_memory", lambda *args, **kwargs: None)
    monkeypatch.setattr(training, "get_num_microbatches", lambda: num_microbatches)
    monkeypatch.setattr(training, "get_current_global_batch_size", lambda: 1)
    # The check sizes its message from the run config, not from the global grid.
    _forbid_global_reads(monkeypatch, training)

    with pytest.raises(ValueError, match="too small for"):
        training.setup_model_and_optimizer(
            None, lambda: None, cfg_container=run_config, pg_collection=ProcessGroupCollection()
        )


@pytest.mark.parametrize("multi_module", [False, True])
def test_train_passes_the_collection_to_its_helpers(monkeypatch, run_config, multi_module):
    """One skipped and one rollout-only iteration of train(), with its helpers recorded."""
    grid_pg_collection = sample_pg_collection = pg_collection = ProcessGroupCollection()
    if multi_module:
        # The grid readers fall back to the global grid; samples count on the language model.
        grid_pg_collection = None
        pg_collection = MultiModuleProcessGroupCollection(
            module_pgs={"language": sample_pg_collection}, language_model_module_name="language"
        )
    run_config.train.gpu_sniff_test_interval = None if multi_module else 1
    run_config.train.micro_batch_size, run_config.train.train_iters = 1, 2
    run_config.train.iterations_to_skip = [1]
    args = SimpleNamespace(seq_length=8, decoder_seq_length=None, logits_save_dir=None)
    train_state, calls = TrainState(), []

    def record(name, result=None):
        def recorder(*args, **kwargs):
            calls.append((name, kwargs["pg_collection"] if "pg_collection" in kwargs else args[0]))
            return result

        monkeypatch.setattr(training, name, recorder)

    record("_run_gpu_sniff_test")
    record("dummy_train_step")
    record("_get_data_parallel_world_size", 1)
    record("consume_seqlen_stats_in_iteration", (None, None))
    record("post_training_step_callbacks", 0.0)
    # Everything else that train() calls on this path: no-ops, unless it needs their result.
    no_ops = """get_timers get_energy_monitor get_rerun_state_machine one_logger_utils
        ft_integration write_args_to_tensorboard print_datetime report_memory
        _get_forward_backward_func maybe_finalize_async_save update_num_microbatches
        maybe_verify_choices training_log _end_otel_startup_span _start_otel_train_span
        _maybe_reroot_otel_interval _end_otel_train_span"""
    falsy = """get_one_logger get_tensorboard_writer should_disable_forward_pre_hook
        is_batch_size_transition _should_compute_params_norm _otel_sg_enabled
        checkpoint_and_decide_exit"""
    sizes = (
        "get_num_microbatches get_current_global_batch_size get_current_running_global_batch_size"
    )
    for names, stub in (
        (no_ops, MagicMock()),
        (falsy, Mock(return_value=None)),
        (sizes, lambda: 1),
    ):
        for name in names.split():
            monkeypatch.setattr(training, name, stub)
    monkeypatch.setattr(training, "get_args", lambda: args)
    monkeypatch.setattr(training, "get_train_state", lambda: train_state)
    monkeypatch.setattr(training, "_otel_managed_span", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(training, "num_floating_point_operations", lambda *args, **kwargs: 0.0)
    _forbid_global_reads(monkeypatch, training)

    training.train(
        None,
        [MagicMock()],
        None,
        None,
        None,
        SimpleNamespace(finalize_model_grads_func=None),
        None,
        next_train_iterator=lambda iteration, iterator: iterator,
        run_eval=lambda **kwargs: None,
        pg_collection=pg_collection,
        run_train_loop_without_updates=True,
    )

    assert calls == [
        *([] if multi_module else [("_run_gpu_sniff_test", grid_pg_collection)]),
        ("dummy_train_step", grid_pg_collection),
        ("_get_data_parallel_world_size", sample_pg_collection),
        ("_get_data_parallel_world_size", sample_pg_collection),
        ("consume_seqlen_stats_in_iteration", grid_pg_collection),
        ("post_training_step_callbacks", grid_pg_collection),
    ]
    assert train_state.iteration == 2
