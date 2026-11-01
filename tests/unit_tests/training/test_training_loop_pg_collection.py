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


def _subgroup(rank_lists):
    """This rank's group from a partition of the world; every rank creates every group."""
    group, _ = torch.distributed.new_subgroups_by_enumeration(rank_lists)
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

        args = SimpleNamespace(gtp_weight_remat_size=2, gtp_remat_num_sequence_shards=1)
        assert (
            training._get_data_parallel_world_size(ProcessGroupCollection(dp_gtp_remat=group), args)
            == 2
        )
        # Without the sample group, the weight replicas times the GTP weight-remat size.
        assert training._get_data_parallel_world_size(ProcessGroupCollection(dp=group), args) == 4
        args.gtp_remat_num_sequence_shards = 2
        with pytest.raises(ValueError, match="dp_gtp_remat"):
            training._get_data_parallel_world_size(ProcessGroupCollection(dp=group), args)

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
            dynamic_context_parallel=False,
            create_attention_mask_in_dataloader=False,
            context_parallel_size=1,
            seq_length=8,
            pipeline_model_parallel_size=2,
        )
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
    def test_full_dataset_length_reduces_over_the_collection(self, monkeypatch, run_config):
        pg_collection = ProcessGroupCollection(
            dp_gtp_remat=_subgroup(_pairs()), mp=_subgroup(_singletons())
        )
        rank = torch.distributed.get_rank()
        provider = Mock(
            return_value=(SimpleNamespace(_dataloader=range(rank + 1)), None, None),
            supports_train_full_dataset=True,
        )
        args = SimpleNamespace(deduplicate_dataloader_across_context_parallel=False)
        _forbid_global_reads(monkeypatch, training)

        sample_count = training._get_train_full_dataset_sample_count(
            args, provider, pg_collection=pg_collection
        )

        first_rank_of_pair = rank - rank % 2
        assert sample_count == (first_rank_of_pair + 1) + (first_rank_of_pair + 2)

    @_needs_4_ranks
    def test_logits_saver_attaches_on_the_model_last_stage(self, monkeypatch, run_config):
        pp_group = _subgroup(_strided_pairs())
        model = [SimpleNamespace(pg_collection=ProcessGroupCollection(pp=pp_group))]
        args = SimpleNamespace(
            logits_save_dir="unused",
            logits_save_top_k=1,
            logits_save_top_p=None,
            logits_save_top_p_min_k=None,
            logits_save_dtype=None,
            logits_load_dir=None,
            moe_use_upcycling=False,
            load=None,
            pretrained_checkpoint=None,
            ckpt_convert_format=None,
        )
        attached = []

        class _LogitsSaverHooks:
            def __init__(self, **kwargs):
                pass

            def attach_hooks(self, model_chunk):
                attached.append(model_chunk)

        run_config.validation.skip_train = True
        monkeypatch.setattr("megatron.training.distillation.LogitsSaverHooks", _LogitsSaverHooks)
        monkeypatch.setattr(training, "get_args", lambda: args)
        monkeypatch.setattr(training, "get_timers", Mock())
        monkeypatch.setattr(training, "get_one_logger", lambda: None)
        monkeypatch.setattr(training, "get_train_state", TrainState)
        monkeypatch.setattr(training, "has_nvidia_modelopt", False)
        monkeypatch.setattr(training, "is_gtp_remat_active", lambda args: False)
        monkeypatch.setattr(training, "get_model", Mock(return_value=model))
        monkeypatch.setattr(training, "get_num_microbatches", lambda: 1)
        monkeypatch.setattr(training, "get_current_global_batch_size", lambda: 1)
        _forbid_global_reads(monkeypatch, training)

        training.setup_model_and_optimizer(None, lambda: None, cfg_container=run_config)

        assert attached == (model if pp_group.rank() == 1 else [])

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

        monkeypatch.setattr(training, "evaluate", evaluate)
        monkeypatch.setattr(training, "get_args", SimpleNamespace)
        monkeypatch.setattr(training, "get_train_state", TrainState)
        monkeypatch.setattr(training, "get_wandb_writer", lambda: None)
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
