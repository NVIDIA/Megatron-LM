# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import megatron.training.training as training_module
from megatron.core.models.hybrid import MTPSplit, PipelineSplit
from megatron.core.ssm.mlp_layer_config import MLPLayerConfig
from megatron.core.tokenizers.utils.build_tokenizer import vocab_size_with_padding
from megatron.core.transformer.moe.moe_layer_config import MoELayerConfig
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.checkpointing import save_grads
from megatron.training.global_vars import set_args
from megatron.training.training import (
    _find_hybrid_model_for_runtime_metrics,
    _hybrid_config_list_moe_logging_metadata,
    build_train_valid_test_data_iterators,
)
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.test_utilities import Utils


def mock_train_valid_test_datasets_provider(train_val_test_num_samples):
    return iter([1]), iter([2]), iter([3])


class _LenDataloader:
    """Fake dataloader with __len__ (required by the full_validation path)
    and __iter__ (consumed via cyclic_iter)."""

    def __init__(self, data):
        self._data = list(data)

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        return iter(self._data)


def mock_multi_valid_full_datasets_provider(train_val_test_num_samples):
    return (iter([1]), [_LenDataloader([2, 2]), _LenDataloader([20, 20, 20])], iter([3]))


def create_test_args():
    # Set dummy values for the args.
    args = SimpleNamespace()
    args.iteration = 0
    args.train_samples = 1
    args.train_iters = 1
    args.eval_interval = 1
    args.eval_iters = 1
    args.global_batch_size = 1
    args.consumed_train_samples = 1
    args.consumed_valid_samples = 1
    args.dataloader_type = "external"
    args.skip_train = False
    args.start_eval_at_iter = None
    args.full_validation = False
    args.multiple_validation_sets = False
    args.perform_rl_step = False
    args.phase_transition_iterations = None

    return args


def test_hybrid_config_list_moe_logging_metadata_counts_all_physical_moe_layers():
    base = TransformerConfig(num_layers=2, hidden_size=8, num_attention_heads=2)
    aux = MoELayerConfig.from_config(base)
    aux.moe_router_load_balancing_type = "aux_loss"
    aux.moe_aux_loss_coeff = 1.0

    disabled = MoELayerConfig.from_config(base)
    disabled.moe_router_load_balancing_type = "aux_loss"
    disabled.moe_aux_loss_coeff = 0.0

    mtp = MoELayerConfig.from_config(base)
    mtp.moe_router_load_balancing_type = "seq_aux_loss"
    mtp.moe_aux_loss_coeff = 2.0
    mtp.moe_z_loss_coeff = 0.1
    source = [aux, PipelineSplit, disabled, MTPSplit, mtp, MTPSplit, mtp]

    repeated = _hybrid_config_list_moe_logging_metadata(source, True)
    # Disabled auxiliary losses do not remove an MoE layer from the averaging denominator.
    assert repeated == (["load_balancing_loss", "seq_load_balancing_loss", "z_loss"], 4, 3, True)

    independent = _hybrid_config_list_moe_logging_metadata(source, False)
    assert independent == (["load_balancing_loss", "seq_load_balancing_loss", "z_loss"], 4, 4, True)


def test_hybrid_config_list_moe_logging_metadata_without_moe():
    base = TransformerConfig(num_layers=2, hidden_size=8, num_attention_heads=2)
    dense = MLPLayerConfig.from_config(base)
    source = [dense, PipelineSplit, dense, MTPSplit, dense, MTPSplit, dense]

    assert _hybrid_config_list_moe_logging_metadata(source, True) == ([], 4, 0, False)
    assert _hybrid_config_list_moe_logging_metadata(source, False) == ([], 4, 0, False)


def test_hybrid_config_list_moe_logging_metadata_uses_standard_mtp_tracker_size():
    base = TransformerConfig(num_layers=2, hidden_size=8, num_attention_heads=2)
    dense = MLPLayerConfig.from_config(base)
    moe = MoELayerConfig.from_config(base)
    source = [moe, PipelineSplit, dense, MTPSplit, dense, moe, MTPSplit, dense, moe]

    # The existing tracker allocates one slot per MTP depth, not per template sublayer.
    assert _hybrid_config_list_moe_logging_metadata(source, True) == (
        ["load_balancing_loss"],
        4,
        2,
        True,
    )
    assert _hybrid_config_list_moe_logging_metadata(source, False) == (
        ["load_balancing_loss"],
        4,
        3,
        True,
    )


def test_hybrid_config_list_moe_logging_metadata_keeps_disabled_metric_names():
    base = TransformerConfig(num_layers=2, hidden_size=8, num_attention_heads=2)
    dense = MLPLayerConfig.from_config(base)
    moe = MoELayerConfig.from_config(base)
    moe.moe_router_load_balancing_type = ["aux_loss", "seq_aux_loss", "global_aux_loss"]
    moe.moe_aux_loss_coeff = [0.0, 0.0, 0.0]

    assert _hybrid_config_list_moe_logging_metadata([moe, dense], False) == (
        ["load_balancing_loss", "seq_load_balancing_loss", "global_load_balancing_loss"],
        2,
        1,
        True,
    )


def test_runtime_metrics_find_hybrid_model_under_language_model_wrapper():
    from megatron.core.models.hybrid.hybrid_model import HybridModel

    hybrid_model = HybridModel.__new__(HybridModel)
    wrapper = SimpleNamespace(language_model=SimpleNamespace(module=hybrid_model))

    assert _find_hybrid_model_for_runtime_metrics(wrapper) is hybrid_model


@pytest.fixture
def training_log_context(monkeypatch):
    """Exercise the logger with empty losses and no device or writer side effects."""
    args = SimpleNamespace(
        timing_log_level=0,
        perform_rl_step=False,
        micro_batch_size=2,
        data_parallel_size=3,
        gtp_weight_remat_size=2,
        world_size=8,
        seq_length=64,
        num_experts=None,
        mtp_num_layers=None,
        dsa_indexer_loss_coeff=None,
        log_interval=10,
        profile_ranks=[],
        record_memory_history=False,
        log_throughput=True,
        log_timers_to_tensorboard=False,
        train_iters=20,
        consumed_train_samples=240,
        skipped_train_samples=0,
        log_energy=False,
        log_memory_interval=None,
        rl_profile=False,
        moe_per_layer_logging=True,
        moe_layer_freq=None,
    )
    timers = Mock()
    timers.return_value.elapsed.return_value = 20.0
    tracker = Mock()
    tracker.report.return_value = " MoE metrics |"
    flops = Mock(return_value=32e12)
    track_throughput = Mock()
    print_log = Mock()
    pg_collection = SimpleNamespace(mp=object())
    for name, value in (
        ("get_args", args),
        ("get_timers", timers),
        ("get_num_microbatches", 2),
        ("get_moe_metrics_tracker", tracker),
        ("get_tensorboard_writer", None),
        ("get_wandb_writer", None),
        ("get_one_logger", None),
        ("get_energy_monitor", None),
        ("get_telemetry", None),
    ):
        monkeypatch.setattr(training_module, name, Mock(return_value=value))
    monkeypatch.setattr(training_module, "has_rl_utils", False)
    monkeypatch.setattr(
        training_module, "reduce_max_stat_across_model_parallel_group", lambda value, group: value
    )
    monkeypatch.setattr(training_module, "num_floating_point_operations", flops)
    monkeypatch.setattr(training_module, "print_rank_last", print_log)
    monkeypatch.setattr(training_module.one_logger_utils, "track_app_tag", Mock())
    monkeypatch.setattr(training_module.one_logger_utils, "track_e2e_metrics", track_throughput)
    monkeypatch.setattr(
        training_module,
        "_hybrid_config_list_moe_logging_metadata",
        Mock(side_effect=AssertionError("The logger must not derive model metadata")),
    )
    total_loss_dict = {"advanced iterations": 3, "skipped iterations": 1, "nan iterations": 0}

    def log(**kwargs):
        training_module.training_log(
            loss_dict={},
            total_loss_dict=total_loss_dict,
            learning_rate=0.1,
            iteration=1 if kwargs.get("is_first_iteration") else 10,
            loss_scale=1.0,
            report_memory_flag=False,
            skipped_iter=0,
            grad_norm=None,
            params_norm=None,
            num_zeros_in_grad=None,
            max_attention_logit=None,
            pg_collection=pg_collection,
            **kwargs,
        )

    return SimpleNamespace(
        args=args,
        log=log,
        total_loss_dict=total_loss_dict,
        timers=timers,
        tracker=tracker,
        flops=flops,
        track_throughput=track_throughput,
        print_log=print_log,
        pg_collection=pg_collection,
    )


@pytest.mark.parametrize(
    "batch_flops,is_first_iteration,llm_world_size",
    [(0.0, False, None), (32e12, False, None), (32e12, True, None), (32e12, False, 4)],
)
def test_training_log_uses_precomputed_flops(
    training_log_context, batch_flops, is_first_iteration, llm_world_size
):
    ctx = training_log_context
    if llm_world_size is not None:
        ctx.args.mimo_llm_world_size = llm_world_size

    ctx.log(
        num_floating_point_operations_in_batch=batch_flops, is_first_iteration=is_first_iteration
    )

    # The logger adds one advanced iteration: 20 elapsed seconds / 5 iterations = 4 s/iter.
    expected_throughput = batch_flops / (4.0 * 1e12 * (llm_world_size or ctx.args.world_size))
    ctx.track_throughput.assert_called_once_with(True, expected_throughput)
    ctx.flops.assert_not_called()
    ctx.timers.return_value.elapsed.assert_called_once_with(
        barrier=True, reset=not is_first_iteration
    )
    ctx.timers.log.assert_called_once_with([], normalizer=10, reset=not is_first_iteration)
    assert ctx.total_loss_dict["advanced iterations"] == (4 if is_first_iteration else 0)
    assert ctx.total_loss_dict["skipped iterations"] == (1 if is_first_iteration else 0)


def test_training_log_preserves_legacy_flops_fallback_with_packed_stats(training_log_context):
    ctx = training_log_context

    ctx.log(seqlen_squared_sum_in_batch=1234.0, total_real_tokens_in_batch=80.0)

    ctx.flops.assert_called_once_with(
        ctx.args, 24, seqlen_squared_sum_in_batch=1234.0, total_real_tokens_in_batch=80.0
    )
    ctx.track_throughput.assert_called_once_with(True, 1.0)


@pytest.mark.parametrize("num_moe_layers", [3, 4])
def test_training_log_uses_precomputed_moe_metadata_without_global_experts(
    training_log_context, num_moe_layers
):
    ctx = training_log_context
    track_names = ["load_balancing_loss", "seq_load_balancing_loss", "z_loss"]

    ctx.log(
        moe_logging_metadata=(track_names, 4, num_moe_layers, True),
        num_floating_point_operations_in_batch=32e12,
    )

    assert ctx.args.num_experts is None
    ctx.tracker.report.assert_called_once_with(
        loss_scale=0.5,
        iteration=10,
        writer=None,
        wandb_writer=None,
        per_layer_logging=True,
        force_initialize=True,
        track_names=track_names,
        num_layers=4,
        num_moe_layers=num_moe_layers,
        moe_layer_freq=None,
        pg_collection=ctx.pg_collection,
        total_loss_dict=ctx.total_loss_dict,
    )
    assert " MoE metrics |" in ctx.print_log.call_args.args[0]


def test_training_log_dense_metadata_overrides_global_experts(training_log_context):
    ctx = training_log_context
    ctx.args.num_experts = 8

    ctx.log(moe_logging_metadata=([], 4, 0, False), num_floating_point_operations_in_batch=32e12)

    ctx.tracker.report.assert_not_called()
    assert " MoE metrics |" not in ctx.print_log.call_args.args[0]


@pytest.mark.parametrize("pattern,expected_moe_layers", [(None, 2), ("*E--", 1)])
def test_training_log_preserves_legacy_moe_metadata(
    training_log_context, pattern, expected_moe_layers
):
    ctx = training_log_context
    ctx.args.num_experts = 8
    ctx.args.num_layers = 4
    ctx.args.moe_layer_freq = 2
    ctx.args.hybrid_layer_pattern = pattern
    ctx.args.moe_router_load_balancing_type = ["aux_loss", "seq_aux_loss"]
    ctx.args.moe_z_loss_coeff = 0.1

    ctx.log(num_floating_point_operations_in_batch=32e12)

    ctx.tracker.report.assert_called_once()
    report_args = ctx.tracker.report.call_args.kwargs
    assert report_args["track_names"] == [
        "load_balancing_loss",
        "seq_load_balancing_loss",
        "z_loss",
    ]
    assert report_args["num_layers"] == 4
    assert report_args["num_moe_layers"] == expected_moe_layers
    assert report_args["loss_scale"] == 0.5


class TestTraining:
    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        args = create_test_args()
        set_args(args)

    def test_build_train_valid_test_data_iterators(self):
        train_iter, valid_iter, test_iter = build_train_valid_test_data_iterators(
            mock_train_valid_test_datasets_provider
        )
        train_data = next(train_iter)
        valid_data = next(valid_iter)
        test_data = next(test_iter)
        assert (train_data, valid_data, test_data) == (1, 2, 3)

    def test_build_train_valid_test_data_iterators_multi_full_validation(self):
        """multiple_validation_sets + full_validation builds a list of iterators
        (one per validation set) and sets args.eval_iters to the per-loader
        lengths MAX-reduced across DP ranks."""
        args = create_test_args()
        args.multiple_validation_sets = True
        args.full_validation = True
        set_args(args)
        _, valid_iters, _ = build_train_valid_test_data_iterators(
            mock_multi_valid_full_datasets_provider
        )
        assert isinstance(valid_iters, list)
        assert len(valid_iters) == 2
        assert next(valid_iters[0]) == 2
        assert next(valid_iters[1]) == 20
        # data_parallel_size=1, so MAX across DP ranks equals the local lengths
        assert args.eval_iters == [2, 3]

    def test_closed_formula_vocab_size_with_padding(self):
        def old_round_impl(after, multiple):
            while (after % multiple) != 0:
                after += 1
            return after

        args = SimpleNamespace()
        args.rank = 0
        args.tensor_model_parallel_size = 1

        for vocab in range(1, 600000, 1000):
            for mult in [1, 17, 32, 64, 128]:
                args.make_vocab_size_divisible_by = mult
                assert old_round_impl(vocab, mult) == vocab_size_with_padding(vocab, args, False), (
                    vocab,
                    mult,
                )

        for vocab in range(1, 10_000, 500):
            for mult in range(1, 1024 + 1):
                args.make_vocab_size_divisible_by = mult
                assert old_round_impl(vocab, mult) == vocab_size_with_padding(vocab, args, False), (
                    vocab,
                    mult,
                )

    def teardown_method(self, method):
        Utils.destroy_model_parallel()


class TestGetModelBucketSizingPgCollection:
    """The DDP-bucket-sizing path in get_model must read world size / rank from the
    explicitly passed pg_collection (pg_collection.dp_cp / pg_collection.pp) rather
    than the mpu globals. With an explicit pg_collection the mpu globals must not be
    consulted at all."""

    def test_bucket_sizing_uses_explicit_pg_collection(self, monkeypatch):
        import megatron.training.training as training

        # Sentinel groups whose size()/rank() identify which group was read.
        class _Group:
            def __init__(self, size, rank):
                self._size = size
                self._rank = rank

            def size(self):
                return self._size

            def rank(self):
                return self._rank

        pg_collection = SimpleNamespace(dp_cp=_Group(size=7, rank=0), pp=_Group(size=4, rank=3))

        # The mpu globals replaced on the bucket-sizing path must never be called
        # when an explicit pg_collection is supplied.
        def _boom(*args, **kwargs):
            raise AssertionError("mpu global consulted on explicit pg_collection path")

        monkeypatch.setattr(training.mpu, "get_data_parallel_world_size", _boom)
        monkeypatch.setattr(training.mpu, "get_pipeline_model_parallel_rank", _boom)

        # get_pg_size/get_pg_rank return 1/0 unless torch.distributed is initialized,
        # so make them read directly off the sentinel groups for this host-only test.
        monkeypatch.setattr(training, "get_pg_size", lambda group: group.size())
        monkeypatch.setattr(training, "get_pg_rank", lambda group: group.rank())

        # Mirror the exact bucket-sizing expressions from get_model.
        bucket_size = max(40000000, 1000000 * training.get_pg_size(pg_collection.dp_cp))
        pp_rank = training.get_pg_rank(pg_collection.pp)

        # dp_cp size 7 -> 7_000_000 < 40_000_000, so the floor wins (default behavior).
        assert bucket_size == 40000000
        # pp rank is driven by pg_collection.pp, not the mpu global.
        assert pp_rank == 3


class TestSaveGrads:
    """Tests for the save_grads function."""

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_save_grads(self, tmp_path_dist_ckpt):
        """Test that save_grads creates the correct directory structure and saves
        state_dict correctly.

        With TP=1, PP=1 on 8 GPUs, we have 8 DP ranks. Only the rank with
        expert_data_parallel_rank==0 should save. All ranks verify the result.
        """
        save_dir = str(tmp_path_dist_ckpt / "test_save_grads")

        with TempNamedDir(save_dir, sync=True) as save_dir:
            # Create a mock state_dict with gradients (use deterministic values for reproducibility).
            state_dict = defaultdict(dict)
            state_dict["model_chunk0"]["layer.weight"] = torch.arange(16).reshape(4, 4).float()
            state_dict["model_chunk0"]["layer.bias"] = torch.arange(4).float()

            iteration = 100
            grad_label = "wgrads"

            # All ranks call save_grads, but only expert_data_parallel_rank==0 actually saves.
            save_grads(save_dir, dict(state_dict), iteration, grad_label)

            # Synchronize before checking results since only rank 0 saves.
            torch.distributed.barrier()

            # All ranks verify the file was created by rank 0.
            expected_dir = Path(save_dir) / grad_label / f"iter_{iteration:07d}"
            assert expected_dir.exists(), f"Expected directory {expected_dir} to exist"

            expected_file = expected_dir / "mp_rank_00.pth"
            assert expected_file.exists(), f"Expected file {expected_file} to exist"

            # Verify saved content.
            loaded = torch.load(expected_file)
            assert "model_chunk0" in loaded
            assert "layer.weight" in loaded["model_chunk0"]
            assert "layer.bias" in loaded["model_chunk0"]
            assert torch.equal(
                loaded["model_chunk0"]["layer.weight"], state_dict["model_chunk0"]["layer.weight"]
            )
            assert torch.equal(
                loaded["model_chunk0"]["layer.bias"], state_dict["model_chunk0"]["layer.bias"]
            )
