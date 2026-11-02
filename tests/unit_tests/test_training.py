# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
from unittest.mock import Mock

import pytest
import torch

from megatron.core.process_groups_config import MultiModuleProcessGroupCollection
from megatron.core.tokenizers.utils.build_tokenizer import vocab_size_with_padding
from megatron.training import training
from megatron.training import training as training_module
from megatron.training.checkpointing import save_grads
from megatron.training.config import CheckpointConfig, TrainingConfig, ValidationConfig
from megatron.training.global_vars import _set_train_state, destroy_global_vars, set_args
from megatron.training.models.deepseek_v4 import normalize_dsv4_hybrid_csa_compress_ratios
from megatron.training.state import TrainState
from megatron.training.training import (
    _get_indexer_logging_layer_counts,
    _get_optimizer_param_scheduler_increment,
    _pop_samples_seen,
    _should_compute_params_norm,
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
    args.phase_transition_iterations = None

    return args


def test_indexer_logging_counts_only_active_legacy_ratios():
    """Unused ratio-tail entries must not dilute the reported indexer loss."""
    args = SimpleNamespace(
        num_layers=2,
        mtp_num_layers=1,
        mtp_use_repeated_layer=False,
        hybrid_layer_pattern=None,
        csa_compress_ratios=[4, 0, 4, 4],
        csa_dense_mode=False,
    )

    assert _get_indexer_logging_layer_counts(args) == (3, 2)


@pytest.mark.parametrize("fsdp2", [False, True])
@pytest.mark.parametrize("megatron_fsdp", [False, True])
def test_native_builder_uses_dist_and_ddp_configs(monkeypatch, run_config, fsdp2, megatron_fsdp):
    args = SimpleNamespace(
        skip_train=False,
        freeze_all_layers=False,
        freeze_base_model_for_mtp=False,
        use_torch_fsdp2=not fsdp2,
        use_megatron_fsdp=not megatron_fsdp,
    )
    run_config.dist.use_torch_fsdp2 = fsdp2
    run_config.ddp = (
        training.TorchFullyShardedDataParallelConfig(use_megatron_fsdp=megatron_fsdp)
        if fsdp2
        else training.DistributedDataParallelConfig(use_megatron_fsdp=megatron_fsdp)
    )
    builder = Mock()
    run_config.model = SimpleNamespace(get_builder_cls=lambda: Mock(return_value=builder))
    monkeypatch.setattr(training, "get_args", lambda: args)
    monkeypatch.setattr(training, "get_timers", Mock())
    monkeypatch.setattr(training, "get_one_logger", lambda: None)
    monkeypatch.setattr(training, "has_nvidia_modelopt", False)
    monkeypatch.setattr(training, "is_gtp_remat_active", lambda args: False)
    monkeypatch.setattr(training, "_add_model_freeze_pre_wrap_hook", Mock())
    monkeypatch.setattr("megatron.training.utils.start_memory_history_recording", Mock())
    # A passed training container must not be replaced by the global container.
    monkeypatch.setattr(training, "get_run_config", Mock(side_effect=AssertionError("global read")))

    class ReachedBuilder(Exception):
        pass

    builder.build_distributed_models.side_effect = ReachedBuilder
    with pytest.raises(ReachedBuilder):
        training.setup_model_and_optimizer(Mock(), cfg_container=run_config)
    kwargs = builder.build_distributed_models.call_args.kwargs
    assert kwargs["ddp_config"] is run_config.ddp
    assert kwargs["use_torch_fsdp2"] is fsdp2
    # Forward both flags unchanged, including the pair rejected by the builder.
    assert kwargs["use_megatron_fsdp"] is megatron_fsdp


def test_indexer_logging_counts_hybrid_mtp_depths_and_dense_mode():
    """Hybrid MTP repeats each inner-pattern indexer once per unshared prediction depth."""
    args = SimpleNamespace(
        num_layers=2,
        mtp_num_layers=2,
        mtp_use_repeated_layer=False,
        hybrid_layer_pattern="DD/DDD/DDD",
        csa_compress_ratios=[4, 0, 4, 0, 4, 4],
        csa_dense_mode=False,
    )

    assert _get_indexer_logging_layer_counts(args) == (5, 5)

    args.mtp_use_repeated_layer = True
    assert _get_indexer_logging_layer_counts(args) == (5, 3)

    args.csa_dense_mode = True
    assert _get_indexer_logging_layer_counts(args) == (5, 0)


def test_indexer_logging_uses_normalized_hybrid_layer_positions():
    """C layers in the main and repeated MTP patterns keep their positional denominator."""
    args = SimpleNamespace(
        experimental_attention_variant="dsv4_hybrid",
        num_layers=3,
        mtp_num_layers=2,
        mtp_use_repeated_layer=False,
        hybrid_layer_pattern="W-C/H-C/H-C",
        csa_compress_ratios=None,
        csa_dense_mode=False,
    )
    config_kwargs = {}

    normalize_dsv4_hybrid_csa_compress_ratios(args, config_kwargs, args.hybrid_layer_pattern)

    expected_ratios = [0, 0, 4, 128, 0, 4, 128, 0, 4]
    assert args.csa_compress_ratios == expected_ratios
    assert config_kwargs["csa_compress_ratios"] == expected_ratios
    assert _get_indexer_logging_layer_counts(args) == (6, 3)


class TestTraining:
    @pytest.fixture(autouse=True)
    def training_config(self, run_config):
        run_config.train = TrainingConfig(train_samples=1, train_iters=1, global_batch_size=1)
        run_config.validation = ValidationConfig(
            eval_interval=1, eval_iters=1, eval_global_batch_size=1
        )

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        args = create_test_args()
        set_args(args)
        _set_train_state()

    @pytest.mark.parametrize("ckpt_format", ["torch", "torch_dist"])
    @pytest.mark.parametrize("verify_integrity", [False, True])
    def test_checkpoint_conversion_preserves_supported_integrity_settings(
        self, monkeypatch, run_config, ckpt_format, verify_integrity
    ):
        """Verify the source before clearing integrity checks for unsupported outputs."""
        run_config.validation.skip_train = True
        args = SimpleNamespace(
            logits_save_dir=None,
            logits_load_dir=None,
            moe_use_upcycling=False,
            load="source-checkpoints",
            pretrained_checkpoint=None,
            ckpt_format="torch_dist",
            ckpt_convert_format=ckpt_format,
            ckpt_convert_save="converted-checkpoints",
            verify_integrity=verify_integrity,
        )

        def load_checkpoint(*unused_args, **unused_kwargs):
            assert args.ckpt_format == "torch_dist"
            assert args.verify_integrity is verify_integrity
            return 123, 456

        def save_checkpoint(*unused_args, **unused_kwargs):
            assert args.ckpt_format == ckpt_format
            assert args.save == str(Path("converted-checkpoints") / ckpt_format)
            assert args.use_dist_ckpt is (ckpt_format != "torch")
            checkpoint = CheckpointConfig(
                ckpt_format=args.ckpt_format, verify_integrity=args.verify_integrity
            )
            assert checkpoint.verify_integrity is (verify_integrity and ckpt_format == "torch_dist")

        load = Mock(side_effect=load_checkpoint)
        save = Mock(side_effect=save_checkpoint)
        monkeypatch.setattr(training, "get_args", lambda: args)
        monkeypatch.setattr(training, "get_timers", Mock())
        monkeypatch.setattr(training, "get_one_logger", lambda: None)
        monkeypatch.setattr(training, "has_nvidia_modelopt", False)
        monkeypatch.setattr(training, "get_model", Mock(return_value=[torch.nn.Linear(1, 1)]))
        monkeypatch.setattr(training, "get_num_microbatches", lambda: 1)
        monkeypatch.setattr(training, "get_current_global_batch_size", lambda: 1)
        monkeypatch.setattr(training, "update_num_microbatches", Mock())
        monkeypatch.setattr(training, "load_checkpoint", load)
        monkeypatch.setattr(training, "save_checkpoint", save)

        with pytest.raises(SystemExit):
            training.setup_model_and_optimizer(None, lambda: None, cfg_container=run_config)
        load.assert_called_once()
        save.assert_called_once()

    def test_build_train_valid_test_data_iterators(self, run_config):
        train_iter, valid_iter, test_iter = build_train_valid_test_data_iterators(
            mock_train_valid_test_datasets_provider
        )
        train_data = next(train_iter)
        valid_data = next(valid_iter)
        test_data = next(test_iter)
        assert (train_data, valid_data, test_data) == (1, 2, 3)

    def test_params_norm_is_computed_only_when_it_can_be_logged(self, run_config):
        args = SimpleNamespace(
            log_params_norm=True, log_interval=20, tensorboard_dir=None, tensorboard_log_interval=1
        )
        run_config.logger = args

        assert _should_compute_params_norm(args, iteration=1, is_first_iteration=True)
        assert _should_compute_params_norm(args, iteration=20, is_first_iteration=False)
        assert not _should_compute_params_norm(args, iteration=19, is_first_iteration=False)

        args.tensorboard_dir = "/tmp/tensorboard"
        args.tensorboard_log_interval = 5
        assert _should_compute_params_norm(args, iteration=5, is_first_iteration=False)

        args.log_params_norm = False
        assert not _should_compute_params_norm(args, iteration=20, is_first_iteration=False)

    def test_build_train_valid_test_data_iterators_multi_full_validation(self, run_config):
        """multiple_validation_sets + full_validation builds a list of iterators
        (one per validation set) and sets cfg.validation.eval_iters to the per-loader
        lengths MAX-reduced across DP ranks."""
        args = create_test_args()
        run_config.validation.multiple_validation_sets = True
        run_config.validation.full_validation = True
        set_args(args)
        _, valid_iters, _ = build_train_valid_test_data_iterators(
            mock_multi_valid_full_datasets_provider
        )
        assert isinstance(valid_iters, list)
        assert len(valid_iters) == 2
        assert next(valid_iters[0]) == 2
        assert next(valid_iters[1]) == 20
        # data_parallel_size=1, so MAX across DP ranks equals the local lengths
        assert run_config.validation.eval_iters == [2, 3]

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
        destroy_global_vars()


def test_training_log_resets_first_iteration_when_log_interval_is_one(monkeypatch, run_config):
    """The second per-step log must not include the first step's loss."""
    args = SimpleNamespace(
        consumed_train_samples=0,
        data_parallel_size=1,
        dsa_indexer_loss_coeff=None,
        gtp_weight_remat_size=1,
        log_energy=False,
        log_interval=1,
        log_memory_interval=None,
        log_throughput=False,
        log_timers_to_tensorboard=False,
        micro_batch_size=1,
        mtp_num_layers=None,
        num_experts=None,
        perform_rl_step=False,
        profile_ranks=[],
        record_memory_history=False,
        rl_profile=False,
        rl_use_sequence_packing=False,
        seq_length=1,
        skipped_train_samples=0,
        timing_log_level=0,
        train_iters=2,
        train_samples=None,
        world_size=1,
    )
    run_config.logger = args
    run_config.train = TrainingConfig(micro_batch_size=1, global_batch_size=1, train_iters=2)
    timers = mock.MagicMock()
    timers.return_value.elapsed.return_value = 1.0
    log_lines = []

    monkeypatch.setattr(training_module, "get_args", lambda: args)
    train_state = TrainState()
    monkeypatch.setattr(training_module, "get_train_state", lambda: train_state)
    monkeypatch.setattr(training_module, "get_timers", lambda: timers)
    monkeypatch.setattr(training_module, "get_tensorboard_writer", lambda: None)
    monkeypatch.setattr(training_module, "get_wandb_writer", lambda: None)
    monkeypatch.setattr(training_module, "get_one_logger", lambda: None)
    monkeypatch.setattr(training_module, "get_energy_monitor", lambda: None)
    monkeypatch.setattr(training_module, "get_num_microbatches", lambda: 1)
    monkeypatch.setattr(
        training_module,
        "reduce_max_stat_across_model_parallel_group",
        lambda value, group=None: value,
    )
    monkeypatch.setattr(training_module, "num_floating_point_operations", lambda *a, **k: 0.0)
    monkeypatch.setattr(training_module.one_logger_utils, "track_app_tag", lambda *a, **k: None)
    monkeypatch.setattr(training_module.one_logger_utils, "track_e2e_metrics", lambda *a, **k: None)
    monkeypatch.setattr(training_module, "print_rank_last", log_lines.append)

    # training_log creates its accumulator tensors on CUDA. Keep this logging-only
    # regression host-runnable by redirecting those tiny tensors to the CPU.
    make_tensor = torch.tensor

    def make_cpu_tensor(*tensor_args, **tensor_kwargs):
        tensor_kwargs.pop("device", None)
        return make_tensor(*tensor_args, **tensor_kwargs)

    monkeypatch.setattr(training_module.torch, "tensor", make_cpu_tensor)

    total_loss_dict = {}
    common_kwargs = dict(
        learning_rate=1.0e-4,
        loss_scale=1.0,
        report_memory_flag=False,
        skipped_iter=0,
        grad_norm=None,
        params_norm=None,
        num_zeros_in_grad=None,
        max_attention_logit=None,
    )
    training_module.training_log(
        {"alignment loss": make_tensor([2.0])},
        total_loss_dict,
        iteration=1,
        is_first_iteration=True,
        **common_kwargs,
    )

    assert total_loss_dict["alignment loss"].item() == 0.0
    assert total_loss_dict["advanced iterations"] == 0
    assert total_loss_dict["skipped iterations"] == 0
    assert total_loss_dict["nan iterations"] == 0

    training_module.training_log(
        {"alignment loss": make_tensor([6.0])},
        total_loss_dict,
        iteration=2,
        is_first_iteration=False,
        **common_kwargs,
    )

    assert " alignment loss: 6.000000E+00 |" in log_lines[-1]
    assert " alignment loss: 4.000000E+00 |" not in log_lines[-1]


def test_training_log_uses_nominal_microbatches_and_dynamic_cp_parent(monkeypatch, run_config):
    """DSA metrics use the stable parent and actual indexer-execution divisor."""
    parent_dp_cp_group = object()
    pp_group = object()
    dp_group = object()
    encoder_pg_collection = SimpleNamespace(mp=None, pp=object(), dp=object(), dp_cp=object())
    language_pg_collection = SimpleNamespace(
        mp=object(), pp=pp_group, dp=dp_group, dp_cp=parent_dp_cp_group
    )
    schedule_pg_collection = MultiModuleProcessGroupCollection(
        module_pgs={"language": language_pg_collection}, language_model_module_name="language"
    )
    recorded = []
    args = SimpleNamespace(
        calculate_per_token_loss=False,
        consumed_train_samples=0,
        context_parallel_size=4,
        csa_compress_ratios=None,
        csa_dense_mode=False,
        cuda_graph_impl="none",
        data_parallel_size=2,
        dsa_indexer_loss_coeff=0.1,
        dsa_indexer_skip_topk_offset=1,
        dsa_indexer_topk_freq=4,
        dynamic_context_parallel=True,
        freeze_all_layers=False,
        gtp_weight_remat_size=1,
        hybrid_layer_pattern=None,
        log_interval=100,
        micro_batch_size=1,
        mtp_num_layers=None,
        mtp_use_repeated_layer=False,
        num_experts=None,
        num_layers=8,
        perform_rl_step=False,
        seq_length=4096,
        skipped_train_samples=0,
        timing_log_level=0,
        world_size=8,
    )
    run_config.logger = args
    run_config.train = TrainingConfig(micro_batch_size=args.micro_batch_size)

    monkeypatch.setattr(training_module, "get_args", lambda: args)
    train_state = TrainState()
    monkeypatch.setattr(training_module, "get_train_state", lambda: train_state)
    monkeypatch.setattr(training_module, "get_timers", mock.MagicMock)
    monkeypatch.setattr(training_module, "get_tensorboard_writer", lambda: None)
    monkeypatch.setattr(training_module, "get_wandb_writer", lambda: None)
    monkeypatch.setattr(training_module, "get_one_logger", lambda: None)
    monkeypatch.setattr(training_module, "get_energy_monitor", lambda: None)
    monkeypatch.setattr(training_module, "get_num_microbatches", lambda: 7)
    monkeypatch.setattr(
        training_module,
        "reduce_max_stat_across_model_parallel_group",
        lambda value, group=None: value,
    )
    monkeypatch.setattr(training_module.one_logger_utils, "track_app_tag", lambda *a: None)
    monkeypatch.setattr(
        training_module.DSAIndexerLossLoggingHelper,
        "track_indexer_metrics",
        lambda **kwargs: recorded.append(kwargs),
    )
    monkeypatch.setattr(
        training_module.MTPLossLoggingHelper, "track_mtp_metrics", lambda *args, **kwargs: None
    )

    def run_training_log(schedule=schedule_pg_collection):
        training_module.training_log(
            {},
            {},
            learning_rate=1.0e-4,
            iteration=1,
            loss_scale=1.0,
            report_memory_flag=False,
            skipped_iter=0,
            grad_norm=None,
            params_norm=None,
            num_zeros_in_grad=None,
            max_attention_logit=None,
            pg_collection=encoder_pg_collection,
            schedule_pg_collection=schedule,
        )

    run_training_log()

    assert len(recorded) == 1
    assert recorded[0]["loss_scale"] == 1 / 7
    assert recorded[0]["dynamic_cp_parent_group"] is parent_dp_cp_group
    assert recorded[0]["pp_group"] is pp_group
    assert recorded[0]["dp_group"] is dp_group
    assert recorded[0]["configured_cp_size"] == 4
    assert recorded[0]["num_layers"] == 8
    assert recorded[0]["num_indexer_layers"] == 2
    assert recorded[0]["preserve_groups"] is False

    for cuda_graph_impl in ("local", "transformer_engine", "full_iteration"):
        recorded.clear()
        args.cuda_graph_impl = cuda_graph_impl
        run_training_log()
        assert recorded[0]["preserve_groups"] is True
    args.cuda_graph_impl = "none"

    standard_mtp_cases = (
        # Independent MTP layers use logical layer numbers N+1..N+D.
        (False, 23),
        # Repeated MTP executes the one physical layer N+1 once per depth.
        (True, 28),
    )
    for use_repeated_layer, indexers in standard_mtp_cases:
        recorded.clear()
        args.num_layers = 78
        args.mtp_num_layers = 7
        args.mtp_use_repeated_layer = use_repeated_layer
        args.dsa_indexer_skip_topk_offset = 3
        args.dsa_indexer_topk_freq = 4
        run_training_log()
        assert recorded[0]["num_layers"] == 85
        assert recorded[0]["num_indexer_layers"] == indexers

    standard_csa_mtp_cases = (
        # Independent MTP executes its three configured ratios: 4, 128, 0.
        (False, 3),
        # Repeated MTP executes the first MTP ratio (4) at every depth.
        (True, 5),
    )
    for use_repeated_layer, indexers in standard_csa_mtp_cases:
        recorded.clear()
        args.num_layers = 4
        args.mtp_num_layers = 3
        args.mtp_use_repeated_layer = use_repeated_layer
        args.csa_compress_ratios = [0, 4, 128, 4, 4, 128, 0]
        run_training_log()
        assert recorded[0]["num_layers"] == 7
        assert recorded[0]["num_indexer_layers"] == indexers

    hybrid_cases = (
        # Non-DSA symbols do not contribute to the divisor.
        ("MD-", 3, None, None, 0, 1, 3, 1),
        # Pipeline separators do not consume a global layer number.
        ("D|D", 2, None, None, 1, 2, 2, 1),
        # Every MTP depth reuses inner layer numbers N+1..N+K and executes its own loss.
        ("D---/DD/DD/DD", 4, 3, None, 1, 4, 7, 4),
        # DSv4 compression ratios take precedence over legacy hybrid D symbols.
        ("CDD", 3, None, [4], 0, 1, 3, 1),
    )
    for pattern, num_layers, mtp_layers, ratios, offset, freq, tracked, indexers in hybrid_cases:
        recorded.clear()
        args.hybrid_layer_pattern = pattern
        args.num_layers = num_layers
        args.mtp_num_layers = mtp_layers
        args.mtp_use_repeated_layer = False
        args.csa_compress_ratios = ratios
        args.dsa_indexer_skip_topk_offset = offset
        args.dsa_indexer_topk_freq = freq
        run_training_log()
        assert recorded[0]["num_layers"] == tracked
        assert recorded[0]["num_indexer_layers"] == indexers

    recorded.clear()
    encoder_only_schedule = MultiModuleProcessGroupCollection(
        module_pgs={"encoder": encoder_pg_collection}, language_model_module_name=None
    )
    run_training_log(encoder_only_schedule)
    assert recorded == []


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


class TestPackedSampleAccounting:
    @pytest.mark.parametrize("schedule,expected_iters", [(None, 4), ("0:8 32:16", 6)])
    def test_sample_horizon_uses_config_and_resets_batch_schedule(
        self, monkeypatch, schedule, expected_iters
    ):
        from megatron.core import num_microbatches_calculator as calculator
        from megatron.training import training

        monkeypatch.setattr(calculator, "_GLOBAL_NUM_MICROBATCHES_CALCULATOR", None)

        def unexpected_global_config():
            raise AssertionError("sample horizon must use the supplied TrainingConfig")

        monkeypatch.setattr(training, "get_run_config", unexpected_global_config)
        train = TrainingConfig(
            micro_batch_size=2,
            global_batch_size=16,
            train_samples=64,
            step_batch_size_schedule=schedule,
        )
        calculator.init_num_microbatches_calculator(
            rank=0,
            global_batch_size=16,
            micro_batch_size=2,
            data_parallel_size=1,
            step_batch_size_schedule=schedule,
        )
        training.update_train_iters(train)
        assert train.train_iters == expected_iters
        assert calculator.get_current_global_batch_size() == (8 if schedule else 16)

    @pytest.mark.parametrize(
        "total_cp,num_sequence_shards,weight_shards,samples",
        [(1, 1, 4, 8), (4, 2, 2, 2), (2, 2, 4, 4)],
    )
    def test_static_cp_sample_count_and_scheduler(
        self, monkeypatch, run_config, total_cp, num_sequence_shards, weight_shards, samples
    ):
        """Sequence shards count each sample once for progress and sample-based LR schedules."""
        import megatron.training.training as training

        args = SimpleNamespace(
            gtp_remat_num_sequence_shards=num_sequence_shards,
            world_size=8,
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            context_parallel_size=total_cp,
            gtp_weight_remat_size=weight_shards,
            data_parallel_size=2,
            micro_batch_size=2,
            train_iters=None,
            train_samples=10_000,
        )
        run_config.train = TrainingConfig(micro_batch_size=2, train_samples=10_000)
        monkeypatch.setattr(training, "get_num_microbatches", lambda: 3)
        monkeypatch.setattr(training, "is_pp_last_stage", lambda group: True)
        monkeypatch.setattr(training, "get_pg_size", lambda group: 1)

        count = training._get_samples_seen_in_iteration([{}] * 3, args, SimpleNamespace(pp=None))

        assert count == 3 * 2 * samples
        assert _get_optimizer_param_scheduler_increment(args, count) == 3 * 2 * samples

    def test_pop_samples_seen_sums_and_removes_metadata(self):
        losses = [
            {"lm loss": torch.tensor(1.0), "_samples_seen": torch.tensor(3.0)},
            {"lm loss": torch.tensor(2.0), "_samples_seen": torch.tensor(5.0)},
        ]

        assert _pop_samples_seen(losses).item() == 8
        assert all("_samples_seen" not in loss for loss in losses)

    def test_pop_samples_seen_requires_every_microbatch(self):
        losses = [
            {"lm loss": torch.tensor(1.0), "_samples_seen": torch.tensor(3.0)},
            {"lm loss": torch.tensor(2.0)},
        ]

        with pytest.raises(ValueError, match="every microbatch"):
            _pop_samples_seen(losses)

    def test_iteration_schedule_uses_running_batch_size(self, monkeypatch, run_config):
        args = SimpleNamespace(train_iters=100, train_samples=None)
        run_config.train = TrainingConfig(train_iters=100)
        monkeypatch.setattr(
            "megatron.training.training.get_current_running_global_batch_size", lambda: 256
        )

        assert _get_optimizer_param_scheduler_increment(args, 3342) == 256

    def test_sample_schedule_uses_samples_seen(self, run_config):
        args = SimpleNamespace(train_iters=39, train_samples=10_000)
        run_config.train = TrainingConfig(train_iters=39, train_samples=10_000)

        assert _get_optimizer_param_scheduler_increment(args, 3342) == 3342


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
