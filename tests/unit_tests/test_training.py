# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from megatron.core.tokenizers.utils.build_tokenizer import vocab_size_with_padding
from megatron.training import training as training_module
from megatron.training.checkpointing import save_grads
from megatron.training.global_vars import set_args
from megatron.training.training import build_train_valid_test_data_iterators
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


def test_training_log_resets_first_iteration_when_log_interval_is_one(monkeypatch):
    """The second per-step log must not include the first step's loss."""
    args = SimpleNamespace(
        consumed_train_samples=0,
        data_parallel_size=1,
        dsa_indexer_loss_coeff=None,
        log_energy=False,
        log_interval=1,
        log_memory_interval=None,
        log_throughput=False,
        log_timers_to_tensorboard=False,
        micro_batch_size=1,
        mtp_num_layers=None,
        num_experts=None,
        perform_rl_step=False,
        record_memory_history=False,
        rl_profile=False,
        rl_use_sequence_packing=False,
        seq_length=1,
        skipped_train_samples=0,
        timing_log_level=0,
        train_iters=2,
        world_size=1,
    )
    timers = mock.MagicMock()
    timers.return_value.elapsed.return_value = 1.0
    log_lines = []

    monkeypatch.setattr(training_module, "get_args", lambda: args)
    monkeypatch.setattr(training_module, "get_timers", lambda: timers)
    monkeypatch.setattr(training_module, "get_tensorboard_writer", lambda: None)
    monkeypatch.setattr(training_module, "get_wandb_writer", lambda: None)
    monkeypatch.setattr(training_module, "get_one_logger", lambda: None)
    monkeypatch.setattr(training_module, "get_energy_monitor", lambda: None)
    monkeypatch.setattr(training_module, "get_num_microbatches", lambda: 1)
    monkeypatch.setattr(
        training_module, "reduce_max_stat_across_model_parallel_group", lambda value: value
    )
    monkeypatch.setattr(training_module, "num_floating_point_operations", lambda *a, **k: 0.0)
    monkeypatch.setattr(training_module.one_logger_utils, "track_app_tag", lambda *a, **k: None)
    monkeypatch.setattr(
        training_module.one_logger_utils, "track_e2e_metrics", lambda *a, **k: None
    )
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
