# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

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

    def test_first_iteration_loss_and_timer_reset_after_resume(self, monkeypatch):
        args = create_test_args()
        vars(args).update(
            consumed_train_samples=1,
            data_parallel_size=1,
            dsa_indexer_loss_coeff=None,
            log_energy=False,
            log_memory_interval=None,
            log_throughput=False,
            log_timers_to_tensorboard=False,
            micro_batch_size=1,
            mtp_num_layers=None,
            num_experts=None,
            record_memory_history=False,
            rl_use_sequence_packing=False,
            seq_length=1,
            skipped_train_samples=0,
            timing_log_level=0,
            world_size=1,
        )
        set_args(args)

        timers = MagicMock()
        timers('interval-time').elapsed.return_value = 1.0
        monkeypatch.setattr(training_module, 'get_timers', lambda: timers)
        monkeypatch.setattr(training_module, 'get_tensorboard_writer', lambda: None)
        monkeypatch.setattr(training_module, 'get_wandb_writer', lambda: None)
        monkeypatch.setattr(training_module, 'get_one_logger', lambda: None)
        monkeypatch.setattr(training_module, 'get_energy_monitor', lambda: None)
        monkeypatch.setattr(training_module, 'get_num_microbatches', lambda: 1)
        monkeypatch.setattr(
            training_module, 'reduce_max_stat_across_model_parallel_group', lambda value: value
        )
        monkeypatch.setattr(training_module, 'num_floating_point_operations', lambda *_: 1)
        log_strings = []
        monkeypatch.setattr(training_module, 'print_rank_last', log_strings.append)
        monkeypatch.setattr(training_module.one_logger_utils, 'track_app_tag', lambda *_: None)
        monkeypatch.setattr(training_module.one_logger_utils, 'track_e2e_metrics', lambda *_: None)

        original_tensor = torch.tensor

        def cpu_tensor(*tensor_args, **tensor_kwargs):
            tensor_kwargs.pop('device', None)
            return original_tensor(*tensor_args, **tensor_kwargs)

        monkeypatch.setattr(training_module.torch, 'tensor', cpu_tensor)

        for loaded_iteration, log_interval, expected_loss in (
            (0, 10, 1.0),
            (50, 1, 0.0),
            (50, 10, 0.0),
        ):
            args.iteration = loaded_iteration
            args.log_interval = log_interval
            total_loss_dict = {}
            timers.reset_mock()
            timers('interval-time').elapsed.return_value = 1.0

            training_module.training_log(
                loss_dict={'lm loss': original_tensor([1.0])},
                total_loss_dict=total_loss_dict,
                learning_rate=1e-4,
                iteration=loaded_iteration + 1,
                loss_scale=1.0,
                report_memory_flag=False,
                skipped_iter=0,
                grad_norm=None,
                params_norm=None,
                num_zeros_in_grad=None,
                max_attention_logit=None,
                is_first_iteration=True,
            )

            assert total_loss_dict['lm loss'].item() == expected_loss
            assert total_loss_dict['advanced iterations'] == int(loaded_iteration == 0)
            timers('interval-time').elapsed.assert_called_once_with(barrier=True, reset=False)
            timers.log.assert_called_once_with([], normalizer=1, reset=False)

        args.iteration = 50
        args.log_interval = 10
        total_loss_dict = {}
        log_strings.clear()
        timers.reset_mock()
        timers('interval-time').elapsed.side_effect = [1.0, 10.0]

        for iteration in range(51, 61):
            training_module.training_log(
                loss_dict={'lm loss': original_tensor([1.0])},
                total_loss_dict=total_loss_dict,
                learning_rate=1e-4,
                iteration=iteration,
                loss_scale=1.0,
                report_memory_flag=False,
                skipped_iter=0,
                grad_norm=None,
                params_norm=None,
                num_zeros_in_grad=None,
                max_attention_logit=None,
                is_first_iteration=iteration == 51,
            )

        assert 'elapsed time per iteration (ms): 1000.0' in log_strings[-1]
        assert 'lm loss: 1.000000E+00' in log_strings[-1]
        assert 'timer iterations' not in log_strings[-1]
        assert [
            call.kwargs['reset'] for call in timers('interval-time').elapsed.call_args_list
        ] == [False, True]
        assert [call.kwargs['normalizer'] for call in timers.log.call_args_list] == [1, 10]

    def teardown_method(self, method):
        Utils.destroy_model_parallel()


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
