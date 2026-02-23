# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from megatron.core.tokenizers.utils.build_tokenizer import vocab_size_with_padding
from megatron.training.checkpointing import save_grads
from megatron.training.global_vars import set_args
from megatron.training.tokenizer.tokenizer import _vocab_size_with_padding
from megatron.training.training import build_train_valid_test_data_iterators, num_floating_point_operations
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


class TestFLOPsCalculation:
    """Tests for FLOPs calculation with different attention patterns."""

    def create_base_args(self):
        """Create base args for FLOPs testing."""
        args = SimpleNamespace()
        args.num_layers = 12
        args.hidden_size = 768
        args.num_attention_heads = 12
        args.kv_channels = 64
        args.seq_length = 2048
        args.ffn_hidden_size = 3072
        args.swiglu = False
        args.group_query_attention = False
        args.num_query_groups = 12
        args.attention_output_gate = False
        args.multi_latent_attention = False
        args.num_experts = None
        args.moe_layer_freq = None
        args.mtp_num_layers = None
        args.experimental_attention_variant = None
        args.linear_attention_freq = None
        args.hybrid_override_pattern = None
        args.hybrid_attention_ratio = 0.0
        args.hybrid_mlp_ratio = 0.0
        return args

    def test_full_causal_attention_baseline(self):
        """Test FLOPs calculation for standard full causal attention."""
        args = self.create_base_args()
        # No window_size or chunk_attention_size
        args.window_size = None
        args.chunk_attention_size = None

        batch_size = 8
        flops = num_floating_point_operations(args, batch_size)

        # FLOPs should be positive
        assert flops > 0, "FLOPs should be positive for baseline case"

        # Store baseline for comparison
        baseline_flops = flops
        return baseline_flops

    def test_sliding_window_attention_reduces_flops(self):
        """Test that sliding window attention reduces FLOPs compared to full attention."""
        args = self.create_base_args()
        batch_size = 8

        # Calculate baseline (full causal attention)
        args.window_size = None
        args.chunk_attention_size = None
        baseline_flops = num_floating_point_operations(args, batch_size)

        # Calculate with sliding window (window much smaller than seq_length)
        args.window_size = (512, 512)  # Much smaller than seq_length=2048
        sliding_window_flops = num_floating_point_operations(args, batch_size)

        # Sliding window should result in fewer FLOPs
        assert sliding_window_flops < baseline_flops, (
            f"Sliding window FLOPs ({sliding_window_flops}) should be less than "
            f"baseline FLOPs ({baseline_flops})"
        )

        # Calculate expected reduction ratio
        # For attention, effective_seq_len changes from 2048 to 512
        # The reduction should be approximately proportional to the window size
        reduction_ratio = sliding_window_flops / baseline_flops
        # Should see significant reduction (at least 20% savings)
        assert reduction_ratio < 0.95, (
            f"Expected significant FLOPs reduction with sliding window, "
            f"but got ratio {reduction_ratio}"
        )

    def test_sliding_window_with_infinite_window(self):
        """Test sliding window with -1 (infinite window) equals full attention."""
        args = self.create_base_args()
        batch_size = 8

        # Full attention baseline
        args.window_size = None
        args.chunk_attention_size = None
        baseline_flops = num_floating_point_operations(args, batch_size)

        # Sliding window with infinite window (-1)
        args.window_size = (-1, -1)
        infinite_window_flops = num_floating_point_operations(args, batch_size)

        # Should be the same as baseline
        assert abs(infinite_window_flops - baseline_flops) < 1e-6, (
            f"Infinite window FLOPs ({infinite_window_flops}) should equal "
            f"baseline FLOPs ({baseline_flops})"
        )

    def test_chunked_attention_reduces_flops(self):
        """Test that chunked attention reduces FLOPs compared to full attention."""
        args = self.create_base_args()
        batch_size = 8

        # Calculate baseline (full causal attention)
        args.window_size = None
        args.chunk_attention_size = None
        baseline_flops = num_floating_point_operations(args, batch_size)

        # Calculate with chunked attention (chunk_size much smaller than seq_length)
        args.chunk_attention_size = 256  # Much smaller than seq_length=2048
        chunked_flops = num_floating_point_operations(args, batch_size)

        # Chunked attention should result in fewer FLOPs
        assert chunked_flops < baseline_flops, (
            f"Chunked attention FLOPs ({chunked_flops}) should be less than "
            f"baseline FLOPs ({baseline_flops})"
        )

        # Calculate expected reduction ratio
        reduction_ratio = chunked_flops / baseline_flops
        # Should see significant reduction (at least 30% savings)
        assert reduction_ratio < 0.9, (
            f"Expected significant FLOPs reduction with chunked attention, "
            f"but got ratio {reduction_ratio}"
        )

    def test_gqa_with_sliding_window(self):
        """Test FLOPs calculation for GQA with sliding window attention."""
        args = self.create_base_args()
        args.group_query_attention = True
        args.num_query_groups = 4  # GQA with 4 groups
        batch_size = 8

        # GQA baseline
        args.window_size = None
        args.chunk_attention_size = None
        gqa_baseline_flops = num_floating_point_operations(args, batch_size)

        # GQA with sliding window
        args.window_size = (512, 512)
        gqa_sliding_flops = num_floating_point_operations(args, batch_size)

        # Sliding window should still reduce FLOPs for GQA
        assert gqa_sliding_flops < gqa_baseline_flops, (
            f"GQA with sliding window FLOPs ({gqa_sliding_flops}) should be less than "
            f"GQA baseline FLOPs ({gqa_baseline_flops})"
        )

    def test_mla_with_sliding_window(self):
        """Test FLOPs calculation for MLA with sliding window attention."""
        args = self.create_base_args()
        # Enable MLA
        args.multi_latent_attention = True
        args.q_lora_rank = None  # Use standard q projection
        args.kv_lora_rank = 512
        args.qk_head_dim = 64
        args.v_head_dim = 64
        args.qk_pos_emb_head_dim = 64
        batch_size = 8

        # MLA baseline
        args.window_size = None
        args.chunk_attention_size = None
        mla_baseline_flops = num_floating_point_operations(args, batch_size)

        # MLA with sliding window
        args.window_size = (512, 512)
        mla_sliding_flops = num_floating_point_operations(args, batch_size)

        # Sliding window should reduce FLOPs for MLA
        assert mla_sliding_flops < mla_baseline_flops, (
            f"MLA with sliding window FLOPs ({mla_sliding_flops}) should be less than "
            f"MLA baseline FLOPs ({mla_baseline_flops})"
        )

    def test_chunk_attention_takes_precedence_over_sliding_window(self):
        """Test that chunk_attention_size takes precedence over window_size."""
        args = self.create_base_args()
        batch_size = 8

        # Only chunk attention
        args.window_size = None
        args.chunk_attention_size = 256
        chunk_only_flops = num_floating_point_operations(args, batch_size)

        # Both chunk and sliding window (chunk should take precedence)
        args.window_size = (1024, 1024)
        args.chunk_attention_size = 256
        both_flops = num_floating_point_operations(args, batch_size)

        # Should be identical since chunk takes precedence
        assert abs(both_flops - chunk_only_flops) < 1e-6, (
            f"Chunk attention should take precedence. "
            f"chunk_only: {chunk_only_flops}, both: {both_flops}"
        )

    @pytest.mark.parametrize("window_size", [
        (128, 128),
        (256, 512),
        (1024, 2048),
        (2048, -1),  # One finite, one infinite
    ])
    def test_various_window_sizes(self, window_size):
        """Test FLOPs calculation with various window sizes."""
        args = self.create_base_args()
        args.window_size = window_size
        args.chunk_attention_size = None
        batch_size = 8

        flops = num_floating_point_operations(args, batch_size)

        # FLOPs should always be positive
        assert flops > 0, f"FLOPs should be positive for window_size={window_size}"

    @pytest.mark.parametrize("chunk_size", [64, 128, 256, 512, 1024])
    def test_various_chunk_sizes(self, chunk_size):
        """Test FLOPs calculation with various chunk sizes."""
        args = self.create_base_args()
        args.window_size = None
        args.chunk_attention_size = chunk_size
        batch_size = 8

        flops = num_floating_point_operations(args, batch_size)

        # FLOPs should always be positive
        assert flops > 0, f"FLOPs should be positive for chunk_size={chunk_size}"

    def test_flops_scale_with_batch_size(self):
        """Test that FLOPs scale linearly with batch size."""
        args = self.create_base_args()
        args.window_size = (512, 512)
        args.chunk_attention_size = None

        batch_size_1 = 1
        batch_size_8 = 8

        flops_1 = num_floating_point_operations(args, batch_size_1)
        flops_8 = num_floating_point_operations(args, batch_size_8)

        # Should scale linearly
        ratio = flops_8 / flops_1
        assert abs(ratio - 8.0) < 0.01, (
            f"FLOPs should scale linearly with batch size, "
            f"expected ratio ~8.0, got {ratio}"
        )
