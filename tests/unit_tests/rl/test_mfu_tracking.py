# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from megatron.core.inference.inference_flops import InferenceFLOPsCalculator
from megatron.training.mfu_tracker import MFUTracker


class TestMFUTracking:
    """End-to-end tests for inference FLOPs calculation and MFU tracking."""

    @pytest.mark.parametrize(
        "decode_tokens,prefill_tokens,active_blocks,active_reqs,num_prefill_reqs",
        [
            (32, 0, 4, 32, 0),  # decode-only
            (0, 128, 2, 2, 2),  # prefill-only
            (16, 64, 4, 18, 2),  # mixed
            (0, 0, 0, 0, 0),  # empty step
        ],
        ids=["decode", "prefill", "mixed", "empty"],
    )
    def test_inference_flops_e2e(
        self, decode_tokens, prefill_tokens, active_blocks, active_reqs, num_prefill_reqs
    ):
        """InferenceFLOPsCalculator.from_args -> compute_step_flops produces consistent results."""
        args = SimpleNamespace(
            hidden_size=256,
            padded_vocab_size=1024,
            num_attention_heads=8,
            num_query_groups=4,
            kv_channels=32,
            ffn_hidden_size=512,
            num_layers=8,
            swiglu=True,
            hybrid_override_pattern="M*M*-EMM*",
            num_experts=4,
            moe_router_topk=2,
            moe_ffn_hidden_size=512,
            moe_shared_expert_intermediate_size=256,
            mamba_num_heads=4,
            mamba_head_dim=64,
            mamba_state_dim=128,
            mamba_num_groups=8,
            mamba_d_conv=4,
            inference_dynamic_batching_block_size=256,
        )
        calc = InferenceFLOPsCalculator.from_args(args)
        total_tokens = decode_tokens + prefill_tokens
        result = calc.compute_step_flops(
            decode_tokens=decode_tokens,
            prefill_tokens=prefill_tokens,
            total_tokens=total_tokens,
            active_blocks=active_blocks,
            active_reqs=active_reqs,
            num_prefill_reqs=num_prefill_reqs,
        )
        assert result['total_flops'] == result['decode_flops'] + result['prefill_flops']
        if total_tokens == 0:
            assert result['total_flops'] == 0.0
        else:
            assert result['total_flops'] > 0
        if active_reqs > 0:
            assert result['t_avg'] == active_blocks * 256 / active_reqs
        # Prefill should grow super-linearly (quadratic attention term)
        if prefill_tokens > 0 and num_prefill_reqs > 0:
            r2 = calc.compute_step_flops(
                decode_tokens=0,
                prefill_tokens=prefill_tokens * 2,
                total_tokens=prefill_tokens * 2,
                active_blocks=active_blocks,
                active_reqs=active_reqs,
                num_prefill_reqs=num_prefill_reqs,
            )
            assert r2['prefill_flops'] / result['prefill_flops'] > 2.0

    @pytest.mark.parametrize(
        "inf_flops,inf_time,inf_tokens,train_flops,train_time,train_tokens,peak",
        [
            (50e12, 5.0, 1000, 50e12, 5.0, 2000, 100.0),  # balanced
            (0, 0, 0, 100e12, 10.0, 5000, 100.0),  # training only
            (100e12, 10.0, 3000, 0, 0, 0, 100.0),  # inference only
            (1e12, 1.0, 100, 1e12, 1.0, 100, 0.0),  # zero peak
        ],
        ids=["balanced", "train-only", "infer-only", "zero-peak"],
    )
    def test_mfu_tracker_e2e(
        self, inf_flops, inf_time, inf_tokens, train_flops, train_time, train_tokens, peak
    ):
        """MFUTracker accumulates per-GPU FLOPs and computes correct MFU."""
        tracker = MFUTracker()
        if inf_flops:
            tracker.add_inference_flops(inf_flops, inf_time, tokens=inf_tokens)
        if train_flops:
            tracker.add_training_flops(train_flops, train_time, tokens=train_tokens)

        # Per-iteration accessors match what was added
        assert tracker.get_iter_inference_flops() == inf_flops
        assert tracker.get_iter_inference_time() == inf_time
        assert tracker.get_iter_inference_tokens() == inf_tokens

        report = tracker.get_report(gpu_peak_tflops=peak)
        total_time = inf_time + train_time
        total_tflops = inf_flops / 1e12 + train_flops / 1e12

        if peak > 0 and total_time > 0:
            expected_mfu = total_tflops / total_time / peak * 100.0
            assert abs(report['total_mfu'] - expected_mfu) < 1e-9
        else:
            assert report['total_mfu'] == 0.0

        # No world_size division — FLOPs are already per-GPU
        if total_time > 0:
            assert abs(report['total_throughput'] - total_tflops / total_time) < 1e-9

        # reset_iter clears per-iteration but keeps cumulative
        tracker.reset_iter()
        assert tracker.get_iter_inference_flops() == 0.0
        assert tracker.get_report(gpu_peak_tflops=peak)['total_tflops'] == total_tflops
