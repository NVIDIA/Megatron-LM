# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Stable regressions for mHC full-recompute failures found by a seed sweep.

CPU seeds select weights; CUDA/tracker seed 123 selects the original inputs and
upstream gradients. Every replica exercises every known failing seed, including
seed 10's alpha_res gradient mismatch, without weakening the original assertions.
"""

import json
import sys
import traceback

import pytest
import torch

from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.hyper_connection import HyperConnectionModule
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.models import test_hybrid_mhc as _original_mhc
from tests.unit_tests.test_utilities import Utils


@pytest.mark.internal
class TestHybridMHCFullRecomputeRegression:
    def setup_method(self):
        Utils.initialize_model_parallel(1, 1)

    def teardown_method(self):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("cpu_seed", [10, 87, 133, 196, 287, 339, 414, 473])
    def test_known_fp32_block_learned_failures(self, cpu_seed):
        torch.manual_seed(cpu_seed)
        model_parallel_cuda_manual_seed(123)
        print(
            "MHC_FIXED_REGRESSION "
            + json.dumps(
                dict(rank=torch.distributed.get_rank(), cpu_seed=cpu_seed, cuda_seed=123),
                sort_keys=True,
            ),
            flush=True,
        )
        try:
            _original_mhc.TestHybridMHCContraction().test_full_recompute_matches_forward_and_all_gradients(
                learned_output_contract=True,
                recompute_method="block",
                fp32_mixing=False,
                autocast_forward=False,
                distribute_saved_activations=False,
            )
        except BaseException:
            traceback.print_exc(file=sys.stdout)
            sys.stdout.flush()
            sys.stderr.flush()
            raise

    @pytest.mark.parametrize("eps_inside_sqrt", [False, True])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
    def test_projection_matches_with_and_without_grad(self, eps_inside_sqrt, dtype):
        torch.manual_seed(10)
        model_parallel_cuda_manual_seed(123)
        config = TransformerConfig(
            num_layers=1,
            hidden_size=32,
            num_attention_heads=4,
            num_residual_streams=4,
            enable_hyper_connections=True,
            mhc_norm_eps_inside_sqrt=eps_inside_sqrt,
            use_cpu_initialization=True,
        )
        module = HyperConnectionModule(config, layer_number=1).cuda()
        hidden = torch.randn(8, 2, 128, device="cuda", dtype=dtype, requires_grad=True)
        saved = []

        def pack(tensor):
            saved.append((tensor.dtype, tuple(tensor.shape)))
            return tensor

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            projection, norm = module._projection_and_get_norm(hidden)
        with torch.no_grad():
            replay_projection, replay_norm = module._projection_and_get_norm(hidden)
        torch.testing.assert_close(replay_projection, projection, rtol=0, atol=0)
        torch.testing.assert_close(replay_norm, norm, rtol=0, atol=0)
        assert (dtype, (16, 128)) in saved
        if dtype == torch.bfloat16:
            assert (torch.float32, (16, 128)) not in saved
