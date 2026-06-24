# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for GTP + Muon (LayerWise) distributed checkpointing.

Covers the optimizer-state checkpoint roundtrip for the
:class:`LayerWiseDistributedOptimizer` (Muon) under GTP, where GTP-replicated
matrix params (e.g. the MoE router) are kept whole and must be disambiguated
by ``replica_id`` so DCP does not see multiple writers for the same shard.
"""

import torch

from megatron.core.dist_checkpointing import load, save
from tests.unit_tests.dist_checkpointing import TempNamedDir, setup_model_and_optimizer
from tests.unit_tests.test_utilities import Utils


def check_equal(input_1, input_2):
    """Check if two inputs are equal, used for checking checkpointing."""
    if isinstance(input_1, dict) and isinstance(input_2, dict):
        assert input_1.keys() == input_2.keys()
        for key in input_1.keys():
            check_equal(input_1[key], input_2[key])
    elif isinstance(input_1, list) and isinstance(input_2, list):
        assert len(input_1) == len(input_2)
        for i in range(len(input_1)):
            check_equal(input_1[i], input_2[i])
    elif isinstance(input_1, torch.Tensor) and isinstance(input_2, torch.Tensor):
        assert torch.all(input_1 == input_2), f"Input 1: {input_1} != Input 2: {input_2}"
    elif type(input_1) != type(input_2):
        assert False, f"Input 1 type: {type(input_1)} != Input 2 type: {type(input_2)}"
    else:
        assert input_1 == input_2, f"Input 1: {input_1} != Input 2: {input_2}"


class TestGTPMuonDCP:
    """GTP + Muon (LayerWise) distributed checkpointing tests."""

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_gtp_muon_moe_save_load(self, tmp_path_dist_ckpt):
        """GTP + Muon (LayerWise) optimizer-state checkpoint roundtrip.

        GTP-REPLICATED, Muon-managed matrix params (e.g. the MoE router, held identically on every
        GTP peer) must not collide on GTP peers during checkpoint save: LayerWise keeps each such
        param whole, so its optimizer-state ShardedTensor has the same key+offset on all GTP peers
        and the replica_id must distinguish them, or DCP validate_sharding_integrity reports 2
        writers ('Invalid access pattern ... [[2]]'). Adam dodges this by sharding the state.
        """
        import os
        from functools import partial

        import pytest

        from megatron.experimental.gtp import HAVE_GTP

        if not HAVE_GTP:
            pytest.skip("GTP requires TE with hook registry")
        if int(os.environ.get('WORLD_SIZE', '1')) != 4:
            pytest.skip("Requires world_size 4 (gtp2 x dp2)")

        os.environ['MEGATRON_GTP_FORCE_ENABLE'] = '1'
        from megatron.core import parallel_state as ps
        from megatron.core.tensor_parallel import model_parallel_cuda_manual_seed
        from megatron.experimental.gtp import GTP_CONFIG, GTPShardedParam, update_gtp_config
        from tests.unit_tests.dist_checkpointing.utils import initialize_moe_model

        Utils.initialize_model_parallel(1, 1)  # bootstrap torch.distributed + model parallel
        ps.destroy_model_parallel()
        ps.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1, gtp_remat_size=2
        )
        model_parallel_cuda_manual_seed(2)
        # Disable GTP alignment padding so the tiny test dims slice cleanly by gtp_size.
        _orig_pad = GTP_CONFIG.pad_for_alignment
        update_gtp_config(pad_for_alignment=0)
        # GTP-friendly dims (divisible by gtp_size=2); GPU init (CPU affine init is not GTP-aware
        # for the strided QKV weight).
        moe_cfg = dict(
            hidden_size=64,
            num_attention_heads=8,
            kv_channels=8,
            ffn_hidden_size=128,
            use_cpu_initialization=False,
        )
        meta = {'distrib_optim_sharding_type': 'dp_reshardable'}
        with TempNamedDir(tmp_path_dist_ckpt / 'gtp_muon_moe_A', sync=True) as ckpt_dir_A:
            with TempNamedDir(tmp_path_dist_ckpt / 'gtp_muon_moe_B', sync=True) as ckpt_dir_B:
                model_A, optimizer_A = setup_model_and_optimizer(
                    seed=2,
                    tp=1,
                    pp=1,
                    bf16=True,
                    dist_opt=True,
                    use_param_layout=True,
                    initialize_fn=partial(initialize_moe_model, use_te=True, **moe_cfg),
                    optimizer='dist_muon',
                )
                assert any(
                    isinstance(p, GTPShardedParam) for p in model_A[0].parameters()
                ), "GTP not active: no GTPShardedParam in the GTP=2 MoE model"

                model_sd_A = model_A[0].sharded_state_dict()
                optim_sd_A = optimizer_A.sharded_state_dict(model_sd_A, metadata=meta)
                save(
                    optim_sd_A, ckpt_dir_A
                )  # fails (2 writers) before the LayerWise replica_id fix

                model_B, optimizer_B = setup_model_and_optimizer(
                    seed=3,
                    tp=1,
                    pp=1,
                    bf16=True,
                    dist_opt=True,
                    use_param_layout=True,
                    initialize_fn=partial(initialize_moe_model, use_te=True, **moe_cfg),
                    optimizer='dist_muon',
                )
                model_sd_B = model_B[0].sharded_state_dict()
                load_sharded_sd = optimizer_B.sharded_state_dict(
                    model_sd_B, is_loading=True, metadata=meta
                )
                state_dict = load(load_sharded_sd, ckpt_dir_A)
                optimizer_B.load_state_dict(state_dict)
                optim_sd_B = optimizer_B.sharded_state_dict(model_sd_B, metadata=meta)
                save(optim_sd_B, ckpt_dir_B)

                update_gtp_config(pad_for_alignment=_orig_pad)

                Utils.destroy_model_parallel()
                Utils.initialize_model_parallel(1, 1)
                from megatron.core.dist_checkpointing import load_plain_tensors

                check_equal(load_plain_tensors(ckpt_dir_A), load_plain_tensors(ckpt_dir_B))
