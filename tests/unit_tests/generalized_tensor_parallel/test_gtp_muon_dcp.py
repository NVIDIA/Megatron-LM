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


def _initialize_native_fp8_moe_model(
    pre_process=True,
    post_process=True,
    seed=0,
    use_glu=True,
    use_sp=False,
    use_te=True,
    use_grouped_mlp=False,
    **config_kwargs,
):
    """``initialize_moe_model`` variant for native-FP8 (fp8_model_init / mxfp8) weights.

    ``params_dtype`` is set up front and the ``.bfloat16()`` / ``.random_()`` post-passes skip FP8
    params — a dtype cast or in-place ``random_`` would replace/destroy the native FP8 storage.
    """
    from megatron.core.fp8_utils import is_float8tensor
    from megatron.core.models.gpt import GPTModel
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
    from megatron.core.tensor_parallel import model_parallel_cuda_manual_seed
    from megatron.core.transformer import TransformerConfig

    # Passed through training.get_model but not part of TransformerConfig.
    config_kwargs.pop("pg_collection", None)
    config_kwargs.pop("config", None)
    torch.manual_seed(seed)
    model_parallel_cuda_manual_seed(seed)
    expert_num = 8

    # Dims sized so every GTP4 / EGTP2 FP8 shard dim stays a multiple of the MXFP8 block (32).
    default_config_kwargs = dict(
        num_layers=2,
        hidden_size=128,
        num_attention_heads=8,
        kv_channels=16,
        ffn_hidden_size=256,
        use_cpu_initialization=False,
        params_dtype=torch.bfloat16,
        num_moe_experts=expert_num,
        sequence_parallel=use_sp,
        moe_grouped_gemm=use_grouped_mlp,
        add_bias_linear=False,
        fp8='e4m3',
        fp8_recipe='mxfp8',
        fp8_param=True,
    )
    default_config_kwargs.update(**config_kwargs)
    transformer_config = TransformerConfig(**default_config_kwargs, gated_linear_unit=use_glu)
    spec = get_gpt_layer_with_transformer_engine_spec(
        num_experts=expert_num, moe_grouped_gemm=use_grouped_mlp
    )
    model = GPTModel(
        config=transformer_config,
        transformer_layer_spec=spec,
        vocab_size=128,
        max_sequence_length=4,
        pre_process=pre_process,
        post_process=post_process,
    )
    with torch.no_grad():
        for p in model.parameters():
            if not is_float8tensor(p):
                p.random_()
    return model


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

        from megatron.core.tensor_parallel.gtp import HAVE_GTP

        if not HAVE_GTP:
            pytest.skip("GTP requires TE with hook registry")
        if int(os.environ.get('WORLD_SIZE', '1')) != 4:
            pytest.skip("Requires world_size 4 (gtp2 x dp2)")

        os.environ['MEGATRON_GTP_FORCE_ENABLE'] = '1'
        from megatron.core import parallel_state as ps
        from megatron.core.tensor_parallel import model_parallel_cuda_manual_seed
        from megatron.core.tensor_parallel.gtp import GTP_CONFIG, GTPShardedParam, update_gtp_config
        from tests.unit_tests.dist_checkpointing.utils import initialize_moe_model

        Utils.initialize_model_parallel(1, 1)  # bootstrap torch.distributed + model parallel
        ps.destroy_model_parallel()
        ps.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1, gtp_remat_size=2
        )
        model_parallel_cuda_manual_seed(2)
        # Disable GTP_remat alignment padding so the tiny test dims slice cleanly by gtp_remat_size.
        _orig_pad = GTP_CONFIG.pad_for_alignment
        update_gtp_config(pad_for_alignment=0)
        # GTP_remat dims (divisible by gtp_remat_size=2); GPU init (CPU affine not GTP_remat-aware
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
                ), "GTP not active: no GTPShardedParam in the GTP_remat_size=2 MoE model"

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

    def test_gtp_muon_moe_native_fp8_save_load(self, tmp_path_dist_ckpt):
        """GTP + Muon (LayerWise) + native-FP8 (fp8_model_init / mxfp8) checkpoint roundtrip.

        Regression guard for the native-FP8 optimizer-state save. Native-FP8 GTP weights are
        dequantized into a NEW bf16 tensor when the model builds its ShardedTensor
        (make_tp_sharded_tensor_for_checkpoint), which breaks the ``id(entry.data) == id(param)``
        match every native-FP8 GTP param relies on -- so ALL of them fall into
        ``_backfill_gtp_sharded_param_map``. The fix reuses each model entry (via the
        ``_gtp_dequant_src`` backlink), preserving its offsets/replica_id; this exercises that
        reuse path end-to-end and asserts a bit-exact save/load/re-save roundtrip.

        Uses the gtp_remat-only MoE grid (no expert-parallel), matching the sibling bf16 test.
        The specific [[2],[2]] cross-expert collision from the production crash needs the GROUPED /
        EGTP expert grid (shared key + per-expert offset, where the old EP-unaware rebuild dropped
        that offset). That grid is intentionally avoided here: LayerWiseDistributedOptimizer's
        whole-param LPT layout over EGTP-sharded expert weights leaves a coverage hole that fails
        the same save even in pure BF16 (independent of native-FP8 and of this fix).
        """
        import os
        from functools import partial
        from unittest import mock

        import pytest

        from megatron.core.tensor_parallel.gtp import HAVE_GTP
        from tests.unit_tests.dist_checkpointing import utils as _dc_utils
        from tests.unit_tests.generalized_tensor_parallel.gtp_test_utils import _requires_mxfp8

        if not HAVE_GTP:
            pytest.skip("GTP requires TE with hook registry")
        if int(os.environ.get('WORLD_SIZE', '1')) != 4:
            pytest.skip("Requires world_size 4 (gtp2 x dp2)")
        _requires_mxfp8()

        os.environ['MEGATRON_GTP_FORCE_ENABLE'] = '1'
        from megatron.core import parallel_state as ps
        from megatron.core.fp8_utils import is_float8tensor
        from megatron.core.tensor_parallel import model_parallel_cuda_manual_seed
        from megatron.core.tensor_parallel.gtp import (
            GTP_CONFIG,
            is_gtp_param,
            tag_gtp_params_with_names,
            update_gtp_config,
        )

        Utils.initialize_model_parallel(1, 1)  # bootstrap torch.distributed + model parallel
        ps.destroy_model_parallel()
        ps.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1, gtp_remat_size=2
        )
        model_parallel_cuda_manual_seed(2)
        # MXFP8 needs shard dims % 32; padding off so dims slice cleanly by the remat size.
        _orig_pad = GTP_CONFIG.pad_for_alignment
        update_gtp_config(pad_for_alignment=0)

        # MXFP8 params can't be aliased into the DDP param buffer (replace_raw_data unsupported);
        # the production native-FP8 path sets reuse_grad_buf_for_mxfp8_param_ag to skip that
        # aliasing. The shared harness builds its own mock args, so wrap init_basic_mock_args to
        # flip the flags before the DDP config is built from them.
        _orig_init_args = _dc_utils.init_basic_mock_args

        def _init_args_fp8(args, tp, pp, bf16=True):
            _orig_init_args(args, tp, pp, bf16=bf16)
            args.fp8_param_gather = True
            args.reuse_grad_buf_for_mxfp8_param_ag = True
            return args

        init_fn = partial(_initialize_native_fp8_moe_model, use_te=True, use_grouped_mlp=False)
        meta = {'distrib_optim_sharding_type': 'dp_reshardable'}
        with (
            mock.patch.object(_dc_utils, 'init_basic_mock_args', _init_args_fp8),
            TempNamedDir(tmp_path_dist_ckpt / 'gtp_muon_fp8_A', sync=True) as ckpt_dir_A,
            TempNamedDir(tmp_path_dist_ckpt / 'gtp_muon_fp8_B', sync=True) as ckpt_dir_B,
        ):
            model_A, optimizer_A = setup_model_and_optimizer(
                seed=2,
                tp=1,
                pp=1,
                bf16=True,
                dist_opt=True,
                use_param_layout=True,
                initialize_fn=init_fn,
                optimizer='dist_muon',
            )
            tag_gtp_params_with_names(model_A[0])
            assert any(
                is_gtp_param(p) and is_float8tensor(p) for p in model_A[0].parameters()
            ), "no native-FP8 GTP param present; test is not exercising the FP8 path"

            model_sd_A = model_A[0].sharded_state_dict()
            # Every native-FP8 GTP param is unmatched (dequantized copy) and reuses its model
            # entry via _backfill_gtp_sharded_param_map; save validates the composed sharding.
            optim_sd_A = optimizer_A.sharded_state_dict(model_sd_A, metadata=meta)
            save(optim_sd_A, ckpt_dir_A)

            model_B, optimizer_B = setup_model_and_optimizer(
                seed=3,
                tp=1,
                pp=1,
                bf16=True,
                dist_opt=True,
                use_param_layout=True,
                initialize_fn=init_fn,
                optimizer='dist_muon',
            )
            tag_gtp_params_with_names(model_B[0])
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
