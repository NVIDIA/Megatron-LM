# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for GTP + Muon (LayerWise) distributed checkpointing.

Covers the optimizer-state checkpoint roundtrip for the
:class:`LayerWiseDistributedOptimizer` (Muon) under GTP, where GTP-replicated
matrix params (e.g. the MoE router) are kept whole and must be disambiguated
by ``replica_id`` so DCP does not see multiple writers for the same shard.
"""

from functools import partial

import pytest
import torch

from megatron.core.tensor_parallel.gtp_api import HAVE_GTP

if not HAVE_GTP:
    pytest.skip("GTP requires TE with hook registry", allow_module_level=True)

from megatron.core.optimizer.emerging_optimizers import HAVE_EMERGING_OPTIMIZERS

if not HAVE_EMERGING_OPTIMIZERS:
    pytest.skip("emerging_optimizers not available", allow_module_level=True)

from megatron.core.dist_checkpointing import load, load_plain_tensors, save
from megatron.core.fp8_utils import is_float8tensor
from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.tensor_parallel import model_parallel_cuda_manual_seed
from megatron.core.tensor_parallel.generalized_tensor_parallelism import (
    is_gtp_param,
    tag_gtp_params_with_names,
)
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.dist_checkpointing import TempNamedDir, setup_model_and_optimizer
from tests.unit_tests.dist_checkpointing.utils import initialize_moe_model
from tests.unit_tests.generalized_tensor_parallel.gtp_test_utils import _requires_mxfp8
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


def _initialize_native_fp8_moe_model(pre_process=True, post_process=True, seed=0, **config_kwargs):
    """MXFP8 ``fp8_param`` variant of ``initialize_moe_model``.

    ``initialize_moe_model``'s ``.bfloat16()`` / ``.random_()`` post-passes would replace the
    native FP8 storage, and ``initialize_quantized_gpt_model`` builds a dense model; the router this
    test needs exists only in a MoE layer. Dims keep every GTP2 shard a multiple of the MXFP8 block.
    """
    # Passed through training.get_model but not part of TransformerConfig.
    config_kwargs.pop("pg_collection", None)
    config_kwargs.pop("config", None)
    torch.manual_seed(seed)
    model_parallel_cuda_manual_seed(seed)
    config = TransformerConfig(
        num_layers=2,
        hidden_size=128,
        num_attention_heads=8,
        kv_channels=16,
        ffn_hidden_size=256,
        params_dtype=torch.bfloat16,
        num_moe_experts=8,
        add_bias_linear=False,
        gated_linear_unit=True,
        fp8='e4m3',
        fp8_recipe='mxfp8',
        fp8_param=True,
        **config_kwargs,
    )
    model = GPTModel(
        config=config,
        transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(num_experts=8),
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


# Dims are multiples of pad_for_alignment(16) * gtp_remat_size(2), so GTP padding is a no-op. GPU
# init: CPU affine init is not GTP-aware for the strided QKV weight.
_BF16_MOE_INIT = partial(
    initialize_moe_model,
    use_te=True,
    hidden_size=64,
    num_attention_heads=8,
    kv_channels=8,
    ffn_hidden_size=128,
    use_cpu_initialization=False,
)


@pytest.mark.usefixtures("run_config")
class TestGTPMuonDCP:
    """GTP + Muon (LayerWise) distributed checkpointing tests."""

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize(
        "native_fp8",
        [
            False,
            # Quarantined: on Blackwell the Muon state of the (non-GTP) native-FP8 expert
            # weights gets two writers per shard at save time.
            pytest.param(True, marks=[pytest.mark.launch_on_gb200, pytest.mark.flaky_in_dev]),
        ],
    )
    def test_gtp_muon_moe_save_load(self, tmp_path_dist_ckpt, native_fp8):
        """GTP + Muon (LayerWise) optimizer-state save/load/re-save roundtrip is bit-exact.

        GTP-replicated, Muon-managed matrix params (the MoE router, held whole on every GTP peer)
        have the same key+offset on all GTP peers, so replica_id must distinguish them or DCP
        validate_sharding_integrity reports 2 writers. Adam dodges this by sharding the state.

        Native FP8: the model's ShardedTensor holds a dequantized copy of each MXFP8 GTP weight, so
        no such param matches by identity and every one goes through
        ``_backfill_gtp_sharded_param_map``, which must reuse the model entry's offsets/replica_id.
        """
        if torch.distributed.get_world_size() != 4:
            pytest.skip("Requires world_size 4 (gtp2 x dp2)")
        if native_fp8:
            _requires_mxfp8()
        Utils.initialize_model_parallel(1, 1, gtp_remat_size=2)
        model_parallel_cuda_manual_seed(2)
        # MXFP8 params can't be aliased into the DDP param buffer; production native FP8 sets
        # reuse_grad_buf_for_mxfp8_param_ag to gather through the grad buffer instead.
        setup = partial(
            setup_model_and_optimizer,
            tp=1,
            pp=1,
            bf16=True,
            dist_opt=True,
            use_param_layout=True,
            initialize_fn=_initialize_native_fp8_moe_model if native_fp8 else _BF16_MOE_INIT,
            optimizer='dist_muon',
            fp8_param_gather=native_fp8,
            reuse_grad_buf_for_mxfp8_param_ag=native_fp8,
        )
        meta = {'distrib_optim_sharding_type': 'dp_reshardable'}
        with (
            TempNamedDir(tmp_path_dist_ckpt / 'gtp_muon_moe_A', sync=True) as ckpt_dir_A,
            TempNamedDir(tmp_path_dist_ckpt / 'gtp_muon_moe_B', sync=True) as ckpt_dir_B,
        ):
            model_A, optimizer_A = setup(seed=2)
            tag_gtp_params_with_names(model_A[0])
            assert any(
                is_gtp_param(p) and is_float8tensor(p) == native_fp8
                for p in model_A[0].parameters()
            ), f"no GTP param with native_fp8={native_fp8}"
            model_sd_A = model_A[0].sharded_state_dict()
            save(optimizer_A.sharded_state_dict(model_sd_A, metadata=meta), ckpt_dir_A)

            model_B, optimizer_B = setup(seed=3)
            tag_gtp_params_with_names(model_B[0])
            model_sd_B = model_B[0].sharded_state_dict()
            load_sharded_sd = optimizer_B.sharded_state_dict(
                model_sd_B, is_loading=True, metadata=meta
            )
            optimizer_B.load_state_dict(load(load_sharded_sd, ckpt_dir_A))
            save(optimizer_B.sharded_state_dict(model_sd_B, metadata=meta), ckpt_dir_B)

            Utils.initialize_model_parallel(1, 1)
            check_equal(load_plain_tensors(ckpt_dir_A), load_plain_tensors(ckpt_dir_B))
