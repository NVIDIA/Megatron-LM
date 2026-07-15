# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
import inspect
import os

import pytest
import torch
import torch.distributed as dist

from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.distributed.finalize_model_grads import (
    _allreduce_non_tensor_model_parallel_grads,
    _allreduce_word_embedding_grads,
    _update_router_qb_beta,
    finalize_model_grads,
    reset_model_temporary_tensors,
)
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_submodules,
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.spec_utils import get_submodules
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.initialize import _set_random_seed
from tests.unit_tests.test_utilities import Utils


class _RouterExpertBiasModel(torch.nn.Module):
    def __init__(self, config, local_tokens_per_expert):
        super().__init__()
        self.config = config
        self.ddp_config = DistributedDataParallelConfig()
        self.router = torch.nn.Module()
        self.router.register_buffer("local_tokens_per_expert", local_tokens_per_expert)
        self.router.register_buffer("expert_bias", torch.zeros_like(local_tokens_per_expert))
        self.finish_grad_sync_calls = 0

    def finish_grad_sync(self, force_all_reduce=False):
        del force_all_reduce
        self.finish_grad_sync_calls += 1


def _router_expert_bias_config():
    return TransformerConfig(
        num_layers=1,
        hidden_size=8,
        num_attention_heads=1,
        use_cpu_initialization=True,
        moe_router_enable_expert_bias=True,
        moe_router_score_function="sigmoid",
        moe_router_bias_update_rate=0.25,
        moe_router_load_balancing_type="none",
    )


_NO_TP_DP_CP = object()


def _router_bias_pg_collection(tp_dp_cp=_NO_TP_DP_CP):
    kwargs = {
        'tp': dist.group.WORLD,
        'pp': dist.group.WORLD,
        'embd': None,
        'pos_embd': None,
        'dp_cp': dist.group.WORLD,
    }
    if tp_dp_cp is not _NO_TP_DP_CP:
        kwargs['tp_dp_cp'] = tp_dp_cp
    return ProcessGroupCollection(**kwargs)


class TestFinalizeModelGradsMoEExpertBias:
    def setup_method(self, method):
        os.environ.pop('NVTE_FUSED_ATTN', None)
        os.environ.pop('NVTE_FLASH_ATTN', None)
        os.environ.pop('NVTE_UNFUSED_ATTN', None)
        Utils.destroy_model_parallel()
        Utils.initialize_distributed()
        parallel_state.destroy_model_parallel()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_finalize_model_grads_updates_router_expert_bias_with_custom_group(self):
        assert not parallel_state.model_parallel_is_initialized()

        config = _router_expert_bias_config()
        device = torch.device("cuda", torch.cuda.current_device())
        local_tokens = torch.tensor(
            [0.0, 2.0] if dist.get_rank() == 0 else [0.0, 0.0], device=device
        )
        model = _RouterExpertBiasModel(config, local_tokens)

        finalize_model_grads(
            [model], pg_collection=_router_bias_pg_collection(tp_dp_cp=dist.group.WORLD)
        )

        expected_bias = torch.tensor([0.25, -0.25], device=device)
        torch.testing.assert_close(model.router.expert_bias, expected_bias)
        torch.testing.assert_close(
            model.router.local_tokens_per_expert, torch.zeros_like(local_tokens)
        )
        assert model.finish_grad_sync_calls == 1

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_finalize_model_grads_requires_custom_group_before_grad_sync(self):
        assert not parallel_state.model_parallel_is_initialized()
        config = _router_expert_bias_config()
        device = torch.device("cuda", torch.cuda.current_device())
        pg_collections = [
            _router_bias_pg_collection(),
            _router_bias_pg_collection(tp_dp_cp=dist.group.WORLD),
        ]
        pg_collections[1].tp_dp_cp = None

        for pg_collection in pg_collections:
            model = _RouterExpertBiasModel(config, torch.tensor([1.0, 0.0], device=device))
            with pytest.raises(AssertionError, match="tp_dp_cp"):
                finalize_model_grads([model], pg_collection=pg_collection)
            assert model.finish_grad_sync_calls == 0


class TestUpdateRouterQBBeta:
    """Exercises the QB bias update in finalize_model_grads against a real MoE router."""

    def setup_method(self, method):
        os.environ.pop('NVTE_FUSED_ATTN', None)
        os.environ.pop('NVTE_FLASH_ATTN', None)
        os.environ.pop('NVTE_UNFUSED_ATTN', None)
        Utils.destroy_model_parallel()
        Utils.initialize_model_parallel(1, 1)
        _set_random_seed(seed_=123, data_parallel_random_init=False)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def _build_moe_layer(self, ema):
        num_experts = 8
        config = TransformerConfig(
            num_layers=2,
            hidden_size=12,
            num_attention_heads=4,
            num_moe_experts=num_experts,
            use_cpu_initialization=True,
            moe_router_load_balancing_type="quantile_balancing",
            moe_router_score_function="softmax",
            moe_router_topk=2,
            moe_aux_loss_coeff=0,
            moe_router_quantile_balancing_ema=ema,
            bf16=True,
            params_dtype=torch.bfloat16,
            add_bias_linear=False,
        )
        submodules = get_submodules(
            get_gpt_layer_local_submodules(num_experts=num_experts, moe_grouped_gemm=False).mlp
        )
        return config, MoELayer(config, submodules).cuda()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize("ema", [0.0, 0.9])
    def test_update_router_qb_beta(self, ema):
        config, moe_layer = self._build_moe_layer(ema)
        router = moe_layer.router
        router.train()
        # Non-zero prior bias so the EMA term is actually exercised.
        router.qb_beta.copy_(torch.randn_like(router.qb_beta))

        # The real router forward populates qb_beta_accum / qb_beta_count.
        hidden = torch.randn((32, 2, config.hidden_size)).cuda().bfloat16()
        router(hidden)
        router(hidden)
        assert router.qb_beta_count.item() == 2
        assert router.qb_beta_accum.abs().sum().item() > 0

        # Expected from the real accumulators: DP-avg(accum/count), EMA-blend, re-center.
        local_avg = router.qb_beta_accum / router.qb_beta_count.clamp(min=1).to(torch.float32)
        torch.distributed.all_reduce(
            local_avg, op=torch.distributed.ReduceOp.AVG, group=dist.group.WORLD
        )
        blended = ema * router.qb_beta + (1.0 - ema) * local_avg
        expected = blended - blended.mean(dim=-1, keepdim=True)

        _update_router_qb_beta([moe_layer], config, dp_cp_group=dist.group.WORLD)

        torch.testing.assert_close(router.qb_beta, expected)
        torch.testing.assert_close(
            router.qb_beta.mean(), torch.zeros((), device=router.qb_beta.device)
        )

        # reset_model_temporary_tensors clears the accumulators for the next global batch.
        reset_model_temporary_tensors(config, [moe_layer])
        torch.testing.assert_close(router.qb_beta_accum, torch.zeros_like(router.qb_beta_accum))
        assert router.qb_beta_count.item() == 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_update_router_qb_beta_skips_eval(self):
        config, moe_layer = self._build_moe_layer(ema=0.0)
        router = moe_layer.router
        # Non-zero prior + non-uniform accumulator, so a broken eval guard would visibly
        # change qb_beta (a uniform accumulator re-centers to zero and hides the bug).
        router.qb_beta.copy_(torch.ones_like(router.qb_beta))
        router.qb_beta_accum.copy_(
            torch.arange(router.qb_beta.numel(), dtype=torch.float32, device=router.qb_beta.device)
        )
        router.qb_beta_count.fill_(1)
        before = router.qb_beta.clone()
        router.eval()

        _update_router_qb_beta([moe_layer], config, dp_cp_group=dist.group.WORLD)

        # Eval-mode modules are skipped, so qb_beta is unchanged.
        torch.testing.assert_close(router.qb_beta, before)


class TestAllReduceLNGrads:

    def init_model(self, share_embeddings_and_output_weights: bool = False):
        self.transformer_config = TransformerConfig(
            num_layers=2,
            hidden_size=12,
            num_attention_heads=4,
            use_cpu_initialization=True,
            tensor_model_parallel_size=self.tp_size,
            pipeline_model_parallel_size=self.pp_size,
            qk_layernorm=True,
            pipeline_dtype=torch.float32,
        )

        self.model = GPTModel(
            config=self.transformer_config,
            transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(qk_layernorm=True),
            vocab_size=100,
            max_sequence_length=4,
            share_embeddings_and_output_weights=share_embeddings_and_output_weights,
        )

    def setup_method(self, method):
        os.environ.pop('NVTE_FUSED_ATTN', None)
        os.environ.pop('NVTE_FLASH_ATTN', None)
        os.environ.pop('NVTE_UNFUSED_ATTN', None)
        Utils.destroy_model_parallel()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("freeze_model,tp_size", [(True, 2), (False, 2)])
    def test_allreduce_layernorm_grads(self, freeze_model, tp_size):
        self.tp_size = tp_size
        self.pp_size = 1
        Utils.initialize_model_parallel(tensor_model_parallel_size=self.tp_size)
        model_parallel_cuda_manual_seed(123)

        self.init_model()
        self.model.cuda()
        self.model.ddp_config = DistributedDataParallelConfig()

        for param in self.model.parameters():
            if freeze_model:
                param.requires_grad = False
            else:
                param.grad = torch.ones_like(param)

        _allreduce_non_tensor_model_parallel_grads(
            [self.model], self.transformer_config, parallel_state.get_tensor_model_parallel_group()
        )

    @pytest.mark.parametrize(
        ("freeze_model", "pp_size", "share_embeddings"),
        [(True, 2, True), (False, 2, True), (True, 2, False), (False, 2, False)],
    )
    def test_allreduce_word_embedding_grads(self, freeze_model, pp_size, share_embeddings):
        self.tp_size = 1
        self.pp_size = pp_size
        Utils.initialize_model_parallel(pipeline_model_parallel_size=self.pp_size)
        model_parallel_cuda_manual_seed(123)

        self.init_model(share_embeddings)
        self.model.cuda()
        self.model.ddp_config = DistributedDataParallelConfig()

        for param in self.model.parameters():
            if freeze_model:
                param.requires_grad = False
            else:
                param.grad = torch.ones_like(param)
        pp_group = parallel_state.get_pipeline_model_parallel_group()
        embd_group = parallel_state.get_embedding_group()

        _allreduce_word_embedding_grads([self.model], self.transformer_config, embd_group, pp_group)
