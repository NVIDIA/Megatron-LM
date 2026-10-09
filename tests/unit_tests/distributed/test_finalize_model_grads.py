# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
import inspect
import os
from contextlib import ExitStack, contextmanager
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
import torch.distributed as dist

from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.distributed.finalize_model_grads import (
    _allreduce_non_tensor_model_parallel_grads,
    _allreduce_word_embedding_grads,
    _update_router_expert_bias,
    _update_router_qb_beta,
    finalize_model_grads,
    reset_model_temporary_tensors,
)
from megatron.core.hyper_comm_grid import HyperCommGrid
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


class _HashRouterWithoutExpertBias(torch.nn.Module):
    """Match hash-router layers, which intentionally do not own expert-bias state."""

    def __init__(self):
        super().__init__()
        self.expert_bias = None
        self.local_tokens_per_expert = None


def test_hash_router_without_expert_bias_is_ignored():
    router = _HashRouterWithoutExpertBias()
    config = SimpleNamespace(
        moe_router_enable_expert_bias=True,
        moe_router_load_balancing_type="none",
        moe_router_bias_update_rate=0.25,
    )

    reset_model_temporary_tensors(config, [router])
    _update_router_expert_bias([router], config)

    assert router.expert_bias is None
    assert router.local_tokens_per_expert is None


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
            with pytest.raises(ValueError, match="tp_dp_cp"):
                finalize_model_grads([model], pg_collection=pg_collection)
            assert model.finish_grad_sync_calls == 0


_PIPELINE_SIZE = 4
_UNSET = object()


@contextmanager
def _forbid_global_groups():
    """Fail on any read of the global parallel grid or any collective over the default group."""

    def _forbidden(name):
        def _raise(*args, **kwargs):
            raise AssertionError(f"read parallel_state.{name}")

        return _raise

    def _without_default_group(collective):
        signature = inspect.signature(collective)

        def _checked(*args, **kwargs):
            group = signature.bind(*args, **kwargs).arguments.get('group')
            assert group is not None, f"{collective.__name__} over the default (WORLD) group"
            return collective(*args, **kwargs)

        return _checked

    accessors = [
        name
        for name in dir(parallel_state)
        if name.startswith('get_')
        and name.endswith(('_group', '_groups', '_gloo', '_rank', '_ranks', '_world_size'))
    ]
    with ExitStack() as stack:
        for name in accessors + ['is_pipeline_first_stage', 'is_pipeline_last_stage']:
            stack.enter_context(mock.patch.object(parallel_state, name, _forbidden(name)))
        for name in ('all_reduce', 'broadcast'):
            stack.enter_context(
                mock.patch.object(dist, name, _without_default_group(getattr(dist, name)))
            )
        yield


def _pipeline_pg_collection():
    """Groups of a TP=1 x DP x PP=4 grid, built without parallel_state.

    The word embeddings live on the first and last stage and the position embeddings on the first
    stage; ranks outside those groups hold None.
    """
    grid = HyperCommGrid(
        [1, dist.get_world_size() // _PIPELINE_SIZE, _PIPELINE_SIZE], ["tp", "dp", "pp"]
    )
    pg_collection = ProcessGroupCollection(
        tp=grid.create_pg("tp"),
        pp=grid.create_pg("pp"),
        dp_cp=grid.create_pg("dp"),
        embd=None,
        pos_embd=None,
    )
    # new_group is collective: every rank creates every group in the same order.
    for stage_ranks in grid.get_rank_enum("pp"):
        embd_ranks = [stage_ranks[0], stage_ranks[-1]]
        embd = dist.new_group(embd_ranks)
        pos_embd = dist.new_group(stage_ranks[:1])
        if dist.get_rank() in embd_ranks:
            pg_collection.embd = embd
        if dist.get_rank() == stage_ranks[0]:
            pg_collection.pos_embd = pos_embd
    return pg_collection


def _stage_config(sync_replicated_params=False, **kwargs):
    config = TransformerConfig(
        num_layers=1, hidden_size=8, num_attention_heads=1, use_cpu_initialization=True, **kwargs
    )
    # Conditional embedders and Flextron routers are replicated on every pipeline stage, and
    # finalize_model_grads all-reduces their gradients across the pipeline.
    config.has_cond_embedder = sync_replicated_params
    config.flextron = sync_replicated_params
    return config


def _num_tokens(chunk, dp_cp_group):
    """Only the last stage counts tokens; finalize_model_grads broadcasts the count to the other
    stages before reducing it across data-parallel replicas."""
    count = 5 + dp_cp_group.rank() if chunk.post_process else 1000
    return torch.tensor(count, dtype=torch.int, device="cuda")


def _assert_finalized(chunk, num_tokens, pp_group, dp_cp_group):
    total_tokens = sum(5 + dp_rank for dp_rank in range(dp_cp_group.size()))
    assert num_tokens.item() == total_tokens
    stage_sum = sum(range(1, pp_group.size() + 1))
    expected = {'cond_embedder': stage_sum / total_tokens, 'router': 10 * stage_sum / total_tokens}
    if chunk.word_embedding is not None:
        expected['word_embedding'] = (1 + pp_group.size()) / total_tokens
    for name, value in expected.items():
        main_grad = getattr(chunk, name).main_grad
        torch.testing.assert_close(main_grad, torch.full_like(main_grad, value))
    assert chunk.finish_grad_sync_calls == 1


class _StageChunk(torch.nn.Module):
    """One pipeline stage with the attributes and hooks that finalize_model_grads uses."""

    def __init__(self, config, pp_group):
        super().__init__()
        self.config = config
        self.ddp_config = DistributedDataParallelConfig()
        self.pre_process = pp_group.rank() == 0
        self.post_process = pp_group.rank() == pp_group.size() - 1
        self.share_embeddings_and_output_weights = True
        stage = pp_group.rank() + 1
        device = torch.cuda.current_device()
        # The tied word embedding has a copy on the first and the last stage.
        self.word_embedding = (
            self._parameter(stage, device) if self.pre_process or self.post_process else None
        )
        self.cond_embedder = self._parameter(stage, device)
        self.cond_embedder.pipeline_parallel = True
        self.router = self._parameter(10 * stage, device)
        self.router.flextron_router_pp_sync = True
        self.finish_grad_sync_calls = 0

    @staticmethod
    def _parameter(grad_value, device):
        param = torch.nn.Parameter(torch.zeros(4, device=device))
        param.main_grad = torch.full_like(param, float(grad_value))
        return param

    def shared_embedding_or_output_weight(self):
        return self.word_embedding

    def finish_grad_sync(self, force_all_reduce=False):
        del force_all_reduce
        self.finish_grad_sync_calls += 1

    def scale_gradients(self, scaling_factor):
        for param in self.parameters():
            param.main_grad.mul_(scaling_factor)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(
    Utils.world_size < _PIPELINE_SIZE or Utils.world_size % _PIPELINE_SIZE != 0,
    reason=f"needs a multiple of {_PIPELINE_SIZE} ranks",
)
class TestFinalizeModelGradsProcessGroups:
    """An explicit collection is the only source of groups; without one, the global grid is."""

    def setup_method(self, method):
        Utils.destroy_model_parallel()
        Utils.initialize_distributed()
        parallel_state.destroy_model_parallel()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_reads_no_global_groups(self):
        assert not parallel_state.model_parallel_is_initialized()
        pg_collection = _pipeline_pg_collection()
        chunk = _StageChunk(_stage_config(sync_replicated_params=True), pg_collection.pp)
        num_tokens = _num_tokens(chunk, pg_collection.dp_cp)

        with _forbid_global_groups():
            finalize_model_grads([chunk], num_tokens=num_tokens, pg_collection=pg_collection)

        _assert_finalized(chunk, num_tokens, pg_collection.pp, pg_collection.dp_cp)

    def test_without_collection_uses_global_groups(self):
        Utils.initialize_model_parallel(pipeline_model_parallel_size=_PIPELINE_SIZE)
        pp_group = parallel_state.get_pipeline_model_parallel_group()
        dp_cp_group = parallel_state.get_data_parallel_group(with_context_parallel=True)
        chunk = _StageChunk(_stage_config(sync_replicated_params=True), pp_group)
        num_tokens = _num_tokens(chunk, dp_cp_group)

        finalize_model_grads([chunk], num_tokens=num_tokens)

        _assert_finalized(chunk, num_tokens, pp_group, dp_cp_group)

    def test_ignores_global_grid_with_other_layout(self):
        # On the global grid every rank is the first or last of two stages, so a lookup there
        # would pair the stages of the explicit four-stage pipeline differently.
        Utils.initialize_model_parallel(pipeline_model_parallel_size=2)
        pg_collection = _pipeline_pg_collection()
        pp_group = pg_collection.pp
        chunk = _StageChunk(_stage_config(sync_replicated_params=True), pp_group)
        is_middle_stage = not (chunk.pre_process or chunk.post_process)
        assert (pg_collection.embd is None) == is_middle_stage

        finalize_model_grads([chunk], pg_collection=pg_collection)

        stage_sum = sum(range(1, pp_group.size() + 1))
        torch.testing.assert_close(
            chunk.router.main_grad, torch.full_like(chunk.router.main_grad, 10.0 * stage_sum)
        )
        if not is_middle_stage:
            main_grad = chunk.word_embedding.main_grad
            torch.testing.assert_close(main_grad, torch.full_like(main_grad, 1.0 + pp_group.size()))

    @pytest.mark.parametrize(
        "field,value,config_kwargs",
        [
            ("tp", _UNSET, {}),
            ("tp", None, {}),
            ("pp", _UNSET, {}),
            ("pp", None, {}),
            ("embd", _UNSET, {}),
            ("dp_cp", _UNSET, {"moe_router_load_balancing_type": "quantile_balancing"}),
        ],
        ids=["tp-unset", "tp-none", "pp-unset", "pp-none", "embd-unset", "dp_cp-unset-qb"],
    )
    def test_rejects_missing_group(self, field, value, config_kwargs):
        pg_collection = _pipeline_pg_collection()
        chunk = _StageChunk(_stage_config(**config_kwargs), pg_collection.pp)
        if value is _UNSET:
            delattr(pg_collection, field)
        else:
            setattr(pg_collection, field, value)

        with _forbid_global_groups(), pytest.raises(ValueError, match=f"pg_collection.{field}"):
            finalize_model_grads([chunk], pg_collection=pg_collection)
        assert chunk.finish_grad_sync_calls == 0

    def test_num_tokens_requires_data_parallel_group(self):
        """Without a DP x CP group, num_tokens must not be all-reduced over every rank."""
        pg_collection = _pipeline_pg_collection()
        delattr(pg_collection, 'dp_cp')
        chunk = _StageChunk(_stage_config(), pg_collection.pp)
        num_tokens = torch.tensor(8, dtype=torch.int, device="cuda")

        with pytest.raises(ValueError, match="dp_cp"):
            finalize_model_grads([chunk], num_tokens=num_tokens, pg_collection=pg_collection)
        assert num_tokens.item() == 8
        assert chunk.finish_grad_sync_calls == 0


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
        qk_layernorm = True
        self.transformer_config = TransformerConfig(
            num_layers=2,
            hidden_size=12,
            num_attention_heads=4,
            use_cpu_initialization=True,
            tensor_model_parallel_size=self.tp_size,
            pipeline_model_parallel_size=self.pp_size,
            qk_layernorm=qk_layernorm,
            pipeline_dtype=torch.float32,
        )

        self.model = GPTModel(
            config=self.transformer_config,
            transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(
                qk_layernorm=qk_layernorm
            ),
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
