# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import importlib
import inspect
import os
from types import SimpleNamespace

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.distributed.finalize_model_grads import (
    _allreduce_non_tensor_model_parallel_grads,
    _allreduce_word_embedding_grads,
    _update_router_expert_bias,
    finalize_model_grads,
)
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


_MISSING = object()
_FINALIZE_MODEL_GRADS_MODULE = importlib.import_module(
    "megatron.core.distributed.finalize_model_grads"
)


class _FinalizeModelGradsModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.finish_grad_sync_calls = 0

    def finish_grad_sync(self, force_all_reduce=False):
        del force_all_reduce
        self.finish_grad_sync_calls += 1


def _finalize_model_grads_config():
    return SimpleNamespace(
        timers=None,
        flextron=False,
        moe_router_enable_expert_bias=True,
        moe_router_load_balancing_type="none",
    )


def _patch_finalize_model_grads_collectives(monkeypatch):
    def no_op(*args, **kwargs):
        del args, kwargs

    for name in (
        "_allreduce_conditional_embedding_grads",
        "_allreduce_non_tensor_model_parallel_grads",
        "_allreduce_word_embedding_grads",
        "_allreduce_position_embedding_grads",
        "reset_model_temporary_tensors",
    ):
        monkeypatch.setattr(_FINALIZE_MODEL_GRADS_MODULE, name, no_op)


def _pg_collection(tp_dp_cp=_MISSING):
    pg_collection = ProcessGroupCollection()
    pg_collection.tp = object()
    pg_collection.pp = object()
    pg_collection.embd = None
    pg_collection.pos_embd = None
    pg_collection.dp_cp = object()
    if tp_dp_cp is not _MISSING:
        pg_collection.tp_dp_cp = tp_dp_cp
    return pg_collection


def test_update_router_expert_bias_uses_explicit_group(monkeypatch):
    class RouterModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.local_tokens_per_expert = torch.tensor([1.0, 3.0])
            self.expert_bias = torch.zeros(2)

    group = object()
    router = RouterModule()
    config = type("Config", (), {"moe_router_bias_update_rate": 0.25})()
    calls = []

    def fake_get_updated_expert_bias(
        tokens_per_expert, expert_bias, expert_bias_update_rate, tp_dp_cp_group=None
    ):
        calls.append((expert_bias_update_rate, tp_dp_cp_group))
        return expert_bias + 1.0

    monkeypatch.setattr(
        _FINALIZE_MODEL_GRADS_MODULE, "get_updated_expert_bias", fake_get_updated_expert_bias
    )

    _update_router_expert_bias([torch.nn.Sequential(router)], config, tp_dp_cp_group=group)

    assert calls == [(0.25, group)]
    torch.testing.assert_close(router.expert_bias, torch.ones(2))


def test_finalize_model_grads_uses_pg_collection_tp_dp_cp(monkeypatch):
    _patch_finalize_model_grads_collectives(monkeypatch)

    group = object()
    pg_collection = _pg_collection(tp_dp_cp=group)

    calls = []

    def fake_update_router_expert_bias(model, config, tp_dp_cp_group=None):
        calls.append((model, config, tp_dp_cp_group))

    monkeypatch.setattr(
        _FINALIZE_MODEL_GRADS_MODULE, "_update_router_expert_bias", fake_update_router_expert_bias
    )

    config = _finalize_model_grads_config()
    model = _FinalizeModelGradsModel(config)
    finalize_model_grads([model], pg_collection=pg_collection)

    assert model.finish_grad_sync_calls == 1
    assert calls == [([model], config, group)]


def test_finalize_model_grads_requires_tp_dp_cp_for_explicit_groups(monkeypatch):
    _patch_finalize_model_grads_collectives(monkeypatch)

    config = _finalize_model_grads_config()
    model = _FinalizeModelGradsModel(config)

    for pg_collection in (_pg_collection(), _pg_collection(tp_dp_cp=None)):
        with pytest.raises(AssertionError, match="tp_dp_cp"):
            finalize_model_grads([model], pg_collection=pg_collection)
    assert model.finish_grad_sync_calls == 0


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
