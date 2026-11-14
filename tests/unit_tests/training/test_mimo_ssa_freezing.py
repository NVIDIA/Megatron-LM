# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Exercise MIMO SSA phase-1 freezing through real topology, DDP, and optimizer steps.

The tiny models supply diagnostic gradients; these tests cover parameter and
router-state freezing, not sparse attention kernels or the indexer KL objective.
The GTP case exercises optimizer groups with replicated diagnostic weights,
not native GTP projection-weight sharding.
Run with torch.distributed.run using four or eight CUDA ranks.
"""

import argparse
import traceback
from contextlib import contextmanager

import pytest
import torch
import torch.distributed as dist
from torch import nn

from examples.mimo.training.runtime import wrap_active_modules_with_ddp
from examples.mimo.training.topology import ModuleGridSpec, create_topology
from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.distributed.finalize_model_grads import (
    _update_router_expert_bias,
    reset_model_temporary_tensors,
)
from megatron.core.models.mimo.config import MimoModelConfig
from megatron.core.models.mimo.config.role import MIMO_LANGUAGE_MODULE_KEY
from megatron.core.models.mimo.model.base import MimoModel
from megatron.core.models.mimo.optimizer import get_mimo_optimizer
from megatron.core.optimizer import OptimizerConfig
from megatron.core.transformer import MegatronModule, TransformerConfig
from megatron.core.transformer.module import Float16Module
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.spec_utils import ModuleSpec
from tests.unit_tests.test_utilities import Utils

ENCODER = "images"


def _config(stage1, pp_size=1, gtp_size=1):
    return TransformerConfig(
        num_layers=pp_size,
        pipeline_model_parallel_size=pp_size,
        pipeline_dtype=torch.bfloat16,
        gtp_weight_remat_size=gtp_size,
        hidden_size=16,
        num_attention_heads=16,
        kv_channels=256,
        num_query_groups=1,
        bf16=True,
        params_dtype=torch.bfloat16,
        add_bias_linear=False,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        experimental_attention_variant="ssa",
        dsa_indexer_topk=512,
        dsa_indexer_loss_coeff=0.1,
        ssa_train_indexer_only=stage1,
        num_moe_experts=2,
        moe_router_topk=1,
        moe_router_dtype="fp32",
        moe_router_score_function="sigmoid",
        moe_router_load_balancing_type="none",
        moe_router_enable_expert_bias=True,
        moe_router_bias_update_rate=0.001,
        moe_aux_loss_coeff=0.0,
        moe_router_fusion=False,
        moe_router_force_load_balancing=False,
    )


class _TinyLanguageModel(MegatronModule):
    def __init__(self, config, pg_collection, with_indexer=True):
        super().__init__(config)
        self.pg_collection = pg_collection
        self.backbone = nn.Linear(16, 16, bias=False, dtype=torch.bfloat16)
        self.router = TopKRouter(config, pg_collection=pg_collection)
        self.router.set_layer_number(1)
        if with_indexer:
            self.indexer = nn.Linear(16, 2, bias=False, dtype=torch.bfloat16)
        with torch.no_grad():
            self.backbone.weight.copy_(torch.eye(16))
            self.router.weight.zero_()
            self.router.weight[0, :8].fill_(torch.logit(torch.tensor(0.6)).item())
            if with_indexer:
                self.indexer.weight.fill_(0.01)

    def forward(self, tokens):
        hidden = self.backbone(tokens)
        probabilities, _ = self.router(hidden)
        loss = (hidden.float() - 0.25).square().mean()
        loss = loss + (probabilities.float() - 0.25).square().mean()
        if hasattr(self, "indexer"):
            loss = loss + (self.indexer(hidden.detach()).float() - 1.0).square().mean()
        for projection in getattr(self, "mimo_input_projections", {}).values():
            loss = loss + (projection(hidden).float() - 0.25).square().mean()
        return loss


class _TinyModality(MegatronModule):
    def __init__(self, config, pg_collection):
        super().__init__(config)
        self.pg_collection = pg_collection
        self.encoders = nn.ModuleDict(
            {"toy": _TinyLanguageModel(config, pg_collection, with_indexer=False)}
        )
        self.input_projections = nn.ModuleList([nn.Linear(16, 16, bias=False)])
        self.output_projections = nn.ModuleList([nn.Linear(16, 16, bias=False)])
        with torch.no_grad():
            for projection in (*self.input_projections, *self.output_projections):
                projection.weight.copy_(torch.eye(16))

    @classmethod
    def from_spec(cls, spec, **kwargs):
        return cls(**spec.params)

    def forward(self, tokens):
        loss = self.encoders["toy"](tokens)
        for projection in (*self.input_projections, *self.output_projections):
            loss = loss + (projection(tokens).float() - 0.25).square().mean()
        return loss


@pytest.fixture
def distributed():
    if not torch.cuda.is_available() or Utils.world_size < 4 or Utils.world_size % 4:
        pytest.skip("requires at least four CUDA ranks and world size divisible by four")
    Utils.initialize_distributed()
    assert not parallel_state.model_parallel_is_initialized()
    yield
    Utils.destroy_model_parallel()


@contextmanager
def _model_and_topology(stage1=True, disjoint=True, pp_size=1, indexers="all", gtp_size=1):
    world = dist.get_world_size()
    language_ranks = world // 2 if disjoint else world
    specs = []
    if disjoint:
        specs.append(ModuleGridSpec(name=ENCODER, num_ranks=world // 2))
    specs.append(
        ModuleGridSpec(
            name=MIMO_LANGUAGE_MODULE_KEY,
            num_ranks=language_ranks,
            rank_offset=world // 2 if disjoint else 0,
            pp=pp_size,
            gtp_remat=gtp_size,
        )
    )
    topology = create_topology(specs)
    model = None
    try:
        config = _config(stage1, pp_size, gtp_size)
        lm_groups = topology.module_pgs[MIMO_LANGUAGE_MODULE_KEY]
        owns_language = topology.grids[MIMO_LANGUAGE_MODULE_KEY].is_current_rank_in_grid()
        with_indexer = indexers != "none"
        if indexers == "last_stage":
            with_indexer = owns_language and lm_groups.pp.rank() == pp_size - 1
        modality_specs = {}
        projection_specs = {}
        if disjoint:
            modality_specs[ENCODER] = ModuleSpec(
                module=_TinyModality,
                params={"config": _config(stage1), "pg_collection": topology.module_pgs[ENCODER]},
            )
            projection_specs[ENCODER] = ModuleSpec(
                module=nn.Linear, params={"in_features": 16, "out_features": 16, "bias": False}
            )
        mimo_config = MimoModelConfig(
            language_model_spec=ModuleSpec(
                module=_TinyLanguageModel,
                params={"config": config, "pg_collection": lm_groups, "with_indexer": with_indexer},
            ),
            modality_submodules_spec=modality_specs,
            language_model_input_projections_spec=projection_specs,
            module_to_grid_map=topology.grids,
        )
        model = MimoModel(
            mimo_config,
            cp_group=lm_groups.cp if owns_language else None,
            tp_group=lm_groups.tp if owns_language else None,
            tp_cp_group=lm_groups.tp_cp if owns_language else None,
        )
        if model.language_model_input_projections is not None:
            with torch.no_grad():
                for projection in model.language_model_input_projections.values():
                    projection.weight.copy_(torch.eye(16))
        yield model, topology
    except Exception as error:
        print(f"rank {dist.get_rank()}: MIMO test failed before cleanup: {error!r}", flush=True)
        traceback.print_exc()
        raise
    finally:
        if model is not None:
            model.destroy()
        topology.destroy()


def _wrap(model, topology, freeze_lm=False):
    wrap_active_modules_with_ddp(
        argparse.Namespace(
            ssa_train_indexer_only=model.config.ssa_train_indexer_only,
            freeze_lm=freeze_lm,
            freeze_projection=False,
            freeze_vit=False,
        ),
        model,
        topology,
        DistributedDataParallelConfig(grad_reduce_in_fp32=True),
    )


def _optimizer(model, stage1):
    optimizer = get_mimo_optimizer(
        model,
        OptimizerConfig(
            optimizer="adam",
            lr=0.01,
            weight_decay=0.0,
            bf16=True,
            params_dtype=torch.bfloat16,
            clip_grad=1.0,
            log_num_zeros_in_grad=True,
        ),
    )
    for name, info in optimizer.module_infos.items():
        expected_active = info.grid.is_current_rank_in_grid() and (
            name == MIMO_LANGUAGE_MODULE_KEY or not stage1
        )
        assert info.is_active == expected_active
        assert (info.optimizer is not None) == expected_active
    if stage1 and model.language_model is None:
        # Entirely frozen encoder grids need no local optimizer, but still
        # participate in the world reductions inside MimoOptimizer.step().
        assert optimizer.is_stub_optimizer
        assert not optimizer.param_groups
    return optimizer


def _assert_step_updates_only_expected_parameters(wrapped, stage1, optimizer):
    assert isinstance(wrapped, DistributedDataParallel)
    assert isinstance(wrapped.module, Float16Module)
    module = wrapped.module.module
    parameters = dict(module.named_parameters())
    expected = {
        name
        for name in parameters
        if not stage1 or name.startswith("indexer.") or ".indexer." in name
    }
    assert {name for name, p in parameters.items() if p.requires_grad} == expected
    captured = {id(p) for p in wrapped.params_with_grad}
    assert {name for name, p in parameters.items() if id(p) in captured} == expected
    routers = [child for child in module.modules() if isinstance(child, TopKRouter)]
    assert routers
    assert all(router.frozen_expert_bias == stage1 for router in routers)
    before = {name: p.detach().clone() for name, p in parameters.items()}
    biases_before = [router.expert_bias.detach().float().clone() for router in routers]
    assert {id(p) for group in optimizer.param_groups for p in group["params"]} == {
        id(getattr(p, "main_param", p)) for name, p in parameters.items() if name in expected
    }

    optimizer.zero_grad()
    wrapped.zero_grad_buffer()
    tokens = torch.eye(16, device="cuda", dtype=torch.bfloat16)[:8].unsqueeze(1)
    tokens = (tokens * (1 + module.pg_collection.dp.rank() / 16)).requires_grad_(True)
    wrapped(tokens).backward()
    wrapped.finish_grad_sync()
    # This update occurs independently of optimizer.step(). Frozen encoder and
    # language routers must both retain their persistent routing state.
    _update_router_expert_bias(
        [wrapped], module.config, tp_dp_cp_group=module.pg_collection.tp_dp_cp
    )
    for router, old_bias in zip(routers, biases_before):
        assert torch.equal(router.expert_bias.float(), old_bias) == stage1
    reset_model_temporary_tensors(module.config, [wrapped])
    for name, parameter in parameters.items():
        assert parameter.grad is None
        if name in expected:
            assert parameter.main_grad.abs().max().item() > 0
        else:
            assert not hasattr(parameter, "main_grad")
    success, grad_norm, num_zeros = optimizer.step()
    assert success
    assert grad_norm > 0
    assert num_zeros >= 0
    stats = torch.tensor([grad_norm, num_zeros], device="cuda", dtype=torch.float64)
    assert torch.isfinite(stats).all()
    world_stats = [torch.empty_like(stats) for _ in range(dist.get_world_size())]
    dist.all_gather(world_stats, stats)
    assert all(torch.equal(stats, other) for other in world_stats)
    for name, parameter in parameters.items():
        assert (not torch.equal(parameter, before[name])) == (name in expected), name
    # Inputs differ between DP ranks, so equality after the update checks that
    # this exercised real DDP communication rather than only local gradients.
    flattened = torch.cat([p.detach().flatten() for p in parameters.values()])
    replicas = [torch.empty_like(flattened) for _ in range(module.pg_collection.dp.size())]
    dist.all_gather(replicas, flattened, group=module.pg_collection.dp)
    assert all(torch.equal(flattened, replica) for replica in replicas)


@pytest.mark.parametrize("stage1", [True, False], ids=["stage1", "stage2"])
@pytest.mark.parametrize("disjoint", [False, True], ids=["language_only", "disjoint_encoder"])
def test_mimo_freezes_before_ddp_and_optimizer(distributed, stage1, disjoint):
    with _model_and_topology(stage1=stage1, disjoint=disjoint) as (model, topology):
        _wrap(model, topology)
        optimizer = _optimizer(model, stage1)
        active = (
            model.language_model
            if model.language_model is not None
            else model.modality_submodules[ENCODER]
        )
        _assert_step_updates_only_expected_parameters(active, stage1, optimizer)
        assert not parallel_state.model_parallel_is_initialized()


@pytest.mark.parametrize("stage1", [True, False], ids=["stage1", "stage2"])
def test_mimo_freezing_with_gtp_optimizer_groups(distributed, stage1):
    with _model_and_topology(stage1=stage1, disjoint=False, gtp_size=2) as (model, topology):
        _wrap(model, topology)
        optimizer = _optimizer(model, stage1)
        groups = topology.module_pgs[MIMO_LANGUAGE_MODULE_KEY]
        assert groups.pp.size() == 1
        assert groups.gtp_remat.size() == 2
        for inner in optimizer.chained_optimizers:
            assert inner.gtp_group is groups.gtp_remat
            assert inner.get_grad_stats_parallel_group() is groups.mp
        _assert_step_updates_only_expected_parameters(model.language_model, stage1, optimizer)
        assert not parallel_state.model_parallel_is_initialized()


def test_stage1_allows_pipeline_stage_without_indexer(distributed):
    with _model_and_topology(pp_size=2, indexers="last_stage") as (model, topology):
        _wrap(model, topology)
        optimizer = _optimizer(model, stage1=True)
        active = (
            model.language_model
            if model.language_model is not None
            else model.modality_submodules[ENCODER]
        )
        if model.language_model is not None:
            language_optimizer = optimizer.module_infos[MIMO_LANGUAGE_MODULE_KEY].optimizer
            assert language_optimizer is not None
            assert bool(language_optimizer.get_parameters()) == (
                topology.module_pgs[MIMO_LANGUAGE_MODULE_KEY].pp.rank() == 1
            )
        _assert_step_updates_only_expected_parameters(active, stage1=True, optimizer=optimizer)
        local_trainable = torch.tensor(len(active.params_with_grad), device="cuda")
        dist.all_reduce(local_trainable)
        assert local_trainable.item() == dist.get_world_size() // 4


def test_stage1_rejects_model_without_any_indexer_on_all_roles(distributed):
    with _model_and_topology(pp_size=2, indexers="none") as (model, topology):
        with pytest.raises(RuntimeError, match="found no simplified sparse attention indexer"):
            _wrap(model, topology)


def test_stage1_rejects_freeze_lm_on_all_roles(distributed):
    with _model_and_topology() as (model, topology):
        with pytest.raises(ValueError, match="freeze"):
            _wrap(model, topology, freeze_lm=True)
