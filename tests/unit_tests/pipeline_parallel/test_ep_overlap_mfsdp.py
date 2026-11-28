# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""MFSDP v2 integration with the expert-parallel overlap schedule."""

import copy
from collections import Counter
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard

from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.distributed.fsdp.mcore_fsdp_adapter import FullyShardedDataParallel
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import (
    FsdpContext,
    GradientFinalizationMode,
    Placements,
    SchedulePolicy,
    fully_shard,
    fully_shard_context,
    fully_shard_optimizer,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.module import FsdpModule
from megatron.core.models.common.combined_1f1b_mfsdp_scheduler import (
    complete_fsdp_unit_backward,
    register_fsdp_ep_overlap_schedule_hooks,
    register_fsdp_module_hooks,
)
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_decoder_block_spec,
    get_gpt_mtp_block_spec,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.pipeline_parallel.combined_1f1b import (
    combined_forward_backward_step,
    setup_mfsdp_v2_ep_overlap,
)
from megatron.core.pipeline_parallel.utils import set_streams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.module import Float16Module
from megatron.core.transformer.multi_token_prediction import MTPLossAutoScaler, MTPLossLoggingHelper
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.distributed.mfsdp_v2.conftest import distributed_setup as distributed_setup
from tests.unit_tests.distributed.mfsdp_v2.mcore_utils import destroy_model_parallel
from tests.unit_tests.test_utilities import Utils


@pytest.mark.parametrize(
    "overlap,mtp_num_layers,error",
    [(True, None, "MoE EP overlap.*tied"), (False, 1, "PP > 1 with MTP and tied embeddings")],
    ids=["ep_overlap", "pipeline_mtp"],
)
def test_rejects_tied_weights_before_sharding(overlap, mtp_num_layers, error):
    """Reject tied weights on every pipeline stage before constructing FSDP state."""
    config = TransformerConfig(
        num_layers=4,
        hidden_size=16,
        num_attention_heads=4,
        bf16=True,
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
        num_moe_experts=4,
        expert_model_parallel_size=2,
        moe_token_dispatcher_type="alltoall",
        overlap_moe_expert_parallel_comm=overlap,
        pipeline_model_parallel_size=2,
        virtual_pipeline_model_parallel_size=2,
        mtp_num_layers=mtp_num_layers,
        gradient_accumulation_fusion=False,
    )
    model = nn.Linear(16, 16)
    model.share_embeddings_and_output_weights = True
    # No usable process group: validation must reject before building a mesh.
    with pytest.raises(ValueError, match=error):
        FullyShardedDataParallel(
            config=config,
            ddp_config=DistributedDataParallelConfig(
                use_megatron_fsdp=True,
                megatron_fsdp_version=2,
                data_parallel_sharding_strategy="optim_grads_params",
            ),
            module=Float16Module(config, model),
            pg_collection=SimpleNamespace(dp_cp=None),
        )


@pytest.mark.parametrize("delay_wgrad", [False, True], ids=["wgrad", "delayed_wgrad"])
@pytest.mark.parametrize("mtp_num_layers", [1, 2], ids=["mtp1", "mtp2"])
def test_ep_overlap_mtp(distributed_setup, monkeypatch, delay_wgrad, mtp_num_layers):
    """Complete each FSDP unit once per backward, including repeated MTP head uses."""
    setup = distributed_setup
    if setup.world_size < 2 or setup.world_size % 2:
        pytest.skip("EP=2 requires an even world size of at least two.")
    Utils.initialize_model_parallel(1, 1, expert_model_parallel_size=2)
    try:
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        torch.manual_seed(123)
        model_parallel_cuda_manual_seed(123)
        config = TransformerConfig(
            num_layers=2,
            hidden_size=64,
            num_attention_heads=4,
            ffn_hidden_size=128,
            num_moe_experts=4,
            expert_model_parallel_size=2,
            moe_layer_freq=[0, 1],
            moe_token_dispatcher_type="alltoall",
            moe_router_topk=2,
            moe_grouped_gemm=True,
            moe_ffn_hidden_size=128,
            mtp_num_layers=mtp_num_layers,
            delay_wgrad_compute=delay_wgrad,
            mtp_loss_scaling_factor=0.1,
            overlap_moe_expert_parallel_comm=True,
            add_bias_linear=False,
            bf16=True,
            params_dtype=torch.bfloat16,
            gradient_accumulation_fusion=False,
            attention_dropout=0.0,
            hidden_dropout=0.0,
            attention_backend=AttnBackend.unfused,
        )
        decoder_spec = get_gpt_decoder_block_spec(config, use_transformer_engine=True)
        model = GPTModel(
            config=config,
            transformer_layer_spec=decoder_spec,
            mtp_block_spec=get_gpt_mtp_block_spec(config, decoder_spec.layer_specs[-1], True),
            vocab_size=128,
            max_sequence_length=16,
            share_embeddings_and_output_weights=False,
            position_embedding_type="rope",
            pg_collection=pg_collection,
        ).cuda()
        wrapped = FullyShardedDataParallel(
            config=config,
            ddp_config=DistributedDataParallelConfig(
                use_megatron_fsdp=True,
                megatron_fsdp_version=2,
                data_parallel_sharding_strategy="optim_grads_params",
            ),
            module=Float16Module(config, model),
            pg_collection=pg_collection,
        )
        setup_mfsdp_v2_ep_overlap([wrapped])
        units = [module for module in wrapped.modules() if isinstance(module, FsdpModule)]
        expected_completions = dict.fromkeys(units, 1)
        completions = Counter()
        post_accumulate_grad = FsdpModule.post_accumulate_grad

        def record_completion(unit):
            post_accumulate_grad(unit)
            completions[unit] += 1

        monkeypatch.setattr(FsdpModule, "post_accumulate_grad", record_completion)
        monkeypatch.setattr(MTPLossLoggingHelper, "tracker", {})
        monkeypatch.setattr(
            MTPLossAutoScaler, "main_loss_backward_scale", torch.tensor(1.0, device=setup.device)
        )
        set_streams()

        def forward_step(_iterator, inner_model, return_schedule_plan):
            assert return_schedule_plan
            assert inner_model is model
            plan = inner_model.build_schedule_plan(
                input_ids=torch.randint(0, 128, (1, 16), device=setup.device),
                labels=torch.randint(0, 128, (1, 16), device=setup.device),
                position_ids=torch.arange(16, device=setup.device).unsqueeze(0),
                attention_mask=None,
            )
            return plan, lambda output: (output.mean(), {})

        def run_step(forward, previous_loss):
            loss, _, _ = combined_forward_backward_step(
                forward_step_func=forward_step,
                data_iterator=None,
                f_model=wrapped if forward else None,
                num_microbatches=3,
                input_tensor=None,
                forward_data_store=[],
                b_model=wrapped if previous_loss is not None else None,
                b_input_tensor=None,
                b_output_tensor=previous_loss,
                b_output_tensor_grad=None,
                config=config,
            )
            return loss

        previous_loss = None
        for microbatch in range(3):
            completions.clear()
            with wrapped.no_sync():
                previous_loss = run_step(True, previous_loss)
            if microbatch == 0:
                # No backward has run yet: every layer, including its independent
                # expert unit, must already have released its forward weights.
                for unit in units:
                    if not unit.is_root():
                        assert all(not group._unshard_events for group in unit.parameter_groups)
            assert completions == (expected_completions if microbatch else {})
        completions.clear()
        run_step(False, previous_loss)
        wrapped.finish_grad_sync()
        assert completions == expected_completions
    finally:
        destroy_model_parallel()


def test_train_validate_train_reshards_and_matches_dense(distributed_setup):
    """Train/eval preserve dense updates and release gathered weights."""
    setup = distributed_setup
    torch.manual_seed(0)
    reference = nn.Sequential(nn.Linear(16, 16), nn.Linear(16, 16)).to(setup.device)
    model = copy.deepcopy(reference)
    mesh = init_device_mesh(setup.device.type, (setup.world_size,))
    placements = Placements([0], [Shard(0)], [Shard(0)], [Shard(0)])
    units = (model[1], model)
    with fully_shard_context(
        FsdpContext(
            device=setup.device, gradient_finalization_mode=GradientFinalizationMode.REDUCE_OUTER
        )
    ):
        for unit in units:
            fully_shard(
                unit,
                mesh=mesh,
                placements=placements,
                schedule_policy=SchedulePolicy(register_hooks=False),
            )
    register_fsdp_module_hooks(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    fully_shard_optimizer(optimizer)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)

    torch.manual_seed(123 + setup.rank)
    for training in (True, False, True, False):
        model.train(training)
        reference.train(training)
        inputs = torch.randn(4, 16, device=setup.device)
        with torch.set_grad_enabled(training):
            if training:
                optimizer.zero_grad()
                reference_optimizer.zero_grad()
            expected = reference(inputs)
            actual = model(inputs)
            torch.testing.assert_close(actual, expected)
            if training:
                # Training releases weights at schedule boundaries; evaluation
                # bypasses that schedule and must release them through its hooks.
                for unit in units:
                    unit.reshard()
                actual.square().mean().backward()
                expected.square().mean().backward()
                for unit in units:
                    complete_fsdp_unit_backward(unit)
                model.context.finish_grad_sync()
                for parameter in reference.parameters():
                    dist.all_reduce(parameter.grad, op=dist.ReduceOp.AVG)
                optimizer.step()
                reference_optimizer.step()
        for unit in units:
            assert all(not group._unshard_events for group in unit.parameter_groups)


def test_schedule_finishes_shared_and_nested_units_once():
    """Schedule boundaries own completion across GraphTasks, not parameter hooks."""
    reductions = []

    class Unit(nn.Module, FsdpModule):
        def __init__(self, name):
            nn.Module.__init__(self)
            self.label = name
            self.weight = nn.Parameter(torch.ones(()))
            self.unsharded = False

        def forward(self, x):
            x = x * self.weight
            for child in self.children():
                x = child(x)
            return x

        def is_root(self):
            return False

        def unshard(self):
            self.unsharded = True

        def reshard(self):
            self.unsharded = False

        def post_accumulate_grad(self):
            reductions.append((self.label, self.weight.grad.item()))
            self.weight.grad = None

    class Plan:
        def __init__(self, layer):
            self.layer = layer

        def set_fsdp_schedule_callbacks(self, forward, backward):
            self.finish_forward = lambda: forward(self.layer)
            self.finish = lambda: backward(self.layer)

    root, shared, expert, other = [Unit(name) for name in ("root", "shared", "expert", "other")]
    shared.add_module("expert", expert)
    model = nn.Module()
    model.register_parameter("weight", root.weight)
    model.add_module("shared", shared)
    model.add_module("other", other)
    root.add_module("model", model)
    register_fsdp_module_hooks(root)
    plans = [Plan(shared), Plan(other), Plan(shared)]
    finish_chunk = register_fsdp_ep_overlap_schedule_hooks(nn.Sequential(root), plans)

    # Exercise the actual schedule callback: the outer layer's reshard must also
    # release nested experts, including each repeated forward of a shared layer.
    root.unshard()
    for plan in plans:
        with torch.no_grad():
            plan.layer(torch.ones(()))
        layer_units = [unit for unit in plan.layer.modules() if isinstance(unit, FsdpModule)]
        assert all(unit.unsharded for unit in layer_units)
        plan.finish_forward()
        assert all(not unit.unsharded for unit in layer_units)
    assert root.unsharded

    # The root weight receives contributions before and after layer backward.
    (root.weight * 2).backward()
    shared(torch.tensor(5.0)).backward()
    plans[-1].finish()
    assert reductions == []
    other(torch.tensor(4.0)).backward()
    plans[1].finish()
    assert reductions == [("other", 4.0)]
    shared(torch.tensor(3.0)).backward()
    plans[0].finish()
    (root.weight * 7).backward()
    finish_chunk()
    assert reductions == [("other", 4.0), ("shared", 8.0), ("expert", 8.0), ("root", 9.0)]
