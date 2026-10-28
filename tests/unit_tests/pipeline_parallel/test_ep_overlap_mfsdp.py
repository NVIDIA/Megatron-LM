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
    Placements,
    fully_shard,
    fully_shard_context,
    fully_shard_optimizer,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.module import FsdpModule
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.parameter_group import (
    FsdpParameterGroup,
)
from megatron.core.models.common.combined_1f1b_mfsdp_scheduler import (
    finish_fsdp_backward,
    register_combined_1f1b_hooks,
)
from megatron.core.models.common.model_chunk_schedule_plan import TransformerModelChunkSchedulePlan
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_decoder_block_spec,
    get_gpt_mtp_block_spec,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.pipeline_parallel.combined_1f1b import combined_forward_backward_step
from megatron.core.pipeline_parallel.utils import set_streams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.module import Float16Module
from megatron.core.transformer.multi_token_prediction import MTPLossAutoScaler
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.distributed.mfsdp_v2.conftest import distributed_setup as distributed_setup
from tests.unit_tests.distributed.mfsdp_v2.mcore_utils import destroy_model_parallel
from tests.unit_tests.test_utilities import Utils

pytestmark = pytest.mark.mfsdp


@pytest.mark.parametrize("delay_wgrad", [False, True])
def test_ep_overlap_mtp(distributed_setup, monkeypatch, delay_wgrad):
    """Reduce every trainable weight once per backward, including tied and MTP uses."""
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
            mtp_num_layers=1,
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
            share_embeddings_and_output_weights=True,
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
        units = [module for module in wrapped.modules() if isinstance(module, FsdpModule)]
        expected_reductions = {
            id(parameter): 1
            for unit in units
            for group in unit.parameter_groups
            if group.requires_grad
            for parameter in group.fsdp_parameters
        }
        reductions = Counter()
        reduce_gradients = FsdpParameterGroup.reduce_partial_gradients

        def record_reduction(group, *args, **kwargs):
            reduce_gradients(group, *args, **kwargs)
            reductions.update(id(parameter) for parameter in group.fsdp_parameters)

        monkeypatch.setattr(FsdpParameterGroup, "reduce_partial_gradients", record_reduction)
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
            reductions.clear()
            with wrapped.no_sync():
                previous_loss = run_step(True, previous_loss)
            assert reductions == (expected_reductions if microbatch else {})
        reductions.clear()
        run_step(False, previous_loss)
        wrapped.finish_grad_sync()
        assert reductions == expected_reductions
    finally:
        destroy_model_parallel()


@pytest.mark.parametrize("tied", [False, True])
def test_train_validate_train_reshards_and_matches_dense(distributed_setup, tied):
    """Train/eval preserve dense updates, including externally supplied tied weights."""
    setup = distributed_setup
    torch.manual_seed(0)

    class OutputHead(nn.Module):
        def forward(self, inputs, weight):
            return torch.nn.functional.linear(inputs, weight)

    class TiedModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.share_embeddings_and_output_weights = True
            self.embedding = nn.Module()
            self.embedding.word_embeddings = nn.Embedding(16, 16)
            self.layer = nn.Linear(16, 16)
            self.output_layer = OutputHead()

        def forward(self, inputs):
            return self.output_layer(
                self.layer(inputs), weight=self.embedding.word_embeddings.weight
            )

    reference = (TiedModel() if tied else nn.Sequential(nn.Linear(16, 16), nn.Linear(16, 16))).to(
        setup.device
    )
    model = copy.deepcopy(reference)
    mesh = init_device_mesh(setup.device.type, (setup.world_size,))
    placements = Placements([0], [Shard(0)], [Shard(0)], [Shard(0)])
    units = (model.layer if tied else model[1], model)
    with fully_shard_context(device=setup.device, caller_managed_grad_sync=True):
        for unit in units:
            fully_shard(unit, mesh=mesh, placements=placements, register_hooks=False)
    register_combined_1f1b_hooks(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    fully_shard_optimizer(optimizer)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)

    torch.manual_seed(123 + setup.rank)
    for training in (True, False, True, False, True, False):
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
                    finish_fsdp_backward(unit)
                model.context.finish_grad_sync()
                for parameter in reference.parameters():
                    dist.all_reduce(parameter.grad, op=dist.ReduceOp.AVG)
                optimizer.step()
                reference_optimizer.step()
        for unit in units:
            assert all(not group._unshard_events for group in unit.parameter_groups)


@pytest.mark.parametrize("repeated", [False, True])
@pytest.mark.parametrize("wrapped", [False, True])
def test_schedule_finishes_shared_and_nested_units_once(repeated, wrapped):
    """Schedule boundaries own completion across GraphTasks, not parameter hooks."""
    reductions = []

    class Unit(nn.Module, FsdpModule):
        def __init__(self, name):
            nn.Module.__init__(self)
            self.label = name
            self.weight = nn.Parameter(torch.ones(()))

        def forward(self, x):
            x = x * self.weight
            for child in self.children():
                x = child(x)
            return x

        def is_root(self):
            return False

        def unshard(self):
            pass

        def reshard(self):
            pass

        def _reduce_gradient_groups(self):
            reductions.append((self.label, self.weight.grad.item()))
            self.weight.grad = None

    class Plan:
        def __init__(self, layer):
            self.layer = layer

        def set_fsdp_reshard_hooks(self, forward, backward):
            self.finish = lambda: backward(self.layer)

    root, shared, expert, other = [Unit(name) for name in ("root", "shared", "expert", "other")]
    shared.add_module("expert", expert)
    model = nn.Module()
    model.register_parameter("weight", root.weight)
    model.add_module("shared", shared)
    model.add_module("other", other)
    root.add_module("model", model)
    register_combined_1f1b_hooks(root)
    plans = [Plan(shared), Plan(other)] + ([Plan(shared)] if repeated else [])
    outer_model = nn.Sequential(root) if wrapped else root
    chunk_plan = SimpleNamespace(_transformer_layers=plans)
    TransformerModelChunkSchedulePlan.register_fsdp_hooks(chunk_plan, outer_model)
    remaining = chunk_plan._fsdp_remaining_modules
    assert remaining == (root,)
    for unit in root.modules():
        if isinstance(unit, FsdpModule):
            assert not unit._backward_hooks
            assert not unit.weight._post_accumulate_grad_hooks

    # The tied root weight receives contributions before and after layer backward.
    (root.weight * 2).backward()
    if repeated:
        shared(torch.tensor(5.0)).backward()
        plans[-1].finish()
        assert reductions == []
    other(torch.tensor(4.0)).backward()
    plans[1].finish()
    assert reductions == [("other", 4.0)]
    shared(torch.tensor(3.0)).backward()
    plans[0].finish()
    (root.weight * 7).backward()
    for unit in remaining:
        finish_fsdp_backward(unit)
    expected_shared = 8.0 if repeated else 3.0
    assert reductions == [
        ("other", 4.0),
        ("shared", expected_shared),
        ("expert", expected_shared),
        ("root", 9.0),
    ]
