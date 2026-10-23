# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""MFSDP v2 integration with the expert-parallel overlap schedule."""

from collections import Counter

import pytest
import torch

from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.distributed.fsdp.mcore_fsdp_adapter import FullyShardedDataParallel
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.module import FsdpModule
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.parameter_group import (
    FsdpParameterGroup,
)
from megatron.core.models.common.model_chunk_schedule_plan import TransformerModelChunkSchedulePlan
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_decoder_block_spec,
    get_gpt_mtp_block_spec,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.pipeline_parallel.utils import set_streams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.module import Float16Module
from megatron.core.transformer.multi_token_prediction import MTPLossAutoScaler
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.distributed.mfsdp_v2.mcore_utils import destroy_model_parallel
from tests.unit_tests.test_utilities import Utils


def test_ep_overlap_mtp(distributed_setup, monkeypatch):
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
        previous_plan = previous_loss = None
        for microbatch in range(3):
            reductions.clear()
            plan = model.build_schedule_plan(
                input_ids=torch.randint(0, 128, (1, 16), device=setup.device),
                labels=torch.randint(0, 128, (1, 16), device=setup.device),
                position_ids=torch.arange(16, device=setup.device).unsqueeze(0),
                attention_mask=None,
            )
            with wrapped.no_sync():
                loss = TransformerModelChunkSchedulePlan.run(
                    plan,
                    previous_plan,
                    b_grad=torch.ones_like(previous_loss) if microbatch else None,
                )
            assert reductions == (expected_reductions if microbatch else {})
            previous_plan, previous_loss = plan, loss
        reductions.clear()
        TransformerModelChunkSchedulePlan.run(
            None, previous_plan, b_grad=torch.ones_like(previous_loss)
        )
        wrapped.finish_grad_sync()
        assert reductions == expected_reductions
        assert all(unit._trainable_parameter_countdown.is_complete for unit in units)
    finally:
        destroy_model_parallel()
