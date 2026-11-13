# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import gc
from dataclasses import replace

import pytest
import torch

from megatron.core.distributed import DistributedDataParallelConfig, finalize_model_grads
from megatron.core.distributed.fsdp.mcore_fsdp_adapter import FullyShardedDataParallel
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import fully_shard_optimizer
from megatron.core.full_cuda_graph import FullCudaGraphWrapper, StaticBufferLoader
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


def _reset_full_cuda_graph_state():
    """Drop the globally stored full-iteration graph so the next run captures its own."""
    FullCudaGraphWrapper.curr_iteration = {"training": 0, "validation": 0}
    FullCudaGraphWrapper.cuda_graph = {"training": None, "validation": None}
    FullCudaGraphWrapper.result = {"training": None, "validation": None}
    StaticBufferLoader.static_buffers = {"training": [], "validation": []}


class TestMFSDPv2EPOverlapCudaGraph:
    """Capture full iterations of the EP-overlap 1F1B schedule with Megatron-FSDP v2."""

    def setup_method(self, method):
        if Utils.world_size < 2 or Utils.world_size % 2:
            pytest.skip("EP=2 requires an even world size of at least two.")
        Utils.initialize_model_parallel(1, 1, expert_model_parallel_size=2)
        self.process_group_collection = ProcessGroupCollection.use_mpu_process_groups()

    def teardown_method(self, method):
        _reset_full_cuda_graph_state()
        Utils.destroy_model_parallel()
        # Free this test's models now rather than during a later test's memory accounting.
        gc.collect()

    def test_ep_overlap_full_iteration_cuda_graph_matches_default_schedule(self):
        """Capture a full iteration driven by the EP-overlap 1F1B schedule.

        The overlap schedule bypasses FsdpModule's pre-backward hook, which otherwise joins the
        gradient-reduction stream to the capture before the first reduction allocates on it.
        Compare against the captured default schedule, which must train identically.
        """
        process_group_collection = self.process_group_collection
        config = TransformerConfig(
            num_layers=2,
            hidden_size=64,
            num_attention_heads=4,
            attention_backend=AttnBackend.unfused,
            num_moe_experts=4,
            expert_model_parallel_size=2,
            moe_router_topk=2,
            moe_ffn_hidden_size=128,
            moe_grouped_gemm=True,
            moe_token_dispatcher_type="alltoall",
            # Drop-padding MoE keeps token counts static, as full-iteration capture requires.
            moe_expert_capacity_factor=1.0,
            moe_pad_expert_input_to_capacity=True,
            add_bias_linear=False,
            bf16=True,
            params_dtype=torch.bfloat16,
            attention_dropout=0.0,
            hidden_dropout=0.0,
            gradient_accumulation_fusion=False,
            cuda_graph_impl="full_iteration",
        )

        def build_model_and_optimizer(config: TransformerConfig):
            # Seed identically so both models start from the same weights.
            torch.manual_seed(123)
            model_parallel_cuda_manual_seed(123, te_rng_tracker=True, force_reset_rng=True)
            model = GPTModel(
                config=config,
                transformer_layer_spec=get_gpt_decoder_block_spec(
                    config, use_transformer_engine=True
                ),
                vocab_size=128,
                max_sequence_length=32,
                pg_collection=process_group_collection,
            ).cuda()
            model = FullyShardedDataParallel(
                config=config,
                ddp_config=DistributedDataParallelConfig(
                    use_megatron_fsdp=True,
                    megatron_fsdp_version=2,
                    use_distributed_optimizer=False,
                    data_parallel_sharding_strategy="optim_grads_params",
                    megatron_fsdp_cuda_graph_mode=True,
                ),
                module=model,
                pg_collection=process_group_collection,
            )
            # Like MCore training, the schedule finishes gradient synchronization.
            config.no_sync_func = model.no_sync
            config.finalize_model_grads_func = finalize_model_grads
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            fully_shard_optimizer(optimizer)
            return model, optimizer

        def loss_func(output_tensor):
            loss = output_tensor.float().sum()
            return loss, {"loss": loss}

        def forward_step(data_iterator, model, return_schedule_plan=False):
            data = next(data_iterator)
            if return_schedule_plan:
                return model.build_schedule_plan(**data), loss_func
            return model(**data), loss_func

        torch.manual_seed(4321 + torch.distributed.get_rank())
        steps = [
            [
                {
                    "input_ids": torch.randint(0, 128, (1, 32), device="cuda"),
                    "labels": torch.randint(0, 128, (1, 32), device="cuda"),
                    "position_ids": torch.arange(32, device="cuda").unsqueeze(0),
                    "attention_mask": torch.ones((1, 1, 32, 32), dtype=torch.bool, device="cuda"),
                }
                for _ in range(2)
            ]
            for _ in range(4)
        ]

        def train(overlap: bool):
            _reset_full_cuda_graph_state()
            model, optimizer = build_model_and_optimizer(
                replace(config, overlap_moe_expert_parallel_comm=overlap)
            )
            forward_backward_func = FullCudaGraphWrapper(
                get_forward_backward_func(), cuda_graph_warmup_steps=1
            )
            losses = []
            for microbatches in steps:
                optimizer.zero_grad(set_to_none=False)
                microbatch_losses = forward_backward_func(
                    forward_step_func=forward_step,
                    data_iterator=[iter(microbatches)],
                    model=[model],
                    num_microbatches=len(microbatches),
                    seq_length=32,
                    micro_batch_size=1,
                    forward_only=False,
                )
                optimizer.step()
                losses.append(torch.stack([entry["loss"] for entry in microbatch_losses]))
            assert FullCudaGraphWrapper.cuda_graph["training"] is not None
            return model, torch.stack(losses)

        reference_model, reference_losses = train(overlap=False)
        model, losses = train(overlap=True)

        torch.testing.assert_close(losses, reference_losses, rtol=0, atol=0)
        for (name, parameter), (_, reference_parameter) in zip(
            model.named_parameters(), reference_model.named_parameters()
        ):
            torch.testing.assert_close(
                parameter.to_local(),
                reference_parameter.to_local(),
                rtol=0,
                atol=0,
                msg=lambda m: f"{name}: {m}",
            )
