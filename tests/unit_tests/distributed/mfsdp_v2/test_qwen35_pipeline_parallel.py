# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Qwen 3.5 end-to-end training test for experimental Megatron-FSDP v2 and PP."""

from functools import partial

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed import DeviceMesh
from torch.distributed.tensor import DTensor

from megatron.core import parallel_state
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import (
    Flat,
    Placements,
    fully_shard,
    fully_shard_optimizer,
)
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_linear_attention_pattern,
    get_moe_layer_pattern,
)
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.core.pipeline_parallel.p2p_communication import P2PCommunicator
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.gated_delta_net import HAVE_FLA, GatedDeltaNet
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.attention import SelfAttention
from megatron.training.models.gpt import GPTModelBuilder, GPTModelConfig
from tests.unit_tests.test_utilities import Utils


def _flat_placements() -> Placements:
    return Placements(dp_axes=[0], parameter=[Flat()], gradient=[Flat()], optimizer=[Flat()])


def _scaled_qwen35_config() -> GPTModelConfig:
    """Scale Qwen3.5-35B-A3B down while preserving its defining layer structure."""
    transformer = TransformerConfig(
        num_layers=4,
        hidden_size=256,
        num_attention_heads=8,
        num_query_groups=2,
        kv_channels=32,
        ffn_hidden_size=256,
        normalization="RMSNorm",
        layernorm_epsilon=1e-6,
        layernorm_zero_centered_gamma=True,
        qk_layernorm=True,
        attention_output_gate=True,
        add_bias_linear=False,
        gated_linear_unit=True,
        activation_func=F.silu,
        experimental_attention_variant="gated_delta_net",
        linear_attention_freq=4,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_num_key_heads=4,
        linear_num_value_heads=8,
        num_moe_experts=4,
        moe_ffn_hidden_size=64,
        moe_shared_expert_intermediate_size=64,
        moe_shared_expert_gate=True,
        moe_layer_freq=1,
        moe_router_load_balancing_type="aux_loss",
        moe_router_topk=2,
        moe_aux_loss_coeff=1e-3,
        moe_token_dispatcher_type="allgather",
        # Keep this test focused on the requested decoder FSDP/PP integration.
        # The current GPT MTP builder rejects experimental attention variants.
        mtp_num_layers=None,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        bf16=True,
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=2,
        expert_model_parallel_size=1,
        expert_tensor_parallel_size=1,
        transformer_impl="transformer_engine",
        cross_entropy_loss_fusion=False,
    )
    # Experimental FSDP v2 finalizes reductions from autograd hooks. The legacy
    # finalizer expects the DDP/Megatron-FSDP-v1 wrapper contract.
    transformer.finalize_model_grads_func = None
    return GPTModelConfig(
        transformer=transformer,
        vocab_size=256,
        seq_length=32,
        parallel_output=True,
        share_embeddings_and_output_weights=False,
        position_embedding_type="rope",
        rotary_percent=0.25,
        rotary_base=10_000_000,
    )


def _make_microbatches(
    iteration: int,
    num_microbatches: int,
    micro_batch_size: int,
    sequence_length: int,
    vocab_size: int,
    device: torch.device,
) -> list[dict[str, torch.Tensor]]:
    generator = torch.Generator(device=device)
    generator.manual_seed(12_345 + iteration)
    position_ids = (
        torch.arange(sequence_length, device=device).unsqueeze(0).expand(micro_batch_size, -1)
    )
    microbatches = []
    for _ in range(num_microbatches):
        tokens = torch.randint(
            0, vocab_size, (micro_batch_size, sequence_length), generator=generator, device=device
        )
        labels = torch.roll(tokens, shifts=-1, dims=1)
        microbatches.append(
            {
                "tokens": tokens,
                "labels": labels,
                "loss_mask": torch.ones_like(labels, dtype=torch.float32),
                "position_ids": position_ids,
            }
        )
    return microbatches


@pytest.mark.skipif(not HAVE_FLA, reason="Qwen 3.5 Gated DeltaNet requires FLA.")
def test_qwen35_mfsdp_v2_pipeline_parallel_trains_end_to_end(distributed_setup):
    """Train scaled Qwen 3.5 for two iterations with PP=2, DP=2, and ZeRO-3."""
    world_size = distributed_setup.world_size
    device = distributed_setup.device
    if world_size != 4 or device.type != "cuda":
        pytest.skip("This test requires exactly four CUDA ranks.")

    pipeline_parallel_size = 2
    num_microbatches = 2
    micro_batch_size = 1
    model_config = _scaled_qwen35_config()
    sequence_length = model_config.seq_length
    vocab_size = model_config.vocab_size
    assert vocab_size is not None

    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=pipeline_parallel_size,
        expert_model_parallel_size=1,
    )
    try:
        model_parallel_cuda_manual_seed(1234)
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        dp_group = pg_collection.dp_cp
        assert dp_group.size() == 2
        mesh = DeviceMesh.from_group(dp_group, device.type)

        model = GPTModelBuilder(model_config).build_model(pg_collection)
        model.train()
        model.to(device)

        expected_linear_attention = [1, 1, 1, 0]
        assert get_linear_attention_pattern(model.config) == expected_linear_attention
        assert get_moe_layer_pattern(model.config) == [1, 1, 1, 1]
        pp_rank = pg_collection.pp.rank()
        local_pattern = expected_linear_attention[2 * pp_rank : 2 * (pp_rank + 1)]
        assert len(model.decoder.layers) == len(local_pattern)
        for layer, is_gdn in zip(model.decoder.layers, local_pattern, strict=True):
            expected_attention_type = GatedDeltaNet if is_gdn else SelfAttention
            assert isinstance(layer.self_attention, expected_attention_type)
            assert layer.mlp.__class__.__name__ == "MoELayer"

        fully_shard(model, mesh=mesh, placements=_flat_placements())
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        fully_shard_optimizer(optimizer)

        p2p_communicator = P2PCommunicator(pg_collection.pp, model.config)
        forward_backward_func = get_forward_backward_func(pp_size=pipeline_parallel_size)
        losses_by_iteration = []

        def forward_step_func(data_iterator, stage):
            data = next(data_iterator)
            tokens = data["tokens"] if parallel_state.is_pipeline_first_stage() else None
            position_ids = (
                data["position_ids"] if parallel_state.is_pipeline_first_stage() else None
            )
            labels = data["labels"] if parallel_state.is_pipeline_last_stage() else None
            loss_mask = data["loss_mask"] if parallel_state.is_pipeline_last_stage() else None

            def loss_func(mask, output_tensor):
                loss = (output_tensor.float() * mask).sum() / mask.sum()
                return loss, {"lm loss": loss.detach()}

            output_tensor = stage(tokens, position_ids, None, labels=labels)
            return output_tensor, partial(loss_func, loss_mask)

        for iteration in range(2):
            optimizer.zero_grad(set_to_none=True)
            data_iterator = iter(
                _make_microbatches(
                    iteration,
                    num_microbatches,
                    micro_batch_size,
                    sequence_length,
                    vocab_size,
                    device,
                )
            )
            losses = forward_backward_func(
                forward_step_func=forward_step_func,
                data_iterator=data_iterator,
                model=model,
                num_microbatches=num_microbatches,
                seq_length=sequence_length,
                micro_batch_size=micro_batch_size,
                forward_only=False,
                pg_collection=pg_collection,
                p2p_communicator=p2p_communicator,
            )

            parameters = list(model.parameters())
            assert parameters
            for parameter in parameters:
                assert isinstance(parameter, DTensor)
                assert isinstance(parameter.grad, DTensor)
                assert torch.isfinite(parameter.grad.to_local()).all()

            params_before_step = [parameter.to_local().detach().clone() for parameter in parameters]
            optimizer.step()
            assert any(
                not torch.equal(before, parameter.to_local())
                for before, parameter in zip(params_before_step, parameters, strict=True)
            )

            if parallel_state.is_pipeline_last_stage():
                assert len(losses) == num_microbatches
                iteration_losses = torch.stack(
                    [loss_dict["lm loss"].float() for loss_dict in losses]
                )
                assert torch.isfinite(iteration_losses).all()
                losses_by_iteration.append(iteration_losses.mean())
            else:
                assert losses == []

        if parallel_state.is_pipeline_last_stage():
            assert len(losses_by_iteration) == 2
        dist.barrier()
    finally:
        Utils.destroy_model_parallel()
