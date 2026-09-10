# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Real non-interleaved pipeline loss/gradient parity with EP all-to-all overlap.

Run on eight GPUs to cover PP2 and PP4, both with EP2::

    CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --standalone --nproc-per-node=8 \
        -m pytest tests/unit_tests/a2a_overlap/test_pipeline_schedule.py \
        --experimental -q --disable-warnings --tb=short

Four GPUs also run the PP2 cases. Fixed and changing sequence lengths exercise
both P2P payload-only and shape-negotiation protocols. Regular unfused attention
keeps this test independent of the DSv4 kernels' device requirements.
"""

import gc

import pytest
import torch

from megatron.core.enums import ModelType
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.pipeline_parallel.p2p_communication import P2PCommunicator
from megatron.core.pipeline_parallel.schedules import (
    forward_backward_pipelining_without_interleaving,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.moe.moe_logging import destroy_moe_metrics_tracker
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


def _build_pipeline_model(pp_size, pg_collection, overlap, variable_seq_lengths):
    config = TransformerConfig(
        num_layers=2 * pp_size,
        hidden_size=128,
        num_attention_heads=4,
        ffn_hidden_size=256,
        num_moe_experts=4,
        moe_ffn_hidden_size=128,
        moe_shared_expert_intermediate_size=128,
        moe_grouped_gemm=True,
        moe_router_topk=2,
        moe_router_dtype="fp32",
        moe_token_dispatcher_type="alltoall",
        expert_model_parallel_size=2,
        pipeline_model_parallel_size=pp_size,
        bf16=True,
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
        attention_backend=AttnBackend.unfused,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        add_bias_linear=False,
        gated_linear_unit=True,
        activation_func=torch.nn.functional.silu,
        normalization="RMSNorm",
        overlap_moe_expert_parallel_comm=overlap,
        deallocate_pipeline_outputs=True,
        variable_seq_lengths=variable_seq_lengths,
        # Compare local parameter gradients directly; no DDP main_grad buffers
        # or optimizer step should obscure missing/wrong pipeline backwards.
        gradient_accumulation_fusion=False,
    )
    model = GPTModel(
        config=config,
        transformer_layer_spec=get_gpt_decoder_block_spec(
            config, use_transformer_engine=True, pp_rank=pg_collection.pp.rank()
        ),
        vocab_size=256,
        max_sequence_length=64,
        pre_process=pg_collection.pp.rank() == 0,
        post_process=pg_collection.pp.rank() == pp_size - 1,
        share_embeddings_and_output_weights=False,
        position_embedding_type="rope",
        pg_collection=pg_collection,
    ).cuda()
    model.model_type = ModelType.encoder_or_decoder
    return model


def _run_pipeline(model, data, pg_collection):
    def forward_step(data_iterator, module, return_schedule_plan=False):
        batch = next(data_iterator)
        forward = module.build_schedule_plan if return_schedule_plan else module
        output = forward(**batch)

        def loss_func(losses):
            loss = losses.float().mean()
            return loss, {"loss": loss.detach().clone()}

        return output, loss_func

    losses = forward_backward_pipelining_without_interleaving(
        forward_step_func=forward_step,
        data_iterator=iter(data),
        model=model,
        num_microbatches=len(data),
        seq_length=64,
        micro_batch_size=2,
        forward_only=False,
        p2p_communicator=P2PCommunicator(pg_collection.pp, model.config),
        pg_collection=pg_collection,
    )
    torch.cuda.synchronize()
    gradients = {
        name: None if parameter.grad is None else parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
    }
    return losses, gradients


@pytest.mark.internal
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("pp_size", [2, 4], ids=["pp2", "pp4"])
@pytest.mark.parametrize("short", [True, False], ids=["short", "steady"])
@pytest.mark.parametrize("variable_seq_lengths", [False, True], ids=["fixed", "variable"])
def test_noninterleaved_pipeline_ep_overlap_loss_and_grad_parity(
    pp_size, short, variable_seq_lengths
):
    """Compare actual P2P and EP collectives for short and steady-state batches.

    Every rank compares all its local parameters, including attention, router,
    routed/shared experts and the first/last stage's embedding/output parameters.
    The reference is the existing non-overlapped pipeline backward schedule.
    """
    if Utils.world_size < 2 * pp_size or Utils.world_size % (2 * pp_size):
        pytest.skip(f"PP{pp_size}/EP2 requires a world size divisible by {2 * pp_size}")
    Utils.initialize_model_parallel(
        pipeline_model_parallel_size=pp_size,
        expert_model_parallel_size=2,
        distributed_timeout_minutes=2,
    )
    models = []
    try:
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        torch.manual_seed(123)
        model_parallel_cuda_manual_seed(123)
        reference = _build_pipeline_model(
            pp_size, pg_collection, overlap=False, variable_seq_lengths=variable_seq_lengths
        )
        models.append(reference)
        overlapped = _build_pipeline_model(
            pp_size, pg_collection, overlap=True, variable_seq_lengths=variable_seq_lengths
        )
        models.append(overlapped)
        overlapped.load_state_dict(reference.state_dict())

        num_microbatches = 1 if short else pp_size + 2
        # Distinct DP inputs exercise real EP exchange. Each corresponding PP
        # rank uses the same generator seed, so tokens and labels stay aligned.
        generator = torch.Generator(device="cuda").manual_seed(765 + pg_collection.dp_cp.rank())
        lengths = [
            (32, 64, 48)[microbatch % 3] if variable_seq_lengths else 64
            for microbatch in range(num_microbatches)
        ]
        data = [
            {
                "input_ids": torch.randint(0, 256, (2, length), device="cuda", generator=generator),
                "labels": torch.randint(0, 256, (2, length), device="cuda", generator=generator),
                "position_ids": torch.arange(length, device="cuda").expand(2, -1),
                "attention_mask": None,
            }
            for length in lengths
        ]

        expected_losses, expected_grads = _run_pipeline(reference, data, pg_collection)
        actual_losses, actual_grads = _run_pipeline(overlapped, data, pg_collection)
        expected_loss_count = num_microbatches if reference.post_process else 0
        assert len(expected_losses) == len(actual_losses) == expected_loss_count
        for expected, actual in zip(expected_losses, actual_losses):
            torch.testing.assert_close(actual["loss"], expected["loss"], rtol=0, atol=0)

        assert actual_grads.keys() == expected_grads.keys()
        assert any("experts" in name and grad is not None for name, grad in expected_grads.items())
        assert any(
            "shared_experts" in name and grad is not None for name, grad in expected_grads.items()
        )
        for name, expected in expected_grads.items():
            actual = actual_grads[name]
            assert (actual is None) == (expected is None), name
            if expected is None:
                continue
            # As in Hybrid overlap parity, BF16 accumulation changes with the
            # backward ordering. Scale atol to each parameter's gradient so a
            # missing small router/shared-expert gradient cannot pass on atol.
            scale = expected.float().abs().max().item()
            torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.002 * scale, msg=name)
    finally:
        for model in models:
            model.zero_grad(set_to_none=True)
        models.clear()
        reference = overlapped = None
        destroy_moe_metrics_tracker()
        Utils.destroy_model_parallel()
        gc.collect()
        torch.cuda.empty_cache()
