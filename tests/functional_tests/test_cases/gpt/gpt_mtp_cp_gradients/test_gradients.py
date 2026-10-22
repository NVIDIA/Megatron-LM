# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Compare complete GPT+MTP training gradients with CP=1 and CP=2.

Run with two GPUs using ``python -m torch.distributed.run --nproc-per-node=2
-m tests.functional_tests.test_cases.gpt.gpt_mtp_cp_gradients.test_gradients``.
Synthetic data and identical initial weights provide an in-run reference;
no dataset, checkpoint, or stored golden values are required.
"""

import argparse
import copy
import json
import os
from pathlib import Path

import torch

from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.distributed.finalize_model_grads import finalize_model_grads
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_spec,
    get_gpt_mtp_block_spec,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.multi_token_prediction import MTPLossLoggingHelper
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import get_batch_on_this_cp_rank


def _batches(mask_name):
    """Include a rank with zero LM targets receiving a rolled MTP target."""
    positions = torch.arange(32, device="cuda")
    masks = {
        "full": torch.ones_like(positions, dtype=torch.bool),
        "empty": torch.zeros_like(positions, dtype=torch.bool),
        "first_only": positions == 0,
        "cross_rank": positions == 8,
        "answer_only": positions >= 25,
        "alternating": positions.remainder(2) == 0,
        "prefix": positions < 19,
    }
    # Accumulate two distinct microbatches, with unequal target counts except
    # for the full/empty controls. The CP1 DP replicas use identical data.
    for microbatch in range(2):
        mask = masks[mask_name].clone()
        if microbatch and mask_name not in ("full", "empty", "first_only", "cross_rank"):
            mask[17:] = False
        tokens = ((positions * 7 + microbatch * 11) % 128).unsqueeze(0)
        yield {
            "tokens": tokens,
            "labels": (tokens + 7) % 128,
            "position_ids": positions.unsqueeze(0),
            "loss_mask": mask.float().unsqueeze(0),
            "attention_mask": None,
            "cu_seqlens": None,
        }


def _train(cp_size, per_token, depth, initial_state=None):
    parallel_state.initialize_model_parallel(context_parallel_size=cp_size)
    torch.manual_seed(6103)
    model_parallel_cuda_manual_seed(6103)
    config = TransformerConfig(
        num_layers=2,
        hidden_size=128,
        num_attention_heads=4,
        ffn_hidden_size=256,
        mtp_num_layers=depth,
        mtp_loss_scaling_factor=1.0,
        calculate_per_token_loss=per_token,
        context_parallel_size=cp_size,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        attention_backend=AttnBackend.flash,
        bf16=True,
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
        use_cpu_initialization=True,
        gradient_accumulation_fusion=False,
        masked_softmax_fusion=False,
    )
    layer_spec = get_gpt_layer_with_transformer_engine_spec()
    model = (
        GPTModel(
            config=config,
            transformer_layer_spec=layer_spec,
            mtp_block_spec=get_gpt_mtp_block_spec(config, layer_spec, use_transformer_engine=True),
            vocab_size=128,
            max_sequence_length=32,
            share_embeddings_and_output_weights=True,
            position_embedding_type="rope",
        )
        .cuda()
        .bfloat16()
    )
    if initial_state is None:
        initial_state = {
            name: value.cpu().clone() if isinstance(value, torch.Tensor) else copy.deepcopy(value)
            for name, value in model.state_dict().items()
        }
    else:
        model.load_state_dict(initial_state)
    model = DistributedDataParallel(
        config,
        DistributedDataParallelConfig(grad_reduce_in_fp32=True, overlap_grad_reduce=False),
        model,
    )
    config.finalize_model_grads_func = finalize_model_grads
    config.no_sync_func = model.no_sync

    def forward_step(iterator, wrapped_model):
        batch = get_batch_on_this_cp_rank(
            next(iterator), is_hybrid_cp=False, cp_group=parallel_state.get_context_parallel_group()
        )
        mask = batch["loss_mask"]
        output = wrapped_model(
            batch["tokens"],
            batch["position_ids"],
            batch["attention_mask"],
            labels=batch["labels"],
            loss_mask=mask,
        )

        def loss_func(losses):
            loss = (losses.float() * mask).sum()
            count = mask.sum().int()
            if per_token:
                return loss, count, {}
            # Legacy schedules multiply this mean by CP size before DDP
            # averages. Preserve a differentiable local numerator.
            torch.distributed.all_reduce(count, group=parallel_state.get_context_parallel_group())
            return loss / count.clamp(min=1), {}

        return output, loss_func

    gradients = {}
    for mask_name in (
        "full",
        "empty",
        "first_only",
        "cross_rank",
        "answer_only",
        "alternating",
        "prefix",
    ):
        model.zero_grad_buffer()
        model.zero_grad(set_to_none=True)
        MTPLossLoggingHelper.tracker = {}
        get_forward_backward_func()(
            forward_step_func=forward_step,
            data_iterator=iter(_batches(mask_name)),
            model=[model],
            num_microbatches=2,
            seq_length=32,
            micro_batch_size=1,
        )
        gradients[mask_name] = {
            name: parameter.main_grad.detach().float().cpu().clone()
            for name, parameter in model.module.named_parameters()
            if parameter.requires_grad
        }
        assert all(torch.isfinite(value).all() for value in gradients[mask_name].values())
        if mask_name == "empty":
            assert all(torch.count_nonzero(value) == 0 for value in gradients[mask_name].values())
    del model
    MTPLossLoggingHelper.tracker = {}
    parallel_state.destroy_model_parallel()
    return initial_state, gradients


def main():
    """Compare every parameter, and fail after reporting all failing cases."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.distributed.init_process_group("nccl")
    assert torch.distributed.get_world_size() == 2, "Launch with exactly two GPUs"
    results = []
    for per_token in (False, True):
        for depth in (1, 3):
            state, reference = _train(1, per_token, depth)
            _, actual = _train(2, per_token, depth, state)
            for mask_name, expected_grads in reference.items():
                failures = []
                worst_relative_error = 0.0
                for name, expected in expected_grads.items():
                    observed = actual[mask_name][name]
                    # BF16 attention kernels have different reduction orders
                    # across CP sizes. Bound the error relative to each tensor,
                    # including small MTP parameters, rather than the whole model.
                    difference = (observed - expected).norm().item()
                    relative_error = difference / max(expected.norm().item(), 1e-12)
                    worst_relative_error = max(worst_relative_error, relative_error)
                    if difference > 0.025 * expected.norm().item() + 1e-7:
                        failures.append(name)
                results.append(
                    {
                        "per_token": per_token,
                        "depth": depth,
                        "mask": mask_name,
                        "worst_relative_error": worst_relative_error,
                        "failed_parameters": failures,
                    }
                )
                if torch.distributed.get_rank() == 0:
                    print(json.dumps(results[-1]), flush=True)
    if torch.distributed.get_rank() == 0 and args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2) + "\n")
    torch.distributed.destroy_process_group()
    assert all(not result["failed_parameters"] for result in results), "CP1/CP2 gradients differ"


if __name__ == "__main__":
    main()
