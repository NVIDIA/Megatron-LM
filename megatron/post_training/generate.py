# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from typing import List, Optional

import torch
from tqdm import tqdm

from megatron.core import mpu
from megatron.core.inference.communication_utils import broadcast_from_last_pipeline_stage
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.core.tensor_parallel.mappings import gather_from_tensor_model_parallel_region
from megatron.post_training.utils import get_current_memory_info


def simple_generate(
    model,
    input_ids: torch.Tensor,
    images: Optional[torch.Tensor] = None,
    osl: int = 32,
    eos_token_id: List[int] = [],
    disable_tqdm: bool = False,
):
    """A simple generate function without using KV-cache."""
    model.eval()

    def _dummy_loss_func(output_tensor, non_loss_data=True):
        return output_tensor

    def _forward_step_func(data, model):
        batch_size = data["tokens"].shape[0]
        seq_len = data["tokens"].shape[-1]
        device = data["tokens"].device

        attention_mask = (
            torch.triu(torch.ones((batch_size, seq_len, seq_len), device=device), diagonal=1)
            .bool()
            .view(batch_size, 1, seq_len, seq_len)
        )
        position_ids = torch.arange(
            data["tokens"].shape[0], dtype=torch.long, device=data["tokens"].device
        )
        output_tensor = model(data["tokens"], position_ids, attention_mask)
        return output_tensor, _dummy_loss_func

    disable_tqdm = disable_tqdm or torch.distributed.get_rank() > 0

    output_ids = None
    step_pbar = tqdm(range(osl), disable=disable_tqdm, leave=False)

    for step in step_pbar:
        step_pbar.set_description(get_current_memory_info())

        # When --sequence-parallel is used, sequence_len must be a multiple of
        # --tensor-parallel. We pad eos tokens on the left to be multiple of 32.
        num_pad_tokens = input_ids.shape[-1] % 32

        if num_pad_tokens > 0:
            num_pad_tokens = 32 - num_pad_tokens
            padding_shape = (input_ids.shape[0], num_pad_tokens)
            padded_tokens = torch.full(
                padding_shape, 0, dtype=input_ids.dtype, device=input_ids.device
            )
            tokens = torch.cat((input_ids, padded_tokens), dim=-1)
        else:
            tokens = input_ids

        list_of_logits = get_forward_backward_func()(
            forward_step_func=_forward_step_func,
            data_iterator=[{"tokens": tokens}],
            model=model,
            num_microbatches=1,
            seq_length=tokens.shape[-1],
            micro_batch_size=1,
            decoder_seq_length=tokens.shape[-1],
            forward_only=True,
            collect_non_loss_data=True,
        )

        if mpu.is_pipeline_last_stage():
            logits = gather_from_tensor_model_parallel_region(list_of_logits[0])
            eager_ids = logits[:, input_ids.shape[-1] - 1, :].argmax(dim=-1, keepdim=True).detach()
        else:
            eager_ids = None

        eager_ids = broadcast_from_last_pipeline_stage(
            [input_ids.shape[0], 1], input_ids.dtype, eager_ids
        )

        input_ids = torch.cat([input_ids, eager_ids], dim=-1)

        if output_ids is None:
            output_ids = eager_ids
        else:
            output_ids = torch.cat([output_ids, eager_ids], dim=-1)

        if eager_ids.item() in eos_token_id:
            break

    return output_ids


def simple_speculative_generate(
    model,
    input_ids: torch.Tensor,
    images: Optional[torch.Tensor] = None,
    osl: int = 32,
    draft_length: int = 0,
    eos_token_id: List[int] = [],
    disable_tqdm: bool = False,
):
    """A simple generate function without using KV-cache."""
    output_ids = simple_generate(
        model,
        input_ids,
        images=images,
        osl=osl,
        eos_token_id=eos_token_id,
        disable_tqdm=disable_tqdm,
    )
    output_ids = torch.cat((input_ids, output_ids), dim=-1)

    actual_osl = output_ids.shape[-1] - input_ids.shape[-1]
    total_steps = 0
    while input_ids.shape[-1] < output_ids.shape[-1]:
        total_steps += 1
        new_token, draft_tokens = model.pseudo_speculative_generate(input_ids, steps=draft_length)
        idx = input_ids.shape[-1]
        if not torch.equal(new_token, output_ids[:, idx : idx + 1]):
            if torch.distributed.get_rank() == 0:
                print(
                    "Rank {:3}/{:3} total_steps {} new {} ref {}".format(
                        torch.distributed.get_rank(),
                        torch.distributed.get_world_size(),
                        total_steps,
                        new_token,
                        output_ids[:, idx : idx + 1],
                    ),
                    flush=True,
                )
        input_ids = output_ids[:, : idx + 1]

        if input_ids.shape[-1] >= output_ids.shape[-1]:
            break

        offset = input_ids.shape[-1]

        for i in range(draft_tokens.shape[-1]):
            if torch.equal(draft_tokens[:, i : i + 1], output_ids[:, offset + i : offset + i + 1]):
                input_ids = output_ids[:, : offset + i + 1]
            else:
                break

    return output_ids, actual_osl, total_steps
