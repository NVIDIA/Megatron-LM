# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from typing import List, Optional

import torch
from modelopt.torch.utils.plugins import megatron_generate


def simple_speculative_generate(
    model,
    input_ids: torch.Tensor,
    images: Optional[torch.Tensor] = None,
    osl: int = 32,
    steps: int = 0,
    eos_token_id: List[int] = [],
    disable_tqdm: bool = False,
):
    """A simple speculative-decoding generate that drives the draft model via the
    target's greedy outputs and counts accepted draft tokens.
    """
    output_ids = megatron_generate(
        model,
        input_ids,
        osl=osl,
        eos_token_id=eos_token_id,
        enable_kv_cache=False,
        disable_tqdm=disable_tqdm,
    )
    output_ids = torch.cat((input_ids, output_ids), dim=-1)
    actual_osl = output_ids.shape[-1] - input_ids.shape[-1]

    total_steps = 0
    while input_ids.shape[-1] < output_ids.shape[-1]:
        total_steps += 1
        offset = input_ids.shape[-1] + 1

        # Speculative decoding forward
        # NOTE: PP is not yet supported.
        new_token, draft_tokens = model.pseudo_speculative_generate(input_ids, steps=steps)

        # Always accept the first token.
        input_ids = output_ids[:, : offset]

        if input_ids.shape[-1] >= output_ids.shape[-1]:
            break

        for i in range(draft_tokens.shape[-1]):
            if torch.equal(draft_tokens[:, i : i + 1], output_ids[:, offset: offset + 1]):
                offset += 1
            else:
                break

        # Broadcast the accepted offset from the last rank.
        offset = [offset]
        torch.distributed.broadcast_object_list(
            offset,
            src=torch.distributed.get_world_size() - 1,
        )

        input_ids = output_ids[:, : offset[0]]

    return output_ids, actual_osl, total_steps
