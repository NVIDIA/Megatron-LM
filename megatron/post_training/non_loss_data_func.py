# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import torch

from megatron.post_training.generate import simple_speculative_generate
from megatron.post_training.utils import get_mtbench_chat_data
from megatron.training import get_tokenizer
from megatron.training.utils import unwrap_model


def report_draft_acceptance_length(model, osl: int = 64, draft_length: int = 7):
    """Report MTBench acceptance length."""
    tokenizer = get_tokenizer()._tokenizer
    unwrapped_model = unwrap_model(model)[0]

    if unwrapped_model.training:
        return
    if not hasattr(unwrapped_model, "pseudo_speculative_generate"):
        return

    dataset = get_mtbench_chat_data()

    category_and_prompt = {}

    for example in dataset:
        if example["category"] not in category_and_prompt:
            category_and_prompt[example["category"]] = [example["conversations"][0]]

    total_osl = 0
    total_steps = 0
    for category, conversations in category_and_prompt.items():
        input_ids = tokenizer.apply_chat_template(
            conversations, return_tensors="pt", add_generation_prompt=True
        ).to(torch.cuda.current_device())
        output_ids, actual_osl, steps = simple_speculative_generate(
            unwrapped_model, input_ids, osl=osl, draft_length=draft_length, disable_tqdm=True
        )
        total_osl += actual_osl
        total_steps += steps
        if torch.distributed.get_rank() == 0:
            al = actual_osl / steps
            ar = al / draft_length
            print(
                "Rank {:3}/{:3} {:12} AL {:.1f} AR {:.2f} STEPS {:5}/{:5} DRAFT {:2}".format(
                    torch.distributed.get_rank(),
                    torch.distributed.get_world_size(),
                    category,
                    al,
                    ar,
                    steps,
                    actual_osl,
                    draft_length,
                ),
                flush=True,
            )
    if torch.distributed.get_rank() == 0:
        al = total_osl / total_steps
        ar = al / draft_length
        print(
            "Rank {:3}/{:3} {:12} AL {:.1f} AR {:.2f} STEPS {:5}/{:5} DRAFT {:2}".format(
                torch.distributed.get_rank(),
                torch.distributed.get_world_size(),
                "average",
                al,
                ar,
                total_steps,
                total_osl,
                draft_length,
            ),
            flush=True,
        )
    torch.distributed.barrier()
