# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import torch
from datasets import load_dataset


def get_current_memory_info():
    """Get current memory usage."""
    remaining_mem, total_mem = torch.cuda.mem_get_info()
    info = "rank {:3}/{:3}  memory remaining {:03}% ({}/{} MB) ".format(
        torch.distributed.get_rank(),
        torch.distributed.get_world_size(),
        int(remaining_mem * 100 / total_mem),
        remaining_mem // 1048576,
        total_mem // 1048576,
    )
    return info


def report_current_memory_info():
    """Report current memory usage."""
    print(get_current_memory_info(), flush=True)
    torch.distributed.barrier()


def get_mtbench_chat_data():
    """Return a MTBench dataset."""

    def mtbench_to_oai_chat(example):
        """Convert MTBench data to OpenAI chat completion format."""
        conversations = []
        for prompt in example["prompt"]:
            conversations.append({"role": "user", "content": prompt})
        example["conversations"] = conversations
        return example

    dataset = load_dataset("HuggingFaceH4/mt_bench_prompts", split="train")
    return dataset.map(mtbench_to_oai_chat)
