# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import os
import torch
import modelopt.torch.quantization as mtq
from megatron.core import parallel_state
from megatron.training.utils import unwrap_model
from modelopt.torch.quantization.utils import is_quantized

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
    from datasets import load_dataset

    def mtbench_to_oai_chat(example):
        """Convert MTBench data to OpenAI chat completion format."""
        conversations = []
        for prompt in example["prompt"]:
            conversations.append({"role": "user", "content": prompt})
        example["conversations"] = conversations
        return example

    dataset = load_dataset("HuggingFaceH4/mt_bench_prompts", split="train", token=os.environ.get("HF_TOKEN", None))
    return dataset.map(mtbench_to_oai_chat)

def to_empty_if_meta(module: torch.nn.Module, *, device: torch.device, recurse=True):
    """Move tensors to device if not meta device; otherwise materialize with empty_like().
   
    Args:
        module: The target module to apply this transformation.
        device: The desired device of the parameters
            and buffers in this module.
        recurse: Whether parameters and buffers of submodules should
            be recursively moved to the specified device.
    """

    def _empty_like_if_meta(tensor: torch.Tensor, *, device: torch.device):
        if tensor.device == torch.device("meta"):
            return torch.empty_like(tensor, device=device)
        else:
            return tensor.to(device)

    module._apply(
        lambda t: _empty_like_if_meta(t, device=device), recurse=recurse
    )

def print_distributed_quant_summary(model, msg=""):
    from megatron.core import parallel_state
    from megatron.training import print_rank_0
    from megatron.training.utils import unwrap_model

    unwrapped_model = unwrap_model(model)
    if isinstance(unwrapped_model, list):
        unwrapped_model = unwrapped_model[0]

    if not is_quantized(unwrapped_model):
        return

    print_rank_0(f"{msg}\nQuantization summary of unwrapped model: {unwrapped_model}\n{'_'*80}")

    if not torch.distributed.is_initialized():
        mtq.print_quant_summary(unwrapped_model)
        return

    # Only print from unique TP ranks of [0, 1]
    if parallel_state.get_data_parallel_rank(with_context_parallel=True) == 0 and parallel_state.get_tensor_model_parallel_rank() in [0, 1]:
        TP_rank = parallel_state.get_tensor_model_parallel_rank()
        EP_rank = parallel_state.get_expert_model_parallel_rank()
        PP_rank = parallel_state.get_pipeline_model_parallel_rank()
        print(f"\nTP rank {TP_rank}, EP rank {EP_rank}, PP rank {PP_rank}")
        print("_" * 80)
        mtq.print_quant_summary(unwrapped_model)
