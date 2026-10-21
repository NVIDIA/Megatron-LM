# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import inspect
import os

import modelopt
import modelopt.torch.quantization as mtq
import torch
from modelopt.torch.distill import DistillationModel
from modelopt.torch.quantization.utils import is_quantized
from packaging.version import Version

from megatron.core.utils import unwrap_model


def is_distillation_model(model) -> bool:
    """Return whether a model or model-chunk list contains a distillation model."""
    unwrapped = unwrap_model(model)
    if isinstance(unwrapped, list):
        return any(isinstance(model_chunk, DistillationModel) for model_chunk in unwrapped)
    return isinstance(unwrapped, DistillationModel)


def should_use_modelopt_config(args) -> bool:
    """Return whether CLI inputs require the ModelOpt-aware model configuration.

    ModelOpt is used when ``--modelopt-enabled`` is set, for a checkpoint with ModelOpt
    state, or for distillation.
    """
    from megatron.post_training.checkpointing import has_modelopt_state
    from megatron.training import print_rank_0

    if getattr(args, "modelopt_enabled", False):
        return True
    if args.load is not None and has_modelopt_state(args.load, args.ckpt_format):
        print_rank_0("ModelOpt checkpoint detected")
        return True
    if getattr(args, "export_kd_teacher_load", None) is not None:
        return True
    return False


def modelopt_version_higher_than(target_version: str):
    """Check if Model-Optimizer is greater than this version."""
    info = "rank {:3}/{:3} checking if nvidia-modelopt-{} is higher than {}".format(
        torch.distributed.get_rank(),
        torch.distributed.get_world_size(),
        str(modelopt.__version__),
        target_version,
    )
    print(info)
    return Version(modelopt.__version__) > Version(target_version)


def modelopt_version_at_least(target_version: str):
    """Check if Model-Optimizer is greater or equal than this version."""
    info = "rank {:3}/{:3} checking if nvidia-modelopt-{} is at least {}".format(
        torch.distributed.get_rank(),
        torch.distributed.get_world_size(),
        str(modelopt.__version__),
        target_version,
    )
    print(info)
    return Version(modelopt.__version__) >= Version(target_version)


def function_has_parameter(function, argument_name: str) -> bool:
    """Check if a function has a specific argument."""
    sig = inspect.signature(function)
    return argument_name in sig.parameters


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

    dataset = load_dataset(
        "HuggingFaceH4/mt_bench_prompts", split="train", token=os.environ.get("HF_TOKEN", None)
    )
    return dataset.map(mtbench_to_oai_chat)


def print_distributed_quant_summary(model, msg=""):
    from megatron.core import parallel_state
    from megatron.core.utils import unwrap_model
    from megatron.training import print_rank_0

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
    if parallel_state.get_data_parallel_rank(
        with_context_parallel=True
    ) == 0 and parallel_state.get_tensor_model_parallel_rank() in [0, 1]:
        TP_rank = parallel_state.get_tensor_model_parallel_rank()
        EP_rank = parallel_state.get_expert_model_parallel_rank()
        PP_rank = parallel_state.get_pipeline_model_parallel_rank()
        print(f"\nTP rank {TP_rank}, EP rank {EP_rank}, PP rank {PP_rank}")
        print("_" * 80)
        mtq.print_quant_summary(unwrapped_model)
