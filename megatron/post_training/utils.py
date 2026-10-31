# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import inspect
import os

import modelopt
import modelopt.torch.quantization as mtq
import torch
from modelopt.torch.quantization.utils import is_quantized
from packaging.version import Version


def maybe_enable_modelopt(args):
    """Set `args.modelopt_enabled` if a ModelOpt checkpoint or distillation teacher is
    configured. Idempotent and safe to call multiple times (e.g. once early in
    `pretrain_gpt.py` before building the model config, and again as a fallback in
    `training.py` for callers that don't go through that entrypoint).
    """
    if getattr(args, "modelopt_enabled", False):
        return

    from megatron.post_training.checkpointing import has_modelopt_state
    from megatron.training import print_rank_0

    if args.load is not None and has_modelopt_state(args.load):
        print_rank_0("ModelOpt checkpoint detected")
        args.modelopt_enabled = True
    if getattr(args, "export_kd_teacher_load", None):
        # For distillation ckpts without ModelOpt state
        args.modelopt_enabled = True


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

    module._apply(lambda t: _empty_like_if_meta(t, device=device), recurse=recurse)


def print_distributed_quant_summary(model, msg=""):
    """Print the quantization summary of a quantized model.

    The ranks of the first data-parallel replica whose tensor-parallel rank is 0 or 1 print the
    summary, labeled with their tensor-, expert- and pipeline-parallel ranks.

    Args:
        model: The model, or a list of model chunks, possibly wrapped. The ranks come from the
            ``dp_cp_gtp_remat``, ``tp``, ``ep`` and ``pp`` groups of the ``pg_collection`` of
            its first chunk. A chunk without a ``ProcessGroupCollection`` that sets all four
            groups falls back to the global process groups with a deprecation warning.
        msg: Header printed above the summary.
    """

    from megatron.core.process_groups_config import ProcessGroupCollection, resolve_process_groups
    from megatron.core.utils import get_pg_rank, unwrap_model
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

    required = ("dp_cp_gtp_remat", "tp", "ep", "pp")
    # A Megatron Core language model carries the collection it was built with; other models, for
    # example MimoModel, may have none.
    pg_collection = getattr(unwrapped_model, "pg_collection", None)
    if not isinstance(pg_collection, ProcessGroupCollection) or any(
        name not in vars(pg_collection) for name in required
    ):
        # Deprecated fallback: without a collection that sets these groups, this diagnostic
        # picks its ranks from the global groups.
        pg_collection = None
    pg_collection = resolve_process_groups(
        pg_collection,
        owner="print_distributed_quant_summary",
        required=required,
        argument="model.pg_collection",
        deprecated_in="0.21",
        removed_in="0.23",
    )
    TP_rank = get_pg_rank(pg_collection.tp)
    # Only print from unique TP ranks of [0, 1]
    if get_pg_rank(pg_collection.dp_cp_gtp_remat) == 0 and TP_rank in [0, 1]:
        EP_rank = get_pg_rank(pg_collection.ep)
        PP_rank = get_pg_rank(pg_collection.pp)
        print(f"\nTP rank {TP_rank}, EP rank {EP_rank}, PP rank {PP_rank}")
        print("_" * 80)
        mtq.print_quant_summary(unwrapped_model)
