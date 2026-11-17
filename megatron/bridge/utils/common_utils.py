# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Adapted from https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/53c20f81cb575f19af34ff0ae1749fb13b07999b/src/megatron/bridge/utils/common_utils.py
# Source license: Apache-2.0.

import os
import re
import types
import warnings
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed

from megatron.bridge.utils.packed_seq_utils import get_packed_seq_cp_partition_indices
from megatron.bridge.utils.slurm_utils import resolve_slurm_master_addr, resolve_slurm_master_port
from megatron.core import DistributedDataParallel as DDP
from megatron.core._rank_utils import safe_get_world_size as get_world_size_safe  # noqa: F401
from megatron.core._slurm_utils import resolve_slurm_rank
from megatron.core.transformer.module import Float16Module
from megatron.core.utils import get_batch_on_this_cp_rank
from megatron.training.utils.common_utils import get_local_rank_preinit  # noqa: F401

try:
    from megatron.core.distributed import TorchFullyShardedDataParallel as torch_FSDP

    ALL_MODULE_WRAPPER_CLASSNAMES = (DDP, torch_FSDP, Float16Module)
except ImportError:
    ALL_MODULE_WRAPPER_CLASSNAMES = (DDP, Float16Module)


def get_rank_safe() -> int:
    """Match MCore main rank resolution while keeping malformed launcher values fail-fast on MCore dev."""
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    if "RANK" in os.environ:
        return int(os.environ["RANK"])
    slurm_rank = resolve_slurm_rank()
    if slurm_rank is not None:
        return slurm_rank
    warnings.warn(
        "Could not determine rank from torch.distributed, RANK, or SLURM_PROCID. Defaulting to rank 0."
    )
    return 0


def get_last_rank() -> int:
    """Get the last rank in the distributed group"""
    if not torch.distributed.is_initialized():
        return 0
    return torch.distributed.get_world_size() - 1


def get_master_addr_safe() -> str:
    """Get the master address for distributed initialization.

    Fallback order:
    1. MASTER_ADDR environment variable (torchrun/torchelastic)
    2. SLURM_NODELIST parsed (SLURM)
    3. Default: localhost (with warning)

    Returns:
        The master node address.
    """
    if "MASTER_ADDR" in os.environ:
        return os.environ["MASTER_ADDR"]

    slurm_addr = resolve_slurm_master_addr()
    if slurm_addr is not None:
        return slurm_addr

    warnings.warn(
        "Could not determine master address from MASTER_ADDR or SLURM_NODELIST. Defaulting to 'localhost'."
    )
    return "localhost"


def get_master_port_safe() -> int:
    """Get the master port for distributed initialization.

    Fallback order:
    1. MASTER_PORT environment variable (torchrun/torchelastic)
    2. SLURM job-based port (SLURM_JOB_ID derived)
    3. Default: 29500 (with warning)

    Returns:
        The master port.
    """
    if "MASTER_PORT" in os.environ:
        return int(os.environ["MASTER_PORT"])

    slurm_port = resolve_slurm_master_port()
    if slurm_port is not None:
        return slurm_port

    warnings.warn(
        "Could not determine master port from MASTER_PORT or SLURM environment. Defaulting to 29500."
    )
    return 29500


def print_rank_0(message: str) -> None:
    """Print a message only on global rank 0.

    Args:
        message: The message string to print.
    """
    rank = get_rank_safe()
    if rank == 0:
        print(message, flush=True)


def warn_rank_0(message):
    """Warn only on rank 0."""
    rank = get_rank_safe()
    if rank == 0:
        warnings.warn(message)


def is_last_rank() -> bool:
    """Check if the current rank is the last rank in the default process group.

    Returns:
        True if the current rank is the last one, False otherwise.
    """
    return torch.distributed.get_rank() == (torch.distributed.get_world_size() - 1)


def print_rank_last(message: str) -> None:
    """Print a message only on the last rank of the default process group.

    Args:
        message: The message string to print.
    """
    if torch.distributed.is_initialized():
        if is_last_rank():
            print(message, flush=True)
    else:
        print(message, flush=True)


def maybe_initialize_distributed(timeout_minutes: int = 60) -> None:
    """Initialize the default process group from the standard launcher env vars.

    No-op when torch.distributed is unavailable or already initialized. This is the minimal
    bring-up used by standalone (e.g. inference) entry points that set up model parallelism
    separately; it does not perform MPU initialization (see ``training.initialize`` for the
    full training path).

    Args:
        timeout_minutes: Process-group timeout in minutes (useful for slow multi-node setup).
    """
    if not torch.distributed.is_available() or torch.distributed.is_initialized():
        return

    os.environ["RANK"] = os.environ.get("RANK", str(get_rank_safe()))
    os.environ["WORLD_SIZE"] = os.environ.get("WORLD_SIZE", str(get_world_size_safe()))
    os.environ["LOCAL_RANK"] = os.environ.get("LOCAL_RANK", str(get_local_rank_preinit()))
    os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", get_master_addr_safe())
    os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", str(get_master_port_safe()))
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.distributed.init_process_group("nccl", timeout=timedelta(minutes=timeout_minutes))


def hook_hf_module_setattr_for_tp_grad_sync(module: torch.nn.Module) -> torch.nn.Module:
    """Mark params for TP grad sync and hook __setattr__ on a module and its children.

    This ensures that all existing parameters under the provided module have the
    attribute ``average_gradients_across_tp_domain=True`` and that any future
    submodules assigned onto this module (or any of its current children) will
    also have their parameters marked automatically.

    Args:
        module: The root module (typically a Hugging Face module instance).

    Returns:
        The same module instance for convenience.
    """
    if module is None:
        return module

    # Mark all existing parameters recursively
    for param in module.parameters(recurse=True):
        setattr(param, "average_gradients_across_tp_domain", True)

    def _wrap_setattr(original_setattr):
        def _wrapped(self, name, value):
            original_setattr(name, value)
            if isinstance(value, torch.nn.Module):
                for p in value.parameters(recurse=True):
                    setattr(p, "average_gradients_across_tp_domain", True)

        return _wrapped

    # Hook __setattr__ on the module and all existing submodules to catch
    # future dynamic assignments anywhere in the hierarchy.
    for submodule in module.modules():
        if getattr(submodule, "_tp_grad_sync_setattr_wrapped", False):
            continue
        original_setattr = submodule.__setattr__
        wrapped = _wrap_setattr(original_setattr)
        submodule.__setattr__ = types.MethodType(wrapped, submodule)
        setattr(submodule, "_tp_grad_sync_setattr_wrapped", True)

    return module


def extract_expert_number_from_param(param_name: str) -> int:
    """Extract the expert number from a parameter name.

    Args:
        param_name: The parameter name to extract the expert number from.

    Returns:
        The expert number.
    """
    patterns = (r"local_experts\.(\d+)", r"(?:weight|bias)(\d+)", r"experts\.(\d+)")
    for pattern in patterns:
        match = re.search(pattern, param_name)
        if match:
            return int(match.group(1))
    raise ValueError(
        f"No expert number found in parameter name: {param_name}. "
        f"Please update the regex patterns {patterns} if necessary."
    )


def disable_mtp_for_inference(m: torch.nn.Module) -> None:
    """Disable Multi-Token Prediction (MTP) on a Megatron model for inference.

    MTP is only used during training. Setting ``mtp_num_layers = None`` and
    clearing ``mtp_process`` on the language model prevents hangs on NCCL
    collectives during inference.

    Args:
        m: A Megatron model (or DDP-wrapped model) to disable MTP on.
    """
    m.config.mtp_num_layers = None
    m.config.grad_scale_func = None
    inner = m.module if hasattr(m, "module") else m
    lang = getattr(inner, "language_model", inner)
    if hasattr(lang, "mtp_process"):
        lang.mtp_process = False


def resolve_path(path: str) -> Path:
    """Resolve a path to an absolute path."""

    return Path(path).expanduser().absolute().resolve()


def slice_batch_for_context_parallel(
    inputs_embeds: torch.Tensor,
    labels: torch.Tensor,
    loss_mask: torch.Tensor,
    position_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    packed_seq_params,
    pg_collection,
):
    """Slice batch tensors for Context Parallelism (CP) in VLM models.

    This function handles CP slicing AFTER vision-text embedding merge, ensuring
    image token positions are correctly preserved. It supports both:
    - THD format (packed sequences): Uses MCore's packed-sequence CP partitioning
    - BSHD format: Uses Megatron's get_batch_on_this_cp_rank with zigzag pattern

    Args:
        inputs_embeds: Input embeddings tensor in (T, B, D) format.
        labels: Labels tensor.
        loss_mask: Loss mask tensor.
        position_ids: Position IDs tensor.
        attention_mask: Attention mask tensor.
        packed_seq_params: PackedSeqParams for THD format, or None for BSHD.
        pg_collection: ProcessGroupCollection containing CP group info.

    Returns:
        Tuple of (inputs_embeds, labels, loss_mask, position_ids, attention_mask)
        with all tensors sliced for this CP rank. inputs_embeds remains in (T, B, D) format.
    """
    cp_size = pg_collection.cp.size()
    if cp_size <= 1:
        return inputs_embeds, labels, loss_mask, position_ids, attention_mask

    cp_rank = pg_collection.cp.rank()

    # (T, B, D) -> (B, T, D) for slicing
    if inputs_embeds is not None:
        inputs_embeds = inputs_embeds.transpose(0, 1).contiguous()

    # MCore's THD path slices within each packed sequence rather than across them.
    if packed_seq_params is not None and packed_seq_params.qkv_format == "thd":
        reference_tensor = inputs_embeds
        if reference_tensor is None:
            reference_tensor = next(
                (tensor for tensor in (labels, loss_mask, position_ids) if tensor is not None), None
            )
        if reference_tensor is None:
            raise ValueError("At least one sequence tensor is required for THD CP slicing")

        seq_len = reference_tensor.size(1)
        index = get_packed_seq_cp_partition_indices(
            packed_seq_params,
            total_tokens=seq_len,
            cp_size=cp_size,
            cp_rank=cp_rank,
            device=reference_tensor.device,
            cp_group=pg_collection.cp,
        )

        # Slice all tensors using THD indices
        if inputs_embeds is not None:
            inputs_embeds = inputs_embeds.index_select(1, index)
        if labels is not None:
            labels = labels.index_select(1, index)
        if loss_mask is not None:
            loss_mask = loss_mask.index_select(1, index)
        if position_ids is not None:
            position_ids = position_ids.index_select(1, index)
        # Note: attention_mask and packed_seq_params stay unchanged for ring attention
    else:
        # For BSHD format, use standard zigzag slicing
        cp_group = pg_collection.cp
        cp_batch = get_batch_on_this_cp_rank(
            {
                "decoder_input": inputs_embeds,
                "labels": labels,
                "loss_mask": loss_mask,
                "position_ids": position_ids,
                "attention_mask": attention_mask,
            },
            is_hybrid_cp=False,
            cp_group=cp_group,
        )

        inputs_embeds = cp_batch.get("decoder_input")
        labels = cp_batch.get("labels")
        loss_mask = cp_batch.get("loss_mask")
        position_ids = cp_batch.get("position_ids")
        attention_mask = cp_batch.get("attention_mask")

    # Transpose back to (T, B, D)
    if inputs_embeds is not None:
        inputs_embeds = inputs_embeds.transpose(0, 1).contiguous()

    return inputs_embeds, labels, loss_mask, position_ids, attention_mask
