# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

import logging


logger = logging.getLogger(__name__)

from typing import Any, Callable

import torch
from megatron.core import tensor_parallel
from megatron.core.distributed import (
    DistributedDataParallel,
    DistributedDataParallelConfig,
    FullyShardedDataParallel,
    TorchFullyShardedDataParallel,
)
from megatron.core.enums import ModelType
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer import MegatronModule, TransformerConfig
from megatron.core.transformer.module import Float16Module
from megatron.core.utils import get_model_config


try:
    from megatron.core.fp8_utils import correct_amax_history_if_needed
except ImportError:
    correct_amax_history_if_needed = None



def unimodal_build_distributed_models(
    build_model_func: Callable,
    transformer_config: TransformerConfig,
    pg_collection: ProcessGroupCollection,
    ddp_config: DistributedDataParallelConfig | None = None,
    overlap_param_gather_with_optimizer_step: bool = False,
    use_megatron_fsdp: bool = False,
    use_torch_fsdp2: bool = False,
    wrap_with_ddp: bool = True,
    data_parallel_random_init: bool = False,
    mixed_precision_wrapper: Callable[[Any, MegatronModule], MegatronModule] | None = Float16Module,
    pre_wrap_hook: Callable[[list[MegatronModule]], list[MegatronModule]] | None = None,
    model_type: ModelType = ModelType.encoder_or_decoder,
) -> list[MegatronModule]:
    """Build model stages and wrap for distributed training.

    Shared helper for unimodal models (GPT, Hybrid, etc.) that share the same procedure
    for distributed model initialization. Performs the following steps in order:

    1. Build virtual pipeline stages (one per VP rank, or a single stage if no VP)
    2. Apply ``pre_wrap_hook``
    3. Set tensor model parallel attributes on all parameters
    4. Move model to GPU (unless using FSDP2 or CPU/meta-device initialization)
    5. Apply mixed precision wrapper (e.g. ``Float16Module``)
    6. Materialize meta-device tensors if ``init_model_with_meta_device`` is set
    7. Optionally wrap with DDP/FSDP

    Args:
        build_model_func: Callable that builds a single model stage (e.g. ``ModelBuilder.build_model``).
        transformer_config: TransformerConfig; used for VP size, precision, and device placement.
        pg_collection: Model communication process groups.
        ddp_config: DistributedDataParallel configuration. Required when ``wrap_with_ddp=True``.
        overlap_param_gather_with_optimizer_step: Whether to overlap parameter gather with optimizer step.
        use_megatron_fsdp: Whether to use Megatron FSDP.
        use_torch_fsdp2: Whether to use Torch FSDP 2.0.
        wrap_with_ddp: Set to False to skip the DDP/FSDP wrapper.
        data_parallel_random_init: Whether to broadcast parameters from data-parallel rank 0.
        mixed_precision_wrapper: Mixed precision wrapper applied per model stage, e.g. ``Float16Module``.
            Pass ``None`` to skip.
        pre_wrap_hook: Hook applied to the model stage list before any wrapping.
        model_type: Deprecated flag, only used for backwards compatibility.

    Returns:
        List of model stages, wrapped and ready for distributed training.
    """
    if wrap_with_ddp and not ddp_config:
        raise ValueError("ddp_config is required when wrap_with_ddp is True")

    vp_size = transformer_config.virtual_pipeline_model_parallel_size
    init_model_with_meta_device = transformer_config.init_model_with_meta_device
    if init_model_with_meta_device:
        with torch.device("meta"):
            model_list = build_virtual_pipeline_stages(build_model_func, pg_collection, vp_size, model_type)
    else:
        model_list = build_virtual_pipeline_stages(build_model_func, pg_collection, vp_size, model_type)

    # Apply pre wrap hooks
    if pre_wrap_hook is not None:
        if not callable(pre_wrap_hook):
            raise TypeError("pre_wrap_hook must be a callable")
        _model = pre_wrap_hook(model_list)
        if _model is not None:
            model_list = _model
        else:
            logger.warning("Final pre wrap hook returned None, skipping pre wrap hooks.")

    # Set tensor model parallel attributes if not set.
    # Only parameters that are already tensor model parallel have these
    # attributes set for them. We should make sure the default attributes
    # are set for all params so the optimizer can use them.
    for model_module in model_list:
        for param in model_module.parameters():
            tensor_parallel.set_defaults_if_not_set_tensor_model_parallel_attributes(param)

    _print_num_params(model_list, pg_collection=pg_collection)

    # GPU allocation.
    # For FSDP2, we don't allocate GPU memory here. We allocate GPU memory
    # in the fully_shard function of FSDP2 instead.
    use_cpu_initialization = transformer_config.use_cpu_initialization
    if not use_torch_fsdp2 and not use_cpu_initialization and not init_model_with_meta_device:
        for model_module in model_list:
            model_module.cuda(torch.cuda.current_device())

    model_list = _wrap_with_mp_wrapper(model_list, transformer_config, mixed_precision_wrapper)

    # Materialize tensors on meta device (GPU allocation) if not using FSDP2 and not using Megatron FSDP.
    if init_model_with_meta_device and not use_torch_fsdp2 and not use_megatron_fsdp:
        model_list = [
            to_empty_if_meta_device(model_module, device=torch.device("cuda")) for model_module in model_list
        ]

    if correct_amax_history_if_needed is not None:
        correct_amax_history_if_needed(model_list)

    if wrap_with_ddp:
        model_list = _ddp_wrap(
            model_list,
            data_parallel_random_init,
            ddp_config,
            overlap_param_gather_with_optimizer_step,
            use_megatron_fsdp=use_megatron_fsdp,
            use_torch_fsdp2=use_torch_fsdp2,
            pg_collection=pg_collection,
        )

    return model_list


def _print_num_params(model: list[MegatronModule], pg_collection: ProcessGroupCollection) -> None:
    """Print the number of parameters in the model on rank 0.

    Only prints on data parallel rank 0 to avoid duplicate output.
    Shows parameter count per (tensor parallel, pipeline parallel) rank.

    Args:
        model: List of model modules to count parameters from
        pg_collection: Model communication process groups.
    """
    if (pg_collection.dp.rank() == 0) and (pg_collection.cp.rank() == 0):
        print(
            " > number of parameters on (tensor, pipeline) model parallel rank ({}, {}): {}".format(
                pg_collection.tp.rank(),
                pg_collection.pp.rank(),
                sum([sum([p.nelement() for p in model_module.parameters()]) for model_module in model]),
            ),
            flush=True,
        )


def _wrap_with_mp_wrapper(
    model_list: list[MegatronModule],
    transformer_config: TransformerConfig,
    mixed_precision_wrapper: Callable[[Any, MegatronModule], MegatronModule] | None = Float16Module,
) -> list[MegatronModule]:
    fp16 = transformer_config.fp16
    bf16 = transformer_config.bf16
    if (fp16 or bf16) and mixed_precision_wrapper is not None:
        model_list = [mixed_precision_wrapper(transformer_config, model_module) for model_module in model_list]

        # Maintain expert bias in float32 wrapped in Float16Module
        for model_module in model_list:
            for submodule in model_module.modules():
                if hasattr(submodule, "_maintain_float32_expert_bias"):
                    submodule._maintain_float32_expert_bias()

    return model_list


def _ddp_wrap(
    model: list[MegatronModule],
    data_parallel_random_init: bool,
    ddp_config: DistributedDataParallelConfig,
    overlap_param_gather_with_optimizer_step: bool,
    use_megatron_fsdp: bool = False,
    use_torch_fsdp2: bool = False,
    *,
    pg_collection: ProcessGroupCollection,
) -> list[MegatronModule]:
    """Wrap model with Distributed Data Parallel (DDP) or Fully Sharded Data Parallel (FSDP).

    Args:
        model: List of model modules to wrap
        data_parallel_random_init: Whether to broadcast parameters from rank 0
        ddp_config: Configuration for distributed data parallel
        overlap_param_gather_with_optimizer_step: Whether to disable bucketing
            for overlapping parameter gathering with optimizer step
        use_megatron_fsdp: Whether to use Megatron FSDP.
        use_torch_fsdp2: Whether to use PyTorch FSDP v2 instead of DDP
        pg_collection: Model communication process groups.

    Returns:
        list[MegatronModule]: List of DDP/FSDP wrapped model modules
    """
    if use_megatron_fsdp:
        DP = FullyShardedDataParallel
        if use_torch_fsdp2:
            raise ValueError("Using use_megatron_fsdp and use_torch_fsdp2 at the same time is not supported.")
    elif use_torch_fsdp2:
        DP = TorchFullyShardedDataParallel
    else:
        DP = DistributedDataParallel

    # DDP initialization is required to be on a side-stream for the full-iteration CUDA graph.
    #  this side-stream may be nested if being called from within the get_model function, but it
    #  is here in case someone wants to use this directly outside of get_model.
    ddp_stream = torch.cuda.Stream()
    ddp_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(ddp_stream):
        model = [
            DP(
                config=get_model_config(model_chunk),
                ddp_config=ddp_config,
                module=model_chunk,
                # Turn off bucketing for model_chunk 2 onwards, since communication for these
                # model chunks is overlapped with compute anyway.
                disable_bucketing=(model_chunk_idx > 0) or overlap_param_gather_with_optimizer_step,
                pg_collection=pg_collection,
            )
            for (model_chunk_idx, model_chunk) in enumerate(model)
        ]
    # Critical: ensure side-stream work completes before touching params on default stream
    torch.cuda.current_stream().wait_stream(ddp_stream)

    # Broadcast params from data parallel src rank to other data parallel ranks.
    if data_parallel_random_init:
        for model_module in model:
            model_module.broadcast_params()

    return model


def build_virtual_pipeline_stages(
    build_model_func: Callable,
    pg_collection: ProcessGroupCollection,
    vp_size: int | None,
    model_type: ModelType = ModelType.encoder_or_decoder,
) -> list[MegatronModule]:
    """Build virtual pipeline stages if using virtual pipeline parallelism.

    Args:
        build_model_func: Function from ``ModelBuilder`` that builds a single stage of the model.
        pg_collection: Model communication process groups.
        vp_size: Virtual pipeline parallel size. If ``None`` or PP size is 1, a single stage is built.
        model_type: Deprecated flag, only used for backwards compatibility.

    Returns:
        List of model stages. Contains one entry per VP rank, or a single entry if VP is not enabled.
    """
    from megatron.core.pipeline_parallel.utils import (
        is_pp_first_stage,
        is_pp_last_stage,
        is_vp_first_stage,
        is_vp_last_stage,
    )

    pp_group = pg_collection.pp
    if pp_group.size() > 1 and vp_size is not None:
        # Create multiple model stages for virtual pipeline
        model_list = []
        for i in range(vp_size):
            pre_process = is_vp_first_stage(vp_stage=i, vp_size=vp_size) and is_pp_first_stage(pp_group)
            post_process = is_vp_last_stage(vp_stage=i, vp_size=vp_size) and is_pp_last_stage(pp_group)
            model = build_model_func(
                pg_collection,
                pre_process=pre_process,
                post_process=post_process,
                vp_stage=i,
            )
            model.model_type = model_type
            model_list.append(model)
    else:
        # Single stage, no VP
        pre_process = is_pp_first_stage(pp_group)
        post_process = is_pp_last_stage(pp_group)
        model = build_model_func(pg_collection, pre_process=pre_process, post_process=post_process)
        model.model_type = model_type
        model_list = [model]

    return model_list


def to_empty_if_meta_device(module: torch.nn.Module, *, device: torch.device, recurse=True):
    """Move tensors to device if not meta device; otherwise materialize with empty_like().

    Officially, torch suggests to_empty() for meta device materialization. Under the hood,
    torch.empty_like() is applied to all parameters or buffers (see _apply). This may
    accidently overwrite buffers with precomputed values during construction. Given the
    goal is to only materialize those tensors on meta device, this function checks the
    device first and only move the tensor to the destination if it is not on meta device.

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

    return module._apply(lambda t: _empty_like_if_meta(t, device=device), recurse=recurse)
