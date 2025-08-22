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

import logging
import types
from enum import IntEnum
from typing import Optional, Sequence, Type, TypeVar

import torch
from torch.distributed import DeviceMesh

from .megatron_fsdp import MegatronFSDP
from .uneven_dtensor import preprocess_state_dict_for_uneven_dtensor
from .utils import FSDPDistributedIndex, create_updated_function_signature

try:
    # Default to Megatron-LM FW.
    from megatron.core.distributed.distributed_data_parallel_config import (
        DistributedDataParallelConfig,
    )
except ImportError:
    # Megatron-LM is not installed, use Megatron-FSDP as a standalone module.
    from .distributed_data_parallel_config import DistributedDataParallelConfig


logger = logging.getLogger(__name__)


class ShardingStrategy(IntEnum):
    """
    IntEnum to track the abbreviated sharding strategy for Megatron-FSDP.

    - `0` or `no_shard` implies that your model is not sharded. Similar memory usage to `DDP`.
    - `1` or `optim` implies that your optimizer state is sharded. Similar to optimizer
        state sharding in `ZeRO-DP`.
    - `2` or `optim_grads` implies that your optimizer state and gradients are sharded.
        Similar to optimizer state and gradient sharding in `ZeRO-2`.
    - `3` or `optim_grads_params` implies that your optimizer state, gradients, and
        training parameters are sharded. Similar to optimizer state, gradient, and
        training parameter sharding in `ZeRO-3`.
    """

    NO_SHARD = 0
    OPTIM = 1
    OPTIM_GRADS = 2
    OPTIM_GRADS_PARAMS = 3


# Hints input-output consistency - if an optimizer is provided, it must be returned.
# If an optimizer is not provided, None must be returned.
MaybeOptimizer = TypeVar("MaybeOptimizer", bound=Optional[torch.optim.Optimizer])


def fully_shard(
    module: torch.nn.Module,
    optimizer: MaybeOptimizer = None,
    zero_dp_strategy: str | int = 3,
    fsdp_unit_modules: Optional[Sequence[Type[torch.nn.Module]] | Sequence[str]] = None,
    use_hybrid_fsdp: bool = False,
    outer_dp_sharding_strategy: str = "no_shard",
    device_mesh: Optional[DeviceMesh] = None,
    dp_shard_dim: Optional[str] = None,
    dp_inter_dim: Optional[str] = None,
    tp_dim: Optional[str] = None,
    hybrid_fsdp_group: Optional[torch.distributed.ProcessGroup] = None,
    device: Optional[torch.device] = None,
    init_model_with_meta_device: bool = False,
    grad_reduce_in_fp32: bool = False,
    preserve_fp32_weights: bool = True,
    overlap_grad_reduce: bool = True,
    overlap_param_gather: bool = True,
    sync_grads_each_step: bool = True,
    check_for_nan_in_grad: bool = True,
    average_in_collective: bool = False,
    disable_bucketing: bool = False,
    calculate_per_token_loss: bool = False,
    keep_fp8_transpose_cache: bool = False,
    nccl_ub: bool = False,
    fsdp_double_buffer: bool = False,
    preproc_state_dict_for_dcp_ckpt: bool = True,
) -> tuple[MegatronFSDP, MaybeOptimizer]:
    """
    Fully shard the model and the optimizer for Megatron-FSDP.

    Wraps the model as an Megatron-FSDP module, and modifies the optimizer to
    be compatible with the Megatron-FSDP training strategy.

    Args:
        module (torch.nn.Module):
            The PyTorch module fully-sharded and managed by Megatron-FSDP.

        optimizer (Optional[torch.optim.Optimizer]):
            (Distributed) optimizer for training the model, which is extended to automatically
            execute necessary Megatron-FSDP operations during the training loop. If not provided,
            the user is expected to utilize the Megatron-FSDP API to manually prepare the model for
            optimization. Defaults to None.

        zero_dp_strategy (str | int):
            Zero-redundancy sharding strategy for sharding data parallel parameters and gradients.
            - "no_shard" / 0: No optimizer, gradient, or parameter sharding. Similar
                memory usage to DDP.
            - "optim" / 1: Shards optimizer states (and main weights for mixed precision training),
                which is conceptually similar to optimizer state sharding in `ZeRO-DP`.
            - "optim_grads" / 2: Shards gradients and optimizer states, which is conceptually
                similar to "ZeRO-2".
            - "optim_grads_params" / 3: Shards parameters, gradients and optimizer states, which
                is conceptually similar to "ZeRO-3".
            Defaults to "optim_grads_params" / 3.

        fsdp_unit_modules (Optional[List[torch.nn.Module] | List[str]]):
            List of module classes or module class import paths to be treated as FSDP units,
            which are modules that do not have their parameters modified during their forward().
            This information is utilized by Megatron-FSDP to shard, gather, and overlap
            communications during the forward and backward pass of the module. Defaults to None.

        use_hybrid_fsdp (bool):
            Whether to use hybrid FSDP, i.e. a combination of replicate and
            sharded data parallel groups.
            Defaults to False.

        outer_dp_sharding_strategy (str):
            Sharding strategy for outer data parallel group in Hybrid Sharded Data Parallel (HSDP).
            Valid values are 'no_shard', 'optim', 'optim_grads', 'optim_grads_params'.
            This option is only effective when Hybrid FSDP is enabled.

        device_mesh (Optional[DeviceMesh]):
            Device mesh object defining the topology for distributed training. Defaults to None,
            in which case the {dp_shard,dp_replicate,tp}_group(s) are required to
            use Megatron-FSDP. If device_mesh is None, Megatron-FSDP will automatically use
            torch.distributed.group.WORLD for sharded data parallelism.

        dp_shard_dim (Optional[str]):
            Name of the data parallel sharding sub-mesh in the device_mesh. Supports
            a flattened DP-CP sub-mesh, in which case parameters, gradients, and
            optimizer state will be sharded across both DP and CP ranks.
            Defaults to None. Required to enable the core functionality of Megatron-FSDP.

        dp_inter_dim (Optional[str]):
            Name of the "inter-FSDP" sub-mesh in the device_mesh. When
            outer_dp_sharding_strategy="optim", this process group will be used
            as a DP replicate group for HSDP. Otherwise, it will be used
            as a DP sharding group, i.e. full-sharding with HSDP.
            Defaults to None. Required for HSDP, i.e. when use_hybrid_fsdp=True.

        tp_dim (Optional[str]):
            Name of the tensor parallel sub-mesh in the device_mesh, which is necessary
            for strided sharding between TP and FSDP (and fully-sharded HSDP) dimensions.
            Defaults to None. Required if TP is used in the model.

        hybrid_fsdp_group (Optional[torch.distributed.ProcessGroup]):
            Cumulative data parallel process group for hybrid FSDP that can be manufactured
            by flattening the inter-FSDP (dp_inter_dim) and FSDP (dp_shard_dim)
            process groups or sub-meshes.
            Defaults to None. Required for HSDP, i.e. when use_hybrid_fsdp=True.

        device (Optional[torch.device]):
            Target device for the sharded model. Used to migrate all parameters in the model
            to an expected device. If init_model_with_meta_device=True, this argument is ignored.
            Defaults to None.

        init_model_with_meta_device (bool):
            Utilized to initialize large models that do not fit on a single device, and requires
            implementing a custom Module.reset_parameters() or Module._reset_parameters() method.
            Defaults to False.

        grad_reduce_in_fp32 (bool):
            Whether to perform gradient reduction in FP32. Defaults to False.

        preserve_fp32_weights (bool):
            Whether to preserve FP32 optimization weights. Defaults to True.

        overlap_grad_reduce (bool):
            Whether to overlap gradient reduce-scatter (or all-reduce) with backward compute.
            Defaults to True.

        overlap_param_gather (bool):
            Whether to overlap parameter all-gather with forward and backward compute.
            Defaults to True.

        sync_grads_each_step (bool):
            Whether to synchronize and install optimizer gradients on each training step.
            When disabled, Megatron-FSDP will overlap reduce-scatter calls with subsequent compute,
            which improves performance and throughput when utilizing delayed optimization
            techniques such as gradient accumulation. Defaults to True for the fully_shard API.

        check_for_nan_in_grad (bool):
            Whether to check for NaN values in gradients. Defaults to True.

        average_in_collective (bool):
            Whether to average gradients in collective communication. Defaults to False.
            TODO: This is currently NOT supported!

        disable_bucketing (bool):
            Whether to disable gradient bucketing optimization, which permits more granular
            and precise communication of parameters and gradients. Defaults to False.

        calculate_per_token_loss (bool):
            Whether to calculate loss per token, which deactivates gradient scaling.
            Defaults to False.

        keep_fp8_transpose_cache (bool):
            Whether to keep the FP8 transpose cache when using a Megatron FSDP.
            Defaults to False.

        nccl_ub (bool):
            Whether to use NCCL UCC for communication. Defaults to False.

        fsdp_double_buffer (bool):
            Whether to use double buffer for FSDP. Defaults to False.

        preproc_state_dict_for_dcp_ckpt (bool):
            Whether to preprocess the state dict for DCP checkpointing. Defaults to True.

    Returns:
        torch.nn.Module: The wrapped Megatron-FSDP model configured for distributed training.
        torch.optim.Optimizer: The Megatron-FSDP-compliant optimizer for training the model.

    Note:
        This implementation uses NVIDIA's FSDP which includes optimizations specific
        to NVIDIA hardware and software stack.
    """

    # Parse zero_dp_strategy.
    # TODO(@cspades): Integrate this Enum into the MegatronFSDP class.
    if zero_dp_strategy == ShardingStrategy.NO_SHARD:
        zero_dp_strategy = "no_shard"
    elif zero_dp_strategy == ShardingStrategy.OPTIM:
        zero_dp_strategy = "optim"
    elif zero_dp_strategy == ShardingStrategy.OPTIM_GRADS:
        zero_dp_strategy = "optim_grads"
    elif zero_dp_strategy == ShardingStrategy.OPTIM_GRADS_PARAMS:
        zero_dp_strategy = "optim_grads_params"

    # Validate arguments.
    if zero_dp_strategy not in ["no_shard", "optim", "optim_grads", "optim_grads_params"]:
        raise ValueError(
            f"Invalid MegatronFSDP Sharding Strategy: {zero_dp_strategy}\n"
            f"Valid Sharding Strategies: {ShardingStrategy.NO_SHARD}, {ShardingStrategy.OPTIM}, "
            f"{ShardingStrategy.OPTIM_GRADS}, {ShardingStrategy.OPTIM_GRADS_PARAMS}, "
            f"no_shard, optim, optim_grads, optim_grads_params"
        )
    if init_model_with_meta_device and zero_dp_strategy != "optim_grads_params":
        raise ValueError(
            "Meta device initialization (init_model_with_meta_device=True) is only "
            "supported for the 'optim_grads_params' sharding strategy."
        )

    # DDP Config for Megatron FSDP.
    ddp_config = DistributedDataParallelConfig(
        check_for_nan_in_grad=check_for_nan_in_grad,
        data_parallel_sharding_strategy=zero_dp_strategy,
        grad_reduce_in_fp32=grad_reduce_in_fp32,
        preserve_fp32_weights=preserve_fp32_weights,
        overlap_grad_reduce=overlap_grad_reduce,
        overlap_param_gather=overlap_param_gather,
        average_in_collective=average_in_collective,
        keep_fp8_transpose_cache=keep_fp8_transpose_cache,  # pylint: disable=C0301
        nccl_ub=nccl_ub,
        fsdp_double_buffer=fsdp_double_buffer,
        outer_dp_sharding_strategy=outer_dp_sharding_strategy,
    )

    # Create FSDPDistributedIndex.
    dist_index = FSDPDistributedIndex(
        device_mesh=device_mesh,
        use_hybrid_fsdp=use_hybrid_fsdp,
        hsdp_outer_dp_shard=outer_dp_sharding_strategy != "no_shard",
        dp_shard_dim=dp_shard_dim,
        dp_inter_dim=dp_inter_dim,
        tp_dim=tp_dim,
        hybrid_fsdp_group=hybrid_fsdp_group,
    )

    # Wrap model in Megatron FSDP.
    model = MegatronFSDP(
        module=module,
        dist_index=dist_index,
        ddp_config=ddp_config,
        fsdp_unit_modules=fsdp_unit_modules,
        disable_bucketing=disable_bucketing,
        device=device,
        calculate_per_token_loss=calculate_per_token_loss,
        init_model_with_meta_device=init_model_with_meta_device,
        sync_grads_each_step=sync_grads_each_step,
    )

    # Extend optimizer methods to support Megatron-FSDP operations.
    if optimizer is not None:
        # Replace the optimizer module parameter references with the Megatron-FSDP-managed
        # parameters.
        optimizer.param_groups.clear()
        optimizer.state.clear()
        optimizer.add_param_group({"params": model.module.parameters()})

        # Save a reference to the optimizer.step() and optimizer.zero_grad() methods.
        optimizer_step_base_func = type(optimizer).step
        optimizer_zero_grad_base_func = type(optimizer).zero_grad

        # Define a new optimizer.step() method that distributes optimizer state and gradients,
        # waits for asynchronous gradient reduce-scatter work to be completed, and
        # updates model weights.
        def megatron_fsdp_optimizer_step(optimizer, *args, **kwargs):
            # Extract extended kwargs.
            sync_grad_before_optimizer_step = kwargs.pop(
                # If sync_grads_each_step is enabled, gradients are synchronized automatically
                # during the post-backward hook, so we do not need to synchronize them here.
                "sync_grad_before_optimizer_step",
                not sync_grads_each_step,
            )
            install_optimized_model_weights = kwargs.pop("install_optimized_model_weights", True)

            # Synchronize reduce-scatter and all-gather operations for all model gradients
            # and parameters, attach gradients to the optimizer state, and replace the raw
            # module parameters with Megatron-FSDP-managed optimizer parameters & states in
            # preparation for (distributed) optimization.
            if sync_grad_before_optimizer_step:
                model.finish_grad_sync()

            # Execute the base optimizer.step() on the model optimizer named parameters.
            optimizer_step_base_func(optimizer, *args, **kwargs)

            # Update the raw module training parameters with optimized values.
            if install_optimized_model_weights:
                model.install_optimized_model_weights()

        # Define a new optimizer.zero_grad() method that zeros the gradient in both
        # the optimizer as well as the Megatron-FSDP gradient buffer.
        def megatron_fsdp_optimizer_zero_grad(optimizer, *args, **kwargs):
            # Extract extended kwargs.
            zero_grad_buffer = kwargs.pop("zero_grad_buffer", True)

            # Execute the base optimizer.zero_grad() on the model optimizer named parameters.
            optimizer_zero_grad_base_func(optimizer, *args, **kwargs)

            # Zero out the gradient in the Megatron-FSDP gradient buffer.
            if zero_grad_buffer:
                model.zero_grad_buffer()

        # Override the optimizer.step() and optimizer.zero_grad() methods to support
        # Megatron-FSDP operations.
        megatron_fsdp_optimizer_step.__signature__ = create_updated_function_signature(
            optimizer_step_base_func,
            sync_grad_before_optimizer_step=True,
            install_optimized_model_weights=True,
        )
        optimizer.step = types.MethodType(megatron_fsdp_optimizer_step, optimizer)
        megatron_fsdp_optimizer_zero_grad.__signature__ = create_updated_function_signature(
            optimizer_zero_grad_base_func, zero_grad_buffer=True
        )
        optimizer.zero_grad = types.MethodType(megatron_fsdp_optimizer_zero_grad, optimizer)

    # Register a state dict post-hook to add Torch DCP metadata for writing checkpoints.
    if preproc_state_dict_for_dcp_ckpt and zero_dp_strategy != "no_shard":
        # Store a reference to the model state dict to avoid an infinite loop
        # when registering the state dict post-hook to the model.
        model_state_dict = model.state_dict()
        model._register_state_dict_hook(
            lambda *args, **kwargs: preprocess_state_dict_for_uneven_dtensor(model_state_dict)
        )

        if optimizer is not None:
            optimizer_state_dict = optimizer.state_dict()
            optimizer.register_state_dict_post_hook(
                lambda *args, **kwargs: preprocess_state_dict_for_uneven_dtensor(
                    optimizer_state_dict
                )
            )

    # FIXME(@shjwudp, @cspades): Checkpointing for `no_shard` is not supported yet.
    if zero_dp_strategy == "no_shard":
        logger.warning(
            "[Megatron-FSDP no_shard] Torch DCP checkpointing for no_shard is not supported yet."
        )

    # Return model and optimizer.
    return model, optimizer
