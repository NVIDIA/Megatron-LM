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
from typing import Callable, Optional, Sequence, Type

import torch
from torch.distributed import DeviceMesh
from torch.distributed.device_mesh import init_device_mesh

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


def experimental_api(func: Callable) -> Callable:
    """
    Mark a function or class as experimental API in Megatron CI/CD.

    TODO(@cspades): Copied from megatron.core.utils to avoid depending on MCore
    for Megatron-FSDP. Should remove when the API is no longer experimental.
    """
    func._experimental_api = True
    return func


@experimental_api
def fully_shard_model(
    module: torch.nn.Module,
    device_mesh: Optional[DeviceMesh] = None,
    dp_shard_dim: Optional[str] = None,
    dp_outer_dim: Optional[str] = None,
    tp_dim: Optional[str] = None,
    hybrid_fsdp_group: Optional[torch.distributed.ProcessGroup] = None,
    expt_device_mesh: Optional[DeviceMesh] = None,
    fsdp_unit_modules: Optional[Sequence[Type[torch.nn.Module]] | Sequence[str]] = None,
    zero_dp_strategy: str | int = 3,
    outer_dp_sharding_strategy: str | int = 0,
    device: Optional[torch.device] = None,
    init_model_with_meta_device: bool = False,
    grad_reduce_in_fp32: bool = False,
    preserve_fp32_weights: bool = True,
    overlap_grad_reduce: bool = True,
    overlap_param_gather: bool = True,
    sync_model_each_microbatch: bool = True,
    preproc_state_dict_for_dcp_ckpt: bool = True,
    check_for_nan_in_grad: bool = True,
    average_in_collective: bool = False,
    disable_bucketing: bool = False,
    calculate_per_token_loss: bool = False,
    keep_fp8_transpose_cache: bool = False,
    nccl_ub: bool = False,
    fsdp_double_buffer: bool = False,
    disable_symmetric_registration: bool = False,
) -> torch.nn.Module:
    """
    Fully-shard the model for Megatron-FSDP. This wraps the model in a MegatronFSDP
    class that schedules the sharding lifecycle of the model parameters and gradients
    during training and inference.

    The original `torch.nn.Module` can be accessed at `MegatronFSDP.module`.

    Args:
        module (torch.nn.Module):
            The PyTorch module fully-sharded and managed by Megatron-FSDP.

        device_mesh (Optional[DeviceMesh]):
            Device mesh object defining the topology for distributed training. If not provided,
            Megatron-FSDP will build a default FSDP DeviceMesh.

        dp_shard_dim (Optional[str]):
            Name of the data parallel sharding sub-mesh in the device_mesh. Supports
            a flattened DP-CP sub-mesh, in which case parameters, gradients, and
            optimizer state will be sharded across both DP and CP ranks.

        dp_outer_dim (Optional[str]):
            Name of the "outer" DP sub-mesh in the device_mesh for hybrid-sharding (HSDP),
            which supports "DP-Replicate" as well as optimizer state sharding (HFSDP).
            Defaults to None. Required for HSDP, which is enabled by this argument.

        tp_dim (Optional[str]):
            Name of the tensor parallel sub-mesh in the device_mesh, which is necessary
            for strided sharding between TP and FSDP (and fully-sharded HSDP) dimensions.
            Defaults to None. Required if TP is used in the model, or if TransformerEngine
            layers are utilized, as TE defaults to "TP=1".

        hybrid_fsdp_group (Optional[torch.distributed.ProcessGroup]):
            Cumulative data parallel process group for hybrid FSDP that can be manufactured
            by flattening the outer-FSDP (dp_outer_dim) and FSDP (dp_shard_dim) process groups
            or sub-meshes. Defaults to None. Required for HSDP, i.e. if dp_outer_dim is not None.

        expt_device_mesh (Optional[DeviceMesh]):
            Expert parallel device mesh object defining the topology for MoE distributed training.
            Utilizes the mesh dimension names specified by the *_dim arguments.

        fsdp_unit_modules (Optional[Sequence[Type[torch.nn.Module]] | Sequence[str]]):
            List of (sub-)module classes or (sub-)module class import paths that are "units",
            which are torch.nn.Module(s) that are sharded and scheduled by Megatron-FSDP.
            In particular, FSDP unit module parameters can be "safely" deallocated after
            the forward() or backward() pass without interfering with other computational
            operations that rely on those parameters in the complete PyTorch model.
            This information is utilized by Megatron-FSDP to optimally shard, gather, and
            overlap communications during the forward and backward pass of the module.
            Defaults to None, which is peak-memory-equivalent to DDP / "no_shard".

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

        outer_dp_sharding_strategy (str | int):
            Sharding strategy for outer data parallel group in Hybrid Sharded Data Parallel (HSDP).
            Shares the same semantics as zero_dp_strategy, but only 'no_shard' / 0 (DP Replication)
            and 'optim' / 1 (Optimizer State Hybrid Sharding) are supported, and 'optim' / 1 is only
            supported when zero_dp_strategy='optim_grads_params'.
            This option is only effective when HSDP is enabled, i.e. when dp_outer_dim is not None.
            Defaults to "no_shard" / 0, which replicates model parameters across the dp_outer group.

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

        sync_model_each_microbatch (bool): Whether to sync parameters and install gradients on
            each training step. When disabled, Megatron-FSDP will overlap reduce-scatter with
            subsequent compute and delay HSDP gather and reduce operations per optimization cycle,
            which improves performance and throughput when using delayed optimization strategies
            such as gradient accumulation. Defaults to True, can be modified before the model
            forward / backward pass via MegatronFSDP.set_model_auto_sync(bool) or controlled
            with the (no_)sync context managers or microbatch_count and is_last_microbatch.

        preproc_state_dict_for_dcp_ckpt (bool):
            Whether to preprocess the unevenly-sharded state dict for DCP checkpointing,
            for both the model and the optimizer.
            Defaults to True.

        check_for_nan_in_grad (bool):
            Whether to check for NaN values in gradients. Defaults to True.

        average_in_collective (bool):
            Whether to average gradients in collective communication. Defaults to False.

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

        disable_symmetric_registration (bool):
            Whether to disable symmetric (window) registration for NCCL UB registration.
            This option forces conventional (local) UB registration when nccl_ub is set.

    Returns:
        model (MegatronFSDP): The wrapped Megatron-FSDP model configured for FSDP.
    """
    # If no DeviceMesh or FSDP dimension is provided, then build an FSDP DeviceMesh.
    # Modify arguments into arguments necessary for vanilla FSDP.
    if device_mesh is None:
        if dp_shard_dim is None:
            dp_shard_dim = "fsdp"
        # Deactivate DP-Outer, which needs to be consistent with Expert DeviceMesh.
        dp_outer_dim = None
        hybrid_fsdp_group = None
        outer_dp_sharding_strategy = ShardingStrategy.NO_SHARD
        device_mesh = init_device_mesh(
            device_type="cuda",
            mesh_shape=(torch.distributed.get_world_size(),),
            mesh_dim_names=(dp_shard_dim,),
        )

    # Parse zero_dp_strategy and outer_dp_sharding_strategy.
    # TODO(@cspades): Integrate this Enum into MegatronFSDP.
    if zero_dp_strategy == ShardingStrategy.NO_SHARD:
        zero_dp_strategy = "no_shard"
    elif zero_dp_strategy == ShardingStrategy.OPTIM:
        zero_dp_strategy = "optim"
    elif zero_dp_strategy == ShardingStrategy.OPTIM_GRADS:
        zero_dp_strategy = "optim_grads"
    elif zero_dp_strategy == ShardingStrategy.OPTIM_GRADS_PARAMS:
        zero_dp_strategy = "optim_grads_params"
    elif zero_dp_strategy in ["no_shard", "optim", "optim_grads", "optim_grads_params"]:
        # Valid string sharding strategy.
        pass
    else:
        # Invalid sharding strategy.
        raise ValueError(
            f"Invalid FSDP / Inner DP Sharding Strategy: {zero_dp_strategy}\n"
            f"Valid Sharding Strategies: {ShardingStrategy.NO_SHARD}, "
            f"{ShardingStrategy.OPTIM}, {ShardingStrategy.OPTIM_GRADS}, "
            f"{ShardingStrategy.OPTIM_GRADS_PARAMS}, "
            "no_shard, optim, optim_grads, optim_grads_params"
        )
    if outer_dp_sharding_strategy == ShardingStrategy.NO_SHARD:
        outer_dp_sharding_strategy = "no_shard"
    elif outer_dp_sharding_strategy == ShardingStrategy.OPTIM:
        outer_dp_sharding_strategy = "optim"
    elif outer_dp_sharding_strategy in ["no_shard", "optim"]:
        # Valid string sharding strategy.
        pass
    else:
        # Invalid sharding strategy.
        raise ValueError(
            f"Invalid Hybrid DP-Outer Sharding Strategy: {outer_dp_sharding_strategy}\n"
            f"Valid Sharding Strategies: {ShardingStrategy.NO_SHARD}, "
            f"{ShardingStrategy.OPTIM}, no_shard, optim"
        )

    # Validate more arguments.
    _outer_fsdp_sharding = outer_dp_sharding_strategy == "optim"
    if _outer_fsdp_sharding and zero_dp_strategy != "optim_grads_params":
        # If sharding on outer DP using HSDP, then we must use HSDP buffers and
        # we must be fully-sharding on inner DP. HSDP is an extension of FSDP.
        # FIXME(@shjwudp, @cspades): This is an unexpected lack of support.
        raise ValueError(
            f"Sharding with Hybrid (Fully) Sharded Data Parallel (HSDP) requires "
            "zero_dp_strategy to use FSDP ('optim_grads_params', 3), because "
            "outer sharding is dependent on inner sharding."
        )
    if (dp_outer_dim is None) ^ (hybrid_fsdp_group is None):
        # XOR - HSDP requires both or neither of dp_outer_dim and hybrid_fsdp_group
        # to be specified, so if XOR then raise an error.
        raise ValueError(
            f"dp_outer_dim={dp_outer_dim} and hybrid_fsdp_group={hybrid_fsdp_group} must be "
            "specified together for Hybrid FSDP (HSDP), or both set to None (for FSDP)."
        )
    if init_model_with_meta_device and zero_dp_strategy == "no_shard":
        raise ValueError(
            "Meta device initialization (init_model_with_meta_device=True) is not "
            "supported or necessary for the 'no_shard' / 0 sharding strategy."
        )

    # DDP Config for Megatron FSDP.
    ddp_config = DistributedDataParallelConfig(
        data_parallel_sharding_strategy=zero_dp_strategy,
        outer_dp_sharding_strategy=outer_dp_sharding_strategy,
        grad_reduce_in_fp32=grad_reduce_in_fp32,
        preserve_fp32_weights=preserve_fp32_weights,
        overlap_grad_reduce=overlap_grad_reduce,
        overlap_param_gather=overlap_param_gather,
        average_in_collective=average_in_collective,
        keep_fp8_transpose_cache=keep_fp8_transpose_cache,  # pylint: disable=C0301
        nccl_ub=nccl_ub,
        fsdp_double_buffer=fsdp_double_buffer or nccl_ub,
        disable_symmetric_registration=disable_symmetric_registration,
        check_for_nan_in_grad=check_for_nan_in_grad,
    )

    # Create FSDPDistributedIndex.
    dist_index = FSDPDistributedIndex(
        device_mesh=device_mesh,
        # Always required for Megatron-FSDP.
        dp_shard_dim=dp_shard_dim,
        # Only required for HSDP.
        dp_outer_dim=dp_outer_dim,
        # TODO(@cspades): TP sub-mesh should be optional if not using TP, but is
        # required for Megatron, TransformerEngine (default TP=1), and strided
        # sharding when using DTensor-based TP.
        tp_dim=tp_dim,
        # Only required for HSDP.
        hybrid_fsdp_group=hybrid_fsdp_group,
        # Access to flattened DP rank assignments for HSDP.
        hsdp_outer_dp_shard=_outer_fsdp_sharding,
        # Only required for Megatron-FSDP + EP.
        expt_device_mesh=expt_device_mesh,
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
        sync_model_each_microbatch=sync_model_each_microbatch,
    )

    # Register a state dict post-hook to add Torch DCP metadata for writing checkpoints.
    if preproc_state_dict_for_dcp_ckpt and zero_dp_strategy != "no_shard":

        def remove_te_extra_state(state_dict):
            # Megatron-FSDP does not support FP8 extra state checkpointing in TE.
            extra_state_keys = [k for k in state_dict.keys() if k.endswith("_extra_state")]
            for key in extra_state_keys:
                state_dict.pop(key)

        def preprocess_dcp_and_te_extra_state(state_dict):
            # Preprocess the state dict for uneven DTensor checkpointing.
            remove_te_extra_state(state_dict)
            return preprocess_state_dict_for_uneven_dtensor(state_dict)

        model._register_state_dict_hook(
            lambda module, state_dict, prefix, local_metadata: preprocess_dcp_and_te_extra_state(
                state_dict
            )
        )

    # Return the wrapped Megatron-FSDP model.
    return model


@experimental_api
def fully_shard_optimizer(
    optimizer: torch.optim.Optimizer, preproc_state_dict_for_dcp_ckpt: bool = True
) -> torch.optim.Optimizer:
    """
    Fully shard the optimizer for Megatron-FSDP. This is an in-place operation on the optimizer
    instance, which modifies the optimizer to call methods exposed by the MegatronFSDP model API.

    The optimizer should be registered on the MegatronFSDP distributed model parameters:
    ```
        # Fully-shard the model.
        mfsdp_model = fully_shard_model(model, ...)

        # Register the fully-sharded parameters with the optimizer.
        # Use MegatronFSDP._replace_param_with_distributed_if_needed()
        # to swap to the distributed optimizer state parameters.
        optimizer = fully_shard_optimizer(Adam(params=mfsdp_model.parameters()))
    ```

    Args:
        optimizer (torch.optim.Optimizer):
            (Distributed) optimizer for training the model, which is extended to automatically
            execute necessary Megatron-FSDP operations during the training loop.

        preproc_state_dict_for_dcp_ckpt (bool):
            Whether to preprocess the state dict for DCP checkpointing. Defaults to True.

    Returns:
        optimizer (torch.optim.Optimizer): The in-place modified optimizer for Megatron-FSDP.
    """
    # Extract a reference to MegatronFSDP from the first registered Parameter.
    if not optimizer.param_groups:
        raise ValueError(
            f"[MegatronFSDP fully_shard_optimizer()] Provided optimizer doesn't "
            f"have any registered parameters: {optimizer}"
        )
    first_mfsdp_param = optimizer.param_groups[0][next(iter(optimizer.param_groups[0]))][0]
    if not getattr(first_mfsdp_param, "_megatron_fsdp_model", None):
        raise ValueError(
            f"[MegatronFSDP fully_shard_optimizer()] Could not retrieve a reference to "
            f"MegatronFSDP from the first registered Parameter: {first_mfsdp_param} \n"
            "Make sure the optimizer is registered to the MegatronFSDP distributed "
            "parameters via MegatronFSDP._replace_param_with_distributed_if_needed() "
            "before initializing the optimizer on the MegatronFSDP model. "
        )
    mfsdp_model = first_mfsdp_param._megatron_fsdp_model

    # Save a reference to the optimizer.step() and optimizer.zero_grad() methods.
    optimizer_step_base_func = type(optimizer).step
    optimizer_zero_grad_base_func = type(optimizer).zero_grad

    # Define a new optimizer.step() method that distributes optimizer state and gradients,
    # waits for asynchronous gradient reduce-scatter work to be completed, and updates
    # model weights. These options can be turned off via arguments in optimizer.step().
    def megatron_fsdp_optimizer_step(optimizer, *args, **kwargs):
        # Extract extended kwargs.
        sync_grad_before_optimizer_step = kwargs.pop("sync_grad_before_optimizer_step", True)
        install_optimized_model_weights = kwargs.pop("install_optimized_model_weights", True)

        # Synchronize reduce-scatter and all-gather operations for all model gradients
        # and parameters, attach gradients to the optimizer state, and replace the raw
        # module parameters with Megatron-FSDP-managed optimizer parameters & states in
        # preparation for (distributed) optimization.
        # NOTE: Only necessary if MegatronFSDP.model_auto_sync = False, in which case
        # gradient synchronization is not automatically handled by MegatronFSDP during
        # the post-backward hook and we need to synchronize manually.
        if sync_grad_before_optimizer_step and not mfsdp_model.model_auto_sync:
            mfsdp_model.finish_grad_sync()

        # Execute the base optimizer.step() on the model optimizer named parameters.
        optimizer_step_base_func(optimizer, *args, **kwargs)

        # Update the raw module training parameters with optimized values.
        if install_optimized_model_weights:
            mfsdp_model.install_optimized_model_weights()

    # Define a new optimizer.zero_grad() method that zeros the gradient in both
    # the optimizer as well as the Megatron-FSDP gradient buffer. These options
    # can be turned off via arguments in optimizer.zero_grad().
    def megatron_fsdp_optimizer_zero_grad(optimizer, *args, **kwargs):
        # Extract extended kwargs.
        zero_grad_buffer = kwargs.pop("zero_grad_buffer", True)

        # Execute the base optimizer.zero_grad() on the model optimizer named parameters.
        optimizer_zero_grad_base_func(optimizer, *args, **kwargs)

        # Zero out the gradient in the Megatron-FSDP gradient buffer.
        if zero_grad_buffer:
            mfsdp_model.zero_grad_buffer()

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

    if preproc_state_dict_for_dcp_ckpt:
        # Requires a non-empty, DTensor-type optimizer state dictionary.
        # This is satisfied naturally by calling optimizer.state_dict()
        # after the first optimizer.step() initializes the state to match
        # the Megatron-FSDP
        optimizer_state_dict = optimizer.state_dict()
        optimizer.register_state_dict_post_hook(
            lambda *args, **kwargs: preprocess_state_dict_for_uneven_dtensor(optimizer_state_dict)
        )

    # Return the in-place modified optimizer.
    return optimizer


def fully_shard(
    module: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device_mesh: Optional[DeviceMesh] = None,
    dp_shard_dim: Optional[str] = None,
    dp_outer_dim: Optional[str] = None,
    tp_dim: Optional[str] = None,
    hybrid_fsdp_group: Optional[torch.distributed.ProcessGroup] = None,
    expt_device_mesh: Optional[DeviceMesh] = None,
    fsdp_unit_modules: Optional[Sequence[Type[torch.nn.Module]] | Sequence[str]] = None,
    zero_dp_strategy: str | int = 3,
    outer_dp_sharding_strategy: str | int = 0,
    device: Optional[torch.device] = None,
    init_model_with_meta_device: bool = False,
    grad_reduce_in_fp32: bool = False,
    preserve_fp32_weights: bool = True,
    overlap_grad_reduce: bool = True,
    overlap_param_gather: bool = True,
    sync_model_each_microbatch: bool = True,
    preproc_state_dict_for_dcp_ckpt: bool = True,
    check_for_nan_in_grad: bool = True,
    average_in_collective: bool = False,
    disable_bucketing: bool = False,
    calculate_per_token_loss: bool = False,
    keep_fp8_transpose_cache: bool = False,
    nccl_ub: bool = False,
    fsdp_double_buffer: bool = False,
    disable_symmetric_registration: bool = False,
) -> tuple[MegatronFSDP, torch.optim.Optimizer]:
    """
    Fully shard the model and the optimizer for Megatron-FSDP.

    Wraps the model as an Megatron-FSDP module, and modifies the optimizer to
    be compatible with the Megatron-FSDP training strategy.

    Args:
        Union of arguments from fully_shard_model and fully_shard_optimizer.

    Returns:
        torch.nn.Module: The wrapped Megatron-FSDP model configured for distributed training.
        torch.optim.Optimizer: The Megatron-FSDP-compliant optimizer for training the model.

    Note:
        This implementation uses NVIDIA's FSDP which includes optimizations specific
        to NVIDIA hardware and software stack.
    """

    model = fully_shard_model(
        module=module,
        device_mesh=device_mesh,
        dp_shard_dim=dp_shard_dim,
        dp_outer_dim=dp_outer_dim,
        tp_dim=tp_dim,
        hybrid_fsdp_group=hybrid_fsdp_group,
        expt_device_mesh=expt_device_mesh,
        fsdp_unit_modules=fsdp_unit_modules,
        zero_dp_strategy=zero_dp_strategy,
        outer_dp_sharding_strategy=outer_dp_sharding_strategy,
        device=device,
        init_model_with_meta_device=init_model_with_meta_device,
        grad_reduce_in_fp32=grad_reduce_in_fp32,
        preserve_fp32_weights=preserve_fp32_weights,
        overlap_grad_reduce=overlap_grad_reduce,
        overlap_param_gather=overlap_param_gather,
        sync_model_each_microbatch=sync_model_each_microbatch,
        preproc_state_dict_for_dcp_ckpt=preproc_state_dict_for_dcp_ckpt,
        check_for_nan_in_grad=check_for_nan_in_grad,
        average_in_collective=average_in_collective,
        disable_bucketing=disable_bucketing,
        calculate_per_token_loss=calculate_per_token_loss,
        keep_fp8_transpose_cache=keep_fp8_transpose_cache,
        nccl_ub=nccl_ub,
        fsdp_double_buffer=fsdp_double_buffer,
        disable_symmetric_registration=disable_symmetric_registration,
    )

    # Extend optimizer methods to support Megatron-FSDP operations.
    # Replace the optimizer module parameter references with
    # Megatron-FSDP-managed distributed parameters.
    model._replace_param_with_distributed_if_needed()
    optimizer.param_groups.clear()
    optimizer.state.clear()
    optimizer.add_param_group({"params": model.parameters()})
    fully_shard_optimizer(
        optimizer, preproc_state_dict_for_dcp_ckpt=preproc_state_dict_for_dcp_ckpt
    )

    # Return model and optimizer.
    return model, optimizer
