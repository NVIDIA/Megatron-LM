# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""General utilities."""
import json
import os
import sys
import warnings
from contextlib import contextmanager
from datetime import datetime
from collections import defaultdict
from typing import Optional

import torch

from megatron.core.msc_utils import open_file
from megatron.core._rank_utils import safe_get_rank as _safe_get_rank
from megatron.core.dist_checkpointing.strategies.nvrx import has_nvrx_async_support

from megatron.core._slurm_utils import resolve_slurm_local_rank

try:
    from transformer_engine.pytorch.optimizers import multi_tensor_applier, multi_tensor_l2norm
except ImportError:
    try:
        from amp_C import multi_tensor_l2norm
        from apex.multi_tensor_apply import multi_tensor_applier
    except ImportError:
        warnings.warn(
            f'Transformer Engine and Apex are not installed. '
            'Falling back to local implementations of '
            'multi_tensor_applier and multi_tensor_l2norm'
        )

        from megatron.core.utils import (
            local_multi_tensor_l2_norm as multi_tensor_l2norm,
            local_multi_tensor_applier as multi_tensor_applier,
        )

from megatron.training import get_args, get_timers, get_adlr_autoresume
from megatron.core import mpu
from megatron.core.datasets.utils import get_blend_from_list
from megatron.core.tensor_parallel import param_is_not_tensor_parallel_duplicate
from megatron.core.utils import (
    get_data_parallel_group_if_dtensor,
    to_local_if_dtensor,
    unwrap_model,
)

from megatron.core.transformer.module import param_is_not_shared



def _compute_norm_2(params_list):
    """Compute squared L2 norm of a list of tensors. Returns a CUDA scalar."""
    if len(params_list) > 0:
        dummy_overflow_buf = torch.tensor([0], dtype=torch.int, device='cuda')
        norm, _ = multi_tensor_applier(
            multi_tensor_l2norm, dummy_overflow_buf, [params_list], False,
        )
        return norm * norm
    return torch.zeros((1,), dtype=torch.float32, device='cuda')


def _get_param_data(param, force_create_fp32_copy, bf16):
    """Extract the appropriate data tensor from a param for norm computation.

    Returns (data_tensor, is_sharded) where is_sharded indicates the param has
    a sharded main_param from the distributed optimizer.
    """
    if bf16:
        if not force_create_fp32_copy and hasattr(param, 'main_param'):
            if getattr(param, 'main_param_sharded', False):
                if param.main_param is not None:
                    return param.main_param, True
                return None, True
            return param.main_param, False
        return param.data.float(), False
    return param.data, False


def calc_params_l2_norm(model, force_create_fp32_copy=False):
    """Calculate l2 norm of parameters"""
    args = get_args()
    if not isinstance(model, list):
        model = [model]

    if getattr(args, 'use_megatron_fsdp', False):
        # All Megatron FSDP parameters are expected to be PyTorch DTensor.
        # params_data is a dict of device_mesh -> list of local tensors.
        params = []
        for model_chunk in model:
            model_chunk.stop_communication()
            for name, param in model_chunk.named_parameters():
                if not hasattr(param, "_local_tensor"):
                    raise RuntimeError(
                        f"Megatron FSDP requires parameters are PyTorch DTensor. "
                        f"Parameter {name} is not a DTensor."
                    )
                params.append(param)

        return calc_dtensor_params_l2_norm(params)

    # 8 buckets: 4 categories × (non-sharded, sharded optimizer main_param).
    # Each category needs different reduction groups.
    params_data = []                # Dense, non-sharded
    sharded_params_data = []        # Dense, sharded → reduce over dp_cp
    gtp_params_data = []            # GTP, non-sharded
    gtp_sharded_params_data = []    # GTP, sharded → reduce over dp_cp_with_gtp
    moe_params_data = []            # MoE, non-sharded
    moe_sharded_params_data = []    # MoE, sharded → reduce over expert_dp
    moe_gtp_params_data = []        # MoE-GTP, non-sharded
    moe_gtp_sharded_params_data = []  # MoE-GTP, sharded → reduce over expert_dp_with_gtp

    gtp_rank = mpu.get_generalized_tensor_parallel_remat_rank()
    egtp_rank = mpu.get_expert_generalized_tensor_parallel_remat_rank()

    for model_chunk in model:
        for param in model_chunk.parameters():
            is_gtp = getattr(param, 'is_gtp', False)

            # Filter TP duplicates. GTP params are always unique across TP ranks
            # so skip this check for them.
            if not is_gtp and not param_is_not_tensor_parallel_duplicate(param):
                continue
            is_expert = not getattr(param, 'allreduce', True)

            # Filter GTP duplicates: non-GTP params are replicated across GTP ranks.
            if is_expert:
                if not is_gtp and egtp_rank != 0:
                    continue
            else:
                if not is_gtp and gtp_rank != 0:
                    continue

            # Route to the correct bucket.
            if is_expert:
                assert param_is_not_shared(param)
                param = to_local_if_dtensor(param)
                data, is_sharded = _get_param_data(param, force_create_fp32_copy, args.bf16)
                if data is None:
                    continue
                if is_gtp:
                    (moe_gtp_sharded_params_data if is_sharded else moe_gtp_params_data).append(data)
                else:
                    (moe_sharded_params_data if is_sharded else moe_params_data).append(data)
            else:
                if param_is_not_shared(param):
                    param = to_local_if_dtensor(param)
                    data, is_sharded = _get_param_data(param, force_create_fp32_copy, args.bf16)
                    if data is None:
                        continue
                    if is_gtp:
                        (gtp_sharded_params_data if is_sharded else gtp_params_data).append(data)
                    else:
                        (sharded_params_data if is_sharded else params_data).append(data)

    # --- Compute local norm^2 for each bucket ---
    params_norm_2 = _compute_norm_2(params_data)
    sharded_norm_2 = _compute_norm_2(sharded_params_data)
    gtp_norm_2 = _compute_norm_2(gtp_params_data)
    gtp_sharded_norm_2 = _compute_norm_2(gtp_sharded_params_data)
    moe_norm_2 = _compute_norm_2(moe_params_data)
    moe_sharded_norm_2 = _compute_norm_2(moe_sharded_params_data)
    moe_gtp_norm_2 = _compute_norm_2(moe_gtp_params_data)
    moe_gtp_sharded_norm_2 = _compute_norm_2(moe_gtp_sharded_params_data)

    def _sum_reduce(tensor, group):
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM, group=group)

    # --- Sharded optimizer DP reductions (each category uses its own group) ---
    # Reduce over the gtp-EXCLUDED replicate group: the model-parallel reduce below already
    # spans the gtp axis, so a gtp-inclusive group here would over-count by gtp. No-op for
    # non-GTP runs (the with_gtp group aliases the regular DP group).
    _sum_reduce(sharded_norm_2, mpu.get_data_parallel_group(with_context_parallel=True, with_gtp=True))
    _sum_reduce(gtp_sharded_norm_2, mpu.get_data_parallel_group(with_context_parallel=True, with_gtp=True))
    _sum_reduce(moe_sharded_norm_2, mpu.get_expert_data_parallel_group())
    _sum_reduce(moe_gtp_sharded_norm_2, mpu.get_expert_data_parallel_group(with_gtp=True))

    # --- Combine dense + GTP norms ---
    # model_parallel group = TP×PP×GTP, so GTP reduction is implicit.
    norm_2 = params_norm_2 + sharded_norm_2 + gtp_norm_2 + gtp_sharded_norm_2

    # --- Combine MoE + MoE-GTP norms ---
    # expert_model_parallel = TP×EP×PP (does NOT include EGTP), so we need
    # an explicit EGTP reduction for MoE-GTP before the model-parallel reduce.
    moe_gtp_combined_norm_2 = moe_gtp_norm_2 + moe_gtp_sharded_norm_2
    _sum_reduce(moe_gtp_combined_norm_2, mpu.get_expert_generalized_tensor_parallel_remat_group())
    moe_total_norm_2 = moe_norm_2 + moe_sharded_norm_2 + moe_gtp_combined_norm_2

    # --- Model-parallel reductions ---
    dense_reduce_group = mpu.get_model_parallel_group()
    expert_reduce_group = mpu.get_expert_tensor_model_pipeline_parallel_group()
    ranks_in_dense_reduce_group = torch.distributed.get_process_group_ranks(dense_reduce_group)
    ranks_in_expert_reduce_group = torch.distributed.get_process_group_ranks(expert_reduce_group)

    if ranks_in_dense_reduce_group == ranks_in_expert_reduce_group:
        norm_2 += moe_total_norm_2
        _sum_reduce(norm_2, dense_reduce_group)
    else:
        _sum_reduce(norm_2, dense_reduce_group)
        _sum_reduce(moe_total_norm_2, expert_reduce_group)
        norm_2 += moe_total_norm_2

    return norm_2.item() ** 0.5


def calc_dtensor_params_l2_norm(params):
    """Calculate l2 norm of DTensor parameters."""
    params_data = defaultdict(list)
    for param in params:
        params_data[param._spec].append(param._local_tensor)

    total_norm_2 = torch.zeros((1,), dtype=torch.float32, device='cuda')
    dummy_overflow_buf = torch.zeros((1,), dtype=torch.int, device='cuda')
    for dtensor_spec, local_tensors in params_data.items():
        local_tensors = [t for t in local_tensors if t.numel() > 0]
        if len(local_tensors) == 0:
            norm = torch.zeros((1,), dtype=torch.float32, device='cuda')
        else:
            norm, _ = multi_tensor_applier(
                multi_tensor_l2norm, dummy_overflow_buf, [local_tensors], False  # no per-parameter norm.
            )
        norm_2 = norm * norm
        for pg, placement in zip(
            dtensor_spec.device_mesh.get_all_groups(),
            dtensor_spec.placements,
        ):
            if placement.is_shard():
                torch.distributed.all_reduce(
                    norm_2, op=torch.distributed.ReduceOp.SUM, group=pg
                )
            elif placement.is_replicate():
                # Replicated parameters are already summed across all ranks.
                pass
            else:
                raise RuntimeError(
                    f"Unsupported placement {placement} for Megatron FSDP."
                )
        total_norm_2 += norm_2

    return total_norm_2.item() ** 0.5


def average_losses_across_data_parallel_group(
    losses, group: Optional[torch.distributed.ProcessGroup] = None
):
    """Reduce a tensor of losses across all GPUs.

    group: data-parallel process group; defaults to mpu.get_data_parallel_group().
    """
    if group is None:
        group = mpu.get_data_parallel_group()
    averaged_losses = torch.cat([loss.clone().detach().view(1) for loss in losses])
    torch.distributed.all_reduce(averaged_losses, group=group)
    averaged_losses = averaged_losses / group.size()

    return averaged_losses


def reduce_max_stat_across_model_parallel_group(
    stat: float, group: Optional[torch.distributed.ProcessGroup] = None
) -> float | None:
    """
    Ranks without an optimizer will have no grad_norm or num_zeros_in_grad stats.
    We need to ensure the logging and writer rank has those values.
    This function reduces a stat tensor across the model parallel group.

    We use an all_reduce max since the values have already been summed across optimizer ranks where possible

    group: model-parallel process group; defaults to mpu.get_model_parallel_group().
    """
    if group is None:
        group = mpu.get_model_parallel_group()
    if stat is None:
        stat = -1.0
    stat = torch.tensor([stat], dtype=torch.float32, device=torch.cuda.current_device())
    torch.distributed.all_reduce(stat, op=torch.distributed.ReduceOp.MAX, group=group)
    if stat.item() == -1.0:
        # No rank has a valid stat, so return None to indicate that it is None across all ranks.
        return None
    else:
        return stat.item()


def logical_and_across_model_parallel_group(
    input: bool, group: Optional[torch.distributed.ProcessGroup] = None
) -> bool:
    """
    This function gathers a bool value across the model parallel group

    group: model-parallel process group; defaults to mpu.get_model_parallel_group().
    """
    if group is None:
        group = mpu.get_model_parallel_group()
    if input is True:
        input = 1
    else:
        input = 0
    input = torch.tensor([input], dtype=torch.int, device=torch.cuda.current_device())
    torch.distributed.all_reduce(input, op=torch.distributed.ReduceOp.MIN, group=group)
    return bool(input.item())


def report_memory(name):
    """Simple GPU memory report."""
    args = get_args()
    mega_bytes = 1024.0 * 1024.0
    string = name + ' memory (MB)'
    string += f" | allocated: {torch.cuda.memory_allocated() / mega_bytes:.2f}"
    string += f" | max allocated: {torch.cuda.max_memory_allocated() / mega_bytes:.2f}"
    string += f" | reserved: {torch.cuda.memory_reserved() / mega_bytes:.2f}"
    string += f" | max reserved: {torch.cuda.max_memory_reserved() / mega_bytes:.2f}"
    if args.log_device_memory_used:
        string += f" | total device memory used: {torch.cuda.device_memory_used() / mega_bytes:.2f}"
    if mpu.get_data_parallel_rank() == 0:
        print("[Rank {}] {}".format(torch.distributed.get_rank(), string), flush=True)


def print_params_min_max_norm(optimizer, iteration):
    """Print min, max, and norm of all parameters."""
    index = 0
    rank = torch.distributed.get_rank()
    string = 'iteration, rank, index, tensor-model-parallel, min, max, norm\n'
    optimizer_ = optimizer.optimizer
    for param_group in optimizer_.param_groups:
        for param in param_group['params']:
            index += 1
            min_ = param.data.min()
            max_ = param.data.max()
            norm = torch.linalg.norm(param.data)
            string += '{:7d}, {:4d}, {:4d}, {:2d}, '.format(
                iteration, rank, index, int(param.tensor_model_parallel)
            )
            string += '{:.6E}, {:.6E}, {:.6E}\n'.format(min_, max_, norm)
    print(string, flush=True)


def check_adlr_autoresume_termination(iteration, model, optimizer, opt_param_scheduler):
    """Check for autoresume signal and exit if it is received."""
    from megatron.training.checkpointing import save_checkpoint

    args = get_args()
    autoresume = get_adlr_autoresume()
    # Add barrier to ensure consistnecy.
    torch.distributed.barrier()
    if autoresume.termination_requested():
        if args.save:
            save_checkpoint(iteration, model, optimizer, opt_param_scheduler)
        print_rank_0(">>> autoresume termination request found!")
        if torch.distributed.get_rank() == 0:
            autoresume.request_resume()
        print_rank_0(">>> training terminated. Returning")
        sys.exit(0)


def get_ltor_masks_and_position_ids(data,
                                    eod_token,
                                    pad_token,
                                    reset_position_ids,
                                    reset_attention_mask,
                                    eod_mask_loss,
                                    pad_mask_loss):
    """Build masks and position id for left to right model."""

    # Extract batch size and sequence length.
    micro_batch_size, seq_length = data.size()

    # Attention mask (lower triangular).
    if reset_attention_mask:
        att_mask_batch = micro_batch_size
    else:
        att_mask_batch = 1
    attention_mask = torch.tril(
        torch.ones((att_mask_batch, seq_length, seq_length), device=data.device)
    ).view(att_mask_batch, 1, seq_length, seq_length)

    # Loss mask.
    loss_mask = torch.ones(data.size(), dtype=torch.float, device=data.device)
    if eod_mask_loss:
        loss_mask[data == eod_token] = 0.0
    if pad_mask_loss:
        loss_mask[data == pad_token] = 0.0

    # Position ids.
    position_ids = torch.arange(seq_length, dtype=torch.long, device=data.device)
    position_ids = position_ids.unsqueeze(0).expand_as(data)
    # We need to clone as the ids will be modifed based on batch index.
    if reset_position_ids:
        position_ids = position_ids.clone()

    if reset_position_ids or reset_attention_mask:
        # Loop through the batches:
        for b in range(micro_batch_size):

            # Find indecies where EOD token is.
            eod_index = position_ids[b, data[b] == eod_token] & position_ids[b, data[b] == pad_token]
            # Detach indecies from positions if going to modify positions.
            if reset_position_ids:
                eod_index = eod_index.clone()

            # Loop through EOD indecies:
            prev_index = 0
            for j in range(eod_index.size()[0]):
                i = eod_index[j]
                # Mask attention loss.
                if reset_attention_mask:
                    attention_mask[b, 0, (i + 1) :, : (i + 1)] = 0
                # Reset positions.
                if reset_position_ids:
                    position_ids[b, (i + 1) :] -= i + 1 - prev_index
                    prev_index = i + 1

    # Convert attention mask to binary:
    attention_mask = attention_mask < 0.5

    return attention_mask, loss_mask, position_ids


def print_rank_0(message, rank=None):
    """If distributed is initialized or rank is specified, print only on rank 0."""
    if rank is not None:
        if rank == 0:
            print(message, flush=True)
    else:
        if _safe_get_rank() == 0:
            print(message, flush=True)


def warn_rank_0(message, rank=None):
    """If distributed is initialized or rank is specified, warn only on rank 0."""
    if rank is not None:
        if rank == 0:
            warnings.warn(message)
    else:
        if _safe_get_rank() == 0:
            warnings.warn(message)


def is_rank0():
    """Returns true if called in the rank0, false otherwise."""
    return _safe_get_rank() == 0


def is_last_rank():
    """Returns true if called on last rank, false otherwise."""
    assert torch.distributed.is_initialized()
    return _safe_get_rank() == (torch.distributed.get_world_size() - 1)


def print_rank_last(message):
    """If distributed is initialized, print only on last rank."""
    if torch.distributed.is_initialized() and torch.distributed.get_backend() != 'fake':
        if is_last_rank():
            print(message, flush=True)
    else:
        print(message, flush=True)


def is_hybrid_model(args):
    """Returns True if the model is a hybrid Mamba-Transformer model."""
    return args.hybrid_layer_pattern is not None


def is_first_or_last_pipeline_stage(vp_stage):
    """Return True if on first or last pipeline stage, taking into account virtual
    pipeline parallelism."""
    ignore_virtual = True
    if vp_stage is not None:
        ignore_virtual = False
    return (
        mpu.is_pipeline_first_stage(ignore_virtual=ignore_virtual, vp_stage=vp_stage)
        or mpu.is_pipeline_last_stage(ignore_virtual=ignore_virtual, vp_stage=vp_stage)
    )


def get_device_arch_version():
    """Returns GPU arch version (8: Ampere, 9: Hopper, 10: Blackwell, ...)"""
    return torch.cuda.get_device_properties(torch.device("cuda:0")).major


def get_blend_and_blend_per_split(args):
    """Get blend and blend_per_split from passed-in arguments."""
    use_data_path = args.data_path is not None or args.data_args_path is not None
    use_per_split_data_path = (
        any(
            elt is not None
            for elt in [args.train_data_path, args.valid_data_path, args.test_data_path]
        )
        or args.per_split_data_args_path is not None
    )

    blend = None
    blend_per_split = None
    if use_data_path:
        if args.data_args_path is not None:
            assert args.data_path is None
            with open_file(args.data_args_path, 'r') as f:
                blend = get_blend_from_list(f.read().split())
        else:
            assert args.data_path is not None
            blend = get_blend_from_list(args.data_path)
    elif use_per_split_data_path:
        if args.per_split_data_args_path is not None:
            with open_file(args.per_split_data_args_path, 'r') as f:
                per_split_data_args = json.load(f)
                # Each element in blend_per_split should be a list of files (and optional
                # weights), so split string if needed.
                for split in ["train", "valid", "test"]:
                    if isinstance(per_split_data_args[split], str):
                        per_split_data_args[split] = per_split_data_args[split].split()

                blend_per_split = [
                    get_blend_from_list(per_split_data_args["train"]),
                    get_blend_from_list(per_split_data_args["valid"]),
                    get_blend_from_list(per_split_data_args["test"]),
                ]
        else:
            blend_per_split = [
                get_blend_from_list(args.train_data_path),
                get_blend_from_list(args.valid_data_path),
                get_blend_from_list(args.test_data_path),
            ]
    else:
        blend, blend_per_split = None, None

    return blend, blend_per_split


def update_use_dist_ckpt(args):
    args.use_dist_ckpt = args.ckpt_format != "torch"


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

    return module._apply(
        lambda t: _empty_like_if_meta(t, device=device), recurse=recurse
    )


def get_nvtx_range():
    """Create an NVTX range context manager.

    Returns a context manager that:
    - Creates an NVTX range for profiling (nsight-systems compatible)
    - Optionally tracks time via Megatron timers when time=True

    Args (for returned context manager):
        msg: Name of the range/timer
        time: If True, also track with Megatron timers (default: False)
        log_level: Timer log level (0=always, 1=default, 2=verbose). Default: 1
    """
    from megatron.core.utils import nvtx_range_pop, nvtx_range_push

    @contextmanager
    def nvtx_range(msg, time=False, log_level=1):
        if time:
            timers = get_timers()
            timers(msg, log_level=log_level).start()
        try:
            nvtx_range_push(msg)
            yield
        finally:
            nvtx_range_pop(msg)
            if time:
                timers(msg, log_level=log_level).stop()

    return nvtx_range


def has_nvrx_installed():
    """Checks if nvidia-resiliency-ext is installed."""
    try:
        import nvidia_resiliency_ext
        return True
    except (ImportError, ModuleNotFoundError):
        return False


def has_nvrx_checkpointing_async_support():
    """Checks whether the installed NVRx package exposes the async checkpointing API Megatron uses."""
    return has_nvrx_async_support()


def get_local_rank_preinit() -> int:
    """Get the local rank from the environment variable, intended for use before full init.

    Fallback order:
    1. LOCAL_RANK environment variable (torchrun/torchelastic)
    2. SLURM_LOCALID environment variable (SLURM)
    3. Default: 0 (with warning)

    Returns:
        The local rank of the current process.
    """
    if "LOCAL_RANK" in os.environ:
        return int(os.environ["LOCAL_RANK"])

    slurm_local_rank = resolve_slurm_local_rank()
    if slurm_local_rank is not None:
        return slurm_local_rank

    warnings.warn("Could not determine local rank from LOCAL_RANK or SLURM_LOCALID. Defaulting to local rank 0.")
    return 0
