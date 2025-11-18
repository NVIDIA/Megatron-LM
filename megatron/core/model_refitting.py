from megatron.core.models.common.language_module.language_module import LanguageModule
import torch
import torch.distributed as dist
from typing import Any
from megatron.core import parallel_state
from mcore_reshard import reshard_with_general_planner
from typing import Any, Optional



def _unwrap_module(module: LanguageModule) -> Any:
    return module.module.module if hasattr(module, 'module') and hasattr(module.module, 'module') else module.module if hasattr(module, 'module') else module


def _move_module(module: Any, device: torch.device | str) -> None:
    for p in module.parameters(recurse=True):
        if p is not None and p.data is not None:
            p.data = p.data.to(device, non_blocking=True)
        if p is not None and p._grad is not None:
            p._grad = p._grad.to(device, non_blocking=True)
    for buf_name, buf in module._buffers.items():  # type: ignore[attr-defined]
        if buf is not None:
            module._buffers[buf_name] = buf.to(device, non_blocking=True)  # type: ignore[index]



def naive_model_swap(src_model: LanguageModule, target_model: LanguageModule):
    print(f"Swapping train to inference model")
    # Handle list-wrapped modules used throughout training utils
    src_lm = src_model[0] if isinstance(src_model, (list, tuple)) else src_model
    target_lm = target_model[0] if isinstance(target_model, (list, tuple)) else target_model

    # Unwrap possible precision/wrapper modules to reach the module that owns parameters
    src_model = _unwrap_module(src_lm)
    target_model = _unwrap_module(target_lm)

    
    src_tp_size = dist.get_world_size(src_model.pg_collection.tp)
    target_tp_size = dist.get_world_size(target_model.pg_collection.tp)
    src_tp_group = src_model.pg_collection.tp
    target_tp_group = target_model.pg_collection.tp

    # Build name->param map for inference module
    infer_params = {name: p for name, p in target_model.named_parameters(recurse=True)}


    # Utility: reconstruct global tensor from TP sharded local tensors
    def _gather_master_from_training(local_shard: torch.Tensor, dim: int, stride: int, group) -> torch.Tensor:
        world_size = dist.get_world_size(group=group)
        if world_size == 1:
            return local_shard.detach().clone()
        # Gather shards from all TP ranks
        gather_list = [torch.empty_like(local_shard) for _ in range(world_size)]
        dist.all_gather(gather_list, local_shard.contiguous(), group=group)
        if stride == 1:
            return torch.cat(gather_list, dim=dim).contiguous()
        # Strided partition: split each shard into stride chunks and interleave
        per_part = local_shard.size(dim)
        assert per_part % stride == 0, "Local shard size must be divisible by stride"
        per_stride = per_part // stride
        blocks: list[torch.Tensor] = [None] * (world_size * stride)  # type: ignore[assignment]
        for r in range(world_size):
            chunks = torch.split(gather_list[r], per_stride, dim=dim)
            assert len(chunks) == stride, "Unexpected number of stride chunks"
            for i in range(stride):
                blocks[r + i * world_size] = chunks[i]
        return torch.cat(blocks, dim=dim).contiguous()

    # Utility: shard master tensor to the target inference layout
    def _shard_master_to_infer(master: torch.Tensor, dim: int, stride: int, group) -> torch.Tensor:
        world_size = dist.get_world_size(group=group)
        if world_size == 1:
            return master
        rank = dist.get_rank(group=group)
        full = master
        assert full.size(dim) % world_size == 0, "Master size not divisible by TP world size"
        per_part = full.size(dim) // world_size
        assert per_part % stride == 0, "Per-part size must be divisible by stride"
        per_stride = per_part // stride
        weight_list = torch.split(full, per_stride, dim=dim)
        # Pick this rank's stride segments and concatenate along partition dim
        my_chunks = weight_list[rank::world_size][:stride]
        return torch.cat(my_chunks, dim=dim).contiguous()

    # Perform transfer
    with torch.no_grad():
        # Simple same-TP copy path for models with the same TP size
        if src_tp_size == target_tp_size:
            for name, src in src_model.named_parameters(recurse=True):
                if name not in infer_params:
                    raise ValueError(f"Parameter {name} in training model not found in inference model")
                dst = infer_params[name]
                if src.shape == dst.shape:
                    dst.copy_(src)
                else:
                    raise ValueError(f"Parameter {name} in training model has different shape than in inference model")
            return

        # General reshard path
        for name, src in src_model.named_parameters(recurse=True):
            dst = infer_params.get(name, None)
            if dst is None:
                raise ValueError(f"Parameter {name} in training model not found in inference model")

            # Non-TP params: direct copy
            is_tp = bool(getattr(src, 'tensor_model_parallel', False))
            if not is_tp:
                if src.shape != dst.shape:
                    # If shapes differ unexpectedly for non-TP, try to broadcast master (rank 0) value
                    #dst.copy_(src.detach().clone().to(dst.dtype))
                    raise ValueError(f"Parameter {name} in training model has different shape than in inference model")
                else:
                    dst.copy_(src)
                continue

            # Resolve sharding attributes and groups
            dim = int(getattr(src, 'partition_dim', 0))
            stride = int(getattr(src, 'partition_stride', 1))
            # Gather training shards -> master
            master = _gather_master_from_training(src, dim=dim, stride=stride, group=src_tp_group)

            # Use dst's own TP attributes for target stride/dim if present
            #TODO when do these just match the the original expecailly dim?
            target_dim = int(getattr(dst, 'partition_dim', dim))
            target_stride = int(getattr(dst, 'partition_stride', stride))
            local_target = _shard_master_to_infer(master, dim=target_dim, stride=target_stride, group=target_tp_group)
            # Cast and copy
            if local_target.dtype != dst.dtype:
                raise ValueError(f"Parameter {name} in training model has different dtype than in inference model")
                # local_target = local_target.to(dst.dtype)
            dst.copy_(local_target)
    print(f"finished Swapped train to inference model")


def swap_model_weights(src_model: LanguageModule, target_model: LanguageModule, refit_method: str):
    if refit_method == "naive":
        naive_model_swap(src_model, target_model)
    elif refit_method == "nccl":
        nccl_model_swap(src_model, target_model)
    else:
        raise ValueError(f"Invalid refit method: {refit_method}")

def nccl_model_swap(src_model: LanguageModule, target_model: LanguageModule):
    # Handle list-wrapped modules used throughout training utils
    src_lm = src_model[0] if isinstance(src_model, (list, tuple)) else src_model
    tgt_lm = target_model[0] if isinstance(target_model, (list, tuple)) else target_model

    # Unwrap to get owning modules (with parameters and pg_collection)
    src_core = _unwrap_module(src_lm)
    tgt_core = _unwrap_module(tgt_lm)

    # Ensure pg_collection exists
    if not hasattr(src_core, "pg_collection") or src_core.pg_collection is None:
        raise RuntimeError("Source model missing pg_collection required for NCCL reshard")
    if not hasattr(tgt_core, "pg_collection") or tgt_core.pg_collection is None:
        raise RuntimeError("Target model missing pg_collection required for NCCL reshard")

    #TODO(Peter): We should figure out why this happens.
    # Fill missing DP group on the source using Megatron's parallel state if not provided
    if getattr(src_core.pg_collection, "dp", None) is None:
        src_core.pg_collection.dp = parallel_state.get_data_parallel_group()
    # caching plan for reuse
    cached_plan: Optional[Any] = getattr(tgt_core, "_cached_reshard_plan", None)
    plan = reshard_with_general_planner(src_core, tgt_core, cached_plan=cached_plan)
    if cached_plan is None:
        setattr(tgt_core, "_cached_reshard_plan", plan)