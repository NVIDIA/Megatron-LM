# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Lifecycle and symmetric workspace support for experimental NVSHMEM CP attention."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from megatron.core import parallel_state

_NVSHMEM_INITIALIZED = False
_NVSHMEM_INITIALIZATION_ELAPSED_MS: float | None = None
_NVSHMEM_WORKSPACES_FROZEN = False
_WORKSPACES: Dict[Tuple[object, ...], "NvshmemCpWorkspace"] = {}

_NVSHMEM_CP_BRANCH_B_PROFILE = {
    "FLASH_ATTN_TWO_SECTION_DIRECT_OWNER_DKV": "1",
    "FLASH_ATTN_UNSAFE_TWO_SECTION_CAUSAL_BWD": "1",
    "MEGATRON_NVSHMEM_CP_BLOCK_READY_PROTOCOL": "1",
    "MEGATRON_NVSHMEM_CP_BLOCK_READY_STREAM_WAIT": "0",
    "MEGATRON_NVSHMEM_CP_BRANCH_B_FA4_GLOBAL": "1",
    "MEGATRON_NVSHMEM_CP_BRANCH_B_FA4_GLOBAL_GRAD_RETURN": "1",
    "MEGATRON_NVSHMEM_CP_BRANCH_B_FA4_PREWARM": "1",
    "MEGATRON_NVSHMEM_CP_BRANCH_B_FA4_SHARED_KV_BATCH": "1",
    "MEGATRON_NVSHMEM_CP_BRANCH_B_FA4_TWO_SECTION_CAUSAL_BWD": "1",
    "MEGATRON_NVSHMEM_CP_BRANCH_B_GRAD_EPOCH_STREAM_WAIT": "1",
    "MEGATRON_NVSHMEM_CP_BRANCH_B_HOST_CALL_EPOCH": "1",
    "MEGATRON_NVSHMEM_CP_BRANCH_B_SKIP_POST_NATIVE_BACKWARD_SYNC": "1",
    "MEGATRON_NVSHMEM_CP_BRANCH_B_SKIP_PRE_GRAD_EPOCH_STREAM_SYNC": "1",
    "MEGATRON_NVSHMEM_CP_BRANCH_B_STREAM_WRITE_GRAD_EPOCH": "1",
    "MEGATRON_NVSHMEM_CP_CORE_ATTN_SAVED_STATE": "1",
    "MEGATRON_NVSHMEM_CP_DIRECT_QKV_PRODUCER": "0",
    "MEGATRON_NVSHMEM_CP_FA_OWNER_IO_V1_BACKWARD_IMPL": "block_ready_native_fused",
    "MEGATRON_NVSHMEM_CP_GRAD_RETURN_DTYPE": "fp32",
    "MEGATRON_NVSHMEM_CP_OWNER": "fa_owner_io_v1",
    "MEGATRON_NVSHMEM_CP_SELF_ATTENTION_BACKEND": "symmetric_qkv_v0",
}


class NvshmemCpAttentionError(RuntimeError):
    """Raised when the experimental NVSHMEM CP attention path cannot run."""


def configure_nvshmem_cp_backend() -> dict[str, str]:
    """Install the validated Branch-B profile behind the typed Megatron selector.

    NVSHMEM runtime settings such as the symmetric heap and CUDA VMM policy remain deployment
    concerns and are intentionally not changed here.
    """

    for name, value in _NVSHMEM_CP_BRANCH_B_PROFILE.items():
        os.environ.setdefault(name, value)
    return dict(_NVSHMEM_CP_BRANCH_B_PROFILE)


class NvshmemCpWorkspace:
    key: torch.Tensor
    value: torch.Tensor
    committed_epoch: torch.Tensor
    block_ready_epoch: torch.Tensor
    grad_key_return: torch.Tensor
    grad_value_return: torch.Tensor
    grad_committed_epoch: torch.Tensor
    raw_key_return: torch.Tensor
    raw_value_return: torch.Tensor
    carrier_key_slots: torch.Tensor
    carrier_value_slots: torch.Tensor
    carrier_epoch: torch.Tensor
    peer_keys: List[torch.Tensor]
    peer_values: List[torch.Tensor]
    peer_committed_epochs: List[torch.Tensor]
    peer_block_ready_epochs: List[torch.Tensor]
    peer_grad_key_returns: List[torch.Tensor]
    peer_grad_value_returns: List[torch.Tensor]
    peer_grad_committed_epochs: List[torch.Tensor]
    peer_raw_key_returns: List[torch.Tensor]
    peer_raw_value_returns: List[torch.Tensor]
    peer_carrier_key_slots: List[torch.Tensor]
    peer_carrier_value_slots: List[torch.Tensor]
    peer_carrier_epochs: List[torch.Tensor]
    cp_group_ranks: List[int]


def _import_nvshmem():
    try:
        import cupy
        import nvshmem
        import nvshmem.core
        try:
            from cuda.core import Device
        except ImportError:
            from cuda.core.experimental import Device
        from nvshmem import bindings as nvshmem_bindings
        from nvshmem.core.interop import torch as nvshmem_torch
        from nvshmem.core.interop.torch import get_peer_tensor
    except Exception as err:
        raise NvshmemCpAttentionError(
            "Experimental NVSHMEM CP attention requires nvshmem4py-cu13, "
            "cuda-core, and cupy in the active Megatron Python environment."
        ) from err
    return cupy, nvshmem, Device, nvshmem_bindings, nvshmem_torch, get_peer_tensor


def _ensure_nvshmem_initialized() -> object:
    global _NVSHMEM_INITIALIZED, _NVSHMEM_INITIALIZATION_ELAPSED_MS
    cupy, nvshmem, Device, nvshmem_bindings, _, _ = _import_nvshmem()
    if _NVSHMEM_INITIALIZED:
        return nvshmem_bindings

    init_start = time.perf_counter()

    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        raise NvshmemCpAttentionError("torch.distributed must be initialized before NVSHMEM.")

    device = torch.cuda.current_device()
    nv_device = Device(device)
    nv_device.set_current()
    cupy.cuda.Device(device).use()

    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    uid_holder = [nvshmem.core.get_unique_id(empty=True)]
    if rank == 0:
        uid_holder[0] = nvshmem.core.get_unique_id()
    torch.distributed.broadcast_object_list(uid_holder, src=0)
    torch.distributed.barrier()

    nvshmem.core.init(
        device=nv_device,
        uid=uid_holder[0],
        rank=rank,
        nranks=world_size,
        initializer_method="uid",
    )
    _NVSHMEM_INITIALIZED = True
    _NVSHMEM_INITIALIZATION_ELAPSED_MS = (time.perf_counter() - init_start) * 1000.0
    return nvshmem_bindings


def eager_initialize_nvshmem_cp_backend_if_enabled() -> bool:
    """Initialize the world-scoped NVSHMEM runtime before pipeline execution."""

    if os.getenv("MEGATRON_NVSHMEM_CP_SELF_ATTENTION_BACKEND") != "symmetric_qkv_v0":
        return False

    was_initialized = _NVSHMEM_INITIALIZED
    _ensure_nvshmem_initialized()
    return not was_initialized


def validate_nvshmem_cp_microbatch_contract(num_microbatches: int) -> None:
    """Reject workspace reuse that can overlap across pipeline microbatches."""

    if os.getenv("MEGATRON_NVSHMEM_CP_SELF_ATTENTION_BACKEND") != "symmetric_qkv_v0":
        return
    if int(num_microbatches) != 1:
        raise NvshmemCpAttentionError(
            "The symmetric_qkv_v0 backend currently owns one workspace per layer and "
            "requires exactly one pipeline microbatch; "
            f"num_microbatches={num_microbatches}."
        )


def _workspace_allocation_manifest(
    shape: Tuple[int, ...],
    dtype: torch.dtype,
    cp_size: int,
) -> Dict[str, object]:
    """Describe every symmetric allocation in its required collective order."""

    grad_shape = (cp_size,) + tuple(shape)

    allocations = [
        {"name": "key", "shape": list(shape), "dtype": str(dtype)},
        {"name": "value", "shape": list(shape), "dtype": str(dtype)},
        {"name": "committed_epoch", "shape": [1], "dtype": str(torch.int32)},
        {
            "name": "block_ready_epoch",
            "shape": [_ready_block_count(shape)],
            "dtype": str(torch.int32),
        },
        {"name": "grad_key_return", "shape": list(grad_shape), "dtype": str(dtype)},
        {"name": "grad_value_return", "shape": list(grad_shape), "dtype": str(dtype)},
    ]
    allocations.append(
        {"name": "grad_committed_epoch", "shape": [cp_size], "dtype": str(torch.int32)}
    )

    return {
        "shape": list(shape),
        "dtype": str(dtype),
        "cp_size": cp_size,
        "ready_block_count": _ready_block_count(shape),
        "abi": {
            "raw_return_sideband": False,
            "cross_rank_carrier": False,
            "writer_indexed_carrier": False,
            "carrier_slot_count": 0,
        },
        "allocations": allocations,
    }


def eager_allocate_nvshmem_cp_workspaces_if_enabled(
    model,
    *,
    seq_length: int,
    micro_batch_size: int,
) -> int:
    """Collectively allocate every local layer workspace before pipeline execution."""

    global _NVSHMEM_WORKSPACES_FROZEN

    if os.getenv("MEGATRON_NVSHMEM_CP_SELF_ATTENTION_BACKEND") != "symmetric_qkv_v0":
        return 0
    if not _NVSHMEM_INITIALIZED:
        raise NvshmemCpAttentionError(
            "NVSHMEM must be eagerly initialized before workspace allocation."
        )

    allocation_start = time.perf_counter()

    roots = model if isinstance(model, (list, tuple)) else [model]
    specs = []
    local_signature = []
    local_error = None
    try:
        attention_modules = []
        seen = set()
        for root in roots:
            for module in root.modules():
                if id(module) in seen or module.__class__.__name__ != "SelfAttention":
                    continue
                seen.add(id(module))
                if not hasattr(module, "layer_number"):
                    continue
                attention_modules.append(module)
        attention_modules.sort(key=lambda module: int(module.layer_number or 0))

        if not attention_modules:
            raise NvshmemCpAttentionError(
                "The symmetric_qkv_v0 backend requires every pipeline rank to own "
                "at least one SelfAttention layer."
            )
        layer_numbers = [int(module.layer_number or 0) for module in attention_modules]
        duplicate_layers = sorted(
            layer for layer in set(layer_numbers) if layer_numbers.count(layer) > 1
        )
        if duplicate_layers:
            raise NvshmemCpAttentionError(
                "The symmetric_qkv_v0 workspace cache requires unique local layer numbers; "
                f"duplicates={duplicate_layers}."
            )

        for module in attention_modules:
            cp_size = int(module.config.context_parallel_size)
            if cp_size != 4 or seq_length % cp_size != 0:
                raise NvshmemCpAttentionError(
                    "The symmetric_qkv_v0 workspace preallocator requires CP=4 and "
                    "a sequence length divisible by CP."
                )
            shape = (
                int(seq_length) // cp_size,
                int(micro_batch_size),
                int(module.num_query_groups_per_partition),
                int(module.hidden_size_per_attention_head),
            )
            dtype = module.config.params_dtype
            specs.append((shape, dtype, int(module.layer_number or 0), cp_size))
        local_signature = [
            _workspace_allocation_manifest(shape, dtype, cp_size)
            for shape, dtype, _, cp_size in specs
        ]
    except Exception as error:
        local_error = f"{type(error).__name__}: {error}"

    local_validation = {"error": local_error, "signature": local_signature}
    world_validations = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(world_validations, local_validation)
    rank_errors = {
        rank: validation["error"]
        for rank, validation in enumerate(world_validations)
        if validation["error"] is not None
    }
    if rank_errors:
        raise NvshmemCpAttentionError(
            "NVSHMEM workspace validation failed before symmetric allocation; "
            f"rank_errors={rank_errors}."
        )

    canonical_signature = world_validations[0]["signature"]
    mismatched = [
        rank
        for rank, validation in enumerate(world_validations)
        if validation["signature"] != canonical_signature
    ]
    if mismatched:
        raise NvshmemCpAttentionError(
            "NVSHMEM symmetric allocations must be collective and identically ordered; "
            f"workspace signature differs on global ranks {mismatched}."
        )

    device = torch.device("cuda", torch.cuda.current_device())
    for shape, dtype, layer_number, _ in specs:
        _workspace(shape, dtype, device, layer_number)
    torch.cuda.current_stream().synchronize()
    torch.distributed.barrier()
    _NVSHMEM_WORKSPACES_FROZEN = True
    audit_dir = os.getenv("MEGATRON_NVSHMEM_CP_STARTUP_AUDIT_DIR")
    if audit_dir:
        global_rank = torch.distributed.get_rank()

        def group_members(group, name: str) -> List[int]:
            if isinstance(group, (list, tuple)):
                raise NvshmemCpAttentionError(
                    f"Startup receipt requires one {name} process group, got {type(group).__name__}."
                )
            return [int(rank) for rank in torch.distributed.get_process_group_ranks(group)]

        tp_group_ranks = group_members(
            parallel_state.get_tensor_model_parallel_group(), "TP"
        )
        cp_group_ranks = group_members(parallel_state.get_context_parallel_group(), "CP")
        pp_group_ranks = group_members(
            parallel_state.get_pipeline_model_parallel_group(), "PP"
        )
        dp_group_ranks = group_members(
            parallel_state.get_data_parallel_group(
                with_context_parallel=False, partial_data_parallel=False
            ),
            "dense DP",
        )
        ep_group_ranks = group_members(
            parallel_state.get_expert_model_parallel_group(), "EP"
        )
        expert_dp_group_ranks = group_members(
            parallel_state.get_expert_data_parallel_group(
                partial_expert_data_parallel=False
            ),
            "expert DP",
        )
        parallel_state_path = Path(parallel_state.__file__).resolve()
        parallel_state_sha256 = hashlib.sha256(parallel_state_path.read_bytes()).hexdigest()
        payload = {
            "schema_version": 2,
            "backend": "symmetric_qkv_v0",
            "global_rank": int(global_rank),
            "world_size": int(torch.distributed.get_world_size()),
            "local_device": int(torch.cuda.current_device()),
            "rank_order": os.getenv("MEGATRON_NVSHMEM_CP_RANK_ORDER"),
            "tp_size": len(tp_group_ranks),
            "pp_size": len(pp_group_ranks),
            "cp_size": len(cp_group_ranks),
            "dense_dp_size": len(dp_group_ranks),
            "ep_size": len(ep_group_ranks),
            "expert_dp_size": len(expert_dp_group_ranks),
            "tp_rank": tp_group_ranks.index(global_rank),
            "tp_group_ranks": tp_group_ranks,
            "cp_rank": cp_group_ranks.index(global_rank),
            "cp_group_ranks": cp_group_ranks,
            "pp_rank": pp_group_ranks.index(global_rank),
            "pp_group_ranks": pp_group_ranks,
            "dp_rank": dp_group_ranks.index(global_rank),
            "dp_group_ranks": dp_group_ranks,
            "ep_rank": ep_group_ranks.index(global_rank),
            "ep_group_ranks": ep_group_ranks,
            "expert_dp_rank": expert_dp_group_ranks.index(global_rank),
            "expert_dp_group_ranks": expert_dp_group_ranks,
            "parallel_state_source": str(parallel_state_path),
            "parallel_state_source_sha256": parallel_state_sha256,
            "nvshmem_initialized": bool(_NVSHMEM_INITIALIZED),
            "nvshmem_initialization_elapsed_ms": _NVSHMEM_INITIALIZATION_ELAPSED_MS,
            "workspace_preallocated": True,
            "workspace_count": len(specs),
            "workspace_signature": local_signature,
            "workspace_preallocation_elapsed_ms": (
                time.perf_counter() - allocation_start
            )
            * 1000.0,
            "workspaces_frozen": bool(_NVSHMEM_WORKSPACES_FROZEN),
            "nvshmem_disable_cuda_vmm": os.getenv("NVSHMEM_DISABLE_CUDA_VMM"),
            "nvshmem_symmetric_size": os.getenv("NVSHMEM_SYMMETRIC_SIZE"),
        }
        audit_root = Path(audit_dir)
        audit_root.mkdir(parents=True, exist_ok=True)
        output = audit_root / f"rank{global_rank:05d}.json"
        temporary = audit_root / f".{output.name}.{os.getpid()}.tmp"
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.replace(output)
    return len(specs)


def _cp_group_and_ranks() -> Tuple[torch.distributed.ProcessGroup, List[int], int]:
    cp_group = parallel_state.get_context_parallel_group()
    cp_group_ranks = torch.distributed.get_process_group_ranks(cp_group)
    cp_rank = parallel_state.get_context_parallel_rank()
    return cp_group, list(cp_group_ranks), int(cp_rank)


def _workspace(
    shape: Tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    layer_number: int | None,
) -> NvshmemCpWorkspace:
    _, cp_group_ranks, _ = _cp_group_and_ranks()
    _, _, _, _, nvshmem_torch, get_peer_tensor = _import_nvshmem()
    cp_size = len(cp_group_ranks)
    allocation_manifest = _workspace_allocation_manifest(shape, dtype, cp_size)
    abi = allocation_manifest["abi"]
    raw_return_abi = bool(abi["raw_return_sideband"])
    key = (
        shape,
        str(dtype),
        int(device.index or 0),
        int(layer_number or 0),
        json.dumps(allocation_manifest, sort_keys=True),
    )
    if key in _WORKSPACES:
        return _WORKSPACES[key]
    if _NVSHMEM_WORKSPACES_FROZEN:
        raise NvshmemCpAttentionError(
            "NVSHMEM CP workspace was requested after collective preallocation: "
            f"shape={shape}, dtype={dtype}, device={device}, layer={layer_number}. "
            "Refusing an unmatched symmetric allocation during pipeline execution."
        )

    sym_key = nvshmem_torch.tensor(shape, dtype=dtype)
    sym_value = nvshmem_torch.tensor(shape, dtype=dtype)
    committed_epoch = nvshmem_torch.tensor((1,), dtype=torch.int32)
    committed_epoch.fill_(0)
    block_ready_epoch = nvshmem_torch.tensor((_ready_block_count(shape),), dtype=torch.int32)
    block_ready_epoch.fill_(0)
    grad_shape = (cp_size,) + tuple(shape)
    grad_key_return = nvshmem_torch.tensor(grad_shape, dtype=dtype)
    grad_value_return = nvshmem_torch.tensor(grad_shape, dtype=dtype)
    if raw_return_abi:
        raw_key_return = nvshmem_torch.tensor(grad_shape, dtype=dtype)
        raw_value_return = nvshmem_torch.tensor(grad_shape, dtype=dtype)
    else:
        raw_key_return = torch.empty((0,), dtype=dtype, device=device)
        raw_value_return = torch.empty((0,), dtype=dtype, device=device)
    grad_committed_epoch = nvshmem_torch.tensor((len(cp_group_ranks),), dtype=torch.int32)
    grad_committed_epoch.fill_(0)
    carrier_abi = bool(abi["cross_rank_carrier"])
    if carrier_abi:
        writer_indexed_carrier = bool(abi["writer_indexed_carrier"])
        carrier_slot_count = int(abi["carrier_slot_count"])
        carrier_shape = (
            (cp_size, carrier_slot_count) + tuple(shape)
            if writer_indexed_carrier
            else (carrier_slot_count,) + tuple(shape)
        )
        carrier_key_slots = nvshmem_torch.tensor(carrier_shape, dtype=dtype)
        carrier_value_slots = nvshmem_torch.tensor(carrier_shape, dtype=dtype)
        carrier_epoch_shape = (
            (cp_size, carrier_slot_count)
            if writer_indexed_carrier
            else (carrier_slot_count,)
        )
        carrier_epoch = nvshmem_torch.tensor(carrier_epoch_shape, dtype=torch.int32)
        carrier_epoch.fill_(0)
    else:
        carrier_key_slots = torch.empty((0,), dtype=dtype, device=device)
        carrier_value_slots = torch.empty((0,), dtype=dtype, device=device)
        carrier_epoch = torch.empty((0,), dtype=torch.int32, device=device)
    rank = torch.distributed.get_rank()
    peer_keys = [sym_key if pe == rank else get_peer_tensor(sym_key, pe) for pe in cp_group_ranks]
    peer_values = [
        sym_value if pe == rank else get_peer_tensor(sym_value, pe) for pe in cp_group_ranks
    ]
    peer_committed_epochs = [
        committed_epoch if pe == rank else get_peer_tensor(committed_epoch, pe)
        for pe in cp_group_ranks
    ]
    peer_block_ready_epochs = [
        block_ready_epoch if pe == rank else get_peer_tensor(block_ready_epoch, pe)
        for pe in cp_group_ranks
    ]
    peer_grad_key_returns = [
        grad_key_return if pe == rank else get_peer_tensor(grad_key_return, pe)
        for pe in cp_group_ranks
    ]
    peer_grad_value_returns = [
        grad_value_return if pe == rank else get_peer_tensor(grad_value_return, pe)
        for pe in cp_group_ranks
    ]
    peer_grad_committed_epochs = [
        grad_committed_epoch if pe == rank else get_peer_tensor(grad_committed_epoch, pe)
        for pe in cp_group_ranks
    ]
    if raw_return_abi:
        peer_raw_key_returns = [
            raw_key_return if pe == rank else get_peer_tensor(raw_key_return, pe)
            for pe in cp_group_ranks
        ]
        peer_raw_value_returns = [
            raw_value_return if pe == rank else get_peer_tensor(raw_value_return, pe)
            for pe in cp_group_ranks
        ]
    else:
        peer_raw_key_returns = [raw_key_return for _ in cp_group_ranks]
        peer_raw_value_returns = [raw_value_return for _ in cp_group_ranks]
    if carrier_abi:
        peer_carrier_key_slots = [
            carrier_key_slots if pe == rank else get_peer_tensor(carrier_key_slots, pe)
            for pe in cp_group_ranks
        ]
        peer_carrier_value_slots = [
            carrier_value_slots if pe == rank else get_peer_tensor(carrier_value_slots, pe)
            for pe in cp_group_ranks
        ]
        peer_carrier_epochs = [
            carrier_epoch if pe == rank else get_peer_tensor(carrier_epoch, pe)
            for pe in cp_group_ranks
        ]
    else:
        peer_carrier_key_slots = [carrier_key_slots for _ in cp_group_ranks]
        peer_carrier_value_slots = [carrier_value_slots for _ in cp_group_ranks]
        peer_carrier_epochs = [carrier_epoch for _ in cp_group_ranks]
    ws = NvshmemCpWorkspace(
        key=sym_key,
        value=sym_value,
        committed_epoch=committed_epoch,
        block_ready_epoch=block_ready_epoch,
        grad_key_return=grad_key_return,
        grad_value_return=grad_value_return,
        grad_committed_epoch=grad_committed_epoch,
        raw_key_return=raw_key_return,
        raw_value_return=raw_value_return,
        carrier_key_slots=carrier_key_slots,
        carrier_value_slots=carrier_value_slots,
        carrier_epoch=carrier_epoch,
        peer_keys=peer_keys,
        peer_values=peer_values,
        peer_committed_epochs=peer_committed_epochs,
        peer_block_ready_epochs=peer_block_ready_epochs,
        peer_grad_key_returns=peer_grad_key_returns,
        peer_grad_value_returns=peer_grad_value_returns,
        peer_grad_committed_epochs=peer_grad_committed_epochs,
        peer_raw_key_returns=peer_raw_key_returns,
        peer_raw_value_returns=peer_raw_value_returns,
        peer_carrier_key_slots=peer_carrier_key_slots,
        peer_carrier_value_slots=peer_carrier_value_slots,
        peer_carrier_epochs=peer_carrier_epochs,
        cp_group_ranks=cp_group_ranks,
    )
    _WORKSPACES[key] = ws
    return ws


def _block_size(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value <= 0:
        raise NvshmemCpAttentionError(f"{name} must be positive, got {value}.")
    return value


def _ready_block_size() -> int:
    return _block_size(
        "MEGATRON_NVSHMEM_CP_READY_BLOCK_SIZE",
        _block_size("MEGATRON_NVSHMEM_CP_FUSED_BLOCK_N", 64),
    )


def _ready_block_count(shape: Tuple[int, ...]) -> int:
    if not shape:
        return 1
    return max(1, int(math.ceil(int(shape[0]) / float(_ready_block_size()))))


def _publish_block_ready_epoch(ws: NvshmemCpWorkspace, epoch: int) -> None:
    ws.block_ready_epoch[:].fill_(int(epoch))


def _wait_for_block_ready_epoch(ws: NvshmemCpWorkspace, epoch: int, timeout: float) -> None:
    block_count = int(ws.block_ready_epoch.numel())
    deadline = time.perf_counter() + timeout
    while True:
        all_ready = True
        for peer_rank, peer_epoch in zip(ws.cp_group_ranks, ws.peer_block_ready_epochs):
            if bool((peer_epoch[:block_count] < int(epoch)).any().item()):
                all_ready = False
                break
        if all_ready:
            return
        if time.perf_counter() >= deadline:
            observed = [
                [int(value) for value in peer_epoch[:block_count].detach().cpu().tolist()]
                for peer_epoch in ws.peer_block_ready_epochs
            ]
            raise NvshmemCpAttentionError(
                f"Timed out waiting for block_ready_epoch >= {epoch}; "
                f"cp_group_ranks={ws.cp_group_ranks}, block_count={block_count}, "
                f"observed={observed}."
            )
        time.sleep(0.00001)


def _sync_timeout_seconds() -> float:
    value = float(os.getenv("MEGATRON_NVSHMEM_CP_SYNC_TIMEOUT_SECONDS", "30"))
    if value <= 0:
        raise NvshmemCpAttentionError(
            f"MEGATRON_NVSHMEM_CP_SYNC_TIMEOUT_SECONDS must be positive, got {value}."
        )
    return value


def _wait_for_grad_committed_epoch(ws: NvshmemCpWorkspace, epoch: int, timeout: float) -> None:
    deadline = time.perf_counter() + timeout
    while True:
        observed = [int(x) for x in ws.grad_committed_epoch.detach().cpu().tolist()]
        if all(value >= epoch for value in observed):
            return
        if time.perf_counter() >= deadline:
            raise NvshmemCpAttentionError(
                f"Timed out waiting for grad_committed_epoch >= {epoch}; "
                f"cp_group_ranks={ws.cp_group_ranks}, observed={observed}."
            )
        time.sleep(0.00001)


def _next_grad_return_epoch(ws: NvshmemCpWorkspace) -> int:
    return int(ws.grad_committed_epoch.min().item()) + 1
