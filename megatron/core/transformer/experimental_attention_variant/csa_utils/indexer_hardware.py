# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Hardware estimates for the eager CSA indexer layout solver."""

import ctypes
import hashlib
import logging
import os
import socket
from threading import Lock
from typing import NamedTuple
from weakref import WeakKeyDictionary

import torch
import torch.distributed as dist


class LayoutCosts(NamedTuple):
    """Effective rates and additional latency for one routed indexer invocation."""

    compute_tflops: float
    exchange_gbps: float
    copy_gbps: float
    overhead_us: float


class _DeviceInfo(NamedTuple):
    """Small integer record exchanged once by the members of a real CP group."""

    bf16_tflops: int
    hbm_gbps: int
    nvlink_gbps: int
    host: int
    fabric: int


# Rounded per-GPU dense BF16 TFLOP/s, HBM GB/s and one-way NVLink GB/s for
# switch-connected parts. PCIe/NVL boards use the PCIe estimate for local CP.
_PEAK_RATES = {
    "H20": (148, 4000, 450),
    "H100": (989, 3350, 450),
    "H200": (989, 4800, 450),
    "B200": (2250, 8000, 900),
    "B300": (2250, 7700, 900),
    "GB200": (2500, 8000, 900),
    "GB300": (2500, 8000, 900),
}
_GROUP_COSTS: WeakKeyDictionary[dist.ProcessGroup, LayoutCosts | None] = WeakKeyDictionary()
_GROUP_LOCKS: WeakKeyDictionary = WeakKeyDictionary()
_logger = logging.getLogger(__name__)


def _tag(value: bytes) -> int:
    """Produce the same compact host or fabric identifier in every process."""
    return int.from_bytes(hashlib.blake2b(value, digest_size=8).digest(), "little", signed=True)


def _fabric_tag(device_uuid) -> int:
    """Identify a registered NVLink fabric clique when NVML exposes it."""
    if device_uuid is None or os.environ.get("NCCL_MNNVL_ENABLE") == "0":
        return 0
    try:
        import pynvml
    except ImportError:
        return 0
    try:
        pynvml.nvmlInit()
        try:
            gpu = "GPU-" + str(device_uuid).removeprefix("GPU-")
            handle = pynvml.nvmlDeviceGetHandleByUUID(gpu.encode())
            info = pynvml.c_nvmlGpuFabricInfoV_t()
            pynvml.nvmlDeviceGetGpuFabricInfoV(handle, ctypes.byref(info))
            # NVML UUIDs are 16 binary bytes. Some bindings expose a C string,
            # which truncates an ordinary attribute read at the first NUL.
            uuid = ctypes.string_at(ctypes.byref(info, type(info).clusterUuid.offset), 16)
            if (
                info.state == pynvml.NVML_GPU_FABRIC_STATE_COMPLETED
                and info.status == pynvml.NVML_SUCCESS
                and any(uuid)
            ):
                return _tag(uuid + int(info.cliqueId).to_bytes(4, "little"))
        finally:
            pynvml.nvmlShutdown()
    except (pynvml.NVMLError, AttributeError, TypeError, ctypes.ArgumentError):
        # A driver, container or binding without fabric telemetry can still score packs.
        pass
    return 0


def _device_info(device: torch.device) -> _DeviceInfo:
    """Use the assigned CUDA device, including when visible ordinals are remapped."""
    props = torch.cuda.get_device_properties(device)
    words = props.name.upper().replace("-", " ").split()
    peak, memory, link = next((_PEAK_RATES[w] for w in words if w in _PEAK_RATES), (0, 0, 0))
    if not peak or "MIG" in words:
        return _DeviceInfo(0, 0, 0, 0, 0)
    if "NVL" in words and ("H100" in words or "H200" in words):
        peak, memory, link = 835, 3900 if "H100" in words else 4800, 0
    elif "PCIE" in words and "H100" in words:
        peak, memory, link = 756, 2000, 0
    # CUDA reports memory clock in kHz and bus width in bits. Older PyTorch
    # releases only expose the board name, for which the table supplies HBM bandwidth.
    try:
        clock = getattr(props, "memory_clock_rate", 0)
        width = getattr(props, "memory_bus_width", 0)
        if clock > 0 and width > 0:
            memory = round(2 * clock * width / 8e6) or memory
    except RuntimeError:
        pass  # Some drivers do not implement the optional clock query.
    if os.environ.get("NCCL_P2P_DISABLE") == "1":
        link = 0
    return _DeviceInfo(
        peak,
        memory,
        link,
        _tag(socket.gethostname().encode()),
        _fabric_tag(getattr(props, "uuid", None)) if link else 0,
    )


def _group_costs(members: list[_DeviceInfo]) -> LayoutCosts | None:
    """Price routing conservatively for the actual CP group, including mixed GPUs."""
    if not all(d.bf16_tflops and d.hbm_gbps for d in members):
        return None
    first = members[0]
    same_host = first.host != 0 and all(d.host == first.host for d in members)
    same_fabric = first.fabric != 0 and all(d.fabric == first.fabric for d in members)
    link = min(d.nvlink_gbps for d in members)
    if link and (same_fabric or (same_host and not any(d.fabric for d in members))):
        exchange_gbps, overhead_us = 0.5 * link, 20.0
    elif same_host:
        exchange_gbps, overhead_us = 32.0, 30.0
    else:
        exchange_gbps, overhead_us = 25.0, 50.0
    # The fastest scorer bounds the saving. The slowest HBM sets the copy cost.
    # Charge multiple reads/writes per payload byte and discount peak dot-product rates.
    return LayoutCosts(
        0.5 * max(d.bf16_tflops for d in members),
        exchange_gbps,
        0.25 * min(d.hbm_gbps for d in members),
        overhead_us,
    )


def _discover_costs(group: dist.ProcessGroup) -> LayoutCosts | None:
    """Exchange five integers per member on the group used for the indexer."""
    if "cuda:nccl" not in dist.get_backend_config(group).split(","):
        return None
    device = torch.device("cuda", torch.cuda.current_device())
    try:
        local = _device_info(device)
    except (RuntimeError, OSError):
        # A failed local query must still participate in the group decision.
        local = _DeviceInfo(0, 0, 0, 0, 0)
    info = torch.tensor(local, dtype=torch.int64, device=device)
    peers = info.new_empty(group.size() * info.numel())
    dist.all_gather_into_tensor(peers, info, group=group)
    members = [_DeviceInfo(*row) for row in peers.view(group.size(), -1).tolist()]
    return _group_costs(members)


def group_layout_costs(group: dist.ProcessGroup) -> LayoutCosts | None:
    """Resolve one common estimate on first use; release it with the process group."""
    if not dist.is_initialized() or not isinstance(group, dist.ProcessGroup):
        return None
    if group not in _GROUP_COSTS:
        # Concurrent first users must issue exactly one collective for this group.
        with _GROUP_LOCKS.setdefault(group, Lock()):
            if group not in _GROUP_COSTS:
                costs = _discover_costs(group)
                _GROUP_COSTS[group] = costs
                if os.environ.get("MCORE_DSA_CP_BAL_DEBUG") == "1" and group.rank() == 0:
                    _logger.info(
                        "CSA auto indexer hardware estimate for CP=%d: %s", group.size(), costs
                    )
    return _GROUP_COSTS[group]
