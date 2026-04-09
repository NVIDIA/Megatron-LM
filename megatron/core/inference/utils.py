# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import asyncio
import logging
import math
import multiprocessing
import sys
from importlib.metadata import PackageNotFoundError, version

import torch

from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.utils import get_model_config

try:
    FLASHINFER_JIT_CACHE_VERSION = version("flashinfer-jit-cache")
except PackageNotFoundError:
    FLASHINFER_JIT_CACHE_VERSION = None


def device_memory_summary() -> str:
    """One-line GPU memory summary for torch_memory_saver logging."""
    dev = torch.cuda.current_device()
    stats = torch.cuda.memory_stats(dev)
    try:
        segs = torch.cuda.memory_snapshot(include_traces=False)
    except TypeError:  # include_traces was added in PyTorch 2.11
        segs = torch.cuda.memory_snapshot()
    M = 1024**2
    private = sum(
        s.get("active_size", 0)
        for s in segs
        if s.get("device", dev) == dev and tuple(s.get("segment_pool_id", (0, 0))) != (0, 0)
    )
    alloc = stats.get("allocated_bytes.all.current", 0)
    resv = stats.get("reserved_bytes.all.current", 0)
    dev_mem = torch.cuda.device_memory_used()
    return (
        f"alloc={alloc/M:.0f}MiB private={private/M:.0f}MiB "
        f"resv-alloc={(resv-alloc)/M:.0f}MiB resv={resv/M:.0f}MiB dev_mem={dev_mem/M:.0f}MiB"
    )


def measure_hbm_bandwidth(device=None, iters=50):
    """Measure observed GPU HBM bandwidth using L2-cache-busting tensor copies.

    Allocates multiple tensor pairs whose total working set exceeds 2x the L2 cache,
    then times round-trip copies to measure sustained memory bandwidth.

    Args:
        device: CUDA device (defaults to current device).
        iters: Number of copy iterations for timing.

    Returns:
        Measured bandwidth in bytes per second.
    """
    if device is None:
        device = torch.cuda.current_device()

    props = torch.cuda.get_device_properties(device)
    l2_cache_bytes = getattr(props, 'l2_cache_size', 64 * 1024 * 1024)

    # Use a large tensor (256 MB) to amortize launch overhead and bust L2 cache
    n = 64 * 1024 * 1024  # 64M float32 elements = 256 MB
    tensor_bytes = n * 4

    # Enough tensors so total working set > 2x L2 cache
    k = max(2, int((l2_cache_bytes * 2) // tensor_bytes) + 1)

    xs = [torch.randn(n, dtype=torch.float32, device=device) for _ in range(k)]
    zs = [torch.empty_like(x) for x in xs]

    # Warmup
    for i in range(k):
        zs[i].copy_(xs[i])
    torch.cuda.synchronize(device)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        for i in range(k):
            zs[i].copy_(xs[i])
    end.record()
    torch.cuda.synchronize(device)

    elapsed_s = start.elapsed_time(end) / 1000.0
    # Each copy reads + writes the tensor: 2 * tensor_bytes per copy
    total_bytes = iters * k * 2 * tensor_bytes
    bandwidth_bytes_per_sec = total_bytes / elapsed_s

    del xs, zs
    torch.cuda.empty_cache()

    return bandwidth_bytes_per_sec


def get_model_weight_bytes(model):
    """Compute total bytes of model weights on GPU.

    Walks all modules checking parameters, buffers, and plain tensor
    attributes to capture weights that may not be registered as nn.Parameter
    (e.g., expert weight buffers in MoE inference, MXFP8 quantized weights).
    Deduplicates by underlying storage to avoid double-counting views.

    Args:
        model: A PyTorch model (possibly wrapped).

    Returns:
        Total weight memory in bytes on GPU.
    """
    seen_storages = {}  # storage data_ptr -> nbytes

    def _account_tensor(tensor):
        """Add a CUDA tensor's storage to the seen set."""
        if tensor.is_cuda:
            ptr = tensor.untyped_storage().data_ptr()
            seen_storages[ptr] = tensor.untyped_storage().nbytes()

    for module in model.modules():
        # nn.Parameter objects
        for tensor in module._parameters.values():
            if tensor is not None and tensor.is_cuda:
                _account_tensor(tensor)
        # Registered buffers (e.g., concatenated expert weight buffers)
        for tensor in module._buffers.values():
            if tensor is not None and tensor.is_cuda:
                _account_tensor(tensor)
        # Plain tensor attributes and quantized tensor wrappers.
        # MXFP8Tensor is not a torch.Tensor subclass — it wraps .data and .scale
        # tensors and is stored as a plain attribute via setattr().
        for attr in module.__dict__.values():
            if isinstance(attr, torch.Tensor):
                _account_tensor(attr)
            elif hasattr(attr, 'data') and isinstance(attr.data, torch.Tensor):
                _account_tensor(attr.data)
                if hasattr(attr, 'scale') and isinstance(attr.scale, torch.Tensor):
                    _account_tensor(attr.scale)
    return sum(seen_storages.values())


class Counter:
    """A simple counter class

    This class is responsible for assigning request ids to incoming requests
    """

    def __init__(self, start: int = 0) -> None:
        self.counter = start

    def __next__(self) -> int:
        i = self.counter
        self.counter += 1
        return i

    def reset(self) -> None:
        """Reset counter"""
        self.counter = 0


def get_attention_mask(seq_length: int) -> torch.Tensor:
    """Constructs an attention mask given the input sequence length."""
    attention_mask = torch.tril(
        torch.ones((1, seq_length, seq_length), device=torch.cuda.current_device())
    ).view(1, 1, seq_length, seq_length)

    # Convert to boolean
    attention_mask = attention_mask < 0.5

    return attention_mask


# Initialize cache for sequence parallel modules
moe_layer_cache = None


def _init_moe_expert_cache(model):
    """
    Initialize the cache of MoE layers once
    """
    global moe_layer_cache
    if moe_layer_cache is not None:
        return  # already initialized

    # Cache for moe layers.
    moe_layer_cache = []
    seen_modules = set()

    def walk(module):
        # Collect from MoELayer fields
        if isinstance(module, MoELayer):
            oid = id(module)
            if oid not in seen_modules:
                moe_layer_cache.append(module)

        for child in module.children():
            walk(child)

    walk(model)


def set_decode_expert_padding(model, set_to: bool = False, capacity_factor: int = None):
    """
    Toggle MoE drop-and-pad for decode.

    Applies ``capacity_factor`` to the router and all token dispatchers so
    decode runs with fixed shapes (CUDA graph-safe). When enabling
    (``set_to=True``), clears variable-size dispatcher metadata from prefill.
    For no-drop decode, use ``capacity_factor = num_moe_experts / moe_router_topk``.

    Args:
    - model: Module containing MoE layers.
    - set_to: Enable (True) or disable (False) padding.
    - capacity_factor: Capacity scaling shared by router and dispatchers.
    """
    global moe_layer_cache
    if moe_layer_cache is None:
        _init_moe_expert_cache(model)

    cfg = get_model_config(model)

    # Flip global/config knobs read by the router
    cfg.moe_pad_expert_input_to_capacity = bool(set_to)
    cfg.moe_expert_capacity_factor = capacity_factor

    # Update all token dispatchers
    for moe_layer in moe_layer_cache:

        dispatcher = moe_layer.token_dispatcher
        # turn padding on/off
        dispatcher.drop_and_pad = bool(set_to)

        # make sure attribute exists even if class didn't define it
        setattr(dispatcher, "moe_expert_capacity_factor", capacity_factor)

        # Check fliping the modules config
        if hasattr(dispatcher, "config"):
            dispatcher.config.moe_pad_expert_input_to_capacity = bool(set_to)
            dispatcher.config.moe_expert_capacity_factor = capacity_factor

        if set_to:
            # clear any variable-size metadata from dropless prefill
            for attr in (
                "input_splits",
                "output_splits",
                "output_splits_tp",
                "tokens_per_expert",
                "num_global_tokens_per_local_expert",
                "reversed_local_input_permutation_mapping",
                "capacity",
            ):
                if hasattr(dispatcher, attr):
                    setattr(dispatcher, attr, None)
            if hasattr(dispatcher, "cuda_sync_point"):
                dispatcher.cuda_sync_point = "no_sync"

        router = moe_layer.router
        setattr(router, "moe_expert_capacity_factor", capacity_factor)
        if hasattr(router, "config"):
            router.config.moe_expert_capacity_factor = capacity_factor
            router.config.moe_pad_expert_input_to_capacity = bool(set_to)


def check_flashinfer_jit_cache_installed(log_version: bool = False):
    """Verify that the flashinfer-jit-cache package is installed.

    The flashinfer-jit-cache package provides pre-compiled CUTLASS fused MoE kernels
    so they don't need to be JIT-compiled at runtime. This avoids a multi-minute
    compilation step during CUDA graph warmup.

    Raises:
        RuntimeError: If flashinfer-jit-cache is not installed and CUDA version is 12 or 13.
    """
    if FLASHINFER_JIT_CACHE_VERSION is not None:
        if log_version:
            logging.info(
                f"Found flashinfer-jit-cache {FLASHINFER_JIT_CACHE_VERSION} with "
                "pre-compiled CUTLASS kernels."
            )
        return

    cuda_major = torch.version.cuda.split(".")[0] if torch.version.cuda else None

    if cuda_major == "12":
        install_cmd = (
            "Install it with:\n\npip install flashinfer-jit-cache "
            "--index-url https://flashinfer.ai/whl/cu129\n"
        )
    elif cuda_major == "13":
        install_cmd = (
            "Install it with:\n\npip install flashinfer-jit-cache "
            "--index-url https://flashinfer.ai/whl/cu130\n"
        )
    else:
        install_cmd = ""

    raise RuntimeError(
        "The 'flashinfer-jit-cache' package is required for expert parallel inference "
        f"but is not installed. {install_cmd}"
    )


def set_inference_cuda_graphed_iteration_for_ep_inference(model):
    """Enable CUDA graph compatibility for expert parallel inference.

    Sets a flag in all MoELayers indicating the current iteration is being
    captured/executed in a CUDA graph. This allows the dispatcher to adjust
    its behavior for CUDA graph compatibility.
    """
    global moe_layer_cache
    if moe_layer_cache is None:
        _init_moe_expert_cache(model)

    for moe_layer in moe_layer_cache:
        moe_layer.set_inference_cuda_graphed_iteration()


def unset_inference_cuda_graphed_iteration_for_ep_inference(model):
    """Disable CUDA graph compatibility for expert parallel inference.

    Clears the flag in all MoELayers, restoring standard dispatcher behavior.
    """
    global moe_layer_cache
    if moe_layer_cache is None:
        _init_moe_expert_cache(model)

    for moe_layer in moe_layer_cache:
        moe_layer.unset_inference_cuda_graphed_iteration()


def tensor_swap(x, src_idxs, dst_idxs):
    """
    Swap x[src_idxs] and x[dst_idxs]
    """
    x[dst_idxs], x[src_idxs] = x[src_idxs], x[dst_idxs]


async def await_process_call(call, process: multiprocessing.Process, timeout: float = 1.0):
    """Repeatedly wait for a multiprocessing callable to resolve, aborting upon process failure.

    Note that the timeout in this function is only for checking process liveness.
    Its value should be set to a relatively high number. The only problem a high timeout
    introduces is that an error is raised slighly later.
    The timeout does not have any effect on the event-waiting, only on process failure detection.

    Args:
        event: The multiprocessing event to wait on.
        process: The process to monitor for failure.
        timeout: The timeout for each wait iteration in seconds.
    """
    while True:
        if await asyncio.to_thread(call, timeout):
            return
        if not process.is_alive():
            raise RuntimeError(
                f"Process {process.name} (pid {process.pid}) has exited unexpectedly."
            )


# Compatibility for Python < 3.13 asyncio Queue functionality.
# This is necessary because asyncio Queues are broken in Python < 3.13.
if sys.version_info < (3, 13):

    _SHUTDOWN_SENTINEL = object()

    class asyncio_QueueShutDown(Exception):
        """Compatibility exception for Python < 3.13."""

        pass

    class asyncio_Queue(asyncio.Queue):
        """An asyncio.Queue with Python 3.13 compatibility features for Python < 3.13."""

        def __init__(self, maxsize: int = 0):
            super().__init__(maxsize)
            self._is_shutdown = False

        async def get(self):
            """Get an item from the queue with Python < 3.13 compatibility."""
            if self._is_shutdown and self.empty():
                raise asyncio_QueueShutDown
            ret = await super().get()
            if ret is _SHUTDOWN_SENTINEL:
                super().put_nowait(_SHUTDOWN_SENTINEL)
                super().task_done()
                raise asyncio_QueueShutDown
            return ret

        def put_nowait(self, item):
            """Put an item into the queue without blocking"""
            if self._is_shutdown:
                raise asyncio_QueueShutDown
            if item is _SHUTDOWN_SENTINEL:
                raise ValueError(f"{item} is reserved for shutdown purposes for Python < 3.13")
            super().put_nowait(item)

        def shutdown(self):
            """Shutdown the queue for Python < 3.13.

            Note that the listening side of the queue can continue to get old data
            off the queue even after it has already been shutdown. The listener only
            shutdowns when the queue is BOTH shutdown AND empty.
            """
            if not self._is_shutdown:
                super().put_nowait(_SHUTDOWN_SENTINEL)
                super().task_done()
                self._is_shutdown = True

else:
    asyncio_QueueShutDown = asyncio.QueueShutDown
    asyncio_Queue = asyncio.Queue
