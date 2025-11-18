# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import asyncio
import multiprocessing

import torch

from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.utils import get_model_config


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


def tensor_swap(x, src_idxs, dst_idxs):
    """
    Swap x[src_idxs] and x[dst_idxs]
    """
    x[dst_idxs], x[src_idxs] = x[src_idxs], x[dst_idxs]


async def await_process_event(
    event: multiprocessing.Event, process: multiprocessing.Process, timeout: float = 1.0
) -> None:
    """Repeatedly wait for a multiprocessing event to be set, aborting upon process failure.

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
        signal = await asyncio.to_thread(event.wait, timeout)
        if signal:
            return
        if not process.is_alive():
            raise RuntimeError(
                f"Process {process.name} (pid {process.pid}) has exited unexpectedly."
            )
