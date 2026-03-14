# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

"""Forward activation logging using forward hooks."""

from collections import defaultdict
import os
from typing import Callable, List, Tuple

import torch
import torch.nn as nn

from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.moe.router import Router

from .checkpointing import save_grads
from .utils import unwrap_model


def _discover_te_types():
    """Discover available Transformer Engine layer types.

    Returns (all_types, grouped_types) where grouped_types is the subset of
    TEGroupedLinear variants used for tokens-per-expert capture.
    """
    all_types = []
    grouped_types = []

    try:
        from megatron.core.extensions.transformer_engine import (
            TELinear,
            TENorm,
            TEColumnParallelLinear,
            TERowParallelLinear,
            TELayerNormColumnParallelLinear,
        )
        all_types.extend([TELinear, TENorm, TEColumnParallelLinear, TERowParallelLinear,
                          TELayerNormColumnParallelLinear])
    except ImportError:
        pass

    try:
        from megatron.core.extensions.transformer_engine import (
            TEGroupedLinear,
            TEColumnParallelGroupedLinear,
            TERowParallelGroupedLinear,
        )
        if TEGroupedLinear is not None:
            grouped = [TEGroupedLinear, TEColumnParallelGroupedLinear,
                       TERowParallelGroupedLinear]
            all_types.extend(grouped)
            grouped_types.extend(grouped)
    except ImportError:
        pass

    return tuple(all_types), tuple(grouped_types)


_TE_TYPES, _GROUPED_LINEAR_TYPES = _discover_te_types()

LINEAR_TYPES = (nn.Linear, nn.Embedding, ColumnParallelLinear, RowParallelLinear,
                Router, *_TE_TYPES)


def _register_hooks(model, module_types, hook_factory, *, name_filter=None):
    """Walk *model* and register a forward hook on every module matching *module_types*.

    Args:
        model: Iterable of model chunks (possibly wrapped).
        module_types: Tuple of types to match via ``isinstance``.
        hook_factory: ``(model_chunk_name, module_name) -> hook_fn``.
        name_filter: Optional ``str -> bool`` predicate on the module name.

    Returns:
        List of hook handles.
    """
    handles = []
    for model_chunk_id, model_chunk in enumerate(model):
        model_chunk_name = f"model_chunk{model_chunk_id}"
        unwrapped = unwrap_model(model_chunk)
        for module_name, module in unwrapped.named_modules():
            if isinstance(module, module_types) and (name_filter is None or name_filter(module_name)):
                handle = module.register_forward_hook(
                    hook_factory(model_chunk_name, module_name),
                    with_kwargs=True,
                )
                handles.append(handle)
    return handles

class ActivationLogger:
    """Captures and saves forward activations using forward hooks.

    Manages two independent hook sets:

    - **Full activation hooks** capture all inputs / outputs / kwargs for every
      ``LINEAR_TYPES`` module.
    - **Tokens-per-expert (TPE) hooks** are lightweight hooks that only capture
      the tokens-per-expert routing metadata from MoE.
    """

    def __init__(self, save_dir: str):
        self._save_dir = save_dir

        # Full activation state.
        self._activations_state_dict: defaultdict = defaultdict(dict)
        self._activation_hooks: List[torch.utils.hooks.RemovableHook] = []

        # Tokens-per-expert state.
        self._tpe_records: List[Tuple[str, object]] = []
        self._tpe_hooks: List[torch.utils.hooks.RemovableHook] = []

    # ------------------------------------------------------------------
    # Full activation hooks
    # ------------------------------------------------------------------

    def _make_activation_hook(self, model_chunk_name: str, module_name: str) -> Callable:
        """Forward hook that captures all inputs, outputs and kwargs."""
        sd = self._activations_state_dict

        def hook(_, args, kwargs, output):
            input_tuple = args if isinstance(args, tuple) else (args,)
            for idx, inp in enumerate(input_tuple):
                if inp is None:
                    continue
                key = f"{module_name}/input{idx}"
                sd[model_chunk_name][key] = inp.detach().cpu() if isinstance(inp, torch.Tensor) else inp
            for idx, out in enumerate(output if isinstance(output, tuple) else (output,)):
                if out is not None and isinstance(out, torch.Tensor):
                    sd[model_chunk_name][f"{module_name}/output{idx}"] = out.detach().cpu()
            for kwarg_key, kwarg_value in kwargs.items():
                key = f"{module_name}/{kwarg_key}"
                sd[model_chunk_name][key] = (
                    kwarg_value.detach().cpu() if isinstance(kwarg_value, torch.Tensor) else kwarg_value
                )

        return hook

    def register_activation_hooks(self, model):
        assert not self._activation_hooks
        self._activation_hooks = _register_hooks(model, LINEAR_TYPES, self._make_activation_hook)

    def remove_activation_hooks(self):
        for hook in self._activation_hooks:
            hook.remove()
        self._activation_hooks.clear()

    def save_activations(self, iteration: int):
        if not self._activations_state_dict:
            return
        save_grads(self._save_dir, self._activations_state_dict, iteration, "activations")
        self._activations_state_dict.clear()

    # ------------------------------------------------------------------
    # Tokens-per-expert hooks
    # ------------------------------------------------------------------

    def _make_tpe_hook(self, _model_chunk_name: str, module_name: str) -> Callable:
        """Forward hook that captures only the non-Tensor ``input1`` (tokens_per_expert).

        The layer number is extracted from *module_name*
        (e.g. ``decoder.layers.3.mlp.experts.linear_fc1`` → ``3``).
        """
        parts = module_name.split(".")
        layer = parts[2] if len(parts) > 2 else module_name

        def hook(_, args, kwargs, output):
            input_tuple = args if isinstance(args, tuple) else (args,)
            if len(input_tuple) > 1 and input_tuple[1] is not None:
                inp = input_tuple[1]
                if not isinstance(inp, torch.Tensor):
                    self._tpe_records.append((layer, inp))

        return hook

    def register_tpe_hooks(self, model):
        assert not self._tpe_hooks
        self._tpe_hooks = _register_hooks(
            model, _GROUPED_LINEAR_TYPES, self._make_tpe_hook,
            name_filter=lambda name: name.endswith("linear_fc1"),
        )

    def remove_tpe_hooks(self):
        for hook in self._tpe_hooks:
            hook.remove()
        self._tpe_hooks.clear()

    def save_tpe(self, iteration: int):
        """Append captured tokens-per-expert records to a shared plain text file.
        All ranks append to a single file ``{save_dir}/tokens_per_expert.log``.
        Lines are formatted as::
            Rank={rank} iter={iter} layer={layer} tokens_per_expert={list}
        """
        if not self._tpe_records:
            return
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        os.makedirs(self._save_dir, exist_ok=True)
        filepath = os.path.join(self._save_dir, "tokens_per_expert.log")
        lines = "".join(
            f"Rank={rank} iter={iteration} layer={layer} tokens_per_expert={inp}\n"
            for layer, inp in self._tpe_records
        )
        with open(filepath, "a") as f:
            f.write(lines)
        self._tpe_records.clear()

_LOGGER: ActivationLogger | None = None


def _get_logger(save_dir: str) -> ActivationLogger:
    global _LOGGER
    if _LOGGER is None:
        _LOGGER = ActivationLogger(save_dir)
    return _LOGGER


def _require_logger() -> ActivationLogger:
    assert _LOGGER is not None, "No ActivationLogger has been initialised"
    return _LOGGER


# -- Full activation logging -------------------------------------------

def enable_activation_logging(model: torch.nn.Module, save_dir: str):
    _get_logger(save_dir).register_activation_hooks(model)


def disable_activation_logging():
    _require_logger().remove_activation_hooks()


def save_activations(iteration: int):
    _require_logger().save_activations(iteration)


# -- Tokens-per-expert logging ----------------------------------------

def enable_tokens_per_expert_logging(model: torch.nn.Module, save_dir: str):
    _get_logger(save_dir).register_tpe_hooks(model)


def disable_tokens_per_expert_logging():
    _require_logger().remove_tpe_hooks()


def save_tokens_per_expert(iteration: int):
    _require_logger().save_tpe(iteration)
