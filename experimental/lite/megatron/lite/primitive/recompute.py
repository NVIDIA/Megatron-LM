# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Model-agnostic activation recompute and offload wrappers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch  # pyright: ignore[reportMissingImports]
import torch.nn as nn  # pyright: ignore[reportMissingImports]

# ── CheckpointWithoutOutput ───────────────────────────────────────────────────
# Zero-copy C++ extension: makes dst's UntypedStorage point to src's data.
# Operates below TensorImpl level → ALL views/reshapes that share dst's StorageImpl
# (e.g. TE GroupedLinear's inp.reshape() saved for backward) see the restored data.
# Equivalent to MC's share_storage in megatron/core/tensor_parallel/random.py.
_SHARE_STORAGE_SRC = r"""
#include <torch/extension.h>

void share_storage(at::Tensor dst, at::Tensor src) {
    auto* dst_impl = dst.storage().unsafeGetStorageImpl();
    auto* src_ref  = new c10::Storage(src.storage());
    void*       data   = src_ref->data_ptr().get();
    size_t      nbytes = src_ref->nbytes();
    c10::Device device = src_ref->device();
    c10::DataPtr shared(
        data,
        static_cast<void*>(src_ref),
        [](void* ctx) { delete static_cast<c10::Storage*>(ctx); },
        device);
    dst_impl->set_data_ptr(std::move(shared));
    dst_impl->set_nbytes(nbytes);
}
"""

_share_storage_ext = None


def _get_share_storage() -> Callable:
    global _share_storage_ext
    if _share_storage_ext is None:
        from torch.utils.cpp_extension import load_inline

        _share_storage_ext = load_inline(
            name="share_storage_ext",
            cpp_sources=_SHARE_STORAGE_SRC,
            functions=["share_storage"],
            verbose=False,
        )
    return _share_storage_ext.share_storage


class _CheckpointWithoutOutputFn(torch.autograd.Function):
    """Autograd Function for CheckpointWithoutOutput.

    Forward: runs function with no_grad, saves inputs.
    Backward: uses outputs/inputs set by CheckpointWithoutOutput._recompute().
    """

    @staticmethod
    def forward(ctx, run_function, ckpt_obj, *args):
        ctx.run_function = run_function
        ctx.preserve_rng_state = ckpt_obj.preserve_rng_state
        if ckpt_obj.preserve_rng_state:
            ctx.cpu_rng_state = torch.get_rng_state()
            ctx.cuda_rng_state = torch.cuda.get_rng_state()

        ctx.tensor_indices = [i for i, a in enumerate(args) if isinstance(a, torch.Tensor)]
        ctx.non_tensor_args = [
            (i, a) for i, a in enumerate(args) if not isinstance(a, torch.Tensor)
        ]
        ctx.num_args = len(args)
        ctx.save_for_backward(*[a for a in args if isinstance(a, torch.Tensor)])

        with torch.no_grad():
            outputs = run_function(*args)

        ckpt_obj.ctx = ctx
        return outputs

    @staticmethod
    def backward(ctx, *grad_outputs):
        # inputs and outputs are set by CheckpointWithoutOutput._recompute()
        # before this backward runs (via the hook registered on the downstream tensor).
        inputs = ctx.inputs
        outputs = ctx.outputs
        torch.autograd.backward(outputs, grad_outputs)
        ctx.outputs = None
        ctx.inputs = None
        grads = tuple(inp.grad if isinstance(inp, torch.Tensor) else None for inp in inputs)
        return (None, None) + grads


class CheckpointWithoutOutput:
    """Checkpoint a function and discard its output to save memory.

    Equivalent to MC's CheckpointWithoutOutput from megatron/core/tensor_parallel/random.py.
    The output tensor's storage is freed immediately after downstream computation;
    it is recomputed just-in-time during backward via a hook on the downstream output.

    The C++ share_storage extension restores the output at the StorageImpl level so
    that ALL aliases (including views saved by TE GroupedLinear's backward) see the data.

    Usage (mirrors MC's moe_act pattern)::

        ckpt = CheckpointWithoutOutput()
        h = ckpt.checkpoint(activation_func, fc1_out, probs)
        fc2_out = fc2(h, m_splits)
        ckpt.discard_output_and_register_recompute(fc2_out)
    """

    def __init__(self, preserve_rng_state: bool = True):
        self.preserve_rng_state = preserve_rng_state
        self.run_function: Callable | None = None
        self._cpu_rng: torch.Tensor | None = None
        self._cuda_rng: torch.Tensor | None = None
        self.ctx: Any | None = None
        self.outputs: tuple[torch.Tensor, ...] | None = None

    def checkpoint(self, run_function: Callable, *args) -> Any:
        self.run_function = run_function
        if self.preserve_rng_state:
            self._cpu_rng = torch.get_rng_state()
            self._cuda_rng = torch.cuda.get_rng_state()

        outputs = _CheckpointWithoutOutputFn.apply(run_function, self, *args)
        self.outputs = (outputs,) if isinstance(outputs, torch.Tensor) else tuple(outputs)
        return outputs

    def _recompute(self, _) -> None:
        if self.ctx is None:
            return

        # Reconstruct args from saved context.
        tensors = list(self.ctx.saved_tensors)
        args: list = [None] * self.ctx.num_args
        t_it = iter(tensors)
        for i in self.ctx.tensor_indices:
            t = next(t_it)
            args[i] = t.detach().requires_grad_(t.requires_grad)
        for i, val in self.ctx.non_tensor_args:
            args[i] = val

        # Recompute with forward-time RNG states.
        if self.preserve_rng_state:
            saved_cpu = torch.get_rng_state()
            saved_cuda = torch.cuda.get_rng_state()
            torch.set_rng_state(self._cpu_rng)
            torch.cuda.set_rng_state(self._cuda_rng)
        with torch.enable_grad():
            outputs = self.run_function(*args)
        if self.preserve_rng_state:
            torch.set_rng_state(saved_cpu)
            torch.cuda.set_rng_state(saved_cuda)

        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)

        # Zero-copy: make original output's StorageImpl point to recomputed data.
        share_storage = _get_share_storage()
        for orig, new in zip(self.outputs, outputs, strict=False):
            share_storage(orig, new)

        self.ctx.outputs = list(outputs)
        self.ctx.inputs = args
        self.run_function = None
        self._cpu_rng = None
        self._cuda_rng = None
        self.outputs = None
        self.ctx = None

    def reset(self) -> None:
        """Reset state so the instance can be reused across forward passes."""
        self.run_function = None
        self._cpu_rng = None
        self._cuda_rng = None
        self.ctx = None
        self.outputs = None

    def discard_output_and_register_recompute(self, hook_tensor: torch.Tensor) -> None:
        """Free output tensor storage; recompute when hook_tensor's grad is computed."""
        for out in self.outputs:
            out.untyped_storage().resize_(0)
        if hook_tensor.requires_grad:
            hook_tensor.register_hook(self._recompute)


ModuleMap = dict[str, Callable[[nn.Module], nn.Module | None]]
"""Maps module name → lambda that extracts a sub-module from a layer."""


def apply_recompute(
    layers: nn.ModuleList,
    module_names: list[str],
    module_map: ModuleMap,
    no_rng_modules: set[str] | None = None,
) -> None:
    """Wrap specified sub-modules with activation checkpointing for recomputation."""
    if not module_names:
        return
    no_rng = no_rng_modules or set()
    for layer in layers:
        if "full" in module_names:
            wrap_checkpoint(layer)
        else:
            for mod_name in module_names:
                if mod_name in module_map:
                    submod = module_map[mod_name](layer)
                    if submod is not None:
                        wrap_checkpoint(submod, preserve_rng_state=mod_name not in no_rng)


def apply_offload(layers: nn.ModuleList, module_names: list[str], module_map: ModuleMap) -> None:
    """Wrap specified sub-modules with activation offloading to CPU."""
    if not module_names:
        return
    try:
        from torch.utils.checkpoint import CheckpointPolicy  # noqa: F401
    except ImportError:
        log_rank0("WARNING: torch.utils.checkpoint policy_fn not available, skipping offload")
        return
    for layer in layers:
        for mod_name in module_names:
            if mod_name in module_map:
                submod = module_map[mod_name](layer)
                if submod is not None:
                    wrap_offload(submod)


class CheckpointFunction(torch.autograd.Function):
    """Reentrant activation checkpoint using custom autograd.Function.

    Adapted from Megatron-Core's CheckpointFunction. Key differences:
    - No distribute_saved_activations (not needed for our TP implementation)
    - No model-parallel RNG tracker (we use TE's built-in RNG management)
    - Handles expert_bias restore for MoE router determinism
    """

    @staticmethod
    def forward(
        ctx: Any, run_function: Callable, preserve_rng_state: bool, *args: torch.Tensor
    ) -> Any:
        ctx.run_function = run_function
        ctx.preserve_rng_state = preserve_rng_state

        # Save RNG states for deterministic recomputation.
        if preserve_rng_state:
            ctx.cpu_rng_state = torch.get_rng_state()
            ctx.cuda_rng_state = torch.cuda.get_rng_state()

        # Run forward without gradient tracking — discard intermediate activations.
        with torch.no_grad():
            outputs = run_function(*args)

        # Save inputs for recomputation in backward.
        ctx.save_for_backward(*args)
        return outputs

    @staticmethod
    def backward(ctx: Any, *grad_outputs: torch.Tensor) -> tuple:
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError(
                "Checkpointing is not compatible with .grad(), use .backward() instead"
            )

        inputs = ctx.saved_tensors

        # Fork RNG: restore forward-time states, then reset to current after recompute.
        if ctx.preserve_rng_state:
            current_cpu_rng = torch.get_rng_state()
            current_cuda_rng = torch.cuda.get_rng_state()
            torch.set_rng_state(ctx.cpu_rng_state)
            torch.cuda.set_rng_state(ctx.cuda_rng_state)

        # Recompute forward pass with gradients enabled.
        detached = tuple(
            t.detach().requires_grad_(t.requires_grad) if isinstance(t, torch.Tensor) else t
            for t in inputs
        )
        with torch.enable_grad():
            outputs = ctx.run_function(*detached)

        # Restore RNG states.
        if ctx.preserve_rng_state:
            torch.set_rng_state(current_cpu_rng)
            torch.cuda.set_rng_state(current_cuda_rng)

        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)

        # Filter to outputs that need gradients.
        outputs_with_grad = []
        grad_for_outputs = []
        for out, grad in zip(outputs, grad_outputs, strict=False):
            if torch.is_tensor(out) and out.requires_grad:
                outputs_with_grad.append(out)
                grad_for_outputs.append(grad)

        if outputs_with_grad:
            torch.autograd.backward(outputs_with_grad, grad_for_outputs)

        grads = tuple(inp.grad if isinstance(inp, torch.Tensor) else None for inp in detached)
        # None for run_function, None for preserve_rng_state, then grads for each input.
        return (None, None) + grads


def wrap_checkpoint(module: nn.Module, *, preserve_rng_state: bool = True) -> None:
    """Wrap a module's forward with reentrant activation checkpointing."""
    original_forward = module.forward
    _routers = [m for m in module.modules() if hasattr(m, "expert_bias")]

    def _checkpointed_forward(*args, **kwargs):
        # expert_bias is modified in-place by the router during forward.
        # Save and restore it so the recomputation in backward sees the same values.
        if _routers:
            saved = [r.expert_bias.clone() for r in _routers]
            call_count = [0]

            def _fwd(*a, **kw):
                call_count[0] += 1
                if call_count[0] > 1:
                    for r, s in zip(_routers, saved, strict=False):
                        r.expert_bias.copy_(s)
                return original_forward(*a, **kw)

        else:
            _fwd = original_forward

        # CheckpointFunction.apply only accepts positional tensor args.
        # Wrap kwargs into the function closure.
        if kwargs:

            def _fn(*a):
                return _fwd(*a, **kwargs)

        else:
            _fn = _fwd

        return CheckpointFunction.apply(_fn, preserve_rng_state, *args)

    module.forward = _checkpointed_forward


def wrap_offload(module: nn.Module) -> None:
    """Wrap a module's forward with activation offloading."""
    original_forward = module.forward

    def _offloaded_forward(*args, **kwargs):
        return torch.utils.checkpoint.checkpoint(
            original_forward, *args, use_reentrant=False, **kwargs
        )

    module.forward = _offloaded_forward


def log_rank0(msg: str) -> None:
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        print(f"[megatron.lite] {msg}", flush=True)


def parse_recompute_spec(recompute: str | list[str] | None) -> list[str]:
    """Parse a recompute spec into a list of module names."""
    if recompute is None or recompute == "none":
        return []
    if recompute == "full":
        return ["full"]
    if isinstance(recompute, list):
        return recompute
    return recompute.split(",")


__all__ = [
    "CheckpointFunction",
    "CheckpointWithoutOutput",
    "ModuleMap",
    "apply_offload",
    "apply_recompute",
    "parse_recompute_spec",
    "wrap_checkpoint",
    "wrap_offload",
]
