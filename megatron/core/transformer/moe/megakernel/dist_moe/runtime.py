# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Explicit post-DDP lifetime for Dist-MoE communication and activation storage."""

from __future__ import annotations

import logging
import sys
import warnings
from dataclasses import dataclass
from functools import wraps

import torch
import torch.nn.functional as F

from megatron.core.transformer.moe.megakernel.parameter_bridge import (
    finish_weight_gradient,
    main_grad_buffer,
)


@dataclass(frozen=True)
class _DistMoECall:
    """Bindings and pipeline completion belonging to one live forward."""

    weight_binding: tuple
    ticket: object = None


class _DistMoEAutograd(torch.autograd.Function):
    """Connect native MCore parameter edges to Dist's functional execution."""

    @staticmethod
    def forward(ctx, module, call, x, ids, scores, *owners):
        from dist_moe import functional

        output, state = functional.forward(
            x, ids, scores, *module._prepared, module.context, save_for_backward=call is not None
        )
        if call is not None:
            ctx.module = module
            ctx.call = call
            ctx.owners = owners
            ctx.state = state
            ctx.save_for_backward(x, ids, scores, *owners)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        from dist_moe import functional

        if torch.is_grad_enabled():
            raise RuntimeError("dist_moe supports only first-order gradients")
        module = ctx.module
        if module is None:
            raise RuntimeError("dist_moe supports one backward per forward")
        if ctx.call.ticket is not None:
            ctx.call.ticket.before_backward()
        grad_w13, grad_w2 = module._check_backward_bindings(ctx.call.weight_binding)
        # Trigger input version checks, but complete the canonical parameters:
        # saved-tensor hooks may unpack aliases without native DDP metadata.
        _ = ctx.saved_tensors
        dx, dscores = functional.backward(
            ctx.state, grad_output.contiguous(), grad_w13=grad_w13, grad_w2=grad_w2
        )
        owner_grads = tuple(finish_weight_gradient(owner) for owner in ctx.owners)
        if ctx.call.ticket is not None:
            ctx.call.ticket.complete()
        module._live -= 1
        ctx.module = ctx.state = ctx.call = ctx.owners = None
        return None, None, dx, None, dscores, *owner_grads


@dataclass(frozen=True)
class DistMoEBackendOptions:
    """Tuning knobs; dimensions and gradient ownership come from MCore."""

    num_sms: int | None = None
    device_scratch_capacity_factor: float | None = None
    activation_slot_bytes: int | None = None
    activation_slot_capacity_factor: float | None = None
    bf16_grouped_gemm_preset: str | None = None
    mxfp8_pipeline: str = "mega"
    shared_expert_bf16: bool = False


def validate_config(config):
    """Reject execution modes without an integration ownership/scheduling contract."""
    required = (
        "bf16",
        "moe_grouped_gemm",
        "moe_use_grouped_tensor",
        "moe_single_grouped_weight",
        "gradient_accumulation_fusion",
    )
    for name in required:
        if not getattr(config, name):
            raise ValueError(f"dist_moe requires {name}=True")
    unsupported = (
        "fp16",
        "fp4",
        "fp4_param",
        "first_last_layers_bf16",
        "add_bias_linear",
        "moe_apply_probs_on_input",
        "moe_pad_expert_input_to_capacity",
        "glu_linear_offset",
        "overlap_moe_expert_parallel_comm",
        "moe_shortcut_connection",
        "delay_wgrad_compute",
        "overlap_dispatch_backward_with_experts_wgrad",
    )
    for name in unsupported:
        if getattr(config, name, None):
            raise ValueError(f"dist_moe does not support {name}")
    for name in (
        "moe_mlp_glu_interleave_size",
        "activation_func_clamp_value",
        "activation_func_tanh_clamp_scale",
        "moe_latent_size",
        "recompute_granularity",
    ):
        if getattr(config, name, None) is not None:
            raise ValueError(f"dist_moe does not support {name}")
    for name in (
        "tensor_model_parallel_size",
        "expert_tensor_parallel_size",
        "context_parallel_size",
        "gtp_weight_remat_size",
        "expert_gtp_weight_remat_size",
    ):
        if getattr(config, name) != 1:
            raise ValueError(f"dist_moe requires {name}=1")
    if config.cuda_graph_impl not in ("none", "local") or (
        config.cuda_graph_impl == "local" and config.cuda_graph_modules
    ):
        raise ValueError("dist_moe supports only local whole-layer CUDA graphs")
    if config.transformer_impl == "inference_optimized":
        raise ValueError("dist_moe requires training modules")
    if config.moe_token_dispatcher_type == "flex":
        raise ValueError("dist_moe replaces the dispatcher; disable Flex/HybridEP/DeepEP")
    if not config.gated_linear_unit or config.activation_func != F.silu:
        raise ValueError("dist_moe requires ordinary SwiGLU")
    if config.fp8 is not None and (config.fp8 != "e4m3" or config.fp8_recipe != "mxfp8"):
        raise ValueError("dist_moe supports BF16 or E4M3 TE MXFP8 compute")
    if config.fp8_param and config.fp8 is None:
        raise ValueError("dist_moe FP8 parameters require MXFP8 compute")
    if config.cuda_graph_impl == "local" and config.fp8 is not None and not config.fp8_param:
        raise ValueError("dist_moe MXFP8 CUDA graphs require fp8_param=True")
    DistMoEBackendOptions(**(config.moe_megakernel_backend_config or {}))


def storage_signature(tensor):
    """Identify an ordinary tensor binding without reading its values."""
    return (tensor.data_ptr(), tensor.device, tensor.dtype, tuple(tensor.shape), tensor.stride())


_POOLS = []


@dataclass
class _RuntimePool:
    """Equal-shape layers share scratch; the scheduler sizes saved-state slots."""

    modules: list
    max_num_local_input_tokens: int
    context: object = None

    def configure(self, num_slots):
        import dist_moe

        if self.context is not None and self.context.config.num_activation_slots == num_slots:
            return
        if any(module._live for module in self.modules):
            raise RuntimeError(
                "cannot resize dist_moe activation slots with an unfinished backward"
            )
        if any(module._graph_bound for module in self.modules):
            # Evaluation and smaller schedules can borrow the existing allocation.
            # Captured slot views must keep their addresses until graphs are deleted.
            if self.context is not None and self.context.config.num_activation_slots >= num_slots:
                return
            raise RuntimeError("delete_cuda_graphs before resizing dist_moe activation slots")
        if self.context is not None:
            torch.cuda.synchronize(self.modules[0].owners[0].device)
            self.context.close()
            self.context = None
            for module in self.modules:
                module.context = None
        first = self.modules[0]
        options = first.options
        # One lease holds one chunk/microbatch, not every virtual stage together.
        depth = max(
            sum(module._chunk_index == chunk for module in self.modules)
            for chunk in {module._chunk_index for module in self.modules}
        )
        config = dist_moe.Config(
            max_num_local_input_tokens=self.max_num_local_input_tokens,
            hidden_dim=first.hidden_size,
            intermediate_dim=first.intermediate_size,
            top_k=first.topk,
            num_experts=first.config.num_moe_experts,
            max_moe_layers_per_activation_slot=depth,
            num_activation_slots=num_slots,
            num_sms=options.num_sms,
            device_scratch_capacity_factor=(
                first.ep_group.size()
                if options.device_scratch_capacity_factor is None
                else options.device_scratch_capacity_factor
            ),
            activation_slot_bytes=options.activation_slot_bytes,
            activation_slot_capacity_factor=(
                options.activation_slot_capacity_factor
                if options.activation_slot_capacity_factor is not None
                else (1.0 if options.activation_slot_bytes is None else None)
            ),
            bf16_grouped_gemm_preset=options.bf16_grouped_gemm_preset,
            block_scaled=(
                dist_moe.BlockScaledConfig(pipeline=options.mxfp8_pipeline)
                if first.use_mxfp8
                else None
            ),
            wgrad_dtype=first.owners[0].main_grad.dtype,
        )
        self.context = dist_moe.create_context(
            group=first.ep_group, config=config, device=first.owners[0].device
        )
        logging.getLogger(__name__).info(
            "Dist-MoE runtime: layers=%d, primary=%s, config=%s; %s",
            len(self.modules),
            "MXFP8" if first.config.fp8_param else "BF16",
            config,
            self.context.memory_plan.explain(),
        )
        for module in self.modules:
            module.context = self.context
            module._stream = torch.cuda.current_stream(module.owners[0].device).cuda_stream


def _backends(model_chunks):
    from .backend import DistMoEMegakernel

    chunks = [model_chunks] if isinstance(model_chunks, torch.nn.Module) else model_chunks
    return [
        module
        for chunk in chunks
        for module in chunk.modules()
        if isinstance(module, DistMoEMegakernel)
    ]


def has_dist_moe_backend(model_chunks):
    """Check whether explicit runtime initialization is needed."""
    return bool(_backends(model_chunks))


def initialize_dist_moe_runtime(model_chunks, *, ddp_config, max_num_local_input_tokens: int):
    """Collectively initialize after DDP and checkpoint/optimizer materialization.

    All EP ranks must enumerate the same layers and token bound. PP=1 creates
    its single-slot contexts immediately. PP/VPP registers native bindings here;
    its schedule later creates contexts with enough slots for live microbatches.
    """
    backends = _backends(model_chunks)
    if not backends:
        return
    owner_ids = [id(owner) for module in backends for owner in module.owners]
    if len(owner_ids) != len(set(owner_ids)):
        raise ValueError("dist_moe does not support tied expert parameters across layers")
    if ddp_config is None or getattr(ddp_config, "use_megatron_fsdp", False):
        raise ValueError("dist_moe runtime requires native MCore DDP")
    if max_num_local_input_tokens <= 0:
        raise ValueError("dist_moe requires a positive physical token bound")
    if any(module._runtime_pool is not None for module in backends):
        raise RuntimeError("dist_moe runtime is already initialized; close it before rebinding")
    chunks = [model_chunks] if isinstance(model_chunks, torch.nn.Module) else model_chunks
    for chunk_index, chunk in enumerate(chunks):
        for module in _backends(chunk):
            module._chunk_index = chunk_index
    pools = {}
    for module in backends:
        if module.config.fp8_param and not ddp_config.reuse_grad_buf_for_mxfp8_param_ag:
            raise ValueError("dist_moe FP8 parameters require reuse_grad_buf_for_mxfp8_param_ag")
        grads = tuple(main_grad_buffer(param) for param in module.owners)
        if grads[0].dtype != grads[1].dtype:
            raise ValueError("dist_moe FC1 and FC2 main_grad must have the same dtype")
        module._gradient_binding = tuple(storage_signature(grad) for grad in grads)
        key = (
            module.ep_group,
            module.hidden_size,
            module.intermediate_size,
            module.topk,
            module.num_local_experts,
            module.owners[0].device,
            module.use_mxfp8,
            grads[0].dtype,
            module.options,
        )
        pools.setdefault(key, []).append(module)
    for modules in pools.values():
        pool = _RuntimePool(modules, max_num_local_input_tokens)
        _POOLS.append(pool)
        for module in modules:
            module._runtime_pool = pool
        if modules[0].config.pipeline_model_parallel_size == 1:
            pool.configure(1)


def configure_dist_moe_activation_slots(model_chunks, num_slots):
    """Materialize/reconfigure drained contexts once the PP schedule is known."""
    seen = set()
    for module in _backends(model_chunks):
        pool = module._runtime_pool
        if pool is None:
            raise RuntimeError("call initialize_dist_moe_runtime before the pipeline schedule")
        if id(pool) not in seen:
            capacity = num_slots[id(pool)] if isinstance(num_slots, dict) else num_slots
            if capacity < 1:
                raise ValueError("dist_moe requires at least one activation slot")
            pool.configure(capacity)
            seen.add(id(pool))


def close_dist_moe_runtime(model_chunks=None):
    """Synchronize and release idle contexts before process-group teardown."""
    selected = None if model_chunks is None else set(_backends(model_chunks))
    for pool in list(_POOLS):
        context, modules = pool.context, pool.modules
        if selected is not None and not selected.intersection(modules):
            continue
        if any(module._live for module in modules):
            raise RuntimeError("cannot close dist_moe with an unfinished backward")
        if any(module._graph_bound for module in modules):
            raise RuntimeError("delete_cuda_graphs before closing the dist_moe runtime")
        if context is not None:
            torch.cuda.synchronize(modules[0].owners[0].device)
            context.close()
        for module in modules:
            module.context = None
            module._runtime_pool = None
            module._pipeline_schedule = None
            module._weight_cache = module._prepared = module._weight_binding = None
            module.is_first_microbatch = True
        _POOLS.remove(pool)


def dist_moe_runtime_lifecycle(function):
    """Close initialized runtimes when the training entrypoint exits."""

    @wraps(function)
    def wrapped(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        finally:
            training_failed = sys.exc_info()[0] is not None
            try:
                if any(module._graph_bound for pool in _POOLS for module in pool.modules):
                    from megatron.core.transformer.cuda_graphs import delete_cuda_graphs

                    delete_cuda_graphs()
                close_dist_moe_runtime()
            except RuntimeError as error:
                if not training_failed:
                    raise
                # Do not mask the training failure or retry a potentially failed
                # GPU invocation. Its process owns teardown of the live context.
                warnings.warn(f"dist_moe cleanup after training failure: {error}")

    return wrapped
