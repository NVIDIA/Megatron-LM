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

from megatron.core.transformer.moe.megakernel.parameter_bridge import main_grad_buffer


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
        "moe_shared_expert_overlap",
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
        "virtual_pipeline_model_parallel_size",
        "recompute_granularity",
    ):
        if getattr(config, name, None) is not None:
            raise ValueError(f"dist_moe does not support {name}")
    for name in (
        "tensor_model_parallel_size",
        "expert_tensor_parallel_size",
        "context_parallel_size",
        "pipeline_model_parallel_size",
        "gtp_weight_remat_size",
        "expert_gtp_weight_remat_size",
    ):
        if getattr(config, name) != 1:
            raise ValueError(f"dist_moe requires {name}=1")
    if config.cuda_graph_impl != "none":
        raise ValueError("dist_moe does not support CUDA graphs")
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
    DistMoEBackendOptions(**(config.moe_megakernel_backend_config or {}))


def storage_signature(tensor):
    """Identify an ordinary tensor binding without reading its values."""
    return (tensor.data_ptr(), tensor.device, tensor.dtype, tuple(tensor.shape), tensor.stride())


_CONTEXTS = []


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

    All EP ranks must enumerate the same layers and token bound. The supported
    PP=1 schedule completes each microbatch backward before starting the next.
    Equal-shape layers share one context and its stack of saved activations.
    """
    backends = _backends(model_chunks)
    if not backends:
        return
    owner_ids = [id(owner) for module in backends for owner in module.owners]
    if len(owner_ids) != len(set(owner_ids)):
        raise ValueError("dist_moe does not support tied expert parameters across layers")
    import dist_moe

    if ddp_config is None or getattr(ddp_config, "use_megatron_fsdp", False):
        raise ValueError("dist_moe runtime requires native MCore DDP")
    if max_num_local_input_tokens <= 0:
        raise ValueError("dist_moe requires a positive physical token bound")
    pools = {}
    for module in backends:
        if module.context is not None:
            raise RuntimeError("dist_moe runtime is already initialized; close it before rebinding")
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
        first = modules[0]
        options = first.options
        config = dist_moe.Config(
            max_num_local_input_tokens=max_num_local_input_tokens,
            hidden_dim=first.hidden_size,
            intermediate_dim=first.intermediate_size,
            top_k=first.topk,
            num_experts=first.config.num_moe_experts,
            max_moe_layers_per_activation_slot=len(modules),
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
        context = dist_moe.create_context(
            group=first.ep_group, config=config, device=first.owners[0].device
        )
        logging.getLogger(__name__).info(
            "Dist-MoE runtime: layers=%d, primary=%s, config=%s; %s",
            len(modules),
            "MXFP8" if first.config.fp8_param else "BF16",
            config,
            context.memory_plan.explain(),
        )
        _CONTEXTS.append((context, modules))
        for module in modules:
            module.context = context
            module._stream = torch.cuda.current_stream(module.owners[0].device).cuda_stream


def close_dist_moe_runtime(model_chunks=None):
    """Synchronize and release idle contexts before process-group teardown."""
    selected = None if model_chunks is None else set(_backends(model_chunks))
    for context, modules in list(_CONTEXTS):
        if selected is not None and not selected.intersection(modules):
            continue
        if any(module._live for module in modules):
            raise RuntimeError("cannot close dist_moe with an unfinished backward")
        torch.cuda.synchronize(modules[0].owners[0].device)
        context.close()
        for module in modules:
            module.context = None
            module._weight_cache = module._prepared = None
            module.is_first_microbatch = True
        _CONTEXTS.remove((context, modules))


def dist_moe_runtime_lifecycle(function):
    """Close initialized runtimes when the training entrypoint exits."""

    @wraps(function)
    def wrapped(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        finally:
            if sys.exc_info()[0] is None:
                close_dist_moe_runtime()
            else:
                # Do not mask the training failure or retry a potentially failed
                # GPU invocation. Its process owns teardown of the live context.
                try:
                    close_dist_moe_runtime()
                except RuntimeError as error:
                    warnings.warn(f"dist_moe cleanup after training failure: {error}")

    return wrapped
