# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Attach Core adapters to a model before data-parallel wrapping."""

from functools import partial

import torch
from torch import nn

from megatron.core.distributed import finalize_model_grads
from megatron.core.transformer.module import mark_keep_in_fp32
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.peft.adapters import LinearAdapter, ParallelLinearAdapter
from megatron.core.transformer.peft.expert_adapters import GroupedExpertLinearAdapter
from megatron.core.transformer.peft.grad_sync import (
    enable_expert_parallel_grad_sync_in_finalize,
    finalize_model_grads_with_expert_adapter_sync,
)
from megatron.core.transformer.peft.linear_attributes import (
    align_expert_dim_for_tp,
    get_adapter_attributes_from_linear,
    get_effective_lora_dim,
    is_expert_linear,
    is_grouped_expert_linear,
)
from megatron.core.transformer.peft.lora import (
    LoRALinear,
    LoRALinearSplitFC1UpGate,
    LoRALinearSplitQKV,
    ModuleDict,
)
from megatron.core.transformer.peft.wrappers import AdapterWrapper

from .config import LoRA
from .matching import matches, selected_components


def _wrap(module, path, components, config, pg_collection):
    if isinstance(module, TopKRouter):
        raise NotImplementedError("Router LoRA is outside this migration")
    if isinstance(module, nn.Linear):
        adapter = LinearAdapter(
            module,
            dim=config.dim,
            alpha=config.alpha,
            dropout=config.dropout,
            dropout_position=config.dropout_position,
            lora_A_init_method=config.lora_A_init_method,
            lora_dtype=config.lora_dtype,
        )
        ParallelLinearAdapter._get_init_fn(
            adapter, config.lora_A_init_method, module.in_features, config.dim
        )(adapter.linear_in.weight)
        ParallelLinearAdapter._get_init_fn(
            adapter, config.lora_B_init_method, config.dim, module.out_features
        )(adapter.linear_out.weight)
        return LoRALinear(module, adapter, pg_collection=pg_collection)
    expert = is_expert_linear(path)
    attrs = get_adapter_attributes_from_linear(
        module, is_expert=expert, pg_collection=pg_collection
    )
    dim = get_effective_lora_dim(
        module, dim=config.dim, normalize_moe_lora=config.normalize_moe_lora, is_expert=expert
    )
    dim = align_expert_dim_for_tp(
        module,
        dim,
        normalize_moe_lora=config.normalize_moe_lora,
        is_expert=expert,
        input_is_parallel=attrs.input_is_parallel,
        pg_collection=pg_collection,
    )
    per_expert = is_grouped_expert_linear(path) and not config.share_expert_adapters
    adapter_type = GroupedExpertLinearAdapter if per_expert else ParallelLinearAdapter
    kwargs = dict(
        dim=dim,
        base_linear_name=path,
        activation="identity",
        alpha=config.alpha,
        column_init_method=config.lora_A_init_method,
        row_init_method=config.lora_B_init_method,
        input_is_parallel=attrs.input_is_parallel,
        dropout=config.dropout,
        dropout_position=config.dropout_position,
        model_parallel_config=module.config,
        base_linear_is_parallel=attrs.base_linear_is_parallel,
        pg_collection=pg_collection,
    )
    if per_expert:
        param = next(module.parameters())
        kwargs.update(
            num_local_experts=module.num_gemms,
            params_device=param.device,
            params_dtype=config.lora_dtype or param.dtype,
        )
    else:
        kwargs.update(
            is_expert=expert,
            disable_tensor_parallel_comm=attrs.disable_tensor_parallel_comm,
            disable_sequence_parallel_comm=attrs.disable_sequence_parallel_comm,
            replicate_adapter=attrs.replicate_adapter,
        )

    def make(out_features, name=path):
        adapter = adapter_type(
            attrs.in_features, out_features, **{**kwargs, "base_linear_name": name}
        )
        return adapter.to(dtype=config.lora_dtype) if config.lora_dtype is not None else adapter

    leaf = path.rsplit(".", 1)[-1]
    if leaf == "linear_qkv" and config.canonical_qkv_projections:
        groups = module.config.num_query_groups
        if groups < pg_collection.tp.size():
            raise NotImplementedError(
                "Canonical QKV adapters require at least one query group per TP rank"
            )
        kv_size = module.config.kv_channels * groups
        q_size = module.config.kv_channels * module.config.num_attention_heads
        adapters = ModuleDict(
            {
                "adapter_q": make(q_size) if "linear_q" in components else None,
                "adapter_k": make(kv_size) if "linear_k" in components else None,
                "adapter_v": make(kv_size) if "linear_v" in components else None,
            }
        )
        return LoRALinearSplitQKV(module, adapters)
    if leaf == "linear_fc1" and config.canonical_mlp_projections:
        if expert or not module.config.gated_linear_unit:
            raise NotImplementedError(
                "Canonical gate/up adapters currently require a dense gated MLP"
            )
        # Each adapter already represents one half; do not split its checkpoint tensor again.
        adapters = ModuleDict(
            {
                "adapter_gate": (
                    make(attrs.out_features // 2, path + ".gate")
                    if "linear_fc1_gate" in components
                    else None
                ),
                "adapter_up": (
                    make(attrs.out_features // 2, path + ".up")
                    if "linear_fc1_up" in components
                    else None
                ),
            }
        )
        for adapter in adapters.values():
            if adapter is not None:
                adapter.split_output_for_checkpoint = False
        return LoRALinearSplitFC1UpGate(module, adapters)
    return LoRALinear(module, make(attrs.out_features))


def apply_lora(model, config: LoRA, *, pg_collection, training=True):
    """Freeze a built model and attach fresh adapters, respecting current recipe settings."""
    config.validate()
    chunks = model if isinstance(model, list) else [model]
    matched = set()
    for chunk in chunks:
        if any(isinstance(module, AdapterWrapper) for module in chunk.modules()):
            raise ValueError("LoRA has already been applied to this model")
        declared = getattr(chunk, "peft_excluded_module_paths", lambda: ())()
        if isinstance(declared, str):
            raise TypeError("peft_excluded_module_paths must return a collection of paths")
        excluded = tuple(declared)
        chunk.requires_grad_(False)

        def visit(parent, prefix=""):
            for name, module in list(parent.named_children()):
                path = f"{prefix}.{name}" if prefix else name
                if any(path == item or path.startswith(item + ".") for item in excluded):
                    continue
                if any(matches(pattern, path) for pattern in config.exclude_modules):
                    continue
                components, patterns = selected_components(config, path)
                if components:
                    wrapper = _wrap(module, path, components, config, pg_collection)
                    if config.lora_dtype == torch.float32:
                        for parameter in wrapper.adapter.parameters():
                            mark_keep_in_fp32(parameter)
                    setattr(parent, name, wrapper)
                    matched.update(patterns)
                else:
                    visit(module, path)

        visit(chunk)
        chunk.train(training)
        if not training:
            chunk.requires_grad_(False)
    # Pipeline stages need not contain every selected projection. Report absent targets
    # across the full model separately; never reject a valid stage with no local adapters.
    if not matched and (
        pg_collection is None
        or getattr(pg_collection, "pp", None) is None
        or pg_collection.pp.size() == 1
    ):
        raise ValueError(f"No modules matched LoRA targets {config.target_modules}")
    if enable_expert_parallel_grad_sync_in_finalize(chunks):
        transformers = {id(chunk.config): chunk.config for chunk in chunks}
        for transformer in transformers.values():
            if transformer.finalize_model_grads_func not in (None, finalize_model_grads):
                raise ValueError("Shared expert LoRA cannot replace a custom gradient finalizer")
            transformer.finalize_model_grads_func = partial(
                finalize_model_grads_with_expert_adapter_sync, pg_collection=pg_collection
            )
    if training:
        from .recompute import enable_recompute_input_grad

        enable_recompute_input_grad(chunks)
    return model
