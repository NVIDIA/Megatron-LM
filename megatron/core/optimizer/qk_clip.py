# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""QK-Clip call flow for standard and CPU-offloaded optimizers.

Training loop:
    prepare_qk_clip(model, optimizer)
        -> Reduce per-head logits across DP/CP and register attention layers.
    optimizer.step()
    clear_qk_clip(optimizer)
        -> Clear statistics and pending tasks, including on skipped updates.

Standard optimizer:
    prepare_qk_clip registers layers without resolving master weights.
    step_with_ready_grads:
        optimizer.step() -> apply_qk_clip(wrapper)
            -> Resolve updated FP32 masters and local shard ranges.
            -> Scale masters in place before model parameter copy-back/all-gather.

CPU offload (HybridDeviceOptimizer, distributed optimizer required):
    prepare_qk_clip -> _prepare_qk_clip_main_params
        -> Resolve CPU/GPU masters and move row factors to their devices.
    HDO.step(): each sub-optimizer updates its masters, then its copy-back hook:
        apply_qk_clip(hdo, sub_optimizer=...) -> copy parameters back to GPU.
    The outer apply_qk_clip(wrapper) detects HDO and skips duplicate clipping.

"""

import torch

from megatron.core import parallel_state


def _iter_qk_clip_modules(model_chunk):
    """Yield attention modules that need QK clipping.

    Preserve the legacy traversal for other model types. HybridModel may own
    nested MTP stacks, so discover its attention modules recursively.
    """
    # Import lazily to avoid optimizer/attention circular imports.
    from megatron.core.models.hybrid.hybrid_model import HybridModel
    from megatron.core.transformer.attention import Attention
    model_module = model_chunk.module.module
    if isinstance(model_module, HybridModel):
        yield from (
            module
            for module in model_module.modules()
            if isinstance(module, Attention) and hasattr(module, 'get_qk_clip_factors')
        )
        return

    for transformer_layer in model_module.decoder.layers:
        if hasattr(transformer_layer.self_attention, 'get_qk_clip_factors'):
            yield transformer_layer.self_attention


@torch.no_grad()
def prepare_qk_clip(model, optimizer=None, *, log_max_only: bool = False) -> float:
    """Reduce logits and register layers before stepping the optimizer.

    Args:
        model: Wrapped model chunks containing attention layers.
        optimizer: Megatron optimizer, required when clipping is enabled.
        log_max_only: Collect statistics without preparing weight updates.

    Returns:
        Maximum attention logit across local layers and heads after DP/CP reduction.
    """
    if not log_max_only and optimizer is None:
        raise ValueError("QK-Clip requires an optimizer for clipping before parameter copy-back.")
    dp_cp_group = parallel_state.get_data_parallel_group(with_context_parallel=True)
    layers = []
    log_max_attention_logit = 0.0
    seen = set()
    for model_chunk in model:
        for attention in _iter_qk_clip_modules(model_chunk):
            if id(attention) in seen:
                continue
            seen.add(id(attention))
            logits = attention.core_attention.current_max_attn_logits
            if logits is None:
                continue
            torch.distributed.all_reduce(
                logits, op=torch.distributed.ReduceOp.MAX, group=dp_cp_group
            )
            if not torch.isfinite(logits).all():
                raise ValueError("Non-finite attention logits detected (NaN or Inf).")
            log_max_attention_logit = max(log_max_attention_logit, logits.max().item())
            if log_max_only:
                attention.core_attention.current_max_attn_logits = None
            else:
                layers.append(attention)
    if not log_max_only:
        for child in _optimizers(optimizer):
            child._qk_clip_layers = layers
            if not layers or getattr(child, 'is_stub_optimizer', False):
                continue

            if _is_hybrid_optimizer(child.optimizer):
                _prepare_qk_clip_main_params(child)
    return log_max_attention_logit


@torch.no_grad()
def _prepare_qk_clip_main_params(optimizer):
    """Resolve masters and factors: before HDO updates, after standard updates."""
    hybrid = _is_hybrid_optimizer(optimizer.optimizer)
    target = optimizer.optimizer if hybrid else optimizer
    target._qk_clip_shards = {}
    layers = getattr(optimizer, '_qk_clip_layers', [])
    if not layers:
        return
    distributed = hasattr(optimizer, 'model_param_gbuf_map')
    if hybrid and not distributed:
        raise ValueError('QK-Clip with CPU offload currently requires the distributed optimizer.')
    if not distributed:
        owned = {id(p) for group in optimizer.optimizer.param_groups for p in group['params']}
    for layer in layers:
        for param, row_factor in layer.get_qk_clip_factors():
            if not param.requires_grad:
                continue
            if distributed:
                if param not in optimizer.model_param_gbuf_map:
                    continue  # No shard on this DP rank / sub-optimizer.
                span = optimizer._get_model_param_range_map(param)['param']
                start = span.start
                if hybrid:
                    # Resolve anew each step: checkpoint loading may rebuild HDO masters.
                    group, index = optimizer.model_param_group_index_map[param]
                    original = target.param_groups[group]['params'][index]
                    master = target.param_to_inner_param[original]
                elif param.dtype == torch.float32:
                    master = param.view(-1)[span.start : span.end]
                else:
                    master = getattr(param, 'main_param', None)
                if master is None or master.numel() != span.end - span.start:
                    raise ValueError(
                        'QK-Clip requires an explicit main-weight shard matching its range.'
                    )
            else:
                start = 0
                master = getattr(
                    param, 'main_param', param if param.dtype == torch.float32 else None
                )
                if master is None or id(master) not in owned:
                    continue
            if master.dtype != torch.float32 or not master.is_contiguous():
                raise ValueError('QK-Clip requires a contiguous FP32 master parameter.')
            # Blocking transfer makes CPU factors ready before HDO copy-back hooks.
            factors = row_factor.detach().to(device=master.device, dtype=torch.float32)
            target._qk_clip_shards[master] = (factors, start, param.shape[1])


def _optimizers(optimizer):
    for child in getattr(optimizer, 'chained_optimizers', [optimizer]):
        if child is optimizer:
            yield child
        else:
            yield from _optimizers(child)


def _is_hybrid_optimizer(optimizer):
    """Recognize HDO by its master mappings, without a circular import of its class."""
    return hasattr(optimizer, 'param_to_inner_param') and hasattr(
        optimizer, 'gpu_params_map_cpu_copy'
    )


@torch.no_grad()
def apply_qk_clip(optimizer, *, sub_optimizer=None):
    """Clip updated masters; HDO wrappers skip clipping handled by internal hooks."""
    hybrid = _is_hybrid_optimizer(optimizer)
    if hybrid:
        if sub_optimizer is None:
            raise ValueError('HDO QK-Clip requires the just-updated sub_optimizer.')
    else:
        if getattr(optimizer, 'is_stub_optimizer', False):
            return
        inner = getattr(optimizer, 'optimizer', None)
        if inner is None:
            raise TypeError('QK-Clip requires a Megatron optimizer wrapper.')
        if _is_hybrid_optimizer(inner):
            # HDO clips each sub-optimizer's masters before its parameter copy-back.
            return
        _prepare_qk_clip_main_params(optimizer)
    master_param_to_clip_info = getattr(optimizer, '_qk_clip_shards', {})
    if not master_param_to_clip_info:
        return
    if hybrid:
        # Other HDO sub-optimizers may not have updated their parameters yet.
        main_params = (p for group in sub_optimizer.param_groups for p in group['params'])
    else:
        main_params = list(master_param_to_clip_info)
    for main_param in main_params:
        clip_info = master_param_to_clip_info.pop(main_param, None)
        if clip_info is not None:
            row_factor, start, row_width = clip_info
            _apply_qk_clip_main_param(main_param, row_factor, start, row_width)


@torch.no_grad()
def _apply_qk_clip_main_param(master, row_factor, start, row_width):
    """Scale a contiguous FP32 shard on its own device, without full-size factors."""
    if master.dtype != torch.float32 or not master.is_contiguous():
        raise ValueError('QK-Clip requires a contiguous FP32 master shard.')
    if row_width <= 0 or start < 0 or start + master.numel() > row_factor.numel() * row_width:
        raise ValueError('QK-Clip shard range exceeds its row factors.')
    factors = row_factor.detach().to(device=master.device, dtype=master.dtype).reshape(-1)
    flat = master.view(-1)
    row, column = divmod(start, row_width)
    offset = 0
    # First partial row, full rows, then final partial row.
    if column and flat.numel():
        size = min(row_width - column, flat.numel())
        flat[:size].mul_(factors[row])
        offset = size
        row += 1
    rows = (flat.numel() - offset) // row_width
    if rows:
        size = rows * row_width
        flat[offset : offset + size].view(rows, row_width).mul_(factors[row : row + rows, None])
        offset += size
        row += rows
    if offset < flat.numel():
        flat[offset:].mul_(factors[row])


def clear_qk_clip(optimizer):
    """Reset statistics only after all sub-optimizers finish (also on skipped steps)."""
    for child in _optimizers(optimizer):
        for layer in getattr(child, '_qk_clip_layers', []):
            layer.core_attention.current_max_attn_logits = None
        child._qk_clip_layers = []
        target = child.optimizer if _is_hybrid_optimizer(child.optimizer) else child
        target._qk_clip_shards = {}
