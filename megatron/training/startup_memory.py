# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# License for these changes: BSD-3-Clause.

"""Opt-in allocator preparation before data workers and initial gradient finalization."""

from __future__ import annotations

import gc
import logging
import random
from dataclasses import replace


def _clone_tracker_states(states: dict) -> dict:
    """Clone tensor-backed and graph-safe generator-backed tracker states."""
    import torch

    cloned = {}
    for key, state in states.items():
        if isinstance(state, torch.Generator):
            cloned[key] = state.clone_state()
        elif isinstance(state, torch.Tensor):
            cloned[key] = state.clone()
        else:
            raise TypeError(f'Unsupported CUDA RNG tracker state: {type(state)}')
    return cloned


def _tracker_state_bytes(state):
    import torch

    return state.get_state() if isinstance(state, torch.Generator) else state


def prewarm_gdp_kernels(models: list, sequence_length: int, micro_batch_size: int) -> bool:
    """Warm a separate native BF16 GDP mixer before starting data-loader workers.

    Original parameters, existing gradients, precision and all RNG streams are
    preserved. The temporary mixer uses the original process-group collection;
    only its gradient-accumulation fusion is disabled because it has no optimizer
    main-gradient buffers. Unsupported precision/parallelism raises explicitly.
    """
    import numpy as np
    import torch

    from megatron.core.models.hybrid.hybrid_layer_specs import gdp_stack_spec
    from megatron.core.ssm.gated_delta_product import GatedDeltaProductMixer
    from megatron.core.tensor_parallel.random import get_cuda_rng_tracker
    from megatron.core.transformer.module import Float16Module

    template = next(
        (
            module
            for model in models
            for module in model.modules()
            if isinstance(module, GatedDeltaProductMixer)
        ),
        None,
    )
    if template is None:
        return False
    config = template.config
    if (
        config.tensor_model_parallel_size != 1
        or config.context_parallel_size != 1
        or config.pipeline_model_parallel_size != 1
        or config.dynamic_context_parallel
        or not config.bf16
        or config.fp8
        or config.fp4
        or config.cuda_graph_impl not in (None, 'none')
    ):
        raise ValueError('GDP prewarming requires eager unquantized BF16 TP1/CP1/PP1.')
    if template.pg_collection.tp.size() != 1 or template.pg_collection.cp.size() != 1:
        raise ValueError('GDP prewarming requires actual singleton TP/CP module groups.')
    if sequence_length <= 0 or micro_batch_size <= 0:
        raise ValueError('GDP prewarming requires positive sequence and microbatch sizes.')
    parameters = [parameter for model in models for parameter in model.parameters()]
    watched = [
        value
        for parameter in parameters
        for value in (parameter, parameter.grad, getattr(parameter, 'main_grad', None))
        if value is not None
    ]
    versions = [value._version for value in watched]
    storage = {value.untyped_storage().data_ptr() for value in watched}
    gradients = [
        (parameter.grad, getattr(parameter, 'main_grad', None)) for parameter in parameters
    ]
    tracker = get_cuda_rng_tracker()
    original_tracked = tracker.get_states()
    warmup_tracked = _clone_tracker_states(original_tracked)
    tracked_bytes = {
        key: _tracker_state_bytes(state).clone() for key, state in original_tracked.items()
    }
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state().clone()
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    precision = torch.get_float32_matmul_precision()
    temporary = value = output = None
    try:
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            tracker.set_states(warmup_tracked)
            temporary_config = replace(config, gradient_accumulation_fusion=False)
            temporary = GatedDeltaProductMixer(
                temporary_config,
                gdp_stack_spec.submodules.mamba_layer.submodules.mixer.submodules,
                config.hidden_size,
                layer_number=1,
                pg_collection=template.pg_collection,
            ).cuda()
            temporary = Float16Module(temporary_config, temporary).module
            if any(
                parameter.untyped_storage().data_ptr() in storage
                for parameter in temporary.parameters()
            ):
                raise RuntimeError('Temporary GDP parameters alias original model or gradients.')
            with torch.enable_grad(), torch.autocast(device_type='cuda', enabled=False):
                value = torch.randn(
                    sequence_length,
                    micro_batch_size,
                    config.hidden_size,
                    device='cuda',
                    dtype=torch.bfloat16,
                    requires_grad=True,
                )
                output, _ = temporary(value)
                output.backward(torch.zeros_like(output))
                torch.cuda.synchronize()
    finally:
        temporary = value = output = None
        tracker.set_states(original_tracked)
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        torch.set_float32_matmul_precision(precision)
    gc.collect()
    torch.cuda.empty_cache()
    assert torch.equal(torch.get_rng_state(), cpu_rng)
    assert torch.equal(torch.cuda.get_rng_state(), cuda_rng)
    restored = tracker.get_states()
    assert set(restored) == set(original_tracked)
    assert all(restored[key] is state for key, state in original_tracked.items())
    assert all(
        torch.equal(state, _tracker_state_bytes(restored[key]))
        for key, state in tracked_bytes.items()
    )
    assert all(value._version == version for value, version in zip(watched, versions))
    assert all(
        parameter.grad is grad and getattr(parameter, 'main_grad', None) is main_grad
        for parameter, (grad, main_grad) in zip(parameters, gradients)
    )
    logging.getLogger(__name__).info("GDP prewarm preserved RNG and model state")
    return True
