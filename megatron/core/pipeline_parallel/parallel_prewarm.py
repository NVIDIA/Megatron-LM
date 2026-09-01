# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Concurrent lazy-kernel initialization across pipeline stages."""

import gc
import logging
import time
from contextlib import ExitStack, nullcontext

import torch

from megatron.core.enums import Fp8Recipe
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.utils import get_attr_wrapped_model, log_on_each_pipeline_stage

logger = logging.getLogger(__name__)


def _collect_differentiable_tensors(value):
    """Flatten differentiable tensors from a nested layer output."""
    if torch.is_tensor(value):
        return [value] if value.requires_grad else []
    if isinstance(value, dict):
        tensors = []
        for item in value.values():
            tensors.extend(_collect_differentiable_tensors(item))
        return tensors
    if isinstance(value, (tuple, list)):
        tensors = []
        for item in value:
            tensors.extend(_collect_differentiable_tensors(item))
        return tensors
    return []


def _discover_local_layers(model):
    """Return local decoder and MTP layers in PP/VPP model-chunk order."""
    local_layers = []
    for model_chunk in model:
        try:
            chunk_with_decoder = get_attr_wrapped_model(
                model_chunk, 'decoder', allow_none=False, return_model_obj=True
            )
        except RuntimeError:
            continue

        for layer in chunk_with_decoder.decoder.layers:
            local_layers.append((layer, False, chunk_with_decoder))
        if hasattr(chunk_with_decoder, 'mtp'):
            for mtp_layer in chunk_with_decoder.mtp.layers:
                local_layers.append((mtp_layer.mtp_model_layer, True, chunk_with_decoder))
    return local_layers


def _get_quantization_context(layer, is_mtp):
    """Return the per-layer FP8/FP4 context used by normal model execution."""
    config = layer.config
    layer_number = -1 if is_mtp else layer.layer_number - 1
    if config.fp8:
        from megatron.core.fp8_utils import get_fp8_context

        return get_fp8_context(config, layer_number)
    if config.fp4:
        from megatron.core.fp4_utils import get_fp4_context

        return get_fp4_context(config, layer_number)
    return nullcontext()


def _build_packed_seq_params(config, kwargs):
    """Convert synthetic THD metadata tensors into eager PackedSeqParams."""
    if 'cu_seqlens_q' not in kwargs:
        return

    max_seqlen = config.max_seqlen_per_dp_cp_rank * config.context_parallel_size
    kwargs['packed_seq_params'] = PackedSeqParams(
        qkv_format='thd',
        cp_partition_mode=config.cp_partition_mode,
        cu_seqlens_q=kwargs.pop('cu_seqlens_q'),
        cu_seqlens_kv=kwargs.pop('cu_seqlens_kv'),
        cu_seqlens_q_padded=kwargs.pop('cu_seqlens_q_padded'),
        cu_seqlens_kv_padded=kwargs.pop('cu_seqlens_kv_padded'),
        max_seqlen_q=max_seqlen,
        max_seqlen_kv=max_seqlen,
        pad_between_seqs=True,
    )


def _reuse_model_chunk_request_carriers(model_chunk, kwargs, request_carrier_cache):
    """Reuse per-request state carriers across layers in one model chunk.

    Pipeline prewarm invokes local layers independently, but DSA IndexShare uses the
    request's packed-sequence metadata (or its explicit attention mask for SBHD) to
    carry top-k state from a computing layer to its sharing layers.  Reuse only those
    carriers; layer-local activations and any MTP-specific context remain independent.
    """
    chunk_carriers = request_carrier_cache.setdefault(id(model_chunk), {})
    for name in ('packed_seq_params', 'attention_mask'):
        if name not in kwargs:
            continue
        if name in chunk_carriers:
            kwargs[name] = chunk_carriers[name]
        else:
            chunk_carriers[name] = kwargs[name]


def _build_layer_inputs(
    layer,
    model_chunk,
    config,
    seq_length,
    micro_batch_size,
    rotary_pos_emb_cache,
    request_carrier_cache,
):
    """Build one eager synthetic sample for a local transformer layer."""
    if not hasattr(layer, 'get_layer_static_inputs'):
        raise TypeError(
            f'Pipeline prewarm requires {type(layer).__name__}.get_layer_static_inputs().'
        )

    static_inputs = layer.get_layer_static_inputs(
        seq_length, micro_batch_size, for_pipeline_prewarm=True
    )
    hidden_states = static_inputs.pop('hidden_states')
    kwargs = static_inputs
    _build_packed_seq_params(config, kwargs)
    _reuse_model_chunk_request_carriers(model_chunk, kwargs, request_carrier_cache)

    if (
        getattr(model_chunk, 'position_embedding_type', None) == 'rope'
        and not config.multi_latent_attention
        and hasattr(model_chunk, 'rotary_pos_emb')
    ):
        rotary_seq_len = model_chunk.rotary_pos_emb.get_rotary_seq_len(
            None, model_chunk.decoder, hidden_states, config, None
        )
        cache_key = (id(model_chunk), rotary_seq_len)
        if cache_key not in rotary_pos_emb_cache:
            rotary_pos_emb_cache[cache_key] = model_chunk.rotary_pos_emb(rotary_seq_len)
        kwargs['rotary_pos_emb'] = rotary_pos_emb_cache[cache_key]

    return (hidden_states,), kwargs


def _reset_temporary_state(model, config, optimizers):
    """Discard gradients and metric state produced by the synthetic pass."""
    from megatron.core.distributed.finalize_model_grads import reset_model_temporary_tensors
    from megatron.core.transformer.moe.moe_logging import get_moe_metrics_tracker

    for model_chunk in model:
        model_chunk.zero_grad_buffer()
    for optimizer in optimizers:
        optimizer.zero_grad()
    get_moe_metrics_tracker().clear()
    reset_model_temporary_tensors(config, model)


def prewarm_pipeline_model_parallel(
    model, config, seq_length, micro_batch_size, optimizers=(), pg_collection=None
):
    """Initialize local lazy kernels concurrently on all pipeline stages.

    Every rank executes one eager forward/backward pass for each locally owned transformer layer.
    The synthetic passes have no pipeline P2P edges, while collectives within each stage's
    TP/CP/EP groups remain matched. This function neither captures nor replays CUDA Graphs.
    """
    if getattr(config, 'moe_paged_stash', False):
        from megatron.core.transformer.moe.paged_stash import PagedStashManager

        assert (
            not PagedStashManager.get_instance().enabled
        ), "Pipeline prewarm must run before the first Paged Stash schedule."

    if pg_collection is None:
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
    local_layers = _discover_local_layers(model)

    torch.distributed.barrier()
    torch.cuda.synchronize()
    start_time = time.time()
    log_on_each_pipeline_stage(
        logger=logger,
        tp_group=pg_collection.tp,
        dp_cp_group=pg_collection.dp_cp,
        level=logging.INFO,
        msg=(
            f'Rank {torch.distributed.get_rank()}: starting pipeline-parallel prewarm for '
            f'{len(local_layers)} local layers.'
        ),
    )

    buffer_backups = []
    first_microbatch_backups = []
    saved_quantization_tensors = None
    offload_disabled = False

    from megatron.core.transformer.experimental_attention_variant.dsa import (
        DSAIndexerLossAutoScaler,
        DSAIndexerLossLoggingHelper,
    )
    from megatron.core.transformer.moe.moe_utils import MoEAuxLossAutoScaler

    dsa_tracker = DSAIndexerLossLoggingHelper.tracker
    dsa_tracker_backup = {
        key: value.clone() if torch.is_tensor(value) else value
        for key, value in dsa_tracker.items()
    }
    loss_scale_backups = (
        (DSAIndexerLossAutoScaler, DSAIndexerLossAutoScaler.main_loss_backward_scale),
        (MoEAuxLossAutoScaler, MoEAuxLossAutoScaler.main_loss_backward_scale),
    )

    layers = [layer for layer, _is_mtp, _model_chunk in local_layers]
    try:
        seen_buffers = set()
        seen_modules = set()
        for layer in layers:
            for module in layer.modules():
                module_id = id(module)
                if module_id not in seen_modules:
                    seen_modules.add(module_id)
                    if hasattr(module, 'is_first_microbatch'):
                        first_microbatch_backups.append((module, module.is_first_microbatch))
                        module.is_first_microbatch = True
                for buffer in module.buffers(recurse=False):
                    buffer_id = id(buffer)
                    if buffer_id not in seen_buffers:
                        seen_buffers.add(buffer_id)
                        buffer_backups.append((buffer, buffer.clone()))

        if config.fp8 or config.fp4:
            from transformer_engine.pytorch.graph import save_fp8_tensors

            if config.fp8:
                from megatron.core.fp8_utils import get_fp8_recipe

                recipe = get_fp8_recipe(config)
            else:
                from megatron.core.fp4_utils import get_fp4_recipe

                recipe = get_fp4_recipe(config)
            saved_quantization_tensors = save_fp8_tensors(layers, recipe)

        if config.fine_grained_activation_offloading:
            from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
                FineGrainedActivationOffloadingInterface as off_interface,
            )

            off_interface.disable_offload()
            offload_disabled = True

        from megatron.core.fp8_utils import get_fp8_context
        from megatron.core.tensor_parallel.random import _fork_rng

        outer_quantization_context = (
            get_fp8_context(config)
            if config.fp8 and config.fp8_recipe == Fp8Recipe.delayed
            else nullcontext()
        )
        rotary_pos_emb_cache = {}
        request_carrier_cache = {}

        with ExitStack() as stack:
            for model_chunk in model:
                no_sync = getattr(model_chunk, 'no_sync', None)
                if callable(no_sync):
                    stack.enter_context(no_sync())

            with _fork_rng(), outer_quantization_context:
                for layer, is_mtp, model_chunk in local_layers:
                    args, kwargs = _build_layer_inputs(
                        layer,
                        model_chunk,
                        config,
                        seq_length,
                        micro_batch_size,
                        rotary_pos_emb_cache,
                        request_carrier_cache,
                    )
                    inner_quantization_context = (
                        nullcontext()
                        if config.fp8 and config.fp8_recipe == Fp8Recipe.delayed
                        else _get_quantization_context(layer, is_mtp)
                    )
                    with inner_quantization_context:
                        outputs = layer.forward(*args, **kwargs)
                    differentiable_outputs = _collect_differentiable_tensors(outputs)
                    assert (
                        differentiable_outputs
                    ), "Pipeline prewarm must produce at least one differentiable tensor."
                    torch.autograd.backward(
                        differentiable_outputs,
                        grad_tensors=[
                            torch.zeros_like(output) for output in differentiable_outputs
                        ],
                    )
                    del args, kwargs, outputs, differentiable_outputs

        torch.cuda.synchronize()
    finally:
        if saved_quantization_tensors is not None:
            from transformer_engine.pytorch.graph import restore_fp8_tensors

            restore_fp8_tensors(layers, saved_quantization_tensors)
        with torch.no_grad():
            for buffer, backup in buffer_backups:
                buffer.copy_(backup)
        for module, is_first_microbatch in first_microbatch_backups:
            module.is_first_microbatch = is_first_microbatch
        if offload_disabled:
            off_interface.enable_offload()
        _reset_temporary_state(model, config, optimizers)
        dsa_tracker.clear()
        dsa_tracker.update(dsa_tracker_backup)
        for loss_scaler, loss_scale in loss_scale_backups:
            loss_scaler.main_loss_backward_scale = loss_scale

        del buffer_backups
        gc.collect()
        torch.cuda.empty_cache()

    elapsed = time.time() - start_time
    log_on_each_pipeline_stage(
        logger=logger,
        tp_group=pg_collection.tp,
        dp_cp_group=pg_collection.dp_cp,
        level=logging.INFO,
        msg=(
            f'Rank {torch.distributed.get_rank()}: pipeline-parallel prewarm completed '
            f'{len(local_layers)} local layers in {elapsed:.2f}s.'
        ),
    )
    torch.distributed.barrier()
