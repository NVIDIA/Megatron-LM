# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from typing import List

import torch
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

from .. import parallel_state
from ..transformer.transformer_config import TransformerConfig
from ..utils import get_attr_wrapped_model, get_model_config
from megatron import get_args


embedding_grad_counter = 0

def _allreduce_word_embedding_grads(model: List[torch.nn.Module], config: TransformerConfig, async_op=False):
    """
    All-reduce word embedding grads.

    Reduce grads across first and last stages to ensure that word_embeddings parameters stay in
    sync. This should only run for models that support pipelined model parallelism (BERT and GPT).
    """
    handles = []
    ignore_virtual = not get_args().zero_bubble_v_schedule
    if parallel_state.is_rank_in_embedding_group(ignore_virtual=ignore_virtual):
        if parallel_state.is_pipeline_first_stage(ignore_virtual=ignore_virtual):
            model_module = model[0]
        elif parallel_state.is_pipeline_last_stage(ignore_virtual=ignore_virtual):
            model_module = model[-1]
        else:  # We do not support the interleaved schedule for T5 yet.
            assert ignore_virtual
            model_module = model[0]

        # Look for module with 'pre_process' attribute to get around the fact that DDP and
        # other wrapper classes inherit from non-core MegatronModule that has
        # 'share_embeddings_and_output_weights' and 'shared_embedding_or_output_weight'
        # attributes already, causing get_attr_wrapped_model() to not unwrap anything here.
        # TODO: Clean this up once the wrapper classes inherit from core MegatronModule.
        model_module = get_attr_wrapped_model(model_module, 'pre_process', return_model_obj=True)
        if model_module.share_embeddings_and_output_weights:
            weight = model_module.shared_embedding_or_output_weight()
            grad = weight.main_grad
            if get_args().zero_bubble_v_schedule:
                from megatron.model.module import local_binary_reduction
                global embedding_grad_counter
                local_binary_reduction(grad.data, key=f"embedding_grads_{int(embedding_grad_counter // 2)}")
                embedding_grad_counter += 1
            else:
                handle = torch.distributed.all_reduce(grad, group=parallel_state.get_embedding_group(), async_op=async_op)
                handles.append(handle)
    return handles


def _allreduce_position_embedding_grads(model: List[torch.nn.Module], config: TransformerConfig, async_op=False):
    """
    All-reduce position_embeddings grad across first (encoder) and split (decoder) stages to
    ensure that position embeddings parameters stay in sync. This should only run for T5 models
    with pipeline parallelism.
    """
    handles = []
    if (
        parallel_state.is_rank_in_position_embedding_group()
        and parallel_state.get_pipeline_model_parallel_world_size() > 1
        and config.pipeline_model_parallel_split_rank is not None
    ):
        model_module = model[0]
        grad = get_attr_wrapped_model(
            model_module, 'language_model.embedding.position_embeddings.weight.main_grad'
        )
        handle = torch.distributed.all_reduce(grad, group=parallel_state.get_position_embedding_group(), async_op=async_op)
        handles.append(handle)
    return handles


def _allreduce_embedding_grads(model: List[torch.nn.Module], config: TransformerConfig, async_op=False):
    """
    All-reduce both word and position embeddings.
    """
    handles = _allreduce_word_embedding_grads(model, config, async_op=async_op)
    handles += _allreduce_position_embedding_grads(model, config, async_op=async_op)
    return handles


def _allreduce_layernorm_grads(model: List[torch.nn.Module], config: TransformerConfig):
    """
    All-reduce layernorm grads (for sequence parallelism).
    """

    # All-reduce layernorm parameters across model parallel nodes
    # when sequence parallelism is used
    if parallel_state.get_tensor_model_parallel_world_size() > 1 and config.sequence_parallel:
        grads = []
        for model_chunk in model:
            for param in get_attr_wrapped_model(model_chunk, 'parameters')():
                if getattr(param, 'sequence_parallel', False):
                    grad = param.main_grad
                    grads.append(grad.data)
        coalesced = _flatten_dense_tensors(grads)
        torch.distributed.all_reduce(
            coalesced, group=parallel_state.get_tensor_model_parallel_group()
        )
        for buf, synced in zip(grads, _unflatten_dense_tensors(coalesced, grads)):
            buf.copy_(synced)


def _allreduce_expert_grads(model: List[torch.nn.Module], config: TransformerConfig):
    """
    All-reduce expert grads (for expert parallelism).
    """

    # All-reduce switchmlp parameters across data modulo expert parallel nodes
    if (
        config.expert_model_parallel_size > 1
        and config.expert_model_parallel_size < parallel_state.get_data_parallel_world_size()
    ):
        grads = []
        for model_chunk in model:
            for param in get_attr_wrapped_model(model_chunk, 'parameters')():
                if not getattr(param, 'allreduce', True):
                    grad = param.main_grad
                    grads.append(grad.data)
        coalesced = _flatten_dense_tensors(grads)
        torch.distributed.all_reduce(
            coalesced, group=parallel_state.get_data_modulo_expert_parallel_group()
        )
        for buf, synced in zip(grads, _unflatten_dense_tensors(coalesced, grads)):
            buf.copy_(synced)


def finalize_model_grads(model: List[torch.nn.Module]):
    """
    All-reduce all model grads across DP replicas, layernorm grads for sequence parallelism,
    embedding grads across first and last pipeline stages (if not tied), and expert grads
    for expert parallelism.
    """

    config = get_model_config(model[0])

    # All-reduce / reduce-scatter across DP replicas.
    if config.timers is not None:
        config.timers('all-grads-sync', log_level=1).start(barrier=config.barrier_with_L1_time)
    for model_chunk in model:
        model_chunk.finish_grad_sync()
    if config.timers is not None:
        config.timers('all-grads-sync').stop()

    # All-reduce layer-norm grads (for sequence parallelism).
    if config.timers is not None:
        config.timers('layernorm-grads-all-reduce', log_level=1).start(
            barrier=config.barrier_with_L1_time
        )
    _allreduce_layernorm_grads(model, config)
    if config.timers is not None:
        config.timers('layernorm-grads-all-reduce').stop()

    # All-reduce embedding grads (for pipeline parallelism).
    if config.timers is not None:
        config.timers('embedding-grads-all-reduce', log_level=1).start(
            barrier=config.barrier_with_L1_time
        )
    if not get_args().enable_zero_bubble:
        # For zero bubble schedules, we do async all-reduce for embedding grads
        # in WeightGradStore.clear() so that it won't generate bubbles
        _allreduce_embedding_grads(model, config)
    if config.timers is not None:
        config.timers('embedding-grads-all-reduce').stop()

    # All-reduce expert grads (for expert parallelism).
    if config.timers is not None:
        config.timers('expert-grads-all-reduce', log_level=1).start(
            barrier=config.barrier_with_L1_time
        )
    _allreduce_expert_grads(model, config)
    if config.timers is not None:
        config.timers('expert-grads-all-reduce').stop()
