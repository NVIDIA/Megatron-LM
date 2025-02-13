# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from typing import List, Optional, Union

import torch
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

try:
    from torch.distributed._tensor import DTensor, distribute_tensor

    HAVE_DTENSOR = True
except ImportError:
    HAVE_DTENSOR = False

from .. import parallel_state
from ..transformer.moe.moe_utils import get_updated_expert_bias
from ..transformer.transformer_config import TransformerConfig
from ..utils import get_attr_wrapped_model, get_model_config


def _unshard_if_dtensor(tensor: Union[torch.Tensor, "DTensor"]) -> torch.Tensor:
    """
    Unshards the input tensor if it is a DTensor and otherwise returns the
    tensor unmodified.

    Args:
        tensor (Union[torch.Tensor, DTensor]): The tensor to potentially unshard.

    Returns:
        An unsharded version of the input tensor if it is a DTensor, or the
        input tensor unmodified if it is not a DTensor.
    """
    if HAVE_DTENSOR and isinstance(tensor, DTensor):
        unsharded_tensor = tensor.full_tensor()
        for k, v in vars(tensor).items():
            setattr(unsharded_tensor, k, v)
        return unsharded_tensor
    return tensor


def _reshard_if_dtensor(
    tensor_to_shard: torch.Tensor, reference_tensor: Union[torch.Tensor, "DTensor"]
) -> Union[torch.Tensor, "DTensor"]:
    """
    Reshards the input tensor to match the sharding configuration of the
    reference tensor if the reference tensor is a DTensor. Otherwise, returns
    the reference tensor unmodified.

    Args:
        tensor_to_shard (torch.Tensor): The tensor to be potentially sharded.
        reference_tensor (Union[torch.Tensor, DTensor]): The reference tensor
            for the sharding configuration.

    Returns:
        Union[torch.Tensor, DTensor]: The sharded tensor matching the reference tensor's
        configuration, or the reference tensor itself if it is not a DTensor.
    """
    if HAVE_DTENSOR and isinstance(reference_tensor, DTensor):
        sharded_tensor = distribute_tensor(
            tensor_to_shard,
            device_mesh=reference_tensor.device_mesh,
            placements=reference_tensor.placements,
        )
        for k, v in vars(reference_tensor).items():
            setattr(sharded_tensor, k, v)
        return sharded_tensor
    return reference_tensor


def _allreduce_conditional_embedding_grads(model: List[torch.nn.Module], config: TransformerConfig):
    """
    All-reduce conditional embedding grads.

    Reduce grads across all the pp stages to ensure that parameters of the conditional embedders
    (e.g., timestep embedder, FPS embedder, label embedder) stay in sync.
    This is for the models with replicated embedders on each PP / VPP rank, like diffusion models.
    """

    if parallel_state.get_pipeline_model_parallel_world_size() > 1 and getattr(
        config, "has_cond_embedder", False
    ):
        grads_dict = {}
        for model_chunk in model:
            for name, param in get_attr_wrapped_model(model_chunk, 'named_parameters')():
                if param.requires_grad and getattr(param, 'pipeline_parallel', False):
                    grad = param.main_grad
                    if name in grads_dict:
                        # Add all the virtual PP rank's gradients to
                        # the first local virtual PP rank.
                        grads_dict[name][0].add_(grad)
                        # Append to the end for later update after cross-rank reduce.
                        grads_dict[name].append(grad)
                    else:
                        grads_dict[name] = [grad]
        if grads_dict:
            # All-reduce the gradient on the first VPP rank.
            grads = [param_grad[0] for _, param_grad in grads_dict.items()]
            coalesced = _flatten_dense_tensors(grads)
            torch.distributed.all_reduce(
                coalesced, group=parallel_state.get_pipeline_model_parallel_group()
            )
            for buf, synced in zip(grads, _unflatten_dense_tensors(coalesced, grads)):
                buf.copy_(synced)

            # Update the gradients on other VPP ranks.
            for grads in grads_dict.values():
                for grad in grads[1:]:
                    grad.copy_(grads[0])


def _allreduce_word_embedding_grads(model: List[torch.nn.Module], config: TransformerConfig):
    """
    All-reduce word embedding grads.

    Reduce grads across first and last stages to ensure that word_embeddings parameters stay in
    sync.
    """

    if (
        parallel_state.is_rank_in_embedding_group(ignore_virtual=True)
        and torch.distributed.get_world_size(parallel_state.get_embedding_group()) > 1
    ):
        if parallel_state.is_pipeline_first_stage(ignore_virtual=True):
            model_module = model[0]
        elif parallel_state.is_pipeline_last_stage(ignore_virtual=True):
            model_module = model[-1]
        else:  # We do not support an interleaved schedule for models with encoders yet.
            model_module = model[0]

        model_module = get_attr_wrapped_model(model_module, 'pre_process', return_model_obj=True)
        if model_module.share_embeddings_and_output_weights:
            weight = model_module.shared_embedding_or_output_weight()
            grad_attr = "main_grad" if hasattr(weight, "main_grad") else "grad"
            orig_grad = getattr(weight, grad_attr)
            grad = _unshard_if_dtensor(orig_grad)
            torch.distributed.all_reduce(grad, group=parallel_state.get_embedding_group())
            setattr(weight, grad_attr, _reshard_if_dtensor(grad, orig_grad))


def _allreduce_position_embedding_grads(model: List[torch.nn.Module], config: TransformerConfig):
    """
    All-reduce position_embeddings grad across encoder and decoder stages to ensure that position
    embeddings parameters stay in sync.
    """
    if (
        parallel_state.is_rank_in_position_embedding_group()
        and torch.distributed.get_world_size(parallel_state.get_position_embedding_group()) > 1
    ):
        if parallel_state.is_pipeline_first_stage(ignore_virtual=True):
            model_module = model[0]
        elif parallel_state.is_pipeline_last_stage(ignore_virtual=True):
            model_module = model[-1]
        else:  # We do not support an interleaved schedule for models with encoders yet.
            model_module = model[0]

        model_module = get_attr_wrapped_model(model_module, 'pre_process', return_model_obj=True)
        assert hasattr(model_module, 'position_embeddings')
        weight = model_module.position_embeddings.weight
        grad_attr = "main_grad" if hasattr(weight, "main_grad") else "grad"
        orig_grad = getattr(weight, grad_attr)
        grad = _unshard_if_dtensor(orig_grad)
        torch.distributed.all_reduce(grad, group=parallel_state.get_position_embedding_group())
        setattr(weight, grad_attr, _reshard_if_dtensor(grad, orig_grad))


def _allreduce_embedding_grads(model: List[torch.nn.Module], config: TransformerConfig):
    """
    All-reduce both word and position embeddings.
    """
    _allreduce_word_embedding_grads(model, config)
    _allreduce_position_embedding_grads(model, config)


def _allreduce_layernorm_grads(model: List[torch.nn.Module], config: TransformerConfig):
    """
    All-reduce layernorm grads (for sequence parallelism).
    """

    # All-reduce layernorm parameters across model parallel nodes
    # when sequence parallelism is used
    if parallel_state.get_tensor_model_parallel_world_size() > 1 and (
        config.sequence_parallel or config.qk_layernorm
    ):
        params = []
        grads = []
        for model_chunk in model:
            for name, param in get_attr_wrapped_model(model_chunk, 'named_parameters')():
                if param.requires_grad and (
                    getattr(param, 'sequence_parallel', False)
                    or 'q_layernorm' in name
                    or 'k_layernorm' in name
                ):
                    params.append(param)
                    grad_attr = "main_grad" if hasattr(param, "main_grad") else "grad"
                    grad = getattr(param, grad_attr)
                    grad = _unshard_if_dtensor(grad)
                    grads.append(grad.data)
        if grads:
            coalesced = _flatten_dense_tensors(grads)
            torch.distributed.all_reduce(
                coalesced, group=parallel_state.get_tensor_model_parallel_group()
            )
            for param, buf, synced in zip(
                params, grads, _unflatten_dense_tensors(coalesced, grads)
            ):
                buf.copy_(synced)
                grad_attr = "main_grad" if hasattr(param, "main_grad") else "grad"
                orig_grad = getattr(param, grad_attr)
                setattr(param, grad_attr, _reshard_if_dtensor(buf, orig_grad))


def _update_router_expert_bias(model: List[torch.nn.Module], config: TransformerConfig):
    """
    Update the expert bias of the router for a global batch.
    This requires all-reduce of local_tokens_per_expert across TPxCPxDP ranks
    """
    tokens_per_expert_list = []
    expert_bias_list = []
    for model_chunk in model:
        for module in get_attr_wrapped_model(model_chunk, 'modules')():
            if hasattr(module, 'expert_bias'):
                tokens_per_expert_list.append(module.local_tokens_per_expert)
                expert_bias_list.append(module.expert_bias)
    # For hybrid models with both MoE and Dense layers, this list can be empty.
    if len(expert_bias_list) == 0:
        return
    stacked_tokens_per_expert = torch.stack(tokens_per_expert_list, dim=0)
    stacked_expert_bias = torch.stack(expert_bias_list, dim=0)
    stacked_updated_expert_bias = get_updated_expert_bias(
        stacked_tokens_per_expert, stacked_expert_bias, config.moe_router_bias_update_rate
    )

    for tokens_per_expert, expert_bias, updated_expert_bias in zip(
        tokens_per_expert_list, expert_bias_list, stacked_updated_expert_bias
    ):
        tokens_per_expert.zero_()
        expert_bias.copy_(updated_expert_bias)


def finalize_model_grads(model: List[torch.nn.Module], num_tokens: Optional[torch.Tensor] = None):
    """
    All-reduce all model grads across DP replicas, layernorm grads for sequence parallelism,
    embedding grads across first and last pipeline stages (if not tied),
    scale gradients by `num_tokens`.
    """

    config = get_model_config(model[0])

    # All-reduce / reduce-scatter across DP replicas.
    if config.timers is not None:
        config.timers('all-grads-sync', log_level=1).start(barrier=config.barrier_with_L1_time)
    for model_chunk in model:
        model_chunk.finish_grad_sync()
    if config.timers is not None:
        config.timers('all-grads-sync').stop()

    # All-reduce t_embedder grads (for pp & vpp of DiT).
    if config.timers is not None:
        config.timers('conditional-embedder-grads-all-reduce', log_level=1).start(
            barrier=config.barrier_with_L1_time
        )
    _allreduce_conditional_embedding_grads(model, config)
    if config.timers is not None:
        config.timers('conditional-embedder-grads-all-reduce').stop()

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
    _allreduce_embedding_grads(model, config)
    if config.timers is not None:
        config.timers('embedding-grads-all-reduce').stop()

    if config.moe_router_enable_expert_bias:
        _update_router_expert_bias(model, config)

    # normalize gradients for per-token loss normalization.
    # if we are using by the number of tokens, then we use that as a divisor. this number
    # will be the total number of non-padded tokens in the global batch.
    if num_tokens is not None:

        # the number of tokens is only present on the last stage, so broadcast it
        # to the other ranks in the pipeline parallel group.
        last_rank = parallel_state.get_pipeline_model_parallel_last_rank()
        pp_group = parallel_state.get_pipeline_model_parallel_group()

        if not isinstance(last_rank, list):
            assert not isinstance(last_rank, list)
            last_rank = [last_rank]
            assert not isinstance(pp_group, list)
            pp_group = [pp_group]

        # need to do a broadcast for every pp group, even though num_tokens should be the same.
        num_tokens_list = []
        for lr, group in zip(last_rank, pp_group):
            torch.distributed.broadcast(num_tokens, src=lr, group=group)
            num_tokens_list.append(torch.clone(num_tokens))
        assert all(x.item() == num_tokens_list[0] for x in num_tokens_list)

        # all-reduce across DP ranks.
        torch.distributed.all_reduce(num_tokens, group=parallel_state.get_data_parallel_group())
        for model_chunk in model:
            if num_tokens > 0:
                scaling = 1.0 / num_tokens
                model_chunk.scale_gradients(scaling)
