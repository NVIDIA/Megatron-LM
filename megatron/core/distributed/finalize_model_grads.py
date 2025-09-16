# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from functools import partial
from typing import Callable, List, Optional, Union

import torch
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

try:
    from torch.distributed._tensor import DTensor, distribute_tensor

    HAVE_DTENSOR = True
except ImportError:
    HAVE_DTENSOR = False

from megatron.core.pipeline_parallel.utils import (
    get_pp_last_rank,
    is_pp_first_stage,
    is_pp_last_stage,
)
from megatron.core.process_groups_config import ProcessGroupCollection

from .. import parallel_state
from ..transformer.moe.moe_utils import get_updated_expert_bias
from ..transformer.transformer_config import TransformerConfig
from ..utils import (
    get_attr_wrapped_model,
    get_model_config,
    get_pg_size,
    get_tensor_model_parallel_group_if_none,
)


def _get_main_grad_attr(param: torch.nn.Parameter):
    if hasattr(param, "main_grad"):
        return "main_grad"
    return "grad"


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


def _allreduce_conditional_embedding_grads(
    model: List[torch.nn.Module],
    config: TransformerConfig,
    pp_group: Optional[torch.distributed.ProcessGroup] = None,
):
    """
    All-reduce conditional embedding grads.

    Reduce grads across all the pp stages to ensure that parameters of the conditional embedders
    (e.g., timestep embedder, FPS embedder, label embedder) stay in sync.
    This is for the models with replicated embedders on each PP / VPP rank, like diffusion models.
    """
    if pp_group is None:
        pp_group = parallel_state.get_pipeline_model_parallel_group()

    if pp_group.size() > 1 and getattr(config, "has_cond_embedder", False):
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
            torch.distributed.all_reduce(coalesced, group=pp_group)
            for buf, synced in zip(grads, _unflatten_dense_tensors(coalesced, grads)):
                buf.copy_(synced)

            # Update the gradients on other VPP ranks.
            for grads in grads_dict.values():
                for grad in grads[1:]:
                    grad.copy_(grads[0])


def _get_shared_word_embedding_weight(
    model_module: torch.nn.Module, config: TransformerConfig
) -> Optional[torch.nn.Parameter]:
    """Return the shared word-embedding weight if it is duplicated across stages.

    Args:
        model_module: The model module from which to extract the
            word-embedding weight.
        config: Transformer config.

    Returns:
        The shared embedding or output weight if available; otherwise ``None``.
    """
    # Only reduce if weights are duplicated across stages.
    if model_module.share_embeddings_and_output_weights or getattr(config, 'mtp_num_layers', 0):
        return model_module.shared_embedding_or_output_weight()
    return None


def _get_position_embedding_weight(model_module: torch.nn.Module) -> torch.nn.Parameter:
    """Return the position-embedding weight tensor from the given model module.

    Args:
        model_module: The model module that owns the
            position-embedding parameter.

    Returns:
        The position-embedding weight tensor.
    """
    return getattr(model_module, 'position_embeddings').weight  # type: ignore[attr-defined]


def _allreduce_word_embedding_grads(
    model: List[torch.nn.Module],
    config: TransformerConfig,
    embd_group: Optional[torch.distributed.ProcessGroup] = None,
    pp_group: Optional[torch.distributed.ProcessGroup] = None,
):
    """All-reduce word-embedding gradients across the first and last PP stages.

    This ensures that the ``word_embeddings`` parameters stay in sync when they
    are shared between the input and output layers.

    Args:
        model: A list containing the pipeline chunks
            that constitute the model on the current rank (including any
            virtual pipeline chunks).
        config: Transformer configuration. Used for edge
            cases like MTP where embeddings might be shared differently.
        embd_group: The process
            group over which to all-reduce the word-embedding gradients. If
            ``None``, it will be looked up based on the current pipeline model
            parallel group.
        pp_group: The pipeline
            parallel process group used to identify first/last stages. If
            ``None``, it will be looked up.
    """
    if embd_group is None:
        embd_group = parallel_state.get_embedding_group(check_initialized=False)
        if get_pg_size(embd_group) > 1:
            assert pp_group is None
            pp_group = parallel_state.get_pipeline_model_parallel_group()

    _allreduce_embedding_grad(
        model, embd_group, pp_group, partial(_get_shared_word_embedding_weight, config=config)
    )


def _allreduce_embedding_grad(
    model: List[torch.nn.Module],
    embd_group: torch.distributed.ProcessGroup,
    pp_group: torch.distributed.ProcessGroup,
    weight_getter: Callable[[torch.nn.Module], Optional[torch.nn.Parameter]],
    skip_if_none: bool = True,
):
    """Unified helper to all-reduce embedding parameters across pipeline stages.

    Args:
        model (List[torch.nn.Module]): A list of model chunks (PP/VPP).
        embd_group (torch.distributed.ProcessGroup): The process group over which to reduce.
        pp_group (torch.distributed.ProcessGroup): The pipeline parallel process group for
            first/last stage detection.
        weight_getter (Callable[[torch.nn.Module], Optional[torch.nn.Parameter]]): A function
            that takes the *pre-process* model chunk and returns the parameter to be reduced
            (or ``None`` if not applicable).
        skip_if_none (bool, optional): If True, quietly returns when the parameter or its
            gradient is ``None``. Defaults to True.
    """

    if (
        # embd_group can be None in cases there is no embd_group
        # get_pg_size(embd_group) will return 1 and the all-reduce will be skipped.
        get_pg_size(embd_group) > 1
        and torch.distributed.get_rank() in torch.distributed.get_process_group_ranks(embd_group)
    ):

        if is_pp_first_stage(pp_group):
            model_module = model[0]
        elif is_pp_last_stage(pp_group):
            model_module = model[-1]
        else:  # We do not support an interleaved schedule for models with encoders yet.
            model_module = model[0]

        ddp_config = model_module.ddp_config
        model_module = get_attr_wrapped_model(model_module, 'pre_process', return_model_obj=True)

        weight = weight_getter(model_module)
        if weight is None and skip_if_none:
            return

        grad_attr = _get_main_grad_attr(weight)
        orig_grad = getattr(weight, grad_attr)
        if ddp_config.use_megatron_fsdp:
            orig_grad = orig_grad._local_tensor if orig_grad is not None else None
        grad = _unshard_if_dtensor(orig_grad)
        # When the embedding is frozen, the grad is None.
        if grad is None and skip_if_none:
            return
        torch.distributed.all_reduce(grad, group=embd_group)
        setattr(weight, grad_attr, _reshard_if_dtensor(grad, orig_grad))


def _allreduce_position_embedding_grads(
    model: List[torch.nn.Module],
    config: TransformerConfig,
    pos_emb_group: torch.distributed.ProcessGroup,
    pp_group: torch.distributed.ProcessGroup,
):
    """
    All-reduce position_embeddings grad across encoder and decoder stages to ensure that position
    embeddings parameters stay in sync.
    """

    _allreduce_embedding_grad(
        model, pos_emb_group, pp_group, _get_position_embedding_weight, skip_if_none=False
    )


def _reset_global_aux_loss_tracker(model: List[torch.nn.Module]):
    """
    Reset the global aux loss tracker.
    """
    for model_chunk in model:
        for module in get_attr_wrapped_model(model_chunk, 'modules')():
            if hasattr(module, 'reset_global_aux_loss_tracker'):
                module.reset_global_aux_loss_tracker()


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


def _allreduce_non_tensor_model_parallel_grads(
    model: List[torch.nn.Module],
    config: TransformerConfig,
    tp_group: Optional[torch.distributed.ProcessGroup] = None,
):
    """
    All-reduce both layernorm grads (for sequence parallelism) and
    gradients from modules with average_gradients_across_tp_domain=True
    across tensor-model-parallel ranks.
    """
    tp_group = get_tensor_model_parallel_group_if_none(tp_group)
    if tp_group.size() <= 1:
        return

    params_sum = []
    grads_sum = []
    params_avg = []
    grads_avg = []

    for model_chunk in model:
        ddp_config = model_chunk.ddp_config
        for name, param in get_attr_wrapped_model(model_chunk, 'named_parameters')():
            if param.requires_grad:
                # Check if this param needs average reduction (average_gradients_across_tp_domain)
                if getattr(param, "average_gradients_across_tp_domain", False):
                    grad_attr = _get_main_grad_attr(param)
                    grad = getattr(param, grad_attr)
                    if grad is None:
                        continue
                    params_avg.append(param)
                    if ddp_config.use_megatron_fsdp:
                        grads_avg.append(grad._local_tensor.data)
                    else:
                        grad = _unshard_if_dtensor(grad)
                        grads_avg.append(grad.data)
                # Check if this param needs sum reduction (sequence parallel or qk_layernorm)
                elif (config.sequence_parallel and getattr(param, "sequence_parallel", False)) or (
                    config.qk_layernorm and ("q_layernorm" in name or "k_layernorm" in name)
                ):
                    grad_attr = _get_main_grad_attr(param)
                    grad = getattr(param, grad_attr)
                    if grad is None:
                        continue
                    params_sum.append(param)
                    if ddp_config.use_megatron_fsdp:
                        grads_sum.append(grad._local_tensor.data)
                    else:
                        grad = _unshard_if_dtensor(grad)
                        grads_sum.append(grad.data)

    # Loop grads and perform correct all-reduce
    for params, grads, all_reduce_op in zip(
        [params_sum, params_avg],
        [grads_sum, grads_avg],
        [torch.distributed.ReduceOp.SUM, torch.distributed.ReduceOp.AVG],
    ):
        if grads:
            coalesced = _flatten_dense_tensors(grads)
            torch.distributed.all_reduce(coalesced, op=all_reduce_op, group=tp_group)
            for param, buf, synced in zip(
                params, grads, _unflatten_dense_tensors(coalesced, grads)
            ):
                buf.copy_(synced)
                grad_attr = _get_main_grad_attr(param)
                orig_grad = getattr(param, grad_attr)
                if ddp_config.use_megatron_fsdp:
                    setattr(param, grad_attr, orig_grad)
                else:
                    setattr(param, grad_attr, _reshard_if_dtensor(buf, orig_grad))


"""
This is an alias to _allreduce_non_tensor_model_parallel_grads that we must
maintain for legacy tests. We can remove this proxy in mcore 0.14.
"""
_allreduce_layernorm_grads = _allreduce_non_tensor_model_parallel_grads


def finalize_model_grads(
    model: List[torch.nn.Module],
    num_tokens: Optional[torch.Tensor] = None,
    pg_collection: Optional[ProcessGroupCollection] = None,
):
    """
    All-reduce all model grads across DP replicas, layernorm grads for sequence parallelism,
    embedding grads across first and last pipeline stages (if not tied),
    scale gradients by `num_tokens`.
    """

    config = get_model_config(model[0])
    if pg_collection is not None:
        assert hasattr(pg_collection, 'tp')
        assert hasattr(pg_collection, 'pp')
        assert hasattr(pg_collection, 'embd'), (
            "pg_collection must have a embd. In previous version, it is used default "
            "`parallel_state.default_embedding_ranks` to create the process group."
            " If you are using the default process group, please use"
            " `parallel_state.get_embedding_group()` "
            "If you don't need embd_group, you need to explicitly set it to None."
        )
        assert hasattr(pg_collection, 'pos_embd'), (
            "pg_collection must have a pos_embd. In previous version, it is used default "
            "`parallel_state.default_position_embedding_ranks` to create the process group."
            " If you are using the default process group, please use "
            " `parallel_state.get_position_embedding_group()` "
            "If you don't need pos_embd_group, you need to explicitly set it to None."
        )
        assert hasattr(pg_collection, 'dp_cp')
        tp_group = pg_collection.tp
        pp_group = pg_collection.pp
        embd_group = pg_collection.embd
        pos_emb_group = pg_collection.pos_embd
        dp_cp_group = pg_collection.dp_cp
    else:
        tp_group = parallel_state.get_tensor_model_parallel_group()
        pp_group = parallel_state.get_pipeline_model_parallel_group()
        embd_group = parallel_state.get_embedding_group(check_initialized=False)
        pos_emb_group = parallel_state.get_position_embedding_group(check_initialized=False)
        dp_cp_group = parallel_state.get_data_parallel_group(with_context_parallel=True)

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
    _allreduce_conditional_embedding_grads(model, config, pp_group)
    if config.timers is not None:
        config.timers('conditional-embedder-grads-all-reduce').stop()

    # All-reduce layer-norm grads (for sequence parallelism) and non-tensor parallel modules.
    if config.timers is not None:
        config.timers('non-tensor-parallel-grads-all-reduce', log_level=1).start(
            barrier=config.barrier_with_L1_time
        )
    _allreduce_non_tensor_model_parallel_grads(model, config, tp_group)
    if config.timers is not None:
        config.timers('non-tensor-parallel-grads-all-reduce').stop()

    # All-reduce embedding grads (for pipeline parallelism).
    if config.timers is not None:
        config.timers('embedding-grads-all-reduce', log_level=1).start(
            barrier=config.barrier_with_L1_time
        )
    _allreduce_word_embedding_grads(model, config, embd_group, pp_group)
    _allreduce_position_embedding_grads(model, config, pos_emb_group, pp_group)

    if config.timers is not None:
        config.timers('embedding-grads-all-reduce').stop()

    if config.moe_router_enable_expert_bias:
        _update_router_expert_bias(model, config)

    if (
        config.moe_router_load_balancing_type == "global_aux_loss"
        or "global_aux_loss" in config.moe_router_load_balancing_type
    ):
        _reset_global_aux_loss_tracker(model)

    # normalize gradients for per-token loss normalization.
    # if we are using by the number of tokens, then we use that as a divisor. this number
    # will be the total number of non-padded tokens in the global batch.
    if num_tokens is not None:

        # the number of tokens is only present on the last stage, so broadcast it
        # to the other ranks in the pipeline parallel group.
        assert not isinstance(pp_group, list)
        last_rank = get_pp_last_rank(pp_group)
        torch.distributed.broadcast(num_tokens, src=last_rank, group=pp_group)

        # all-reduce across DP ranks.
        torch.distributed.all_reduce(num_tokens, group=dp_cp_group)
        for model_chunk in model:
            if num_tokens > 0:
                scaling = 1.0 / num_tokens
                model_chunk.scale_gradients(scaling)
