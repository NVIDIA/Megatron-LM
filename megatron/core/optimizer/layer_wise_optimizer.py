# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
import re
from typing import Any, Callable, Tuple

import torch
from absl import logging
from emerging_optimizers import utils
from emerging_optimizers.orthogonalized_optimizers import muon

from megatron.core import optimizer as mcore_optimizer
from megatron.core.models.gpt import gpt_model as mcore_gpt_model
from megatron.core.optimizer import Adam as AdamW
from megatron.core.optimizer import clip_grads


def default_params_group_func(
    mcore_te_model: mcore_gpt_model.GPTModel,
) -> Tuple[list[torch.nn.Parameter], ...]:
    """Default parameter group function for LayerWiseDistributedOptimizer.

    It is designed to be used with Megatron-core models with transformer engine backend. Filtering
    is strongly based on layer names.
    """
    # Strip out module inside DDP and other wrappers
    module = mcore_te_model.module if hasattr(mcore_te_model, "module") else mcore_te_model
    while hasattr(module, "module"):
        module = module.module
    assert hasattr(module, "decoder"), "MCore GPTModel should have a decoder"

    attn_linear_params_list = []
    mlp_linear_params_list = []
    norm_params_list = []
    # Also collect layer names for debugging purposes
    non_ep_linear_names = []
    ep_linear_names = []
    for name, param in module.decoder.named_parameters():
        # NOTE: Megatron-core uses allreduce to indicate experts MLP layers that are in EP, not
        # actually needs allreduce. And not all of the parameters have the "allreduce" attribute
        # patched.
        is_ep_mlp = getattr(param, "allreduce", False)
        if "norm" not in name and re.search(r".weight(\d*)$", name):
            if not is_ep_mlp:
                attn_linear_params_list.append(param)
                non_ep_linear_names.append(name.replace(".weight", ""))
            else:
                mlp_linear_params_list.append(param)
                ep_linear_names.append(name.replace(".weight", ""))
        elif "norm" in name:
            norm_params_list.append(param)
        else:
            raise ValueError(f"Unsupported parameter: {name}")
    logging.info(f"{len(attn_linear_params_list)} attention linear layers collected in total.")
    logging.info(f"{len(mlp_linear_params_list)} MLP linear layers collected in total.")
    logging.debug(f"Non EP linear layer names: {non_ep_linear_names}")
    logging.debug(f"EP linear layer names: {ep_linear_names}")

    # TODO(skyw): Revisit whether we should put positional embedding on separate GPU or together
    # with word embedding
    embed_params_list = []
    if hasattr(module, "embedding"):
        embed_params_list.append(module.embedding.word_embeddings.weight)

        assert not hasattr(
            module.embedding, "token_type_embeddings"
        ), "Token type embedding is \
            not supported"
        assert not hasattr(
            module.embedding, "position_embeddings"
        ), "Traditional position embedding is not supported in favor of RoPE"

    if hasattr(module, "output_layer"):
        embed_params_list.append(module.output_layer.weight)

    collected_num_params = (
        len(attn_linear_params_list)
        + len(mlp_linear_params_list)
        + len(norm_params_list)
        + len(embed_params_list)
    )
    total_num_params = sum(1 for _ in module.parameters())
    assert (
        total_num_params == collected_num_params
    ), "Total number of parameters should be equal to the sum of linear, norm and embedding \
        parameters"

    # NOTE: Moving to param_groups is WIP. The goal is to pass necessary information to the
    # optimizer.
    linear_param_groups = [{"params": attn_linear_params_list}, {"params": mlp_linear_params_list}]
    return linear_param_groups, norm_params_list, embed_params_list


def group_params_layer_wise(
    linear_params_list: list[torch.nn.Parameter],
    optimizer_wrapper: type[torch.optim.Optimizer],
    linear_optimizer_cls: type[torch.optim.Optimizer],
    linear_opt_kwargs: dict,
    wrapper_args: list[Any],
    group: torch.distributed.ProcessGroup,
) -> tuple[torch.optim.Optimizer, list[list[torch.nn.Parameter]]]:
    """Group parameters layer-wise and distribute to different ranks.

    Args:
        linear_params_list: List of linear parameters to be distributed.
        optimizer_wrapper: Optimizer wrapper class.
        linear_optimizer_cls: Linear optimizer class.
        linear_opt_kwargs: Keyword arguments for linear optimizer.
        wrapper_args: Arguments for optimizer wrapper.
        group: Process group.

    Returns:
        A tuple of the layer-wise optimizer and the list of broadcast parameters.
    """
    layer_wise_optimizer = None
    broadcast_params_list = []
    num_linear_params_per_rank = math.ceil(len(linear_params_list) / group.size())
    for i, offset in enumerate(range(0, len(linear_params_list), num_linear_params_per_rank)):
        local_linear_params_list = linear_params_list[offset : offset + num_linear_params_per_rank]
        broadcast_params_list.append(local_linear_params_list)
        if group.rank() == i:
            layer_wise_optimizer = optimizer_wrapper(
                linear_optimizer_cls(local_linear_params_list, **linear_opt_kwargs), *wrapper_args
            )
            logging.info(f"Rank {group.rank()} has {num_linear_params_per_rank} layers")

    assert (
        l := len(broadcast_params_list)
    ) <= group.size(), (
        f"Broadcast params list ({l}) should be smaller than world size ({group.size()})"
    )

    return layer_wise_optimizer, broadcast_params_list


class LayerWiseDistributedOptimizer(mcore_optimizer.ChainedOptimizer):
    """Layer-wise distributed optimizer for Megatron-core models.

    Warning:
        This is a experimental optimizer that distributes optimizers of linear and embedding layers 
        to different ranks with the option to use different optimizer for different layer types.
        Generic tensor parallelism support is still work in progress.

    Args:
        module: A megatron core model chunk.
        linear_optimizer_cls: Optimizer class for linear layers.
        opt_kwargs_list: List of keyword arguments for different optimizer types. Must have same 
            length as what returned by params_group_func.
        mcore_optimizer_config: Megatron-core optimizer configuration.
        pg_collection: A collection of process groups.
            It is currently implemented as a dictionary with keys matching mcore convention. It will
            be updated to mcore ProcessGroupCollection after release of Mcore r0.14.1.
        params_group_func: A function that returns a list of parameters iterables.
    """

    def __init__(
        self,
        module: torch.nn.Module,
        linear_optimizer_cls: type[torch.optim.Optimizer],
        opt_kwargs_list: list[dict],
        mcore_optimizer_config: mcore_optimizer.OptimizerConfig,
        pg_collection: dict[str, torch.distributed.ProcessGroup],
        params_group_func: (
            Callable[[torch.nn.Module], Tuple[list[torch.nn.Parameter], ...]] | None
        ) = None,
    ) -> None:
        if isinstance(module, list):
            assert len(module) == 1, "LayerWiseDistributedOptimizer only supports one model chunk"
            module = module[0]
        assert len(opt_kwargs_list) == 3, "LayerWiseDistributedOptimizer requires 3 optimizer types"
        assert all(
            isinstance(opt_kwargs, dict) for opt_kwargs in opt_kwargs_list
        ), "All optimizer kwargs must be dicts"
        linear_opt_kwargs, embed_opt_kwargs, default_opt_kwargs = opt_kwargs_list

        # TODO(skyw): Add consistency check for optimizer kwargs and optimizer config.
        self.dp_cp_group = pg_collection.get("dp_cp", None)
        self.expt_dp_group = pg_collection.get("expt_dp", None)
        assert (
            self.dp_cp_group is not None
        ), f"dp_cp group is required for {self.__class__.__name__}"
        logging.debug(f"dp_cp_group : {self.dp_cp_group}, expt_dp_group: {self.expt_dp_group}")

        if params_group_func is None:
            params_group_func = default_params_group_func
        linear_param_groups, norm_params_list, embed_params_list = params_group_func(module)
        logging.info(
            f"Collected {len(linear_param_groups[0]['params'])} attention linear parameters and "
            f"{len(linear_param_groups[1]['params'])} MLP linear parameters "
            f"and {len(norm_params_list)} norm parameters "
            f"and {len(embed_params_list)} embedding parameters."
        )

        if mcore_optimizer_config.bf16:
            optimizer_wrapper = mcore_optimizer.Float16OptimizerWithFloat16Params
            wrapper_args = [mcore_optimizer_config, None, None]
        elif not mcore_optimizer_config.fp16:
            optimizer_wrapper = mcore_optimizer.FP32Optimizer
            wrapper_args = [mcore_optimizer_config, None]
        else:
            raise ValueError("Unsupported optimizer precision")

        self.is_moe = module.config.num_moe_experts is not None

        non_moe_linear_params_list = []
        moe_linear_params_list = []
        # If MoE is used, attention and MLP layers will have different DP groups.
        if not self.is_moe:
            non_moe_linear_params_list = (
                linear_param_groups[0]["params"] + linear_param_groups[1]["params"]
            )
        else:
            non_moe_linear_params_list = linear_param_groups[0]["params"]
            moe_linear_params_list = linear_param_groups[1]["params"]

        non_moe_layer_wise_optimizer, self.non_moe_broadcast_params_list = group_params_layer_wise(
            non_moe_linear_params_list,
            optimizer_wrapper,
            linear_optimizer_cls,
            linear_opt_kwargs,
            wrapper_args,
            self.dp_cp_group,
        )
        if self.is_moe:
            moe_layer_wise_optimizer, self.moe_broadcast_params_list = group_params_layer_wise(
                moe_linear_params_list,
                optimizer_wrapper,
                linear_optimizer_cls,
                linear_opt_kwargs,
                wrapper_args,
                self.expt_dp_group,
            )
        else:
            moe_layer_wise_optimizer = None
            self.moe_broadcast_params_list = []

        if len(embed_params_list) > 0:
            embed_optimizer = optimizer_wrapper(
                AdamW(embed_params_list, **embed_opt_kwargs), *wrapper_args
            )
        else:
            embed_optimizer = None
        norm_optimizer = optimizer_wrapper(
            AdamW(norm_params_list, **default_opt_kwargs), *wrapper_args
        )
        assert norm_optimizer, "Norm optimizer should not be empty."

        self.has_non_moe_layer_wise_opt = False
        self.has_moe_layer_wise_opt = False
        all_optimizers = []
        if non_moe_layer_wise_optimizer is not None:
            all_optimizers.append(non_moe_layer_wise_optimizer)
            self.has_non_moe_layer_wise_opt = True
        if moe_layer_wise_optimizer is not None:
            all_optimizers.append(moe_layer_wise_optimizer)
            self.has_moe_layer_wise_opt = True
        if norm_optimizer is not None:
            all_optimizers.append(norm_optimizer)
        if embed_optimizer is not None:
            all_optimizers.append(embed_optimizer)

        super().__init__(all_optimizers)

    @torch.no_grad()
    def step(self):  # type: ignore[no-untyped-def]
        """step function for layer-wise optimizer

        Note:
            A lot of code from ChainedOptimizer.step() is copied here because there are bug fixes 
            after core 0.12.
        """
        found_inf_flag = self.prepare_grads()
        if found_inf_flag:
            return False, None, None

        grad_norm = self.get_grad_norm()

        # Clip gradients.
        for optimizer in self.chained_optimizers:
            if hasattr(optimizer, "is_stub_optimizer") and optimizer.is_stub_optimizer:
                continue

            if optimizer.config.clip_grad > 0.0:
                clip_grads.clip_grad_by_total_norm_fp32(
                    optimizer.get_parameters(),
                    max_norm=optimizer.config.clip_grad,
                    total_norm=grad_norm,
                    use_decoupled_grad=optimizer.config.use_precision_aware_optimizer,
                )

        # # Count the zeros in the grads.
        num_zeros_in_grad = self.count_zeros() if self.config.log_num_zeros_in_grad else None

        update_successful = self.step_with_ready_grads()

        # Broadcast linear layer weights to all other ranks.
        # This may not be slower than PyTorch allgatherv which calls broadcast internally.
        # TODO(skyw): Profile and implement more efficient version.
        self.broadcast_params(self.non_moe_broadcast_params_list, self.dp_cp_group)
        if self.is_moe and utils.get_pg_size(self.expt_dp_group) > 1:
            self.broadcast_params(self.moe_broadcast_params_list, self.expt_dp_group)

        return update_successful, grad_norm, num_zeros_in_grad

    def broadcast_params(
        self, params_list: list[list[torch.nn.Parameter]], group: torch.distributed.ProcessGroup
    ) -> None:
        """Broadcast parameters to all other ranks."""
        for i, params_list in enumerate(params_list):
            src_global_rank = torch.distributed.get_global_rank(group, i)
            for param in params_list:
                torch.distributed.broadcast(param, src_global_rank, group)

    def get_grad_norm(self) -> float:
        """Get the gradient norm of the optimizer"""
        opt_offset = 0
        if self.has_non_moe_layer_wise_opt:
            non_moe_layer_wise_grad_norm = self.chained_optimizers[opt_offset].get_grad_norm()
            opt_offset += 1
        else:
            non_moe_layer_wise_grad_norm = 0.0
        if self.has_moe_layer_wise_opt:
            moe_layer_wise_grad_norm = self.chained_optimizers[opt_offset].get_grad_norm()
            opt_offset += 1
        else:
            moe_layer_wise_grad_norm = 0.0
        non_moe_out_list = [0 for _ in range(utils.get_pg_size(self.dp_cp_group))]
        moe_out_list = [0 for _ in range(utils.get_pg_size(self.expt_dp_group))]

        # TODO(skyw): Determine whether it is too big of an overhead to use all_gather_object.
        torch.distributed.all_gather_object(
            non_moe_out_list, non_moe_layer_wise_grad_norm**2, group=self.dp_cp_group
        )
        if self.is_moe:
            torch.distributed.all_gather_object(
                moe_out_list, moe_layer_wise_grad_norm**2, group=self.expt_dp_group
            )

        # Add the gradient norm of the norm layers which is duplicated among all ranks.
        duplicated_grad_square_sum = 0.0
        for optimizer in self.chained_optimizers[opt_offset:]:
            duplicated_grad_square_sum += optimizer.get_grad_norm() ** 2
        final_norm = math.sqrt(
            sum(non_moe_out_list) + sum(moe_out_list) + duplicated_grad_square_sum
        )

        return final_norm

    def save_state_dict_to_file(self, filename: str) -> None:
        """Save the parameter state of the optimizer.

        Args:
            filename: The filename to save the parameter state.
        """
        torch.save(super().state_dict(), filename)

    def load_state_dict_from_file(self, filename: str) -> None:
        """Load the parameter state of the optimizer."""
        super().load_state_dict(torch.load(filename))


def get_shower_optimizer_for_mcore(
    model: torch.nn.Module,
    config: mcore_optimizer.OptimizerConfig,
    linear_optimizer: str = "muon",
    split_qkv: bool = False,
    **kwargs: Any,
) -> LayerWiseDistributedOptimizer:
    """Get a layer-wise optimizer for a MCore GPTModel.

    Warning:
        Megatron monkey patches a lot of attributes to variety of objects.
        This function tries to encapsulate fragile functionalities that depends on those fragile 
        patches and isolate them from the main LayerWiseDistributedOptimizer class.

    Args:
        model: A MCore GPTModel.
        config: A Megatron-core optimizer configuration.
        linear_optimizer: The linear optimizer to use. Must be one of ['soap', 'muon'].
        split_qkv: Whether to split QKV tensors into separate Q, K, V tensors. Defaults to False.
        **kwargs: Additional arguments for LayerWiseOptimizer.

    Returns:
        A LayerWiseOptimizer.
    """
    linear_optimizer_cls: type[torch.optim.Optimizer]
    if linear_optimizer == "soap":
        raise NotImplementedError("SOAP is not supported yet.")
    elif linear_optimizer == "muon":
        transformer_config = model.config if not isinstance(model, list) else model[0].config
        qkv_split_shapes = (
            # Note: We need to check this when TP is used with GQA/MHSA.
            transformer_config.kv_channels * transformer_config.num_attention_heads,
            transformer_config.kv_channels * transformer_config.num_query_groups,
            transformer_config.kv_channels * transformer_config.num_query_groups,
        )
        fused_qkv_shape = torch.Size([sum(qkv_split_shapes), transformer_config.hidden_size])

        def is_qkv_fn(x: torch.Tensor) -> bool:
            """Check if a tensor is a fused QKV tensor by shape.

            Checks all tensors that is a parameter of the model and has the same shape as defined by
             `fused_qkv_shape`.
            Note: Other tensors might have the same shape but are not fused QKV tensors and this 
                function will still return True.

            Args:
                x: tensor to check.

            Returns:
                True if the tensor is a fused QKV tensor only by shape `fused_qkv_shape`, False 
                otherwise.
            """
            return x.shape == fused_qkv_shape

        linear_optimizer_cls = muon.Muon
        linear_opt_kwargs = dict(
            lr=config.lr,
            weight_decay=getattr(config, "muon_linear_wd", config.weight_decay),
            split_qkv=split_qkv,
            is_qkv_fn=is_qkv_fn,
            qkv_split_shapes=qkv_split_shapes,
        )
        muon_config_keys = [
            "momentum_beta",
            "num_ns_steps",
            "scale_mode",
            "extra_scale_factor",
            "fp32_matmul_prec",
            "use_nesterov",
            "use_decoupled_weight_decay",
        ]
        for attr_name in muon_config_keys:
            prefixed_attr_name = f"muon_{attr_name}"
            if hasattr(config, prefixed_attr_name):
                linear_opt_kwargs[attr_name] = getattr(config, prefixed_attr_name)
            elif hasattr(config, attr_name):
                linear_opt_kwargs[attr_name] = getattr(config, attr_name)
            else:
                logging.warning(
                    f"Config attribute {attr_name} (or {prefixed_attr_name}) not found in config."
                )
    else:
        raise ValueError(
            f"Unsupported linear optimizer: {linear_optimizer}, must be one of ['soap', 'muon']"
        )

    # strip out arguments
    embed_adam_kwargs = {
        "lr": config.decoupled_lr if config.decoupled_lr is not None else config.lr,
        "weight_decay": config.weight_decay,
        "betas": (config.adam_beta1, config.adam_beta2),
        "eps": config.adam_eps,
    }

    norm_adam_kwargs = {
        "lr": config.lr,
        "weight_decay": 0.0,  # No weight decay for norm layers
        "betas": (config.adam_beta1, config.adam_beta2),
        "eps": config.adam_eps,
    }

    logging.debug(f"embed_adam_kwargs: {embed_adam_kwargs}")
    logging.debug(f"norm_adam_kwargs: {norm_adam_kwargs}")
    logging.debug(f"linear_opt_kwargs: {linear_opt_kwargs}")

    optimizer = LayerWiseDistributedOptimizer(
        model,
        linear_optimizer_cls,
        [linear_opt_kwargs, embed_adam_kwargs, norm_adam_kwargs],
        config,
        **kwargs,
    )

    # Patch necessary keys that MCore depends on.
    for param_group in optimizer.param_groups:
        param_group["min_lr"] = config.min_lr
        param_group["max_lr"] = config.lr
        param_group["is_decoupled_lr"] = False
        if config.decoupled_lr is not None and getattr(
            param_group["params"][0], "is_embedding_or_output_parameter", False
        ):
            param_group["min_lr"] = config.decoupled_min_lr
            param_group["max_lr"] = config.decoupled_lr
            param_group["is_decoupled_lr"] = True

    return optimizer
