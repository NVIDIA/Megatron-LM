# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Union

import torch

from megatron.core import parallel_state, tensor_parallel
from megatron.core.process_groups_config import ModelCommProcessGroups
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe.moe_utils import get_default_model_comm_pgs
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.moe.token_dispatcher import (
    MoEAllGatherTokenDispatcher,
    MoEAlltoAllTokenDispatcher,
    MoEFlexTokenDispatcher,
    MoETokenDispatcher,
)
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig

try:
    import transformer_engine as te  # pylint: disable=unused-import

    from megatron.core.extensions.transformer_engine import te_checkpoint

    HAVE_TE = True
except ImportError:
    HAVE_TE = False

# === save_tensor 插桩 ===
from megatron.core.align_dump_utils import (
    _mg_tensor_info,
    _mg_grad_info,
    is_log_enabled as _is_log_enabled,
    mg_dump_grad_hook as _mg_dump_grad_hook,
)
# === save_tensor 插桩结束 ===


@dataclass
class MoESubmodules:
    """MoE Layer Submodule spec"""

    experts: Union[ModuleSpec, type] = None
    shared_experts: Union[ModuleSpec, type] = None


class BaseMoELayer(MegatronModule, ABC):
    """Base class for a mixture of experts layer.

    Args:
        config (TransformerConfig): Configuration object for the transformer model.
    """

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: Optional[int] = None,
        model_comm_pgs: Optional[ModelCommProcessGroups] = None,
    ):
        super(BaseMoELayer, self).__init__(config)
        self.config = config
        self.layer_number = layer_number
        self.ep_group = model_comm_pgs.ep
        # use model_comm_pgs.expt_tp_group as tensor parallel group in this module.
        self.attn_tp_group = model_comm_pgs.tp
        ep_size = self.ep_group.size()
        ep_rank = self.ep_group.rank()
        assert ep_size > 0, "Expected non-negative expert parallel size"

        assert self.config.num_moe_experts % ep_size == 0
        self.num_local_experts = self.config.num_moe_experts // ep_size
        local_expert_indices_offset = ep_rank * self.num_local_experts

        self.use_shared_expert = self.config.moe_shared_expert_intermediate_size is not None
        self.shared_expert_overlap = self.config.moe_shared_expert_overlap

        self.local_expert_indices = [
            local_expert_indices_offset + i for i in range(self.num_local_experts)
        ]
        assert all(map(lambda x: x < self.config.num_moe_experts, self.local_expert_indices))
        self.router: TopKRouter = None
        self.experts = None
        self.shared_experts = None
        self.token_dispatcher: Optional[MoETokenDispatcher] = None
        self.layer_number = layer_number

    @abstractmethod
    def forward(self, hidden_states):
        """Forward method for the MoE layer."""
        pass

    def set_layer_number(self, layer_number: int):
        """Set the layer number for the MoE layer."""
        self.layer_number = layer_number
        self.router.set_layer_number(layer_number)


class MoELayer(BaseMoELayer):
    """Mixture of Experts layer.

    This layer implements a Mixture of Experts model, where each token is routed to a
    subset of experts. This implementation supports different token dispatching
    strategies such as All-to-All and All-Gather.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: Optional[MoESubmodules] = None,
        layer_number: Optional[int] = None,
        model_comm_pgs: Optional[ModelCommProcessGroups] = None,
    ):
        self.submodules = submodules
        # TODO(Hepteract): delete the usage of the global parallel_state.
        # Initialize process groups with the global parallel_state.
        if model_comm_pgs is None:
            model_comm_pgs = get_default_model_comm_pgs()
        super(MoELayer, self).__init__(
            config=config, layer_number=layer_number, model_comm_pgs=model_comm_pgs
        )
        self.moe_layer_recompute = (
            config.recompute_granularity == 'selective' and "moe" in config.recompute_modules
        )

        # Initialize router
        self.router = TopKRouter(config=self.config, model_comm_pgs=model_comm_pgs)

        # Initialize token dispatcher
        if config.moe_token_dispatcher_type == "allgather":
            self.token_dispatcher = MoEAllGatherTokenDispatcher(
                self.num_local_experts,
                self.local_expert_indices,
                config=self.config,
                model_comm_pgs=model_comm_pgs,
            )
        elif config.moe_token_dispatcher_type == "alltoall":
            self.token_dispatcher = MoEAlltoAllTokenDispatcher(
                self.num_local_experts,
                self.local_expert_indices,
                config=self.config,
                model_comm_pgs=model_comm_pgs,
            )
        elif config.moe_token_dispatcher_type == "flex":
            self.token_dispatcher = MoEFlexTokenDispatcher(
                self.num_local_experts,
                self.local_expert_indices,
                config=self.config,
                model_comm_pgs=model_comm_pgs,
            )
        else:
            raise ValueError(
                f"Unsupported token dispatcher type: {config.moe_token_dispatcher_type}"
            )

        # Initialize experts
        self.experts = build_module(
            self.submodules.experts,
            self.num_local_experts,
            self.config,
            model_comm_pgs=model_comm_pgs,
        )

        # Initialize shared experts
        if self.use_shared_expert:
            self.shared_experts = build_module(
                self.submodules.shared_experts, config=self.config, model_comm_pgs=model_comm_pgs
            )
            if self.shared_expert_overlap:
                self.token_dispatcher.set_shared_experts(self.shared_experts)

    def router_and_preprocess(self, hidden_states: torch.Tensor):
        """Compute and preprocess token routing for dispatch.

        This method uses the router to determine which experts to send each token to,
        producing routing probabilities and a mapping. It then preprocesses the
        hidden states and probabilities for the token dispatcher. The original
        hidden states are returned as a residual connection.
        """
        # === [反向对齐 align-grad] 探针: 把 hidden_states 拆三路独立 grad 节点 ===
        # 与 PF moe_layer.py 中相同, clone 三份:
        #   - residual (shared_expert 路)
        #   - hidden_states (dispatcher 路, 在 dispatch_preprocess 后输出)
        #   - hidden_states_for_router (router 路)
        # 反向时三个 clone.grad 之和 = 原 hidden_states.grad.
        if _is_log_enabled() and hidden_states.requires_grad:
            _DUMP_TAG_TP_MG = f"moe_three_paths_LMG{self.layer_number}"

            _hs_router_path_mg = hidden_states.clone()
            _hs_dispatcher_path_mg = hidden_states.clone()
            _hs_shared_path_mg = hidden_states.clone()

            _hs_router_path_mg.register_hook(_mg_grad_info(
                "cp11b_three_paths_router_grad",
                layer_num=self.layer_number, prefix="GRAD MG MoE"))
            _hs_dispatcher_path_mg.register_hook(_mg_grad_info(
                "cp11b_three_paths_dispatcher_grad",
                layer_num=self.layer_number, prefix="GRAD MG MoE"))
            _hs_shared_path_mg.register_hook(_mg_grad_info(
                "cp11b_three_paths_shared_grad",
                layer_num=self.layer_number, prefix="GRAD MG MoE"))
            # 跨框架 npy dump (与 PF 共享文件命名), 由 align_dump_utils 统一管理
            _hs_router_path_mg.register_hook(_mg_dump_grad_hook(_DUMP_TAG_TP_MG, "router"))
            _hs_dispatcher_path_mg.register_hook(_mg_dump_grad_hook(_DUMP_TAG_TP_MG, "dispatcher"))
            _hs_shared_path_mg.register_hook(_mg_dump_grad_hook(_DUMP_TAG_TP_MG, "shared"))

            residual = _hs_shared_path_mg
            hidden_states_router = _hs_router_path_mg
            hidden_states_dispatch = _hs_dispatcher_path_mg
        else:
            residual = hidden_states
            hidden_states_router = hidden_states
            hidden_states_dispatch = hidden_states
        # === [反向对齐 align-grad] 探针结束 ===
        probs, routing_map = self.router(hidden_states_router)
        hidden_states, probs = self.token_dispatcher.dispatch_preprocess(
            hidden_states_dispatch, routing_map, probs
        )
        return hidden_states, probs, residual

    def dispatch(self, hidden_states: torch.Tensor, probs: torch.Tensor):
        """Dispatches tokens to assigned expert ranks via communication.
        This method performs the actual communication (e.g., All-to-All) to distribute
        tokens and their associated probabilities to the devices hosting their assigned
        experts.
        """
        return self.token_dispatcher.token_dispatch(hidden_states, probs)

    def experts_compute(
        self, hidden_states: torch.Tensor, probs: torch.Tensor, residual: torch.Tensor
    ):
        """Computes the output of the experts on the dispatched tokens.

        This method first post-processes the dispatched input to get permuted tokens
        for each expert. It then passes the tokens through the local experts.
        If a shared expert is configured and not overlapped with communication,
        it is also applied. The output from the experts is preprocessed for the
        combine step.
        """
        shared_expert_output = None
        if self.use_shared_expert and not self.shared_expert_overlap:
            # Compute the shared expert separately when not overlapped with communication.
            shared_expert_output = self.shared_experts(residual)
            _mg_tensor_info("shared_expert_output", shared_expert_output, self.layer_number)
            if shared_expert_output.requires_grad:
                shared_expert_output.register_hook(_mg_grad_info("shared_expert_output", layer_num=self.layer_number, prefix="GRAD MG MoE"))
            # === GRAD DEBUG: shared expert 内部梯度 ===
            if residual.requires_grad:
                residual.register_hook(_mg_grad_info("shared_expert_input_grad", layer_num=self.layer_number, prefix="GRAD MG MoE"))
            # === GRAD DEBUG END ===
        dispatched_input, tokens_per_expert, permuted_probs = (
            self.token_dispatcher.dispatch_postprocess(hidden_states, probs)
        )
        _mg_tensor_info("permuted_input", dispatched_input, self.layer_number)
        _mg_tensor_info("tokens_per_expert", tokens_per_expert, self.layer_number)
        if dispatched_input.requires_grad:
            dispatched_input.register_hook(_mg_grad_info("permuted_input", layer_num=self.layer_number, prefix="GRAD MG MoE"))

        expert_output, mlp_bias = self.experts(dispatched_input, tokens_per_expert, permuted_probs)
        assert mlp_bias is None, f"mlp_bias is not supported for {type(self.token_dispatcher)}"
        _mg_tensor_info("expert_raw_output", expert_output, self.layer_number)
        if expert_output.requires_grad:
            expert_output.register_hook(_mg_grad_info("expert_raw_output", layer_num=self.layer_number, prefix="GRAD MG MoE"))
        output = self.token_dispatcher.combine_preprocess(expert_output)

        return output, shared_expert_output, mlp_bias

    def combine(self, output: torch.Tensor, shared_expert_output: Optional[torch.Tensor]):
        """Combines expert outputs via communication and adds shared expert output.

        This method uses the token dispatcher to combine the outputs from different
        experts (e.g., via an All-to-All communication). It then adds the output
        from the shared expert if it exists.
        """
        output = self.token_dispatcher.token_combine(output)
        output = self.token_dispatcher.combine_postprocess(output)
        _mg_tensor_info("routed_output", output, self.layer_number)
        if output.requires_grad:
            output.register_hook(_mg_grad_info("routed_output", layer_num=self.layer_number, prefix="GRAD MG MoE"))

        if shared_expert_output is not None:
            output = output + shared_expert_output
        _mg_tensor_info("final_moe_output", output, self.layer_number)
        if output.requires_grad:
            output.register_hook(_mg_grad_info("final_moe_output", layer_num=self.layer_number, prefix="GRAD MG MoE"))
        return output

    def forward(self, hidden_states: torch.Tensor):
        """Forward pass for the MoE layer.

        The forward pass comprises four main steps:
        1. Routing & Preprocessing: Route tokens to the assigned experts and prepare for dispatch.
        2. Dispatch: Tokens are sent to the expert devices using communication collectives.
        3. Expert Computation: Experts process the dispatched tokens.
        4. Combine: The outputs from the experts are combined and returned.

        Args:
            hidden_states (torch.Tensor): The input tensor to the MoE layer.

        Returns:
            A tuple containing the output tensor and the MLP bias, if any.
        """
        if self.training and self.attn_tp_group.size() > 1 and not self.config.sequence_parallel:
            raise ValueError(
                "During training, performance may degrade if MoE and tensor parallelism"
                "are enabled without also enabling sequence parallelism."
            )

        _mg_tensor_info("input", hidden_states, self.layer_number)

        # === [反向对齐 align-grad] 探针: 在 forward 入口注册聚合 grad hook 并 dump,
        # 用于和 PF cp11b_grad_aggregated 比对 ===
        if _is_log_enabled() and hidden_states.requires_grad:
            hidden_states.register_hook(_mg_grad_info(
                "cp11b_grad_aggregated",
                layer_num=self.layer_number, prefix="GRAD MG MoE"))
        # === [反向对齐 align-grad] 探针结束 ===

        # MoE forward: route -> dispatch -> compute -> combine
        def custom_forward(hidden_states):
            hidden_states, probs, residual = self.router_and_preprocess(hidden_states)
            dispatched_input, probs = self.dispatch(hidden_states, probs)
            output, shared_expert_output, mlp_bias = self.experts_compute(
                dispatched_input, probs, residual
            )
            output = self.combine(output, shared_expert_output)
            return output, mlp_bias

        if self.moe_layer_recompute:
            if self.config.fp8:
                output, mlp_bias = te_checkpoint(
                    custom_forward,
                    False,
                    tensor_parallel.random.get_cuda_rng_tracker,
                    parallel_state.get_tensor_model_parallel_group(),
                    hidden_states,
                )
            else:
                output, mlp_bias = tensor_parallel.checkpoint(custom_forward, False, hidden_states)
        else:
            output, mlp_bias = custom_forward(hidden_states)

        return output, mlp_bias

    def backward_dw(self):
        """Compute weight gradients for experts and shared experts."""
        self.experts.backward_dw()
        if self.use_shared_expert and not self.shared_expert_overlap:
            self.shared_experts.backward_dw()
