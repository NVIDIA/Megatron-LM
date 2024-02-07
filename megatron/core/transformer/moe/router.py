# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import math
from abc import ABC, abstractmethod
from typing import Callable, List

import torch

from megatron.core import parallel_state, tensor_parallel
from megatron.core.parallel_state import get_tensor_and_expert_parallel_group
from megatron.core.tensor_parallel import get_cuda_rng_tracker, get_data_parallel_rng_tracker_name
from megatron.core.tensor_parallel.random import (
    get_cuda_rng_tracker,
    get_data_parallel_rng_tracker_name,
)
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe.moe_utils import (
    MoEAuxLossAutoScaler,
    sinkhorn,
    switch_load_balancing_loss_func,
    z_loss_func,
)
from megatron.core.transformer.transformer_config import TransformerConfig


class Router(ABC, MegatronModule):
    """Base Router class"""

    def __init__(self, config: TransformerConfig) -> None:
        """
        Initialize the Router module.

        Args:
            config (TransformerConfig): Configuration object for the Transformer model.
        """
        super().__init__(config)
        self.config = config
        self.num_experts = self.config.num_moe_experts
        self.moe_aux_loss_func = None

        # Initialize the gate weights.
        self.weight = torch.nn.Parameter(
            torch.empty((self.config.num_moe_experts, self.config.hidden_size))
        )
        with get_cuda_rng_tracker().fork(get_data_parallel_rng_tracker_name()):
            config.init_method(self.weight)
        setattr(self.weight, 'sequence_parallel', config.sequence_parallel)

    def gating(self, input: torch.Tensor):
        """Forward pass of the router gate.

        Args:
            input (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Logits tensor.
        """
        logits = torch.nn.functional.linear(input, self.weight)
        return logits

    @abstractmethod
    def routing(self, logits: torch.Tensor):
        """Routing function.

        Args:
            logits (torch.Tensor): Logits tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple of tensors representing max probs and the indices.
        """
        raise NotImplementedError("Routing function not implemented.")

    def forward(self, input: torch.Tensor):
        """
        Forward pass of the router.

        Args:
            input (torch.Tensor): Input tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: scores and indices.
        """
        self.hidden = input.shape[-1]

        logits = self.gating(input)
        logits = logits.view(-1, self.config.num_moe_experts)

        scores, indices = self.routing(logits)

        return scores, indices


class TopKRouter(Router):
    """Route each token to the top-k experts."""

    def __init__(self, config: TransformerConfig,) -> None:
        """Initialize the zero token dropping router.

        Args:
            config (TransformerConfig): The configuration for the transformer model.
        """
        super().__init__(config=config)
        assert config.moe_token_dropping is False
        self.topk = self.config.moe_router_topk
        self.routing_type = self.config.moe_router_load_balancing_type
        self.moe_aux_loss_func = switch_load_balancing_loss_func
        self.input_jitter = None

    def sinkhorn_load_balancing(self, logits: torch.Tensor):
        """Apply sinkhorn routing to the logits tensor.

        Args:
            logits (torch.Tensor): The logits tensor.

        Returns:
            torch.Tensor: The logits tensor after applying sinkhorn routing.
        """

        def _sinkhorn_activation(logits):
            if self.topk == 1:
                logits = torch.sigmoid(logits)
            else:  # k > 1
                logits = torch.softmax(logits, dim=-1, dtype=torch.float32).type_as(logits)
            return logits

        assert self.config.moe_aux_loss_coeff == 0, "Sinkhorn routing does not support aux loss."
        if self.training:
            with torch.no_grad():
                norm_logits = sinkhorn(
                    logits.to(dtype=torch.float32)
                )  # explicit fp32 conversion for stability
                _, indices = torch.topk(norm_logits, k=self.topk, dim=1)
            logits = _sinkhorn_activation(logits)
            scores = torch.gather(logits, 1, indices)
        else:
            logits = _sinkhorn_activation(logits)
            scores, indices = torch.topk(logits, k=self.topk, dim=1)
        return scores, indices

    def aux_loss_load_balancing(self, logits: torch.Tensor):
        """Apply loss-based load balancing to the logits tensor.

        Args:
            logits (torch.Tensor): The logits tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The scores and the indices tensor after applying load balancing.
        """
        top_logits, indices = torch.topk(logits, k=self.topk, dim=1)
        scores = torch.softmax(top_logits, dim=-1, dtype=torch.float32).type_as(logits)
        # Apply load balancing loss
        probs = torch.softmax(logits, dim=-1, dtype=torch.float32)
        scores = self.apply_aux_loss(self.moe_aux_loss_func, probs, indices, activation=scores)
        return scores, indices

    def apply_aux_loss(
        self,
        loss_func: Callable,
        probs: torch.Tensor,
        indices: torch.Tensor,
        activation: torch.Tensor,
    ):
        """Applies auxiliary loss to the MoE layer.

        Args:
            loss_func (callable): The loss function to be used.
            probs (torch.Tensor): The probabilities output by the MoE layer.
            indices (torch.Tensor): The indices of the selected experts.
            activation (torch.Tensor): The activation tensor to attach the gradient function to.

        Returns:
            torch.Tensor: The activation tensor with the attached gradient function.
        """
        mask = torch.nn.functional.one_hot(indices, num_classes=self.num_experts).sum(dim=1)
        aux_loss = loss_func(probs, mask, self.config.moe_aux_loss_coeff)
        activation = MoEAuxLossAutoScaler.apply(activation, aux_loss)
        return activation

    def apply_z_loss(self, logits):
        """Encourages the router's logits to remain small to enhance stability.
        Please refer to the ST-MoE paper (https://arxiv.org/pdf/2202.08906.pdf) for details.
        
        Args:
            logits (torch.Tensor): The logits of the router.
        
        Returns:
            torch.Tensor: The logits after applying the z-loss.
        """
        if self.config.moe_z_loss_coeff is not None:
            z_loss = z_loss_func(logits, self.config.moe_z_loss_coeff)
            logits = MoEAuxLossAutoScaler.apply(logits, z_loss)
        return logits

    def apply_input_jitter(self, input: torch.Tensor):
        """Add noise to the input tensor.
        Refer to https://arxiv.org/abs/2101.03961.

        Args:
            input (Tensor): Input tensor.

        Returns:
            Tensor: Jittered input.
        """
        if self.config.moe_input_jitter_eps is not None:
            eps = self.config.moe_input_jitter_eps
            if self.input_jitter is None:
                self.input_jitter = torch.distributions.uniform.Uniform(
                    torch.tensor(1.0 - eps, device=input.device),
                    torch.tensor(1.0 + eps, device=input.device),
                ).rsample
            return input * self.input_jitter(input.shape)
        else:
            return input

    def routing(self, logits: torch.Tensor):
        """Top-k routing function

        Args:
            logits (torch.Tensor): Logits tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Probs and the indices tensor.
        """
        logits = logits.view(-1, self.config.num_moe_experts)

        # Apply Z-Loss
        logits = self.apply_z_loss(logits)
        # Apply input jitter
        logits = self.apply_input_jitter(logits)

        if self.routing_type == "sinkhorn":
            scores, indices = self.sinkhorn_load_balancing(logits)
        elif self.routing_type == "aux_loss":
            scores, indices = self.aux_loss_load_balancing(logits)
        elif self.routing_type == "none":
            # A naive top-k routing without load balancing
            top_logits, indices = torch.topk(logits, k=self.topk, dim=1)
            scores = torch.softmax(top_logits, dim=-1, dtype=torch.float32).type_as(logits)
        else:
            raise ValueError(f"Unsupported MoE routing type: {self.routing_type}")

        return scores, indices
