# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from typing import List, Optional, Tuple

import torch

from megatron.core import parallel_state, tensor_parallel
from megatron.core.tensor_parallel.mappings import _gather_along_first_dim_expert_parallel
from megatron.core.transformer.moe.moe_utils import permute, unpermute
from megatron.core.transformer.moe.token_dispatcher import MoETokenDispatcher
from megatron.core.transformer.transformer_config import TransformerConfig


class MoEAlltoAllSEQTokenDispatcher(MoETokenDispatcher):
    """
    The legacy implementation of the AlltoAll-based token dispatcher, which handles token dispatching on the sequence level instead of token level. The core of this implementation lies each device dispatching on the entire sequence, with the hidden state being partitioned.
    Note: This class is a replica of the MoEAlltoAllTokenDispatcher from version 0.8.
    """

    def __init__(
        self, num_local_experts: int, local_expert_indices: List[int], config: TransformerConfig
    ) -> None:
        """
        Initialize the AlltoAll token dispatcher.

        Args:
            num_local_experts (int): Number of local experts on the current device.
            local_expert_indices (List[int]): Indices of local experts on the current device.
            config (TransformerConfig): Configuration for the transformer model.
        """
        super().__init__(config=config)
        self.hidden_shape = None
        self.num_input_tokens = None
        self.num_local_experts = num_local_experts
        self.num_experts = config.num_moe_experts
        assert self.num_local_experts > 0, "Expected at least one expert"
        if self.num_local_experts > 1:
            self.expert_ids_per_ep_rank = torch.tensor(
                [i % self.num_local_experts for i in range(self.num_experts)],
                dtype=torch.int32,
                device=torch.cuda.current_device(),
            )
        self.local_expert_indices = local_expert_indices
        assert (
            len(self.local_expert_indices) == self.num_local_experts
        ), "Invalid local expert indices"
        for i in range(len(self.local_expert_indices) - 1):
            assert (
                self.local_expert_indices[i] == self.local_expert_indices[i + 1] - 1
            ), "local_expert_indices must be continous"
        self.router_topk = config.moe_router_topk
        self.add_bias = config.add_bias_linear
        self.ep_size = config.expert_model_parallel_size
        self.probs = None
        self.input_splits = None
        self.output_splits = None
        self.num_global_tokens_per_local_expert = None

        # Token drop and padding.
        # We need to keep track of the token num if we drop tokens without padding them.
        self.num_out_tokens = None
        # Drop and pad the input to capacity.
        self.drop_and_pad = self.config.moe_pad_expert_input_to_capacity
        if self.drop_and_pad:
            assert self.config.moe_expert_capacity_factor is not None
        self.capacity = None

        # A cuda stream synchronization is needed in self.token_permutation() in some cases,
        # because there are several non-blocking DtoH data transfers called in self.preprocess().
        # The synchronization happens at different points based on MoE settings as late as possible.
        # Valid sync points are "before_permutation_1", "before_ep_alltoall", "before_finish", and "no_sync".
        self.cuda_sync_point = "no_sync"

    def preprocess(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Preprocess token indices for AlltoAll communication and token permutation. This method computes the number of tokens assigned to each expert based on the input indices.
        It also initializes the necessary data structures for AlltoAll communication, such as input
        and output splits, and the mapping between global tokens and local experts.

        Args:
            indices (torch.Tensor): Tensor of indices mapping tokens to experts.

        Returns:
            torch.Tensor: Tensor containing the number of tokens assigned to local expert.
        """
        num_local_tokens_per_expert = torch.histc(
            indices, bins=self.num_experts, min=0, max=self.num_experts
        )
        # num_local_tokens_per_expert: [num_experts]

        ep_size = self.config.expert_model_parallel_size
        if self.drop_and_pad:
            # probs: [num_experts, capacity]
            self.capacity = self.probs.size(1)
            num_tokens_per_local_expert = torch.full(
                (self.num_local_experts,), self.capacity * self.ep_size, dtype=torch.long
            )
            return num_tokens_per_local_expert
        elif self.config.moe_expert_capacity_factor is not None:
            # Token drop but no pad. A synchronization is needed before the first
            # permutation to get the `num_out_tokens` CPU value.
            self.num_out_tokens = num_local_tokens_per_expert.sum().to(
                torch.device("cpu"), non_blocking=True
            )
            self.cuda_sync_point = "before_permutation_1"
        elif ep_size > 1:
            # Token dropless and enable ep. A synchronization is needed before expert parallel
            # AlltoAll communication to get the `input_splits` and `output_splits` CPU values.
            self.cuda_sync_point = "before_ep_alltoall"
        else:
            # Token dropless and no ep. A synchronization is needed before the token_permutation()
            # function returns to get the `tokens_per_expert` CPU value.
            self.cuda_sync_point = "before_finish"

        if ep_size > 1:
            # ===================================================
            # Calculate input_splits, output_splits for alltoall-v.
            # ===================================================
            self.input_splits = (
                num_local_tokens_per_expert.reshape(ep_size, self.num_local_experts)
                .sum(axis=1)
                .to(torch.device("cpu"), non_blocking=True)
                .numpy()
            )
            num_global_tokens_per_expert = _gather_along_first_dim_expert_parallel(
                num_local_tokens_per_expert
            ).reshape(ep_size, self.num_experts)
            self.num_global_tokens_per_local_expert = num_global_tokens_per_expert[
                :, self.local_expert_indices[0] : self.local_expert_indices[-1] + 1
            ]
            self.output_splits = (
                self.num_global_tokens_per_local_expert.sum(axis=-1)
                .to(torch.device("cpu"), non_blocking=True)
                .numpy()
            )
            num_tokens_per_local_expert = self.num_global_tokens_per_local_expert.sum(axis=0).to(
                torch.device("cpu"), non_blocking=True
            )
            # ===================================================
            # num_global_tokens_per_expert: [ep_size, num_experts]
            # num_global_tokens_per_local_expert: [ep_size, num_local_experts]
            # num_tokens_per_local_expert: [num_local_experts]
            # ===================================================
        else:
            self.num_global_tokens_per_local_expert = num_local_tokens_per_expert.reshape(
                -1, self.num_experts
            )
            num_tokens_per_local_expert = num_local_tokens_per_expert.to(
                torch.device("cpu"), non_blocking=True
            )

        if self.num_local_experts > 1:
            # No further synchronization is needed because torch.repeat_interleave() calls stream
            # synchronization internally when the `output_size` parameter is not provided.
            self.cuda_sync_point = "no_sync"
            self.global_input_tokens_local_experts_indices = torch.repeat_interleave(
                self.expert_ids_per_ep_rank, self.num_global_tokens_per_local_expert.ravel()
            )

        return num_tokens_per_local_expert

    def token_permutation(
        self, hidden_states: torch.Tensor, probs: torch.Tensor, indices: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Dispatch tokens to local experts using AlltoAll communication.

        Args:
            hidden_states (torch.Tensor): Input token embeddings.
            probs (torch.Tensor): Probs of tokens assigned to experts.
            indices (torch.Tensor): Indices of tokens assigned to experts.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Permuted token embeddings for local experts.
                - Number of tokens per expert.
        """
        # Preprocess: Get the metadata for communication, permutation and computation operations.
        self.hidden_shape = hidden_states.shape
        self.probs = probs
        assert probs.dim() == 2, "Expected 2D tensor for probs"
        assert indices.dim() == 2, "Expected 2D tensor for indices"
        hidden_states = hidden_states.view(-1, self.hidden_shape[-1])
        tokens_per_expert = self.preprocess(indices)

        # Perform tensor parallel AlltoAll communication
        # hidden_states: [S*B/TP, H] -> [S*B, H/TP]
        if parallel_state.get_tensor_model_parallel_world_size() > 1:
            hidden_states = tensor_parallel.all_to_all_sp2hp(hidden_states)

        # Permutation 1: input to AlltoAll input
        self.hidden_shape_before_permute = hidden_states.shape
        if self.cuda_sync_point == "before_permutation_1":
            torch.cuda.current_stream().synchronize()
        permutated_local_input_tokens, self.reversed_local_input_permutation_mapping = permute(
            hidden_states,
            indices,
            num_out_tokens=self.num_out_tokens,
            padded_mode=self.drop_and_pad,
        )

        # Perform expert parallel AlltoAll communication
        if self.cuda_sync_point == "before_ep_alltoall":
            torch.cuda.current_stream().synchronize()
        global_input_tokens = tensor_parallel.all_to_all(
            parallel_state.get_expert_model_parallel_group(),
            permutated_local_input_tokens,
            self.output_splits,
            self.input_splits,
        )

        # Permutation 2: Sort alltoall output by local experts when num_local_experts > 1.
        if self.num_local_experts > 1:
            if not self.drop_and_pad:
                global_input_tokens, self.reversed_global_input_permutation_mapping = permute(
                    global_input_tokens, self.global_input_tokens_local_experts_indices
                )
            else:
                global_input_tokens = global_input_tokens.reshape(
                    self.ep_size, self.num_local_experts, self.capacity, -1
                )
                global_input_tokens = (
                    global_input_tokens.transpose(0, 1)
                    .reshape(self.num_local_experts * self.ep_size * self.capacity, -1)
                    .contiguous()
                )

        # Perform tensor parallel AllGather on the hidden dimension to obtain the input tokens.
        # global_input_tokens: [SEQL, H/TP] -> [SEQL, H]
        if parallel_state.get_tensor_model_parallel_world_size() > 1:
            global_input_tokens = tensor_parallel.all_gather_last_dim_from_tensor_parallel_region(
                global_input_tokens
            )
        if self.cuda_sync_point == "before_finish":
            torch.cuda.current_stream().synchronize()

        return global_input_tokens, tokens_per_expert

    def token_unpermutation(
        self, hidden_states: torch.Tensor, bias: torch.Tensor = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Reverse the token permutation to restore the original order.

        Args:
            hidden_states (torch.Tensor): Output from local experts.
            bias (torch.Tensor, optional): Bias tensor (not supported).

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                - Unpermuted token embeddings in the original order.
                - None (bias is not supported).
        """
        assert bias is None, "Bias is not supported in MoEAlltoAllTokenDispatcher"

        # Perform tensor parallel Reduce-Scatter
        # hidden_states: [SEQL, H] -> [SEQL, H/TP]
        if parallel_state.get_tensor_model_parallel_world_size() > 1:
            hidden_states = tensor_parallel.reduce_scatter_last_dim_to_tensor_parallel_region(
                hidden_states
            )

        # Unpermutation 2: expert output to AlltoAll input
        if self.num_local_experts > 1:
            if not self.drop_and_pad:
                hidden_states = unpermute(
                    hidden_states, self.reversed_global_input_permutation_mapping
                )
            else:
                hidden_states = hidden_states.reshape(
                    self.num_local_experts, self.ep_size, self.capacity, -1
                )
                hidden_states = (
                    hidden_states.transpose(0, 1)
                    .reshape(self.ep_size * self.num_local_experts * self.capacity, -1)
                    .contiguous()
                )

        # Perform expert parallel AlltoAll communication
        # hidden_states: [SEQL, H] -> [SEQL, H/TP]
        permutated_local_input_tokens = tensor_parallel.all_to_all(
            parallel_state.get_expert_model_parallel_group(),
            hidden_states,
            self.input_splits,
            self.output_splits,
        )

        # Unpermutation 1: AlltoAll output to output
        output = unpermute(
            permutated_local_input_tokens,
            self.reversed_local_input_permutation_mapping,
            probs=self.probs,
            padded_mode=self.drop_and_pad,
            restore_shape=self.hidden_shape_before_permute,
        )

        # Perform tensor parallel AlltoAll communication
        # output: [S*B, H/TP] -> [S*B/TP, H]
        if parallel_state.get_tensor_model_parallel_world_size() > 1:
            output = tensor_parallel.all_to_all_hp2sp(output)

        # Reshape the output tensor
        output = output.view(self.hidden_shape)
        return output, None
