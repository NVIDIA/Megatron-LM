# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from abc import abstractmethod
from typing import List, Optional, Tuple

import torch

from megatron.core import parallel_state, tensor_parallel
from megatron.core.tensor_parallel.mappings import _gather_along_first_dim_expert_parallel
from megatron.core.transformer.moe.moe_utils import permute, unpermute
from megatron.core.transformer.transformer_config import TransformerConfig


class MoETokenDispatcher:
    """
    MoE Token Dispatcher
    """

    def __init__(self, config: TransformerConfig) -> None:
        """
        Initialize the MoE Token Dispatcher.
        """
        self.config = config

    @abstractmethod
    def token_permutation(
        self, tokens: torch.Tensor, indices: torch.Tensor,
    ):
        """Dispatch tokens to experts.

        Args:
            tokens (torch.Tensor): Input tokens.
            indices (torch.Tensor): indices tensor.

        Returns:
            torch.Tensor: Tokens tensor.
        """
        raise NotImplementedError("Dispatch function not implemented.")

    @abstractmethod
    def token_unpermutation(
        self, expert_output: torch.Tensor, scores: torch.Tensor, indices: torch.Tensor,
    ):
        """Restores the expert output to its original ordering.

        Args:
            expert_output (torch.Tensor): The output tensor from the expert models.
            scores (torch.Tensor): Each token's score with each expert.
            indices (torch.Tensor): The indices used to reorder the expert output.

        Returns: 
            (torch.Tensor, torch.Tensor): Unpermuted activation and optional bias.            
        """
        raise NotImplementedError("Restore function not implemented.")


class MoEAllGatherTokenDispatcher(MoETokenDispatcher):
    """
    AllGather Based Token dispatcher.
    """

    def __init__(
        self, num_local_experts: int, local_expert_indices: List[int], config: TransformerConfig,
    ) -> None:
        """
        Initialize the zero token dropping router.
        """
        super().__init__(config=config)
        self.num_local_experts = num_local_experts
        assert self.num_local_experts > 0, "Expected at least one expert"
        self.local_expert_indices = local_expert_indices
        assert len(self.local_expert_indices) > 0, "Expected at least one local expert index"
        self.router_topk = config.moe_router_topk
        self.add_bias = config.add_bias_linear

        # self.local_probs: probs of global token assignment to local experts.
        self.local_probs = None

        # self.indices: The indices of `local_indices` (which holds the un-sorted expert indices of tokens that local expert can process) that give its sorted order along dim 0.
        self.indices = None

        # self.global_local_map: 2D tensor. A mask of mapping between global and local tokens where each element is True if it's between the local_expert_indices. Only useful when cross device token permutation is enabled and **AllGahter** is performed.
        self.global_local_map = None

    def token_permutation(
        self, hidden_states: torch.Tensor, max_prob: torch.Tensor, max_ind: torch.Tensor
    ):
        """Dispatch tokens to local experts. It's composed of two stages:
        (1) Permute the tokens across the expert parallel devices. After this stage,
        each device receives all of the tokens assigned to its local set of experts
        in its local HBM.
        (2) Permute the tokens locally so that they are grouped by their expert
        assignment. After the stage (1), the tokens are grouped by which device
        they came from. We re-order them locally for subsequent efficient computation.

        Args:
            hidden_states: input tokens of shape [SeqLen/TP, MBS, HiddenSize]
            max_prob: probs of local token assignment to global experts.
            max_ind: token assignment to local experts.

        Returns:
            permuted_local_hidden_states: Permutation of tokens to local experts group.
            tokens_per_expert: the number of tokens each local expert to process.
        """
        self.hidden_shape = hidden_states.shape
        # [S/TP, B, H] -> [S*B/TP, H]
        hidden_states = hidden_states.view(-1, self.hidden_shape[-1])

        # Permute the tokens across the expert parallel devices.
        if self.config.sequence_parallel or (self.config.expert_model_parallel_size > 1):
            # [S*B/TP, H] -> [S*B, H]
            global_hidden_states = tensor_parallel.gather_from_sequence_parallel_region_to_moe(
                hidden_states
            )
            with torch.no_grad():
                global_indices = tensor_parallel.gather_from_sequence_parallel_region_to_moe(
                    max_ind
                )
                # Create a mask of mapping between global and local tokens where each
                # element is True if it's between the local_expert_indices
                global_local_mask = (global_indices >= self.local_expert_indices[0]) & (
                    global_indices <= self.local_expert_indices[-1]
                )
                local_indices = global_indices.masked_select(global_local_mask)

            if self.router_topk > 1:  # k > 1
                global_probs = tensor_parallel.gather_from_sequence_parallel_region_to_moe(max_prob)
                self.local_probs = global_probs.masked_select(global_local_mask)
            else:
                self.local_probs = max_prob

            # Reshape global_local_mask to be compatible with Tensor.gather
            global_local_map = global_local_mask.nonzero()[:, 0]
            self.global_local_map = global_local_map.view(-1, 1).expand(-1, hidden_states.shape[-1])
            local_hidden_states = torch.gather(global_hidden_states, 0, self.global_local_map)
        else:
            if self.router_topk > 1:
                global_local_mask = torch.ones_like(max_ind).bool()
                local_indices = max_ind.masked_select(global_local_mask)
                self.local_probs = max_prob.masked_select(global_local_mask)
                global_local_map = global_local_mask.nonzero()[:, 0]
                self.global_local_map = global_local_map.view(-1, 1).expand(
                    -1, hidden_states.shape[-1]
                )
                local_hidden_states = torch.gather(hidden_states, 0, self.global_local_map)
            else:
                local_indices = max_ind
                self.local_probs = max_prob
                local_hidden_states = hidden_states
                self.global_local_map = None

        with torch.no_grad():
            # The indices of local_indices that give its sorted order along dim 0.
            self.indices = torch.argsort(local_indices, dim=0)
            tokens_per_expert = torch.histc(
                local_indices,
                bins=self.num_local_experts,
                min=self.local_expert_indices[0],
                max=self.local_expert_indices[-1],
            )
            tokens_per_expert = tokens_per_expert.cpu().to(torch.long)

        # Stage2: permute the tokens locally so that they are grouped by their expert assignment
        # Reshape indices to be compatible with Tensor.gather
        self.indices = self.indices.view(-1, 1).expand(-1, hidden_states.shape[-1])
        permuted_local_hidden_states = torch.gather(local_hidden_states, 0, self.indices)
        return (
            permuted_local_hidden_states,
            tokens_per_expert,
        )

    def token_unpermutation(
        self, hidden_states: torch.Tensor, bias: torch.Tensor = None,
    ):
        """
        Reverse process of `dispatch()` which permutes the ouput of local
        experts locallay and across expert parallel rank into the original order to
        produce the final output.

        Args:
            hidden_states: 2D tensor of shape [sum_tokens_of_all_local_experts, HiddenSize],
            ouput of local experts.
            bias (optional): The bias tensor.

        Returns:
            output_total: un-permuted updated hidden states output from all local experts
            with shape of [SeqLen/TP, MBS, HiddenSize]
        """
        # Stage1: unpermute the tokens and bias locally respectively.
        scores = self.local_probs.to(dtype=hidden_states.dtype)
        unpermuted_local_hidden = torch.zeros_like(hidden_states)
        assert self.indices.shape == hidden_states.shape
        unpermuted_local_hidden = unpermuted_local_hidden.scatter(0, self.indices, hidden_states)

        # Scale the expert output prior to reduction and subsequent to local unpermutation if k > 1.
        if self.router_topk > 1:
            unpermuted_local_hidden = unpermuted_local_hidden * scores.view(-1, 1)

        unpermuted_local_bias = None
        if self.add_bias:
            assert bias is not None
            unpermuted_local_bias = torch.zeros_like(hidden_states)
            assert self.indices.shape == bias.shape
            unpermuted_local_bias = unpermuted_local_bias.scatter(0, self.indices, bias)
            if self.router_topk > 1:
                unpermuted_local_bias = unpermuted_local_bias * scores.view(-1, 1)

        output_total = unpermuted_local_hidden
        output_bias_total = unpermuted_local_bias

        # Unpermute the tokens across expert parallel devices.
        if self.config.sequence_parallel or (self.config.expert_model_parallel_size > 1):
            assert (
                self.global_local_map is not None
            ), "global_local_map is necessary for `AllGather`."
            ep_group_size = parallel_state.get_tensor_and_expert_parallel_world_size()
            # hidden_shape: [SeqLen/TP, MBS, HiddenSize], glboal_num_tokens = SeqLen/TP*MBS*(TP*EP)
            global_num_tokens = self.hidden_shape[0] * self.hidden_shape[1] * ep_group_size
            global_hidden_shape = [global_num_tokens, hidden_states.shape[-1]]
            unpermuted_global_hidden = torch.zeros(
                global_hidden_shape, dtype=hidden_states.dtype, device=torch.cuda.current_device()
            )
            # Reshape global_local_map to be compatible with Tensor.scatter
            assert self.global_local_map.shape == unpermuted_local_hidden.shape
            unpermuted_global_hidden = unpermuted_global_hidden.scatter_add(
                0, self.global_local_map, unpermuted_local_hidden
            )
            output_total = tensor_parallel.reduce_scatter_to_sequence_parallel_region_from_moe(
                unpermuted_global_hidden
            )
            if self.add_bias:
                # Unpermute the bias across expert parallel devices.
                unpermuted_global_bias = torch.zeros_like(unpermuted_global_hidden)
                unpermuted_global_bias = unpermuted_global_bias.scatter_add(
                    0, self.global_local_map, unpermuted_local_bias
                )
                output_bias_total = tensor_parallel.reduce_scatter_to_sequence_parallel_region_from_moe(
                    unpermuted_global_bias
                )
                # bias is duplicated across tensor parallelism ranks;
                # reduce scatter reduces bias across tensor parallel_ranks
                output_bias_total = (
                    output_bias_total / parallel_state.get_tensor_model_parallel_world_size()
                )
        else:
            if self.router_topk > 1:
                global_num_tokens = self.hidden_shape[0] * self.hidden_shape[1]
                global_hidden_shape = [global_num_tokens, hidden_states.shape[-1]]
                unpermuted_global_hidden = torch.zeros(
                    global_hidden_shape,
                    dtype=hidden_states.dtype,
                    device=torch.cuda.current_device(),
                )
                output_total = unpermuted_global_hidden.scatter_add(
                    0, self.global_local_map, unpermuted_local_hidden
                )
                if self.add_bias:
                    unpermuted_global_bias = torch.zeros_like(unpermuted_global_hidden)
                    output_bias_total = unpermuted_global_bias.scatter_add(
                        0, self.global_local_map, unpermuted_local_bias
                    )

        if self.router_topk == 1:
            output_total = output_total * scores
        output_total = output_total.view(self.hidden_shape)
        if self.add_bias:
            assert output_bias_total is not None
            if self.router_topk == 1:
                output_bias_total = output_bias_total * scores
            output_bias_total = output_bias_total.view(self.hidden_shape)
        else:
            output_bias_total = None

        return output_total, output_bias_total


class MoEAlltoAllTokenDispatcher(MoETokenDispatcher):
    """
    AlltoAll Based Token dispatcher.
    """

    def __init__(
        self, num_local_experts: int, local_expert_indices: List[int], config: TransformerConfig,
    ) -> None:
        """
        Initialize the AlltoAll token dispatcher.

        Args:
            num_local_experts (int): Number of local experts on the current device.
            local_expert_indices (List[int]): Indices of local experts on the current device.
            config (TransformerConfig): Configuration for the transformer model.
        """
        super().__init__(config=config)
        self.num_local_experts = num_local_experts
        self.num_experts = config.num_moe_experts
        assert self.num_local_experts > 0, "Expected at least one expert"
        self.local_expert_indices = local_expert_indices
        assert (
            len(self.local_expert_indices) == self.num_local_experts
        ), "Invalid local expert indices"
        self.router_topk = config.moe_router_topk
        self.add_bias = config.add_bias_linear
        self.ep_size = config.expert_model_parallel_size
        self.scores: torch.Tensor = None
        self.input_splits = None
        self.output_splits = None
        self.num_global_tokens_per_local_expert = None

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
        if ep_size > 1:
            # ===================================================
            # Calculate input_splits, output_splits for alltoall-v.
            # ===================================================
            self.input_splits = (
                num_local_tokens_per_expert.reshape(ep_size, self.num_local_experts)
                .sum(axis=1)
                .to(torch.device("cpu"))
                .numpy()
            )
            num_global_tokens_per_expert = _gather_along_first_dim_expert_parallel(
                num_local_tokens_per_expert
            ).reshape(ep_size, self.num_experts)
            self.num_global_tokens_per_local_expert = num_global_tokens_per_expert[
                :, self.local_expert_indices
            ]
            self.output_splits = (
                self.num_global_tokens_per_local_expert.sum(axis=-1).to(torch.device("cpu")).numpy()
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
            expert_ids_per_ep_rank = torch.tensor(
                [i % self.num_local_experts for i in range(self.config.num_moe_experts)],
                dtype=torch.int32,
                device=torch.cuda.current_device(),
            )
            self.global_input_tokens_local_experts_indices = torch.repeat_interleave(
                expert_ids_per_ep_rank, self.num_global_tokens_per_local_expert.ravel()
            )

        return num_tokens_per_local_expert

    def token_permutation(
        self, hidden_states: torch.Tensor, scores: torch.Tensor, indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Dispatch tokens to local experts using AlltoAll communication.

        Args:
            hidden_states (torch.Tensor): Input token embeddings.
            scores (torch.Tensor): Scores of tokens assigned to experts.
            indices (torch.Tensor): Indices of tokens assigned to experts.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Permuted token embeddings for local experts.
                - Number of tokens per expert.
        """
        self.hidden_shape = hidden_states.shape
        self.scores = scores
        assert scores.dim() == 2, "Expected 2D tensor for scores"
        assert indices.dim() == 2, "Expected 2D tensor for indices"
        tokens_per_expert = self.preprocess(indices)

        # TODO Optimize EP=1 case
        # Flatten the input tensor
        # hidden_states: [S/TP, B, H] -> [S*B/TP, H]
        hidden_states = hidden_states.view(-1, self.hidden_shape[-1])

        # Perform tensor parallel AlltoAll communication
        # hidden_states: [S*B/TP, H] -> [S*B, H/TP]
        if parallel_state.get_tensor_model_parallel_world_size() > 1:
            hidden_states = tensor_parallel.all_to_all_sp2hp(hidden_states)

        # Permutation 1: input to AlltoAll input
        self.local_input_tokens_global_experts_indices = indices
        permutated_local_input_tokens, self.reversed_local_input_permutation_mapping = permute(
            hidden_states, self.local_input_tokens_global_experts_indices, topk=self.router_topk,
        )

        # Perform expert parallel AlltoAll communication
        global_input_tokens = tensor_parallel.all_to_all(
            parallel_state.get_expert_model_parallel_group(),
            permutated_local_input_tokens,
            self.output_splits,
            self.input_splits,
        )

        # Permutation 2: AlltoAll output to expert input if num_local_experts > 1
        if self.num_local_experts > 1:
            global_input_tokens, self.reversed_global_input_permutation_mapping = permute(
                global_input_tokens, self.global_input_tokens_local_experts_indices
            )

        # Perform tensor parallel All-Gather
        # global_input_tokens: [SEQL, H/TP] -> [SEQL, H]
        if parallel_state.get_tensor_model_parallel_world_size() > 1:
            global_input_tokens = tensor_parallel.all_gather_last_dim_from_tensor_parallel_region(
                global_input_tokens
            )

        return global_input_tokens, tokens_per_expert

    def token_unpermutation(
        self, hidden_states: torch.Tensor, bias: torch.Tensor = None,
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
        # hidden_states: [SEQL, H] -> [SEQL, H/TP]
        if self.num_local_experts > 1:
            hidden_states = unpermute(
                hidden_states, self.reversed_global_input_permutation_mapping,
            )

        # Perform expert parallel AlltoAll communication
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
            probs=self.scores,
            topk=self.router_topk,
        )

        # Perform tensor parallel AlltoAll communication
        if parallel_state.get_tensor_model_parallel_world_size() > 1:
            # output: [S*B, H/TP] -> [S*B/TP, H]
            output = tensor_parallel.all_to_all_hp2sp(output)

        # Reshape the output tensor
        output = output.view(self.hidden_shape)
        return output, None
