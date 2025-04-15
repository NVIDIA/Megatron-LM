# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import torch

from megatron.core.parallel_state import (
    get_expert_model_parallel_group,
    get_expert_tensor_and_model_parallel_group,
    get_expert_tensor_parallel_group,
    get_expert_tensor_parallel_rank,
)
from megatron.core.tensor_parallel import (
    all_to_all,
    gather_from_sequence_parallel_region,
    reduce_scatter_to_sequence_parallel_region,
)
from megatron.core.transformer.moe.fused_a2a import fused_combine, fused_dispatch
from megatron.core.transformer.moe.moe_utils import (
    get_capacity,
    maybe_move_tensor_to_cpu,
    permute,
    sort_chunks_by_idxs,
    unpermute,
)
from megatron.core.transformer.moe.shared_experts import SharedExpertMLP
from megatron.core.transformer.transformer_config import TransformerConfig

""" We use the following notation throughout this file:
     H: hidden size
     B: micro batch size
     S: sequence length
     TP: tensor model parallel size
     EP: expert model parallel size
     num_local_tokens: S/TP*B
     num_global_tokens: num_local_tokens*TP*EP
"""


class MoETokenDispatcher:
    """
    MoE Token Dispatcher
    """

    def __init__(self, config: TransformerConfig) -> None:
        """
        Initialize the MoE Token Dispatcher.
        """
        self.config = config
        self.shared_experts: Optional[SharedExpertMLP] = None

        self.tp_size = config.expert_tensor_parallel_size
        self.ep_size = config.expert_model_parallel_size

    @property
    def ep_group(self):
        """Get expert model parallel group."""
        return get_expert_model_parallel_group()

    @property
    def tp_group(self):
        """Get expert tensor parallel group."""
        return get_expert_tensor_parallel_group()

    @property
    def tp_rank(self):
        """Get expert tensor parallel rank."""
        return get_expert_tensor_parallel_rank()

    @property
    def tp_ep_group(self):
        """Get expert tensor and model parallel group."""
        return get_expert_tensor_and_model_parallel_group()

    @abstractmethod
    def token_permutation(
        self, tokens: torch.Tensor, probs: torch.Tensor, routing_map: torch.Tensor
    ):
        """Dispatch tokens to experts.

        Args:
            tokens (torch.Tensor): Input tokens.
            probs (torch.Tensor): The routing probability tensor [num_tokens, num_experts].
            routing_map (torch.Tensor): Token to expert mapping tensor.

        Returns:
            torch.Tensor: Tokens tensor.
        """
        raise NotImplementedError("Dispatch function not implemented.")

    @abstractmethod
    def token_unpermutation(self, expert_output: torch.Tensor, bias: torch.Tensor = None):
        """Restores the expert output to its original ordering.

        Args:
            expert_output (torch.Tensor): The output tensor from the expert models.
            bias (torch.Tensor): The bias tensor.

        Returns:
            (torch.Tensor, torch.Tensor): Unpermuted activation and optional bias.
        """
        raise NotImplementedError("Restore function not implemented.")

    def set_shared_experts(self, shared_experts):
        """Set shared expert to the dispatcher."""
        assert self.config.moe_shared_expert_overlap
        self.shared_experts = shared_experts


class MoEAllGatherTokenDispatcher(MoETokenDispatcher):
    """
    AllGather Based Token dispatcher.
    Note that this allgather spans the communication domain of TP*EP:
    """

    def __init__(
        self, num_local_experts: int, local_expert_indices: List[int], config: TransformerConfig
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

        # self.global_local_map: 2D tensor. A mask of mapping between global and local tokens where
        # each element is True if it's between the local_expert_indices. Only useful when cross
        # device token permutation is enabled and **AllGahter** is performed.
        self.global_local_map = None

    def token_permutation(
        self, hidden_states: torch.Tensor, probs: torch.Tensor, routing_map: torch.Tensor
    ):
        """Dispatch tokens to local experts. It's composed of two stages:
        (1) Gather the tokens across the expert parallel devices. After this stage,
        each device receives all of the tokens assigned to its local set of experts
        in its local HBM.
        (2) Permute the tokens locally so that they are grouped by their expert
        assignment.

        Args:
            hidden_states: 3D tensor [S/TP, B, H]. Input tokens.
            probs: 2D tensor [S/TP*B, num_experts]. Each row of probs contains
            the probility distribution across `topk` experts for one local token.
            routing_map: 2D tensor [S/TP*B, num_experts], representing token assignment to
            global experts.

        Returns:
            permuted_local_hidden_states: Permutation of tokens to local experts group.
            tokens_per_expert: the number of tokens each local expert to process.
        """
        self.hidden_shape = hidden_states.shape
        # [S/TP, B, H] -> [S*B/TP, H]
        hidden_states = hidden_states.view(-1, self.hidden_shape[-1])

        # Permute the tokens across the expert parallel devices.
        if self.tp_size > 1 or self.ep_size > 1:
            ## local_indices calculation
            with torch.no_grad():
                # [num_local_tokens, num_experts] -> [num_global_tokens, num_experts], where:
                #     num_local_tokens=(S/TP)*B, num_global_tokens=S*B*EP
                routing_map = gather_from_sequence_parallel_region(
                    routing_map, group=self.tp_ep_group
                )

            ## local_probs calculation
            # max_prob: [S/TP*B, num_experts] -> global_probs: [S*B*EP, num_experts]
            probs = gather_from_sequence_parallel_region(probs, group=self.tp_ep_group)

            # Note that this allgather spans the communication domain of TP*EP.
            #  [(S/TP)*B, H] -> [((S/TP)*B)*(TP*EP), H] = [S*B*EP, H]
            hidden_states = gather_from_sequence_parallel_region(
                hidden_states, group=self.tp_ep_group, use_global_buffer=True
            )
        self.hidden_shape_before_permute = hidden_states.shape

        # The routing map and probs that for local experts.
        self.local_map = routing_map[
            :, self.local_expert_indices[0] : self.local_expert_indices[-1] + 1
        ].contiguous()
        # probs of global token assignment to local experts.
        self.local_probs = probs[
            :, self.local_expert_indices[0] : self.local_expert_indices[-1] + 1
        ].contiguous()

        tokens_per_expert = self.local_map.sum(dim=0).long().cpu()

        (permuted_local_hidden_states, self.reversed_local_input_permutation_mapping) = permute(
            hidden_states,
            self.local_map,
            num_out_tokens=tokens_per_expert.sum(),
            fused=self.config.moe_permute_fusion,
        )

        return permuted_local_hidden_states, tokens_per_expert

    def token_unpermutation(self, hidden_states: torch.Tensor, bias: torch.Tensor = None):
        """
        Reverse process of `dispatch()` which permutes the output of local
        experts locallay and across expert parallel rank into the original order to
        produce the final output.

        Args:
            hidden_states: 2D tensor [num_permuted_tokens_for_local_experts, H],
            output of local experts.
            bias (optional): The bias tensor.

        Returns:
            output_total: un-permuted updated hidden states output from all local experts
            with shape of [S/TP, B, H]
        """
        # Scale the expert output prior to reduction and subsequent to local unpermutation if k > 1.
        # Unpermute the expert output and bias
        permuted_probs = self.local_probs.T.contiguous().masked_select(
            self.local_map.T.contiguous()
        )
        # Here may change permuted_tokens to higher precision if probs use fp32/fp64.
        weighted_hidden_states = hidden_states * permuted_probs.unsqueeze(-1)
        unpermuted_local_hidden = unpermute(
            weighted_hidden_states,
            self.reversed_local_input_permutation_mapping,
            restore_shape=self.hidden_shape_before_permute,
            routing_map=self.local_map,
            fused=self.config.moe_permute_fusion,
        )

        unpermuted_local_bias = None
        if self.add_bias:
            assert bias is not None
            weighted_bias = bias * permuted_probs.unsqueeze(-1)
            unpermuted_local_bias = unpermute(
                weighted_bias,
                self.reversed_local_input_permutation_mapping,
                restore_shape=self.hidden_shape_before_permute,
                routing_map=self.local_map,
                fused=self.config.moe_permute_fusion,
            )

        output_total = unpermuted_local_hidden
        output_bias_total = unpermuted_local_bias

        # Unpermute the tokens across ranks.
        if self.tp_size > 1 or self.ep_size > 1:
            output_total = reduce_scatter_to_sequence_parallel_region(
                output_total, group=self.tp_ep_group
            )
            if self.add_bias:
                # Unpermute the bias across expert parallel devices.
                # bias is duplicated across tensor parallelism ranks;
                output_bias_total = (
                    reduce_scatter_to_sequence_parallel_region(
                        output_bias_total, group=self.tp_ep_group
                    )
                    / self.tp_size
                )

        output_total = output_total.view(self.hidden_shape)
        if self.add_bias:
            output_bias_total = output_bias_total.view(self.hidden_shape)

        # Restore the dtype of the output to the original dtype.
        output_total = output_total.to(hidden_states.dtype)
        if bias is not None:
            output_bias_total = output_bias_total.to(bias.dtype)
        return output_total, output_bias_total


class MoEAlltoAllTokenDispatcher(MoETokenDispatcher):
    """
    AlltoAll-based token dispatcher.

    The workflow of AlltoAll token dispatcher is as follows:
    (1) preprocess(): calculate necessary metadata for communication and permute
    (2) token_permutation(): permute->A2A(EP)->AG(TP)->sort_chunk(if num_local_experts>1)
    (3) token_unpermutation(): sort_chunk(if num_local_experts>1)->RS(TP)->A2A(EP)->unpermute
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
        self.num_local_experts = num_local_experts
        assert config.num_moe_experts is not None
        self.num_experts = config.num_moe_experts
        assert self.num_local_experts > 0, "Expected at least one expert"
        self.local_expert_indices = local_expert_indices
        assert (
            len(self.local_expert_indices) == self.num_local_experts
        ), "Invalid local expert indices"
        for i in range(len(self.local_expert_indices) - 1):
            assert (
                self.local_expert_indices[i] == self.local_expert_indices[i + 1] - 1
            ), "local_expert_indices must be continous"

        # [ep_size]. Represents the number of tokens sent by the current rank to other
        # EP ranks.
        self.input_splits = None
        # [ep_size]. Represents the number of tokens received by the current rank from
        # other EP ranks.
        self.output_splits = None
        # [tp_size]. Represents the number of tokens received by the current rank from
        # other TP ranks.
        self.output_splits_tp = None
        self.permute_idx_device = torch.device("cuda") if self.config.moe_permute_fusion else None
        input_chunk_idxs = torch.arange(
            self.num_experts * self.tp_size, device=self.permute_idx_device
        )
        # [num_local_experts, tp_size * ep_size]. Sort the input chunks by local experts.
        self.sort_input_by_local_experts = input_chunk_idxs.reshape(
            -1, self.num_local_experts
        ).T.ravel()
        # [tp_size * ep_size, num_local_experts]. Restore the output chunks by local experts.
        self.restore_output_by_local_experts = input_chunk_idxs.reshape(
            self.num_local_experts, -1
        ).T.ravel()

        # Token drop and padding.
        # Drop and pad the input to capacity.
        self.drop_and_pad = self.config.moe_pad_expert_input_to_capacity
        if self.drop_and_pad:
            assert self.config.moe_expert_capacity_factor is not None
            self.moe_expert_capacity_factor = self.config.moe_expert_capacity_factor
        self.capacity = None

        # A cuda stream synchronization is needed in self.token_permutation() in some cases,
        # because there are several non-blocking DtoH data transfers called at
        # `self.cuda_dtoh_point`. The synchronization happens at `self.cuda_sync_point`, which is
        # decided based on the MoE and parallel settings. Valid points are "before_permutation_1",
        # "before_ep_alltoall", "before_permutation_2", "before_finish", and "no_sync".
        self.cuda_sync_point = "no_sync"
        self.cuda_sync_point_priority = {
            "before_permutation_1": 0,
            "before_ep_alltoall": 1,
            "before_permutation_2": 2,
            "before_finish": 3,
            "no_sync": 4,
        }
        self.cuda_dtoh_point = "before_permutation_1"
        self.cuda_dtoh_stream = torch.cuda.Stream()

        self.shared_experts = None

    def preprocess(self, routing_map: torch.Tensor) -> torch.Tensor:
        """
        Preprocess token routing map for AlltoAll communication and token permutation.

        This method computes the number of tokens assigned to each expert based on the routing_map.
        It also initializes the necessary data structures for AlltoAll communication, such as input
        and output splits, and the mapping between global tokens and local experts. This method
        should not call any DtoH data copying due to performance consideration. The necessary DtoH
        copies are made on the `self.cuda_dtoh_stream` at `self.cuda_dtoh_point`.

        Args:
            routing_map (torch.Tensor): The mapping of tokens to experts, with shape
                [num_tokens, num_experts].

        Returns:
            torch.Tensor: Tensor containing the number of tokens assigned to local expert.
        """
        if self.drop_and_pad:
            # Drop and pad the input to capacity.
            num_tokens = routing_map.size(0) * self.config.moe_router_topk
            self.capacity = get_capacity(
                num_tokens=num_tokens,
                num_experts=self.num_experts,
                capacity_factor=self.moe_expert_capacity_factor,
            )
            self.num_out_tokens = self.capacity * self.num_experts
            # [num_local_experts], number of tokens processed by each expert.
            num_tokens_per_local_expert = torch.full(
                (self.num_local_experts,),
                self.capacity * self.tp_size * self.ep_size,
                dtype=torch.long,
            )
            # [tp_size * ep_size, num_local_experts]. Represents the number of tokens sent
            # to each local expert by all ranks.
            self.num_global_tokens_per_local_expert = torch.full(
                (self.num_experts * self.tp_size,),
                self.capacity,
                dtype=torch.long,
                device=self.permute_idx_device,
            )
            return num_tokens_per_local_expert

        # [num_experts], number of tokens assigned to each expert from the current rank's input.
        num_local_tokens_per_expert = routing_map.sum(dim=0).long()

        if self.config.moe_expert_capacity_factor is not None:
            # Drop tokens to capacity, no padding.
            self.num_out_tokens = num_local_tokens_per_expert.sum()

            # A synchronization is needed before the first permutation
            # to get the `num_out_tokens` CPU value.
            self._maybe_update_cuda_sync_point("before_permutation_1")
        else:
            # Dropless
            self.num_out_tokens = routing_map.size(0) * self.config.moe_router_topk

        if self.ep_size > 1 or self.tp_size > 1:
            # ===================================================
            # Calculate input_splits, output_splits for alltoall/allgather in variable size.
            # ===================================================
            # [ep_size]. Represents the number of tokens sent by the current rank to other
            # EP ranks.
            self.input_splits = num_local_tokens_per_expert.reshape(
                self.ep_size, self.num_local_experts
            ).sum(axis=1)
            # Gather the global distribution of tokens across ranks.
            # num_global_tokens_per_expert represents the number of tokens sent to each
            # expert by all ranks.
            # [tp_size, ep_size, num_experts]
            num_global_tokens_per_expert = (
                gather_from_sequence_parallel_region(
                    num_local_tokens_per_expert, group=self.tp_ep_group
                )
                .reshape(self.ep_size, self.tp_size, self.num_experts)
                .transpose(0, 1)
            )
            # [tp_size, ep_size, num_experts] -> [tp_size, ep_size, num_local_experts]
            num_global_tokens_per_local_expert = num_global_tokens_per_expert[
                :, :, self.local_expert_indices[0] : self.local_expert_indices[-1] + 1
            ].contiguous()
            # [tp_size, ep_size, num_local_experts] -> [tp_size, ep_size]
            num_global_tokens_per_rank = num_global_tokens_per_local_expert.sum(axis=2)
            # [tp_size, ep_size] -> [ep_size]
            # self.output_splits represents the number of tokens received by the current rank
            # from other EP rank.
            self.output_splits = num_global_tokens_per_rank[self.tp_rank]
            # [tp_size, ep_size] -> [tp_size]
            # self.output_splits_tp represents the number of tokens received by the current
            # rank from other TP rank.
            self.output_splits_tp = num_global_tokens_per_rank.sum(axis=1)
            # [tp_size, ep_size, num_local_experts] -> [num_local_experts]
            num_tokens_per_local_expert = num_global_tokens_per_local_expert.sum(dim=(0, 1))

            # A synchronization is needed before expert parallel AlltoAll communication
            # to get the `input_splits` and `output_splits` CPU values.
            self._maybe_update_cuda_sync_point("before_ep_alltoall")
        else:
            num_global_tokens_per_local_expert = num_local_tokens_per_expert.reshape(
                self.num_experts
            )
            num_tokens_per_local_expert = num_local_tokens_per_expert

            # A synchronization is needed before the returns
            # to get the `num_tokens_per_local_expert` CPU value.
            self._maybe_update_cuda_sync_point("before_finish")

        if self.num_local_experts > 1:
            # [tp_size * ep_size, num_local_experts]. Represents the number of tokens sent
            # to each local expert by all ranks.
            self.num_global_tokens_per_local_expert = num_global_tokens_per_local_expert.view(
                -1, self.num_local_experts
            )
            if not self.config.moe_permute_fusion:
                # A synchronization is needed before permutation 2
                # to get the `num_global_tokens_per_local_expert` CPU value.
                self._maybe_update_cuda_sync_point("before_permutation_2")

        assert (
            self.cuda_sync_point_priority[self.cuda_dtoh_point]
            <= self.cuda_sync_point_priority[self.cuda_sync_point]
        ), "cuda_sync_point must be after cuda_dtoh_point."
        return num_tokens_per_local_expert

    def token_permutation(
        self, hidden_states: torch.Tensor, probs: torch.Tensor, routing_map: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Dispatch tokens to local experts using AlltoAll communication.

        This method performs the following steps:
        1. Preprocess the routing map to get metadata for communication and permutation.
        2. Permute input tokens for AlltoAll communication.
        3. Perform expert parallel AlltoAll communication.
        4. Sort tokens by local expert (if multiple local experts exist).

        Args:
            hidden_states (torch.Tensor): Input token embeddings.
            probs (torch.Tensor): The probabilities of token to experts assignment.
            routing_map (torch.Tensor): The mapping of token to experts assignment.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Permuted token embeddings for local experts.
                - Number of tokens per expert.
        """
        # Preprocess: Get the metadata for communication, permutation and computation operations.
        self.hidden_shape = hidden_states.shape
        self.probs = probs
        self.routing_map = routing_map
        assert probs.dim() == 2, "Expected 2D tensor for probs"
        assert routing_map.dim() == 2, "Expected 2D tensor for token2expert mask"
        assert routing_map.dtype == torch.bool, "Expected bool tensor for mask"
        hidden_states = hidden_states.view(-1, self.hidden_shape[-1])
        tokens_per_expert = self.preprocess(self.routing_map)

        if self.shared_experts is not None:
            self.shared_experts.pre_forward_comm(hidden_states.view(self.hidden_shape))

        # Permutation 1: input to AlltoAll input
        tokens_per_expert = self._maybe_dtoh_and_synchronize(
            "before_permutation_1", tokens_per_expert
        )
        self.hidden_shape_before_permute = hidden_states.shape
        permutated_local_input_tokens, self.reversed_local_input_permutation_mapping = permute(
            hidden_states,
            routing_map,
            num_out_tokens=self.num_out_tokens,
            fused=self.config.moe_permute_fusion,
            drop_and_pad=self.drop_and_pad,
        )

        # Perform expert parallel AlltoAll communication
        tokens_per_expert = self._maybe_dtoh_and_synchronize(
            "before_ep_alltoall", tokens_per_expert
        )
        global_input_tokens = all_to_all(
            self.ep_group, permutated_local_input_tokens, self.output_splits, self.input_splits
        )
        if self.shared_experts is not None:
            self.shared_experts.linear_fc1_forward_and_act(global_input_tokens)

        if self.tp_size > 1:
            if self.output_splits_tp is None:
                output_split_sizes = None
            else:
                output_split_sizes = self.output_splits_tp.tolist()
            global_input_tokens = gather_from_sequence_parallel_region(
                global_input_tokens, group=self.tp_group, output_split_sizes=output_split_sizes
            )

        # Permutation 2: Sort tokens by local expert.
        tokens_per_expert = self._maybe_dtoh_and_synchronize(
            "before_permutation_2", tokens_per_expert
        )
        if self.num_local_experts > 1:
            if self.drop_and_pad:
                global_input_tokens = (
                    global_input_tokens.view(
                        self.tp_size * self.ep_size,
                        self.num_local_experts,
                        self.capacity,
                        *global_input_tokens.size()[1:],
                    )
                    .transpose(0, 1)
                    .contiguous()
                    .flatten(start_dim=0, end_dim=2)
                )
            else:
                global_input_tokens = sort_chunks_by_idxs(
                    global_input_tokens,
                    self.num_global_tokens_per_local_expert.ravel(),
                    self.sort_input_by_local_experts,
                    fused=self.config.moe_permute_fusion,
                )

        tokens_per_expert = self._maybe_dtoh_and_synchronize("before_finish", tokens_per_expert)

        return global_input_tokens, tokens_per_expert

    def token_unpermutation(
        self, hidden_states: torch.Tensor, bias: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Reverse the token permutation to restore the original order.

        This method performs the following steps:
        1. Unsort tokens by local expert (if multiple local experts exist).
        2. Perform expert parallel AlltoAll communication to restore the original order.
        3. Unpermute tokens to restore the original order.

        Args:
            hidden_states (torch.Tensor): Output from local experts.
            bias (torch.Tensor, optional): Bias tensor (not supported).

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                - Unpermuted token embeddings in the original order.
                - None (bias is not supported).
        """
        assert bias is None, "Bias is not supported in MoEAlltoAllTokenDispatcher"

        # Unpermutation 2: Unsort tokens by local expert.
        if self.num_local_experts > 1:
            if self.drop_and_pad:
                hidden_states = (
                    hidden_states.view(
                        self.num_local_experts,
                        self.tp_size * self.ep_size,
                        self.capacity,
                        *hidden_states.size()[1:],
                    )
                    .transpose(0, 1)
                    .contiguous()
                    .flatten(start_dim=0, end_dim=2)
                )
            else:
                hidden_states = sort_chunks_by_idxs(
                    hidden_states,
                    self.num_global_tokens_per_local_expert.T.ravel(),
                    self.restore_output_by_local_experts,
                    fused=self.config.moe_permute_fusion,
                )

        if self.tp_size > 1:
            if self.output_splits_tp is None:
                input_split_sizes = None
            else:
                input_split_sizes = self.output_splits_tp.tolist()
            hidden_states = reduce_scatter_to_sequence_parallel_region(
                hidden_states, group=self.tp_group, input_split_sizes=input_split_sizes
            )

        # Perform expert parallel AlltoAll communication
        # hidden_states: [SEQL, H] -> [SEQL, H/TP]
        permutated_local_input_tokens = all_to_all(
            self.ep_group, hidden_states, self.input_splits, self.output_splits
        )
        if self.shared_experts is not None:
            self.shared_experts.linear_fc2_forward(permutated_local_input_tokens)
            self.shared_experts.post_forward_comm()

        # Unpermutation 1: AlltoAll output to output
        output = unpermute(
            permutated_local_input_tokens,
            self.reversed_local_input_permutation_mapping,
            restore_shape=self.hidden_shape_before_permute,
            probs=self.probs,
            routing_map=self.routing_map,
            fused=self.config.moe_permute_fusion,
            drop_and_pad=self.drop_and_pad,
        )

        # Reshape the output tensor
        output = output.view(self.hidden_shape)

        # Add shared experts output
        if self.shared_experts is not None:
            shared_expert_output = self.shared_experts.get_output()
            output += shared_expert_output
        return output, None

    def _maybe_update_cuda_sync_point(self, point: str):
        """
        Update the CUDA sync point if the priority of the new point is higher than the current
        sync point, which means the new point is reached earlier than the current sync point.
        """
        if (
            self.cuda_sync_point_priority[point]
            < self.cuda_sync_point_priority[self.cuda_sync_point]
        ):
            self.cuda_sync_point = point

    def _maybe_dtoh_and_synchronize(
        self, point: str, tokens_per_expert: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Move all possible GPU tensors to CPU and make a synchronization at the expected point.
        """
        if not self.drop_and_pad:
            if point == self.cuda_dtoh_point:
                # Move all possible GPU tensors to CPU at self.cuda_dtoh_point.
                on_side_stream = torch.cuda.current_stream() != self.cuda_dtoh_stream
                if on_side_stream:
                    self.cuda_dtoh_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(self.cuda_dtoh_stream):
                    # TODO: use MemcpyBatchAsync instead.
                    tokens_per_expert = maybe_move_tensor_to_cpu(
                        tokens_per_expert, record_stream=on_side_stream
                    )
                    self.input_splits = maybe_move_tensor_to_cpu(
                        self.input_splits, as_numpy=True, record_stream=on_side_stream
                    )
                    self.output_splits = maybe_move_tensor_to_cpu(
                        self.output_splits, as_numpy=True, record_stream=on_side_stream
                    )
                    self.output_splits_tp = maybe_move_tensor_to_cpu(
                        self.output_splits_tp, as_numpy=True, record_stream=on_side_stream
                    )
                    self.num_out_tokens = maybe_move_tensor_to_cpu(
                        self.num_out_tokens, record_stream=on_side_stream
                    )
                    if self.num_local_experts > 1 and not self.config.moe_permute_fusion:
                        self.num_global_tokens_per_local_expert = maybe_move_tensor_to_cpu(
                            self.num_global_tokens_per_local_expert, record_stream=on_side_stream
                        )

            if point == self.cuda_sync_point:
                # Synchronize with the dtoh stream at self.cuda_sync_point.
                self.cuda_dtoh_stream.synchronize()

        return tokens_per_expert


class _DispatchManager(ABC):
    """
    A manager class to handle dispatch and combine processes for MoE models.

    DispatcherManager handles token dispatching according to the routing_map of format
    [num_local_tokens, world_size, num_instances]. The routing_map is a 3D tensor where each
    element indicates whether a token should be sent to a specific rank.

    num_instances is the maximum number of tokens instances dispatched into a target rank, it
    can be the number of local experts, or the size of sub_group.
    """

    @abstractmethod
    def setup_metadata(self, routing_map: torch.Tensor, probs: torch.Tensor):
        """Set up metadata of routing_map and probs."""
        pass

    @abstractmethod
    def dispatch(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Dispatch the hidden_states according to the routing_map."""
        pass

    @abstractmethod
    def combine(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Combine the hidden_states after expert processing."""
        pass

    @abstractmethod
    def get_dispached_metadata(self) -> torch.Tensor:
        """Get the metadata of the dispatched hidden_states."""
        pass

    @abstractmethod
    def get_permuted_hidden_states_by_experts(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Get the permuted hidden states by instances."""
        pass

    @abstractmethod
    def get_restored_hidden_states_by_experts(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Get the restored hidden states by instances."""
        pass


class _DeepepManager(_DispatchManager):
    """
    A manager class to handle fused all-to-all communication processes for MoE models using
    DeepEP backend. See https://github.com/deepseek-ai/deepep for more details.

    The workflow of the DeepEP dispatcher is:
    (1) setup_metadata(): Process routing map and probabilities to prepare dispatch metadata
    (2) dispatch():
        - Use fused kernel to permute tokens and perform all-to-all communication in single step
    (3) get_permuted_hidden_states_by_instances():
        - Convert routing map and probabilities to multihot format
        - Permute tokens using fused kernel
    (4) get_restored_hidden_states_by_instances():
        - Reverse permutation using fused kernel
    (5) combine():
        - Reverse process using fused kernel to unpermute and perform all-to-all in single step

    This implementation uses fused communication kernels (fused_dispatch/fused_combine) that
    combine permutation and communication operations for improved efficiency compared to
    separate permute+alltoall steps.
    """

    def __init__(
        self,
        group: torch.distributed.ProcessGroup,
        router_topk: int,
        permute_fusion: bool = False,
        capacity_factor: float = None,
        num_experts: int = None,
        num_local_experts: int = None,
    ):
        self.group = group
        self.router_topk = router_topk
        self.capacity_factor = capacity_factor
        self.permute_fusion = permute_fusion
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.router_dtype = router_dtype

        # Metadata
        self.token_indices = None
        self.token_probs = None
        # Handle used for combine operation
        self.handle = None

        if fused_dispatch is None:
            raise ImportError(
                "DeepEP is not installed. Please install DeepEP package from "
                "https://github.com/deepseek-ai/deepep."
            )

    def setup_metadata(self, routing_map: torch.Tensor, probs: torch.Tensor):
        num_tokens = routing_map.shape[0]

        routing_map = routing_map.reshape(num_tokens, self.num_experts)
        probs = probs.reshape(num_tokens, self.num_experts)
        # Convert the format of routing map from multihot to indices.
        self.token_probs, self.token_indices = torch.topk(probs, self.router_topk, dim=-1)
        # Mask the indices of dropped tokens with -1
        if self.capacity_factor is not None:
            mask = self.token_probs == 0
            self.token_indices = self.token_indices.masked_fill(mask, -1)

    def dispatch(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # DeepEP only supports float32 probs
        if self.token_probs.dtype != torch.float32:
            if self.token_probs.dtype in [torch.bfloat16, torch.float16]:
                print("DeepEP only supports float32 probs, please set --moe-router-dtype=fp32")
            self.token_probs = self.token_probs.float()  # downcast or upcast
        hidden_states, dispatched_indices, dispatched_probs, num_tokens_per_expert, handle = (
            fused_dispatch(
                hidden_states, self.token_indices, self.token_probs, self.num_experts, self.group
            )
        )
        self.handle = handle
        self.tokens_per_expert = num_tokens_per_expert
        self.dispatched_indices = dispatched_indices
        self.dispatched_probs = dispatched_probs

        return hidden_states

    def _indices_to_multihot(self, indices, probs):
        """
        Converts a tensor of indices to a multihot vector.

        Args:
            indices (torch.Tensor): [num_tokens, topk] token indices, where -1 means masked out.
            probs (torch.Tensor): [num_tokens, topk] token probabilities.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - routing_map: Multihot vector.
                - probs: Multihot probabilities.
        """
        batch_size = indices.shape[0]
        multihot_routing_map = torch.zeros(
            (batch_size, self.num_local_experts), dtype=torch.long, device=indices.device
        )

        multihot_probs = torch.zeros(
            (batch_size, self.num_local_experts), dtype=torch.float, device=indices.device
        )

        mask = indices != -1
        valid_indices = indices[mask]
        row_indices = torch.arange(batch_size, device=indices.device).repeat_interleave(
            mask.sum(dim=1)
        )
        multihot_routing_map[row_indices, valid_indices] = 1
        multihot_probs[row_indices, valid_indices] = probs[mask]
        return multihot_routing_map.bool(), multihot_probs

    def get_dispached_metadata(self) -> torch.Tensor:
        return self.dispatched_indices, self.dispatched_probs

    def get_number_of_tokens_per_expert(self) -> torch.Tensor:
        """
        Get the number of tokens per expert.
        """
        return self.tokens_per_expert

    def combine(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, event = fused_combine(hidden_states, self.group, self.handle)
        # Release the handle after combine operation
        self.handle = None
        return hidden_states

    def get_permuted_hidden_states_by_experts(self, hidden_states: torch.Tensor) -> torch.Tensor:
        self.dispatched_routing_map, self.dispatched_probs = self._indices_to_multihot(
            self.dispatched_indices, self.dispatched_probs
        )
        self.hidden_shape_before_permute = hidden_states.shape
        hidden_states, self.reversed_mapping_for_combine = permute(
            hidden_states,
            self.dispatched_routing_map,
            num_out_tokens=sum(self.tokens_per_expert),
            fused=self.permute_fusion,
        )
        return hidden_states

    def get_restored_hidden_states_by_experts(self, hidden_states: torch.Tensor) -> torch.Tensor:
        assert self.dispatched_probs.dtype == torch.float32, "DeepEP only supports float32 probs"
        if self.router_dtype == "fp64":
            self.dispatched_probs = self.dispatched_probs.to(torch.float64)
        hidden_states = unpermute(
            hidden_states,
            self.reversed_mapping_for_combine,
            restore_shape=self.hidden_shape_before_permute,
            routing_map=self.dispatched_routing_map,
            probs=self.dispatched_probs,
            fused=self.permute_fusion,
        )
        return hidden_states


class MoEFlexTokenDispatcher(MoETokenDispatcher):
    """
    Flexible token dispatcher for MoE models with Efficient-A2A communication kernels.
    """

    def __init__(
        self, num_local_experts: int, local_expert_indices: List[int], config: TransformerConfig
    ):
        super().__init__(config)

        self.num_local_experts = num_local_experts
        self.local_expert_indices = local_expert_indices
        assert self.tp_size * self.ep_size > 1, "Flex token dispatcher requires TPxEP > 1"
        assert (
            self.config.moe_enable_deepep
        ), "DeepEP is not enabled. Please set --moe-enable-deepep to use DeepEP backend."
        assert (
            self.config.moe_pad_expert_input_to_capacity is False
        ), "Flex token dispatcher does not support --moe-pad-expert-input-to-capacity"
        self._comm_manager = _DeepepManager(
            group=self.tp_ep_group,
            router_topk=self.tp_size * self.config.moe_router_topk,
            permute_fusion=self.config.moe_permute_fusion,
            capacity_factor=self.config.moe_expert_capacity_factor,
            num_experts=self.tp_size * self.config.num_moe_experts,
            num_local_experts=self.num_local_experts,
            router_dtype=self.config.moe_router_dtype,
        )

    def set_shared_experts(self, shared_experts):
        raise NotImplementedError(
            "Shared expert overlap is not supported in Flex Token Dispatcher."
        )

    def _initialize_metadata(self, routing_map: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
        """
        Initialize the routing map and probs to a unified format covering the TPxEP group.
        This design decouples the communication group from underlying model parallelism groups,
        such that the communication strategy of tokens can be agnostic of TP size and EP size.

        This function expands the routing_map from shape [num_local_tokens, num_experts] to
        [num_local_tokens, world_size, num_local_experts]. Each element in the routing_map
        indicates whether a token should be sent to a specific rank. Specifically, the
        routing_map is replicated across TP group since each TP ranks in a TP group should
        receive the same tokens.
        """
        num_local_tokens = routing_map.shape[0]
        world_size = self.tp_size * self.ep_size
        # Organize routing map and probs to [num_local_tokens, world_size, num_local_experts]
        routing_map = (
            routing_map.reshape(num_local_tokens, self.ep_size, 1, self.num_local_experts)
            .expand(-1, -1, self.tp_size, -1)
            .reshape(num_local_tokens, world_size, self.num_local_experts)
        ).contiguous()
        probs = (
            probs.reshape(num_local_tokens, self.ep_size, 1, self.num_local_experts)
            .expand(-1, -1, self.tp_size, -1)
            .reshape(num_local_tokens, world_size, self.num_local_experts)
        ).contiguous()
        return routing_map, probs

    def token_permutation(
        self, hidden_states: torch.Tensor, probs: torch.Tensor, routing_map: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.hidden_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_shape[-1])

        # Initialize metadata
        routing_map, probs = self._initialize_metadata(routing_map, probs)

        self._comm_manager.setup_metadata(routing_map, probs)
        hidden_states = self._comm_manager.dispatch(hidden_states)
        global_input_tokens = self._comm_manager.get_permuted_hidden_states_by_experts(
            hidden_states
        )
        tokens_per_expert = self._comm_manager.get_number_of_tokens_per_expert()

        return global_input_tokens, tokens_per_expert

    def token_unpermutation(
        self, hidden_states: torch.Tensor, bias: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        assert bias is None, "Bias is not supported in MoEFlexTokenDispatcher"
        hidden_states = self._comm_manager.get_restored_hidden_states_by_experts(hidden_states)
        hidden_states = self._comm_manager.combine(hidden_states)

        return hidden_states.view(self.hidden_shape), None
