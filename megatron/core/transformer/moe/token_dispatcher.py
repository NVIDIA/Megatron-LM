# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import torch

from megatron.core import utils
from megatron.core.config import is_experimental_enabled
from megatron.core.fp8_utils import get_fp8_align_size
from megatron.core.fusions.fused_indices_converter import fused_indices_to_multihot
from megatron.core.fusions.fused_pad_routing_map import fused_pad_routing_map
from megatron.core.tensor_parallel import (
    all_to_all,
    gather_from_sequence_parallel_region,
    reduce_scatter_to_sequence_parallel_region,
)
from megatron.core.transformer.moe.fused_a2a import (
    fused_combine,
    fused_dispatch,
    set_deepep_num_sms,
)
from megatron.core.transformer.moe.moe_utils import (
    ProcessGroupCollection,
    get_capacity,
    maybe_move_tensor_to_cpu,
    pad_routing_map,
    permute,
    sort_chunks_by_idxs,
    unpermute,
)
from megatron.core.transformer.moe.shared_experts import SharedExpertMLP
from megatron.core.transformer.transformer_config import TransformerConfig

logger = logging.getLogger(__name__)

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

    def __init__(
        self, config: TransformerConfig, pg_collection: Optional[ProcessGroupCollection] = None
    ) -> None:
        """
        Initialize the MoE Token Dispatcher.

        Args:
            config (TransformerConfig): Configuration for the MoE layer.
            pg_collection (ProcessGroupCollection, optional): Process groups for MoE operations.
        """
        self.config = config
        self.shared_experts: Optional[SharedExpertMLP] = None

        self.ep_group = pg_collection.ep
        # use pg_collection.expt_tp_group as tensor parallel group in this module.
        self.tp_group = pg_collection.expt_tp
        self.tp_ep_group = pg_collection.tp_ep

        self.tp_size = utils.get_pg_size(self.tp_group)
        self.tp_rank = utils.get_pg_rank(self.tp_group)
        self.ep_size = utils.get_pg_size(self.ep_group)

        # Attributes that need to be captured in cudagraph. These attributes are returned
        # as cudagraph outputs when the cuda_graph_scope contains moe_preprocess.
        self.cudagraph_attrs = []
        self.valid_cudagraph_attrs = None

    @abstractmethod
    def dispatch_preprocess(
        self, tokens: torch.Tensor, routing_map: torch.Tensor, probs: torch.Tensor
    ):
        """Prepares tokens for dispatch without inter-device communication.

        This method should handle all local computations like tensor rearrangement and
        metadata extraction before the main communication step.

        Note:
            Try to avoid any communication here to enable optimal computation-communication
            overlapping when enabling communication overlap, since communications in the
            same stream runs sequentially and may get exposed.

        Args:
            tokens (torch.Tensor): Input tokens.
            routing_map (torch.Tensor): Token to expert mapping tensor.
            probs (torch.Tensor): The routing probability tensor, [num_tokens, num_experts].

        Returns:
            A tuple of preprocessed tokens and probabilities.
        """
        raise NotImplementedError("dispatch_preprocess function not implemented.")

    @abstractmethod
    def token_dispatch(self, hidden_states: torch.Tensor, probs: torch.Tensor):
        """Dispatches tokens to expert devices using communication.

        This method performs the main communication (e.g., All-to-All) to send
        tokens to the devices where their assigned experts reside.

        Args:
            hidden_states (torch.Tensor): Preprocessed hidden states to be dispatched.
            probs (torch.Tensor): Preprocessed probabilities for each token-expert pair.

        Returns:
            A tuple of dispatched tokens and probabilities.
        """
        raise NotImplementedError("token_dispatch function not implemented.")

    @abstractmethod
    def dispatch_postprocess(self, hidden_states: torch.Tensor, probs: torch.Tensor):
        """Performs local processing after token dispatch communication.

        This method handles post-communication tasks like token reordering and
        preparing metadata for the expert forward pass.

        Note:
            Try to avoid any communication here to enable optimal computation-communication
            overlapping when enabling communication overlap, since communications in the
            same stream runs sequentially and may get exposed.

        Args:
            hidden_states (torch.Tensor): Dispatched hidden states.
            probs (torch.Tensor): Dispatched probabilities.

        Returns:
            A tuple containing the permuted tokens for experts, the number of
            tokens per expert, and the permuted probabilities.
        """
        raise NotImplementedError("dispatch_postprocess function not implemented.")

    @abstractmethod
    def combine_preprocess(self, hidden_states):
        """Prepares expert outputs for the combine step.

        This method performs local computations on expert outputs before the
        communication step for combining them.

        Note:
            Try to avoid any communication here to enable optimal computation-communication
            overlapping when enabling communication overlap, since communications in the
            same stream runs sequentially and may get exposed.

        Args:
            hidden_states (torch.Tensor): The output tensor from the experts.

        Returns:
            The preprocessed expert output.
        """
        raise NotImplementedError("combine_preprocess function not implemented.")

    @abstractmethod
    def token_combine(self, hidden_states):
        """Combines expert outputs across devices using communication.

        This method aggregates expert outputs from different devices via
        communication (e.g., All-to-All or Reduce-Scatter).

        Args:
            hidden_states (torch.Tensor): Preprocessed output from experts.

        Returns:
            The combined expert outputs.
        """
        raise NotImplementedError("token_combine function not implemented.")

    @abstractmethod
    def combine_postprocess(self, hidden_states):
        """Performs local processing after token combine.

        This method handles post-communication tasks like unpermuting and
        reshaping to restore the original tensor structure.

        Note:
            Try to avoid any communication here to enable optimal computation-communication
            overlapping when enabling communication overlap, since communications in the
            same stream runs sequentially and may get exposed.

        Args:
            hidden_states (torch.Tensor): Combined hidden states from token combination

        Returns:
            The final output tensor.
        """
        raise NotImplementedError("combine_postprocess function not implemented.")

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
        self,
        num_local_experts: int,
        local_expert_indices: List[int],
        config: TransformerConfig,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ) -> None:
        """Initialize the AllGather based token dispatcher.

        Args:
            num_local_experts (int): Number of local experts.
            local_expert_indices (List[int]): Indices of local experts.
            config (TransformerConfig): Configuration for the MoE layer.
            pg_collection (ProcessGroupCollection, optional): Process groups for MoE operations.
        """
        super().__init__(config=config, pg_collection=pg_collection)
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

        # Attributes that need to be captured in cudagraph. These attributes are returned
        # as cudagraph outputs when the cuda_graph_scope contains moe_preprocess.
        self.cudagraph_attrs = ['routing_map']

    def dispatch_preprocess(
        self, hidden_states: torch.Tensor, routing_map: torch.Tensor, probs: torch.Tensor
    ):
        """Reshapes hidden states and caches the routing map."""
        self.hidden_shape = hidden_states.shape
        # [S/TP, B, H] -> [S*B/TP, H]
        hidden_states = hidden_states.view(-1, self.hidden_shape[-1])
        self.routing_map = routing_map
        return hidden_states, probs

    def token_dispatch(self, hidden_states, probs):
        """Gathers tokens from all TP*EP ranks using AllGather."""

        # Permute the tokens across the expert parallel devices.
        if self.tp_size > 1 or self.ep_size > 1:
            ## local_indices calculation
            with torch.no_grad():
                # [num_local_tokens, num_experts] -> [num_global_tokens, num_experts], where:
                #     num_local_tokens=(S/TP)*B, num_global_tokens=S*B*EP
                self.routing_map = gather_from_sequence_parallel_region(
                    self.routing_map, group=self.tp_ep_group
                )

            ## local_probs calculation
            # max_prob: [S/TP*B, num_experts] -> global_probs: [S*B*EP, num_experts]
            probs = gather_from_sequence_parallel_region(probs, group=self.tp_ep_group)
            # Note that this allgather spans the communication domain of TP*EP.
            #  [(S/TP)*B, H] -> [((S/TP)*B)*(TP*EP), H] = [S*B*EP, H]
            hidden_states = gather_from_sequence_parallel_region(
                hidden_states, group=self.tp_ep_group, use_global_buffer=True
            )

        return hidden_states, probs

    def dispatch_postprocess(self, hidden_states, probs):
        """After gathering in token_dispatch, this method identifies tokens for local experts and
        permutes them for expert processing.
        """
        self.hidden_shape_before_permute = hidden_states.shape

        # The routing map and probs that for local experts.
        self.local_map = self.routing_map[
            :, self.local_expert_indices[0] : self.local_expert_indices[-1] + 1
        ].contiguous()
        # probs of global token assignment to local experts.
        self.local_probs = probs[
            :, self.local_expert_indices[0] : self.local_expert_indices[-1] + 1
        ].contiguous()

        tokens_per_expert = self.local_map.sum(dim=0).long().cpu()

        (permuted_local_hidden_states, _, self.reversed_local_input_permutation_mapping) = permute(
            hidden_states,
            self.local_map,
            num_out_tokens=tokens_per_expert.sum(),
            fused=self.config.moe_permute_fusion,
        )

        self.local_probs = self.local_probs.T.contiguous().masked_select(
            self.local_map.T.contiguous()
        )
        self.routing_map = None
        return permuted_local_hidden_states, tokens_per_expert, self.local_probs

    def combine_preprocess(self, hidden_states):
        """
        Reverses token permutation to restore original ordering before reduction operations.

        This method unpermutes the expert outputs using the cached permutation mapping
        from the dispatch phase. The unpermutation operation restores tokens to their
        original sequence positions, preparing them for the subsequent reduction scatter
        operation that will aggregate results across ranks.
        """
        unpermuted_local_hidden = unpermute(
            hidden_states,
            self.reversed_local_input_permutation_mapping,
            restore_shape=self.hidden_shape_before_permute,
            routing_map=self.local_map,
            fused=self.config.moe_permute_fusion,
        )
        return unpermuted_local_hidden

    def token_combine(self, hidden_states):
        """Combines expert outputs using Reduce-Scatter.

        This method performs the ReduceScatter communication operation to collect expert
        outputs from their processing ranks and redistribute tokens back to the ranks that
        originally held them. This completes the expert processing
        communication pattern and prepares tokens for final unpermutation.
        """
        # Unpermute the tokens across ranks.
        if self.tp_size > 1 or self.ep_size > 1:
            hidden_states = reduce_scatter_to_sequence_parallel_region(
                hidden_states.to(self.local_probs.dtype), group=self.tp_ep_group
            ).to(hidden_states.dtype)
        return hidden_states

    def combine_postprocess(self, hidden_states):
        """Restores the original tensor shape."""
        return hidden_states.view(self.hidden_shape)


class MoEAlltoAllTokenDispatcher(MoETokenDispatcher):
    """
    AlltoAll-based token dispatcher.

    The workflow of AlltoAll token dispatcher is as follows:
    (1) preprocess: calculate necessary metadata for communication and permute
    (2) dispatch process: permute tokens
    (3) token dispatch: A2A(EP)
    (4) dispatch postprocess: AG(TP)->sort_chunk(if num_local_experts>1)
    (5) combine preprocess: sort_chunk(if num_local_experts>1)->RS(TP)
    (6) token combine: A2A(EP)
    (7) combine postprocess: unpermute tokens
    """

    # DtoH copies are performed on this stream for overlapping with the main stream.
    cuda_dtoh_stream = None

    def __init__(
        self,
        num_local_experts: int,
        local_expert_indices: List[int],
        config: TransformerConfig,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ) -> None:
        """
        Initialize the AlltoAll token dispatcher.

        Args:
            num_local_experts (int): Number of local experts on the current device.
            local_expert_indices (List[int]): Indices of local experts on the current device.
            config (TransformerConfig): Configuration for the transformer model.
            pg_collection (ProcessGroupCollection, optional): Process groups for MoE operations.
        """
        super().__init__(config=config, pg_collection=pg_collection)
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
            ), "local_expert_indices must be continuous"

        # [ep_size]. Represents the number of tokens sent by the current rank to other
        # EP ranks.
        self.input_splits = None
        # [ep_size]. Represents the number of tokens received by the current rank from
        # other EP ranks.
        self.output_splits = None
        # [tp_size]. Represents the number of tokens received by the current rank from
        # other TP ranks.
        self.output_splits_tp = None
        self.permute_idx_device = torch.device("cuda") if self.config.moe_permute_fusion else "cpu"
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

        # A cuda stream synchronization is needed in during token permutation in some cases,
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
        if (
            config.cuda_graph_impl == "transformer_engine"
            and 'moe_preprocess' in config.cuda_graph_scope
        ):
            self.cuda_dtoh_point = "before_ep_alltoall"
        else:
            self.cuda_dtoh_point = "before_permutation_1"
        if MoEAlltoAllTokenDispatcher.cuda_dtoh_stream is None:
            MoEAlltoAllTokenDispatcher.cuda_dtoh_stream = torch.cuda.Stream()

        # Attributes that need to be captured in cudagraph. These attributes are returned
        # as cudagraph outputs when the cuda_graph_scope contains moe_preprocess.
        self.cudagraph_attrs = [
            'tokens_per_expert',
            'input_splits',
            'output_splits',
            'output_splits_tp',
            'num_out_tokens',
            'num_global_tokens_per_local_expert',
            'reversed_local_input_permutation_mapping',
            'routing_map',
        ]

        self.shared_experts = None

    def set_shared_experts(self, shared_experts):
        """Set shared expert to the dispatcher."""
        super().set_shared_experts(shared_experts)
        if shared_experts.use_shared_expert_gate:
            self.cudagraph_attrs.append('shared_experts.gate_score')
        self.cudagraph_attrs.append('shared_experts.cached_fc1_input')

    def preprocess(self, routing_map: torch.Tensor) -> torch.Tensor:
        """
        Preprocesses the token routing map for All-to-All communication and token permutation.

        This method computes the number of tokens assigned to each expert based on the routing_map.
        It also initializes necessary data structures for All-to-All communication, such as input
        and output splits, and the mapping between global tokens and local experts. This method
        should not call any DtoH data copying due to performance consideration. The necessary DtoH
        copies are made on the `self.cuda_dtoh_stream` at `self.cuda_dtoh_point`.

        Args:
            routing_map (torch.Tensor): The mapping of tokens to experts.

        Returns:
            A tensor with the number of tokens for each local expert.
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

        if (
            self.config.moe_expert_capacity_factor is not None
            or self.config.moe_router_padding_for_fp8
        ):
            # When using token dropping or router padding, output size is dynamic.
            # Need to sync output size GPU->CPU before allocating output buffer
            self.num_out_tokens = num_local_tokens_per_expert.sum()
            self._maybe_update_cuda_sync_point("before_permutation_1")
        else:
            # For dropless training, output size is static (num_tokens * topk)
            # No explicit sync needed
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

    def dispatch_preprocess(
        self, hidden_states: torch.Tensor, routing_map: torch.Tensor, probs: torch.Tensor
    ):
        """Prepares hidden states and probabilities for dispatch.

        This method reshapes the hidden states, computes communication metadata,
        and permutes the tokens and probabilities before the All-to-All communication.

        Args:
            hidden_states (torch.Tensor): Input token embeddings.
            routing_map (torch.Tensor): The mapping of tokens to experts.
            probs (torch.Tensor): Routing probabilities.

        Returns:
            A tuple of permuted hidden states and probabilities.
        """
        # Preprocess: Get the metadata for communication, permutation and computation operations.
        self.hidden_shape = hidden_states.shape
        self.probs = probs
        self.routing_map = routing_map
        assert probs.dim() == 2, "Expected 2D tensor for probs"
        assert routing_map.dim() == 2, "Expected 2D tensor for token2expert mask"
        assert routing_map.dtype == torch.bool, "Expected bool tensor for mask"
        hidden_states = hidden_states.view(-1, self.hidden_shape[-1])

        if self.config.moe_router_padding_for_fp8:
            pad_multiple = get_fp8_align_size(self.config.fp8_recipe)
            if is_experimental_enabled() and self.config.moe_permute_fusion:
                self.routing_map = fused_pad_routing_map(self.routing_map, pad_multiple)
            else:
                self.routing_map = pad_routing_map(self.routing_map, pad_multiple)
        self.tokens_per_expert = self.preprocess(self.routing_map)

        if self.shared_experts is not None:
            self.shared_experts.pre_forward_comm(hidden_states.view(self.hidden_shape))

        # Permutation 1: input to AlltoAll input
        self.tokens_per_expert = self._maybe_dtoh_and_synchronize(
            "before_permutation_1", self.tokens_per_expert
        )
        self.hidden_shape_before_permute = hidden_states.shape
        (
            permutated_local_input_tokens,
            permuted_probs,
            self.reversed_local_input_permutation_mapping,
        ) = permute(
            hidden_states,
            self.routing_map,
            probs=probs,
            num_out_tokens=self.num_out_tokens,
            fused=self.config.moe_permute_fusion,
            drop_and_pad=self.drop_and_pad,
        )
        return permutated_local_input_tokens, permuted_probs

    def token_dispatch(self, permutated_local_input_tokens, permuted_probs):
        """
        Perform all-to-all communication for dispatching tokens.

        This method performs the all-to-all communication step to dispatch tokens across
        expert parallel ranks. It synchronizes metadata at the appropriate point before
        performing the communication.

        Args:
            permutated_local_input_tokens (torch.Tensor): Pre-permuted input tokens.
            permuted_probs (torch.Tensor): Pre-permuted probabilities.

        Returns:
            A tuple of tokens and probabilities after All-to-All.
        """
        # Perform expert parallel AlltoAll communication
        self.tokens_per_expert = self._maybe_dtoh_and_synchronize(
            "before_ep_alltoall", self.tokens_per_expert
        )
        global_input_tokens = all_to_all(
            self.ep_group, permutated_local_input_tokens, self.output_splits, self.input_splits
        )
        global_probs = all_to_all(
            self.ep_group, permuted_probs, self.output_splits, self.input_splits
        )

        return global_input_tokens, global_probs

    def dispatch_postprocess(self, global_input_tokens, global_probs):
        """Post-processes tokens after All-to-All communication.

        This involves an All-Gather in the tensor parallel dimension and sorting
        tokens by expert if there are multiple local experts.

        Args:
            global_input_tokens (torch.Tensor): Tokens after All-to-All.
            global_probs (torch.Tensor): Probabilities after All-to-All.

        Returns:
            A tuple of processed tokens, token counts per expert, and processed probabilities.
        """
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
            global_probs = gather_from_sequence_parallel_region(
                global_probs, group=self.tp_group, output_split_sizes=output_split_sizes
            )

        # Permutation 2: Sort tokens by local expert.
        self.tokens_per_expert = self._maybe_dtoh_and_synchronize(
            "before_permutation_2", self.tokens_per_expert
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
                global_probs = (
                    global_probs.view(
                        self.tp_size * self.ep_size,
                        self.num_local_experts,
                        self.capacity,
                        *global_probs.size()[1:],
                    )
                    .transpose(0, 1)
                    .contiguous()
                    .flatten(start_dim=0, end_dim=2)
                )
            else:
                global_input_tokens, global_probs = sort_chunks_by_idxs(
                    global_input_tokens,
                    self.num_global_tokens_per_local_expert.ravel(),
                    self.sort_input_by_local_experts,
                    probs=global_probs,
                    fused=self.config.moe_permute_fusion,
                )

        tokens_per_expert = self._maybe_dtoh_and_synchronize(
            "before_finish", self.tokens_per_expert
        )
        self.tokens_per_expert = None
        return global_input_tokens, tokens_per_expert, global_probs

    def combine_preprocess(self, hidden_states):
        """Prepares hidden states for token combination after expert computations.

        This may involve un-sorting tokens and a Reduce-Scatter in the tensor
        parallel dimension.
        """
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
                hidden_states, _ = sort_chunks_by_idxs(
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
                hidden_states.to(self.probs.dtype),
                group=self.tp_group,
                input_split_sizes=input_split_sizes,
            ).to(hidden_states.dtype)

        return hidden_states

    def token_combine(
        self,
        hidden_states: torch.Tensor,
        async_finish: bool = True,
        allocate_on_comm_stream: bool = True,
    ):
        """Executes fused un-permutation and communication using DeepEP kernels.

        This method performs the inverse AlltoAll communication operation to collect expert
        outputs from their processing ranks and redistribute them back to the ranks that
        originally held the corresponding tokens. This completes the expert processing
        communication pattern and prepares tokens for final unpermutation.

        Args:
            hidden_states (torch.Tensor): Expert outputs ready for combination
            async_finish (bool): Whether to use asynchronous communication completion
            allocate_on_comm_stream (bool): Whether to allocate buffers on communication stream

        Returns:
            Tokens after the All-to-All communication for combining.
        """
        # Perform expert parallel AlltoAll communication
        # hidden_states: [SEQL, H] -> [SEQL, H/TP]
        permutated_local_input_tokens = all_to_all(
            self.ep_group, hidden_states, self.input_splits, self.output_splits
        )
        return permutated_local_input_tokens

    def combine_postprocess(self, permutated_local_input_tokens):
        """Finalizes token reconstruction with un-permutation and reshaping.

        This method un-permutes the tokens back to their original order,
        reshapes the tensor to its original shape, and adds the shared
        expert output if enabled.

        Args:
            permutated_local_input_tokens (torch.Tensor): Permuted hidden states from token combine.

        Returns:
            The final MoE layer output reshaped to its original dimensions.
        """
        if self.shared_experts is not None:
            self.shared_experts.linear_fc2_forward(permutated_local_input_tokens)
            self.shared_experts.post_forward_comm()

        # Unpermutation 1: AlltoAll output to output
        output = unpermute(
            permutated_local_input_tokens,
            self.reversed_local_input_permutation_mapping,
            restore_shape=self.hidden_shape_before_permute,
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
        return output

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
                self.d2h_event = self.cuda_dtoh_stream.record_event()

            if point == self.cuda_sync_point:
                # Synchronize with the DtoH stream at self.cuda_sync_point.
                self.d2h_event.synchronize()

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
        num_local_experts: int,
        router_topk: int,
        num_experts: int,
        config: TransformerConfig,
    ):
        """
        Initialize the DeepEP dispatcher.

        Args:
            group (torch.distributed.ProcessGroup): The process group to use for communication.
                This should be the ETPxEP group.
            num_local_experts (int): The number of local experts.
            router_topk (int): The number of experts for each token to select.
            num_experts (int): The total number of experts in the group.
            config (TransformerConfig): The configuration for the transformer model.
        """
        self.group = group
        self.num_local_experts = num_local_experts
        self.config = config

        self.router_topk = router_topk
        self.num_experts = num_experts
        self.router_dtype = config.moe_router_dtype
        self.capacity_factor = config.moe_expert_capacity_factor
        self.permute_fusion = config.moe_permute_fusion

        # Metadata
        self.token_indices: Optional[torch.Tensor] = None
        self.token_probs: Optional[torch.Tensor] = None
        # Handle used for combine operation
        self.handle = None

        if fused_dispatch is None:
            raise ImportError(
                "DeepEP is not installed. Please install DeepEP package from "
                "https://github.com/deepseek-ai/deepep."
            )
        set_deepep_num_sms(config.moe_deepep_num_sms)

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

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        async_finish: bool = False,
        allocate_on_comm_stream: bool = False,
    ) -> torch.Tensor:
        # DeepEP only supports float32 probs
        if self.token_probs.dtype != torch.float32:
            if self.token_probs.dtype in [torch.bfloat16, torch.float16]:
                logger.info(
                    "DeepEP only supports float32 probs, please set --moe-router-dtype=fp32"
                )
            self.token_probs = self.token_probs.float()  # downcast or upcast
        hidden_states, dispatched_indices, dispatched_probs, num_tokens_per_expert, handle = (
            fused_dispatch(
                hidden_states,
                self.token_indices,
                self.token_probs,
                self.num_experts,
                self.group,
                async_finish=async_finish,
                allocate_on_comm_stream=allocate_on_comm_stream,
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
            A tuple of (routing_map, probs), where routing_map is the multihot vector
            and probs is the multihot probabilities.
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

    def combine(
        self,
        hidden_states: torch.Tensor,
        async_finish: bool = False,
        allocate_on_comm_stream: bool = False,
    ) -> torch.Tensor:
        hidden_states, _ = fused_combine(
            hidden_states,
            self.group,
            self.handle,
            async_finish=async_finish,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )
        # Release the handle after combine operation
        self.handle = None
        return hidden_states

    def _pad_routing_map(
        self, routing_map: torch.Tensor, tokens_per_expert: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pad the routing map to the nearest multiple of the pad_multiple.
        """
        pad_multiple = get_fp8_align_size(self.config.fp8_recipe)

        num_input_tokens = routing_map.shape[0]
        target_tokens_per_expert = (
            torch.ceil(tokens_per_expert / pad_multiple) * pad_multiple
        ).long()

        # Check if there are enough tokens to pad
        enough_tokens_to_pad = torch.all(target_tokens_per_expert <= num_input_tokens)
        if not enough_tokens_to_pad:
            logger = logging.getLogger(__name__)
            logger.warning(
                "Not enough tokens to pad. The total number of tokens received in this rank "
                "is smaller than the target number of tokens for each expert. "
                "Falling back to explicit padding within GroupedMLP"
            )
        else:
            if is_experimental_enabled() and self.permute_fusion:
                from megatron.core.fusions.fused_pad_routing_map import fused_pad_routing_map

                routing_map = fused_pad_routing_map(routing_map, pad_multiple)
            else:
                routing_map = pad_routing_map(routing_map, pad_multiple)
            tokens_per_expert = target_tokens_per_expert
        return routing_map, tokens_per_expert

    def get_permuted_hidden_states_by_experts(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if is_experimental_enabled() and self.permute_fusion:
            self.dispatched_routing_map, self.dispatched_probs = fused_indices_to_multihot(
                self.dispatched_indices, self.dispatched_probs, self.num_local_experts
            )
        else:
            self.dispatched_routing_map, self.dispatched_probs = self._indices_to_multihot(
                self.dispatched_indices, self.dispatched_probs
            )
        if self.config.moe_router_padding_for_fp8:
            self.dispatched_routing_map, self.tokens_per_expert = self._pad_routing_map(
                self.dispatched_routing_map, self.tokens_per_expert
            )

        self.hidden_shape_before_permute = hidden_states.shape
        assert self.dispatched_probs.dtype == torch.float32, "DeepEP only supports float32 probs"
        hidden_states, permuted_probs, self.reversed_mapping_for_combine = permute(
            hidden_states,
            self.dispatched_routing_map,
            probs=self.dispatched_probs,
            num_out_tokens=self.tokens_per_expert.sum().item(),
            fused=self.permute_fusion,
        )
        if self.router_dtype == "fp64":
            permuted_probs = permuted_probs.to(torch.float64)
        return hidden_states, permuted_probs

    def get_restored_hidden_states_by_experts(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = unpermute(
            hidden_states,
            self.reversed_mapping_for_combine,
            restore_shape=self.hidden_shape_before_permute,
            routing_map=self.dispatched_routing_map,
            fused=self.permute_fusion,
        )
        return hidden_states


class MoEFlexTokenDispatcher(MoETokenDispatcher):
    """A flexible token dispatcher that abstracts the underlying tensor and expert
    parallelism. It uses a single communication group over all TP and EP ranks,
    making the dispatch logic independent of the specific parallelism strategy.
    """

    def __init__(
        self,
        num_local_experts: int,
        local_expert_indices: List[int],
        config: TransformerConfig,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):
        """
        Initialize the Flex token dispatcher.

        Args:
            num_local_experts (int): Number of local experts on the current device.
            local_expert_indices (List[int]): Indices of local experts on the current device.
            config (TransformerConfig): Configuration for the transformer model.
            pg_collection (ProcessGroupCollection, optional): Process groups for MoE operations.
        """
        super().__init__(config=config, pg_collection=pg_collection)

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
            num_local_experts=self.num_local_experts,
            router_topk=self.tp_size * self.config.moe_router_topk,
            num_experts=self.tp_size * self.config.num_moe_experts,
            config=self.config,
        )

        # Attributes that need to be captured in cudagraph. These attributes are returned
        # as cudagraph outputs when the cuda_graph_scope contains moe_preprocess.
        if isinstance(self._comm_manager, _DeepepManager):
            self.cudagraph_attrs = ['_comm_manager.token_probs', '_comm_manager.token_indices']
        else:
            # TODO: more dispatcher types to be supported in the future.
            assert (
                config.cuda_graph_impl == "none"
            ), "Cudagraphs are not supported for non-DeepEP flex token dispatchers."
            self.cudagraph_attrs = []

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

    def dispatch_preprocess(
        self, hidden_states: torch.Tensor, routing_map: torch.Tensor, probs: torch.Tensor
    ):
        """Initializes routing metadata and prepares tensors for fused dispatch.

        This method reshapes input tensors and processes routing information into a
        unified format, where the routing map is expanded to cover the TPxEP communication domain,
        enabling the token dispatch logic to be agnostic to parallelism strategies.

        Args:
            hidden_states (torch.Tensor): Input hidden states to be processed
            routing_map (torch.Tensor): Map indicating which expert each token should be routed to
            probs (torch.Tensor): Routing probabilities for each token-expert pair

        Returns:
            A tuple of reshaped hidden states and token probabilities.
        """
        self.hidden_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_shape[-1])

        # Initialize metadata
        routing_map, probs = self._initialize_metadata(routing_map, probs)

        self._comm_manager.setup_metadata(routing_map, probs)
        return hidden_states, self._comm_manager.token_probs

    def token_dispatch(
        self,
        hidden_states: torch.Tensor,
        probs: torch.Tensor = None,
        async_finish: bool = True,
        allocate_on_comm_stream: bool = True,
    ):
        """
        Execute fused permutation and AlltoAll communication.

        This method currently leverages DeepEP's fused dispatch kernel, which combines token
        permutation and AlltoAll communication into a single optimized operation.
        The fused approach reduces memory bandwidth requirements and enables better
        overlap between computation and communication operations.

        Args:
            hidden_states (torch.Tensor): Preprocessed hidden states to be dispatched
            probs (torch.Tensor): Routing probabilities (unused in current implementation)
            async_finish (bool): Whether to use asynchronous communication completion
            allocate_on_comm_stream (bool): Whether to allocate buffers on communication stream

        Returns:
            A tuple of dispatched tokens and probabilities.
        """
        return (
            self._comm_manager.dispatch(hidden_states, async_finish, allocate_on_comm_stream),
            self._comm_manager.dispatched_probs,
        )

    def dispatch_postprocess(self, hidden_states: torch.Tensor, probs: torch.Tensor):
        """Converts dispatched tokens to a per-expert format for expert processing.

        This method transforms the output of the fused dispatch into the tensor
        organization required for the expert computation.

        Args:
            hidden_states (torch.Tensor): Hidden states after fused dispatch
            probs (torch.Tensor): Routing probabilities after fused dispatch

        Returns:
            A tuple of permuted tokens, token counts per expert, and permuted probabilities.
        """
        global_input_tokens, permuted_probs = (
            self._comm_manager.get_permuted_hidden_states_by_experts(hidden_states)
        )
        tokens_per_expert = self._comm_manager.get_number_of_tokens_per_expert()
        return global_input_tokens, tokens_per_expert, permuted_probs

    def combine_preprocess(self, hidden_states: torch.Tensor):
        """Pre-processes hidden states before combining them after expert processing.

        This method restores the hidden states to their original ordering before expert processing
        by using the communication manager's restoration function.
        """
        hidden_states = self._comm_manager.get_restored_hidden_states_by_experts(hidden_states)
        return hidden_states

    def token_combine(
        self,
        hidden_states: torch.Tensor,
        async_finish: bool = True,
        allocate_on_comm_stream: bool = True,
    ):
        """Executes fused un-permutation and communication using DeepEP kernels.

        This is the inverse of the `token_dispatch` operation.

        Args:
            hidden_states (torch.Tensor): Expert outputs ready for combination
            async_finish (bool): Whether to use asynchronous communication completion
            allocate_on_comm_stream (bool): Whether to allocate buffers on communication stream

        Returns:
            Combined tokens after fused un-permutation and communication.
        """
        return self._comm_manager.combine(hidden_states, async_finish, allocate_on_comm_stream)

    def combine_postprocess(self, hidden_states: torch.Tensor):
        """
        Restores the original tensor shape and finalizes the MoE layer output.

        This method performs the final step of the MoE token processing pipeline
        by reshaping the combined tokens back to their original input dimensions.

        Args:
            hidden_states (torch.Tensor): Combined tokens.

        Returns:
            The final MoE layer output reshaped to its original dimensions.
        """
        return hidden_states.view(self.hidden_shape)
