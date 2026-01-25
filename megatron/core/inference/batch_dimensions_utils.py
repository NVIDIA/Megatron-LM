# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""
Batch dimensions utilities.

This module contains utilities for managing batch dimensions,
including the InferenceBatchDimensions dataclass and CUDAGraphBatchDimensionBuilder for generating
and matching CUDA graph batch dimensions.
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

from megatron.core import parallel_state


@dataclass(order=True, frozen=True)
class InferenceBatchDimensions:
    """Batch dimensions for dynamic inference.

    Attributes:
        token_count : number of total input tokens
        prefill_req_count : number of prefill requests
        decode_req_count : number of decode requests

    The batch dimensions are ordered by token_count, then by prefill_req_count,
    then by decode_req_count.

    """

    token_count: int = 0
    prefill_req_count: int = 0
    decode_req_count: int = 0

    def __str__(self):
        """
        Returns a string representation of the batch dimensions.
        """
        return f"[{self.token_count}]: {self.prefill_req_count} P + {self.decode_req_count} D"

    def is_applicable_for_batch_dim(
        self, real_batch_dim: "InferenceBatchDimensions", strict: bool = False
    ) -> bool:
        """
        Checks if this batch dimension is applicable for the given real batch dimension.
        Applicable batch dimensions are those that have enough tokens and
        requests budget to handle the real batch dimensions.

        Note that if strict is False, prefill slots can be used
        for prefill or decode requests. Otherwise, prefill slots
        can only be used for prefill requests.
        """
        if real_batch_dim.prefill_req_count == 0:
            return (
                self.token_count >= real_batch_dim.token_count
                and self.decode_req_count >= real_batch_dim.decode_req_count
                and self.prefill_req_count == 0  # keep decode only property
            )
        if strict:
            return (
                self.token_count >= real_batch_dim.token_count
                and self.prefill_req_count >= real_batch_dim.prefill_req_count
                and self.decode_req_count >= real_batch_dim.decode_req_count
            )
        else:
            return (
                self.token_count >= real_batch_dim.token_count
                and self.prefill_req_count >= real_batch_dim.prefill_req_count
                and self.prefill_req_count + self.decode_req_count
                >= real_batch_dim.prefill_req_count + real_batch_dim.decode_req_count
            )

    def is_valid(self, max_requests: int, max_sequence_length: int) -> bool:
        """
        Checks if the batch dimension is valid based on resource constraints.

        Args:
            max_requests: Maximum number of requests allowed

        Returns:
            True if the config is valid, False otherwise
        """
        # Check if total requests exceed maximum
        if self.prefill_req_count + self.decode_req_count > max_requests:
            return False

        # Check for negative request counts
        if self.prefill_req_count < 0 or self.decode_req_count < 0:
            return False

        # Check if token count is sufficient for requests
        if self.token_count < self.prefill_req_count + self.decode_req_count:
            return False

        # Check if the prefill requests are shorter than the max sequence length
        if self.token_count > self.prefill_req_count * max_sequence_length + self.decode_req_count:
            return False

        return True

    def __hash__(self):
        """
        Returns a hash of the batch dimension.
        In cuda graph quick matching, the batch dimension is used as a key in a dictionary.
        """
        return hash((self.token_count, self.prefill_req_count, self.decode_req_count))

    def __eq__(self, other: "InferenceBatchDimensions") -> bool:
        """
        Checks if this batch dimension is equal to another batch dimension.
        """
        if other is None:
            return False
        return (self.token_count, self.prefill_req_count, self.decode_req_count) == (
            other.token_count,
            other.prefill_req_count,
            other.decode_req_count,
        )

    @property
    def req_count(self) -> int:
        """
        Returns the total number of requests.
        """
        return self.prefill_req_count + self.decode_req_count

    @staticmethod
    def adjust_batch_dims_for_expert_parallelism(
        local_batch_dims, decode_only_cuda_graphs: bool
    ) -> "InferenceBatchDimensions":
        """Adjusted cuda graph batch dimensions for expert parallelism.
            We take the max token count across expert model parallel group.
        Return:
            (InferenceBatchDimensions) A new InferenceBatchDimensions object with
            adjusted dimensions.
        """

        ep_size = parallel_state.get_expert_model_parallel_world_size()
        if ep_size <= 1:
            return local_batch_dims

        expert_model_parallel_group = parallel_state.get_expert_model_parallel_group()
        # all reduce local work across expert model parallel group

        is_non_decode = local_batch_dims.prefill_req_count > 0

        sync_tensor = torch.tensor(
            [local_batch_dims.token_count, int(is_non_decode)],
            dtype=torch.int32,
            device=torch.cuda.current_device(),
        )
        torch.distributed.all_reduce(
            sync_tensor, op=torch.distributed.ReduceOp.MAX, group=expert_model_parallel_group
        )
        sync_tensor = sync_tensor.cpu()
        is_any_ep_rank_in_non_decode = sync_tensor[1].item() == 1
        if decode_only_cuda_graphs and is_any_ep_rank_in_non_decode:
            return None  # indicate no match, run in eager mode

        adjusted_batch_dim = InferenceBatchDimensions(
            token_count=int(sync_tensor[0].item()),
            prefill_req_count=local_batch_dims.prefill_req_count,
            decode_req_count=local_batch_dims.decode_req_count,
        )
        return adjusted_batch_dim


class CUDAGraphBatchDimensionBuilder:
    """Builder for creating and managing CUDA graph batch dimensions.

    This class provides static methods for generating lists of CUDA graph batch dimensions
    and matching the best batch dimension for a given real batch dimension.
    """

    # Constant for rounding token counts when generating CUDA graph batch dimensions
    CUDA_GRAPH_ROUNDER = 8

    @staticmethod
    def _calculate_cuda_graph_token_counts(
        tp_size: int, num_cuda_graphs: int, cuda_graph_max_tokens: int
    ) -> List[int]:
        """
        Calculate CUDA graph token counts for a given configuration.

        This method computes evenly-spaced token counts from step_size up to
        cuda_graph_max_tokens, ensuring proper rounding and TP alignment.

        Args:
            tp_size: Tensor parallel size (for alignment)
            num_cuda_graphs: Number of CUDA graphs to generate (must be >= 1)
            cuda_graph_max_tokens: Maximum token count for CUDA graphs (must be > 0)

        Returns:
            List of token counts in descending order

        Example:
            >>> _calculate_cuda_graph_token_counts
            (tp_size=2, num_cuda_graphs=4, cuda_graph_max_tokens=1000)
            [1000, 752, 504, 256]
        """
        assert num_cuda_graphs >= 1, f"num_cuda_graphs must be >= 1, got {num_cuda_graphs}"
        assert (
            cuda_graph_max_tokens > 0
        ), f"cuda_graph_max_tokens must be > 0, got {cuda_graph_max_tokens}"

        # Cuda graph step size.
        cuda_graph_step_size = cuda_graph_max_tokens / num_cuda_graphs
        cuda_graph_step_size = CUDAGraphBatchDimensionBuilder.CUDA_GRAPH_ROUNDER * int(
            math.ceil(int(cuda_graph_step_size) / CUDAGraphBatchDimensionBuilder.CUDA_GRAPH_ROUNDER)
        )
        # Make sure divisible by TP size
        cuda_graph_step_size = math.ceil(cuda_graph_step_size / tp_size) * tp_size

        # round down cuda graph max tokens to be multiple of TP size
        cuda_graph_max_tokens = (cuda_graph_max_tokens // tp_size) * tp_size

        # Cuda graph token counts.
        if num_cuda_graphs == 1:
            cuda_graph_token_counts = [cuda_graph_max_tokens]
        else:
            cuda_graph_token_counts = list(
                range(cuda_graph_step_size, cuda_graph_max_tokens, cuda_graph_step_size)
            )
            if (
                len(cuda_graph_token_counts) == 0
                or cuda_graph_token_counts[-1] != cuda_graph_max_tokens
            ):
                cuda_graph_token_counts.append(cuda_graph_max_tokens)
            cuda_graph_token_counts.reverse()

        return cuda_graph_token_counts

    @staticmethod
    def generate_cuda_graph_batch_dimensions_list(
        tp_size: int,
        num_cuda_graphs: Optional[int],
        cuda_graph_max_tokens: int,
        cuda_graph_mixed_prefill_count: Optional[int],
        max_requests: int,
        max_tokens: int,
        max_sequence_length: int,
        use_cuda_graphs_for_non_decode_steps: bool,
    ) -> Tuple[List[InferenceBatchDimensions], Optional[List[int]]]:
        """
        Generate CUDA graph batch dimensions.

        This function constructs CUDA graph batch dimensions for different token counts
        and request patterns, then filters them based on resource constraints.
        The construction process involves:

        Construction Rules:
        1. Token count generation: Creates token counts from step_size to max_tokens,
           rounded to multiples of 8
        2. Tensor parallelism alignment: Ensures step_size is divisible by tensor parallel size
        3. Batch dimension creation: For each token count, creates three types of batch dimensions:
           - Decode-only: (token_count, 0, token_count) - all tokens used for decode requests
           - Mixed prefill+decode: (token_count, prefill_req_count, token_count - prefill_req_count)
           - Prefill-only:
             (token_count, max(prefill_req_count, ceil(token_count/(max_seq_len-1))), 0)

        Filtering Rules:
        1. Request limit: prefill_req_count + decode_req_count <= max_requests
        2. Non-negative counts: Both prefill_req_count and decode_req_count must be >= 0
        3. Token sufficiency: token_count >= prefill_req_count + decode_req_count

        Sorting Rules for Attention Metadata Construction:
        1. Batch dimensions are sorted by prefill token count (token_count - decode_req_count)
           in descending order

        Args:
            tp_size: Tensor parallel size
            num_cuda_graphs: Number of CUDA graphs to generate
            cuda_graph_max_tokens: Maximum tokens for CUDA graphs
            cuda_graph_mixed_prefill_count: Number of mixed prefill requests for CUDA graphs
            max_requests: Maximum number of requests
            max_tokens: Maximum total tokens
            max_sequence_length: Maximum sequence length
            use_cuda_graphs_for_non_decode_steps: Whether to use CUDA graphs for non-decode steps

        Returns:
            Tuple containing:
            - List of InferenceBatchDimensions objects,
              sorted by prefill token count in descending order
            - Optional list of CUDA graph token counts
        """

        def add_if_valid(token_count: int, prefill_req_count: int, decode_req_count: int) -> None:
            """Helper to create and append batch dimension to list only if it's valid."""
            batch_dim = InferenceBatchDimensions(token_count, prefill_req_count, decode_req_count)
            if batch_dim.is_valid(max_requests, max_sequence_length):
                cuda_graph_batch_dimensions_list.append(batch_dim)

        # Cuda graph token-counts
        # (i.e., token counts used by cuda-graph steps, both decode and non-decode).
        cuda_graph_prefill_token_counts = None
        cuda_graph_decode_token_counts = None
        if num_cuda_graphs is not None:

            # Ensure valid num_cuda_graphs.
            if (
                cuda_graph_max_tokens is None
                or cuda_graph_max_tokens > max_tokens
                or cuda_graph_max_tokens <= 0
            ):
                cuda_graph_max_tokens = max_tokens
            num_cuda_graphs = min(max(num_cuda_graphs, 1), cuda_graph_max_tokens)

            # Calculate token counts for prefill and mixed graphs.
            # These need the full cuda_graph_max_tokens to handle variable-length sequences.
            cuda_graph_prefill_token_counts = (
                CUDAGraphBatchDimensionBuilder._calculate_cuda_graph_token_counts(
                    tp_size=tp_size,
                    num_cuda_graphs=num_cuda_graphs,
                    cuda_graph_max_tokens=cuda_graph_max_tokens,
                )
            )

            # Calculate separate token counts for decode-only graphs.
            # Decode graphs can be more conservative since each request uses exactly 1 token.
            cuda_graph_max_tokens_decode = min(cuda_graph_max_tokens, max_requests)
            cuda_graph_decode_token_counts = (
                CUDAGraphBatchDimensionBuilder._calculate_cuda_graph_token_counts(
                    tp_size=tp_size,
                    num_cuda_graphs=num_cuda_graphs,
                    cuda_graph_max_tokens=cuda_graph_max_tokens_decode,
                )
            )

        cuda_graph_batch_dimensions_list = []
        if num_cuda_graphs is None:
            cuda_graph_batch_dimensions_list = []
        elif (
            not cuda_graph_mixed_prefill_count
            or cuda_graph_mixed_prefill_count <= 0
            or not use_cuda_graphs_for_non_decode_steps
        ):  # decode only
            # Use decode-specific token counts for decode-only graphs
            for size in cuda_graph_decode_token_counts:
                add_if_valid(
                    token_count=min(size, max_requests),
                    prefill_req_count=0,
                    decode_req_count=min(size, max_requests),
                )
        else:
            # Mixed prefill and decode mode
            # Create prefill and mixed dimensions with full token counts
            for size in cuda_graph_prefill_token_counts:
                add_if_valid(
                    token_count=size,
                    prefill_req_count=min(cuda_graph_mixed_prefill_count, max_requests),
                    decode_req_count=min(size, max_requests)
                    - min(cuda_graph_mixed_prefill_count, max_requests),
                )
                # We need to ensure the prefill requests are shorter than the max sequence length,
                # considering the one decode token is used for prefill request construction
                prefill_only_minimal_num = max(
                    cuda_graph_mixed_prefill_count,
                    math.ceil(size / max(1, max_sequence_length - 1)),
                )
                if prefill_only_minimal_num < max_requests:
                    add_if_valid(
                        token_count=size,
                        prefill_req_count=max(prefill_only_minimal_num, min(max_requests, size)),
                        decode_req_count=0,
                    )

            # Create decode-only dimensions with optimized token counts
            for size in cuda_graph_decode_token_counts:
                add_if_valid(
                    token_count=min(size, max_requests),
                    prefill_req_count=0,
                    decode_req_count=min(size, max_requests),
                )

        # Remove duplicates and sort by prefill token count
        cuda_graph_batch_dimensions_list = list(set(cuda_graph_batch_dimensions_list))
        cuda_graph_batch_dimensions_list.sort(
            key=lambda x: ((x.token_count - x.decode_req_count), x.decode_req_count), reverse=True
        )

        # Collect actual token counts from batch dimensions, then unique and sort
        if num_cuda_graphs is None or len(cuda_graph_batch_dimensions_list) == 0:
            # No CUDA graphs or no valid batch dimensions
            cuda_graph_token_counts = None
        else:
            # Extract unique token counts from the batch dimensions we actually created
            token_counts_set = {
                batch_dim.token_count for batch_dim in cuda_graph_batch_dimensions_list
            }
            cuda_graph_token_counts = sorted(list(token_counts_set), reverse=True)

        return cuda_graph_batch_dimensions_list, cuda_graph_token_counts

    @staticmethod
    def match_graph_config(
        real_batch_dim: InferenceBatchDimensions,
        cuda_graph_batch_dimensions_list: List[InferenceBatchDimensions],
        strict: bool = False,
        decode_only_cuda_graphs: bool = False,
    ) -> Optional[InferenceBatchDimensions]:
        """
        Matches the best CUDA graph batch dimension for the given real batch dimension.

        Args:
            real_batch_dim: The real batch dimension to match
            cuda_graph_batch_dimensions_list: List of available CUDA graph batch dimensions
            strict: If False, prefill slots can be used for prefill or decode requests.
                   If True, prefill slots can only be used for prefill requests.
            decode_only_cuda_graphs: Used by expert parallel matching. If this is true,
            and one of the EP ranks is running a non-decode step, we elect to run in
            eager mode instead of matching a decode-only cuda graph.
        Returns:
            The best matching CUDA graph batch dimension, or None if no applicable match is found
        """

        if not cuda_graph_batch_dimensions_list:
            # no need to match if no cuda graph batch dimensions are provided
            return None

        adjusted_batch_dim = InferenceBatchDimensions.adjust_batch_dims_for_expert_parallelism(
            real_batch_dim, decode_only_cuda_graphs
        )

        if adjusted_batch_dim is None:
            # we hit this scenario if decode_only_cuda_graphs is true,
            # and one of the EP ranks is running a non-decode step
            # in that case, all ranks have to run in eager mode
            return None

        # first filter out batch dimensions with smaller token count, prefill req count,
        # or decode req count, as they are not applicable
        graph_batch_dims_applicable = [
            graph_batch_dim
            for graph_batch_dim in cuda_graph_batch_dimensions_list
            if graph_batch_dim.is_applicable_for_batch_dim(adjusted_batch_dim, strict=strict)
        ]
        if len(graph_batch_dims_applicable) == 0:
            return None
        # then find the best batch dimension
        best_batch_dim = min(graph_batch_dims_applicable)
        return best_batch_dim
