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

from megatron.core.utils import get_pg_size, round_up_to_nearest_multiple


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

    def is_valid(
        self, max_requests: int, max_sequence_length: int, num_speculative_tokens: int
    ) -> bool:
        """
        Checks if the batch dimension is valid based on resource constraints.

        Args:
            max_requests: Maximum number of requests allowed

        Returns:
            True if the config is valid, False otherwise
        """
        # A dimension with no tokens serves no requests.
        if self.token_count <= 0:
            return False

        # Check if total requests exceed maximum
        if self.prefill_req_count + self.decode_req_count > max_requests:
            return False

        # Check for negative request counts
        if self.prefill_req_count < 0 or self.decode_req_count < 0:
            return False

        # Check if token count is sufficient for requests
        if self.token_count < self.prefill_req_count + self.decode_req_count * (
            num_speculative_tokens + 1
        ):
            return False

        # Check if the prefill requests are shorter than the max sequence length
        if (
            self.token_count
            > self.prefill_req_count * max_sequence_length
            + self.decode_req_count * (num_speculative_tokens + 1)
        ):
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
        local_batch_dims,
        ep_group: Optional[torch.distributed.ProcessGroup] = None,
        ep_zmq_communicator=None,
    ) -> Optional["InferenceBatchDimensions"]:
        """Adjust CUDA graph batch dimensions for expert parallelism.

        All-reduce-max the token count and non-decode flag across the EP group.
        If any rank has a prefill (non-decode) step, all ranks fall back to eager
        mode (return None) — the non-CG path handles variable token counts via
        use_allgather_v. Otherwise return adjusted dims with the max token count.

        Args:
            local_batch_dims: The local batch dimensions to adjust.
            ep_group: Optional expert parallel process group. If None, uses global parallel state.
                      When using different EP sizes for inference vs training, pass the
                      inference EP group explicitly.
            ep_zmq_communicator: Optional AsyncZMQCommunicator over the EP group. When
                      provided, the cross-rank MAX reduction runs on the CPU via ZMQ
                      (no GPU kernel, no H2D/D2H), avoiding a per-step NCCL AllReduce
                      on the compute stream. When absent, falls back to
                      torch.distributed.all_reduce on a GPU tensor.

        Returns:
            InferenceBatchDimensions with max token count, or None for eager mode.
        """
        ep_size = get_pg_size(ep_group)
        if ep_size <= 1:
            return local_batch_dims

        is_non_decode = local_batch_dims.prefill_req_count > 0

        if ep_zmq_communicator is not None:
            # CPU-only sync via ZMQ: avoids a NCCL AllReduce kernel on the
            # compute stream plus the H2D/D2H pair that sandwiches it.
            (max_token_count, max_is_non_decode) = ep_zmq_communicator.sync_all_reduce_max(
                local_batch_dims.token_count, int(is_non_decode)
            )
        else:
            sync_tensor = torch.tensor(
                [local_batch_dims.token_count, int(is_non_decode)],
                dtype=torch.int32,
                device=torch.cuda.current_device(),
            )
            torch.distributed.all_reduce(
                sync_tensor, op=torch.distributed.ReduceOp.MAX, group=ep_group
            )
            sync_tensor = sync_tensor.cpu()
            max_token_count = int(sync_tensor[0].item())
            max_is_non_decode = int(sync_tensor[1].item())

        is_any_ep_rank_in_non_decode = max_is_non_decode == 1

        if is_any_ep_rank_in_non_decode:
            return None  # any rank has prefill → eager mode

        adjusted_batch_dim = InferenceBatchDimensions(
            token_count=max_token_count,
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
    CUDA_GRAPH_ROUNDER = 2

    @staticmethod
    def _calculate_cuda_graph_token_counts(
        tp_size: int, num_cuda_graphs: int, cuda_graph_max_tokens: int
    ) -> List[int]:
        """
        Calculate CUDA graph token counts for a given configuration.

        This method computes exponentially-decreasing token counts (powers of 2)
        from cuda_graph_max_tokens down to CUDA_GRAPH_ROUNDER, ensuring proper
        rounding and TP alignment.

        Args:
            tp_size: Tensor parallel size (for alignment)
            num_cuda_graphs: Number of CUDA graphs to generate (must be >= 1)
            cuda_graph_max_tokens: Maximum token count for CUDA graphs (must be > 0)

        Returns:
            List of token counts in descending order

        Example:
            >>> _calculate_cuda_graph_token_counts
            (tp_size=1, num_cuda_graphs=8, cuda_graph_max_tokens=128)
            [128, 64, 32, 16, 8, 4, 2, 1]
        """
        if num_cuda_graphs == -1:
            # Each step in the exponential-decay loop below halves the cudagraph size, so we need
            # ~log2(max_tokens) steps with an extra +2 to leave headroom for dedup/trim.
            auto_n = max(4, int(math.log2(max(2, cuda_graph_max_tokens))) + 2)
            return CUDAGraphBatchDimensionBuilder._calculate_cuda_graph_token_counts(
                tp_size=tp_size,
                num_cuda_graphs=auto_n,
                cuda_graph_max_tokens=cuda_graph_max_tokens,
            )

        assert num_cuda_graphs >= 1, f"num_cuda_graphs must be >= 1, got {num_cuda_graphs}"
        assert (
            cuda_graph_max_tokens > 0
        ), f"cuda_graph_max_tokens must be > 0, got {cuda_graph_max_tokens}"

        rounder = CUDAGraphBatchDimensionBuilder.CUDA_GRAPH_ROUNDER

        # Cuda graph step size.
        cuda_graph_step_size = cuda_graph_max_tokens / num_cuda_graphs
        cuda_graph_step_size = CUDAGraphBatchDimensionBuilder.CUDA_GRAPH_ROUNDER * int(
            math.ceil(int(cuda_graph_step_size) / CUDAGraphBatchDimensionBuilder.CUDA_GRAPH_ROUNDER)
        )
        # Make sure divisible by TP size
        cuda_graph_step_size = round_up_to_nearest_multiple(cuda_graph_step_size, tp_size)
        # Ensure non-zero step size (can happen when max_tokens < num_cuda_graphs).
        cuda_graph_step_size = max(cuda_graph_step_size, tp_size)

        # Round down cuda graph max tokens to be multiple of TP size
        cuda_graph_max_tokens = (cuda_graph_max_tokens // tp_size) * tp_size

        if num_cuda_graphs == 1:
            return [cuda_graph_max_tokens]

        # Exponentially decreasing, stops after num_cuda_graphs entries
        # or when below the minimum size.
        # TODO(helenn/lmcafee): Extend upper range of distribution to be linearly-spaced.
        cuda_graph_token_counts = []
        val = cuda_graph_max_tokens
        for _ in range(num_cuda_graphs):
            # Round down to multiple of rounder, then up to multiple of TP size
            rounded = max(rounder, (val // rounder) * rounder)
            rounded = math.ceil(rounded / tp_size) * tp_size
            if rounded not in cuda_graph_token_counts:
                cuda_graph_token_counts.append(rounded)
            val //= 2
            if val < 1:
                break

        # Ensure cuda_graph_max_tokens is always included
        if cuda_graph_token_counts[0] != cuda_graph_max_tokens:
            cuda_graph_token_counts.insert(0, cuda_graph_max_tokens)

        # Include a (possibly extra) size-1 graph
        if cuda_graph_token_counts[-1] != tp_size:
            cuda_graph_token_counts.append(tp_size)

        # Trim from the middle if we exceed num_cuda_graphs requested by the user
        #  Since num_cuda_graphs >= 1, this only runs when we have at least 2 elements.
        while len(cuda_graph_token_counts) > num_cuda_graphs:
            cuda_graph_token_counts.pop(-2)

        assert len(cuda_graph_token_counts) <= num_cuda_graphs
        assert cuda_graph_max_tokens in cuda_graph_token_counts

        return cuda_graph_token_counts

    @staticmethod
    def generate_cuda_graph_batch_dimensions_list(
        tp_size: int,
        num_cuda_graphs: Optional[int],
        cuda_graph_max_tokens: int,
        cuda_graph_mixed_prefill_request_count: Optional[int],
        max_requests: int,
        max_tokens: int,
        max_sequence_length: int,
        use_cuda_graphs_for_non_decode_steps: bool,
        num_speculative_tokens: int = 0,
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
            cuda_graph_mixed_prefill_request_count: Number of mixed prefill requests for CUDA graphs
            max_requests: Maximum number of requests
            max_tokens: Maximum total tokens
            max_sequence_length: Maximum sequence length
            use_cuda_graphs_for_non_decode_steps: Whether to use CUDA graphs for non-decode steps
            num_speculative_tokens: Number of speculative tokens

        Returns:
            Tuple containing:
            - List of InferenceBatchDimensions objects,
              sorted by prefill token count in descending order
            - Optional list of CUDA graph token counts
        """

        def add_if_valid(token_count: int, prefill_req_count: int, decode_req_count: int) -> None:
            """Helper to create and append batch dimension to list only if it's valid."""
            batch_dim = InferenceBatchDimensions(token_count, prefill_req_count, decode_req_count)
            if batch_dim.is_valid(max_requests, max_sequence_length, num_speculative_tokens):
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

            assert cuda_graph_max_tokens >= max_requests * (num_speculative_tokens + 1), (
                f"cuda_graph_max_tokens ({cuda_graph_max_tokens}) must be >= max_requests * "
                f"(num_speculative_tokens + 1) ({max_requests * (num_speculative_tokens + 1)})."
            )

            if num_cuda_graphs != -1:
                # if -1, no need to adjust. This will be taken care of in
                # the _calculate_cuda_graph_token_counts function where we will generate
                # the token counts based on the max_tokens value and the step size.
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
            cuda_graph_max_tokens_decode = min(
                cuda_graph_max_tokens, max_requests * (num_speculative_tokens + 1)
            )
            cuda_graph_decode_token_counts = (
                CUDAGraphBatchDimensionBuilder._calculate_cuda_graph_token_counts(
                    tp_size=tp_size,
                    num_cuda_graphs=num_cuda_graphs,
                    cuda_graph_max_tokens=cuda_graph_max_tokens_decode,
                )
            )

            # Include the smallest decode-only graphs when auto-sizing (num_cuda_graphs == -1).
            # Without this, TP alignment (size 1 -> tp_size) and the num_speculative_tokens floor
            # division may drop the size 1 and size 2 graph sizes.
            if num_cuda_graphs == -1:
                spec_unit = num_speculative_tokens + 1
                min_decode_tokens = math.lcm(spec_unit, tp_size)
                for req_count_multiple in (1, 2):
                    floor_tokens = min_decode_tokens * req_count_multiple
                    if (
                        floor_tokens <= cuda_graph_max_tokens_decode
                        and floor_tokens not in cuda_graph_decode_token_counts
                    ):
                        cuda_graph_decode_token_counts.append(floor_tokens)
                cuda_graph_decode_token_counts = sorted(
                    set(cuda_graph_decode_token_counts), reverse=True
                )

        cuda_graph_batch_dimensions_list = []
        if num_cuda_graphs is None:
            cuda_graph_batch_dimensions_list = []
        elif (
            not cuda_graph_mixed_prefill_request_count
            or cuda_graph_mixed_prefill_request_count <= 0
            or not use_cuda_graphs_for_non_decode_steps
        ):  # decode only
            # Use decode-specific token counts for decode-only graphs
            for size in cuda_graph_decode_token_counts:
                decode_req_count = min(size // (num_speculative_tokens + 1), max_requests)
                token_count = decode_req_count * (num_speculative_tokens + 1)
                token_count = token_count // tp_size * tp_size
                add_if_valid(
                    token_count=token_count, prefill_req_count=0, decode_req_count=decode_req_count
                )
        else:
            # Mixed prefill and decode mode
            # Create prefill and mixed dimensions with full token counts
            for size in cuda_graph_prefill_token_counts:
                assert size % tp_size == 0
                prefill_req_count = min(cuda_graph_mixed_prefill_request_count, max_requests)
                decode_req_count = max(
                    0,
                    min(
                        (size - prefill_req_count) // (num_speculative_tokens + 1),
                        max_requests - prefill_req_count,
                    ),
                )
                add_if_valid(
                    token_count=size,
                    prefill_req_count=prefill_req_count,
                    decode_req_count=decode_req_count,
                )
                # We need to ensure the prefill requests are shorter than the max sequence length,
                # considering the one decode token is used for prefill request construction
                prefill_only_minimal_num = max(
                    cuda_graph_mixed_prefill_request_count,
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
                decode_req_count = min(size // (num_speculative_tokens + 1), max_requests)
                token_count = decode_req_count * (num_speculative_tokens + 1)
                token_count = token_count // tp_size * tp_size
                add_if_valid(
                    token_count=token_count, prefill_req_count=0, decode_req_count=decode_req_count
                )

        # Remove duplicates and sort by prefill token count
        cuda_graph_batch_dimensions_list = list(set(cuda_graph_batch_dimensions_list))
        cuda_graph_batch_dimensions_list.sort(
            key=lambda x: (
                (x.token_count - x.decode_req_count * (num_speculative_tokens + 1)),
                x.decode_req_count,
            ),
            reverse=True,
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
        ep_group: Optional[torch.distributed.ProcessGroup] = None,
        ep_zmq_communicator=None,
        match_ep_token_counts: bool = True,
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
            ep_group: Optional expert parallel process group. If None, uses global parallel state.
                      When using different EP sizes for inference vs training, pass the
                      inference EP group explicitly.
            ep_zmq_communicator: Optional AsyncZMQCommunicator over the EP group. When
                      provided, batch-dimension MAX reduction uses a CPU-only ZMQ sync
                      instead of a GPU NCCL AllReduce. Forwarded to
                      adjust_batch_dims_for_expert_parallelism.
            match_ep_token_counts: If True (default), token counts are synced across EP ranks via
                all-reduce-max so all ranks select the same CUDA graph. Set to False when the
                dispatcher handles per-rank token variation internally (e.g. AGV/RSV in the NVLS
                path) and external EP sync is not needed.
        Returns:
            The best matching CUDA graph batch dimension, or None if no applicable match is found
        """

        if not cuda_graph_batch_dimensions_list:
            # no need to match if no cuda graph batch dimensions are provided
            return None

        if match_ep_token_counts:
            # NCCL dispatcher: all EP ranks must select the same CUDA graph. Sync batch dims
            # across the EP group so graph selection is consistent.
            adjusted_batch_dim = InferenceBatchDimensions.adjust_batch_dims_for_expert_parallelism(
                real_batch_dim, ep_group=ep_group, ep_zmq_communicator=ep_zmq_communicator
            )

            if adjusted_batch_dim is None:
                # we hit this scenario if decode_only_cuda_graphs is true,
                # and one of the EP ranks is running a non-decode step
                # in that case, all ranks have to run in eager mode
                return None
        else:
            adjusted_batch_dim = real_batch_dim

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
