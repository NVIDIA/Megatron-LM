from typing import Callable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_pipeline_model_parallel_rank,
    get_pipeline_model_parallel_world_size,
)

from ..dist_checkpointing.mapping import ShardedStateDict
from ..utils import make_tp_sharded_tensor_for_checkpoint
from .mappings import (
    reduce_from_tensor_model_parallel_region,
    reduce_scatter_to_sequence_parallel_region,
)
from .utils import VocabUtility
from .layers import (
    _initialize_affine_weight_cpu,
    _initialize_affine_weight_gpu,
)

def _get_vocab_parallel_rank():
    return (
        get_pipeline_model_parallel_rank() * get_tensor_model_parallel_world_size()
        + get_tensor_model_parallel_rank()
    )

def _get_vocab_parallel_world_size():
    return (
        get_pipeline_model_parallel_world_size() * get_tensor_model_parallel_world_size()
    )

class VocabParallelInput(torch.nn.Module):
    """Embedding parallelized in the vocabulary dimension.

    This is mainly adapted from torch.nn.Embedding and all the default
    values are kept.

    Args:
        num_embeddings: vocabulary size.
        embedding_dim: size of hidden state.
        reduce_scatter_embeddings: Decides whether to perform ReduceScatter after embedding lookup

    Keyword Args:
        config: A megatron.core.ModelParallelConfig object
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        init_method: Callable,
        reduce_scatter_embeddings: bool = False,
        config: ModelParallelConfig,
    ):
        super(VocabParallelInput, self).__init__()
        # Keep the input dimensions.
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.reduce_scatter_embeddings = reduce_scatter_embeddings
        self.vocab_parallel_world_size = _get_vocab_parallel_world_size()
        # Divide the weight matrix along the vocaburaly dimension.
        (
            self.vocab_start_index,
            self.vocab_end_index,
        ) = VocabUtility.vocab_range_from_global_vocab_size(
            self.num_embeddings, _get_vocab_parallel_rank(), self.vocab_parallel_world_size
        )
        self.num_embeddings_per_partition = self.vocab_end_index - self.vocab_start_index
        self.deterministic_mode = config.deterministic_mode

        # Allocate weights and initialize.
        if config.use_cpu_initialization:
            self.weight = Parameter(
                torch.empty(
                    self.num_embeddings_per_partition, self.embedding_dim, dtype=config.params_dtype
                )
            )
            if config.perform_initialization:
                _initialize_affine_weight_cpu(
                    self.weight,
                    self.num_embeddings,
                    self.embedding_dim,
                    self.num_embeddings_per_partition,
                    0,
                    init_method,
                    params_dtype=config.params_dtype,
                    rank=_get_vocab_parallel_rank(),
                    world_size=_get_vocab_parallel_world_size(),
                )
        else:
            self.weight = Parameter(
                torch.empty(
                    self.num_embeddings_per_partition,
                    self.embedding_dim,
                    device=torch.cuda.current_device(),
                    dtype=config.params_dtype,
                )
            )
            if config.perform_initialization:
                _initialize_affine_weight_gpu(self.weight, init_method, partition_dim=0, stride=1)

    def forward(self, input_):
        if self.vocab_parallel_world_size > 1:
            # Build the mask.
            input_mask = (input_ < self.vocab_start_index) | (input_ >= self.vocab_end_index)
            # Mask the input.
            masked_input = input_.clone() - self.vocab_start_index
            masked_input[input_mask] = 0
        else:
            masked_input = input_
        # Get the embeddings.
        if self.deterministic_mode:
            output_parallel = self.weight[masked_input]
        else:
            # F.embedding currently has a non-deterministic backward function
            output_parallel = F.embedding(masked_input, self.weight)
        # Mask the output embedding.
        if self.vocab_parallel_world_size > 1:
            output_parallel[input_mask, :] = 0.0

        if self.reduce_scatter_embeddings:
            # Data format change to avoid explicit tranposes : [b s h] --> [s b h].
            output_parallel = output_parallel.transpose(0, 1).contiguous()
            output = reduce_scatter_to_sequence_parallel_region(output_parallel)
        else:
            # Reduce across all the model parallel GPUs.
            output = reduce_from_tensor_model_parallel_region(output_parallel)

        output = output.clone() # TODO (benson): temporary workaround.
        return output

    def sharded_state_dict(
        self,
        prefix: str = '',
        sharded_offsets: Tuple[Tuple[int, int, int]] = (),
        metadata: Optional[dict] = None,
    ) -> ShardedStateDict:
        """Non-default implementation for embeddings due to `allow_shape_mismatch` param"""
        state_dict = self.state_dict(prefix='', keep_vars=True)

        weight_prefix = f'{prefix}weight'
        return {
            weight_prefix: make_tp_sharded_tensor_for_checkpoint(
                tensor=state_dict['weight'],
                key=weight_prefix,
                allow_shape_mismatch=True,
                prepend_offsets=sharded_offsets,
            )
        }
