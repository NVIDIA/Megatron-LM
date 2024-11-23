# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
from .cross_entropy import vocab_parallel_cross_entropy
from .data import broadcast_data
from .layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
    copy_tensor_model_parallel_attributes,
    linear_with_grad_accumulation_and_async_allreduce,
    param_is_not_tensor_parallel_duplicate,
    set_defaults_if_not_set_tensor_model_parallel_attributes,
    set_tensor_model_parallel_attributes,
)
from .mappings import (
    all_gather_last_dim_from_tensor_parallel_region,
    all_to_all,
    all_to_all_hp2sp,
    all_to_all_sp2hp,
    copy_to_tensor_model_parallel_region,
    gather_from_sequence_parallel_region,
    gather_from_tensor_model_parallel_region,
    reduce_from_tensor_model_parallel_region,
    reduce_scatter_last_dim_to_tensor_parallel_region,
    reduce_scatter_to_sequence_parallel_region,
    scatter_to_sequence_parallel_region,
    scatter_to_tensor_model_parallel_region,
)
from .random import (
    checkpoint,
    get_cuda_rng_tracker,
    get_data_parallel_rng_tracker_name,
    get_expert_parallel_rng_tracker_name,
    model_parallel_cuda_manual_seed,
)
from .utils import (
    gather_split_1d_tensor,
    split_tensor_along_last_dim,
    split_tensor_into_1d_equal_chunks,
)

__all__ = [
    # cross_entropy.py
    "vocab_parallel_cross_entropy",
    # data.py
    "broadcast_data",
    # layers.py
    "ColumnParallelLinear",
    "RowParallelLinear",
    "VocabParallelEmbedding",
    "set_tensor_model_parallel_attributes",
    "set_defaults_if_not_set_tensor_model_parallel_attributes",
    "copy_tensor_model_parallel_attributes",
    "param_is_not_tensor_parallel_duplicate",
    "linear_with_grad_accumulation_and_async_allreduce",
    # mappings.py
    "copy_to_tensor_model_parallel_region",
    "gather_from_tensor_model_parallel_region",
    "gather_from_sequence_parallel_region",
    "reduce_from_tensor_model_parallel_region",
    "reduce_scatter_to_sequence_parallel_region",
    "scatter_to_tensor_model_parallel_region",
    "scatter_to_sequence_parallel_region",
    # random.py
    "checkpoint",
    "get_cuda_rng_tracker",
    "model_parallel_cuda_manual_seed",
    "get_expert_parallel_rng_tracker_name",
    # utils.py
    "split_tensor_along_last_dim",
    "split_tensor_into_1d_equal_chunks",
    "gather_split_1d_tensor",
]
