# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Shared parameter metadata used by sharding and parameter replacement paths."""

# Metadata that must survive replacing a parameter or exposing member views of a
# grouped parameter. Refit and tensor-parallel planning inspect these attributes.
PARAMETER_SHARDING_ATTRIBUTES = (
    "allreduce",
    "expert_parallel",
    "expert_tp",
    "group",
    "is_embedding_or_output_parameter",
    "is_gtp_weight_remat",
    "is_qkv",
    "pad_length",
    "partition_dim",
    "partition_sizes",
    "partition_stride",
    "qkv_split_shapes",
    "sequence_parallel",
    "tensor_model_parallel",
)
