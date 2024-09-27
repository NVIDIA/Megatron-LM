# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from dataclasses import dataclass


@dataclass
class ExportConfig:
    """Base configuration for Megatron Core Export

    These parameters control the export setting for trtllm
    """

    inference_tp_size: int = 1

    inference_pp_size: int = 1

    use_parallel_embedding: bool = False

    use_embedding_sharing: bool = False
