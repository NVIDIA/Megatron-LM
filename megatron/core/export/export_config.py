# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import warnings
from dataclasses import dataclass
from typing import Optional


@dataclass
class ExportConfig:
    """Base configuration for Megatron Core Export

    These parameters control the export setting for trtllm
    """

    inference_tp_size: int = 1

    inference_pp_size: int = 1

    use_parallel_embedding: bool = False

    use_embedding_sharing: Optional[bool] = None

    def __post_init__(self):
        if self.use_embedding_sharing is not None:
            with warnings.catch_warnings():
                warnings.simplefilter("always")
                warnings.warn(
                    "use_embedding_sharing is deprecated in ExportConfig, "
                    "use share_embeddings_and_output_weights in TRTLLMHelper instead",
                    DeprecationWarning,
                    stacklevel=3,
                )
