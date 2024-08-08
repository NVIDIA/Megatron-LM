# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Configuration dataclass for a RetroModel."""

import os
import types
from dataclasses import dataclass
from importlib.metadata import version

from pkg_resources import packaging

from megatron.core.transformer import TransformerConfig


@dataclass
class RetroConfig(TransformerConfig):
    """Configuration object for Retro models."""

    # Retro.
    retro_project_dir: str = None
    """Retro project directory, which contains the preprocessed data for for pretraining. This
       directory is built during preprocessing (see tools/retro/README.md), and contains
       subdirectories for the chunk database and pretraining neighbors.
    """

    retro_block_size: int = None
    """Number of records to load per data file, as saved during preprocessing. Block processing is
       used for efficient data preprocessing.
    """

    retro_chunk_length: int = None
    """Chunk length used for performing chunked- cross-attention (CCA)."""

    retro_encoder_num_layers: int = 2
    """Number of layers to use for the retrieval encoder."""

    retro_encoder_hidden_dropout: float = 0.1
    """Hidden dropout for retrieval encoder."""

    retro_encoder_attention_dropout: float = 0.1
    """Attention dropout for retrieval encoder."""

    retro_neighbor_dirs: dict = None
    """Directory names of saved neighbor id files for train, valid, and test datasets."""

    retro_num_neighbors: int = 2
    """Number of neighbors to retrieve during pretraining."""

    retro_num_retrieved_chunks: int = 2
    """Number of chunks to retrieve from the retrieval database."""

    retro_retrieved_length: int = None
    """Cached value of retro_num_retrieved_chunks * retro_chunk_length (i.e., the total number of
       retrieved tokens; neighbor + continuation).
    """

    retro_split_preprocessing: str = None
    """Data split used during data preprocessing."""

    retro_verify_neighbor_count: bool = True
    """Verify that len(GPT dataset) == len(saved neighbors)."""

    def __post_init__(self) -> None:
        """Validate Retro config."""

        super().__post_init__()

        # Validate Transformer Engine version.
        te_version = packaging.version.Version(version("transformer-engine"))
        if te_version >= packaging.version.Version("1.3"):
            try:
                assert os.getenv("NVTE_FLASH_ATTN") == "0"
                assert os.getenv("NVTE_FUSED_ATTN") == "0"
            except Exception as e:
                raise Exception(
                    "When using Transformer Engine >= 1.3, environment vars NVTE_FLASH_ATTN and NVTE_FUSED_ATTN most both be defined and set to '0'. Currently, NVTE_FLASH_ATTN == %s, NVTE_FUSED_ATTN == %s."
                    % (
                        os.getenv("NVTE_FLASH_ATTN", "[unset]"),
                        os.getenv("NVTE_FUSED_ATTN", "[unset]"),
                    )
                )

        # Preprocessing split should be defined.
        assert self.retro_split_preprocessing is not None

        # Pre-compute retrieved length.
        self.retro_retrieved_length = self.retro_num_retrieved_chunks * self.retro_chunk_length
