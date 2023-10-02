# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from dataclasses import dataclass
import types

from megatron.core.transformer import TransformerConfig


@dataclass
class RetroConfig(TransformerConfig):

    """Configuration object for Retro models.

    Attributes:

        retro_preprocess (SimpleNamespace): Retro preprocess arguments.
        retro_workdir (str): Retro working directory, which contains the
            preprocessed data for for pretraining. This directory is built during
            preprocessing (see tools/retro/README.md), and contains subdirectories
            for the chunk database and pretraining neighbors.
        retro_encoder_layers (int): Number of layers to use for the retrieval
            encoder.
        retro_encoder_hidden_dropout (float): Hidden dropout for retrieval
            encoder.
        retro_encoder_attention_dropout (float): Attention dropout for retrieval
            encoder.
        retro_num_neighbors (int): Number of neighbors to retrieve during
            pretraining.
        retro_num_retrieved_chunks (int): Number of chunks to retrieve from the
            retrieval database.
        retro_verify_neighbor_count (bool): Verify that len(GPT dataset) ==
            len(saved neighbors).
    """

    # Retro.
    retro_preprocess: types.SimpleNamespace = None
    retro_workdir: str = None
    retro_encoder_num_layers: int = 2
    retro_encoder_hidden_dropout: float = 0.1
    retro_encoder_attention_dropout: float = 0.1
    retro_num_neighbors: int = 2
    retro_num_retrieved_chunks: int = 2
    retro_verify_neighbor_count: bool = True
