# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import torch

from megatron.core import tensor_parallel
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import (
    make_sharded_tensor_for_checkpoint,
    make_tp_sharded_tensor_for_checkpoint,
)


class T5Embedding(MegatronModule):
    """Language model embeddings.

    Arguments:
        config (TransformerConfig): config object with all necessary configs for TransformerBlock
        vocab_size (int): vocabulary size
        max_sequence_length (int): maximum size of sequence. This
                             is used for positional embedding
        add_position_embedding (bool): Add a position embedding.
        embedding_dropout_prob float): dropout probability for embeddings
    """

    def __init__(
        self,
        config: TransformerConfig,
        vocab_size: int,
        max_sequence_length: int,
        add_position_embedding: bool,
    ):
        super().__init__(config=config)

        self.config: TransformerConfig = config
        self.vocab_size: int = vocab_size
        self.max_sequence_length: int = max_sequence_length
        self.add_position_embedding: bool = add_position_embedding

        # Word embeddings (parallel).
        self.word_embeddings = tensor_parallel.VocabParallelEmbedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.config.hidden_size,
            init_method=self.config.init_method,
            config=self.config,
        )

        # Position embedding (serial).
        if self.add_position_embedding:
            self.position_embeddings = torch.nn.Embedding(
                self.max_sequence_length, self.config.hidden_size
            )

            # Initialize the position embeddings.
            if self.config.perform_initialization:
                self.config.init_method(self.position_embeddings.weight)

        # Embeddings dropout
        self.embedding_dropout = torch.nn.Dropout(self.config.hidden_dropout)

    def zero_parameters(self):
        """Zero out all parameters in embedding."""
        self.word_embeddings.weight.data.fill_(0)
        self.word_embeddings.weight.shared = True
        self.position_embeddings.weight.data.fill_(0)
        self.position_embeddings.weight.shared = True

    def forward(self, input_ids, position_ids):
        # Embeddings.
        word_embeddings = self.word_embeddings(input_ids)
        if self.add_position_embedding:
            position_embeddings = self.position_embeddings(position_ids)
            embeddings = word_embeddings + position_embeddings
        else:
            embeddings = word_embeddings

        # Data format change to avoid explicit tranposes : [b s h] --> [s b h].
        embeddings = embeddings.transpose(0, 1).contiguous()

        # If the input flag for fp32 residual connection is set, convert for float.
        if self.config.fp32_residual_connection:
            embeddings = embeddings.float()

        # Dropout.
        if self.config.sequence_parallel:
            embeddings = tensor_parallel.scatter_to_sequence_parallel_region(embeddings)
            with tensor_parallel.get_cuda_rng_tracker().fork():
                embeddings = self.embedding_dropout(embeddings)
        else:
            embeddings = self.embedding_dropout(embeddings)

        return embeddings

    def sharded_state_dict(self, prefix=''):

        sharded_state_dict = {}

        word_embeddings_prefix = f'{prefix}word_embeddings.'
        word_embeddings_state_dict = self.word_embeddings.state_dict(
            prefix=word_embeddings_prefix, keep_vars=True
        )

        sharded_word_embeddings_key = f'{word_embeddings_prefix}weight'
        sharded_word_embeddings_tensor = make_tp_sharded_tensor_for_checkpoint(
            tensor=word_embeddings_state_dict[sharded_word_embeddings_key],
            key=sharded_word_embeddings_key,
            allow_shape_mismatch=True,
        )
        sharded_state_dict[sharded_word_embeddings_key] = sharded_word_embeddings_tensor

        if self.add_position_embedding:
            position_embeddings_prefix = f'{prefix}position_embeddings.'
            position_embeddings_state_dict = self.position_embeddings.state_dict(
                prefix=position_embeddings_prefix, keep_vars=True
            )
            sharded_position_embeddings_key = f'{position_embeddings_prefix}weight'
            sharded_position_embeddings_tensor = make_sharded_tensor_for_checkpoint(
                tensor=position_embeddings_state_dict[sharded_position_embeddings_key],
                key=sharded_position_embeddings_key,
            )
            sharded_state_dict[sharded_position_embeddings_key] = sharded_position_embeddings_tensor

        return sharded_state_dict
