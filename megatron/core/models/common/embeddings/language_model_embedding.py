# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from typing import Literal

import torch
from torch import Tensor

from megatron.core import tensor_parallel
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core import mpu


class LanguageModelEmbedding(MegatronModule):
    """Language model embeddings.

    Args:
        config (TransformerConfig): config object with all necessary configs for TransformerBlock
        vocab_size (int): vocabulary size
        max_sequence_length (int): maximum size of sequence. This
                             is used for positional embedding
        add_position_embedding (bool): Add a position embedding.
        embedding_dropout_prob (float): dropout probability for embeddings
        num_tokentypes (int): Set to 0 without binary head, and 2 with a binary head . Defaults to 0.
    """

    def __init__(
        self,
        config: TransformerConfig,
        vocab_size: int,
        max_sequence_length: int,
        position_embedding_type: Literal['learned_absolute', 'rope'] = 'learned_absolute',
        num_tokentypes: int = 0,
        parallel_word_embedding: bool = True,
    ):
        super().__init__(config=config)

        self.config: TransformerConfig = config
        self.vocab_size: int = vocab_size
        self.max_sequence_length: int = max_sequence_length
        self.add_position_embedding: bool = position_embedding_type == 'learned_absolute'
        self.num_tokentypes = num_tokentypes

        # Word embeddings (parallel or not).
        if parallel_word_embedding:
            self.word_embeddings = tensor_parallel.VocabParallelEmbedding(
                num_embeddings=self.vocab_size,
                embedding_dim=self.config.hidden_size,
                init_method=self.config.init_method,
                config=self.config,
            )
        else:
            self.word_embeddings = torch.nn.Embedding(
                num_embeddings=vocab_size,
                embedding_dim=self.config.hidden_size,
            )

        # Position embedding (serial).
        if self.add_position_embedding:
            self.position_embeddings = torch.nn.Embedding(
                self.max_sequence_length, self.config.hidden_size
            )

            # Initialize the position embeddings.
            if self.config.perform_initialization:
                self.config.init_method(self.position_embeddings.weight)

        if self.num_tokentypes > 0:
            self.tokentype_embeddings = torch.nn.Embedding(
                self.num_tokentypes, self.config.hidden_size
            )
            # Initialize the token-type embeddings.
            if self.config.perform_initialization:
                self.config.init_method(self.tokentype_embeddings.weight)
        else:
            self.tokentype_embeddings = None

        # Embeddings dropout
        self.embedding_dropout = torch.nn.Dropout(self.config.hidden_dropout)

    def zero_parameters(self):
        """Zero out all parameters in embedding."""
        self.word_embeddings.weight.data.fill_(0)
        self.word_embeddings.weight.shared = True
        self.position_embeddings.weight.data.fill_(0)
        self.position_embeddings.weight.shared = True
        if self.num_tokentypes > 0:
            self.tokentype_embeddings.weight.data.fill_(0)
            self.tokentype_embeddings.weight.shared = True

    def forward(self, input_ids: Tensor, position_ids: Tensor, tokentype_ids: int = None, external_feature_dict: dict = {}) -> Tensor:
        """Forward pass of the embedding module.

        Args:
            input_ids (Tensor): The input tokens
            position_ids (Tensor): The position id's used to calculate position embeddings
            tokentype_ids (int): The token type ids. Used when args.bert_binary_head is set to True. Defaults to None

        Returns:
            Tensor: The output embeddings
        """
        word_embeddings = self.word_embeddings(input_ids)
        if external_feature_dict:
            assert 'features' in external_feature_dict \
                and (len(external_feature_dict) ==1 \
                     or len(external_feature_dict) == 2 and 'pre_len' in external_feature_dict \
                     or len(external_feature_dict) == 2 and 'indices' in external_feature_dict \
                     or len(external_feature_dict) == 3 and 'src_indices' in external_feature_dict and 'tgt_indices' in external_feature_dict), "The format of external_feature_dict is not right!"
            word_embeddings = word_embeddings.clone()
            features = external_feature_dict['features']
            if 'indices' in external_feature_dict:
                indices_b, indices_s = external_feature_dict['indices'].unbind(dim=0)
                word_embeddings[indices_b.view(-1), indices_s.view(-1)] = features.view(-1, features.shape[-1])
            elif 'pre_len' in external_feature_dict:
                pre_len = external_feature_dict['pre_len']
                word_embeddings[:,pre_len:pre_len+features.shape[1]] = features
            elif "src_indices" in external_feature_dict and "tgt_indices" in external_feature_dict:
                src_indices_b, src_indices_s = external_feature_dict['src_indices']
                tgt_indices_b, tgt_indices_s = external_feature_dict['tgt_indices']
                word_embeddings[tgt_indices_b, tgt_indices_s] = features[src_indices_b, src_indices_s]
            else:
                word_embeddings += features.mean() * 0 # To avoid backward hanging.
        if self.add_position_embedding:
            position_embeddings = self.position_embeddings(position_ids)
            embeddings = word_embeddings + position_embeddings
        else:
            embeddings = word_embeddings

        # Data format change to avoid explicit tranposes : [b s h] --> [s b h].
        embeddings = embeddings.transpose(0, 1).contiguous()

        if tokentype_ids is not None:
            assert self.tokentype_embeddings is not None
            # [b s h] -> [s b h] (So that it can be added with embeddings)
            tokentype_embedding = self.tokentype_embeddings(tokentype_ids).permute(1, 0, 2)
            embeddings = embeddings + tokentype_embedding
        else:
            assert self.tokentype_embeddings is None

        # If the input flag for fp32 residual connection is set, convert for float.
        if self.config.fp32_residual_connection:
            embeddings = embeddings.float()

        # Dropout.
        if self.config.sequence_parallel:
            if isinstance(self.word_embeddings, torch.nn.Embedding):
                embeddings += self.word_embeddings.weight[0].mean() * 0
            embeddings = tensor_parallel.scatter_to_sequence_parallel_region(embeddings)
            # `scatter_to_sequence_parallel_region` returns a view, which prevents
            # the original tensor from being garbage collected. Clone to facilitate GC.
            # Has a small runtime cost (~0.5%).
            if self.config.clone_scatter_output_in_embedding:
                embeddings = embeddings.clone()
            with tensor_parallel.get_cuda_rng_tracker().fork():
                embeddings = self.embedding_dropout(embeddings)
        else:
            if mpu.get_context_parallel_world_size() != 1:
                if isinstance(self.word_embeddings, torch.nn.Embedding):
                    embeddings += self.word_embeddings.weight[0].mean() * 0
            embeddings = self.embedding_dropout(embeddings)

        return embeddings
