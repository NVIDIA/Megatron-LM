# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from dataclasses import dataclass
import math
from typing import List, Literal, Optional

import torch
from torch import Tensor

from megatron.core import tensor_parallel
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import (
    make_sharded_tensor_for_checkpoint,
    make_tp_sharded_tensor_for_checkpoint,
)

@dataclass
class NoiseSchedulerConfig:
    class_name: str
    milestones: Optional[List[int]] = None
    max_steps: Optional[int] = None
    num_milestones: Optional[int] = None
    gamma: Optional[float] = None
    verbose: Optional[bool] = False
    starting_value: Optional[float] = 0.0

class NoiseScheduler:
    def __init__(self, noise_scheduler_config):
        self.noise_scheduler_config = noise_scheduler_config
        self.current_step = 0

    def step(self):
        self.current_step += 1
        
class MultiStepNoiseScheduler(NoiseScheduler):
    def __init__(self, noise_scheduler_config):
        super().__init__(noise_scheduler_config)
        self.milestones = noise_scheduler_config.get('milestones')
        if self.milestones is None:
            num_milestones = noise_scheduler_config.get('num_milestones')
            max_steps = noise_scheduler_config.get('max_steps')
            if num_milestones is not None and max_steps is not None:
                self.milestones =  [(i + 1) * max_steps // (num_milestones + 1) for i in range(num_milestones)]
            else:
                raise ValueError("Configuration for MultiStepNoiseScheduler must include either 'milestones' or both 'num_milestones' and 'max_steps'.")

        self.gamma = noise_scheduler_config.get('gamma')
        self.verbose = noise_scheduler_config.get('verbose', True)
        self.starting_value = noise_scheduler_config.get('starting_value', 1.0)
        self.current_value = self.starting_value
        if self.verbose:
            print(f'Noise value initialized to {self.current_value}')
            print(f'Noise milestones: {self.milestones}')
            milestone_to_val = {milestone: self.starting_value * (self.gamma ** i) for i, milestone in enumerate(self.milestones)}
            print(f'Noise will change: {milestone_to_val}')

    def get_noise(self):
        if self.current_step in self.milestones:
            if self.verbose:
                print(f'Noise step {self.current_step} reached milestone')
            self.current_value *= self.gamma
            if self.verbose:
                print(f'Noise value changed to {self.current_value}')
        return self.current_value

def get_noise_scheduler(class_name):
    if class_name == 'MultiStepNoiseScheduler':
        return MultiStepNoiseScheduler
    else:
        raise ValueError(f'Noise scheduler {class_name} not implemented')


class LanguageModelEmbedding(MegatronModule):
    """Language model embeddings.

    Arguments:
        config (TransformerConfig): config object with all necessary configs for TransformerBlock
        vocab_size (int): vocabulary size
        max_sequence_length (int): maximum size of sequence. This
                             is used for positional embedding
        add_position_embedding (bool): Add a position embedding.
        embedding_dropout_prob (float): dropout probability for embeddings
        num_tokentypes (int): Set to 0 without binary head, and 2 with a binary head . Defaults to 0.
        embedding_noise (bool): Add noise to the embeddings. Defaults to False.
        embedding_noise_mean (float): Mean of the noise added to the embeddings. Defaults to 0.0.
        embedding_noise_std (float): Standard deviation of the noise added to the embeddings. Defaults to 0.001.
        embedding_noise_type (str): Type of noise added to the embeddings. Defaults to 'uniform'.
        neft (bool): Add noise to the embeddings using NEFT. Defaults to False.
        neft_alpha (float): Scaling factor for NEFT. Defaults to 5.0.
        noise_positonal_embedding (bool): Add noise to the positional embeddings. Defaults to False.
        noise_scaling_factor (float): Scaling factor for the noise added to the embeddings. Defaults to None.
        noise_ascend (bool): Ascend the noise added to the embeddings. Defaults to None.
    """

    def __init__(
        self,
        config: TransformerConfig,
        vocab_size: int,
        max_sequence_length: int,
        position_embedding_type: Literal['learned_absolute', 'rope'] = 'learned_absolute',
        num_tokentypes: int = 0,
        embedding_noise=False,
        embedding_noise_mean=0.0,
        embedding_noise_std=0.001,
        embedding_noise_type='uniform',
        neft=False,
        neft_alpha=5.0,
        noise_positonal_embedding=False,
        noise_scheduler_config=None,
    ):
        super().__init__(config=config)

        self.config: TransformerConfig = config
        self.vocab_size: int = vocab_size
        self.max_sequence_length: int = max_sequence_length
        self.add_position_embedding: bool = position_embedding_type == 'learned_absolute'
        self.num_tokentypes = num_tokentypes

        self.embedding_noise = embedding_noise
        self.embedding_noise_mean = embedding_noise_mean
        self.embedding_noise_std = embedding_noise_std
        self.embedding_noise_type = embedding_noise_type
        self.neft = neft
        self.neft_alpha = neft_alpha
        self.noise_positonal_embedding = noise_positonal_embedding

        if noise_scheduler_config is not None:
            self.noise_scheduler = get_noise_scheduler(noise_scheduler_config.class_name)(noise_scheduler_config)

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

    def _noise(self, embeddings):
        if self.training:
            if self.embedding_noise and not self.neft:
                    noise_scale = self.embedding_noise_std
                    if hasattr(self, 'noise_scheduler') and self.noise_scheduler is not None:
                        noise_scale = self.noise_scheduler.get_noise()
                    if self.embedding_noise_type == 'uniform':
                        noise = torch.empty_like(embeddings).uniform_(self.embedding_noise_mean, noise_scale).detach()
                    elif self.embedding_noise_type == 'normal':
                        noise = torch.empty_like(embeddings).normal_(self.embedding_noise_mean, noise_scale).detach()
                    else:
                        raise NotImplementedError(f"embedding noise type {self.embedding_noise_type} not implemented")

                    original_norm = torch.norm(embeddings, p=2, dim=1, keepdim=True)
                    embeddings = embeddings + noise
                    noisy_norm = torch.norm(embeddings, p=2, dim=1, keepdim=True)
                    embeddings = embeddings * (original_norm / noisy_norm)
            elif self.neft:
                    current_alpha = self.neft_alpha
                    if hasattr(self, 'noise_scheduler') and self.noise_scheduler is not None:
                        current_alpha = self.noise_scheduler.get_noise()

                    epsilon = torch.empty_like(embeddings).uniform_(-1, 1).detach()
                    scaled_noise = (current_alpha / math.sqrt(embeddings.shape[0] * embeddings.shape[-1])) * epsilon
                    embeddings = embeddings + scaled_noise
        return embeddings

    def forward(self, input_ids: Tensor, position_ids: Tensor, tokentype_ids: int = None) -> Tensor:
        """Forward pass of the embedding module
        Args:
            input_ids (Tensor): The input tokens
            position_ids (Tensor): The position id's used to calculate position embeddings
            tokentype_ids (int): The token type ids. Used when args.bert_binary_head is set to True. Defaults to None

        Returns:
            Tensor: The output embeddings
        """
        word_embeddings = self.word_embeddings(input_ids)

        if not self.noise_positonal_embedding:
            word_embeddings = self._noise(word_embeddings)

        if self.add_position_embedding:
            position_embeddings = self.position_embeddings(position_ids)
            embeddings = word_embeddings + position_embeddings
        else:
            embeddings = word_embeddings

        if self.noise_positonal_embedding:
            embeddings = self._noise(embeddings)
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
            embeddings = tensor_parallel.scatter_to_sequence_parallel_region(embeddings)
            # `scatter_to_sequence_parallel_region` returns a view, which prevents
            # the original tensor from being garbage collected. Clone to facilitate GC.
            # Has a small runtime cost (~0.5%).
            if self.config.clone_scatter_output_in_embedding:
                embeddings = embeddings.clone()
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
