# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import logging
import random
from typing import Literal, Optional, Union

import torch
from torch import Tensor

from megatron.core import InferenceParams, parallel_state, tensor_parallel
from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding, NoiseSchedulerConfig
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.models.common.language_module.language_module import LanguageModule
from megatron.core.transformer.enums import AttnMaskType, ModelType
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import make_tp_sharded_tensor_for_checkpoint


class GPTModel(LanguageModule):
    """GPT Transformer language model.

    Args:
        config (TransformerConfig): Transformer config
        transformer_layer_spec (ModuleSpec): Specifies module to use for transformer layers
        vocab_size (int): Vocabulary size
        max_sequence_length (int): maximum size of sequence. This is used for positional embedding
        pre_process (bool, optional): Include embedding layer (used with pipeline parallelism). Defaults to True.
        post_process (bool, optional): Include an output layer (used with pipeline parallelism). Defaults to True.
        fp16_lm_cross_entropy (bool, optional): Defaults to False.
        parallel_output (bool, optional): Do not gather the outputs, keep them split across tensor parallel ranks. Defaults to True.
        share_embeddings_and_output_weights (bool, optional): When True, input embeddings and output logit weights are shared. Defaults to False.
        position_embedding_type (Literal[learned_absolute,rope], optional):  Position embedding type.. Defaults to 'learned_absolute'.
        rotary_percent (float, optional): Percent of rotary dimension to use for rotary position embeddings. Ignored unless position_embedding_type is 'rope'. Defaults to 1.0.
        rotary_base (int, optional): Base period for rotary position embeddings. Ignored unless position_embedding_type is 'rope'. Defaults to 10000.
        seq_len_interpolation_factor (Optional[float], optional): scale of linearly interpolating RoPE for longer sequences. The value must be a float larger than 1.0. Defaults to None.
        embedding_noise (bool, optional): Add noise to embedding layer. Defaults to False.
        embedding_noise_mean (float, optional): Mean of embedding noise. Defaults to 0.0.
        embedding_noise_std (float, optional): Standard deviation of embedding noise. Defaults to 0.001.
        embedding_noise_type (str, optional): Type of embedding noise. Defaults to 'uniform'.
        neft (bool, optional): Use NEFT. Defaults to False.
        neft_alpha (float, optional): NEFT alpha. Defaults to 5.0.
        noise_positonal_embedding (bool, optional): Add noise to positional embedding. Defaults to False.
        adversarial_training (bool, optional): Adversarial training. Defaults to False.
        adversarial_training_epsilon (float, optional): Adversarial training scaling factor. Defaults to 0.01.
        noise_scheduler_config (Optional[dict], optional): Noise scheduler configuration. Defaults to None.
        cre_adversarial_training (bool, optional): Contextualized representation adversarial training. Defaults to False.
        creat_init_var (float, optional): Contextualized representation adversarial training initial variance. Used to scale the initial adversarial perturbation. Defaults to 0.01.
        creat_num_adv_steps (int, optional): Contextualized representation adversarial training number of adversarial steps. Used to refine the adversarial perturbation. Defaults to 2.
        creat_adv_temp (float, optional): Contextualized representation adversarial training adversarial temperature. Used to control the strength of the adversarial perturbation. Defaults to 1.0.
        creat_lambda (float, optional): Contextualized representation adversarial training lambda. Used to balance the adversarial loss and the original loss. Defaults to 0.5.
        creat_lr (float, optional): Contextualized representation adversarial training learning rate. Used to scale the gradient of the adversarial perturbation. Defaults to 0.1.
        creat_max_norm (float, optional): Contextualized representation adversarial training max norm. Used to clip the adversarial perturbation. Defaults to 0.1.
    """

    def __init__(
        self,
        config: TransformerConfig,
        transformer_layer_spec: ModuleSpec,
        vocab_size: int,
        max_sequence_length: int,
        pre_process: bool = True,
        post_process: bool = True,
        fp16_lm_cross_entropy: bool = False,
        parallel_output: bool = True,
        share_embeddings_and_output_weights: bool = False,
        position_embedding_type: Literal['learned_absolute', 'rope'] = 'learned_absolute',
        rotary_percent: float = 1.0,
        rotary_base: int = 10000,
        seq_len_interpolation_factor: Optional[float] = None,
        embedding_noise=False,
        embedding_noise_mean=0.0,
        embedding_noise_std=0.001,
        embedding_noise_type='uniform',
        neft=False,
        neft_alpha=5.0,
        noise_positonal_embedding=False,
        adversarial_training=False,
        adversarial_training_epsilon=0.01,
        noise_scheduler_config: NoiseSchedulerConfig = None,
        cre_adversarial_training=False,
        creat_init_var=1e-2,
        creat_num_adv_steps=2,
        creat_adv_temp=1.0,
        creat_lambda=0.5,
        creat_lr=0.1,
        creat_max_norm=0.1,
        neft_reimplement=False,
        embedmix=False,
        embedmix_subset_p=0.0,
        embedmix_embed_p=0.0,
        embedmix_tokens_p=0.0,
        embedmix_type=None,
        embedmix_alpha=0.5,
    ) -> None:
        super().__init__(config=config)

        self.transformer_layer_spec: ModuleSpec = transformer_layer_spec
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.pre_process = pre_process
        self.post_process = post_process
        self.fp16_lm_cross_entropy = fp16_lm_cross_entropy
        self.parallel_output = parallel_output
        self.share_embeddings_and_output_weights = share_embeddings_and_output_weights
        self.position_embedding_type = position_embedding_type

        self.embedding_noise = embedding_noise
        self.embedding_noise_mean = embedding_noise_mean
        self.embedding_noise_std = embedding_noise_std
        self.embedding_noise_type = embedding_noise_type
        self.neft = neft
        self.neft_alpha = neft_alpha
        self.noise_positonal_embedding = noise_positonal_embedding
        
        self.adversarial_training = adversarial_training
        self.adversarial_training_epsilon = adversarial_training_epsilon

        self.cre_adversarial_training = cre_adversarial_training
        self.creat_init_var = creat_init_var
        self.creat_num_adv_steps = creat_num_adv_steps
        self.creat_adv_temp = creat_adv_temp
        self.creat_lambda = creat_lambda
        self.creat_lr = creat_lr
        self.creat_max_norm = creat_max_norm
        self.neft_reimplement = neft_reimplement
        

        self.embedmix = embedmix
        self.embedmix_subset_p = embedmix_subset_p
        self.embedmix_embed_p = embedmix_embed_p
        self.embedmix_tokens_p = embedmix_tokens_p
        self.embedmix_type = embedmix_type
        self.embedmix_alpha = embedmix_alpha

        # megatron core pipelining currently depends on model type
        # TODO: remove this dependency ?
        self.model_type = ModelType.encoder_or_decoder

        if self.pre_process:
            self.embedding = LanguageModelEmbedding(
                config=self.config,
                vocab_size=self.vocab_size,
                max_sequence_length=self.max_sequence_length,
                position_embedding_type=position_embedding_type,
                embedding_noise=self.embedding_noise,
                embedding_noise_mean=self.embedding_noise_mean,
                embedding_noise_std=self.embedding_noise_std,
                embedding_noise_type=self.embedding_noise_type,
                neft=self.neft,
                neft_alpha=self.neft_alpha,
                noise_positonal_embedding=self.noise_positonal_embedding,
                noise_scheduler_config=noise_scheduler_config,
                neft_reimplement=self.neft_reimplement,
            )

        if self.position_embedding_type == 'rope':
            self.rotary_pos_emb = RotaryEmbedding(
                kv_channels=self.config.kv_channels,
                rotary_percent=rotary_percent,
                seq_len_interpolation_factor=seq_len_interpolation_factor,
                rotary_base=rotary_base,
            )

        # Transformer.
        self.decoder = TransformerBlock(
            config=self.config,
            spec=transformer_layer_spec,
            pre_process=self.pre_process,
            post_process=self.post_process,
        )

        # Output
        if post_process:
            self.output_layer = tensor_parallel.ColumnParallelLinear(
                config.hidden_size,
                self.vocab_size,
                config=config,
                init_method=config.init_method,
                bias=False,
                skip_bias_add=False,
                gather_output=not self.parallel_output,
                skip_weight_param_allocation=self.pre_process
                and self.share_embeddings_and_output_weights,
            )

        if self.share_embeddings_and_output_weights and (self.pre_process or self.post_process):
            self.initialize_last_stage_with_word_embeddings()

    def set_input_tensor(self, input_tensor: Tensor) -> None:
        """Sets input tensor to the model.

        See megatron.model.transformer.set_input_tensor()

        Args:
            input_tensor (Tensor): Sets the input tensor for the model.
        """
        # This is usually handled in schedules.py but some inference code still
        # gives us non-lists or None
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]

        assert len(input_tensor) == 1, 'input_tensor should only be length 1 for gpt/bert'
        self.decoder.set_input_tensor(input_tensor[0])

    def embedmix_forward(self, decoder_input: torch.Tensor) -> torch.Tensor:
        if self.embedmix_subset_p == 0.0:
            return decoder_input

        decoder_input_clone = decoder_input.clone()
        seq_length, batch_size, embed_size = decoder_input_clone.shape
        
        num_batches_to_perturb = max(1, round(self.embedmix_subset_p * batch_size))
        num_tokens_to_perturb = max(2, round(self.embedmix_tokens_p * seq_length))
        
        if num_tokens_to_perturb % 2 == 1:
            num_tokens_to_perturb += 1
        
        perturb_batches_indices = torch.randperm(batch_size)[:num_batches_to_perturb]
        perturb_tokens_indices = torch.randperm(seq_length)[:num_tokens_to_perturb].view(-1, 2)
        
        for batch_idx in perturb_batches_indices:
            for token_pair in perturb_tokens_indices:
                if self.embedmix_type == 'swap':
                    perturb_mask_len = round(embed_size * self.embedmix_embed_p)
                    start_index = random.randint(0, embed_size - perturb_mask_len)
                    end_index = start_index + perturb_mask_len

                    decoder_input_clone[token_pair[0], batch_idx, start_index:end_index], \
                    decoder_input_clone[token_pair[1], batch_idx, start_index:end_index] = \
                    decoder_input_clone[token_pair[1], batch_idx, start_index:end_index].clone(), \
                    decoder_input_clone[token_pair[0], batch_idx, start_index:end_index].clone()
                elif self.embedmix_type == 'mix':
                    decoder_input_clone[token_pair[0], batch_idx] = self.embedmix_alpha * decoder_input_clone[token_pair[0], batch_idx] + \
                                                                    (1 - self.embedmix_alpha) * decoder_input_clone[token_pair[1], batch_idx]
        return decoder_input_clone

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
        decoder_input: Tensor = None,
        labels: Tensor = None,
        inference_params: InferenceParams = None,
        extra_block_kwargs: dict = None,
        input_lengths: Optional[Tensor] = None
    ) -> Tensor:
        """Forward function of the GPT Model This function passes the input tensors
        through the embedding layer, and then the decoeder and finally into the post
        processing layer (optional).

        It either returns the Loss values if labels are given  or the final hidden units
        """
        # If decoder_input is provided (not None), then input_ids and position_ids are ignored.
        # Otherwise, apply embedding layer on input_ids and position_ids to get decoder_input.

        # Decoder embedding.
        if decoder_input is not None:
            pass
        elif self.pre_process:
            decoder_input = self.embedding(input_ids=input_ids, position_ids=position_ids, input_lengths=input_lengths)
        else:
            # intermediate stage of pipeline
            # decoder will get hidden_states from encoder.input_tensor
            decoder_input = None

        # Rotary positional embeddings (embedding is None for PP intermediate devices)
        rotary_pos_emb = None
        if self.position_embedding_type == 'rope':
            rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
                inference_params, self.decoder, decoder_input, self.config
            )
            rotary_pos_emb = self.rotary_pos_emb(rotary_seq_len)
        if self.adversarial_training and decoder_input is not None and self.training:
            decoder_input_clone = decoder_input.clone().detach().requires_grad_(True)

            hidden_states = self.decoder(
                hidden_states=decoder_input_clone,
                attention_mask=attention_mask,
                inference_params=inference_params,
                rotary_pos_emb=rotary_pos_emb,
                **(extra_block_kwargs or {}),
            )

            output_weight = None
            if self.share_embeddings_and_output_weights:
                output_weight = self.shared_embedding_or_output_weight()
            logits, _ = self.output_layer(hidden_states, weight=output_weight)

            loss = self.compute_language_model_loss(labels, logits)
            self.zero_grad() # Ensure that the gradients are zeroed out before backpropagation
            loss.mean().backward(retain_graph=True)
            perturbation = self.adversarial_training_epsilon * decoder_input_clone.grad.sign()
            decoder_input = decoder_input + perturbation.detach()

        if self.embedmix:
            decoder_input = self.embedmix_forward(decoder_input)

        # Run decoder.
        hidden_states = self.decoder(
            hidden_states=decoder_input,
            attention_mask=attention_mask,
            inference_params=inference_params,
            rotary_pos_emb=rotary_pos_emb,
            **(extra_block_kwargs or {}),
        )

        if not self.post_process:
            return hidden_states

        # logits and loss
        output_weight = None
        if self.share_embeddings_and_output_weights:
            output_weight = self.shared_embedding_or_output_weight()
        logits, _ = self.output_layer(hidden_states, weight=output_weight)

        if labels is None:
            # [s b h] => [b s h]
            return logits.transpose(0, 1).contiguous()

        loss = self.compute_language_model_loss(labels, logits)

        if self.cre_adversarial_training and self.training:
            cos_sim = torch.nn.CosineSimilarity(dim=-1)
            delta = torch.zeros_like(decoder_input).normal_(0, 1) * self.creat_init_var
            delta.requires_grad_()
            for j in range(self.creat_num_adv_steps):
                hidden_states_p = self.decoder(
                    hidden_states=decoder_input + delta,
                    attention_mask=attention_mask,
                    inference_params=inference_params,
                    rotary_pos_emb=rotary_pos_emb,
                    **(extra_block_kwargs or {}),
                )
                if self.share_embeddings_and_output_weights:
                    output_weight = self.shared_embedding_or_output_weight()

                logits_p, _ = self.output_layer(hidden_states_p, weight=output_weight)
                loss_p = self.compute_language_model_loss(labels, logits_p)

                # No need to compute loss and gradients for the last step
                if j == self.creat_num_adv_steps - 1:
                    break

                cos_loss_p = cos_sim(hidden_states_p, hidden_states.detach()).transpose(0, 1)
                loss_p = loss_p - cos_loss_p * self.creat_adv_temp
                delta = self._inner_update(delta, loss_p)
                delta.requires_grad_()
            loss = loss_p * self.creat_lambda + loss * (1 - self.creat_lambda)

        return loss

    def _inner_update(self, delta, loss):
        loss = loss.transpose(0, 1).sum(-1).mean()
        if delta.grad is not None:
            delta.grad.zero_()
    
    # Compute gradients using backward
        loss.backward(retain_graph=True, create_graph=True)
    
    # Extract the computed gradient from delta
        delta_grad = delta.grad
        _shape = None
        if delta.dim() > 3:
            # e.g. multi-choice
            _shape = delta.shape
            delta, delta_grad = delta.view(-1, _shape[-2], _shape[-1]), delta_grad.view(-1, _shape[-2], _shape[-1])

        grad_norm = torch.norm(delta_grad.view(delta_grad.shape[0], -1), dim=-1, p="fro")
        grad_norm = torch.clamp(grad_norm, min=1e-8).view(-1, 1, 1)
        delta = (delta + self.creat_lr * delta_grad / grad_norm).detach()

        delta_norm = torch.norm(delta.view(delta.shape[0], -1), dim=-1, p="fro").detach()
        clip_mask = (delta_norm > self.creat_max_norm).to(delta)
        clip_weights = self.creat_max_norm / delta_norm * clip_mask + (1 - clip_mask)
        delta = (delta * clip_weights.view(-1, 1, 1)).detach()

        if _shape is not None:
            delta = delta.view(_shape)
        delta.grad.zero_()
        return delta

    def sharded_state_dict(self, prefix: str = '') -> dict:
        sharded_state_dict = {}

        if self.pre_process:
            embedding_prefix = f'{prefix}embedding.'
            embedding_sharded_state_dict = self.embedding.sharded_state_dict(
                prefix=embedding_prefix
            )
            sharded_state_dict.update(embedding_sharded_state_dict)

        decoder_prefix = f'{prefix}decoder.'
        decoder_sharded_state_dict = self.decoder.sharded_state_dict(prefix=decoder_prefix)
        sharded_state_dict.update(decoder_sharded_state_dict)

        if self.post_process:
            output_layer_prefix = f'{prefix}output_layer.'
            output_layer_key = f'{output_layer_prefix}weight'
            if self.share_embeddings_and_output_weights:
                if not self.pre_process:
                    # when sharing embeddings with last stage, we need to use the weights from the first stage
                    # on pipeline first rank, word embeddings are saved to {prefix}embedding.word_embeddings.weight
                    tensor = self.shared_embedding_or_output_weight()
                    first_stage_word_emb_key = f'{prefix}embedding.word_embeddings.weight'
                    last_stage_word_emb_replica_id = (
                        1,  # copy of first stage embedding
                        0,
                        parallel_state.get_data_parallel_rank(),
                    )

                    sharded_output_layer_tensor = make_tp_sharded_tensor_for_checkpoint(
                        tensor=tensor,
                        key=first_stage_word_emb_key,
                        replica_id=last_stage_word_emb_replica_id,
                        allow_shape_mismatch=True,
                    )

                    sharded_state_dict[output_layer_key] = sharded_output_layer_tensor

            else:
                output_layer_state_dict = self.output_layer.state_dict(
                    prefix=output_layer_prefix, keep_vars=True
                )
                output_layer_tensor = output_layer_state_dict[output_layer_key]
                # independent output layer
                sharded_output_layer_tensor = make_tp_sharded_tensor_for_checkpoint(
                    tensor=output_layer_tensor, key=output_layer_key, allow_shape_mismatch=True,
                )

                sharded_state_dict[output_layer_key] = sharded_output_layer_tensor

        return sharded_state_dict
