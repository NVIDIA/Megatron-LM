# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
from typing import Literal, Optional

import torch
from torch import Tensor

from megatron.core.models.bert.bert_lm_head import BertLMHead
from megatron.core.models.bert.pooler import Pooler
from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding
from megatron.core.models.common.language_module.language_module import LanguageModule
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.transformer.enums import AttnMaskType, ModelType
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.utils import get_linear_layer
from megatron.model.bert_model import bert_extended_attention_mask, bert_position_ids


class BertModel(LanguageModule):
    """Transformer language model.

    Args:
        config (TransformerConfig): transformer config
        transformer_layer_spec (ModuleSpec): Specifies module to use for transformer layers
        vocab_size (int): vocabulary size
        max_sequence_length (int): maximum size of sequence. This is used for positional embedding
        pre_process (bool): Include embedding layer (used with pipeline parallelism)
        post_process (bool): Include an output layer (used with pipeline parallelism)
        parallel_output (bool): Do not gather the outputs, keep them split across tensor parallel ranks
        share_embeddings_and_output_weights (bool): When True, input embeddings and output logit weights are shared. Defaults to False.
        position_embedding_type (string): Position embedding type. Options ['learned_absolute', 'rope'].
            Defaults is 'learned_absolute'.
        rotary_percent (float): Percent of rotary dimension to use for rotary position embeddings.
            Defaults to 1.0 (100%). Ignored unless position_embedding_type is 'rope'.
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
        seq_len_interpolation_factor: Optional[float] = None,
        add_binary_head=True,
        return_embeddings=False,
    ):
        super(BertModel, self).__init__(config=config)

        if return_embeddings:
            assert self.post_process and self.add_binary_head

        self.config: TransformerConfig = config
        self.transformer_layer_spec: ModuleSpec = transformer_layer_spec
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.pre_process = pre_process
        self.post_process = post_process
        self.fp16_lm_cross_entropy = fp16_lm_cross_entropy
        self.parallel_output = parallel_output
        self.share_embeddings_and_output_weights = share_embeddings_and_output_weights
        self.position_embedding_type = position_embedding_type
        self.add_binary_head = add_binary_head
        self.return_embeddings = return_embeddings

        # megatron core pipelining currently depends on model type
        self.model_type = ModelType.encoder_or_decoder

        # Embeddings.
        if self.pre_process:
            self.embedding = LanguageModelEmbedding(
                config=self.config,
                vocab_size=self.vocab_size,
                max_sequence_length=self.max_sequence_length,
                position_embedding_type=position_embedding_type,
            )

        if self.position_embedding_type == 'rope':
            self.rotary_pos_emb = RotaryEmbedding(
                self.config.kv_channels, rotary_percent, seq_len_interpolation_factor
            )

        # Transformer.
        self.transformer = TransformerBlock(
            config=self.config,
            transformer_layer_spec=self.transformer_layer_spec,
            self_attn_mask_type=AttnMaskType.padding,
            pre_process=self.pre_process,
            post_process=self.post_process,
        )

        # Output
        if post_process:
            self.lm_head = BertLMHead(
                config.hidden_size,
                config,
                parallel_output,
                self.vocab_size,
                self.pre_process,
                self.share_embeddings_and_output_weights,
            )

            self.output_layer = self.lm_head.output_layer

            self.binary_head = None
            if self.add_binary_head:
                # TODO: Shoudl switch this to TE ?
                self.binary_head = get_linear_layer(config.hidden_size, 2, config.init_method)

                self.pooler = Pooler(
                    config.hidden_size, config.init_method, config.sequence_parallel, config
                )

        if self.share_embeddings_and_output_weights and (self.pre_process or self.post_process):
            self.initialize_last_stage_with_word_embeddings()

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        tokentype_ids: Tensor = None,
        lm_labels: Tensor = None,
        inference_params=None,
    ):
        extended_attention_mask = bert_extended_attention_mask(attention_mask)

        position_ids = bert_position_ids(input_ids)

        # Encoder embedding.
        if self.pre_process:
            # TODO : tokentype_ids should be used to be consistant with non core bert model
            encoder_input = self.embedding(input_ids=input_ids, position_ids=position_ids)
        else:
            # intermediate stage of pipeline
            # decoder will get hidden_states from encoder.input_tensor
            encoder_input = None

        # Rotary positional embeddings (Why not move this into BERT/GPTEmberdding ?)
        rotary_pos_emb = None
        if self.position_embedding_type == 'rope':
            rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
                inference_params, self.transformer, encoder_input, self.config
            )
            rotary_pos_emb = self.rotary_pos_emb(rotary_seq_len)

        # Run decoder.
        hidden_states = self.transformer(
            hidden_states=encoder_input,
            attention_mask=extended_attention_mask,
            inference_params=inference_params,
            rotary_pos_emb=rotary_pos_emb,
        )
        if not self.post_process:
            return hidden_states

        if self.add_binary_head:
            pooled_output = self.pooler(hidden_states, 0)

        if self.return_embeddings:
            embeddings = torch.transpose(hidden_states, 0, 1)
            masks = torch.sum(attention_mask, dim=1)
            # Collect masked embeddings.
            output = torch.zeros(
                size=(embeddings.shape[0], embeddings.shape[2]),
                dtype=torch.float32,
                device=torch.cuda.current_device(),
            )
            for i, (embedding, mask) in enumerate(zip(embeddings, masks)):
                output[i, :] = torch.mean(embedding[1 : mask - 1], dim=0)
            return output

        # logits and loss
        output_weight = None
        if self.share_embeddings_and_output_weights:
            output_weight = self.shared_embedding_or_output_weight()

        logits = self.lm_head(hidden_states=hidden_states, word_embeddings_weight=output_weight)

        binary_logits = None
        if self.binary_head is not None:
            binary_logits = self.binary_head(pooled_output)

        if lm_labels is None:
            # [s b h] => [b s h]
            return logits.transpose(0, 1).contiguous(), binary_logits

        loss = self.compute_language_model_loss(lm_labels, logits)

        return loss, binary_logits

    # TODO: add distributed checkpointing
    def state_dict_for_save_checkpoint(self, prefix='', keep_vars=False):
        pass

    # TODO: add distributed checkpointing
    def load_state_dict(self, state_dict, strict=True):
        pass
