# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
import os
from importlib.metadata import version
from typing import Dict, Literal, Optional

import torch
from pkg_resources import packaging
from torch import Tensor

from megatron.core import parallel_state, tensor_parallel
from megatron.core.config_logger import has_config_logger_enabled, log_config_to_disk
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.models.bert.bert_layer_specs import bert_layer_with_transformer_engine_spec
from megatron.core.models.bert.bert_lm_head import BertLMHead
from megatron.core.models.bert.pooler import Pooler
from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.models.common.language_module.language_module import LanguageModule
from megatron.core.transformer.enums import AttnMaskType, ModelType
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.utils import get_linear_layer


def get_te_version():
    return packaging.version.Version(version("transformer-engine"))


class BertModel(LanguageModule):
    """Transformer language model.

    Args:
        config (TransformerConfig): transformer config
        num_tokentypes (int) : Set to 2 when args.bert_binary_head is True, and 0 otherwise. Defaults to 0.
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
        num_tokentypes: int,
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

        if has_config_logger_enabled(config):
            log_config_to_disk(config, locals(), prefix=type(self).__name__)

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

        self.attn_mask_dimensions = self._santiy_check_attention_and_get_attn_mask_dimension(
            transformer_layer_spec
        )

        # Embeddings.
        if self.pre_process:
            self.embedding = LanguageModelEmbedding(
                config=self.config,
                vocab_size=self.vocab_size,
                max_sequence_length=self.max_sequence_length,
                position_embedding_type=position_embedding_type,
                num_tokentypes=num_tokentypes,
            )

        if self.position_embedding_type == 'rope':
            self.rotary_pos_emb = RotaryEmbedding(
                kv_channels=self.config.kv_channels,
                rotary_percent=rotary_percent,
                rotary_interleaved=self.config.rotary_interleaved,
                seq_len_interpolation_factor=seq_len_interpolation_factor,
                use_cpu_initialization=self.config.use_cpu_initialization,
            )

        # Transformer.
        self.encoder = TransformerBlock(
            config=self.config,
            spec=self.transformer_layer_spec,
            pre_process=self.pre_process,
            post_process=self.post_process,
        )

        # Output
        if post_process:
            # TODO: Make sure you are passing in the mpu_vocab_size properly
            self.lm_head = BertLMHead(
                config.hidden_size,
                config,
            )

            self.output_layer = tensor_parallel.ColumnParallelLinear(
                config.hidden_size,
                self.vocab_size,
                config=config,
                init_method=config.init_method,
                bias=True,
                skip_bias_add=False,
                gather_output=not self.parallel_output,
                skip_weight_param_allocation=pre_process and share_embeddings_and_output_weights,
            )

            self.binary_head = None
            if self.add_binary_head:
                # TODO: Shoudl switch this to TE ?
                self.binary_head = get_linear_layer(
                    config.hidden_size, 2, config.init_method, config.perform_initialization
                )

                self.pooler = Pooler(
                    config.hidden_size, config.init_method, config, config.sequence_parallel
                )

        if self.pre_process or self.post_process:
            self.setup_embeddings_and_output_layer()

    def _santiy_check_attention_and_get_attn_mask_dimension(
        self, transformer_layer_spec: ModuleSpec
    ) -> str:
        """We do some checks and return attention mask dimensions for self attention

        Transformer engine library underwent a lot of change. So we need to change dimensions of the attention mask depending on the TE version. We also santiy check some arguments.
        1. If we use local version of attention dimension of the mask is [b,1,s,s]
        2. If we use transformer engine < 1.7 (Flash and Fused attention not supported. We use unfused path). Attn mask dimension is  [b,1,s,s]
        2. If we use transformer engine >= 1.7 (Flash and fused attention supported with attn mask dimension [b,1,1,s]). Unfused path will use attn mask dimension [b,1,s,s] with attn mask type arbitrary. Default if you dont set any NVTE_ATTN flag will just use unfused path.

        Args:
            transformer_layer_spec (ModuleSpec): _description_

        Returns:
            str: _description_
        """
        attn_mask_dimensions = "b1ss"
        if transformer_layer_spec == bert_layer_with_transformer_engine_spec:
            if get_te_version() >= packaging.version.Version("1.7.0"):
                if os.getenv('NVTE_FLASH_ATTN') == '0' and os.getenv('NVTE_FUSED_ATTN') == '0':
                    assert (
                        transformer_layer_spec.submodules.self_attention.params['attn_mask_type']
                        == AttnMaskType.arbitrary
                    ), "Set env variable NVTE_FLASH_ATTN to 1 or NVTE_FUSED_ATTN to 1 to use a more optimized attention kernal. Currently using unfused attention path. If you want to proceed with this path set AttnMaskType in module spec to be arbitrary"
                else:
                    attn_mask_dimensions = "b11s"
            else:
                assert os.getenv('NVTE_ALLOW_NONDETERMINISTIC_ALGO') == '0' or (
                    os.getenv('NVTE_FLASH_ATTN') == '0' and os.getenv('NVTE_FUSED_ATTN') == '0'
                ), "Flash and fused attention is not supported with transformer engine version < 1.7. Set NVTE_FLASH_ATTN=0 and NVTE_FUSED_ATTN=0 or upgrade transformer engine >= 1.7 or set NVTE_ALLOW_NONDETERMINISTIC_ALGO=0"
        return attn_mask_dimensions

    def bert_extended_attention_mask(self, attention_mask: Tensor) -> Tensor:
        """Creates the extended attention mask

        Converts the attention mask of dimension [batch size, 1, seq len] to [batch size, 1, seq len, seq len] or [batch size, 1, 1, seq_len] and makes it binary

        Args:
            attention_mask (Tensor): The input attention mask

        Returns:
            Tensor: The extended binary attention mask
        """
        # We create a 3D attention mask from a 2D tensor mask.
        if self.attn_mask_dimensions == "b1ss":
            # [b, 1, s]
            attention_mask_b1s = attention_mask.unsqueeze(1)
            # [b, s, 1]
            attention_mask_bs1 = attention_mask.unsqueeze(2)
            # [b, s, s]
            attention_mask_bss = attention_mask_b1s * attention_mask_bs1
            # [b, 1, s, s]
            extended_attention_mask = attention_mask_bss.unsqueeze(1)
        else:
            # [b, 1, 1, s]
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)

        # Convert attention mask to binary:
        extended_attention_mask = extended_attention_mask < 0.5

        return extended_attention_mask

    def bert_position_ids(self, token_ids):
        # Create position ids
        seq_length = token_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=token_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(token_ids)

        return position_ids

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
        self.encoder.set_input_tensor(input_tensor[0])

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        tokentype_ids: Tensor = None,
        lm_labels: Tensor = None,
        inference_params=None,
    ):
        """Forward function of BERT model

        Forward function of the BERT Model This function passes the input tensors
        through the embedding layer, and then the encoder and finally into the post
        processing layer (optional).

        It either returns the Loss values if labels are given  or the final hidden units
        """
        extended_attention_mask = self.bert_extended_attention_mask(attention_mask)

        if parallel_state.is_pipeline_first_stage():
            input_ids = input_ids
            position_ids = self.bert_position_ids(input_ids)
        else:
            position_ids = None
            input_ids = None

        # Encoder embedding.
        if self.pre_process:
            encoder_input = self.embedding(
                input_ids=input_ids, position_ids=position_ids, tokentype_ids=tokentype_ids
            )
        else:
            # intermediate stage of pipeline
            # encoder will get hidden_states from encoder.input_tensor
            encoder_input = None

        # Rotary positional embeddings (Why not move this into BERT/GPTEmberdding ?)
        rotary_pos_emb = None
        if self.position_embedding_type == 'rope':
            rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
                inference_params, self.encoder, encoder_input, self.config
            )
            rotary_pos_emb = self.rotary_pos_emb(rotary_seq_len)

        # Run encoder.
        hidden_states = self.encoder(
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

        hidden_states_after_lm_head = self.lm_head(hidden_states=hidden_states)
        logits, _ = self.output_layer(hidden_states_after_lm_head, weight=output_weight)

        binary_logits = None
        if self.binary_head is not None:
            binary_logits = self.binary_head(pooled_output)

        if lm_labels is None:
            # [s b h] => [b s h]
            return logits.transpose(0, 1).contiguous(), binary_logits

        loss = self.compute_language_model_loss(lm_labels, logits)

        return loss, binary_logits
