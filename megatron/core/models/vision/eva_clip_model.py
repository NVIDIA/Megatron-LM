# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import logging
from typing import Dict, Literal, Optional, Tuple, Union

import torch
from torch import Tensor

from megatron.core import InferenceParams, parallel_state, tensor_parallel
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.models.common.language_module.language_module import LanguageModule
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.enums import AttnMaskType, ModelType
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import make_tp_sharded_tensor_for_checkpoint
from megatron.core import mpu

from megatron.core.transformer.transformer_layer import TransformerLayer, make_viewless_tensor

class EVAClipTransformerLayer(TransformerLayer):
    def forward(
        self,
        hidden_states,
        attention_mask,
        context=None,
        context_mask=None,
        rotary_pos_emb=None,
        inference_params=None,
        packed_seq_params=None,
    ):
        # hidden_states: [s, b, h]

        attention_input = hidden_states

        # Self attention.
        attention_output, attention_bias = self.self_attention(
            attention_input,
            attention_mask=attention_mask,
            inference_params=inference_params,
            rotary_pos_emb=rotary_pos_emb,
            packed_seq_params=packed_seq_params,
        )

        if attention_bias is not None:
            attention_output = attention_output + attention_bias

        attention_output_after_layernorm = self.input_layernorm(attention_output)

        # Residual connection
        mlp_input = attention_input + attention_output_after_layernorm

        # MLP.
        mlp_output, mlp_bias = self.mlp(mlp_input)

        if mlp_bias is not None:
            mlp_output = mlp_output + mlp_bias

        mlp_output_after_layernorm = self.pre_mlp_layernorm(mlp_output)

        # Residual connection
        hidden_states = mlp_input + mlp_output_after_layernorm

        # Jit compiled function creates 'view' tensor. This tensor
        # potentially gets saved in the MPU checkpoint function context,
        # which rejects view tensors. While making a viewless tensor here
        # won't result in memory savings (like the data loader, or
        # p2p_communication), it serves to document the origin of this
        # 'view' tensor.
        output = make_viewless_tensor(
            inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True
        )

        return output, context



class Eva2ClipModel(LanguageModule):
    """Adapted from GPT Transformer language model.
    """
    def __init__(
        self,
        config: TransformerConfig,
        transformer_layer_spec: ModuleSpec,
        vocab_size: int,
        pre_process: bool = True,

    ) -> None:
        super().__init__(config=config)

        self.transformer_layer_spec: ModuleSpec = transformer_layer_spec
        self.vocab_size = vocab_size
        self.pre_process = pre_process
        
        self.visual_hidden_size = config.hidden_size
        self.patch_dim = config.patch_dim
        self.img_h = config.img_h
        self.img_w = config.img_w
        assert self.img_h % self.patch_dim == 0
        assert self.img_w % self.patch_dim == 0
        self.num_patches_per_dim_h = self.img_h // self.patch_dim
        self.num_patches_per_dim_w = self.img_w // self.patch_dim
        self.num_patches = self.num_patches_per_dim_h * self.num_patches_per_dim_w

        self.seq_length = self.num_patches + self.vocab_size
        self.max_sequence_length = self.seq_length

        # megatron core pipelining currently depends on model type
        # TODO: remove this dependency ?
        self.model_type = ModelType.encoder_or_decoder

        # These 2 attributes are needed for TensorRT-LLM export.
        self.max_position_embeddings = self.max_sequence_length

        if self.pre_process:
            self.conv1 = torch.nn.Conv2d(
                in_channels=3,
                out_channels=self.visual_hidden_size,
                kernel_size=self.patch_dim,
                stride=self.patch_dim,
                bias=True,
            )
            self.embedding = LanguageModelEmbedding(
                config=self.config,
                vocab_size=self.vocab_size,
                max_sequence_length=self.max_sequence_length,
                position_embedding_type='learned_absolute',
                parallel_word_embedding=False,
            )

        self.post_process = False
        # Transformer.
        self.encoder = TransformerBlock(
            config=self.config,
            spec=transformer_layer_spec,
            pre_process=self.pre_process,
            post_process=self.post_process,
        )

        self.share_embeddings_and_output_weights = False
        if self.pre_process or self.post_process:
            self.setup_embeddings_and_output_layer()
        

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
        position_ids: Tensor,
        attention_mask: Tensor,
        encoder_input: Tensor = None,
        labels: Tensor = None,
        inference_params: InferenceParams = None,
        packed_seq_params: PackedSeqParams = None,
        extra_block_kwargs: dict = None,
        external_inputs: dict = {},
    ) -> Tensor:
        """Forward function of the GPT Model This function passes the input tensors
        through the embedding layer, and then the decoeder and finally into the post
        processing layer (optional).

        It either returns the Loss values if labels are given  or the final hidden units
        """
        # If encoder_input is provided (not None), then input_ids and position_ids are ignored.
        # Otherwise, apply embedding layer on input_ids and position_ids to get encoder_input.

        # encoder embedding.
        if encoder_input is not None:
            pass
        elif self.pre_process:
            if hasattr(inference_params, 'external_inputs') and inference_params.external_inputs is not None and not inference_params.key_value_memory_dict:
                external_inputs = inference_params.external_inputs
                print("setting vision inputs to inference_params.external_inputs......")
            if external_inputs:
                images = external_inputs["images"]
                embeddings = self.conv1(images)
                embeddings = embeddings.flatten(2).transpose(1, 2)
                if mpu.get_context_parallel_world_size() != 1:
                    cp_size = mpu.get_context_parallel_world_size()
                    cp_rank = mpu.get_context_parallel_rank()
                    val = embeddings
                    seq_dim = 1
                    val = val.view(
                        *val.shape[0:seq_dim],
                        2 * cp_size,
                        val.shape[seq_dim] // (2 * cp_size),
                        *val.shape[(seq_dim + 1) :],
                    )
                    index = torch.tensor([cp_rank, (2 * cp_size - cp_rank - 1)], 
                                        device="cpu", pin_memory=True).cuda(non_blocking=True)
                    val = val.index_select(seq_dim, index)
                    val = val.view(*val.shape[0:seq_dim], -1, *val.shape[(seq_dim + 2) :])
                    embeddings = val
                external_feature_dict = {
                    "features": embeddings,
                    "pre_len": input_ids.shape[1] - embeddings.shape[1]
                }
                encoder_input = self.embedding(input_ids=input_ids, position_ids=position_ids, external_feature_dict=external_feature_dict)
            else:
                encoder_input = self.embedding(input_ids=input_ids, position_ids=position_ids)
        else:
            # intermediate stage of pipeline
            # encoder will get hidden_states from encoder.input_tensor
            encoder_input = None

        # Run encoder.
        hidden_states = self.encoder(
            hidden_states=encoder_input,
            attention_mask=attention_mask,
            inference_params=inference_params,
            packed_seq_params=packed_seq_params,
            **(extra_block_kwargs or {}),
        )

        return hidden_states
