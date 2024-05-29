# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
import logging
from collections import namedtuple
from functools import partial
from typing import List

import torch

from megatron.core import InferenceParams, parallel_state
from megatron.core.models.gpt import GPTModel
from megatron.core.models.vision.clip_vit_model import CLIPViTModel
from megatron.core.models.vision.multimodal_projector import MultimodalProjector
from megatron.core.transformer import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig


# Note: This is under development and may be missing features.
class LLaVAModel(MegatronModule):
    """LLaVA multi-modal model.

    Args:
        language_transformer_config (TransformerConfig): Transformer config for the language model.
        language_transformer_layer_spec (ModuleSpec): Specifies module to use for transformer layers of the language model.
        language_vocab_size (int): Language model vocabulary size.
        language_max_sequence_length (int): Language model maximum sequence length. This is used for positional embedding.
        vision_transformer_config (TransformerConfig): Transformer config for the vision model.
        vision_transformer_layer_spec (ModuleSpec): Specifies module to use for transformer layers of the vision model.
        drop_vision_class_token (bool): Drop vision class token(s) before input to the language model.
        vision_projection_config (TransformerConfig): Config for the projection from vision model outputs to language model inputs.
        vision_projection_layer_spec (ModuleSpec): Specifies the module to use for the vision projection.
        vision_projection_type (str): Type of the vision projection to use. Default is a 2-layer MLP.
        allow_missing_vision_projection_checkpoint (bool): Allow vision projection weights to be missing when loading a checkpoint. Default False.
        parallel_output (bool): Do not gather the outputs, keep them split across tensor parallel ranks. This is typically True for training and False for inference.
        language_position_embedding_type (str): Position embedding type to use in the language model. Default learned absolute.
        language_rotary_percent (float): Percent of rotary dimension to use for rotary position embeddings in the language model. Defaults to 1.0.
    """

    def __init__(
        self,
        language_transformer_config: TransformerConfig,
        language_transformer_layer_spec: ModuleSpec,
        language_vocab_size: int,
        language_max_sequence_length: int,
        vision_transformer_config: TransformerConfig,
        vision_transformer_layer_spec: ModuleSpec,
        drop_vision_class_token: bool,
        vision_projection_config: TransformerConfig,
        vision_projection_layer_spec: ModuleSpec,
        vision_projection_type: str = "mlp",
        allow_missing_vision_projection_checkpoint: bool = False,
        parallel_output: bool = True,
        language_position_embedding_type: str = 'learned_absolute',
        language_rotary_percent: float = 1.0,
    ) -> None:
        super().__init__(config=language_transformer_config)

        logging.getLogger(__name__).warning(
            "LLaVA model is under development and may be missing features."
        )

        if parallel_state.get_pipeline_model_parallel_world_size() > 1:
            raise NotImplementedError("pipeline parallelism is not supported in this model yet.")

        self.language_model = GPTModel(
            config=language_transformer_config,
            transformer_layer_spec=language_transformer_layer_spec,
            vocab_size=language_vocab_size,
            max_sequence_length=language_max_sequence_length,
            parallel_output=parallel_output,
            position_embedding_type=language_position_embedding_type,
            rotary_percent=language_rotary_percent,
        )

        self.vision_model = CLIPViTModel(vision_transformer_config, vision_transformer_layer_spec)
        self._drop_vision_class_token = drop_vision_class_token

        # Map (intermediate) vision model outputs to the language model input dimension.
        self.vision_projection = MultimodalProjector(
            vision_projection_config,
            vision_projection_layer_spec,
            vision_projection_type,
            vision_transformer_config.hidden_size,  # input size to the projection.
        )

        # This allows ignoring missing weights for the vision projection during checkpoint loading.
        # This should be disabled by default but can be enabled if your checkpoint contains pretrained
        # vision and language models but not the projection from vision model outputs to language model inputs.
        if allow_missing_vision_projection_checkpoint:
            vision_projection_param_names = [
                f"vision_projection.{name}" for name in self.vision_projection.state_dict().keys()
            ]
            self.vision_projection.register_load_state_dict_post_hook(
                partial(_load_state_dict_hook_ignore_param_names, vision_projection_param_names)
            )

    def set_input_tensor(self, input_tensor: torch.Tensor) -> None:
        """Sets input tensor to the model.

        NOTE: Pipeline parallelism is not supported in this model yet. This is just a placeholder implementation.

        Args:
            input_tensor (Tensor): Sets the input tensor for the model.
        """
        self.vision_model.set_input_tensor(input_tensor)

    def freeze(
        self, freeze_language_model: bool, freeze_vision_model: bool, freeze_vision_projection: bool
    ):
        """Freeze model modules.

        Make specific modules non-trainable by setting requires_grad to False for the module's parameters.

        Args:
            freeze_language_model (bool): Freeze the language model module.
            freeze_vision_model (bool): Freeze the vision model module.
            freeze_vision_projection (bool): Freeze the vision projection module.
        """
        modules = []
        if freeze_language_model:
            modules.append(self.language_model)
        if freeze_vision_model:
            modules.append(self.vision_model)
        if freeze_vision_projection:
            modules.append(self.vision_projection)

        for module in modules:
            for param in module.parameters():
                param.requires_grad = False

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor = None,
        inference_params: InferenceParams = None,
    ) -> torch.Tensor:
        """Forward function of the LLaVA model.

        Args:
            images (torch.Tensor): input image of shape [batch, img_h, img_w].
            input_ids (torch.Tensor): input text ids [batch, text_seq_len].
            position_ids (torch.Tensor): input text position ids [batch, text_seq_len].
            attention_mask (torch.Tensor): attention mask for the language model [batch, 1, combined_seq_len, combined_seq_len].
            labels (torch.Tensor): Optional target text labels [batch, combined_seq_len].
            inference_params (InferenceParams): Inference-time parameters including KV cache.

        Returns:
            output (torch.Tensor): Loss of shape [b, s] if labels are provided, otherwise logits of shape [b, s, vocab_size].
        """
        language_embeddings = self.language_model.embedding(
            input_ids=input_ids, position_ids=position_ids
        )  # [text_seq_len, b, h_language]

        # If running inference, we can skip image token computation if they were computed already earlier for this sample.
        if (
            inference_params is not None
            and "image_tokens_count" in inference_params.key_value_memory_dict
        ):
            combined_embeddings = language_embeddings
        else:
            image_embeddings = self.vision_model(images)  # [b, img_seq_len, h_vision]

            if self._drop_vision_class_token:
                image_embeddings = image_embeddings[:, self.vision_model.class_token_len :, :]

            image_embeddings = image_embeddings.permute(1, 0, 2)  # [img_seq_len, b, h_vision]

            # map vision model output size to language model input size.
            image_embeddings = self.vision_projection(
                image_embeddings
            )  # [img_seq_len, b, h_vision]

            # If running inference, the language model KV cache will be updated for image token positions.
            # Here we store the image tokens sequence length, which can be used as an offset to the KV cache later.
            if inference_params is not None:
                inference_params.key_value_memory_dict[
                    "image_tokens_count"
                ] = image_embeddings.shape[0]

            combined_embeddings = torch.cat(
                [image_embeddings, language_embeddings], dim=0
            )  # [combined_seq_len, b, h_language]

        # Embedding is computed above so we can discard input and position ids.
        input_ids = None
        position_ids = None

        # Note: This returns loss if labels are provided, otherwise logits.
        output = self.language_model(
            input_ids,
            position_ids,
            attention_mask,
            decoder_input=combined_embeddings,
            labels=labels,
            inference_params=inference_params,
        )

        return output


def _load_state_dict_hook_ignore_param_names(
    param_names: List[str], module: torch.nn.Module, incompatible_keys: namedtuple
):
    """Hook to ignore missing keys during checkpoint loading.

    By default, this should not be used to avoid accidentally missing weights in checkpoint loading.

    Example use case: Use this for the vision projection if you want to load a checkpoint that contains vision and language model weights
    but not the vision projection weights.

    Args:
        param_names (list of str): Parameter names allowed to be missing when calling load_state_dict.
        module (torch.nn.Module): The torch module this hook applies to. Unused here but required by the torch API.
        incompatible_keys (namedtuple): Namedtuple with fields missing_keys and unexpected_keys, which collect the missing and unexpected
            keys when calling load_state_dict on this torch module, respectively.
    """
    for param_name in param_names:
        incompatible_keys.missing_keys.remove(param_name)
