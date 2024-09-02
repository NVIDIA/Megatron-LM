# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
import logging
from collections import namedtuple
from functools import partial
from typing import List, Optional

import torch

from megatron.core import InferenceParams
from megatron.core.config_logger import has_config_logger_enabled, log_config_to_disk
from megatron.core.models.gpt import GPTModel
from megatron.core.models.vision.clip_vit_model import CLIPViTModel, get_image_sequence_length
from megatron.core.models.vision.multimodal_projector import MultimodalProjector
from megatron.core.transformer import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig

IMAGE_TOKEN_INDEX = -200  # ID for images in the input sequence.
IGNORE_INDEX = -100  # ID for labels that should be ignored.


# Note: This is under development and may be missing features.
class LLaVAModel(MegatronModule):
    """LLaVA multi-modal model.

    Args:
        language_transformer_config (TransformerConfig): Transformer config for the language model.
        language_transformer_layer_spec (ModuleSpec): Language model spec.
        language_vocab_size (int): Language model vocabulary size.
        language_max_sequence_length (int): Language model maximum sequence length.
        vision_transformer_config (TransformerConfig): Transformer config for the vision model.
        vision_transformer_layer_spec (ModuleSpec): Vision model spec.
        drop_vision_class_token (bool): Drop vision class token(s) before the language model.
        vision_projection_config (TransformerConfig): Vision projection config.
        vision_projection_layer_spec (ModuleSpec): Vision projection spec.
        vision_projection_type (str): Type of the vision projection. Default: 2-layer MLP.
        allow_missing_vision_projection_checkpoint (bool): Allow vision projection weights to be
            missing when loading a checkpoint. Default False.
        parallel_output (bool): Keep outputs split across tensor parallel ranks.
            This is typically True for training and False for inference.
        language_position_embedding_type (str): Language model position embedding type.
        language_rotary_percent (float): RoPE percent. Defaults to 1.0.
        pre_process (bool): Include embedding layer in the decoder (used with pipeline parallel).
        post_process (bool): Include output layer in the decoder (used with pipeline parallel).
        add_encoder (bool): Construct the encoder (used with pipeline parallel).
            When we use pipelining, the encoder will live on only the first stage
        add_decoder (bool): Construct the decoder (used with pipeline parallel).
            When we use pipelining, the decoder will live on every stage after the first one.
        img_h (int): Input image height.
        img_w (int): Input image width.
        patch_dim (int): The size of each image patch side.
        language_rotary_base (int): RoPE base.
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
        pre_process: bool = True,
        post_process: bool = True,
        add_encoder: bool = True,
        add_decoder: bool = True,
        img_h: int = 336,
        img_w: int = 336,
        patch_dim: int = 14,
        language_rotary_base: int = 10000,
    ) -> None:
        super().__init__(config=language_transformer_config)

        if has_config_logger_enabled(language_transformer_config):
            log_config_to_disk(language_transformer_config, locals(), prefix=type(self).__name__)

        logging.getLogger(__name__).warning(
            "LLaVA model is under active development. "
            "It may be missing features and its methods may change."
        )

        self.pre_process = pre_process
        self.post_process = post_process
        self.add_encoder = add_encoder
        self.add_decoder = add_decoder

        self.encoder_hidden_state = None
        self.vision_model = None
        self.vision_projection = None
        self.language_model = None

        # This attribute is needed to check if an all-reduce is required
        # on the word embeddings inside `finalize_model_grads._allreduce_word_embedding_grads`.
        self.share_embeddings_and_output_weights = False
        if self.add_decoder:
            self.language_model = GPTModel(
                config=language_transformer_config,
                transformer_layer_spec=language_transformer_layer_spec,
                vocab_size=language_vocab_size,
                max_sequence_length=language_max_sequence_length,
                parallel_output=parallel_output,
                position_embedding_type=language_position_embedding_type,
                rotary_percent=language_rotary_percent,
                pre_process=self.pre_process,
                post_process=self.post_process,
                rotary_base=language_rotary_base,
            )
            self.share_embeddings_and_output_weights = (
                self.language_model.share_embeddings_and_output_weights
            )
            self._language_max_sequence_length = language_max_sequence_length

        class_token_len = 1
        if self.add_encoder:
            self.vision_model = CLIPViTModel(
                vision_transformer_config,
                vision_transformer_layer_spec,
                img_h=img_h,
                img_w=img_w,
                class_token_len=class_token_len,
                patch_dim=patch_dim,
            )
            self._drop_vision_class_token = drop_vision_class_token
            # Map (intermediate) vision model outputs to the language model input dimension.
            self.vision_projection = MultimodalProjector(
                vision_projection_config,
                vision_projection_layer_spec,
                vision_projection_type,
                vision_transformer_config.hidden_size,  # input size to the projection.
            )
            # Ignore missing weights for the vision projection during checkpoint loading.
            # This should be disabled by default but can be enabled if your checkpoint contains
            # pretrained vision and language models but not the projection from vision model
            # outputs to language model inputs.
            if allow_missing_vision_projection_checkpoint:
                vision_projection_param_names = [
                    f"vision_projection.{name}"
                    for name in self.vision_projection.state_dict().keys()
                ]
                self.vision_projection.register_load_state_dict_post_hook(
                    partial(_load_state_dict_hook_ignore_param_names, vision_projection_param_names)
                )

        self._img_seq_len = get_image_sequence_length(
            img_h, img_w, patch_dim, not drop_vision_class_token, class_token_len
        )

    def shared_embedding_or_output_weight(self):
        """This is a convenience method to surface the language model's word embeddings, which is
        necessary for `finalize_model_grads._allreduce_word_embedding_grads`."""
        if self.add_decoder:
            return self.language_model.shared_embedding_or_output_weight()
        return None

    def set_input_tensor(self, input_tensor) -> None:
        """Set model chunk input tensor."""
        # This is usually handled in schedules.py but some inference code still
        # gives us non-lists or None
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]
        assert len(input_tensor) == 1, 'input_tensor should only be length 1 for llava'

        if self.add_encoder and self.add_decoder:
            self.vision_model.set_input_tensor(input_tensor[0])
        elif self.add_encoder:
            self.vision_model.set_input_tensor(input_tensor[0])
        elif self.pre_process:
            self.encoder_hidden_state = input_tensor[0]
        else:
            self.language_model.set_input_tensor(input_tensor[0])

    def freeze(
        self, freeze_language_model: bool, freeze_vision_model: bool, freeze_vision_projection: bool
    ):
        """Freeze model modules.

        Make specific modules non-trainable by setting requires_grad to False.

        Args:
            freeze_language_model (bool): Freeze the language model module.
            freeze_vision_model (bool): Freeze the vision model module.
            freeze_vision_projection (bool): Freeze the vision projection module.
        """
        modules = []
        if freeze_language_model and self.language_model is not None:
            modules.append(self.language_model)
        if freeze_vision_model and self.vision_model is not None:
            modules.append(self.vision_model)
        if freeze_vision_projection and self.vision_projection is not None:
            modules.append(self.vision_projection)

        for module in modules:
            for param in module.parameters():
                param.requires_grad = False

    def _preprocess_data(
        self,
        image_embeddings,
        language_embeddings,
        input_ids,
        loss_mask,
        labels,
        use_inference_kv_cache,
        image_token_index,
        num_image_tiles,
    ):
        """Preprocess input data before input to language model.

        This function is adopted from
        https://github.com/huggingface/transformers/blob/85817d98fb60977c97e3014196a462b732d2ed1a/src/transformers/models/llava_next/modeling_llava_next.py#L409
        for our input data conventions.

        image_token_index = -200 indicates the image position in the input_ids = [0, 1, -200, 2, 3]
        and labels = [1, -200, 2, 3, 4], for example.
        We want to replace the image position (-200) with image_embeddings and return the following:
        - final_embeddings = [0, 1, image_embeddings, 2, 3],
        - final_labels = [1, -100, 2, 3, 4]
        - final_loss_mask = [1, 0, 0, 1, 1]

        This function handles samples without images (text-only sample). It also handles samples
        with images that are split into multiples tiles.

        If pipeline parallelism is not used, then self.pre_process and self.post_process
        are both True and we update both input embeddings, labels and loss masks (if available).

        If pipeline parallelism is used, then we do the following
        - the first language model chunk has self.pre_process = True and
          self.post_process = False. We update input embeddings.
        - the middle language model chunk(s) has self.pre_process = False and
          self.post_process = False. We don't need to update anything.
        - the last language model chunk has self.pre_process = False and
          self.post_process = True. We update labels and loss mask.

        TODO: This function should adjust the attention mask too.
        Currently, we assume the language model uses a causal mask.

        Returns:
            final_embedding (torch.Tensor): image and text embeddings [combined_seq_len, b, h].
            final_labels (torch.Tensor): labels for image and text positions [b, combined_seq_len].
            final_loss_mask (torch.Tensor): loss mask [b, combined_seq_len].
        """
        assert self.add_decoder, "input text preprocessing is only needed for the language model"

        # No pre- or postprocessing needed.
        # With pipeline parallel > 2, this means a chunk in the middle of the model.
        if not self.pre_process and not self.post_process:
            return language_embeddings, loss_mask, labels

        # If using the inference KV cache, the image tokens are already computed.
        if use_inference_kv_cache:
            return language_embeddings, loss_mask, labels

        img_seq_len = self._img_seq_len
        batch_size, text_seq_len = input_ids.shape

        has_labels = labels is not None
        if has_labels:
            assert (
                labels.shape == loss_mask.shape
            ), f"mismatching labels shape {labels.shape} and loss mask shape {loss_mask.shape}"

        # Create indices for new text and label positions.
        with torch.no_grad():
            image_token_mask = input_ids == image_token_index
            num_images_per_sample = torch.sum(image_token_mask, dim=-1)

            # Number of tiles per sample.
            num_image_tiles_batch = num_image_tiles.split(num_images_per_sample.tolist(), dim=0)
            num_image_tiles_batch = torch.tensor(
                [x.sum() for x in num_image_tiles_batch], device=input_ids.device
            )

            # Sequence length for each sample is the image sequence length multiplied by
            # the number of tiles for that image, minus image token indices,
            # plus text sequence length.
            seq_lens = num_image_tiles_batch * img_seq_len - num_images_per_sample + text_seq_len
            max_seq_len = seq_lens.max()
            batch_indices, non_image_indices = torch.where(input_ids != image_token_index)

            # New position ids for the text tokens, shifted by the image sequence length.
            # E.g. for input_ids = [-200, 1, 2, 3] and img_seq_len = 576, we get
            # new_position_ids = [576, 577, 578, 579]. text_position_ids are then [577, 578, 579].
            image_token_mask_lens = image_token_mask.int().clone()
            # -1 is for the removed image token index.
            image_token_mask_lens[image_token_mask] = num_image_tiles * img_seq_len - 1
            # +1 is needed here for the cumulative sum. -1 is adjusting for zero-based indexing.
            new_position_ids = torch.cumsum((image_token_mask_lens + 1), dim=-1) - 1
            text_position_ids = new_position_ids[batch_indices, non_image_indices]

            # Labels are shifted to left by one.
            # So, shift text position ids and non-image indices to left by one.
            if has_labels:
                label_text_position_ids = text_position_ids - 1
                valid_label_text_position_ids = label_text_position_ids >= 0
                label_text_position_ids = label_text_position_ids[valid_label_text_position_ids]

                label_batch_indices = batch_indices[valid_label_text_position_ids]

                label_non_image_indices = non_image_indices - 1
                valid_label_non_image_indices = label_non_image_indices >= 0
                label_non_image_indices = label_non_image_indices[valid_label_non_image_indices]

            # Create a mask for the image embedding positions.
            images_mask = torch.full(
                (batch_size, max_seq_len), True, dtype=torch.bool, device=input_ids.device
            )
            # No images in the text positions.
            images_mask[batch_indices, text_position_ids] = False
            # Samples can have different amount of images tokens.
            # new_position_ids[:, -1] gives the last text position id for each sample.
            # Padding is needed when the number of image tokens differs.
            first_padding_idx = new_position_ids[:, -1] + 1
            images_mask[
                torch.arange(max_seq_len, device=first_padding_idx.device).repeat(batch_size, 1)
                >= first_padding_idx.unsqueeze(1)
            ] = False

        # Create the final input embedding (if this is the first language model stage).
        final_embedding = None
        if self.pre_process:
            embed_dim = language_embeddings.shape[-1]
            final_embedding = torch.zeros(
                batch_size,
                max_seq_len,
                embed_dim,
                dtype=language_embeddings.dtype,
                device=language_embeddings.device,
            )

            # Put text embeddings to the text positions in the result tensor.
            final_embedding[batch_indices, text_position_ids] = language_embeddings[
                batch_indices, non_image_indices
            ]

            # Put image embeddings to image positions.
            final_embedding[images_mask] = image_embeddings.reshape(-1, embed_dim).contiguous()

        # Create the final labels and loss mask (if this is the last language model stage).
        final_labels, final_loss_mask = None, None
        if has_labels:
            final_labels = torch.full(
                (batch_size, max_seq_len), IGNORE_INDEX, dtype=labels.dtype, device=labels.device
            )
            final_loss_mask = torch.full(
                (batch_size, max_seq_len), 0, dtype=loss_mask.dtype, device=loss_mask.device
            )

            # Put text labels and loss mask to the text positions.
            final_labels[label_batch_indices, label_text_position_ids] = labels[
                label_batch_indices, label_non_image_indices
            ]

            final_loss_mask[batch_indices, text_position_ids] = loss_mask[
                batch_indices, non_image_indices
            ]

            # For labels, pick the last label index that got dropped by the shift to left.
            label_extra_text_position_ids = seq_lens - 1
            batch_range = torch.arange(len(label_extra_text_position_ids))
            final_labels[batch_range, label_extra_text_position_ids] = labels[batch_range, -1]

            # Loss mask the image positions.
            final_loss_mask[images_mask] = 0

            # Loss mask last text position just before an image
            # so that text token does not need to predict the first image token.
            batch_image_indices, image_indices = torch.where(image_token_mask)
            # Indices just before image tokens. If it's -1, skip it.
            before_image_indices = image_indices - 1
            valid = before_image_indices >= 0
            valid_batch_image_indices = batch_image_indices[valid]
            valid_before_image_indices = before_image_indices[valid]
            # Map those indices those position ids.
            valid_before_image_indices = new_position_ids[
                valid_batch_image_indices, valid_before_image_indices
            ]

            final_loss_mask[valid_batch_image_indices, valid_before_image_indices] = 0

        if final_embedding is not None and has_labels:
            assert (
                final_embedding.shape[:2] == final_labels.shape == final_loss_mask.shape
            ), "unexpected shapes after data preprocessing"

        if final_embedding is not None:
            final_embedding = final_embedding.transpose(1, 0).contiguous()

        # Truncate if exceeding the language model's max sequence length.
        if (
            final_embedding is not None
            and final_embedding.shape[0] > self._language_max_sequence_length
        ):
            final_embedding = final_embedding[: self._language_max_sequence_length]

        if has_labels and final_labels.shape[1] > self._language_max_sequence_length:
            final_labels = final_labels[:, : self._language_max_sequence_length]
            final_loss_mask = final_loss_mask[:, : self._language_max_sequence_length]

        return final_embedding, final_labels, final_loss_mask

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        loss_mask: Optional[torch.Tensor] = None,
        inference_params: Optional[InferenceParams] = None,
        num_image_tiles: Optional[List[int]] = None,
        image_token_index: Optional[int] = IMAGE_TOKEN_INDEX,
    ) -> torch.Tensor:
        """Forward function of the LLaVA model.

        Args:
            images (torch.Tensor): input images of shape [num_tiles, img_h, img_w].
                num_tiles means the number of image tiles in this batch.
                num_tiles = 0 if the batch doesn't contain images.
            input_ids (torch.Tensor): input text ids [batch, text_seq_len].
            position_ids (torch.Tensor): input text position ids [batch, text_seq_len].
            attention_mask (torch.Tensor): Language model attention mask
                [batch, 1, combined_seq_len, combined_seq_len].
            labels (torch.Tensor): Optional target text labels [batch, combined_seq_len].
            loss_mask (torch.Tensor): Text loss mask [batch, text_seq_len].
            inference_params (InferenceParams): Inference-time parameters including KV cache.
            num_image_tiles (list of int): Number of tiles per image. Default 1 tile per image.
            image_token_index (int): ID for input images.

        Returns:
            output (torch.Tensor): Loss of shape [b, s] if labels are provided,
                otherwise logits of shape [b, s, vocab_size].
            loss_mask (torch.Tensor): Loss mask expanded to combined sequence length. Shape [b, s].
        """
        use_inference_kv_cache = (
            inference_params is not None
            and "image_tokens_count" in inference_params.key_value_memory_dict
        )
        has_images = images.shape[0] > 0

        # If running inference, we can skip image token computation
        # if they were computed already earlier for this sample.
        if use_inference_kv_cache:
            image_embeddings = None
        elif self.add_encoder and not has_images:
            # If no images provided, use an empty image embeddings tensor.
            image_embeddings = torch.tensor([], dtype=images.dtype, device=images.device)
        elif self.add_encoder and has_images:
            image_embeddings = self.vision_model(images)  # [num_tiles, img_seq_len, h_vision]
            if self._drop_vision_class_token:
                image_embeddings = image_embeddings[:, self.vision_model.class_token_len :, :]
            # contiguous() required as `permute` can sparsify the tensor and this breaks pipelining
            image_embeddings = image_embeddings.permute(
                1, 0, 2
            ).contiguous()  # [img_seq_len, num_tiles, h_vision]

            # map vision model output size to language model input size.
            image_embeddings = self.vision_projection(
                image_embeddings
            )  # [img_seq_len, num_tiles, h_language]

            # TODO: Support batched inference.
            # In inference, the language model KV cache will be updated for image token positions.
            # Store the image tokens sequence length to be used as an offset to the KV cache later.
            if inference_params is not None:
                inference_params.key_value_memory_dict["image_tokens_count"] = (
                    image_embeddings.shape[0] * image_embeddings.shape[1]
                )
        else:
            image_embeddings = self.encoder_hidden_state

        if not self.add_decoder:
            return image_embeddings, loss_mask

        language_embeddings = None
        if self.pre_process:
            input_ids_text = input_ids.clone()
            input_ids_text[input_ids_text == image_token_index] = 0
            # Note: This adds absolute position embedding but not RoPE.
            # Each image is counted as one position.
            # RoPE is added in language_model forward. Each image embedding is one position.
            language_embeddings = self.language_model.embedding(
                input_ids=input_ids_text, position_ids=position_ids
            )  # [text_seq_len, b, h_language]
            language_embeddings = language_embeddings.transpose(
                1, 0
            ).contiguous()  # [b, text_seq_len, h_language]

        # Assume 1 tile per image if the number of tiles is not provided.
        if num_image_tiles is None:
            num_image_tiles = torch.ones(images.shape[0], dtype=torch.int, device=input_ids.device)

        # Preprocess input, labels and loss mask.
        combined_embeddings, new_labels, new_loss_mask = self._preprocess_data(
            image_embeddings,
            language_embeddings,
            input_ids,
            loss_mask,
            labels,
            use_inference_kv_cache,
            image_token_index,
            num_image_tiles,
        )  # [combined_seq_len, b, h_language], [b, combined_seq_len], [b, combined_seq_len]

        output = self.language_model(
            input_ids=None,
            position_ids=None,
            attention_mask=attention_mask,
            decoder_input=combined_embeddings,
            labels=new_labels,
            inference_params=inference_params,
        )

        if labels is None or loss_mask is None:
            return output

        return output, new_loss_mask


def _load_state_dict_hook_ignore_param_names(
    param_names: List[str], module: torch.nn.Module, incompatible_keys: namedtuple
):
    """Hook to ignore missing keys during checkpoint loading.

    By default, this should not be used to avoid accidentally missing weights in checkpoint loading.

    Example use case: Use this if you want to load a checkpoint that contains vision and language
    model weights but not the vision projection weights.

    Args:
        param_names (list str): Parameter names allowed to be missing when calling load_state_dict.
        module (torch.nn.Module): The torch module this hook applies to. Required by the torch API.
        incompatible_keys (namedtuple): Namedtuple with fields missing_keys and unexpected_keys,
            which collect the missing and unexpected keys, respectively.
    """
    for param_name in param_names:
        if param_name in incompatible_keys.missing_keys:
            logging.getLogger(__name__).warning(
                f"{param_name} being removed from incompatible_keys.missing_keys in LlavaModel"
            )
            incompatible_keys.missing_keys.remove(param_name)
