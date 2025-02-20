# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
import logging
from collections import namedtuple
from functools import partial
from typing import List, Optional

import torch

from megatron.core import InferenceParams, tensor_parallel
from megatron.core.config_logger import has_config_logger_enabled, log_config_to_disk
from megatron.core.models.gpt import GPTModel
from megatron.core.models.vision.clip_vit_model import CLIPViTModel, get_num_image_embeddings
from megatron.core.models.vision.multimodal_projector import MultimodalProjector
from megatron.core.models.vision.radio import RADIOViTModel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.parallel_state import get_context_parallel_rank, get_context_parallel_world_size
from megatron.core.transformer import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import log_single_rank

try:
    import transformer_engine  # pylint: disable=unused-import

    from megatron.core.extensions.transformer_engine import TEDotProductAttention
    from megatron.core.utils import is_te_min_version

    HAVE_TE = True
    try:
        import transformer_engine_torch as tex

        HAVE_TEX = True
    except:
        HAVE_TEX = False
except:
    HAVE_TE = False
    if get_context_parallel_world_size() > 1:
        raise RuntimeError("ContextParallelism requires TransformerEngine support, but not found.")


IGNORE_INDEX = -100  # ID for labels that should be ignored.
# Image token index can be tokenizer dependent so the default value does not work in all cases.
DEFAULT_IMAGE_TOKEN_INDEX = -200
IMAGE_TOKEN = "<image>"
VIDEO_TOKEN = "<video>"


class _get_data_on_this_cp_rank(torch.autograd.Function):
    """Performs sharding for Context Parallelism in THD format

    In the forward pass, indices are selected for each CP rank and remaining tokens are dropped.
    In the backward pass, this class takes care of managing gradients for dropped tokens on each
    CP rank.
    """

    @staticmethod
    def forward(ctx, batch, packed_seq_params):
        """Context Parallelism forward support for THD format"""
        cp_size = get_context_parallel_world_size()
        cp_rank = get_context_parallel_rank()
        for key, data in batch.items():
            index = tex.thd_get_partitioned_indices(
                packed_seq_params.cu_seqlens_q_padded, data.size(1), cp_size, cp_rank
            )
            if key == "combined_embeddings":
                ctx.decoder_emb_index = index
                ctx.decoder_emb_seqlen = data.size(1)
            batch[key] = data.index_select(1, index)
            batch[key].requires_grad = data.requires_grad

        return batch

    @staticmethod
    def backward(ctx, grad_out, grad_label, grad_loss):
        """Context Parallelism backward support for THD format"""
        seqlen = ctx.decoder_emb_seqlen
        index = ctx.decoder_emb_index
        assert grad_out.size(1) == index.size(
            0
        ), f"Shape mismatch in incoming gradient {grad_out.shape} and \
                index from THD CP sharding {index.shape}"
        grad_in = torch.zeros(
            grad_out.size(0),
            seqlen,
            *grad_out.size()[2:],
            dtype=grad_out.dtype,
            device=grad_out.device,
        )
        grad_in[:, ctx.decoder_emb_index, :] = grad_out

        return (grad_in, None, None, None)


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
        share_embeddings_and_output_weights (bool): Input embedding and output layer share weights.
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
        language_rope_scaling (bool): Toggle RoPE scaling.
        language_rope_scaling_factor (float): RoPE scaling factor. Defaults to 8.
        image_token_index (int): Token ID for image token such as <image>.
        pixel_shuffle (bool): Enable pixel shuffle.
        tile_tags (list): Optional tile tags.
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
        share_embeddings_and_output_weights: bool = False,
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
        language_rope_scaling: bool = False,
        language_rope_scaling_factor: float = 8.0,
        image_token_index: int = DEFAULT_IMAGE_TOKEN_INDEX,
        pixel_shuffle: bool = False,
        tile_tags: Optional[list] = None,
    ) -> None:
        super().__init__(config=language_transformer_config)

        if has_config_logger_enabled(language_transformer_config):
            log_config_to_disk(language_transformer_config, locals(), prefix=type(self).__name__)

        log_single_rank(
            logging.getLogger(__name__),
            logging.WARNING,
            "LLaVA is work in progress. Features are missing and methods can change.",
        )

        self.pre_process = pre_process
        self.post_process = post_process
        self.add_encoder = add_encoder
        self.add_decoder = add_decoder

        self.encoder_hidden_state = None
        self.vision_model = None
        self.vision_projection = None
        self.language_model = None

        self.sequence_parallel_lm = language_transformer_config.sequence_parallel
        self.tp_comm_overlap_lm = language_transformer_config.tp_comm_overlap
        self.context_parallel_lm = language_transformer_config.context_parallel_size
        if self.sequence_parallel_lm or self.context_parallel_lm > 1:
            assert (
                language_transformer_layer_spec.submodules.self_attention.submodules.core_attention
                == TEDotProductAttention
                and HAVE_TE
            ), "Sequence/Context Parallelism is supported only with TE DotProductAttention."
            if self.context_parallel_lm > 1:
                assert is_te_min_version(
                    "1.10.0"
                ), "Context Parallelism in LLaVA requires TE v1.10 or higher"
        self.tensor_model_parallel_size_lm = language_transformer_config.tensor_model_parallel_size

        # This attribute is needed to check if an all-reduce is required
        # on the word embeddings inside `finalize_model_grads._allreduce_word_embedding_grads`.
        self.share_embeddings_and_output_weights = share_embeddings_and_output_weights
        if self.add_decoder:
            if hasattr(
                language_transformer_config, "language_model_type"
            ) and language_transformer_config.language_model_type.startswith("huggingface"):
                from megatron.core.models.huggingface.module import build_hf_model

                self.language_model = build_hf_model(language_transformer_config)
            else:
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
                    rope_scaling=language_rope_scaling,
                    scatter_embedding_sequence_parallel=False,
                )

                self.share_embeddings_and_output_weights = (
                    self.language_model.share_embeddings_and_output_weights
                )
            self._language_max_sequence_length = language_max_sequence_length
            self._language_is_pipeline_parallel = (
                language_transformer_config.pipeline_model_parallel_size > 1
            )

            # Newer Transformer Engine versions add _extra_state keys in state_dict when using FP8.
            # Older models may not have _extra_state and can be ignored.
            self.language_model.register_load_state_dict_post_hook(
                _load_state_dict_hook_ignore_extra_state
            )

        class_token_len = 1
        if self.add_encoder:
            self._drop_vision_class_token = drop_vision_class_token
            add_class_token = True
            if vision_transformer_config.vision_model_type.startswith(
                ("clip", "siglip", "internvit")
            ):
                if vision_transformer_config.vision_model_type == "siglip":
                    class_token_len = 0
                    add_class_token = False
                    error_msg = (
                        "Siglip does not support vision class token, "
                        "set disable-vision-class-token to False."
                    )
                    assert not self._drop_vision_class_token, error_msg
                self.vision_model = CLIPViTModel(
                    vision_transformer_config,
                    vision_transformer_layer_spec,
                    img_h=img_h,
                    img_w=img_w,
                    class_token_len=class_token_len,
                    patch_dim=patch_dim,
                    model_subtype=vision_transformer_config.vision_model_type,
                    add_class_token=add_class_token,
                )
            elif vision_transformer_config.vision_model_type in ("radio"):
                # TODO: should refactor into model code itself?
                class_token_len = 8
                max_img_h = 2048
                max_img_w = 2048
                embedder_bias = False
                use_mask_token = False
                self.vision_model = RADIOViTModel(
                    vision_transformer_config,
                    vision_transformer_layer_spec,
                    img_h=img_h,
                    img_w=img_w,
                    max_img_h=max_img_h,
                    max_img_w=max_img_w,
                    class_token_len=class_token_len,
                    patch_dim=patch_dim,
                    add_class_token=add_class_token,
                    embedder_bias=embedder_bias,
                    use_mask_token=use_mask_token,
                )
            elif vision_transformer_config.vision_model_type.startswith("huggingface"):
                from megatron.core.models.huggingface.module import build_hf_model

                self.vision_model = build_hf_model(vision_transformer_config)
            else:
                raise ValueError(
                    "Vision model "
                    f"{vision_transformer_config.vision_model_type} is not "
                    "supported."
                )

            self.vision_model.register_load_state_dict_post_hook(
                _load_state_dict_hook_ignore_extra_state
            )

            vision_projection_input_size = vision_transformer_config.hidden_size
            vision_projection_input_size *= 4 if pixel_shuffle else 1

            # Map (intermediate) vision model outputs to the language model input dimension.
            self.vision_projection = MultimodalProjector(
                vision_projection_config,
                vision_projection_layer_spec,
                vision_projection_type,
                vision_projection_input_size,
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

        self.img_seq_len = get_num_image_embeddings(
            img_h,
            img_w,
            patch_dim,
            vision_transformer_config.vision_model_type,
            drop_vision_class_token,
            class_token_len,
            pixel_shuffle,
            tile_tags is not None,  # Tile tags enabled/disabled.
        )

        self.image_token_index = image_token_index
        self._pixel_shuffle = pixel_shuffle
        self._tile_tags = tile_tags

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
        inference_params,
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
            return None, None, None

        # If using the inference KV cache, the image tokens are already computed.
        if use_inference_kv_cache:
            return language_embeddings, loss_mask, labels

        img_seq_len = self.img_seq_len
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
            # Pipeline parallel expects fixed input size. Check if we need to pad.
            if (
                self._language_is_pipeline_parallel
                and max_seq_len < self._language_max_sequence_length
                and inference_params is None
            ):
                max_seq_len = self._language_max_sequence_length

            batch_indices, non_image_indices = torch.where(image_token_mask != True)

            # New position ids for the text tokens, shifted by the image sequence length.
            # E.g. for input_ids = [-200, 1, 2, 3] and img_seq_len = 576, we get
            # new_position_ids = [576, 577, 578, 579]. text_position_ids are then [577, 578, 579].
            image_token_mask_lens = image_token_mask.int().clone()
            # -1 is for the removed image token index.
            image_token_mask_lens[image_token_mask] = num_image_tiles * img_seq_len - 1
            # +1 is needed here for the cumulative sum. -1 is adjusting for zero-based indexing.
            new_position_ids = torch.cumsum((image_token_mask_lens + 1), dim=-1) - 1
            text_position_ids = new_position_ids[batch_indices, non_image_indices]

            label_batch_indices = None  # dummy value to pass formatting
            # Labels are shifted to left by one.
            # So, shift text position ids and non-image indices to left by one.
            label_batch_indices = None
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
            # NOTE: FSDP can hang with text-only samples so we use a workaround to run a dummy image
            # through the vision model and then zero-out the impact of the output here.
            if num_image_tiles.shape[0] == 0 and image_embeddings.shape[0] > 0:
                assert images_mask.sum() == 0 and getattr(
                    self.vision_model, "_is_fsdp_managed_module", False
                ), "expected FSDP and dummy image"
                final_embedding[:1, :1, :1] += 0 * image_embeddings[:1, :1, :1]
            else:
                final_embedding[images_mask] = (
                    image_embeddings.permute(1, 0, 2).reshape(-1, embed_dim).contiguous()
                )

        # Create the final labels and loss mask (if this is the last language model stage).
        final_labels, final_loss_mask = None, None
        if self.post_process and has_labels:
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

        if final_embedding is not None and final_labels is not None:
            assert (
                final_embedding.shape[:2] == final_labels.shape == final_loss_mask.shape
            ), "unexpected shapes after data preprocessing"

        if final_embedding is not None:
            # Truncate if exceeding the language model's max sequence length.
            if final_embedding.shape[1] > self._language_max_sequence_length:
                final_embedding = final_embedding[:, : self._language_max_sequence_length]
            # Transpose to [s,b,h] only if not using CP because CP Sharding expects seq in dim=1
            if self.context_parallel_lm == 1:
                final_embedding = final_embedding.transpose(1, 0).contiguous()

        truncate_labels = (
            final_labels is not None and final_labels.shape[1] > self._language_max_sequence_length
        )
        if truncate_labels:
            final_labels = final_labels[:, : self._language_max_sequence_length]
            final_loss_mask = final_loss_mask[:, : self._language_max_sequence_length]

        return final_embedding, final_labels, final_loss_mask

    def _process_embedding_token_parallel(
        self, combined_embeddings, new_labels, new_loss_mask, packed_seq_params
    ):
        """Processes the input data for model parallelism support.

        When using sequence parallelism (SP) or context parallelism (CP), the sequence is sharded
        across different GPUs. This function performs the sharding and distributes the sequence
        across GPUs for SP and CP

        Context Parallelism is a feature that helps improve memory efficiency for
        long sequence training by distributing sequence across CP ranks.
        It requires token length to be divisible by (CP size *2) to ensure proper load balance.

        Sequence Parallelism is a feature that helps improve memory efficiency for
        long sequence training by distributing sequence across TP ranks.
        It requires token length to be divisible by TP size.

        Returns:
            combined_embeddings (torch.Tensor): image and text embeddings combined and distributed.
            new_labels (torch.Tensor): Distributed labels for image and text positions.
            new_loss_mask (torch.Tensor): Distributed loss mask.
            packed_seq_params (PackedSeqParams): Dict with padded token information.

        """

        # No pre or post processing needed with PP middle chunks.
        if not self.pre_process and not self.post_process:
            return combined_embeddings, new_labels, new_loss_mask, packed_seq_params

        shard_factor = seq_dim = None
        if self.pre_process:
            if self.context_parallel_lm > 1 and self.sequence_parallel_lm:
                shard_factor = self.tensor_model_parallel_size_lm * self.context_parallel_lm * 2
                seq_dim = 1
            elif self.context_parallel_lm > 1:
                shard_factor = self.context_parallel_lm * 2
                seq_dim = 1
            elif self.sequence_parallel_lm:
                shard_factor = self.tensor_model_parallel_size_lm
                seq_dim = 0

            assert (
                combined_embeddings.shape[seq_dim] % shard_factor == 0
            ), f"Sequence length should be divisible by {shard_factor} for \
                Sequence/Context parallelism"
            if self.sequence_parallel_lm and self.tp_comm_overlap_lm:
                assert (
                    combined_embeddings.shape[seq_dim] == self._language_max_sequence_length
                ), f"TP Comm overlap either requires Vision+Text token length \
                == language_max_sequence_length"

        if self.context_parallel_lm > 1:
            batch = dict()
            if self.pre_process:
                batch["combined_embeddings"] = combined_embeddings
            if self.post_process:
                batch["new_labels"] = new_labels
                batch["new_loss_mask"] = new_loss_mask
            # Distribute sequence across CP ranks
            if packed_seq_params is None or packed_seq_params.qkv_format == 'sbhd':
                from megatron.training.utils import get_batch_on_this_cp_rank

                batch = get_batch_on_this_cp_rank(batch)
            else:
                assert HAVE_TEX and is_te_min_version(
                    "1.10.0"
                ), "Please update Transformer Engine to >= 1.10 to use \
                    Context Parallel with THD format data"
                batch = _get_data_on_this_cp_rank.apply(batch, packed_seq_params)

            if self.pre_process:
                combined_embeddings = batch["combined_embeddings"]  # [B, S/CP, H]
                combined_embeddings = combined_embeddings.transpose(
                    1, 0
                ).contiguous()  # [B,S/CP,H] -> [S/CP,B,H]
            if self.post_process:
                new_labels = batch["new_labels"]
                new_loss_mask = batch["new_loss_mask"]

        if self.sequence_parallel_lm and self.pre_process:
            combined_embeddings = tensor_parallel.scatter_to_sequence_parallel_region(
                combined_embeddings
            )  # [S/(CP*TP),B,H]

        return combined_embeddings, new_labels, new_loss_mask, packed_seq_params

    def _apply_tile_tagging(self, image_embeddings, num_image_tiles):
        """Apply tile tagging.

        The image embeddings of multiple tiles are prepended with tile tags such as <tile_1>.
        This implements the method used in NVLM https://arxiv.org/pdf/2409.11402.

        Args:
            image_embeddings (torch.Tensor): [img_seq_len, num_tiles, h_language].
            num_image_tiles (torch.Tensor): Number of tiles for each input image [num_images].

        Returns:
            torch.Tensor: Tile tags prepended to image embeddings.
                [tile_seq_len (=5) + img_seq_len, num_tiles, h_language]
        """
        assert (
            num_image_tiles.shape[0] == 1 and len(num_image_tiles) == 1
        ), "multiple input images are not supported yet."

        num_tiles = num_image_tiles[0].item()
        tile_tags = self._tile_tags[: num_tiles - 1] + [self._tile_tags[-1]]

        # [num_tiles, tile_seq_len (=5)]
        tile_tag_input_ids = torch.tensor(
            tile_tags, dtype=torch.int64, device=num_image_tiles.device
        )

        # [tile_seq_len, num_tiles, h_language]
        tile_tag_embeds = self.language_model.embedding(tile_tag_input_ids, position_ids=None)

        # [num_tiles, dim] should be the same same
        assert tile_tag_embeds.shape[1:] == image_embeddings.shape[1:]

        image_embeddings = torch.cat([tile_tag_embeds, image_embeddings])

        return image_embeddings  # [tile_seq_len + img_seq_len, num_tiles, h_language]

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
        image_token_index: Optional[int] = None,
        runtime_gather_output: Optional[bool] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
    ) -> torch.Tensor:
        """Forward function of the LLaVA model.

        Args:
            images (torch.Tensor): input images of shape [num_tiles, img_h, img_w].
                num_tiles means the number of image tiles in this batch.
                num_tiles = 0 if the batch doesn't contain images.
            input_ids (torch.Tensor): input text ids [batch, text_seq_len].
            position_ids (torch.Tensor): input text position ids [batch, text_seq_len].
            attention_mask (torch.Tensor): Language model attention mask
                [batch, 1, 1, combined_seq_len]. NOTE: attention_mask is typically None and
                attn_mask_type in layer specs determines the attention mask used.
            labels (torch.Tensor): Optional target text labels [batch, combined_seq_len].
            loss_mask (torch.Tensor): Text loss mask [batch, text_seq_len].
            inference_params (InferenceParams): Inference-time parameters including KV cache.
            num_image_tiles (list of int): Number of tiles per image. Default 1 tile per image.
            image_token_index (int): ID for input images. Default None means `image_token_index`
                arg in the constructor will be used.
            runtime_gather_output (bool): Gather output at runtime. Default None means
                `parallel_output` arg in the constructor will be used.
            packed_seq_params (PackedSeqParams): 1) If using sequence packing, must contain
                subsample length information. 2) If using SP/CP with padding mask type,
                must contain padded token information.

        Returns:
            output (torch.Tensor): Loss of shape [b, s] if labels are provided,
                otherwise logits of shape [b, s, vocab_size].
            loss_mask (torch.Tensor): Loss mask expanded to combined sequence length. Shape [b, s].
        """
        use_inference_kv_cache = (
            inference_params is not None
            and "image_tokens_count" in inference_params.key_value_memory_dict
        )
        has_images = images is not None and images.shape[0] > 0

        # If running inference, we can skip image token computation
        # if they were computed already earlier for this sample.
        if use_inference_kv_cache:
            image_embeddings = None
        elif self.add_encoder and not has_images:
            # If no images provided, use an empty image embeddings tensor.
            image_embeddings = torch.tensor([], dtype=images.dtype, device=images.device).reshape(
                0, 0, 0
            )
        elif self.add_encoder and has_images:
            image_embeddings = self.vision_model(images)  # [num_tiles, img_seq_len, h_vision]
            if self._drop_vision_class_token:
                image_embeddings = image_embeddings[:, self.vision_model.class_token_len :, :]

            if self._pixel_shuffle:
                image_embeddings = pixel_shuffle(
                    image_embeddings
                )  # [num_tiles, img_seq_len_shuffled, h_vision_shuffled]

            # contiguous() required as `permute` can sparsify the tensor and this breaks pipelining
            image_embeddings = image_embeddings.permute(
                1, 0, 2
            ).contiguous()  # [img_seq_len, num_tiles, h_vision]

            # map vision model output size to language model input size.
            image_embeddings = self.vision_projection(
                image_embeddings
            )  # [img_seq_len, num_tiles, h_language]

            # Apply tile tagging if enabled and an image token is present.
            if self._tile_tags is not None and torch.any(input_ids == self.image_token_index):
                image_embeddings = self._apply_tile_tagging(image_embeddings, num_image_tiles)

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
            input_ids_text[input_ids_text == self.image_token_index] = 0
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
        if num_image_tiles is None and images is not None:
            num_image_tiles = torch.ones(images.shape[0], dtype=torch.int, device=input_ids.device)

        combined_embeddings, new_labels, new_loss_mask = self._preprocess_data(
            image_embeddings,
            language_embeddings,
            input_ids,
            loss_mask,
            labels,
            use_inference_kv_cache,
            inference_params,
            image_token_index if image_token_index is not None else self.image_token_index,
            num_image_tiles,
        )  # [combined_seq_len, b, h_language], [b, combined_seq_len], [b, combined_seq_len]

        if self.context_parallel_lm > 1 or self.sequence_parallel_lm:
            combined_embeddings, new_labels, new_loss_mask, packed_seq_params = (
                self._process_embedding_token_parallel(
                    combined_embeddings, new_labels, new_loss_mask, packed_seq_params
                )
            )

        output = self.language_model(
            input_ids=None,
            position_ids=None,
            attention_mask=attention_mask,
            decoder_input=combined_embeddings,
            labels=new_labels,
            inference_params=inference_params,
            runtime_gather_output=runtime_gather_output,
            packed_seq_params=packed_seq_params,
        )

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


def _load_state_dict_hook_ignore_extra_state(
    module: torch.nn.Module, incompatible_keys: namedtuple
):
    """Hook to ignore Transformer Engine _extra_state used for FP8.

    This is for backwards-compatibility. Newer TE versions add _extra_state keys to the state dict,
    while older models might not have those keys. Those keys can be ignored when not using FP8.

    Args:
        module (torch.nn.Module): The torch module this hook applies to. Required by the torch API.
        incompatible_keys (namedtuple): Namedtuple with fields missing_keys and unexpected_keys,
            which collect the missing and unexpected keys, respectively.
    """
    for name, keys in incompatible_keys._asdict().items():
        for key in keys[::-1]:
            if "extra_state" in key:
                logging.getLogger(__name__).warning(
                    f"_extra_state key {key} being removed from {name}"
                )
                keys.remove(key)


# pylint: disable-next=line-too-long
# Based on https://github.com/OpenGVLab/InternVL/blob/c7c5af1a8930b4862afe8ed14672307082ef61fa/internvl_chat/internvl/model/internvl_chat/modeling_internvl_chat.py#L218
# Copyright (c) 2023 OpenGVLab.
def pixel_shuffle(x, scale_factor=0.5, version=2):
    """Pixel shuffle based on InternVL but adapted for our use case.

    Args:
        x (torch.Tensor): Vision model outputs [num_tiles, img_seq_len, h_vision]
        version (int): Implementation version.

    Returns:
        Shuffled vision model outputs [num_tiles, (sq ** 2) * (scale ** 2), h_vision / (scale ** 2)]
    """
    h = w = int(x.shape[1] ** 0.5)  # sq
    x = x.reshape(x.shape[0], h, w, -1)  # [num_tiles, sq, sq, h_vision]

    n, w, h, c = x.size()
    # N, W, H, C --> N, W, H * scale, C // scale
    x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
    # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
    x = x.permute(0, 2, 1, 3).contiguous()
    # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
    x = x.view(
        n, int(h * scale_factor), int(w * scale_factor), int(c / (scale_factor * scale_factor))
    )

    if version == 2:
        x = x.permute(0, 2, 1, 3).contiguous()

    x = x.reshape(x.shape[0], -1, x.shape[-1])

    return x
