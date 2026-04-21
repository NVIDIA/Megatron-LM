# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
import copy
import logging
from collections import namedtuple
from copy import deepcopy
from functools import partial
import os
from typing import List, Optional, Tuple

import torch

from megatron.core import tensor_parallel
from megatron.core.config_logger import has_config_logger_enabled, log_config_to_disk
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.models.gpt import GPTModel
from megatron.core.models.mamba import MambaModel
from megatron.core.models.multimodal.context_parallel import (
    gather_from_context_parallel_ranks,
    split_to_context_parallel_ranks,
    get_padding,
    split_to_context_parallel_ranks_dynamic_res,
    gather_from_context_parallel_ranks_dynamic_res,
)
from megatron.core.models.multimodal.efficient_video_sampling import EVSVariant
from megatron.core.models.vision.clip_vit_model import CLIPViTModel, get_num_image_embeddings
from megatron.core.models.vision.conv_merging import ConvTokenMerge
from megatron.core.models.vision.multimodal_projector import MultimodalProjector
from megatron.core.models.vision.radio import RADIOViTModel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.parallel_state import get_context_parallel_group
from megatron.core.parallel_state import get_tensor_model_parallel_rank
from megatron.core.transformer import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.heterogeneous.heterogeneous_config import HeterogeneousTransformerConfig
from megatron.core.utils import deprecate_inference_params, log_single_rank

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


IGNORE_INDEX = -100  # ID for labels that should be ignored.
# Image token index can be tokenizer dependent so the default value does not work in all cases.
DEFAULT_IMAGE_TOKEN_INDEX = -200
DEFAULT_SOUND_TOKEN_INDEX = -300
IMAGE_TOKEN = "<image>"
VIDEO_TOKEN = "<video>"
SOUND_TOKEN = "<so_embedding>"


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
        hybrid_attention_ratio (float): Ratio of attention heads in hybrid attention.
        hybrid_mlp_ratio (float): Ratio of MLP heads in hybrid attention.
        hybrid_override_pattern (str): Pattern for hybrid attention override.
        fp16_lm_cross_entropy (bool): Use FP16 for language model cross-entropy loss.
        image_token_index (int): Token ID for image token such as <image>.
        pixel_shuffle (bool): Enable pixel shuffle.
        tile_tags (list): Optional tile tags.
        cp_group (torch.distributed.ProcessGroup): Process group for context parallelism.
        max_num_tiles (int): Maximum number of tiles per image.
        tokenizer_type (str): Tokenizer type.
        dynamic_resolution (bool): Enable dynamic resolution.
        image_break_token (str): Token for image break.
        conv_merging (bool): Enable conv merging.
        allow_missing_conv_merge_checkpoint (bool): Allow missing conv merge checkpoint.
        force_eval_mode: (bool): Force RADIO to stay in eval mode, optional for pre-training. Defaults to False.
        force_cpe_eval_mode: (bool): Force RADIO to use cropped PE in eval mode, optional for SFT. Defaults to False.
        disable_cpe: (bool): Disable RADIO cropped position embeddings, optional for pre-training and/or SFT. Defaults to False.
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
        hybrid_attention_ratio: float = 1.0,
        hybrid_mlp_ratio: float = 1.0,
        hybrid_override_pattern: str = None,
        fp16_lm_cross_entropy: bool = False,
        image_token_index: int = DEFAULT_IMAGE_TOKEN_INDEX,
        pixel_shuffle: bool = False,
        tile_tags: Optional[list] = None,
        cp_group: Optional[torch.distributed.ProcessGroup] = None,
        max_num_tiles: int = 0,
        tokenizer_type: str = "",
        use_vision_backbone_fp8_arch: bool = False,
        dynamic_resolution: bool = False,
        image_break_token: Optional[str] = None,
        conv_merging: bool = False,
        allow_missing_conv_merge_checkpoint: bool = False,
        efficient_video_sampling_variant: Optional[str] = None,
        sound_model: Optional[torch.nn.Module] = None,
        sound_projection: Optional[torch.nn.Module] = None,
        sound_token_index: int = DEFAULT_SOUND_TOKEN_INDEX,
        class_token_len: Optional[int] = None,
        radio_force_eval_mode: bool = False,
        radio_force_cpe_eval_mode: bool = False,
        radio_interpolate_only_cpe: bool = False,
        radio_cpe_aspect_ratio_select: bool = False,
        radio_disable_cpe: bool = False,
        use_loss_scaling: bool = False,
        log_model_grad_norms: bool = False,
        log_model_act_norms: bool = False,
        video_temporal_patch_size: int = 1,  # Default 1 = no temporal compression
        allow_checkpoint_without_temporal_compression: bool = False,
        separate_video_embedder: bool = False,  # Use separate embedders for images (C*P*P) and videos (C*T*P*P)
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

        self.sound_model = sound_model
        self.sound_projection = sound_projection
        self.sound_token_index = sound_token_index

        self.use_loss_scaling = use_loss_scaling

        language_model_type = getattr(language_transformer_config, "language_model_type", "")

        # Output gradient/activation norms are set dynamically during forward/backward pass
        # if using --log-model-grad-norms. See get_model_grad_norms() and get_model_act_norms().
        self.log_model_grad_norms = log_model_grad_norms
        self.log_model_act_norms = log_model_act_norms
        self._grad_norms: dict = {}
        self._act_norms: dict = {}

        self.sequence_parallel_lm = language_transformer_config.sequence_parallel
        self.tp_comm_overlap_lm = language_transformer_config.tp_comm_overlap
        self.context_parallel_lm = language_transformer_config.context_parallel_size
        if self.sequence_parallel_lm or self.context_parallel_lm > 1:
            # TODO: maybe need a better check for when using heterogeneous layer config
            if not language_model_type.startswith('nemotron5-hybrid') and not language_model_type.startswith('nemotron6-moe') and not isinstance(language_transformer_config, HeterogeneousTransformerConfig):
                attn_module = language_transformer_layer_spec.submodules.self_attention
                assert (
                    attn_module.submodules.core_attention == TEDotProductAttention and HAVE_TE
                ), "Sequence/Context Parallelism is supported only with TE DotProductAttention."
            if self.context_parallel_lm > 1:
                self.cp_group = get_context_parallel_group() if cp_group is None else cp_group
                assert (
                    self.cp_group.size() == self.context_parallel_lm
                ), "CP Group size should match the Language Model CP size"
                assert is_te_min_version(
                    "1.10.0"
                ), "Context Parallelism in LLaVA requires TE v1.10 or higher"
            else:
                self.cp_group = None
        self.tensor_model_parallel_size_lm = language_transformer_config.tensor_model_parallel_size

        # This attribute is needed to check if an all-reduce is required
        # on the word embeddings inside `finalize_model_grads._allreduce_word_embedding_grads`.
        self.share_embeddings_and_output_weights = share_embeddings_and_output_weights

        # Store CLI class_token_len value (will be None if not provided)
        cli_class_token_len = class_token_len
        self._drop_vision_class_token = drop_vision_class_token
        if self.add_encoder:
            self._vision_fp8 = vision_transformer_config.fp8 or use_vision_backbone_fp8_arch
            self._vision_fp8_no_arch = vision_transformer_config.fp8
            vision_projection_input_size = vision_transformer_config.hidden_size
            add_class_token = True

            # Set class_token_len: CLI parameter takes precedence, then model-specific defaults
            if cli_class_token_len is not None:
                class_token_len = cli_class_token_len
            # Otherwise, use model-specific defaults (will be set in each section)
            if vision_transformer_config.vision_model_type.startswith(
                ("clip", "siglip", "internvit")
            ):
                # Use CLI value if provided, otherwise use config default
                if cli_class_token_len is None:
                    class_token_len = getattr(vision_transformer_config, 'class_token_len', 1)

                if vision_transformer_config.vision_model_type == "siglip":
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
            elif "radio" in vision_transformer_config.vision_model_type:
                # TODO: should refactor into model code itself?
                # Initialize defaults - use CLI parameter if provided, otherwise use config default
                max_img_h = 0
                max_img_w = 0
                embedder_bias = False
                ln_post_impl = None
                use_mask_token = False

                # Use CLI value if provided, otherwise use config default
                if cli_class_token_len is None:
                    class_token_len = getattr(vision_transformer_config, 'class_token_len', 8)

                # Set other model-specific parameters
                if vision_transformer_config.vision_model_type == "radio-g":
                    max_img_h = 1792
                    max_img_w = 1792
                    embedder_bias = True
                    from megatron.core.extensions.transformer_engine import TENorm

                    ln_post_impl = TENorm
                    use_mask_token = True
                else:
                    max_img_h = 2048
                    max_img_w = 2048
                    embedder_bias = False
                    ln_post_impl = None
                    use_mask_token = False

                # Apply FP8 override (FP8 has technical requirements that override user preferences)
                if vision_transformer_config.fp8 or use_vision_backbone_fp8_arch:
                    # FP8 padding for final sequence length to be a multiple of 16 or 32.
                    class_token_len = 32 if vision_transformer_config.fp8_recipe == "mxfp8" else 16

                self.vision_model = RADIOViTModel(
                    vision_transformer_config,
                    vision_transformer_layer_spec,
                    ln_post_impl=ln_post_impl,
                    img_h=img_h,
                    img_w=img_w,
                    max_img_h=max_img_h,
                    max_img_w=max_img_w,
                    class_token_len=class_token_len,
                    patch_dim=patch_dim,
                    add_class_token=add_class_token,
                    embedder_bias=embedder_bias,
                    dynamic_resolution=dynamic_resolution,
                    use_mask_token=use_mask_token,
                    force_eval_mode=radio_force_eval_mode,
                    force_cpe_eval_mode=radio_force_cpe_eval_mode,
                    interpolate_only_cpe=radio_interpolate_only_cpe,
                    cpe_aspect_ratio_select=radio_cpe_aspect_ratio_select,
                    has_cpe=not radio_disable_cpe,
                    temporal_patch_dim=video_temporal_patch_size,
                    allow_checkpoint_without_temporal_compression=allow_checkpoint_without_temporal_compression,
                    separate_video_embedder=separate_video_embedder,
                )
            elif vision_transformer_config.vision_model_type.startswith("hf://"):
                from megatron.core.models.huggingface.module import build_hf_model

                self.vision_model = build_hf_model(
                    vision_transformer_config, vision_transformer_config.vision_model_type
                )
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

            self.vision_projection.register_load_state_dict_post_hook(
                _load_state_dict_hook_ignore_extra_state
            )

            if conv_merging:
                # TODO: use different config (ie not vision_projection config), and maybe allow replacing the vision projection itself??
                conv_merge_config = deepcopy(vision_projection_config)
                conv_merge_config.hidden_size = vision_transformer_config.hidden_size
                self.conv_merge = ConvTokenMerge(conv_merge_config)

                if allow_missing_conv_merge_checkpoint:
                    conv_merge_param_names = [
                        f"conv_merge.{name}" for name in self.conv_merge.state_dict().keys()
                    ]
                    self.conv_merge.register_load_state_dict_post_hook(
                        partial(_load_state_dict_hook_ignore_param_names, conv_merge_param_names)
                    )

                self.conv_merge.register_load_state_dict_post_hook(
                    _load_state_dict_hook_ignore_extra_state
                )
            else:
                self.conv_merge = None

        if self.add_decoder:
            if getattr(language_transformer_config, "language_model_type", "").startswith("hf://"):
                from megatron.core.models.huggingface.module import build_hf_model

                self.language_model = build_hf_model(
                    language_transformer_config, language_transformer_config.language_model_type
                )
                self.language_model = build_hf_model(language_transformer_config)
            elif language_model_type.startswith('nemotron5-hybrid') or language_model_type.startswith('nemotron6-moe'):
                self.language_model = MambaModel(
                    config=language_transformer_config,
                    mamba_stack_spec=language_transformer_layer_spec,
                    vocab_size=language_vocab_size,
                    max_sequence_length=language_max_sequence_length,
                    parallel_output=parallel_output,
                    position_embedding_type=language_position_embedding_type,
                    pre_process=self.pre_process,
                    hybrid_attention_ratio=hybrid_attention_ratio,
                    hybrid_mlp_ratio=hybrid_mlp_ratio,
                    hybrid_override_pattern=hybrid_override_pattern,
                    post_process=self.post_process,
                    rotary_percent=language_rotary_percent,
                    rotary_base=language_rotary_base,
                    fp16_lm_cross_entropy=fp16_lm_cross_entropy,
                    scatter_embedding_sequence_parallel=False,
                    share_embeddings_and_output_weights=share_embeddings_and_output_weights,
                )
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
                    rope_scaling_factor=language_rope_scaling_factor,
                    scatter_embedding_sequence_parallel=False,
                    share_embeddings_and_output_weights=share_embeddings_and_output_weights,
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

        self.img_seq_len = get_num_image_embeddings(
            img_h=img_h,
            img_w=img_w,
            patch_dim=patch_dim,
            vision_model_type=vision_transformer_config.vision_model_type,
            disable_vision_class_token=drop_vision_class_token,
            class_token_len=class_token_len,
            pixel_shuffle=pixel_shuffle,
            use_tile_tags=tile_tags is not None,  # Tile tags enabled/disabled.
            max_num_tiles=max_num_tiles,
            tokenizer_type=tokenizer_type,
            use_image_break_token=image_break_token is not None,
            conv_merging=conv_merging,
        )

        self.image_token_index = image_token_index
        self.image_break_token = image_break_token
        self._patch_dim = patch_dim
        self._class_token_len = class_token_len #WARNING: this will be wrong for PP > 1 when encoder isn't created. On most models we remove class token so it's okay, but keep this in mind.
        self._use_conv_merging = getattr(self, "conv_merge", None) is not None
        self._pixel_shuffle = pixel_shuffle
        self._tile_tags = tile_tags
        self._max_num_tiles = max_num_tiles
        self._dynamic_resolution = dynamic_resolution
        self._img_h = img_h
        self._img_w = img_w
        self._video_temporal_patch_size = video_temporal_patch_size
        self.efficient_video_sampler: EVSVariant = self._init_efficient_video_sampling(efficient_video_sampling_variant)

    def _init_efficient_video_sampling(self, evs_variant: str) -> EVSVariant | None:
        evs = EVSVariant.from_string(evs_variant)
        if evs is None:
            return None
        if self._dynamic_resolution:
            raise NotImplementedError("EVS does not support dynamic resolution, yet")
        if self.image_break_token is not None:
            raise NotImplementedError("EVS does not support image break token, yet")
        return evs

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
        self, freeze_language_model: bool, freeze_vision_model: bool, freeze_vision_projection: bool,
        freeze_sound_model: bool, freeze_sound_projection: bool, unfreeze_router: bool
    ):
        """Freeze model modules.

        Make specific modules non-trainable by setting requires_grad to False.

        Args:
            freeze_language_model (bool): Freeze the language model module.
            freeze_vision_model (bool): Freeze the vision model module.
            freeze_vision_projection (bool): Freeze the vision projection module.
            freeze_sound_model (bool): Freeze the sound model module.
            freeze_sound_projection (bool): Freeze the sound projection module.
            unfreeze_router (bool): Keep router weights trainable even if LLM is frozen.
        """
        modules = []
        if freeze_language_model and self.language_model is not None:
            modules.append(self.language_model)
        if freeze_vision_model and self.vision_model is not None:
            modules.append(self.vision_model)
        if freeze_vision_projection and self.vision_projection is not None:
            modules.append(self.vision_projection)
        if freeze_sound_model and self.sound_model is not None:
            modules.append(self.sound_model)
        if freeze_sound_projection and self.sound_projection is not None:
            modules.append(self.sound_projection)

        for module in modules:
            for name, param in module.named_parameters():
                # Option to leave router weights unfrozen even if LLM is frozen.
                if unfreeze_router and "router" in name:
                    continue
                param.requires_grad = False

    def _preprocess_data(
        self,
        image_embeddings,
        language_embeddings,
        input_ids,
        loss_mask,
        labels,
        use_inference_kv_cache,
        inference_context,
        image_token_index,
        num_image_tiles,
        imgs_sizes,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
        insertion_nums=None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        vision_tokens_retention_mask: Optional[torch.Tensor] = None,
        sound_embeddings: Optional[torch.Tensor] = None,
        sound_embeddings_len: Optional[torch.Tensor] = None,
        sound_timestamps: Optional[torch.Tensor] = None,
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
            final_position_ids (Optional[torch.Tensor]): position_ids [b, combined_seq_len].
            packed_seq_params (Optional[PackedSeqParams]): updated PackedSeqParams in case pruning took place.
        """

        inference_context = deprecate_inference_params(inference_context, inference_params)

        assert self.add_decoder, "input text preprocessing is only needed for the language model"

        # No pre- or postprocessing needed.
        # With pipeline parallel > 2, this means a chunk in the middle of the model.
        if not self.pre_process and not self.post_process:
            return None, None, None, None, packed_seq_params

        # If using the inference KV cache, the image tokens are already computed.
        if use_inference_kv_cache:
            return language_embeddings.transpose(0, 1).contiguous(), loss_mask, labels, None, packed_seq_params

        img_seq_len = self.img_seq_len
        if self._dynamic_resolution:
            img_seq_len = torch.prod(
                imgs_sizes // self._patch_dim, dim=-1, dtype=torch.int32
            ) + (0 if self._drop_vision_class_token else self._class_token_len)
            if self._pixel_shuffle:
                img_seq_len = (img_seq_len * (0.5**2)).int()
            if self._use_conv_merging:
                img_seq_len = (img_seq_len * (0.5**2)).int()
            if self.image_break_token is not None and insertion_nums is not None:
                img_seq_len = img_seq_len + insertion_nums.to(device=img_seq_len.device)
            # Logic for when multiple images per image (eg thumbnail image test)
            out_img_seq_len = torch.zeros_like(num_image_tiles)
            start = 0
            for i, c in enumerate(num_image_tiles):
                out_img_seq_len[i] = torch.sum(img_seq_len[start : start + c])
                start += c
            img_seq_len = out_img_seq_len

            img_seq_len = img_seq_len.to("cuda")

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
            if self._dynamic_resolution:
                # Currently makes assumption that mbz is length 1
                # assert (
                #     inference_context is not None or num_images_per_sample.shape[0] == 1
                # ), "Dynamic resolution only works for mbz=1"
                packed_length_per_batch = torch.sum(img_seq_len, dim=-1)

                # In inference, we assume mbz > 1 implies that we're using padding (e.g. for fp8)
                # so we spoof the number of images per sample to be the same as the max,
                # which we assume is the "real" batch.
                # For mbz > 1 with padding (e.g., chosen/rejected pairs in preference learning,
                # or FP8 calibration), we require all samples to have the same number of images.
                # This allows batching while maintaining consistent sequence structure.
                if num_images_per_sample.shape[0] > 1:
                    # Check that all samples have the same number of images (required for batching)
                    assert torch.all(num_images_per_sample == num_images_per_sample[0]), (
                        f"Dynamic resolution with mbz > 1 requires all samples to have the same "
                        f"number of images, got: {num_images_per_sample.tolist()}"
                    )

                if inference_context is not None and num_images_per_sample.shape[0] > 1:
                    num_images_per_sample = num_images_per_sample[0]
                seq_lens = packed_length_per_batch - num_images_per_sample + text_seq_len
            else:
                seq_lens = (
                    num_image_tiles_batch * img_seq_len - num_images_per_sample + text_seq_len
                )
            max_seq_len = seq_lens.max()
            #TODO: should remove this code and force people to us dataloader-seq-length when using pipeline parallelism?
            # Pipeline parallel expects fixed input size. Check if we need to pad.
            if (
                self._language_is_pipeline_parallel
                and max_seq_len < self._language_max_sequence_length
                and inference_context is None
            ):
                max_seq_len = self._language_max_sequence_length

            # Pad combined sequence length to be divisible by shard_factor for SP/CP.
            shard_factor = self._calc_shard_factor()
            if shard_factor is not None and max_seq_len % shard_factor != 0:
                max_seq_len = ((max_seq_len + shard_factor - 1) // shard_factor) * shard_factor

            batch_indices, non_image_indices = torch.where(image_token_mask != True)

            # New position ids for the text tokens, shifted by the image sequence length.
            # E.g. for input_ids = [-200, 1, 2, 3] and img_seq_len = 576, we get
            # new_position_ids = [576, 577, 578, 579]. text_position_ids are then [577, 578, 579].
            image_token_mask_lens = image_token_mask.int().clone()
            # -1 is for the removed image token index.
            if self._dynamic_resolution:
                image_token_mask_lens[image_token_mask] = img_seq_len - 1
            else:
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

        # Build 3D position ids for M-RoPE if enabled: [3, B, max_seq_len]
        final_position_ids = None

        lm_position_type = getattr(self.language_model, "position_embedding_type", None)
        if lm_position_type == "mrope":
            # Temporal: 0..S-1 per sequence
            pos_dtype = torch.int32
            seq_positions = torch.arange(max_seq_len, device=input_ids.device, dtype=pos_dtype).unsqueeze(0).expand(batch_size, -1)
            mrope_pos = torch.zeros((3, batch_size, max_seq_len), dtype=pos_dtype, device=input_ids.device)
            mrope_pos[0] = seq_positions

            # Spatial H/W for image tokens; text tokens remain 0
            total_tiles = int(num_image_tiles.sum().item()) if num_image_tiles is not None and num_image_tiles.numel() > 0 else 0

            if total_tiles > 0:
                # Derive per-tile patch sizes (H, W) at the current tokenization resolution
                if self._dynamic_resolution:
                    # imgs_sizes has shape [num_tiles, 2] with pixel sizes; convert to patch grid
                    tile_hw = (imgs_sizes // self._patch_dim).to(dtype=torch.int32)
                else:
                    h_patches = torch.tensor(self._img_h // self._patch_dim, dtype=torch.int32, device=input_ids.device)
                    w_patches = torch.tensor(self._img_w // self._patch_dim, dtype=torch.int32, device=input_ids.device)
                    tile_hw = torch.stack([
                        h_patches.repeat(total_tiles),
                        w_patches.repeat(total_tiles),
                    ], dim=1)

                # Adjust for pixel shuffle and conv merging (each halves H and W)
                if self._pixel_shuffle:
                    tile_hw = tile_hw // 2
                if self._use_conv_merging:
                    tile_hw = tile_hw // 2

                # Currently, image break token insertion is not supported with M-RoPE mapping
                if self.image_break_token is not None:
                    raise NotImplementedError("M-RoPE is not supported with image_break_token enabled.")

                class_tokens = 0 if self._drop_vision_class_token else int(self._class_token_len)

                h_list: list[torch.Tensor] = []
                w_list: list[torch.Tensor] = []
                for i in range(tile_hw.shape[0]):
                    H_i = int(tile_hw[i, 0].item())
                    W_i = int(tile_hw[i, 1].item())

                    # Prepend class tokens if present
                    if class_tokens > 0:
                        if class_tokens > 0:
                            zeros = torch.zeros(class_tokens, dtype=pos_dtype, device=input_ids.device)
                            h_list.append(zeros)
                            w_list.append(zeros)

                    if H_i > 0 and W_i > 0:
                        h_idx = torch.arange(H_i, device=input_ids.device, dtype=pos_dtype).repeat_interleave(W_i)
                        w_idx = torch.arange(W_i, device=input_ids.device, dtype=pos_dtype).repeat(H_i)
                        h_list.append(h_idx)
                        w_list.append(w_idx)

                if len(h_list) > 0:
                    h_flat = torch.cat(h_list, dim=0)
                    w_flat = torch.cat(w_list, dim=0)
                    # Assign into spatial planes at image token positions
                    mrope_pos[1][images_mask] = h_flat
                    mrope_pos[2][images_mask] = w_flat

            final_position_ids = mrope_pos

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

            sound_mask = input_ids == self.sound_token_index
            # Replace with sound embeddings where needed
            if sound_mask is not False and sound_mask.any():
                # Get the positions where sounds should be placed
                sound_batch_indices, sound_token_indices = torch.where(sound_mask)
                # Map the original token positions to the new (expanded) positions
                sound_new_position_ids = new_position_ids[sound_batch_indices, sound_token_indices]
                # Remove the padding from sound embeddings
                if self.sound_model.config.sound_pad_to_clip_duration:  # ignore lengths, feed padding tokens to the LLM
                    sound_embeddings = sound_embeddings.permute(1, 0, 2).reshape(-1, embed_dim)
                else:
                    sound_embeddings = torch.cat([se[:sel] for se, sel in zip(sound_embeddings.permute(1, 0, 2), sound_embeddings_len)], dim=0)
                final_embedding[sound_batch_indices, sound_new_position_ids] = sound_embeddings.reshape(-1, embed_dim)
            else:
                # TODO: Sound encoder from HF/Nemo can hang with text-only samples. Find a better way to handle this.
                # Note(pzelasko): This should actually be fixed with dynamic shape MR since it disabled NCCL sync of max
                #                 observed seq lengths on DP ranks in FastConformer; but I have no way to test it at the moment.
                if sound_embeddings.shape[0] > 0:
                    assert sound_embeddings.shape[:2] == torch.Size([2, 1]) and sound_timestamps.shape == torch.Size([0])
                    final_embedding[:1, :1, :1] += 0 * sound_embeddings[:1, :1, :1]

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

        if vision_tokens_retention_mask is not None:
            assert self.efficient_video_sampler is not None
            assert images_mask.sum() == vision_tokens_retention_mask.numel()
            final_retention_mask = torch.ones_like(images_mask)
            final_retention_mask[images_mask] = vision_tokens_retention_mask.view(-1)
            assert final_retention_mask.sum() == vision_tokens_retention_mask.sum() + text_position_ids.numel()

            initial_packed_seq_params = copy.deepcopy(packed_seq_params) if packed_seq_params is not None else None
            shard_factor = self._calc_shard_factor() if self.training else None

            sequence_pad_to_divisibility = None
            if self.training and (self._vision_fp8 or self._vision_fp8_no_arch):
                base_factor = shard_factor or 1
                fp8_factor = 16 * base_factor
                sequence_pad_to_divisibility = (base_factor + fp8_factor - 1) // fp8_factor * fp8_factor

            final_embedding, final_position_ids, packed_seq_params = self.efficient_video_sampler.mask_embeddings(
                embeddings=final_embedding, evs_mask=final_retention_mask, packed_seq_params=initial_packed_seq_params,
                per_sample_pad_to_divisibility=shard_factor, sequence_pad_to_divisibility=sequence_pad_to_divisibility,
            )
            if has_labels:
                final_labels, final_loss_mask = self.efficient_video_sampler.mask_labels_and_loss_mask(
                    labels=final_labels, loss_mask=final_loss_mask, evs_mask=final_retention_mask, packed_seq_params=initial_packed_seq_params,
                    per_sample_pad_to_divisibility=shard_factor, sequence_pad_to_divisibility=sequence_pad_to_divisibility,
                    labels_padding_value=IGNORE_INDEX, loss_padding_value=0
                )

        if final_embedding is not None and final_labels is not None:
            assert (
                final_embedding.shape[:2] == final_labels.shape == final_loss_mask.shape
            ), "unexpected shapes after data preprocessing"

        if final_embedding is not None:
            # Truncate if exceeding the language model's max sequence length.
            if inference_context is not None and final_embedding.shape[1] > self._language_max_sequence_length:
                raise ValueError(
                    f"Final embedding shape {final_embedding.shape} exceeds language max sequence length {self._language_max_sequence_length}",
                    "You might want to increase the value of the args decoder-seq-length, max-position-embeddings, inference-max-seq-length, and max-tokens-to-oom.")
            if final_embedding.shape[1] > self._language_max_sequence_length:
                final_embedding = final_embedding[:, : self._language_max_sequence_length]
            # Transpose to [s,b,h] only if not using CP because CP Sharding expects seq in dim=1
            if self.context_parallel_lm == 1:
                final_embedding = final_embedding.transpose(1, 0).contiguous()

        if final_position_ids is not None:
            # Truncate if exceeding the language model's max sequence length.
            if final_position_ids.shape[1] > self._language_max_sequence_length:
                final_position_ids = final_position_ids[:, : self._language_max_sequence_length]
            # Transpose to [s,b,h] only if not using CP because CP Sharding expects seq in dim=1
            if self.context_parallel_lm == 1:
                final_position_ids = final_position_ids.transpose(1, 0).contiguous()

        truncate_labels = (
            final_labels is not None and final_labels.shape[1] > self._language_max_sequence_length
        )
        if truncate_labels:
            final_labels = final_labels[:, : self._language_max_sequence_length]
            final_loss_mask = final_loss_mask[:, : self._language_max_sequence_length]

        return final_embedding, final_labels, final_loss_mask, final_position_ids, packed_seq_params

    @staticmethod
    def calc_shard_factor_and_seq_dim_for_preprocessing(context_parallel_lm, sequence_parallel_lm, tensor_model_parallel_size_lm):
        shard_factor = seq_dim = None
        if context_parallel_lm > 1 and sequence_parallel_lm:
            shard_factor = max(tensor_model_parallel_size_lm * context_parallel_lm, context_parallel_lm * 2)
            seq_dim = 1
        elif context_parallel_lm > 1:
            shard_factor = context_parallel_lm * 2
            seq_dim = 1
        elif sequence_parallel_lm:
            shard_factor = tensor_model_parallel_size_lm
            seq_dim = 0
        return shard_factor, seq_dim

    def _calc_shard_factor(self, *, validate_with_combined_embeddings=None):
        shard_factor = seq_dim = None
        if not self.pre_process:
            return None

        shard_factor, seq_dim = self.calc_shard_factor_and_seq_dim_for_preprocessing(
            context_parallel_lm=self.context_parallel_lm,
            sequence_parallel_lm=self.sequence_parallel_lm,
            tensor_model_parallel_size_lm=self.tensor_model_parallel_size_lm,
        )

        if validate_with_combined_embeddings is not None and shard_factor is not None:
            assert (
                    validate_with_combined_embeddings.shape[seq_dim] % shard_factor == 0
            ), f"Sequence length {validate_with_combined_embeddings.shape[seq_dim]} should be divisible by {shard_factor} for \
                        Sequence/Context parallelism"
            if self.sequence_parallel_lm and self.tp_comm_overlap_lm:
                assert (
                        validate_with_combined_embeddings.shape[seq_dim] == self._language_max_sequence_length
                ), f"TP Comm overlap either requires Vision+Text token length \
                        == language_max_sequence_length"

        return shard_factor

    def _process_embedding_token_parallel(
        self, combined_embeddings, new_labels, new_loss_mask, loss_weight, position_ids: Optional[torch.Tensor], packed_seq_params
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
            position_ids (torch.Tensor): Distributed position ids. If input is None, it will be None.
            packed_seq_params (PackedSeqParams): Dict with padded token information.

        """
        # No pre or post processing needed with PP middle chunks.
        if not self.pre_process and not self.post_process:
            return combined_embeddings, new_labels, new_loss_mask, position_ids, packed_seq_params

        _ = self._calc_shard_factor(validate_with_combined_embeddings=combined_embeddings)  # used just to assert we're good

        if self.context_parallel_lm > 1:
            batch = dict()
            if self.pre_process:
                batch["combined_embeddings"] = combined_embeddings
                # Only include 2D position_ids ([B,S]) in CP splitting; 3D mRoPE ids are handled in the model
                if position_ids is not None and position_ids.dim() == 2:
                    batch["position_ids"] = position_ids
            if self.post_process and new_labels is not None:
                batch["new_labels"] = new_labels
                batch["new_loss_mask"] = new_loss_mask
                if loss_weight is not None:
                    batch["loss_weight"] = loss_weight
            # Distribute sequence across CP ranks
            if packed_seq_params is None or packed_seq_params.qkv_format == 'sbhd':
                from megatron.training.utils import get_batch_on_this_cp_rank

                batch = get_batch_on_this_cp_rank(batch)
            else:
                assert HAVE_TEX and is_te_min_version(
                    "1.10.0"
                ), "Please update Transformer Engine to >= 1.10 to use \
                    Context Parallel with THD format data"
                cp_size = self.cp_group.size()
                cp_rank = self.cp_group.rank()
                for key, data in batch.items():
                    index = tex.thd_get_partitioned_indices(
                        packed_seq_params.cu_seqlens_q_padded, data.size(1), cp_size, cp_rank
                    )
                    batch[key] = data.index_select(1, index)

            if self.pre_process:
                combined_embeddings = batch["combined_embeddings"]  # [B, S/CP, H]
                combined_embeddings = combined_embeddings.transpose(
                    1, 0
                ).contiguous()  # [B,S/CP,H] -> [S/CP,B,H]
                if "position_ids" in batch:
                    position_ids = batch["position_ids"]
                    position_ids = position_ids.transpose(1, 0).contiguous()  # [B,S/CP] -> [S/CP,B]
            if self.post_process and new_labels is not None:
                new_labels = batch["new_labels"]
                new_loss_mask = batch["new_loss_mask"]
                if "loss_weight" in batch:
                    new_loss_mask = new_loss_mask * batch["loss_weight"]

        if self.sequence_parallel_lm and self.pre_process:
            combined_embeddings = tensor_parallel.scatter_to_sequence_parallel_region(
                combined_embeddings
            )  # [S/(CP*TP),B,H]

        return combined_embeddings, new_labels, new_loss_mask, position_ids, packed_seq_params

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

    def _add_fp8_padding_for_inference(self, images, imgs_sizes, vision_packed_seq_params, has_pad_img):
        """Add FP8 padding for inference when not using context parallelism.

        This method applies FP8 padding to images during inference to ensure proper alignment
        for FP8 operations, similar to split_to_context_parallel_ranks_dynamic_res but for
        the non-context-parallel case.

        Args:
            images (torch.Tensor): Input images tensor.
            imgs_sizes (torch.Tensor): Image sizes tensor.
            vision_packed_seq_params (PackedSeqParams): Vision packed sequence parameters.
            has_pad_img (bool): Whether padding image is already present.

        Returns:
            tuple: (images, imgs_sizes, vision_packed_seq_params, has_pad_img) with FP8 padding applied.
        """
        final_seqlen = images.shape[1]

        padding_needed = get_padding(final_seqlen, 1, 1, False, fp8_enabled=True)

        if padding_needed > 0:
            patch_dim = self.vision_model.patch_dim

            pad_img = torch.zeros([1, padding_needed, patch_dim * patch_dim * 3], device=images.device, dtype=images.dtype)

            # Concatenate padding image to the batch
            images = torch.cat([images, pad_img], dim=1)

            # Update imgs_sizes with padding dimensions
            pad_img_size = torch.tensor([[patch_dim, patch_dim * padding_needed]], device=imgs_sizes.device, dtype=imgs_sizes.dtype)

            imgs_sizes = torch.cat([imgs_sizes, pad_img_size])

            # Update vision_packed_seq_params
            if vision_packed_seq_params is not None:
                cu_seqlens = vision_packed_seq_params.cu_seqlens_q

                new_cu_seqlens = torch.cat([cu_seqlens, torch.tensor([final_seqlen + padding_needed], device=cu_seqlens.device, dtype=cu_seqlens.dtype)])
                vision_packed_seq_params.cu_seqlens_q = new_cu_seqlens
                vision_packed_seq_params.cu_seqlens_kv = new_cu_seqlens

                # Update padded sequence lengths if they exist
                if vision_packed_seq_params.cu_seqlens_q_padded is not None:
                    cu_seqlens_padded = vision_packed_seq_params.cu_seqlens_q_padded
                    new_cu_seqlens_padded = torch.cat([cu_seqlens_padded, torch.tensor([final_seqlen + padding_needed], device=cu_seqlens_padded.device, dtype=cu_seqlens_padded.dtype)])
                    vision_packed_seq_params.cu_seqlens_q_padded = new_cu_seqlens_padded
                    vision_packed_seq_params.cu_seqlens_kv_padded = new_cu_seqlens_padded

                # Update max sequence lengths
                seqlens = new_cu_seqlens[1:] - new_cu_seqlens[:-1]
                max_seqlen = max(seqlens).to(torch.int32)
                vision_packed_seq_params.max_seqlen_q = max_seqlen
                vision_packed_seq_params.max_seqlen_kv = max_seqlen

            has_pad_img = True

        return images, imgs_sizes, vision_packed_seq_params, has_pad_img

    def _register_grad_norm_hook(self, output: torch.Tensor, name: str) -> None:
        """Create a backward hook that stores the gradient norm for a given component."""
        if not isinstance(output, torch.Tensor):
            raise ValueError(
                f"Output variable {name} has type {type(output)}, expected torch.Tensor"
                f" for _register_grad_norm_hook()."
            )

        if not output.requires_grad:
            return

        def _hook(grad: torch.Tensor) -> None:
            self._grad_norms[name] = grad.norm().item()
        output.register_hook(_hook)

    def _store_activation_norm(self, activation: torch.Tensor, name: str) -> None:
        """Store the activation norm for a given component."""
        if not isinstance(activation, torch.Tensor):
            return
        self._act_norms[name] = activation.norm().item()

    def get_model_grad_norms(self) -> dict:
        """Get output grad norms for available components.

        Returns dict with norms for components that had hooks registered during forward pass.
        Possible components (depending on config and PP rank):
        - vision_model: after class token removal, before conv_merge/pixel_shuffle
        - conv_merge: after conv_merge (only if conv_merge enabled)
        - vision_projection: after projection
        - language_model: language model output

        Returns empty dict if no grads are available (e.g., on PP ranks without these components).
        """
        return self._grad_norms.copy()

    def get_model_act_norms(self) -> dict:
        """Get output activation norms for available components.

        Returns dict with norms for components that were recorded during forward pass.
        Same components as get_model_grad_norms().

        Returns empty dict if no activations are available.
        """
        return self._act_norms.copy()

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        loss_mask: Optional[torch.Tensor] = None,
        inference_context: Optional[BaseInferenceContext] = None,
        num_image_tiles: Optional[List[int]] = None,
        num_frames: Optional[List[int]] = None,
        image_token_index: Optional[int] = None,
        runtime_gather_output: Optional[bool] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        imgs_sizes: Optional[Tuple[int, int]] = None,
        vision_packed_seq_params: Optional[PackedSeqParams] = None,
        has_pad_img: bool = False,
        sound_clips: Optional[torch.Tensor] = None,
        sound_length: Optional[torch.Tensor] = None,
        sound_timestamps: Optional[torch.Tensor] = None,
        num_sound_clips: Optional[torch.Tensor] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
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
            inference_context (BaseInferenceContext): Inference-time parameters including KV cache.
            num_image_tiles (list of int): Number of tiles per image. Default 1 tile per image.
            num_frames (list of int): Number of frames. Images have a single frame, video clips can have multiple frames.
            image_token_index (int): ID for input images. Default None means `image_token_index`
                arg in the constructor will be used.
            runtime_gather_output (bool): Gather output at runtime. Default None means
                `parallel_output` arg in the constructor will be used.
            packed_seq_params (PackedSeqParams): 1) If using sequence packing, must contain
                subsample length information. 2) If using SP/CP with padding mask type,
                must contain padded token information.
            vision_packed_seq_params (PackedSeqParams): Vision packed sequence parameters containing
                cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, and max_seqlen_kv for vision model.

        Returns:
            output (torch.Tensor): Loss of shape [b, s] if labels are provided,
                otherwise logits of shape [b, s, vocab_size].
            loss_mask (torch.Tensor): Loss mask expanded to combined sequence length. Shape [b, s].
        """
        inference_context = deprecate_inference_params(inference_context, inference_params)

        # Keep a copy of the original imgs_sizes and num_frames in case we split to context parallel ranks later.
        global_imgs_sizes = imgs_sizes.clone() if imgs_sizes is not None else None
        if num_frames is not None:
            if not isinstance(num_frames, torch.Tensor):
                num_frames = torch.tensor(num_frames, dtype=torch.int32, device=imgs_sizes.device)
            global_num_frames = num_frames.clone()
        else:
            global_num_frames = None

        use_inference_kv_cache = (
            inference_context is not None
            and (
                "image_tokens_count" in inference_context.key_value_memory_dict
                or "sound_tokens_count" in inference_context.key_value_memory_dict
            )
        )
        has_images = images is not None and images.shape[0] > 0

        has_sounds = (sound_clips is not None and
                      sound_clips.numel() > 1 and  # Not just a single element
                      not (sound_clips.shape == torch.Size([1, 1]) and sound_clips[0,0].item() == 0))  # Not the dummy tensor

        insertion_nums = None
        image_tokens_retention_mask = None

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
            pad = None
            if self._dynamic_resolution:
                if self.context_parallel_lm > 1:
                    # This will split the images and imgs_sizes to context parallel ranks. Each rank will have a different imgs_sizes.
                    # If there are fewer images than CP ranks, dummy images are added to keep all ranks active.
                    dummy_img_size = self.vision_model.patch_dim
                    if self._pixel_shuffle:
                        dummy_img_size = dummy_img_size * 2
                    if self.conv_merge is not None:
                        dummy_img_size = dummy_img_size * 2

                    # Split images and imgs_sizes across context parallel ranks
                    # When using temporal compression, split on tubelet boundaries to avoid mid-tubelet splits
                    images, imgs_sizes, vision_packed_seq_params, has_pad_img, num_padded_imgs, local_num_frames = \
                        split_to_context_parallel_ranks_dynamic_res(
                            images, imgs_sizes, vision_packed_seq_params, self._vision_fp8, dummy_img_size,
                            num_frames=num_frames, temporal_patch_size=self._video_temporal_patch_size
                        )

                    # Update num_frames if it was split
                    if local_num_frames is not None:
                        num_frames = local_num_frames
                else:
                    num_padded_imgs = 0  # No CP padding when not using context parallelism
                    # TODO: should we replace the dataset_helper/task_encoder code in training that adds the padding image with this?
                    # Add FP8 padding for inference when not using context parallelism
                    if inference_context is not None and not has_pad_img and self._vision_fp8_no_arch:
                        images, imgs_sizes, vision_packed_seq_params, has_pad_img = self._add_fp8_padding_for_inference(
                            images, imgs_sizes, vision_packed_seq_params, has_pad_img
                        )

                if self._video_temporal_patch_size > 1:
                    image_embeddings, imgs_sizes, num_frames = self.vision_model(
                        images, imgs_sizes=imgs_sizes, packed_seq_params=vision_packed_seq_params,
                        num_frames=num_frames,
                    )  # [num_tiles, img_seq_len, h_vision]

                    # Because we only support dynamic res, num_image_tiles is a list of all ones
                    #   with length equal to len(imgs_sizes) == sum(num_frames), where each entry
                    #   in num_frames is 1 for images, and >1 for videos. We must update this after
                    #   updating imgs_sizes and num_frames
                    num_image_tiles = torch.ones(
                        len(imgs_sizes), dtype=torch.int, device=images.device
                    )
                else:
                    image_embeddings = self.vision_model(
                        images, imgs_sizes=imgs_sizes, packed_seq_params=vision_packed_seq_params,
                    )  # [num_tiles, img_seq_len, h_vision]

            else:
                assert self._video_temporal_patch_size == 1, "Temporal compression is not supported for tiling"
                if self.context_parallel_lm > 1 and images.shape[0] >= 2:
                    cp_images, pad = split_to_context_parallel_ranks(images)
                    image_embeddings = self.vision_model(cp_images)
                else:
                    image_embeddings = self.vision_model(images)  # [num_tiles, img_seq_len, h_vision]

            if self._drop_vision_class_token:
                if self._dynamic_resolution:
                    remove_class_token_mask = torch.full(
                        (image_embeddings.shape[-2],), True, dtype=torch.bool
                    )
                    seq_lens = torch.prod(imgs_sizes // self.vision_model.patch_dim, dim=-1)
                    # TODO: efficiency, torch.cumsum?, broadcast a mask instead of full_like?
                    current_length = 0
                    for seq_len in seq_lens:
                        remove_class_token_mask[
                            current_length : current_length + self.vision_model.class_token_len
                        ] = False
                        current_length += seq_len + self.vision_model.class_token_len
                    image_embeddings = image_embeddings[:, remove_class_token_mask, :]
                else:
                    image_embeddings = image_embeddings[:, self.vision_model.class_token_len :, :]

            # If we used a fake image to pad for fp8 with dynamic resolution, remove if from the
            # image embeddings and imgs_sizes.
            if has_pad_img and self._dynamic_resolution:
                pad_len = imgs_sizes[-1][0] // self.vision_model.patch_dim * imgs_sizes[-1][1] // self.vision_model.patch_dim
                image_embeddings = image_embeddings[:, :-pad_len, :]
                imgs_sizes = imgs_sizes[:-1]

            # Track norms for vision model output (after class token removal, before conv_merge/pixel_shuffle)
            if self.log_model_act_norms and self.training:
                self._store_activation_norm(image_embeddings, name="vision_model")
            if self.log_model_grad_norms and self.training:
                self._register_grad_norm_hook(image_embeddings, name="vision_model")

            # Apply conv-merge before pixel-shuffle to keep ConvTokenMerge input dims consistent
            if self.conv_merge is not None:
                if self._dynamic_resolution:
                    image_embeddings = self.conv_merge(
                        image_embeddings, imgs_sizes // self.vision_model.patch_dim
                    )
                else:
                    image_embeddings = self.conv_merge(
                        image_embeddings,
                        [
                            (
                                self._img_h // self.vision_model.patch_dim,
                                self._img_w // self.vision_model.patch_dim,
                            )
                        ],
                    )

                # Track norms for conv_merge output (after conv_merge, before pixel_shuffle)
                if self.log_model_act_norms and self.training:
                    self._store_activation_norm(image_embeddings, name="conv_merge")
                if self.log_model_grad_norms and self.training:
                    self._register_grad_norm_hook(image_embeddings, name="conv_merge")

            if self._pixel_shuffle:
                if self._dynamic_resolution:
                    # After conv-merge, the effective patch size doubles for token layout
                    eff_patch_dim = (
                        self.vision_model.patch_dim * (2 if self.conv_merge is not None else 1)
                    )
                    image_embeddings = pixel_shuffle_dynamic_res(
                        image_embeddings, imgs_sizes, eff_patch_dim
                    )  # [num_tiles, img_seq_len_shuffled, h_vision_shuffled]
                else:
                    image_embeddings = pixel_shuffle(
                        image_embeddings
                    )  # [num_tiles, img_seq_len_shuffled, h_vision_shuffled]

                # NOTE: Pixel shuffle has no params, act/grad norm is same as prev embeddings

            # contiguous() required as `permute` can sparsify the tensor and this breaks pipelining
            image_embeddings = image_embeddings.permute(
                1, 0, 2
            ).contiguous()  # [img_seq_len, num_tiles, h_vision]

            vision_projection_padding_needed = 0
            if self._vision_fp8_no_arch and self._dynamic_resolution:
                vision_projection_padding_needed = get_padding(image_embeddings.shape[0], self.context_parallel_lm, self.tensor_model_parallel_size_lm, self.sequence_parallel_lm, fp8_enabled=self._vision_fp8)
                if vision_projection_padding_needed > 0:
                    padding_image_embeddings = torch.zeros([vision_projection_padding_needed, image_embeddings.shape[1], image_embeddings.shape[2]]).to(image_embeddings.device).to(image_embeddings.dtype)
                    image_embeddings = torch.cat([image_embeddings, padding_image_embeddings], dim=0)

            # map vision model output size to language model input size.
            image_embeddings = self.vision_projection(
                image_embeddings
            )  # [img_seq_len, num_tiles, h_language]

            # Track norms for vision_projection output
            if self.log_model_act_norms and self.training:
                self._store_activation_norm(image_embeddings, name="vision_projection")
            if self.log_model_grad_norms and self.training:
                self._register_grad_norm_hook(image_embeddings, name="vision_projection")

            if vision_projection_padding_needed > 0:
                image_embeddings = image_embeddings[:-vision_projection_padding_needed, :, :]

            if self.context_parallel_lm > 1 and self._dynamic_resolution:
                image_embeddings = gather_from_context_parallel_ranks_dynamic_res(image_embeddings, num_padded_imgs)
                # For temporal compression, gather the post-compression imgs_sizes and num_frames
                # For non-temporal, use the saved global values for backwards compatibility
                if self._video_temporal_patch_size > 1:
                    imgs_sizes = gather_from_context_parallel_ranks_dynamic_res(imgs_sizes, num_padded_imgs)
                    if num_frames is not None:
                        # Convert to tensor if it's a list
                        if not isinstance(num_frames, torch.Tensor):
                            num_frames = torch.tensor(num_frames, dtype=torch.int, device=imgs_sizes.device)
                        num_frames = gather_from_context_parallel_ranks_dynamic_res(
                            num_frames.unsqueeze(-1) if num_frames.dim() == 1 else num_frames, num_padded_imgs
                        ).squeeze(-1)
                else:
                    imgs_sizes = global_imgs_sizes
                    if global_num_frames is not None:
                        num_frames = global_num_frames
                # num_image_tiles: one entry per frame/tubelet, all 1s (no tiling for dynamic res)
                num_image_tiles = torch.ones(len(imgs_sizes), dtype=torch.int, device=imgs_sizes.device)

            if self.image_break_token is not None:
                patch_sizes = imgs_sizes // self.vision_model.patch_dim
                if self._pixel_shuffle:
                    patch_sizes = patch_sizes // 2
                if self.conv_merge is not None:
                    patch_sizes = patch_sizes // 2
                image_break_token = self.language_model.embedding(
                    input_ids=torch.tensor(
                        [self.image_break_token], device=image_embeddings.device
                    ),
                    position_ids=None,
                ).transpose(1, 0)
                image_embeddings, insertion_nums = insert_image_break_tokens(
                    image_embeddings, patch_sizes, image_break_token
                )

            # Apply tile tagging if enabled and an image token is present.
            if self._tile_tags is not None and torch.any(input_ids == self.image_token_index):
                if self.efficient_video_sampler is not None:
                    raise NotImplementedError  # TODO: must rearrange `masks_seqlen` as well
                image_embeddings = self._apply_tile_tagging(image_embeddings, num_image_tiles)

            if self.context_parallel_lm > 1 and pad is not None and not self._dynamic_resolution:
                image_embeddings = gather_from_context_parallel_ranks(image_embeddings, pad)

            # Here, `image_embeddings` and `images` represent entire batch (not cp chunk).

            image_tokens_retention_mask, masks_seqlen = None, None
            if self.efficient_video_sampler is not None and self.efficient_video_sampler.enabled:
                if num_frames is None:
                    raise ValueError("`num_frames` is not available, however, Efficient Video Sampling is enabled, and requires it.")

                is_video = [_ > 1 for _ in num_frames]
                evs_masks, masks_seqlen = self.efficient_video_sampler.calculate_mask(
                    images=images, embeddings=image_embeddings, num_tiles=num_image_tiles, num_frames=num_frames, is_video=is_video, is_training=self.training,
                    # noqa
                )
                # For simplicity of dealing with list of images, we concat all
                image_tokens_retention_mask = torch.cat(evs_masks, dim=0)
                _, actual_mask_seqlen = image_tokens_retention_mask.shape
                assert sum(map(len, evs_masks)) == len(images)
                if masks_seqlen != actual_mask_seqlen:
                    raise ValueError(f"Mismatch between actual EVS mask seqlen ({actual_mask_seqlen}) and expected EVS mask seqlen ({masks_seqlen}) ")
                if masks_seqlen != image_embeddings.shape[0]:
                    raise ValueError(f"Mismatch between EVS mask seqlen ({masks_seqlen}) to the image embeddings seqlen ({image_embeddings.shape[0]})")

            # TODO: Support batched inference.
            # In inference, the language model KV cache will be updated for image token positions.
            # Store the image tokens sequence length to be used as an offset to the KV cache later.
            if inference_context is not None:
                if image_tokens_retention_mask is not None:
                    image_tokens_count = torch.count_nonzero(image_tokens_retention_mask)
                else:
                    image_tokens_count = image_embeddings.shape[0] * image_embeddings.shape[1]
                inference_context.key_value_memory_dict["image_tokens_count"] = image_tokens_count
        else:
            image_embeddings = self.encoder_hidden_state

        if use_inference_kv_cache:
            sound_embeddings = None
            sound_embeddings_len = None
        elif self.add_encoder and not has_sounds:
            device = sound_clips.device if sound_clips is not None else "cuda"
            dtype = sound_clips.dtype if sound_clips is not None else torch.float32
            sound_embeddings = torch.tensor([], dtype=dtype, device=device).reshape(
                0, 0, 0
            )
            sound_embeddings_len = torch.tensor([], dtype=torch.long, device=device).reshape(0)
        elif self.add_encoder and has_sounds:
            sound_pad = None
            is_parakeet = "parakeet" in self.sound_model.config.sound_model_type.lower()

            if self.context_parallel_lm > 1 and sound_clips.shape[0] > self.context_parallel_lm:
                sound_clips, sound_pad = split_to_context_parallel_ranks(sound_clips)
                if is_parakeet:
                    # Parakeet needs sound lengths. Minimum sound length is the hop length.
                    sound_length, sound_pad2 = split_to_context_parallel_ranks(sound_length, pad_value=1600)
                    assert sound_pad == sound_pad2, "something went wrong with splitting to context parallel ranks"

            if is_parakeet:
                # note(pzelasko): With dynamic shapes we are getting much larger batch sizes (throughput ~2.5x) but still dominated by padding.
                #                 Unless bucketing is enabled, set this to 2 or higher to avoid OOMs.
                #                 Bucketing via '--packing-knapsack-algorithm bucketing_greedy_knapsack' resolves this issue and increases throughput another ~1.65x.
                if (split_factor := self.sound_model.config.sound_batch_split) <= 1:
                    sound_embeddings, sound_embeddings_len = self.sound_model(sound_clips, sound_length) # [num_clips, sound_seq_len, h_sound]
                else:
                    sound_clips = torch.chunk(sound_clips, split_factor, dim=0)
                    sound_length = torch.chunk(sound_length, split_factor, dim=0)
                    se = []
                    sel = []
                    for sound_clip, sound_length in zip(sound_clips, sound_length):
                        maxlen = sound_length.max()
                        sound_clip = sound_clip[:, :maxlen]  # save time and memory
                        sound_embeddings, sound_embeddings_len = self.sound_model(sound_clip, sound_length) # [num_clips, sound_seq_len, h_sound]
                        se.append(sound_embeddings)
                        sel.append(sound_embeddings_len)
                    maxlen = max([emb.shape[1] for emb in se])
                    se = [torch.cat([emb, torch.zeros(emb.shape[0], maxlen - emb.shape[1], emb.shape[2], device=emb.device, dtype=emb.dtype)], dim=1) for emb in se]
                    sound_embeddings = torch.cat(se, dim=0)
                    sound_embeddings_len = torch.cat(sel, dim=0)
            else:
                sound_embeddings = self.sound_model(sound_clips) # [num_clips, sound_seq_len, h_sound]

            # contiguous() required as `permute` can sparsify the tensor and this breaks pipelining
            sound_embeddings = sound_embeddings.permute(
                1, 0, 2
            ).contiguous()  # [sound_seq_len, num_clips, h_sound]

            # map audio model output size to language model input size.
            sound_embeddings = self.sound_projection(
                sound_embeddings
            ).contiguous()  # [sound_seq_len, num_clips, h_language]

            if self.context_parallel_lm > 1 and sound_pad is not None:
                sound_embeddings = gather_from_context_parallel_ranks(sound_embeddings, sound_pad)
                if sound_embeddings_len is not None:
                    # Gather sound_embeddings_len along the clips dimension (unsqueeze to 2D, gather, squeeze back)
                    sound_embeddings_len = gather_from_context_parallel_ranks(sound_embeddings_len.unsqueeze(0), sound_pad).squeeze(0)

            if inference_context is not None:
                inference_context.key_value_memory_dict["sound_tokens_count"] = sound_embeddings.shape[1]
        else:
            sound_embeddings = self.encoder_hidden_state
            sound_embeddings_len = None

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

        combined_embeddings, new_labels, new_loss_mask, position_ids, packed_seq_params = self._preprocess_data(
            image_embeddings,
            language_embeddings,
            input_ids,
            loss_mask,
            labels,
            use_inference_kv_cache,
            inference_context,
            image_token_index if image_token_index is not None else self.image_token_index,
            num_image_tiles,
            imgs_sizes,
            insertion_nums=insertion_nums,
            packed_seq_params=packed_seq_params,
            vision_tokens_retention_mask=image_tokens_retention_mask,
            sound_embeddings=sound_embeddings,
            sound_embeddings_len=sound_embeddings_len,
            sound_timestamps=sound_timestamps,
        )  # [combined_seq_len, b, h_language], [b, combined_seq_len], [b, combined_seq_len]

        if self.context_parallel_lm > 1 or self.sequence_parallel_lm:
            loss_weight = None
            if new_labels is not None:
                acc_lengths = (
                    packed_seq_params.cu_seqlens_q_padded
                    if packed_seq_params is not None
                    else [0, combined_embeddings.shape[0]]
                )
                num_samples = len(acc_lengths) - 1
                if self.use_loss_scaling:
                    loss_weight = pre_calc_loss_weight(num_samples, acc_lengths, new_labels[0])
                else:
                    loss_weight = torch.ones(new_labels[0].shape[0], device=new_labels[0].device, dtype=torch.float32)
                    loss_weight[new_labels[0]==IGNORE_INDEX] = 0
                loss_weight = loss_weight.unsqueeze(0)

            combined_embeddings, new_labels, new_loss_mask, position_ids, packed_seq_params = (
                self._process_embedding_token_parallel(
                    combined_embeddings, new_labels, new_loss_mask, loss_weight, position_ids, packed_seq_params
                )
            )

        output = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            decoder_input=combined_embeddings,
            labels=new_labels,
            inference_context=inference_context,
            runtime_gather_output=runtime_gather_output,
            packed_seq_params=packed_seq_params,
        )
        # Track norms for language_model output
        if self.log_model_act_norms and self.training:
            self._store_activation_norm(output, name="language_model")
        if self.log_model_grad_norms and self.training:
            self._register_grad_norm_hook(output, name="language_model")

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
                logging.getLogger(__name__).debug(
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


def pixel_shuffle_dynamic_res(x, imgs_sizes, patch_dim, scale_factor=0.5, version=2):
    """From InternVL.

    Args:
        x (torch.Tensor): Vision model outputs [num_tiles, img_seq_len, h_vision]
        imgs_sizes (torch.Tensor): Size of images in packed sequence for dynamic resolution
        version (int): Implementation version.

    Returns:
        x (torch.Tensor): Shuffled vision model outputs
            [num_tiles, (sq ** 2) * (scale ** 2), h_vision / (scale ** 2)]
    """
    seq_lens = torch.prod(imgs_sizes // patch_dim, dim=-1)
    splits = torch.split(x, seq_lens.tolist(), dim=-2)

    out = []
    for i, sv in enumerate(splits):
        h = imgs_sizes[i][0] // patch_dim
        w = imgs_sizes[i][1] // patch_dim
        sv = sv.reshape(sv.shape[0], h, w, -1)

        n, h, w, c = sv.size()

        sv = sv.view(n, h, int(w * scale_factor), int(c / scale_factor))
        sv = sv.permute(0, 2, 1, 3).contiguous()
        sv = sv.view(
            n, int(w * scale_factor), int(h * scale_factor), int(c / (scale_factor * scale_factor))
        )

        if version == 2:
            sv = sv.permute(0, 2, 1, 3).contiguous()

        sv = sv.reshape(sv.shape[0], -1, sv.shape[-1])
        out.append(sv)

    x = torch.cat(out, dim=-2)

    return x


def insert_image_break_tokens(x, patch_sizes, image_break_token):
    num_insertions = [h - 1 for h, _ in patch_sizes]
    new_shape = list(x.shape)
    new_shape[-3] += sum(num_insertions)
    new_image_embeddings = torch.zeros(new_shape, dtype=x.dtype, device=x.device)

    start = 0
    start_new = 0
    for how_many, interval in patch_sizes:
        for i in range(how_many - 1):
            new_image_embeddings[start_new : start_new + interval, :, :] = x[
                start : start + interval, :, :
            ]
            new_image_embeddings[start_new + interval, :, :] = image_break_token
            start += interval
            start_new += interval + 1
        new_image_embeddings[start_new : start_new + interval, :, :] = x[
            start : start + interval, :, :
        ]
        start += interval
        start_new += interval

    return new_image_embeddings, torch.tensor(num_insertions, device=new_image_embeddings.device)


def pre_calc_loss_weight(num_samples, acc_lengths, shift_labels):
    """Loss weighting when using context parallel with packed sequences."""
    loss_weight = torch.ones(shift_labels.shape[0], device=shift_labels.device, dtype=torch.float32)
    num_valid_labels_list = []
    loss_weight[shift_labels==IGNORE_INDEX] = 0
    all_num_valid_labels = (shift_labels!=IGNORE_INDEX).sum()
    for sample_idx in range(num_samples):
        weight_this_sample = loss_weight[acc_lengths[sample_idx]: acc_lengths[sample_idx+1]]
        shift_labels_this_sample = shift_labels[acc_lengths[sample_idx]:acc_lengths[sample_idx+1]]
        num_valid_labels = (shift_labels_this_sample!=IGNORE_INDEX).sum(-1)
        if num_valid_labels > 0:
            weight_this_sample = weight_this_sample / num_valid_labels * num_valid_labels.sqrt()
            num_valid_labels_list.append(num_valid_labels)
        loss_weight[acc_lengths[sample_idx]: acc_lengths[sample_idx+1]] = weight_this_sample
    base_num = torch.stack(num_valid_labels_list).sqrt().sum()
    loss_weight = loss_weight / base_num
    return loss_weight
