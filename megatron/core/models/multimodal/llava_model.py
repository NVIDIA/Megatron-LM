# Copyright (c) 2024-2026, NVIDIA CORPORATION. All rights reserved.
import logging
from collections import namedtuple
from functools import partial
from typing import List, Optional

import torch

from megatron.core import tensor_parallel
from megatron.core.config_logger import has_config_logger_enabled, log_config_to_disk
from megatron.core.extensions.transformer_engine import HAVE_TE
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.models.gpt import GPTModel
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.models.vision.clip_vit_model import CLIPViTModel, get_num_image_embeddings
from megatron.core.models.vision.multimodal_projector import MultimodalProjector
from megatron.core.models.vision.radio import RADIOViTModel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer import MegatronModule
from megatron.core.transformer.attention import SelfAttentionSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayerSubmodules
from megatron.core.utils import deprecate_inference_params, is_te_min_version, log_single_rank

if HAVE_TE:
    from megatron.core.extensions.transformer_engine import TEDotProductAttention

    try:
        import transformer_engine_torch as tex

        HAVE_TEX = True
    except ImportError:
        tex = None
        HAVE_TEX = False
else:
    TEDotProductAttention = None
    tex = None
    HAVE_TEX = False


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
        image_token_index (int): Token ID for image token such as <image>.
        pixel_shuffle (bool): Enable pixel shuffle.
        conv_merging (bool): Account for a native 2x2 vision-token merger.
        tile_tags (list): Optional tile tags.
        pg_collection (ProcessGroupCollection): Model communication process groups.
        vp_stage (int): Virtual pipeline stage.
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
        hybrid_layer_pattern: str = None,
        fp16_lm_cross_entropy: bool = False,
        image_token_index: int = DEFAULT_IMAGE_TOKEN_INDEX,
        pixel_shuffle: bool = False,
        conv_merging: bool = False,
        tile_tags: Optional[list] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
        max_num_tiles: int = 0,
        tokenizer_type: str = "",
        vp_stage: Optional[int] = None,
        use_vision_backbone_fp8_arch: bool = False,
        dynamic_resolution: bool = False,
        class_token_len: Optional[int] = None,
        radio_force_eval_mode: bool = False,
        radio_force_cpe_eval_mode: bool = False,
        radio_interpolate_only_cpe: bool = False,
        radio_cpe_aspect_ratio_select: bool = False,
        radio_disable_cpe: bool = False,
        # Audio/video params kept for API compatibility with upstream LLaVAModel.
        # Not exercised by the VLM inference path this PR adds; they are accepted
        # and stored on ``self`` but do not otherwise affect behavior here.
        sound_model: Optional[torch.nn.Module] = None,
        sound_projection: Optional[torch.nn.Module] = None,
        sound_token_index: int = DEFAULT_SOUND_TOKEN_INDEX,
        temporal_patch_dim: int = 1,
        separate_video_embedder: bool = False,
        temporal_ckpt_compat: bool = False,
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
        self.vp_stage = vp_stage
        self._dynamic_resolution = dynamic_resolution
        self.patch_dim = patch_dim
        self._conv_merging = conv_merging

        self.encoder_hidden_state = None
        self.vision_model = None
        self.vision_projection = None
        self.language_model = None
        self._vision_projection_input_size = None

        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        self.pg_collection = pg_collection

        language_model_type = getattr(language_transformer_config, "language_model_type", "")
        self.sequence_parallel_lm = language_transformer_config.sequence_parallel
        self.tp_comm_overlap_lm = language_transformer_config.tp_comm_overlap
        self.context_parallel_lm = language_transformer_config.context_parallel_size
        if self.sequence_parallel_lm or self.context_parallel_lm > 1:
            if not (
                language_model_type.startswith('nemotron5-hybrid')
                or language_model_type == 'nemotron6-moe'
            ):  # pylint: disable=line-too-long
                assert isinstance(
                    language_transformer_layer_spec.submodules, TransformerLayerSubmodules
                )
                assert isinstance(
                    language_transformer_layer_spec.submodules.self_attention.submodules,
                    SelfAttentionSubmodules,
                )
                attn_submodules = (
                    language_transformer_layer_spec.submodules.self_attention.submodules
                )
                assert (
                    attn_submodules.core_attention == TEDotProductAttention and HAVE_TE
                ), "Sequence/Context Parallelism is supported only with TE DotProductAttention."
            if self.context_parallel_lm > 1:
                self.cp_group = self.pg_collection.cp
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

        if self.add_decoder:
            if getattr(language_transformer_config, "language_model_type", "").startswith("hf://"):
                from megatron.core.models.huggingface.module import build_hf_model

                self.language_model = build_hf_model(
                    language_transformer_config, language_transformer_config.language_model_type
                )
            elif language_model_type.startswith(('nemotron5-hybrid', 'nemotron6-moe')):
                self.language_model = HybridModel(
                    config=language_transformer_config,
                    hybrid_stack_spec=language_transformer_layer_spec,
                    vocab_size=language_vocab_size,
                    max_sequence_length=language_max_sequence_length,
                    parallel_output=parallel_output,
                    position_embedding_type=language_position_embedding_type,
                    pre_process=self.pre_process,
                    hybrid_layer_pattern=hybrid_layer_pattern,
                    post_process=self.post_process,
                    rotary_percent=language_rotary_percent,
                    rotary_base=language_rotary_base,
                    fp16_lm_cross_entropy=fp16_lm_cross_entropy,
                    scatter_embedding_sequence_parallel=False,
                    share_embeddings_and_output_weights=share_embeddings_and_output_weights,
                    pg_collection=self.pg_collection,
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
                    pg_collection=self.pg_collection,
                    vp_stage=self.vp_stage,
                )

            self._language_max_sequence_length = language_max_sequence_length
            self.max_sequence_length = language_max_sequence_length
            self.position_embedding_type = language_position_embedding_type
            self._language_is_pipeline_parallel = (
                language_transformer_config.pipeline_model_parallel_size > 1
            )

            # Newer Transformer Engine versions add _extra_state keys in state_dict when using FP8.
            # Older models may not have _extra_state and can be ignored.
            self.language_model.register_load_state_dict_post_hook(
                _load_state_dict_hook_ignore_extra_state
            )

        # Save the constructor arg before local reassignment shadows it.
        _class_token_len_override = class_token_len
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
                    pg_collection=self.pg_collection,
                    vp_stage=self.vp_stage,
                )
            elif vision_transformer_config.vision_model_type in ("radio", "radio-g", "cradio-g"):
                # TODO: should refactor into model code itself?
                radio_class_token_len = 0
                max_img_h = 0
                max_img_w = 0
                embedder_bias = False
                ln_post_impl = None
                use_mask_token = False

                if vision_transformer_config.vision_model_type == "radio":
                    radio_class_token_len = 8
                    max_img_h = 2048
                    max_img_w = 2048
                    embedder_bias = False
                    ln_post_impl = None
                    use_mask_token = False
                elif vision_transformer_config.vision_model_type == "radio-g":
                    radio_class_token_len = 5
                    max_img_h = 1792
                    max_img_w = 1792
                    embedder_bias = True
                    from megatron.core.extensions.transformer_engine import TENorm

                    ln_post_impl = TENorm
                    use_mask_token = True
                elif vision_transformer_config.vision_model_type == "cradio-g":
                    radio_class_token_len = 8
                    max_img_h = 2048
                    max_img_w = 2048
                    embedder_bias = False
                    ln_post_impl = None
                    use_mask_token = False

                # Allow overriding class_token_len from constructor arg.
                if _class_token_len_override is not None:
                    radio_class_token_len = _class_token_len_override

                if vision_transformer_config.fp8 or use_vision_backbone_fp8_arch:
                    # FP8 padding for final sequence length to be a multiple of 16 or 32.
                    radio_class_token_len = (
                        32 if vision_transformer_config.fp8_recipe == "mxfp8" else 16
                    )

                self.vision_model = RADIOViTModel(
                    vision_transformer_config,
                    vision_transformer_layer_spec,
                    ln_post_impl=ln_post_impl,
                    img_h=img_h,
                    img_w=img_w,
                    max_img_h=max_img_h,
                    max_img_w=max_img_w,
                    class_token_len=radio_class_token_len,
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
                    pg_collection=self.pg_collection,
                    vp_stage=self.vp_stage,
                )
            elif vision_transformer_config.vision_model_type in (
                "pixtral-vit",
                "pixtral-vit-large",
                "qwen-vl",
                "kimi-vit",
            ):
                from megatron.core.models.vision.vit_model import (
                    KimiViTModel,
                    QwenVLViTModel,
                    ViTModel,
                )

                add_class_token = False
                class_token_len = 0
                vmt = vision_transformer_config.vision_model_type
                if vmt in ("pixtral-vit", "pixtral-vit-large"):
                    self.vision_model = ViTModel(
                        transformer_config=vision_transformer_config,
                        transformer_layer_spec=vision_transformer_layer_spec,
                        patch_dim=patch_dim,
                        img_h=img_h,
                        img_w=img_w,
                        pos_emb_type='rope2d',
                        ln_pre=True,
                        use_merger=(vmt == "pixtral-vit-large"),
                        pg_collection=self.pg_collection,
                        vp_stage=self.vp_stage,
                    )
                elif vmt == "qwen-vl":
                    num_pos_per_side = int(
                        getattr(vision_transformer_config, 'num_position_embeddings', 2304) ** 0.5
                    )
                    self.vision_model = QwenVLViTModel(
                        transformer_config=vision_transformer_config,
                        transformer_layer_spec=vision_transformer_layer_spec,
                        patch_dim=patch_dim,
                        img_h=img_h,
                        img_w=img_w,
                        num_pos_per_side=num_pos_per_side,
                        pg_collection=self.pg_collection,
                        vp_stage=self.vp_stage,
                    )
                elif vmt == "kimi-vit":
                    pos_h = getattr(vision_transformer_config, 'pos_embed_height', 64)
                    pos_w = getattr(vision_transformer_config, 'pos_embed_width', 64)
                    self.vision_model = KimiViTModel(
                        transformer_config=vision_transformer_config,
                        transformer_layer_spec=vision_transformer_layer_spec,
                        patch_dim=patch_dim,
                        img_h=img_h,
                        img_w=img_w,
                        pos_embed_height=pos_h,
                        pos_embed_width=pos_w,
                        pg_collection=self.pg_collection,
                        vp_stage=self.vp_stage,
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

            # Native mcore encoders also support dynamic pre-patchified inputs.
            # RADIO stores this flag internally; setting it here keeps the
            # downstream packed-sequence path uniform across encoder families.
            self.vision_model.dynamic_resolution = dynamic_resolution

            self.vision_model.register_load_state_dict_post_hook(
                _load_state_dict_hook_ignore_extra_state
            )

            vision_encoder_output_size = getattr(
                self.vision_model, 'out_hidden_size', vision_transformer_config.hidden_size
            )
            vision_projection_input_size = vision_encoder_output_size
            vision_projection_input_size *= 4 if pixel_shuffle else 1
            self._vision_projection_input_size = vision_projection_input_size

            # Map (intermediate) vision model outputs to the language model input dimension.
            self.vision_projection = MultimodalProjector(
                vision_projection_config,
                vision_projection_layer_spec,
                vision_projection_type,
                vision_projection_input_size,
                tp_group=self.pg_collection.tp,
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

        self.img_seq_len = get_num_image_embeddings(
            img_h,
            img_w,
            patch_dim,
            vision_transformer_config.vision_model_type,
            drop_vision_class_token,
            class_token_len,
            pixel_shuffle,
            tile_tags is not None,  # Tile tags enabled/disabled.
            max_num_tiles,
            tokenizer_type,
        )

        self.image_token_index = image_token_index
        self._pixel_shuffle = pixel_shuffle
        self._tile_tags = tile_tags
        self._max_num_tiles = max_num_tiles
        self.patch_dim = patch_dim
        self._class_token_len = class_token_len

        # Audio/video attributes kept for API compatibility with upstream. The
        # VLM inference path in this PR does not exercise them.
        self.sound_model = sound_model
        self.sound_projection = sound_projection
        self.sound_token_index = sound_token_index
        self.temporal_patch_dim = temporal_patch_dim
        self.separate_video_embedder = separate_video_embedder
        self.temporal_ckpt_compat = temporal_ckpt_compat

    @property
    def decoder(self):
        """Expose the language model's decoder for inference utilities."""
        return self.language_model.decoder

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

        if self.add_encoder and self.vision_model is not None and self.add_decoder:
            self.vision_model.set_input_tensor(input_tensor[0])
        elif self.add_encoder and self.vision_model is not None:
            self.vision_model.set_input_tensor(input_tensor[0])
        elif self.pre_process:
            self.encoder_hidden_state = input_tensor[0]
        else:
            self.language_model.set_input_tensor(input_tensor[0])

    def freeze(
        self,
        freeze_language_model: bool,
        freeze_vision_model: bool,
        freeze_vision_projection: bool,
        freeze_sound_model: bool = False,
        freeze_sound_projection: bool = False,
    ):
        """Freeze model modules.

        Make specific modules non-trainable by setting requires_grad to False.

        Args:
            freeze_language_model (bool): Freeze the language model module.
            freeze_vision_model (bool): Freeze the vision model module.
            freeze_vision_projection (bool): Freeze the vision projection module.
            freeze_sound_model (bool): Freeze the sound model module.
            freeze_sound_projection (bool): Freeze the sound projection module.
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
            for param in module.parameters():
                param.requires_grad = False

    def _vision_projection_dtype(self) -> torch.dtype:
        for module in (self.vision_projection, self.vision_model):
            if module is None:
                continue
            for param in module.parameters():
                return param.dtype
        return torch.float32

    def _build_zero_projection_anchor(self, device):
        if self.vision_projection is None or self._vision_projection_input_size is None:
            return None
        if not any(param.requires_grad for param in self.vision_projection.parameters()):
            return None

        projection_input = torch.zeros(
            1,
            1,
            self._vision_projection_input_size,
            dtype=self._vision_projection_dtype(),
            device=device,
        )
        return self.vision_projection(projection_input)

    def _preprocess_data(
        self,
        image_embeddings,
        language_embeddings,
        input_ids,
        position_ids,
        loss_mask,
        labels,
        use_inference_kv_cache,
        inference_context,
        image_token_index,
        num_image_tiles,
        imgs_sizes=None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
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
            final_embedding (torch.Tensor): image and text embeddings [combined_seq_len, b, h] if
                context_parallel_lm == 1 else [b, combined_seq_len, h].
            final_labels (torch.Tensor): labels for image and text positions [b, combined_seq_len].
            final_loss_mask (torch.Tensor): loss mask [b, combined_seq_len].
            final_input_ids (torch.Tensor): token ids expanded to match combined embeddings.
            final_position_ids (torch.Tensor): position ids expanded to match combined embeddings.
        """

        inference_context = deprecate_inference_params(inference_context, inference_params)

        assert self.add_decoder, "input text preprocessing is only needed for the language model"

        # No pre- or postprocessing needed.
        # With pipeline parallel > 2, this means a chunk in the middle of the model.
        if not self.pre_process and not self.post_process:
            return None, None, None, None, None

        # If using the inference KV cache, the image tokens are already computed.
        if use_inference_kv_cache:
            return language_embeddings, labels, loss_mask, input_ids, position_ids

        if num_image_tiles.numel() == 0:
            final_embedding = None
            if self.pre_process:
                final_embedding = language_embeddings
                if image_embeddings is not None and image_embeddings.numel() > 0:
                    final_embedding = final_embedding + (image_embeddings.sum() * 0).to(
                        dtype=final_embedding.dtype
                    )
                if self.context_parallel_lm == 1:
                    final_embedding = final_embedding.transpose(1, 0).contiguous()
            return final_embedding, labels, loss_mask, input_ids, position_ids

        img_seq_len = self.img_seq_len
        if self._dynamic_resolution and imgs_sizes is not None:
            # Per-tile token counts for dynamic resolution.
            img_seq_len = torch.prod(imgs_sizes // self.patch_dim, dim=-1, dtype=torch.int32) + (
                0 if self._drop_vision_class_token else self.vision_model.class_token_len
            )
            if self._pixel_shuffle:
                img_seq_len = (img_seq_len * (0.5**2)).int()
            if self._conv_merging:
                img_seq_len = (img_seq_len * (0.5**2)).int()
            # Group per-tile counts into per-image totals.
            if self._max_num_tiles > 1:
                out_img_seq_len = torch.zeros_like(num_image_tiles)
                start = 0
                for i, c in enumerate(num_image_tiles):
                    out_img_seq_len[i] = torch.sum(img_seq_len[start : start + c])
                    start += c
                img_seq_len = out_img_seq_len.to(input_ids.device)
            else:
                img_seq_len = img_seq_len.to(input_ids.device)

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

            # If num_image_tiles is empty but <image> tokens exist, the images were
            # all filtered (e.g. too small for dynamic resolution -> 0 patches).
            # Fall back to text-only: strip <image> tokens from the mask so the
            # split below doesn't crash.
            if num_image_tiles.numel() == 0 and num_images_per_sample.sum() > 0:
                image_token_mask = torch.zeros_like(image_token_mask)
                num_images_per_sample = torch.zeros_like(num_images_per_sample)

            # Number of tiles per sample.
            num_image_tiles_batch = num_image_tiles.split(num_images_per_sample.tolist(), dim=0)
            num_image_tiles_batch = torch.tensor(
                [x.sum() for x in num_image_tiles_batch], device=input_ids.device
            )

            # Sequence length for each sample is the image sequence length multiplied by
            # the number of tiles for that image, minus image token indices,
            # plus text sequence length.
            if self._dynamic_resolution and imgs_sizes is not None:
                packed_length_per_batch = torch.sum(img_seq_len, dim=-1)
                seq_lens = packed_length_per_batch - num_images_per_sample + text_seq_len
            else:
                seq_lens = (
                    num_image_tiles_batch * img_seq_len - num_images_per_sample + text_seq_len
                )
            max_seq_len = seq_lens.max()
            # Pipeline parallel expects fixed input size. Check if we need to pad.
            if (
                self._language_is_pipeline_parallel
                and max_seq_len < self._language_max_sequence_length
                and inference_context is None
            ):
                max_seq_len = self._language_max_sequence_length

            batch_indices, non_image_indices = torch.where(image_token_mask != True)

            # New position ids for the text tokens, shifted by the image sequence length.
            # E.g. for input_ids = [-200, 1, 2, 3] and img_seq_len = 576, we get
            # new_position_ids = [576, 577, 578, 579]. text_position_ids are then [577, 578, 579].
            image_token_mask_lens = image_token_mask.int().clone()
            # -1 is for the removed image token index.
            if self._dynamic_resolution and imgs_sizes is not None:
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

            final_input_ids = torch.zeros(
                batch_size, max_seq_len, dtype=input_ids.dtype, device=input_ids.device
            )
            final_input_ids[batch_indices, text_position_ids] = input_ids[
                batch_indices, non_image_indices
            ]
            # position_ids may be None (e.g. text-free callers, or the dynamic
            # inference path that supplies decoder_input directly). In that case
            # we don't need position ids for the combined sequence either.
            # TODO: image-aware positions (new_position_ids computed above) are
            # not yet propagated into the combined sequence; arange is only
            # correct for RoPE-with-decoder_input where the LM ignores
            # position_ids. For learned_absolute / mrope, the combined-sequence
            # position ids should be derived from new_position_ids instead.
            if position_ids is None:
                final_position_ids = None
            else:
                final_position_ids = (
                    torch.arange(max_seq_len, dtype=position_ids.dtype, device=position_ids.device)
                    .unsqueeze(0)
                    .expand(batch_size, -1)
                    .contiguous()
                )

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
            # NOTE: DDP/FSDP can hang with text-only samples because vision projection
            # params have no gradient path. Workaround: add a zero-contribution from
            # image_embeddings so they participate in the backward graph.
            if num_image_tiles.shape[0] == 0 and image_embeddings.shape[0] > 0:
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
        if final_input_ids.shape[1] > self._language_max_sequence_length:
            final_input_ids = final_input_ids[:, : self._language_max_sequence_length]
            final_position_ids = final_position_ids[:, : self._language_max_sequence_length]

        return final_embedding, final_labels, final_loss_mask, final_input_ids, final_position_ids

    def _process_embedding_token_parallel(
        self, combined_embeddings, expanded_labels, expanded_loss_mask, packed_seq_params
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
            expanded_labels (torch.Tensor): Distributed labels for image and text positions.
            expanded_loss_mask (torch.Tensor): Distributed loss mask.
            packed_seq_params (PackedSeqParams): Dict with padded token information.

        """

        # No pre or post processing needed with PP middle chunks.
        if not self.pre_process and not self.post_process:
            return combined_embeddings, expanded_labels, expanded_loss_mask, packed_seq_params

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
                shard_factor is not None and seq_dim is not None
            ), "_process_embedding_token_parallel called without SP or CP enabled"

            # VLM combined embeddings (text + vision tokens) may not align
            # naturally to shard_factor under dynamic resolution — the number of
            # image tokens is data-dependent. Pad up to a multiple of
            # shard_factor so SP/CP can shard evenly. Zero for embeddings, -100
            # for labels (ignore-index), 0 for the loss mask.
            seq_len = combined_embeddings.shape[seq_dim]
            pad_len = (shard_factor - seq_len % shard_factor) % shard_factor
            if pad_len > 0:
                pad_shape = list(combined_embeddings.shape)
                pad_shape[seq_dim] = pad_len
                combined_embeddings = torch.cat(
                    [
                        combined_embeddings,
                        torch.zeros(
                            pad_shape,
                            dtype=combined_embeddings.dtype,
                            device=combined_embeddings.device,
                        ),
                    ],
                    dim=seq_dim,
                )
                if self.post_process and expanded_labels is not None:
                    expanded_labels = torch.nn.functional.pad(
                        expanded_labels, (0, pad_len), value=-100
                    )
                if self.post_process and expanded_loss_mask is not None:
                    expanded_loss_mask = torch.nn.functional.pad(
                        expanded_loss_mask, (0, pad_len), value=0
                    )

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
                batch["expanded_labels"] = expanded_labels
                batch["expanded_loss_mask"] = expanded_loss_mask
            # Distribute sequence across CP ranks
            if packed_seq_params is None or packed_seq_params.qkv_format == 'sbhd':
                from megatron.core.utils import get_batch_on_this_cp_rank

                batch = get_batch_on_this_cp_rank(
                    batch, is_hybrid_cp=False, cp_group=self.pg_collection.cp
                )
            else:
                assert HAVE_TEX and is_te_min_version(
                    "1.10.0"
                ), "Please update Transformer Engine to >= 1.10 to use \
                    Context Parallel with THD format data"
                index = tex.thd_get_partitioned_indices(
                    packed_seq_params.cu_seqlens_q_padded,
                    batch[next(iter(batch))].size(1),
                    self.cp_group.size(),
                    self.cp_group.rank(),
                )
                for key, data in batch.items():
                    batch[key] = data.index_select(1, index)

            if self.pre_process:
                combined_embeddings = batch["combined_embeddings"]  # [B, S/CP, H]
                combined_embeddings = combined_embeddings.transpose(
                    1, 0
                ).contiguous()  # [B,S/CP,H] -> [S/CP,B,H]
            if self.post_process:
                expanded_labels = batch["expanded_labels"]
                expanded_loss_mask = batch["expanded_loss_mask"]

        if self.sequence_parallel_lm and self.pre_process:
            combined_embeddings = tensor_parallel.scatter_to_sequence_parallel_region(
                combined_embeddings
            )  # [S/(CP*TP),B,H]

        return combined_embeddings, expanded_labels, expanded_loss_mask, packed_seq_params

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
        inference_context: Optional[BaseInferenceContext] = None,
        num_image_tiles: Optional[List[int]] = None,
        image_token_index: Optional[int] = None,
        runtime_gather_output: Optional[bool] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        imgs_sizes: Optional[torch.Tensor] = None,
        vision_packed_seq_params: Optional[PackedSeqParams] = None,
        # Audio/video params kept for API compatibility with the upstream
        # LLaVAModel.forward signature. The VLM inference path this PR adds does
        # not consume them; matching stubs exist on the constructor (sound_model,
        # sound_projection, sound_token_index, temporal_patch_dim,
        # separate_video_embedder, temporal_ckpt_compat).
        sound_clips: Optional[torch.Tensor] = None,
        sound_length: Optional[torch.Tensor] = None,
        sound_timestamps: Optional[torch.Tensor] = None,
        num_sound_clips: Optional[List[int]] = None,
        num_frames: Optional[int] = None,
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

        inference_context = deprecate_inference_params(inference_context, inference_params)

        use_inference_kv_cache = (
            inference_context is not None
            and hasattr(inference_context, 'key_value_memory_dict')
            and "image_tokens_count" in inference_context.key_value_memory_dict
        )
        has_images = images is not None and images.shape[0] > 0

        # If running inference, we can skip image token computation
        # if they were computed already earlier for this sample.
        if use_inference_kv_cache:
            image_embeddings = None
        elif self.add_encoder and not has_images:
            # Keep trainable projection/adapters in the graph for DDP even when
            # Energon drops every image from this microbatch.
            image_device = images.device if images is not None else input_ids.device
            image_embeddings = self._build_zero_projection_anchor(image_device)
        elif self.add_encoder and has_images:
            # Build packed_seq_params for dynamic-resolution vision fprop.
            if (
                vision_packed_seq_params is None
                and imgs_sizes is not None
                and getattr(self.vision_model, 'dynamic_resolution', False)
            ):
                patch_dim = self.vision_model.patch_dim
                if torch.is_tensor(imgs_sizes):
                    seq_lens = torch.prod(imgs_sizes // patch_dim, dim=-1)
                else:
                    seq_lens = torch.tensor(
                        [(h // patch_dim) * (w // patch_dim) for h, w in imgs_sizes],
                        device=images.device,
                    )
                cu_seqlens = torch.zeros(
                    seq_lens.shape[0] + 1, dtype=torch.int32, device=images.device
                )
                torch.cumsum(seq_lens.int(), dim=0, out=cu_seqlens[1:])
                max_seqlen = int(seq_lens.max().item())
                vision_packed_seq_params = PackedSeqParams(
                    qkv_format='thd',
                    cu_seqlens_q=cu_seqlens,
                    cu_seqlens_kv=cu_seqlens,
                    max_seqlen_q=max_seqlen,
                    max_seqlen_kv=max_seqlen,
                )

            image_embeddings = self.vision_model(
                images, imgs_sizes=imgs_sizes, packed_seq_params=vision_packed_seq_params
            )  # [num_tiles, img_seq_len, h_vision]

            if self._drop_vision_class_token:
                if (
                    imgs_sizes is not None
                    and getattr(self.vision_model, 'dynamic_resolution', False)
                    and self.vision_model.class_token_len > 0
                ):
                    # Class tokens are interleaved between tiles; build mask to remove them.
                    remove_mask = torch.full(
                        (image_embeddings.shape[-2],),
                        True,
                        dtype=torch.bool,
                        device=image_embeddings.device,
                    )
                    patch_dim = self.vision_model.patch_dim
                    if torch.is_tensor(imgs_sizes):
                        seq_lens = torch.prod(
                            imgs_sizes.to(device=image_embeddings.device) // patch_dim, dim=-1
                        )
                    else:
                        seq_lens = torch.tensor(
                            [(h // patch_dim) * (w // patch_dim) for h, w in imgs_sizes],
                            device=image_embeddings.device,
                        )
                    seq_lens = seq_lens.to(torch.long)
                    class_token_len = self.vision_model.class_token_len
                    segment_starts = torch.cumsum(
                        torch.cat([seq_lens.new_zeros(1), seq_lens + class_token_len]), dim=0
                    )[:-1]
                    class_offsets = segment_starts.unsqueeze(1) + torch.arange(
                        class_token_len, device=image_embeddings.device, dtype=seq_lens.dtype
                    ).unsqueeze(0)
                    remove_mask[class_offsets.reshape(-1)] = False
                    image_embeddings = image_embeddings[:, remove_mask, :]
                else:
                    image_embeddings = image_embeddings[:, self.vision_model.class_token_len :, :]

            if self._pixel_shuffle:
                if imgs_sizes is not None and getattr(
                    self.vision_model, 'dynamic_resolution', False
                ):
                    image_embeddings = pixel_shuffle_dynamic_res(
                        image_embeddings, imgs_sizes, self.vision_model.patch_dim
                    )
                else:
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
            if inference_context is not None and hasattr(
                inference_context, 'key_value_memory_dict'
            ):  # pylint: disable=line-too-long
                inference_context.key_value_memory_dict["image_tokens_count"] = (
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

        # if context_parallel_lm == 1:
        #   [combined_seq_len, b, h_language], [b, combined_seq_len], [b, combined_seq_len]
        # else:
        #   [b, combined_seq_len, h_language], [b, combined_seq_len], [b, combined_seq_len]
        (
            combined_embeddings,
            expanded_labels,
            expanded_loss_mask,
            combined_input_ids,
            combined_position_ids,
        ) = self._preprocess_data(
            image_embeddings,
            language_embeddings,
            input_ids,
            position_ids,
            loss_mask,
            labels,
            use_inference_kv_cache,
            inference_context,
            image_token_index if image_token_index is not None else self.image_token_index,
            num_image_tiles,
            imgs_sizes=imgs_sizes,
        )  # [combined_seq_len, b, h_language], [b, combined_seq_len], [b, combined_seq_len]

        # Rebuild packed_seq_params to match post-truncation tensor dims.
        # _preprocess_data expands image placeholders into full embeddings and
        # truncates to _language_max_sequence_length. The dataloader's
        # cu_seqlens were computed in pre-truncation space, so they may exceed
        # the actual sequence length. Clamp and deduplicate so downstream
        # layers (Mamba seq_idx, attention masks) see valid boundaries.
        if packed_seq_params is not None and combined_embeddings is not None:
            if self.context_parallel_lm == 1:
                actual_seq_len = combined_embeddings.shape[0]  # [S, B, H]
            else:
                actual_seq_len = combined_embeddings.shape[1]  # [B, S, H]

            cu = packed_seq_params.cu_seqlens_q
            if cu is not None and cu[-1] > actual_seq_len:
                cu = cu.clamp(max=actual_seq_len)
                # Remove duplicate consecutive values from clamping.
                keep = torch.ones(len(cu), dtype=torch.bool, device=cu.device)
                keep[1:] = cu[1:] != cu[:-1]
                cu = cu[keep]
                if cu[-1] != actual_seq_len:
                    cu = torch.cat([cu, cu.new_tensor([actual_seq_len])])
                packed_seq_params = PackedSeqParams(
                    qkv_format=packed_seq_params.qkv_format,
                    cu_seqlens_q=cu,
                    cu_seqlens_kv=cu,
                    max_seqlen_q=(cu[1:] - cu[:-1]).max().item(),
                    max_seqlen_kv=(cu[1:] - cu[:-1]).max().item(),
                )

        if self.context_parallel_lm > 1 or self.sequence_parallel_lm:
            combined_embeddings, expanded_labels, expanded_loss_mask, packed_seq_params = (
                self._process_embedding_token_parallel(
                    combined_embeddings, expanded_labels, expanded_loss_mask, packed_seq_params
                )
            )

        language_model_kwargs = {
            "input_ids": combined_input_ids,
            "position_ids": combined_position_ids,
            "attention_mask": attention_mask,
            "decoder_input": combined_embeddings,
            "labels": expanded_labels,
            "inference_context": inference_context,
            "runtime_gather_output": runtime_gather_output,
            "packed_seq_params": packed_seq_params,
        }
        if isinstance(self.language_model, (GPTModel, HybridModel)):
            # MTP is a training-time feature (multi-token prediction loss). Only
            # pass loss_mask / mtp_source_loss_mask to the language model when
            # labels are present (i.e. we're computing loss); at inference we
            # skip them so we don't collide with LM forward signatures that
            # don't accept the MTP kwargs.
            if expanded_labels is not None:
                language_model_kwargs["loss_mask"] = expanded_loss_mask
                language_model_kwargs["mtp_source_loss_mask"] = expanded_loss_mask

        output = self.language_model(**language_model_kwargs)

        return output, expanded_loss_mask

    def forward_lm_only(
        self,
        combined_embeddings,
        attention_mask=None,
        labels=None,
        inference_context=None,
        runtime_gather_output=None,
        packed_seq_params=None,
    ):
        """Forward pre-combined embeddings through the language model only.

        Callers provide ``combined_embeddings`` in batch-first
        ``[batch, sequence, hidden]`` layout. This method owns the conversion
        to the language model's sequence-first layout and any SP/CP sharding
        when configured. Bypasses the vision encoder and ``_preprocess_data``.

        Args:
            combined_embeddings: [batch, seq_len, hidden] or None on non-first PP stages.
            attention_mask: Optional attention mask.
            labels: Optional target labels.
            inference_context: Inference context (KV cache).
            runtime_gather_output: Whether to gather output across TP ranks.
            packed_seq_params: Packed sequence parameters.
        """
        if combined_embeddings is not None:
            if self.context_parallel_lm == 1:
                combined_embeddings = combined_embeddings.transpose(0, 1).contiguous()
            if self.context_parallel_lm > 1 or self.sequence_parallel_lm:
                combined_embeddings, _, _, _ = self._process_embedding_token_parallel(
                    combined_embeddings, None, None, packed_seq_params
                )

        try:
            from megatron.core.models.mamba.mamba_model import MambaModel

            is_mamba = isinstance(self.language_model, MambaModel)
        except ImportError:
            is_mamba = False

        if is_mamba:
            output = self.language_model(
                input_ids=None,
                position_ids=None,
                attention_mask=attention_mask,
                decoder_input=combined_embeddings,
                labels=labels,
                inference_context=inference_context,
                runtime_gather_output=runtime_gather_output,
            )
        else:
            output = self.language_model(
                input_ids=None,
                position_ids=None,
                attention_mask=attention_mask,
                decoder_input=combined_embeddings,
                labels=labels,
                inference_context=inference_context,
                runtime_gather_output=runtime_gather_output,
                packed_seq_params=packed_seq_params,
            )

        return output


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


def pixel_shuffle_dynamic_res(x, imgs_sizes, patch_dim, scale_factor=0.5, version=2):
    """Pixel shuffle for dynamic resolution (variable tile sizes).

    Splits the packed sequence by per-tile lengths, applies pixel shuffle to each tile,
    then re-concatenates.

    Args:
        x (torch.Tensor): Vision model outputs [batch, total_seq_len, h_vision]
        imgs_sizes (torch.Tensor): Per-tile (H, W) pixel sizes [num_tiles, 2]
        patch_dim (int): Patch size used by the vision encoder
        scale_factor (float): Pixel shuffle scale factor
        version (int): Implementation version

    Returns:
        x (torch.Tensor): Shuffled outputs [batch, total_shuffled_seq_len, h_vision / (scale**2)]
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
