# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
from functools import partial
import warnings
import logging
from copy import deepcopy

import torch

from config import (
    get_language_model_config,
    get_vision_model_config,
    get_vision_projection_config,
    get_sound_model_config,
    get_sound_projection_config,
)
from layer_specs import (get_layer_spec, get_layer_spec_te, get_mlp_module_spec, get_norm_mlp_module_spec_te,
                         get_mamba_layer_spec_te)

from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_decoder_block_spec,
)
from megatron.core.models.gpt.heterogeneous.heterogeneous_layer_specs import get_gpt_heterogeneous_layer_spec
from megatron.core.models.multimodal.efficient_video_sampling import EVSVariant
from megatron.core.models.multimodal.llava_model import IMAGE_TOKEN, SOUND_TOKEN, LLaVAModel
from megatron.core.models.vision.multimodal_projector import MultimodalProjector
from megatron.core.models.vision.clip_vit_model import get_num_image_embeddings
from megatron.training import get_args, get_tokenizer, print_rank_0
from megatron.training.arguments import core_transformer_config_from_args
from megatron.core.utils import log_single_rank
from megatron.core.transformer.spec_utils import import_module


def model_provider(
    pre_process=True, post_process=True, add_encoder=True, add_decoder=True, parallel_output=True
) -> LLaVAModel:
    """Builds the model.

    Args:
        pre_process (bool): Include the embedding layer in the gpt decoder (used with pipeline parallelism). Defaults to True.
        post_process (bool): Include an output layer and a layernorm in the gpt decoder (used with pipeline parallelism). Defaults to True.
        add_encoder (bool): Construct the encoder module (used with pipeline parallelism). Defaults to True. When we use pipelining, the encoder
            will live on only a subset of the pipeline stages (specifically, only the first stage).
        add_decoder (bool): Construct the decoder module (used with pipeline parallelism). Defaults to True. When we use pipelining, the decoder
            will live on only a subset of the pipeline stages (specifically, every stage after the first one).
        parallel_output (bool): Enable parallel model output.

    Returns:
        model: A multimodal model.
    """
    args = get_args()

    # Deprecation warning for encoder pipeline parallelism
    if args.encoder_pipeline_model_parallel_size > 0 or args.encoder_tensor_model_parallel_size > 0:
        warnings.warn(
            "Encoder-specific pipeline parallelism functionality is deprecated and will be removed in core_r0.14.0. "
            "This includes the parameters 'encoder_tensor_model_parallel_size' and 'encoder_pipeline_model_parallel_size', "
            "as well as all associated encoder pipeline parallel logic and infrastructure. "
            "This functionality is being replaced by the new 'orthotope' parallelism management system, which provides "
            "a more general and flexible approach to handling complex parallelism configurations including encoder-decoder models. "
            "Please refrain from building new features or dependencies on encoder pipeline parallelism as this entire "
            "capability will not be supported in future releases. For migration guidance and information on the orthotope "
            "system, please refer to the Megatron-LM documentation.",
            DeprecationWarning,
            stacklevel=2
        )

    assert args.encoder_pipeline_model_parallel_size <= 1, "LLaVA does not support pp>1 for encoder on it's own pipeline rank"

    use_te = args.use_te

    print_rank_0('building a multimodal model ...')
    if args.dynamic_resolution:
        max_num_image_embeddings = args.seq_length
        num_image_embeddings = args.seq_length
        if args.pixel_shuffle:
            max_num_image_embeddings //= 4
            num_image_embeddings //= 4
        elif args.conv_merging:
            max_num_image_embeddings //= 4
            num_image_embeddings //= 4
    else:
        # For tiling only, need a fixed number of embeddings for num_image_embeddings_per_tile
        # We don't pass in `is_video` to calculate `self.num_image_embeddings_per_tile`,
        #   so we currently require no temporal compression because we can't verify the number of frames
        video_temporal_patch_size=getattr(args, 'video_temporal_patch_size', 1)
        if video_temporal_patch_size != 1:
            raise NotImplementedError(
                f"When using tiling, temporal compression is not supported."
                f" Found video_temporal_patch_size={video_temporal_patch_size}."
            )

        num_image_embeddings = get_num_image_embeddings(
            img_h=args.img_h,
            img_w=args.img_w,
            patch_dim=args.patch_dim,
            vision_model_type=args.vision_model_type,
            disable_vision_class_token=args.disable_vision_class_token,
            class_token_len=1,
            pixel_shuffle=args.pixel_shuffle,
            use_tile_tags=args.use_tile_tags,
            max_num_tiles=args.max_num_tiles,
            tokenizer_type=args.tokenizer_prompt_format,
            use_image_break_token=args.image_break_token is not None,
            conv_merging=args.conv_merging,
        )
        old_seq_length = args.seq_length
        args.seq_length = args.encoder_seq_length = num_image_embeddings
        if old_seq_length != args.seq_length:
            log_single_rank(
                logging.getLogger(__name__),
                logging.WARNING,
                f"Changed seq_length and encoder_seq_length (vision model sequence length) from {old_seq_length} to num_image_tokens ({num_image_embeddings})"
            )

        max_num_image_embeddings = max((args.max_num_tiles + int(args.use_thumbnail)), args.num_frames) * num_image_embeddings

    assert (
        args.decoder_seq_length is not None
    ), "Please provide --decoder-seq-length to set the language model sequence length"
    assert (
        args.decoder_seq_length > max_num_image_embeddings
    ), "Language model sequence length must be greater than the maximum number of image embeddings"
    if args.decoder_seq_length > args.max_position_embeddings:
        args.max_position_embeddings = args.decoder_seq_length
        warnings.warn(
            f"Expanded max_position_embeddings to {args.max_position_embeddings} to accommodate the maximum language model sequence length"
        )

    language_model_type = args.language_model_type
    vision_model_type = args.vision_model_type

    base_config = core_transformer_config_from_args(get_args())
    base_config.language_model_type = args.language_model_type
    base_config.vision_model_type = args.vision_model_type
    base_config.sound_model_type = getattr(args, "sound_model_type", None)
    base_config.calculate_per_token_loss = False if getattr(args, "no_calculate_per_token_loss", False) else True
    base_config.no_load_balancing_sequence_scaling = getattr(args, "no_load_balancing_sequence_scaling", False)

    if getattr(args, "no_calculate_per_token_loss", False):
        assert not args.use_loss_scaling, "Cannot disable calculating per-token loss and use loss scaling at the same time, either remove --no-calculate-per-token-loss or --use-loss-scaling"

    language_config, language_transformer_layer_spec = get_language_config_and_spec(base_config)

    if args.heterogeneous_layers_config_path is not None:
        without_hetero = get_args()
        without_hetero.heterogeneous_layers_config_path = None
        without_hetero.heterogeneous_layers_config_encoded_json = None
        vision_config = core_transformer_config_from_args(without_hetero)
        vision_config.language_model_type = args.language_model_type
        vision_config.vision_model_type = args.vision_model_type
        vision_config.calculate_per_token_loss = base_config.calculate_per_token_loss
        vision_config.num_layers_in_first_pipeline_stage = None
        vision_config.num_layers_in_last_pipeline_stage = None
    else:
        vision_config = deepcopy(base_config)

    vision_config = get_vision_model_config(
        vision_config, enable_fusions=args.enable_fusions
    )
    if vision_model_type.startswith("hf://"):
        assert args.encoder_tensor_model_parallel_size < 2, "Huggingface vision encoders do not support --encoder-tensor-model-parallel-size > 1"
        assert args.encoder_pipeline_model_parallel_size == 0, "Huggingface vision encoders do not support --encoder-pipeline-model-parallel-size > 0"
        assert not args.sequence_parallel, "Huggingface models do not support --sequence-parallel"
        assert args.context_parallel_size < 2, "Huggingface models do not support --context-parallel-size > 1"

    if vision_model_type == "radio-g":
        if use_te:
            from radio.radio_g import get_radio_g_layer_spec_te
            vision_transformer_layer_spec = get_radio_g_layer_spec_te()  # TENorm detects LayerNorm/RMS automatically.
        else:
            from radio.radio_g import get_radio_g_layer_spec
            vision_transformer_layer_spec = get_radio_g_layer_spec(
                normalization=vision_config.normalization
            )
    elif vision_model_type in ["clip", "siglip"] or "radio" in vision_model_type:
        if use_te:
            vision_transformer_layer_spec = get_layer_spec_te(
                is_vit=True
            )  # TENorm detects LayerNorm/RMS automatically.
        else:
            vision_transformer_layer_spec = get_layer_spec(
                is_vit=True, normalization=vision_config.normalization
            )
    elif vision_model_type == "internvit":
        from nvlm.internvit import get_internvit_layer_spec
        vision_transformer_layer_spec = get_internvit_layer_spec(use_te=use_te)
    elif vision_model_type == "internvit300M":
        from nvlm.internvit import get_internvit300M_layer_spec
        vision_transformer_layer_spec = get_internvit300M_layer_spec(use_te=use_te)
    elif vision_model_type.startswith("hf://"):
        vision_transformer_layer_spec = None
    else:
        raise RuntimeError("unsupported vision model type", vision_model_type)


    if args.heterogeneous_layers_config_path is not None:
        without_hetero = get_args()
        without_hetero.heterogeneous_layers_config_path = None
        without_hetero.heterogeneous_layers_config_encoded_json = None
        vision_projection_config = core_transformer_config_from_args(without_hetero)
        vision_projection_config.language_model_type = args.language_model_type
        vision_projection_config.vision_model_type = args.vision_model_type
        vision_projection_config.calculate_per_token_loss = base_config.calculate_per_token_loss
    else:
        vision_projection_config = deepcopy(base_config)

    vision_projection_config = get_vision_projection_config(
        vision_projection_config, language_config.hidden_size, enable_fusions=args.enable_fusions
    )

    # --encoder-pipeline-model-parallel-size 1 will enable a separate pipeline stage for the vision model.
    if args.encoder_pipeline_model_parallel_size > 0:
        assert (
            args.encoder_pipeline_model_parallel_size == 1
        ), "vision model and projection can only live on 1 pipeline stage."

        if args.encoder_tensor_model_parallel_size > 0:
            vision_config.tensor_model_parallel_size = args.encoder_tensor_model_parallel_size
            vision_projection_config.tensor_model_parallel_size = (
                args.encoder_tensor_model_parallel_size
            )

    # Make sure vision model pipeline parallel size is not inherited from the language model pipeline parallel size.
    # 0 is not a valid for the config value, hence max(1, ).
    vision_config.pipeline_model_parallel_size = max(1, args.encoder_pipeline_model_parallel_size)
    vision_projection_config.pipeline_model_parallel_size = vision_config.pipeline_model_parallel_size

    # Make sure the vision model does not inherit first and last pipeline num layers from the language model.
    vision_config.num_layers_in_first_pipeline_stage = vision_config.num_layers_in_last_pipeline_stage = None

    if vision_projection_config.normalization:
        vision_projection_layer_spec = get_norm_mlp_module_spec_te().submodules
    else:
        vision_projection_layer_spec = get_mlp_module_spec(use_te=use_te).submodules

    # Toggle --recompute* for the vision and language model separately.
    if args.recompute_vision:
        recompute_vision_num_layers = args.recompute_vision_num_layers if getattr(args, "recompute_vision_num_layers", 0) > 0 else vision_config.num_layers
        vision_config.recompute_num_layers = recompute_vision_num_layers

        if getattr(args, "recompute_granularity_vision", None) is not None:
            vision_config.recompute_granularity = args.recompute_granularity_vision

        if getattr(args, "recompute_method_vision", None) is not None:
            vision_config.recompute_method = args.recompute_method_vision
    else:
        vision_config.recompute_granularity = None
        vision_config.recompute_method = None
        vision_config.recompute_num_layers = None

    # TODO: Vision model and projection do not use CP or TP comm overlap yet.
    vision_config.sequence_parallel = False
    vision_config.context_parallel_size = 1
    vision_config.tp_comm_overlap = False

    vision_projection_config.sequence_parallel = False
    vision_projection_config.context_parallel_size = 1
    vision_projection_config.tp_comm_overlap = False

    # Toggle --recompute* for the vision projection layer.
    if getattr(args, "recompute_vision_projection", False):
        vision_projection_config.recompute_granularity = "full"
    else:
        vision_projection_config.recompute_granularity = None
        vision_projection_config.recompute_method = None
        vision_projection_config.recompute_num_layers = None

    tokenizer = get_tokenizer()
    image_token_index = tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
    assert image_token_index is not None, f"IMAGE_TOKEN={IMAGE_TOKEN} needs to be added using the --special-tokens arg."

    # Validate that image break token is included in special tokens if specified
    if args.image_break_token is not None:
        assert args.image_break_token in args.special_tokens, f"IMAGE_BREAK_TOKEN='{args.image_break_token}' needs to be added to the --special-tokens list."

    # Not used anymore.
    tile_tags = None

    sound_model, sound_projection, sound_token_index = sound_model_provider(base_config, language_config.hidden_size)

    model = LLaVAModel(
        language_transformer_config=language_config,
        language_transformer_layer_spec=language_transformer_layer_spec,
        language_vocab_size=args.padded_vocab_size,
        language_max_sequence_length=args.decoder_seq_length,
        vision_transformer_config=vision_config,
        vision_transformer_layer_spec=vision_transformer_layer_spec,
        drop_vision_class_token=args.disable_vision_class_token,
        vision_projection_config=vision_projection_config,
        vision_projection_layer_spec=vision_projection_layer_spec,
        vision_projection_type="mlp",
        allow_missing_vision_projection_checkpoint=args.allow_missing_vision_projection_checkpoint,
        parallel_output=parallel_output,
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        language_position_embedding_type=args.position_embedding_type,
        language_rotary_percent=args.rotary_percent,
        pre_process=pre_process,
        post_process=post_process,
        add_encoder=add_encoder,
        add_decoder=add_decoder,
        img_h=args.img_h,
        img_w=args.img_w,
        patch_dim=args.patch_dim,
        language_rotary_base=args.rotary_base,
        language_rope_scaling=args.use_rope_scaling,
        hybrid_attention_ratio=args.hybrid_attention_ratio,
        hybrid_mlp_ratio=args.hybrid_mlp_ratio,
        hybrid_override_pattern=args.hybrid_override_pattern,
        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
        image_token_index=image_token_index,
        pixel_shuffle=args.pixel_shuffle,
        tile_tags=tile_tags,
        dynamic_resolution=args.dynamic_resolution,
        max_num_tiles=args.max_num_tiles,
        tokenizer_type=args.tokenizer_prompt_format,
        use_vision_backbone_fp8_arch=args.use_vision_backbone_fp8_arch,
        image_break_token=tokenizer.convert_tokens_to_ids(args.image_break_token) if args.image_break_token is not None else None,
        conv_merging=args.conv_merging,
        allow_missing_conv_merge_checkpoint=args.allow_missing_conv_merge_checkpoint,
        efficient_video_sampling_variant=args.efficient_video_sampling_variant,
        sound_model=sound_model,
        sound_projection=sound_projection,
        sound_token_index=sound_token_index,
        class_token_len=getattr(args, 'class_token_len', None),
        radio_force_eval_mode=getattr(args, "radio_force_eval_mode", False),
        radio_force_cpe_eval_mode=getattr(args, "radio_force_cpe_eval_mode", False),
        radio_interpolate_only_cpe=getattr(args, "radio_interpolate_only_cpe", False),
        radio_cpe_aspect_ratio_select=getattr(args, "radio_cpe_aspect_ratio_select", False),
        radio_disable_cpe=getattr(args, "radio_disable_cpe", False),
        use_loss_scaling=getattr(args, "use_loss_scaling", False),
        log_model_grad_norms=getattr(args, "log_model_grad_norms", False),
        log_model_act_norms=getattr(args, "log_model_act_norms", False),
        video_temporal_patch_size=getattr(args, "video_temporal_patch_size", 1),
        allow_checkpoint_without_temporal_compression=getattr(args, "allow_checkpoint_without_temporal_compression", False),
        separate_video_embedder=getattr(args, "separate_video_embedder", False),
    )

    model.freeze(
        freeze_language_model=args.freeze_LM,
        freeze_vision_model=args.freeze_ViT,
        freeze_vision_projection=getattr(args, "freeze_vision_projection", False),
        freeze_sound_model=getattr(args, "freeze_sound_model", False),
        freeze_sound_projection=getattr(args, "freeze_sound_projection", False),
        unfreeze_router=getattr(args, "unfreeze_router", False),
    )

    return model


def get_language_config_and_spec(base_config):
    """Get the language config and spec."""
    args = get_args()

    use_te = args.use_te

    language_config = deepcopy(base_config)
    language_config = get_language_model_config(
        language_config, args.enable_fusions,
        apply_rope_fusion=not EVSVariant.uses_special_position_ids(args.efficient_video_sampling_variant)
    )
    language_model_type = language_config.language_model_type

    if language_model_type.startswith("hf://"):
        assert args.tensor_model_parallel_size == 1, "Huggingface models do not support --tensor-model-parallel-size > 1"
        assert args.pipeline_model_parallel_size < 2, "Huggingface models do not support --pipeline-model-parallel-size > 1"
        assert not args.sequence_parallel, "Huggingface models do not support --sequence-parallel"
        assert args.context_parallel_size < 2, "Huggingface models do not support --context-parallel-size > 1"

    if language_model_type.startswith("hf://"):
        language_transformer_layer_spec = None
    elif args.heterogeneous_layers_config_path is not None:
        language_transformer_layer_spec = get_gpt_heterogeneous_layer_spec(language_config, use_te)
    elif use_te:
        vp_stage = None
        if args.num_experts:
            if args.spec is not None:
                language_transformer_layer_spec = import_module(args.spec)
            else:
                language_transformer_layer_spec = get_gpt_decoder_block_spec(
                    language_config, use_transformer_engine=use_te, normalization=args.normalization, qk_l2_norm=args.qk_l2_norm, vp_stage=vp_stage
                )
        else:
            # Padding mask needed for SP/CP.
            padding = args.context_parallel_size > 1 and args.sequence_parallel
            if args.language_model_type.startswith('nemotron5-hybrid'):
                language_transformer_layer_spec = get_mamba_layer_spec_te(padding=padding)
            else:
                language_transformer_layer_spec = get_layer_spec_te(
                    is_vit=False, padding=padding
                )  # TENorm detects LayerNorm/RMS automatically.
    else:
        language_transformer_layer_spec = get_layer_spec(
            is_vit=False, normalization=language_config.normalization
        )

    return language_config, language_transformer_layer_spec


def sound_model_provider(base_config, language_hidden_size):
    args = get_args()

    if getattr(args, "sound_model_type", None) is None:
        return None, None, None

    sound_config = deepcopy(base_config)
    sound_config = get_sound_model_config(sound_config)

    sound_projection_config = deepcopy(base_config)

    sound_projection_config = get_sound_projection_config(
        sound_projection_config, language_hidden_size, enable_fusions=args.enable_fusions
    )

    if args.recompute_sound:
        if sound_config.recompute_method is not None and sound_config.recompute_granularity is not None:
            sound_config.recompute_num_layers = sound_config.num_layers
    else:
        sound_config.recompute_granularity = None
        sound_config.recompute_method = None
        sound_config.recompute_num_layers = None

    # Toggle --recompute* for the sound projection layer.
    if getattr(args, "recompute_sound_projection", False):
        sound_projection_config.recompute_granularity = "full"
    else:
        sound_projection_config.recompute_granularity = None
        sound_projection_config.recompute_method = None
        sound_projection_config.recompute_num_layers = None

    sound_config.sound_pad_to_clip_duration = getattr(args, "sound_pad_to_clip_duration", True)
    sound_config.sound_batch_split = getattr(args, "sound_batch_split", 1)

    sound_projection_config.sequence_parallel = False
    sound_projection_config.context_parallel_size = 1
    sound_projection_config.tp_comm_overlap = False

    if sound_projection_config.normalization:
        sound_projection_layer_spec = get_norm_mlp_module_spec_te().submodules
    else:
        sound_projection_layer_spec = get_mlp_module_spec(use_te=args.use_te).submodules

    if sound_config.sound_model_type.startswith("hf://") or sound_config.sound_model_type.startswith("nemo://"):
        from megatron.core.models.huggingface.module import build_hf_model

        sound_model = build_hf_model(
            sound_config, sound_config.sound_model_type
        )
    else:
        raise ValueError(
            "Sound model "
            f"{sound_config.sound_model_type} is not "
            "supported."
        )

    from megatron.core.models.multimodal.llava_model import _load_state_dict_hook_ignore_extra_state, _load_state_dict_hook_ignore_param_names

    sound_model.register_load_state_dict_post_hook(
        _load_state_dict_hook_ignore_extra_state
    )
    sound_projection_input_size = sound_config.hidden_size

    # Map (intermediate) sound model outputs to the language model input dimension.
    sound_projection = MultimodalProjector(
        sound_projection_config,
        sound_projection_layer_spec,
        "mlp",
        sound_projection_input_size,
    )
    # Ignore missing weights for the sound projection during checkpoint loading.
    # This should be disabled by default but can be enabled if your checkpoint contains
    # pretrained sound and language models but not the projection from sound model
    # outputs to language model inputs.
    if args.allow_missing_sound_projection_checkpoint:
        sound_projection_param_names = [
            f"sound_projection.{name}"
            for name in sound_projection.state_dict().keys()
        ]
        sound_projection.register_load_state_dict_post_hook(
            partial(_load_state_dict_hook_ignore_param_names, sound_projection_param_names)
        )

    if args.allow_missing_sound_model_checkpoint:
        sound_model_param_names = [
            f"sound_model.{name}"
            for name in sound_model.state_dict().keys()
        ]
        sound_model.register_load_state_dict_post_hook(
            partial(_load_state_dict_hook_ignore_param_names, sound_model_param_names)
        )

    tokenizer = get_tokenizer()
    sound_token_index = tokenizer.convert_tokens_to_ids(SOUND_TOKEN)

    return sound_model, sound_projection, sound_token_index
