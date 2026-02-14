# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
# Copyright (c) 2025, Qwen3-VL pretraining script.
"""Pretrain Qwen3-VL vision language model.

This script supports training Qwen3-VL models with real multimodal data.
It uses the Qwen3VLModel from megatron.core.models.multimodal which includes:
- HuggingFace Qwen3-VL visual encoder
- DeepStack mechanism for multi-layer visual feature injection
- Proper sequence expansion for image embeddings

Usage:
    torchrun --nproc_per_node=8 pretrain_qwenvl.py \
        --vision-model-type qwen3_vl \
        --hf-model-name Qwen/Qwen3-VL-8B-Instruct \
        --img-h 384 --img-w 384 \
        ...

References:
    - megatron/core/models/multimodal/qwen3_vl_model.py
    - Megatron-LM pretrain_vlm.py for LLaVA architecture
"""
from copy import deepcopy
from functools import partial

import torch

from megatron.core import parallel_state, tensor_parallel
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig
from megatron.core.datasets.multimodal_dataset import MockMultimodalDataset, MultimodalDatasetConfig
from megatron.core.datasets.qwen3vl_dataset import Qwen3VLDataset, Qwen3VLDatasetConfig, Qwen3VLDatasetBuilder
from megatron.core.enums import ModelType
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
)
# [ModelOpt]: Import modelopt layer spec for QAT training
try:
    from megatron.core.post_training.modelopt.gpt.model_specs import get_gpt_modelopt_spec
    has_modelopt_spec = True
except ImportError:
    has_modelopt_spec = False
from megatron.core.models.multimodal.qwen3_vl_model import (
    Qwen3VLModel,
    DEFAULT_IMAGE_TOKEN_INDEX,
    DEFAULT_VIDEO_TOKEN_INDEX,
)
from megatron.core.transformer.spec_utils import import_module
from megatron.training import get_args, get_timers, get_tokenizer, pretrain, print_rank_0
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.utils import get_blend_and_blend_per_split
from pretrain_gpt import loss_func

# [ModelOpt]: Import for loading quantization state from PTQ checkpoints
try:
    from megatron.post_training.checkpointing import load_modelopt_state, has_modelopt_state
    has_modelopt_support = True
except ImportError:
    has_modelopt_support = False

# [ModelOpt]: Import modelopt arguments for QAT training
try:
    from megatron.post_training.arguments import add_modelopt_args
    has_modelopt_args = True
except ImportError:
    has_modelopt_args = False

# [ModelOpt]: Import for knowledge distillation (QAD)
try:
    import modelopt.torch.distill as mtd
    import modelopt.torch.distill.plugins.megatron as mtd_mcore
    import modelopt.torch.opt as mto
    from megatron.post_training.model_builder import (
        _load_teacher_model_config,
    )
    has_kd_support = True
except ImportError:
    has_kd_support = False


class _VLMTeacherWrapper(torch.nn.Module):
    """Temporary wrapper to load a teacher GPTModel from a VLM checkpoint.

    VLM checkpoints store the language model under the 'language_model.' prefix
    (e.g., language_model.embedding.word_embeddings.weight). This wrapper places
    the teacher GPTModel as self.language_model so that sharded_state_dict()
    generates keys matching the checkpoint structure.

    The visual.* keys in the checkpoint are simply not requested (only language_model.*
    keys are generated), so they are ignored during loading.
    """

    def __init__(self, language_model):
        super().__init__()
        self.language_model = language_model

    def sharded_state_dict(self, prefix='', sharded_offsets=(), metadata=None):
        language_prefix = f'{prefix}language_model.'
        return self.language_model.sharded_state_dict(
            prefix=language_prefix,
            sharded_offsets=sharded_offsets,
            metadata=metadata,
        )


def _load_vlm_teacher_model(config, config_raw, model_kwargs):
    """Load teacher GPTModel from a VLM checkpoint.

    Unlike _load_teacher_model in model_builder.py (designed for text-only GPT checkpoints),
    this function handles loading from a VLM checkpoint where the language model weights
    are stored under the 'language_model.' prefix (e.g., language_model.embedding.word_embeddings.weight).

    The teacher is a bare MCoreGPTModel, but the checkpoint was saved from a Qwen3VLModel
    which stores GPTModel as self.language_model. We wrap the teacher in a _VLMTeacherWrapper
    that mirrors the Qwen3VLModel structure, then use Megatron's load_checkpoint which
    handles all the distributed checkpoint complexity (TP resharding, factory merging, etc.).

    Uses a regular (non-modelopt) layer spec because the VLM checkpoint was saved from
    a regular ms-swift mcore model, not a PTQ/modelopt checkpoint.
    """
    from megatron.core.models.gpt import GPTModel as MCoreGPTModel
    from megatron.training.checkpointing import load_checkpoint

    args = get_args()

    # Use regular layer spec (NOT modelopt spec) because the VLM teacher checkpoint
    # was saved from a regular mcore model (ms-swift export), not a PTQ checkpoint.
    if args.transformer_impl == "transformer_engine":
        model_kwargs["transformer_layer_spec"] = get_gpt_layer_with_transformer_engine_spec(
            num_experts=args.num_experts,
            moe_grouped_gemm=args.moe_grouped_gemm,
            qk_layernorm=args.qk_layernorm,
        )
    else:
        model_kwargs["transformer_layer_spec"] = get_gpt_layer_local_spec(
            num_experts=args.num_experts,
            moe_grouped_gemm=args.moe_grouped_gemm,
            qk_layernorm=args.qk_layernorm,
        )
    teacher = MCoreGPTModel(config=config, **model_kwargs)

    teacher_load_path = args.export_kd_teacher_load
    print_rank_0(f"Loading VLM teacher as {type(teacher).__name__} from {teacher_load_path} ...")

    # Wrap teacher in _VLMTeacherWrapper so its sharded_state_dict() generates
    # 'language_model.' prefixed keys matching the VLM checkpoint structure.
    wrapper = _VLMTeacherWrapper(teacher)

    # Temporarily set finetune=True to avoid checkpoint arg/rng validation
    original_finetune = args.finetune
    args.finetune = True

    # Use Megatron's load_checkpoint which handles TP resharding, factory merging,
    # and all the distributed checkpoint complexity automatically.
    # The wrapper's sharded_state_dict() only requests language_model.* keys,
    # so visual.* keys in the checkpoint are silently ignored (strict=False).
    load_checkpoint([wrapper], None, None, strict=False, load_arg='export_kd_teacher_load')

    args.finetune = original_finetune

    # Reset global checkpoint version set by teacher's load_checkpoint.
    # Without this, the student's checkpoint loading would fail with
    # "checkpoint versions do not match" if the versions differ.
    import megatron.training.checkpointing as _ckpt_module
    _ckpt_module._CHECKPOINT_VERSION = None

    print_rank_0("...VLM teacher loaded successfully.")
    return teacher


def model_provider(
    pre_process: bool = True,
    post_process: bool = True,
    parallel_output: bool = True,
    config=None,
    pg_collection=None,
    vp_stage: int = None,
) -> Qwen3VLModel:
    """Build the Qwen3-VL model.

    This uses the Qwen3VLModel from megatron.core.models.multimodal which
    combines a Megatron-LM GPT language model with HuggingFace Qwen3-VL
    visual encoder and supports DeepStack multi-layer visual injection.

    Args:
        pre_process: Include embedding layer (for pipeline parallelism)
        post_process: Include output layer (for pipeline parallelism)
        parallel_output: Enable model parallel output
        config: TransformerConfig passed by Megatron-LM training
        pg_collection: Process group collection
        vp_stage: Virtual pipeline stage

    Returns:
        Qwen3VLModel instance
    """
    args = get_args()

    print_rank_0('building Qwen3-VL model ...')

    # Get transformer config for language model
    # Use provided config if available, otherwise create from args
    if config is not None:
        language_transformer_config = config
    else:
        language_transformer_config = core_transformer_config_from_args(args)

    # Configure heterogeneous distributed checkpoint based on checkpoint type
    # - PTQ checkpoints (from quantize.py) use per-layer format: "layers.0.", "layers.1.", etc.
    # - ms-swift mcore checkpoints use factory-merged format: "layers.self_attention..." with merged tensors
    # Only enable hetereogenous_dist_checkpoint for PTQ checkpoints
    is_ptq_checkpoint = (
        args.load is not None and
        has_modelopt_support and
        has_modelopt_state(args.load)
    )
    if is_ptq_checkpoint:
        language_transformer_config.hetereogenous_dist_checkpoint = True
        print_rank_0('Loading PTQ checkpoint: using hetereogenous_dist_checkpoint=True (per-layer format)')
    else:
        # For ms-swift mcore checkpoints, use factory-merged format
        language_transformer_config.hetereogenous_dist_checkpoint = False
        print_rank_0('Loading mcore checkpoint: using hetereogenous_dist_checkpoint=False (factory-merged format)')

    # Override num_layers if decoder_num_layers is specified
    if hasattr(args, 'decoder_num_layers') and args.decoder_num_layers is not None:
        language_transformer_config.num_layers = args.decoder_num_layers

    # Get transformer layer spec
    # [ModelOpt]: Use modelopt spec when loading from PTQ checkpoint for QAT training
    use_modelopt_spec = getattr(args, 'use_modelopt_spec', False)
    if args.load is not None and has_modelopt_support and has_modelopt_state(args.load):
        if has_modelopt_spec:
            use_modelopt_spec = True
            print_rank_0('Detected PTQ checkpoint, using modelopt layer spec for QAT training')
        else:
            print_rank_0('Warning: PTQ checkpoint detected but modelopt spec not available')

    if args.spec is not None:
        language_transformer_layer_spec = import_module(args.spec)
    elif use_modelopt_spec and has_modelopt_spec:
        # Use modelopt spec for QAT training with quantized checkpoints
        # remap_te_layernorm=True for compatibility with checkpoints saved with --export-te-mcore-model
        language_transformer_layer_spec = get_gpt_modelopt_spec(
            config=language_transformer_config,
            local_core_attention=getattr(args, 'use_local_attention', False),
            remap_te_layernorm=True,
            real_quant_cfg=getattr(args, 'export_real_quant_cfg', 'None'),
            use_arbitrary_attention_mask=True if language_transformer_config.context_parallel_size == 1 else False,
        )
    elif args.transformer_impl == "transformer_engine":
        language_transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
            args.num_experts, args.moe_grouped_gemm
        )
    else:
        language_transformer_layer_spec = get_gpt_layer_local_spec(
            args.num_experts, args.moe_grouped_gemm
        )

    # Vision transformer config (use same as language but with vision-specific settings)
    vision_transformer_config = deepcopy(language_transformer_config)
    if hasattr(args, 'encoder_num_layers') and args.encoder_num_layers is not None:
        vision_transformer_config.num_layers = args.encoder_num_layers
    vision_transformer_config.context_parallel_size = 1  # Force CP=1 for vision

    # When no data path is provided, we run in mock-data mode (no real images).
    # Skip the HuggingFace vision encoder to avoid model downloads.
    has_data = (getattr(args, 'data_path', None) is not None or
                getattr(args, 'per_split_data_args_path', None) is not None)
    add_encoder = parallel_state.is_pipeline_first_stage() and has_data

    # Build the Qwen3-VL model
    model = Qwen3VLModel(
        language_transformer_config=language_transformer_config,
        language_transformer_layer_spec=language_transformer_layer_spec,
        language_vocab_size=args.padded_vocab_size,
        language_max_sequence_length=args.max_position_embeddings,
        vision_transformer_config=vision_transformer_config,
        hf_model_name=args.hf_model_name,
        freeze_vision=args.freeze_ViT,
        freeze_language=args.freeze_LM,
        freeze_lm_embedding=args.freeze_lm_embedding,
        parallel_output=parallel_output,
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        language_position_embedding_type=args.position_embedding_type,
        language_rotary_percent=args.rotary_percent,
        language_rotary_base=args.rotary_base,
        pre_process=pre_process,
        post_process=post_process,
        add_encoder=add_encoder,
        add_decoder=True,
        img_h=args.img_h,
        img_w=args.img_w,
        image_token_index=args.image_token_index,
        video_token_index=args.video_token_index,
        num_deepstack_layers=args.num_deepstack_layers,
        pg_collection=pg_collection,
        vp_stage=vp_stage,
    )

    # [ModelOpt]: Load modelopt state from PTQ checkpoint for QAT training
    # This converts the model to have quantization-aware modules before weights are loaded.
    # The quantization config (in modelopt_state) specifies which layers to quantize.
    # For VLM, only the language_model is quantized (visual encoder is not quantized).
    #
    # The main checkpoint has keys like: language_model.decoder.layers.0.*._extra_state
    # We must NOT use a prefix here because Qwen3VLModel.sharded_state_dict() already
    # adds "language_model." internally. Using prefix="" ensures the keys match correctly.
    # (Using prefix="language_model." would create "language_model.language_model.*" keys)
    if has_modelopt_support and args.load is not None:
        if has_modelopt_state(args.load):
            print_rank_0(f'Loading modelopt state from {args.load} for QAT training...')
            from megatron.post_training.checkpointing import get_sharded_load_dir
            from modelopt.torch.opt.plugins.mcore_dist_checkpointing import restore_sharded_modelopt_state
            sharded_load_dir, _ = get_sharded_load_dir(args.load)
            if sharded_load_dir is not None:
                # No prefix needed - checkpoint keys already have language_model. prefix
                # and Qwen3VLModel.sharded_state_dict() generates matching keys
                restore_sharded_modelopt_state([model], sharded_load_dir, prefix="")
                print_rank_0('Modelopt state loaded. Model converted to quantization-aware format.')
            else:
                print_rank_0('No sharded checkpoint found. Skipping modelopt state loading.')

    # [ModelOpt]: Knowledge Distillation (QAD) setup
    # For VLM, we wrap model.language_model (not the full Qwen3VLModel) with DistillationModel.
    # The teacher is a text-only GPTModel that receives the same embeddings (text + image features)
    # as the student's language_model. The vision encoder is NOT distilled.
    if has_kd_support and getattr(args, 'export_kd_teacher_load', None):
        print_rank_0("VLM Distillation: Enabled (wrapping language_model only).")

        assert not getattr(args, 'manual_gc', False), \
            "ModelOpt Distillation currently incompatible with `--manual-gc` option."
        assert not getattr(args, 'tp_comm_overlap', False), \
            "ModelOpt Distillation currently incompatible with `--tp-comm-overlap` option."

        teacher_config_raw = _load_teacher_model_config(args.export_kd_teacher_load)
        teacher_config = core_transformer_config_from_args(teacher_config_raw)

        distill_cfg = mtd_mcore.setup_distillation_config(
            getattr(args, 'export_kd_cfg', None),
            student_cfg=language_transformer_config,
            teacher_cfg=teacher_config,
        )

        # Build model_kwargs for teacher (same structure as language_model)
        teacher_model_kwargs = {
            "vocab_size": args.padded_vocab_size,
            "max_sequence_length": args.max_position_embeddings,
            "parallel_output": parallel_output,
            "share_embeddings_and_output_weights": not args.untie_embeddings_and_output_weights,
            "position_embedding_type": args.position_embedding_type,
            "rotary_percent": args.rotary_percent,
            "pre_process": pre_process,
            "post_process": post_process,
        }

        kd_config = {
            "teacher_model": _load_vlm_teacher_model(teacher_config, teacher_config_raw, teacher_model_kwargs),
            "criterion": distill_cfg.criterion,
            "loss_balancer": distill_cfg.loss_balancer,
        }

        # Wrap only language_model with DistillationModel
        model.language_model = mtd.convert(model.language_model, mode=[("kd_loss", kd_config)])
        mtd_mcore.adjust_distillation_model_for_mcore(model.language_model, distill_cfg)

        # Fix "Model has multiple modelopt states!" during checkpoint saving.
        # restore_sharded_modelopt_state (above) sets _modelopt_state on the root Qwen3VLModel,
        # then mtd.convert creates a second _modelopt_state on language_model. Two modules
        # with state triggers an assertion in save_sharded_modelopt_state. Remove language_model's
        # state so only the root has it, then pop the quantization state from root to match
        # text-only QAD behavior (see model_builder.py:338).
        if hasattr(model.language_model, '_modelopt_state'):
            mto.ModeloptStateManager.remove_state(model.language_model)
        mto.ModeloptStateManager(model).state_dict().pop()
        print_rank_0("VLM Distillation: language_model wrapped with DistillationModel.")

    elif getattr(args, 'export_kd_teacher_load', None) and not has_kd_support:
        raise RuntimeError(
            "--export-kd-teacher-load is set but knowledge distillation dependencies are not available. "
            "Please install modelopt with distillation support: pip install nvidia-modelopt[torch]"
        )

    # [ModelOpt]: Memory-efficient quantizer (--me-quantizer)
    # Monkey-patches TensorQuantizer._fake_quantize for inplace NVFP4 fake quantization.
    # Incompatible with --use-distributed-optimizer / --overlap-param-gather because the
    # bound method leaks into modelopt's extra_state pickle and hits unpicklable DDP hooks.
    if getattr(args, 'me_quantizer', False):
        from patch_tensor_quantizer import inject_memory_efficient_dynamic_block_quant
        inject_memory_efficient_dynamic_block_quant(model)

    return model


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build training, validation, and test datasets.

    This function creates datasets that can load real multimodal data
    from the specified data blend path. Supports both --data-path and
    --per-split-data-args-path for specifying data.

    For VLM training with images, use --use-vlm-dataset to load images
    from JSONL files instead of preprocessed binary files.

    Args:
        train_val_test_num_samples: List of sample counts [train, val, test]

    Returns:
        Tuple of (train_ds, val_ds, test_ds)
    """
    args = get_args()

    print_rank_0("> building train, validation, and test datasets for Qwen3-VL ...")

    # Determine sequence length for dataloader
    seq_length = args.dataloader_seq_length if args.dataloader_seq_length else args.seq_length

    # Get blend paths from args (handles both --data-path and --per-split-data-args-path)
    blend, blend_per_split = get_blend_and_blend_per_split(args)

    # When no data paths are provided, use MockMultimodalDataset (mock mode).
    if blend is None and blend_per_split is None:
        print_rank_0(">   No data paths provided. Using MockMultimodalDataset (mock data mode)")
        config = MultimodalDatasetConfig(
            random_seed=args.seed,
            split=args.split,
            sequence_length=seq_length,
            tokenizer=get_tokenizer(),
            reset_position_ids=args.reset_position_ids,
            reset_attention_mask=args.reset_attention_mask,
            eod_mask_loss=args.eod_mask_loss,
            image_h=args.img_h,
            image_w=args.img_w,
        )

        train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
            MockMultimodalDataset,
            train_val_test_num_samples,
            lambda: parallel_state.get_tensor_model_parallel_rank() == 0,
            config,
        ).build()

        print_rank_0("> finished creating mock Qwen3-VL datasets ...")
        return train_ds, valid_ds, test_ds

    print_rank_0(f">   blend: {blend}")
    print_rank_0(f">   blend_per_split: {blend_per_split}")

    # Check if we should use VLM dataset with images
    use_vlm_dataset = getattr(args, 'use_vlm_dataset', False)

    if use_vlm_dataset:
        # Use Qwen3VL multimodal dataset that loads images from JSONL
        print_rank_0(">   Using Qwen3VL multimodal dataset with images")

        # Get the blend path (for VLM, we expect per-split-data-args-path)
        blend_path = getattr(args, 'per_split_data_args_path', None)
        if not blend_path:
            raise ValueError("--per-split-data-args-path is required for --use-vlm-dataset")

        # Create VLM dataset config
        vlm_config = Qwen3VLDatasetConfig(
            image_base_dir=getattr(args, 'image_base_dir', None),
            img_h=args.img_h,
            img_w=args.img_w,
            processor_name=args.hf_model_name,
            sequence_length=seq_length,
            random_seed=args.seed,
        )

        # Build VLM datasets
        builder = Qwen3VLDatasetBuilder(
            config=vlm_config,
            blend_path=blend_path,
            train_val_test_num_samples=train_val_test_num_samples,
        )
        train_ds, valid_ds, test_ds = builder.build()

    else:
        # Use standard GPTDataset (text-only, no images)
        print_rank_0(">   Using GPTDataset (text-only, no images)")
        print_rank_0(">   NOTE: Add --use-vlm-dataset to load images from JSONL files")

        # Create dataset config
        # Note: split and blend_per_split are incompatible - only use split when blend is provided
        config = GPTDatasetConfig(
            random_seed=args.seed,
            sequence_length=seq_length,
            blend=blend,
            blend_per_split=blend_per_split,
            split=args.split if blend_per_split is None else None,
            num_dataset_builder_threads=args.num_dataset_builder_threads,
            path_to_cache=args.data_cache_path,
            mmap_bin_files=args.mmap_bin_files,
            tokenizer=get_tokenizer(),
            reset_position_ids=args.reset_position_ids,
            reset_attention_mask=args.reset_attention_mask,
            eod_mask_loss=args.eod_mask_loss,
            create_attention_mask=args.create_attention_mask_in_dataloader,
        )

        # Build datasets
        train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
            GPTDataset,
            train_val_test_num_samples,
            lambda: parallel_state.get_tensor_model_parallel_rank() == 0,
            config,
        ).build()

    print_rank_0("> finished creating Qwen3-VL datasets ...")

    return train_ds, valid_ds, test_ds


def get_batch(data_iterator):
    """Generate a batch for Qwen3-VL training.

    Args:
        data_iterator: Iterator over the dataset

    Returns:
        Tuple of batch components for Qwen3VLModel.forward()
    """
    args = get_args()

    # Get data from iterator
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None

    # Broadcast data across tensor parallel ranks
    data_i = tensor_parallel.broadcast_data(
        ["tokens", "position_ids", "labels"], data, torch.int64
    )
    data_f = tensor_parallel.broadcast_data(
        ["loss_mask"], data, torch.float32
    )

    # Extract batch components
    tokens = data_i["tokens"].long()
    position_ids = data_i["position_ids"].long()
    labels = data_i["labels"].long()
    loss_mask = data_f["loss_mask"].float()

    # Handle image data.
    # All broadcast_data calls must be unconditional — every TP rank must
    # participate in the NCCL collective, even when data is None on non-TP0 ranks.
    images = None
    grid_thw = None

    has_data_path = (getattr(args, 'data_path', None) is not None or
                     getattr(args, 'per_split_data_args_path', None) is not None)

    if not has_data_path:
        # Mock data mode: MockMultimodalDataset always returns "image" key
        image_data = tensor_parallel.broadcast_data(
            ["image"], data, torch.float32
        )
        images = image_data["image"].to(torch.bfloat16)
    else:
        # Real data mode: Qwen3VLDataset collate always includes pixel_values
        # and image_grid_thw (empty tensors when no images in batch).
        pixel_data = tensor_parallel.broadcast_data(
            ["pixel_values"], data, torch.float32
        )
        grid_data = tensor_parallel.broadcast_data(
            ["image_grid_thw"], data, torch.int64
        )
        pv = pixel_data["pixel_values"]
        gt = grid_data["image_grid_thw"]
        if pv.numel() > 0:
            images = pv.to(torch.bfloat16)
            grid_thw = gt if gt.numel() > 0 else None

    attention_mask = None  # Use mask type from layer spec
    packed_seq_params = None

    return (
        images,
        tokens,
        position_ids,
        labels,
        loss_mask,
        attention_mask,
        grid_thw,
        packed_seq_params,
    )


def forward_step(data_iterator, model: Qwen3VLModel):
    """Forward training step for Qwen3-VL.

    Args:
        data_iterator: Iterator over the dataset
        model: The Qwen3-VL model

    Returns:
        Tuple of (output_tensor, loss_func)
    """
    timers = get_timers()

    # Get batch
    timers('batch-generator', log_level=2).start()
    (
        images,
        tokens,
        position_ids,
        labels,
        loss_mask,
        attention_mask,
        grid_thw,
        packed_seq_params,
    ) = get_batch(data_iterator)
    timers('batch-generator').stop()

    # Forward pass through Qwen3VLModel
    # The model handles:
    # - Image embedding extraction via vision encoder
    # - Sequence expansion to insert image embeddings
    # - DeepStack visual feature injection
    # - Labels and loss_mask expansion
    output_tensor, expanded_loss_mask = model(
        images=images,
        input_ids=tokens,
        position_ids=position_ids,
        attention_mask=attention_mask,
        labels=labels,
        loss_mask=loss_mask,
        grid_thw=grid_thw,
        packed_seq_params=packed_seq_params,
    )

    # For QAD (knowledge distillation): the DistillationModel wraps the inner
    # language_model (GPTModel), not the outer Qwen3VLModel. Pass language_model
    # to loss_func so compute_kd_loss() is reachable. For regular QAT (no KD),
    # pass the full model as-is.
    args = get_args()
    if getattr(args, 'export_kd_teacher_load', None):
        from megatron.training.utils import unwrap_model as _unwrap
        _unwrapped = _unwrap(model)
        loss_model = getattr(_unwrapped, 'language_model', model)
    else:
        loss_model = model
    return output_tensor, partial(loss_func, expanded_loss_mask, model=loss_model)


def add_qwen3vl_extra_args(parser):
    """Add Qwen3-VL specific arguments and modelopt arguments for QAT.

    Args:
        parser: Argument parser

    Returns:
        Updated parser
    """
    # [ModelOpt]: Add modelopt arguments for QAT training (export_kd_teacher_load, etc.)
    if has_modelopt_args:
        parser = add_modelopt_args(parser)

    group = parser.add_argument_group(title='Qwen3-VL specific arguments')

    # Visual encoder arguments
    group.add_argument(
        '--hf-model-name',
        type=str,
        default='Qwen/Qwen3-VL-8B-Instruct',
        help='HuggingFace model name for loading the visual encoder'
    )
    group.add_argument(
        '--num-deepstack-layers',
        type=int,
        default=5,
        help='Number of deepstack layers for visual feature injection'
    )
    group.add_argument(
        '--image-token-index',
        type=int,
        default=DEFAULT_IMAGE_TOKEN_INDEX,
        help='Token index used to mark image positions (-200 for Qwen3-VL)'
    )
    group.add_argument(
        '--video-token-index',
        type=int,
        default=DEFAULT_VIDEO_TOKEN_INDEX,
        help='Token index used to mark video positions (-300 for Qwen3-VL)'
    )

    # Note: --img-h, --img-w, --patch-dim, --encoder-num-layers, --decoder-num-layers
    # are already defined in megatron/training/arguments.py

    # Training arguments
    group.add_argument(
        '--freeze-LM',
        action='store_true',
        default=False,
        help='Freeze language model weights during training'
    )
    group.add_argument(
        '--freeze-ViT',
        action='store_true',
        default=False,
        help='Freeze vision encoder weights during training'
    )
    group.add_argument(
        '--freeze-lm-embedding',
        action='store_true',
        default=False,
        help='Freeze language model embedding layer during training '
             '(not quantized, saves memory and avoids drift)'
    )
    group.add_argument(
        '--dataloader-seq-length',
        type=int,
        help='Sequence length for dataloader (text tokens only, before adding image tokens)'
    )

    # VLM Dataset arguments
    group.add_argument(
        '--use-vlm-dataset',
        action='store_true',
        default=False,
        help='Use VLM dataset that loads images from JSONL files instead of preprocessed binary files'
    )
    group.add_argument(
        '--image-base-dir',
        type=str,
        default=None,
        help='Base directory for resolving relative image paths in JSONL files'
    )

    # Vision model type
    group.add_argument(
        '--vision-model-type',
        type=str,
        default='qwen3_vl',
        choices=['qwen3_vl', 'clip'],
        help='Type of vision model to use'
    )

    # memory efficient quantizer
    group.add_argument(
        "--me-quantizer",
        action="store_true",
        default=False,
        help="Inject custom NVFP4 quantizer into the model.",
    )
    return parser


def qwen3vl_embedding_ranks(pp_ranks):
    """Get embedding ranks for Qwen3-VL with pipeline parallelism.

    For Qwen3-VL, embeddings are needed on:
    - First rank: for input embeddings and visual encoder
    - Last rank: for output layer

    Args:
        pp_ranks: List of global ranks in a pipeline group

    Returns:
        List of ranks that hold embeddings
    """
    first_rank = pp_ranks[0]
    last_rank = pp_ranks[-1]

    if len(pp_ranks) == 1:
        return [first_rank]
    return [first_rank, last_rank]


def qwen3vl_position_embedding_ranks(pp_ranks):
    """Get position embedding ranks for Qwen3-VL.

    Position embeddings are only on the first pipeline stage.

    Args:
        pp_ranks: List of global ranks in a pipeline group

    Returns:
        List of ranks that hold position embeddings
    """
    return [pp_ranks[0]]


if __name__ == "__main__":
    # Mark dataset provider as distributed
    train_valid_test_datasets_provider.is_distributed = True

    # Run pretraining
    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        args_defaults={'tokenizer_type': 'HuggingFaceTokenizer'},
        extra_args_provider=add_qwen3vl_extra_args,
        get_embedding_ranks=qwen3vl_embedding_ranks,
        get_position_embedding_ranks=qwen3vl_position_embedding_ranks,
    )
