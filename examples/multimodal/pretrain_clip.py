# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Pretrain vision language model."""
import json
import warnings
from copy import deepcopy
from functools import partial
from pathlib import Path

import torch

from megatron.core import mpu, parallel_state, tensor_parallel
from megatron.core.datasets.blended_megatron_dataset_builder import (
    BlendedMegatronDatasetBuilder,
)
from megatron.core.datasets.multimodal_dataset import (
    MockMultimodalDataset,
    MultimodalDatasetConfig,
)
from megatron.core.enums import ModelType
from megatron.core.models.multimodal.clip_model import ClipModel
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_spec,
    get_gpt_layer_local_spec,
)
from megatron.core.models.vision.clip_vit_model import get_num_image_embeddings
from megatron.core.models.vision.vit_layer_specs import (
    get_vit_layer_with_local_spec,
    get_vit_layer_with_transformer_engine_spec,
)
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.spec_utils import import_module
from megatron.training import (
    get_args,
    get_timers,
    get_tokenizer,
    pretrain,
    print_rank_0,
)
from megatron.training.arguments import core_transformer_config_from_args


def model_provider(
    parallel_output=True,
    config=None,
    **kwargs,  # absorbs pre_process/post_process passed by Megatron training framework
) -> ClipModel:
    """Builds the model.

    Note: currently, only Clip model is supported. Follow-up changes will make this configurable.

    Args:
        parallel_output (bool): Enable model parallel output.

    Returns:
        model (megatron.core.models.multimodal.clip_model.ClipModel): A multimodal model
    """
    args = get_args()
    vision_model_type = "clip"

    assert args.ckpt_format == 'torch', "Only ckpt-format torch is supported for VLM training currently."
    assert not (args.context_parallel_size > 1 and args.pipeline_model_parallel_size > 1), "PP+CP is not yet supported by this script. \
    Current mock dataset does not support natively packed sequence dataset required for correct PP comm shapes."

    num_image_embeddings = get_num_image_embeddings(
        args.img_h, args.img_w, args.patch_dim, vision_model_type,
        args.disable_vision_class_token, class_token_len=1,
        pixel_shuffle=False, use_tile_tags=False
    )

    # Vision sequence length
    args.encoder_seq_length = num_image_embeddings

    # Text sequence length (no image embeddings added)
    if args.decoder_seq_length is None:
        args.decoder_seq_length = args.seq_length

    # Update max position embeddings if needed
    args.max_position_embeddings = max(args.max_position_embeddings, args.decoder_seq_length)

    print_rank_0('building a multimodal model ...')
    if config is None:
        language_transformer_config = core_transformer_config_from_args(get_args())
    else:
        language_transformer_config = config
    if args.decoder_num_layers is not None:
        language_transformer_config.num_layers = args.decoder_num_layers
    else:
        language_transformer_config.num_layers = args.num_layers
    if getattr(args, "decoder_tp_comm_overlap", False):
        assert args.transformer_impl == "transformer_engine", \
            "TransformerEngine is needed to support Decoder TP Comm overlap"
        language_transformer_config.tp_comm_overlap = getattr(args, "decoder_tp_comm_overlap", False)

    if args.spec is not None:
        language_transformer_layer_spec = import_module(args.spec)
    elif args.transformer_impl == "transformer_engine":
        language_transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
            args.num_experts, args.moe_grouped_gemm, args.qk_layernorm
        )
    else:  # transformer_impl == "local"
        language_transformer_layer_spec = get_gpt_layer_local_spec(
            args.num_experts, args.moe_grouped_gemm, args.qk_layernorm
        )

    if args.transformer_impl == "transformer_engine":
        vision_transformer_layer_spec = get_vit_layer_with_transformer_engine_spec()
    else:  # transformer_impl == "local"
        vision_transformer_layer_spec = get_vit_layer_with_local_spec()

    # TODO: Make these configurable via input .yaml config.
    vision_transformer_config = deepcopy(language_transformer_config)
    vision_transformer_config.num_layers = args.encoder_num_layers
    vision_transformer_config.first_pipeline_num_layers = None
    vision_transformer_config.last_pipeline_num_layers = None
    vision_transformer_config.vision_model_type = vision_model_type
    vision_transformer_config.context_parallel_size = 1 # Force CP=1 for Vision Transformer
    if vision_transformer_config.sequence_parallel:
        print_rank_0("> Disabling Sequence parallelism in Vision Transformer. Not yet supported")
        vision_transformer_config.sequence_parallel = False
    if vision_transformer_config.tp_comm_overlap:
        print_rank_0("> Disabling TP Comm overlap in Vision Transformer. Not yet supported")
        vision_transformer_config.tp_comm_overlap = False

    vision_projection_type = "affine"
    vision_projection_config = deepcopy(language_transformer_config)
    vision_projection_config.context_parallel_size = 1 # Force CP=1 for Vision Projection
    if vision_projection_config.sequence_parallel:
        print_rank_0("> Disabling Sequence parallelism in Vision Projection. Not yet supported")
        vision_projection_config.sequence_parallel = False
    if vision_projection_config.tp_comm_overlap:
        print_rank_0("> Disabling TP Comm overlap in Vision Projection. Not yet supported")
        vision_projection_config.tp_comm_overlap = False

    # Vision Encoder and Projection should live on PP rank0
    vision_transformer_config.pipeline_model_parallel_size = 1
    vision_projection_config.pipeline_model_parallel_size = 1

    vision_projection_modules = deepcopy(language_transformer_layer_spec.submodules.mlp.submodules)

    language_max_sequence_length = args.decoder_seq_length
    # Add language projection config (same structure as vision projection)
    language_projection_type = "affine"
    language_projection_config = deepcopy(language_transformer_config)
    language_projection_config.context_parallel_size = 1
    language_projection_config.pipeline_model_parallel_size = 1
    if language_projection_config.sequence_parallel:
        print_rank_0("> Disabling Sequence parallelism in Language Projection. Not yet supported")
        language_projection_config.sequence_parallel = False
    if language_projection_config.tp_comm_overlap:
        print_rank_0("> Disabling TP Comm overlap in Language Projection. Not yet supported")
        language_projection_config.tp_comm_overlap = False

    language_projection_modules = deepcopy(language_transformer_layer_spec.submodules.mlp.submodules)

    model = ClipModel(
        language_transformer_config=language_transformer_config,
        language_transformer_layer_spec=language_transformer_layer_spec,
        language_vocab_size=args.padded_vocab_size,
        language_max_sequence_length=language_max_sequence_length,
        language_projection_config=language_projection_config,
        language_projection_layer_spec=language_projection_modules,
        language_projection_type=language_projection_type,
        vision_transformer_config=vision_transformer_config,
        vision_transformer_layer_spec=vision_transformer_layer_spec,
        drop_vision_class_token=args.disable_vision_class_token,
        vision_projection_config=vision_projection_config,
        vision_projection_layer_spec=vision_projection_modules,
        vision_projection_type=vision_projection_type,
        parallel_output=parallel_output,
        language_position_embedding_type=args.position_embedding_type,
        language_rotary_percent=args.rotary_percent,
        language_rope_scaling=args.use_rope_scaling,
        pre_process=parallel_state.is_pipeline_first_stage(),
        post_process=False,
        img_h=args.img_h,
        img_w=args.img_w,
        patch_dim=args.patch_dim,
        local_loss=getattr(args, 'clip_local_loss', False),
    )

    model.freeze(
        freeze_language_model=getattr(args, "freeze_LM", False),
        freeze_vision_model=getattr(args, "freeze_ViT", False),
        freeze_vision_projection=False,
        freeze_language_projection=False,
    )

    return model


class EnergoClipDataset(torch.utils.data.Dataset):
    """Map-style dataset wrapping energon CaptioningSample for CLIP pretraining.

    Streams image-caption pairs from an energon WebDataset (or Metadataset YAML),
    applies CLIP image preprocessing and CLIPTokenizer, and returns dicts with keys
    'image', 'tokens', 'attention_mask' matching the format expected by get_batch().

    __getitem__ ignores the index argument and advances an internal energon iterator.
    This is intentional: energon handles shuffling and DP-rank routing internally.
    Requires num_workers=0 in the DataLoader (set via --num-workers 0).
    """

    CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
    CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]

    def __init__(self, data_path, tokenizer_path, image_size=224, seq_length=77,
                 dp_rank=0, dp_world_size=1, shuffle_buffer=1000, num_samples=None):
        import torchvision.transforms as T
        from PIL import Image as PILImage
        from transformers import CLIPTokenizer
        from megatron.energon import get_train_dataset, get_loader, WorkerConfig

        self.seq_length = seq_length
        self._pil_image = PILImage  # keep reference for __getitem__

        # CLIP standard image preprocessing
        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=self.CLIP_MEAN, std=self.CLIP_STD),
        ])

        # Offline CLIPTokenizer (no network access on compute nodes)
        self.tokenizer = CLIPTokenizer(
            vocab_file=f"{tokenizer_path}/vocab.json",
            merges_file=f"{tokenizer_path}/merges.txt",
        )

        # Energon dataset: data_path is either a directory or a Metadataset YAML
        worker_config = WorkerConfig(rank=dp_rank, world_size=dp_world_size, num_workers=0)
        energon_ds = get_train_dataset(
            data_path,
            batch_size=1,
            shuffle_buffer_size=shuffle_buffer,
            max_samples_per_sequence=100,
            worker_config=worker_config,
        )
        self._loader = get_loader(energon_ds, worker_config=worker_config)
        self._iter = None

        # Total samples reported to Megatron's sampler via __len__.
        # Use the caller-supplied num_samples (= train_val_test_num_samples[0]) so
        # the sampler never runs dry before TRAIN_ITERS is reached.  Energon
        # handles its own cycling at epoch boundaries via StopIteration restart.
        self._total = num_samples if num_samples is not None else 1_000_000

    def __len__(self):
        return self._total

    def _next_energon_sample(self):
        """Advance the energon iterator, restarting at epoch boundary."""
        if self._iter is None:
            self._iter = iter(self._loader)
        try:
            return next(self._iter)
        except StopIteration:
            self._iter = iter(self._loader)
            return next(self._iter)

    def __getitem__(self, idx):
        """Return the next energon sample as a dict; idx is ignored."""
        import numpy as np

        sample = self._next_energon_sample()  # CaptioningSample, batch_size=1

        # Image: energon decodes JPEG → float tensor [1, 3, H, W] in [0, 1]
        img_tensor = sample.image[0]  # [3, H, W]
        img_np = (img_tensor.permute(1, 2, 0).cpu().float().numpy() * 255).clip(0, 255).astype(np.uint8)
        img_pil = self._pil_image.fromarray(img_np)
        img_out = self.transform(img_pil)  # [3, 224, 224] float32

        # Text: tokenize caption with padding to seq_length
        caption = sample.caption[0]
        encoded = self.tokenizer(
            caption,
            max_length=self.seq_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt',
        )
        tokens = encoded['input_ids'][0]              # [77] int64
        attn_mask = encoded['attention_mask'][0].bool()  # [77] bool

        return {
            'image': img_out,          # [3, 224, 224] float32
            'tokens': tokens,          # [77] int64
            'attention_mask': attn_mask,  # [77] bool
        }


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build the train, validation, and test datasets.

    Supports two modes:
      - Mock (--mock-data): random image-text pairs via MockMultimodalDataset.
        No files required. Use for smoke tests.
      - Real (--clip-data-path): energon WebDataset or Metadataset YAML.
        Requires --clip-tokenizer-path. Image preprocessing and tokenization
        are done inside EnergoClipDataset.__getitem__. Pass --num-workers 0.

    Args:
        train_val_test_num_samples: [n_train, n_val, n_test] from Megatron.

    Returns:
        train_ds, val_ds, test_ds
    """
    args = get_args()

    # ── Real data path (energon) ──────────────────────────────────────────────
    clip_data_path = getattr(args, 'clip_data_path', None)
    if clip_data_path is not None:
        clip_tokenizer_path = getattr(args, 'clip_tokenizer_path', None)
        if clip_tokenizer_path is None:
            raise ValueError("--clip-tokenizer-path is required when using --clip-data-path")

        dp_rank  = parallel_state.get_data_parallel_rank()
        dp_world = parallel_state.get_data_parallel_world_size()

        print_rank_0(f"> building energon CLIP dataset from: {clip_data_path}")
        train_ds = EnergoClipDataset(
            data_path=clip_data_path,
            tokenizer_path=clip_tokenizer_path,
            image_size=args.img_h,
            seq_length=args.decoder_seq_length,
            dp_rank=dp_rank,
            dp_world_size=dp_world,
            num_samples=train_val_test_num_samples[0],
        )
        print_rank_0(f"> energon dataset ready, reporting {len(train_ds)} samples to sampler")
        return train_ds, None, None

    # ── Mock data path ────────────────────────────────────────────────────────
    config = MultimodalDatasetConfig(
        random_seed=args.seed,
        split=args.split,
        sequence_length=args.decoder_seq_length,
        tokenizer=get_tokenizer(),
        reset_position_ids=args.reset_position_ids,
        reset_attention_mask=args.reset_attention_mask,
        eod_mask_loss=args.eod_mask_loss,
        image_h=args.img_h,
        image_w=args.img_w,
        preprocess_func=_preprocess_data_for_clip,
        mid_level_dataset_surplus=getattr(args, "mid_level_dataset_surplus", 0.0),
        allow_ambiguous_pad_tokens=getattr(args, "allow_ambiguous_pad_tokens", False),
    )

    print_rank_0("> building train, validation, and test datasets for multimodal ...")

    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        MockMultimodalDataset,
        train_val_test_num_samples,
        lambda: parallel_state.get_tensor_model_parallel_rank() == 0,
        config,
    ).build()

    print_rank_0("> finished creating multimodal datasets ...")

    return train_ds, valid_ds, test_ds


def _preprocess_data_for_clip(data):
    """Preprocess data sample to the format expected by a Clip model.

    Args:
        data (dict): Data sample with keys like 'image', 'tokens', etc.

    Returns:
        data (dict): Processed data sample suitable for the model.
    """
    return data


def get_batch(data_iterator):
    """Generate a batch for Clip training.
    Clip uses simple pairs of image-text data.

    Args:
        data_iterator: Iterable dataset.

    Returns:
        images
        input_ids
        attention_mask
    """
    args = get_args()
    cp_size = args.context_parallel_size
    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None

    data_i = tensor_parallel.broadcast_data(["tokens"], data, torch.int64)
    img_dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)
    data_f = tensor_parallel.broadcast_data(["image"], data, torch.float32)  # broadcast native dtype, cast after

    data_attn = tensor_parallel.broadcast_data(["attention_mask"], data, torch.bool)

    input_ids = data_i["tokens"].long()
    images = data_f["image"].to(dtype=img_dtype)
    attention_mask = data_attn.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.long()

    return images, input_ids, attention_mask

def forward_step(data_iterator, model: ClipModel):
    """Forward training step.

    Args:
        data_iterator: Iterable dataset.
        model (megatron.core.models.multimodal.llava_model.ClipModel)

    Returns:
        loss (torch.Tensor): contrastive loss
        loss_func (callable): None, done internally.
    """
    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    images, input_ids, attention_mask = get_batch(data_iterator)
    timers('batch-generator').stop()

    # Note: the GPT dataset returns a 4D causal attention mask [batch, 1, seq, seq],
    # not a 1D padding mask. Pass None so ClipModel uses language_output[:, -1, :] (EOS).
    loss = model(
        images=images,
        input_ids=input_ids,
        attention_mask=None,
        position_ids=None,
    )

    # loss_func must return (output_tensor, dict_of_losses) for training.py averaging
    return loss, lambda x: (x, {"loss": x})


def add_clip_extra_args(parser):
    """Extra arguments."""
    group = parser.add_argument_group(title='ClipModel specific arguments')
    group.add_argument(
        '--freeze-LM', action='store_true', default=False, help="Freeze language model weights"
    )
    group.add_argument(
        '--freeze-ViT', action='store_true', default=False, help="Freeze vision model (ViT) weights"
    )
    group.add_argument(
        '--freeze-language-projection', action='store_true', default=False,
        help="Freeze language projection weights"
    )
    group.add_argument(
        '--freeze-vision-projection', action='store_true', default=False,
        help="Freeze vision projection weights"
    )
    group.add_argument(
        "--disable-vision-class-token",
        action="store_true",
        default=False,
        help="Drop vision model class token (not recommended for CLIP)",
    )
    group.add_argument(
        "--clip-data-path",
        type=str,
        default=None,
        help="Path to energon WebDataset directory or Metadataset YAML for real data training. "
             "Mutually exclusive with --mock-data.",
    )
    group.add_argument(
        "--clip-tokenizer-path",
        type=str,
        default=None,
        help="Path to directory containing offline CLIPTokenizer files "
             "(vocab.json, merges.txt). Required when --clip-data-path is set.",
    )
    group.add_argument(
        "--clip-local-loss",
        action="store_true",
        default=False,
        help="Use local contrastive loss: each DP rank computes cross-entropy only over its own "
             "local batch rows against the globally-gathered embeddings. Reduces cross-entropy "
             "compute by dp_world_size× while keeping the full negative pool. "
             "Default False (full global loss, all ranks compute the same NxN matrix).",
    )

    return parser

def clip_embedding_ranks(pp_ranks):
    """CLIP embedding ranks.

    For CLIP, embeddings are on first stage (vision + text encoders).

    Args:
        pp_ranks: A list of global ranks that constitute a pipeline group.
    """
    # CLIP typically has vision and text encoders on first stage
    return [pp_ranks[0]]


def clip_position_embedding_ranks(pp_ranks):
    """CLIP positional embeddings are on the first rank stage.

    Args:
        pp_ranks: A list of global ranks that constitute a pipeline group.
    """
    return [pp_ranks[0]]

if __name__ == "__main__":
    train_valid_test_datasets_provider.is_distributed = True

    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,  # CLIP has dual encoders
        forward_step,
        args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
        extra_args_provider=add_clip_extra_args,  # Changed!
        get_embedding_ranks=clip_embedding_ranks,  # Changed!
        get_position_embedding_ranks=clip_position_embedding_ranks,  # Changed!
    )
