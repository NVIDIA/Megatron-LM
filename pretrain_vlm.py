# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
"""Pretrain vision language model."""
from copy import deepcopy
from functools import partial
import warnings

import torch

from megatron.core import parallel_state, tensor_parallel
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.multimodal_dataset import MockMultimodalDataset, MultimodalDatasetConfig
from megatron.core.enums import ModelType
from megatron.core.models.vision.clip_vit_model import get_num_image_embeddings
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.models.multimodal.llava_model import LLaVAModel, DEFAULT_IMAGE_TOKEN_INDEX
from megatron.core.models.multimodal.llava_spec import (
    decoder_model_with_transformer_engine_default_spec,
    decoder_model_with_local_default_spec,
)
from megatron.core.models.vision.vit_layer_specs import (
    get_vit_layer_with_transformer_engine_spec,
    get_vit_layer_with_local_spec,
)
from megatron.core.transformer.spec_utils import import_module
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.training import get_args, get_timers, get_tokenizer, pretrain, print_rank_0
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.utils import get_batch_on_this_cp_rank
from megatron.core import mpu
from pretrain_gpt import loss_func

def calculate_model_parallel_padding(decoder_seq_len, text_only=False):
    args = get_args()
    cp_size = args.context_parallel_size
    tp_size = args.tensor_model_parallel_size

    mp_padding_needed = 0
    # TP Comm overlap is performed with combined text+image embeddings.
    # text_only flag skips using the full sequence length to calculate padding and uses
    # the provided decoder_seq_len
    if args.sequence_parallel and args.decoder_tp_comm_overlap and not text_only:
        # If TP Comm Overlap is enabled for combined text+image embedding in LM backbone,
        # user needs to provide decoder_seq_length with any potential padding needed for SP+CP
        assert args.decoder_seq_length is not None, \
            "Please provide --decoder-seq-length when using TP Comm overlap for LM backbone"
        mp_padding_needed = args.decoder_seq_length - decoder_seq_len
    elif args.sequence_parallel or cp_size > 1:
        if args.sequence_parallel and cp_size > 1:
            # Padding to multiple of tp_size * cp_size*2 when using sequence parallel and context parallel
            padding_factor = tp_size * cp_size * 2
        elif cp_size > 1:
            padding_factor = cp_size * 2
        elif args.sequence_parallel:
            padding_factor = tp_size
        mp_padding_needed = int((decoder_seq_len + padding_factor - 1) // (padding_factor) * (padding_factor)) - decoder_seq_len
        args.decoder_seq_length = decoder_seq_len + mp_padding_needed
    else:
        args.decoder_seq_length = decoder_seq_len

    return mp_padding_needed

def model_provider(
    pre_process=True, post_process=True, add_encoder=True, add_decoder=True, parallel_output=True
) -> LLaVAModel:
    """Builds the model.

    Note: currently, only LLaVA model is supported. Follow-up changes will make this configurable.

    Args:
        pre_process (bool): Include the embedding layer in the gpt decoder (used with pipeline parallelism). Defaults to True.
        post_process (bool): Include an output layer and a layernorm in the gpt decoder (used with pipeline parallelism). Defaults to True.
        add_encoder (bool): Construct the encoder module (used with pipeline parallelism). Defaults to True. When we use pipelining, the encoder
            will live on only a subset of the pipeline stages (specifically, only the first stage).
        add_decoder (bool): Construct the decoder module (used with pipeline parallelism). Defaults to True. When we use pipelining, the decoder
            will live on only a subset of the pipeline stages (specifically, every stage after the first one).
        parallel_output (bool): Enable model parallel output.

    Returns:
        model (megatron.core.models.multimodal.llava_model.LLaVAModel): A multimodal model
    """
    args = get_args()
    vision_model_type = "clip"

    assert args.ckpt_format == 'torch', "Only ckpt-format torch is supported for VLM training currently."

    if args.pipeline_model_parallel_size > 1:
        assert not args.freeze_LM, "Freezing a pipeline parallel language model is not currently supported"

    if args.encoder_pipeline_model_parallel_size == 1:
        assert not args.freeze_ViT, "Freezing a vision encoder on its own pipeline rank is not currently supported"

    num_image_embeddings = get_num_image_embeddings(
        args.img_h, args.img_w, args.patch_dim, vision_model_type, args.disable_vision_class_token,
        class_token_len=1, pixel_shuffle=False, use_tile_tags=False
    )

    old_seq_length = args.seq_length
    # dataloader-seq-length is required to determine the length of text seq len
    if args.dataloader_seq_length is None:
        args.dataloader_seq_length = args.seq_length

    # decoder_seq_len denotes the language model sequence length.
    decoder_seq_len = args.dataloader_seq_length + num_image_embeddings

    # seq_length and encoder_seq_length denote the vision model sequence length. Override if the user provided something else.
    args.seq_length = args.encoder_seq_length = num_image_embeddings
    if torch.distributed.get_rank() == 0 and old_seq_length != args.seq_length:
        warnings.warn(
            f"Changed seq_length and encoder_seq_length (vision model sequence length) from {old_seq_length} to num_image_tokens ({num_image_embeddings})"
        )
    mp_padding_needed = calculate_model_parallel_padding(decoder_seq_len)

    args.max_position_embeddings = max(args.max_position_embeddings, args.decoder_seq_length)

    print_rank_0('building a multimodal model ...')
    language_transformer_config = core_transformer_config_from_args(get_args())
    if args.decoder_tp_comm_overlap:
        assert args.transformer_impl == "transformer_engine", \
            "TransformerEngine is needed to support Decoder TP Comm overlap"
        language_transformer_config.tp_comm_overlap = args.decoder_tp_comm_overlap

    if args.spec is not None:
        language_transformer_layer_spec = import_module(args.spec)
    elif args.transformer_impl == "transformer_engine":
        language_transformer_layer_spec = decoder_model_with_transformer_engine_default_spec(
            args.num_experts, args.moe_grouped_gemm
        )
    else:  # transformer_impl == "local"
        language_transformer_layer_spec = decoder_model_with_local_default_spec(
            args.num_experts, args.moe_grouped_gemm
        )
    
    # Prepare mask type for any required padding to support CP/SP sequence sharding.
    if mp_padding_needed > 0:
        if language_transformer_layer_spec.submodules.self_attention.params.get('attn_mask_type', '') == AttnMaskType.causal:
            language_transformer_layer_spec.submodules.self_attention.params['attn_mask_type'] = AttnMaskType.padding_causal
        elif language_transformer_layer_spec.submodules.self_attention.params.get('attn_mask_type', '') == AttnMaskType.no_mask:
            language_transformer_layer_spec.submodules.self_attention.params['attn_mask_type'] = AttnMaskType.padding

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

    vision_projection_type = "mlp"
    vision_projection_config = deepcopy(language_transformer_config)
    vision_projection_config.context_parallel_size = 1 # Force CP=1 for Vision Projection
    if vision_projection_config.sequence_parallel:
        print_rank_0("> Disabling Sequence parallelism in Vision Projection. Not yet supported")
        vision_projection_config.sequence_parallel = False
    if vision_projection_config.tp_comm_overlap:
        print_rank_0("> Disabling TP Comm overlap in Vision Projection. Not yet supported")
        vision_projection_config.tp_comm_overlap = False

    if args.encoder_pipeline_model_parallel_size > 0:
        assert (
            args.encoder_pipeline_model_parallel_size == 1
        ), "ViT can only live on 1 pipeline stage."
        vision_transformer_config.pipeline_model_parallel_size = (
            args.encoder_pipeline_model_parallel_size
        )
        vision_projection_config.pipeline_model_parallel_size = (
            args.encoder_pipeline_model_parallel_size
        )
        if args.encoder_tensor_model_parallel_size > 0:
            vision_transformer_config.tensor_model_parallel_size = (
                args.encoder_tensor_model_parallel_size
            )
            vision_projection_config.tensor_model_parallel_size = (
                args.encoder_tensor_model_parallel_size
            )

    vision_projection_modules = deepcopy(language_transformer_layer_spec.submodules.mlp.submodules)

    if args.virtual_pipeline_model_parallel_size:
        raise NotImplementedError("virtual pipeline model parallelism is not supported yet.")

    model = LLaVAModel(
        language_transformer_config=language_transformer_config,
        language_transformer_layer_spec=language_transformer_layer_spec,
        language_vocab_size=args.padded_vocab_size,
        language_max_sequence_length=args.decoder_seq_length,
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
        pre_process=pre_process,
        post_process=post_process,
        add_encoder=add_encoder,
        add_decoder=add_decoder,
        img_h=args.img_h,
        img_w=args.img_w,
        patch_dim=args.patch_dim,
    )

    model.freeze(
        freeze_language_model=args.freeze_LM,
        freeze_vision_model=args.freeze_ViT,
        freeze_vision_projection=False,
    )

    return model


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build the train test and validation datasets.

    Args:
        train_val_test_num_samples : A list containing the number of samples in train, validation, and test sets.

    Returns:
        train_ds, val_ds, test_ds (megatron.core.datasets.multimodal_dataset.MockMultimodalDataset): Train, validation, and test datasets, respectively.
    """
    args = get_args()

    config = MultimodalDatasetConfig(
        random_seed=args.seed,
        split=args.split,
        sequence_length=args.dataloader_seq_length,
        tokenizer=get_tokenizer(),
        reset_position_ids=args.reset_position_ids,
        reset_attention_mask=args.reset_attention_mask,
        eod_mask_loss=args.eod_mask_loss,
        image_h=args.img_h,
        image_w=args.img_w,
        preprocess_func=_preprocess_data_for_llava,
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


def _preprocess_data_for_llava(data):
    """Preprocess data sample to the format expected by a LLaVA model.

    Note: This doesn't support all the different modes in the official LLaVA repo yet.

    Args:
        data (dict): Data sample with keys like 'image', 'tokens', etc.

    Returns:
        data (dict): Processed data sample suitable for the model.
    """
    # Prepend image token index to tokens.
    data["tokens"] = torch.cat(
        [
            DEFAULT_IMAGE_TOKEN_INDEX
            * torch.ones(1, dtype=data["tokens"].dtype, device=data["tokens"].device),
            data["tokens"],
        ]
    )
    # Prepend labels accordingly.
    data["labels"] = torch.cat([data["tokens"][1].unsqueeze(0), data["labels"]])
    # Zero loss mask for the image token index.
    data["loss_mask"] = torch.cat(
        [
            torch.zeros(1, dtype=data["loss_mask"].dtype, device=data["loss_mask"].device),
            data["loss_mask"],
        ]
    )
    # Add one more position id.
    data["position_ids"] = torch.cat(
        [data["position_ids"], data["position_ids"][-1].unsqueeze(0) + 1]
    )

    return data

def get_batch(data_iterator):
    """Generate a batch.

    Args:
        data_iterator: Iterable dataset.

    Returns:
        sample: A data sample with images, tokens, etc.
    """
    def _get_packed_seq_params(tokens, img_seq_len, mp_padding_needed):
        batch_size = tokens.shape[0]
        # Calculate the valid token seq len that LM backbone should compute on
        combined_valid_seqlen = tokens.shape[1] + img_seq_len - mp_padding_needed
        cu_seqlens = torch.arange(
            0, (batch_size + 1) * (combined_valid_seqlen), step=(combined_valid_seqlen), dtype=torch.int32, device=tokens.device)
        # Calculate the total padded token seq len
        combined_padded_seqlen = tokens.shape[1] + img_seq_len
        cu_seqlens_padded = None
        qkv_format = 'sbhd'
        if cp_size > 1:
            # Provide cu_seqlens_<q/kv>_padded for CP support
            cu_seqlens_padded = torch.arange(
                0, (batch_size + 1) * (combined_padded_seqlen), step=(combined_padded_seqlen), dtype=torch.int32, device=tokens.device)
            # CP with padding mask type requires THD format
            qkv_format = 'thd'
        packed_seq_params = PackedSeqParams(
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            cu_seqlens_q_padded=cu_seqlens_padded,
            cu_seqlens_kv_padded=cu_seqlens_padded,
            max_seqlen_q=combined_padded_seqlen,
            max_seqlen_kv=combined_padded_seqlen,
            qkv_format=qkv_format,
        )
        return packed_seq_params

    args = get_args()
    cp_size = args.context_parallel_size
    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None

    data_i = tensor_parallel.broadcast_data(["tokens", "position_ids", "labels"], data, torch.int64)
    data_f = tensor_parallel.broadcast_data(["image", "loss_mask"], data, torch.float32)

    batch = dict()
    packed_seq_params = None
    image_token_mask = None
    # Create batch with tokens and position_ids for CP sharding.
    tokens = data_i["tokens"].long()
    position_ids = data_i["position_ids"].long()
    labels = data_i["labels"].long()
    loss_mask = data_f["loss_mask"].float()
    images = data_f["image"].float()
    
    if cp_size > 1 or args.sequence_parallel:
        vision_model_type = "clip"
        # Calculate the number of image embedding tokens will be added to text tokens 
        num_image_embeddings_per_tile = get_num_image_embeddings(
            args.img_h, args.img_w, args.patch_dim, vision_model_type, args.disable_vision_class_token, 1
        )
        # Pad to make sure the text sequence can be sharded equally by CP chunks.
        mp_padding_needed_for_text = calculate_model_parallel_padding(tokens.shape[1], text_only=True)
        if mp_padding_needed_for_text > 0:
            tokens, position_ids, labels, loss_mask = [torch.nn.functional.pad(item, (0, mp_padding_needed_for_text)) for item in (tokens, position_ids, labels, loss_mask)]
        # Image token mask must be supplied before distributed sequence to CP ranks.
        image_token_mask = tokens == DEFAULT_IMAGE_TOKEN_INDEX
        num_images_per_sample = torch.sum(image_token_mask, dim=-1)
        img_seq_len = (num_image_embeddings_per_tile * num_images_per_sample - num_images_per_sample).max()
        packed_seq_params = _get_packed_seq_params(tokens, img_seq_len, mp_padding_needed_for_text)
    
    # slice batch along sequence dimension for context parallelism
    batch = get_batch_on_this_cp_rank({"tokens": tokens, "position_ids": position_ids})
    attention_mask = None  # Use the attention mask type defined in layer spec. Typically no mask for the vision model and causal mask for the vision model.

    return batch["tokens"], batch["position_ids"], labels, images, loss_mask, attention_mask, image_token_mask, packed_seq_params


def forward_step(data_iterator, model: LLaVAModel):
    """Forward training step.

    Args:
        data_iterator: Iterable dataset.
        model (megatron.core.models.multimodal.llava_model.LLaVAModel): Multimodal model

    Returns:
        output_tensor (torch.Tensor): Loss of shape [b, s] if labels are provided, otherwise logits of shape [b, s, vocab_size].
        loss_func (callable): Loss function with a loss mask specified.
    """
    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    tokens, position_ids, labels, images, loss_mask, attention_mask, image_token_mask, packed_seq_params = get_batch(data_iterator)
    timers('batch-generator').stop()

    output_tensor, loss_mask = model(
        images, tokens, position_ids, attention_mask, labels, loss_mask, image_token_mask=image_token_mask, packed_seq_params=packed_seq_params
    )

    return output_tensor, partial(loss_func, loss_mask)


def add_vlm_extra_args(parser):
    """Extra arguments."""
    group = parser.add_argument_group(title='vision language model specific arguments')
    group.add_argument(
        '--freeze-LM', action='store_true', default=False, help="Freeze language model weights"
    )
    group.add_argument(
        '--freeze-ViT', action='store_true', default=False, help="Freeze vision model (ViT) weights"
    )
    group.add_argument(
        "--disable-vision-class-token",
        action="store_true",
        default=False,
        help="Drop vision model class token",
    )
    group.add_argument("--dataloader-seq-length", type=int, help="Make dataloader to produce sequences of specific length.")
    group.add_argument("--decoder-tp-comm-overlap", action="store_true", default=False, help="Enables the overlap of "
                        "Tensor parallel communication and GEMM kernels in Decoder only. "
                        "Please provide decoder-seq-length when using this feature.")
    return parser


def llava_embedding_ranks(pp_ranks):
    """LLava's embedding ranks consist of the decoder's first and last ranks (ie, the ViT has no embeddings).
    Args:
        pp_ranks: A list of global ranks that constitute a pipeline group.
    """
    args = get_args()

    # encoder size is also the index to the first rank of the decoder.
    epp = args.encoder_pipeline_model_parallel_size

    last_rank = pp_ranks[-1]
    if len(pp_ranks) == 1 or pp_ranks[epp] == last_rank:
        return [last_rank]
    else:
        return [pp_ranks[epp], last_rank]


def llava_position_embedding_ranks(pp_ranks):
    """LLava's embedding ranks consist of the singular rank of the model or the decoder's first rank.
    Args:
        pp_ranks: A list of global ranks that constitute a pipeline group.
    """
    args = get_args()

    # encoder size is also the index to the first rank of the decoder.
    epp = args.encoder_pipeline_model_parallel_size

    last_rank = pp_ranks[-1]
    if len(pp_ranks) == 1:
        return [last_rank]
    else:
        return [pp_ranks[epp]]


if __name__ == "__main__":
    train_valid_test_datasets_provider.is_distributed = True

    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_and_decoder,
        forward_step,
        args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
        extra_args_provider=add_vlm_extra_args,
        get_embedding_ranks=llava_embedding_ranks,
        get_position_embedding_ranks=llava_position_embedding_ranks,
    )
