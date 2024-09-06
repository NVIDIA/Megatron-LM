# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
"""Pretrain vision language model."""
from copy import deepcopy
from functools import partial

import torch

from megatron.core import parallel_state, tensor_parallel
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.multimodal_dataset import MockMultimodalDataset, MultimodalDatasetConfig
from megatron.core.enums import ModelType
from megatron.core.models.multimodal.llava_model import LLaVAModel, IMAGE_TOKEN_INDEX
from megatron.core.models.multimodal.llava_spec import (
    decoder_model_with_transformer_engine_default_spec,
    decoder_model_with_local_default_spec,
)
from megatron.core.models.vision.vit_layer_specs import (
    get_vit_layer_with_transformer_engine_spec,
    get_vit_layer_with_local_spec,
)
from megatron.core.transformer.spec_utils import import_module
from megatron.training import get_args, get_timers, get_tokenizer, pretrain, print_rank_0
from megatron.training.arguments import core_transformer_config_from_args
from pretrain_gpt import loss_func


def get_num_image_tokens():
    args = get_args()
    add_class_token = not args.disable_vision_class_token

    num_patches_per_dim_h = args.img_h // args.patch_dim
    num_patches_per_dim_w = args.img_w // args.patch_dim
    num_patches = num_patches_per_dim_h * num_patches_per_dim_w
    num_image_tokens = num_patches + (1 if add_class_token else 0)
    return num_image_tokens


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

    num_image_tokens = get_num_image_tokens()
    args.decoder_seq_length = args.seq_length + num_image_tokens
    args.seq_length = num_image_tokens
    args.max_position_embeddings = max(args.max_position_embeddings, args.decoder_seq_length)

    print_rank_0('building a multimodal model ...')
    language_transformer_config = core_transformer_config_from_args(get_args())

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

    if args.transformer_impl == "transformer_engine":
        vision_transformer_layer_spec = get_vit_layer_with_transformer_engine_spec()
    else:  # transformer_impl == "local"
        vision_transformer_layer_spec = get_vit_layer_with_local_spec()

    # TODO: Make these configurable via input .yaml config.
    vision_transformer_config = deepcopy(language_transformer_config)
    vision_transformer_config.num_layers = args.encoder_num_layers
    vision_transformer_config.first_pipeline_num_layers = None
    vision_transformer_config.last_pipeline_num_layers = None

    vision_projection_type = "mlp"
    vision_projection_config = deepcopy(language_transformer_config)

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

    model = LLaVAModel(
        language_transformer_config=language_transformer_config,
        language_transformer_layer_spec=language_transformer_layer_spec,
        language_vocab_size=args.padded_vocab_size,
        language_max_sequence_length=args.max_position_embeddings,
        vision_transformer_config=vision_transformer_config,
        vision_transformer_layer_spec=vision_transformer_layer_spec,
        drop_vision_class_token=args.disable_vision_class_token,
        vision_projection_config=vision_projection_config,
        vision_projection_layer_spec=vision_projection_modules,
        vision_projection_type=vision_projection_type,
        parallel_output=parallel_output,
        language_position_embedding_type=args.position_embedding_type,
        language_rotary_percent=args.rotary_percent,
        pre_process=pre_process,
        post_process=post_process,
        add_encoder=add_encoder,
        add_decoder=add_decoder,
        img_h=args.img_h,
        img_w=args.img_w,
        patch_dim=args.patch_dim,
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
        sequence_length=args.decoder_seq_length - args.seq_length,
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
            IMAGE_TOKEN_INDEX
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
    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None

    data_i = tensor_parallel.broadcast_data(["tokens", "position_ids", "labels"], data, torch.int64)
    data_f = tensor_parallel.broadcast_data(["image", "loss_mask"], data, torch.float32)

    tokens = data_i["tokens"].long()
    position_ids = data_i["position_ids"].long()
    labels = data_i["labels"].long()
    images = data_f["image"].float()
    loss_mask = data_f["loss_mask"].float()
    attention_mask = None  # Use the attention mask type defined in layer spec. Typically no mask for the vision model and causal mask for the vision model.

    return tokens, position_ids, labels, images, loss_mask, attention_mask


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
    tokens, position_ids, labels, images, loss_mask, attention_mask = get_batch(data_iterator)
    timers('batch-generator').stop()

    output_tensor, loss_mask = model(
        images, tokens, position_ids, attention_mask, labels, loss_mask
    )

    return output_tensor, partial(loss_func, loss_mask)


def add_vlm_extra_args(parser):
    """Extra arguments."""
    group = parser.add_argument_group(title='vision language model specific arguments')
    group.add_argument("--disable-vision-class-token", action="store_true", default=False)
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
