# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
"""Pretrain vision language model."""

from functools import partial

import torch

from megatron import get_args, get_timers, get_tokenizer, print_rank_0
from megatron.arguments import core_transformer_config_from_args
from megatron.core import tensor_parallel
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.multimodal_dataset import MockMultimodalDataset, MultimodalDatasetConfig
from megatron.core.enums import ModelType
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.multimodal.llava_model import LLaVAModel
from megatron.core.transformer.spec_utils import import_module
from megatron.training import pretrain
from pretrain_gpt import is_dataset_built_on_rank, loss_func


def model_provider(pre_process=True, post_process=True) -> LLaVAModel:
    """Builds the model.

    Note: currently, only LLaVA model is supported. Follow-up changes will make this configurable.

    Args:
        pre_process (bool): Enable preprocessing in the model. NOTE: Not used at the moment.
        post_process (bool): Enable postprocessing in the model. NOTE: Not used at the moment.

    Returns:
        model (megatron.core.models.multimodal.llava_model.LLaVAModel): A multimodal model
    """
    args = get_args()

    print_rank_0('building a multimodal model ...')
    config = core_transformer_config_from_args(get_args())

    if args.spec is not None:
        transformer_layer_spec = import_module(args.spec)
    else:
        transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
            args.num_experts, args.moe_grouped_gemm
        )

    model = LLaVAModel(
        language_transformer_config=config,
        language_transformer_layer_spec=transformer_layer_spec,
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        vision_transformer_config=config,
        vision_transformer_layer_spec=transformer_layer_spec,
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

    tokenizer = get_tokenizer()

    config = MultimodalDatasetConfig(
        random_seed=args.seed,
        sequence_length=args.seq_length,
        tokenizer=tokenizer,
        reset_position_ids=args.reset_position_ids,
        reset_attention_mask=args.reset_attention_mask,
        eod_mask_loss=args.eod_mask_loss,
        mock=True,
        image_h=args.img_h,
        image_w=args.img_w,
        preprocess_func=_preprocess_data_for_llava,
    )

    dataset_type = MockMultimodalDataset

    print_rank_0("> building train, validation, and test datasets for multimodal ...")

    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        dataset_type, train_val_test_num_samples, is_dataset_built_on_rank, config
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
    args = get_args()

    # TODO: Move these to multimodal spec (added in a separate code change).
    class_token_len = 1
    add_class_token = True

    num_patches_per_dim_h = args.img_h // args.patch_dim
    num_patches_per_dim_w = args.img_w // args.patch_dim
    num_patches = num_patches_per_dim_h * num_patches_per_dim_w
    num_image_tokens = num_patches + (class_token_len if add_class_token else 0)

    data["loss_mask"] = torch.cat(
        [torch.zeros(num_image_tokens, dtype=torch.float32), data["loss_mask"]]
    )
    data["labels"] = torch.cat([torch.zeros(num_image_tokens, dtype=torch.int64), data["labels"]])

    full_seq_length = len(data["labels"])
    attention_mask = torch.tril(torch.ones((1, full_seq_length, full_seq_length)))
    attention_mask = attention_mask < 0.5
    attention_mask[:, num_image_tokens:, num_image_tokens:] = data["attention_mask"]
    data["attention_mask"] = attention_mask

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
    data_b = tensor_parallel.broadcast_data(["attention_mask"], data, torch.bool)

    tokens = data_i["tokens"].long()
    position_ids = data_i["position_ids"].long()
    labels = data_i["labels"].long()
    images = data_f["image"].float()
    loss_mask = data_f["loss_mask"].float()
    attention_mask = data_b["attention_mask"].bool()

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

    output_tensor = model(images, tokens, position_ids, attention_mask, labels=labels)

    return output_tensor, partial(loss_func, loss_mask)


if __name__ == "__main__":
    train_valid_test_datasets_provider.is_distributed = True

    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
    )
