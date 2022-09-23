# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.

"""Pretrain VIT"""

import torch
import torch.nn.functional as F
from functools import partial
from megatron import get_args, get_timers, print_rank_0
from megatron.data.vit_dataset import build_train_valid_datasets
from megatron.model import ModelType
from megatron.model.vision.classification import VitClassificationModel
from megatron.model.vision.classification import MitClassificationModel
from megatron.training import pretrain
from megatron.utils import average_losses_across_data_parallel_group


def model_provider(pre_process=True, post_process=True):
    """Build the model."""

    args = get_args()

    if args.vision_backbone_type == 'vit':
        print_rank_0("building VIT model ...")
        model = VitClassificationModel(num_classes=args.num_classes,
                                       pre_process=pre_process,
                                       post_process=post_process)
    elif args.vision_backbone_type == 'mit':
        print_rank_0("building MIT model ...")
        model = MitClassificationModel(num_classes=args.num_classes,
                                       pre_process=pre_process,
                                       post_process=post_process)
    else:
        raise Exception('{} vision backbone is not supported.'.format(
                              args.vision_backbone_type))
    return model


def get_batch(data_iterator):
    """Build the batch."""
    data = next(data_iterator)

    # only data parallelism; no need for broadcast
    images = data[0].cuda()
    labels = data[1].cuda()

    return images, labels


def loss_func(labels, output_tensor):
    logits = output_tensor.contiguous().float()
    loss = F.cross_entropy(logits, labels)

    outputs = torch.argmax(logits, -1)
    correct = (outputs == labels).float()
    accuracy = torch.mean(correct)

    averaged_loss = average_losses_across_data_parallel_group([loss, accuracy])

    return loss, {"loss": averaged_loss[0], "accuracy": averaged_loss[1]}


def forward_step(data_iterator, model):
    """Forward step."""
    timers = get_timers()

    # Get the batch.
    timers("batch-generator", log_level=2).start()
    (
        images,
        labels,
    ) = get_batch(data_iterator)
    timers("batch-generator").stop()

    # Forward model. lm_labels
    output_tensor = model(images)

    return output_tensor, partial(loss_func, labels)

def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()

    print_rank_0(
        "> building train, validation, and test datasets " "for VIT ..."
    )
    train_ds, valid_ds = build_train_valid_datasets(
        data_path=args.data_path,
        image_size=(args.img_h, args.img_w)
    )
    print_rank_0("> finished creating VIT datasets ...")

    return train_ds, valid_ds, None


if __name__ == "__main__":

    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        args_defaults={'dataloader_type': 'cyclic', 'vision_pretraining': True}
    )
