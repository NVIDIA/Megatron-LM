# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.

import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch.distributed as dist
from functools import partial
from megatron import get_args, get_timers, print_rank_0
from megatron.data.vit_dataset import build_train_valid_datasets
from megatron.model.vision.dino import DINOPretrainModel
from megatron.model.vision.knn_monitor import knn_predict, get_feature_bank
from megatron.training import pretrain
from megatron.utils import average_losses_across_data_parallel_group, unwrap_model
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP
from megatron.model import DistributedDataParallel as LocalDDP
from megatron.model import Float16Module
from megatron.model import ModelType

def model_provider(pre_process=True, post_process=True):
    """Build the model."""
    return DINOPretrainModel(pre_process=pre_process, post_process=post_process)

def get_batch(data_iterator):
    """Build the batch."""
    data = next(data_iterator)

    # only data parallelism; no need for broadcast
    if isinstance(data[0], list):
        images = [aug.cuda() for aug in data[0]]
    else:
        images = data[0].cuda()
    labels = data[1].cuda()

    return images, labels


def loss_func(model, labels, output_tensor, collect_data=False):
    args = get_args()
    
    model = unwrap_model(
        model,
        (torchDDP, LocalDDP, Float16Module)
    )
    if model.training:
        student_output, teacher_output = output_tensor
        loss = model.dino_loss(student_output, teacher_output, args.curr_iteration)
        averaged_loss = average_losses_across_data_parallel_group([loss])
        return loss, {"loss": averaged_loss[0]}
    else:
        _, teacher_feature = output_tensor
        feature_bank, feature_labels, classes = get_feature_bank()
        feature = F.normalize(teacher_feature.float(), dim=1)

        knn_accs = []
        for k in [10, 20, 100, 200]:
            pred_labels = knn_predict(feature, feature_bank,
                                      feature_labels, classes, k, 0.07)
            knn_acc = (pred_labels[:, 0] == labels).float().mean()
            knn_accs.append(knn_acc)

        averaged_loss = average_losses_across_data_parallel_group(knn_accs)
        return 0, {"knn_acc_10": averaged_loss[0],
                   "knn_acc_20": averaged_loss[1],
                   "knn_acc_100": averaged_loss[2],
                   "knn_acc_200": averaged_loss[3]}


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

    return model(images), partial(loss_func, model, labels)


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

