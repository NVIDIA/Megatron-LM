# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pretrain VIT"""

import torch
import torch.nn.functional as F
from megatron import get_args, get_timers, mpu, print_rank_0
from megatron.data.vit_dataset import build_train_valid_datasets
from megatron.model import VitModel
from megatron.training import pretrain
from megatron.utils import average_losses_across_data_parallel_group


def model_provider():
    """Build the model."""

    print_rank_0("building VIT model ...")
    args = get_args()

    model = VitModel(num_classes=args.num_classes)
    return model


def get_batch(data_iterator):
    """Build the batch."""

    # Items and their type.
    keys = ["image", "label"]
    datatype = torch.half

    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None

    dict_data = {}
    dict_data["image"] = data[0].half()
    dict_data["label"] = data[1].half()
    data_b = mpu.broadcast_data(keys, dict_data, datatype)

    # Unpack.
    images = data_b["image"]
    labels = data_b["label"].long()
    return images, labels


def forward_step(data_iterator, model, input_tensor):
    """Forward step."""
    timers = get_timers()
    assert input_tensor is None

    # Get the batch.
    timers("batch generator").start()
    (
        images,
        labels,
    ) = get_batch(data_iterator)
    timers("batch generator").stop()

    # Forward model. lm_labels
    logits = model(images).contiguous().float()
    loss = F.cross_entropy(logits, labels)

    outputs = torch.argmax(logits, -1)
    correct = (outputs == labels).float()
    accuracy = torch.mean(correct)

    averaged_loss = average_losses_across_data_parallel_group([loss, accuracy])

    return loss, {"loss": averaged_loss[0], "accuracy": averaged_loss[1]}


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()

    print_rank_0(
        "> building train, validation, and test datasets " "for VIT ..."
    )
    train_ds, valid_ds = build_train_valid_datasets(data_path=args.data_path)
    print_rank_0("> finished creating VIT datasets ...")

    return train_ds, valid_ds, None


if __name__ == "__main__":

    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        forward_step,
        random_sample=True
    )
