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
import os
import torch
from torchvision import datasets, transforms
from megatron.data.autoaugment import ImageNetPolicy


def build_train_valid_datasets(data_path, crop_size=224, color_jitter=True):

    # training dataset
    train_data_path = os.path.join(data_path[0], "train")
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    process = [
        transforms.RandomResizedCrop(crop_size),
        transforms.RandomHorizontalFlip(),
    ]
    if color_jitter:
        process += [
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
            )
        ]
    fp16_t = transforms.ConvertImageDtype(torch.half)
    process += [ImageNetPolicy(), transforms.ToTensor(), normalize, fp16_t]
    transform_train = transforms.Compose(process)
    train_data = datasets.ImageFolder(
        root=train_data_path, transform=transform_train
    )

    # validation dataset
    val_data_path = os.path.join(data_path[0], "val")
    transform_val = transforms.Compose(
        [
            transforms.Resize(crop_size),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            normalize,
            fp16_t
        ]
    )
    val_data = datasets.ImageFolder(
        root=val_data_path, transform=transform_val
    )

    return train_data, val_data
