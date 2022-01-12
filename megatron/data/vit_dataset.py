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
import random
import numpy as np
import torch
import torchvision.transforms as T
from torchvision import datasets
from megatron import get_args
from megatron.data.image_folder import ImageFolder
from megatron.data.autoaugment import ImageNetPolicy
from megatron.data.data_samplers import RandomSeedDataset

class ClassificationTransform():
    def __init__(self, image_size, train=True):
        args = get_args()
        assert args.fp16 or args.bf16
        self.data_type = torch.half if args.fp16 else torch.bfloat16
        if train:
            self.transform = T.Compose([
                T.RandomResizedCrop(image_size),
                T.RandomHorizontalFlip(),
                T.ColorJitter(0.4, 0.4, 0.4, 0.1),
                ImageNetPolicy(),
                T.ToTensor(),
                T.Normalize(*self.mean_std),
                T.ConvertImageDtype(self.data_type)
            ])
        else:
            self.transform = T.Compose([
                T.Resize(image_size),
                T.CenterCrop(image_size),
                T.ToTensor(),
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                T.ConvertImageDtype(self.data_type)
            ])

    def __call__(self, input):
        output = self.transform(input)
        return output



def build_train_valid_datasets(data_path, image_size=224):
    args = get_args()
    train_transform = ClassificationTransform(image_size)
    val_transform = ClassificationTransform(image_size, train=False)

    # training dataset
    train_data_path = data_path[0]
    train_data = ImageFolder(
        root=train_data_path,
        transform=train_transform,
        classes_fraction=args.classes_fraction,
        data_per_class_fraction=args.data_per_class_fraction
    )
    train_data = RandomSeedDataset(train_data)

    # validation dataset
    val_data_path = data_path[1]
    val_data = ImageFolder(
        root=val_data_path,
        transform=val_transform
    )
    val_data = RandomSeedDataset(val_data)

    return train_data, val_data
