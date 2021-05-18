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

"""Evaluation utilities."""

import os
from functools import partial

import torch

from megatron import get_args
from megatron import print_rank_0, print_rank_last
from megatron import mpu
from megatron.schedules import get_forward_backward_func
from tasks.vision.finetune_utils import build_data_loader
from tasks.vision.finetune_utils import process_batch
from torchvision import datasets, transforms


def accuracy_func_provider():
    """Provide function that calculates accuracies."""
    args = get_args()
    data_path = args.data_path
    crop_size = args.img_dim

    # mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    # Build dataloaders.
    val_data_path = os.path.join(data_path[0], "val")
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    transform_val = transforms.Compose(
        [
            transforms.Resize(crop_size),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            normalize,
        ]
    )
    dataset = datasets.ImageFolder(root=val_data_path, transform=transform_val)

    dataloader = build_data_loader(
        dataset,
        args.micro_batch_size,
        num_workers=args.num_workers,
        drop_last=(mpu.get_data_parallel_world_size() > 1),
    )

    def metrics_func(model, epoch):
        print_rank_0("calculating metrics ...")
        correct, total = calculate_correct_answers(model, dataloader, epoch)
        percent = float(correct) * 100.0 / float(total)
        print_rank_last(
            " >> |epoch: {}| overall: correct / total = {} / {} = "
            "{:.4f} %".format(epoch, correct, total, percent)
        )

    return metrics_func


def calculate_correct_answers(model, dataloader, epoch):
    """Calculate correct over total answers"""

    args = get_args()
    forward_backward_func = get_forward_backward_func()
    for m in model:
        m.eval()

    def loss_func(labels, output_tensor):
        logits = output_tensor

        loss_dict = {}
        # Compute the correct answers.
        predicted = torch.argmax(logits, dim=-1)
        corrects = (predicted == labels).float()
        # Add to the counters.
        loss_dict['total'] = labels.size(0)
        loss_dict['correct'] = corrects.sum().item()

        return 0, loss_dict

    #defined inside to capture output_predictions
    def correct_answers_forward_step(batch, model):
        try:
            batch_ = next(batch)
        except BaseException:
            batch_ = batch
        images, labels = process_batch(batch_)

        # Forward model.
        args = get_args()
        output_tensor = model(images)

        return output_tensor, partial(loss_func, labels)

    with torch.no_grad():
        # For all the batches in the dataset.
        total = 0
        correct = 0
        for _, batch in enumerate(dataloader):

            loss_dicts = forward_backward_func(correct_answers_forward_step, batch, model,
                                               optimizer=None, timers=None, forward_only=True)

            for loss_dict in loss_dicts:
                total += loss_dict['total']
                correct += loss_dict['correct']

    for m in model:
        m.train()

    # Reduce.
    if mpu.is_pipeline_last_stage():
        unreduced = torch.cuda.LongTensor([correct, total])
        torch.distributed.all_reduce(unreduced,
                                     group=mpu.get_data_parallel_group())

        # Print on screen.
        correct_ans = unreduced[0].item()
        total_count = unreduced[1].item()
        return correct_ans, total_count
