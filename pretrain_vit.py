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
from functools import partial
from megatron.mpu.initialize import get_tensor_model_parallel_group
from megatron.mpu.initialize import get_tensor_model_parallel_rank
from megatron.mpu.initialize import get_tensor_model_parallel_src_rank
from megatron.mpu import cls_parallel_cross_entropy
import torch
import torch.nn.functional as F
from megatron import get_args, get_timers, mpu, print_rank_0
from megatron.data.vit_dataset import build_train_valid_datasets
from megatron.model.vit_model import VitModel, VitModelPipe
from megatron.training import pretrain
from megatron.utils import average_losses_across_data_parallel_group

import deepspeed
from deepspeed.runtime.utils import see_memory_usage
from deepspeed.accelerator.real_accelerator import get_accelerator

def model_provider(pre_process=True, post_process=True):
    """Build the model."""

    print_rank_0("building VIT model ...")
    args = get_args()
    with deepspeed.zero.Init(data_parallel_group=mpu.get_data_parallel_group(),
                             remote_device=None if args.remote_device == 'none' else args.remote_device,
                             config_dict_or_path=args.deepspeed_config,
                             enabled=args.zero_stage == 3,
                             mpu=mpu):
        if args.deepspeed and not args.no_pipeline_parallel:
            model = VitModelPipe(
                num_tokentypes=0,
                parallel_output=True
            )
            # This is a hack to give us a reference to get_batch_pipe from within training.py
            # We need to call model.set_batch_fn after deepspeed.initialize
            model._megatron_batch_fn = get_batch_pipe
        else:
            model = VitModel(num_classes=args.num_classes, pre_process=pre_process, post_process=post_process)
    return model

def get_batch(data_iterator):
    """Build the batch."""
    args = get_args()
    datatype = torch.int64
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    batch_size = args.micro_batch_size // args.data_parallel_size
    if get_tensor_model_parallel_rank() == 0:
        images = data[0].to(get_accelerator().device_name())
        labels = data[1].to(get_accelerator().device_name())
    else:
        images = torch.empty((batch_size, 3, 224, 224),
                                   device=get_accelerator().current_device_name(),
                                   dtype=torch.float16)
        labels = torch.empty((batch_size),
                                   device=get_accelerator().current_device_name(),
                                   dtype=datatype)
    keys = ['text']
    datatype = torch.float32

    # Broadcast
    torch.distributed.broadcast(images, get_tensor_model_parallel_src_rank(),
                                group=get_tensor_model_parallel_group())
    torch.distributed.broadcast(labels, get_tensor_model_parallel_src_rank(),
                                group=get_tensor_model_parallel_group())
    images = images.contiguous()
    labels = labels.contiguous()
    return images, labels

def get_batch_pipe(data):
    """Modification of `get_batch` to work on `next(data_iterator)` instead of `data_iterator`"""
    args = get_args()

    datatype = torch.int64
    batch_size = args.micro_batch_size // args.data_parallel_size
    if get_tensor_model_parallel_rank() == 0:
        images = data[0].to(get_accelerator().device_name())
        labels = data[1].to(get_accelerator().device_name())
    else:
        images = torch.empty((batch_size, 3, 224, 224),
                                   device=get_accelerator().current_device_name(),
                                   dtype=torch.float16)
        labels = torch.empty((batch_size),
                                   device=get_accelerator().current_device_name(),
                                   dtype=datatype)
    # Broadcast data.
    torch.distributed.broadcast(images, get_tensor_model_parallel_src_rank(),
                                group=get_tensor_model_parallel_group())
    torch.distributed.broadcast(labels, get_tensor_model_parallel_src_rank(),
                                group=get_tensor_model_parallel_group())

    return images, labels

def loss_func(labels, logits):
    loss = cls_parallel_cross_entropy(logits.contiguous(), labels).mean()
    outputs = torch.argmax(logits, -1)
    correct = (outputs == labels).float()
    accuracy = torch.mean(correct)

    averaged_loss = average_losses_across_data_parallel_group([loss, accuracy])
    return loss, {"loss": averaged_loss[0], "accuracy": averaged_loss[1]}

def forward_step(data_iterator, model):
    """Forward step."""
    timers = get_timers()
#    assert input_tensor is None

    # Get the batch.
    timers("batch-generator").start()
    (
        images,
        labels,
    ) = get_batch(data_iterator)
    timers("batch-generator").stop()

    # Forward model. lm_labels
    logits = model(images).contiguous()#.float()
    return logits, partial(loss_func, labels)

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
        args_defaults={'dataloader_type': 'cyclic'}
    )
