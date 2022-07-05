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

"""Vision-classification finetuning/evaluation."""

import torch.nn.functional as F
from functools import partial
from megatron import get_args, get_timers
from megatron import print_rank_0
from megatron.model.vision.classification import VitClassificationModel
from megatron.data.vit_dataset import build_train_valid_datasets
from tasks.vision.classification.eval_utils import accuracy_func_provider
from tasks.vision.finetune_utils import finetune
from megatron.utils import average_losses_across_data_parallel_group


def classification():
    def train_valid_datasets_provider():
        """Build train and validation dataset."""
        args = get_args()

        train_ds, valid_ds = build_train_valid_datasets(
            data_path=args.data_path,
            image_size=(args.img_h, args.img_w),
        )
        return train_ds, valid_ds

    def model_provider(pre_process=True, post_process=True):
        """Build the model."""
        args = get_args()

        print_rank_0("building classification model for ImageNet ...")

        return VitClassificationModel(num_classes=args.num_classes, finetune=True,
                                      pre_process=pre_process, post_process=post_process)

    def process_batch(batch):
        """Process batch and produce inputs for the model."""
        images = batch[0].cuda().contiguous()
        labels = batch[1].cuda().contiguous()
        return images, labels

    def cross_entropy_loss_func(labels, output_tensor):
        logits = output_tensor

        # Cross-entropy loss.
        loss = F.cross_entropy(logits.contiguous().float(), labels)

        # Reduce loss for logging.
        averaged_loss = average_losses_across_data_parallel_group([loss])

        return loss, {'lm loss': averaged_loss[0]}

    def _cross_entropy_forward_step(batch, model):
        """Simple forward step with cross-entropy loss."""
        timers = get_timers()

        # Get the batch.
        timers("batch generator").start()
        try:
            batch_ = next(batch)
        except BaseException:
            batch_ = batch
        images, labels = process_batch(batch_)
        timers("batch generator").stop()

        # Forward model.
        output_tensor = model(images)
      
        return output_tensor, partial(cross_entropy_loss_func, labels)

    """Finetune/evaluate."""
    finetune(
        train_valid_datasets_provider,
        model_provider,
        forward_step=_cross_entropy_forward_step,
        end_of_epoch_callback_provider=accuracy_func_provider,
    )

def main():
    classification()

