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

from megatron import get_args
from megatron import print_rank_0
from megatron.model.vit_model import VitModel
from megatron.data.vit_dataset import build_train_valid_datasets
from tasks.vision.eval_utils import accuracy_func_provider
from tasks.vision.finetune_utils import finetune


def classification():
    def train_valid_datasets_provider():
        """Build train and validation dataset."""
        args = get_args()

        train_ds, valid_ds = build_train_valid_datasets(
            data_path=args.data_path,
            crop_size=args.img_dim,
        )
        return train_ds, valid_ds

    def model_provider(pre_process=True, post_process=True):
        """Build the model."""
        args = get_args()

        print_rank_0("building classification model for ImageNet ...")

        return VitModel(num_classes=args.num_classes, finetune=True,
                        pre_process=pre_process, post_process=post_process)

    """Finetune/evaluate."""
    finetune(
        train_valid_datasets_provider,
        model_provider,
        end_of_epoch_callback_provider=accuracy_func_provider,
    )


def main():
    classification()
