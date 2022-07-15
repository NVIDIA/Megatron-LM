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
import math
import einops
import torch
import apex
import torch.nn.functional as F
from megatron import get_args
from megatron.model.module import MegatronModule
from megatron.model.vision.vit_backbone import VitBackbone, VitMlpHead
from megatron.model.vision.mit_backbone import mit_b3, mit_b5
from tasks.vision.segmentation.seg_heads import SetrSegmentationHead, SegformerSegmentationHead


class SetrSegmentationModel(MegatronModule):

    def __init__(self,
                 num_classes,
                 pre_process=True,
                 post_process=True):
        super(SetrSegmentationModel, self).__init__()
        args = get_args()
        assert post_process & pre_process
        self.hidden_size = args.hidden_size
        self.num_classes = num_classes
        self.backbone = VitBackbone(
            pre_process=pre_process,
            post_process=post_process,
            class_token=False,
            post_layer_norm=False,
            drop_path_rate=0.1
        )

        self.head = SetrSegmentationHead(
            self.hidden_size,
            self.num_classes
        )

    def set_input_tensor(self, input_tensor):
        """See megatron.model.transformer.set_input_tensor()"""
        pass

    def forward(self, input):
        # [b hw c]
        hidden_states = self.backbone(input)
        result_final = self.head(hidden_states)
        return result_final


class SegformerSegmentationModel(MegatronModule):

    def __init__(self,
                 num_classes,
                 pre_process=True,
                 post_process=True):
        super(SegformerSegmentationModel, self).__init__()
        args = get_args()
        self.hidden_size = args.hidden_size
        self.num_classes = num_classes
        self.pre_process = pre_process
        self.post_process = post_process

        self.backbone = mit_b5()
        self.head = SegformerSegmentationHead(
            feature_strides=[4, 8, 16, 32],
            in_channels=[64, 128, 320, 512],
            embedding_dim=768,
            dropout_ratio=0.1
        )

    def set_input_tensor(self, input_tensor):
        """See megatron.model.transformer.set_input_tensor()"""
        pass

    def forward(self, input):
        # [b hw c]
        hidden_states = self.backbone(input)
        hidden_states = self.head(hidden_states)
        return hidden_states

