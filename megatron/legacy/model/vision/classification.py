# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Vision Transformer(VIT) model."""

import torch
from torch.nn.init import trunc_normal_
from megatron.training import get_args
from megatron.legacy.model.utils import get_linear_layer
from megatron.legacy.model.vision.vit_backbone import VitBackbone, VitMlpHead
from megatron.legacy.model.vision.mit_backbone import mit_b3_avg
from megatron.legacy.model.module import MegatronModule

class VitClassificationModel(MegatronModule):
    """Vision Transformer Model."""

    def __init__(self, config, num_classes, finetune=False,
                 pre_process=True, post_process=True):
        super(VitClassificationModel, self).__init__()
        args = get_args()
        self.config = config

        self.hidden_size = args.hidden_size
        self.num_classes = num_classes
        self.finetune = finetune
        self.pre_process = pre_process
        self.post_process = post_process
        self.backbone = VitBackbone(
            config=config,
            pre_process=self.pre_process,
            post_process=self.post_process,
            single_token_output=True
        )

        if self.post_process:
            if not self.finetune:
                self.head = VitMlpHead(config, self.hidden_size, self.num_classes)
            else:
                self.head = get_linear_layer(
                    self.hidden_size,
                    self.num_classes,
                    torch.nn.init.zeros_
                )

    def set_input_tensor(self, input_tensor):
        """See megatron.legacy.model.transformer.set_input_tensor()"""
        self.backbone.set_input_tensor(input_tensor)

    def forward(self, input):
        hidden_states = self.backbone(input)

        if self.post_process:
            hidden_states = self.head(hidden_states)

        return hidden_states


class MitClassificationModel(MegatronModule):
    """Mix vision Transformer Model."""

    def __init__(self, num_classes,
                 pre_process=True, post_process=True):
        super(MitClassificationModel, self).__init__()
        args = get_args()

        self.hidden_size = args.hidden_size
        self.num_classes = num_classes

        self.backbone = mit_b3_avg()
        self.head = torch.nn.Linear(512, num_classes)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, torch.nn.Linear) and m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)

    def set_input_tensor(self, input_tensor):
        """See megatron.legacy.model.transformer.set_input_tensor()"""
        pass

    def forward(self, input):
        hidden_states = self.backbone(input)
        hidden_states = self.head(hidden_states)

        return hidden_states
