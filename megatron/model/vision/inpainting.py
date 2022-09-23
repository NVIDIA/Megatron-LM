# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.
i
import math
import apex
import einops
import torch
import torch.nn.functional as F
from megatron import get_args, print_rank_0
from megatron.model.utils import get_linear_layer
from megatron.model.vision.vit_backbone import VitBackbone
from megatron.model.module import MegatronModule
from megatron.model.vision.mit_backbone import mit_b3
from megatron.model.vision.utils import resize_


class VitInpaintingModel(MegatronModule):

    def __init__(self, pre_process=True, post_process=True):
        super(VitInpaintingModel, self).__init__()
        args = get_args()

        self.pre_process = pre_process
        self.post_process = post_process
        self.hidden_size = args.hidden_size
        self.backbone = VitBackbone(
            pre_process=self.pre_process,
            post_process=self.post_process,
            class_token=False,
        )
        self.patch_dim = args.patch_dim
        self.img_h = args.img_h
        self.img_w = args.img_w
        self.seq_length = args.seq_length
        # full mask

        if self.post_process:
            self.linear_decoder = get_linear_layer(
                self.hidden_size,
                self.backbone.flatten_dim,
                torch.nn.init.zeros_
            )

    def set_input_tensor(self, input_tensor):
        self.backbone.set_input_tensor(input_tensor)

    def forward(self, input):

        hidden_states = self.backbone(input)

        if not self.post_process:
            return hidden_states
        decoded_output = self.linear_decoder(hidden_states)
        output = einops.rearrange(
                decoded_output,
                "b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
                p1=self.patch_dim,
                p2=self.patch_dim,
                h=self.img_h//self.patch_dim,
                w=self.img_w//self.patch_dim,
            )

        return output


class MLP(torch.nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = torch.nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class MitInpaintingModel(MegatronModule):
    """Mix vision Transformer Model."""

    def __init__(self, pre_process=True, post_process=True):
        super(MitInpaintingModel, self).__init__()
        self.pre_process = pre_process
        self.post_process = post_process

        args = get_args()
        self.patch_dim = args.patch_dim
        self.img_h = args.img_h
        self.img_w = args.img_w
        self.flatten_dim = self.patch_dim * self.patch_dim * 3
        self.backbone = mit_b3()

        self.in_channels = [64, 128, 320, 512]
        self.embedding_dim = 768

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=self.embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=self.embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=self.embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=self.embedding_dim)

        self.conv_fuse = torch.nn.Conv2d(self.embedding_dim*4, self.embedding_dim, 1, 1, bias=False)
        self.norm = apex.parallel.SyncBatchNorm(self.embedding_dim)
        self.dropout = torch.nn.Dropout2d(0.1)
        
        self.linear_pred = torch.nn.Conv2d(self.embedding_dim, self.flatten_dim, kernel_size=1)
    
    def set_input_tensor(self, input_tensor):
        """See megatron.model.transformer.set_input_tensor()"""
        pass

    def forward(self, input):
        c1, c2, c3, c4 = self.backbone(input)

        n, _, h, w = c4.shape
        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = resize(_c4, size=c1.size()[2:], mode='bilinear', align_corners=False)
    
        _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = resize(_c3, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = resize(_c2, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c = torch.cat([_c4, _c3, _c2, _c1], dim=1)
        _c = self.conv_fuse(_c)
 
        x = self.norm(_c)
        x = F.relu(x, inplace=True)
        x = self.dropout(x)

        x = self.linear_pred(x)

        output = einops.rearrange(
            x,
            "b (c p1 p2) h w -> b c (h p1) (w p2)",
            p1=self.patch_dim,
            p2=self.patch_dim,
            h=self.img_h//self.patch_dim,
            w=self.img_w//self.patch_dim,
        )

        return output
