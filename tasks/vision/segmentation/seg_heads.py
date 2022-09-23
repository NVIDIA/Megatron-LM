# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
import math
import einops
import torch
import apex
import torch.nn.functional as F
from megatron import get_args
from megatron.model import LayerNorm
from megatron.model.module import MegatronModule
from megatron.model.vision.utils import resize


class SetrSegmentationHead(MegatronModule):
    def __init__(self, hidden_size, num_classes):
        super(SetrSegmentationHead, self).__init__()
        args = get_args()
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.img_h = args.img_h
        self.img_w = args.img_w
        self.patch_dim = args.patch_dim

        self.layernorm = LayerNorm(hidden_size, eps=args.layernorm_epsilon)
        self.conv_0 = torch.nn.Conv2d(hidden_size, hidden_size,
                                      1, 1, bias=False)
        self.norm_0 = apex.parallel.SyncBatchNorm(hidden_size)
        self.conv_1 = torch.nn.Conv2d(hidden_size, num_classes, 1, 1)

    def to_2D(self, x):
        n, hw, c = x.shape
        h = self.img_h // self.patch_dim
        w = self.img_w // self.patch_dim
        assert(hw == h * w)
        x = x.transpose(1, 2).reshape(n, c, h, w)
        return x

    def forward(self, hidden_states):
        # [b c h w]
        hidden_states = self.layernorm(hidden_states)
        hidden_states = self.to_2D(hidden_states)

        hidden_states = self.conv_0(hidden_states)
        hidden_states = self.norm_0(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.conv_1(hidden_states)

        # [b c h w]
        result = F.interpolate(hidden_states,
                               size=(self.img_h, self.img_w),
                               mode='bilinear')

        return result


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


class SegformerSegmentationHead(MegatronModule):
    def __init__(self, feature_strides, in_channels,
                 embedding_dim, dropout_ratio):
        super(SegformerSegmentationHead, self).__init__()
        assert len(feature_strides) == len(in_channels)
        assert min(feature_strides) == feature_strides[0]
        args = get_args()
        self.feature_strides = feature_strides
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim
        self.num_classes = args.num_classes
        self.dropout_ratio = dropout_ratio

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = \
            self.in_channels

        self.linear_c4 = MLP(input_dim=c4_in_channels,
                             embed_dim=self.embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels,
                             embed_dim=self.embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels,
                             embed_dim=self.embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels,
                             embed_dim=self.embedding_dim)

        self.conv_fuse = torch.nn.Conv2d(self.embedding_dim*4,
                                         self.embedding_dim, 1, 1)
        self.norm = apex.parallel.SyncBatchNorm(self.embedding_dim)

        self.dropout = torch.nn.Dropout2d(self.dropout_ratio)
        self.linear_pred = torch.nn.Conv2d(self.embedding_dim,
                                           self.num_classes,
                                           kernel_size=1)

    def forward(self, inputs):
        c1, c2, c3, c4 = inputs

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = resize(_c4, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = resize(_c3, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = resize(_c2, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c = self.conv_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))
        x = self.norm(_c)
        x = F.relu(x, inplace=True)
        x = self.dropout(x)
        x = self.linear_pred(x)

        return x

