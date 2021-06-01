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

"""Vision Transformer(VIT) model."""

import math
import einops
import torch
import torch.nn.functional as F
from megatron import get_args
from megatron.model.transformer import ParallelTransformer
from megatron.model.utils import (
    get_linear_layer,
    init_method_normal,
    scaled_init_method_normal,
)
from .module import MegatronModule


class VitMlpHead(MegatronModule):
    """Pooler layer.

    Pool hidden states of a specific token (for example start of the
    sequence) and add a linear transformation followed by a tanh.

    Arguments:
        hidden_size: hidden size
        init_method: weight initialization method for the linear layer.
            bias is set to zero.
    """

    def __init__(self, hidden_size, num_classes):
        super(VitMlpHead, self).__init__()
        self.dense_in = torch.nn.Linear(hidden_size, hidden_size)
        self.dense_out = torch.nn.Linear(hidden_size, num_classes)
        torch.nn.init.constant_(self.dense_out.bias, -10)

    def forward(self, hidden_states, sequence_index=0):
        # hidden_states: [b, s, h]
        # sequence_index: index of the token to pool.
        hidden_state = hidden_states[:, sequence_index, :]
        dense_in_result = self.dense_in(hidden_state)
        tanh_result = torch.tanh(dense_in_result)
        dense_out_result = self.dense_out(tanh_result)
        return dense_out_result


def twod_interpolate_position_embeddings_hook(
    state_dict,
    prefix,
    local_metadata,
    strict,
    missing_keys,
    unexpected_keys,
    error_msgs,
):

    args = get_args()
    num_patches_per_dim = args.img_dim // args.patch_dim
    num_patches = num_patches_per_dim ** 2
    seq_length = num_patches + 1
    hidden_size = args.hidden_size

    key = prefix + "weight"
    # import pdb
    # pdb.set_trace()
    assert key in state_dict
    if key in state_dict:
        input_param = state_dict[key]

        assert input_param.shape[1] == hidden_size
        if input_param.shape[0] != seq_length:
            # update input_param and load it to state_dict[key]

            num_tok_input = input_param.shape[0] - 1
            num_tok_new = seq_length - 1
            input_param_tok, input_param_grid = (
                input_param[:1, :],
                input_param[1:, :],
            )

            gs_input = int(math.sqrt(num_tok_input))
            gs_new = int(math.sqrt(num_tok_new))

            input_param_grid = input_param_grid.transpose(0, 1).contiguous()
            input_param_grid = input_param_grid.reshape(
                (1, -1, gs_input, gs_input)
            )
            input_param_grid = input_param_grid.float()
            scale_factor = gs_new / gs_input

            input_param_grid = F.interpolate(
                input_param_grid, scale_factor=scale_factor, mode="bilinear"
            )

            input_param_grid = input_param_grid.half()
            input_param_grid = input_param_grid.reshape((-1, gs_new * gs_new))
            input_param_grid = input_param_grid.transpose(0, 1).contiguous()

            assert input_param_grid.shape[1] == hidden_size
            input_param = torch.cat((input_param_tok, input_param_grid), dim=0)
            assert (
                input_param.shape[0] == seq_length
                and input_param.shape[1] == hidden_size
            )

            state_dict[key] = input_param


class VitModel(MegatronModule):
    """Vision Transformer Model."""

    def __init__(self, 
                 num_classes,
                 finetune=False,
                 pre_process=True,
                 post_process=True):
        super(VitModel, self).__init__(share_word_embeddings=False)
        args = get_args()

        self.fp16_lm_cross_entropy = args.fp16_lm_cross_entropy
        if args.init_method_xavier_uniform:
            self.init_method = torch.nn.init.xavier_uniform_
            self.scaled_init_method = torch.nn.init.xavier_uniform_
        else:
            self.init_method = init_method_normal(args.init_method_std)
            self.scaled_init_method = scaled_init_method_normal(
                args.init_method_std, args.num_layers
            )

        self.pre_process = pre_process
        self.post_process = post_process
        self.hidden_size = args.hidden_size
        self.num_classes = num_classes
        self.patch_dim = args.patch_dim
        self.img_dim = args.img_dim
        self.finetune = finetune

        assert self.img_dim % self.patch_dim == 0
        self.num_patches_per_dim = self.img_dim // self.patch_dim
        self.num_patches = self.num_patches_per_dim ** 2
        self.seq_length = self.num_patches + 1
        self.flatten_dim = self.patch_dim * self.patch_dim * args.num_channels

        if self.pre_process:
            # cls_token
            self.cls_token = torch.nn.Parameter(
                torch.randn(1, 1, self.hidden_size)
            )
            torch.nn.init.zeros_(self.cls_token)

            # Linear encoder
            self.linear_encoder = torch.nn.Linear(
                self.flatten_dim, self.hidden_size
            )

            # embedding
            self.position_embeddings = torch.nn.Embedding(
                self.seq_length, self.hidden_size
            )
            init_method_normal(args.init_method_std)(
                self.position_embeddings.weight
            )
            self.position_ids = torch.arange(self.seq_length).expand(1, -1).cuda()

            self.position_embeddings._register_load_state_dict_pre_hook(
                twod_interpolate_position_embeddings_hook
            )

            self.embedding_dropout = torch.nn.Dropout(args.hidden_dropout)

        # Transformer
        self.transformer = ParallelTransformer(
            self.init_method, 
            self.scaled_init_method,
            pre_process=self.pre_process,
            post_process=self.post_process
        )

        if self.post_process:
            # MLP head
            if not self.finetune:
                self.mlp_head = VitMlpHead(self.hidden_size, self.num_classes)
            else:
                self.class_head = get_linear_layer(
                    self.hidden_size, num_classes, torch.nn.init.zeros_
                )

    def set_input_tensor(self, input_tensor):
        """See megatron.model.transformer.set_input_tensor()"""
        self.transformer.set_input_tensor(input_tensor)

    def forward(self, input):

        if self.pre_process:
            rearranged_input = einops.rearrange(
                input,
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=self.patch_dim,
                p2=self.patch_dim,
            )

            assert rearranged_input.dtype == torch.half
            encoder_output = self.linear_encoder(rearranged_input)
            cls_tokens = self.cls_token.expand(encoder_output.shape[0], -1, -1)
            concatenated_tokens = torch.cat((cls_tokens, encoder_output), dim=1)

            token_embeddings = concatenated_tokens + \
                self.position_embeddings(self.position_ids)
            hidden_states = self.embedding_dropout(token_embeddings)
        else:
            hidden_states = input

        hidden_states = self.transformer(hidden_states, None)

        if self.post_process:
            if not self.finetune:
                hidden_states = self.mlp_head(hidden_states)
            else:
                hidden_states = self.class_head(hidden_states[:, 0, :])

        return hidden_states
