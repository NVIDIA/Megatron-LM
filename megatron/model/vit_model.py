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

import torch
import torch.nn.functional as F
from megatron import get_args
from megatron.model.transformer import ParallelTransformer
from deepspeed.pipe import PipelineModule, LayerSpec, TiedLayerSpec
from .enums import AttnMaskType
from megatron.model.utils import (
    get_linear_layer,
    init_method_normal,
    scaled_init_method_normal,
)

from megatron.mpu.utils import ClsUtility
from megatron.mpu.initialize import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
from megatron.mpu.mappings import reduce_from_tensor_model_parallel_region
from .vit_embedding import VitEmbedding, VitEmbeddingPipe
from .module import MegatronModule, fp32_to_float16
from megatron.model.module import float16_to_fp32
from .transformer import ParallelTransformerLayerPipe
from megatron.model import LayerNorm
from megatron import mpu
from megatron.mpu.layers import ColumnParallelLinear

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
        super(VitMlpHead, self).__init__(share_word_embeddings=False)
        self.tensor_model_parallel_size = get_tensor_model_parallel_world_size()

        self.dense_in = ColumnParallelLinear(hidden_size, hidden_size,gather_output=True)
        self.dense_out = ColumnParallelLinear(hidden_size, num_classes, gather_output=False)
        torch.nn.init.constant_(self.dense_out.bias, -10)

    def forward(self, hidden_states, sequence_index=0):
        x = hidden_states[:, sequence_index, :]
        x, _ = self.dense_in(x)
        x = torch.tanh(x)
        x, _ = self.dense_out(x)
        return x

    @property
    def dense_in_weight(self):
        """Easy accessory for the DeepSpeed pipeline engine to tie embeddings across stages."""
        return self.dense_in.weight


class VitModel(MegatronModule):
    """Vision Transformer Model."""

    def __init__(self, num_classes, finetune=False,
                 pre_process=True,
                 post_process=True):
        super(VitModel, self).__init__(share_word_embeddings=False)
        args = get_args()
        self.pre_process = pre_process
        self.post_process = post_process
        self.fp16_lm_cross_entropy = args.fp16_lm_cross_entropy



        if args.init_method_xavier_uniform:
            self.init_method = torch.nn.init.xavier_uniform_
            self.scaled_init_method = torch.nn.init.xavier_uniform_
        else:
            self.init_method = init_method_normal(args.init_method_std)
            self.scaled_init_method = scaled_init_method_normal(
                args.init_method_std, args.num_layers
            )

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

        self.tensor_model_parallel_size = get_tensor_model_parallel_world_size()
        self.cls_start_index, self.cls_end_index = \
            ClsUtility.cls_range_from_global_num_classes(
                self.num_classes, get_tensor_model_parallel_rank(),
                self.tensor_model_parallel_size)
        self.num_classes_per_partition = self.cls_end_index - \
                                         self.cls_start_index
        if self.pre_process:
            self.embedding = VitEmbedding()

        # Transformer
        self.transformer = ParallelTransformer(
            self.init_method, self.scaled_init_method, pre_process=pre_process, post_process=post_process
        )
        
        if self.post_process:
            # MLP head
            if not self.finetune:
                self.mlp_head = VitMlpHead(self.hidden_size, self.num_classes)
            else:
                self.class_head = get_linear_layer(
                    self.hidden_size, self.num_classes_per_partition, torch.nn.init.zeros_
                )

    def set_input_tensor(self, input_tensor):
        """ See megatron.model.transformer.set_input_tensor()"""
        self.transformer.set_input_tensor(input_tensor)

    def forward(self, x):
        if self.pre_process:
            embedding_output = self.embedding(x)
            encoder_input = embedding_output
        else:
            encoder_input = None
        x = self.transformer(encoder_input, None)[0]

        if self.post_process:
            x = self.mlp_head(x)
        return x

def CrossEntropy(output, labels):
    labels, loss_mask = labels[0], labels[1]

    args = get_args()

    losses = mpu.vocab_parallel_cross_entropy(output.contiguous().float(), labels)
    loss_mask = loss_mask.view(-1)
    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()
    return loss

def vitCrossEntropy(output, labels):
    losses = mpu.cls_parallel_cross_entropy(output.contiguous().float(), labels)
    return losses.mean()

class VitModelPipe(PipelineModule, MegatronModule):
    """GPT-2 Language model."""

    def __init__(self,
                 num_tokentypes=0,
                 parallel_output=True):
        args = get_args()
        self.parallel_output = parallel_output
        init_method = init_method_normal(args.init_method_std)

        self.specs = []

        def _to_float16(inputs):
            if args.fp16:
                return fp32_to_float16(inputs, lambda v: v.half())
            elif args.bf16:
                return fp32_to_float16(inputs, lambda v: v.bfloat16())
            else:
                return inputs

        self.specs.append(_to_float16)

        # Embedding layer
        self.specs.append(TiedLayerSpec('embed',
                                        VitEmbeddingPipe,
                                        tied_weight_attr='linear_encoder_weight'))

        if args.fp32_residual_connection:
            self.specs.append(lambda x: x.transpose(0, 1).contiguous().float())
        else:
            self.specs.append(lambda x: x.transpose(0, 1).contiguous())

        for layer_idx in range(args.num_layers):
            self.specs.append(
                LayerSpec(ParallelTransformerLayerPipe,
                          init_method=init_method,
                          output_layer_init_method=scaled_init_method_normal(args.init_method_std,
                                                                             args.num_layers),
                          layer_number=layer_idx,
                          isvit=True,
                          self_attn_mask_type=AttnMaskType.causal))

        # Undo data format change
        self.specs.append(lambda x: x.transpose(0, 1).contiguous())

        # Final layernorm after transformer layers
        self.specs.append(
            LayerSpec(LayerNorm,
                      args.hidden_size,
                      eps=args.layernorm_epsilon))

        self.specs.append(
            TiedLayerSpec('linear',
                   VitMlpHead,
                   args.hidden_size,
                   args.num_classes,
                   tied_weight_attr='dense_in_weight'
                   )
        )

        # Convert to fp32 if needed
        # if args.fp16 or args.bf16:
        #     self.specs.append(float16_to_fp32)

        if args.checkpoint_activations:
            interval = args.checkpoint_num_layers
        else:
            interval = 0

        from deepspeed.runtime.pipe.topology import PipeModelDataParallelTopology
        topo = PipeModelDataParallelTopology(num_pp=mpu.get_pipeline_model_parallel_world_size(),
                                             num_mp=mpu.get_tensor_model_parallel_world_size(),
                                             num_dp=mpu.get_data_parallel_world_size())

        super().__init__(layers=self.specs,
                         loss_fn=vitCrossEntropy,
                         topology=topo,
                         activation_checkpoint_interval=interval,
                         partition_method='type:transformer')
