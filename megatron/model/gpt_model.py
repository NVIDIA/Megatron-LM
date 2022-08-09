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

"""GPT-2 model."""

from functools import partial
import torch

from megatron import get_args
from megatron import mpu
from .module import MegatronModule, fp32_to_float16

from .enums import AttnMaskType
from .language_model import parallel_lm_logits
from .language_model import get_language_model
from .utils import init_method_normal
from .utils import scaled_init_method_normal

from deepspeed.pipe import PipelineModule, LayerSpec, TiedLayerSpec
from megatron.model import LayerNorm
from megatron.model.module import float16_to_fp32
from .language_model import EmbeddingPipe
from .transformer import ParallelTransformerLayerPipe


def post_language_model_processing(lm_output, labels, logit_weights,
                                   get_key_value, parallel_output,
                                   forward_method_parallel_output,
                                   fp16_lm_cross_entropy):
    if get_key_value:
        lm_output, presents = lm_output

    # Output.
    if forward_method_parallel_output is not None:
        parallel_output = forward_method_parallel_output
    output = parallel_lm_logits(
        lm_output,
        logit_weights,
        parallel_output)

    if get_key_value:
        output = [output, presents]

    if labels is None:
        return output
    else:
        if fp16_lm_cross_entropy:
            assert output.dtype == torch.half
            loss = mpu.vocab_parallel_cross_entropy(output, labels)
        else:
            loss = mpu.vocab_parallel_cross_entropy(output.float(), labels)
        return loss


class GPTModel(MegatronModule):
    """GPT-2 Language model."""

    def __init__(self,
                 num_tokentypes=0,
                 parallel_output=True,
                 pre_process=True,
                 post_process=True,
                 return_moe_loss=True):
        super(GPTModel, self).__init__()
        args = get_args()

        self.parallel_output = parallel_output
        self.pre_process = pre_process
        self.post_process = post_process
        self.fp16_lm_cross_entropy = args.fp16_lm_cross_entropy
        self.return_moe_loss = return_moe_loss
        self.language_model, self._language_model_key = get_language_model(
            num_tokentypes=num_tokentypes,
            add_pooler=False,
            encoder_attn_mask_type=AttnMaskType.causal,
            init_method=init_method_normal(args.init_method_std),
            scaled_init_method=scaled_init_method_normal(args.init_method_std, args.num_layers),
            num_experts=args.num_experts,
            pre_process=self.pre_process,
            post_process=self.post_process)

        self.initialize_word_embeddings(init_method_normal)

    def set_input_tensor(self, input_tensor):
        """See megatron.model.transformer.set_input_tensor()"""
        self.language_model.set_input_tensor(input_tensor)

    def forward(self, input_ids, position_ids, attention_mask, labels=None,
                tokentype_ids=None, layer_past=None, get_key_value=False,
                forward_method_parallel_output=None, curriculum_seqlen=None):
        args = get_args()
        if curriculum_seqlen is not None:
            args.curriculum_seqlen = curriculum_seqlen
            if curriculum_seqlen < input_ids.size()[1]:
                # seqlen-based curriculum learning
                # input_ids, position_ids, labels have size [batch size, seqlen]
                input_ids = input_ids[:, :curriculum_seqlen].contiguous()
                position_ids = position_ids[:, :curriculum_seqlen].contiguous()
                if labels is not None:
                    labels = labels[:, :curriculum_seqlen].contiguous()

                # attention_mask has size [1, 1, seqlen, seqlen]
                attention_mask = attention_mask[:, :, :curriculum_seqlen, :curriculum_seqlen].contiguous()
        else:
            if args.curriculum_learning:
                # If got a None input, need to reset curriculum_seqlen on user side
                args.curriculum_seqlen = args.seq_length

        lm_output, *moe_losses = self.language_model(
            input_ids,
            position_ids,
            attention_mask,
            layer_past=layer_past,
            get_key_value=get_key_value)

        if self.post_process:
            lm_output = post_language_model_processing(
                    lm_output, labels,
                    self.word_embeddings_weight(),
                    get_key_value,
                    self.parallel_output,
                    forward_method_parallel_output,
                    self.fp16_lm_cross_entropy)
        
        if self.return_moe_loss:
            return (lm_output, *moe_losses)
        else:
            return lm_output

    def state_dict_for_save_checkpoint(self, destination=None, prefix='',
                                       keep_vars=False):

        state_dict_ = {}
        language_model_state_dict = self.language_model.state_dict_for_save_checkpoint(
                destination, prefix, keep_vars)
        # MoE states need to be handled separately by DeepSpeed engine, thus
        # moving them to the top level dictionary
        if "moe_state_dict" in language_model_state_dict:
            for key in list(language_model_state_dict["moe_state_dict"].keys()):
                state_dict_[key] = language_model_state_dict["moe_state_dict"].pop(key)
            del language_model_state_dict["moe_state_dict"]
        state_dict_[self._language_model_key] = language_model_state_dict
        # Save word_embeddings.
        if self.post_process and not self.pre_process:
            state_dict_[self._word_embeddings_for_head_key] \
                = self.word_embeddings.state_dict(destination, prefix, keep_vars)
        return state_dict_

    def load_state_dict(self, state_dict, strict=True):
        """Customized load."""

        # Load word_embeddings.
        if self.post_process and not self.pre_process:
            self.word_embeddings.load_state_dict(
                state_dict[self._word_embeddings_for_head_key], strict=strict)
        # Gather MoE states and move under language model
        moe_state_dict = {}
        for key in list(state_dict.keys()):
            if 'expert' in key and 'moe.gate.wg.weight' not in key:
                moe_state_dict[key] = state_dict.pop(key)
        if self._language_model_key in state_dict:
            state_dict = state_dict[self._language_model_key]
        if len(moe_state_dict) > 0:
            state_dict["moe_state_dict"] = moe_state_dict
        self.language_model.load_state_dict(state_dict, strict=strict)


def CrossEntropy(output, labels):
    labels, loss_mask = labels[0], labels[1]

    args = get_args()

    losses = mpu.vocab_parallel_cross_entropy(output.contiguous().float(), labels)
    loss_mask = loss_mask.view(-1)
    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()
    return loss


class GPTModelPipe(PipelineModule,MegatronModule):
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
                                        EmbeddingPipe,
                                        args.hidden_size,
                                        args.padded_vocab_size,
                                        args.max_position_embeddings,
                                        args.hidden_dropout,
                                        init_method=init_method,
                                        num_tokentypes=num_tokentypes,
                                        tied_weight_attr='word_embeddings_weight'))
        
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
                    self_attn_mask_type=AttnMaskType.causal))
                
        
        # Undo data format change
        self.specs.append(lambda x: x.transpose(0, 1).contiguous())

        # Final layernorm after transformer layers
        self.specs.append(
            LayerSpec(LayerNorm,
                      args.hidden_size,
                      eps=args.layernorm_epsilon))

        def _logits_helper(embedding, lm_output):
            """A wrapper to massage inputs/outputs from pipeline. """
            return parallel_lm_logits(
                lm_output,
                embedding.word_embeddings_weight,
                self.parallel_output)

        self.specs.append(
            TiedLayerSpec('embed',
                          EmbeddingPipe,
                          args.hidden_size,
                          args.padded_vocab_size,
                          args.max_position_embeddings,
                          args.hidden_dropout,
                          init_method=init_method,
                          num_tokentypes=num_tokentypes,
                          forward_fn=_logits_helper,
                          tied_weight_attr='word_embeddings_weight')
        )

        # Convert to fp32 if needed
        if args.fp16 or args.bf16:
            self.specs.append(float16_to_fp32)

        if args.checkpoint_activations:
            interval = args.checkpoint_num_layers
        else:
            interval = 0
        
        from deepspeed.runtime.pipe.topology import PipeModelDataParallelTopology
        topo = PipeModelDataParallelTopology(num_pp=mpu.get_pipeline_model_parallel_world_size(),
                                             num_mp=mpu.get_tensor_model_parallel_world_size(),
                                             num_dp=mpu.get_data_parallel_world_size())

        super().__init__(layers=self.specs,
                         loss_fn=CrossEntropy,
                         topology=topo,
                         activation_checkpoint_interval=interval,
                         partition_method='type:transformer')
