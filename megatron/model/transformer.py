# coding=utf-8
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
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

"""Transformer."""

import math

import torch
from apex.normalization.fused_layer_norm import FusedLayerNorm as LayerNorm

from megatron import mpu
from megatron.module import MegatronModule


""" We use the following notation throughout this file:
     h: hidden size
     n: number of attention heads
     p: number of model parallel partitions
     np: n/p
     hp: h/p
     hn: h/n
     b: batch size
     s: sequence length
     l: number of layers
    Transformer takes input of size [b, s, h] and returns a
    tensor of the same size. We use the following arguments:
        hyperparameters: transformer hyperparameters
        attention_mask_func: a function that takes `unmaksed-attention-scores`
            with size [b, np, s, s] and an `attention-mask` and will apply
            the masking. The function should return a masked score of the
            same size [b, np, s, s].
               masked-attention-scores = attention_mask_func(
                                     unmaksed-attention-scores, attention-mask)
"""


class TransformerHyperparameters:
    """Hyperparameters used to build and run the transformer.

    Arguments:
        hidden_size: hidden size (h)
        num_layers: number of layers (l)
        num_attention_heads: number of attention heads (n)
        attention_dropout_prob: dropout probability for the attention
                                probabiliies
        output_dropout_prob: dropout probability for the output
                             layers (attention output and mlp output)
        mlp_activation_func: activation function for the mlp layer
        layernorm_epsilon: tolerance parameters used for layer norm
                           dividions
        init_method: init method used for all weights except layer
                     norm and output weights
        output_layer_init_method: init method for output weights (
                                  attention output and mlp output)
        checkpoint_activations: flag to use activation checkpointing
        checkpoint_num_layers: number of layers use in each chunk of
                               activation checkpointing
        apply_residual_connection_post_layernorm: Take the post layer-norm
            values for resudual connecton. BERT: True, GPT-2: False
    """
    def __init__(self,
                 hidden_size=None,
                 num_layers=None,
                 num_attention_heads=None,
                 attention_dropout_prob=None,
                 output_dropout_prob=None,
                 mlp_activation_func=None,
                 layernorm_epsilon=None,
                 init_method=None,
                 output_layer_init_method=None,
                 checkpoint_activations=None,
                 checkpoint_num_layers=None,
                 apply_residual_connection_post_layernorm=None,
                 apply_query_key_layer_scaling=None,
                 attention_softmax_in_fp32=None):
        self.params_dict = {}
        self.params_dict['hidden_size'] = hidden_size
        self.params_dict['num_layers'] = num_layers
        self.params_dict['num_attention_heads'] = num_attention_heads
        self.params_dict['attention_dropout_prob'] = attention_dropout_prob
        self.params_dict['output_dropout_prob'] = output_dropout_prob
        self.params_dict['mlp_activation_func'] = mlp_activation_func
        self.params_dict['layernorm_epsilon'] = layernorm_epsilon
        self.params_dict['init_method'] = init_method
        self.params_dict['output_layer_init_method'] = output_layer_init_method
        self.params_dict['checkpoint_activations'] = checkpoint_activations
        self.params_dict['checkpoint_num_layers'] = checkpoint_num_layers
        self.params_dict['apply_residual_connection_post_layernorm'] \
            = apply_residual_connection_post_layernorm
        self.params_dict['apply_query_key_layer_scaling'] \
            = apply_query_key_layer_scaling
        self.params_dict['attention_softmax_in_fp32'] \
            = attention_softmax_in_fp32


    def __getitem__(self, key):
        """Custom retrieval with error checks."""
        try:
            value = self.params_dict[key]
        except KeyError:
            raise Exception(
                'could not find {} in transformer hyperparameters'.format(key))
        except Exception as e:
            print('unexpected error in transformer hyperparameters:', e)
            raise Exception()
        else:
            assert value is not None, \
                'parameter value for {} is not set in transformer '\
                'hyperparameters'.format(key)
            return value
        raise Exception('should not be here')



class ParallelMLP(MegatronModule):
    """MLP.

    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension. At the end, dropout is also
    applied.
    """

    def __init__(self, hyperparameters):
        super(ParallelMLP, self).__init__()

        # Project to 4h.
        self.dense_h_to_4h = mpu.ColumnParallelLinear(
            hyperparameters['hidden_size'],
            4*hyperparameters['hidden_size'],
            gather_output=False,
            init_method=hyperparameters['init_method'])

        self.activation_func = hyperparameters['mlp_activation_func']

        # Project back to h.
        self.dense_4h_to_h = mpu.RowParallelLinear(
            4*hyperparameters['hidden_size'],
            hyperparameters['hidden_size'],
            input_is_parallel=True,
            init_method=hyperparameters['output_layer_init_method'])

        self.dropout = torch.nn.Dropout(hyperparameters['output_dropout_prob'])


    def forward(self, hidden_states):

        # [b, s, 4hp]
        intermediate_parallel = self.dense_h_to_4h(hidden_states)
        intermediate_parallel = self.activation_func(intermediate_parallel)

        # [b, s, h]
        output = self.dense_4h_to_h(intermediate_parallel)
        output = self.dropout(output)
        return output



class ParallelSelfAttention(MegatronModule):
    """Parallel self-attention layer abstract class.

    Self-attention layer takes input with size [b, s, h]
    and returns output of the same size.
    """

    def __init__(self, hyperparameters, attention_mask_func, layer_number):
        super(ParallelSelfAttention, self).__init__()

        self.attention_mask_func = attention_mask_func
        self.apply_query_key_layer_scaling \
            = hyperparameters['apply_query_key_layer_scaling']
        self.attention_softmax_in_fp32 \
            = hyperparameters['attention_softmax_in_fp32']
        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True
        self.layer_number = max(1, layer_number)

        # Per attention head and per partition values.
        world_size = mpu.get_model_parallel_world_size()
        self.hidden_size_per_partition = mpu.divide(
            hyperparameters['hidden_size'], world_size)
        self.hidden_size_per_attention_head = mpu.divide(
            hyperparameters['hidden_size'],
            hyperparameters['num_attention_heads'])
        self.num_attention_heads_per_partition = mpu.divide(
            hyperparameters['num_attention_heads'], world_size)

        # Strided linear layer.
        self.query_key_value = mpu.ColumnParallelLinear(
            hyperparameters['hidden_size'],
            3*hyperparameters['hidden_size'],
            stride=3,
            gather_output=False,
            init_method=hyperparameters['init_method'])

        # Dropout. Note that for a single iteration, this layer will generate
        # different outputs on different number of parallel partitions but
        # on average it should not be partition dependent.
        self.attention_dropout = torch.nn.Dropout(
            hyperparameters['attention_dropout_prob'])

        # Output.
        self.dense = mpu.RowParallelLinear(
            hyperparameters['hidden_size'],
            hyperparameters['hidden_size'],
            input_is_parallel=True,
            init_method=hyperparameters['output_layer_init_method'])
        self.output_dropout = torch.nn.Dropout(
            hyperparameters['output_dropout_prob'])


    def _transpose_for_scores(self, tensor):
        """Transpose a 3D tensor [b, s, np*hn] into a 4D tensor with
        size [b, np, s, hn].
        """
        new_tensor_shape = tensor.size()[:-1] + \
                           (self.num_attention_heads_per_partition,
                            self.hidden_size_per_attention_head)
        tensor = tensor.view(*new_tensor_shape)
        return tensor.permute(0, 2, 1, 3)


    def _get_query_key_value(self, hidden_states):
        """Get query, key, and value and transpose to
        get size [b, np, s, hn].
        """
        # Attention heads. [b, s, hp]
        mixed_x_layer = self.query_key_value(hidden_states)
        (mixed_query_layer,
         mixed_key_layer,
         mixed_value_layer) = mpu.split_tensor_along_last_dim(mixed_x_layer, 3)

        # Reshape and transpose [b, np, s, hn]
        query_layer = self._transpose_for_scores(mixed_query_layer)
        key_layer = self._transpose_for_scores(mixed_key_layer)
        value_layer = self._transpose_for_scores(mixed_value_layer)

        return query_layer, key_layer, value_layer


    def _get_unmasked_attention_scores(self, query_layer, key_layer):
        """Unmasked attention scores with size [b, np, s, s]."""
        coeff = 1
        if self.apply_query_key_layer_scaling:
            coeff = self.layer_number
        norm_factor = math.sqrt(coeff *
                                math.sqrt(self.hidden_size_per_attention_head))
        # Raw attention scores. [b, np, s, s]
        return torch.matmul(query_layer/norm_factor,
                            key_layer.transpose(-1, -2)/norm_factor)


    def _get_attention_probs(self, attention_scores):
        """Attention probabilies with dropout. The output has
        the size [b, np, s, s].
        """
        # Attention probabilities. [b, np, s, s]
        if self.apply_query_key_layer_scaling:
            attention_scores = attention_scores * self.layer_number
        attention_probs = torch.nn.Softmax(dim=-1)(attention_probs)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        with mpu.get_cuda_rng_tracker().fork():
            attention_probs = self.attention_dropout(attention_probs)

        return attention_probs


    def _get_attended_context(self, attention_probs, value_layer):
        """Final attended tesnor and transposed back to [b, s, hp]."""
        # Context layer.
        # [b, np, s, hn]
        context_layer = torch.matmul(attention_probs, value_layer)
        # [b, s, np, hn]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + \
                                  (self.hidden_size_per_partition,)
        # [b, s, hp]
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer


    def _get_output(self, context_layer):
        """Output layer with dropout."""
        # Output. [b, s, h]
        output = self.dense(context_layer)
        output = self.output_dropout(output)

        return output


    def forward(self, hidden_states, attention_mask, layer_past=None,
                get_key_value=False):
        # hidden_states: [b, s, h]

        # Attention heads. [b, np, s, hn]
        query_layer, key_layer, value_layer = self._get_query_key_value(
            hidden_states)

        if layer_past is not None:
            past_key, past_value = layer_past
            key_layer = torch.cat((past_key.type_as(key_layer),
                                   key_layer), dim=-2)
            value_layer = torch.cat((past_value.type_as(value_layer),
                                     value_layer), dim=-2)
        if get_key_value:
            present = (key_layer, value_layer)

        # Raw attention scores. [b, np, s, s]
        attention_scores = self._get_unmasked_attention_scores(
            query_layer, key_layer)

        # fp32 conversion.
        if self.attention_softmax_in_fp32:
            attention_scores = attention_scores.float()

        # Apply attention mask. [b, np, s, s]
        if get_key_value:
            with torch.no_grad():
                if layer_past is not None:
                    attention_mask = attention_mask[
                        ...,
                        attention_scores.size(3)-1,
                        :attention_scores.size(3)].unsqueeze(2)
                else:
                    attention_mask = attention_mask[
                        ...,
                        :attention_scores.size(3),
                        :attention_scores.size(3)]
        attention_scores = self.attention_mask_func(attention_scores,
                                                    attention_mask)

        # Attention probabilities. [b, np, s, s]
        attention_probs = self._get_attention_probs(attention_scores)

        # fp16 conversion
        if self.attention_softmax_in_fp32:
            attention_probs = attention_probs.half()

        # Context layer. [b, s, hp]
        context_layer = self._get_attended_context(attention_probs, value_layer)

        # Output. [b, s, h]
        output = self._get_output(context_layer)

        if get_key_value:
            output = [output, present]

        return output



class ParallelTransformerLayer(MegatronModule):
    """A single transformer layer.

    Transformore layer takes input with size [b, s, h] and returns an
    output of the same size.
    """
    def __init__(self, hyperparameters, attention_mask_func, layer_number):

        super(ParallelTransformerLayer, self).__init__()

        self.apply_residual_connection_post_layernorm \
            = hyperparameters['apply_residual_connection_post_layernorm']

        # Layernorm on the input data.
        self.input_layernorm = LayerNorm(
            hyperparameters['hidden_size'],
            eps=hyperparameters['layernorm_epsilon'])

        # Self attention.
        self.attention = ParallelSelfAttention(
            hyperparameters, attention_mask_func, layer_number)

        # Layernorm on the input data.
        self.post_attention_layernorm = LayerNorm(
            hyperparameters['hidden_size'],
            eps=hyperparameters['layernorm_epsilon'])

        # MLP
        self.mlp = ParallelMLP(hyperparameters)


    def forward(self, hidden_states, attention_mask, layer_past=None,
                get_key_value=False):
        # hidden_states: [b, s, h]

        # Layer norm at the begining of the transformer layer.
        layernorm_output = self.input_layernorm(hidden_states)
        # Self attention.
        attention_output = self.attention(layernorm_output,
                                          attention_mask,
                                          layer_past=layer_past,
                                          get_key_value=get_key_value)
        if get_key_value:
            attention_output, presents = attention_output

        # Residual connection.
        if self.apply_residual_connection_post_layernorm:
            layernorm_input = layernorm_output + attention_output
        else:
            layernorm_input = hidden_states + attention_output
        # Layer norm post the self attention.
        layernorm_output = self.post_attention_layernorm(layernorm_input)

        # MLP.
        mlp_output = self.mlp(layernorm_output)
        # Second residual connection.
        if self.apply_residual_connection_post_layernorm:
            output = layernorm_output + mlp_output
        else:
            output = layernorm_input + mlp_output

        if get_key_value:
            output = [output, presents]

        return output


class ParallelTransformer(MegatronModule):
    """Transformer class."""

    def __init__(self, hyperparameters, attention_mask_func):
        super(ParallelTransformer, self).__init__()

        # Store activation checkpoiting flag.
        self.checkpoint_activations = hyperparameters['checkpoint_activations']
        self.checkpoint_num_layers = hyperparameters['checkpoint_num_layers']

        def get_layer(layer_number):
            return ParallelTransformerLayer(
                hyperparameters, attention_mask_func, layer_number)

        # Transformer layers.
        self.layers = torch.nn.ModuleList(
            [get_layer(i+1) for i in range(hyperparameters['num_layers'])])

        # Final layer norm before output.
        self.final_layernorm = LayerNorm(
            hyperparameters['hidden_size'],
            eps=hyperparameters['layernorm_epsilon'])


    def _checkpointed_forward(self, hidden_states, attention_mask):
        """Forward method with activation checkpointing."""
        def custom(start, end):
            def custom_forward(*inputs):
                layers_ = self.layers[start:end]
                x_ = inputs[0]
                for layer in layers_:
                    x_ = layer(x_, inputs[1])
                return x_
            return custom_forward

        l = 0
        num_layers = len(self.layers)
        while l < num_layers:
            hidden_states = mpu.checkpoint(
                custom(l, l+self.checkpoint_num_layers),
                hidden_states, attention_mask)
            l += self.checkpoint_num_layers

        return hidden_states


    def forward(self, hidden_states, attention_mask, layer_past=None,
                get_key_value=False):

        # Checks
        if layer_past is not None:
            assert get_key_value, \
                'for not None values in layer_past, ' \
                'expected get_key_value to be set'
        if get_key_value:
            assert not self.checkpoint_activations, \
                'get_key_value does not work with ' \
                'activation checkpointing'

        if self.checkpoint_activations:
            hidden_states = self._checkpointed_forward(hidden_states,
                                                       attention_mask)
        else:
            if get_key_value:
                presents = []
            for i, layer in enumerate(self.layers):
                past = None
                if layer_past is not None:
                    past = layer_past[i]
                hidden_states = layer(hidden_states,
                                      attention_mask,
                                      layer_past=past,
                                      get_key_value=get_key_value)
                if get_key_value:
                    hidden_states, present = hidden_states
                    presents.append(present)

        # Final layer norm.
        output = self.final_layernorm(hidden_states)
        if get_key_value:
            output = [output, presents]

        return output
