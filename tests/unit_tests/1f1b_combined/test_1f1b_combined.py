# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
"""Pretrain GPT."""

import logging
import os
import sys

import torch
from torch import Tensor
from megatron.training import get_args
from megatron.training.initialize import initialize_megatron
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer import transformer_layer
from megatron.core.pipeline_parallel.combined_1f1b import ScheduleNode, StreamRelease, StreamAcquire
from megatron.training.arguments import core_transformer_config_from_args
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.transformer.moe.token_dispatcher import MoEAlltoAllPerBatchState
class TransformerScheduleNode(ScheduleNode):

    def __init__(self, common_state, layer, stream, event):
        super().__init__(self.forward_impl, stream, event)
        self.common_state = common_state
        self.layer = layer



class AttnScheduleNode(TransformerScheduleNode):

    def forward_impl(self, hidden_states):
        attention_mask=None
        context=None
        context_mask=None
        rotary_pos_emb=None
        rotary_pos_cos=None
        rotary_pos_sin=None
        attention_bias=None
        inference_params=None
        packed_seq_params=None

        # Residual connection.
        residual = hidden_states

        # Optional Input Layer norm
        input_layernorm_output = self.layer.input_layernorm(hidden_states)

        # Self attention.
        attention_output_with_bias = self.layer.self_attention(
            input_layernorm_output,
            attention_mask=attention_mask,
            inference_params=inference_params,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            attention_bias=attention_bias,
            packed_seq_params=packed_seq_params,
        )

        # TODO: could we move `bias_dropout_add_exec_handler` itself
        # inside the module provided in the `bias_dropout_add_spec` module?
        with self.layer.bias_dropout_add_exec_handler():
            hidden_states = self.layer.self_attn_bda(self.layer.training, self.layer.config.bias_dropout_fusion)(
                attention_output_with_bias, residual, self.layer.hidden_dropout
            )

        # Residual connection.
        residual = hidden_states

        # Optional Layer norm after self-attention
        pre_cross_attn_layernorm_output = self.layer.pre_cross_attn_layernorm(hidden_states)

        # Cross attention.
        attention_output_with_bias = self.layer.cross_attention(
            pre_cross_attn_layernorm_output,
            attention_mask=context_mask,
            key_value_states=context,
            inference_params=inference_params,
        )

        if isinstance(attention_output_with_bias, dict) and "context" in attention_output_with_bias:
            context = attention_output_with_bias["context"]

        # TODO: could we move `bias_dropout_add_exec_handler` itself
        # inside the module provided in the `bias_dropout_add_spec` module?
        with self.layer.bias_dropout_add_exec_handler():
            hidden_states = self.layer.cross_attn_bda(self.layer.training, self.layer.config.bias_dropout_fusion)(
                attention_output_with_bias, residual, self.layer.hidden_dropout
            )

        # Residual connection.
        residual = hidden_states

        # Optional Layer norm post the cross-attention.
        pre_mlp_layernorm_output = self.layer.pre_mlp_layernorm(hidden_states)

        # residual, pre_mlp_layernorm_output
        # MLP.
        probs, routing_map = self.layer.mlp.router(pre_mlp_layernorm_output)
        self.common_state.probs = probs
        token_dispatcher = self.layer.mlp.token_dispatcher
        with token_dispatcher.per_batch_state_context(self.common_state):
            tokens_per_expert = token_dispatcher.meta_prepare(pre_mlp_layernorm_output, probs, routing_map)
            permutated_local_input_tokens = token_dispatcher.dispatch_preprocess(pre_mlp_layernorm_output, routing_map)
        self.common_state.tokens_per_expert = tokens_per_expert
        return residual, pre_mlp_layernorm_output, permutated_local_input_tokens, probs


class DispatchScheduleNode(TransformerScheduleNode):

    def forward_impl(self, residual, pre_mlp_layernorm_output, permutated_local_input_tokens, probs):

        self.common_state.probs = probs
        token_dispatcher = self.layer.mlp.token_dispatcher
        with token_dispatcher.per_batch_state_context(self.common_state):
            dispatched_input = token_dispatcher.dispatch_all_to_all(permutated_local_input_tokens)
        return residual, pre_mlp_layernorm_output, dispatched_input, probs


class MlPScheduleNode(TransformerScheduleNode):
    def forward_impl(self, residual, pre_mlp_layernorm_output, dispatched_input, probs):
        self.common_state.probs = probs
        token_dispatcher = self.layer.mlp.token_dispatcher
        with token_dispatcher.per_batch_state_context(self.common_state):
            dispatched_input = token_dispatcher.dispatch_postprocess(dispatched_input)
            expert_output, mlp_bias = self.layer.mlp.experts(dispatched_input, self.common_state.tokens_per_expert)
            assert mlp_bias is None
            permutated_local_input_tokens = token_dispatcher.combine_preprocess(expert_output)
        shared_output = self.layer.mlp.shared_experts(pre_mlp_layernorm_output)
        return residual, permutated_local_input_tokens, shared_output, probs


class CombinedSchedule(TransformerScheduleNode):
    def forward_impl(self, residual, permutated_local_input_tokens, shared_output, probs):
        token_dispatcher = self.layer.mlp.token_dispatcher
        self.common_state.probs = probs
        with token_dispatcher.per_batch_state_context(self.common_state):
            permutated_local_input_tokens = token_dispatcher.combine_all_to_all(permutated_local_input_tokens)
        return residual, permutated_local_input_tokens, shared_output, probs


class CombinePostProcessSchedule(TransformerScheduleNode):
    def forward_impl(self, residual, permutated_local_input_tokens, shared_output, probs):
        token_dispatcher = self.layer.mlp.token_dispatcher
        self.common_state.probs = probs
        with token_dispatcher.per_batch_state_context(self.common_state):
            output = token_dispatcher.combine_postprocess(permutated_local_input_tokens)
        output = output.type_as(residual)
        output += shared_output
        mlp_output_with_bias = (output, None)
        with self.layer.bias_dropout_add_exec_handler():
            hidden_states = self.layer.mlp_bda(self.layer.training, self.layer.config.bias_dropout_fusion)(
                mlp_output_with_bias, residual, self.layer.hidden_dropout
            )
        output = transformer_layer.make_viewless_tensor(
            inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True
        )
        return output


class CommonState(MoEAlltoAllPerBatchState):
    pass

def build_schedule_nodes(layer, event, comp_stream, com_stream):
    common_state = CommonState()
    attn = AttnScheduleNode(common_state, layer, comp_stream, event)
    dispatch = DispatchScheduleNode(common_state, layer, com_stream, event)
    mlp = MlPScheduleNode(common_state, layer,  comp_stream, event)
    combine = CombinedSchedule(common_state, layer,  com_stream, event)
    post_combine = CombinePostProcessSchedule(common_state, layer,  comp_stream, event)
    return [attn, dispatch, mlp, combine, post_combine]


def set_deterministic():
    torch.use_deterministic_algorithms(True)


def build_data(args):
    s = args.seq_length
    if args.sequence_parallel:
        s = s // args.tensor_model_parallel_size
    b = 1
    h = args.hidden_size

    hidden_states = torch.randn(*(s, b, h), dtype=torch.bfloat16, device="cuda") * h
    return hidden_states


def build_transformer_layer(args):
    config = core_transformer_config_from_args(args)
    model_spec = get_gpt_layer_with_transformer_engine_spec(
        args.num_experts,
        args.moe_grouped_gemm,
        multi_latent_attention=args.multi_latent_attention,
        moe_use_legacy_grouped_gemm=True,
    )
    transformer_layer = build_module(model_spec, config=config, layer_number=1)
    return transformer_layer



def test_1f1b_overlap(args):
    layer = build_transformer_layer(args)
    for param in layer.parameters():
        param.grad = None
        param.main_grad = torch.zeros_like(param, dtype=torch.float32)
    datas = [ build_data(args) for _ in range(16)]
    events = [torch.cuda.Event() for _ in range(16)]
    com_stream = torch.cuda.Stream(device="cuda")
    comp_stream = torch.cuda.Stream(device="cuda") # torch.cuda.current_stream()
    schedule_1f1b_overlap(datas, events, comp_stream, com_stream, layer)



def schedule_1f1b_overlap(datas, events, comp_stream, com_stream, layer):
    l = len(datas)
    assert l == len(events), f"{l} vs {len(events)}"
    # first f
    pre_stream = torch.cuda.current_stream()
    data_0 = StreamRelease.apply(events[0], pre_stream, datas[0])
    pre_graphs = build_schedule_nodes(layer, events[0], comp_stream, com_stream)
    output = (data_0,)
    for g in pre_graphs:
        output = g.forward(*output)
    pre_output = output

    # 1f1b
    for i in range(1, l):
        grad = torch.ones_like(pre_output)
        grad = StreamRelease.apply(events[i-1], pre_stream, grad)
        data = StreamRelease.apply(events[i], pre_stream, datas[i])
        graphs = build_schedule_nodes(layer, events[i], comp_stream, com_stream)

        output = (data,)
        grad = (grad,)
        torch.cuda.nvtx.range_push(f"1f1b")
        # post combine_backward
        print("post combine_backward")
        grad = pre_graphs[4].backward(*grad)
        # combine_backward
        print("combine_backward")
        grad = pre_graphs[3].backward(*grad)
        # attn
        print("attn")
        output = graphs[0].forward(*output)
        # dispatch
        print("dispatch")
        output = graphs[1].forward(*output)
        # mlp backward
        print("mlp backward")
        grad = pre_graphs[2].backward(*grad)
        # mlp forward
        print("mlp forward")
        output = graphs[2].forward(*output)
        # dispatch backward
        print("dispatch backward")
        grad = pre_graphs[1].backward(*grad)
        # combine forward
        print("combine forward")
        output = graphs[3].forward(*output)
        # attn backward
        print("attn backward")
        grad = pre_graphs[0].backward(*grad)
        # post combine
        print("post combine")
        output = graphs[4].forward(*output)



        pre_output = output
        pre_graphs = graphs
        torch.cuda.nvtx.range_pop()

    # last b
    grad = torch.ones_like(pre_output)
    grad = StreamRelease.apply(events[l-1], pre_stream, grad)
    grad = (grad,)
    grad = pre_graphs[4].backward(*grad)
    grad = pre_graphs[3].backward(*grad)
    grad = pre_graphs[2].backward(*grad)
    grad = pre_graphs[1].backward(*grad)
    grad = pre_graphs[0].backward(*grad)

    for (i, e) in enumerate(events):
        e.wait(torch.cuda.current_stream())

    torch.cuda.synchronize()
    print("finish")

def main():
    initialize_megatron()
    args = get_args()
    torch.cuda.cudart().cudaProfilerStart()
    test_1f1b_overlap(args)
    torch.cuda.cudart().cudaProfilerStop()



if __name__ == "__main__":
    main()
