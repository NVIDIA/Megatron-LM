# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

"""Pretrain Retro with Megatron Core"""

from functools import partial

from megatron import get_args
from megatron.arguments import core_transformer_config_from_args
from megatron.core.enums import ModelType
from megatron.core.models.retro import get_retro_decoder_block_spec
from megatron.training import pretrain

from pretrain_gpt_core import model_provider as gpt_model_provider
from pretrain_retro import (
    forward_step,
    train_valid_test_datasets_provider,
)


# >>>
# import torch
# from lutil import pax, tp

# def hasnan(t):
#     if isinstance(t, torch.Tensor):
#         return torch.sum(torch.isnan(t)).item() > 0 if isinstance(t, torch.Tensor) else False
#     elif isinstance(t, (list, tuple, set)):
#         return any(hasnan(a) for a in t)
#     else:
#         return False

# def forward_hook(module, inputs, outputs):
#     return
#     # if any(hasnan(t) for t in [*inputs, *outputs] if isinstance(t, torch.Tensor)):
#     if hasnan([ inputs, outputs ]):
#         pax({"module": type(module).__name__}, "inputs", "outputs")

# def backward_hook(module, input_grads, output_grads):
#     return
#     if hasnan([ input_grads, output_grads ]):
#         pax({"module": type(module).__name__}, "input_grads", "output_grads")

# # decoder = model[0].module.module
# # encoder = decoder.decoder.layers[5].cross_attention.encoder

# def print_grads(top_key, top_model, depth):
#     print("%s~~~~ %s ~~~~" % ("  " * depth, top_key))
#     for sub_key, sub_param in top_model.named_parameters(recurse=False):
#         prefix = "%s%s" % ("  " * (depth + 1), sub_key)
#         print("%s / p : %s" % (prefix, tp(sub_param)))
#         print("%s / g : %s" % (prefix, tp(sub_param.main_grad)))
#     # for sub_key, sub_model in top_model.named_modules():
#     for sub_key, sub_model in top_model.named_children():
#         assert top_model != sub_model, f"{top_key} == {sub_key}."
#         print_grads(sub_key, sub_model, depth + 1)

# # print_grads("decoder", decoder, 0)
# # print_grads("encoder", encoder, 0)
# <<<


def model_provider(pre_process=True, post_process=True):
    args = get_args()
    config = core_transformer_config_from_args(args)
    model = gpt_model_provider(pre_process, post_process,
                               block_spec=get_retro_decoder_block_spec(config))

    # >>>
    # pax("model")
    # self.encoder.register_backward_hook(encoder_backward_hook)
    # self.encoder.layers[-1].ln_mlp.register_backward_hook(encoder_backward_hook)
    # module = model.decoder.layers[5].cross_attention
    # module = model.decoder.layers[5].cross_attn_bda
    # module = model.decoder.layers[11]
    # module = model.decoder.final_layernorm

    # for k, m in model.named_modules():
    #     if "bda" in k:
    #         # raise Exception("hi.")
    #         continue
    #     m.register_forward_hook(backward_hook)
    #     m.register_backward_hook(backward_hook)

    # encoder = cross_attn.encoder
    # encoder.layers[-1].ln_mlp.register_backward_hook(backward_hook)
    # <<<

    return model


def get_forward_kwargs(input_ids, position_ids, attn_mask):
    return {
        "context_input_ids" : input_ids,
        "context_position_ids" : position_ids,
        "context_mask" : attn_mask,
    }


if __name__ == "__main__":

    pretrain(train_valid_test_datasets_provider, model_provider,
             ModelType.encoder_or_decoder,
             partial(forward_step, get_forward_kwargs=get_forward_kwargs),
             args_defaults={'tokenizer_type': 'GPT2BPETokenizer'}
    )
