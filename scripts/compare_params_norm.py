# lawrence mcafee

# ~~~~~~~~ import ~~~~~~~~
from megatron.core.enums import ModelType
from megatron.training import get_model
from pretrain_gpt import model_provider as default_model_provider
from pretrain_gpt_core import model_provider as core_model_provider

from .compare_models import (
    compare_top_nparams,
    # get_default_and_core_models,
    print_model,
)

from lutil import pax

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def get_default_and_core_models():

    # >>>
    if 0:
        import os
        os.environ["NVTE_FLASH_ATTN"] = "0"
    # <<<

    # model, optimizer, opt_param_scheduler = setup_model_and_optimizer(
    #     model_provider, model_type)
    return [
        get_model(fn, ModelType.encoder_or_decoder)[0].module.module
        for fn in (default_model_provider, core_model_provider)
    ]
    # unwrapped_model = unwrap_model(model)

def copy_embedding(default_model, core_model):

    default_emb = default_model.language_model.embedding # .word_embeddings.weight
    core_emb = core_model.embedding # .word_embeddings.weight
    # core_emb.data.copy_(default_emb)
    core_emb.word_embeddings.weight.data.copy_(default_emb.word_embeddings.weight)
    core_emb.position_embeddings.weight.data.copy_(default_emb.position_embeddings.weight)
    # pax("default_emb, core_emb")

    # >>>
    # print_model("default emb", default_model.language_model.embedding)
    # print_model("core emb", core_model.embedding)
    # exit()
    # <<<

def copy_self_attn_block(default_layer, core_layer):

    # >>>
    # print_model("default layer", default_layer)
    # print_model("core layer", core_layer)
    # <<<

    default_norm = default_layer.input_norm
    core_norm = core_layer.input_layernorm
    default_attn = default_layer.self_attention
    core_attn = core_layer.self_attention
    # default_bda = default_layer.self_attn_bda
    # core_bda = core_layer.self_attn_bda

    # core_attn

    print_model("default_norm", default_norm)
    print_model("core_norm", core_norm)
    print_model("default_attn", default_attn)
    print_model("core_attn", core_attn)
    exit()

    pax(
        "default_norm",
        "core_norm",
        # "default_attn",
        "core_attn",
    )

def copy_layer(default_layer, core_layer):

    copy_self_attn_block(default_layer, core_layer)
    copy_cross_attn_block(default_layer, core_layer)
    copy_mlp_attn_block(default_layer, core_layer)

    pax({
        "default_layer" : type(default_layer).__name__,
        "core_layer" : type(core_layer).__name__,
    })

def copy_layers(default_model, core_model):
    default_layers = list(default_model.language_model.encoder.layers)
    core_layers = list(core_model.decoder.layers)
    assert len(default_layers) == len(core_layers)
    for i in range(len(default_layers)):
        copy_layer(default_layers[i], core_layers[i])
    pax("default_layers, core_layers")

# def copy_params_default_to_core(default_model, core_model):
# def copy_params(default_model, core_model):
def copy_model(default_model, core_model):

    copy_embedding(default_model, core_model)
    copy_layers(default_model, core_model)


def compare_params_norm():

    default_model, core_model = get_default_and_core_models()

    compare_top_nparams("model", default_model, core_model)

    copy_model(default_model, core_model)

    pax({
        "default_model" : type(default_model).__name__,
        "core_model" : type(core_model).__name__,
    })

# eof
