# lawrence mcafee

# ~~~~~~~~ import ~~~~~~~~
from megatron import get_args
from megatron.core.enums import ModelType
from megatron.training import get_model
from pretrain_retro import core_model_provider, default_model_provider

from lutil import pax, tp

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    
# def print_model_with_params(key, model, depth=0):
def print_model(key, model, depth=0):
    if depth == 0:
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("%s%s%s" % (
        "  " * depth,
        "" if key is None else f"({key}) ",
        type(model).__name__,
    ))
    for k, p in model.named_parameters(recurse=False):
        print("%s* %s : %s ... [%s]." % (
            "  " * (depth + 1),
            k,
            list(p.shape),
            # ",".join(map(str, p.view(-1)[None:None:p.numel()//4].tolist())),
            tp(p),
        ))
    for k, m in model.named_children():
        print_model(k, m, depth + 1)
    if depth == 0:
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("%s nparams : %d." % (key, sum(t.numel() for t in model.parameters())))
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

def compare_top_nparams(key, default_module, core_module):
    get_nparams = lambda m : "--" if m is None else sum(t.numel() for t in m.parameters())
    # >>>
    # get_param_shapes = lambda m : "--" if m is None else ", ".join(str(tuple(p.shape)) for p in m.parameters())
    get_param_shapes = lambda m : "--"
    # <<<
    # get_param_shapes = lambda m : "--" if m is None else "-some-"
    default_nparams = get_nparams(default_module)
    core_nparams = get_nparams(core_module)
    print("%10s : d %10s, c %10s ... %s ---- d %s, c %s." % (
        key,
        default_nparams,
        core_nparams,
        default_nparams - core_nparams if isinstance(default_nparams, int) and isinstance(core_nparams, int) else "--",
        get_param_shapes(default_module),
        get_param_shapes(core_module),
    ))

def compare_preprocess_nparams(default_model, core_model):
    default_embedding = default_model.language_model.embedding
    core_embedding = core_model.embedding
    compare_top_nparams("emb", default_embedding, core_embedding)

    # pax({
    #     "default_embedding" : type(default_embedding).__name__,
    #     "core_embedding" : type(core_embedding).__name__,
    # })

# def compare_sub_nparams(key, default_module, core_module):
def compare_xattn_nparams(key, default_xattn, core_xattn):

    # default_map = dict(default_module.named_children())
    # core_map = dict(core_module.named_children())

    compare_top_nparams(
        f"{key} xattn /    q",
        default_xattn.query,
        core_xattn.linear_q,
    )
    compare_top_nparams(
        f"{key} xattn /   kv",
        default_xattn.key_value,
        core_xattn.linear_kv,
    )
    compare_top_nparams(
        f"{key} xattn / core",
        default_xattn.core_attention,
        core_xattn.core_attention,
    )
    compare_top_nparams(
        f"{key} xattn /    o",
        default_xattn.dense,
        core_xattn.linear_proj,
    )

    # default_q = default_xattn.query
    # core_q = core_xattn.linear_q
    # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    # print(default_xattn)
    # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    # print(core_xattn)
    # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    # print(default_q)
    # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    # print(core_q)
    # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    # print(lift_params(default_xattn))
    # print(lift_params(core_xattn))

    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print_model(None, default_xattn)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print_model(None, core_xattn)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    # pax({
    #     "default
    # })
    # pax("default_map, core_map")

# def compare_retro_decoder_layer_0(default_layer, core_layer):
# def compare_retro_decoder_layer(layer_idx, default_layers, core_layers):
def compare_layer_nparams(key, layer_idx, default_layers, core_layers):

    default_layer = default_layers[layer_idx]
    core_layer = core_layers[layer_idx]

    compare_top_nparams(
        f"{key} {layer_idx} / pre sattn norm",
        default_layer.input_norm,
        core_layer.input_layernorm,
    )
    compare_top_nparams(
        f"{key} {layer_idx} /      self attn",
        default_layer.self_attention,
        core_layer.self_attention,
    )
    compare_top_nparams(
        f"{key} {layer_idx} / pre cattn norm",
        default_layer.post_attention_norm,
        core_layer.pre_cross_attn_layernorm,
    )
    compare_top_nparams(
        f"{key} {layer_idx} /     cross attn",
        default_layer.inter_attention,
        core_layer.cross_attention,
    )
    compare_top_nparams(
        f"{key} {layer_idx} /   pre mlp norm",
        default_layer.post_inter_attention_norm,
        core_layer.pre_mlp_layernorm,
    )
    compare_top_nparams(
        f"{key} {layer_idx} /            mlp",
        default_layer.mlp,
        core_layer.mlp,
    )
    compare_top_nparams(
        f"{key} {layer_idx} /      retriever",
        default_layer.retriever,
        None,
    )

    # pax({
    #     "default children" : list(dict(default_layer.named_children()).keys()),
    #     "core children" : list(dict(core_layer.named_children()).keys()),
    # })

    # compare_top_nparams(f"{key} {layer_idx}", default_layer, core_layer)

def compare_block_nparams(key, default_layers, core_layers):
    assert len(default_layers) == len(core_layers)
    for i in range(len(default_layers)):
        compare_top_nparams(
            f"{key} block / {i}",
            default_layers[i],
            core_layers[i],
        )

def get_default_and_core_models():

    # model, optimizer, opt_param_scheduler = setup_model_and_optimizer(
    #     model_provider, model_type)
    return [
        get_model(fn, ModelType.retro_decoder)[0].module.module
        for fn in (default_model_provider, core_model_provider)
    ]
    # unwrapped_model = unwrap_model(model)

def compare_models():

    args = get_args()

    default_model, core_model = get_default_and_core_models()

    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print(default_model)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print(core_model)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    default_layers = list(default_model.language_model.encoder.layers)
    core_layers = list(core_model.decoder.layers)

    default_encoder_layers = list(default_layers[5].retriever.layers)
    core_encoder_layers = list(core_layers[5].cross_attention.encoder.layers)
    default_encoder_xattn = default_encoder_layers[0].inter_attention
    core_encoder_xattn = core_encoder_layers[0].cross_attention.attn

    # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    # print_model("default norm", default_encoder_layers[0].post_attention_norm)
    # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    # print_model("core norm", core_encoder_layers[0].pre_cross_attn_layernorm)
    # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    # print_model("default xattn", default_encoder_xattn)
    # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    # print_model("core xattn", core_encoder_xattn)
    # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    # exit()

    # pax("default_encoder_layers, core_encoder_layers")

    compare_preprocess_nparams(default_model, core_model)
    compare_block_nparams("decoder", default_layers, core_layers)
    compare_layer_nparams("decoder layer", 5, default_layers, core_layers) # 5, 8
    compare_block_nparams("encoder", default_encoder_layers, core_encoder_layers)
    compare_layer_nparams("encoder layer", 0, default_encoder_layers, core_encoder_layers)
    # compare_sub_nparams("encoder xattn", default_encoder_xattn, core_encoder_xattn)
    compare_xattn_nparams("encoder", default_encoder_xattn, core_encoder_xattn)
    compare_top_nparams("model", default_model, core_model)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    exit()

    pax(
        # "default_model, core_model",
        {
            "n default" : len(list(default_model.parameters())),
            "n core" : len(list(core_model.parameters())),
            "d children" : dict(default_model.named_children()),
            "c children" : dict(core_model.named_children()),
        },
    )

# eof
