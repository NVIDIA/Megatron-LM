# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import json
import os
import sys
from pathlib import Path

import torch

from utils import _ConverterFakeProcessGroup

try:
    import transformers
except ImportError:
    raise ImportError("The 'transformers' package is not installed.")
import types

from tqdm import tqdm


def add_arguments(parser):
    group = parser.add_argument_group(title="Llama Nemotron loader.")

    parser.add_argument(
        "--bf16", action="store_true", help="Whether to load weights in bf16."
    )
    parser.add_argument(
        "--fp16", action="store_true", help="Whether to load weights in fp16."
    )
    group.add_argument(
        "--true-vocab-size",
        type=int,
        default=None,
        help="original size of vocab, if specified will trim padding from embedding table.",
    )
    group.add_argument(
        "--vocab-file",
        type=str,
        default=None,
        help="Path to the vocab file. If specified will use this to get vocab size and "
        "trim padding from the embedding table.",
    )
    group.add_argument("--tokenizer-model", required=True, help="Tokenizer model file.")
    group.add_argument(
        "--megatron-path",
        type=str,
        default=None,
        help="Base directory of Megatron repository",
    )
    group.add_argument(
        "--make-vocab-size-divisible-by",
        type=int,
        default=None,
        help="Make vocab size divisible by",
    )


def verify_transformers_version():
    major, minor, patch = map(int, transformers.__version__.split("."))
    assert major >= 4 and minor >= 48, (
        f"transformers version must be >= 4.48.0, but got {transformers.__version__}"
    )


def load_args_from_checkpoint(args):
    # Read Llama args.
    model_args_path = Path(args.load) / "config.json"
    model_args = json.loads(model_args_path.read_text())

    # Update Megatron args.
    args.seq_length = 4096
    args.hidden_size = model_args["hidden_size"]
    args.num_attention_heads = model_args["num_attention_heads"]
    args.max_position_embeddings = model_args["max_position_embeddings"]
    args.num_layers = model_args["num_hidden_layers"]
    args.global_batch_size = 1024
    args.norm_epsilon = model_args["rms_norm_eps"]
    args.iteration = 1  # '0', 'release' don't work
    args.position_embedding_type = "rope"
    args.use_rope_scaling = True
    args.use_rotary_position_embeddings = True
    args.rotary_base = model_args["rope_theta"]
    args.rope_scaling_factor = model_args["rope_scaling"]["factor"]
    args.swiglu = True
    args.normalization = "RMSNorm"
    args.add_bias_linear = False
    args.untie_embeddings_and_output_weights = not model_args.get(
        "tie_word_embeddings", False
    )
    args.vocab_size = model_args["vocab_size"]
    args.padded_vocab_size = model_args["vocab_size"]
    args.ffn_hidden_size = model_args["intermediate_size"]
    args.group_query_attention = True
    # num query head must be consistent across all layers
    args.num_query_groups = (
        model_args["num_attention_heads"]
        // model_args["block_configs"][0]["attention"]["n_heads_in_group"]
    )

    args.heterogeneous_layers_config_path = model_args_path.as_posix()


def set_preprocess_state(args, model, hf_model):
    """Set embedding params."""
    model.embedding.word_embeddings.weight.data.copy_(
        hf_model.model.embed_tokens.weight
    )


def set_postprocess_state(args, model, hf_model):
    """Set output layer & norm params."""
    model.decoder.final_layernorm.weight.data.copy_(hf_model.model.norm.weight)
    model.output_layer.weight.data.copy_(hf_model.lm_head.weight)


def set_attn_state(args, layer, hf_layer):
    """Set self-attention params."""

    if hf_layer.attention_config.no_op:
        return

    if hf_layer.attention_config.replace_with_linear:
        layer.self_attention.layer_norm_weight.data.copy_(
            hf_layer.input_layernorm.weight
        )
        layer.self_attention.weight.data.copy_(hf_layer.self_attn.linear_attn.weight)
    else:
        # Get attention layer & state.
        attn = layer.self_attention
        hf_attn = hf_layer.self_attn

        # Layer norm weight
        attn.linear_qkv.layer_norm_weight.data.copy_(hf_layer.input_layernorm.weight)

        # Reshape loaded weights.
        tp = args.tensor_model_parallel_size
        num_heads = args.num_attention_heads // tp
        num_query_groups = (
            args.num_query_groups
            if args.group_query_attention
            else args.num_attention_heads
        ) // tp
        num_querys_per_group = num_heads // num_query_groups
        dim = args.kv_channels
        assert num_heads % num_querys_per_group == 0

        # Copy weights (re-order dimensions for Megatron).
        attn.linear_qkv.weight.data.copy_(
            torch.cat(
                [
                    hf_attn.q_proj.weight.reshape(
                        (num_query_groups, num_querys_per_group * dim, -1)
                    ),
                    hf_attn.k_proj.weight.reshape((num_query_groups, dim, -1)),
                    hf_attn.v_proj.weight.reshape((num_query_groups, dim, -1)),
                ],
                dim=1,
            ).reshape((-1, args.hidden_size))
        )
        attn.linear_proj.weight.data.copy_(hf_attn.o_proj.weight)


def set_mlp_state(args, layer, hf_layer):
    """Set MLP params."""

    if hf_layer.ffn_config.no_op:
        return

    if hf_layer.ffn_config.replace_with_linear:
        layer.mlp.layer_norm_weight.data.copy_(hf_layer.post_attention_layernorm.weight)
        layer.mlp.weight.data.copy_(hf_layer.mlp.linear_mlp.weight)
    else:
        # Layer norm weight
        layer.mlp.linear_fc1.layer_norm_weight.data.copy_(
            hf_layer.post_attention_layernorm.weight
        )

        layer.mlp.linear_fc1.weight.data.copy_(
            torch.cat(
                [
                    hf_layer.mlp.gate_proj.weight,
                    hf_layer.mlp.up_proj.weight,
                ],
                dim=0,
            )
        )
        layer.mlp.linear_fc2.weight.data.copy_(hf_layer.mlp.down_proj.weight)


def set_layer_state(args, model, hf_model, layer_idx):
    """Set transformer layer params."""

    layer = model.decoder.layers[layer_idx]
    hf_layer = hf_model.model.layers[layer_idx]

    set_attn_state(args, layer, hf_layer)
    set_mlp_state(args, layer, hf_layer)


def load_checkpoint_to_model(args):
    """Set model params."""

    from transformers import AutoModelForCausalLM

    from pretrain_gpt import model_provider

    # Load Huggingface model.
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.load,
        torch_dtype=args.params_dtype,
        low_cpu_mem_usage=True,
        device_map="cpu",
        trust_remote_code=True,
    )

    # Init Megatron model.
    model = model_provider(True, True).to(args.params_dtype)

    # Set model state.
    set_preprocess_state(args, model, hf_model)
    set_postprocess_state(args, model, hf_model)
    for layer_idx in tqdm(range(args.num_layers), "set layer states"):
        set_layer_state(args, model, hf_model, layer_idx)

    return model


def _load_checkpoint(queue, args):
    verify_transformers_version()

    # Search in directory above this.
    sys.path.append(
        os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)
        )
    )
    if args.megatron_path is not None:
        sys.path.insert(0, args.megatron_path)

    try:
        from megatron.core import mpu
        from megatron.core.enums import ModelType
        from megatron.training.arguments import parse_args, validate_args
        from megatron.training.global_vars import set_args, set_global_variables
    except ModuleNotFoundError:
        print(
            "Unable to import Megatron, please specify the path to Megatron using --megatron-path. Exiting."
        )
        queue.put("exit")
        exit(1)

    # We want all arguments to come from us.
    sys.argv = [
        "script.py",
        "--no-masked-softmax-fusion",
        "--no-bias-gelu-fusion",
        "--no-bias-dropout-fusion",
        "--no-async-tensor-model-parallel-allreduce",
        "--use-cpu-initialization",
        "--micro-batch-size",
        "1",
        "--no-load-optim",
        "--no-load-rng",
        "--no-save-optim",
        "--no-save-rng",
        "--mock-data",  # To pass the "blend data checks" in arguments.py
        "--no-initialization",
        "--load",
        args.load_dir,
        "--no-one-logger",
        "--transformer-impl",
        "transformer_engine",
    ]

    if args.make_vocab_size_divisible_by is not None:
        sys.argv.extend(
            ["--make-vocab-size-divisible-by", str(args.make_vocab_size_divisible_by)]
        )

    margs = parse_args()
    margs.tokenizer_model = args.tokenizer_model
    load_args_from_checkpoint(margs)

    margs.tokenizer_type = "HuggingFaceTokenizer"

    # Arguments do sanity checks on the world size, but we don't care,
    # so trick it into thinking we are plenty of processes.
    margs.world_size = (
        margs.tensor_model_parallel_size * margs.pipeline_model_parallel_size
    )

    margs = validate_args(margs)

    def check_for_arg(arg_name, default=None):
        if getattr(margs, arg_name, None) is None:
            if default is not None:
                setattr(margs, arg_name, default)
            else:
                print(f"Checkpoint does not specify the argument {arg_name}. Exiting.")
                print(f"Arguments: {margs}")
                queue.put("exit")
                exit(1)

    check_for_arg("tensor_model_parallel_size")
    check_for_arg("pipeline_model_parallel_size")
    check_for_arg("num_layers")
    check_for_arg("hidden_size")
    check_for_arg("seq_length")
    check_for_arg("num_attention_heads")
    check_for_arg("max_position_embeddings")
    check_for_arg("position_embedding_type")
    check_for_arg("iteration")
    check_for_arg("bert_binary_head")
    check_for_arg("disable_bias_linear", True)
    check_for_arg("params_dtype")
    check_for_arg("swiglu", True)
    check_for_arg("heterogeneous_layers_config_path")

    # Determine how to make our models.
    assert args.model_type == "GPT", (
        "Llama-2, Llama-3, Llama-Nemotron and Mistral are GPT models."
    )
    margs.model_type = ModelType.encoder_or_decoder
    margs.params_dtype = (
        torch.bfloat16 if args.bf16 else torch.float16 if args.fp16 else torch.float32
    )

    set_global_variables(margs, build_tokenizer=False)
    mpu.set_tensor_model_parallel_world_size(margs.tensor_model_parallel_size)
    mpu.set_pipeline_model_parallel_world_size(margs.pipeline_model_parallel_size)
    mpu.set_virtual_pipeline_model_parallel_world_size(
        margs.virtual_pipeline_model_parallel_size
    )

    # For backward compatibility during local parallel states refactoring
    fake_tp_group = _ConverterFakeProcessGroup(size=margs.tensor_model_parallel_size)
    fake_ep_group = _ConverterFakeProcessGroup(size=margs.expert_model_parallel_size)
    mpu._TENSOR_MODEL_PARALLEL_GROUP = fake_tp_group
    mpu._EXPERT_MODEL_PARALLEL_GROUP = fake_ep_group

    # Short aliases.
    tp_size = margs.tensor_model_parallel_size
    pp_size = margs.pipeline_model_parallel_size
    vp_size = margs.virtual_pipeline_model_parallel_size
    if vp_size is None:
        vp_size = 1

    # Metadata.
    md = types.SimpleNamespace()
    md.model_type = args.model_type
    md.num_layers = margs.num_layers
    md.hidden_size = margs.hidden_size
    md.seq_length = margs.seq_length
    md.num_attention_heads = margs.num_attention_heads
    md.max_position_embeddings = margs.max_position_embeddings
    md.tokenizer_type = margs.tokenizer_type
    md.iteration = margs.iteration
    md.params_dtype = margs.params_dtype
    md.bert_binary_head = margs.bert_binary_head
    md.output_layer = margs.untie_embeddings_and_output_weights
    md.position_embedding_type = margs.position_embedding_type
    md.linear_bias = margs.add_bias_linear
    md.qkv_bias = margs.add_qkv_bias
    md.norm_has_bias = False
    md.swiglu = margs.swiglu
    md.previous_tensor_parallel_size = margs.tensor_model_parallel_size
    md.previous_pipeline_parallel_size = margs.pipeline_model_parallel_size
    md.make_vocab_size_divisible_by = margs.make_vocab_size_divisible_by
    md.checkpoint_args = margs
    md.consumed_train_samples = 0
    md.consumed_valid_samples = 0

    # Get true (non-padded) vocab size
    tokenizer = transformers.AutoTokenizer.from_pretrained(margs.tokenizer_model)
    md.true_vocab_size = tokenizer._tokenizer.get_vocab_size(with_added_tokens=True)

    # Get first pipe stage.
    mpu.set_tensor_model_parallel_rank(0)
    mpu.set_pipeline_model_parallel_rank(0)
    model = load_checkpoint_to_model(margs)

    queue.put(md)

    def queue_put(name, msg):
        print(f"sending {name}")
        msg["name"] = name
        queue.put(msg)

    # Send embeddings.
    message = {"word embeddings": model.embedding.word_embeddings.weight.data}
    if md.position_embedding_type == "learned_absolute":
        message["position embeddings"] = model.embedding.position_embeddings.weight.data
    else:
        assert not hasattr(model.embedding, "position_embeddings")

    queue_put("embeddings", message)

    with open(margs.heterogeneous_layers_config_path, "r") as f:
        heterogeneous_layers_config = json.load(f)

    for layer_num in range(margs.num_layers):
        message = {}

        # add heterogeneous layers config
        layer_config = heterogeneous_layers_config["block_configs"][layer_num]
        message["heterogeneous layer config"] = layer_config

        layer = model.decoder.layers[layer_num]

        # Handle self-attention.
        if not layer_config["attention"]["no_op"]:
            if layer_config["attention"]["replace_with_linear"]:
                message["input norm weight"] = (
                    layer.self_attention.layer_norm_weight.data
                )
                message["linear attention weight"] = layer.self_attention.weight.data
                if md.linear_bias:
                    message["linear attention bias"] = layer.self_attention.bias.data
            else:
                message["input norm weight"] = (
                    layer.self_attention.linear_qkv.layer_norm_weight.data
                )
                message["qkv weight"] = layer.self_attention.linear_qkv.weight.data
                if md.qkv_bias:
                    message["qkv bias"] = layer.self_attention.linear_qkv.bias.data
                message["dense weight"] = layer.self_attention.linear_proj.weight.data
                if md.linear_bias:
                    message["dense bias"] = layer.self_attention.linear_proj.bias.data

        # Handle MLP.
        if not layer_config["ffn"]["no_op"]:
            if layer_config["ffn"]["replace_with_linear"]:
                message["post norm weight"] = layer.mlp.layer_norm_weight.data
                message["linear mlp weight"] = layer.mlp.weight.data
                if md.linear_bias:
                    message["linear mlp bias"] = layer.mlp.bias.data
            else:
                message["post norm weight"] = (
                    layer.mlp.linear_fc1.layer_norm_weight.data
                )
                # Handle gated linear units.
                if md.swiglu:
                    message["mlp l0 weight W"], message["mlp l0 weight V"] = (
                        torch.chunk(layer.mlp.linear_fc1.weight.data, 2, dim=0)
                    )
                    if md.linear_bias:
                        message["mlp l0 bias W"], message["mlp l0 bias V"] = (
                            torch.chunk(layer.mlp.linear_fc1.bias.data, 2, dim=0)
                        )
                else:
                    message["mlp l0 weight"] = layer.mlp.linear_fc1.weight.data
                    if md.linear_bias:
                        message["mlp l0 bias"] = layer.mlp.linear_fc1.bias.data

                message["mlp l1 weight"] = layer.mlp.linear_fc2.weight.data
                if md.linear_bias:
                    message["mlp l1 bias"] = layer.mlp.linear_fc2.bias.data

        queue_put(f"transformer layer {layer_num}", message)

    # Send final norm from tp_rank 0.
    message = {
        "weight": model.decoder.final_layernorm.weight.data,
    }
    queue_put("final norm", message)

    if md.output_layer:
        message = {"weight": model.output_layer.weight.data}
        queue_put("output layer", message)

    queue.put("done")


def load_checkpoint(queue, args):
    try:
        _load_checkpoint(queue, args)
    except Exception:
        queue.put("exit")
        raise 