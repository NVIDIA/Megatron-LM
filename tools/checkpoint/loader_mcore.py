# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import json
import os
import sys
import types
import torch
import packaging

from schema_mcore import get_model_schema
from utils import print_memory_usage


def add_arguments(parser):
    group = parser.add_argument_group(title='Megatron loader')

    group.add_argument('--true-vocab-size', type=int, default=None,
                       help='original size of vocab, if specified will trim padding from embedding table.')
    group.add_argument('--vocab-file', type=str, default=None,
                       help='Path to the vocab file. If specified will use this to get vocab size and '
                       'trim padding from the embedding table.')
    group.add_argument('--megatron-path', type=str, default=None,
                       help='Base directory of Megatron repository')
    group.add_argument('--position-embedding-type',
                       type=str,
                       default='learned_absolute',
                       choices=['learned_absolute', 'rope'],
                       help='Position embedding type.')
    group.add_argument('--loader-transformer-impl', default='transformer_engine',
                       choices=['local', 'transformer_engine'],
                       help='Which Transformer implementation to use.')


def _load_checkpoint(queue, args):

    # Search in directory above this
    sys.path.append(os.path.abspath(
        os.path.join(os.path.dirname(__file__),
                     os.path.pardir,
                     os.path.pardir)))
    if args.megatron_path is not None:
        sys.path.insert(0, args.megatron_path)

    try:
        from megatron.training.arguments import parse_args, validate_args
        from megatron.training.global_vars import set_global_variables
        from megatron.training.checkpointing import load_args_from_checkpoint, load_checkpoint
        from megatron.core.parallel_state import initialize_model_parallel
        from megatron.legacy.model import module
        from megatron.core import mpu
        from megatron.core.enums import ModelType
        from megatron.legacy import fused_kernels
    except ModuleNotFoundError:
        print("Unable to import Megatron, please specify the path to Megatron using --megatron-path. Exiting.")
        queue.put("exit")
        exit(1)

    # We want all arguments to come from us
    sys.argv = ['script.py',
                '--no-masked-softmax-fusion',
                '--no-bias-gelu-fusion',
                '--no-bias-dropout-fusion',
                '--no-async-tensor-model-parallel-allreduce',
                '--use-cpu-initialization',
                '--auto-detect-ckpt-format',
                '--micro-batch-size', '1',
                '--no-load-optim',
                '--no-load-rng',
                '--no-save-optim',
                '--no-save-rng',
                '--no-initialization',
                '--mock-data', # To pass the "blend data checks" in arguments.py
                '--load', args.load_dir,
                '--position-embedding-type', args.position_embedding_type,
                '--exit-on-missing-checkpoint',
                '--no-one-logger',
                ]

    margs = parse_args()

    device_count = torch.cuda.device_count()
    if device_count > 0:
        torch.cuda.set_device(0)
        device_id = torch.device(f'cuda:0')
    else:
        device_id = None

    margs, checkpoint_args = load_args_from_checkpoint(margs)

    # for now, if load dist ckpt, we load it as tp1pp1ep1vp1 for convenience
    if checkpoint_args.use_dist_ckpt:
        # Call the init process
        init_process_group_kwargs = {
            'backend': 'nccl',
            'world_size': 1,
            'rank': 0,
        }

        if packaging.version.Version(torch.__version__) >= packaging.version.Version("2.3.0"):
            init_process_group_kwargs['device_id'] = device_id
        margs.tensor_model_parallel_size = 1
        margs.pipeline_model_parallel_size = 1
        margs.expert_model_parallel_size = 1
        margs.virtual_pipeline_model_parallel_size = 1
        torch.distributed.init_process_group(**init_process_group_kwargs)
        initialize_model_parallel()
        print(f"real initializing distributed")
    else:
        print(f"fake initializing distributed")
        margs.tensor_model_parallel_size = checkpoint_args.tensor_model_parallel_size
        margs.pipeline_model_parallel_size = checkpoint_args.pipeline_model_parallel_size
        margs.expert_model_parallel_size = checkpoint_args.expert_model_parallel_size
        margs.virtual_pipeline_model_parallel_size = checkpoint_args.virtual_pipeline_model_parallel_size
        margs.sequence_parallel = checkpoint_args.sequence_parallel
        margs.ckpt_format = checkpoint_args.ckpt_format

    # Arguments do sanity checks on the world size, but we don't care,
    # so trick it into thinking we are plenty of processes
    margs.world_size = margs.tensor_model_parallel_size * margs.pipeline_model_parallel_size * margs.expert_model_parallel_size

    # Explicitly copy data types from checkpoint.
    margs.fp16 = checkpoint_args.fp16
    margs.bf16 = checkpoint_args.bf16

    margs.use_legacy_models = False
    margs.transformer_impl = args.loader_transformer_impl
    margs.norm_epsilon = checkpoint_args.norm_epsilon
    margs.rotary_base = checkpoint_args.rotary_base
    if checkpoint_args.num_experts:
        margs.moe_shared_expert_intermediate_size = checkpoint_args.moe_shared_expert_intermediate_size
        margs.num_experts = checkpoint_args.num_experts
        margs.moe_router_topk = checkpoint_args.moe_router_topk

    # Expert parallelism requires sequence parallelism.
    if margs.expert_model_parallel_size > 1:
        margs.sequence_parallel = True

    # Validate margs.
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

    check_for_arg('tensor_model_parallel_size')
    check_for_arg('pipeline_model_parallel_size')
    check_for_arg('expert_model_parallel_size')
    check_for_arg('num_layers')
    check_for_arg('hidden_size')
    check_for_arg('seq_length')
    check_for_arg('num_attention_heads')
    check_for_arg('max_position_embeddings')
    check_for_arg('position_embedding_type')
    check_for_arg('tokenizer_type')
    check_for_arg('iteration')
    check_for_arg('bert_binary_head')
    check_for_arg('disable_bias_linear', False)
    check_for_arg('params_dtype')
    check_for_arg('swiglu', False)
    if checkpoint_args.num_experts:
        check_for_arg('num_experts')
    print(f"checkpoint_args {checkpoint_args}")
    # Determine how to make our models
    if args.model_type == 'GPT':
        from pretrain_gpt import model_provider
        margs.model_type = ModelType.encoder_or_decoder
    elif args.model_type == 'BERT':
        from pretrain_bert import model_provider
        margs.model_type = ModelType.encoder_or_decoder
    else:
        raise Exception(f'unrecognized model type: {args.model_type}')

    # supress warning about torch.distributed not being initialized
    module.MegatronModule.embedding_warning_printed = True

    consumed_train_samples = None
    consumed_valid_samples = None
    def get_models(tp_size, ep_size, dtype):
        nonlocal consumed_train_samples
        nonlocal consumed_valid_samples
        model_array_len = margs.virtual_pipeline_model_parallel_size
        if model_array_len is None:
            model_array_len = 1
        models = [[[] for _ in range(ep_size)] for _ in range(model_array_len)]
        pre_process = mpu.is_pipeline_first_stage()
        post_process = mpu.is_pipeline_last_stage()
        for ep_rank in range(ep_size):
            mpu.set_expert_model_parallel_rank(ep_rank)
            for tp_rank in range(tp_size):
                mpu.set_tensor_model_parallel_rank(tp_rank)
                if margs.virtual_pipeline_model_parallel_size is not None:
                    model_ = []
                    for i in range(margs.virtual_pipeline_model_parallel_size):
                        mpu.set_virtual_pipeline_model_parallel_rank(i)
                        # Set pre_process and post_process only after virtual rank is set.
                        pre_process = mpu.is_pipeline_first_stage()
                        post_process = mpu.is_pipeline_last_stage()
                        this_model = model_provider(
                            pre_process=pre_process,
                            post_process=post_process
                        ).to(dtype)
                        model_.append(this_model)
                else:
                    pre_process = mpu.is_pipeline_first_stage()
                    post_process = mpu.is_pipeline_last_stage()
                    model_ = [model_provider(pre_process, post_process).to(dtype)]
                margs.consumed_train_samples = 0
                margs.consumed_valid_samples = 0
                margs.exit_on_missing_checkpoint = True
                load_checkpoint(model_, None, None, strict=False)

                if consumed_train_samples is not None:
                    assert(margs.consumed_train_samples == consumed_train_samples)
                else:
                    consumed_train_samples = margs.consumed_train_samples
                if consumed_valid_samples is not None:
                    assert(margs.consumed_valid_samples == consumed_valid_samples)
                else:
                    consumed_valid_samples = margs.consumed_valid_samples
                for vp_rank in range(model_array_len):
                    models[vp_rank][ep_rank].append(model_[vp_rank])

                # Print memory usage.
                print_memory_usage("loader", tp_rank, tp_size)

        return models

    set_global_variables(margs, build_tokenizer=False)
    mpu.set_tensor_model_parallel_world_size(margs.tensor_model_parallel_size)
    mpu.set_pipeline_model_parallel_world_size(margs.pipeline_model_parallel_size)
    mpu.set_expert_model_parallel_world_size(margs.expert_model_parallel_size)
    mpu.set_virtual_pipeline_model_parallel_world_size(margs.virtual_pipeline_model_parallel_size)
    fused_kernels.load(margs)
    print(f"loader's margs {margs}")
    # Get true (non-padded) vocab size
    if args.true_vocab_size is not None:
        true_vocab_size = args.true_vocab_size
    elif args.vocab_file is not None:
        with open(args.vocab_file) as vocab_file_handler:
            vocab = json.load(vocab_file_handler)
        true_vocab_size = len(vocab)
        if args.true_vocab_size is not None and true_vocab_size != args.true_vocab_size:
            print("Both --true-vocab-size and --vocab-file specified and the vocab size does not match, aborting.")
            queue.put("exit")
            exit(1)
    else:
        true_vocab_size = None

    # short aliases
    tp_size = margs.tensor_model_parallel_size
    pp_size = margs.pipeline_model_parallel_size
    ep_size = margs.expert_model_parallel_size
    vp_size = margs.virtual_pipeline_model_parallel_size
    if vp_size is None:
        vp_size = 1

    # Layernorm has bias; RMSNorm does not.
    if hasattr(checkpoint_args, 'normalization'):
        norm_has_bias = checkpoint_args.normalization == "LayerNorm"
    else:
        # older models only supported LayerNorm
        norm_has_bias = True

    # Metadata.
    md = types.SimpleNamespace()
    md.model_type = args.model_type
    md.num_layers = margs.num_layers
    md.hidden_size = margs.hidden_size
    md.ffn_hidden_size = margs.ffn_hidden_size
    md.seq_length = margs.seq_length
    md.num_attention_heads = margs.num_attention_heads
    md.max_position_embeddings = margs.max_position_embeddings
    md.tokenizer_type = margs.tokenizer_type
    md.iteration = margs.iteration
    md.params_dtype = margs.params_dtype
    md.bert_binary_head = margs.bert_binary_head
    md.output_layer = margs.untie_embeddings_and_output_weights
    md.untie_embeddings_and_output_weights = margs.untie_embeddings_and_output_weights
    md.position_embedding_type = margs.position_embedding_type
    md.linear_bias = margs.add_bias_linear
    md.qkv_bias = margs.add_qkv_bias
    md.norm_has_bias = norm_has_bias
    md.swiglu = margs.swiglu
    md.previous_tensor_parallel_size = margs.tensor_model_parallel_size
    md.previous_pipeline_parallel_size = margs.pipeline_model_parallel_size
    md.previous_expert_parallel_size = margs.expert_model_parallel_size
    md.true_vocab_size = true_vocab_size
    md.make_vocab_size_divisible_by = margs.make_vocab_size_divisible_by
    md.checkpoint_args = checkpoint_args
    md.use_legacy_models = margs.use_legacy_models
    md.num_query_groups = margs.num_query_groups
    md.group_query_attention = margs.group_query_attention
    md.norm_epsilon = margs.norm_epsilon
    md.rotary_base = margs.rotary_base
    md.padded_vocab_size = margs.padded_vocab_size
    md.num_experts = margs.num_experts
    md.moe_router_topk = margs.moe_router_topk
    md.moe_shared_expert_intermediate_size = margs.moe_shared_expert_intermediate_size

    # Get first pipe stage
    mpu.set_pipeline_model_parallel_rank(0)
    # all_models: pp_rank, vp_rank, ep_rank, tp_rank
    all_models = [get_models(tp_size, ep_size, md.params_dtype)]
    models = all_models[0][0]
    if ep_size == 1:
        assert len(models) == 1

    md.consumed_train_samples = consumed_train_samples
    md.consumed_valid_samples = consumed_valid_samples
    queue.put(md)

    def queue_put(name, msg):
        print(f"sending {name}")
        msg["name"] = name
        queue.put(msg)

    # Model schema.
    schema = get_model_schema(
        md.model_type,
        margs.transformer_impl,
        margs.num_experts,
        margs.expert_model_parallel_size,
    )

    # Send embeddings.
    embeddings = [ schema.get("embeddings", model) for model in models[0] ]
    message = {
        "word embeddings": torch.cat([ e["word"] for e in embeddings ], dim=0)
    }
    if md.position_embedding_type == 'learned_absolute':
        message["position embeddings"] = embeddings[0]["pos"]
    else:
        assert embeddings[0]["pos"] is None
    queue_put("embeddings", message)

    def set_common_message(message):
        # Get non-parallel tensors from tp_rank 0
        layer = schema.get_layer(models[0][0], layer_num)
        message["input norm weight"] = layer["self_attn_norm_weight"]
        message["post norm weight"] = layer["mlp_norm_weight"]
        if norm_has_bias:
            message["input norm bias"] = layer["self_attn_norm_bias"]
            message["post norm bias"] = layer["mlp_norm_bias"]
        if md.linear_bias:
            message["dense bias"] = layer["self_attn_proj_bias"]

        # Grab attention parallel tensors for this layer
        qkv_weight = []
        qkv_bias = []
        dense_weight = []
        for tp_rank, model in enumerate(models[0]):
            layer = schema.get_layer(model, layer_num)
            qkv_weight.append(layer["self_attn_qkv_weight"])
            dense_weight.append(layer["self_attn_proj_weight"])
            if md.qkv_bias:
                qkv_bias.append(layer["self_attn_qkv_bias"])

        # simple concat of the rest
        message["qkv weight"] = torch.cat(qkv_weight, dim=0)
        message["dense weight"] = torch.cat(dense_weight, dim=1)
        if md.qkv_bias:
            message["qkv bias"] = torch.cat(qkv_bias, dim=0)

    def set_message_for_dense_model(message):
        # Get non-parallel tensors from tp_rank 0
        layer = schema.get_layer(models[0][0], layer_num)
        if md.linear_bias:
            message["mlp l1 bias"] = layer["mlp_fc2_bias"]

        # Grab mlp parallel tensors for this layer
        mlp_l0_weight = []
        mlp_l0_bias = []
        mlp_l1_weight = []
        for tp_rank, model in enumerate(models[0]):
            layer = schema.get_layer(model, layer_num)
            mlp_l0_weight.append(layer["mlp_fc1_weight"])
            mlp_l1_weight.append(layer["mlp_fc2_weight"])
            if md.linear_bias:
                mlp_l0_bias.append(layer["mlp_fc1_bias"])

        # Handle gated linear units
        if md.swiglu:
            # concat all the first halves ('W's) and all the second halves ('V's)
            for tp_rank in range(tp_size):
                mlp_l0_weight[tp_rank] = torch.chunk(mlp_l0_weight[tp_rank], 2, dim=0)
            message["mlp l0 weight W"] = torch.cat([w[0] for w in mlp_l0_weight], dim=0)
            message["mlp l0 weight V"] = torch.cat([w[1] for w in mlp_l0_weight], dim=0)
        else:
            message["mlp l0 weight"] = torch.cat(mlp_l0_weight, dim=0)

        # simple concat of the rest
        message["mlp l1 weight"] = torch.cat(mlp_l1_weight, dim=1)
        if md.linear_bias:
            if md.swiglu:
                for tp_rank in range(tp_size):
                    mlp_l0_bias[tp_rank] = torch.chunk(mlp_l0_bias[tp_rank], 2, dim=0)
                message["mlp l0 bias W"] = torch.cat([b[0] for b in mlp_l0_bias], dim=0)
                message["mlp l0 bias V"] = torch.cat([b[1] for b in mlp_l0_bias], dim=0)
            else:
                message["mlp l0 bias"] = torch.cat(mlp_l0_bias, dim=0)

    def set_message_for_moe_model(args, message):
        use_shared_expert = args.moe_shared_expert_intermediate_size is not None

        # Get non-parallel tensors from tp_rank 0
        layer = schema.get_layer(models[0][0], layer_num)
        router_weight = layer["router_weight"]
        if use_shared_expert:
            shared_mlp_gate_weight = layer["shared_mlp_gate_weight"]

        # Grab all parallel tensors for this layer
        shared_expert_mlp_l0_weight = []
        shared_expert_mlp_l1_weight = []
        mlp_l0_weight_list = [[] for _ in range(margs.num_experts)]
        mlp_l1_weight_list = [[] for _ in range(margs.num_experts)]

        # Routed Experts modules
        num_experts_per_rank = margs.num_experts // ep_size
        for ep_rank, tp_models in enumerate(models):
            for tp_rank, model in enumerate(tp_models):
                layer = schema.get_layer(model, layer_num)
                for local_expert_idx in range(num_experts_per_rank):
                    expert_idx = int(ep_rank * num_experts_per_rank + local_expert_idx)
                    mlp_l0_weight_list[expert_idx].append(layer[f"mlp_fc1_weight.{local_expert_idx}"])
                    mlp_l1_weight_list[expert_idx].append(layer[f"mlp_fc2_weight.{local_expert_idx}"])

        mlp_l0_weight_w_list = [[] for _ in range(margs.num_experts)]
        mlp_l0_weight_v_list = [[] for _ in range(margs.num_experts)]
        # Concat along the tensor parallel dimension
        for expert_idx in range(margs.num_experts):
            mlp_l0_weight = mlp_l0_weight_list[expert_idx]
            if md.swiglu:
                for tp_rank in range(tp_size):
                    mlp_l0_weight[tp_rank] = torch.chunk(mlp_l0_weight[tp_rank], 2, dim=0)
                mlp_l0_weight_w_list[expert_idx] = torch.cat([w[0] for w in mlp_l0_weight], dim=0)
                mlp_l0_weight_v_list[expert_idx] = torch.cat([w[1] for w in mlp_l0_weight], dim=0)
            else:
                mlp_l0_weight_list[expert_idx] = torch.cat(mlp_l0_weight, dim=0)
            mlp_l1_weight_list[expert_idx] = torch.cat(mlp_l1_weight_list[expert_idx], dim=1)

        if md.swiglu:
            # Stack along the expert parallel dimension
            message["mlp l0 weight W"] = torch.stack(mlp_l0_weight_w_list)
            message["mlp l0 weight V"] = torch.stack(mlp_l0_weight_v_list)
        else:
            message["mlp l0 weight"] = torch.stack(mlp_l0_weight_list)
        message["mlp l1 weight"] = torch.stack(mlp_l1_weight_list)

        # Share Experts modules
        if use_shared_expert:
            for tp_rank, model in enumerate(models[0]):
                layer = schema.get_layer(model, layer_num)
                shared_expert_mlp_l0_weight.append(layer["shared_mlp_fc1_weight"])
                shared_expert_mlp_l1_weight.append(layer["shared_mlp_fc2_weight"])

            if md.swiglu:
                for tp_rank in range(tp_size):
                    shared_expert_mlp_l0_weight[tp_rank] = torch.chunk(shared_expert_mlp_l0_weight[tp_rank], 2, dim=0)
                message["shared mlp l0 weight W"] = torch.cat([w[0] for w in shared_expert_mlp_l0_weight], dim=0)
                message["shared mlp l0 weight V"] = torch.cat([w[1] for w in shared_expert_mlp_l0_weight], dim=0)
            else:
                message["shared mlp l0 weight"] = torch.cat(shared_expert_mlp_l0_weight, dim=0)
            message["shared mlp l1 weight"] = torch.cat(shared_expert_mlp_l1_weight, dim=1)
            message["shared gate weight"] = shared_mlp_gate_weight

        # Do nothing to router
        message["router weight"] = router_weight


    total_layer_num = 0
    for vp_rank in range(vp_size):
        mpu.set_virtual_pipeline_model_parallel_rank(vp_rank)
        for pp_rank in range(pp_size):
            if pp_rank > 0:
                mpu.set_pipeline_model_parallel_rank(pp_rank)
                if vp_rank == 0:
                    all_models.append(get_models(tp_size, ep_size, md.params_dtype))
            models = all_models[pp_rank][vp_rank]
            for layer_num in range(schema.get_num_layers(models[0][0])):
                message = {}
                set_common_message(message)
                if margs.num_experts:
                    set_message_for_moe_model(margs, message)
                else:
                    set_message_for_dense_model(message)

                queue_put(f"transformer layer {total_layer_num}", message)

                total_layer_num = total_layer_num + 1

    # Send final norm from tp_rank 0.
    final_norm = schema.get("final_norm", models[0][0])
    message = {
        "weight": final_norm["weight"],
    }
    if norm_has_bias:
        message["bias"] = final_norm["bias"]
    queue_put("final norm", message)

    # Send output layer.
    if md.output_layer:
        output_layer_ranks = [ schema.get("output_layer", m) for m in models[0] ]
        message = {
            "weight": torch.cat([r["weight"] for r in output_layer_ranks], dim=0),
        }
        queue_put("output layer", message)

    # Send BERT params.
    if md.model_type == 'BERT':

        # Pooler.
        pooler = schema.get("pooler", models[0][0])
        message = {
            "weight": pooler["weight"],
            "bias": pooler["bias"],
        }
        queue_put("pooler", message)

        # LM head.
        lm_head = schema.get("lm_head", models[0][0])
        message = {
            "dense weight": lm_head["dense_weight"],
            "dense bias": lm_head["dense_bias"],
            "norm weight": lm_head["norm_weight"],
        }
        if norm_has_bias:
            message["norm bias"] = lm_head["norm_bias"],
        queue_put("lm head", message)

        # Binary head.
        if md.bert_binary_head:
            binary_head = schema.get("binary_head", models[0][0])
            message = {
                "weight": binary_head["weight"],
                "bias": binary_head["bias"],
            }
            queue_put("binary head", message)

    # Done.
    queue.put("done")

def load_checkpoint(queue, args):
    try:
        _load_checkpoint(queue, args)
    except Exception:
        queue.put("exit")
        raise
