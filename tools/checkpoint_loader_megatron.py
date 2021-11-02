import os
import sys
import types

import torch

def add_arguments(parser):
    group = parser.add_argument_group(title='Megatron loader')

    group.add_argument('--megatron-path', type=str, default=None,
                       help='Base directory of deepspeed repository')

def _load_checkpoint(queue, args):

    # Search in directory above this
    sys.path.append(os.path.abspath(
        os.path.join(os.path.dirname(__file__),
                     os.path.pardir)))
    if args.megatron_path is not None:
        sys.path.insert(0, args.megatron_path)

    try:
        from megatron.arguments import parse_args, validate_args
        from megatron.global_vars import set_args, set_global_variables, rebuild_tokenizer
        from megatron.checkpointing import load_args_from_checkpoint, load_checkpoint
        from megatron.model import ModelType
        from megatron import mpu, fused_kernels
    except ModuleNotFoundError:
        print("Unable to import Megatron, please specify the path to Megatron using --megatron-path. Exiting.")
        queue.put("exit")
        exit(1)

    # We want all arguments to come from us
    sys.argv = ['script.py',
                '--no-masked-softmax-fusion',
                '--no-bias-gelu-fusion',
                '--no-bias-dropout-fusion',
                '--use-cpu-initialization',
                '--micro-batch-size', '1',
                '--no-load-optim',
                '--no-load-rng',
                '--no-save-optim',
                '--no-save-rng',
                '--no-initialization',
                '--load', args.load_dir
                ]

    margs = parse_args(validate=False)
    margs = load_args_from_checkpoint(margs)

    def check_for_arg(arg_name):
        if getattr(margs, arg_name, None) is None:
            print(f"Checkpoint does not specify the argument {arg_name}. Exiting.")
            print(f"Arguments: {margs}")
            queue.put("exit")
            exit(1)

    check_for_arg('tensor_model_parallel_size')
    check_for_arg('pipeline_model_parallel_size')
    check_for_arg('num_layers')
    check_for_arg('hidden_size')
    check_for_arg('seq_length')
    check_for_arg('num_attention_heads')
    check_for_arg('max_position_embeddings')
    check_for_arg('tokenizer_type')
    check_for_arg('iteration')
    check_for_arg('bert_binary_head')

    # Arguments do sanity checks on the world size, but we don't care,
    # so trick it into thinking we are plenty of processes
    os.environ["WORLD_SIZE"] = f'{margs.tensor_model_parallel_size * margs.pipeline_model_parallel_size}'

    margs = validate_args(margs)

    check_for_arg('params_dtype')

    # Determine how to make our models
    if args.model_type == 'GPT':
        from pretrain_gpt import model_provider
        margs.model_type = ModelType.encoder_or_decoder
    elif args.model_type == 'BERT':
        from pretrain_bert import model_provider
        margs.model_type = ModelType.encoder_or_decoder
    else:
        raise Exception(f'unrecognized model type: {args.model_type}')

    def get_models(count, dtype, pre_process, post_process):
        # with concurrent.futures.ThreadPoolExecutor(max_workers=count) as executor:
        #     futures = [executor.submit(model_provider, pre_process, post_process) for _ in range(count)]
        #     models = [f.result().bfloat16() for f in futures]
        models = []
        for rank in range(count):
            mpu.initialize.set_tensor_model_parallel_rank(rank)
            model_ = [model_provider(pre_process, post_process).to(dtype)]
            margs.consumed_train_samples = 0
            margs.consumed_valid_samples = 0
            load_checkpoint(model_, None, None)
            assert(len(model_) == 1)
            models.append(model_[0])
        return models

    set_args(margs)

    if margs.num_layers_per_virtual_pipeline_stage is not None:
        print("Model with an interleaved pipeline schedule are not yet supported.")
        queue.put("exit")
        exit(1)

    set_global_variables(parse_args=False)
    mpu.initialize.set_tensor_model_parallel_world_size(margs.tensor_model_parallel_size)
    mpu.initialize.set_pipeline_model_parallel_world_size(margs.pipeline_model_parallel_size)
    fused_kernels.load(margs)

    # short aliases
    tp_size = margs.tensor_model_parallel_size
    pp_size = margs.pipeline_model_parallel_size

    # metadata
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
    md.previous_tensor_parallel_size = margs.tensor_model_parallel_size
    md.previous_pipeline_parallel_size = margs.pipeline_model_parallel_size
    queue.put(md)

    # Get first pipe stage
    mpu.initialize.set_pipeline_model_parallel_rank(0)
    post_process = pp_size == 1
    models = get_models(tp_size, md.params_dtype, True, post_process)

    # Send embeddings
    word_embed = []
    for tp_rank in range(tp_size):
        if tp_rank == 0:
            print("Sending position embeddings")
            queue.put(models[tp_rank].language_model.embedding.position_embeddings.weight.data)
        word_embed.append(models[tp_rank].language_model.embedding.word_embeddings.weight.data)
    full_word_embed = torch.cat(word_embed, dim=0)
    print("Sending word embeddings")
    queue.put(full_word_embed)

    total_layer_num = 0
    for pp_rank in range(pp_size):
        if pp_rank > 0:
            mpu.initialize.set_pipeline_model_parallel_rank(pp_rank)
            post_process = pp_rank == pp_size - 1
            models = get_models(tp_size, md.params_dtype, False, post_process)
        for layer_num in range(len(models[0].language_model.encoder.layers)):
            qkv_weight = []
            qkv_bias = []
            dense_weight = []
            mlp_l0_weight = []
            mlp_l0_bias = []
            mlp_l1_weight = []

            # Get non-parallel tensors from tp_rank 0
            layer = models[0].language_model.encoder.layers[layer_num]
            input_layernorm_weight = layer.input_layernorm.weight.data
            input_layernorm_bias = layer.input_layernorm.bias.data
            dense_bias = layer.self_attention.dense.bias.data
            post_layernorm_weight = layer.post_attention_layernorm.weight.data
            post_layernorm_bias = layer.post_attention_layernorm.bias.data
            mlp_l1_bias = layer.mlp.dense_4h_to_h.bias.data

            # Grab all parallel tensors for this layer
            for tp_rank, model in enumerate(models):
                layer = model.language_model.encoder.layers[layer_num]
                qkv_weight.append(layer.self_attention.query_key_value.weight.data)
                qkv_bias.append(layer.self_attention.query_key_value.bias.data)
                dense_weight.append(layer.self_attention.dense.weight.data)
                mlp_l0_weight.append(layer.mlp.dense_h_to_4h.weight.data)
                mlp_l0_bias.append(layer.mlp.dense_h_to_4h.bias.data)
                mlp_l1_weight.append(layer.mlp.dense_4h_to_h.weight.data)

            # send everything in order while concatenating them
            print(f"Sending layer {layer_num} of pipeline rank {pp_rank} (total layer {total_layer_num})")
            queue.put(input_layernorm_weight)
            queue.put(input_layernorm_bias)
            queue.put(torch.cat(qkv_weight, dim=0))
            queue.put(torch.cat(qkv_bias, dim=0))
            queue.put(torch.cat(dense_weight, dim=1))
            queue.put(dense_bias)
            queue.put(post_layernorm_weight)
            queue.put(post_layernorm_bias)
            queue.put(torch.cat(mlp_l0_weight, dim=0))
            queue.put(torch.cat(mlp_l0_bias, dim=0))
            queue.put(torch.cat(mlp_l1_weight, dim=1))
            queue.put(mlp_l1_bias)

            total_layer_num = total_layer_num + 1

    # Send final layernorm from tp_rank 0
    print("Sending final layernorm")
    queue.put(models[0].language_model.encoder.final_layernorm.weight.data)
    queue.put(models[0].language_model.encoder.final_layernorm.bias.data)

    # Send BERT lm head and binary head if it exists
    if md.model_type == 'BERT':
        print("Sending LM Pooler")
        queue.put("pooler")
        queue.put(models[0].language_model.pooler.dense.weight.data)
        queue.put(models[0].language_model.pooler.dense.bias.data)

        print("Sending BERT LM head")
        queue.put("lm head")
        queue.put(models[0].lm_head.dense.weight.data)
        queue.put(models[0].lm_head.dense.bias.data)
        queue.put(models[0].lm_head.layernorm.weight.data)
        queue.put(models[0].lm_head.layernorm.bias.data)

        if md.bert_binary_head:
            print("Sending BERT Binary head")
            queue.put("binary head")
            queue.put(models[0].binary_head.weight.data)
            queue.put(models[0].binary_head.bias.data)
    queue.put("done")

def load_checkpoint(queue, args):
    try:
        _load_checkpoint(queue, args)
    except:
        queue.put("exit")
        raise
