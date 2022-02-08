import argparse
import concurrent.futures
import os
import sys

import torch

def add_arguments(parser):
    group = parser.add_argument_group(title='Megatron saver')

    group.add_argument('--megatron-path', type=str, default=None,
                       help='Base directory of Megatron repository')

    group.add_argument('--target-tensor-parallel-size', type=int,
                       help='Target tensor model parallel size, defaults to the tensor parallel size '
                       'in the input checkpoint if provided by the loader, otherwise to 1')
    group.add_argument('--target-pipeline-parallel-size', type=int,
                       help='Target tensor model parallel size, default to the pipeline parall size '
                       'in the input checkpoint if provided by the loader, otherwise to 1')

def save_checkpoint(queue, args):

    # Search in directory above this
    sys.path.append(os.path.abspath(
        os.path.join(os.path.dirname(__file__),
                     os.path.pardir)))
    if args.megatron_path is not None:
        sys.path.insert(0, args.megatron_path)

    try:
        from megatron.checkpointing import save_checkpoint
        from megatron.global_vars import set_global_variables, get_args
        from megatron.model import ModelType
        from megatron import mpu, fused_kernels
    except ModuleNotFoundError:
        print("Unable to import Megatron, please specify the path to Megatron using --megatron-path. Exiting.")
        exit(1)

    def queue_get():
        val = queue.get()
        if val == "exit":
            print("Loader exited, exiting saver")
            exit(1)
        return val

    md = queue_get()


    if args.target_tensor_parallel_size is None:
        if hasattr(md, 'previous_tensor_parallel_size'):
            args.target_tensor_parallel_size = md.previous_tensor_parallel_size
        else:
            print("loader did not provide a tensor parallel size and --target-tensor-parallel-size not provided on command line. "
                  "Default to 1.")
            args.target_tensor_parallel_size = 1

    if args.target_pipeline_parallel_size is None:
        if hasattr(md, 'previous_pipeline_parallel_size'):
            args.target_pipeline_parallel_size = md.previous_pipeline_parallel_size
        else:
            print("loader did not provide a pipeline parallel size and --target-pipeline-parallel-size not provided on command line. "
                  "Default to 1.")
            args.target_pipeline_parallel_size = 1


    # Arguments do sanity checks on the world size, but we don't care,
    # so trick it into thinking we are plenty of processes
    if args.target_tensor_parallel_size is not None and args.target_pipeline_parallel_size is not None:
        os.environ["WORLD_SIZE"] = f'{args.target_tensor_parallel_size * args.target_pipeline_parallel_size}'

    # We want all arguments to come from us
    sys.argv = ['script.py',
                '--num-layers', str(md.num_layers),
                '--hidden-size', str(md.hidden_size),
                '--seq-length', str(md.seq_length),
                '--num-attention-heads', str(md.num_attention_heads),
                '--max-position-embeddings', str(md.max_position_embeddings),
                '--tokenizer-type', str(md.tokenizer_type),
                '--tensor-model-parallel-size', str(args.target_tensor_parallel_size),
                '--pipeline-model-parallel-size', str(args.target_pipeline_parallel_size),
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
                '--save-interval', '1',
                '--save', args.save_dir
                ]
    if md.params_dtype == torch.float16:
        sys.argv.append('--fp16')
    elif md.params_dtype == torch.bfloat16:
        sys.argv.append('--bf16')

    if md.model_type == 'BERT' and not md.bert_binary_head:
        sys.argv.append('--bert-no-binary-head')
    set_global_variables()

    # margs = megatron args
    margs = get_args()

    # Determine how to make our models
    if md.model_type == 'GPT':
        from pretrain_gpt import model_provider
        margs.model_type = ModelType.encoder_or_decoder
    elif md.model_type == 'BERT':
        from pretrain_bert import model_provider
        margs.model_type = ModelType.encoder_or_decoder
    else:
        raise Exception(f'unrecognized model type: {args.model_type}')

    def get_models(count, dtype, pre_process, post_process):
        models = [model_provider(pre_process, post_process).to(dtype) for _ in range(count)]
        return models

    # fake initializing distributed
    mpu.initialize.set_tensor_model_parallel_world_size(args.target_tensor_parallel_size)
    mpu.initialize.set_pipeline_model_parallel_world_size(args.target_pipeline_parallel_size)
    mpu.initialize.set_tensor_model_parallel_rank(0)
    mpu.initialize.set_pipeline_model_parallel_rank(0)
    fused_kernels.load(margs)

    # Embeddings
    #-----------
    pos_embed = queue_get()
    full_word_embed = queue_get()

    # Tell Megatron what our full size is
    margs.padded_vocab_size = full_word_embed.shape[0]
    if margs.padded_vocab_size % args.target_tensor_parallel_size != 0:
        print("source vocab size is not evenly divisble by target tensor parallel size")
        exit(1)

    # Split into new tensor model parallel sizes
    out_word_embed = torch.chunk(full_word_embed, args.target_tensor_parallel_size, dim=0)

    # Make models for first pipeline stage and fill in embeddings
    mpu.initialize.set_pipeline_model_parallel_rank(0)
    post_process = args.target_pipeline_parallel_size == 1
    models = get_models(args.target_tensor_parallel_size, md.params_dtype, True, post_process)
    for tp_rank, model in enumerate(models):
        model.language_model.embedding.word_embeddings.weight.data.copy_(out_word_embed[tp_rank])
        model.language_model.embedding.position_embeddings.weight.data.copy_(pos_embed)

    # Transformer layers
    #-------------------
    for pp_rank in range(args.target_pipeline_parallel_size):
        # For later pipeline parallel ranks, make the new models
        if pp_rank > 0:
            mpu.initialize.set_pipeline_model_parallel_rank(pp_rank)
            post_process = pp_rank == args.target_pipeline_parallel_size - 1
            models = get_models(args.target_tensor_parallel_size, md.params_dtype, False, post_process)

        for layer in range(len(models[0].language_model.encoder.layers)):
            # get full tensors
            input_layernorm_weight = queue_get()
            input_layernorm_bias = queue_get()
            full_qkv_weight = queue_get()
            full_qkv_bias = queue_get()
            full_dense_weight = queue_get()
            dense_bias = queue_get()
            post_layernorm_weight = queue_get()
            post_layernorm_bias = queue_get()
            full_mlp_l0_weight = queue_get()
            full_mlp_l0_bias = queue_get()
            full_mlp_l1_weight = queue_get()
            mlp_l1_bias = queue_get()

            # Split up the parallel tensors
            out_qkv_weight = torch.chunk(full_qkv_weight, args.target_tensor_parallel_size, dim=0)
            out_qkv_bias = torch.chunk(full_qkv_bias, args.target_tensor_parallel_size, dim=0)
            out_dense_weight = torch.chunk(full_dense_weight, args.target_tensor_parallel_size, dim=1)
            out_mlp_l0_weight = torch.chunk(full_mlp_l0_weight, args.target_tensor_parallel_size, dim=0)
            out_mlp_l0_bias = torch.chunk(full_mlp_l0_bias, args.target_tensor_parallel_size, dim=0)
            out_mlp_l1_weight = torch.chunk(full_mlp_l1_weight, args.target_tensor_parallel_size, dim=1)

            # Save them to the model
            for tp_rank in range(args.target_tensor_parallel_size):
                l = models[tp_rank].language_model.encoder.layers[layer]
                l.input_layernorm.weight.data.copy_(input_layernorm_weight)
                l.input_layernorm.bias.data.copy_(input_layernorm_bias)
                l.self_attention.query_key_value.weight.data.copy_(out_qkv_weight[tp_rank])
                l.self_attention.query_key_value.bias.data.copy_(out_qkv_bias[tp_rank])
                l.self_attention.dense.weight.data.copy_(out_dense_weight[tp_rank])
                l.self_attention.dense.bias.data.copy_(dense_bias)
                l.post_attention_layernorm.weight.data.copy_(post_layernorm_weight)
                l.post_attention_layernorm.bias.data.copy_(post_layernorm_bias)
                l.mlp.dense_h_to_4h.weight.data.copy_(out_mlp_l0_weight[tp_rank])
                l.mlp.dense_h_to_4h.bias.data.copy_(out_mlp_l0_bias[tp_rank])
                l.mlp.dense_4h_to_h.weight.data.copy_(out_mlp_l1_weight[tp_rank])
                l.mlp.dense_4h_to_h.bias.data.copy_(mlp_l1_bias)

        if post_process:
            final_layernorm_weight = queue_get()
            final_layernorm_bias = queue_get()
            for tp_rank in range(args.target_tensor_parallel_size):
                models[tp_rank].language_model.encoder.final_layernorm.weight.data.copy_(final_layernorm_weight)
                models[tp_rank].language_model.encoder.final_layernorm.bias.data.copy_(final_layernorm_bias)
                if pp_rank != 0:
                    # Copy word embeddings to final pipeline rank
                    models[tp_rank].word_embeddings.weight.data.copy_(out_word_embed[tp_rank])
            del final_layernorm_weight
            del final_layernorm_bias

            name = queue_get()
            if name == "pooler":
                if not hasattr(models[0].language_model, 'pooler'):
                    print("ERROR: got a pooler, but model does not have one")
                    exit(1)
                pooler_weight = queue_get()
                pooler_bias = queue_get()
                for tp_rank in range(args.target_tensor_parallel_size):
                    models[tp_rank].language_model.pooler.dense.weight.data.copy_(pooler_weight)
                    models[tp_rank].language_model.pooler.dense.bias.data.copy_(pooler_bias)
                name = queue_get()
                del pooler_weight
                del pooler_bias

            if name == "lm head":
                if not hasattr(models[0], 'lm_head'):
                    print("ERROR: got an lm head, but model does not have one")
                    exit(1)
                lm_head_dense_weight = queue_get()
                lm_head_dense_bias = queue_get()
                lm_head_layernorm_weight = queue_get()
                lm_head_layernorm_bias = queue_get()
                for tp_rank in range(args.target_tensor_parallel_size):
                    models[tp_rank].lm_head.dense.weight.data.copy_(lm_head_dense_weight)
                    models[tp_rank].lm_head.dense.bias.data.copy_(lm_head_dense_bias)
                    models[tp_rank].lm_head.layernorm.weight.data.copy_(lm_head_layernorm_weight)
                    models[tp_rank].lm_head.layernorm.bias.data.copy_(lm_head_layernorm_bias)
                name = queue_get()

            if name == "binary head":
                if not hasattr(models[0], 'binary_head'):
                    print("ERROR: got a binary head, but model does not have one")
                    exit(1)
                binary_head_weight = queue_get()
                binary_head_bias = queue_get()
                for tp_rank in range(args.target_tensor_parallel_size):
                    models[tp_rank].binary_head.weight.data.copy_(binary_head_weight)
                    models[tp_rank].binary_head.bias.data.copy_(binary_head_bias)
                name = queue_get()

            if name != "done":
                print("ERROR: got some more data but were expecting to be done")

        for tp_rank in range(args.target_tensor_parallel_size):
            mpu.initialize.set_tensor_model_parallel_rank(tp_rank)
            save_checkpoint(md.iteration, [models[tp_rank]], None, None)
    print("Done!")
