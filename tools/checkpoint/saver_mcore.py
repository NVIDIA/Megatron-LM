# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import os
import sys
import torch

from setter import ModelSetter
from utils import get_mcore_transformer_block_key, print_memory_usage
from megatron.core.utils import get_te_version, is_te_min_version


class MCoreSetter(ModelSetter):

    transformer_block_key = None

    @classmethod
    def get_transformer_block(cls, model):
        return getattr(model, cls.transformer_block_key)

    @classmethod
    def has_position_embeddings(cls, model):
        return hasattr(model.embedding, "position_embeddings")

    @classmethod
    def set_embeddings(
        cls,
        model,
        word=None,
        pos=None,
    ):
        cls.set_tensor(model.embedding.word_embeddings.weight, word)
        if pos is not None:
            cls.set_tensor(model.embedding.position_embeddings.weight, pos)

    @classmethod
    def set_final_norm(
        cls,
        model,
        weight=None,
        bias=None,
    ):
        block = cls.get_transformer_block(model)
        cls.set_tensor(block.final_layernorm.weight, weight)
        if bias is not None:
            cls.set_tensor(block.final_layernorm.bias, bias)

    @classmethod
    def set_output_word_embeddings(
        cls,
        model,
        emb=None,
    ):
        cls.set_tensor(model.embedding.word_embeddings.weight, emb)

    @classmethod
    def set_output_layer(
        cls,
        model,
        weight=None,
    ):
        cls.set_tensor(model.output_layer.weight, weight)

    @classmethod
    def set_pooler(
        cls,
        model,
        weight=None,
        bias=None,
    ):
        cls.set_tensor(model.pooler.dense.weight, weight)
        if bias is not None:
            cls.set_tensor(model.pooler.dense.bias, bias)

    @classmethod
    def set_lm_head(
        cls,
        model,
        dense_weight=None,
        dense_bias=None,
        norm_weight=None,
        norm_bias=None,
    ):

        cls.set_tensor(model.lm_head.dense.weight, dense_weight)
        if dense_bias is not None:
            cls.set_tensor(model.lm_head.dense.bias, dense_bias)

        cls.set_tensor(model.lm_head.layer_norm.weight, norm_weight)
        if norm_bias is not None:
            cls.set_tensor(model.lm_head.layer_norm.bias, norm_bias)

    @classmethod
    def set_binary_head(
        cls,
        model,
        weight=None,
        bias=None,
    ):
        cls.set_tensor(model.binary_head.weight, weight)
        if bias is not None:
            cls.set_tensor(model.binary_head.bias, bias)


class MCoreLocalSetter(MCoreSetter):

    @classmethod
    def set_layer(
        cls,
        model,
        layer_idx,
        self_attn_norm_weight=None,
        self_attn_norm_bias=None,
        self_attn_qkv_weight=None,
        self_attn_qkv_bias=None,
        self_attn_proj_weight=None,
        self_attn_proj_bias=None,
        mlp_norm_weight=None,
        mlp_norm_bias=None,
        mlp_fc1_weight=None,
        mlp_fc1_bias=None,
        mlp_fc2_weight=None,
        mlp_fc2_bias=None,
    ):

        block = cls.get_transformer_block(model)
        l = block.layers[layer_idx]

        # Self attention.
        cls.set_tensor(l.input_layernorm.weight, self_attn_norm_weight)
        if self_attn_norm_bias is not None:
            cls.set_tensor(l.input_layernorm.bias, self_attn_norm_bias)

        cls.set_tensor(l.self_attention.linear_qkv.weight, self_attn_qkv_weight)
        if self_attn_qkv_bias is not None:
            cls.set_tensor(l.self_attention.linear_qkv.bias, self_attn_qkv_bias)

        cls.set_tensor(l.self_attention.linear_proj.weight, self_attn_proj_weight)
        if self_attn_proj_bias is not None:
            cls.set_tensor(l.self_attention.linear_proj.bias, self_attn_proj_bias)

        # MLP.
        cls.set_tensor(l.pre_mlp_layernorm.weight, mlp_norm_weight)
        if mlp_norm_bias is not None:
            cls.set_tensor(l.pre_mlp_layernorm.bias, mlp_norm_bias)

        cls.set_tensor(l.mlp.linear_fc1.weight, mlp_fc1_weight)
        if mlp_fc1_bias is not None:
            cls.set_tensor(l.mlp.linear_fc1.bias, mlp_fc1_bias)

        cls.set_tensor(l.mlp.linear_fc2.weight, mlp_fc2_weight)
        if mlp_fc2_bias is not None:
            cls.set_tensor(l.mlp.linear_fc2.bias, mlp_fc2_bias)


class MCoreTESetter(MCoreSetter):

    @classmethod
    def set_layer(
        cls,
        model,
        layer_idx,
        self_attn_norm_weight=None,
        self_attn_norm_bias=None,
        self_attn_qkv_weight=None,
        self_attn_qkv_bias=None,
        self_attn_proj_weight=None,
        self_attn_proj_bias=None,
        mlp_norm_weight=None,
        mlp_norm_bias=None,
        mlp_fc1_weight=None,
        mlp_fc1_bias=None,
        mlp_fc2_weight=None,
        mlp_fc2_bias=None,
    ):

        block = cls.get_transformer_block(model)
        l = block.layers[layer_idx]

        # Self attention.
        cls.set_tensor(l.self_attention.linear_qkv.layer_norm_weight, self_attn_norm_weight)
        if self_attn_norm_bias is not None:
            cls.set_tensor(l.self_attention.linear_qkv.layer_norm_bias, self_attn_norm_bias)

        cls.set_tensor(l.self_attention.linear_qkv.weight, self_attn_qkv_weight)
        if self_attn_qkv_bias is not None:
            cls.set_tensor(l.self_attention.linear_qkv.bias, self_attn_qkv_bias)

        cls.set_tensor(l.self_attention.linear_proj.weight, self_attn_proj_weight)
        if self_attn_proj_bias is not None:
            cls.set_tensor(l.self_attention.linear_proj.bias, self_attn_proj_bias)

        # MLP.
        cls.set_tensor(l.mlp.linear_fc1.layer_norm_weight, mlp_norm_weight)
        if mlp_norm_bias is not None:
            cls.set_tensor(l.mlp.linear_fc1.layer_norm_bias, mlp_norm_bias)

        cls.set_tensor(l.mlp.linear_fc1.weight, mlp_fc1_weight)
        if mlp_fc1_bias is not None:
            cls.set_tensor(l.mlp.linear_fc1.bias, mlp_fc1_bias)

        cls.set_tensor(l.mlp.linear_fc2.weight, mlp_fc2_weight)
        if mlp_fc2_bias is not None:
            cls.set_tensor(l.mlp.linear_fc2.bias, mlp_fc2_bias)

class MCoreMoETESetter(MCoreSetter):

    @classmethod
    def set_layer(
        cls,
        model,
        layer_idx,
        router_weight=None,
        self_attn_norm_weight=None,
        self_attn_norm_bias=None,
        self_attn_qkv_weight=None,
        self_attn_qkv_bias=None,
        self_attn_proj_weight=None,
        self_attn_proj_bias=None,
        mlp_norm_weight=None,
        mlp_norm_bias=None,
        mlp_fc1_weight=None,
        mlp_fc1_bias=None,
        mlp_fc2_weight=None,
        mlp_fc2_bias=None,
    ):

        block = cls.get_transformer_block(model)
        l = block.layers[layer_idx]

        # Self attention.
        cls.set_tensor(l.self_attention.linear_qkv.layer_norm_weight, self_attn_norm_weight)
        if self_attn_norm_bias is not None:
            cls.set_tensor(l.self_attention.linear_qkv.layer_norm_bias, self_attn_norm_bias)
        cls.set_tensor(l.self_attention.linear_qkv.weight, self_attn_qkv_weight)
        if self_attn_qkv_bias is not None:
            cls.set_tensor(l.self_attention.linear_qkv.bias, self_attn_qkv_bias)
        cls.set_tensor(l.self_attention.linear_proj.weight, self_attn_proj_weight)
        if self_attn_proj_bias is not None:
            cls.set_tensor(l.self_attention.linear_proj.bias, self_attn_proj_bias)

        # MLP.
        cls.set_tensor(l.pre_mlp_layernorm.weight, mlp_norm_weight)
        if model.config.normalization == "LayerNorm":
            cls.set_tensor(l.pre_mlp_layernorm.bias, mlp_norm_bias)

        cls.set_tensor(l.mlp.router.weight, router_weight)

        num_local_experts = mlp_fc1_weight.shape[0]
        for expert_idx in range(num_local_experts):
            cls.set_tensor(l.mlp.experts.local_experts[expert_idx].linear_fc1.weight, mlp_fc1_weight[expert_idx])
            cls.set_tensor(l.mlp.experts.local_experts[expert_idx].linear_fc2.weight, mlp_fc2_weight[expert_idx])


def get_model_setter(model_type, transformer_impl, num_experts=0):
    if num_experts is not None and num_experts > 0:
        # Only support TE setter for MOE
        assert transformer_impl == "transformer_engine"
        setter = MCoreMoETESetter
    else:
        setter = {
            "local" : MCoreLocalSetter,
            "transformer_engine" : MCoreTESetter,
        }[transformer_impl]
    setter.transformer_block_key = get_mcore_transformer_block_key(model_type)
    return setter


def add_arguments(parser):
    group = parser.add_argument_group(title='M-Core saver')

    group.add_argument('--megatron-path', type=str, default=None,
                       help='Base directory of Megatron repository')

    group.add_argument('--target-tensor-parallel-size', type=int,
                       help='Target tensor model parallel size, defaults to the tensor parallel size '
                       'in the input checkpoint if provided by the loader, otherwise to 1')
    group.add_argument('--target-pipeline-parallel-size', type=int,
                       help='Target tensor model parallel size, default to the pipeline parall size '
                       'in the input checkpoint if provided by the loader, otherwise to 1')
    group.add_argument('--saver-transformer-impl', default='transformer_engine',
                       choices=['local', 'transformer_engine'],
                       help='Which Transformer implementation to use.')
    group.add_argument('--target-expert-parallel-size', type=int, default=1,
                       help='Target expert model parallel size, default to 1')


def save_checkpoint(queue, args):

    # Transformer engine >= 0.12.0, for CPU initialization.
    assert is_te_min_version("0.12.0"), \
        "transformer engine version: %s (>=0.12.0 required)." % get_te_version()

    # Search in directory above this
    sys.path.append(os.path.abspath(
        os.path.join(os.path.dirname(__file__),
                     os.path.pardir,
                     os.path.pardir)))
    if args.megatron_path is not None:
        sys.path.insert(0, args.megatron_path)

    try:
        from megatron.training.arguments import (parse_args, validate_args)
        from megatron.training.checkpointing import save_checkpoint
        from megatron.training.global_vars import set_global_variables, get_args
        from megatron.core.enums import ModelType
        from megatron.training.tokenizer.tokenizer import _vocab_size_with_padding
        from megatron.legacy import fused_kernels
        from megatron.core import mpu
    except ModuleNotFoundError:
        print("Unable to import Megatron, please specify the path to Megatron using --megatron-path. Exiting.")
        exit(1)

    def queue_get(name=None):
        val = queue.get()
        if val == "exit":
            print("Loader exited, exiting saver")
            exit(1)
        if name is not None and args.checking and val["name"] != name:
            val_name = val["name"]
            print(f'Unexpected message. Expecting "{name}" but got "{val_name}". Exiting saver.')
            exit(1)
        if name is not None:
            print(f"received {name}")
        return val

    def check_message(msg):
        if not args.checking:
            return
        msg_name = msg.pop("name")
        if len(msg.keys()) > 0:
            print(f"Unexpected values in {msg_name}:")
            for key in msg.keys():
                print(f"   {key}")
            print(f"Exiting. If you want to ignore this, use the argument --no-checking.")
            exit(1)


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
        if args.target_expert_parallel_size is not None:
            os.environ["WORLD_SIZE"] = f'{args.target_tensor_parallel_size * args.target_pipeline_parallel_size * args.target_expert_parallel_size}'
        else:
            os.environ["WORLD_SIZE"] = f'{args.target_tensor_parallel_size * args.target_pipeline_parallel_size}'

    # We want all arguments to come from us
    sys.argv = ['script.py',
                '--num-layers', str(md.num_layers),
                '--hidden-size', str(md.hidden_size),
                '--seq-length', str(md.seq_length),
                '--num-experts', str(getattr(md, "num_experts", 0)),
                '--num-attention-heads', str(md.num_attention_heads),
                '--max-position-embeddings', str(md.max_position_embeddings),
                '--position-embedding-type', str(md.position_embedding_type),
                '--tokenizer-type', str(md.tokenizer_type),
                '--tensor-model-parallel-size', str(args.target_tensor_parallel_size),
                '--pipeline-model-parallel-size', str(args.target_pipeline_parallel_size),
                '--expert-model-parallel-size', str(args.target_expert_parallel_size),
                '--no-masked-softmax-fusion',
                '--no-bias-gelu-fusion',
                '--no-bias-dropout-fusion',
                '--no-async-tensor-model-parallel-allreduce',
                '--use-cpu-initialization',
                '--micro-batch-size', '1',
                '--no-load-optim',
                '--no-load-rng',
                '--no-save-optim',
                '--no-save-rng',
                '--no-initialization',
                '--save-interval', '1',
                '--save', args.save_dir,
                '--ckpt-format', 'torch', # only 'torch' supported for conversion
                ]

    if md.make_vocab_size_divisible_by is not None:
        sys.argv.extend(['--make-vocab-size-divisible-by', str(md.make_vocab_size_divisible_by)])
    if md.params_dtype == torch.float16:
        sys.argv.append('--fp16')
    elif md.params_dtype == torch.bfloat16:
        sys.argv.append('--bf16')

    if md.output_layer:
        sys.argv.append('--untie-embeddings-and-output-weights')
    if not md.linear_bias:
        sys.argv.append('--disable-bias-linear')

    if md.model_type == 'BERT' and not md.bert_binary_head:
        sys.argv.append('--bert-no-binary-head')

    margs = parse_args()

    if hasattr (md, 'checkpoint_args'):
        # These are arguments that we are either changing, or cause problems for validation if they are set
        # Note that some of these deal with T5 so will need to be changed if we support T5.
        args_to_keep = ['tensor_model_parallel_size', 'pipeline_model_parallel_size', 'expert_model_parallel_size', 'world_size', 'params_dtype',
                        'num_layers_per_virtual_pipeline_stage', 'virtual_pipeline_model_parallel_size',
                        'masked_softmax_fusion', 'bias_gelu_fusion', 'bias_dropout_fusion',
                        'sequence_parallel', 'async_tensor_model_parallel_allreduce',
                        'no_load_optim', 'no_load_rng', 'no_save_optim', 'no_save_rng',
                        'vocab_file', 'tokenizer_model',
                        'save_interval', 'save',
                        'perform_initialization', 'use_cpu_initialization',
                        'recompute_granularity', 'recompute_num_layers', 'recompute_method',
                        'encoder_num_layers', 'encoder_seq_length',
                        'distribute_saved_activations',
                        'train_iters', 'lr_decay_iters', 'lr_warmup_iters', 'lr_warmup_fraction',
                        'start_weight_decay', 'end_weight_decay',
                        'ckpt_format',
        ]

        for arg, value in vars(md.checkpoint_args).items():
            if arg in args_to_keep:
                continue
            if not hasattr(margs, arg):
                print(f"Checkpoint had argument {arg} but new arguments does not have this.")
                continue
            if getattr(margs, arg) != value:
                print(f"Overwriting default {arg} value {getattr(margs, arg)} with value from checkpoint {value}.")
                setattr(margs, arg, value)

    # Explicitly copy sequence_parallel, apply_query_key_layer_scaling.
    margs.sequence_parallel = md.checkpoint_args.sequence_parallel
    margs.apply_query_key_layer_scaling = md.checkpoint_args.apply_query_key_layer_scaling

    # Sequence parallel is required if use both tensor-parallel and Moe.
    if margs.num_experts is not None and args.target_tensor_parallel_size is not None:
        if margs.num_experts > 1 and args.target_tensor_parallel_size > 1:
            margs.sequence_parallel = True

    validate_args(margs)

    # Use M-core models & unset loaded paths.
    margs.use_legacy_models = False
    margs.blendable_index_path = None
    margs.data_path = []
    margs.load = None
    margs.save = args.save_dir
    margs.tensorboard_dir = None
    margs.tokenizer_model = None
    margs.transformer_impl = args.saver_transformer_impl

    set_global_variables(margs, build_tokenizer=False)

    # Megatron args. (i.e., 'margs')
    margs = get_args()

    if hasattr(md, 'consumed_train_samples'):
        margs.consumed_train_samples = md.consumed_train_samples
        margs.consumed_valid_samples = md.consumed_valid_samples
        print(f"Setting consumed_train_samples to {margs.consumed_train_samples}"
              f" and consumed_valid_samples to {margs.consumed_valid_samples}")
    else:
        print("consumed_train_samples not provided.")

    # Determine how to make our models
    if md.model_type == 'GPT':
        from pretrain_gpt import model_provider
        margs.model_type = ModelType.encoder_or_decoder
    elif md.model_type == 'BERT':
        from pretrain_bert import model_provider
        margs.model_type = ModelType.encoder_or_decoder
    else:
        raise Exception(f'unrecognized model type: {args.model_type}')

    # fake initializing distributed
    mpu.set_tensor_model_parallel_world_size(args.target_tensor_parallel_size)
    mpu.set_pipeline_model_parallel_world_size(args.target_pipeline_parallel_size)
    mpu.set_expert_model_parallel_world_size(args.target_expert_parallel_size)
    mpu.set_tensor_model_parallel_rank(0)
    mpu.set_pipeline_model_parallel_rank(0)
    mpu.set_expert_model_parallel_rank(0)
    fused_kernels.load(margs)

    # Embeddings
    #-----------
    embeddings_msg = queue_get("embeddings")

    pos_embed = None
    if md.position_embedding_type == 'learned_absolute':
        pos_embed = embeddings_msg.pop("position embeddings")
    orig_word_embed = embeddings_msg.pop("word embeddings")
    check_message(embeddings_msg)

    # Deal with padding
    def pad_weight(orig_word_embed, true_vocab_size):
        if true_vocab_size is not None:
            # figure out what our padded vocab size is
            orig_vocab_size = orig_word_embed.shape[0]
            margs.padded_vocab_size = _vocab_size_with_padding(true_vocab_size, margs)

            # Cut out extra padding we don't need
            if orig_vocab_size > margs.padded_vocab_size:
                full_word_embed = orig_word_embed[0:margs.padded_vocab_size,:]

            # Expanding embedding to larger size by replicating final entry
            elif orig_vocab_size < margs.padded_vocab_size:
                padding_size = margs.padded_vocab_size - orig_vocab_size

                full_word_embed = torch.cat((
                    orig_word_embed,
                    orig_word_embed[-1].unsqueeze(0).expand(padding_size, -1)))

            # Same size!
            else:
                full_word_embed = orig_word_embed
        else:
            print("Original vocab size not specified, leaving embedding table as-is. "
                "If you've changed the tensor parallel size this could cause problems.")
            margs.padded_vocab_size = orig_word_embed.shape[0]
            full_word_embed = orig_word_embed
        return full_word_embed

    full_word_embed = pad_weight(orig_word_embed, md.true_vocab_size)

    # Split into new tensor model parallel sizes
    out_word_embed = torch.chunk(full_word_embed, args.target_tensor_parallel_size, dim=0)

    # Parameter setter class.
    setter = get_model_setter(md.model_type, margs.transformer_impl, margs.num_experts)

    # Construct a 3D(PPxEPxTP) arry for models, fill it with None
    models = [[[None for _ in range(args.target_tensor_parallel_size)] for _ in range(args.target_expert_parallel_size)] for _ in range(args.target_pipeline_parallel_size)]

    # Model is lazy instantiated at firstly using
    def get_local_model(pp_rank, ep_rank, tp_rank):
        if models[pp_rank][ep_rank][tp_rank] is None:
            pre_process = True if pp_rank == 0 else False
            post_process = True if pp_rank == args.target_pipeline_parallel_size - 1 else False
            models[pp_rank][ep_rank][tp_rank] = model_provider(pre_process, post_process).to(md.params_dtype)
        return models[pp_rank][ep_rank][tp_rank]

    # Set embeddings.
    # --------------
    for ep_rank in range(args.target_expert_parallel_size):
        for tp_rank in range(args.target_tensor_parallel_size):
            model = get_local_model(0, ep_rank, tp_rank)
            if pos_embed is None:
                assert not setter.has_position_embeddings(model)
            setter.set_embeddings(
                model,
                word=out_word_embed[tp_rank],
                pos=pos_embed,
            )

    def chunk_weight(weight, parallel_mode, tp_size=1, ep_size=1):
        assert parallel_mode in ["row", "column"]
        if weight.dim() == 3:
            num_experts, out_features, in_features = weight.shape
            if parallel_mode == "column":
                weight = weight.reshape(ep_size, num_experts // ep_size, tp_size, out_features // tp_size, in_features)
                weight = weight.permute(0, 2, 1, 3, 4)
            else:
                weight = weight.reshape(ep_size, num_experts // ep_size, out_features, tp_size, in_features // tp_size)
                weight = weight.permute(0, 3, 1, 2, 4)
            return weight # (ep_size, tp_size, local_eps, output_features, in_features)
        else:
            out_features, in_features = weight.shape
            if parallel_mode == "column":
                weight = weight.reshape(tp_size, out_features // tp_size, in_features)
            else:
                weight = weight.reshape(out_features, tp_size, in_features // tp_size).permute(1, 0, 2)
            return weight # (tp_size, output_features, in_features)

    def chunk_bias(bias, parallel_mode, tp_size=1, ep_size=1):
        assert parallel_mode in ["row", "column"]
        if bias.dim() == 2:
            num_experts, hidden_size = bias.shape
            if parallel_mode == 'column':
                bias = bias.reshape(ep_size, num_experts // ep_size, tp_size, hidden_size // tp_size)
                bias = bias.permute(0, 2, 1, 3) # (ep_size, tp_size, local_eps, hidden_size)
            else:
                bias = bias.reshape(ep_size, num_experts // ep_size, hidden_size) # (ep_size, local_eps, hidden_size)
            return bias
        else:
            hidden_size = bias.shape
            if parallel_mode == "column":
                bias = bias.reshape(tp_size, hidden_size[0] // tp_size) # (tp_size, hidden_size)
            return bias

    # Transformer layers.
    # ------------------
    total_layer_num = 0
    for pp_rank in range(args.target_pipeline_parallel_size):
        # initial the first module in pp stage to get the layer_num, pooler, lm_head. binary_head
        get_local_model(pp_rank,0,0)
        for layer_id in range(len(setter.get_transformer_block(models[pp_rank][0][0]).layers)):
            msg = queue_get(f"transformer layer {total_layer_num}")

            # duplicated tensors
            input_norm_weight = msg.pop("input norm weight")
            post_norm_weight = msg.pop("post norm weight")
            if md.norm_has_bias:
                input_norm_bias = msg.pop("input norm bias")
                post_norm_bias = msg.pop("post norm bias")

            # Split up the parallel tensors
            qkv_weight = chunk_weight(msg.pop("qkv weight"), "column", args.target_tensor_parallel_size)
            dense_weight = chunk_weight(msg.pop("dense weight"), "row", args.target_tensor_parallel_size)
            mlp_l1_weight = chunk_weight(msg.pop("mlp l1 weight"), "row", args.target_tensor_parallel_size, args.target_expert_parallel_size)

            if margs.num_experts:
                router = msg.pop("router weight")

            # Special handling for swiglu
            if md.swiglu:
                mlp_l0_weight_W = chunk_weight(msg.pop("mlp l0 weight W"), "column", args.target_tensor_parallel_size, args.target_expert_parallel_size)
                mlp_l0_weight_V = chunk_weight(msg.pop("mlp l0 weight V"), "column", args.target_tensor_parallel_size, args.target_expert_parallel_size)
                mlp_l0_weight = torch.cat((mlp_l0_weight_W, mlp_l0_weight_V), dim=-2)
            else:
                mlp_l0_weight = chunk_weight(msg.pop("mlp l0 weight"), "column", args.target_tensor_parallel_size, args.target_expert_parallel_size)

            if md.linear_bias:
                dense_bias = msg.pop("dense bias")
                mlp_l1_bias = chunk_bias(msg.pop("mlp l1 bias"), 'row', args.target_tensor_parallel_size, args.target_expert_parallel_size)
                qkv_bias = chunk_bias(msg.pop("qkv bias"), 'column', args.target_tensor_parallel_size)
                if md.swiglu:
                    mlp_l0_bias_W = chunk_bias(msg.pop("mlp l0 bias W"), 'column', args.target_tensor_parallel_size, args.target_expert_parallel_size)
                    mlp_l0_bias_V = chunk_bias(msg.pop("mlp l0 bias V"), 'column', args.target_tensor_parallel_size, args.target_expert_parallel_size)
                    mlp_l0_bias = torch.cat((mlp_l0_bias_W, mlp_l0_bias_V), dim=-1)
                else:
                    mlp_l0_bias = chunk_bias(msg.pop("mlp l0 bias"), 'column', args.target_tensor_parallel_size, args.target_expert_parallel_size)

            # Save them to the model
            for ep_rank in range(args.target_expert_parallel_size):
                for tp_rank in range(args.target_tensor_parallel_size):
                    params_dict = {
                        "self_attn_norm_weight" : input_norm_weight,
                        "self_attn_qkv_weight" : qkv_weight[tp_rank],
                        "self_attn_proj_weight" : dense_weight[tp_rank],
                        "mlp_norm_weight" : post_norm_weight
                    }
                    if margs.num_experts:
                        params_dict.update({
                            "mlp_fc1_weight" : mlp_l0_weight[ep_rank][tp_rank],
                            "mlp_fc2_weight" : mlp_l1_weight[ep_rank][tp_rank]
                        })
                    else:
                        params_dict.update({
                            "mlp_fc1_weight" : mlp_l0_weight[tp_rank],
                            "mlp_fc2_weight" : mlp_l1_weight[tp_rank]
                        })
                    params_dict.update({
                        "self_attn_norm_bias" : input_norm_bias if md.norm_has_bias else None,
                        "mlp_norm_bias" : post_norm_bias if md.norm_has_bias else None,
                    })
                    if md.linear_bias:
                        params_dict.update({
                            "self_attn_qkv_bias" : qkv_bias[tp_rank],
                            "self_attn_proj_bias" : dense_bias
                        })
                        if margs.num_experts:
                            params_dict.update({
                                "mlp_fc1_bias" : mlp_l0_bias[ep_rank][tp_rank],
                                "mlp_fc2_bias" : mlp_l1_bias[ep_rank]
                            })
                        else :
                            params_dict.update({
                                "mlp_fc1_bias" : mlp_l0_bias[tp_rank],
                                "mlp_fc2_bias" : mlp_l1_bias
                            })
                    if margs.num_experts:
                        params_dict.update({
                            "router_weight":  router
                        })
                    model = get_local_model(pp_rank, ep_rank, tp_rank)
                    setter.set_layer(model, layer_id, **params_dict)

            total_layer_num = total_layer_num + 1
            check_message(msg)


        if pp_rank == args.target_pipeline_parallel_size - 1:
            msg = queue_get("final norm")
            final_norm_weight = msg.pop("weight")
            if md.norm_has_bias:
                final_norm_bias = msg.pop("bias")
            pp_local_models = [get_local_model(pp_rank, ep_rank, tp_rank) for ep_rank in range(args.target_expert_parallel_size)
                for tp_rank in range(args.target_tensor_parallel_size)]
            for eptp_rank, model in enumerate(pp_local_models):
                tp_rank = eptp_rank % args.target_tensor_parallel_size
                setter.set_final_norm(
                    model,
                    weight=final_norm_weight,
                    bias=final_norm_bias if md.norm_has_bias else None,
                )
                if pp_rank != 0 and not md.output_layer:
                    # Copy word embeddings to final pipeline rank
                    setter.set_output_word_embeddings(
                        model,
                        emb=out_word_embed[tp_rank],
                    )
            del final_norm_weight
            if md.norm_has_bias:
                del final_norm_bias
            check_message(msg)

            if md.output_layer:
                msg = queue_get("output layer")
                if not hasattr(pp_local_models[0], 'output_layer'):
                    print("ERROR: got an output layer, but model does not have one")
                    exit(1)
                output_layer_weight = pad_weight(msg.pop("weight"), md.true_vocab_size)
                output_layer_weight = torch.chunk(output_layer_weight, args.target_tensor_parallel_size, dim=0)
                for eptp_rank, model in enumerate(pp_local_models):
                    tp_rank = eptp_rank % args.target_tensor_parallel_size
                    setter.set_output_layer(model, output_layer_weight[tp_rank])
                check_message(msg)

            msg = queue_get()
            if msg != "done" and msg["name"] == "pooler":
                if not hasattr(models[pp_rank][0][0], 'pooler'):
                    print("ERROR: got a pooler, but model does not have one")
                    exit(1)
                print("received pooler")
                pooler_weight = msg.pop("weight")
                pooler_bias = msg.pop("bias")
                for model in pp_local_models:
                    setter.set_pooler(
                        model=model,
                        weight=pooler_weight,
                        bias=pooler_bias,
                    )
                del pooler_weight
                del pooler_bias
                check_message(msg)
                msg = queue_get()

            if msg != "done" and msg["name"] == "lm head":
                if not hasattr(models[pp_rank][0][0], 'lm_head'):
                    print("ERROR: got an lm head, but model does not have one")
                    exit(1)
                print("received lm head")
                lm_head_dense_weight = msg.pop("dense weight")
                lm_head_dense_bias = msg.pop("dense bias")
                lm_head_norm_weight = msg.pop("norm weight")
                if md.norm_has_bias:
                    lm_head_norm_bias = msg.pop("norm bias")
                for model in pp_local_models:
                    setter.set_lm_head(
                        model=model,
                        dense_weight=lm_head_dense_weight,
                        dense_bias=lm_head_dense_bias,
                        norm_weight=lm_head_norm_weight,
                        norm_bias=lm_head_norm_bias if md.norm_has_bias else None,
                    )
                check_message(msg)
                msg = queue_get()

            if msg != "done" and msg["name"] == "binary head":
                if not hasattr(models[pp_rank][0][0], 'binary_head'):
                    print("ERROR: got a binary head, but model does not have one")
                    exit(1)
                print("received binary head")
                binary_head_weight = msg.pop("weight")
                binary_head_bias = msg.pop("bias")
                for model in pp_local_models:
                    setter.set_binary_head(
                        model=model,
                        weight=binary_head_weight,
                        bias=binary_head_bias,
                    )
                check_message(msg)
                msg = queue_get()

            # TODO: delete weight when not used
            if msg != "done":
                print("ERROR: got some more data but was expecting to be done")

        for ep_rank in range(args.target_expert_parallel_size):
            for tp_rank in range(args.target_tensor_parallel_size):
                save_checkpoint(md.iteration, [get_local_model(pp_rank, ep_rank, tp_rank)], None, None, num_floating_point_operations_so_far=0,
                    pipeline_rank=pp_rank, pipeline_parallel=args.target_pipeline_parallel_size > 1,
                    expert_rank=ep_rank, expert_parallel=args.target_expert_parallel_size > 1,
                    tensor_rank=tp_rank)
                # release the uselese model parts
                models[pp_rank][ep_rank][tp_rank] = None

    print("Done!")
