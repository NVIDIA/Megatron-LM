# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
import json
import os
from importlib.metadata import version
from packaging.version import Version as PkgVersion
import sys
import torch

from utils import chunk_bias, chunk_weight

class MegatronCheckpointSaverBase:
    """Orchestrates saving a Megatron checkpoint using parameters received on a multiprocessing queue.

    Args:
        args: argparse Namespace with Megatron checkpoint configurations.
        queue: A multiprocessing.Queue (or similar) used to send out loaded tensors.
        build_tokenizer: Whether to build a tokenizer for the model to be saved
    """

    def __init__(self, args, queue, build_tokenizer=False):
        self.args = args
        self.queue = queue
        self.build_tokenizer = build_tokenizer

        self.margs = None            # Will hold Megatron's main args
        self.md = None               # Metadata received from the loader

        self.models = None
        self.model_provider = None   # model_provider function either from pretrain_gpt or pretrain_bert

    def _maybe_parse_additional_megatron_args(self, margs):
        """
        Method used to optionally add arguments from the checkpoint to the main args.
        For instance, using margs.some_arg = checkpoint_args.some_arg
        """
        return margs

    def insert_megatron_path_and_check_te(self):
        """
        Check for an appropriate installation of transformer engine and add megatron to sys path.
        """
        # Transformer engine >= 0.12.0, for CPU initialization.
        te_version = PkgVersion(version("transformer-engine"))
        assert te_version >= PkgVersion("0.12.0"), \
            "transformer engine version: %s (>=0.12.0 required)." % te_version

        # Search in directory above this
        sys.path.append(os.path.abspath(
            os.path.join(os.path.dirname(__file__),
                         os.path.pardir,
                         os.path.pardir)))
        if self.args.megatron_path is not None:
            sys.path.insert(0, self.args.megatron_path)

    def _load_checkpoint_args(self, margs):
        """
        Load arguments from checkpoint to margs.
        """
        if hasattr(self.md, 'checkpoint_args'):
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

            for arg, value in vars(self.md.checkpoint_args).items():
                if arg in args_to_keep:
                    continue
                if not hasattr(margs, arg):
                    print(f"Checkpoint had argument {arg} but new arguments does not have this.")
                    continue
                if getattr(margs, arg) != value:
                    print(f"Overwriting default {arg} value {getattr(margs, arg)} with value from checkpoint {value}.")
                    setattr(margs, arg, value)

        return margs

    def parse_megatron_args(self):
        """
        Parse Megatron arguments by forcibly overwriting sys.argv.
        Populates self.margs and self.checkpoint_args.
        """
        try:
            from megatron.training.arguments import parse_args, validate_args
        except ModuleNotFoundError:
            print("Unable to import Megatron. Please specify --megatron-path. Exiting.")
            sys.exit(1)

        sys.argv = self.build_sys_argv()

        margs = parse_args()
        margs = self._load_checkpoint_args(margs)

        margs.inference_batch_times_seqlen_threshold = -1

        # Explicitly copy sequence_parallel, apply_query_key_layer_scaling.
        margs.sequence_parallel = self.md.checkpoint_args.sequence_parallel
        margs.apply_query_key_layer_scaling = self.md.checkpoint_args.apply_query_key_layer_scaling

        # Sequence parallel is required if use both tensor-parallel and Moe.
        if margs.num_experts is not None and self.args.target_tensor_parallel_size is not None:
            if margs.num_experts > 1 and self.args.target_tensor_parallel_size > 1:
                margs.sequence_parallel = True

        margs = self._maybe_parse_additional_megatron_args(margs)

        validate_args(margs)

        # Use M-core models & unset loaded paths.
        margs.use_legacy_models = False
        margs.blendable_index_path = None
        margs.data_path = []
        margs.load = None
        margs.save = self.args.save_dir
        margs.tensorboard_dir = None
        if not self.build_tokenizer:
            margs.tokenizer_model = None
        margs.transformer_impl = self.args.saver_transformer_impl
        if self.args.saver_transformer_impl == "local" and margs.normalization == "RMSNorm":
            margs.no_persist_layer_norm = True

        self.margs = margs

    def initialize_megatron_env(self):
        """
        Initialize Megatron global variables and fused kernels.
        """
        try:
            from megatron.training.global_vars import set_global_variables, get_args
            from megatron.core import mpu
            from megatron.legacy import fused_kernels
        except ModuleNotFoundError as e:
            print(f"Unable to import required Megatron modules: {e}")
            sys.exit(1)

        set_global_variables(self.margs, build_tokenizer=self.build_tokenizer)

        # Megatron args. (i.e., 'margs')
        self.margs = get_args()

        if hasattr(self.md, 'consumed_train_samples'):
            self.margs.consumed_train_samples = self.md.consumed_train_samples
            self.margs.consumed_valid_samples = self.md.consumed_valid_samples
            print(f"Setting consumed_train_samples to {self.margs.consumed_train_samples}"
                  f" and consumed_valid_samples to {self.margs.consumed_valid_samples}")
        else:
            print("consumed_train_samples not provided.")

        self.import_model_provider()

        # fake initializing distributed
        mpu.set_tensor_model_parallel_world_size(self.args.target_tensor_parallel_size)
        mpu.set_pipeline_model_parallel_world_size(self.args.target_pipeline_parallel_size)
        mpu.set_expert_model_parallel_world_size(self.args.target_expert_parallel_size)
        mpu.set_tensor_model_parallel_rank(0)
        mpu.set_pipeline_model_parallel_rank(0)
        mpu.set_expert_model_parallel_rank(0)
        fused_kernels.load(self.margs)

    def queue_get(self, name=None):
        """
        Receive a message over the multiprocessing queue.
        """
        val = self.queue.get()
        if val == "exit":
            print("Loader exited, exiting saver")
            exit(1)
        if name is not None and self.args.checking and val["name"] != name:
            val_name = val["name"]
            print(f'Unexpected message. Expecting "{name}" but got "{val_name}". Exiting saver.')
            exit(1)
        if name is not None:
            print(f"received {name}")
        return val

    def check_message(self, msg):
        """
        Check that a field exists on queue message if necessary.
        """
        if not self.args.checking:
            return
        msg_name = msg.pop("name")
        if len(msg.keys()) > 0:
            print(f"Unexpected values in {msg_name}:")
            for key in msg.keys():
                print(f"   {key}")
            print(f"Exiting. If you want to ignore this, use the argument --no-checking.")
            exit(1)

    def build_sys_argv(self): 
        """
        Construct a sys.argv list for Megatron's argument parser.
        This centralizes the hack of overwriting sys.argv.
        """
        # We want all arguments to come from us
        my_argv = ['script.py',
                    '--num-layers', str(self.md.num_layers),
                    '--hidden-size', str(self.md.hidden_size),
                    '--seq-length', str(self.md.seq_length),
                    '--num-experts', str(getattr(self.md, "num_experts", 0)),
                    '--num-attention-heads', str(self.md.num_attention_heads),
                    '--max-position-embeddings', str(self.md.max_position_embeddings),
                    '--position-embedding-type', str(self.md.position_embedding_type),
                    '--tokenizer-type', str(self.md.tokenizer_type),
                    '--tensor-model-parallel-size', str(self.args.target_tensor_parallel_size),
                    '--pipeline-model-parallel-size', str(self.args.target_pipeline_parallel_size),
                    '--expert-model-parallel-size', str(self.args.target_expert_parallel_size),
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
                    '--save', self.args.save_dir,
                    '--ckpt-format', 'torch', # only 'torch' supported for conversion
                    '--no-one-logger',
                    ]

        if self.md.make_vocab_size_divisible_by is not None:
            my_argv.extend(['--make-vocab-size-divisible-by', str(self.md.make_vocab_size_divisible_by)])
        if self.md.params_dtype == torch.float16:
            my_argv.append('--fp16')
        elif self.md.params_dtype == torch.bfloat16:
            my_argv.append('--bf16')
        if self.md.output_layer:
            my_argv.append('--untie-embeddings-and-output-weights')
        if not self.md.linear_bias:
            my_argv.append('--disable-bias-linear')

        if self.md.model_type == 'BERT' and not self.md.bert_binary_head:
            my_argv.append('--bert-no-binary-head')

        return my_argv

    def receive_checkpoint_metadata(self):
        """
        Receive and populate model metadata.
        """
        self.md = self.queue_get()

        if self.args.target_tensor_parallel_size is None:
            if hasattr(self.md, 'previous_tensor_parallel_size'):
                self.args.target_tensor_parallel_size = self.md.previous_tensor_parallel_size
            else:
                print("loader did not provide a tensor parallel size and --target-tensor-parallel-size not provided on command line. "
                      "Default to 1.")
                self.args.target_tensor_parallel_size = 1

        if self.args.target_pipeline_parallel_size is None:
            if hasattr(self.md, 'previous_pipeline_parallel_size'):
                self.args.target_pipeline_parallel_size = self.md.previous_pipeline_parallel_size
            else:
                print("loader did not provide a pipeline parallel size and --target-pipeline-parallel-size not provided on command line. "
                      "Default to 1.")
                self.args.target_pipeline_parallel_size = 1

        # Arguments do sanity checks on the world size, but we don't care,
        # so trick it into thinking we are plenty of processes
        if self.args.target_tensor_parallel_size is not None and self.args.target_pipeline_parallel_size is not None:
            if self.args.target_expert_parallel_size is not None:
                os.environ["WORLD_SIZE"] = f'{self.args.target_tensor_parallel_size * self.args.target_pipeline_parallel_size * self.args.target_expert_parallel_size}'
            else:
                os.environ["WORLD_SIZE"] = f'{self.args.target_tensor_parallel_size * self.args.target_pipeline_parallel_size}'

    def initialize_models(self):
        """Construct a 3D(PPxEPxTP) array for models, fill it with None"""
        return [[[None for _ in range(self.args.target_tensor_parallel_size)] for _ in range(self.args.target_expert_parallel_size)] for _ in range(self.args.target_pipeline_parallel_size)]

    def get_local_model(self, pp_rank, ep_rank, tp_rank):
        """
        Get the local model for a certain (pp,ep,tp).
        """
        if self.models[pp_rank][ep_rank][tp_rank] is None:
            pre_process = True if pp_rank == 0 else False
            post_process = True if pp_rank == self.args.target_pipeline_parallel_size - 1 else False
            self.models[pp_rank][ep_rank][tp_rank] = self.model_provider(pre_process, post_process).to(self.md.params_dtype)
        return self.models[pp_rank][ep_rank][tp_rank]

    def save(self):
        """
        Orchestrate the entire flow of saving the Megatron checkpoint.
        """
        self.insert_megatron_path_and_check_te()

        self.receive_checkpoint_metadata()

        self.parse_megatron_args()

        self.initialize_megatron_env()

        self.models = self.initialize_models()

        self.receive_model()

        self.save_local_models_to_checkpoint()

        print("Done!")

    def save_local_models_to_checkpoint(self):
        """
        Save local models in self.models to a megatron checkpoint.
        """
        try:
            from megatron.training.checkpointing import save_checkpoint
            from megatron.core import mpu
        except ModuleNotFoundError as e:
            print(f"Unable to import required Megatron modules: {e}")
            sys.exit(1)

        for pp_rank in range(self.args.target_pipeline_parallel_size):
            mpu.set_pipeline_model_parallel_rank(pp_rank)
            # initial the first module in pp stage to get the layer_num, pooler, lm_head. binary_head
            self.get_local_model(pp_rank,0,0)
            for ep_rank in range(self.args.target_expert_parallel_size):
                for tp_rank in range(self.args.target_tensor_parallel_size):
                    save_checkpoint(self.md.iteration, [self.get_local_model(pp_rank, ep_rank, tp_rank)], None, None, num_floating_point_operations_so_far=0,
                        pipeline_rank=pp_rank, pipeline_parallel=self.args.target_pipeline_parallel_size > 1,
                        expert_rank=ep_rank, expert_parallel=self.args.target_expert_parallel_size > 1,
                        tensor_rank=tp_rank)
                    # release the uselese model parts
                    self.models[pp_rank][ep_rank][tp_rank] = None

    def receive_lm(self, schema, prefix=None):
        """
        Receive LM model parameters over queue and save them in self.models
        """
        try:
            from megatron.core import mpu
            from megatron.training.tokenizer.tokenizer import _vocab_size_with_padding
        except ModuleNotFoundError as e:
            print(f"Unable to import required Megatron modules: {e}")
            sys.exit(1)

        # Embeddings
        #-----------
        embeddings_msg = self.queue_get("embeddings")
        pos_embed = None
        if self.md.position_embedding_type == 'learned_absolute':
            pos_embed = embeddings_msg.pop("position embeddings")
        orig_word_embed = embeddings_msg.pop("word embeddings")
        self.check_message(embeddings_msg)

        # Deal with padding
        def pad_weight(orig_word_embed, true_vocab_size):
            if true_vocab_size is not None:
                # figure out what our padded vocab size is
                orig_vocab_size = orig_word_embed.shape[0]
                self.margs.padded_vocab_size = _vocab_size_with_padding(true_vocab_size, self.margs)

                # Cut out extra padding we don't need
                if orig_vocab_size > self.margs.padded_vocab_size:
                    full_word_embed = orig_word_embed[0:self.margs.padded_vocab_size,:]

                # Expanding embedding to larger size by replicating final entry
                elif orig_vocab_size < self.margs.padded_vocab_size:
                    padding_size = self.margs.padded_vocab_size - orig_vocab_size

                    full_word_embed = torch.cat((
                        orig_word_embed,
                        orig_word_embed[-1].unsqueeze(0).expand(padding_size, -1)))

                # Same size!
                else:
                    full_word_embed = orig_word_embed
            else:
                print("Original vocab size not specified, leaving embedding table as-is. "
                    "If you've changed the tensor parallel size this could cause problems.")
                self.margs.padded_vocab_size = orig_word_embed.shape[0]
                full_word_embed = orig_word_embed
            return full_word_embed

        full_word_embed = pad_weight(orig_word_embed, self.md.true_vocab_size)

        # Split into new tensor model parallel sizes
        out_word_embed = torch.chunk(full_word_embed, self.args.target_tensor_parallel_size, dim=0)

        # Set embeddings.
        # --------------
        for ep_rank in range(self.args.target_expert_parallel_size):
            for tp_rank in range(self.args.target_tensor_parallel_size):
                model = self.get_local_model(0, ep_rank, tp_rank)
                if pos_embed is None:
                    assert not schema.has_position_embeddings(model)
                schema.set("embeddings", model, {
                    "pos" : pos_embed,
                    "word" : out_word_embed[tp_rank],
                })

        # Transformer layers.
        # ------------------
        total_layer_num = 0
        for pp_rank in range(self.args.target_pipeline_parallel_size):
            mpu.set_pipeline_model_parallel_rank(pp_rank)
            # initial the first module in pp stage to get the layer_num, pooler, lm_head. binary_head
            self.get_local_model(pp_rank,0,0)
            for layer_id in range(schema.get_num_layers(self.models[pp_rank][0][0])):
                msg = self.queue_get(f"transformer layer {total_layer_num}")

                # duplicated tensors
                input_norm_weight = msg.pop("input norm weight")
                post_norm_weight = msg.pop("post norm weight")
                if self.md.norm_has_bias:
                    input_norm_bias = msg.pop("input norm bias")
                    post_norm_bias = msg.pop("post norm bias")

                # Split up the parallel tensors
                qkv_weight = chunk_weight(msg.pop("qkv weight"), "column", self.args.target_tensor_parallel_size)
                dense_weight = chunk_weight(msg.pop("dense weight"), "row", self.args.target_tensor_parallel_size)
                mlp_l1_weight = chunk_weight(msg.pop("mlp l1 weight"), "row", self.args.target_tensor_parallel_size, self.args.target_expert_parallel_size)

                if self.margs.num_experts:
                    router = msg.pop("router weight")

                # Special handling for swiglu
                if self.md.swiglu:
                    mlp_l0_weight_W = chunk_weight(msg.pop("mlp l0 weight W"), "column", self.args.target_tensor_parallel_size, self.args.target_expert_parallel_size)
                    mlp_l0_weight_V = chunk_weight(msg.pop("mlp l0 weight V"), "column", self.args.target_tensor_parallel_size, self.args.target_expert_parallel_size)
                    mlp_l0_weight = torch.cat((mlp_l0_weight_W, mlp_l0_weight_V), dim=-2)
                else:
                    mlp_l0_weight = chunk_weight(msg.pop("mlp l0 weight"), "column", self.args.target_tensor_parallel_size, self.args.target_expert_parallel_size)

                if self.md.qkv_bias:
                    qkv_bias = chunk_bias(msg.pop("qkv bias"), 'column', self.args.target_tensor_parallel_size)
                if self.md.linear_bias:
                    dense_bias = msg.pop("dense bias")
                    mlp_l1_bias = chunk_bias(msg.pop("mlp l1 bias"), 'row', self.args.target_tensor_parallel_size, self.args.target_expert_parallel_size)
                    if self.md.swiglu:
                        mlp_l0_bias_W = chunk_bias(msg.pop("mlp l0 bias W"), 'column', self.args.target_tensor_parallel_size, self.args.target_expert_parallel_size)
                        mlp_l0_bias_V = chunk_bias(msg.pop("mlp l0 bias V"), 'column', self.args.target_tensor_parallel_size, self.args.target_expert_parallel_size)
                        mlp_l0_bias = torch.cat((mlp_l0_bias_W, mlp_l0_bias_V), dim=-1)
                    else:
                        mlp_l0_bias = chunk_bias(msg.pop("mlp l0 bias"), 'column', self.args.target_tensor_parallel_size, self.args.target_expert_parallel_size)

                # Save them to the model
                for ep_rank in range(self.args.target_expert_parallel_size):
                    for tp_rank in range(self.args.target_tensor_parallel_size):
                        params_dict = {
                            "self_attn_norm_weight" : input_norm_weight,
                            "self_attn_qkv_weight" : qkv_weight[tp_rank],
                            "self_attn_proj_weight" : dense_weight[tp_rank],
                            "mlp_norm_weight" : post_norm_weight
                        }
                        if self.margs.num_experts:
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
                            "self_attn_norm_bias" : input_norm_bias if self.md.norm_has_bias else None,
                            "mlp_norm_bias" : post_norm_bias if self.md.norm_has_bias else None,
                        })
                        if self.md.qkv_bias:
                            params_dict.update({
                                "self_attn_qkv_bias" : qkv_bias[tp_rank]
                            })
                        if self.md.linear_bias:
                            params_dict.update({
                                "self_attn_proj_bias" : dense_bias
                            })
                            if self.margs.num_experts:
                                params_dict.update({
                                    "mlp_fc1_bias" : mlp_l0_bias[ep_rank][tp_rank],
                                    "mlp_fc2_bias" : mlp_l1_bias[ep_rank]
                                })
                            else :
                                params_dict.update({
                                    "mlp_fc1_bias" : mlp_l0_bias[tp_rank],
                                    "mlp_fc2_bias" : mlp_l1_bias
                                })
                        if self.margs.num_experts:
                            params_dict.update({
                                "router_weight":  router
                            })
                        model = self.get_local_model(pp_rank, ep_rank, tp_rank)
                        schema.set_layer(model, layer_id, params_dict)

                total_layer_num = total_layer_num + 1
                self.check_message(msg)


            if pp_rank == self.args.target_pipeline_parallel_size - 1:
                msg = self.queue_get("final norm")
                final_norm_weight = msg.pop("weight")
                if self.md.norm_has_bias:
                    final_norm_bias = msg.pop("bias")
                pp_local_models = [self.get_local_model(pp_rank, ep_rank, tp_rank) for ep_rank in range(self.args.target_expert_parallel_size)
                    for tp_rank in range(self.args.target_tensor_parallel_size)]
                for eptp_rank, model in enumerate(pp_local_models):
                    tp_rank = eptp_rank % self.args.target_tensor_parallel_size
                    schema.set("final_norm", model, {
                        "weight" : final_norm_weight,
                        "bias" : final_norm_bias if self.md.norm_has_bias else None,
                    })
                    if pp_rank != 0 and not self.md.output_layer:
                        # Copy word embeddings to final pipeline rank
                        schema.set("output_layer", model, {
                            "weight" : out_word_embed[tp_rank],
                        })
                del final_norm_weight
                if self.md.norm_has_bias:
                    del final_norm_bias
                self.check_message(msg)

                if self.md.output_layer:
                    msg = self.queue_get("output layer")
                    if not hasattr(pp_local_models[0] if prefix is None else getattr(pp_local_models[0], prefix), 'output_layer'):
                        print("ERROR: got an output layer, but model does not have one")
                        exit(1)
                    output_layer_weight = pad_weight(msg.pop("weight"), self.md.true_vocab_size)
                    output_layer_weight = torch.chunk(output_layer_weight, self.args.target_tensor_parallel_size, dim=0)
                    for eptp_rank, model in enumerate(pp_local_models):
                        tp_rank = eptp_rank % self.args.target_tensor_parallel_size
                        schema.set("output_layer", model, {
                            "weight" : output_layer_weight[tp_rank],
                        })
                    self.check_message(msg)

                msg = self.queue_get()
                if msg != "done" and msg["name"] == "pooler":
                    if not hasattr(self.models[pp_rank][0][0] if prefix is None else getattr(self.models[pp_rank][0][0], prefix), 'pooler'):
                        print("ERROR: got a pooler, but model does not have one")
                        exit(1)
                    print("received pooler")
                    pooler_weight = msg.pop("weight")
                    pooler_bias = msg.pop("bias")
                    for model in pp_local_models:
                        schema.set("pooler", model, {
                            "weight" : pooler_weight,
                            "bias" : pooler_bias,
                        })
                    del pooler_weight
                    del pooler_bias
                    self.check_message(msg)
                    msg = self.queue_get()

                if msg != "done" and msg["name"] == "lm head":
                    if not hasattr(self.models[pp_rank][0][0] if prefix is None else getattr(self.models[pp_rank][0][0], prefix), 'lm_head'):
                        print("ERROR: got an lm head, but model does not have one")
                        exit(1)
                    print("received lm head")
                    lm_head_dense_weight = msg.pop("dense weight")
                    lm_head_dense_bias = msg.pop("dense bias")
                    lm_head_norm_weight = msg.pop("norm weight")
                    if self.md.norm_has_bias:
                        lm_head_norm_bias = msg.pop("norm bias")
                    for model in pp_local_models:
                        schema.set("lm_head", model, {
                            "dense_weight" : lm_head_dense_weight,
                            "dense_bias" : lm_head_dense_bias,
                            "norm_weight" : lm_head_norm_weight,
                            "norm_bias" : lm_head_norm_bias if self.md.norm_has_bias else None,
                        })
                    self.check_message(msg)
                    msg = self.queue_get()

                if msg != "done" and msg["name"] == "binary head":
                    if not hasattr(self.models[pp_rank][0][0] if prefix is None else getattr(self.models[pp_rank][0][0], prefix), 'binary_head'):
                        print("ERROR: got a binary head, but model does not have one")
                        exit(1)
                    print("received binary head")
                    binary_head_weight = msg.pop("weight")
                    binary_head_bias = msg.pop("bias")
                    for model in pp_local_models:
                        schema.set("binary_head", model, {
                            "weight" : binary_head_weight,
                            "bias" : binary_head_bias,
                        })
                    self.check_message(msg)
                    msg = self.queue_get()

                # TODO: delete weight when not used
                if msg != "done":
                    print("ERROR: got some more data but was expecting to be done")

    def import_model_provider(self):
        """Return the correct model_provider function."""
        raise NotImplementedError

    def receive_model(self):
        """Creates model scheme and receives model over the queue"""
        raise NotImplementedError
