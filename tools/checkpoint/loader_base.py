# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
import json
import os
import sys
import types
import torch

from utils import print_memory_usage

class MegatronCheckpointLoaderBase:
    """Orchestrates loading a Megatron checkpoint and sending
    model parameters over a given multiprocessing queue.

    Args:
        args: argparse Namespace with Megatron checkpoint configurations.
        queue: A multiprocessing.Queue (or similar) used to send out loaded tensors.
    """

    def __init__(self, args, queue, build_tokenizer=False):
        self.args = args
        self.queue = queue
        self.build_tokenizer = build_tokenizer
        self.margs = None            # Will hold Megatron's main args
        self.checkpoint_args = None  # Will hold additional checkpoint args
        self.all_models = None       # Model sharded over different parallelism
        self.md = None               # Metadata sent to the saver
        self.consumed_train_samples = None
        self.consumed_valid_samples = None

    def _maybe_parse_additional_megatron_args(self, margs, checkpoint_args):
        """
        Method used to optionally add arguments from the checkpoint to the main args.
        For instance, using margs.some_arg = checkpoint_args.some_arg
        """
        return margs

    def parse_megatron_args(self):
        """
        Parse Megatron arguments by forcibly overwriting sys.argv.
        Populates self.margs and self.checkpoint_args.
        """
        # Ensure we can import Megatron
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
        if self.args.megatron_path is not None:
            sys.path.insert(0, self.args.megatron_path)

        try:
            from megatron.training.arguments import parse_args, validate_args
            from megatron.training.checkpointing import load_args_from_checkpoint
        except ModuleNotFoundError:
            print("Unable to import Megatron. Please specify --megatron-path. Exiting.")
            self.queue.put("exit")
            sys.exit(1)

        # Overwrite sys.argv
        sys.argv = self.build_sys_argv()

        margs = parse_args()
        margs, checkpoint_args = load_args_from_checkpoint(margs)

        # Adjust world size so validation doesn't fail
        margs.world_size = margs.tensor_model_parallel_size * margs.pipeline_model_parallel_size

        # Copy data types from checkpoint
        margs.fp16 = checkpoint_args.fp16
        margs.bf16 = checkpoint_args.bf16

        # Expert parallelism requires sequence parallelism
        if margs.expert_model_parallel_size > 1:
            margs.sequence_parallel = True
        
        margs = self._maybe_parse_additional_megatron_args(margs, checkpoint_args)

        # Validate final arguments
        try:
            from megatron.training.arguments import validate_args
            margs = validate_args(margs)
        except Exception as e:
            print(f"Error validating Megatron arguments: {e}")
            self.queue.put("exit")
            sys.exit(1)

        margs.use_legacy_models = False
        margs.transformer_impl = self.args.loader_transformer_impl
        if self.args.loader_transformer_impl == "local" and margs.normalization == "RMSNorm":
            margs.no_persist_layer_norm = True

        self.margs = margs
        self.checkpoint_args = checkpoint_args

    def _maybe_ensure_additional_required_arguments(self):
        """
        Can be used to ensure some expected args are present.
        For instance, use self.check_for_arg('some_arg')
        """
        pass

    def check_for_arg(self, arg_name, default=None):
        if getattr(self.margs, arg_name, None) is None:
            if default is not None:
                setattr(self.margs, arg_name, default)
            else:
                print(f"Checkpoint does not specify argument {arg_name}. Exiting.")
                print(f"Arguments: {self.margs}")
                self.queue.put("exit")
                sys.exit(1)

    def ensure_required_arguments(self):
        """
        Ensure that certain Megatron arguments (from checkpoint) are present.
        If missing, either set defaults or exit.
        """

        self.check_for_arg('tensor_model_parallel_size')
        self.check_for_arg('pipeline_model_parallel_size')
        self.check_for_arg('num_layers')
        self.check_for_arg('hidden_size')
        self.check_for_arg('seq_length')
        self.check_for_arg('num_attention_heads')
        self.check_for_arg('max_position_embeddings')
        self.check_for_arg('position_embedding_type')
        self.check_for_arg('tokenizer_type')
        self.check_for_arg('iteration')
        self.check_for_arg('bert_binary_head')
        self.check_for_arg('disable_bias_linear', False)
        self.check_for_arg('params_dtype')
        self.check_for_arg('swiglu', False)

        self._maybe_ensure_additional_required_arguments()

    def initialize_megatron_env(self):
        """
        Initialize Megatron global variables and fused kernels.
        """
        try:
            from megatron.training.global_vars import set_global_variables
            from megatron.core import mpu
            from megatron.legacy import fused_kernels
        except ModuleNotFoundError as e:
            print(f"Unable to import required Megatron modules: {e}")
            self.queue.put("exit")
            sys.exit(1)

        set_global_variables(self.margs, build_tokenizer=self.build_tokenizer)
        mpu.set_tensor_model_parallel_world_size(self.margs.tensor_model_parallel_size)
        mpu.set_pipeline_model_parallel_world_size(self.margs.pipeline_model_parallel_size)
        mpu.set_virtual_pipeline_model_parallel_world_size(self.margs.virtual_pipeline_model_parallel_size)
        mpu.set_expert_model_parallel_world_size(self.margs.expert_model_parallel_size)
        fused_kernels.load(self.margs)

    def compute_true_vocab_size(self):
        """Determine the 'true' (non-padded) vocab size."""
        if self.args.true_vocab_size is not None:
            return self.args.true_vocab_size
        elif self.args.vocab_file is not None:
            vocab = json.load(open(self.args.vocab_file))
            return len(vocab)
        else:
            return None

    def verify_vocabs_match(self, true_vocab_size):
        """
        If both --true-vocab-size and --vocab-file are specified, verify they match.
        Return False (and exit) if they don't match; True otherwise.
        """
        if self.args.true_vocab_size is not None and self.args.vocab_file is not None:
            vocab = json.load(open(self.args.vocab_file))
            if len(vocab) != self.args.true_vocab_size:
                print("Both --true-vocab-size and --vocab-file specified but vocab sizes do not match. Aborting.")
                return False
        return True

    def load_model_shards(self, model_provider, dtype):
        """
        Build and load model shards for each tensor-parallel rank, returning:
          - A nested list of loaded models by [pipeline_rank][virtual_pipeline_rank].
          - consumed_train_samples, consumed_valid_samples
        """
        from megatron.core import mpu
        from megatron.training.checkpointing import load_checkpoint

        consumed_train_samples = None
        consumed_valid_samples = None
        tp_size = self.margs.tensor_model_parallel_size
        pp_size = self.margs.pipeline_model_parallel_size
        vp_size = self.margs.virtual_pipeline_model_parallel_size or 1

        all_models = []  # all_models[pp_rank][vp_rank] = [list of models across TP ranks]

        def get_models_for_pipeline_stage(count, dtype):
            local_models_for_stage = [[] for _ in range(vp_size)]
            for tp_rank in range(count):
                mpu.set_tensor_model_parallel_rank(tp_rank)
                model_list = []

                for i in range(vp_size):
                    mpu.set_virtual_pipeline_model_parallel_rank(i)
                    pre_process = mpu.is_pipeline_first_stage()
                    post_process = mpu.is_pipeline_last_stage()
                    this_model = model_provider(pre_process=pre_process,
                                                post_process=post_process).to(dtype)
                    model_list.append(this_model)

                # Each time we load, we set counters to 0, pass None for optimizer/ LR
                self.margs.consumed_train_samples = 0
                self.margs.consumed_valid_samples = 0
                self.margs.exit_on_missing_checkpoint = True
                load_checkpoint(model_list, None, None)

                # Validate that train/valid samples match across ranks
                nonlocal consumed_train_samples, consumed_valid_samples
                if consumed_train_samples is not None:
                    assert self.margs.consumed_train_samples == consumed_train_samples
                else:
                    consumed_train_samples = self.margs.consumed_train_samples

                if consumed_valid_samples is not None:
                    assert self.margs.consumed_valid_samples == consumed_valid_samples
                else:
                    consumed_valid_samples = self.margs.consumed_valid_samples

                for vp_rank in range(vp_size):
                    local_models_for_stage[vp_rank].append(model_list[vp_rank])

                # Print memory usage
                print_memory_usage("loader", tp_rank, count)

            return local_models_for_stage

        # Load shards for each pipeline rank
        mpu.set_virtual_pipeline_model_parallel_rank(0)
        for pp_rank in range(pp_size):
            mpu.set_pipeline_model_parallel_rank(pp_rank)
            all_models.append(get_models_for_pipeline_stage(tp_size, dtype))

        return all_models, consumed_train_samples, consumed_valid_samples
    
    def send_metadata_over_queue(self):
        # Let the consumer know the overall metadata:
        self.md.consumed_train_samples = self.consumed_train_samples
        self.md.consumed_valid_samples = self.consumed_valid_samples
        self.queue.put(self.md)

    def queue_put(self, name, msg):
        print(f"sending {name}")
        msg["name"] = name
        self.queue.put(msg)

    def send_llm_over_queue(self, schema):
        """
        Using self.all_models, extract model parameters and send them over the queue.
        """
        # 2) Transformer layers
        tp_size = self.margs.tensor_model_parallel_size
        pp_size = self.margs.pipeline_model_parallel_size
        vp_size = self.margs.virtual_pipeline_model_parallel_size or 1

        # all_models[pp_rank][vp_rank] is a list across TP ranks
        # We'll start with pipeline=0, vp=0 for embeddings/final norm
        first_pipeline_models = self.all_models[0][0]

        # 1) Embeddings
        embeddings = [schema.get("embeddings", m) for m in first_pipeline_models]
        message = {
            "word embeddings": torch.cat([e["word"] for e in embeddings], dim=0)
        }
        if self.md.position_embedding_type == 'learned_absolute':
            # Only send one set from rank 0
            message["position embeddings"] = embeddings[0]["pos"]
        else:
            assert embeddings[0]["pos"] is None
        self.queue_put("embeddings", message)

        total_layer_num = 0
        for vp_rank in range(vp_size):
            for pp_rank in range(pp_size):
                models = self.all_models[pp_rank][vp_rank]
                num_layers = schema.get_num_layers(models[0])
                for layer_idx in range(num_layers):
                    message = {}
                    layer = schema.get_layer(models[0], layer_idx)

                    # Non-parallel params
                    message["input norm weight"] = layer["self_attn_norm_weight"]
                    message["post norm weight"] = layer["mlp_norm_weight"]
                    if self.md.norm_has_bias:
                        message["input norm bias"] = layer["self_attn_norm_bias"]
                        message["post norm bias"] = layer["mlp_norm_bias"]
                    if self.md.linear_bias:
                        message["dense bias"] = layer["self_attn_proj_bias"]
                        message["mlp l1 bias"] = layer["mlp_fc2_bias"]

                    # Collect parallel parameters
                    qkv_weight, qkv_bias = [], []
                    dense_weight = []
                    mlp_l0_weight, mlp_l0_bias = [], []
                    mlp_l1_weight = []

                    for model_tp in models:
                        layer_p = schema.get_layer(model_tp, layer_idx)
                        qkv_weight.append(layer_p["self_attn_qkv_weight"])
                        dense_weight.append(layer_p["self_attn_proj_weight"])
                        mlp_l0_weight.append(layer_p["mlp_fc1_weight"])
                        mlp_l1_weight.append(layer_p["mlp_fc2_weight"])
                        if self.md.qkv_bias:
                            qkv_bias.append(layer_p["self_attn_qkv_bias"])
                        if self.md.linear_bias:
                            mlp_l0_bias.append(layer_p["mlp_fc1_bias"])

                    # If we are using SwiGLU, chunk each mlp_l0_weight
                    if self.md.swiglu:
                        for i in range(tp_size):
                            mlp_l0_weight[i] = torch.chunk(mlp_l0_weight[i], 2, dim=0)
                        message["mlp l0 weight W"] = torch.cat([w[0] for w in mlp_l0_weight], dim=0)
                        message["mlp l0 weight V"] = torch.cat([w[1] for w in mlp_l0_weight], dim=0)
                    else:
                        message["mlp l0 weight"] = torch.cat(mlp_l0_weight, dim=0)

                    # Standard concatenations
                    message["qkv weight"] = torch.cat(qkv_weight, dim=0)
                    message["dense weight"] = torch.cat(dense_weight, dim=1)
                    message["mlp l1 weight"] = torch.cat(mlp_l1_weight, dim=1)

                    if self.md.qkv_bias:
                        message["qkv bias"] = torch.cat(qkv_bias, dim=0)
                    if self.md.linear_bias:
                        if self.md.swiglu:
                            for i in range(tp_size):
                                mlp_l0_bias[i] = torch.chunk(mlp_l0_bias[i], 2, dim=0)
                            message["mlp l0 bias W"] = torch.cat([b[0] for b in mlp_l0_bias], dim=0)
                            message["mlp l0 bias V"] = torch.cat([b[1] for b in mlp_l0_bias], dim=0)
                        else:
                            message["mlp l0 bias"] = torch.cat(mlp_l0_bias, dim=0)

                    self.queue_put(f"transformer layer {total_layer_num}", message)
                    total_layer_num += 1

        # 3) Final norm
        final_norm = schema.get("final_norm", models[0])
        message = {"weight": final_norm["weight"]}
        if self.md.norm_has_bias:
            message["bias"] = final_norm["bias"]
        self.queue_put("final norm", message)

        # 4) Output layer
        if self.md.output_layer:
            output_layers = [schema.get("output_layer", m) for m in models]
            message = {
                "weight": torch.cat([layer["weight"] for layer in output_layers], dim=0),
            }
            self.queue_put("output layer", message)

        # 5) BERT-specific parameters
        if self.md.model_type == 'BERT':
            # Pooler
            pooler = schema.get("pooler", models[0])
            message = {
                "weight": pooler["weight"],
                "bias": pooler["bias"],
            }
            self.queue_put("pooler", message)

            # LM head
            lm_head = schema.get("lm_head", models[0])
            message = {
                "dense weight": lm_head["dense_weight"],
                "dense bias": lm_head["dense_bias"],
                "norm weight": lm_head["norm_weight"],
            }
            if self.md.norm_has_bias:
                message["norm bias"] = lm_head["norm_bias"]
            self.queue_put("lm head", message)

            # Binary head
            if self.md.bert_binary_head:
                binary_head = schema.get("binary_head", models[0])
                message = {
                    "weight": binary_head["weight"],
                    "bias": binary_head["bias"],
                }
                self.queue_put("binary head", message)

        # Done
        self.queue.put("done")

    def load(self):
        """
        Orchestrate the entire flow of loading the Megatron checkpoint.
        """
        # 1) Parse Megatron arguments
        self.parse_megatron_args()

        # 2) Ensure required arguments are present
        self.ensure_required_arguments()

        # 3) Import the correct model provider (GPT or BERT)
        model_provider = self.import_model_provider()

        # 4) Initialize the Megatron environment
        self.initialize_megatron_env()

        # 5) Determine the true vocab size and verify if both sources match
        true_vocab_size = self.compute_true_vocab_size()
        if not self.verify_vocabs_match(true_vocab_size):
            self.queue.put("exit")
            sys.exit(1)

        # 6) Build metadata
        self.md = self.build_checkpoint_metadata(true_vocab_size)

        # 7) Load all model shards
        self.all_models, self.consumed_train_samples, self.consumed_valid_samples = self.load_model_shards(
            model_provider,
            self.md.params_dtype
        )

        # 8) Send model over the queue        
        self.send_model_over_queue()

    def build_checkpoint_metadata(self, true_vocab_size):
        """
        Construct a simple namespace for all relevant model metadata.
        """
        norm_has_bias = True
        if hasattr(self.checkpoint_args, 'normalization'):
            # For older models, normalization was always "LayerNorm".
            norm_has_bias = (self.checkpoint_args.normalization == "LayerNorm")

        md = types.SimpleNamespace()
        md.model_type = self.args.model_type
        md.num_layers = self.margs.num_layers
        md.hidden_size = self.margs.hidden_size
        md.seq_length = self.margs.seq_length
        md.num_attention_heads = self.margs.num_attention_heads
        md.max_position_embeddings = self.margs.max_position_embeddings
        md.tokenizer_type = self.margs.tokenizer_type
        md.iteration = self.margs.iteration
        md.params_dtype = self.margs.params_dtype
        md.bert_binary_head = self.margs.bert_binary_head
        md.output_layer = self.margs.untie_embeddings_and_output_weights
        md.position_embedding_type = self.margs.position_embedding_type
        md.linear_bias = self.margs.add_bias_linear
        md.qkv_bias = self.margs.add_qkv_bias
        md.norm_has_bias = norm_has_bias
        md.swiglu = self.margs.swiglu
        md.previous_tensor_parallel_size = self.margs.tensor_model_parallel_size
        md.previous_pipeline_parallel_size = self.margs.pipeline_model_parallel_size
        md.true_vocab_size = true_vocab_size
        md.make_vocab_size_divisible_by = self.margs.make_vocab_size_divisible_by
        md.checkpoint_args = self.checkpoint_args
        md.use_legacy_models = self.margs.use_legacy_models
        return md

    def build_sys_argv(self):
        """
        Construct a sys.argv list for Megatron's argument parser.
        This centralizes the hack of overwriting sys.argv.
        """

        return [
            'script.py',
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
            '--mock-data',  # To pass the "blend data checks" in arguments.py
            '--load', self.args.load_dir,
            '--exit-on-missing-checkpoint',
            '--use-mp-args-from-checkpoint-args',
            '--no-one-logger',
        ]

    def import_model_provider(self):
        """Return the correct model_provider function depending on GPT vs. BERT."""
        raise NotImplementedError

    def send_model_over_queue(self):
        """Creates model schema and sends the model over the queue"""
        raise NotImplementedError

