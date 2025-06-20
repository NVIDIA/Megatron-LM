# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
from importlib.metadata import version
import os
from packaging.version import Version as PkgVersion
import sys

import torch

from schema_core import get_model_schema
from saver_base import MegatronCheckpointSaverBase
from utils import chunk_bias, chunk_weight

os.environ["NVTE_ALLOW_NONDETERMINISTIC_ALGO"] = "0"
os.environ["NCCL_ALGO"] = "^NVLS"

"""
python tools/checkpoint/convert.py \
    --model-type GPT --loader llava --saver llava \
    --megatron-path . --target-tensor-parallel-size 1 --true-vocab-size 151680 \
    --load-dir <load_path> \
    --save-dir <save_path> --megatron-path . --target-tensor-parallel-size 1 --true-vocab-size 151680
"""


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


class MegatronCheckpointSaverLLaVA(MegatronCheckpointSaverBase):
    """Orchestrates saving a LLaVA Megatron checkpoint using parameters received on a multiprocessing queue.

    Args:
        args: argparse Namespace with Megatron checkpoint configurations.
        queue: A multiprocessing.Queue (or similar) used to send out loaded tensors.
        build_tokenizer: Whether to build a tokenizer for the model to be saved
    """

    def _load_checkpoint_args(self, margs):
        if hasattr (self.md, 'checkpoint_args'):
            # These are arguments that we are either changing, or cause problems for validation if they are set
            # Note that some of these deal with T5 so will need to be changed if we support T5.
            args_to_keep = ['tensor_model_parallel_size', 'pipeline_model_parallel_size', 'expert_model_parallel_size', 'world_size', 'params_dtype',
                            'num_layers_per_virtual_pipeline_stage', 'virtual_pipeline_model_parallel_size',
                            'masked_softmax_fusion', 'bias_gelu_fusion', 'bias_dropout_fusion',
                            'sequence_parallel', 'async_tensor_model_parallel_allreduce',
                            'no_load_optim', 'no_load_rng', 'no_save_optim', 'no_save_rng',
                            'vocab_file',
                            'save_interval', 'save',
                            'perform_initialization', 'use_cpu_initialization',
                            'recompute_granularity', 'recompute_num_layers', 'recompute_method',
                            'encoder_num_layers', 'encoder_seq_length',
                            'distribute_saved_activations',
                            'train_iters', 'lr_decay_iters', 'lr_warmup_iters', 'lr_warmup_fraction',
                            'start_weight_decay', 'end_weight_decay',
                            'ckpt_format', 'inference_batch_times_seqlen_threshold',
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

        print("im here")
        return margs
    def build_sys_argv(self):
        my_argv = ['script.py',
                    '--use-checkpoint-args',
                    '--use-mp-args-from-checkpoint-args', # need this since we're loading torch ckpts
                    '--num-layers', str(self.md.num_layers),
                    '--hidden-size', str(self.md.hidden_size),
                    '--seq-length', str(self.md.seq_length),
                    '--num-experts', str(getattr(self.md, "num_experts", 0)),
                    '--num-attention-heads', str(self.md.num_attention_heads),
                    '--max-position-embeddings', str(self.md.max_position_embeddings),
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
                    '--mock-data',
                    '--load', self.args.load_dir,
                    '--exit-on-missing-checkpoint',
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

    def _maybe_parse_additional_megatron_args(self, margs):
        # Copy values for llava model from checkpoint, should only need to be dummy values
        margs.use_te = getattr(self.md.checkpoint_args, "use_te", margs.transformer_impl == "transformer_engine")
        margs.language_model_type = self.md.checkpoint_args.language_model_type
        margs.vision_model_type = self.md.checkpoint_args.vision_model_type
        margs.tokenizer_prompt_format = getattr(self.md.checkpoint_args, "tokenizer_prompt_format", "dummy")
        margs.disable_vision_class_token = getattr(self.md.checkpoint_args, "disable_vision_class_token", False)
        margs.use_tiling = getattr(self.md.checkpoint_args, "use_tiling", False)
        margs.pixel_shuffle = getattr(self.md.checkpoint_args, "pixel_shuffle", False)
        margs.use_tile_tags = getattr(self.md.checkpoint_args, "use_tile_tags", False)
        margs.max_num_tiles = getattr(self.md.checkpoint_args, "max_num_tiles", 1)
        margs.use_thumbnail = getattr(self.md.checkpoint_args, "use_thumbnail", False)
        margs.img_h = getattr(self.md.checkpoint_args, "img_h", 448)
        margs.img_w = getattr(self.md.checkpoint_args, "img_w", 448)
        margs.patch_dim = getattr(self.md.checkpoint_args, "patch_dim", 16)
        margs.decoder_seq_length = getattr(self.md.checkpoint_args, "decoder_seq_length", 4096)
        margs.special_tokens = getattr(self.md.checkpoint_args, "special_tokens", "")
        margs.image_tag_type = getattr(self.md.checkpoint_args, "image_tag_type", "")
        margs.allow_missing_vision_projection_checkpoint = getattr(self.md.checkpoint_args, "allow_missing_vision_projection_checkpoint", False)
        margs.freeze_LM = getattr(self.md.checkpoint_args, "freeze_LM", False)
        margs.freeze_ViT = getattr(self.md.checkpoint_args, "freeze_ViT", False)
        margs.encoder_tensor_model_parallel_size = getattr(self.md.checkpoint_args, "encoder_tensor_model_parallel_size", 0)
        margs.force_system_message = getattr(self.md.checkpoint_args, "force_system_message", False)
        margs.image_tag_type = getattr(self.md.checkpoint_args, "image_tag_type", "")
        margs.num_frames = getattr(self.md.checkpoint_args, "num_frames", 8)
        margs.recompute_vision = getattr(self.md.checkpoint_args, "recompute_vision", False)
        margs.padded_vocab_size = self.md.padded_vocab_size

        return margs

    def import_model_provider(self):
        try:
            from megatron.core.enums import ModelType
        except ModuleNotFoundError as e:
            print(f"Unable to import required Megatron modules: {e}")
            sys.exit(1)

        if self.md.model_type == 'GPT':
            sys.path.insert(0, './examples/multimodal')
            from examples.multimodal.model import model_provider
            from examples.multimodal.config import get_vision_model_config, get_vision_projection_config
            self.model_provider = model_provider
            self.margs.model_type = ModelType.encoder_or_decoder
        elif self.md.model_type == 'BERT':
            from pretrain_bert import model_provider
            self.margs.model_type = ModelType.encoder_or_decoder
            self.model_provider = model_provider
        else:
            raise Exception(f'unrecognized model type: {self.args.model_type}')

    def receive_vision_backbone(self, schema):

        # ViT Embeddings.
        #-----------
        # The ViT embeddings are put on the PP / EP / TP 0 
        vit_embeddings_msg = self.queue_get("vit embeddings")

        if self.md.vision_model_type in ("radio", "radio-g"):
            embedder_weight = chunk_weight(vit_embeddings_msg["embedder weight"], "column", self.args.target_tensor_parallel_size, self.args.target_expert_parallel_size)
            if self.md.vision_model_type == "radio-g":
                embedder_bias = chunk_bias(vit_embeddings_msg["embedder bias"], "column", self.args.target_tensor_parallel_size, self.args.target_expert_parallel_size)

        for ep_rank in range(self.args.target_expert_parallel_size):
            for tp_rank in range(self.args.target_tensor_parallel_size):
                model = self.get_local_model(0, ep_rank, tp_rank)
                if self.md.vision_model_type in ("internvit", "clip", "siglip"):
                    model.vision_model.conv1.weight.data.copy_(vit_embeddings_msg["conv1 weight"])
                    if self.md.vision_model_type in ("internvit", "siglip"):
                        model.vision_model.conv1.bias.data.copy_(vit_embeddings_msg["conv1 bias"])
                    model.vision_model.position_embeddings.weight.data.copy_(vit_embeddings_msg["position embeddings"])

                if self.md.vision_model_type == "radio-g":
                    model.vision_model.mask_token.data.copy_(vit_embeddings_msg["mask token"])

                if self.md.vision_model_type in ("radio", "radio-g"):
                    model.vision_model.embedder.weight.data.copy_(embedder_weight[tp_rank])
                    if self.md.vision_model_type == "radio-g":
                        model.vision_model.embedder.bias.data.copy_(embedder_bias[tp_rank])
                    model.vision_model.position_embeddings.data.copy_(vit_embeddings_msg["position embeddings"])

                if self.md.vision_model_type in ("clip"):
                    model.vision_model.ln_pre.weight.data.copy_(vit_embeddings_msg["ln pre weight"])
                    model.vision_model.ln_pre.bias.data.copy_(vit_embeddings_msg["ln pre bias"])

                if self.md.vision_model_type in ("siglip", "radio-g"):
                    model.vision_model.ln_post.weight.data.copy_(vit_embeddings_msg["ln post weight"])
                    model.vision_model.ln_post.bias.data.copy_(vit_embeddings_msg["ln post bias"])

                if self.md.vision_model_type in ("internvit", "clip", "radio", "radio-g"):
                    model.vision_model.class_token.data.copy_(vit_embeddings_msg["class token"])

        # ViT Transformer layers.
        #-----------
        total_layer_num = 0
        # ViT will only ever be on first pp rank
        pp_rank = 0
        ep_rank = 0
        for layer_id in range(schema.get_num_layers(self.get_local_model(pp_rank, 0, 0))):
            msg = self.queue_get(f"vit transformer layer {total_layer_num}")

            input_norm_weight = msg.pop("input norm weight")
            pre_mlp_norm_weight = msg.pop("pre mlp norm weight")
            if self.md.vision_norm_has_bias:
                input_norm_bias = msg.pop("input norm bias")
                pre_mlp_norm_bias = msg.pop("pre mlp norm bias")

            # Split up the parallel tensors
            qkv_weight = chunk_weight(msg.pop("qkv weight"), "column", self.args.target_tensor_parallel_size)
            dense_weight = chunk_weight(msg.pop("dense weight"), "row", self.args.target_tensor_parallel_size)
            mlp_l1_weight = chunk_weight(msg.pop("mlp l1 weight"), "row", self.args.target_tensor_parallel_size, self.args.target_expert_parallel_size)

            # Special handling for swiglu
            if self.md.vision_swiglu:
                mlp_l0_weight_W = chunk_weight(msg.pop("mlp l0 weight W"), "column", self.args.target_tensor_parallel_size, self.args.target_expert_parallel_size)
                mlp_l0_weight_V = chunk_weight(msg.pop("mlp l0 weight V"), "column", self.args.target_tensor_parallel_size, self.args.target_expert_parallel_size)
                mlp_l0_weight = torch.cat((mlp_l0_weight_W, mlp_l0_weight_V), dim=-2)
            else:
                mlp_l0_weight = chunk_weight(msg.pop("mlp l0 weight"), "column", self.args.target_tensor_parallel_size, self.args.target_expert_parallel_size)

            if self.md.vision_qkv_bias:
                qkv_bias = chunk_bias(msg.pop("qkv bias"), 'column', self.args.target_tensor_parallel_size)
            if self.md.vision_linear_bias:
                dense_bias = msg.pop("dense bias")
                mlp_l1_bias = chunk_bias(msg.pop("mlp l1 bias"), 'row', self.args.target_tensor_parallel_size, self.args.target_expert_parallel_size)
                if self.md.vision_swiglu:
                    mlp_l0_bias_W = chunk_bias(msg.pop("mlp l0 bias W"), 'column', self.args.target_tensor_parallel_size, self.args.target_expert_parallel_size)
                    mlp_l0_bias_V = chunk_bias(msg.pop("mlp l0 bias V"), 'column', self.args.target_tensor_parallel_size, self.args.target_expert_parallel_size)
                    mlp_l0_bias = torch.cat((mlp_l0_bias_W, mlp_l0_bias_V), dim=-1)
                else:
                    mlp_l0_bias = chunk_bias(msg.pop("mlp l0 bias"), 'column', self.args.target_tensor_parallel_size, self.args.target_expert_parallel_size)
            if self.md.vision_model_type in ("internvit", "radio-g"):
                ls1 = msg.pop("ls1")
                ls2 = msg.pop("ls2")
            if self.md.vision_model_type == "internvit":
                # chunk_bias is intentional here, since these weights only have 1-dim
                k_norm_weight = chunk_bias(msg.pop("k norm weight"), "column", self.args.target_tensor_parallel_size)
                q_norm_weight = chunk_bias(msg.pop("q norm weight"), "column", self.args.target_tensor_parallel_size)
            if self.md.vision_norm_has_bias and self.md.vision_model_type == "internvit":
                k_norm_bias = chunk_bias(msg.pop("k norm bias"), "column", self.args.target_tensor_parallel_size)
                q_norm_bias = chunk_bias(msg.pop("q norm bias"), "column", self.args.target_tensor_parallel_size)

            # Save them to the model
            for tp_rank in range(self.args.target_tensor_parallel_size):
                params_dict = {
                    "self_attn_norm_weight" : input_norm_weight,
                    "self_attn_norm_bias" : input_norm_bias if self.md.vision_norm_has_bias else None,

                    "self_attn_qkv_weight" : qkv_weight[tp_rank],
                    "self_attn_proj_weight" : dense_weight[tp_rank],
                    "mlp_norm_weight" : pre_mlp_norm_weight,
                    "mlp_norm_bias" : pre_mlp_norm_bias if self.md.vision_norm_has_bias else None,
                    "mlp_fc1_weight" : mlp_l0_weight[tp_rank],
                    "mlp_fc2_weight" : mlp_l1_weight[tp_rank],
                }
                if self.md.vision_qkv_bias:
                    params_dict.update({
                        "self_attn_qkv_bias" : qkv_bias[tp_rank]
                    })
                if self.md.vision_linear_bias:
                    params_dict.update({
                        "self_attn_proj_bias" : dense_bias,
                        "mlp_fc1_bias" : mlp_l0_bias[tp_rank],
                        "mlp_fc2_bias" : mlp_l1_bias
                    })
                if self.md.vision_model_type == "radio-g":
                    params_dict.update({
                        "ls1": ls1,
                        "ls2": ls2,
                    })
                if self.md.vision_model_type == "internvit":
                    params_dict.update({
                        "k_layernorm_weight": k_norm_weight[tp_rank],
                        "q_layernorm_weight": q_norm_weight[tp_rank],
                        "ls1": ls1,
                        "ls2": ls2,
                    })
                if self.md.vision_norm_has_bias and self.md.vision_model_type == "internvit":
                    params_dict.update({
                        "k_layernorm_bias": k_norm_bias[tp_rank],
                        "q_layernorm_bias": q_norm_bias[tp_rank],
                    })
                model = self.get_local_model(pp_rank, ep_rank, tp_rank)
                schema.set_layer(model, layer_id, params_dict)

            total_layer_num = total_layer_num + 1
            self.check_message(msg)

    def receive_vision_projection(self):
        vision_projection_msg = self.queue_get("vision projection")

        vision_projection_l0_weight = chunk_weight(
            vision_projection_msg.pop("vision projection l0 weight"), "column", self.args.target_tensor_parallel_size)
        vision_projection_l1_weight = chunk_weight(
            vision_projection_msg.pop("vision projection l1 weight"), "row", self.args.target_tensor_parallel_size)
        # Check for this explicitly, since don't have any gurantees based on our model types
        has_vision_projection_norm_weight = False
        if "vision projection norm weight" in vision_projection_msg:
            vision_projection_norm_weight = vision_projection_msg.pop("vision projection norm weight")
            has_vision_projection_norm_weight = True
        has_vision_projection_norm_bias = False
        if "vision projection norm bias" in vision_projection_msg:
            vision_projection_norm_bias = vision_projection_msg.pop("vision projection norm bias")
            has_vision_projection_norm_bias = True
        if self.md.vision_projection_linear_bias:
            vision_projection_l0_bias = chunk_bias(
            vision_projection_msg.pop("vision projection l0 bias"), "column", self.args.target_tensor_parallel_size)
            vision_projection_l1_bias = vision_projection_msg.pop("vision projection l1 bias")
        for tp_rank in range(self.args.target_tensor_parallel_size):
            # The vision projection is on the PP / EP 0 
            model = self.get_local_model(0, 0, tp_rank)
            model.vision_projection.encoder.linear_fc1.weight.data.copy_(
                vision_projection_l0_weight[tp_rank])
            model.vision_projection.encoder.linear_fc2.weight.data.copy_(
                vision_projection_l1_weight[tp_rank])
            if has_vision_projection_norm_weight:
                model.vision_projection.encoder.linear_fc1.layer_norm_weight.data.copy_(
                    vision_projection_norm_weight)
            if has_vision_projection_norm_bias:
                model.vision_projection.encoder.linear_fc1.layer_norm_bias.data.copy_(
                    vision_projection_norm_bias)
            if self.md.vision_projection_linear_bias:
                model.vision_projection.encoder.linear_fc1.bias.data.copy_(
                    vision_projection_l0_bias[tp_rank])
                model.vision_projection.encoder.linear_fc2.bias.data.copy_(vision_projection_l1_bias)

    def receive_model(self):
        extra_layer_schema = {}

        if self.md.vision_model_type == "internvit":
            extra_layer_schema = {
                "ls1": "ls1",
                "ls2": "ls2",
                "k_layernorm_weight": "self_attention.k_layernorm.weight",
                "k_layernorm_bias": "self_attention.k_layernorm.bias",
                "q_layernorm_weight": "self_attention.q_layernorm.weight",
                "q_layernorm_bias": "self_attention.q_layernorm.bias",
            }
        elif self.md.vision_model_type == "radio-g":
            extra_layer_schema = {
                "ls1": "ls1",
                "ls2": "ls2",
            }

        schema_vision_backbone = get_model_schema(
            "GPT",
            self.margs.transformer_impl,
            self.margs.num_experts,
            self.margs.expert_model_parallel_size,
            prefix="vision_model.",
            extra_layer_schema=extra_layer_schema,
        )
        self.receive_vision_backbone(schema_vision_backbone)

        self.receive_vision_projection()

        schema = get_model_schema(
            self.md.model_type,
            self.margs.transformer_impl,
            self.margs.num_experts,
            self.margs.expert_model_parallel_size,
            prefix="language_model."
        )
        self.receive_lm(schema, prefix="language_model")

def save_checkpoint(queue, args):
    """
    Required top-level function that creates the saver and calls its .save().
    """
    saver = MegatronCheckpointSaverLLaVA(args, queue, build_tokenizer=True)
    try:
        saver.save()
    except Exception as e:
        raise e
    
