# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import json
import os
import sys
import torch
import types

from copy import deepcopy

from schema_core import get_model_schema
from loader_base import MegatronCheckpointLoaderBase


def add_arguments(parser):
    group = parser.add_argument_group(title="Mcore LLaVA loader")

    group.add_argument('--true-vocab-size', type=int, default=None,
                       help='original size of vocab, if specified will trim padding from embedding table.')
    group.add_argument('--vocab-file', type=str, default=None,
                       help='Path to the vocab file. If specified will use this to get vocab size and '
                       'trim padding from the embedding table.')
    group.add_argument('--megatron-path', type=str, default=None,
                       help='Base directory of Megatron repository')
    group.add_argument('--loader-transformer-impl', default='transformer_engine',
                       choices=['local', 'transformer_engine'],
                       help='Which Transformer implementation to use.')
    group.add_argument('--vit-dummy-head-count', type=int, default=0,
                       help='Number of vit dummy heads to remove')


class MegatronCheckpointLoaderLLaVA(MegatronCheckpointLoaderBase):
    """Orchestrates loading a Megatron checkpoint and sending
    model parameters over a given multiprocessing queue.

    Args:
        args: argparse Namespace with Megatron checkpoint configurations.
        queue: A multiprocessing.Queue (or similar) used to send out loaded tensors.
    """

    def build_sys_argv(self):
        """
        Construct a sys.argv list for Megatron's argument parser.
        This centralizes the hack of overwriting sys.argv.
        """
        return [
            *super().build_sys_argv(),
            '--ckpt-format', 'torch',
            '--use-checkpoint-args',
        ]

        return argv

    def _maybe_parse_additional_megatron_args(self, margs, checkpoint_args):
        """
        Parse Megatron arguments by forcibly overwriting sys.argv.
        Populates self.margs and self.checkpoint_args.
        """
        # Copy values for llava model from checkpoint, should only need to be dummy values
        margs.use_te = getattr(checkpoint_args, "use_te", margs.transformer_impl == "transformer_engine")
        margs.language_model_type = checkpoint_args.language_model_type
        margs.vision_model_type = checkpoint_args.vision_model_type
        margs.tokenizer_prompt_format = getattr(checkpoint_args, "tokenizer_prompt_format", "dummy")
        margs.disable_vision_class_token = getattr(checkpoint_args, "disable_vision_class_token", False)
        margs.use_tiling = getattr(checkpoint_args, "use_tiling", False)
        margs.pixel_shuffle = getattr(checkpoint_args, "pixel_shuffle", False)
        margs.use_tile_tags = getattr(checkpoint_args, "use_tile_tags", False)
        margs.max_num_tiles = getattr(checkpoint_args, "max_num_tiles", 1)
        margs.use_thumbnail = getattr(checkpoint_args, "use_thumbnail", False)
        margs.img_h = getattr(checkpoint_args, "img_h", 448)
        margs.img_w = getattr(checkpoint_args, "img_w", 448)
        margs.patch_dim = getattr(checkpoint_args, "patch_dim", 16)
        margs.decoder_seq_length = getattr(checkpoint_args, "decoder_seq_length", 4096)
        margs.special_tokens = getattr(checkpoint_args, "special_tokens", "")
        margs.image_tag_type = getattr(checkpoint_args, "image_tag_type", "")
        margs.allow_missing_vision_projection_checkpoint = getattr(checkpoint_args, "allow_missing_vision_projection_checkpoint", False)
        margs.freeze_LM = getattr(checkpoint_args, "freeze_LM", False)
        margs.freeze_ViT = getattr(checkpoint_args, "freeze_ViT", False)
        margs.encoder_tensor_model_parallel_size = getattr(checkpoint_args, "encoder_tensor_model_parallel_size", 0)
        margs.force_system_message = getattr(checkpoint_args, "force_system_message", False)
        margs.image_tag_type = getattr(checkpoint_args, "image_tag_type", "")
        margs.num_frames = getattr(checkpoint_args, "num_frames", 8)
        margs.recompute_vision = getattr(checkpoint_args, "recompute_vision", False)
        margs.vocab_size = getattr(checkpoint_args, "vocab_size", None)
        margs.padded_vocab_size = checkpoint_args.padded_vocab_size
        
        return margs

    def _maybe_ensure_additional_required_arguments(self):
        """
        Ensure that certain Megatron arguments (from checkpoint) are present.
        If missing, either set defaults or exit.
        """
        self.check_for_arg('num_query_groups')
        self.check_for_arg('kv_channels')

    def import_model_provider(self):
        if self.args.megatron_path is not None:
            sys.path.insert(0, os.path.join(self.args.megatron_path, 'examples/multimodal'))
        else:
            sys.path.insert(0, './examples/multimodal')
        from examples.multimodal.model import model_provider
        return model_provider

    def build_checkpoint_metadata(self, true_vocab_size):
        """
        Construct a simple namespace for all relevant model metadata.
        """
        md = super().build_checkpoint_metadata(true_vocab_size)

        try:
            from megatron.training.arguments import core_transformer_config_from_args
            from examples.multimodal.config import get_language_model_config, get_vision_model_config, get_vision_projection_config
        except ModuleNotFoundError:
            print("Unable to import Megatron, please specify the path to Megatron using --megatron-path. Exiting.")
            queue.put("exit")
            exit(1)

        # checkpoint_args.cp_comm_type = ["p2p"]
        base_config = core_transformer_config_from_args(self.checkpoint_args)
        base_config.language_model_type = self.margs.language_model_type
        base_config.vision_model_type = self.margs.vision_model_type

        language_config = get_language_model_config(deepcopy(base_config))

        vision_config = deepcopy(base_config)
        vision_config = get_vision_model_config(base_config, apply_query_key_layer_scaling=self.checkpoint_args.apply_query_key_layer_scaling)

        vision_projection_config = deepcopy(base_config)
        vision_projection_config = get_vision_projection_config(
            vision_projection_config, self.margs.hidden_size
        )

        md.num_query_groups = self.margs.num_query_groups
        md.kv_channels = self.margs.kv_channels
        # Swiglu is used to chunk linear layer weight in a specific way, and this is guarded by the
        # gated_linear_unit config in the MLP code.
        md.swiglu = self.margs.swiglu and language_config.gated_linear_unit
        md.previous_encoder_tensor_parallel_size = self.margs.tensor_model_parallel_size if self.margs.encoder_tensor_model_parallel_size == 0 else self.margs.encoder_tensor_model_parallel_size
        md.vision_model_type = self.margs.vision_model_type
        md.language_model_type = self.margs.language_model_type
        md.vision_projection_linear_bias = vision_projection_config.add_bias_linear
        md.vision_num_layers = vision_config.num_layers
        #TODO: check below line is actually correct, seems like it should be
        md.vision_swiglu = vision_config.gated_linear_unit
        md.vision_num_attention_heads = vision_config.num_attention_heads
        md.vision_kv_channels = vision_config.kv_channels
        md.vision_hidden_size = vision_config.hidden_size
        md.vision_dummy_head_count = self.args.vit_dummy_head_count
        md.vision_linear_bias = vision_config.add_bias_linear
        md.vision_qkv_bias = vision_config.add_qkv_bias
        md.padded_vocab_size = self.margs.padded_vocab_size
        if hasattr(vision_config, 'normalization'):
            md.vision_norm_has_bias = vision_config.normalization == "LayerNorm"
        else:
            # older models only supported LayerNorm
            md.vision_norm_has_bias = True
        
        return md

    def send_vision_backbone_over_queue(self, schema):
        """
        Using self.all_models, extract model parameters and send them over the queue.
        """
        from megatron.core import mpu

        vp_size = self.margs.virtual_pipeline_model_parallel_size or 1
        encoder_tp_size = self.md.previous_encoder_tensor_parallel_size

        if self.md.vision_model_type not in ("internvit", "siglip", "radio", "radio-g"):
            raise Exception(f'unrecognized vision model type: {md.vision_model_type}')

        message = {}
        if self.md.vision_model_type in ("internvit", "siglip"):
            message["conv1 weight"] = self.all_models[0][0][0].vision_model.conv1.weight.data
            if self.md.vision_model_type in ("internvit", "siglip"):
                message["conv1 bias"] = self.all_models[0][0][0].vision_model.conv1.bias.data
            message["position embeddings"] = self.all_models[0][0][0].vision_model.position_embeddings.weight.data

        if self.md.vision_model_type == "radio-g":
            message["mask token"] = self.all_models[0][0][0].vision_model.mask_token.data

        if self.md.vision_model_type in ("radio", "radio-g"):
            message["embedder weight"] = torch.cat([self.all_models[0][0][tp_rank].vision_model.embedder.weight.data for tp_rank in range(encoder_tp_size)], dim=0)
            if self.md.vision_model_type == "radio-g":
                message["embedder bias"] = torch.cat([self.all_models[0][0][tp_rank].vision_model.embedder.bias.data for tp_rank in range(encoder_tp_size)], dim=0)
            message["position embeddings"] = self.all_models[0][0][0].vision_model.position_embeddings.data

        if self.md.vision_model_type in ("siglip", "radio-g"):
            message["ln post weight"] = self.all_models[0][0][0].vision_model.ln_post.weight.data
            message["ln post bias"] = self.all_models[0][0][0].vision_model.ln_post.bias.data

        if self.md.vision_model_type in ("internvit", "radio", "radio-g"):
            message["class token"] = self.all_models[0][0][0].vision_model.class_token.data

        self.queue_put("vit embeddings", message)


        total_layer_num = 0
        #TODO: Maybe not worth dealing with 'weird' encoder tp sizes, but make sure works properly with different encoder tp sizes
        #TODO: Do I need this vp loop for vision model?
        #TODO: CHECK THAT PARAMS ARE THE SAME WITH OTHER ENCODERS, THSI SHOULD WORK FOR INTERNVIT BUT I SUSPECT WILL FAIL FOR SIGLIP
        for vp_rank in range(vp_size):
            mpu.set_virtual_pipeline_model_parallel_rank(vp_rank)
            # ViT will only ever be on first pp rank
            models = self.all_models[0][vp_rank]
            for layer_num in range(schema.get_num_layers(models[0])):
                message = {}

                # Get non-parallel tensors from tp_rank 0
                layer = schema.get_layer(models[0], layer_num)

                message["input norm weight"] = layer["self_attn_norm_weight"]
                message["pre mlp norm weight"] = layer["mlp_norm_weight"]
                if self.md.vision_norm_has_bias:
                    message["input norm bias"] = layer["self_attn_norm_bias"]
                    message["pre mlp norm bias"] = layer["mlp_norm_bias"]
                if self.md.vision_linear_bias:
                    message["dense bias"] = layer["self_attn_proj_bias"]
                    message["mlp l1 bias"] = layer["mlp_fc2_bias"]
                if self.md.vision_model_type in ("internvit", "radio-g"):
                    message["ls1"] = layer["ls1"]
                    message["ls2"] = layer["ls2"]

                # Grab all parallel tensors for this layer
                qkv_weight = []
                qkv_bias = []
                k_norm_weight = []
                k_norm_bias = []
                q_norm_weight = []
                q_norm_bias = []
                dense_weight = []
                mlp_l0_weight = []
                mlp_l0_bias = []
                mlp_l1_weight = []
                for tp_rank, model in enumerate(models):
                    layer = schema.get_layer(model, layer_num)
                    qkv_weight.append(layer["self_attn_qkv_weight"])
                    dense_weight.append(layer["self_attn_proj_weight"])
                    mlp_l0_weight.append(layer["mlp_fc1_weight"])
                    mlp_l1_weight.append(layer["mlp_fc2_weight"])
                    if self.md.vision_model_type == "internvit":
                        k_norm_weight.append(layer["k_layernorm_weight"])
                        q_norm_weight.append(layer["q_layernorm_weight"])
                        if self.md.vision_norm_has_bias:
                            k_norm_bias.append(layer["k_layernorm_bias"])
                            q_norm_bias.append(layer["q_layernorm_bias"])
                    if self.md.vision_qkv_bias:
                        qkv_bias.append(layer["self_attn_qkv_bias"])
                    if self.md.vision_linear_bias:
                        mlp_l0_bias.append(layer["mlp_fc1_bias"])

                # Handle gated linear units
                if self.md.vision_swiglu:
                    # concat all the first halves ('W's) and all the second halves ('V's)
                    for tp_rank in range(encoder_tp_size):
                        mlp_l0_weight[tp_rank] = torch.chunk(mlp_l0_weight[tp_rank], 2, dim=0)
                    message["mlp l0 weight W"] = torch.cat([w[0] for w in mlp_l0_weight], dim=0)
                    message["mlp l0 weight V"] = torch.cat([w[1] for w in mlp_l0_weight], dim=0)
                else:
                    message["mlp l0 weight"] = torch.cat(mlp_l0_weight, dim=0)

                # simple concat of the rest
                message["qkv weight"] = torch.cat(qkv_weight, dim=0)
                message["dense weight"] = torch.cat(dense_weight, dim=1)
                message["mlp l1 weight"] = torch.cat(mlp_l1_weight, dim=1)
                if self.md.vision_model_type == "internvit":
                    message["k norm weight"] = torch.cat(k_norm_weight, dim=0)
                    message["q norm weight"] = torch.cat(q_norm_weight, dim=0)
                if self.md.vision_qkv_bias:
                    message["qkv bias"] = torch.cat(qkv_bias, dim=0)
                if self.md.vision_linear_bias:
                    if self.md.vision_swiglu:
                        for tp_rank in range(encoder_tp_size):
                            mlp_l0_bias[tp_rank] = torch.chunk(mlp_l0_bias[tp_rank], 2, dim=0)
                        message["mlp l0 bias W"] = torch.cat([b[0] for b in mlp_l0_bias],dim=0)
                        message["mlp l0 bias V"] = torch.cat([b[1] for b in mlp_l0_bias],dim=0)
                    else:
                        message["mlp l0 bias"] = torch.cat(mlp_l0_bias, dim=0)
                if self.md.vision_norm_has_bias and self.md.vision_model_type == "internvit":
                    message["k norm bias"] = torch.cat(k_norm_bias, dim=0)
                    message["q norm bias"] = torch.cat(q_norm_bias, dim=0)

                self.queue_put(f"vit transformer layer {total_layer_num}", message)

                total_layer_num = total_layer_num + 1
    
    def send_vision_projection_over_queue(self):
        encoder_tp_size = self.md.previous_encoder_tensor_parallel_size
        message = {
            "vision projection l0 weight": torch.cat([self.all_models[0][0][tp_rank].vision_projection.encoder.linear_fc1.weight.data for tp_rank in range(encoder_tp_size)], dim=0),
            "vision projection l1 weight": torch.cat([self.all_models[0][0][tp_rank].vision_projection.encoder.linear_fc2.weight.data for tp_rank in range(encoder_tp_size)], dim=1),
        }
        # Check for this explicitly, since don't have any gurantees based on our model types
        # if hasattr(self.all_models[0][0][0].vision_projection.encoder.linear_fc1.layer_norm_weight, "data"):
        try:
            message["vision projection norm weight"] = self.all_models[0][0][0].vision_projection.encoder.linear_fc1.layer_norm_weight.data
        except:
            pass
        try:
        # if hasattr(self.all_models[0][0][0].vision_projection.encoder.linear_fc1.layer_norm_bias, "data"):
            message["vision projection norm bias"] = self.all_models[0][0][0].vision_projection.encoder.linear_fc1.layer_norm_bias.data
        except:
            pass
        if self.md.vision_projection_linear_bias:
            message["vision projection l0 bias"] = torch.cat([self.all_models[0][0][tp_rank].vision_projection.encoder.linear_fc1.bias.data for tp_rank in range(encoder_tp_size)], dim=0)
            message["vision projection l1 bias"] = self.all_models[0][0][0].vision_projection.encoder.linear_fc2.bias.data
        
        self.queue_put("vision projection", message)
        
    def send_model_over_queue(self):
        self.send_metadata_over_queue()

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
        self.send_vision_backbone_over_queue(schema_vision_backbone)

        self.send_vision_projection_over_queue()

        schema = get_model_schema(
            "GPT",
            self.margs.transformer_impl,
            self.margs.num_experts,
            self.margs.expert_model_parallel_size,
            prefix="language_model."
        )
        self.send_llm_over_queue(schema)
        self.queue.put("done")

def load_checkpoint(queue, args):
    """
    Required top-level function that creates the loader,
    calls its .load(), and handles exceptions by signaling 'exit'.
    """
    loader = MegatronCheckpointLoaderLLaVA(args, queue, build_tokenizer=True)
    try:
        loader.load()
    except Exception as e:
        queue.put("exit")
        raise e
