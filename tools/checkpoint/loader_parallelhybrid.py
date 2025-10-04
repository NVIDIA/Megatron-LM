# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import json
import os
import sys
import torch
import types

from loader_base import MegatronCheckpointLoaderBase


def add_arguments(parser):
    """Add command-line arguments relevant to Falcon-H1 model loading."""
    group = parser.add_argument_group(title='Falcon-H1 loader')

    group.add_argument('--true-vocab-size', type=int, default=None,
                       help='Original size of vocab; if specified, trims padding from embedding table.')
    group.add_argument('--vocab-file', type=str, default=None,
                       help='Path to a vocab file. If specified, determines vocab size to trim padding.')
    group.add_argument('--megatron-path', type=str, default=None,
                       help='Base directory of Megatron repository')
    group.add_argument('--position-embedding-type',
                       type=str,
                       default='learned_absolute',
                       choices=['learned_absolute', 'rope'],
                       help='Type of position embedding.')
    group.add_argument('--loader-transformer-impl', default='local',
                       choices=['local', 'transformer_engine'],
                       help='Which Transformer implementation to use.')


class MegatronCheckpointLoaderFalconH1(MegatronCheckpointLoaderBase):
    """
    Falcon-H1 specific checkpoint loader that handles hybrid architecture
    with alternating Mamba+Attention layers and MLP-only layers.
    
    Architecture:
    - Even layers (0,2,4,...): Hybrid (Mamba mixer + Self-attention)
    - Odd layers (1,3,5,...): MLP-only
    """

    def build_sys_argv(self):
        """
        Construct a sys.argv list for Megatron's argument parser.
        """
        return [
            *super().build_sys_argv(),
            '--position-embedding-type', self.args.position_embedding_type,
        ]

    def import_model_provider(self):
        """Return the Mamba model provider for Falcon-H1."""
        from pretrain_mamba import model_provider
        return model_provider

    def is_hybrid_layer(self, layer_idx):
        """Determine if a layer is hybrid (Mamba + Attention) or MLP-only."""
        return layer_idx % 2 == 0

    def extract_mamba_weights(self, model, layer_idx):
        """Extract Mamba mixer weights from a hybrid layer."""
        layer_name = f"decoder.layers.{layer_idx}.mamba_mixer"
        
        mamba_weights = {}
        
        # Get the mamba mixer module
        mamba_mixer = None
        for name, module in model.named_modules():
            if name == layer_name:
                mamba_mixer = module
                break
        
        if mamba_mixer is None:
            raise ValueError(f"Could not find mamba_mixer at layer {layer_idx}")
        
        # Extract Mamba-specific parameters
        mamba_weights["A_log"] = getattr(mamba_mixer, 'A_log', None)
        mamba_weights["D"] = getattr(mamba_mixer, 'D', None)
        mamba_weights["dt_bias"] = getattr(mamba_mixer, 'dt_bias', None)
        
        # Conv1D weights
        if hasattr(mamba_mixer, 'conv1d'):
            mamba_weights["conv1d_weight"] = mamba_mixer.conv1d.weight
            mamba_weights["conv1d_bias"] = mamba_mixer.conv1d.bias
        
        # Input and output projections
        if hasattr(mamba_mixer, 'in_proj'):
            mamba_weights["in_proj_weight"] = mamba_mixer.in_proj.weight
            # Note: pre_norm_weight is extracted separately above
        
        if hasattr(mamba_mixer, 'out_proj'):
            mamba_weights["out_proj_weight"] = mamba_mixer.out_proj.weight
        
        # Norm weights - GET BOTH TYPES
        if hasattr(mamba_mixer, 'norm'):
            mamba_weights["internal_norm_weight"] = mamba_mixer.norm.weight
        
        # Pre-norm weight (from in_proj layer norm)
        if hasattr(mamba_mixer, 'in_proj') and hasattr(mamba_mixer.in_proj, 'layer_norm_weight'):
            mamba_weights["pre_norm_weight"] = mamba_mixer.in_proj.layer_norm_weight
        
        return mamba_weights

    def extract_attention_weights(self, model, layer_idx):
        """Extract self-attention weights from a hybrid layer."""
        layer_name = f"decoder.layers.{layer_idx}.self_attention"
        
        attention_weights = {}
        
        # Get the self attention module
        self_attention = None
        for name, module in model.named_modules():
            if name == layer_name:
                self_attention = module
                break
        
        if self_attention is None:
            raise ValueError(f"Could not find self_attention at layer {layer_idx}")
        
        # QKV projection
        if hasattr(self_attention, 'linear_qkv'):
            attention_weights["qkv_weight"] = self_attention.linear_qkv.weight
            attention_weights["qkv_norm_weight"] = getattr(self_attention.linear_qkv, 'layer_norm_weight', None)
        
        # Output projection
        if hasattr(self_attention, 'linear_proj'):
            attention_weights["proj_weight"] = self_attention.linear_proj.weight
        
        return attention_weights

    def extract_mlp_weights(self, model, layer_idx):
        """Extract MLP weights from an MLP-only layer."""
        layer_name = f"decoder.layers.{layer_idx}.mlp"
        
        mlp_weights = {}
        
        # Get the MLP module
        mlp = None
        for name, module in model.named_modules():
            if name == layer_name:
                mlp = module
                break
        
        if mlp is None:
            raise ValueError(f"Could not find mlp at layer {layer_idx}")
        
        # FC1 (first linear layer)
        if hasattr(mlp, 'linear_fc1'):
            mlp_weights["fc1_weight"] = mlp.linear_fc1.weight
            mlp_weights["fc1_norm_weight"] = getattr(mlp.linear_fc1, 'layer_norm_weight', None)
        
        # FC2 (second linear layer)  
        if hasattr(mlp, 'linear_fc2'):
            mlp_weights["fc2_weight"] = mlp.linear_fc2.weight
        
        return mlp_weights

    def send_model_over_queue(self):
        """Send Falcon-H1 model over the queue with proper hybrid layer handling."""
        # Send metadata first
        self.send_metadata_over_queue()

        # Get model parameters
        tp_size = self.margs.tensor_model_parallel_size
        pp_size = self.margs.pipeline_model_parallel_size
        vp_size = self.margs.virtual_pipeline_model_parallel_size or 1

        # Get first pipeline models for embeddings/final norm
        first_pipeline_models = self.all_models[0][0]

        # 1) Send embeddings
        message = {}
        for i, model in enumerate(first_pipeline_models):
            # Extract embedding weights
            for name, param in model.named_parameters():
                if 'embedding.word_embeddings.weight' in name:
                    if i == 0:
                        message["word embeddings"] = param
                    else:
                        message["word embeddings"] = torch.cat([message["word embeddings"], param], dim=0)
                elif 'position_embeddings.weight' in name and self.md.position_embedding_type == 'learned_absolute':
                    if i == 0:  # Only take from rank 0
                        message["position embeddings"] = param
        
        if "position embeddings" not in message:
            message["position embeddings"] = None
            
        self.queue_put("embeddings", message)

        # 2) Process each layer based on type
        total_layer_num = 0
        for vp_rank in range(vp_size):
            for pp_rank in range(pp_size):
                models = self.all_models[pp_rank][vp_rank]
                
                # Determine number of layers in this model shard
                model = models[0]
                layer_count = 0
                max_layer_idx = -1
                for name, _ in model.named_parameters():
                    if 'decoder.layers.' in name:
                        # Extract layer index
                        parts = name.split('.')
                        if len(parts) > 2 and parts[2].isdigit():
                            layer_idx = int(parts[2])
                            max_layer_idx = max(max_layer_idx, layer_idx)
                
                num_layers = max_layer_idx + 1 if max_layer_idx >= 0 else 0
                
                for layer_idx in range(num_layers):
                    if self.is_hybrid_layer(total_layer_num):
                        # Process hybrid layer (Mamba + Attention)
                        message = {}
                        
                        # Collect Mamba weights across TP ranks
                        mamba_weights_per_rank = []
                        attention_weights_per_rank = []
                        
                        for model_tp in models:
                            mamba_weights = self.extract_mamba_weights(model_tp, layer_idx)
                            attention_weights = self.extract_attention_weights(model_tp, layer_idx)
                            mamba_weights_per_rank.append(mamba_weights)
                            attention_weights_per_rank.append(attention_weights)
                        
                        # Mamba components (typically not sharded across TP)
                        message["mamba A_log"] = mamba_weights_per_rank[0]["A_log"]
                        message["mamba D"] = mamba_weights_per_rank[0]["D"] 
                        message["mamba dt_bias"] = mamba_weights_per_rank[0]["dt_bias"]
                        message["mamba conv1d weight"] = mamba_weights_per_rank[0]["conv1d_weight"]
                        message["mamba conv1d bias"] = mamba_weights_per_rank[0]["conv1d_bias"]
                        message["mamba pre norm weight"] = mamba_weights_per_rank[0]["pre_norm_weight"]
                        message["mamba internal norm weight"] = mamba_weights_per_rank[0]["internal_norm_weight"]
                        
                        # Mamba projections (may be sharded)
                        if len(mamba_weights_per_rank) > 1 and mamba_weights_per_rank[1]["in_proj_weight"] is not None:
                            # Concatenate across TP ranks
                            message["mamba in_proj weight"] = torch.cat([w["in_proj_weight"] for w in mamba_weights_per_rank], dim=0)
                            message["mamba out_proj weight"] = torch.cat([w["out_proj_weight"] for w in mamba_weights_per_rank], dim=1)
                        else:
                            message["mamba in_proj weight"] = mamba_weights_per_rank[0]["in_proj_weight"]
                            message["mamba out_proj weight"] = mamba_weights_per_rank[0]["out_proj_weight"]
                        
                        # Attention components (sharded across TP)
                        message["attention input norm weight"] = attention_weights_per_rank[0]["qkv_norm_weight"]
                        
                        # Concatenate QKV and dense weights across TP ranks
                        if len(attention_weights_per_rank) > 1:
                            message["attention qkv weight"] = torch.cat([w["qkv_weight"] for w in attention_weights_per_rank], dim=0)
                            message["attention dense weight"] = torch.cat([w["proj_weight"] for w in attention_weights_per_rank], dim=1)
                        else:
                            message["attention qkv weight"] = attention_weights_per_rank[0]["qkv_weight"]
                            message["attention dense weight"] = attention_weights_per_rank[0]["proj_weight"]
                        
                        self.queue_put(f"hybrid layer {total_layer_num}", message)
                        
                    else:
                        # Process MLP-only layer
                        message = {}
                        
                        # Collect MLP weights across TP ranks
                        mlp_weights_per_rank = []
                        for model_tp in models:
                            mlp_weights = self.extract_mlp_weights(model_tp, layer_idx)
                            mlp_weights_per_rank.append(mlp_weights)
                        
                        # MLP norm (not sharded)
                        message["mlp input norm weight"] = mlp_weights_per_rank[0]["fc1_norm_weight"]
                        
                        # MLP weights (sharded across TP)
                        if len(mlp_weights_per_rank) > 1:
                            message["mlp fc1 weight"] = torch.cat([w["fc1_weight"] for w in mlp_weights_per_rank], dim=0)
                            message["mlp fc2 weight"] = torch.cat([w["fc2_weight"] for w in mlp_weights_per_rank], dim=1)
                        else:
                            message["mlp fc1 weight"] = mlp_weights_per_rank[0]["fc1_weight"]
                            message["mlp fc2 weight"] = mlp_weights_per_rank[0]["fc2_weight"]
                        
                        self.queue_put(f"mlp layer {total_layer_num}", message)
                    
                    total_layer_num += 1

        # 3) Send final norm
        message = {}
        for name, param in models[0].named_parameters():
            if 'decoder.final_norm.weight' in name:
                message["weight"] = param
                break
        self.queue_put("final norm", message)

        # 4) Send output layer
        if self.md.output_layer:
            message = {}
            output_weights = []
            for model in models:
                for name, param in model.named_parameters():
                    if 'output_layer.weight' in name:
                        output_weights.append(param)
                        break
            
            if output_weights:
                if len(output_weights) > 1:
                    message["weight"] = torch.cat(output_weights, dim=0)
                else:
                    message["weight"] = output_weights[0]
                self.queue_put("output layer", message)

        self.queue.put("done")


def load_checkpoint(queue, args):
    """
    Required top-level function that creates the loader,
    calls its .load(), and handles exceptions by signaling 'exit'.
    """
    loader = MegatronCheckpointLoaderFalconH1(args, queue)
    try:
        loader.load()
    except Exception as e:
        queue.put("exit")
        raise e