# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import json
import os
import sys
import types
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from loader_hf_hybrid import HuggingFaceCheckpointLoaderHybrid
from loader_base import MegatronCheckpointLoaderBase


def add_arguments(parser):
    """Add command-line arguments relevant to HuggingFace model loading."""
    group = parser.add_argument_group(title='HuggingFace loader')
    
    group.add_argument('--true-vocab-size', type=int, default=None,
                       help='Original size of vocab; if specified, trims padding from embedding table.')
    group.add_argument('--megatron-path', type=str, default=None,
                       help='Base directory of Megatron repository')
    group.add_argument('--tokenizer-model', type=str, default=None,
                       help='Tokenizer model file.')
    group.add_argument('--tokenizer-type', type=str, default="MultimodalTokenizer",
                       help='Tokenizer type.')
    group.add_argument('--tokenizer-prompt-format', type=str, default="nemotron6-moe",
                       help='Tokenizer prompt format.')
    group.add_argument('--target-tensor-parallel-size', type=int,
                       help='Target tensor model parallel size, defaults to the tensor parallel size '
                       'in the input checkpoint if provided by the loader, otherwise to 1')


class HuggingFaceCheckpointLoaderMoELLaVA(HuggingFaceCheckpointLoaderHybrid):
    def __init__(self, args, queue, build_tokenizer=False):
        super().__init__(args, queue, build_tokenizer)
        self.hf_model = None
        self.hf_config = None

    def parse_megatron_args(self):
        """
        Parse Megatron arguments by loading HF config and building equivalent args.
        """
        # Ensure we can import Megatron
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
        if self.args.megatron_path is not None:
            sys.path.insert(0, self.args.megatron_path)

        try:
            from megatron.training.arguments import parse_args, validate_args
        except ModuleNotFoundError:
            print("Unable to import Megatron. Please specify --megatron-path. Exiting.")
            self.queue.put("exit")
            sys.exit(1)

        # Load HF config
        self.hf_config = AutoConfig.from_pretrained(self.args.load_dir, trust_remote_code=True)
        
        # Build sys.argv based on HF config
        sys.argv = self.build_sys_argv()

        margs = parse_args()
        
        # Create fake checkpoint args based on HF config
        checkpoint_args = types.SimpleNamespace()
        checkpoint_args.fp16 = self.hf_config.torch_dtype == torch.float16
        checkpoint_args.bf16 = self.hf_config.torch_dtype == torch.bfloat16
        checkpoint_args.normalization = "RMSNorm"  # Based on config showing rms_norm_eps
        checkpoint_args.sequence_parallel = False
        checkpoint_args.apply_query_key_layer_scaling = False
        checkpoint_args.ffn_hidden_size = self.hf_config.llm_config.intermediate_size
        checkpoint_args.num_attention_heads = self.hf_config.llm_config.num_attention_heads
        checkpoint_args.num_query_groups = self.hf_config.llm_config.num_key_value_heads
        checkpoint_args.kv_channels = self.hf_config.llm_config.head_dim
        checkpoint_args.group_query_attention = checkpoint_args.num_query_groups < self.hf_config.llm_config.num_attention_heads
        checkpoint_args.position_embedding_type = "none"
        if self.hf_config.llm_config.mlp_hidden_act == "relu2":
            checkpoint_args.squared_relu = True
        if self.args.model_type == "hybrid":
            checkpoint_args.spec = ['megatron.core.models.mamba.mamba_layer_specs', 'mamba_stack_spec']
            checkpoint_args.hybrid_attention_ratio = 0.0 # Will be computed from pattern
            checkpoint_args.hybrid_mlp_ratio = 0.0
            checkpoint_args.hybrid_override_pattern = self.hf_config.llm_config.hybrid_override_pattern
            checkpoint_args.mamba_state_dim = self.hf_config.llm_config.ssm_state_size
            checkpoint_args.mamba_num_groups = self.hf_config.llm_config.n_groups
            checkpoint_args.mamba_head_dim = self.hf_config.llm_config.mamba_head_dim
            checkpoint_args.mamba_num_heads = self.hf_config.llm_config.mamba_num_heads
            checkpoint_args.is_hybrid_model = True
        checkpoint_args.num_experts = self.hf_config.llm_config.n_routed_experts
        checkpoint_args.moe_router_topk = self.hf_config.llm_config.num_experts_per_tok
        checkpoint_args.moe_shared_expert_intermediate_size = self.hf_config.llm_config.moe_shared_expert_intermediate_size
        checkpoint_args.moe_router_topk_scaling_factor = self.hf_config.llm_config.routed_scaling_factor
        checkpoint_args.moe_router_enable_expert_bias = True
        checkpoint_args.moe_router_score_function = "sigmoid"
        checkpoint_args.tokenizer_model = self.args.tokenizer_model
        checkpoint_args.tokenizer_prompt_format = self.args.tokenizer_prompt_format

        # TODO: Hard coding these for now
        checkpoint_args.language_model_type = "nemotron6-moe"
        checkpoint_args.vision_model_type = "radio"
        checkpoint_args.pixel_shuffle = True
        
        # Set key attributes from HF config
        margs.num_layers = self.hf_config.llm_config.num_hidden_layers
        margs.hidden_size = self.hf_config.llm_config.hidden_size
        margs.ffn_hidden_size = self.hf_config.llm_config.intermediate_size
        margs.num_attention_heads = self.hf_config.llm_config.num_attention_heads
        margs.num_query_groups = self.hf_config.llm_config.num_key_value_heads
        margs.kv_channels = self.hf_config.llm_config.head_dim
        margs.group_query_attention = checkpoint_args.group_query_attention
        margs.seq_length = 2048 # Default, will be overridden
        margs.max_position_embeddings = self.hf_config.llm_config.max_position_embeddings
        margs.iteration = 1  # Dummy value
        margs.params_dtype = self.hf_config.torch_dtype
        margs.add_bias_linear = self.hf_config.llm_config.use_bias
        margs.add_qkv_bias = self.hf_config.llm_config.attention_bias
        margs.swiglu = self.hf_config.llm_config.mlp_hidden_act == "swiglu"  # Check actual activation
        margs.untie_embeddings_and_output_weights = not self.hf_config.llm_config.tie_word_embeddings
        margs.bert_binary_head = False
        margs.tokenizer_type = self.args.tokenizer_type
        margs.tokenizer_model = self.args.tokenizer_model
        margs.tokenizer_prompt_format = self.args.tokenizer_prompt_format
        margs.position_embedding_type = "none"
        margs.make_vocab_size_divisible_by = 128
        margs.vocab_size = self.hf_config.llm_config.vocab_size
        margs.padded_vocab_size = self.hf_config.llm_config.vocab_size

        # Adjust world size so validation doesn't fail
        margs.world_size = 1
        margs.tensor_model_parallel_size = 1
        margs.pipeline_model_parallel_size = 1
        margs.expert_model_parallel_size = 1
        margs.expert_tensor_parallel_size = 1
        margs.data_parallel_size = 1
        margs.context_parallel_size = 1
        margs.micro_batch_size = 1
        margs.global_batch_size = 1
        margs.virtual_pipeline_model_parallel_size = None

        margs.use_legacy_models = False
        margs.transformer_impl = "local"
        margs.no_persist_layer_norm = True

        margs.use_cpu_initialization = False

        self.margs = margs
        self.checkpoint_args = checkpoint_args

    def build_sys_argv(self):
        """
        Construct a sys.argv list for Megatron's argument parser.
        """
        # Build base arguments and add hybrid-specific ones
        base_args = MegatronCheckpointLoaderBase.build_sys_argv(self)
        
        hybrid_args = [
            '--position-embedding-type', 'none',
            '--hybrid-override-pattern', self.hf_config.llm_config.hybrid_override_pattern,
            '--mamba-state-dim', str(self.hf_config.llm_config.ssm_state_size),
            '--mamba-num-groups', str(self.hf_config.llm_config.n_groups),
            '--mamba-head-dim', str(self.hf_config.llm_config.mamba_head_dim),
            '--mamba-num-heads', str(self.hf_config.llm_config.mamba_num_heads),
        ]
        
        return base_args + hybrid_args

    def build_checkpoint_metadata(self, true_vocab_size):
        """
        Construct metadata based on HuggingFace config.
        """
        md = types.SimpleNamespace()
        md.model_type = "hybrid"
        md.num_layers = self.hf_config.llm_config.num_hidden_layers
        md.hidden_size = self.hf_config.llm_config.hidden_size
        md.seq_length = 2048 # Default will be overridden
        md.decoder_seq_length = 16384 # Default will be overridden
        md.num_attention_heads = self.hf_config.llm_config.num_attention_heads
        md.kv_channels = self.hf_config.llm_config.head_dim
        md.num_query_groups = self.hf_config.llm_config.num_key_value_heads
        md.max_position_embeddings = self.hf_config.llm_config.max_position_embeddings
        md.tokenizer_type = self.args.tokenizer_type
        md.tokenizer_model = self.args.tokenizer_model
        md.tokenizer_prompt_format = self.args.tokenizer_prompt_format
        md.iteration = 1
        md.params_dtype = self.hf_config.llm_config.torch_dtype
        md.bert_binary_head = False
        md.output_layer = not self.hf_config.llm_config.tie_word_embeddings
        md.position_embedding_type = "none"
        md.linear_bias = self.hf_config.llm_config.use_bias
        md.qkv_bias = self.hf_config.llm_config.attention_bias
        md.norm_has_bias = False  # RMSNorm typically doesn't have bias
        md.swiglu = False
        md.previous_tensor_parallel_size = 1
        md.previous_pipeline_parallel_size = 1
        md.true_vocab_size = true_vocab_size or self.hf_config.llm_config.vocab_size
        md.make_vocab_size_divisible_by = 128
        md.padded_vocab_size = self.hf_config.llm_config.vocab_size
        md.vocab_size = md.true_vocab_size # TODO: not sure if this is correct
        md.language_model_type = "nemotron6-moe"
        md.checkpoint_args = self.checkpoint_args
        md.use_legacy_models = False
        md.use_cpu_initialization = False
        
        # Hybrid-specific metadata
        if self.args.model_type == "hybrid":
            md.hybrid_attention_ratio = None
            md.hybrid_mlp_ratio = None
            md.hybrid_override_pattern = self.hf_config.llm_config.hybrid_override_pattern
            md.mamba_state_dim = self.hf_config.llm_config.ssm_state_size
            md.mamba_num_groups = self.hf_config.llm_config.n_groups
            md.mamba_head_dim = self.hf_config.llm_config.mamba_head_dim
            md.mamba_num_heads = self.hf_config.llm_config.mamba_num_heads

        md.num_experts = self.hf_config.llm_config.n_routed_experts
        md.moe_router_topk = self.hf_config.llm_config.num_experts_per_tok
        md.moe_shared_expert_intermediate_size = self.hf_config.llm_config.moe_shared_expert_intermediate_size
        md.moe_router_topk_scaling_factor = self.hf_config.llm_config.routed_scaling_factor

        # TODO: hard coding these for now
        md.vision_model_type = "radio"
        md.vision_norm_has_bias = True
        md.vision_swiglu = False
        md.vision_hidden_size = self.hf_config.vit_hidden_size
        md.vision_qkv_bias = True
        md.vision_linear_bias = True

        md.vision_projection_linear_bias = False
        md.conv_merging = False
        
        md.moe_router_topk_scaling_factor = None
        md.moe_router_dtype = None
        md.moe_router_padding_for_fp8 = False
        md.moe_router_num_groups = self.hf_config.llm_config.n_groups
        md.moe_router_group_topk = self.hf_config.llm_config.num_experts_per_tok
        md.moe_router_pre_softmax = False
        md.moe_router_topk_scaling_factor = None
        md.moe_router_dtype = None
        md.moe_router_padding_for_fp8 = False
        md.moe_router_num_groups = self.hf_config.llm_config.n_groups
        md.moe_router_group_topk = self.hf_config.llm_config.num_experts_per_tok
        md.moe_router_pre_softmax = False
        md.moe_router_enable_expert_bias = True
        md.moe_router_score_function = "sigmoid"
        return md

    def compute_true_vocab_size(self):
        """Determine the 'true' (non-padded) vocab size."""
        if self.args.true_vocab_size is not None:
            return self.args.true_vocab_size
        else:
            return self.hf_config.llm_config.vocab_size

    def send_model_over_queue(self):
        """
        Send the HuggingFace model over the queue in Megatron format.
        """
        self.send_metadata_over_queue()
        
        self.send_hf_vision_backbone_over_queue()
        self.send_hf_vision_projection_over_queue()
        self.send_hf_moe_lm_over_queue()
        self.queue.put("done")

    def send_hf_vision_backbone_over_queue(self):
        """
        Send the HuggingFace vision backbone over the queue in Megatron format.
        """
        # Embeddings (RADIO)
        vit_msg = {}
        vit_msg["embedder weight"] = self.hf_model.vision_model.radio_model.model.patch_generator.embedder.weight.detach()
        vit_msg["class token"] = self.hf_model.vision_model.radio_model.model.patch_generator.cls_token.token.detach()
        vit_msg["position embeddings"] = self.hf_model.vision_model.radio_model.model.patch_generator.pos_embed.detach()
        # Normalization constants for RADIO input conditioner (unused by Megatron saver but harmless to include)
        vit_msg["input_conditioner_norm_mean"] = torch.tensor([0.48145466, 0.4578275, 0.40821073]).unsqueeze(-1).unsqueeze(-1)
        vit_msg["input_conditioner_norm_std"] = torch.tensor([0.26862954, 0.26130258, 0.27577711]).unsqueeze(-1).unsqueeze(-1)
        self.queue_put("vit embeddings", vit_msg)

        # Transformer blocks (RADIO)
        blocks = self.hf_model.vision_model.radio_model.model.blocks
        num_layers = len(blocks)
        for layer_idx in range(num_layers):
            block = blocks[layer_idx]
            msg = {}

            # Pre-attention norm
            msg["input norm weight"] = block.norm1.weight.detach()
            if hasattr(block.norm1, 'bias') and block.norm1.bias is not None:
                msg["input norm bias"] = block.norm1.bias.detach()

            # Attention projections (qkv combined, and output proj)
            # Inverse of saver_hf_llava RADIO ordering: hf_qkv = megatron_qkv[order]
            # Here we compute inv_order s.t. megatron_qkv = hf_qkv[inv_order]
            hidden_size = block.attn.proj.weight.shape[0]
            num_heads = getattr(block.attn, 'num_heads', None)
            if num_heads is None:
                num_heads = getattr(block.attn, 'heads', None)
            if num_heads is None or hidden_size % num_heads != 0:
                raise ValueError("Unable to determine vision attention num_heads for RADIO reordering")
            dim = hidden_size // num_heads
            order = torch.ones(3 * hidden_size, dtype=torch.long)
            for j in range(num_heads):
                base = dim * 3 * j
                for i in range(dim):
                    order[j * dim + i] = i + base
                    order[j * dim + i + num_heads * dim] = dim + i + base
                    order[j * dim + i + num_heads * dim * 2] = 2 * dim + i + base
            inv_order = torch.empty_like(order)
            inv_order[order] = torch.arange(order.numel(), dtype=torch.long)

            qkv_w = block.attn.qkv.weight.detach()[inv_order]
            msg["qkv weight"] = qkv_w
            if hasattr(block.attn.qkv, 'bias') and block.attn.qkv.bias is not None:
                qkv_b = block.attn.qkv.bias.detach()[inv_order]
                msg["qkv bias"] = qkv_b
            msg["dense weight"] = block.attn.proj.weight.detach()
            if hasattr(block.attn.proj, 'bias') and block.attn.proj.bias is not None:
                msg["dense bias"] = block.attn.proj.bias.detach()

            # Pre-MLP norm
            msg["pre mlp norm weight"] = block.norm2.weight.detach()
            if hasattr(block.norm2, 'bias') and block.norm2.bias is not None:
                msg["pre mlp norm bias"] = block.norm2.bias.detach()

            # MLP
            msg["mlp l0 weight"] = block.mlp.fc1.weight.detach()
            if hasattr(block.mlp.fc1, 'bias') and block.mlp.fc1.bias is not None:
                msg["mlp l0 bias"] = block.mlp.fc1.bias.detach()
            msg["mlp l1 weight"] = block.mlp.fc2.weight.detach()
            if hasattr(block.mlp.fc2, 'bias') and block.mlp.fc2.bias is not None:
                msg["mlp l1 bias"] = block.mlp.fc2.bias.detach()

            self.queue_put(f"vit transformer layer {layer_idx}", msg)

    def send_hf_vision_projection_over_queue(self):
        """
        Send the HuggingFace vision projection over the queue in Megatron format.
        """
        message = {}
        message["vision projection norm weight"] = self.hf_model.mlp1[0].weight
        if hasattr(self.hf_model.mlp1[0], 'bias'):
            message["vision projection norm bias"] = self.hf_model.mlp1[0].bias
        message["vision projection l0 weight"] = self.hf_model.mlp1[1].weight
        message["vision projection l1 weight"] = self.hf_model.mlp1[3].weight
        if hasattr(self.hf_model.mlp1[1], 'bias'):
            message["vision projection l0 bias"] = self.hf_model.mlp1[1].bias
        if hasattr(self.hf_model.mlp1[3], 'bias'):
            message["vision projection l1 bias"] = self.hf_model.mlp1[3].bias
        self.queue_put("vision projection", message)

    def send_hf_moe_lm_over_queue(self):
        """
        Convert HuggingFace hybrid model weights to Megatron format and send over queue.
        """
        model = self.hf_model.language_model
        
        # 1) Embeddings
        word_embeddings = model.backbone.embeddings.weight
        message = {
            "word embeddings": word_embeddings
        }
        # No position embeddings for RoPE
        self.queue_put("embeddings", message)

        # 2) Determine layer types by inspecting actual model weights
        layer_types = []
        for i in range(self.hf_config.llm_config.num_hidden_layers):
            layer_weights = model.backbone.layers[i].mixer
            
            # Determine layer type by checking what weights exist
            if hasattr(layer_weights, 'A_log'):
                layer_types.append('MAMBA')
            elif hasattr(layer_weights, 'q_proj'):
                layer_types.append('ATTENTION')  
            elif hasattr(layer_weights, 'up_proj'):
                layer_types.append('MLP')
            elif hasattr(layer_weights, 'gate'):
                layer_types.append('MOE')
            else:
                raise ValueError(f"Couldn't detect layer type of layer {i}")

        # 3) Send transformer layers
        for layer_idx in range(self.hf_config.llm_config.num_hidden_layers):
            layer = model.backbone.layers[layer_idx]
            layer_type = layer_types[layer_idx]
            message = {}

            if layer_type == 'MAMBA':
                # Mamba layer weights
                message["in proj norm weight"] = layer.norm.weight
                message["dt bias"] = layer.mixer.dt_bias
                message["D"] = layer.mixer.D
                message["A log"] = layer.mixer.A_log
                message["in proj weight"] = layer.mixer.in_proj.weight
                message["conv1d weight"] = layer.mixer.conv1d.weight
                message["conv1d bias"] = layer.mixer.conv1d.bias
                message["norm weight"] = layer.mixer.norm.weight
                message["out proj weight"] = layer.mixer.out_proj.weight
                
            elif layer_type == 'ATTENTION':
                # Attention layer weights
                message["input norm weight"] = layer.norm.weight
                
                # Combine q, k, v projections into qkv weight
                q_weight = layer.mixer.q_proj.weight
                k_weight = layer.mixer.k_proj.weight  
                v_weight = layer.mixer.v_proj.weight
                
                # Calculate head dimension for attention layers
                # head_dim = self.hf_config.llm_config.hidden_size // self.hf_config.llm_config.num_attention_heads
                head_dim = self.hf_config.llm_config.head_dim
                qkv_weight = self.combine_hf_qkv_weight(q_weight, k_weight, v_weight, self.hf_config.llm_config.num_attention_heads, self.hf_config.llm_config.num_key_value_heads, head_dim, self.args.target_tensor_parallel_size)
                
                message["qkv weight"] = qkv_weight
                message["dense weight"] = layer.mixer.o_proj.weight
                
                # Add bias if present
                if self.hf_config.llm_config.attention_bias:
                    q_bias = layer.mixer.q_proj.bias if hasattr(layer.mixer.q_proj, 'bias') else None
                    k_bias = layer.mixer.k_proj.bias if hasattr(layer.mixer.k_proj, 'bias') else None
                    v_bias = layer.mixer.v_proj.bias if hasattr(layer.mixer.v_proj, 'bias') else None
                    
                    if q_bias is not None and k_bias is not None and v_bias is not None:
                        qkv_bias = self.combine_hf_qkv_bias(q_bias, k_bias, v_bias, self.hf_config.llm_config.num_attention_heads, self.hf_config.llm_config.num_key_value_heads, head_dim, self.args.target_tensor_parallel_size)
                        message["qkv bias"] = qkv_bias
                        
                if hasattr(layer.mixer.o_proj, 'bias') and layer.mixer.o_proj.bias is not None:
                    message["dense bias"] = layer.mixer.o_proj.bias
                
            elif layer_type == 'MLP':
                # MLP layer weights - this model uses ReLU^2 activation, not SwiGLU
                message["post norm weight"] = layer.norm.weight
                
                # For this model, it seems to be standard MLP (not SwiGLU)
                # up_proj corresponds to the first linear layer, down_proj to the second
                message["mlp l0 weight"] = layer.mixer.up_proj.weight
                message["mlp l1 weight"] = layer.mixer.down_proj.weight
                # Add bias if present
                if hasattr(layer.mixer, 'up_proj') and hasattr(layer.mixer.up_proj, 'bias') and layer.mixer.up_proj.bias is not None:
                    message["mlp l0 bias"] = layer.mixer.up_proj.bias
                if hasattr(layer.mixer, 'down_proj') and hasattr(layer.mixer.down_proj, 'bias') and layer.mixer.down_proj.bias is not None:
                    message["mlp l1 bias"] = layer.mixer.down_proj.bias
            elif layer_type == 'MOE':
                # MoE layer weights
                message["pre mlp norm weight"] = layer.norm.weight
                if hasattr(layer.norm, 'bias') and layer.norm.bias is not None:
                    message["pre mlp norm bias"] = layer.norm.bias
                message["router weight"] = layer.mixer.gate.weight
                message["router bias"] = layer.mixer.gate.e_score_correction_bias
                message["shared mlp l0 weight"] = layer.mixer.shared_experts.up_proj.weight
                message["shared mlp l1 weight"] = layer.mixer.shared_experts.down_proj.weight

                experts_up = []
                experts_down = []
                # Recombine experts
                for expert_idx in range(self.hf_config.llm_config.n_routed_experts):
                    experts_up.append(layer.mixer.experts[expert_idx].up_proj.weight)
                    experts_down.append(layer.mixer.experts[expert_idx].down_proj.weight)
                message["mlp l0 weight"] = torch.stack(experts_up, dim=0)
                message["mlp l1 weight"] = torch.stack(experts_down, dim=0)

            message = {k: v.detach() for k, v in message.items()}
            self.queue_put(f"transformer layer {layer_idx}", message)

        # 4) Final norm
        message = {
            "weight": model.backbone.norm_f.weight
        }
        self.queue_put("final norm", message)

        # 5) Output layer
        if self.md.output_layer:
            message = {
                "weight": model.lm_head.weight
            }
            self.queue_put("output layer", message)



def load_checkpoint(queue, args):
    """
    Required top-level function that creates the loader,
    calls its .load(), and handles exceptions by signaling 'exit'.
    """
    loader = HuggingFaceCheckpointLoaderMoELLaVA(args, queue)
    try:
        loader.load()
    except Exception as e:
        queue.put("exit")
        raise e 