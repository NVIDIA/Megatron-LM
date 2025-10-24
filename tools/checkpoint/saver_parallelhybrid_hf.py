import sys
import os
import gc
import math
import json
from pathlib import Path
from shutil import rmtree

import torch
import torch.multiprocessing as mp
from transformers import (
    AutoModelForCausalLM,
    FalconH1Config,
    FalconH1ForCausalLM,
    GenerationConfig,
)

sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__),
                    os.path.pardir,
                    os.path.pardir)))
try:
    from megatron.training.tokenizer.tokenizer import _vocab_size_with_padding
except ModuleNotFoundError:
    print("Unable to import Megatron. Exiting.")
    exit(1)

def add_arguments(parser):
    group = parser.add_argument_group(title="Parallel Hybrid HF saver.")
    group.add_argument(
        "--hf-tokenizer",
        type=str,
        default=None,
        help="HF tokenizer (example: tiiuae/Falcon-H1-0.5B-Instruct)",
    )
    group.add_argument(
        "--check-eq-hf",
        type=str,
        default=None,
        help="check equality with HF model, example: tiiuae/Falcon-H1-1.5B-Instruct",
    )
    group.add_argument(
        "--save-chat-model",
        action='store_true',
        help="flag to save chat model or not",
    )

def perform_check(
    state_dict: dict[str, torch.Tensor], ref_state_dict: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """
    Given a reference state dict, check that state_dict is equal to it
    then pop the keys from ref_state_dict
    """
    for key in state_dict:
        if key in ref_state_dict:
            if not torch.equal(ref_state_dict[key], state_dict[key]):
                print(f"Warning: Mismatch found in {key}")
            ref_state_dict.pop(key)
        else:
            print(f"Warning: Key {key} not found in reference model")
    return ref_state_dict

def save_layer(
    state_dict: dict[str, torch.Tensor],
    index_dict: dict,
    dir_path: str,
    filename: str,
    check_reference: bool = False,
    ref_state_dict: dict[str, torch.Tensor] = None,
) -> tuple[dict, dict[str, torch.Tensor]]:
    """check state dict against a reference one if needed
    update index_dict
    save state dict
    """
    if check_reference and ref_state_dict is not None:
        ref_state_dict = perform_check(state_dict, ref_state_dict)
    for layer_name, weight_matrix in state_dict.items():
        index_dict["weight_map"][layer_name] = filename
        index_dict["metadata"]["total_size"] += weight_matrix.numel()
    print(f"saving state dict to {dir_path}/{filename}")
    torch.save(state_dict, f"{dir_path}/{filename}")
    return index_dict, ref_state_dict

def is_hybrid_layer(layer_idx: int) -> bool:
    """Determine if a layer is hybrid (Mamba + Attention) or MLP-only"""
    return layer_idx % 2 == 0

def process_hybrid_layer_weights(message: dict, layer_idx: int, falcon_h1_config: FalconH1Config) -> dict[str, torch.Tensor]:
    """Process weights for hybrid layers (Mamba + Attention)"""
    state_dict = {}
    
    # Mamba mixer components
    state_dict[f"model.layers.{layer_idx}.mamba.A_log"] = message["mamba A_log"]
    state_dict[f"model.layers.{layer_idx}.mamba.D"] = message["mamba D"]
    state_dict[f"model.layers.{layer_idx}.mamba.dt_bias"] = message["mamba dt_bias"]
    state_dict[f"model.layers.{layer_idx}.mamba.conv1d.weight"] = message["mamba conv1d weight"]
    state_dict[f"model.layers.{layer_idx}.mamba.conv1d.bias"] = message["mamba conv1d bias"]
    state_dict[f"model.layers.{layer_idx}.mamba.in_proj.weight"] = message["mamba in_proj weight"]
    state_dict[f"model.layers.{layer_idx}.mamba.out_proj.weight"] = message["mamba out_proj weight"]
    
    state_dict[f"model.layers.{layer_idx}.input_layernorm.weight"] = message["mamba pre norm weight"]
    state_dict[f"model.layers.{layer_idx}.mamba.norm.weight"] = message["mamba internal norm weight"]
    
    # Self-attention components - PROPER QKV SPLITTING
    qkv_weight = message["attention qkv weight"]
    
    # using standard Llama QKV layout
    head_size = falcon_h1_config.hidden_size // falcon_h1_config.num_attention_heads  # 128
    heads_per_group = falcon_h1_config.num_attention_heads // falcon_h1_config.num_key_value_heads  # 4
    qkv_total_heads = falcon_h1_config.num_attention_heads + 2 * falcon_h1_config.num_key_value_heads  # 12
    
    # Reshape QKV to [12, 128, 1024] like Llama does
    qkv_weights = qkv_weight.reshape([qkv_total_heads, head_size, falcon_h1_config.hidden_size])
    
    # Create slices for Q, K, V exactly like Llama saver
    q_slice = torch.cat([
        torch.arange(
            (heads_per_group + 2) * i,
            (heads_per_group + 2) * i + heads_per_group,
        )
        for i in range(falcon_h1_config.num_key_value_heads)
    ])
    k_slice = torch.arange(heads_per_group, qkv_total_heads, (heads_per_group + 2))
    v_slice = torch.arange(heads_per_group + 1, qkv_total_heads, (heads_per_group + 2))
    
    # Extract Q, K, V using Llama's slicing approach
    state_dict[f"model.layers.{layer_idx}.self_attn.q_proj.weight"] = qkv_weights[q_slice].reshape(-1, falcon_h1_config.hidden_size)
    state_dict[f"model.layers.{layer_idx}.self_attn.k_proj.weight"] = qkv_weights[k_slice].reshape(-1, falcon_h1_config.hidden_size)
    state_dict[f"model.layers.{layer_idx}.self_attn.v_proj.weight"] = qkv_weights[v_slice].reshape(-1, falcon_h1_config.hidden_size)
    
    # Attention output projection
    state_dict[f"model.layers.{layer_idx}.self_attn.o_proj.weight"] = message["attention dense weight"]
    
    # Attention layer norm 
    state_dict[f"model.layers.{layer_idx}.input_layernorm.weight"] = message["attention input norm weight"]
    
    return state_dict

def process_mlp_layer_weights(message: dict, layer_idx: int, falcon_h1_config: FalconH1Config) -> dict[str, torch.Tensor]:
    """Process weights for MLP-only layers"""
    state_dict = {}
    
    # MLP components - FIXED NAMES TO FEED_FORWARD
    mlp_fc1_weight = message["mlp fc1 weight"]
    
    # Split gate and up projections (assuming SwiGLU like Llama)
    intermediate_size = falcon_h1_config.intermediate_size
    
    # Split the fc1 weight into gate_proj and up_proj
    gate_proj_weight = mlp_fc1_weight[:intermediate_size, :]
    up_proj_weight = mlp_fc1_weight[intermediate_size:, :]
    
    state_dict[f"model.layers.{layer_idx}.feed_forward.gate_proj.weight"] = gate_proj_weight
    state_dict[f"model.layers.{layer_idx}.feed_forward.up_proj.weight"] = up_proj_weight
    state_dict[f"model.layers.{layer_idx}.feed_forward.down_proj.weight"] = message["mlp fc2 weight"]
    
    # MLP layer norm - FIXED NAME
    state_dict[f"model.layers.{layer_idx}.pre_ff_layernorm.weight"] = message["mlp input norm weight"]
    
    return state_dict

def save_checkpoint(queue: mp.Queue, args):
    def queue_get(name=None):
        val = queue.get()
        if val == "exit":
            print("Loader exited, exiting saver")
            exit(1)
        if name is not None and args.checking and val["name"] != name:
            val_name = val["name"]
            print(
                f'Unexpected message. Expecting "{name}" but got "{val_name}". Exiting saver.'
            )
            exit(1)
        if name is not None:
            print(f"received {name}")
        return val

    md = queue_get()

    ### Verify compatibility of args
    if not hasattr(md, "checkpoint_args"):
        raise ValueError("missing checkpoint_args in metadata")
    
    # Falcon-H1 specific validations
    if not hasattr(md.checkpoint_args, 'hybrid_architecture'):
        print("Warning: hybrid_architecture not specified in checkpoint_args, assuming Falcon-H1")
    
    torch_dtype = torch.float32
    if md.checkpoint_args.bf16:
        torch_dtype = torch.bfloat16
        if md.checkpoint_args.fp16:
            raise ValueError("bf16 and fp16 cannot be both set.")
    elif md.checkpoint_args.fp16:
        torch_dtype = torch.float16
        if md.checkpoint_args.bf16:
            raise ValueError("bf16 and fp16 cannot be both set.")

    ### init
    save_dir = Path(args.save_dir)
    tmp_save_dir = save_dir / "tmp"
    save_dir.mkdir(exist_ok=True)
    tmp_save_dir.mkdir(exist_ok=True)
    index_dict = {
        "weight_map": {},
        "metadata": {"total_size": 0},
    }
    tokenizer = None
    ref_state_dict = None

    ### prepare a reference model if needed
    if args.check_eq_hf:
        print(f"preparing checks with given HF model {args.check_eq_hf}")
        ref_model = AutoModelForCausalLM.from_pretrained(args.check_eq_hf, trust_remote_code=True)
        ref_state_dict = ref_model.state_dict()

    ### save tokenizer conf files
    if args.hf_tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.hf_tokenizer)
        print(f"saving tokenizer to {args.save_dir}")
        tokenizer.save_pretrained(args.save_dir)

    ### save config.json
    falcon_h1_config = FalconH1Config(
        # Basic model parameters from checkpoint
        vocab_size=md.true_vocab_size if md.true_vocab_size else md.checkpoint_args.padded_vocab_size,
        hidden_size=md.checkpoint_args.hidden_size,                    
        intermediate_size=md.checkpoint_args.ffn_hidden_size,           
        num_hidden_layers=md.checkpoint_args.num_layers,               
        num_attention_heads=md.checkpoint_args.num_attention_heads,  
        num_key_value_heads=md.checkpoint_args.num_query_groups,   
        max_position_embeddings=md.checkpoint_args.max_position_embeddings, 
        rms_norm_eps=md.checkpoint_args.norm_epsilon,                  
        tie_word_embeddings=not md.checkpoint_args.untie_embeddings_and_output_weights,
        attention_dropout=md.checkpoint_args.attention_dropout,        
        
        # Mamba parameters from checkpoint
        mamba_d_state=md.checkpoint_args.mamba_state_dim,              
        mamba_d_conv=md.checkpoint_args.d_conv,                        
        mamba_expand=md.checkpoint_args.expand,                        
        mamba_d_ssm=md.checkpoint_args.d_inner,                        
        mamba_n_heads=md.checkpoint_args.d_inner // md.checkpoint_args.mamba_head_dim, 
        mamba_d_head=md.checkpoint_args.mamba_head_dim,          
        mamba_n_groups=md.checkpoint_args.mamba_num_groups,            
        mamba_chunk_size=md.checkpoint_args.chunk_size,                
        mamba_conv_bias=md.checkpoint_args.conv_bias,                  
        mamba_proj_bias=md.checkpoint_args.add_bias_linear,            
        mamba_norm_before_gate=md.checkpoint_args.norm_before_gate,   
        mamba_rms_norm=md.checkpoint_args.rmsnorm,                     
        
        # RoPE parameters from checkpoint
        rope_theta=md.checkpoint_args.rotary_base,                  
        
        # Bias parameters from checkpoint
        attention_bias=md.checkpoint_args.add_bias_linear,            
        mlp_bias=md.checkpoint_args.add_bias_linear,                 
        projectors_bias=md.checkpoint_args.add_bias_linear,           
        
        # Token IDs - from tokenizer if available, otherwise defaults
        pad_token_id=getattr(tokenizer, 'pad_token_id', 0) if tokenizer else 0,
        bos_token_id=getattr(tokenizer, 'bos_token_id', 1) if tokenizer else 1,
        eos_token_id=getattr(tokenizer, 'eos_token_id', 2) if tokenizer else 2,
        
        # Parameters using FalconH1Config defaults (not in checkpoint)
        hidden_act="silu",                                             
        initializer_range=0.02,                                     
        use_cache=True,                                                 
        num_logits_to_keep=1,                                           
        rope_scaling=None,                                            
        
        # Model metadata
        torch_dtype=torch_dtype,
        architectures=["FalconH1ForCausalLM"],
        model_type="falcon_h1",
        transformers_version="4.52.0",
    )
    
    if args.hf_tokenizer:
        falcon_h1_config.eos_token_id = tokenizer.eos_token_id
        falcon_h1_config.bos_token_id = tokenizer.bos_token_id
    
    print(f"saving config.json to {tmp_save_dir}")
    falcon_h1_config.save_pretrained(tmp_save_dir)

    ### save embedding layer
    def pad_weight(orig_word_embed, true_vocab_size):
        if true_vocab_size is not None:
            # figure out what our padded vocab size is
            orig_vocab_size = orig_word_embed.shape[0]
            md.checkpoint_args.padded_vocab_size = _vocab_size_with_padding(true_vocab_size, md.checkpoint_args)

            # Cut out extra padding we don't need
            if orig_vocab_size > md.checkpoint_args.padded_vocab_size:
                full_word_embed = orig_word_embed[0:md.checkpoint_args.padded_vocab_size,:]

            # Expanding embedding to larger size by replicating final entry
            elif orig_vocab_size < md.checkpoint_args.padded_vocab_size:
                padding_size = md.checkpoint_args.padded_vocab_size - orig_vocab_size
                full_word_embed = torch.cat((
                    orig_word_embed,
                    orig_word_embed[-1].unsqueeze(0).expand(padding_size, -1)))

            # Same size!
            else:
                full_word_embed = orig_word_embed
        else:
            print("Original vocab size not specified, leaving embedding table as-is. "
                "If you've changed the tensor parallel size this could cause problems.")
            md.checkpoint_args.padded_vocab_size = orig_word_embed.shape[0]
            full_word_embed = orig_word_embed
        return full_word_embed

    state_dict = {
        "model.embed_tokens.weight": pad_weight(queue_get("embeddings")["word embeddings"], md.true_vocab_size)
    }
    index_dict, ref_state_dict = save_layer(
        state_dict,
        index_dict,
        dir_path=tmp_save_dir,
        filename="pytorch_model-embedding.bin",
        check_reference=args.check_eq_hf,
        ref_state_dict=ref_state_dict,
    )

    for i_layer in range(falcon_h1_config.num_hidden_layers):
        state_dict = {}
        
        if is_hybrid_layer(i_layer):
            # Process hybrid layer (Mamba + Attention) - EVEN layers
            message = queue_get(f"hybrid layer {i_layer}")
            
            # Add Mamba + Attention components from Megatron
            hybrid_weights = process_hybrid_layer_weights(message, i_layer, falcon_h1_config)
            state_dict.update(hybrid_weights)
            
            # Add MISSING MLP components (configured to output zeros = identity for addition)
            mlp_intermediate_size = falcon_h1_config.intermediate_size 
            state_dict.update({
                # Gate and up can be anything since down_proj will zero everything out
                f"model.layers.{i_layer}.feed_forward.gate_proj.weight": torch.randn(
                    mlp_intermediate_size, falcon_h1_config.hidden_size, 
                    dtype=torch_dtype
                ) * 0.01,
                f"model.layers.{i_layer}.feed_forward.up_proj.weight": torch.randn(
                    mlp_intermediate_size, falcon_h1_config.hidden_size, 
                    dtype=torch_dtype
                ) * 0.01,
                # KEY: down_proj = 0 makes entire MLP output zero
                f"model.layers.{i_layer}.feed_forward.down_proj.weight": torch.zeros(
                    falcon_h1_config.hidden_size, mlp_intermediate_size, 
                    dtype=torch_dtype
                ),
                f"model.layers.{i_layer}.pre_ff_layernorm.weight": torch.ones(
                    falcon_h1_config.hidden_size, dtype=torch_dtype
                ),
            })
            
        else:
            # Process MLP-only layer - ODD layers
            message = queue_get(f"mlp layer {i_layer}")
            
            # Add MLP components from Megatron
            mlp_weights = process_mlp_layer_weights(message, i_layer, falcon_h1_config)
            state_dict.update(mlp_weights)
            
            # Add MISSING Mamba components (configured to output zeros = identity for addition)
            mamba_intermediate_size = (
                falcon_h1_config.mamba_d_ssm if falcon_h1_config.mamba_d_ssm 
                else int(falcon_h1_config.mamba_expand * falcon_h1_config.hidden_size)
            )
            conv_dim = mamba_intermediate_size + 2 * falcon_h1_config.mamba_n_groups * falcon_h1_config.mamba_d_state
            projection_size = mamba_intermediate_size + conv_dim + falcon_h1_config.mamba_n_heads
            
            state_dict.update({
                f"model.layers.{i_layer}.mamba.A_log": torch.log(torch.arange(1, falcon_h1_config.mamba_n_heads + 1, dtype=torch_dtype)),
                f"model.layers.{i_layer}.mamba.D": torch.ones(falcon_h1_config.mamba_n_heads, dtype=torch_dtype),
                f"model.layers.{i_layer}.mamba.dt_bias": torch.ones(falcon_h1_config.mamba_n_heads, dtype=torch_dtype),
                f"model.layers.{i_layer}.mamba.conv1d.weight": torch.randn(
                    conv_dim, 1, falcon_h1_config.mamba_d_conv, dtype=torch_dtype
                ) * 0.01,
                f"model.layers.{i_layer}.mamba.conv1d.bias": torch.zeros(conv_dim, dtype=torch_dtype),
                f"model.layers.{i_layer}.mamba.in_proj.weight": torch.randn(
                    projection_size, falcon_h1_config.hidden_size, dtype=torch_dtype
                ) * 0.01,
                # KEY: out_proj = 0 makes entire Mamba output zero
                f"model.layers.{i_layer}.mamba.out_proj.weight": torch.zeros(
                    falcon_h1_config.hidden_size, mamba_intermediate_size, dtype=torch_dtype
                ),
                f"model.layers.{i_layer}.mamba.norm.weight": torch.ones(mamba_intermediate_size, dtype=torch_dtype),
            })
            
            # Add MISSING Attention components (configured to output zeros = identity for addition)
            head_dim = falcon_h1_config.hidden_size // falcon_h1_config.num_attention_heads
            state_dict.update({
                f"model.layers.{i_layer}.self_attn.q_proj.weight": torch.randn(
                    falcon_h1_config.num_attention_heads * head_dim,
                    falcon_h1_config.hidden_size, dtype=torch_dtype
                ) * 0.01,
                f"model.layers.{i_layer}.self_attn.k_proj.weight": torch.randn(
                    falcon_h1_config.num_key_value_heads * head_dim,
                    falcon_h1_config.hidden_size, dtype=torch_dtype
                ) * 0.01,
                f"model.layers.{i_layer}.self_attn.v_proj.weight": torch.randn(
                    falcon_h1_config.num_key_value_heads * head_dim,
                    falcon_h1_config.hidden_size, dtype=torch_dtype
                ) * 0.01,
                # KEY: o_proj = 0 makes entire attention output zero
                f"model.layers.{i_layer}.self_attn.o_proj.weight": torch.zeros(
                    falcon_h1_config.hidden_size,
                    falcon_h1_config.num_attention_heads * head_dim,
                    dtype=torch_dtype
                ),
                f"model.layers.{i_layer}.input_layernorm.weight": torch.ones(
                    falcon_h1_config.hidden_size, dtype=torch_dtype
                ),
            })        
        index_dict, ref_state_dict = save_layer(
            state_dict,
            index_dict,
            dir_path=tmp_save_dir,
            filename=f"pytorch_model-{i_layer + 1}.bin",
            check_reference=args.check_eq_hf,
            ref_state_dict=ref_state_dict,
        )


    ### save final norm and output layer
    state_dict = {
    "model.final_layernorm.weight": queue_get("final norm")["weight"]
}
    if md.checkpoint_args.untie_embeddings_and_output_weights:
        state_dict["lm_head.weight"] = pad_weight(queue_get("output layer")["weight"], md.true_vocab_size)
    
    index_dict, ref_state_dict = save_layer(
        state_dict,
        index_dict,
        dir_path=tmp_save_dir,
        filename="pytorch_model-lm-head.bin",
        check_reference=args.check_eq_hf,
        ref_state_dict=ref_state_dict,
    )
    
    # final check
    if ref_state_dict:
        remaining_keys = list(ref_state_dict.keys())
        print(f"Warning: reference state dict has {len(remaining_keys)} additional layers not present in converted model:")
        for key in remaining_keys[:10]:  # Show first 10
            print(f"  - {key}")
        if len(remaining_keys) > 10:
            print(f"  ... and {len(remaining_keys) - 10} more")

    ### save index dict
    index_dict["metadata"]["total_size"] *= {
        torch.float32: 4,
        torch.float16: 2,
        torch.bfloat16: 2,
    }[torch_dtype]
    print(f"saving {tmp_save_dir}/pytorch_model.bin.index.json")
    with open(f"{tmp_save_dir}/pytorch_model.bin.index.json", "w") as f:
        json.dump(index_dict, f)

    ### load then save model in HF format
    # Make space so we can load the model properly now.
    del state_dict
    gc.collect()
    print(f"Loading the converted pytorch checkpoint in a Falcon-H1 HF model from {tmp_save_dir}")
    model = FalconH1ForCausalLM.from_pretrained(
        str(tmp_save_dir), torch_dtype=torch_dtype, low_cpu_mem_usage=True, trust_remote_code=True
    )
    
    # Avoid saving this as part of the config.
    if hasattr(model.config, '_name_or_path'):
        del model.config._name_or_path
    model.config.torch_dtype = torch_dtype
    print(f"Saving in the Transformers safe tensors format to {args.save_dir}")
    model.save_pretrained(args.save_dir, safe_serialization=True)

    ### save chat config
    generation_config = (
        GenerationConfig(
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            bos_token_id=falcon_h1_config.bos_token_id,
            eos_token_id=falcon_h1_config.eos_token_id,
        )
        if args.save_chat_model
        else GenerationConfig(
            _from_model_config=True,
            bos_token_id=falcon_h1_config.bos_token_id,
            eos_token_id=falcon_h1_config.eos_token_id,
        )
    )
    print(f"Saving generation config to {args.save_dir}")
    generation_config.save_pretrained(args.save_dir)
    
    ### cleanup tmp
    print(f"Deleting {tmp_save_dir}")
    rmtree(tmp_save_dir)
