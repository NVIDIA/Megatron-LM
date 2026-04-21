import os
import sys

import torch

from schema_hf import get_vision_model_schema, get_language_model_schema
from saver_hf import HFCheckpointSaver

from huggingface_hub import save_torch_state_dict

def add_arguments(parser):
    group = parser.add_argument_group(title='HuggingFace LLaVA saver')

    group.add_argument('--megatron-path', type=str, default=None,
                       help='Base directory of Megatron repository')

def _convert_nemo_parakeet_key_to_hf(key: str) -> str:
    """Convert NeMo Parakeet key names to HuggingFace Parakeet format.
    
    The NeMo checkpoint uses different naming conventions than the HuggingFace
    transformers ParakeetEncoder. This function converts keys during checkpoint saving.
    
    Key mappings:
    - pre_encode.out.* -> subsampling.linear.*
    - pre_encode.conv.N.* -> subsampling.layers.N.*
    - layers.N.self_attn.linear_q.* -> layers.N.self_attn.q_proj.*
    - layers.N.self_attn.linear_k.* -> layers.N.self_attn.k_proj.*
    - layers.N.self_attn.linear_v.* -> layers.N.self_attn.v_proj.*
    - layers.N.self_attn.linear_out.* -> layers.N.self_attn.o_proj.*
    - layers.N.self_attn.linear_pos.* -> layers.N.self_attn.relative_k_proj.*
    - layers.N.self_attn.pos_bias_u -> layers.N.self_attn.bias_u
    - layers.N.self_attn.pos_bias_v -> layers.N.self_attn.bias_v
    - layers.N.conv.batch_norm.* -> layers.N.conv.norm.*
    """
    # Subsampling conversions
    if ".pre_encode.out." in key:
        key = key.replace(".pre_encode.out.", ".subsampling.linear.")
    if ".pre_encode.conv." in key:
        key = key.replace(".pre_encode.conv.", ".subsampling.layers.")
    
    # Self-attention projections
    key = key.replace(".self_attn.linear_q.", ".self_attn.q_proj.")
    key = key.replace(".self_attn.linear_k.", ".self_attn.k_proj.")
    key = key.replace(".self_attn.linear_v.", ".self_attn.v_proj.")
    key = key.replace(".self_attn.linear_out.", ".self_attn.o_proj.")
    key = key.replace(".self_attn.linear_pos.", ".self_attn.relative_k_proj.")
    
    # Position bias names
    key = key.replace(".self_attn.pos_bias_u", ".self_attn.bias_u")
    key = key.replace(".self_attn.pos_bias_v", ".self_attn.bias_v")
    
    # Batch norm -> norm
    key = key.replace(".conv.batch_norm.", ".conv.norm.")
    
    return key


def recover_qkv(new_tensor, num_head, head_dim):
    # Step 1: Reshape back to (num_head, 3*head_dim, -1)
    temp = new_tensor.view(num_head, 3 * head_dim, -1)
    
    # Step 2: Slice along the head_dim dimension to get q, k, v
    q = temp[:, 0:head_dim, :]
    k = temp[:, head_dim:2*head_dim, :]
    v = temp[:, 2*head_dim:3*head_dim, :]
    
    # Step 3: Reshape each back to (num_head * head_dim, -1)
    q_proj_params = q.contiguous().view(num_head * head_dim, -1)
    k_proj_params = k.contiguous().view(num_head * head_dim, -1)
    v_proj_params = v.contiguous().view(num_head * head_dim, -1)
    
    return q_proj_params, k_proj_params, v_proj_params

class HFCheckpointSaverLLaVA(HFCheckpointSaver):
    def __init__(self, args, queue):
        super().__init__(args, queue)

    def receive_vision_backbone(self, schema):
        vision_embeddings_msg = self.queue_get("vit embeddings")

        params_dict = {}

        if self.md.vision_model_type == "radio":
            params_dict["embedder_weight"] = vision_embeddings_msg["embedder weight"]
            if getattr(self.md, 'separate_video_embedder', False):
                params_dict["video_embedder_weight"] = vision_embeddings_msg["video_embedder weight"]
            params_dict["class_token"] = vision_embeddings_msg["class token"]
            params_dict["position_embeddings"] = vision_embeddings_msg["position embeddings"]
            params_dict["input_conditioner_norm_mean"] = torch.tensor([0.48145466, 0.4578275, 0.40821073]).unsqueeze(-1).unsqueeze(-1)
            params_dict["input_conditioner_norm_std"] = torch.tensor([0.26862954, 0.26130258, 0.27577711]).unsqueeze(-1).unsqueeze(-1)
        elif self.md.vision_model_type == "internvit":
            params_dict["patch_embedding_weight"] = vision_embeddings_msg["conv1 weight"]
            params_dict["patch_embedding_bias"] = vision_embeddings_msg["conv1 bias"]
            params_dict["position_embeddings"] = vision_embeddings_msg["position embeddings"]
            params_dict["class_token"] = vision_embeddings_msg["class token"]
        elif self.md.vision_model_type == "siglip":
            params_dict["patch_embedding_weight"] = vision_embeddings_msg["conv1 weight"]
            params_dict["patch_embedding_bias"] = vision_embeddings_msg["conv1 bias"]
            params_dict["ln_post_weight"] = vision_embeddings_msg["ln post weight"]
            params_dict["ln_post_bias"] = vision_embeddings_msg["ln post bias"]
            params_dict["position_embeddings"] = vision_embeddings_msg["position embeddings"]

        schema.set(self.state_dict, params_dict)

        # Creates indices for reordering of qkv weights for Internvit and RADIO
        if self.md.vision_model_type in ("internvit", "radio"):
            order = torch.ones(3 * self.md.vision_hidden_size).long()

            num_heads = self.md.vision_num_attention_heads
            if self.md.vision_model_type == "internvit":
                num_heads = self.md.vision_num_attention_heads - self.md.vision_dummy_head_count
            dim = self.md.vision_kv_channels
            for j in range(num_heads):
                for i in range(dim):
                    order[j*dim+i] = i + dim*3*j
                    order[j*dim+i+num_heads*dim] = dim + i + dim*3*j
                    order[j*dim+i+num_heads*dim*2] = dim*2 + i + dim*3*j

        for i in range(self.md.vision_num_layers):
            message = self.queue_get(f"vit transformer layer {i}")
            params_dict = {}

            if self.md.vision_model_type == "internvit":
                params_dict["ls1"] = message["ls1"] 
                params_dict["ls2"] = message["ls2"] 
                params_dict["k_norm_weight"] = message["k norm weight"][:self.md.vision_hidden_size]
                params_dict["q_norm_weight"] = message["q norm weight"][:self.md.vision_hidden_size]
                if self.md.vision_norm_has_bias:
                    params_dict["k_norm_bias"] = message["k norm bias"]
                    params_dict["q_norm_bias"] = message["q norm bias"]

            if self.md.vision_model_type in ("internvit", "siglip", "radio"):
                params_dict["input_norm_weight"] = message["input norm weight"]
                params_dict["pre_mlp_norm_weight"] = message["pre mlp norm weight"]
                if self.md.vision_norm_has_bias:
                    params_dict["input_norm_bias"] = message["input norm bias"]
                    params_dict["pre_mlp_norm_bias"] = message["pre mlp norm bias"]

            if self.md.vision_swiglu:
                params_dict["mlp_l0_weight_W"] = message["mlp l0 weight W"]
                params_dict["mlp_l0_weight_V"] = message["mlp l0 weight V"]
            else:
                params_dict["mlp_l0_weight"] = message["mlp l0 weight"] 
            if self.md.vision_linear_bias:
                if self.md.vision_swiglu:
                    params_dict["mlp_l0_bias_W"] = message["mlp l0 bias W"]
                    params_dict["mlp_l0_bias_V"] = message["mlp l0 bias V"]
                else:
                    params_dict["mlp_l0_bias"] = message["mlp l0 bias"] 

            if self.md.vision_model_type == "internvit":
                params_dict["qkv_weight"] = message["qkv weight"][:self.md.vision_hidden_size * 3][order]
            elif self.md.vision_model_type in ("siglip"):
                # Split the Q/K/V
                query, key, value = recover_qkv(message["qkv weight"], num_head=16, head_dim=72)
                params_dict["q_proj_weight"] = query
                params_dict["k_proj_weight"] = key
                params_dict["v_proj_weight"] = value
            elif self.md.vision_model_type == "radio":
                params_dict["qkv_weight"] = message["qkv weight"][order]
            if self.md.vision_qkv_bias:
                if self.md.vision_model_type == "internvit":
                    params_dict["qkv_bias"] = message["qkv bias"][order]
                if self.md.vision_model_type in ("siglip"):
                    query_bias, key_bias, value_bias = recover_qkv(message["qkv bias"], num_head=16, head_dim=72)
                    assert query_bias.shape[-1] == 1, "expected query_bias last dimension after recovery to be 1"
                    params_dict["q_proj_bias"] = query_bias[:, 0]
                    params_dict["k_proj_bias"] = key_bias[:, 0]
                    params_dict["v_proj_bias"] = value_bias[:, 0]
                if self.md.vision_model_type == "radio":
                    params_dict["qkv_bias"] = message["qkv bias"][order]

            if self.md.vision_model_type == "internvit":
                params_dict["dense_weight"] = message["dense weight"][..., :self.md.vision_hidden_size]
            elif self.md.vision_model_type in ("siglip", "radio"):
                params_dict["dense_weight"] = message["dense weight"]
            params_dict["mlp_l1_weight"] = message["mlp l1 weight"]
            if self.md.vision_linear_bias:
                params_dict["mlp_l1_bias"] = message["mlp l1 bias"]
                if self.md.vision_model_type in ("siglip", "radio"):
                    params_dict["dense_bias"] = message["dense bias"]
                elif self.md.vision_model_type == "internvit":
                    params_dict["dense_bias"] = message["dense bias"][:self.md.vision_hidden_size]
            
            schema.set_layer(self.state_dict, i, params_dict)

    def receive_vision_projection(self):
        projection_msg = self.queue_get("vision projection")
        self.state_dict["mlp1.0.weight"] = projection_msg["vision projection norm weight"]
        if "vision projection norm bias" in projection_msg:
            self.state_dict["mlp1.0.bias"] = projection_msg["vision projection norm bias"]
        self.state_dict["mlp1.1.weight"] = projection_msg["vision projection l0 weight"]
        self.state_dict["mlp1.3.weight"] = projection_msg["vision projection l1 weight"]
        if self.md.vision_projection_linear_bias:
            self.state_dict["mlp1.1.bias"] = projection_msg["vision projection l0 bias"]
            self.state_dict["mlp1.3.bias"] = projection_msg["vision projection l1 bias"]

        # Handle conv merge weights if conv_merging is enabled
        if self.md.conv_merging:
            self.state_dict["conv_merge.mlp.0.weight"] = projection_msg["conv merge l0 weight"]
            self.state_dict["conv_merge.mlp.2.weight"] = projection_msg["conv merge l1 weight"]
            if self.md.vision_projection_linear_bias:
                self.state_dict["conv_merge.mlp.0.bias"] = projection_msg["conv merge l0 bias"]
                self.state_dict["conv_merge.mlp.2.bias"] = projection_msg["conv merge l1 bias"]

    def receive_sound_projection(self):
        """Receive and save sound projection MLP weights."""
        if getattr(self.md, 'sound_model_type', None) is None:
            return
        
        try:
            projection_msg = self.queue_get("sound projection")
        except:
            print("No sound projection data available, skipping")
            return
        
        print("Receiving sound projection weights...")
        
        # Map to HuggingFace SoundProjection structure
        # Megatron: sound_mlp1.{0,1,3} -> HF: sound_projection.{norm, linear1, linear2}
        # Sound projection is an MLP: linear1 -> norm -> activation -> linear2
        self.state_dict["sound_projection.linear1.weight"] = projection_msg["sound projection l0 weight"]
        self.state_dict["sound_projection.norm.weight"] = projection_msg["sound projection norm weight"]
        self.state_dict["sound_projection.linear2.weight"] = projection_msg["sound projection l1 weight"]
        
        if "sound projection norm bias" in projection_msg:
            self.state_dict["sound_projection.norm.bias"] = projection_msg["sound projection norm bias"]
        
        if getattr(self.md, 'sound_projection_linear_bias', False):
            self.state_dict["sound_projection.linear1.bias"] = projection_msg["sound projection l0 bias"]
            self.state_dict["sound_projection.linear2.bias"] = projection_msg["sound projection l1 bias"]
        
        print("Loaded sound projection parameters")

    def receive_sound_model(self):
        """Receive the sound_model weights (e.g., Parakeet encoder) and map them to HF layout.

        The loader (`loader_llava.py`) sends the underlying sound model state dict in chunks
        with keys prefixed by:
          - ``sound_model.feature_extractor.*``
          - ``sound_model.model.*``

        In the Hugging Face NemotronH checkpoint, the audio encoder lives under
        ``sound_encoder.encoder`` (see ``audio_model.SoundEncoder``), so we rewrite
        those prefixes accordingly:

          sound_model.feature_extractor.X -> sound_encoder.encoder.feature_extractor.X
          sound_model.model.X             -> sound_encoder.encoder.X
        """
        # If no sound model is expected, do nothing
        if getattr(self.md, "sound_model_type", None) is None:
            return

        start_msg = self.queue_get("sound model start")

        total_chunks = start_msg.get("chunks", 0)
        merged = {}
        for idx in range(total_chunks):
            chunk_msg = self.queue_get(f"sound model chunk {idx}")
            merged.update(chunk_msg)

        # Consume end sentinel
        _ = self.queue_get("sound model end")

        fe_prefix = "sound_model.feature_extractor."
        mdl_prefix = "sound_model.model."

        for key, value in merged.items():
            if key.startswith(fe_prefix):
                # Map feature_extractor weights into HF `SoundEncoder.encoder.feature_extractor`
                new_key = "sound_encoder.encoder.feature_extractor." + key[len(fe_prefix):]
            elif key.startswith(mdl_prefix):
                # Map main encoder weights into HF `SoundEncoder.encoder`
                new_key = "sound_encoder.encoder." + key[len(mdl_prefix):]
            else:
                # Unknown prefix; skip rather than breaking conversion.
                continue

            # Convert NeMo key names to HuggingFace format
            new_key = _convert_nemo_parakeet_key_to_hf(new_key)
            self.state_dict[new_key] = value

        print("Loaded sound model parameters")

    def receive_model(self):
        """Override to handle vision, sound, and language models for LLaVA"""
        # Vision model
        vision_model_prefix = "vision_model.vision_model."
        vision_layer_prefix = "encoder.layers"
        if self.md.vision_model_type == "radio":
            vision_model_prefix = "vision_model.radio_model."
            vision_layer_prefix = "model.blocks"
        vision_schema = get_vision_model_schema(
            self.md.vision_model_type,
            prefix=vision_model_prefix,
            layer_prefix=vision_layer_prefix,
            use_swiglu=self.md.vision_swiglu,
        )
        self.receive_vision_backbone(vision_schema)

        self.receive_vision_projection()

        # Sound model (if present)
        # Note: Order must match loader_llava.py lines 455-456
        if getattr(self.md, 'sound_model_type', None) is not None:
            self.receive_sound_projection()
            self.receive_sound_model()

        # Language model
        language_model_prefix = "language_model."
        if self.md.model_type == "hybrid":
            language_layer_prefix = "backbone.layers"
        else:
            language_layer_prefix = "model.layers"
        language_schema = get_language_model_schema(
            model_type=self.args.model_type,
            prefix=language_model_prefix,
            layer_prefix=language_layer_prefix,
            use_swiglu=self.md.swiglu,
        )
        self.receive_lm(language_schema)

def save_checkpoint(queue, args):
    """
    Required top-level function that creates the saver and calls its .save().
    """
    saver = HFCheckpointSaverLLaVA(args, queue)
    try:
        saver.save()
    except Exception as e:
        raise e
    
