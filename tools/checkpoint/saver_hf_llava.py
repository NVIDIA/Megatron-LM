import os
import sys

import torch

from schema_hf import get_vision_model_schema, get_language_model_schema

from transformers.modeling_utils import WEIGHTS_INDEX_NAME, WEIGHTS_NAME, shard_checkpoint

def add_arguments(parser):
    group = parser.add_argument_group(title='HuggingFace LLaVA saver')

    group.add_argument('--megatron-path', type=str, default=None,
                       help='Base directory of Megatron repository')

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

class HFCheckpointSaverLLaVA:
    def __init__(self, args, queue):
        self.args = args
        self.queue = queue

    def insert_megatron_path(self):
        sys.path.append(os.path.abspath(
            os.path.join(os.path.dirname(__file__),
                         os.path.pardir,
                         os.path.pardir)))
        if self.args.megatron_path is not None:
            sys.path.insert(0, self.args.megatron_path)

    def queue_get(self, name=None):
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
        if not self.args.checking:
            return
        msg_name = msg.pop("name")
        if len(msg.keys()) > 0:
            print(f"Unexpected values in {msg_name}:")
            for key in msg.keys():
                print(f"   {key}")
            print(f"Exiting. If you want to ignore this, use the argument --no-checking.")
            exit(1)

    def receive_vision_backbone(self, schema):
        vision_embeddings_msg = self.queue_get("vit embeddings")

        params_dict = {}

        if self.md.vision_model_type == "radio":
            params_dict["embedder_weight"] = vision_embeddings_msg["embedder weight"]
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


    def recover_lm_qkv_weight(self, qkv_weight):
        dim = self.md.kv_channels
        tp = self.md.previous_tensor_parallel_size
        nh = self.md.num_attention_heads // tp
        ng = self.md.num_query_groups // tp
        hidden_size = self.md.hidden_size

        params_per_tp = torch.chunk(qkv_weight, tp, dim=0)

        q = torch.empty(0)
        k = torch.empty(0)
        v = torch.empty(0)

        for t in params_per_tp:
            # 1. Reshape back to (ng, (dim*nh//ng + 2*dim), hidden_size).
            qkv = t.reshape(ng, dim * (nh // ng) + 2 * dim, -1)

            # 2. Slice out q, k, v along dim=1.
            q_t = qkv[:, : dim * (nh // ng), :]  
            k_t = qkv[:, dim * (nh // ng) : dim * (nh // ng) + dim, :]
            v_t = qkv[:, dim * (nh // ng) + dim :, :]

            # 3. Reshape each to match the original HF shapes.
            q_t = q_t.reshape(ng, dim * (nh // ng), -1)
            k_t = k_t.reshape(ng, dim, -1)
            v_t = v_t.reshape(ng, dim, -1)

            qp = q_t.reshape(dim * (nh // ng) * ng, -1)
            kp = k_t.reshape(dim * ng, -1)
            vp = v_t.reshape(dim * ng, -1)

            q = torch.cat([q, qp])
            k = torch.cat([k, kp])
            v = torch.cat([v, vp])

        return q, k, v

    def recover_lm_qkv_bias(self, qkv_bias):
        dim = self.md.kv_channels
        tp = self.md.previous_tensor_parallel_size
        nh = self.md.num_attention_heads // tp
        ng = self.md.num_query_groups // tp

        bias_per_tp = torch.chunk(qkv_bias, tp, dim=0)

        qb = torch.empty(0)
        kb = torch.empty(0)
        vb = torch.empty(0)

        for b in bias_per_tp:
            qkvb = b.reshape(ng, dim * (nh // ng) + 2 * dim)

            q_b = qkvb[:, : dim * (nh // ng)]  
            k_b = qkvb[:, dim * (nh // ng) : dim * (nh // ng) + dim]
            v_b = qkvb[:, dim * (nh // ng) + dim :]

            q_b = q_b.reshape(ng, dim * (nh // ng))
            k_b = k_b.reshape(ng, dim)
            v_b = v_b.reshape(ng, dim)

            q_b = q_b.reshape(-1)
            k_b = k_b.reshape(-1)
            v_b = v_b.reshape(-1)

            qb = torch.cat([qb, q_b]) 
            kb = torch.cat([kb, k_b]) 
            vb = torch.cat([vb, v_b]) 

        return qb, kb, vb,

    def receive_lm(self, schema):
        embeddings_msg = self.queue_get("embeddings")
        params_dict = {}

        params_dict["word_embeddings"] = embeddings_msg["word embeddings"]
        if self.md.position_embedding_type == "learned_absolute":
            # TODO: Below may not be correct, but none of our models use absolute positional embeddings
            params_dict["position_embeddings"] = embeddings_msg["position embeddings"]

        schema.set(self.state_dict, params_dict)

        for i in range(self.md.num_layers):
            message = self.queue_get(f"transformer layer {i}")
            params_dict = {}

            params_dict["input_norm_weight"] = message["input norm weight"]
            params_dict["post_norm_weight"] = message["post norm weight"]
            if self.md.norm_has_bias:
                params_dict["input_norm_bias"] = message["input norm bias"]
                params_dict["post_norm_bias"] = message["post norm bias"]

            if self.md.swiglu:
                params_dict["mlp_l0_weight_W"] = message["mlp l0 weight W"]
                params_dict["mlp_l0_weight_V"] = message["mlp l0 weight V"]
            else:
                params_dict["mlp_l0_weight"] = message["mlp l0 weight"]
            if self.md.linear_bias:
                if self.md.swiglu:
                    params_dict["mlp_l0_bias_W"] = message["mlp l0 bias W"]
                    params_dict["mlp_l0_bias_V"] = message["mlp l0 bias V"]
                else:
                    params_dict["mlp_l0_bias"] = message["mlp l0 bias"] 

            qkv_weight = message["qkv weight"]
            if self.md.qkv_bias:
                qkv_bias = message["qkv bias"]

            q, k, v = self.recover_lm_qkv_weight(qkv_weight)
            q = q.clone().detach().contiguous()
            params_dict["q_proj_weight"] = q
            k = k.clone().detach().contiguous()
            params_dict["k_proj_weight"] = k
            v = v.clone().detach().contiguous()
            params_dict["v_proj_weight"] = v

            if self.md.qkv_bias:
                qb, kb, vb = self.recover_lm_qkv_bias(qkv_bias)
                qb = qb.clone().detach().contiguous()
                params_dict["q_proj_bias"] = qb
                kb = kb.clone().detach().contiguous()
                params_dict["k_proj_bias"] = kb
                vb = vb.clone().detach().contiguous()
                params_dict["v_proj_bias"] = vb

            params_dict["mlp_l1_weight"] = message["mlp l1 weight"]
            params_dict["dense_weight"] = message["dense weight"]
            if self.md.linear_bias:
                params_dict["mlp_l1_bias"] = message["mlp l1 bias"]
                params_dict["dense_bias"] = message["dense bias"]

            schema.set_layer(self.state_dict, i, params_dict)

        params_dict = {
            "final_norm": self.queue_get('final norm')['weight'],
            "output_layer": self.queue_get('output layer')['weight'],

        }
        schema.set(self.state_dict, params_dict)

    def receive_model(self):
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

        language_model_prefix = "language_model."
        language_layer_prefix = "model.layers"
        language_schema = get_language_model_schema(
            prefix=language_model_prefix,
            layer_prefix=language_layer_prefix,
            use_swiglu=self.md.swiglu,
        )
        self.receive_lm(language_schema)

    def save_state_dict_to_hf_checkpoint(self):
        if not os.path.exists(self.args.save_dir):
            os.makedirs(self.args.save_dir)

        # Store the state_dict to file.
        max_shard_size = "4GB"
        shards, index = shard_checkpoint(self.state_dict, max_shard_size=max_shard_size)

        # Save the model
        for shard_file, shard in shards.items():
            torch.save(shard, os.path.join(self.args.save_dir, shard_file))

        if index is None:
            print(f"Model weights saved in {os.path.join(self.args.save_dir, WEIGHTS_NAME)}")
        else:
            import json
            save_index_file = os.path.join(self.args.save_dir, WEIGHTS_INDEX_NAME)
            # Save the index as well
            with open(save_index_file, "w", encoding="utf-8") as f:
                content = json.dumps(index, indent=2, sort_keys=True) + "\n"
                f.write(content)
            print(
                f"The model is bigger than the maximum size per checkpoint ({max_shard_size}) and is going to be "
                    f"split in {len(shards)} checkpoint shards. You can find where each parameters has been saved in the "
                    f"index located at {save_index_file}."
            )

    def save(self):
        self.insert_megatron_path()

        self.md = self.queue_get()
        
        self.state_dict = {}

        self.receive_model()

        self.save_state_dict_to_hf_checkpoint()

        print("Done!")
        

def save_checkpoint(queue, args):
    """
    Required top-level function that creates the saver and calls its .save().
    """
    saver = HFCheckpointSaverLLaVA(args, queue)
    try:
        saver.save()
    except Exception as e:
        raise e
    
