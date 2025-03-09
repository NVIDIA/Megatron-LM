import os
import sys

import torch

from transformers import Qwen2Config
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

def save_checkpoint(queue, args):
    # Search in directory above this
    sys.path.append(os.path.abspath(
        os.path.join(os.path.dirname(__file__),
                     os.path.pardir,
                     os.path.pardir)))
    if args.megatron_path is not None:
        sys.path.insert(0, args.megatron_path)

    try:
        import megatron
    except ModuleNotFoundError:
        print("Unable to import Megatron, please specify the path to Megatron using --megatron-path. Exiting.")
        exit(1)

    try:
        sys.path.insert(0, './examples/multimodal')
        from examples.multimodal.config import get_vision_model_config
    except ModuleNotFoundError:
        print("Unable to import multimodal example code. Please run this from the top directory in the megatron-lm repository")
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

    #TODO: COMPATBILITY OF ARGS ETC ARGS FOR HUGGINFACE
    # if "qwen2" in md.language_model_type:
    #     assert hasattr(md, 'checkpoint_args')
    #     assert md.model_type == 'GPT'
    #     mag_conf = md.checkpoint_args
    #     torch_dtype = torch.float32
    #     if mag_conf.bf16:
    #         assert mag_conf.fp16 == False
    #         torch_dtype = torch.bfloat16
    #     elif mag_conf.fp16:
    #         assert mag_conf.bf16 == False
    #         torch_dtype = torch.float16
    #     assert mag_conf.swiglu == True
    #     assert mag_conf.rotary_percent == 1.0
    #     qwen_config = Qwen2Config(
    #         vocab_size = mag_conf.padded_vocab_size,
    #         hidden_size             = mag_conf.hidden_size,
    #         intermediate_size       = mag_conf.ffn_hidden_size,
    #         num_hidden_layers       = md.num_layers,
    #         num_attention_heads     = mag_conf.num_attention_heads,
    #         num_key_value_heads     = mag_conf.num_query_groups,
    #         max_position_embeddings = mag_conf.max_position_embeddings,
    #         rms_norm_eps            = mag_conf.norm_epsilon,
    #         tie_word_embeddings     = not mag_conf.untie_embeddings_and_output_weights,
    #         torch_dtype             = torch_dtype,
    #         model_type              = "qwen2",
    #         architectures           = ['Qwen2ForCausalLM'],
    #         transformers_version    = "4.41.2",
    #     )
    #     # attention_bias          = mag_conf.add_bias_linear,
    #     qwen_config.save_pretrained(args.save_dir)

    state_dict = {}

    vision_embeddings_msg = queue_get("vit embeddings")
    if md.vision_model_type in ("internvit", "clip", "siglip"):
        state_dict["vision_model.vision_model.embeddings.patch_embedding.weight"] = vision_embeddings_msg["conv1 weight"]
        state_dict["vision_model.vision_model.embeddings.patch_embedding.bias"] = vision_embeddings_msg["conv1 bias"]
    if md.vision_model_type == "internvit":
        state_dict["vision_model.vision_model.embeddings.position_embedding"] = vision_embeddings_msg["position embeddings"]
    if md.vision_model_type == "siglip":
        state_dict["vision_model.vision_model.post_layernorm.weight"] = vision_embeddings_msg["ln post weight"]
        state_dict["vision_model.vision_model.post_layernorm.bias"] = vision_embeddings_msg["ln post bias"]
    if md.vision_model_type in ("siglip", "clip"):
        state_dict["vision_model.vision_model.embeddings.position_embedding.weight"] = vision_embeddings_msg["position embeddings"]
    if md.vision_model_type in ("internvit", "clip"):
        state_dict["vision_model.vision_model.embeddings.class_embedding"] = vision_embeddings_msg["class token"]


    #TODO: Do we need this reordering for huggingface on siglip and other vision converters as well?
    if md.vision_model_type == "internvit":
        order = torch.ones(3 * md.vision_hidden_size).long()

        num_heads = md.vision_num_attention_heads - md.vision_dummy_head_count
        dim = md.vision_kv_channels
        for j in range(num_heads):
            for i in range(dim):
                order[j*dim+i] = i + dim*3*j
                order[j*dim+i+num_heads*dim] = dim + i + dim*3*j
                order[j*dim+i+num_heads*dim*2] = dim*2 + i + dim*3*j

    for i in range(md.vision_num_layers):
        message = queue_get(f"vit transformer layer {i}")
        prefix = f"vision_model.vision_model.encoder.layers.{i}."

        if md.vision_model_type == "internvit":
            state_dict[prefix + 'ls1'] = message["ls1"] 
            state_dict[prefix + 'ls2'] = message["ls2"] 
            state_dict[prefix + 'norm1.weight'] = message["input norm weight"]
            state_dict[prefix + 'norm2.weight'] = message["pre mlp norm weight"]
            state_dict[prefix + 'attn.k_norm.weight'] = message["k norm weight"][:md.vision_hidden_size]
            state_dict[prefix + 'attn.q_norm.weight'] = message["q norm weight"][:md.vision_hidden_size]
            if md.vision_norm_has_bias:
                state_dict[prefix + 'norm1.bias'] = message["input norm bias"]
                state_dict[prefix + 'norm2.bias'] = message["pre mlp norm bias"]
                state_dict[prefix + 'attn.k_norm.bias'] = message["k norm bias"]
                state_dict[prefix + 'attn.q_norm.bias'] = message["q norm bias"]
        if md.vision_model_type == "siglip":
            state_dict[prefix + 'layer_norm1.weight'] = message["input norm weight"]
            state_dict[prefix + 'layer_norm2.weight'] = message["pre mlp norm weight"]
            if md.vision_norm_has_bias:
                state_dict[prefix + 'layer_norm1.bias'] = message["input norm bias"]
                state_dict[prefix + 'layer_norm2.bias'] = message["pre mlp norm bias"]

        if md.vision_swiglu:
            state_dict[prefix + 'mlp.gate_proj.weight'] = message["mlp l0 weight W"]
            state_dict[prefix + 'mlp.up_proj.weight'] = message["mlp l0 weight V"]
        else:
            state_dict[prefix + 'mlp.fc1.weight'] = message["mlp l0 weight"] 
        if md.vision_linear_bias:
            if md.vision_swiglu:
                state_dict[prefix + 'mlp.gate_proj.bias'] = message["mlp l0 bias W"]
                state_dict[prefix + 'mlp.up_proj.bias'] = message["mlp l0 bias V"]
            else:
                state_dict[prefix + 'mlp.fc1.bias'] = message["mlp l0 bias"] 

        if md.vision_model_type == "internvit":
            state_dict[prefix + "attn.qkv.weight"] = state_dict[prefix + "attn.qkv.weight"][:md.vision_hidden_size * 3][order]
        elif md.vision_model_type in ("siglip", "clip"):
            # Split the Q/K/V
            query, key, value = recover_qkv(message["qkv weight"], num_head=16, head_dim=72)
            state_dict[prefix + "self_attn.q_proj.weight"] = query
            state_dict[prefix + "self_attn.k_proj.weight"] = key
            state_dict[prefix + "self_attn.v_proj.weight"] = value
        if md.vision_qkv_bias:
            if md.vision_model_type == "internvit":
                state_dict[prefix + "attn.qkv.bias"] = message["qkv bias"][order]
            if md.vision_model_type in ("siglip", "clip"):
                query_bias, key_bias, value_bias = recover_qkv(message["qkv bias"], num_head=16, head_dim=72)
                state_dict[prefix + "self_attn.q_proj.bias"] = query_bias[:, 0]
                state_dict[prefix + "self_attn.k_proj.bias"] = key_bias[:, 0]
                state_dict[prefix + "self_attn.v_proj.bias"] = value_bias[:, 0]

        if md.vision_model_type == "internvit":
            state_dict[prefix + "attn.proj.weight"] = state_dict[prefix + "attn.proj.weight"][..., :md.vision_hidden_size]
        elif md.vision_model_type in ("siglip", "clip"):
            state_dict[prefix + "self_attn.out_proj.weight"] = message["dense weight"]
        if md.vision_swiglu:
            state_dict[prefix + 'mlp.down_proj.weight'] = message["mlp l1 weight"]
        else:
            state_dict[prefix + 'mlp.fc2.weight'] = message["mlp l1 weight"]
        if md.vision_linear_bias:
            if md.vision_swiglu:
                state_dict[prefix + 'mlp.down_proj.bias'] = message["mlp l1 bias"]
            else:
                state_dict[prefix + 'mlp.fc2.bias'] = message["mlp l1 bias"]
            if md.vision_model_type in ("siglip", "clip"):
                state_dict[prefix + "self_attn.out_proj.bias"] = message["dense bias"]
            else:
                state_dict[prefix + "attn.proj.bias"] = message["dense bias"]
            if md.vision_model_type == "internvit":
                state_dict[prefix + "attn.proj.bias"] = state_dict[prefix + "attn.proj.bias"][:md.vision_hidden_size]

    projection_msg = queue_get("vision projection")
    state_dict["mlp1.0.weight"] = projection_msg["vision projection norm weight"]
    #TODO: Find a more principled way to determine if vision projection has norm bias
    if md.vision_model_type == "internvit":
        state_dict["mlp1.0.bias"] = projection_msg["vision projection norm bias"]
    state_dict["mlp1.1.weight"] = projection_msg["vision projection l0 weight"]
    state_dict["mlp1.3.weight"] = projection_msg["vision projection l1 weight"]
    if md.vision_projection_linear_bias:
        state_dict["mlp1.1.bias"] = projection_msg["vision projection l0 bias"]
        state_dict["mlp1.3.bias"] = projection_msg["vision projection l1 bias"]

    embeddings_msg = queue_get("embeddings")
    state_dict["language_model.model.embed_tokens.weight"] = embeddings_msg["word embeddings"]
    if md.position_embedding_type == "learned_absolute":
        # TODO: Below may not be correct, but none of our models use absolute positional embeddings
        state_dict["language_model.embeddings.position_embedding"] = embeddings_msg["position embeddings"]

    for i in range(md.num_layers):
        message = queue_get(f"transformer layer {i}")
        prefix = f"language_model.model.layers.{i}."

        state_dict[prefix + 'input_layernorm.weight'] = message["input norm weight"]
        state_dict[prefix + 'post_attention_layernorm.weight'] = message["post norm weight"]
        if md.norm_has_bias:
            state_dict[prefix + 'input_layernorm.bias'] = message["input norm bias"]
            state_dict[prefix + 'post_attention_layernorm.bias'] = message["post norm bias"]

        if md.swiglu:
            state_dict[prefix + 'mlp.gate_proj.weight'] = message["mlp l0 weight W"]
            state_dict[prefix + 'mlp.up_proj.weight'] = message["mlp l0 weight V"]
        else:
            state_dict[prefix + 'mlp.fc1.weight'] = message["mlp l0 weight"]
        if md.linear_bias:
            if md.swiglu:
                state_dict[prefix + 'mlp.gate_proj.bias'] = message["mlp l0 bias W"]
                state_dict[prefix + 'mlp.up_proj.bias'] = message["mlp l0 bias V"]
            else:
                state_dict[prefix + 'mlp.fc1.bias'] = message["mlp l0 bias"] 

        qkv_weight = message["qkv weight"]
        if md.qkv_bias:
            qkv_bias = message["qkv bias"]
        # qkv_weight = qkv_weight.view(qwen_conf.num_attention_heads, 3, -1, qwen_conf.hidden_size)

        nh = md.num_attention_heads
        ng = md.num_query_groups
        dim = md.kv_channels
        tp = md.previous_tensor_parallel_size

        params_per_tp = torch.chunk(qkv_weight, tp, dim=0)
        if md.qkv_bias:
            bias_per_tp = torch.chunk(qkv_bias, tp, dim=0)

        q = torch.empty(0)
        k = torch.empty(0)
        v = torch.empty(0)

        for t in params_per_tp:
            qp = t[:dim * nh // ng, :]
            kp = t[dim * nh // ng:dim * nh // ng + dim, :]
            vp = t[dim * nh // ng + dim:, :]

            q = torch.cat([q, qp])
            k = torch.cat([k, kp])
            v = torch.cat([v, vp])

        if md.qkv_bias:
            qb = torch.empty(0)
            kb = torch.empty(0)
            vb = torch.empty(0)

            for b in bias_per_tp:
                qbp = b[:dim * nh // ng]
                kbp = b[dim * nh // ng:dim * nh // ng + dim]
                vbp = b[dim * nh // ng + dim:]

                qb = torch.cat([qb, qbp])
                kb = torch.cat([kb, kbp])
                vb = torch.cat([vb, vbp])

        # qkv_weight = qkv_weight.transpose(0, 1).reshape(3, qwen_conf.hidden_size, qwen_conf.hidden_size)
        q = q.clone().detach().contiguous()
        state_dict[prefix + 'self_attn.q_proj.weight'] = q
        k = k.clone().detach().contiguous()
        state_dict[prefix + 'self_attn.k_proj.weight'] = k
        v = v.clone().detach().contiguous()
        state_dict[prefix + 'self_attn.v_proj.weight'] = v

        if md.qkv_bias:
            qb = qb.clone().detach().contiguous()
            state_dict[prefix + 'self_attn.q_proj.bias'] = qb
            kb = kb.clone().detach().contiguous()
            state_dict[prefix + 'self_attn.k_proj.bias'] = kb
            vb = vb.clone().detach().contiguous()
            state_dict[prefix + 'self_attn.v_proj.bias'] = vb

        if md.swiglu:
            state_dict[prefix + 'mlp.down_proj.weight'] = message["mlp l1 weight"]
        else:
            state_dict[prefix + 'mlp.fc2.weight'] = message["mlp l1 weight"]
        state_dict[prefix + 'self_attn.o_proj.weight'] = message["dense weight"]
        if md.linear_bias:
            if md.swiglu:
                state_dict[prefix + 'mlp.down_proj.bias'] = message["mlp l1 bias"]
            else:
                state_dict[prefix + 'mlp.fc2.bias'] = message["mlp l1 bias"]
            state_dict[prefix + 'self_attn.o_proj.bias'] = message["dense bias"]

    state_dict["language_model.model.norm.weight"] = queue_get('final norm')['weight']
    state_dict["language_model.lm_head.weight"] = queue_get('output layer')['weight']

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # Store the state_dict to file.
    max_shard_size = "4GB"
    shards, index = shard_checkpoint(state_dict, max_shard_size=max_shard_size)

    # Save the model
    for shard_file, shard in shards.items():
        torch.save(shard, os.path.join(args.save_dir, shard_file))

    if index is None:
        print(f"Model weights saved in {os.path.join(args.save_dir, WEIGHTS_NAME)}")
    else:
        import json
        save_index_file = os.path.join(args.save_dir, WEIGHTS_INDEX_NAME)
        # Save the index as well
        with open(save_index_file, "w", encoding="utf-8") as f:
            content = json.dumps(index, indent=2, sort_keys=True) + "\n"
            f.write(content)
        print(
            f"The model is bigger than the maximum size per checkpoint ({max_shard_size}) and is going to be "
                f"split in {len(shards)} checkpoint shards. You can find where each parameters has been saved in the "
                f"index located at {save_index_file}."
        )

    print("Done!")
