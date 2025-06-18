# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import os, torch, torch.multiprocessing as mp
from transformers import AutoModelForCausalLM, LlamaConfig, AutoTokenizer
from contextlib import contextmanager

def add_arguments(parser):
    group = parser.add_argument_group(title='Llama2_hf saver.')
    group.add_argument('--hf-tokenizer-path', type=str, default=None,
                       help='Huggingface tokenizer path. eg. /models/llama-2-hf/7b-chat.')


@contextmanager
def suspend_nn_inits():
    """
    create context manager for loading without init

    see https://github.com/huggingface/transformers/issues/26258
    """
    skip = lambda *args, **kwargs: None
    saved_inits = torch.nn.init.kaiming_uniform_, torch.nn.init.uniform_, torch.nn.init.normal_  #saving
    torch.nn.init.kaiming_uniform_ = torch.nn.init.uniform_ = torch.nn.init.normal_ = skip  #replacing
    try:
        yield
    finally:
        torch.nn.init.kaiming_uniform_, torch.nn.init.uniform_, torch.nn.init.normal_ = saved_inits  # restoring

def save_checkpoint(queue: mp.Queue, args):
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

    # Verify compatibility of args
    assert hasattr(md, 'checkpoint_args')
    assert md.model_type == 'GPT'
    mag_conf = md.checkpoint_args
    torch_dtype = torch.float16

    llama_conf = LlamaConfig(
        vocab_size              = mag_conf.padded_vocab_size,
        hidden_size             = mag_conf.hidden_size,
        intermediate_size       = mag_conf.ffn_hidden_size,
        num_hidden_layers       = mag_conf.encoder_num_layers,
        num_attention_heads     = mag_conf.num_attention_heads,
        num_key_value_heads     = mag_conf.num_query_groups,
        max_position_embeddings = mag_conf.max_position_embeddings,
        rms_norm_eps            = mag_conf.norm_epsilon,
        tie_word_embeddings     = not mag_conf.untie_embeddings_and_output_weights,
        attention_bias          = mag_conf.add_bias_linear,
        torch_dtype             = torch_dtype
    )

    state_dict = {}
    def set_hf_param(name, tensor: torch.Tensor):
        weight_name = f'{name}.weight'
        state_dict[weight_name] = tensor.to(torch.float16)

    set_hf_param('model.embed_tokens', queue_get("embeddings")["word embeddings"])
    for i_layer in range(llama_conf.num_hidden_layers):
        message = queue_get(f"transformer layer {i_layer}")
        suffix = f'model.layers.{i_layer}.'
        set_hf_param(suffix + 'input_layernorm', message["input norm weight"])
        set_hf_param(suffix + 'post_attention_layernorm', message["post norm weight"])
        set_hf_param(suffix + 'mlp.gate_proj', message["mlp l0 weight W"])
        set_hf_param(suffix + 'mlp.up_proj', message["mlp l0 weight V"])
        qkv_weight = message["qkv weight"]
        qkv_weight = qkv_weight.view(llama_conf.num_key_value_heads, -1, llama_conf.hidden_size)
        qkv_weight = torch.split(qkv_weight, [
            llama_conf.hidden_size // llama_conf.num_key_value_heads,
            llama_conf.hidden_size // llama_conf.num_attention_heads,
            llama_conf.hidden_size // llama_conf.num_attention_heads,
        ], dim=1)
        set_hf_param(suffix + 'self_attn.q_proj', qkv_weight[0].reshape(-1, llama_conf.hidden_size))
        set_hf_param(suffix + 'self_attn.k_proj', qkv_weight[1].reshape(-1, llama_conf.hidden_size))
        set_hf_param(suffix + 'self_attn.v_proj', qkv_weight[2].reshape(-1, llama_conf.hidden_size))
        set_hf_param(suffix + 'self_attn.o_proj', message["dense weight"])
        set_hf_param(suffix + 'mlp.down_proj', message["mlp l1 weight"])
    set_hf_param('model.norm', queue_get('final norm')['weight'])
    set_hf_param('lm_head', queue_get('output layer')['weight'])

    with suspend_nn_inits():
        print("Saving model to disk ...")
        model = AutoModelForCausalLM.from_pretrained(None, config=llama_conf, state_dict=state_dict, torch_dtype=torch_dtype)
        model.save_pretrained(args.save_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.hf_tokenizer_path)
    tokenizer.save_pretrained(args.save_dir)