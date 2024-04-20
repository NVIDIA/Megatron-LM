import sys
import os
import numpy as np
import torch
import torch.multiprocessing as mp
from transformers import AutoModelForCausalLM, MixtralConfig


def add_arguments(parser):
    group = parser.add_argument_group(title="Mixtral-8x7b HF saver.")
    group.add_argument(
        "--megatron-path", type=str, default=None, help="Base directory of megatron checkpoint"
    )
    group.add_argument(
        "--target-tensor-parallel-size", type=int,
        help="Target tensor model parallel size, defaults to the tensor parallel size "
        "in the input checkpoint if provided by the loader, otherwise to 1"
    )
    group.add_argument(
        "--target-pipeline-parallel-size", type=int,
        help="Target tensor model parallel size, default to the pipeline parall size "
        "in the input checkpoint if provided by the loader, otherwise to 1"
    )
    group.add_argument(
        "--check-eq-with-hf", type=str, default="/mnt/nfs/mixtral/models/Mixtral-8x7B-v0.1",
        help="Check the weights with given HF model"
    )
    group.add_argument(
        "--weight-check", action="store_true",
        help="Check the weights value with given HF model"
    )


def save_checkpoint(queue: mp.Queue, args):
    def queue_get(name=None):
        val = queue.get()
        if val == "exit":
            print("Loader exited, exiting saver")
            exit(1)
        if name is not None and args.checking and val["name"] != name:
            val_name = val["name"]
            print(f"Unexpected message. Expecting {name} but got {val_name}. Exiting saver.")
            exit(1)
        if name is not None:
            print(f"received {name}")
        return val

    sys.path.append(os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            os.path.pardir,
            os.path.pardir
        )
    ))
    if args.megatron_path is not None:
        sys.path.insert(0, args.megatron_path)

    md = queue_get()

    # Verify compatibility of args
    assert hasattr(md, "checkpoint_args")
    assert md.model_type == "GPT"
    mag_conf = md.checkpoint_args
    torch_dtype = torch.float32
    if mag_conf.bf16:
        assert mag_conf.fp16 == False
        torch_dtype = torch.bfloat16
    elif mag_conf.fp16:
        assert mag_conf.bf16 == False
        torch_dtype = torch.float16
    assert mag_conf.swiglu == True
    assert mag_conf.rotary_percent == 1.0

    mixtral_conf = MixtralConfig(
        # https://huggingface.co/mistralai/Mixtral-8x7B-v0.1/blob/main/config.json
        vocab_size              = mag_conf.padded_vocab_size,   # 320000 or padded_vocab_size
        hidden_size             = mag_conf.hidden_size,         # 4096
        intermediate_size       = mag_conf.ffn_hidden_size,     # 14336
        num_hidden_layers       = mag_conf.encoder_num_layers,  # 32
        num_key_value_heads     = mag_conf.num_query_groups,    # 8
        num_local_experts       = mag_conf.num_experts,         # 8
        num_experts_per_tok     = mag_conf.num_experts_per_tok, # 2
        output_router_logits    = False,
        rope_theta              = 1000000.0,
        router_aux_loss_coef    = mag_conf.router_aux_loss_coef,  # 0.02
        max_position_embeddings = mag_conf.max_position_embeddings,  # 32768
        rms_norm_eps            = mag_conf.norm_epsilon,        # 1e-05
        tie_word_embeddings     = not mag_conf.untie_embeddings_and_output_weights,  # False
        attention_bias          = mag_conf.add_bias_linear,     # False
        torch_dtype             = torch_dtype,
        model_type              = "mixtral",
        sliding_window          = mag_conf.sliding_window_size,
        architectures           = ["MixtralForCausalLM"],
        transformers_version    = "4.36.0.dev0"
    )
    mixtral_conf.save_pretrained(args.save_dir)

    state_dict = {}
    def set_hf_param(name, tensor: torch.Tensor):
        weight_name = f"{name}.weight"
        state_dict[weight_name] = tensor

    set_hf_param("model.embed_tokens", queue_get("embeddings")["word embeddings"])
    kv_channels = mag_conf.hidden_size // mag_conf.num_attention_heads
    qkv_split_sizes = [
        mag_conf.hidden_size // mag_conf.num_query_groups,
        kv_channels,
        kv_channels,
    ] * mag_conf.num_query_groups

    for i_layer in range(mixtral_conf.num_hidden_layers):
        message = queue_get(f"transformer layer {i_layer}")
        suffix = f"model.layers.{i_layer}."
        set_hf_param(suffix + "input_layernorm", message["input norm weight"])
        set_hf_param(suffix + "post_attention_layernorm", message["post norm weight"])
        if mag_conf.add_bias_linear:
            raise NotImplementedError("add_bias_linear is not supported.")
        set_hf_param(suffix + "block_sparse_moe.gate", message["mlp gate weight"])
        for expert_idx in range(mixtral_conf.num_local_experts):
            set_hf_param(suffix + f"block_sparse_moe.experts.{expert_idx}.w1", message[f"mlp expert{expert_idx} w1"])
            set_hf_param(suffix + f"block_sparse_moe.experts.{expert_idx}.w2", message[f"mlp expert{expert_idx} w2"])
            set_hf_param(suffix + f"block_sparse_moe.experts.{expert_idx}.w3", message[f"mlp expert{expert_idx} w3"])

        qkv_weight = message["qkv weight"]  # [6144, 4096]
        # q, k, v, q, k, v, ...という順番でgroup分並んでいるのでそれぞれ分割
        qkv_split_weights = torch.split(qkv_weight, qkv_split_sizes)
        q_segments = []
        k_segments = []
        v_segments = []
        for i in range(mag_conf.num_query_groups):
            q_segments.append(qkv_split_weights[i * 3])
            k_segments.append(qkv_split_weights[i * 3 + 1])
            v_segments.append(qkv_split_weights[i * 3 + 2])
        q_proj = torch.cat(q_segments, dim=0)
        k_proj = torch.cat(k_segments, dim=0)
        v_proj = torch.cat(v_segments, dim=0)
        set_hf_param(suffix + "self_attn.q_proj", q_proj)
        set_hf_param(suffix + "self_attn.k_proj", k_proj)
        set_hf_param(suffix + "self_attn.v_proj", v_proj)
        set_hf_param(suffix + "self_attn.o_proj", message["dense weight"])
    print("received attention layers")
    set_hf_param("model.norm", queue_get("final norm")["weight"])
    set_hf_param("lm_head", queue_get("output layer")["weight"])

    check_eq_with_hf = args.check_eq_with_hf
    if check_eq_with_hf:
        print(f"Checking with given HF model {check_eq_with_hf}")
        ref_model = AutoModelForCausalLM.from_pretrained(check_eq_with_hf)
        ref_state_dict = ref_model.state_dict()
        for key in ref_state_dict:
            if key not in state_dict:
                print(f'Key {key} not found in state_dict')
        assert sorted(list(ref_state_dict.keys())) == sorted(list(state_dict.keys()))
        for key in state_dict:
            print(f"Checking {key}")
            # shapeは確実に一致することを確認. embeddingのweightはvocab_sizeがおなじの時だけチェック
            if (ref_model.vocab_size != mixtral_conf.vocab_size) and ("embed_tokens" in key or "lm_head" in key):
                print(f"Skip shape check for {key}")
            else:
                assert ref_state_dict[key].shape == state_dict[key].shape, f"Shape of {key} not equal"
            # かなり小さい小数点の違いで一致しないので、np.iscloseでチェック
            # 以下のようなtorch.equalは一致しない
            # assert torch.equal(ref_state_dict[key], state_dict[key]), f"Key {key} not equal"
            if args.weight_check:
                if (ref_model.vocab_size != mixtral_conf.vocab_size) and ("embed_tokens" in key or "lm_head" in key):
                        print(f"Skip weight check for {key}")
                else:
                    check_atol_list = [1e-4, 1e-5]  # 時間かかるので2個だけ
                    ref_numpy = ref_state_dict[key].cpu().numpy()
                    converted_numpy = state_dict[key].cpu().numpy()
                    for atol in check_atol_list:
                        if not np.all(np.isclose(ref_numpy, converted_numpy, atol=atol)):
                            # 1e-5でチェックすればよほどOKなはず
                            if atol >= 1e-5:
                                raise AssertionError(f"Key {key} not equal with atol {atol}")
                            else:
                                print(f"Key {key} not equal with atol {atol}")
        print("Check passed.")

    print("Saving state_dict as bin file...")
    torch.save(state_dict, os.path.join(args.save_dir, "pytorch_model.bin"))
    print("Saving model as HF safetensors...")
    _ = AutoModelForCausalLM.from_pretrained(args.save_dir).save_pretrained(args.save_dir)
    os.remove(os.path.join(args.save_dir, "pytorch_model.bin"))
    print("Done!")
