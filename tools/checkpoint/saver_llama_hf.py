import sys
import os
import gc
import json
from pathlib import Path
from shutil import rmtree
from tempfile import TemporaryDirectory

import torch
import torch.multiprocessing as mp
from transformers import AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM, GenerationConfig

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
    group = parser.add_argument_group(title="Llama-2 HF saver.")
    group.add_argument(
        "--hf-tokenizer",
        type=str,
        default=None,
        help="Example: epfl-llm/meditron-70b",
    )
    group.add_argument(
        "--check-eq-hf",
        type=str,
        default=None,
        help="check equality with HF model, e.g. epfl-llm/meditron-70b",
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
        assert torch.equal(ref_state_dict[key], state_dict[key])
        ref_state_dict.pop(key)
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
    if check_reference:
        ref_state_dict = perform_check(state_dict, ref_state_dict)
    for layer_name, weight_matrix in state_dict.items():
        index_dict["weight_map"][layer_name] = filename
        index_dict["metadata"]["total_size"] += weight_matrix.numel()
    print(f"saving state dict to {dir_path}/{filename}")
    torch.save(state_dict, f"{dir_path}/{filename}")
    return index_dict, ref_state_dict


def save_checkpoint(queue: mp.Queue, args):
    def queue_get(name=None):
        val = queue.get()
        if val == "exit":
            print("Loader exited, exiting saver")
            exit(1)

        print(f'......{val}......') if val=='done' else None
        if name is not None and args.checking and val["name"] != name :  
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
    if md.model_type != "GPT":
        raise ValueError("wrong model_type in metadata. must be GPT")
    if md.checkpoint_args.position_embedding_type != "rope":
        raise ValueError("LLama model must use RoPE")
    if md.checkpoint_args.use_rope_scaling:
        raise ValueError("LLama model must use llama3 RoPE scaling")
    if not md.checkpoint_args.swiglu:
        raise ValueError("LLama model must use gated linear layers")
    if md.checkpoint_args.normalization != "RMSNorm":
        raise ValueError("LLama model must use RMSNorm")
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
    save_dir.mkdir(exist_ok=True)
    with TemporaryDirectory(prefix=str(save_dir/"tmp")) as tmp_save_dir:
        index_dict = {
            "weight_map": {},
            "metadata": {"total_size": 0},
        }
        tokenizer = None
        ref_state_dict = None

        ### prepare a reference model if needed
        if args.check_eq_hf:
            print(f"preparing checks with given HF model {args.check_eq_hf}")
            ref_model = AutoModelForCausalLM.from_pretrained(args.check_eq_hf)
            ref_state_dict = ref_model.state_dict()

        ### save tokenizer conf files
        if args.hf_tokenizer:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(args.hf_tokenizer)
            print(f"saving tokenizer to {args.save_dir}")
            tokenizer.save_pretrained(args.save_dir)

        ### save config.json
        llama_conf = LlamaConfig(
            vocab_size=md.true_vocab_size if md.true_vocab_size else md.checkpoint_args.padded_vocab_size,
            hidden_size=md.checkpoint_args.hidden_size,
            intermediate_size=md.checkpoint_args.ffn_hidden_size,
            num_hidden_layers=md.checkpoint_args.num_layers,
            num_attention_heads=md.checkpoint_args.num_attention_heads,
            num_key_value_heads=md.checkpoint_args.num_query_groups,
            hidden_act="silu",
            max_position_embeddings=md.checkpoint_args.max_position_embeddings,
            rms_norm_eps=md.checkpoint_args.norm_epsilon,
            tie_word_embeddings=not md.checkpoint_args.untie_embeddings_and_output_weights,
            rope_theta=md.checkpoint_args.rotary_base,
            rope_scaling={"rope_type": "llama3",
                          "factor": md.checkpoint_args.rope_scaling_factor,
                          "original_max_position_embeddings": md.checkpoint_args.max_position_embeddings,
                          "factor": md.checkpoint_args.rope_scaling_factor,
                          "high_freq_factor": 4.0,
                          "low_freq_factor": 1.0},  # high/low freq factor: Set defaults, thay aren't used in megatron.
            attention_bias=md.checkpoint_args.add_qkv_bias,
            mlp_bias=md.checkpoint_args.add_bias_linear,
            torch_dtype=torch_dtype,
            model_type="llama",
            architectures=["LlamaForCausalLM"],
            attention_dropout=md.checkpoint_args.attention_dropout,
            hidden_dropout=md.checkpoint_args.hidden_dropout,
        )
        if args.hf_tokenizer:
            llama_conf.eos_token_id = tokenizer.eos_token_id
            llama_conf.bos_token_id = tokenizer.bos_token_id

        print(f"saving config.json to {tmp_save_dir}")
        llama_conf.save_pretrained(tmp_save_dir)

        ### save embedding layer
        # Deal with padding
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

        # We also keep embeddings as we might need them to set the lm_head weight.
        embeddings = pad_weight(queue_get("embeddings")["word embeddings"], md.true_vocab_size)
        state_dict = {
            "model.embed_tokens.weight": embeddings
        }
        index_dict, ref_state_dict = save_layer(
            state_dict,
            index_dict,
            dir_path=tmp_save_dir,
            filename="pytorch_model-embedding.bin",
            check_reference=args.check_eq_hf,
            ref_state_dict=ref_state_dict,
        )

        ### save every transformer layer
        head_size = llama_conf.hidden_size // llama_conf.num_attention_heads
        heads_per_group = llama_conf.num_attention_heads // llama_conf.num_key_value_heads
        qkv_total_heads = llama_conf.num_attention_heads + 2 * llama_conf.num_key_value_heads
        for i_layer in range(llama_conf.num_hidden_layers):
            message = queue_get(f"transformer layer {i_layer}")
            state_dict = {
                f"model.layers.{i_layer}.input_layernorm.weight": message[
                    "input norm weight"
                ],
                f"model.layers.{i_layer}.post_attention_layernorm.weight": message[
                    "post norm weight"
                ],
                f"model.layers.{i_layer}.mlp.gate_proj.weight": message["mlp l0 weight W"],
                f"model.layers.{i_layer}.mlp.up_proj.weight": message["mlp l0 weight V"],
                f"model.layers.{i_layer}.self_attn.o_proj.weight": message["dense weight"],
                f"model.layers.{i_layer}.mlp.down_proj.weight": message["mlp l1 weight"]
            }
            if md.checkpoint_args.add_bias_linear:
                state_dict |= {
                    f"model.layers.{i_layer}.self_attn.o_proj.bias": message["dense bias"],
                    f"model.layers.{i_layer}.mlp.gate_proj.bias": message["mlp l0 bias W"],
                    f"model.layers.{i_layer}.mlp.up_proj.bias": message["mlp l0 bias V"],
                    f"model.layers.{i_layer}.mlp.down_proj.bias": message["mlp l1 bias"],
                }
            q_slice = torch.cat(
                [
                    torch.arange(
                        (heads_per_group + 2) * i,
                        (heads_per_group + 2) * i + heads_per_group,
                    )
                    for i in range(llama_conf.num_key_value_heads)
                ]
            )
            k_slice = torch.arange(heads_per_group, qkv_total_heads, (heads_per_group + 2))
            v_slice = torch.arange(
                heads_per_group + 1, qkv_total_heads, (heads_per_group + 2)
            )
            qkv_weights = message["qkv weight"]
            qkv_weights = qkv_weights.reshape(
                [qkv_total_heads, head_size, llama_conf.hidden_size]
            )
            state_dict |= {
                f"model.layers.{i_layer}.self_attn.q_proj.weight": qkv_weights[
                    q_slice
                ].reshape(-1, llama_conf.hidden_size),
                f"model.layers.{i_layer}.self_attn.k_proj.weight": qkv_weights[
                    k_slice
                ].reshape(-1, llama_conf.hidden_size),
                f"model.layers.{i_layer}.self_attn.v_proj.weight": qkv_weights[
                    v_slice
                ].reshape(-1, llama_conf.hidden_size)
            }
            if md.checkpoint_args.add_bias_linear:
                qkv_bias = message["qkv bias"]
                qkv_bias = qkv_bias.reshape([qkv_total_heads, head_size])
                state_dict |= {
                    f"model.layers.{i_layer}.self_attn.q_proj.bias": qkv_bias[
                        q_slice
                    ].reshape(-1, llama_conf.hidden_size),
                    f"model.layers.{i_layer}.self_attn.k_proj.bias": qkv_bias[
                        k_slice
                    ].reshape(-1, llama_conf.hidden_size),
                    f"model.layers.{i_layer}.self_attn.v_proj.bias": qkv_bias[
                        v_slice
                    ].reshape(-1, llama_conf.hidden_size)
                }
            index_dict, ref_state_dict = save_layer(
                state_dict,
                index_dict,
                dir_path=tmp_save_dir,
                filename=f"pytorch_model-{i_layer + 1}.bin",
                check_reference=args.check_eq_hf,
                ref_state_dict=ref_state_dict,
            )

        if not md.checkpoint_args.untie_embeddings_and_output_weights:  # tied embeddings and lm-head
            state_dict = {
                "model.norm.weight": queue_get("final norm")["weight"],
                "lm_head.weight": embeddings
                }
        else:                                                           # untied embeddings and lm-head
            state_dict = {
                "model.norm.weight": queue_get("final norm")["weight"],
                "lm_head.weight": pad_weight(queue_get("output layer")["weight"], md.true_vocab_size) 
        }
        index_dict, ref_state_dict = save_layer(
            state_dict,
            index_dict,
            dir_path=tmp_save_dir,
            filename="pytorch_model-lm-head.bin",
            check_reference=args.check_eq_hf,
            ref_state_dict=ref_state_dict,
        )
        # final check
        assert (
            not ref_state_dict
        ), "reference state dict has additional layers not present in model."

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
        print(f"Loading the converted pytorch checkpoint in a Llama HF model from {tmp_save_dir}")
        model = LlamaForCausalLM.from_pretrained(
            str(tmp_save_dir), torch_dtype=torch.bfloat16, low_cpu_mem_usage=False # last arg requires a recent version of accelerate
        )

    # Avoid saving this as part of the config.
    del model.config._name_or_path
    model.config.torch_dtype = torch.float16
    print(f"Saving in the Transformers safe tensors format to {args.save_dir}")
    model.save_pretrained(args.save_dir, safe_serialization=True)

    ### save chat config
    generation_config = (
        GenerationConfig(
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            bos_token_id=llama_conf.bos_token_id,
            eos_token_id=llama_conf.eos_token_id,
        )
        if args.save_chat_model
        else GenerationConfig(
            _from_model_config=True,
            bos_token_id=llama_conf.bos_token_id,
            eos_token_id=llama_conf.eos_token_id,
        )
    )
    print(f"Saving chat config to {args.save_dir}")
    generation_config.save_pretrained(args.save_dir)
    queue_get()  # Recv final "exit" message so saver exits gracefully.

