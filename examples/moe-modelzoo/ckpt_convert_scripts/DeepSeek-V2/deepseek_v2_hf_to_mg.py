import json
import os
import random
import re

import numpy as np
import torch
import torch.distributed as dist
from safetensors import safe_open

from megatron.training import get_args
from megatron.training.checkpointing import get_checkpoint_name
from megatron.training.initialize import initialize_megatron
from pretrain_gpt import model_provider

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)


def init_distributed_environment(backend="nccl", port="12355"):
    """Initialize the distributed environment for checkpoint conversion.

    Args:
        backend (str): Distributed backend ('nccl', 'gloo', or 'mpi'). Default: 'nccl'.
        port (str): Port number for distributed communication. Default: '12355'.
    """
    try:
        # Set deterministic behavior
        torch.manual_seed(1234)
        torch.cuda.manual_seed(1234)
        random.seed(1234)
        np.random.seed(1234)

        # Configure distributed environment
        os.environ.update({"MASTER_ADDR": "localhost", "MASTER_PORT": port})

        # Initialize process group
        dist.init_process_group(
            backend=backend, init_method="env://", world_size=1, rank=0
        )
    except Exception as e:
        print(f"Failed to initialize distributed environment: {str(e)}")
        raise


def add_extra_args(parser):
    parser.add_argument("--target-tensor-model-parallel-size", type=int, default=1)
    parser.add_argument("--target-pipeline-model-parallel-size", type=int, default=1)
    parser.add_argument("--target-expert-model-parallel-size", type=int, default=1)

    return parser


def load_tensor(weight_file, weight_map, args):
    file_name = weight_map[weight_file]
    ckpt_file_path = os.path.join(args.load, file_name)

    with safe_open(ckpt_file_path, framework="pt", device=0) as f:
        weight = f.get_tensor(weight_file)

    if args.bf16:
        return weight.bfloat16()
    elif args.fp16:
        return weight.float16()
    else:
        return weight


def convert_ckpt_from_hf_to_megatron(mg_model, hf_index_path, args):
    print("Start copying")
    if args.bf16:
        mg_model = mg_model.bfloat16()
    elif args.fp16:
        mg_model = mg_model.float16()

    # Load weight map
    with open(hf_index_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        weight_map = data["weight_map"]

    with torch.no_grad():
        mg_model.embedding.word_embeddings.weight.copy_(
            load_tensor(f"model.embed_tokens.weight", weight_map, args)
        )

        for mg_layer_idx, mg_layer in enumerate(mg_model.decoder.layers):
            hf_layer_idx = mg_layer_idx
            hf_layer = f"model.layers.{hf_layer_idx}"

            # Input layernorm
            mg_layer.input_layernorm.weight.copy_(
                load_tensor(f"{hf_layer}.input_layernorm.weight", weight_map, args)
            )

            # Multi-latent attention
            if args.q_lora_rank is not None:
                mg_layer.self_attention.linear_q_down_proj.weight.copy_(
                    load_tensor(f"{hf_layer}.self_attn.q_a_proj.weight", weight_map, args)
                )
                mg_layer.self_attention.linear_q_up_proj.weight.copy_(
                    load_tensor(f"{hf_layer}.self_attn.q_b_proj.weight", weight_map, args)
                )
                mg_layer.self_attention.linear_q_up_proj.layer_norm_weight.copy_(
                    load_tensor(f"{hf_layer}.self_attn.q_a_layernorm.weight", weight_map, args)
                )
            else:
                mg_layer.self_attention.linear_q_proj.weight.copy_(
                    load_tensor(f"{hf_layer}.self_attn.q_proj.weight", weight_map, args)
                )

            mg_layer.self_attention.linear_kv_down_proj.weight.copy_(
                load_tensor(f"{hf_layer}.self_attn.kv_a_proj_with_mqa.weight", weight_map, args)
            )
            mg_layer.self_attention.linear_kv_up_proj.weight.copy_(
                load_tensor(f"{hf_layer}.self_attn.kv_b_proj.weight", weight_map, args)
            )
            mg_layer.self_attention.linear_kv_up_proj.layer_norm_weight.copy_(
                load_tensor(f"{hf_layer}.self_attn.kv_a_layernorm.weight", weight_map, args)
            )
            mg_layer.self_attention.linear_proj.weight.copy_(
                load_tensor(f"{hf_layer}.self_attn.o_proj.weight", weight_map, args)
            )

            # Dense layer
            if mg_layer_idx == 0:
                mg_layer.mlp.linear_fc1.layer_norm_weight.copy_(
                    load_tensor(f"{hf_layer}.post_attention_layernorm.weight", weight_map, args)
                )
                gate_proj = load_tensor(f"{hf_layer}.mlp.gate_proj.weight", weight_map, args)
                up_proj = load_tensor(f"{hf_layer}.mlp.up_proj.weight", weight_map, args)
                hf_fc1 = torch.cat([gate_proj, up_proj], dim=0)

                mg_layer.mlp.linear_fc1.weight.copy_(hf_fc1)
                mg_layer.mlp.linear_fc2.weight.copy_(
                    load_tensor(f"{hf_layer}.mlp.down_proj.weight", weight_map, args)
                )

            # MoE layer
            else:
                mg_layer.pre_mlp_layernorm.weight.copy_(
                    load_tensor(f"{hf_layer}.post_attention_layernorm.weight", weight_map, args)
                )
                mg_layer.mlp.router.weight.copy_(
                    load_tensor(f"{hf_layer}.mlp.gate.weight", weight_map, args)
                )

                if args.moe_grouped_gemm:
                    for expert_idx in range(args.num_experts):
                        gate_proj = load_tensor(f"{hf_layer}.mlp.experts.{expert_idx}.gate_proj.weight", weight_map, args)
                        up_proj = load_tensor(f"{hf_layer}.mlp.experts.{expert_idx}.up_proj.weight", weight_map, args)
                        hf_expert_fc1 = torch.cat([gate_proj, up_proj], dim=0)
        
                        getattr(mg_layer.mlp.experts.linear_fc1, f"weight{expert_idx}").copy_(hf_expert_fc1)
                        getattr(mg_layer.mlp.experts.linear_fc2, f"weight{expert_idx}").copy_(
                            load_tensor(f"{hf_layer}.mlp.experts.{expert_idx}.down_proj.weight", weight_map, args)
                        )

                else:
                    for expert_idx in range(args.num_experts):
                        gate_proj = load_tensor(f"{hf_layer}.mlp.experts.{expert_idx}.gate_proj.weight", weight_map, args)
                        up_proj = load_tensor(f"{hf_layer}.mlp.experts.{expert_idx}.up_proj.weight", weight_map, args)
                        hf_expert_fc1 = torch.cat([gate_proj, up_proj], dim=0)
                        
                        expert = getattr(
                            mg_layer.mlp.experts.local_experts, str(expert_idx)
                        )
                        expert.linear_fc1.weight.copy_(hf_expert_fc1)
                        expert.linear_fc2.weight.copy_(
                            load_tensor(f"{hf_layer}.mlp.experts.{expert_idx}.down_proj.weight", weight_map, args)
                        )

                # Shared experts
                shared_gate_proj = load_tensor(f"{hf_layer}.mlp.shared_experts.gate_proj.weight", weight_map, args)
                shared_up_proj = load_tensor(f"{hf_layer}.mlp.shared_experts.up_proj.weight", weight_map, args)
                shared_experts_fc1 = torch.cat([shared_gate_proj, shared_up_proj], dim=0)

                mg_layer.mlp.shared_experts.linear_fc1.weight.copy_(shared_experts_fc1)
                mg_layer.mlp.shared_experts.linear_fc2.weight.copy_(
                    load_tensor(f"{hf_layer}.mlp.shared_experts.down_proj.weight", weight_map, args)
                )

        # Output layer
        mg_model.decoder.final_layernorm.weight.copy_(
            load_tensor("model.norm.weight", weight_map, args)
        )
        mg_model.output_layer.weight.copy_(
            load_tensor("lm_head.weight", weight_map, args)
        )


def save_state_dict(args, model, checkpoint_name):
    state_dict = {
        "args": args,
        "checkpoint_version": 3.0,
        "iteration": 0,
        "model": model,
    }

    checkpoint_dir = os.path.dirname(checkpoint_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"Saving model checkpoint to: {checkpoint_name}")
    torch.save(state_dict, checkpoint_name)


def save_mg_model(mg_model, args):
    print("Start saving")
    args.tensor_model_parallel_size = args.target_tensor_model_parallel_size
    args.pipeline_model_parallel_size = args.target_pipeline_model_parallel_size
    args.expert_model_parallel_size = args.target_expert_model_parallel_size

    os.makedirs(args.save, exist_ok=True)
    os.system("cp -rf " + args.load + "/config*.json " + args.save)
    os.system("cp -rf " + args.load + "/tokenizer* " + args.save)

    tracker_filepath = os.path.join(args.save, "latest_checkpointed_iteration.txt")
    with open(tracker_filepath, "w") as f:
        f.write("release")

    full_model = mg_model.state_dict_for_save_checkpoint()
    for key in list(full_model.keys()):
        if full_model[key] is None:  # or "_extra_state" in k:
            full_model.pop(key)

    if (
        args.tensor_model_parallel_size == 1
        and args.pipeline_model_parallel_size == 1
        and args.expert_model_parallel_size == 1
    ):
        checkpoint_name = get_checkpoint_name(
            args.save,
            iteration=0,
            release=True,
        )
        save_state_dict(args, full_model, checkpoint_name)

    elif (
        args.tensor_model_parallel_size == 1
        and args.pipeline_model_parallel_size == 1
        and args.expert_model_parallel_size > 1
        and args.num_experts % args.expert_model_parallel_size == 0
        and not args.moe_grouped_gemm
    ):
        pattern = r"local_experts\.(\d+)\."
        num_local_experts = args.num_experts // args.expert_model_parallel_size

        for ep_rank in range(args.expert_model_parallel_size):
            model_split = {}
            checkpoint_name = get_checkpoint_name(
                args.save,
                iteration=0,
                release=True,
                pipeline_parallel=None,
                tensor_rank=None,
                pipeline_rank=None,
                expert_parallel=True,
                expert_rank=ep_rank,
            )
            print(f"Saving ep_rank {ep_rank} model to {checkpoint_name}")

            # Process model weights for current expert rank
            for key, value in full_model.items():
                if "local_experts" in key:
                    global_expert_id = int(re.findall(pattern, key)[0])
                    if global_expert_id // num_local_experts != ep_rank:
                        continue

                    local_expert_id = global_expert_id % num_local_experts
                    key = key.replace(
                        f"local_experts.{global_expert_id}",
                        f"local_experts.{local_expert_id}",
                    )
                model_split[key] = value

            save_state_dict(args, model_split, checkpoint_name)

    elif (
        args.tensor_model_parallel_size == 1
        and args.pipeline_model_parallel_size == 1
        and args.expert_model_parallel_size > 1
        and args.num_experts % args.expert_model_parallel_size == 0
        and args.moe_grouped_gemm
    ):
        pattern = r"weight(\d+)"
        num_local_experts = args.num_experts // args.expert_model_parallel_size

        for ep_rank in range(args.expert_model_parallel_size):
            model_split = {}
            checkpoint_name = get_checkpoint_name(
                args.save,
                iteration=0,
                release=True,
                pipeline_parallel=None,
                tensor_rank=None,
                pipeline_rank=None,
                expert_parallel=True,
                expert_rank=ep_rank,
            )
            print(f"[GroupGEMM] Saving ep_rank {ep_rank} model to {checkpoint_name}")

            # Process model weights for current expert rank
            for key, value in full_model.items():
                if "experts" in key and "weight" in key and "shared_experts" not in key:
                    match = re.search(pattern, key)
                    global_expert_id = int(match.group(1))
                    if global_expert_id // num_local_experts != ep_rank:
                        continue

                    local_expert_id = global_expert_id % num_local_experts
                    key = key.replace(
                        f"weight{global_expert_id}", f"weight{local_expert_id}"
                    )
                model_split[key] = value

            save_state_dict(args, model_split, checkpoint_name)

    elif (
        args.tensor_model_parallel_size == 1
        and args.pipeline_model_parallel_size > 1
        and args.num_experts % args.expert_model_parallel_size == 0
    ):
        # Ensure layers can be evenly divided by pipeline model parallel size
        assert args.num_layers % args.pipeline_model_parallel_size == 0
        layers_per_pipeline = args.num_layers // args.pipeline_model_parallel_size

        pattern = r"weight(\d+)"
        num_local_experts = args.num_experts // args.expert_model_parallel_size

        for pp_rank in range(args.pipeline_model_parallel_size):
            # Get the current range of layers for this pipeline stage
            pp_start = pp_rank * layers_per_pipeline
            pp_end = pp_start + layers_per_pipeline - 1

            for ep_rank in range(args.expert_model_parallel_size):
                model_split = {}
                checkpoint_name = get_checkpoint_name(
                    args.save,
                    iteration=0,
                    release=True,
                    pipeline_parallel=True,
                    tensor_rank=None,
                    pipeline_rank=pp_rank,
                    expert_parallel=True,
                    expert_rank=ep_rank,
                )
                print(f"Saving pp_rank {pp_rank}, ep_rank {ep_rank} model to {checkpoint_name}")

                for key, value in full_model.items():
                    # First pipeline stage
                    if "embedding" in key:
                        if pp_rank == 0:
                            model_split[key] = value
                        continue

                    # Last pipeline stage
                    if "final_layernorm" in key or "output_layer" in key:
                        if pp_rank == args.pipeline_model_parallel_size - 1:
                            model_split[key] = value
                        continue

                    # Skip if the layer doesn't belong current pipeline stage
                    original_layer_id = int(key.split(".")[2])
                    if not pp_start <= original_layer_id <= pp_end:
                        continue

                    # Remap layer index for current pipeline stage
                    local_layer_id = original_layer_id % layers_per_pipeline
                    key = key.replace(
                        f"layers.{original_layer_id}", f"layers.{local_layer_id}"
                    )

                    if (
                        "experts" in key
                        and "weight" in key
                        and "shared_experts" not in key
                    ):
                        match = re.search(pattern, key)
                        global_expert_id = int(match.group(1))
                        if global_expert_id // num_local_experts != ep_rank:
                            continue

                        local_expert_id = global_expert_id % num_local_experts
                        key = key.replace(
                            f"weight{global_expert_id}", f"weight{local_expert_id}"
                        )
                    model_split[key] = value

                save_state_dict(args, model_split, checkpoint_name)

    # elif (
    #     args.tensor_model_parallel_size > 1
    #     and args.pipeline_model_parallel_size == 1
    #     and args.num_experts % args.expert_model_parallel_size == 0
    # ):
    #     pattern = r"weight(\d+)"
    #     num_local_experts = args.num_experts // args.expert_model_parallel_size

    #     for tp_rank in range(args.tensor_model_parallel_size):
    #         for ep_rank in range(args.expert_model_parallel_size):
    #             model_split = {}
    #             if args.expert_model_parallel_size > 1:
    #                 checkpoint_name = get_checkpoint_name(
    #                     args.save,
    #                     iteration=0,
    #                     release=True,
    #                     pipeline_parallel=None,
    #                     tensor_rank=tp_rank,
    #                     pipeline_rank=None,
    #                     expert_parallel=True,
    #                     expert_rank=ep_rank,
    #                 )
    #                 print(f"Saving tp_rank {tp_rank}, ep_rank {ep_rank} model to {checkpoint_name}")

    #             elif args.expert_model_parallel_size == 1:
    #                 checkpoint_name = get_checkpoint_name(
    #                     args.save,
    #                     iteration=0,
    #                     release=True,
    #                     pipeline_parallel=None,
    #                     tensor_rank=tp_rank,
    #                     pipeline_rank=None,
    #                     expert_parallel=False,
    #                 )
    #                 print(f"Saving tp_rank {tp_rank} model to {checkpoint_name}")

    #             for key, value in full_model.items():
    #                 if not isinstance(value, torch.Tensor):
    #                     model_split[key] = value
    #                 elif "linear_q_proj" in key or "linear_q_a_proj" in key:
    #                     seg = value.shape[0] // args.tensor_model_parallel_size
    #                     target_value = value[seg * tp_rank : seg * (tp_rank + 1)]
    #                 elif "linear_q_b_proj" in key:
    #                     seg_0 = value.shape[0] // args.tensor_model_parallel_size
    #                     seg_1 = value.shape[1] // args.tensor_model_parallel_size
    #                     target_value = value[
    #                         seg_0 * tp_rank : seg_0 * (tp_rank + 1),
    #                         seg_1 * tp_rank : seg_1 * (tp_rank + 1),
    #                     ]
    #                 elif "q_a_layernorm" in key:
    #                     seg = value.shape[0] // args.tensor_model_parallel_size
    #                     target_value = value[seg * tp_rank : seg * (tp_rank + 1)]
    #                 elif "linear_kv_b_proj" in key:
    #                     seg = value.shape[0] // args.tensor_model_parallel_size
    #                     target_value = value[seg * tp_rank : seg * (tp_rank + 1)]
    #                 elif "linear_proj" in key:
    #                     seg = value.shape[1] // args.tensor_model_parallel_size
    #                     target_value = value[:, seg * tp_rank : seg * (tp_rank + 1)]
    #                 elif "embedding" in key or "output_layer" in key:
    #                     seg = value.shape[0] // args.tensor_model_parallel_size
    #                     target_value = value[seg * tp_rank : seg * (tp_rank + 1)]
    #                 elif "decoder.layers.0.mlp.linear_fc2" in key:
    #                     seg = value.shape[1] // args.tensor_model_parallel_size
    #                     target_value = value[:, seg * tp_rank : seg * (tp_rank + 1)]
    #                 elif "decoder.layers.0.mlp.linear_fc1" in key:
    #                     viewed = value.view(-1, args.ffn_hidden_size, args.hidden_size)
    #                     seg = args.ffn_hidden_size // args.tensor_model_parallel_size
    #                     target_value = viewed[
    #                         :, seg * tp_rank : seg * (tp_rank + 1), :
    #                     ].reshape(-1, args.hidden_size)
    #                 elif "local_experts" in key:
    #                     expert_rank = int(re.findall(pattern, key)[0])
    #                     if expert_rank // num_local_experts != ep_rank:
    #                         continue
    #                     expert_local_rank = expert_rank % num_local_experts
    #                     if "linear_fc1" in key and "norm" not in key:
    #                         viewed = value.view(
    #                             -1, args.moe_ffn_hidden_size, args.hidden_size
    #                         )
    #                         seg = (
    #                             args.moe_ffn_hidden_size
    #                             // args.tensor_model_parallel_size
    #                         )
    #                         target_value = viewed[
    #                             :, seg * tp_rank : seg * (tp_rank + 1), :
    #                         ].reshape(-1, args.hidden_size)
    #                     elif "linear_fc2" in key:
    #                         seg = value.shape[1] // args.tensor_model_parallel_size
    #                         target_value = value[:, seg * tp_rank : seg * (tp_rank + 1)]
    #                     key = key.replace(
    #                         f"local_experts.{expert_rank}",
    #                         f"local_experts.{expert_local_rank}",
    #                     )
    #                 elif "shared_expert" in key and "gate" not in key:
    #                     if "linear_fc1" in key:
    #                         viewed = value.view(
    #                             -1,
    #                             args.moe_ffn_hidden_size * args.num_shared_experts,
    #                             args.hidden_size,
    #                         )
    #                         seg = (
    #                             args.moe_ffn_hidden_size
    #                             * args.num_shared_experts
    #                             // args.tensor_model_parallel_size
    #                         )
    #                         target_value = viewed[
    #                             :, seg * tp_rank : seg * (tp_rank + 1), :
    #                         ].reshape(-1, args.hidden_size)
    #                     elif "linear_fc2" in key:
    #                         seg = value.shape[1] // args.tensor_model_parallel_size
    #                         target_value = value[:, seg * tp_rank : seg * (tp_rank + 1)]
    #                 else:
    #                     target_value = value
    #                 model_split[key] = target_value
    #             save_state_dict(args, model_split, checkpoint_name)

    else:
        raise ValueError(
            f"Unsupported model parallel configuration: "
            f"TP={args.tensor_model_parallel_size}, "
            f"PP={args.pipeline_model_parallel_size}, "
            f"EP={args.expert_model_parallel_size}. "
            f"Currently only supports TP=1 with PP>=1 and EP>=1 conversion."
        )

    print(f"Megatron model is saved to {args.save}")


def main():
    initialize_megatron(extra_args_provider=add_extra_args)
    args = get_args()
    mg_model = model_provider()

    hf_index_path = os.path.join(args.load, "model.safetensors.index.json")
    convert_ckpt_from_hf_to_megatron(mg_model, hf_index_path, args)

    save_mg_model(mg_model, args)


if __name__ == "__main__":
    init_distributed_environment()
    main()
