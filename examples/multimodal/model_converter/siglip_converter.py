# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
import argparse
import os
from transformers import PaliGemmaForConditionalGeneration
import torch


def convert(output_path, tensor_parallel_size, use_te):
    device = "cuda"

    model_id = "google/paligemma-3b-pt-448"
    model = PaliGemmaForConditionalGeneration.from_pretrained(model_id).eval()

    model = model.to(device)

    print(model.config)
    for name, tensor in model.state_dict().items():
        if "vision_model" not in name:
            continue
        shape_str = "(" + ", ".join([str(x) for x in tensor.shape]) + ")"
        print(f"{name:<75} {shape_str:>20}")

    state_dict = model.state_dict()
    new_state_dicts = [{"model": dict()} for _ in range(tensor_parallel_size)]

    def add_chunck_tensor(new_tensor, new_name, chunk_dim=None):
        if chunk_dim is None:
            new_tensors = [new_tensor for _ in range(tensor_parallel_size)]
        else:
            new_tensors = torch.chunk(new_tensor, tensor_parallel_size, dim=chunk_dim)

        for i in range(tensor_parallel_size):
            # chunk() creates a view of a bigger tensor. clone() is used here to avoid excessive storage.
            new_state_dicts[i]["model"][new_name] = new_tensors[i].clone()

            # TE sets _extra_state (for FP8 purposes), so set an empty one here for compatibility.
            extra_state_layers = ("linear_qkv", "linear_proj", "linear_fc1", "linear_fc2")
            is_extra_state_layer = any([l in new_name for l in extra_state_layers])
            if use_te and is_extra_state_layer:
                layer = new_name.split(".")[-2]
                if layer in extra_state_layers:
                    extra_state_name = (
                        new_name[: new_name.rfind(".") + 1] + "_extra_state"
                    )  # Replace the weight name.
                    new_state_dicts[i]["model"][extra_state_name] = None

    for name, tensor in state_dict.items():
        if tensor.dtype == torch.float16:
            state_dict[name] = tensor.to(torch.float32)

    add_chunck_tensor(
        state_dict["vision_tower.vision_model.embeddings.position_embedding.weight"],
        "position_embeddings.weight")
    add_chunck_tensor(
        state_dict["vision_tower.vision_model.embeddings.patch_embedding.weight"],
        "conv1.weight")
    add_chunck_tensor(
        state_dict["vision_tower.vision_model.embeddings.patch_embedding.bias"],
        "conv1.bias")

    head_dim = 72
    num_head = 16
    for layer_idx in range(27):
        origin_base = f"vision_tower.vision_model.encoder.layers.{layer_idx}" 
        target_base = f"decoder.layers.{layer_idx}"
        
        for param_type in ["weight", "bias"]:
            # QKV
            q_proj_params = state_dict[f"{origin_base}.self_attn.q_proj.{param_type}"]
            k_proj_params = state_dict[f"{origin_base}.self_attn.k_proj.{param_type}"]
            v_proj_params = state_dict[f"{origin_base}.self_attn.v_proj.{param_type}"]
            # Do some tensor manipulation because megatron expect one tensor
            # projection for the QKV in the order
            # [(Q1, K1, V1), (Q2, K2, V2), ...] where Qi is the query of the
            # i-th head with dimension num_head.
            new_tensor = torch.concatenate([
                q_proj_params.view(num_head, head_dim, -1),
                k_proj_params.view(num_head, head_dim, -1),
                v_proj_params.view(num_head, head_dim, -1)], axis=1).view(
                    3*head_dim*num_head, -1)
            if param_type == "bias":
                new_tensor = new_tensor[:, 0]
            new_name = f"{target_base}.self_attention.linear_qkv.{param_type}"
            add_chunck_tensor(new_tensor, new_name, chunk_dim=0)
            # linear_proj
            add_chunck_tensor(
                state_dict[f"{origin_base}.self_attn.out_proj.{param_type}"],
                f"{target_base}.self_attention.linear_proj.{param_type}",
                chunk_dim=1 if param_type == "weight" else None)
            # layer_norm
            new_name = f"{target_base}.input_layernorm.{param_type}"
            if use_te:
                new_name = f"{target_base}.self_attention.linear_qkv.layer_norm_{param_type}"
            add_chunck_tensor(
                state_dict[f"{origin_base}.layer_norm1.{param_type}"],
                new_name)
            # FC 1
            add_chunck_tensor(
                state_dict[f"{origin_base}.mlp.fc1.{param_type}"],
                f"{target_base}.mlp.linear_fc1.{param_type}",
                chunk_dim=0)
            # FC 2
            add_chunck_tensor(
                state_dict[f"{origin_base}.mlp.fc2.{param_type}"],
                f"{target_base}.mlp.linear_fc2.{param_type}",
                chunk_dim=1 if param_type=="weight" else None)
            # layer_norm
            new_name = f"{target_base}.pre_mlp_layernorm.{param_type}"
            if use_te:
                new_name = f"{target_base}.mlp.linear_fc1.layer_norm_{param_type}"
            add_chunck_tensor(
                state_dict[f"{origin_base}.layer_norm2.{param_type}"],
                new_name)

    add_chunck_tensor(
        state_dict["vision_tower.vision_model.post_layernorm.weight"],
        "ln_post.weight")
    add_chunck_tensor(
        state_dict["vision_tower.vision_model.post_layernorm.bias"],
        "ln_post.bias")

    for i in range(tensor_parallel_size):
        output_dir_tp = os.path.join(output_path, "iter_0000001", f"mp_rank_0{i}")
        os.makedirs(output_dir_tp)
        output_path_tp = os.path.join(output_dir_tp, "model_optim_rng.pt")
        torch.save(new_state_dicts[i], output_path_tp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
Convert SigLIP weights to megatron format.


Example usage:
python siglip_converter.py --tensor-parallel-size 4 --output google_paligemma_3b_pt_44_mcore_tp_4 --use-te

examples/multimodal/combine_mistral_clip.sh /lustre/fsw/portfolios/llmservice/users/jbarker/workspace/checkpoints/Mistral-7B-Instruct-v0.3-mcore-tp4 google_paligemma_3b_pt_44_mcore_tp_4 mistral_7b_instruct_v0p3_google_paligemma_3b_pt_44_mcore_tp_4
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output", type=str, required=True, help="output directory for megatron state dict file(s)"
    )
    parser.add_argument(
        "--tensor-parallel-size", type=int, default=1, help="model tensor parallel size"
    )
    parser.add_argument("--use-te", action="store_true", help="Use Transformer Engine")

    args = parser.parse_args()

    convert(args.output, args.tensor_parallel_size, args.use_te)

    print("done.")
