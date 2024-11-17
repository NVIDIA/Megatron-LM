import argparse
import os

import torch
from transformers import AutoModel


def convert(model_name, output_path, tensor_parallel_size, use_te):
    """Convert InternViT HF checkpoint to mcore."""
    hf_model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True
    )

    hf_state_dict = hf_model.state_dict()
    new_state_dicts = [{"model": dict()} for _ in range(tensor_parallel_size)]

    hidden_size = 3200
    num_heads = 25
    dim = 128

    order = torch.ones(3 * hidden_size).long()

    for j in range(num_heads):
        for i in range(dim):
            order[i + dim*3*j] = j*dim+i
            order[dim + i + dim*3*j] = j*dim+i+num_heads*dim
            order[dim*2 + i + dim*3*j] = j*dim+i+num_heads*dim*2

    for name, tensor in hf_state_dict.items():
        # Map parameter names to ones used in megatron.
        new_name = ""
        new_tensor = tensor

        # This is used for chunking some tensors to target tensor parallel size.
        chunk_dim = None

        if "embeddings.class_embedding" in name:
            new_name = "class_token"
        elif "embeddings.patch_embedding.weight" in name:
            new_name = "conv1.weight"
        elif "embeddings.patch_embedding.bias" in name:
            new_name = "conv1.bias"
        elif "embeddings.position_embedding" in name:
            new_name = "position_embeddings.weight"
            new_tensor = new_tensor.squeeze(0)
        elif "encoder.layers" in name:
            layer_idx = name.split(".")[2]

            base = f"decoder.layers.{layer_idx}"

            head_dim = 128

            if tensor_parallel_size == 1:
                num_padded_heads = 25
            elif tensor_parallel_size == 8:
                # Note: 25 is not divisible by 8 and we don't currently support uneven heads split with tensor parallelism.
                # So we pad with dummy all-zero heads. Please use a nice even number of attention heads in your model.
                num_padded_heads = 32
            else:
                raise NotImplementedError("invalid tensor parallel size value:", tensor_parallel_size)

            if "ls1" in name:
                new_name = f"{base}.ls1"
            elif "ls2" in name:
                new_name = f"{base}.ls2"
            elif "attn.qkv.weight" in name:
                new_name = f"{base}.self_attention.linear_qkv.weight"
                num_tensors = 3
                padded_dim = head_dim * num_padded_heads * num_tensors
                padded_tensor = torch.zeros((padded_dim, new_tensor.shape[-1]), dtype=new_tensor.dtype, device=new_tensor.device)
                padded_tensor[:new_tensor.shape[0], :] = new_tensor[order]
                new_tensor = padded_tensor
                chunk_dim = 0
            elif "attn.q_norm.weight" in name:
                new_name = f"{base}.self_attention.q_layernorm.weight"
                num_tensors = 1
                padded_dim = head_dim * num_padded_heads * num_tensors
                padded_tensor = torch.zeros(padded_dim, dtype=new_tensor.dtype, device=new_tensor.device)
                padded_tensor[:new_tensor.shape[0]] = new_tensor
                new_tensor = padded_tensor
                chunk_dim = 0
            elif "attn.k_norm.weight" in name:
                new_name = f"{base}.self_attention.k_layernorm.weight"
                num_tensors = 1
                padded_dim = head_dim * num_padded_heads * num_tensors
                padded_tensor = torch.zeros(padded_dim, dtype=new_tensor.dtype, device=new_tensor.device)
                padded_tensor[:new_tensor.shape[0]] = new_tensor
                new_tensor = padded_tensor
                chunk_dim = 0
            elif "attn.proj.weight" in name:
                new_name = f"{base}.self_attention.linear_proj.weight"
                num_tensors = 1
                padded_dim = head_dim * num_padded_heads * num_tensors
                padded_tensor = torch.zeros((new_tensor.shape[0], padded_dim), dtype=new_tensor.dtype, device=new_tensor.device)
                padded_tensor[:, :new_tensor.shape[-1]] = new_tensor
                new_tensor = padded_tensor
                chunk_dim = 1
            elif "attn.proj.bias" in name:
                new_name = f"{base}.self_attention.linear_proj.bias"
            elif "mlp.fc1.weight" in name:
                new_name = f"{base}.mlp.linear_fc1.weight"
                chunk_dim = 0
            elif "mlp.fc1.bias" in name:
                new_name = f"{base}.mlp.linear_fc1.bias"
                chunk_dim = 0
            elif "mlp.fc2.weight" in name:
                new_name = f"{base}.mlp.linear_fc2.weight"
                chunk_dim = 1
            elif "mlp.fc2.bias" in name:
                new_name = f"{base}.mlp.linear_fc2.bias"
            elif "norm1" in name:
                new_name = f"{base}.input_layernorm.weight"
            elif "norm2" in name:
                new_name = f"{base}.pre_mlp_layernorm.weight"
            else:
                raise RuntimeError("unexpected transformer layer name", name)
        else:
            raise RuntimeError("unexpected layer name", name)

        assert new_name != "", f"unexpected layer name {name}"

        # TE sets _extra_state (for FP8 purposes), so set an empty one here for compatibility.
        extra_state_layers = ("linear_qkv", "linear_proj", "linear_fc1", "linear_fc2")
        is_extra_state_layer = any([l in new_name for l in extra_state_layers])
        if use_te and is_extra_state_layer:
            layer = new_name.split(".")[-2]
            if layer in extra_state_layers:
                extra_state_name = (
                    new_name[: new_name.rfind(".") + 1] + "_extra_state"
                )  # Replace the weight name.
                for i in range(tensor_parallel_size):
                    new_state_dicts[i]["model"][extra_state_name] = None

        if chunk_dim is None:
            new_tensors = [new_tensor for _ in range(tensor_parallel_size)]
        else:
            new_tensors = torch.chunk(new_tensor, tensor_parallel_size, dim=chunk_dim)

        for i in range(tensor_parallel_size):
            new_state_dicts[i]["model"][new_name] = new_tensors[i].clone()

    for i in range(tensor_parallel_size):
        output_dir_tp = os.path.join(output_path, f"iter_0000001/mp_rank_0{i}")
        os.makedirs(output_dir_tp, exist_ok=True)
        output_path_tp = os.path.join(output_dir_tp, "model_optim_rng.pt")
        torch.save(new_state_dicts[i], output_path_tp)
        print("saved file", output_path_tp)

    print("done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="InternVIT HuggingFace to Mcore converter")
    parser.add_argument("--model-name", type=str, default="OpenGVLab/InternViT-6B-448px-V1-5", help="Model name in HuggingFace")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for the mcore model.")
    parser.add_argument("--use-te", action="store_true", default=True)
    parser.add_argument("--tensor-parallel-size", type=int, required=True)

    args = parser.parse_args()

    convert(args.model_name, args.output_dir, args.tensor_parallel_size, args.use_te)
