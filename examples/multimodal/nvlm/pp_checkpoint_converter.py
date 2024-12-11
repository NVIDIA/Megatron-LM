# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
import argparse
import os
import sys

import torch

# Add megatron to the path.
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, os.path.pardir))
)


def split(input_dir, base_output_dir, input_pp, output_pp, num_tp, num_layers_per_pp_rank):
    """Split pipeline parallel size = 1 checkpoint to pipeline parallel size N."""
    for tp in range(num_tp):
        path = os.path.join(input_dir, f"mp_rank_0{tp}", "model_optim_rng.pt")
        sd = torch.load(path)

        if num_layers_per_pp_rank is None:
            num_layers = sd["args"].num_layers
            assert num_layers % output_pp == 0, "specify --num-layers-per-pp-rank for an uneven split"
            num_layers_per_pp_rank = [num_layers // output_pp] * output_pp

        layer_lb = 0
        for pp in range(output_pp):
            assert num_layers_per_pp_rank[pp] > 0, "each pp rank must have at least 1 layer"
            layer_ub = layer_lb + num_layers_per_pp_rank[pp]

            new_sd = sd.copy()
            new_sd["model"] = dict()
            for k, v in sd["model"].items():
                # First pp rank has vision model.
                if pp == 0 and ("vision_model" in k or "vision_projection" in k):
                    new_sd["model"][k] = v
                    continue

                # Only the first pp rank has the word embeddings.
                if "language_model.embedding.word_embeddings" in k and pp == 0:
                    new_sd["model"][k] = v

                # Only the last pp rank has the output layer.
                if "language_model.output_layer" in k and pp == output_pp - 1:
                    new_sd["model"][k] = v

                # Only the last pp rank has final layer norm.
                if "language_model.decoder.final_layernorm" in k and pp == output_pp - 1:
                    new_sd["model"][k] = v

                if "language_model.decoder.layers" in k:
                    layer_num = int(k.split(".")[3])

                    if layer_lb <= layer_num and layer_num < layer_ub:
                        # On all pp ranks, megatron starts layer nums from 0!
                        new_layer_num = int(layer_num - layer_lb)

                        k_splitted = k.split(".")
                        k_splitted[3] = str(new_layer_num)
                        new_k = ".".join(k_splitted)

                        new_sd["model"][new_k] = v

            output_dir = os.path.join(base_output_dir, f"iter_0000001/mp_rank_0{tp}_00{pp}")
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, "model_optim_rng.pt")
            torch.save(new_sd, output_path)

            print(f"processed tp rank: {tp}/{num_tp - 1} and pp rank: {pp}/{output_pp - 1}")

            layer_lb = layer_ub

    # This is needed for megatron checkpoint loading.
    with open(os.path.join(base_output_dir, "latest_checkpointed_iteration.txt"), "w") as f:
        f.write("1")


def combine(input_dir, base_output_dir, input_pp, output_pp, num_tp, num_layers_per_pp_rank):
    """Combine pipeline parallel size = N checkpoint to pipeline parallel size 1."""
    for tp in range(num_tp):
        new_sd = None

        layer_num_offset = 0
        max_layer_num = 0

        for pp in range(input_pp):
            path = os.path.join(input_dir, f"mp_rank_0{tp}_00{pp}", "model_optim_rng.pt")
            sd = torch.load(path)

            if pp == 0:
                new_sd = sd.copy()
                new_sd["model"] = dict()
                new_sd["args"].pipeline_model_parallel_size = 1

            assert new_sd is not None

            for k, v in sd["model"].items():
                # First pp rank has vision model.
                if pp == 0 and ("vision_model" in k or "vision_projection" in k):
                    new_sd["model"][k] = v
                    continue

                # Only the first pp rank has the word embeddings.
                if "language_model.embedding.word_embeddings" in k and pp == 0:
                    new_sd["model"][k] = v

                # Only the last pp rank has the output layer.
                if "language_model.output_layer" in k and pp == input_pp - 1:
                    new_sd["model"][k] = v

                # Only the last pp rank has final layer norm.
                if "language_model.decoder.final_layernorm" in k and pp == input_pp - 1:
                    new_sd["model"][k] = v

                if "language_model.decoder.layers" in k:
                    layer_num = int(k.split(".")[3])

                    # On all pp ranks, megatron starts layer nums from 0!
                    new_layer_num = layer_num_offset + layer_num

                    if new_layer_num > max_layer_num:
                        max_layer_num = new_layer_num

                    k_splitted = k.split(".")
                    k_splitted[3] = str(new_layer_num)
                    new_k = ".".join(k_splitted)

                    new_sd["model"][new_k] = v

            print(f"processed tp rank: {tp}/{num_tp - 1} and pp rank: {pp}/{input_pp - 1}")

            layer_num_offset = max_layer_num + 1

        output_dir = os.path.join(base_output_dir, f"iter_0000001/mp_rank_0{tp}")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "model_optim_rng.pt")
        torch.save(new_sd, output_path)

    # This is needed for megatron checkpoint loading.
    with open(os.path.join(base_output_dir, "latest_checkpointed_iteration.txt"), "w") as f:
        f.write("1")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Change pipeline parallelism for a model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--input", type=str, required=True, help="Input model directory"
    )
    parser.add_argument(
        "--input-pipeline-parallel", type=int, required=True, help="Input model pipeline parallelism"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Output model directory"
    )
    parser.add_argument(
        "--output-pipeline-parallel", type=int, required=True, help="Output model pipeline parallelism"
    )
    parser.add_argument(
        "--tensor-parallel", type=int, required=True, help="Model tensor parallel size",
    )
    parser.add_argument(
        "--num-layers-per-pp-rank", type=int, default=None, nargs="*", help="Specify this for uneven pipeline parallel split",
    )

    args = parser.parse_args()

    f = None
    if args.input_pipeline_parallel == 1 and args.output_pipeline_parallel > 1:
        f = split
    elif args.input_pipeline_parallel > 1 and args.output_pipeline_parallel == 1:
        f = combine
    else:
        raise NotImplementedError("Only pipeline parallel 1 to N and N to 1 are supported")

    f(args.input, args.output, args.input_pipeline_parallel, args.output_pipeline_parallel, args.tensor_parallel, args.num_layers_per_pp_rank)

    print("done.")
