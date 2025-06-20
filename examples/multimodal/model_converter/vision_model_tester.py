# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
import argparse
import os
import sys

# Add megatron and the multimodal example to the path.
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, os.path.pardir)
    )
)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import torch
from transformers import AutoModel

from examples.multimodal.model import model_provider
from examples.multimodal.multimodal_args import add_multimodal_extra_args
from megatron.training import get_model
from megatron.training.checkpointing import load_checkpoint
from megatron.training.initialize import initialize_megatron


def run_mcore_vision(model_path):
    """Run mcore vision model."""
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"

    # Megatron has some mandatory flags.
    sys.argv = [
        "ignore_me.py",
        "--micro-batch-size=1",
        "--num-layers=2",
        "--vision-model-type=internvit",
        "--language-model-type=mistral_7b",
        "--tokenizer-prompt-format=mistral",
        "--tokenizer-type=MultimodalTokenizer",
        "--tokenizer-model=mistralai/Mistral-7B-Instruct-v0.3",
        "--vocab-size=1024",
        "--hidden-size=64",
        "--num-attention-heads=8",
        "--seq-length=1024",
        "--decoder-seq-length=2048",
        "--max-position-embeddings=2048",
        "--bf16",
        "--img-h=448",
        "--img-w=448",
        "--patch-dim=14",
        "--tensor-model-parallel-size=8",
        "--use-te",
        f"--pretrained-checkpoint={model_path}",
    ]

    initialize_megatron(extra_args_provider=add_multimodal_extra_args)

    def wrapped_model_provider(pre_process, post_process):
        return model_provider(pre_process, post_process, parallel_output=False)

    # Set up model and load checkpoint.
    model = get_model(wrapped_model_provider, wrap_with_ddp=False)

    vision_model = model[0].module.vision_model

    load_checkpoint([vision_model], None, None)

    vision_model.eval()

    images = torch.ones((1, 3, 448, 448), dtype=torch.bfloat16, device="cuda")

    output = vision_model(images)

    return output


def run_hf_vision(model_name):
    """Run HF vision model."""
    model = (
        AutoModel.from_pretrained(model_name, torch_dtype=torch.bfloat16, trust_remote_code=True)
        .cuda()
        .eval()
    )

    images = torch.ones((1, 3, 448, 448), dtype=torch.bfloat16, device="cuda")

    outputs = model(images, return_dict=True)

    return outputs


def main(mcore_model, hf_model):
    """Compare vision model outputs between mcore and HF given the same fixed input."""
    mcore = run_mcore_vision(mcore_model)

    if torch.distributed.get_rank() == 0:
        hf = run_hf_vision(hf_model)
        hf = hf["last_hidden_state"]

        # Compare logits. Due to different attention implementations and other details,
        # there will be numerical differences.
        diff = (mcore - hf).abs()
        mean_diff = diff.mean().item()
        max_diff = diff.max().item()
        print(f"mean diff {mean_diff}, max diff {max_diff}")
        assert mean_diff < 0.1, "mean output difference is greater than expected"
        assert max_diff < 50, "max output difference is greater than expected"

        print("lgtm")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Check mcore vision model output vs. HF numerically.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mcore-model", type=str, required=True, help="directory for mcore model weights"
    )
    parser.add_argument("--hf-model", type=str, required=True, help="Model name in HF")

    args = parser.parse_args()

    main(args.mcore_model, args.hf_model)
