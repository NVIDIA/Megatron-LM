# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""An example script to run the tensorrt_llm engine."""

import argparse
from pathlib import Path

import numpy as np
import torch
from ammo.deploy.llm import generate, load, unload
from transformers import AutoTokenizer, T5Tokenizer


class CustomSentencePieceTokenizer(T5Tokenizer):
    """This is a custom GPTSentencePiece Tokenizer modified from the T5Tokenizer.

    Note:
        The modification is kept minimal to make `encode` and `batch_decode` working
        properly (used in TensorRT-LLM engine). Other functions have not been tested.
    """

    def __init__(self, model):
        super().__init__(model, extra_ids=0, bos_token="<s>", pad_token="<pad>")

    def encode(self, text, add_special_tokens: bool = True, **kwargs):
        return self.sp_model.encode_as_ids(text)

    def batch_decode(self, sequences, skip_special_tokens: bool = False, **kwargs):
        if isinstance(sequences, np.ndarray) or torch.is_tensor(sequences):
            sequences = sequences.tolist()
        return self.sp_model.decode(sequences)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=str, default="")
    parser.add_argument("--max-output-len", type=int, default=100)
    parser.add_argument("--engine-dir", type=str, default="/tmp/ammo")
    parser.add_argument(
        "--input-texts",
        type=str,
        default=(
            "Born in north-east France, Soyer trained as a|Born in California, Soyer trained as a"
        ),
        help="Input texts. Please use | to separate different batches.",
    )
    parser.add_argument("--max-num-beams", type=int, default=1)
    parser.add_argument("--profiler-output", type=str, default="")
    return parser.parse_args()


def run(args):
    tokenizer_path = Path(args.tokenizer)

    if tokenizer_path.is_dir():
        # For llama models, use local HF tokenizer which is a folder.
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    elif tokenizer_path.is_file():
        # For nextllm and nemotron models, use local Megatron GPTSentencePiece tokenizer which is a model file.
        tokenizer = CustomSentencePieceTokenizer(args.tokenizer)
    else:
        raise ValueError(
            "arg.tokenizer must be a dir to a hf tokenizer checkpoint for llama or a SentencePiece .model file for gptnext"
        )

    if not hasattr(args, "profiler_output"):
        args.profiler_output = ""

    input_texts = args.input_texts.split("|")
    assert input_texts, "input_text not specified"
    print(input_texts)

    free_memory_before = torch.cuda.mem_get_info()

    host_context = load(
        tokenizer=tokenizer, engine_dir=args.engine_dir, num_beams=args.max_num_beams
    )
    torch.cuda.cudart().cudaProfilerStart()
    outputs = generate(input_texts, args.max_output_len, host_context, None, args.profiler_output)
    print(outputs)
    torch.cuda.cudart().cudaProfilerStop()

    free_memory_after = torch.cuda.mem_get_info()
    print(
        f"Use GPU memory: {(free_memory_before[0] - free_memory_after[0]) / 1024 / 1024 / 1024} GB"
    )

    unload(host_context)


if __name__ == "__main__":
    args = parse_arguments()
    run(args)
