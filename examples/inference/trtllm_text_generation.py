# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""An example script to run the tensorrt_llm engine."""

import argparse
from pathlib import Path

import numpy as np
import torch
from modelopt.deploy.llm import LLM, build_tensorrt_llm
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
        return torch.Tensor(self.sp_model.encode_as_ids(text))

    def batch_encode_plus(
        self, batch_text_or_text_pairs, add_special_tokens: bool = True, **kwargs
    ):
        return {'input_ids': self.sp_model.encode_as_ids(batch_text_or_text_pairs)}

    def batch_decode(self, sequences, skip_special_tokens: bool = False, **kwargs):
        if isinstance(sequences, np.ndarray) or torch.is_tensor(sequences):
            sequences = sequences.tolist()
        return self.sp_model.decode(sequences)

    def decode(self, token_ids, skip_special_tokens: bool = False, **kwargs):
        return self.sp_model.decode([token_ids])[0]


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=str, default="")
    parser.add_argument("--max-input-len", type=int, default=4096)
    parser.add_argument("--max-output-len", type=int, default=512)
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument("--tensorrt-llm-checkpoint-dir", type=str, default=None)
    parser.add_argument("--engine-dir", type=str, default="/tmp/trtllm_engine")
    parser.add_argument(
        "--input-texts",
        type=str,
        default=(
            "Born in north-east France, Soyer trained as a|Born in California, Soyer trained as a"
        ),
        help="Input texts. Please use | to separate different batches.",
    )
    parser.add_argument("--max-beam-width", type=int, default=1)
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
    print(tokenizer, tokenizer.vocab_size)

    if not hasattr(args, "profiler_output"):
        args.profiler_output = ""

    input_texts = args.input_texts.split("|")
    assert input_texts, "input_text not specified"
    print(input_texts)

    if args.tensorrt_llm_checkpoint_dir is not None:
        print("Building TensorRT-LLM engines.")
        build_tensorrt_llm(
            args.tensorrt_llm_checkpoint_dir + "/config.json",
            args.engine_dir,
            max_input_len=args.max_input_len,
            max_batch_size=args.max_batch_size,
            max_beam_width=args.max_beam_width,
            num_build_workers=1,
        )
        print(f"TensorRT-LLM engines saved to {args.engine_dir}")

    free_memory_before = torch.cuda.mem_get_info()

    # This is a ModelOpt wrapper on top of tensorrt_llm.hlapi.llm.LLM
    llm_engine = LLM(args.engine_dir, tokenizer)

    torch.cuda.cudart().cudaProfilerStart()
    # outputs = llm_engine.generate_text(input_texts, args.max_output_len, args.max_beam_width)
    outputs = llm_engine.generate(input_texts)
    torch.cuda.cudart().cudaProfilerStop()

    free_memory_after = torch.cuda.mem_get_info()
    print(
        f"Used GPU memory: {(free_memory_before[0] - free_memory_after[0]) / 1024 / 1024 / 1024} GB"
    )
    print(outputs)


if __name__ == "__main__":
    args = parse_arguments()
    run(args)
