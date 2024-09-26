# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""An example script to run the tensorrt_llm engine."""

import argparse
from pathlib import Path
import subprocess
from typing import Optional, Union

import numpy as np
import torch
from modelopt.deploy.llm import LLM
from tensorrt_llm.models import PretrainedConfig
from transformers import AutoTokenizer, T5Tokenizer
import tensorrt_llm


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=str, default="")
    parser.add_argument("--engine-dir", type=str, default="/tmp/trtllm_engine")
    parser.add_argument(
        "--input-texts",
        type=str,
        default=(
            "Born in north-east France, Soyer trained as a|Born in California, Soyer trained as a"
        ),
        help="Input texts. Please use | to separate different batches.",
    )
    return parser.parse_args()


def run(args):
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    except Exception as e:
        raise Exception(f"Failed to load tokenizer: {e}")

    print(tokenizer, tokenizer.vocab_size)

    input_texts = args.input_texts.split("|")
    assert input_texts, "input_text not specified"
    print(input_texts)

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
