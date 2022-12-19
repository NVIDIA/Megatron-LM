import pytest
import os
import re
import subprocess


@pytest.fixture(params=[1])
def moe_num_experts(request):
    return str(request.param)


@pytest.fixture(params=[1])
def mp_size(request):
    return str(request.param)


@pytest.fixture
def params(moe_num_experts, mp_size):
    base_dir = os.getenv("MEGATRON_CKPT_DIR")
    assert base_dir, "Please set MEGATRON_CKPT_DIR in your environment"

    vocab_file = os.path.join(base_dir, "gpt2-vocab.json")
    merge_file = os.path.join(base_dir, "gpt2-merges.txt")
    ckpt_path = os.path.join(base_dir, "checkpoints/gpt2_345m")

    return [
        "--micro-batch-size", "1",
        "--num-layers", "24",
        "--hidden-size", "1024",
        "--num-attention-heads", "16",
        "--max-position-embeddings", "1024",
        "--vocab-file", vocab_file,
        "--merge-file", merge_file,
        "--load", ckpt_path,
        "--seq-length", "1024",
        "--out-seq-length", "1024",
        "--tensor-model-parallel-size", mp_size,
        "--tokenizer-type", "GPT2BPETokenizer",
        "--num-experts", moe_num_experts,
        "--mlp-type", "standard",
        "--num-samples", "0",
        "--fp16",
    ]


def test_moe_megatron(params, mp_size):
    output_re = r"===START OUTPUT===([\S\s]*)===END OUTPUT==="

    # Run the baseline
    baseline_cmd = ["deepspeed", "--num_gpus", mp_size, "./run_megatron.py"] + params
    result = subprocess.run(baseline_cmd, stdout=subprocess.PIPE)
    baseline_output = re.search(output_re, result.stdout.decode("utf-8")).group(1)

    # Run with DeepSpeed
    deepspeed_cmd = baseline_cmd + ["--ds-inference"]
    result = subprocess.run(deepspeed_cmd, stdout=subprocess.PIPE)
    deepspeed_output = re.search(output_re, result.stdout.decode("utf-8")).group(1)

    assert (
        baseline_output == deepspeed_output
    ), f"outputs do not match: {baseline_output}\n{deepspeed_output}"
