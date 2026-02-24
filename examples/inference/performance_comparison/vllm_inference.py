# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import time
import torch
import os

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


def print_cuda_memory_usage(stage: str = ""):
    """Print CUDA memory usage statistics."""
    if not torch.cuda.is_available():
        return
    
    i = 0
    allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)  # GB
    reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)    # GB
    max_allocated = torch.cuda.max_memory_allocated(i) / (1024 ** 3)  # GB
    max_reserved = torch.cuda.max_memory_reserved(i) / (1024 ** 3)    # GB
    
    print(f"[{stage}] GPU {i} Memory:")
    print(f"  Allocated: {allocated:.2f} GB")
    print(f"  Reserved:  {reserved:.2f} GB")
    print(f"  Max Allocated: {max_allocated:.2f} GB")
    print(f"  Max Reserved:  {max_reserved:.2f} GB")


def main():

    prompt_lengths = torch.randint(low=140, high=270, size=(512,))
    
    # Create a base prompt text (similar to Megatron's approach)
    prompt_lengths = [143, 143, 143, 143, 143, 143, 143, 143, 143, 143, 143, 143, 143, 143, 143, 143, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 151, 151, 151, 151, 151, 151, 151, 151, 151, 151, 151, 151, 151, 151, 151, 151, 115, 115, 115, 115, 115, 115, 115, 115, 115, 115, 115, 115, 115, 115, 115, 115, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 93, 93, 93, 93, 93, 93, 93, 93, 93, 93, 93, 93, 93, 93, 93, 93, 106, 106, 106, 106, 106, 106, 106, 106, 106, 106, 106, 106, 106, 106, 106, 106, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 121, 121, 121, 121, 121, 121, 121, 121, 121, 121, 121, 121, 121, 121, 121, 121, 83, 83, 83, 83, 83, 83, 83, 83, 83, 83, 83, 83, 83, 83, 83, 83, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 78, 78, 78, 78, 78, 78, 78, 78, 78, 78, 78, 78, 78, 78, 78, 78, 119, 119, 119, 119, 119, 119, 119, 119, 119, 119, 119, 119, 119, 119, 119, 119, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 122, 122, 122, 122, 122, 122, 122, 122, 122, 122, 122, 122, 122, 122, 122, 122, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 107, 107, 107, 107, 107, 107, 107, 107, 107, 107, 107, 107, 107, 107, 107, 107, 94, 94, 94, 94, 94, 94, 94, 94, 94, 94, 94, 94, 94, 94, 94, 94, 279, 279, 279, 279, 279, 279, 279, 279, 279, 279, 279, 279, 279, 279, 279, 279, 107, 107, 107, 107, 107, 107, 107, 107, 107, 107, 107, 107, 107, 107, 107, 107, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65]

    # Truncate to desired lengths
    vocab_size = 50000
    prompt_tokens_tensor = torch.randint(low=0, high=vocab_size, size=(512, 280))
    prompts = []
    for idx, length in enumerate(prompt_lengths):
        prompts.append({'prompt_token_ids':prompt_tokens_tensor[idx,:length].tolist()})


    llm = LLM(
        model='Qwen/Qwen2.5-1.5B',
        tokenizer='Qwen/Qwen2.5-1.5B',
        tensor_parallel_size=1,
        max_model_len=512, # This makes a little bit of difference
        gpu_memory_utilization=0.6,
        dtype='bfloat16',
        load_format='dummy',
        trust_remote_code=True,
        seed=1234,
        enable_prefix_caching=True,
        enforce_eager=False,
    )
    print("vLLM engine initialized successfully")
    print_cuda_memory_usage("After vLLM Engine Initialization")
    # Create sampling parameters
    sampling_params = SamplingParams(
        temperature=1.0,
        top_k=-1,
        top_p=1.0,  # vLLM requires top_p > 0
        max_tokens=512,
        logprobs=1,  # Return log probabilities
        prompt_logprobs=1,  # Return prompt log probabilities
    )

    # Start profiling if NSIGHT_PREFIX is set
    if os.environ.get("NSIGHT_PREFIX"):
        torch.cuda.cudart().cudaProfilerStart()

    # Reset peak memory stats before inference
    torch.cuda.reset_peak_memory_stats()
    print_cuda_memory_usage("Before Inference")

    start_time = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
    end_time = time.perf_counter()

    # Stop profiling if NSIGHT_PREFIX is set
    if os.environ.get("NSIGHT_PREFIX"):
        torch.cuda.cudart().cudaProfilerStop()

    latency = end_time - start_time
    print_cuda_memory_usage("After Inference")
    
    print("-" * 80)
    print(f"Total time: {latency:.2f} seconds")
    print(f"Throughput: {len(prompts) / latency:.2f} requests/sec")
    
 
    # Calculate token statistics
    total_prompt_tokens = sum(len(output.prompt_token_ids) for output in outputs)
    total_generated_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
    total_tokens = total_prompt_tokens + total_generated_tokens
    
    
    print(f"Total prompt tokens: {total_prompt_tokens}")
    print(f"Total generated tokens: {total_generated_tokens}")
    print(f"Total tokens: {total_tokens}")
    print(f"Token throughput: {total_tokens / latency:.2f} tokens/sec")
    print(f"Generation throughput: {total_generated_tokens / latency:.2f} tokens/sec")
    
    # Print peak memory usage
    if torch.cuda.is_available():
        print()
        i = 0
        peak_allocated = torch.cuda.max_memory_allocated(i) / (1024 ** 3)
        peak_reserved = torch.cuda.max_memory_reserved(i) / (1024 ** 3)
        print(f"GPU {i} Peak Memory - Allocated: {peak_allocated:.2f} GB, Reserved: {peak_reserved:.2f} GB")
    
    print("-" * 80)


if __name__ == "__main__":
    main()

