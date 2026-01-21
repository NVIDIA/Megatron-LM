#!/usr/bin/env python3
"""Convert HuggingFace GPT-OSS checkpoint to Megatron format."""

import os
import argparse

from megatron.bridge import AutoBridge

def _parse_args():
    parser = argparse.ArgumentParser(description="Convert HF LLMs to Megatron format")
    parser.add_argument(
        "--hf-model",
        type=str,
        required=True,
        help="HuggingFace model identifier or path",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Path to save the converted Megatron checkpoint",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    HF_MODEL = args.hf_model
    SAVE_PATH = args.save_path
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))

    if SAVE_PATH is None:
        SAVE_PATH = f"./megatron_checkpoints/{HF_MODEL.replace('/', '_')}"
    
    print(f"Converting {HF_MODEL} to Megatron format...")
    
    bridge = AutoBridge.from_hf_pretrained(HF_MODEL, trust_remote_code=True)
    provider = bridge.to_megatron_provider()
    provider.expert_tensor_parallel_size = 1
    provider.tensor_model_parallel_size = 1
    provider.pipeline_model_parallel_size = WORLD_SIZE
    provider.finalize()
    
    model = provider.provide_distributed_model(wrap_with_ddp=False)
    
    bridge.save_megatron_model(
        model,
        SAVE_PATH,
        hf_tokenizer_path=HF_MODEL
    )
    
    print(f"Saved Megatron checkpoint to {SAVE_PATH}")
