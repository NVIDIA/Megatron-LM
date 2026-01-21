#!/usr/bin/env python3

import os
import sys

"""Convert HuggingFace checkpoint to Megatron format."""

from megatron.bridge import AutoBridge

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python 01_convert_hf.py <hf_model_name>")
        print("Example: python 01_convert_hf.py openai/gpt-oss-20b")
        sys.exit(1)
    
    HF_MODEL = sys.argv[1]
    
    # Auto-generate save path based on model name
    model_name = HF_MODEL.split("/")[-1].replace("-", "_")
    SAVE_PATH = f"./megatron_checkpoints/{model_name}"
    
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
    
    print(f"Converting {HF_MODEL} to Megatron format...")
    print(f"Save path: {SAVE_PATH}")
    
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
