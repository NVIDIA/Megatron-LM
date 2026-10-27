# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Convert HuggingFace checkpoints to Megatron format."""

import os
import argparse

from megatron.bridge import AutoBridge

def _parse_args():
    parser = argparse.ArgumentParser(description="Convert Megatron LLMs to HuggingFace format")
    parser.add_argument(
        "--hf-model",
        type=str,
        required=True,
        help="HuggingFace model identifier or path to load config from",
    )
    parser.add_argument(
        "--megatron-model",
        type=str,
        required=True,
        help="Megatron model identifier or path",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Path to save the converted HuggingFace checkpoint",
    )
    parser.add_argument('--local-rank', '--local_rank', type=int, default=0)
    return parser.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    HF_MODEL = args.hf_model
    MEGATRON_MODEL = args.megatron_model
    SAVE_PATH = args.save_path
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))

    if SAVE_PATH is None:
        SAVE_PATH = f"./huggingface_checkpoints/{MEGATRON_MODEL.replace('/', '_')}"
    
    print(f"Converting {MEGATRON_MODEL} to HuggingFace {HF_MODEL} format...")
    print(f"Save path: {SAVE_PATH}")
    
    bridge = AutoBridge.from_hf_pretrained(HF_MODEL, trust_remote_code=True)
    bridge.export_ckpt(
        MEGATRON_MODEL,
        SAVE_PATH,
    )
    
    print(f"Saved HuggingFace checkpoint to {SAVE_PATH}")
