# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import os
import sys
from pathlib import Path

import datasets


def save_dataset_to_json(base_dir):
    target_path = Path(base_dir, "datasets/HF")
    target_path_cache = Path(base_dir, "datasets/HF")
    demo_dataset_dir = Path(base_dir, "datasets/dmc_demo")
    json_path = Path(demo_dataset_dir, "hf_wiki_20231101_en_train.jsonl")
    megatron_dataset_dir = Path(demo_dataset_dir, "llama2_tokenized")
    print(f'Creating dataset directories for demo at {target_path} and {demo_dataset_dir}')
    Path(target_path_cache).mkdir(parents=True, exist_ok=True)
    Path(demo_dataset_dir).mkdir(parents=True, exist_ok=True)
    Path(megatron_dataset_dir).mkdir(parents=True, exist_ok=True)

    datasets.config.DOWNLOADED_DATASETS_PATH = Path(target_path)
    datasets.config.HF_DATASETS_CACHE = Path(target_path_cache)
    ds = datasets.load_dataset("wikimedia/wikipedia", "20231101.en")
    ds["train"].to_json(json_path)


if __name__ == "__main__":
    demo_repo_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    save_dataset_to_json(demo_repo_dir)
