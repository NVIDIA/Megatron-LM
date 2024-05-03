# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import psutil


def print_memory_usage(key, rank, num_ranks):
    '''Print memory usage.'''
    process = psutil.Process()
    mem_info = process.memory_info()
    print("> memory usage: '%s', rank %d / %d, mem %.1f/%.1f gb." % (
        key,
        rank,
        num_ranks,
        mem_info.rss / 1024**3,
        100 * mem_info.rss / process.memory_percent() / 1024**3,
    ))


def get_mcore_transformer_block_key(model_key):
    return {
        "GPT" : "decoder",
        "BERT" : "encoder",
        "EVA": "encoder"
    }[model_key]
