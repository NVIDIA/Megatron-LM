# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from enum import Enum

ModelType = Enum(
    'ModelType',
    ["gpt", "gptnext", "llama", "falcon", "starcoder", "mixtral", "gemma", "nemotron_nas"],
)
