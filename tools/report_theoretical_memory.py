# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

"""Computes theoretical memory footprint for model training without instantiating
a model and running training iterations on GPU(s)."""

from megatron.training import get_args
from megatron.training.initialize import initialize_megatron
from megatron.training.theoretical_memory_usage import report_theoretical_memory

if __name__ == "__main__":
    initialize_megatron(allow_no_cuda=True, skip_mpu_initialization=True)
    args = get_args()

    report_theoretical_memory(args, verbose=True)
