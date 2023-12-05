# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

"""Computes theoretical memory footprint for model training without instantiating
a model and running training iterations on GPU(s)."""

from megatron import get_args
from megatron.initialize import initialize_megatron
from megatron.theoretical_memory_usage import report_theoretical_memory

if __name__ == "__main__":
    initialize_megatron(allow_no_cuda=True, skip_mpu_initialization=True)
    args = get_args()

    report_theoretical_memory(args, verbose=True)
