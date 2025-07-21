# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.

"""Required external libraries for Retro preprocessing."""

import importlib

required_libs = ["faiss", "h5py", "transformers"]  # for huggingface bert

for lib in required_libs:
    try:
        globals()[lib] = importlib.import_module(lib)
    except ImportError as e:
        pass
