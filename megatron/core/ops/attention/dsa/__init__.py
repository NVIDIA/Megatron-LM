# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""DeepSeek sparse-attention kernels and backend adapters.

Reference attention accepts SBHD or packed THD queries/keys; top-k indices are
BSJ and use -1 for invalid keys. Mask/layout helpers define packed and CP index
spaces. Indexer loss reduces teacher probabilities over the supplied TP group;
the model owns CP all-gathers, parameters, index sharing, logging and checkpoints.

Fused backends may return None for unsupported runtime shapes/layouts, preserving
the reference fallback. Backward, dtype and capture support are backend-specific.
Construction binds concrete hooks; this package adds no runtime registry.
Importing it alone does not load TileLang, cuDNN, FlashMLA or Hadamard kernels.
"""

from .kernel_metadata import KERNELS as KERNELS
