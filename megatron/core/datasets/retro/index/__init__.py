# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.

"""
Exports:

  - train_index: Train an index on representative vectors.
  - add_to_index: Add vectors to a trained index.
  - build_index: Wrapper function that calls above two functions.
"""

from .build import add_to_index, build_index, train_index
