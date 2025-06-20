# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.

"""
Exports:
- FaissBaseIndex: Unoptimized Faiss index wrapper
- FaissParallelAddIndex: Optimized index.add() for Faiss index.
"""

from .faiss_base import FaissBaseIndex
from .faiss_par_add import FaissParallelAddIndex
