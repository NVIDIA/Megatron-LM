# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CSA operation helpers.

Indexer quantization consumes packed BF16 [T,H,D] or [T,D] projections and
returns MXFP8 data with logical or per-sequence E8M0 scales. Callers own the
packed prefixes and temporary storage; these helpers perform no collectives.
"""
