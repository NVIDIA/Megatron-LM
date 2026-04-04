# Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

# Backward-compatible re-export. The canonical location is now
# megatron.core.models.hybrid.hybrid_block.
from megatron.core.models.hybrid.hybrid_block import *  # noqa: F401,F403
from megatron.core.models.hybrid.hybrid_block import (  # noqa: F401
    HybridStack,
    HybridStackSubmodules,
    MambaStack,
    MambaStackSubmodules,
)
