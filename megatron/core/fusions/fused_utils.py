# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.


def propagate_paged_stash_marker(source, target):
    """Preserve Transformer Engine's grouped-tensor marker across view/cast operations."""
    # Lazy import avoids the transformer_engine extension -> MLP -> fusion import cycle.
    from megatron.core.extensions.transformer_engine import (
        is_grouped_tensor_marked,
        mark_grouped_tensor,
    )

    if is_grouped_tensor_marked(source):
        mark_grouped_tensor(target)
    return target
