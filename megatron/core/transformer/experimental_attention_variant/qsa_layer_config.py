# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from megatron.core.transformer.transformer_config import TransformerConfig


class QSALayerConfig(TransformerConfig):
    """Configuration for a Qwen Sparse Attention layer in a hybrid stack.

    QSA is GQA-based (not MLA-family), so this derives from TransformerConfig.
    Due to backwards-compatibility, this config's arguments are defined in TransformerConfig.
    """
