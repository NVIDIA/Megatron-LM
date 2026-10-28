# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from megatron.core.ssm.gdn_layer_config import GDNLayerConfig


class KDALayerConfig(GDNLayerConfig):
    """Configuration view for a Kimi Delta Attention layer in a hybrid stack."""
