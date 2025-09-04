# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.

"""Base class for decoder and encoder attention modules."""

from megatron.core.models.retro.config import RetroConfig
from megatron.core.process_groups_config import ModelCommProcessGroups
from megatron.core.transformer.attention import CrossAttention, CrossAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.module import MegatronModule


class BaseRetroCrossAttention(MegatronModule):
    """Base class for Retro cross attention, for both encoder & decoder layers.

    This class collects the retro arguments below (i.e., num neighbors, chunk
    length, and retrieve length) for use in Retro's custom cross attention
    operators.

    Args:
        config (RetroConfig): Retro config.
        submodules (CrossAttentionSubmodules): Cross attention submodules.
        layer_number (int): Layer number within transformer block.
        attn_mask_type (AttnMaskType): Mask type ('causal' or 'padding').
        model_comm_pgs (ModelCommProcessGroups): Model communication process groups.
    """

    def __init__(
        self,
        config: RetroConfig,
        submodules: CrossAttentionSubmodules,
        layer_number: int = 1,
        attn_mask_type: AttnMaskType = AttnMaskType.padding,
        model_comm_pgs: ModelCommProcessGroups = None,
    ):
        super().__init__(config=config)

        self.attn = CrossAttention(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            model_comm_pgs=model_comm_pgs,
        )

        self.retro_num_neighbors = config.retro_num_neighbors
        self.retro_chunk_length = config.retro_chunk_length
        self.retro_retrieved_length = config.retro_retrieved_length
