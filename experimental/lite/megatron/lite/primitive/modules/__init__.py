"""Shared model modules owned by Megatron Lite."""

from megatron.lite.primitive.modules.dispatcher import TokenDispatcher
from megatron.lite.primitive.modules.experts import Experts
from megatron.lite.primitive.modules.gated_delta_net import GatedDeltaNet
from megatron.lite.primitive.modules.gqa import GQAttention, split_grouped_qkvg
from megatron.lite.primitive.modules.moe import MoEAuxLossAutoScaler, _AllToAll
from megatron.lite.primitive.modules.mrope import MultimodalRotaryEmbedding
from megatron.lite.primitive.modules.mtp import MTPBlock, MTPDecoderLayer, MTPLossAutoScaler
from megatron.lite.primitive.modules.router import SigmoidTopKRouter, TopKRouter

__all__ = [
    "Experts",
    "GatedDeltaNet",
    "GQAttention",
    "MTPBlock",
    "MTPDecoderLayer",
    "MTPLossAutoScaler",
    "MoEAuxLossAutoScaler",
    "MultimodalRotaryEmbedding",
    "SigmoidTopKRouter",
    "split_grouped_qkvg",
    "TokenDispatcher",
    "TopKRouter",
    "_AllToAll",
]
