# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Deployment MoE/mHC/head on the native Lite model lifecycle.

Ratio-zero attention uses native BI projections and sparse FlashMLA. Compressed
attention still uses Core and is explicitly provisional; receipts distinguish
the two providers until each target path has its own parity evidence.
"""
from megatron.core.models.gpt.gpt_layer_specs import get_mlp_module_spec
from megatron.core.tensor_parallel.layers import ColumnParallelLinear
from megatron.lite.model.deepseek_v41.lite.components.block import DS41Block
from megatron.lite.model.deepseek_v41.lite.model import DeepseekV41Model as NativeModel

from .attention import build_attention
from .head import head_logits
from .mhc import joint, post_mix
from .moe import MegaMoE
from .norm import ModelNorm


class Head(ColumnParallelLinear):
    def forward(self, x, weight=None, **kwargs):
        # Core cross entropy owns and mutates its logits buffer in place.
        # A custom Function's view output must not be that mutable buffer.
        return head_logits(x, self.weight if weight is None else weight).clone(), None


class Block(DS41Block):
    def __init__(self, config, layer_number, attention, ffn):
        super().__init__(config, layer_number, attention, ffn)
        self.first_layer = layer_number == 1

    def _sublayer(self, hidden, pre_mix, kind, packed_seq_params, input_ids):
        mixes = self.attn_mixes if kind == 'attn' else self.ffn_mixes
        norm = self.attn_norm if kind == 'attn' else self.ffn_norm
        _, next_pre, post, comb, x = joint(
            hidden, pre_mix, norm.weight, mixes, broadcast=self.first_layer and kind == 'attn'
        )
        x, bias = (
            self.attention(x, attention_mask=None, packed_seq_params=packed_seq_params)
            if kind == 'attn'
            else self.ffn(x, input_ids=input_ids, packed_seq_params=packed_seq_params)
        )
        if bias is not None:
            raise ValueError('Deployment DS4.1 blocks require bias-free sublayers')
        return post_mix(x, hidden, post, comb), next_pre


class DeepseekV41Model(NativeModel):
    deployment_providers = {
        'attention': 'ratio0 native BI + sparse FlashMLA; ratio>0 Core provisional; NOT_TARGET_PARITY',
        'engram': 'Lite frozen-host native FP8 owner; Core projections provisional',
        'moe': 'R1 nonfused MegaMoE + Core native groups/owners/collectives',
        'mhc': 'native BI caller-stream pre + materialized post; parity unverified',
        'head': 'native BI BF16 GEMM + Core cross entropy; parity unverified',
    }

    def _build_attention(self, config, index):
        return build_attention(config, index, self.context)

    def _build_mlp(self, config, release, index):
        spec = get_mlp_module_spec(
            use_te=True, num_experts=release.n_routed_experts, moe_grouped_gemm=True
        )
        return MegaMoE(
            config=config,
            submodules=spec.keywords['submodules'],
            layer_number=index + 1,
            pg_collection=self.context.groups,
            hash_moe_layer_threshold=release.num_hash_layers,
        )

    def _build_block(self, config, index, attention, mlp):
        return Block(config, index + 1, attention, mlp)

    def _build_head(self, config, release):
        norm = ModelNorm(config, release.hidden_size, release.rms_norm_eps)
        head = Head(
            release.hidden_size,
            release.vocab_size,
            config=config,
            init_method=config.init_method,
            bias=False,
            gather_output=False,
            tp_group=self.ps.tp_group,
        )
        head.weight.is_embedding_or_output_parameter = True
        return norm, head
