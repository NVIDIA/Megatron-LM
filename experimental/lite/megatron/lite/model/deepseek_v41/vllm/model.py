# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Native Lite lifecycle with deployment MoE, mHC, attention and head."""
import torch

from megatron.core.models.gpt.gpt_layer_specs import get_mlp_module_spec
from megatron.core.tensor_parallel.layers import ColumnParallelLinear
from megatron.core.utils import ensure_params_ready
from megatron.lite.model.deepseek_v41.lite.components.block import DS41Block
from megatron.lite.model.deepseek_v41.lite.model import DeepseekV41Model as NativeModel

from .attention import build_attention
from .head import head_logits, log_softmax
from .mhc import aggregate, joint, post_mix
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
        'attention': 'native BI SWA + ratio2 compressor/indexer + sparse FlashMLA; ratio2 parity pending',
        'engram': 'Lite frozen-host owner; native BI MXFP8 WKV and Triton post-WKV; parity pending',
        'moe': 'R1 nonfused MegaMoE + Core native groups/owners/collectives',
        'mhc': 'native BI pre/post/collapse with owned VJPs',
        'head': 'native BI BF16 GEMM and native log-softmax; parity pending',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from .engram import forward

        for module in self.engrams.values():
            module.forward = forward.__get__(module, type(module))
        for block in self.layers.values():
            mixes = block.ffn_mixes
            mixes.aggregate = aggregate.__get__(mixes, type(mixes))

    def forward(self, *args, labels=None, loss_mask=None, calculate_entropy=False, **kwargs):
        # Keep the module pipeline, but publish buckets also for functional
        # mHC/expert reads which bypass their owning leaf module's pre-hook.
        ensure_params_ready(self.parameters())
        if labels is None or not self.layout.has_head:
            return super().forward(
                *args,
                labels=labels,
                loss_mask=loss_mask,
                calculate_entropy=calculate_entropy,
                **kwargs,
            )
        if self.ps.tp_size != 1:
            raise NotImplementedError('Deployment log-probs currently require TP1')
        output = super().forward(*args, labels=None, loss_mask=loss_mask, **kwargs)
        distribution = log_softmax(output.pop('logits'))
        picked = distribution.gather(-1, labels.reshape(-1, 1)).squeeze(-1)
        mask = torch.ones_like(picked) if loss_mask is None else loss_mask.reshape(-1).to(picked)
        output.update(
            log_probs=picked,
            loss=-(picked * mask).sum() / mask.sum().clamp_min(1),
            valid_mask=mask.bool(),
        )
        if calculate_entropy:
            output['entropy'] = -(distribution.exp() * distribution).sum(-1)
        return output

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
