# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""V4.1 layer composition and forward-local shared attention state."""

from dataclasses import replace

import torch
from torch import nn

from megatron.core.extensions.transformer_engine import TENorm
from megatron.core.models.deepseek_v41.moe import ModalityRouter, multimodal_moe_forward
from megatron.core.models.gpt.moe_module_specs import get_moe_module_spec
from megatron.core.transformer.experimental_attention_variant.csa2 import CSA2State
from megatron.core.transformer.experimental_attention_variant.csa2_module_spec import (
    csa2_attention_spec,
)
from megatron.core.transformer.experimental_attention_variant.dsv4_layer_config import (
    CSA2LayerConfig,
)
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe.moe_layer_config import MoELayerConfig
from megatron.core.transformer.single_pass_mhc import SinglePassHyperConnection, contract_streams
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.utils import sharded_state_dict_default
from megatron.core.utils import WrappedTensor, make_viewless_tensor


class ShardedLayerList(nn.ModuleList):
    """Preserve each layer's expert/table sharding through the ModuleList boundary."""

    def __init__(self, modules, tp_group) -> None:
        super().__init__(modules)
        self.tp_group = tp_group

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        """Dispatch to each child's checkpoint interface instead of flattening its state."""
        state = {}
        for i, layer in enumerate(self):
            state.update(
                sharded_state_dict_default(
                    layer, f"{prefix}{i}.", sharded_offsets, metadata, tp_group=self.tp_group
                )
            )
        return state


class DeepSeekV41Block(MegatronModule):
    """One logical block: single-pass mHC, CSA2, single-pass mHC, then DeepSeekMoE."""

    def __init__(self, config, layer_idx, pg_collection) -> None:
        super().__init__(config)
        self.tp_group = pg_collection.tp
        self.layer_number = layer_idx + 1
        self.attention = build_module(
            csa2_attention_spec,
            config=config,
            layer_number=self.layer_number,
            pg_collection=pg_collection,
            compress_ratio=config.csa_compress_ratios[layer_idx],
        )
        self.attention_norm = TENorm(config, config.hidden_size, eps=config.layernorm_epsilon)
        self.ffn_norm = TENorm(config, config.hidden_size, eps=config.layernorm_epsilon)
        self.attention_mhc = SinglePassHyperConnection(config, 2 * layer_idx + 1)
        self.ffn_mhc = SinglePassHyperConnection(config, 2 * layer_idx + 2)
        moe_spec = get_moe_module_spec(
            use_te=True,
            num_experts=config.num_moe_experts,
            moe_grouped_gemm=config.moe_grouped_gemm,
        )
        if getattr(config, "vision_config", None):
            moe_spec.keywords["submodules"] = replace(
                moe_spec.keywords["submodules"], router=ModalityRouter
            )
        self.mlp = moe_spec(
            config=config, layer_number=self.layer_number, pg_collection=pg_collection
        )
        self.engram = None

    def forward(
        self,
        hidden_states,
        previous_mix,
        state,
        attention_mask=None,
        padding_mask=None,
        image_mask=None,
    ):
        """Carry the last FFN mix and shared CSA2 tensors without storing either on the layer."""
        branch, next_mix, post, residual = self.attention_mhc(hidden_states, previous_mix)
        branch, bias = self.attention(self.attention_norm(branch), attention_mask, csa2_state=state)
        if bias is not None:
            branch = branch + bias
        hidden_states = self.attention_mhc.combine(branch, hidden_states, post, residual)
        branch, following_mix, post, residual = self.ffn_mhc(hidden_states, next_mix)
        branch = self.ffn_norm(branch)
        if image_mask is not None:
            branch, bias = multimodal_moe_forward(self.mlp, branch, image_mask, padding_mask)
        else:
            branch, bias = self.mlp(branch, padding_mask=padding_mask)
        if bias is not None:
            branch = branch + bias
        return (self.ffn_mhc.combine(branch, hidden_states, post, residual), following_mix)


class DeepSeekV41Stack(MegatronModule):
    """HybridModel stack with V4.1's single-pass residual and CSA2 sharing contracts.

    Keeping the shared state inside one forward also makes multiple outstanding
    microbatches safe. The decoder's Full layer projects global KV once from the
    encoder boundary; later decoder layers retain graph-connected references.
    """

    def __init__(
        self,
        config,
        *,
        pg_collection,
        layer_config_list,
        pre_process=True,
        post_process=True,
        pp_layer_offset=0,
        dtype=None,
        name=None,
    ) -> None:
        super().__init__(config)
        self.tp_group = pg_collection.tp
        if not pre_process or not post_process or pp_layer_offset:
            raise NotImplementedError("V4.1 stack currently requires one pipeline stage")
        if config.pipeline_model_parallel_size != 1 or config.virtual_pipeline_model_parallel_size:
            raise NotImplementedError("V4.1 stack currently requires PP=1 and no VPP")
        if config.recompute_granularity is not None or config.cuda_graph_impl != "none":
            raise NotImplementedError("V4.1 shared state requires eager stack execution")
        if (
            config.overlap_moe_expert_parallel_comm
            or config.delay_wgrad_compute
            or config.fine_grained_activation_offloading
        ):
            raise NotImplementedError(
                "V4.1 does not yet support layer overlap or activation offload"
            )
        if len(layer_config_list) != 2 * len(config.csa_compress_ratios):
            raise ValueError("V4.1 requires exactly one attention and one MoE branch per block")
        for index, layer_config in enumerate(layer_config_list):
            expected = CSA2LayerConfig if index % 2 == 0 else MoELayerConfig
            if type(layer_config) is not expected:
                raise ValueError("The V4.1 stack requires an alternating VE hybrid pattern")
        self.layers = ShardedLayerList(
            (
                DeepSeekV41Block(config, i, pg_collection)
                for i in range(len(config.csa_compress_ratios))
            ),
            pg_collection.tp,
        )
        self.final_norm = TENorm(config, config.hidden_size, eps=config.layernorm_epsilon)
        self.input_tensor = None
        self.pre_process = pre_process
        self.post_process = post_process
        self.num_layers_per_pipeline_rank = len(self.layers)

    def set_input_tensor(self, input_tensor):
        """Implement the HybridModel stack interface."""
        self.input_tensor = input_tensor

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        *,
        inference_context=None,
        rotary_pos_emb=None,
        packed_seq_params=None,
        padding_mask=None,
        packed_seq_params_by_layout=None,
        cp_layout_plan=None,
        engram_hashes=None,
        token_mask=None,
        image_mask=None,
    ):
        """Return normalized backbone hidden states."""
        if inference_context is not None or packed_seq_params is not None:
            raise NotImplementedError("V4.1 stack currently accepts full, unpacked sequences")
        if isinstance(hidden_states, WrappedTensor):
            hidden_states = hidden_states.unwrap()
        n = self.config.mhc_num_residual_streams
        hidden_states = (
            hidden_states.unsqueeze(-2).expand(*hidden_states.shape[:-1], n, -1).flatten(-2)
        )
        state, previous_mix = (CSA2State(), None)
        for _i, layer in enumerate(self.layers):
            if layer.engram is not None:
                hidden_states = layer.engram(hidden_states, engram_hashes, token_mask)
            hidden_states, previous_mix = layer(
                hidden_states, previous_mix, state, attention_mask, padding_mask, image_mask
            )
        hidden_states = self.final_norm(contract_streams(hidden_states, previous_mix, n))
        hidden_states = make_viewless_tensor(
            hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True
        )
        return hidden_states


deepseek_v41_stack_spec = ModuleSpec(module=DeepSeekV41Stack)
