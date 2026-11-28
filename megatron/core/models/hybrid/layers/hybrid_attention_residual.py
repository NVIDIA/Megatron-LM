# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Attention-residual (AttnRes) wrapper for HybridStack entries.

Lives under ``layers/`` like the mHC wrapper (``hybrid_hyper_connection.py``): the
wrapper is applied by ``HybridStack`` at construction time, next to the mHC wrap,
so the stack itself stays generic. ``hybrid_block.py`` re-exports the class so
callers importing it from there keep working.
"""

import warnings
from typing import Optional, Tuple

import torch
from torch import Tensor

from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.attention_residual import (
    AttentionResidual,
    attn_res_final_num_sources,
    attn_res_num_sources,
    is_attn_res_block_start,
)
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_layer import TransformerLayer
from megatron.core.utils import nvtx_range_pop, nvtx_range_push


class AttnResHybridLayer(MegatronModule):
    """Attention-residual wrapper for hybrid stack entries.

    Wraps ANY hybrid entry (Mamba/GDN/KDA/attention/MLP/MoE — each entry is one
    depth sublayer) with the AttnRes depth aggregation::

        partial = None if block-start else hidden_states
        h = AttentionResidual(sources [+ partial])
        delta = inner_layer(h, ..., _return_layer_delta=True)
        new_partial = delta (+ partial)

    TransformerLayer entries (including KDA, MLA, dense MLP and MoE) return the
    bias/dropout contribution directly through their normal module call. This
    avoids cancellation in BF16 ``(input + branch) - input``. Non-transformer
    entries retain their legacy residual reconstruction. Checkpoint keys remain
    nested under ``inner_layer.``.
    """

    def __init__(
        self, config: TransformerConfig, layer: MegatronModule, is_mtp_layer: bool = False
    ):
        super().__init__(config)
        self.inner_layer = layer
        self.layer_number = layer.layer_number
        self.attn_res = AttentionResidual(
            config,
            self.layer_number,
            recompute=(
                config.recompute_granularity == 'selective'
                and 'attn_res' in (config.recompute_modules or [])
            ),
        )
        # Norm fusion: the aggregation emits the entry's pre-norm output directly, using the
        # weight of the inner entry's norm module, which keeps owning it. Only the attribute
        # name is kept here: storing the module itself would register it as a second child
        # and duplicate its weight in the state dict (checkpoint layout must not change).
        self.attn_res_norm_fusion = config.attn_res_norm_fusion
        self._fused_norm_name = (
            self._resolve_fused_norm_name(layer) if self.attn_res_norm_fusion else None
        )
        if self.attn_res_norm_fusion and self._fused_norm_name is None:
            warnings.warn(
                f"attn_res_norm_fusion: hybrid entry {self.layer_number} "
                f"({type(layer).__name__}) has no standalone pre-norm (folded into its first "
                "linear); its AttnRes aggregation stays unfused."
            )
        # Hybrid entries are single sublayers: attn_res_block_layers counts
        # pattern entries here (see attention_residual.py docstrings).
        if is_mtp_layer:
            # Every entry in an MTP depth attends over the same complete trunk
            # history. The depth's eh_proj output is its fresh partial sum; MTP
            # entries never turn that partial into a persistent depth source.
            self.attn_res_is_block_start = False
            self.attn_res_num_sources = attn_res_final_num_sources(
                config.num_layers, config.attn_res_block_layers
            )
        else:
            self.attn_res_is_block_start = is_attn_res_block_start(
                self.layer_number, config.attn_res_block_layers
            )
            self.attn_res_num_sources = attn_res_num_sources(
                self.layer_number, config.attn_res_block_layers
            )

    @property
    def fused_norm(self) -> Optional[MegatronModule]:
        """The inner entry's pre-norm fused into the aggregation, or None without norm fusion."""
        if self._fused_norm_name is None:
            return None
        return getattr(self.inner_layer, self._fused_norm_name)

    @staticmethod
    def _resolve_fused_norm_name(layer: MegatronModule) -> Optional[str]:
        """Name of the inner entry's standalone pre-norm, or None when the entry has none.

        Entries whose norm is folded into their first linear (TE LayerNormLinear specs, e.g.
        the dense MLP entry of Kimi-K3, or Mamba mixers) keep the unfused aggregation: there
        is no separate norm whose weight the fused kernel could apply.
        """
        if not isinstance(layer, TransformerLayer):
            return None
        has_attention = not isinstance(layer.self_attention, IdentityOp)
        has_mlp = not isinstance(layer.mlp, IdentityOp)
        if has_attention == has_mlp:
            raise ValueError(
                "Hybrid AttnRes requires exactly one attention or MLP branch per entry."
            )
        name = "input_layernorm" if has_attention else "pre_mlp_layernorm"
        norm = getattr(layer, name)
        if isinstance(norm, IdentityOp) or not isinstance(
            getattr(norm, "weight", None), torch.Tensor
        ):
            return None
        return name

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        inference_context: Optional[BaseInferenceContext] = None,
        rotary_pos_emb: Optional[Tensor] = None,
        sequence_len_offset: Optional[Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        padding_mask: Optional[Tensor] = None,
        input_ids: Optional[Tensor] = None,
        *,
        attn_res_sources: Optional[Tuple[Tensor, ...]] = None,
        packed_sequence_cp_metadata=None,
    ) -> Tensor:
        """Aggregate depth sources, run the inner entry, and update the partial sum."""
        if attn_res_sources is None:
            raise RuntimeError(
                "AttnResHybridLayer requires the attn_res_sources keyword; it is "
                "provided by HybridStack.forward. Execution paths that do not thread "
                "it (e.g. full recompute or inference) cannot run attention residuals."
            )
        if inference_context is not None:
            raise NotImplementedError("Attention residuals do not support inference yet.")
        assert len(attn_res_sources) == self.attn_res_num_sources, (
            f"hybrid entry {self.layer_number}: expected {self.attn_res_num_sources} depth "
            f"sources, got {len(attn_res_sources)}; block-boundary bookkeeping is broken"
        )

        # At block-start entries the incoming hidden state was just appended to
        # the depth sources by the caller, so there is no partial sum yet.
        partial = None if self.attn_res_is_block_start else hidden_states
        values = list(attn_res_sources) if partial is None else [*attn_res_sources, partial]
        fused_norm = self.fused_norm
        aggregated = self.attn_res(
            values, output_norm_weight=fused_norm.weight if fused_norm is not None else None
        )

        if isinstance(self.inner_layer, TransformerLayer):
            inner_kwargs = dict(
                hidden_states=aggregated,
                attention_mask=attention_mask,
                inference_context=None,
                rotary_pos_emb=rotary_pos_emb,
                sequence_len_offset=sequence_len_offset,
                packed_seq_params=packed_seq_params,
                padding_mask=padding_mask,
                input_ids=input_ids,
                _called_from_hybrid_attn_res_wrapper=True,
                _return_layer_delta=True,
                _input_is_normalized=fused_norm is not None,
            )
            if packed_sequence_cp_metadata is not None:
                raise TypeError(
                    "AttnResHybridLayer: transformer entries do not take "
                    "packed_sequence_cp_metadata (chunkwise-CP Mamba metadata)."
                )
            output = self.inner_layer(**inner_kwargs)
        else:
            # Non-transformer entries (e.g. MambaLayer) accept only the common
            # arguments — same contract as HyperConnectionHybridLayer's inner call.
            inner_kwargs = dict(
                hidden_states=aggregated,
                attention_mask=attention_mask,
                inference_context=None,
                packed_seq_params=packed_seq_params,
            )
            if packed_sequence_cp_metadata is not None:
                inner_kwargs["packed_sequence_cp_metadata"] = packed_sequence_cp_metadata
            output = self.inner_layer(**inner_kwargs)
        if isinstance(output, tuple):
            output = output[0]

        nvtx_range_push(msg="attn_res.hybrid_delta")
        # Only legacy non-transformer entries add a local residual. Split
        # TransformerLayer entries already return the bias/dropout contribution.
        delta = output if isinstance(self.inner_layer, TransformerLayer) else output - aggregated
        new_partial = delta if partial is None else partial + delta
        nvtx_range_pop(msg="attn_res.hybrid_delta")
        # The forward is done with the aggregation output; under 'attn_res' recompute release
        # it now and rebuild it when the gradient of this entry's output arrives, which is
        # before the inner entry's norm checkpoint hook (on the branch output) and backward.
        self.attn_res.discard_output_and_register_recompute(new_partial)
        return new_partial
