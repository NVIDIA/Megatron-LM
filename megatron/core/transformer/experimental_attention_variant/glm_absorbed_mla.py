# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""GLM-5.2 LoRA-aware absorbed MLA (Baseten additive subclass).

Isolates the one durable Baseten delta on the absorbed-MLA path — folding a LoRA adapter
into the kv up-projection's effective weight — into a subclass, so the upstream
``absorbed_mla.py`` carries no LoRA-specific logic and stays merge-clean against NVIDIA dev.

The absorbed path consumes ``linear_kv_up_proj`` as a raw weight (K folded into the query,
V applied after core attention) rather than calling its ``forward``, so a normal LoRA adapter
would never be applied without this override.
"""

import torch

from megatron.core.transformer.experimental_attention_variant.absorbed_mla import (
    AbsorbedMLASelfAttention,
)


class GlmAbsorbedMLASelfAttention(AbsorbedMLASelfAttention):
    """Absorbed MLA that folds a LoRA adapter into the kv up-projection effective weight."""

    def _kv_up_proj_weight(self) -> torch.Tensor:
        """Return ``linear_kv_up_proj``'s effective weight, folding in a LoRA adapter.

        When the module is LoRA-wrapped (``AdapterWrapper``: a ``to_wrap`` base plus an
        ``adapter``), return ``W_base + scale * (B @ A)`` so the adapter both trains
        (gradients flow to A/B through the absorption einsums) and serves consistently. The
        check is duck-typed to avoid a megatron.core -> megatron.bridge dependency.

        GLM-5.2 fused DSA forces TP=1, so the adapter factors are unsharded; TP>1 LoRA on the
        absorbed kv up-projection is not supported.
        """
        module = self.linear_kv_up_proj
        if not hasattr(module, "to_wrap"):
            return module.weight
        weight = module.to_wrap.weight
        if not getattr(module, "_adapter_enabled", True):
            return weight
        if self.config.tensor_model_parallel_size != 1:
            raise NotImplementedError(
                "LoRA on the absorbed kv up-projection is only supported with TP=1."
            )
        adapter = module.adapter
        lora_a = adapter.linear_in.weight  # [lora_dim, kv_lora_rank]
        lora_b = adapter.linear_out.weight  # [num_heads * (qk_head_dim + v_head_dim), lora_dim]
        scale = getattr(adapter, "scale", None)
        if scale is None:
            scale = adapter.alpha / adapter.dim
        return weight + scale * (lora_b @ lora_a)
