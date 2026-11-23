# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Associate direct parameter reads with forward boundaries that actually execute."""

from collections.abc import Iterable

from torch import nn


def collect_param_readiness_dependencies(
    root: nn.Module,
) -> dict[nn.Module, tuple[nn.Parameter, ...]]:
    """Return extra parameters to publish at each called module's entry.

    Fused kernels and split forwards can bypass a parameter owner's __call__. Keep
    their dependencies here, without moving parameters or adding waits to computation.
    Choose the nearest preceding module that runs on every affected execution path;
    recursively waiting on a whole layer would unnecessarily reduce gather overlap.
    """
    # Import only when installing hooks, after model modules have been initialized.
    from ..models.audio.nemo_rope_transformer_encoder import TransformerBlock as AudioBlock
    from ..models.vision.vit_model import KimiViTModel, QwenVLViTModel, ViTModel
    from ..ssm.gated_delta_net.common import _GDNBase
    from ..ssm.gated_delta_product import GatedDeltaProductMixer
    from ..ssm.mamba_mixer import MambaMixer
    from ..transformer.experimental_attention_variant.absorbed_mla import AbsorbedMLASelfAttention
    from ..transformer.hyper_connection import HyperConnectionModule
    from ..transformer.multi_latent_attention import FusedMLASelfAttention, MLASelfAttention

    dependencies: dict[nn.Module, list[nn.Parameter]] = {}

    def add(boundary: nn.Module, params: Iterable[nn.Parameter]) -> None:
        dependencies.setdefault(boundary, []).extend(params)

    for module in root.modules():
        if isinstance(module, (GatedDeltaProductMixer, _GDNBase, MambaMixer)):
            # Split/shortcut execution skips the mixer's own hook too.
            add(module.in_proj, module.parameters(recurse=False))
            if isinstance(module, (GatedDeltaProductMixer, _GDNBase)):
                add(module.in_proj, module.conv1d.parameters())
            if isinstance(module, _GDNBase):
                add(module.in_proj, module.out_norm.parameters())
        elif isinstance(module, HyperConnectionModule):
            add(module, module.mapping_proj.parameters())
        elif isinstance(module, MLASelfAttention) and module._use_fused_q_uproj:
            boundary = (
                module.linear_qkv_down_proj
                if isinstance(module, FusedMLASelfAttention)
                else module.linear_kv_down_proj
            )
            add(boundary, module.linear_q_up_proj.parameters())
        elif isinstance(module, AbsorbedMLASelfAttention):
            boundary = (
                module.linear_q_up_proj
                if module.config.q_lora_rank is not None
                else module.linear_q_proj
            )
            add(boundary, module.linear_kv_up_proj.parameters())
        elif isinstance(module, (ViTModel, QwenVLViTModel, KimiViTModel)):
            # Dynamic-resolution inputs use patch_embed.forward_patches directly.
            add(module, module.patch_embed.proj.parameters())
        elif isinstance(module, AudioBlock):
            # Both the block and attention use forward_packed, bypassing their hooks.
            add(module.norm1, module.attn.w_qkv.parameters())

    return {boundary: tuple(params) for boundary, params in dependencies.items()}
