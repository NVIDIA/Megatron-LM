# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""HF safetensors to mcore weight mapping for Nemotron Omni.

No existing path covers this: `megatron/core/export/` targets TRT-LLM, and
`tools/checkpoint/loader_llava.py` reads RADIO but sources vision weights from `torch.hub`
rather than HF safetensors. The four prefix classes in the checkpoint are:

    language_model.backbone.*                 -> hybrid language model
    mlp1.*                                    -> vision projector
    vision_model.radio_model.*                -> RADIOViTModel
    sound_encoder.* / sound_projection.*      -> audio tower

Two entries are dropped rather than mapped: `input_conditioner.*` (image normalization runs on
the host, in the processor) and `summary_idxs` (a teacher-selection index, unused at inference).

Two divergences from the reference implementation, both deliberate:

1. `ls1` / `ls2` (LayerScale) are *not* silently skipped. The reference loader `continue`s past
   them, which means it drops them if they are non-unit. mcore's RADIO has no LayerScale
   either, so this loader warns instead of hiding the difference.
2. Mamba SSM states are forced to fp32. The reference's config hook for this is keyed to a
   non-Omni architecture string and therefore never fires for Omni checkpoints.
"""

import logging
from typing import Dict, Iterable, List, Optional, Tuple

import torch

from megatron.core.inference.multimodal.nemotron_omni.config import NemotronOmniConfig

logger = logging.getLogger(__name__)

LANGUAGE_MODEL_PREFIX = "language_model."
VISION_PREFIX = "vision_model.radio_model."
VISION_PROJECTOR_PREFIX = "mlp1."
AUDIO_ENCODER_PREFIX = "sound_encoder."
AUDIO_PROJECTOR_PREFIX = "sound_projection."

# Checkpoint entries with no mcore counterpart.
DROPPED_SUBSTRINGS = ("input_conditioner.", "summary_idxs")


def qkv_interleave_indices(hidden_size: int, num_heads: int, kv_channels: int) -> torch.Tensor:
    """Row permutation from PyTorch's fused QKV layout to mcore's.

    PyTorch multi-head attention stores QKV as three contiguous `[hidden, hidden]` blocks;
    mcore interleaves them per head as `[q_head_0, k_head_0, v_head_0, q_head_1, ...]`. Without
    this permutation the tower loads without error and produces garbage.

    Args:
        hidden_size (int): Attention hidden size.
        num_heads (int): Attention head count.
        kv_channels (int): Channels per head.

    Return:
        (torch.Tensor) int64 index tensor of length `3 * hidden_size`.
    """
    indices: List[torch.Tensor] = []
    for head in range(num_heads):
        lower = head * kv_channels
        upper = (head + 1) * kv_channels
        indices.append(torch.arange(lower, upper, dtype=torch.int64))
        indices.append(torch.arange(hidden_size + lower, hidden_size + upper, dtype=torch.int64))
        indices.append(
            torch.arange(2 * hidden_size + lower, 2 * hidden_size + upper, dtype=torch.int64)
        )
    return torch.cat(indices)


def map_radio_key(hf_key: str, use_te: bool = True) -> Optional[str]:
    """Map one RADIO checkpoint key to its `RADIOViTModel` name.

    Args:
        hf_key (str): Key with the `vision_model.radio_model.` prefix already stripped.
        use_te (bool): Whether the tower is built with Transformer Engine fused layers, which
            absorb the pre-attention and pre-MLP norms into the linear modules.

    Return:
        (Optional[str]) mcore parameter name, or None if the entry should be dropped.
    """
    if any(dropped in hf_key for dropped in DROPPED_SUBSTRINGS):
        return None

    if "patch_generator" in hf_key:
        if "embedder" in hf_key:
            return "embedder.weight"
        if "video_embedder" in hf_key:
            return "video_embedder.weight"
        if "cls_token" in hf_key:
            # mcore stores CLS and register tokens as one [class_token_len, hidden] parameter.
            return "class_token"
        if "pos_embed" in hf_key:
            return "position_embeddings"
        return None

    if ".blocks." not in hf_key and not hf_key.startswith("blocks."):
        return None

    parts = hf_key.split(".")
    layer_idx = parts[parts.index("blocks") + 1]
    base = f"decoder.layers.{layer_idx}"
    suffix = ".".join(parts[parts.index("blocks") + 2 :])

    fused_norms = {
        "norm1.weight": f"{base}.self_attention.linear_qkv.layer_norm_weight",
        "norm1.bias": f"{base}.self_attention.linear_qkv.layer_norm_bias",
        "norm2.weight": f"{base}.mlp.linear_fc1.layer_norm_weight",
        "norm2.bias": f"{base}.mlp.linear_fc1.layer_norm_bias",
    }
    separate_norms = {
        "norm1.weight": f"{base}.input_layernorm.weight",
        "norm1.bias": f"{base}.input_layernorm.bias",
        "norm2.weight": f"{base}.pre_mlp_layernorm.weight",
        "norm2.bias": f"{base}.pre_mlp_layernorm.bias",
    }
    mapping = {
        "attn.qkv.weight": f"{base}.self_attention.linear_qkv.weight",
        "attn.qkv.bias": f"{base}.self_attention.linear_qkv.bias",
        "attn.proj.weight": f"{base}.self_attention.linear_proj.weight",
        "attn.proj.bias": f"{base}.self_attention.linear_proj.bias",
        "mlp.fc1.weight": f"{base}.mlp.linear_fc1.weight",
        "mlp.fc1.bias": f"{base}.mlp.linear_fc1.bias",
        "mlp.fc2.weight": f"{base}.mlp.linear_fc2.weight",
        "mlp.fc2.bias": f"{base}.mlp.linear_fc2.bias",
    }
    mapping.update(fused_norms if use_te else separate_norms)
    return mapping.get(suffix)


def map_projector_key(hf_key: str) -> Optional[str]:
    """Map one `mlp1.*` key to the `MultimodalProjector` name.

    The HF projector is a `nn.Sequential`: index 0 is the RMSNorm, 1 the first Linear, and 3
    the second Linear (index 2 is the activation). mcore folds the norm into fc1 via
    `TELayerNormColumnParallelLinear`.
    """
    suffix = (
        hf_key[len(VISION_PROJECTOR_PREFIX) :]
        if hf_key.startswith(VISION_PROJECTOR_PREFIX)
        else hf_key
    )
    return {
        "0.weight": "encoder.linear_fc1.layer_norm_weight",
        "1.weight": "encoder.linear_fc1.weight",
        "3.weight": "encoder.linear_fc2.weight",
    }.get(suffix)


class NemotronOmniWeightMapper:
    """Rewrites a stream of HF checkpoint tensors into mcore-shaped, mcore-named tensors.

    Args:
        config (NemotronOmniConfig): Resolved model configuration.
        use_te (bool): Whether the vision tower uses Transformer Engine fused layers.
    """

    def __init__(self, config: NemotronOmniConfig, use_te: bool = True) -> None:
        self.config = config
        self.use_te = use_te
        hidden = config.vision.hidden_size
        _, _, num_heads, _ = config.vision.dims
        self._qkv_indices = qkv_interleave_indices(hidden, num_heads, hidden // num_heads)
        self._layer_scale_keys: List[str] = []

    def check_layer_scale(self) -> None:
        """Warn if the checkpoint carried LayerScale tensors that nothing consumes.

        `mcore`'s RADIO has no LayerScale, matching the reference implementation, so parity
        between the two engines holds. But if these tensors are non-unit, *both* are dropping a
        real transform relative to the HF reference implementation, and vision features will
        differ from it. Surfaced rather than swallowed.
        """
        if self._layer_scale_keys:
            logger.warning(
                "Checkpoint contains %d LayerScale tensor(s) (e.g. %s) that mcore's RADIO does "
                "not model. This matches the reference vLLM implementation, which also skips "
                "them, but if they are not ~1.0 both engines differ from the HF reference. "
                "Verify before trusting vision-feature parity against HF.",
                len(self._layer_scale_keys),
                self._layer_scale_keys[0],
            )

    def convert(
        self, weights: Iterable[Tuple[str, torch.Tensor]]
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """Split and rename a checkpoint into per-submodule state dicts.

        Args:
            weights (Iterable[Tuple[str, torch.Tensor]]): `(key, tensor)` pairs, e.g. from
                `safetensors.safe_open`.

        Return:
            (Dict[str, Dict[str, torch.Tensor]]) State dicts keyed by
            "language_model", "vision_model", "vision_projection", "audio_model".
        """
        out: Dict[str, Dict[str, torch.Tensor]] = {
            "language_model": {},
            "vision_model": {},
            "vision_projection": {},
            "audio_model": {},
        }

        for key, tensor in weights:
            if any(dropped in key for dropped in DROPPED_SUBSTRINGS):
                continue

            if key.startswith(LANGUAGE_MODEL_PREFIX):
                out["language_model"][key[len(LANGUAGE_MODEL_PREFIX) :]] = tensor

            elif key.startswith(VISION_PREFIX):
                radio_key = key[len(VISION_PREFIX) :]
                if (
                    radio_key.endswith(("ls1", "ls2"))
                    or ".ls1." in radio_key
                    or (".ls2." in radio_key)
                ):
                    self._layer_scale_keys.append(key)
                    continue
                mapped = map_radio_key(radio_key, use_te=self.use_te)
                if mapped is None:
                    continue
                if mapped.endswith("linear_qkv.weight") or mapped.endswith("linear_qkv.bias"):
                    tensor = tensor[self._qkv_indices]
                out["vision_model"][mapped] = tensor

            elif key.startswith(VISION_PROJECTOR_PREFIX):
                mapped = map_projector_key(key)
                if mapped is not None:
                    out["vision_projection"][mapped] = tensor

            elif key.startswith(AUDIO_ENCODER_PREFIX):
                if "encoder.feature_extractor." in key:
                    # Mel-filterbank buffers; the host front-end builds its own.
                    continue
                out["audio_model"][key[len(AUDIO_ENCODER_PREFIX) :]] = tensor

            elif key.startswith(AUDIO_PROJECTOR_PREFIX):
                suffix = key[len(AUDIO_PROJECTOR_PREFIX) :]
                mapped = {
                    "0.weight": "projection.norm.weight",
                    "1.weight": "projection.fc1.weight",
                    "3.weight": "projection.fc2.weight",
                }.get(suffix)
                if mapped is not None:
                    out["audio_model"][mapped] = tensor

        self.check_layer_scale()
        return out
