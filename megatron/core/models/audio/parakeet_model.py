# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Parakeet Conformer audio tower plus its projector, loaded from an Omni checkpoint.

Deliberately not built on `ParakeetHuggingFaceModel`
(`megatron/core/models/huggingface/fastconformer_model.py`): that wrapper calls
`AutoModel.from_pretrained` on a *separate* `hf://` or `nemo://` checkpoint, whereas the tower
here has to be constructed empty and then filled from the Omni checkpoint's `sound_encoder.*`
tensors. Only the dtype and sampling-rate conventions are shared.

The tower is replicated, not tensor-parallel: it is small relative to the language model, runs
once per request at admission rather than once per step, and replicating it avoids an
all-gather on the encoder output.
"""

from typing import List, Optional

import torch

from megatron.core.inference.multimodal.nemotron_omni.config import SoundConfig

# `transformers.ParakeetEncoder` landed in transformers 5.5.3.
MIN_TRANSFORMERS_VERSION = "5.5.3"


def _build_hf_encoder(config: SoundConfig, dtype: torch.dtype) -> torch.nn.Module:
    """Instantiate an empty `transformers.ParakeetEncoder` from a `SoundConfig`."""
    try:
        from transformers import ParakeetEncoder, ParakeetEncoderConfig
    except ImportError as exc:
        import transformers

        raise ImportError(
            f"Nemotron Omni audio support needs transformers >= {MIN_TRANSFORMERS_VERSION} for "
            f"ParakeetEncoder; found {getattr(transformers, '__version__', 'unknown')}. Either "
            "bump the container (see the mcore-build-and-dependency skill) or run the model "
            "without audio."
        ) from exc

    hf_config = ParakeetEncoderConfig(
        hidden_size=config.hidden_size,
        num_attention_heads=config.num_attention_heads,
        num_hidden_layers=config.num_hidden_layers,
        intermediate_size=config.intermediate_size,
        conv_kernel_size=config.conv_kernel_size,
        convolution_bias=config.convolution_bias,
        subsampling_conv_channels=config.subsampling_conv_channels,
        subsampling_conv_kernel_size=config.subsampling_conv_kernel_size,
        subsampling_factor=config.subsampling_factor,
        num_mel_bins=config.num_mel_bins,
    )
    return ParakeetEncoder(hf_config).to(dtype)


class ParakeetProjection(torch.nn.Module):
    """`RMSNorm -> Linear -> ReLU^2 -> Linear`, bias-free.

    Replicated rather than tensor-parallel, matching the tower. Kept as plain torch modules so
    the checkpoint's `sound_projection.*` keys map across without a parallel-layout reindex.
    """

    def __init__(
        self, input_size: int, hidden_size: int, output_size: int, eps: float = 1e-5
    ) -> None:
        super().__init__()
        self.norm = torch.nn.RMSNorm(input_size, eps=eps)
        self.fc1 = torch.nn.Linear(input_size, hidden_size, bias=False)
        self.fc2 = torch.nn.Linear(hidden_size, output_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project encoder output to the language model's hidden size."""
        hidden_states = self.norm(hidden_states)
        hidden_states = self.fc1(hidden_states)
        hidden_states = torch.nn.functional.relu(hidden_states).square()
        return self.fc2(hidden_states)


class ProjectedParakeet(torch.nn.Module):
    """Parakeet encoder plus projection, trimming each clip to its valid length.

    Args:
        config (SoundConfig): Audio tower geometry.
        llm_hidden_size (int): Language model hidden size to project to.
        projector_hidden_size (int): Projector inner width.
        dtype (torch.dtype): Parameter dtype. The towers are unquantized even for FP8/NVFP4
            checkpoints -- the quantized-layer manifest contains no `sound_encoder.*` entries.
        projector_norm_eps (float): RMSNorm epsilon for the projector.
    """

    def __init__(
        self,
        config: SoundConfig,
        llm_hidden_size: int,
        projector_hidden_size: int,
        dtype: torch.dtype = torch.bfloat16,
        projector_norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.config = config
        self.encoder = _build_hf_encoder(config, dtype)
        self.projection = ParakeetProjection(
            input_size=config.hidden_size,
            hidden_size=projector_hidden_size,
            output_size=llm_hidden_size,
            eps=projector_norm_eps,
        ).to(dtype)

    def subsampling_output_length(self, num_frames: torch.Tensor) -> torch.Tensor:
        """Encoder rows produced from a batch of mel frame counts.

        Delegates to the HF encoder so the device-side trim cannot drift from the host-side
        token count that `ParakeetAudioProcessor` computed.
        """
        return self.encoder._get_subsampling_output_length(num_frames.to(torch.float))

    def forward(
        self, mel_features: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> List[torch.Tensor]:
        """Encode a batch of clips and trim each to its valid row count.

        Args:
            mel_features (torch.Tensor): `[num_clips, max_frames, num_mel_bins]`.
            attention_mask (Optional[torch.Tensor]): `[num_clips, max_frames]` bool.

        Return:
            (List[torch.Tensor]) One `[rows, llm_hidden_size]` tensor per clip. Returned
            unconcatenated because the caller groups clips back into items.
        """
        outputs = self.encoder(input_features=mel_features, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        hidden_states = self.projection(hidden_states.to(dtype=self.projection.fc1.weight.dtype))

        if attention_mask is None:
            return list(hidden_states)

        valid_lengths = self.subsampling_output_length(attention_mask.sum(dim=1))
        return [hidden_states[i, : int(valid_lengths[i])] for i in range(hidden_states.shape[0])]
