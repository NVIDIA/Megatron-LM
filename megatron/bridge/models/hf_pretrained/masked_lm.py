# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

# Adapted from https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/53c20f81cb575f19af34ff0ae1749fb13b07999b/src/megatron/bridge/models/hf_pretrained/masked_lm.py
# Source license: Apache-2.0.

#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path
from typing import Any, Dict, Generic, List, Optional, Self, TypeVar, Union

import torch
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForMaskedLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from transformers.models.auto.modeling_auto import MODEL_FOR_MASKED_LM_MAPPING

from megatron.bridge.models.hf_pretrained.base import PreTrainedBase
from megatron.bridge.models.hf_pretrained.safe_config_loader import safe_load_config_with_retry

MaskedLMType = TypeVar("MaskedLMType", bound=PreTrainedModel)


class PreTrainedMaskedLM(PreTrainedBase, Generic[MaskedLMType]):
    """
    A generic class for Pretrained Masked/Encoder-only Language Models with lazy loading.

    Allows type-safe access to specific model implementations like BertForMaskedLM.

    Unlike :class:`~megatron.bridge.models.hf_pretrained.causal_lm.PreTrainedCausalLM`,
    this class makes no generation-specific assumptions (no ``generate()``, no
    ``GenerationConfig``): encoder-only models are typically used for masked-token
    prediction or as feature extractors, not autoregressive decoding.

    The underlying model is loaded via ``AutoModelForMaskedLM``, falling back to the
    architecture-agnostic ``AutoModel`` when the config class has no registered
    masked-LM head (e.g. encoder-only checkpoints that only expose a base encoder).

    Examples:
        Basic usage with lazy loading:
        >>> from megatron.bridge.models.hf_pretrained import PreTrainedMaskedLM
        >>> # Create instance - no model loading happens yet
        >>> model = PreTrainedMaskedLM.from_pretrained("bert-base-uncased")
        >>> # Components are loaded on first access
        >>> config = model.config  # Loads config
        >>> tokenizer = model.tokenizer  # Loads tokenizer
        >>> # Run a forward pass - model is loaded here
        >>> inputs = model.encode("The capital of France is [MASK].")
        >>> outputs = model(**inputs)

        Using specific model types with type hints:
        >>> from transformers import BertForMaskedLM
        >>> from megatron.bridge.models.hf_pretrained import PreTrainedMaskedLM
        >>> bert: PreTrainedMaskedLM[BertForMaskedLM] = PreTrainedMaskedLM.from_pretrained(
        ...     "bert-base-uncased",
        ...     torch_dtype=torch.float16,
        ...     device="cuda",
        ... )
        >>> model_instance = bert.model  # Type is BertForMaskedLM
    """

    ARTIFACTS = ["tokenizer"]

    def __init__(
        self,
        model_name_or_path: Optional[Union[str, Path]] = None,
        device: Optional[Union[str, torch.device]] = None,
        torch_dtype: Optional[torch.dtype] = None,
        trust_remote_code: bool = False,
        **kwargs,
    ):
        """
        Initialize a Pretrained Masked LM with lazy loading.

        Args:
            model_name_or_path: HuggingFace model identifier or local path
            device: Device to load model on (e.g., 'cuda', 'cpu')
            torch_dtype: Data type to load model in (e.g., torch.float16)
            trust_remote_code: Whether to trust remote code when loading
            **kwargs: Additional arguments passed to from_pretrained methods
        """
        self._model_name_or_path = model_name_or_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch_dtype
        self.trust_remote_code = trust_remote_code
        super().__init__(**kwargs)
        # Store the original source path for custom modeling file preservation
        if model_name_or_path and trust_remote_code:
            self._original_source_path = model_name_or_path

    def _load_model(self) -> MaskedLMType:
        """Load the model, preferring AutoModelForMaskedLM and falling back to AutoModel."""
        if self.model_name_or_path is None:
            raise ValueError("model_name_or_path must be provided to load model")

        model_kwargs = {"trust_remote_code": self.trust_remote_code, **self.init_kwargs}
        if self.torch_dtype is not None:
            model_kwargs["torch_dtype"] = self.torch_dtype
        # Loading the config up front (rather than letting `from_pretrained` load it
        # implicitly) lets us decide the model class from the registry instead of
        # relying on catching whatever `ValueError` `from_pretrained` happens to raise.
        config = self.config
        model_kwargs["config"] = config

        if self._has_registered_masked_lm_head(config):
            model = AutoModelForMaskedLM.from_pretrained(self.model_name_or_path, **model_kwargs)
        else:
            # The config class has no registered masked-LM head (e.g. encoder-only
            # checkpoints that only expose a base encoder); fall back to AutoModel.
            model = AutoModel.from_pretrained(self.model_name_or_path, **model_kwargs)
        return model.to(self.device)

    @staticmethod
    def _has_registered_masked_lm_head(config: AutoConfig) -> bool:
        """Return whether ``config``'s class resolves to a masked-LM head.

        Checks the static Transformers registry (``MODEL_FOR_MASKED_LM_MAPPING``) as
        well as a ``trust_remote_code`` config's ``auto_map``, which declares custom
        classes that are not part of the static registry.
        """
        auto_map = getattr(config, "auto_map", None)
        if auto_map and "AutoModelForMaskedLM" in auto_map:
            return True
        # `AutoConfig` is a dynamic dispatcher: at runtime `config` is always an
        # instance of a concrete `PretrainedConfig` subclass (e.g. `BertConfig`), but
        # mypy only sees the static `AutoConfig` annotation, hence the mismatch below.
        return type(config) in MODEL_FOR_MASKED_LM_MAPPING  # type: ignore[comparison-overlap]

    def _load_config(self) -> AutoConfig:
        """Load the model config with thread-safety protection."""
        if self.model_name_or_path is None:
            raise ValueError("model_name_or_path must be provided to load config")
        return safe_load_config_with_retry(
            self.model_name_or_path, trust_remote_code=self.trust_remote_code, **self.init_kwargs
        )

    def _load_tokenizer(self) -> PreTrainedTokenizer:
        """Load the tokenizer."""
        if self.model_name_or_path is None:
            raise ValueError("model_name_or_path must be provided to load tokenizer")
        return AutoTokenizer.from_pretrained(
            self.model_name_or_path, trust_remote_code=self.trust_remote_code, **self.init_kwargs
        )

    @property
    def tokenizer(self) -> PreTrainedTokenizer:
        """Lazy load and return the tokenizer."""
        if not hasattr(self, "_tokenizer"):
            self._tokenizer = self._load_artifact("tokenizer")
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, value: PreTrainedTokenizer):
        """Set the tokenizer manually."""
        self._tokenizer = value

    @property
    def model_name_or_path(self) -> Optional[Union[str, Path]]:
        """Return the model name or path."""
        return self._model_name_or_path

    @property
    def has_model(self) -> bool:
        """Check if model has been loaded."""
        return hasattr(self, "_model") and self._model is not None

    @property
    def model(self) -> MaskedLMType:
        """Lazy load and return the underlying model."""
        return super().model

    @model.setter
    def model(self, value: MaskedLMType):
        """Set the model manually and move it to the appropriate device."""
        self._model = value
        if self._model is not None:
            self._model = self._model.to(self.device)

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: Union[str, Path],
        device: Optional[Union[str, torch.device]] = None,
        torch_dtype: Optional[torch.dtype] = None,
        trust_remote_code: bool = False,
        **kwargs,
    ) -> Self:
        """
        Create a PreTrainedMaskedLM instance for lazy loading.

        Args:
            model_name_or_path: HuggingFace model identifier or local path
            device: Device to load model on
            torch_dtype: Data type to load model in
            trust_remote_code: Whether to trust remote code
            **kwargs: Additional arguments for from_pretrained methods

        Returns:
            PreTrainedMaskedLM instance configured for lazy loading
        """
        return cls(
            model_name_or_path=model_name_or_path,
            device=device,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )

    def __call__(self, *args, **kwargs):
        """Forward call to model."""
        return self.model(*args, **kwargs)

    def encode(self, text: Union[str, List[str]], **kwargs: Any) -> Dict[str, torch.Tensor]:
        """
        Encode text into token IDs using the model's tokenizer.

        Args:
            text: Input text to encode. Can be a single string or a list of
                strings for batch encoding.
            **kwargs: Additional arguments passed to the tokenizer (e.g. padding,
                truncation, max_length, return_attention_mask).

        Returns:
            Dict[str, torch.Tensor]: Tokenizer output, moved to the model's device.
        """
        if "return_tensors" not in kwargs:
            kwargs["return_tensors"] = "pt"
        return self.tokenizer(text, **kwargs).to(self.device)

    def decode(self, token_ids: Union[int, List[int], torch.Tensor], **kwargs: Any) -> str:
        """Decode token IDs back into text using the model's tokenizer."""
        return self.tokenizer.decode(token_ids, **kwargs)

    def to(self, device: Union[str, torch.device]) -> Self:
        """Move model to specified device."""
        self.device = device
        if self.has_model:
            self._model = self._model.to(device)
        return self

    def half(self) -> Self:
        """Convert model to half precision (float16)."""
        if self.has_model:
            self._model = self._model.half()
        return self

    def float(self) -> Self:
        """Convert model to full precision (float32)."""
        if self.has_model:
            self._model = self._model.float()
        return self

    def save_pretrained(self, save_directory: Union[str, Path]):
        """
        Save all components (model, tokenizer, config) to a directory.

        Args:
            save_directory: Path to directory where components will be saved
        """
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)

        if hasattr(self, "_model") and self._model is not None:
            self._model.save_pretrained(save_path)

        self.save_artifacts(save_path)

    @property
    def dtype(self) -> Optional[torch.dtype]:
        """Get model's dtype if loaded."""
        if self.has_model:
            try:
                return next(self.model.parameters()).dtype
            except StopIteration:
                return None
        return None

    @property
    def num_parameters(self) -> Optional[int]:
        """Get total number of parameters if model is loaded."""
        if self.has_model:
            return sum(p.numel() for p in self.model.parameters())
        return None

    def __repr__(self) -> str:
        """Return a string representation of the PreTrainedMaskedLM instance."""
        try:
            _ = self.config
        except Exception:
            pass

        lines = [f"{self.__class__.__name__}("]
        for name, attr_name in sorted(self.get_artifacts().items()):
            is_loaded = hasattr(self, attr_name)
            artifact_instance = getattr(self, attr_name, None) if is_loaded else None

            type_name = "N/A"
            details = "not loaded"
            if is_loaded and artifact_instance is not None:
                type_name = artifact_instance.__class__.__name__
                if name == "tokenizer":
                    vocab = getattr(artifact_instance, "vocab_size", "N/A")
                    details = f"vocab_size={vocab}"
                else:
                    details = "loaded"
            lines.append(f"  ({name}): {type_name} [{details}]")

        model_repr_content: str
        if self.has_model:
            model_class_name = self.model.__class__.__name__
            config = self.config
            layers = getattr(config, "num_hidden_layers", "N/A")
            hidden_size = getattr(config, "hidden_size", "N/A")
            model_repr_content = (
                f"{model_class_name} [layers={layers}, hidden_size={hidden_size}, loaded]"
            )
        elif hasattr(self, "_config") and self._config is not None:
            config = self.config
            model_class_name_from_hf_config = "MaskedLM"
            if hasattr(config, "architectures") and config.architectures:
                model_class_name_from_hf_config = config.architectures[0]
            elif getattr(config, "model_type", None):
                mt = config.model_type
                model_class_name_from_hf_config = f"{mt.capitalize()}Model" if mt else "MaskedLM"

            details_parts = []
            if getattr(config, "num_hidden_layers", None) is not None:
                details_parts.append(f"layers={config.num_hidden_layers}")
            if getattr(config, "hidden_size", None) is not None:
                details_parts.append(f"hidden_size={config.hidden_size}")

            details_str = ", ".join(details_parts)
            if details_str:
                model_repr_content = (
                    f"{model_class_name_from_hf_config}({details_str}) [not loaded]"
                )
            else:
                model_repr_content = f"{model_class_name_from_hf_config} [not loaded]"
        else:
            model_repr_content = "AutoModelForMaskedLM [not loaded]"

        lines.append(f"  (model): {model_repr_content}")

        lines.sort()

        params_str = f"{self.num_parameters:,}" if self.num_parameters is not None else "N/A"
        dtype_str = str(self.dtype).replace("torch.", "") if self.dtype is not None else "N/A"
        lines.extend(
            [
                f"  (parameters): {params_str}",
                f"  (device): {str(self.device)}",
                f"  (dtype): {dtype_str}",
                ")",
            ]
        )
        return "\n".join(lines)
