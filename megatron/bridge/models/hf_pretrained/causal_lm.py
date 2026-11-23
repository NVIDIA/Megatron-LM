# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

# Adapted from https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/53c20f81cb575f19af34ff0ae1749fb13b07999b/src/megatron/bridge/models/hf_pretrained/causal_lm.py
# Source license: Apache-2.0.

#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import sys
from pathlib import Path
from typing import Any, Dict, Generic, List, Optional, TypeVar, Union

import torch
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedTokenizer,
    ProcessorMixin,
)
from transformers.configuration_utils import PretrainedConfig
from transformers.generation.utils import GenerateOutput

from megatron.bridge.models.hf_pretrained.base import PreTrainedBase
from megatron.bridge.models.hf_pretrained.safe_config_loader import safe_load_config_with_retry
from megatron.bridge.models.hf_pretrained.state import StateDict

# Python 3.12+ supports PEP 692 (TypedDict Unpack)
if sys.version_info >= (3, 12):
    from typing import TypedDict, Unpack
else:
    from typing_extensions import TypedDict, Unpack


CausalLMType = TypeVar("CausalLMType", bound=AutoModelForCausalLM)


class PreTrainedCausalLM(PreTrainedBase, Generic[CausalLMType]):
    """
    A generic class for Pretrained Causal Language Models with lazy loading.

    Allows type-safe access to specific model implementations like LlamaForCausalLM.

    Examples:
        Basic usage with lazy loading:
        >>> from mbridge.pretrained import PreTrainedCausalLM
        >>> # Create instance - no model loading happens yet
        >>> model = PreTrainedCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
        >>> # Components are loaded on first access
        >>> config = model.config  # Loads config
        >>> tokenizer = model.tokenizer  # Loads tokenizer
        >>> # Generate text - model is loaded here
        >>> inputs = model.encode("Hello, how are you?")
        >>> outputs = model.generate(**inputs, max_length=50)
        >>> print(model.decode(outputs[0], skip_special_tokens=True))

        Using specific model types with type hints:
        >>> from transformers import LlamaForCausalLM
        >>> from mbridge.pretrained import PreTrainedCausalLM
        >>> # Type-safe access to Llama-specific features
        >>> llama_model: PreTrainedCausalLM[LlamaForCausalLM] = PreTrainedCausalLM.from_pretrained(
        ...     "meta-llama/Llama-3.1-8B-Instruct",
        ...     torch_dtype=torch.float16,
        ...     device="cuda"
        ... )
        >>> # Access Llama-specific attributes
        >>> model_instance = llama_model.model  # Type is LlamaForCausalLM

        Loading with custom configurations:
        >>> # Load model with specific settings
        >>> model = PreTrainedCausalLM.from_pretrained(
        ...     "gpt2",
        ...     device="cuda:0",
        ...     torch_dtype=torch.bfloat16,
        ...     attn_implementation="flash_attention_2",
        ...     load_in_8bit=True
        ... )
        >>> # Override generation config
        >>> from transformers import GenerationConfig
        >>> model.generation_config = GenerationConfig(
        ...     max_length=100,
        ...     temperature=0.7,
        ...     top_p=0.9,
        ...     do_sample=True
        ... )

        Manual component management:
        >>> # Create empty instance
        >>> model = PreTrainedCausalLM()
        >>> # Manually set components
        >>> from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
        >>> model.config = AutoConfig.from_pretrained("microsoft/phi-2")
        >>> model.tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
        >>> model.model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2")
        >>> # Save all components
        >>> model.save_artifacts("./my_model")

        Batch processing example:
        >>> # Process multiple prompts
        >>> prompts = [
        ...     "The capital of France is",
        ...     "Machine learning is",
        ...     "Python programming language was created by"
        ... ]
        >>> # Encode all prompts
        >>> inputs = model.encode(prompts, padding=True, truncation=True)
        >>> # Generate completions
        >>> outputs = model.generate(**inputs, max_new_tokens=20)
        >>> # Decode results
        >>> for i, output in enumerate(outputs):
        ...     print(f"Prompt {i+1}: {model.decode(output, skip_special_tokens=True)}")
    """

    ARTIFACTS = ["tokenizer"]
    OPTIONAL_ARTIFACTS = ["generation_config", "processor", "image_processor"]

    def __init__(
        self,
        model_name_or_path: Optional[Union[str, Path]] = None,
        device: Optional[Union[str, torch.device]] = None,
        torch_dtype: Optional[torch.dtype] = None,
        trust_remote_code: bool = False,
        **kwargs,
    ):
        """
        Initialize a Pretrained Causal LM with lazy loading.

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
        self._text_only = False
        super().__init__(**kwargs)
        # Store the original source path for custom modeling file preservation
        if model_name_or_path and trust_remote_code:
            self._original_source_path = model_name_or_path

    def _get_auto_model_class(self) -> type:
        """Select the HF loader without changing shared lazy-loading behavior."""
        return AutoModelForCausalLM

    def _load_model(self) -> CausalLMType:
        """Load the model."""
        if self._text_only:
            # A filtered checkpoint view is for conversion, not HF's automatic
            # loading of the original multimodal repository and prefixed keys.
            raise NotImplementedError(
                "Use the Megatron text model or load its standalone HF export for inference."
            )
        if self.model_name_or_path is None:
            raise ValueError("model_name_or_path must be provided to load model")

        model_kwargs = {"trust_remote_code": self.trust_remote_code, **self.init_kwargs}
        if self.torch_dtype is not None:
            model_kwargs["torch_dtype"] = self.torch_dtype
        config = getattr(self, "_config", None)
        if config is not None:
            model_kwargs["config"] = config

        model = self._get_auto_model_class().from_pretrained(
            self.model_name_or_path, **model_kwargs
        )
        model = model.to(self.device)

        generation_config = getattr(self, "_generation_config", None)
        if generation_config is not None and hasattr(model, "generation_config"):
            model.generation_config = generation_config
        return model

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
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path, trust_remote_code=self.trust_remote_code, **self.init_kwargs
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def _load_processor(self) -> Optional[ProcessorMixin]:
        """Lazy load and return the processor."""
        if self.model_name_or_path is None:
            return None

        try:
            return AutoProcessor.from_pretrained(
                self.model_name_or_path,
                trust_remote_code=self.trust_remote_code,
                **self.init_kwargs,
            )
        except Exception:
            # Processor is optional - not all models have one
            # VLMs typically have a processor, but it might not be saved during export
            return None

    def _load_image_processor(self) -> Optional[Any]:
        """
        Lazy load and return the image processor.
        For VLMs, the image processor might be included in the processor.
        """
        # Check if image processor is available through processor first
        processor = getattr(self, "_processor", None)
        if processor is not None and hasattr(processor, "image_processor"):
            return processor.image_processor

        # Try to load image processor separately
        if self.model_name_or_path is not None:
            try:
                return AutoImageProcessor.from_pretrained(
                    self.model_name_or_path,
                    trust_remote_code=self.trust_remote_code,
                    **self.init_kwargs,
                )
            except Exception:
                # Some VLMs include image processor only in processor
                pass
        return None

    def _load_generation_config(self) -> Optional[GenerationConfig]:
        """Load the generation config."""
        if self.model_name_or_path is not None:
            try:
                return GenerationConfig.from_pretrained(
                    self.model_name_or_path,
                    trust_remote_code=self.trust_remote_code,
                    **self.init_kwargs,
                )
            except Exception:
                # Not all models have generation configs
                pass
        return None

    @property
    def auto_map_model_class(self) -> Optional[str]:
        """Get the AutoModelForCausalLM class from the config."""
        config = self.config
        auto_map = getattr(config, "auto_map", None)
        if auto_map and "AutoModelForCausalLM" in auto_map:
            auto_map_class = auto_map["AutoModelForCausalLM"]
            return str(auto_map_class)

        return None

    @property
    def generation_config(self) -> Optional[GenerationConfig]:
        """Lazy load and return the generation config."""
        if not hasattr(self, "_generation_config"):
            self._generation_config = self._load_generation_config()
        return self._generation_config

    @generation_config.setter
    def generation_config(self, value: GenerationConfig):
        """Set the generation config manually."""
        self._generation_config = value
        # Update model's generation config if model is already loaded
        model = getattr(self, "_model", None)
        if model is not None and hasattr(model, "generation_config"):
            model.generation_config = value

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
    def processor(self) -> ProcessorMixin:
        """Lazy load and return the processor."""
        if not hasattr(self, "_processor"):
            self._processor = self._load_artifact("processor")
        return self._processor

    @processor.setter
    def processor(self, value: ProcessorMixin):
        """Set the processor manually."""
        self._processor = value

    @property
    def image_processor(self) -> Optional[Any]:
        """Lazy load and return the image processor."""
        if not hasattr(self, "_image_processor"):
            self._image_processor = self._load_artifact("image_processor")
        return self._image_processor

    @image_processor.setter
    def image_processor(self, value: Any):
        """Set the image processor manually."""
        self._image_processor = value

    @property
    def model_name_or_path(self) -> Optional[Union[str, Path]]:
        """Return the model name or path."""
        return self._model_name_or_path

    @property
    def has_model(self) -> bool:
        """Check if model has been loaded."""
        return hasattr(self, "_model") and self._model is not None

    @property
    def model(self) -> CausalLMType:
        """Lazy load and return the underlying model."""
        return super().model

    @model.setter
    def model(self, value: CausalLMType):
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
    ) -> "PreTrainedCausalLM[CausalLMType]":
        """
        Create a PreTrainedCausalLM instance for lazy loading.

        Args:
            model_name_or_path: HuggingFace model identifier or local path
            device: Device to load model on
            torch_dtype: Data type to load model in
            trust_remote_code: Whether to trust remote code
            **kwargs: Additional arguments for from_pretrained methods

        Returns:
            PreTrainedCausalLM instance configured for lazy loading
        """
        return cls(
            model_name_or_path=model_name_or_path,
            device=device,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )

    def generate(
        self, input_ids: Optional[torch.LongTensor] = None, **kwargs: Unpack["GenerateKwargs"]
    ) -> Union[torch.LongTensor, GenerateOutput]:
        """
        Generate text using the underlying language model.

        This method forwards all arguments to the model's generate method,
        supporting all generation strategies provided by the transformers library.

        Common parameters include:
            inputs (torch.LongTensor, optional): Input token IDs. If not provided,
                will generate from the beginning of sequence token.
            max_length (int, optional): Maximum length of generated sequence.
                Defaults to model's max_length configuration.
            min_length (int, optional): Minimum length of generated sequence.
            max_new_tokens (int, optional): Maximum number of tokens to generate,
                ignoring the number of tokens in the prompt.
            do_sample (bool, optional): Whether to use sampling. Defaults to False
                (greedy decoding).
            temperature (float, optional): Temperature for sampling. Higher values
                produce more random outputs. Typical range: 0.1-2.0.
            top_p (float, optional): Nucleus sampling threshold. Only tokens with
                cumulative probability up to top_p are considered. Range: 0.0-1.0.
            top_k (int, optional): Only consider the top k tokens for sampling.
            num_beams (int, optional): Number of beams for beam search. 1 means
                no beam search.
            repetition_penalty (float, optional): Penalty for repeating tokens.
                Values > 1.0 discourage repetition.
            pad_token_id (int, optional): ID of padding token.
            eos_token_id (int or List[int], optional): ID(s) of end-of-sequence token(s).
            use_cache (bool, optional): Whether to use past key values to speed up
                generation. Defaults to True.

        Returns:
            torch.LongTensor or transformers.generation.utils.GenerateOutput:
                Generated token IDs. If return_dict_in_generate=True, returns a
                GenerateOutput object containing generated sequences and additional
                information like scores.

        Examples:
            >>> # Basic generation
            >>> model = PreTrainedCausalLM.from_pretrained("gpt2")
            >>> inputs = model.encode("Hello, how are")
            >>> outputs = model.generate(inputs["input_ids"], max_length=20)
            >>> print(model.decode(outputs[0]))

            >>> # Generation with sampling
            >>> outputs = model.generate(
            ...     inputs["input_ids"],
            ...     max_length=50,
            ...     do_sample=True,
            ...     temperature=0.8,
            ...     top_p=0.9
            ... )

            >>> # Beam search
            >>> outputs = model.generate(
            ...     inputs["input_ids"],
            ...     max_length=50,
            ...     num_beams=5,
            ...     early_stopping=True
            ... )

        Note:
            For detailed documentation of all parameters, see the transformers
            library documentation for generation methods.
        """
        model = self.model  # Ensures model is loaded
        # Sync generation config if it has been set on the wrapper
        generation_config = getattr(self, "_generation_config", None)
        if generation_config is not None and hasattr(model, "generation_config"):
            model.generation_config = generation_config
        return model.generate(input_ids, **kwargs)

    def __call__(self, *args, **kwargs):
        """Forward call to model."""
        return self.model(*args, **kwargs)

    def encode(
        self, text: Union[str, List[str]], **kwargs: Unpack["EncodeKwargs"]
    ) -> Dict[str, torch.Tensor]:
        """
        Encode text into token IDs using the model's tokenizer.

        This method tokenizes input text and returns tensors ready for model input.
        The output is automatically moved to the same device as the model.

        Args:
            text (str or List[str]): Input text to encode. Can be a single string
                or a list of strings for batch encoding.
            **kwargs: Additional arguments passed to the tokenizer. Common options:
                padding (bool or str, optional): Padding strategy.
                    - True or 'longest': Pad to longest sequence in batch
                    - 'max_length': Pad to max_length
                    - False or 'do_not_pad': No padding (default)
                truncation (bool or str, optional): Truncation strategy.
                    - True or 'longest_first': Truncate to max_length
                    - 'only_first': Truncate only first sequence (for pairs)
                    - False: No truncation
                max_length (int, optional): Maximum length of returned sequences.
                    Defaults to model's max_length.
                add_special_tokens (bool, optional): Whether to add special tokens
                    (e.g., [CLS], [SEP]). Defaults to True.
                return_attention_mask (bool, optional): Whether to return attention
                    mask. Defaults to True.
                return_token_type_ids (bool, optional): Whether to return token
                    type IDs (for models like BERT). Defaults to True if model
                    expects them.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing:
                - input_ids: Token IDs tensor of shape (batch_size, sequence_length)
                - attention_mask: Attention mask tensor of same shape (if applicable)
                - token_type_ids: Token type IDs tensor (if applicable)
                Additional keys may be present depending on the tokenizer.

        Examples:
            >>> model = PreTrainedCausalLM.from_pretrained("gpt2")
            >>> # Single text encoding
            >>> tokens = model.encode("Hello world!")
            >>> print(tokens["input_ids"].shape)  # torch.Size([1, 3])

            >>> # Batch encoding with padding
            >>> texts = ["Hello!", "How are you doing today?"]
            >>> tokens = model.encode(texts, padding=True)
            >>> print(tokens["input_ids"].shape)  # torch.Size([2, 6])

            >>> # Encoding with truncation
            >>> tokens = model.encode(
            ...     "This is a very long text that might exceed the maximum length",
            ...     truncation=True,
            ...     max_length=10
            ... )

        Note:
            The returned tensors are on the same device as the model, ready
            for immediate use in forward passes or generation.
        """
        # Only set return_tensors default if not provided
        if "return_tensors" not in kwargs:
            kwargs["return_tensors"] = "pt"

        return self.tokenizer(text, **kwargs).to(self.device)

    def decode(
        self, token_ids: Union[int, List[int], torch.Tensor], **kwargs: Unpack["DecodeKwargs"]
    ) -> str:
        """
        Decode token IDs back into text using the model's tokenizer.

        This method converts token IDs (from model output or encode method)
        back into human-readable text.

        Args:
            token_ids (int, List[int], or torch.Tensor): Token IDs to decode.
                Can be:
                - Single token ID (int)
                - List of token IDs
                - 1D tensor of token IDs
                - 2D tensor (will decode the first sequence)
            **kwargs: Additional arguments passed to the tokenizer's decode method:
                skip_special_tokens (bool, optional): Whether to remove special
                    tokens (e.g., [PAD], [CLS], [SEP]) from output. Defaults to True.
                clean_up_tokenization_spaces (bool, optional): Whether to clean up
                    tokenization artifacts (extra spaces, etc.). Defaults to True.

        Returns:
            str: Decoded text string.

        Examples:
            >>> model = PreTrainedCausalLM.from_pretrained("gpt2")
            >>> # Encode and decode round-trip
            >>> text = "Hello, world!"
            >>> tokens = model.encode(text)
            >>> decoded = model.decode(tokens["input_ids"][0])
            >>> print(decoded)  # "Hello, world!"

            >>> # Decode generated tokens
            >>> inputs = model.encode("The weather is")
            >>> outputs = model.generate(inputs["input_ids"], max_length=10)
            >>> decoded = model.decode(outputs[0])
            >>> print(decoded)  # "The weather is nice today..."

            >>> # Decode without special tokens
            >>> token_ids = [101, 7592, 1010, 2088, 999, 102]  # BERT-style tokens
            >>> decoded = model.decode(token_ids, skip_special_tokens=True)
            >>> print(decoded)  # "Hello, world!"

            >>> # Decode keeping special tokens
            >>> decoded = model.decode(token_ids, skip_special_tokens=False)
            >>> print(decoded)  # "[CLS] Hello, world! [SEP]"

        Note:
            If a 2D tensor is provided (batch of sequences), only the first
            sequence is decoded. For batch decoding, use tokenizer.batch_decode()
            directly or iterate over the sequences.
        """
        return self.tokenizer.decode(token_ids, **kwargs)

    def to(self, device: Union[str, torch.device]):
        """Move model to specified device."""
        self.device = device
        if self.has_model:
            self._model = self._model.to(device)
        return self

    def half(self):
        """Convert model to half precision (float16)."""
        if self.has_model:
            self._model = self._model.half()
        return self

    def float(self):
        """Convert model to full precision (float32)."""
        if self.has_model:
            self._model = self._model.float()
        return self

    def save_pretrained(self, save_directory: Union[str, Path]):
        """
        Save all components (model, tokenizer, config, generation_config) to a directory.

        This method saves:
        - Model weights and config
        - Tokenizer files
        - Generation config (if available)

        Args:
            save_directory: Path to directory where components will be saved
        """
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save model if loaded
        if hasattr(self, "_model") and self._model is not None:
            self._model.save_pretrained(save_path)

        # Use the base class save_artifacts to save config and all artifacts
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
        """Return a string representation of the PreTrainedCausalLM instance."""
        try:
            # Access config to trigger lazy loading for a richer repr
            _ = self.config
        except Exception:
            # If loading fails, repr shouldn't crash.
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
                elif name == "config":
                    m_type = getattr(artifact_instance, "model_type", "N/A")
                    details = f"model_type={m_type}"
                else:
                    details = "loaded"
            lines.append(f"  ({name}): {type_name} [{details}]")

        # Manually add model repr
        model_repr_content: str
        if self.has_model:
            model_class_name = self.model.__class__.__name__
            # Assuming self.config is loaded or available here due to earlier attempt
            config = self.config
            layers = getattr(config, "num_hidden_layers", "N/A")
            hidden_size = getattr(config, "hidden_size", "N/A")
            model_repr_content = (
                f"{model_class_name} [layers={layers}, hidden_size={hidden_size}, loaded]"
            )
        elif "config" in self.__dict__:  # Model not loaded, but config is
            config = self.config
            model_class_name_from_hf_config = "CausalLM"  # Default
            if hasattr(config, "architectures") and config.architectures:
                model_class_name_from_hf_config = config.architectures[0]
            elif getattr(config, "model_type", None):
                mt = config.model_type
                model_class_name_from_hf_config = f"{mt.capitalize()}Model" if mt else "CausalLM"

            details_parts = []
            if getattr(config, "num_hidden_layers", None) is not None:
                details_parts.append(f"layers={config.num_hidden_layers}")
            if getattr(config, "hidden_size", None) is not None:
                details_parts.append(f"hidden_size={config.hidden_size}")

            details_str = ", ".join(details_parts)
            status_suffix = "not loaded"
            if details_str:
                model_repr_content = (
                    f"{model_class_name_from_hf_config}({details_str}) [{status_suffix}]"
                )
            else:
                model_repr_content = f"{model_class_name_from_hf_config} [{status_suffix}]"
        else:  # Model and Config also not loaded
            model_repr_content = "AutoModelForCausalLM [not loaded]"

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


class _ConfigOnlyPretrainedShim(PreTrainedCausalLM):
    """Config-backed causal LM wrapper used for provider construction without weights."""

    def __init__(self, config: PretrainedConfig):
        super().__init__(
            model_name_or_path=getattr(config, "name_or_path", None),
            trust_remote_code=bool(getattr(config, "trust_remote_code", False)),
        )
        self.config = config
        self.generation_config = self._build_generation_config(config)

    @property
    def state(self) -> StateDict:
        """Reject weight-state access so config-only provider construction remains weightless."""
        raise AttributeError("Config-only pretrained wrappers do not have weight state.")

    @staticmethod
    def _build_generation_config(config: PretrainedConfig) -> GenerationConfig:
        try:
            config_dict = config.to_dict() if hasattr(config, "to_dict") else {}
            return GenerationConfig.from_dict(config_dict)
        except Exception:
            return GenerationConfig()

    def __repr__(self) -> str:
        config_name = getattr(self.config, "__class__", type(self.config)).__name__
        return f"_ConfigOnlyPretrainedShim(config={config_name})"


# TypedDict definitions for method parameters
class GenerateKwargs(TypedDict, total=False):
    """TypedDict for generate method parameters."""

    attention_mask: Optional[torch.Tensor]
    max_length: Optional[int]
    max_new_tokens: Optional[int]
    min_length: Optional[int]
    do_sample: Optional[bool]
    temperature: Optional[float]
    top_k: Optional[int]
    top_p: Optional[float]
    repetition_penalty: Optional[float]
    pad_token_id: Optional[int]
    eos_token_id: Optional[Union[int, List[int]]]
    bos_token_id: Optional[int]
    num_beams: Optional[int]
    num_return_sequences: Optional[int]
    early_stopping: Optional[bool]
    use_cache: Optional[bool]
    return_dict_in_generate: Optional[bool]
    output_scores: Optional[bool]
    output_attentions: Optional[bool]


class EncodeKwargs(TypedDict, total=False):
    """TypedDict for encode method parameters."""

    padding: Union[bool, str]
    truncation: Union[bool, str]
    max_length: Optional[int]
    add_special_tokens: bool
    return_attention_mask: bool
    return_token_type_ids: Optional[bool]
    return_tensors: str


class DecodeKwargs(TypedDict, total=False):
    """TypedDict for decode method parameters."""

    skip_special_tokens: bool
    clean_up_tokenization_spaces: bool
