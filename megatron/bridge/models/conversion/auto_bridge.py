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

# Adapted from https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/53c20f81cb575f19af34ff0ae1749fb13b07999b/src/megatron/bridge/models/conversion/auto_bridge.py
# Source license: Apache-2.0.

from __future__ import annotations

import dataclasses
import inspect
import json
import logging
import os
from collections.abc import Callable, Mapping
from contextlib import nullcontext
from functools import cached_property, partial
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Iterable,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import torch
import torch.distributed as dist
import transformers
import yaml
from pydantic import BaseModel

from megatron.core import parallel_state
from megatron.core.process_groups_config import ProcessGroupCollection

if TYPE_CHECKING:
    from megatron.bridge.models.conversion.modelopt_utils import ModelOptExportPlan
    from megatron.bridge.models.conversion.param_mapping import LocalHFParam, LocalMXFP8Param

from safetensors.torch import save_file
from transformers.configuration_utils import PretrainedConfig
from typing_extensions import Unpack

from megatron.bridge.models.conversion import model_bridge
from megatron.bridge.models.conversion.model_bridge import (
    HFWeightTuple,
    MegatronModelBridge,
    MegatronWeightTuple,
    WeightConversionTask,
)
from megatron.bridge.models.conversion.utils import get_causal_lm_class_name_via_auto_map
from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.models.hf_pretrained.causal_lm import (
    PreTrainedCausalLM,
    _ConfigOnlyPretrainedShim,
)
from megatron.bridge.models.hf_pretrained.masked_lm import PreTrainedMaskedLM
from megatron.bridge.models.hf_pretrained.safe_config_loader import safe_load_config_with_retry
from megatron.bridge.models.hf_pretrained.state import (
    SafeTensorsStateSource,
    _existing_safetensors_files,
)
from megatron.bridge.models.hf_pretrained.token_classification import PreTrainedTokenClassification
from megatron.bridge.models.model_provider import (
    GetModelKwargs,
    ModelParallelKwargs,
    ModelProviderMixin,
    build_distributed_models,
)
from megatron.bridge.utils.common_utils import get_local_rank_preinit
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import MLATransformerConfig, TransformerConfig
from megatron.training.models.base import ModelConfig

logger = logging.getLogger(__name__)

MegatronModelT = TypeVar("MegatronModelT", bound=MegatronModule)
DataclassT = TypeVar("DataclassT")

# Supported HuggingFace architecture suffixes for causal generation models, plus a
# small allowlist of non-causal architectures (e.g. masked/encoder-only LMs) that are
# handled via a different hf_pretrained wrapper. See _resolve_pretrained_wrapper_cls.
SUPPORTED_HF_ARCHITECTURES: tuple[str, ...] = (
    "ForCausalLM",
    "ForConditionalGeneration",
    "NemotronH_Nano_VL_V2",
    "NemotronH_Nano_Omni_Reasoning_V3",
    "NemotronH_Super_Omni_Reasoning_V3",
    "NemotronH_Omni_Reasoning_V3",
    "Qwen2_5OmniModel",
    "MiniCPMO",
    "NemotronLabsDiffusionModel",
    "DiffusionGemmaForBlockDiffusion",
    "LLaDAModelLM",  # trust_remote_code class for GSAI-ML LLaDA1.5 (masked-diffusion LLM)
    "ForMaskedLM",  # encoder-only masked LMs (e.g. BertForMaskedLM), loaded via PreTrainedMaskedLM
    "ForTokenClassification",
)

# Error-message metadata only: shared architectures used by supported successors
# must not appear here. Consult this only after normal bridge lookup fails.
_REMOVED_HF_ARCHITECTURES = {
    "DeepseekV2ForCausalLM": "DeepSeek V2",
    "DeciLMForCausalLM": "Llama Nemotron",
    "GemmaForCausalLM": "Gemma 1",
    "Gemma2ForCausalLM": "Gemma 2",
    "NemotronForCausalLM": "Nemotron-4",
}

# hf_pretrained wrapper types that carry both a config and (optionally) loaded weights.
# Used for isinstance checks that should accept any such wrapper, regardless of which
# HF Auto* class it loads the underlying model with.
_PRETRAINED_WRAPPER_TYPES: tuple[type, ...] = (
    PreTrainedCausalLM,
    PreTrainedMaskedLM,
    PreTrainedTokenClassification,
)

# Mapping from non-standard HF architecture names to their actual transformers class names.
# Some HF model configs report architecture names that don't follow the standard
# 'ForCausalLM'/'ForConditionalGeneration' convention and don't directly map to a
# transformers class. This dict resolves those aliases.
HF_ARCHITECTURE_ALIASES: dict[str, str] = {
    "Qwen2_5OmniModel": "Qwen2_5OmniForConditionalGeneration"
}

MTP_CONFIG_FIELDS: tuple[str, ...] = (
    "num_nextn_predict_layers",
    "mtp_num_hidden_layers",
    "mtp_num_layers",
)
_MISSING = object()

_MINIMUM_TRANSFORMERS_BY_ARCHITECTURE: dict[str, str] = {"Qwen3_5ForTokenClassification": "5.9"}


def _validate_hf_model_runtime_support(
    config: PretrainedConfig, *, trust_remote_code: bool
) -> None:
    """Validate dependencies needed to load an actual Hugging Face model.

    Config-only Megatron construction does not need the corresponding HF model
    implementation, so this check is intentionally limited to weight-loading
    entry points and ``can_handle``.

    Args:
        config: Hugging Face model configuration being checked.
        trust_remote_code: Whether a config ``auto_map`` implementation may be used.

    Raises:
        ValueError: If the installed Transformers cannot load the requested model.
    """
    architectures = getattr(config, "architectures", None) or []
    auto_map = getattr(config, "auto_map", None) or {}
    for architecture in architectures:
        minimum_transformers = _MINIMUM_TRANSFORMERS_BY_ARCHITECTURE.get(architecture)
        if minimum_transformers is None:
            continue
        has_remote_implementation = trust_remote_code and bool(
            auto_map.get("AutoModelForTokenClassification")
        )
        if getattr(transformers, architecture, None) is None and not has_remote_implementation:
            raise ValueError(
                f"{architecture} requires transformers >= {minimum_transformers}, "
                f"but found {transformers.__version__}. Please upgrade transformers."
            )


def _get_config_field(config: Any, field: str) -> Any:
    if isinstance(config, dict):
        return config.get(field, _MISSING)

    return getattr(config, "__dict__", {}).get(field, _MISSING)


def _config_disables_mtp(config: Any) -> bool:
    """Return True when a config object or dict explicitly disables MTP layers."""
    if config is None:
        return False

    for field in MTP_CONFIG_FIELDS:
        value = _get_config_field(config, field)
        if value is _MISSING or value is None:
            continue

        return int(value) == 0

    text_config = _get_config_field(config, "text_config")
    return text_config is not _MISSING and _config_disables_mtp(text_config)


def _saved_config_disables_mtp(path: str | Path) -> bool:
    """Check the final config.json written for an HF export."""
    import json

    config_path = Path(path) / "config.json"
    if not config_path.exists():
        return False
    with open(config_path) as f:
        return _config_disables_mtp(json.load(f))


def _mtp_source_key_prefixes(source: Any, *configs: Any) -> tuple[str, ...]:
    """Source-checkpoint key prefixes for MTP/nextn tensors that must be ignored
    when exporting a model built without an MTP head.

    Different HF architectures name their Multi-Token-Prediction (nextn) tensors
    differently:

    * DeepSeek-style: a dedicated ``mtp.*`` prefix.
    * GLM-4.x ``glm4_moe_lite`` and Step3.7: nextn layers are stored as regular
      decoder layers starting at index ``num_hidden_layers`` (i.e. one past the
      last real layer), so their tensors live under consecutive
      ``model.layers.{layer_index}.*`` prefixes.

    When the megatron model is built without an MTP head the generator never
    yields these tensors. If they remain in the source sharding map, the shards
    that co-locate them with real (non-MTP) tensors can never be completed and
    get dropped wholesale (taking real boundary params with them). Stripping
    these prefixes from the expected source map lets those shards complete with
    only their real keys.

    Precondition: only call this when the built model omits MTP (see
    ``_model_omits_mtp``). The returned prefixes are always stripped, so calling
    it for an MTP-enabled export would silently drop real nextn tensors.

    Returns the tuple of prefixes that exist in ``source`` and should be ignored.
    """
    prefixes: list[str] = []

    if source.has_glob("mtp.*"):
        prefixes.append("mtp.")

    # GLM nextn layer at index == num_hidden_layers. HF decoder layers are
    # 0-indexed, so layer ``num_hidden_layers`` is one past the last real layer.
    # ``configs`` is ordered hf_config-before-model_config so the HF value wins
    # on conflict (source keys are HF-shaped).
    num_hidden_layers = _MISSING
    for config in configs:
        value = _get_config_field(config, "num_hidden_layers")
        if value is _MISSING or value is None:
            text_config = _get_config_field(config, "text_config")
            if text_config is not _MISSING:
                value = _get_config_field(text_config, "num_hidden_layers")
        if value is not _MISSING and value is not None:
            num_hidden_layers = int(value)
            break

    if num_hidden_layers is not _MISSING:
        num_nextn_layers = 1
        for config in configs:
            text_config = _get_config_field(config, "text_config")
            config_candidates = (config,) if text_config is _MISSING else (config, text_config)
            for config_candidate in config_candidates:
                for field in MTP_CONFIG_FIELDS:
                    value = _get_config_field(config_candidate, field)
                    if value is not _MISSING and value is not None:
                        num_nextn_layers = max(num_nextn_layers, int(value))

        for layer_index in range(num_hidden_layers, num_hidden_layers + num_nextn_layers):
            nextn_prefix = f"model.layers.{layer_index}."
            if source.has_glob(f"{nextn_prefix}*"):
                prefixes.append(nextn_prefix)

    return tuple(prefixes)


def _model_omits_mtp(model_config: Any) -> bool:
    """True when the *built* megatron model has no MTP head.

    Unlike :func:`_config_disables_mtp` (which treats ``None``/unset as
    "unspecified"), the built model's provider carries an explicit
    ``mtp_num_layers``; a falsy value (``None`` or ``0``) means no MTP head was
    instantiated, so the export generator will not yield any MTP/nextn tensors.
    """
    if model_config is None:
        return False
    value = _get_config_field(model_config, "mtp_num_layers")
    if value is _MISSING:
        return False
    return not value


# Preformatted display string for error/help messages
SUPPORTED_HF_ARCHITECTURES_DISPLAY = " or ".join(f"'{s}'" for s in SUPPORTED_HF_ARCHITECTURES)


def _resolve_pretrained_wrapper_cls(
    config: Any,
) -> Type[PreTrainedCausalLM] | Type[PreTrainedMaskedLM] | Type[PreTrainedTokenClassification]:
    """Select the hf_pretrained wrapper class for a model's architecture.

    Token classifiers (``*ForTokenClassification``) are loaded through
    :class:`PreTrainedTokenClassification`, which uses
    ``AutoModelForTokenClassification``. Masked/encoder-only LMs
    (``*ForMaskedLM``) are loaded through
    :class:`PreTrainedMaskedLM`, which uses ``AutoModelForMaskedLM``. This matters
    because ``AutoModelForCausalLM`` (used by :class:`PreTrainedCausalLM`) resolves
    some masked-LM config classes (e.g. ``BertConfig``) to an unrelated causal-LM
    class instead of raising, silently loading the wrong architecture. Remaining
    supported architectures use :class:`PreTrainedCausalLM`.
    """
    architectures = getattr(config, "architectures", None) or []
    if "NemotronH_Omni_Reasoning_V3" in architectures:
        from megatron.bridge.models.nemotron_super_vl.hf_pretrained import Nemotron35VLPretrained

        return Nemotron35VLPretrained
    if any(arch.endswith("ForTokenClassification") for arch in architectures):
        return PreTrainedTokenClassification
    if any(arch.endswith("ForMaskedLM") for arch in architectures):
        return PreTrainedMaskedLM
    return PreTrainedCausalLM


def _drop_readonly_config_properties(
    config_dict: dict[str, object], config_type: Type[PretrainedConfig]
) -> dict[str, object]:
    """Remove read-only property names before constructing a HuggingFace config."""
    readonly_properties = {
        name
        for cls in config_type.mro()
        for name, attr in vars(cls).items()
        if isinstance(attr, property) and attr.fset is None
    }
    if not readonly_properties:
        return config_dict
    return {key: value for key, value in config_dict.items() if key not in readonly_properties}


def _check_streaming_argument_support(
    stream_fn: Callable[..., Any], owner: object, argument: str
) -> None:
    """Fail early when a streaming override cannot accept an export argument."""
    try:
        parameters = inspect.signature(stream_fn).parameters
    except (TypeError, ValueError):
        # Not introspectable (C callables); let the call itself decide.
        return
    if argument in parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
    ):
        return
    raise TypeError(
        f"{type(owner).__name__}.{getattr(stream_fn, '__name__', 'stream_weights')} does not accept "
        f"'{argument}'; a bridge that overrides the streaming export must forward this flag "
        "before the requested export mode can be used."
    )


def _check_with_megatron_names_support(stream_fn: Callable[..., Any], owner: object) -> None:
    """Fail up front when a streamer cannot accept ``with_megatron_names``."""
    _check_streaming_argument_support(stream_fn, owner, "with_megatron_names")


def _check_current_pp_stage_only_support(stream_fn: Callable[..., Any], owner: object) -> None:
    """Fail early when a streamer cannot accept PP-local export."""
    _check_streaming_argument_support(stream_fn, owner, "current_pp_stage_only")


MODELOPT_QUANTIZER_STATE_FILE = "quantizer_state.pth"
MODELOPT_QUANTIZER_RECIPE_FILE = "quant_recipe.yaml"
MODELOPT_QUANTIZER_RECIPE_FORMAT = "modelopt_quantizer_recipe"
MODELOPT_QUANTIZER_RECIPE_VERSION = 1


def _save_modelopt_quantizer_sidecars(
    path: Path, quantizer_recipe: dict[str, dict[str, Any]], quant_tensors: dict[str, torch.Tensor]
) -> None:
    """Write ModelOpt quantizer ranges and recipe under HF names, e.g. for vLLM fake-quant serving.

    ``quantizer_state.pth`` holds the tensors (e.g. ``_amax``) of the enabled quantizers and
    ``quant_recipe.yaml`` their properties-only ModelOpt state. Neither is written when no
    quantizer is enabled (e.g. weight-only formats whose weights were folded), so the export
    stays a plain checkpoint.
    """

    class _RecipeDumper(yaml.SafeDumper):
        # Spell out every entry: anchors would make an edit of one quantizer change all its aliases.
        def ignore_aliases(self, data):
            return True

    # Quantizer states can record runtime dtypes and shapes; store dtypes by name, shapes as lists,
    # and config objects (e.g. a disabled RotateConfig) as plain mappings.
    _RecipeDumper.add_representer(
        torch.dtype, lambda dumper, dtype: dumper.represent_str(str(dtype))
    )
    _RecipeDumper.add_representer(
        torch.Size, lambda dumper, size: dumper.represent_list(list(size))
    )
    _RecipeDumper.add_multi_representer(
        BaseModel, lambda dumper, config: dumper.represent_dict(config.model_dump(mode="json"))
    )

    state_path = path / MODELOPT_QUANTIZER_STATE_FILE
    recipe_path = path / MODELOPT_QUANTIZER_RECIPE_FILE
    if not quantizer_recipe:
        state_path.unlink(missing_ok=True)
        recipe_path.unlink(missing_ok=True)
        return
    path.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            name: tensor
            for name, tensor in quant_tensors.items()
            if name.rpartition(".")[0] in quantizer_recipe
        },
        state_path,
    )
    recipe = {
        "format": MODELOPT_QUANTIZER_RECIPE_FORMAT,
        "version": MODELOPT_QUANTIZER_RECIPE_VERSION,
        "quantizers": quantizer_recipe,
    }
    recipe_path.write_text(yaml.dump(recipe, Dumper=_RecipeDumper, sort_keys=False))


class AutoBridge(Generic[MegatronModelT]):
    """
    Automatically select and instantiate the appropriate bridge for a model.

    This unified bridge class combines automatic model detection with full bridge
    functionality for converting models between HuggingFace and Megatron formats.
    It handles supported causal and masked language models as well as task-specific
    heads such as token classification. It manages weight mapping, tensor-parallel
    distribution, and configuration translation.

    The bridge supports both directions of conversion:
    - HuggingFace → Megatron: For training or inference with Megatron
    - Megatron → HuggingFace: For saving trained models in HF format

    Args:
        hf_pretrained: A PreTrainedCausalLM, PreTrainedMaskedLM, or
            PreTrainedTokenClassification instance with loaded model, or a
            PretrainedConfig for configuration-only operations.

    Example:
        >>> # Load and convert a model to Megatron format
        >>> bridge = AutoBridge.from_hf_pretrained("meta-llama/Meta-Llama-3-8B")
        >>> provider = bridge.to_megatron_provider()
        >>> megatron_model = provider.provide_distributed_model(wrap_with_ddp=False)

        >>> # Export a Megatron model back to HuggingFace format
        >>> bridge.save_hf_pretrained(megatron_model, "./exported_model")

        >>> # Convert weights with custom settings
        >>> for name, weight in bridge.export_hf_weights(
        ...     megatron_model,
        ...     cpu=True
        ... ):
        ...     print(f"Exported {name}: {weight.shape}")

        >>> # Check if a model is supported before loading
        >>> if AutoBridge.can_handle("microsoft/phi-2"):
        ...     bridge = AutoBridge.from_hf_pretrained("microsoft/phi-2")

    Note:
        The bridge automatically detects the model architecture and applies
        the appropriate weight mappings. Custom architectures require implementing
        a MegatronModelBridge subclass.
    """

    def __init__(
        self,
        hf_pretrained: (
            PreTrainedCausalLM
            | PreTrainedMaskedLM
            | PreTrainedTokenClassification
            | PretrainedConfig
        ),
    ):
        if not isinstance(hf_pretrained, (*_PRETRAINED_WRAPPER_TYPES, PretrainedConfig)):
            raise ValueError(
                "hf_pretrained must be a PreTrainedCausalLM, PreTrainedMaskedLM, "
                "PreTrainedTokenClassification, or PretrainedConfig instance"
            )
        self.hf_pretrained: (
            PreTrainedCausalLM
            | PreTrainedMaskedLM
            | PreTrainedTokenClassification
            | PretrainedConfig
        ) = hf_pretrained
        # Data type for exporting weights
        self.export_weight_dtype: Literal["bf16", "fp16", "fp8"] = "bf16"
        self.text_only = getattr(hf_pretrained, "_text_only", False) is True
        self.hf_model_id: Optional[str] = None
        init_kwargs = getattr(hf_pretrained, "init_kwargs", {})
        revision = init_kwargs.get("revision") if isinstance(init_kwargs, dict) else None
        self.hf_model_revision: str | None = revision if isinstance(revision, str) else None
        trust_remote_code = getattr(hf_pretrained, "trust_remote_code", False)
        self.trust_remote_code = trust_remote_code if isinstance(trust_remote_code, bool) else False

    @classmethod
    def list_supported_models(cls) -> list[str]:
        """
        List all model architectures currently supported by the bridge system.

        Returns:
            List of supported HuggingFace model architecture names
        """
        # Get all registered implementations from the dispatch system
        supported = []

        # Access the dispatch registry to find all registered types

        if hasattr(model_bridge.get_model_bridge, "_exact_types"):
            for arch_type in model_bridge.get_model_bridge._exact_types.keys():
                # Support both type and string registrations
                if isinstance(arch_type, str):
                    supported.append(arch_type)
                elif hasattr(arch_type, "__name__"):
                    supported.append(arch_type.__name__)

        return sorted(supported)

    @classmethod
    def supports(cls, config: Any) -> bool:
        """
        Check if this bridge supports the given model configuration.

        A model is supported if it has at least one architecture ending with one of the
        suffixes listed in SUPPORTED_HF_ARCHITECTURES.

        Args:
            config: HuggingFace model config object

        Returns:
            True if this bridge can handle the model, False otherwise
        """
        architectures = getattr(config, "architectures", [])
        if not architectures:
            return False
        return any(arch.endswith(SUPPORTED_HF_ARCHITECTURES) for arch in architectures)

    @classmethod
    def from_auto_config(
        cls, megatron_path: str, hf_model_id: str, trust_remote_code: bool = False
    ) -> "AutoBridge":
        """
        Create a config-only AutoBridge by synthesizing an HF config from a Megatron checkpoint.

        This method creates a bridge instace from a Megatron checkpoint and reference hf_model_id,
        without loading any weights. This enables exporting of:
        - Custom small models of popular architectures
        - Models pruned from a larger teacher model

        Args:
            megatron_path: Directory path where the Megatron checkpoint is stored
            hf_model_id: HuggingFace model ID or path to model directory
                Examples: "meta-llama/Meta-Llama-3-8B", "./my_model"
            trust_remote_code: Whether to trust remote code when loading the reference config and
                the checkpoint's run config. Defaults to False for security. Set to True only for
                models that require custom modeling code from the repository.

        Returns:
            AutoBridge: Bridge instance configured for the architecture

        Raises:
            FileNotFoundError: If run_config.yaml is not found in the Megatron path
        """

        from transformers import AutoConfig

        from megatron.bridge.models.conversion.checkpoint import (
            get_hf_source_from_model_config,
            load_model_config,
        )
        from megatron.bridge.models.conversion.utils import conform_config_to_reference
        from megatron.bridge.utils.instantiate_utils import register_allowed_target_prefix

        checkpoint_path = Path(megatron_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Megatron checkpoint not found: {checkpoint_path}")

        # Look for configuration files to determine the model type
        run_config = checkpoint_path / "run_config.yaml"
        if not run_config.exists():
            iter_dirs = [
                d for d in checkpoint_path.iterdir() if d.is_dir() and d.name.startswith("iter_")
            ]
            if iter_dirs:
                latest_iter = max(iter_dirs, key=lambda d: int(d.name.replace("iter_", "")))
                run_config = latest_iter / "run_config.yaml"

        if not run_config.exists():
            raise FileNotFoundError(
                f"Could not find run_config.yaml in {checkpoint_path}. Ensure this is a valid Megatron checkpoint."
            )

        # 1. Load config from both sides
        if trust_remote_code:
            logger.warning(
                "Loading a model with trust_remote_code=True allows arbitrary code execution "
                "from the model repository. Only use this with models you trust."
            )
            # The run config can restore config classes from the repository code, such as a VLM's vision config.
            register_allowed_target_prefix("transformers_modules.")
        megatron_cfg, _ = load_model_config(str(run_config.parent))
        text_reference = None
        if getattr(megatron_cfg, "hf_model_text_only", False) is True:
            # A replacement reference repository need not contain the source revision.
            source_id, source_revision = get_hf_source_from_model_config(megatron_cfg)
            revision = source_revision if str(hf_model_id) == str(source_id) else None
            text_reference = cls.from_hf_pretrained(
                hf_model_id, text_only=True, trust_remote_code=trust_remote_code, revision=revision
            )
            hf_cfg = text_reference.hf_pretrained.config
        else:
            hf_cfg = AutoConfig.from_pretrained(hf_model_id, trust_remote_code=trust_remote_code)
        # 2. Translate Megatron config -> HF, conforming to reference config
        bridge = cls.from_hf_config(hf_cfg)
        megatron_hf_cfg_dict = bridge._model_bridge.megatron_to_hf_config(megatron_cfg)
        megatron_hf_cfg_dict = conform_config_to_reference(megatron_hf_cfg_dict, hf_cfg.to_dict())
        megatron_hf_cfg_dict = _drop_readonly_config_properties(megatron_hf_cfg_dict, type(hf_cfg))
        # 3. Build final bridge from the synthesized config
        synthesized_config = type(hf_cfg)(**megatron_hf_cfg_dict)
        synthesized_config.name_or_path = hf_model_id
        bridge = cls.from_hf_config(synthesized_config)
        if text_reference is not None:
            bridge.text_only = text_reference.text_only
            bridge.hf_model_revision = text_reference.hf_model_revision
        bridge.hf_model_id = hf_model_id
        bridge.trust_remote_code = trust_remote_code

        return bridge

    @classmethod
    def from_hf_config(cls, config: PretrainedConfig) -> "AutoBridge":
        """
        Create an AutoBridge from a HuggingFace configuration.

        This method creates a bridge instance from just a model configuration,
        without loading any weights. This is useful for:
        - Creating Megatron models with random initialization
        - Working with model architectures without downloading weights
        - Testing and development scenarios

        Args:
            config: HuggingFace PretrainedConfig instance containing model
                architecture information

        Returns:
            AutoBridge: Bridge instance configured for the architecture

        Raises:
            ValueError: If the configuration does not name a supported model architecture.

        Example:
            >>> from transformers import AutoConfig
            >>>
            >>> # Load just the configuration
            >>> config = AutoConfig.from_pretrained("meta-llama/Meta-Llama-3-8B")
            >>>
            >>> # Create bridge from config (no weights)
            >>> bridge = AutoBridge.from_hf_config(config)
            >>>
            >>> # Create Megatron model with random initialization
            >>> provider = bridge.to_megatron_provider(load_weights=False)
            >>> model = provider.provide_distributed_model(wrap_with_ddp=False)

            >>> # Or use for architecture exploration
            >>> transformer_config = bridge.transformer_config
            >>> print(f"Hidden size: {transformer_config.hidden_size}")
            >>> print(f"Num layers: {transformer_config.num_layers}")

        See Also:
            from_hf_pretrained: Create bridge with loaded weights
            transformer_config: Access the Megatron TransformerConfig
        """
        cls._validate_config(config)
        return cls(config)

    @classmethod
    def from_hf_pretrained(
        cls, path: Union[str, Path], *, text_only: bool = False, **kwargs
    ) -> "AutoBridge":
        """
        Load an AutoBridge from a pretrained model, automatically detecting the model type.

        This method loads a model from HuggingFace Hub or a local directory and
        creates a bridge instance ready for conversion operations. The model
        architecture is validated to ensure compatibility.

        Args:
            path: HuggingFace model ID or path to model directory
                Examples: "meta-llama/Meta-Llama-3-8B", "./my_model"
            text_only: Select only the language checkpoint of an explicitly
                supported multimodal family. Defaults to the full model.
                Unsupported families raise ValueError.
            **kwargs: Additional arguments passed to HuggingFace from_hf_pretrained
                Common options include:
                - torch_dtype: Model precision (torch.float16, torch.bfloat16)
                - device_map: Device placement strategy ("auto", "cuda:0", etc.)
                - trust_remote_code: Allow custom model code execution
                - attn_implementation: Attention implementation ("flash_attention_2", etc.)

        Returns:
            AutoBridge: Bridge instance with loaded model

        Raises:
            ValueError: If the model architecture is not supported

        Example:
            >>> # Basic loading
            >>> bridge = AutoBridge.from_hf_pretrained("gpt2")

            >>> # Load with specific settings
            >>> bridge = AutoBridge.from_hf_pretrained(
            ...     "meta-llama/Meta-Llama-3-8B",
            ...     torch_dtype=torch.float16,
            ...     device_map="auto"
            ... )

            >>> # Works with local paths too
            >>> bridge = AutoBridge.from_hf_pretrained("/path/to/model")
        """

        # First load just the config to check architecture support
        # Use thread-safe config loading to prevent race conditions
        config_kwargs = dict(kwargs)
        trust_remote_code = bool(config_kwargs.pop("trust_remote_code", False))
        if trust_remote_code:
            logger.warning(
                "Loading a model with trust_remote_code=True allows arbitrary code execution "
                "from the model repository. Only use this with models you trust."
            )
        config = safe_load_config_with_retry(
            path, trust_remote_code=trust_remote_code, **config_kwargs
        )

        cls._validate_config(config, str(path))
        _validate_hf_model_runtime_support(config, trust_remote_code=trust_remote_code)

        # Transformers 5.0+ changed `rope_scaling` to a property whose setter
        # does `self.rope_parameters = value`, replacing the entire dict and
        # dropping any fields (e.g. `rope_theta`) that were set during initial
        # construction.  When a `rope_scaling` override is passed as a kwarg,
        # `PretrainedConfig.from_dict` applies it via `setattr` *after* the
        # initial construction, so those fields are silently lost and Megatron
        # falls back to defaults (e.g. `rotary_base=10000`).  Pre-populate the
        # override dict with all base-config rope fields so the setter
        # preserves them.
        if "rope_scaling" in kwargs and isinstance(kwargs["rope_scaling"], dict):
            base_rope = getattr(config, "rope_scaling", None)
            if isinstance(base_rope, dict):
                for key, value in base_rope.items():
                    if key not in kwargs["rope_scaling"]:
                        kwargs["rope_scaling"][key] = value

        wrapper_cls = _resolve_pretrained_wrapper_cls(config)
        try:
            hf_pretrained = wrapper_cls.from_pretrained(path, **kwargs)
            # Reuse the config that was already loaded and validated above.
            # Besides avoiding a second Hub request, this guarantees that the
            # warning and model wrapper observe the same immutable revision.
            hf_pretrained.config = config
            bridge = cls(hf_pretrained)
            if text_only:
                hf_pretrained = bridge._model_bridge.text_only_pretrained(hf_pretrained)
                cls._validate_config(hf_pretrained.config, str(path))
                bridge = cls(hf_pretrained)
            return bridge
        except Exception as e:
            raise ValueError(f"Failed to load model with AutoBridge: {e}") from e

    @classmethod
    def can_handle(cls, path: Union[str, Path], trust_remote_code: bool = False) -> bool:
        """
        Check if the bridge can handle the model at the given path.

        This method allows you to verify model compatibility before attempting
        to load it, which can be useful for validation or UI feedback.

        Args:
            path: Path to model directory or HuggingFace model ID
                Examples: "meta-llama/Meta-Llama-3-8B", "/models/my_model"
            trust_remote_code: Whether to trust remote code when loading config.
                Set to True for models that use custom modeling code.

        Returns:
            bool: True if the bridge supports the model, False otherwise

        Example:
            >>> # Check if a model is supported
            >>> if AutoBridge.can_handle("meta-llama/Meta-Llama-3-8B"):
            ...     print("Model is supported!")
            ... else:
            ...     print("Model requires a custom bridge implementation")
        """
        try:
            config = safe_load_config_with_retry(path, trust_remote_code=trust_remote_code)
            # Reuse the same registry check as from_hf_pretrained (allowlisted
            # architecture suffix *and* a registered MegatronModelBridge) so this
            # preflight can't report True for a model that would then fail to load.
            cls._validate_config(config, str(path))
            _validate_hf_model_runtime_support(config, trust_remote_code=trust_remote_code)
            return True
        except Exception:
            return False

    def load_hf_weights(
        self,
        model: list[MegatronModelT],
        hf_path: str | Path | None = None,
        allowed_mismatched_params: list[str] | None = None,
    ) -> None:
        """
        Load HuggingFace weights into a Megatron model.

        This method handles the conversion and distribution of weights from
        HuggingFace format to Megatron's distributed format, including proper
        tensor parallel and pipeline parallel distribution.

        Args:
            model: List of Megatron model instances (one per virtual pipeline stage)
            hf_path: Optional path to load weights from. If None, uses weights
                from the bridge's hf_pretrained instance
            allowed_mismatched_params: Optional list of parameter names or patterns
                to allow mismatch (skip instead of raise error).

        Returns:
            The input model with loaded weights

        Raises:
            ValueError: If hf_path is None and bridge was created without weights

        Example:
            >>> # Load weights from bridge's pretrained model
            >>> bridge = AutoBridge.from_hf_pretrained("gpt2")
            >>> megatron_model = create_megatron_model()  # Your model creation
            >>> bridge.load_hf_weights(megatron_model)

            >>> # Load weights from a different checkpoint
            >>> bridge.load_hf_weights(megatron_model, "./finetuned_model")

            >>> # Load weights with allowed mismatched parameters
            >>> bridge.load_hf_weights(
            ...     megatron_model,
            ...     allowed_mismatched_params=["*.bias", "decoder.layers.0.*"]
            ... )
        """
        if hf_path is None:
            if not isinstance(self.hf_pretrained, _PRETRAINED_WRAPPER_TYPES):
                raise ValueError(
                    "hf_path is required when hf_pretrained is not a PreTrainedCausalLM, "
                    "PreTrainedMaskedLM, or PreTrainedTokenClassification instance"
                )
            pre_trained = self.hf_pretrained
        else:
            # Preserve trust_remote_code setting from the original bridge instance
            trust_remote_code = getattr(self.hf_pretrained, "trust_remote_code", False)
            if self.text_only:
                pre_trained = self._text_only_pretrained_from_path(hf_path)
            else:
                wrapper_cls = self._pretrained_wrapper_cls
                pre_trained = wrapper_cls.from_pretrained(
                    hf_path, trust_remote_code=trust_remote_code
                )
        bridge = self._model_bridge
        bridge.load_weights_hf_to_megatron(
            pre_trained, model, allowed_mismatched_params=allowed_mismatched_params
        )
        # Get unquantized_state_dict from the bridge instance that was used for optimizer reload
        self.unquantized_state_dict = getattr(bridge, "unquantized_state_dict", None)
        return model

    def get_export_fp8_tasks(
        self, model: MegatronModelT | list[MegatronModelT]
    ) -> list[WeightConversionTask | None]:
        """Build physical FP8 data and scale export tasks."""
        if not isinstance(model, list):
            model = [model]
        return self._model_bridge.build_export_fp8_tasks(self.hf_pretrained, model)

    def get_export_mxfp8_tasks(
        self, model: MegatronModelT | list[MegatronModelT]
    ) -> list[WeightConversionTask]:
        """Build native MXFP8 export tasks, including singular grouped-expert weights."""
        if not isinstance(model, list):
            model = [model]
        return self._model_bridge.build_export_mxfp8_tasks(self.hf_pretrained, model)

    def iter_local_mxfp8_params(
        self, tasks: Iterable[WeightConversionTask]
    ) -> Iterable["LocalMXFP8Param"]:
        """Yield local native MXFP8 projections through the public bridge API."""
        return self._model_bridge.iter_local_mxfp8_params(tasks)

    def iter_local_hf_params(
        self, tasks: Iterable[WeightConversionTask]
    ) -> Iterable["LocalHFParam"]:
        """Yield local unquantized BF16 parameters as canonical HF views.

        Args:
            tasks: Reusable tasks from :meth:`get_conversion_tasks` in the
                deterministic order they should be exported.

        Returns:
            An iterator over live local parameter views and their shard metadata.

        Raises:
            ValueError: If a locally owned task is quantized, is not BF16, or
                cannot be represented without Bridge conversion collectives.
        """
        return self._model_bridge.iter_local_hf_params(tasks)

    def export_hf_weights(
        self,
        model: list[MegatronModelT],
        cpu: bool = False,
        show_progress: bool = True,
        conversion_tasks: Optional[List[WeightConversionTask]] = None,
        weight_dtype: Optional[torch.dtype] = None,
        with_megatron_names: bool = False,
        current_pp_stage_only: bool = False,
    ) -> Iterable["HFWeightTuple"]:
        """
        Export Megatron model weights to HuggingFace format.

        This method yields weight tensors in HuggingFace format, handling the
        gathering of distributed tensors and format conversion. It's useful for
        streaming weight export or custom processing. By default all ranks get full tensors;
        set ``current_pp_stage_only=True`` to yield only the current pipeline stage's tensors.

        Args:
            model: Megatron model instance or list of instances
            cpu: Whether to move tensors to CPU before yielding
            show_progress: Display progress bar during export
            conversion_tasks (Optional[List[WeightConversionTask]]): Pre-built conversion tasks.
                If not provided, tasks will be built automatically from the models.
                *Please note that this is an advanced feature and should be used with caution.
                The tasks needs to be built with the `get_conversion_tasks` method first and
                carefully adjust based on your needs.*
            weight_dtype: Plain export dtype; skips quantized *.scale companions when set.
            with_megatron_names: Yield ``HFSourcedWeightTuple`` (param_name, weight,
                megatron_param_names) instead of the two-field tuple, so each exported weight can
                be traced back to the Megatron parameter(s) it came from (e.g. by RL weight-sync
                loops): one name for a directly converted weight, one per contributing expert for
                a grouped-expert export, and none for HF-only passthrough tensors. Bridges that
                override ``stream_weights_megatron_to_hf`` must accept the flag; otherwise a
                ``TypeError`` is raised before any weight is streamed.
            current_pp_stage_only: Export only parameters owned by the current pipeline stage and
                suppress cross-stage PP broadcasts. TP/EP gathering and HF layout conversion remain
                enabled. Defaults to False. Bridges that override the streamer must accept this
                flag when it is enabled.

        Yields:
            HFWeightTuple: Named tuples of (param_name, weight_tensor), or HFSourcedWeightTuple
            when ``with_megatron_names`` is set.

        Example:
            >>> # Export and process weights
            >>> for name, weight in bridge.export_hf_weights(model):
            ...     print(f"{name}: {weight.shape}")

            >>> # Export with specific settings
            >>> weights = list(bridge.export_hf_weights(
            ...     model,
            ...     cpu=True
            ... ))
        """
        # Build conversion tasks based on export_weight_dtype configuration
        if conversion_tasks is None and self.export_weight_dtype == "fp8":
            if not isinstance(model, list):
                model = [model]
            self._validate_fp8_export_config(model)
            # Use FP8 export tasks for blockwise FP8 weights
            conversion_tasks = self._model_bridge.build_export_fp8_tasks(self.hf_pretrained, model)

        bridge = self._model_bridge
        if with_megatron_names:
            _check_with_megatron_names_support(bridge.stream_weights_megatron_to_hf, bridge)
        if current_pp_stage_only:
            _check_current_pp_stage_only_support(bridge.stream_weights_megatron_to_hf, bridge)
        return bridge.stream_weights_megatron_to_hf(
            model,
            self.hf_pretrained,
            cpu=cpu,
            show_progress=show_progress,
            conversion_tasks=conversion_tasks,
            weight_dtype=weight_dtype,
            # Only forward the flag when set so bridges with a custom streamer keep working by default.
            **({"with_megatron_names": True} if with_megatron_names else {}),
            **({"current_pp_stage_only": True} if current_pp_stage_only else {}),
        )

    def build_hf_modelopt_export_plan(
        self,
        model: MegatronModelT | list[MegatronModelT],
        conversion_tasks: Optional[List[WeightConversionTask]] = None,
    ) -> "ModelOptExportPlan":
        """Prepare canonical real-quant tensors and configuration for streaming export.

        Args:
            model: Megatron model instance or list of instances.
            conversion_tasks: Pre-built conversion tasks. If not provided, tasks will be built
                automatically from the models.
        """
        from megatron.bridge.models.conversion.modelopt_utils import build_modelopt_export_plan

        if not isinstance(model, list):
            model = [model]
        bridge = self._model_bridge
        if conversion_tasks is None:
            from modelopt.torch.quantization.nn import AnyQuantizer

            # ModelOpt reads quantizer state from each owning weight, not separate HF mappings.
            conversion_tasks = bridge.build_conversion_tasks(
                self.hf_pretrained, model, exclude_module_types=AnyQuantizer
            )
        tied_weight_aliases = {}
        unwrapped_model = model_bridge.unwrap_model(model)[0]
        if bridge._share_embeddings_and_output_weights(unwrapped_model):
            registry = bridge.mapping_registry()
            for task in conversion_tasks:
                if task is None:
                    continue
                alias = bridge._get_tied_output_hf_name(task, registry)
                if alias is not None:
                    tied_weight_aliases[alias] = task.mapping.hf_param
        return build_modelopt_export_plan(
            conversion_tasks, model=model, tied_weight_aliases=tied_weight_aliases
        )

    def export_hf_weight_groups_modelopt(
        self,
        model: MegatronModelT | list[MegatronModelT],
        *,
        cpu: bool = False,
        show_progress: bool = True,
        export_plan: Optional["ModelOptExportPlan"] = None,
    ) -> Iterable[tuple["HFWeightTuple", ...]]:
        """Export materialized ModelOpt weight groups.

        Args:
            model: Megatron model instance or virtual-pipeline model list.
            cpu: Move each exported tensor to CPU before yielding it.
            show_progress: Display conversion progress.
            export_plan: Reusable plan built by :meth:`build_hf_modelopt_export_plan`.

        Yields:
            One tuple per conversion task. Empty tuples preserve tasks that emit no
            tensors on a given topology. Bridges may append groups for checkpoint-only
            tensors after the conversion-task groups.

        Note:
            This is a distributed collective stream. Every rank must consume groups
            in identical order, including empty groups and the terminal
            ``StopIteration``. A group is collective-complete before it is yielded;
            callers must synchronize rank-local consumers before requesting the next
            group.

        Raises:
            NotImplementedError: If custom Hugging Face export overrides are requested.
        """
        if type(self).export_hf_weights is not AutoBridge.export_hf_weights:
            raise NotImplementedError(
                "Grouped ModelOpt export does not support AutoBridge subclasses "
                "with custom HuggingFace export behavior"
            )
        bridge = self._model_bridge
        if (
            type(bridge).stream_weights_megatron_to_hf
            is not MegatronModelBridge.stream_weights_megatron_to_hf
        ):
            raise NotImplementedError(
                "Grouped ModelOpt export does not support model bridges with custom HuggingFace export behavior"
            )
        if not isinstance(model, list):
            model = [model]
        if export_plan is None:
            export_plan = self.build_hf_modelopt_export_plan(model)
        from megatron.bridge.models.conversion.modelopt_utils import prepare_modelopt_export_tasks

        yield from bridge._stream_weight_groups_megatron_to_hf(
            model,
            self.hf_pretrained,
            cpu=cpu,
            show_progress=show_progress,
            conversion_tasks=prepare_modelopt_export_tasks(export_plan),
        )

    def export_hf_weights_modelopt(
        self,
        model: MegatronModelT | list[MegatronModelT],
        *,
        cpu: bool = False,
        show_progress: bool = True,
        export_plan: Optional["ModelOptExportPlan"] = None,
    ) -> Iterable["HFWeightTuple"]:
        """Export canonical ModelOpt deployment tensors from a prepared plan."""
        if not isinstance(model, list):
            model = [model]
        if export_plan is None:
            export_plan = self.build_hf_modelopt_export_plan(model)
        from megatron.bridge.models.conversion.modelopt_utils import prepare_modelopt_export_tasks

        yield from self.export_hf_weights(
            model,
            cpu=cpu,
            show_progress=show_progress,
            conversion_tasks=prepare_modelopt_export_tasks(export_plan),
        )

    def export_hf_weights_quant(
        self,
        model: list[MegatronModelT],
        quantization_checker: Callable[[str], bool],
        quant_fn: Callable[..., Tuple[torch.Tensor, torch.Tensor]],
        quant_block_size: Optional[Tuple[int, int]] = None,
        cpu: bool = False,
        show_progress: bool = True,
        conversion_tasks: Optional[List[WeightConversionTask]] = None,
    ) -> Iterable["HFWeightTuple"]:
        """
        Export Megatron model weights to HuggingFace format with quantization.
        """
        dispatch_instance = (self._causal_lm_architecture, self._get_model_instance(model))
        return model_bridge.stream_weights_megatron_to_hf_quant(
            dispatch_instance,
            model,
            self.hf_pretrained,
            quantization_checker,
            quant_fn,
            quant_block_size=quant_block_size,
            cpu=cpu,
            show_progress=show_progress,
            conversion_tasks=conversion_tasks,
        )

    def save_hf_pretrained(
        self,
        model: list[MegatronModelT],
        path: str | Path,
        show_progress: bool = True,
        source_path: Optional[Union[str, Path]] = None,
        strict: bool = True,
        distributed_save: bool = False,
        save_every_n_ranks: int = 1,
        weight_dtype: Optional[torch.dtype] = None,
    ) -> None:
        """
        Save a Megatron model in HuggingFace format.

        This method exports the complete model including configuration, tokenizer,
        and weights to a directory that can be loaded with HuggingFace's
        from_pretrained methods.

        If the original model was loaded with trust_remote_code=True, any custom
        modeling files (e.g., modeling_*.py, configuration_*.py) will be preserved
        to ensure the saved model can be loaded properly.

        Config-only bridges are supported when created via the auto-config
        flow used by ``examples/bridge/conversion/convert.sh``.

        Args:
            model: Megatron model instance or list of instances
            path: Directory path to save the model
            show_progress: Display progress bar during weight export
            source_path: Path to the directory containing custom modeling files to be preserved.
                This is useful when converting from Megatron checkpoints where the original
                HuggingFace model with custom modeling files needs to be referenced. If not specified,
                the path will be automatically determined from the HuggingFace configuration.
            strict: Whether to perform strict validation during weight export
            distributed_save: Whether to enable distributed saving mode where each rank saves
                part of weights independently. When False (default), only rank 0 performs
                the save operation after gathering weights from all ranks.
            save_every_n_ranks: Interval for saving weights across ranks in distributed mode.
                For example, if set to 2, only ranks 0, 2, 4, ... will save weights.
                This is useful for reducing I/O pressure when dealing with large-scale distributed
                training. Only effective when distributed_save=True. Default is 1 (all ranks save).
            weight_dtype: Plain export dtype; skips quantized *.scale companions when set.

        Example:
            >>> # Save model after training
            >>> bridge.save_hf_pretrained(megatron_model, "./my_finetuned_model")

            >>> # Load the saved model with HuggingFace
            >>> from transformers import AutoModelForCausalLM
            >>> hf_model = AutoModelForCausalLM.from_pretrained("./my_finetuned_model")

        Note:
            This method is collective - all ranks must call it. Only rank 0
            saves the configuration files, while weight saving is coordinated
            across all ranks.
        """
        model_bridge = self._save_hf_artifacts(path, source_path)

        self.save_hf_weights(
            model,
            path,
            show_progress,
            strict,
            distributed_save=distributed_save,
            save_every_n_ranks=save_every_n_ranks,
            weight_dtype=weight_dtype,
        )
        if model_bridge is not None and (not dist.is_initialized() or dist.get_rank() == 0):
            weight_postprocessor = getattr(
                type(model_bridge), "postprocess_hf_export_weights", None
            )
            if weight_postprocessor is not None:
                weight_postprocessor(model_bridge, Path(path))

    def save_hf_pretrained_modelopt(
        self,
        model: list[MegatronModelT],
        path: str | Path,
        *,
        export_plan: Optional["ModelOptExportPlan"] = None,
        show_progress: bool = True,
        source_path: Optional[Union[str, Path]] = None,
        max_shard_size: int = 5 * 1024**3,
    ) -> None:
        """Save a ModelOpt-quantized Megatron model as a quantized HuggingFace checkpoint.

        Quantized weights are packed with their scales (e.g. NVFP4 ``weight``, ``weight_scale``,
        ``weight_scale_2`` and ``input_scale``), and ``config.json`` carries the
        ``quantization_config`` that vLLM, SGLang and TensorRT-LLM read. Any quantization config
        of the source checkpoint is replaced. Unlike :meth:`save_hf_pretrained`, which exports
        fake-quantized BF16 weights with quantizer sidecars, this writes the deployment format.

        Args:
            model: Megatron model instance or list of instances.
            path: Directory to save the checkpoint to.
            export_plan: Plan built by :meth:`build_hf_modelopt_export_plan`; built when omitted.
            show_progress: Display progress bar during weight export.
            source_path: Directory with custom modeling files to preserve, as in
                :meth:`save_hf_pretrained`.
            max_shard_size: Maximum size in bytes of each safetensors shard.

        Raises:
            NotImplementedError: If the bridge post-processes exported weights.
            RuntimeError: If an enabled quantizer has no exported tensors (e.g. KV-cache quantizers).
            ValueError: If projections that deployments fuse (e.g. ``q_proj`` / ``k_proj`` /
                ``v_proj``) carry different per-tensor scales.

        Note:
            This method is collective - all ranks must call it. Rank 0 writes the files.
        """
        # TODO: Consider unifying with save_hf_pretrained, choosing the format from the quantizer
        # state: enabled weight quantizers -> this quantized checkpoint, folded weights -> BF16 with
        # quantizer sidecars.
        from megatron.bridge.models.conversion.modelopt_utils import (
            _raise_distributed_errors,
            _world_group,
            check_exportable_quantizers,
            check_fused_sibling_scales,
            save_safetensors_shards,
        )

        if not isinstance(model, list):
            model = [model]
        if (
            type(self._model_bridge).postprocess_hf_export_weights
            is not MegatronModelBridge.postprocess_hf_export_weights
        ):
            raise NotImplementedError(
                f"{type(self._model_bridge).__name__} post-processes exported weights, which a quantized export cannot"
            )
        check_exportable_quantizers(model)
        if export_plan is None:
            export_plan = self.build_hf_modelopt_export_plan(model)
        rank = dist.get_rank() if dist.is_initialized() else 0
        output_path = Path(path).resolve()
        # Rank 0 saves the artifacts; fail the other ranks too before the export collectives.
        error = None
        try:
            self._save_hf_artifacts(path, source_path)
        except Exception as exception:
            error = f"{type(exception).__name__}: {exception}"
        _raise_distributed_errors(
            error, _world_group(), f"ModelOpt HF export to {output_path} failed"
        )
        tensors = self.export_hf_weights_modelopt(
            model, cpu=True, show_progress=show_progress, export_plan=export_plan
        )
        scales = {}

        def _keep_scales(tensors):
            for name, tensor in tensors:
                if name.endswith(("weight_scale_2", "input_scale")):
                    scales[name] = tensor
                yield name, tensor

        try:
            if rank == 0:
                previous_weight_files = _existing_safetensors_files(output_path)
                saved = save_safetensors_shards(
                    _keep_scales(tensors), output_path, max_shard_size=max_shard_size
                )
                check_fused_sibling_scales(scales)
                for filename in previous_weight_files - saved:
                    (output_path / filename).unlink(missing_ok=True)
                # Quantizer sidecars or a separate config would describe other weights.
                for stale in (
                    "hf_quant_config.json",
                    MODELOPT_QUANTIZER_STATE_FILE,
                    MODELOPT_QUANTIZER_RECIPE_FILE,
                ):
                    (output_path / stale).unlink(missing_ok=True)
                config_file = output_path / "config.json"
                config = json.loads(config_file.read_text())
                config["quantization_config"] = export_plan.quantization_config
                config_file.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
            else:
                for _ in tensors:
                    pass
        except Exception as exception:  # Drain so the other ranks finish the export collectives.
            error = f"{type(exception).__name__}: {exception}"
            for _ in tensors:
                pass

        _raise_distributed_errors(
            error, _world_group(), f"ModelOpt HF export to {output_path} failed"
        )

    def _save_hf_artifacts(self, path: str | Path, source_path: Optional[Union[str, Path]]) -> Any:
        """Check the bridge supports HF export and save the config and tokenizer artifacts on rank 0.

        Returns:
            The resolved model bridge, or ``None`` for a config-only save without one.
        """
        # Some bridges (e.g. the language-model-only bridge of a multimodal model) cannot
        # produce a valid standalone Hugging Face checkpoint; gate the export on the
        # resolved bridge's capability flag. Resolving the bridge needs a concrete,
        # registered architecture: a config-only save from a bare ``PretrainedConfig`` has
        # none, so treat an unresolvable bridge as "nothing to gate" and fall through to the
        # normal config-only path instead of failing here.
        try:
            model_bridge = self._model_bridge
        except (ValueError, NotImplementedError):
            model_bridge = None
        if model_bridge is not None and not model_bridge.SUPPORTS_HF_PRETRAINED_EXPORT:
            raise NotImplementedError(
                f"{type(model_bridge).__name__} does not support standalone Hugging Face checkpoint export."
            )
        if not isinstance(self.hf_pretrained, (*_PRETRAINED_WRAPPER_TYPES, PretrainedConfig)):
            raise ValueError(
                "save_hf_pretrained requires a pretrained HuggingFace model or config."
            )
        is_config_only = isinstance(self.hf_pretrained, PretrainedConfig)

        def _save_artifacts():
            if is_config_only:
                artifact_source_path = self.hf_model_id or source_path
                if artifact_source_path is not None:
                    model_bridge_instance = self._model_bridge
                    artifact_kwargs = {}
                    if hasattr(model_bridge_instance, "get_hf_tokenizer_kwargs"):
                        artifact_kwargs.update(
                            model_bridge_instance.get_hf_tokenizer_kwargs() or {}
                        )
                    artifact_kwargs["trust_remote_code"] = self.trust_remote_code

                    # This wrapper loads artifacts lazily; model weights are never materialized here.
                    if self.text_only:
                        artifact_source = (
                            type(self)
                            .from_hf_pretrained(
                                artifact_source_path,
                                text_only=True,
                                revision=self.hf_model_revision,
                                **artifact_kwargs,
                            )
                            .hf_pretrained
                        )
                    else:
                        artifact_cls = _resolve_pretrained_wrapper_cls(self.hf_pretrained)
                        # Preserve specialized causal-family processor loaders without
                        # changing the artifact contract for non-causal model heads.
                        if not issubclass(artifact_cls, PreTrainedCausalLM):
                            artifact_cls = PreTrainedCausalLM
                        artifact_source = artifact_cls.from_pretrained(
                            artifact_source_path, **artifact_kwargs
                        )
                    artifact_source.config = self.hf_pretrained
                    additional_files = (
                        getattr(model_bridge_instance, "ADDITIONAL_FILE_PATTERNS", None) or None
                    )
                    artifact_source.save_artifacts(
                        path, original_source_path=source_path, additional_files=additional_files
                    )
                else:
                    import json

                    # A bridge built directly from a config has no reference artifacts to preserve.
                    Path(path).mkdir(parents=True, exist_ok=True)
                    config_dict = self.hf_pretrained.to_dict()
                    if not self.trust_remote_code:
                        config_dict.pop("auto_map", None)
                    with open(Path(path) / "config.json", "w") as _f:
                        json.dump(config_dict, _f, indent=2, sort_keys=True, allow_nan=True)

            else:
                # Get bridge-level ADDITIONAL_FILE_PATTERNS if configured
                additional_files = (
                    getattr(self._model_bridge, "ADDITIONAL_FILE_PATTERNS", None) or None
                )
                self.hf_pretrained.save_artifacts(
                    path, original_source_path=source_path, additional_files=additional_files
                )

            if model_bridge is not None:
                artifact_postprocessor = getattr(
                    type(model_bridge), "postprocess_hf_export_artifacts", None
                )
                if artifact_postprocessor is not None:
                    artifact_postprocessor(model_bridge, Path(path))

        if not dist.is_initialized() or dist.get_rank() == 0:
            _save_artifacts()
        return model_bridge

    def save_hf_weights(
        self,
        model: list[MegatronModelT],
        path: str | Path,
        show_progress: bool = True,
        strict: bool = True,
        distributed_save: bool = False,
        save_every_n_ranks: int = 1,
        weight_dtype: Optional[torch.dtype] = None,
    ) -> None:
        """
        Save Megatron model weights in HuggingFace safetensors format.

        This method exports only the model weights (not configuration or tokenizer)
        to safetensors files compatible with HuggingFace. It uses streaming save
        to handle large models efficiently without requiring all weights in memory
        at once.

        The weights are gathered from distributed ranks and saved in the standard
        HuggingFace sharded format when the model is large.

        Args:
            model: Megatron model instance or list of instances
            path: Directory path where weight files will be saved
            show_progress: Display progress bar during export
            distributed_save: Whether to enable distributed saving mode where each rank saves
                part of weights independently.
            save_every_n_ranks: Interval for saving weights across ranks in distributed mode.
                For example, if set to 2, only ranks 0, 2, 4, ... will save weights.
            weight_dtype: Plain export dtype; skips quantized *.scale companions when set.

        Raises:
            ValueError: If the state source doesn't support streaming save

        Example:
            >>> # Save just the weights
            >>> bridge.save_hf_weights(megatron_model, "./model_weights")

            >>> # Save without progress bar (useful in scripts)
            >>> bridge.save_hf_weights(megatron_model, "./weights", show_progress=False)

        Note:
            - This method is collective and must be called by all ranks
            - Uses safetensors format for efficient loading and security
            - Automatically handles model sharding for large models
            - The saved weights can be loaded with HuggingFace's from_pretrained
        """
        is_distributed = dist.is_initialized()
        rank = dist.get_rank() if is_distributed else 0
        output_path = Path(path).resolve()
        previous_weight_files = _existing_safetensors_files(output_path) if rank == 0 else set()
        saved_filenames: set[str] = set()
        if is_distributed:
            dist.barrier()
        bridge = self._model_bridge
        generator = bridge.stream_weights_megatron_to_hf(
            model,
            self.hf_pretrained,
            cpu=True,
            show_progress=show_progress,
            weight_dtype=weight_dtype,
        )
        model_instance = self._get_model_instance(model)
        quant_tensors = None
        quantizer_recipe = None
        # Import lazily so Bridge conversion modules can load before ModelOpt
        # registers its Megatron-Bridge plugin hooks.
        from modelopt.torch.quantization.utils import is_quantized

        # Building the recipe is collective, so decide on every rank from all chunks: a pipeline
        # stage can hold no quantizers (e.g. only the embedding).
        quantized = any(is_quantized(chunk) for chunk in model)
        if is_distributed:
            per_rank = [None] * dist.get_world_size()
            dist.all_gather_object(per_rank, quantized)
            quantized = any(per_rank)
        if quantized:
            quantizer_recipe = bridge.build_modelopt_quantizer_recipe(self.hf_pretrained, model)
            quant_tensors = {}

            def _filter_quant(gen):
                for name, tensor in gen:
                    if "_quantizer." in name:
                        quant_tensors[name] = tensor
                        continue
                    yield name, tensor

            generator = _filter_quant(generator)

        # Check if the state source is SafeTensorsStateSource for streaming save.
        if (
            hasattr(self.hf_pretrained, "state")
            and hasattr(self.hf_pretrained.state, "source")
            and isinstance(self.hf_pretrained.state.source, SafeTensorsStateSource)
        ):
            source = self.hf_pretrained.state.source
            model_config = getattr(model_instance, "config", None)
            hf_config = getattr(self.hf_pretrained, "config", self.hf_pretrained)
            mtp_disabled = (
                _saved_config_disables_mtp(path)
                or any(_config_disables_mtp(config) for config in (hf_config, model_config))
                or _model_omits_mtp(model_config)
            )
            ignored_source_key_prefixes = (
                _mtp_source_key_prefixes(source, hf_config, model_config) if mtp_disabled else ()
            ) or None
            declared_prefixes = getattr(
                type(self._model_bridge), "HF_EXPORT_IGNORED_SOURCE_PREFIXES", ()
            )
            if not isinstance(declared_prefixes, tuple) or any(
                not isinstance(prefix, str)
                or not prefix
                or not prefix.endswith(".")
                or "*" in prefix
                for prefix in declared_prefixes
            ):
                raise ValueError(
                    "HF_EXPORT_IGNORED_SOURCE_PREFIXES must contain nonempty module prefixes ending in '.'."
                )
            ignored_source_key_prefixes = (
                tuple(dict.fromkeys((*(ignored_source_key_prefixes or ()), *declared_prefixes)))
                or None
            )
            ignored_source_key_suffixes = tuple(
                dict.fromkeys(
                    (("_scale_inv",) if weight_dtype is not None else ())
                    + tuple(getattr(type(bridge), "DEQUANTIZED_SOURCE_KEY_SUFFIXES", ()))
                )
            )
            saved_filenames = source.save_generator(
                generator,
                path,
                strict=strict,
                distributed_save=distributed_save,
                save_every_n_ranks=save_every_n_ranks,
                ignored_source_key_prefixes=ignored_source_key_prefixes,
                ignored_source_key_suffixes=ignored_source_key_suffixes or None,
            )
        else:
            # Config-only path: shard and write safetensors directly
            import json

            from huggingface_hub import split_torch_state_dict_into_shards

            # NOTE: Collects the full state dict into CPU memory before sharding.
            # For very large models (>70B), this may require significant host RAM.
            if rank == 0:
                state_dict = {name: tensor.contiguous().cpu() for name, tensor in generator}
            else:
                for _ in generator:
                    pass
                state_dict = None

            if rank == 0:
                plan = split_torch_state_dict_into_shards(state_dict)
                safe_dir = Path(path)
                safe_dir.mkdir(parents=True, exist_ok=True)
                for filename, tensors in plan.filename_to_tensors.items():
                    shard = {k: state_dict[k] for k in tensors}
                    save_file(shard, safe_dir / filename)
                    saved_filenames.add(filename)
                if plan.is_sharded:
                    index = {"metadata": plan.metadata, "weight_map": plan.tensor_to_filename}
                    with open(safe_dir / "model.safetensors.index.json", "w") as f:
                        json.dump(index, f, indent=2)
                    saved_filenames.add("model.safetensors.index.json")

        if rank == 0:
            for filename in previous_weight_files - saved_filenames:
                (output_path / filename).unlink(missing_ok=True)

        # Save the quantizer sidecars after the main generator is consumed (rank 0 only). A model
        # without quantizers still removes stale ones, so they never describe other weights.
        if rank == 0:
            _save_modelopt_quantizer_sidecars(
                output_path, quantizer_recipe or {}, quant_tensors or {}
            )

        if is_distributed:
            dist.barrier()

    def save_megatron_model(
        self,
        model: list[MegatronModule],
        path: str | Path,
        hf_tokenizer_path: Optional[str | Path] = None,
        low_memory_save: bool = False,
        hf_tokenizer_kwargs: Optional[dict] = None,
    ) -> None:
        """
        Save a Megatron model in native Megatron checkpoint format without optimizer
        state.

        This method saves the model in Megatron's native checkpoint format, which
        can be loaded directly by Megatron for training or inference. The checkpoint
        includes the model configuration and weights, NO optimizer state or other
        artifacts.

        Args:
            model: Megatron model instance or list of instances
            path: Directory path where the checkpoint will be saved
            hf_tokenizer_path: Optional HuggingFace model ID or path for tokenizer metadata.
                If provided, the tokenizer metadata will be included in the checkpoint.
            low_memory_save: If True, uses a memory-optimized save flow that reduces
                peak memory by ~50% for models with merged weights (e.g., gate+up
                projections). The model is deleted after state dict generation and
                cannot be used afterward. Default is False, preserving the model
                for further use.
            hf_tokenizer_kwargs: Optional dictionary of kwargs to pass to the HuggingFace tokenizer.
                Common options include trust_remote_code=True for models with custom tokenizers,
                or use_fast=True for models that require the fast tokenizer.

        Example:
            >>> # Save model checkpoint after conversion
            >>> bridge.save_megatron_model(megatron_model, "./megatron_checkpoint")

            >>> # Save model checkpoint with tokenizer metadata
            >>> bridge.save_megatron_model(
            ...     megatron_model,
            ...     "./megatron_checkpoint",
            ...     hf_tokenizer_path="meta-llama/Meta-Llama-3-8B"
            ... )

            >>> # Low-memory save (destroys model after save)
            >>> bridge.save_megatron_model(
            ...     megatron_model,
            ...     "./megatron_checkpoint",
            ...     low_memory_save=True
            ... )

        Note:
            - This method is collective and must be called by all ranks
            - The saved checkpoint can be loaded with Megatron's checkpoint loading utilities
            - The checkpoint format follows Megatron's standard structure for compatibility
            - When low_memory_save=True, the model is deleted and cannot be used afterward
        """
        from megatron.bridge.models.conversion.checkpoint import save_megatron_model

        save_megatron_model(
            model,
            path,
            hf_tokenizer_path=hf_tokenizer_path,
            low_memory_save=low_memory_save,
            hf_tokenizer_kwargs=hf_tokenizer_kwargs,
        )

    def load_megatron_model(
        self,
        path: str | Path,
        *,
        mp_overrides: ModelParallelKwargs | None = None,
        **kwargs: Unpack[GetModelKwargs],
    ) -> list[MegatronModelT]:
        """
        Load a Megatron model from a native Megatron checkpoint.

        This method loads a model from a Megatron checkpoint that was saved using
        the save_megatron_model method. It reads the checkpoint configuration,
        creates the appropriate model provider, and loads the weights.

        Args:
            path: Directory path where the Megatron checkpoint is stored
            mp_overrides: Optional model-parallel overrides to apply to the loaded config.
            **kwargs: Additional arguments passed to the model provider

        Returns:
            List of Megatron model instances loaded from the checkpoint

        Example:
            >>> # Load a previously saved Megatron model
            >>> bridge = AutoBridge.from_hf_config(config)
            >>> model = bridge.load_megatron_model("./megatron_checkpoint")

            >>> # Load and specify model configuration
            >>> model = bridge.load_megatron_model(
            ...     "./megatron_checkpoint",
            ...     wrap_with_ddp=False
            ... )

        Note:
            - This method is collective and must be called by all ranks
            - The checkpoint must have been saved with save_megatron_model
            - The model architecture must match the bridge configuration
        """
        from megatron.bridge.models.conversion.checkpoint import load_megatron_model

        if self.trust_remote_code:
            from megatron.bridge.utils.instantiate_utils import register_allowed_target_prefix

            register_allowed_target_prefix("transformers_modules.")

        return load_megatron_model(self, path, mp_overrides=mp_overrides, **kwargs)

    @classmethod
    def import_ckpt(
        cls,
        hf_model_id: str | Path,
        megatron_path: str | Path,
        *,
        low_memory_save: bool = False,
        **kwargs,
    ) -> None:
        """
        Import a HuggingFace model and save it as a Megatron checkpoint.

        This is a convenience method that combines loading a HuggingFace model,
        converting it to Megatron format, and saving it as a native Megatron
        checkpoint. This is useful for preparing models for Megatron training
        or creating Megatron checkpoints from pretrained HuggingFace models.

        Args:
            hf_model_id: HuggingFace model ID or path to model directory
                Examples: "meta-llama/Meta-Llama-3-8B", "./my_model"
            megatron_path: Directory path where the Megatron checkpoint will be saved
            low_memory_save: Reduce peak memory while saving the Megatron checkpoint.
                This destroys the converted model during save and can increase runtime.
            **kwargs: Additional arguments passed to from_hf_pretrained
                Common options include:
                - torch_dtype: Model precision (torch.float16, torch.bfloat16)
                - device_map: Device placement strategy ("auto", "cuda:0", etc.)
                - trust_remote_code: Allow custom model code execution
                - attn_implementation: Attention implementation ("flash_attention_2", etc.)

        Example:
            >>> # Basic import
            >>> AutoBridge.import_ckpt(
            ...     "meta-llama/Meta-Llama-3-8B",
            ...     "./megatron_checkpoints/llama3_8b"
            ... )

            >>> # Import with specific settings
            >>> AutoBridge.import_ckpt(
            ...     "meta-llama/Meta-Llama-3-8B",
            ...     "./megatron_checkpoints/llama3_8b",
            ...     torch_dtype=torch.float16,
            ...     device_map="auto",
            ...     low_memory_save=True
            ... )
        """
        # Load the HuggingFace model before creating temporary distributed state.
        bridge = cls.from_hf_pretrained(hf_model_id, **kwargs)

        from megatron.bridge.models.conversion.checkpoint import temporary_distributed_context

        model_context = (
            nullcontext()
            if dist.is_initialized()
            else temporary_distributed_context(backend="gloo")
        )
        with model_context:
            # Prefer the native ModelConfig/ModelBuilder path for migrated model
            # families while preserving the provider path for legacy bridges.
            if bridge._model_bridge.USE_MODEL_CONFIG_FOR_CONVERSION:
                model_config = bridge.get_model_config()
                model_config.transformer.use_cpu_initialization = True
                megatron_model = bridge.get_model(
                    model_config, wrap_with_ddp=False, mixed_precision_wrapper=None
                )
            else:
                megatron_model = bridge.to_megatron_model(
                    wrap_with_ddp=False, use_cpu_initialization=True
                )

            # Save as Megatron checkpoint
            hf_tokenizer_kwargs = {}
            if hasattr(bridge._model_bridge, "get_hf_tokenizer_kwargs"):
                hf_tokenizer_kwargs = bridge._model_bridge.get_hf_tokenizer_kwargs()
            if hf_tokenizer_kwargs is None:
                hf_tokenizer_kwargs = {}
            if kwargs.get("revision") is not None:
                hf_tokenizer_kwargs.setdefault("revision", kwargs["revision"])
            # Forward trust_remote_code to the tokenizer (needed for repos with custom code)
            if kwargs.get("trust_remote_code"):
                hf_tokenizer_kwargs.setdefault("trust_remote_code", True)
            bridge.save_megatron_model(
                megatron_model,
                megatron_path,
                hf_tokenizer_path=hf_model_id,
                hf_tokenizer_kwargs=hf_tokenizer_kwargs,
                low_memory_save=low_memory_save,
            )

    def export_ckpt(
        self,
        megatron_path: str | Path,
        hf_path: str | Path,
        show_progress: bool = True,
        strict: bool = False,
        source_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """
        Export a Megatron checkpoint to HuggingFace format.

        This is a convenience method that loads a Megatron checkpoint and
        exports it to HuggingFace format. This is useful for sharing trained
        models or deploying them with HuggingFace inference tools.

        Also supports config-only bridges created via the auto-config
        flow used by ``examples/bridge/conversion/convert.sh``.

        Args:
            megatron_path: Directory path where the Megatron checkpoint is stored
            hf_path: Directory path where the HuggingFace model will be saved
            show_progress: Display progress bar during weight export
            strict: Whether to perform strict validation during weight export
            source_path: Path to the directory containing custom modeling files to be preserved.
                This is useful when converting from Megatron checkpoints where the original
                HuggingFace model with custom modeling files needs to be referenced. If not specified,
                the path will be automatically determined from the HuggingFace configuration.

        Example:
            >>> # Basic export
            >>> bridge = AutoBridge.from_hf_pretrained("meta-llama/Meta-Llama-3-8B")
            >>> bridge.export_ckpt(
            ...     "./megatron_checkpoints/my_model",
            ...     "./hf_exports/my_model"
            ... )
            >>> # Export with specific settings
            >>> bridge.export_ckpt(
            ...     "./megatron_checkpoints/my_model",
            ...     "./hf_exports/my_model",
            ...     show_progress=False
            ... )

            >>> # Load the exported model with HuggingFace
            >>> from transformers import AutoModelForCausalLM
            >>> hf_model = AutoModelForCausalLM.from_pretrained("./hf_exports/my_model")
        """
        from megatron.bridge.models.conversion.checkpoint import temporary_distributed_context

        # Export ckpt performs on CPU
        model_context = (
            nullcontext()
            if dist.is_initialized()
            else temporary_distributed_context(backend="gloo")
        )
        with model_context:
            # Load the Megatron model
            megatron_model = self.load_megatron_model(
                megatron_path,
                wrap_with_ddp=False,
                use_cpu_initialization=dist.get_backend() == "gloo",
            )

            # Save in HuggingFace format
            self.save_hf_pretrained(
                megatron_model,
                hf_path,
                show_progress=show_progress,
                source_path=source_path,
                strict=strict,
            )

    def push_to_hub(self, path: str | Path) -> None: ...

    def get_model_config(self) -> ModelConfig:
        """Convert the Hugging Face architecture to a builder-backed config.

        Builder-backed config support is being rolled out incrementally by
        model family. This API is available only for families that have
        migrated to a compatible ``ModelConfig``. Continue to use
        :meth:`to_megatron_provider` for families that have not yet migrated.

        This method performs configuration conversion only. It does not load
        weights, initialize distributed state, finalize the config, or build a
        model.

        The Hugging Face source id and revision are recorded in the config's
        ``extra_checkpoint_metadata``. A source projected to its language model
        with ``text_only=True`` also sets ``hf_model_text_only``, so a restored
        config reopens the same projection.

        Returns:
            Serializable model config linked to its model builder.

        Raises:
            ModelConfigNotSupportedError: If the selected model family has not
                migrated to builder-backed configs.
        """
        hf_config = (
            self.hf_pretrained.config
            if isinstance(self.hf_pretrained, _PRETRAINED_WRAPPER_TYPES)
            else self.hf_pretrained
        )
        model_config = self._model_bridge.hf_config_to_model_config(hf_config)
        # Only families declaring this option need checkpoint code at build time.
        if hasattr(model_config, "trust_remote_code"):
            model_config.trust_remote_code = self.trust_remote_code

        hf_name_or_path = getattr(self.hf_pretrained, "model_name_or_path", None)
        if hf_name_or_path is None and isinstance(self.hf_pretrained, PretrainedConfig):
            hf_name_or_path = getattr(self.hf_pretrained, "name_or_path", None)
        self._set_source_metadata(
            model_config, hf_model_id=hf_name_or_path, hf_model_revision=self.hf_model_revision
        )
        if self.text_only:
            model_config.hf_model_text_only = True

        return model_config

    @staticmethod
    def _set_source_metadata(
        model_config: ModelConfig, *, hf_model_id: str | Path | None, hf_model_revision: str | None
    ) -> None:
        """Record Hugging Face source provenance in the config's checkpoint metadata.

        The source ``hf_model_id`` and ``hf_model_revision`` are stored under the
        inherited serializable ``extra_checkpoint_metadata`` mapping rather than as
        dedicated config fields, so they round-trip through ``run_config.yaml``
        without adding model-specific fields to builder-backed configs.

        Args:
            model_config: Builder-backed config to annotate in place.
            hf_model_id: Source model id or path. Ignored when falsy.
            hf_model_revision: Source revision. When falsy, any recorded revision
                is removed so it does not leak from a previous source.
        """
        metadata = model_config.extra_checkpoint_metadata
        if metadata is None:
            metadata = {}
            model_config.extra_checkpoint_metadata = metadata
        if hf_model_id:
            metadata["hf_model_id"] = str(hf_model_id)
        if hf_model_revision:
            metadata["hf_model_revision"] = hf_model_revision
        else:
            metadata.pop("hf_model_revision", None)

    def get_model(
        self,
        model_config: ModelConfig | None = None,
        *,
        load_weights: bool = True,
        hf_path: str | Path | None = None,
        pg_collection: ProcessGroupCollection | None = None,
        **kwargs: Any,
    ) -> list[MegatronModelT]:
        """Build a Megatron model through its configured ``ModelBuilder``.

        Like :meth:`get_model_config`, this builder-backed API is being rolled
        out incrementally and supports only migrated model families. Continue
        to use :meth:`to_megatron_model` for families that have not yet
        migrated.

        Args:
            model_config: Optional config with user overrides already applied.
                When omitted, :meth:`get_model_config` supplies the defaults.
            load_weights: Load Hugging Face weights before distributed wrapping.
            hf_path: Optional Hugging Face checkpoint path to load instead of
                the bridge's original source.
            pg_collection: Existing process groups. Standalone single-process
                groups are initialized when omitted.
            **kwargs: Arguments for ``ModelBuilder.build_distributed_models``.
                Its ``mixed_precision_wrapper`` (``Float16Module`` when omitted,
                ``None`` to skip wrapping) keeps router expert biases and
                parameters named in ``_keep_in_float32_parameter_names`` in FP32,
                as :meth:`to_megatron_model` does.

        Returns:
            Distributed model stages created by the configured builder.

        Raises:
            ModelConfigNotSupportedError: If defaults are requested for a model
                family that has not migrated to builder-backed configs.
        """
        if model_config is None:
            model_config = self.get_model_config()

        transformer_config = getattr(model_config, "transformer", None)
        if not isinstance(transformer_config, TransformerConfig):
            raise TypeError(
                f"{type(model_config).__name__} does not define a nested TransformerConfig."
            )
        if hf_path is not None and not load_weights:
            raise ValueError("hf_path is only valid when load_weights=True.")
        if (
            load_weights
            and hf_path is None
            and not isinstance(self.hf_pretrained, _PRETRAINED_WRAPPER_TYPES)
        ):
            raise ValueError(
                "AutoBridge.from_hf_config() does not include weights. "
                "Pass load_weights=False or provide hf_path to load weights."
            )

        original_perform_initialization = transformer_config.perform_initialization
        original_pre_wrap_hooks = list(model_config.pre_wrap_hooks)
        original_extra_metadata = model_config.extra_checkpoint_metadata
        original_extra_metadata_snapshot = (
            dict(original_extra_metadata) if original_extra_metadata is not None else None
        )
        succeeded = False
        try:
            if load_weights:
                transformer_config.perform_initialization = False
                if hf_path is None:
                    pretrained = self.hf_pretrained
                else:
                    if self.text_only:
                        pretrained = self._text_only_pretrained_from_path(hf_path)
                    else:
                        pretrained = self._pretrained_wrapper_cls.from_pretrained(
                            hf_path, trust_remote_code=self.trust_remote_code
                        )
                    self._set_source_metadata(
                        model_config, hf_model_id=hf_path, hf_model_revision=None
                    )
                model_config.pre_wrap_hooks[:] = [
                    partial(self._model_bridge.load_weights_hf_to_megatron, pretrained),
                    *original_pre_wrap_hooks,
                ]

            model_config.finalize()
            from megatron.bridge.models.conversion.runtime import (
                classify_gtp_remat_chains,
                configure_gtp_remat,
            )

            configure_gtp_remat(model_config)
            if pg_collection is None:
                pg_collection = self._get_or_initialize_pg_collection(transformer_config)
            kwargs.setdefault("data_parallel_random_init", False)
            models = build_distributed_models(model_config, pg_collection, **kwargs)
            classify_gtp_remat_chains(models, model_config)
            for model in models:
                model.model_config = model_config
            succeeded = True
        finally:
            transformer_config.perform_initialization = original_perform_initialization
            model_config.pre_wrap_hooks[:] = original_pre_wrap_hooks
            if not succeeded:
                model_config.extra_checkpoint_metadata = (
                    dict(original_extra_metadata_snapshot)
                    if original_extra_metadata_snapshot is not None
                    else None
                )

        return models

    @staticmethod
    def _get_or_initialize_pg_collection(
        transformer_config: TransformerConfig, *, seed: int = 0
    ) -> ProcessGroupCollection:
        """Return model-parallel process groups for standalone construction."""
        if not dist.is_initialized():
            os.environ.setdefault("RANK", "0")
            os.environ.setdefault("WORLD_SIZE", "1")
            os.environ.setdefault("MASTER_ADDR", "localhost")
            os.environ.setdefault("MASTER_PORT", "12355")

            use_cpu = transformer_config.use_cpu_initialization or not torch.cuda.is_available()
            backend = "gloo" if use_cpu else "nccl"
            if backend == "nccl":
                torch.cuda.set_device(get_local_rank_preinit())
            dist.init_process_group(backend=backend)

        if not parallel_state.is_initialized():
            parallel_state.initialize_model_parallel(
                tensor_model_parallel_size=transformer_config.tensor_model_parallel_size,
                pipeline_model_parallel_size=transformer_config.pipeline_model_parallel_size,
                virtual_pipeline_model_parallel_size=transformer_config.virtual_pipeline_model_parallel_size,
                context_parallel_size=transformer_config.context_parallel_size or 1,
                expert_model_parallel_size=transformer_config.expert_model_parallel_size or 1,
                expert_tensor_parallel_size=transformer_config.expert_tensor_parallel_size,
                gtp_remat_size=transformer_config.gtp_weight_remat_size,
                expert_gtp_remat_size=transformer_config.expert_gtp_weight_remat_size,
            )
            if torch.cuda.is_available():
                from megatron.core.tensor_parallel import model_parallel_cuda_manual_seed

                model_parallel_cuda_manual_seed(seed)

        return ProcessGroupCollection.use_mpu_process_groups()

    def to_megatron_model(
        self,
        load_weights: bool = True,
        hf_path: str | Path | None = None,
        **kwargs: Unpack[GetModelKwargs],
    ) -> list[MegatronModelT]:
        provider = self.to_megatron_provider(load_weights, hf_path)

        # Finalize the provider before creating models
        if hasattr(provider, "finalize"):
            provider.finalize()

        return provider.provide_distributed_model(**kwargs)

    def to_megatron_provider(
        self, load_weights: bool = True, hf_path: str | Path | None = None
    ) -> GPTModelProvider:
        """
        Convert to a Megatron model provider.

        This method creates a GPTModelProvider configured to match the HuggingFace
        model's architecture. The provider can then be used to instantiate
        Megatron models for training or inference.

        Args:
            load_weights: Whether to configure the provider to load weights
                from HuggingFace format. If False, creates model with random
                initialization.
            hf_path: Optional path to load weights from. If None, uses weights
                from the bridge's hf_pretrained instance. Useful for loading
                weights from a different checkpoint.

        Returns:
            GPTModelProvider: A configured model provider ready to create
                Megatron models

        Example:
            >>> # Create provider and model with loaded weights
            >>> bridge = AutoBridge.from_hf_pretrained("meta-llama/Meta-Llama-3-8B")
            >>> provider = bridge.to_megatron_provider()
            >>> model = provider.get_model()

            >>> # Create provider without loading weights (for training from scratch)
            >>> provider = bridge.to_megatron_provider(load_weights=False)
            >>> model = provider.get_model()  # Random initialization

            >>> # Load weights from a different checkpoint
            >>> bridge = AutoBridge.from_hf_config(config)  # Config only
            >>> provider = bridge.to_megatron_provider(hf_path="./finetuned_model")
            >>> model = provider.get_model()  # Loads finetuned weights

        See Also:
            GPTModelProvider: The provider class for creating models
            load_weights: Method to load weights into existing models
        """
        provider_input = self._provider_bridge_input
        provider: ModelProviderMixin = self._model_bridge.provider_bridge(provider_input)
        if self.text_only:
            provider.hf_model_text_only = True

        if load_weights:
            if hf_path is None and not isinstance(self.hf_pretrained, _PRETRAINED_WRAPPER_TYPES):
                raise ValueError(
                    "AutoBridge.from_hf_config() does not include weights. "
                    "Pass load_weights=False for random initialization or provide hf_path to load weights."
                )
            # Skip weights initialization since we are going to load weights
            provider.perform_initialization = False
            if hf_path is None:
                provider.register_pre_wrap_hook(
                    partial(self._model_bridge.load_weights_hf_to_megatron, self.hf_pretrained)
                )
            else:
                # Load from specified path
                trust_remote_code = getattr(self.hf_pretrained, "trust_remote_code", False)
                if self.text_only:
                    pre_trained = self._text_only_pretrained_from_path(hf_path)
                else:
                    wrapper_cls = self._pretrained_wrapper_cls
                    pre_trained = wrapper_cls.from_pretrained(
                        hf_path, trust_remote_code=trust_remote_code
                    )
                provider.register_pre_wrap_hook(
                    partial(self._model_bridge.load_weights_hf_to_megatron, pre_trained)
                )

        hf_identifier: str | None = None
        if hf_path is not None:
            hf_identifier = str(hf_path)
        else:
            hf_name_or_path = getattr(self.hf_pretrained, "model_name_or_path", None)
            if hf_name_or_path is None and isinstance(self.hf_pretrained, PretrainedConfig):
                hf_name_or_path = getattr(self.hf_pretrained, "name_or_path", None)
            if hf_name_or_path:
                hf_identifier = str(hf_name_or_path)

        if hf_identifier:
            setattr(provider, "hf_model_id", hf_identifier)
        if hf_path is None and self.hf_model_revision:
            setattr(provider, "hf_model_revision", self.hf_model_revision)
        elif self.text_only and load_weights and hf_path is not None:
            provider.hf_model_revision = pre_trained.init_kwargs.get("revision")

        return provider

    @staticmethod
    def get_hf_model_id_from_checkpoint(path: str | Path) -> str | None:
        """Get the HuggingFace model identifier stored in a checkpoint.

        Args:
            path: Path to a Megatron checkpoint directory, either the root directory
                containing iteration subdirectories or a specific iteration directory.

        Returns:
            The HuggingFace model ID or path recorded in the checkpoint metadata if present,
            otherwise `None`.
        """
        from megatron.bridge.models.conversion import checkpoint as _checkpoint_utils

        return _checkpoint_utils.get_hf_model_id_from_checkpoint(path)

    def get_conversion_tasks(
        self,
        megatron_model: Union[MegatronModelT, List[MegatronModelT]],
        hf_path: str | Path | None = None,
        *,
        exclude_param_patterns: tuple[str, ...] = (),
    ) -> List["WeightConversionTask"]:
        """Get the conversion tasks for weight conversion between HuggingFace and Megatron formats.

        This method returns the planned conversion tasks that would be executed during
        weight conversion in either direction. Each task contains information about parameter
        mappings, source and target parameters, and the conversion logic required.

        The tasks can be used for both HF→Megatron and Megatron→HF conversions since they
        contain bidirectional mapping information.

        Args:
            megatron_model: Megatron model instance or list of instances (one per
                virtual pipeline stage) that participate in the conversion.
            hf_path: Optional path to load HF weights from. If None, uses weights
                from the bridge's hf_pretrained instance.
            exclude_param_patterns: Glob patterns for local parameter and persistent-buffer
                names to exclude before mapping validation. Defaults to no exclusions.

        Returns:
            List[WeightConversionTask]: List of conversion tasks that would be executed.
                Each task contains:
                - param_name: Megatron parameter name
                - mapping: The parameter mapping object handling the conversion
                - pp_rank: Pipeline parallel rank that owns the parameter
                - vp_stage: Virtual pipeline stage index
                - megatron_module: Reference to the Megatron module owning the parameter
                - param_weight: The actual parameter tensor

        Example:
            >>> bridge = AutoBridge.from_hf_pretrained("meta-llama/Llama-3.2-1B")
            >>> megatron_model = bridge.to_megatron_model(load_weights=False, wrap_with_ddp=False)
            >>> tasks = bridge.get_conversion_tasks(megatron_model)
            >>>
            >>> for task in tasks:
            ...     # For HF→Megatron direction
            ...     print(f"HF param {task.mapping.hf_param} -> Megatron param {task.param_name}")
            ...
            ...     # For Megatron→HF direction
            ...     hf_params = task.mapping.hf_param
            ...     if isinstance(hf_params, str):
            ...         print(f"Megatron param {task.param_name} -> HF param {hf_params}")
            ...     else:
            ...         print(f"Megatron param {task.param_name} -> HF params {list(hf_params.values())}")
            ...
            ...     print(f"  Mapping type: {type(task.mapping).__name__}")
            ...     print(f"  PP rank: {task.pp_rank}, VP stage: {task.vp_stage}")

        Note:
            This method is useful for:
            - Debugging weight conversion issues in both directions
            - Understanding parameter mappings between formats
            - Custom weight conversion implementations
            - Analyzing model structure differences
            - Verifying parameter alignment and shapes
        """
        if not isinstance(megatron_model, list):
            megatron_model = [megatron_model]

        if hf_path is None:
            if not isinstance(self.hf_pretrained, _PRETRAINED_WRAPPER_TYPES):
                raise ValueError(
                    "hf_path is required when hf_pretrained is not a PreTrainedCausalLM, "
                    "PreTrainedMaskedLM, or PreTrainedTokenClassification instance"
                )
            pre_trained = self.hf_pretrained
        else:
            if self.text_only:
                pre_trained = self._text_only_pretrained_from_path(hf_path)
            else:
                pre_trained = self._pretrained_wrapper_cls.from_pretrained(hf_path)

        return self._model_bridge.build_conversion_tasks(
            pre_trained,
            megatron_model,
            **(
                {"exclude_param_patterns": exclude_param_patterns} if exclude_param_patterns else {}
            ),
        )

    def stream_weights_hf_to_megatron(
        self,
        megatron_model: MegatronModelT | list[MegatronModelT],
        conversion_tasks: list[WeightConversionTask] | None = None,
        *,
        hf_state_dict: Mapping[str, torch.Tensor] | None = None,
    ) -> Iterable[MegatronWeightTuple]:
        """Stream HF-to-Megatron conversions from the configured or external state."""
        return self._model_bridge.stream_weights_hf_to_megatron(
            self._provider_bridge_input,
            megatron_model,
            conversion_tasks,
            hf_state_dict=hf_state_dict,
        )

    def finalize_hf_import(self, megatron_model: MegatronModelT | list[MegatronModelT]) -> None:
        """Finalize tied parameters and parameter-derived caches after import."""
        self._model_bridge.finalize_hf_import(megatron_model)

    @property
    def transformer_config(self) -> TransformerConfig:
        _model_provider = self.to_megatron_provider(load_weights=False)

        # Finalize the provider before extracting config
        if hasattr(_model_provider, "finalize"):
            _model_provider.finalize()

        return self._create_config_from_provider(_model_provider, TransformerConfig)

    @property
    def mla_transformer_config(self) -> MLATransformerConfig:
        _model_provider = self.to_megatron_provider(load_weights=False)

        # Finalize the provider before extracting config
        if hasattr(_model_provider, "finalize"):
            _model_provider.finalize()

        return self._create_config_from_provider(_model_provider, MLATransformerConfig)

    @property
    def _model_bridge(self) -> "MegatronModelBridge":
        hf_config = getattr(self.hf_pretrained, "hf_config", None)
        if hf_config is None:
            if isinstance(self.hf_pretrained, _PRETRAINED_WRAPPER_TYPES):
                hf_config = self.hf_pretrained.config
            else:
                hf_config = self.hf_pretrained

        bridge = model_bridge.get_model_bridge(self._causal_lm_architecture, hf_config=hf_config)
        bridge.export_weight_dtype = self.export_weight_dtype
        bridge.trust_remote_code = self.trust_remote_code
        return bridge

    def _text_only_pretrained_from_path(self, path: str | Path) -> PreTrainedCausalLM:
        """Reuse a pinned source when checkpoint initialization names it again."""
        current = self.hf_pretrained
        if (
            isinstance(current, PreTrainedCausalLM)
            and self.text_only
            and str(path) == str(current.model_name_or_path)
        ):
            return current
        source_kwargs = getattr(current, "init_kwargs", {})
        kwargs = {
            key: source_kwargs[key]
            for key in ("token", "cache_dir", "local_files_only")
            if key in source_kwargs
        }
        # A different repository may not contain the original revision.
        # Config-only reconstruction, however, still references the same source.
        if str(path) == self.hf_model_id and self.hf_model_revision:
            kwargs["revision"] = self.hf_model_revision
        return (
            type(self)
            .from_hf_pretrained(
                path, text_only=True, trust_remote_code=self.trust_remote_code, **kwargs
            )
            .hf_pretrained
        )

    @property
    def _pretrained_wrapper_cls(
        self,
    ) -> Type[PreTrainedCausalLM] | Type[PreTrainedMaskedLM] | Type[PreTrainedTokenClassification]:
        """The hf_pretrained wrapper class matching this bridge's current instance.

        Falls back to resolving from the current config when ``hf_pretrained`` is a
        bare ``PretrainedConfig`` (i.e. the bridge was built with ``from_hf_config``).
        """
        if isinstance(self.hf_pretrained, PreTrainedTokenClassification):
            return PreTrainedTokenClassification
        if isinstance(self.hf_pretrained, PreTrainedMaskedLM):
            return PreTrainedMaskedLM
        if isinstance(self.hf_pretrained, PreTrainedCausalLM):
            return PreTrainedCausalLM
        return _resolve_pretrained_wrapper_cls(self.hf_pretrained)

    @property
    def _provider_bridge_input(
        self,
    ) -> (
        PreTrainedCausalLM
        | PreTrainedMaskedLM
        | PreTrainedTokenClassification
        | _ConfigOnlyPretrainedShim
    ):
        if isinstance(self.hf_pretrained, _PRETRAINED_WRAPPER_TYPES):
            return self.hf_pretrained
        return self._config_only_pretrained

    @property
    def _causal_lm_architecture(self):
        """Resolve the model architecture used for bridge dispatch.

        Behavior:
        - If the model can be imported from transformers directly, return the actual transformers class object.
        - Otherwise, if the model uses HuggingFace auto_map, return the architecture's class name as a string (e.g.,
        "DeepseekV3ForCausalLM").

        Returns:
            str | type: The Transformers model class, or its class name for
            ``auto_map`` and other dynamically registered models.

        Raises:
            ValueError: If no supported architecture is found or resolved.
        """
        if isinstance(self.hf_pretrained, _PRETRAINED_WRAPPER_TYPES):
            config = self.hf_pretrained.config
        else:
            config = self.hf_pretrained

        architectures = getattr(config, "architectures", [])

        if not architectures:
            raise ValueError(
                "\n✗ No architectures found in model config\n\n"
                "The model configuration does not specify any architectures.\n"
                "This is required for determining the model type."
            )

        causal_lm_arch = None
        for architecture_name in architectures:
            # TODO: Can we improve this?
            if architecture_name.endswith(SUPPORTED_HF_ARCHITECTURES):
                causal_lm_arch = architecture_name
                break

        if not causal_lm_arch:
            raise ValueError(
                f"\n✗ No supported architecture found\n\n"
                f"Model architectures: {architectures}\n\n"
                f"None of the architectures end with {SUPPORTED_HF_ARCHITECTURES_DISPLAY}.\n"
                f"This bridge only supports the language-model and task-head architectures "
                f"listed above (for example causal LMs, masked LMs, and token classifiers).\n"
                f"For other model types, use a different bridge class."
            )

        # Try auto_map first (returns class name string if available)
        cls_name = get_causal_lm_class_name_via_auto_map(config=config)
        if cls_name is not None:
            # For auto_map models, return the class name as a string
            return cls_name

        # Resolve non-standard architecture names via alias mapping
        resolved_arch = HF_ARCHITECTURE_ALIASES.get(causal_lm_arch, causal_lm_arch)

        try:
            return getattr(transformers, resolved_arch)
        except AttributeError:
            # Model class not in standard transformers — fall back to class-name string.
            # This handles custom models registered via AutoConfig.register / AutoModelForCausalLM.register
            # in model bridge modules (e.g. BailingMoeV2ForCausalLM).
            return resolved_arch

    @classmethod
    def _validate_config(cls, config: PretrainedConfig, path: str | None = None) -> None:
        # Check if this is a causal LM model
        if not cls.supports(config):
            architectures = getattr(config, "architectures", [])
            raise ValueError(
                f"\n✗ Model architecture not supported by AutoBridge\n\n"
                f"Model: {path}\n"
                f"Architectures: {architectures}\n\n"
                f"AutoBridge only supports models with architectures ending in {SUPPORTED_HF_ARCHITECTURES_DISPLAY}.\n"
                f"Found architectures that don't match this pattern.\n\n"
                f"If this is a different model type (e.g., Vision, Sequence-to-Sequence),\n"
                f"you may need to use a different bridge class."
            )

        # Check if we have an implementation for this specific architecture
        architecture = None
        for arch_name in config.architectures:
            if arch_name.endswith(SUPPORTED_HF_ARCHITECTURES):
                architecture = arch_name
                break

        if architecture:
            # Try auto_map first; returns a class-name string if available
            arch_name = get_causal_lm_class_name_via_auto_map(config=config)
            if arch_name is not None:
                # For auto_map models, use class-name string
                arch_key = arch_name
            else:
                # Resolve non-standard architecture names via alias mapping
                resolved_arch = HF_ARCHITECTURE_ALIASES.get(architecture, architecture)
                arch_class = getattr(transformers, resolved_arch, None)
                # Fall back to name-based registration when the installed
                # Transformers release does not expose the model class.
                arch_key = arch_class if arch_class is not None else architecture

            # Test if we have a registered implementation (type or class-name string)
            has_implementation = False
            if hasattr(model_bridge.get_model_bridge, "_exact_types"):
                registry = model_bridge.get_model_bridge._exact_types
                if isinstance(arch_key, str):
                    has_implementation = arch_key in registry
                else:
                    has_implementation = (arch_key in registry) or (
                        getattr(arch_key, "__name__", None) in registry
                    )

                if not has_implementation:
                    if architecture in _REMOVED_HF_ARCHITECTURES:
                        raise ValueError(
                            f"Support for {_REMOVED_HF_ARCHITECTURES[architecture]} "
                            f"(architecture '{architecture}') was removed in Megatron Bridge 0.7.0. "
                            "Use Megatron Bridge 0.6.x for this model."
                        )

                    # Get list of supported models
                    supported_models = cls.list_supported_models()

                    raise ValueError(
                        f"\n✗ Model architecture '{architecture}' is not yet supported\n\n"
                        f"Model: {path}\n"
                        f"Architecture: {architecture}\n\n"
                        f"Currently supported architectures:\n"
                        + "\n".join(f"  • {model}" for model in supported_models)
                        + f"\n\nTo add support for {architecture}, you need to:\n"
                        f"1. Create a new bridge class that inherits from MegatronModelBridge\n"
                        f"2. Implement the required methods (provider_bridge, mapping_registry)\n"
                        f"3. Register it with @MegatronModelBridge.register_bridge decorator\n\n"
                        f"Example implementation:\n"
                        f"  from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge\n"
                        f"  from transformers import {architecture}\n"
                        f"  from megatron.core.models.gpt import GPTModel\n\n"
                        f"  @MegatronModelBridge.register_bridge(source={architecture}, target=GPTModel)\n"
                        f"  class Megatron{architecture.replace('ForCausalLM', '')}Bridge(MegatronModelBridge):\n"
                        f"      def provider_bridge(self, hf_pretrained):\n"
                        f"          # Return a ModelProvider instance\n"
                        f"          ...\n\n"
                        f"      def mapping_registry(self):\n"
                        f"          # Return a MegatronMappingRegistry with weight mappings\n"
                        f"          ...\n\n"
                        f"For reference implementations, see:\n"
                        f"  • src/megatron/bridge/models/llama/llama_bridge.py\n"
                        f"  • src/megatron/bridge/models/qwen/qwen_2_causal_bridge.py"
                    ) from None

    def _get_model_instance(self, model: list[MegatronModelT]) -> MegatronModelT:
        model_instance = model[0]
        while hasattr(model_instance, "module"):
            model_instance = model_instance.module
        return model_instance

    def _create_config_from_provider(
        self, source_obj: Any, target_dataclass: Type[DataclassT]
    ) -> DataclassT:
        kwargs = {}
        for field in dataclasses.fields(target_dataclass):
            if hasattr(source_obj, field.name):
                kwargs[field.name] = getattr(source_obj, field.name)
        return target_dataclass(**kwargs)

    @cached_property
    def _config_only_pretrained(self) -> PreTrainedCausalLM:
        if not isinstance(self.hf_pretrained, PretrainedConfig):
            raise ValueError(
                "Config-only shim accessed when hf_pretrained is not a PretrainedConfig instance."
            )
        return _ConfigOnlyPretrainedShim(self.hf_pretrained)

    def __repr__(self) -> str:
        class_name = self.__class__.__name__

        lines_for_build = []

        # Format hf_pretrained
        hf_repr_actual_lines = repr(self.hf_pretrained).splitlines()
        if hf_repr_actual_lines:
            # First line of hf_pretrained part
            lines_for_build.append(f"  (hf_pretrained): {hf_repr_actual_lines[0]}")
            # Subsequent lines of hf_pretrained part, indented
            for line in hf_repr_actual_lines[1:]:
                lines_for_build.append(f"  {line}")
        else:
            lines_for_build.append("  (hf_pretrained): ")  # Fallback for empty repr

        # Format model bridge
        mb_repr_actual_lines = repr(self._model_bridge).splitlines()
        if mb_repr_actual_lines:
            # First line of model bridge part
            lines_for_build.append(f"  (model_bridge): {mb_repr_actual_lines[0]}")
            # Subsequent lines of model bridge part, indented
            for line in mb_repr_actual_lines[1:]:
                lines_for_build.append(f"  {line}")
        else:
            lines_for_build.append("  (model_bridge): ")  # Fallback for empty repr

        return f"{class_name}(\n" + "\n".join(lines_for_build) + "\n)"

    def _validate_fp8_export_config(self, model: list[MegatronModelT]) -> None:
        """Validate runtime Megatron config before enabling FP8 export tasks."""
        model_instance = self._get_model_instance(model)
        model_config = getattr(model_instance, "config", None)
        fp8 = getattr(model_config, "fp8", None)
        fp8_recipe = getattr(model_config, "fp8_recipe", None)
        fp8_param = getattr(model_config, "fp8_param", None)
        if fp8 is None or fp8_recipe not in ("blockwise", "mxfp8") or not fp8_param:
            raise ValueError(
                "export_weight_dtype='fp8' only supports blockwise or MXFP8 parameter export. "
                "Expected fp8 to be enabled, fp8_recipe to be 'blockwise' or 'mxfp8', and fp8_param=True, "
                f"but got fp8={fp8!r}, fp8_recipe={fp8_recipe!r}, fp8_param={fp8_param!r}."
            )
