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

# Adapted from https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/53c20f81cb575f19af34ff0ae1749fb13b07999b/src/megatron/bridge/models/conversion/model_bridge.py
# Source license: Apache-2.0.

import abc
import contextlib
import fnmatch
import gc
import itertools
import logging
import math
import re
import warnings
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Generic,
    Iterable,
    List,
    Literal,
    Mapping,
    NamedTuple,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)

import torch
from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn
from torch.distributed._tensor import DTensor
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel

from megatron.bridge.models.common import ModelConfigOverrideMixin
from megatron.bridge.models.conversion.fp8_export import (
    FP8ExportLayout,
    detect_fp8_export_layout,
    get_fp8_export_tensors,
)
from megatron.bridge.models.conversion.gtp import (
    _gather_gtp_weight,
    _get_mapping_shape,
    _gtp_weight_load_context,
    _is_gtp_param,
    _slice_gtp_weight,
)
from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.param_mapping import (
    LocalHFParam,
    LocalHFParamSpec,
    MegatronParamMapping,
)
from megatron.bridge.models.conversion.quant_bridge import MegatronQuantizationBridge
from megatron.bridge.models.conversion.quant_mapping import AmaxMapping, amax_mapping_hf_names
from megatron.bridge.models.conversion.transformers_compat import (
    partial_rotary_factor_from_hf,
    rope_theta_from_hf,
)
from megatron.bridge.models.conversion.utils import (
    extract_sort_key,
    get_module_and_param_from_name,
    persistent_buffers,
    unwrap_model,
)
from megatron.bridge.models.decorators.dispatch import dispatch
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.model_provider import ModelProviderMixin
from megatron.bridge.models.transformer_config import TransformerConfig as BridgeTransformerConfig
from megatron.bridge.utils import fusions
from megatron.bridge.utils.activation_map import ACTIVATION_FUNC_MAP
from megatron.bridge.utils.common_utils import print_rank_0
from megatron.core import parallel_state
from megatron.core.pipeline_parallel.utils import is_pp_first_stage, is_pp_last_stage
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import MLATransformerConfig, TransformerConfig
from megatron.core.utils import get_pg_size
from megatron.training.models.base import ModelConfig

logger = logging.getLogger(__name__)

MappingT = TypeVar("MappingT", bound=MegatronParamMapping)
HFPreTrained = TypeVar("HFPreTrained")
ModelProviderTarget = TypeVar("ModelProviderTarget", bound=ModelProviderMixin)
MegatronModel = TypeVar("MegatronModel", bound=MegatronModule)
_BridgeImplClass = TypeVar("_BridgeImplClass", bound="MegatronModelBridge")


def _named_conversion_parameters(
    model: torch.nn.Module,
    exclude_module_types: tuple[type[torch.nn.Module], ...] = (),
    *,
    exclude_param_patterns: tuple[str, ...] = (),
) -> Iterable[tuple[str, torch.Tensor]]:
    """Iterate parameters and persistent buffers outside excluded module subtrees."""
    parameters: Iterable[tuple[str, torch.Tensor]] = itertools.chain(
        model.named_parameters(), persistent_buffers(model)
    )
    if exclude_param_patterns:
        parameters = (
            (name, tensor)
            for name, tensor in parameters
            if not any(fnmatch.fnmatchcase(name, pattern) for pattern in exclude_param_patterns)
        )
    if not exclude_module_types:
        yield from parameters
        return

    excluded_names = {
        name for name, module in model.named_modules() if isinstance(module, exclude_module_types)
    }
    for name, tensor in parameters:
        parent_name = name.rpartition(".")[0]
        while parent_name and parent_name not in excluded_names:
            parent_name = parent_name.rpartition(".")[0]
        if parent_name not in excluded_names:
            yield name, tensor


class ModelConfigNotSupportedError(NotImplementedError):
    """Raised when a bridge has no builder-backed model config path."""


class MegatronWeightTuple(NamedTuple):
    """Tuple representing a Megatron model weight with its metadata."""

    param_name: str
    weight: torch.Tensor
    vp_stage: int


class HFWeightTuple(NamedTuple):
    """Tuple representing a HuggingFace model weight with its metadata."""

    param_name: str
    weight: torch.Tensor

    def iter_finalized(
        self,
        *,
        cpu: bool,
        export_hook: Callable[[str, torch.Tensor], Iterable["HFWeightTuple"]] | None = None,
        clone_identity_output: bool = False,
        megatron_param_names: tuple[str, ...] | None = None,
    ) -> Iterable["HFWeightTuple | HFSourcedWeightTuple"]:
        """Apply an optional export hook and yield finalized weights.

        Export hooks run on a detached tensor before final device placement and may
        emit zero, one, or multiple weights. Identity outputs can optionally be
        cloned so independently named tied weights do not share storage.

        Args:
            cpu: Whether to move exported tensors to CPU.
            export_hook: Optional transformation applied before device placement.
            clone_identity_output: Clone an output when it is the detached input.
            megatron_param_names: When given (possibly empty), yield :class:`HFSourcedWeightTuple`
                carrying these source Megatron parameter names next to each exported weight.

        Yields:
            Finalized HuggingFace weights in export-hook order.
        """
        source_tensor = self.weight.detach()
        exported_weights = (
            export_hook(self.param_name, source_tensor)
            if export_hook is not None
            else ((self.param_name, source_tensor),)
        )
        for exported_name, exported_tensor in exported_weights:
            is_identity_output = exported_tensor is source_tensor
            exported_tensor = exported_tensor.detach()
            if clone_identity_output and is_identity_output:
                exported_tensor = exported_tensor.clone().detach()
            exported_tensor = exported_tensor.cpu() if cpu else exported_tensor
            if megatron_param_names is not None:
                yield HFSourcedWeightTuple(exported_name, exported_tensor, megatron_param_names)
            else:
                yield HFWeightTuple(exported_name, exported_tensor)


class HFSourcedWeightTuple(NamedTuple):
    """A :class:`HFWeightTuple` that also names the Megatron parameters it was exported from.

    Only produced when a streaming export is called with ``with_megatron_names=True``;
    the default export keeps yielding plain two-field :class:`HFWeightTuple` values so
    ``for name, weight in ...`` unpacking keeps working for existing callers.

    ``megatron_param_names`` holds the unwrapped local Megatron parameter names of the
    conversion tasks behind the weight, in accumulation order:

    * one name for a weight converted from a single Megatron parameter;
    * one name per contributing per-expert task for a grouped-expert export that packs
      several Megatron parameters into one HF tensor;
    * no names for HF-only passthrough tensors that a bridge copies from the source
      checkpoint without any Megatron counterpart.
    """

    param_name: str
    weight: torch.Tensor
    megatron_param_names: tuple[str, ...]

    @property
    def megatron_param_name(self) -> str | None:
        """The single source Megatron parameter name, or ``None`` for zero or several sources."""
        return self.megatron_param_names[0] if len(self.megatron_param_names) == 1 else None


@dataclass(frozen=True)
class WeightConversionTask(Generic[MappingT]):
    """A unified task for converting weights between HuggingFace and Megatron formats.

    This class combines both HF->Megatron and Megatron->HF conversion tasks since they
    have different method names (hf_to_megatron vs megatron_to_hf) and can coexist safely.

    The task encapsulates all information needed for weight conversion in either direction,
    with different fields being relevant depending on the conversion type.

    Attributes:
        param_name (str): *unwrapped, local* parameter name (no ``module.`` prefixes).
        global_param_name (str): *unwrapped, global* parameter name (no ``module.`` prefixes).
        mapping (MappingT): Concrete :pyclass:`MegatronParamMapping` instance responsible
            for weight transformation and distribution.

        pp_rank (Optional[int]): Pipeline-parallel rank that owns the parameter (required for saves).
        vp_stage (Optional[int]): Virtual-pipeline stage index (required for loads).
        megatron_module (Optional[torch.nn.Module]): Reference to the Megatron model or
            sub-module that owns the parameter (required for loads).
        param_weight (Optional[torch.Tensor]): The actual parameter tensor that will
            receive the converted weight (required for loads).
        weight_dtype (Optional[torch.dtype]): Export only. Cast float weights to this
            dtype; bridges that requantize on export skip it (no scale companions).
        export_hook: Export-only transformation applied after mapping conversion and
            before final device placement.
        required_hf_param_names: Import-only source tensors consumed by the loading
            hook. Defaults to the parameter names declared by ``mapping.hf_param``.

    """

    param_name: str
    global_param_name: str
    mapping: MappingT
    pp_rank: Optional[int] = None
    vp_stage: Optional[int] = None
    megatron_module: Optional[torch.nn.Module] = None
    param_weight: Optional[torch.Tensor] = None
    weight_dtype: Optional[torch.dtype] = None
    export_hook: Optional[Callable[[str, torch.Tensor], Iterable[HFWeightTuple]]] = field(
        default=None, compare=False, repr=False
    )
    required_hf_param_names: tuple[str, ...] | None = field(default=None, compare=False)

    @property
    def hf_param_names(self) -> tuple[str, ...]:
        """HF tensors required to import this Megatron parameter."""
        if self.required_hf_param_names is not None:
            return self.required_hf_param_names
        hf_param = self.mapping.hf_param
        names = (hf_param,) if isinstance(hf_param, str) else tuple(hf_param.values())
        return tuple(dict.fromkeys(names))

    def local_hf_param_specs(self) -> tuple[LocalHFParamSpec, ...]:
        """Return this parameter's canonical local HF-compatible views.

        An empty tuple means the caller must use the normal Bridge conversion
        path; it does not declare the model unsupported for HF conversion or
        M-to-N refit. Destination backend and topology compatibility are outside
        this per-parameter contract, including whether MXFP8 is transferred as
        qualified physical data/scales or requantized from a logical view.
        """
        return self.mapping.local_hf_param_specs(self.global_param_name)

    def combine_local_hf_weights(self, weights: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Reassemble transferred local HF views into this Megatron parameter."""
        specs = self.local_hf_param_specs()
        if not specs:
            raise ValueError(f"{self.param_name!r} has no local HF parameter views.")
        if len(specs) == 1:
            return weights[specs[0].name]

        split_dims = {spec.split_dim for spec in specs}
        if len(split_dims) != 1 or None in split_dims:
            raise ValueError(f"{self.param_name!r} local HF views cannot be combined generically.")
        ordered = sorted(specs, key=lambda spec: spec.split_index)
        return torch.cat([weights[spec.name] for spec in ordered], dim=ordered[0].split_dim)


class _HFNameSuffixMapping:
    """A lightweight wrapper mapping that appends a suffix to all HF param names on export.

    This is used by FP8 scale tasks: the task's `param_name/global_param_name` may carry
    `*_scale_inv`, but the exported output names come from `mapping.megatron_to_hf()`.
    Reusing the base mapping would otherwise emit the original HF weight names (no suffix).
    """

    def __init__(self, base_mapping: Any, suffix: str, scale_block_size: int | None = None):
        self._base_mapping = base_mapping
        self._suffix = suffix
        self.scale_block_size = scale_block_size

    def __getattr__(self, name: str) -> Any:
        # Delegate everything else (e.g., broadcast/gather helpers, megatron_param, etc.)
        return getattr(self._base_mapping, name)

    def resolve(self, captures: Tuple[str, ...]) -> "_HFNameSuffixMapping":
        # Preserve wildcard resolution behavior if the base mapping supports it.
        if hasattr(self._base_mapping, "resolve"):
            return _HFNameSuffixMapping(
                self._base_mapping.resolve(captures), self._suffix, self.scale_block_size
            )
        return _HFNameSuffixMapping(self._base_mapping, self._suffix, self.scale_block_size)

    def hf_to_megatron(self, hf_weights: Any, megatron_module: torch.nn.Module) -> torch.Tensor:
        # Pass-through (not used by our export path, but keeps the wrapper mapping "complete").
        return self._base_mapping.hf_to_megatron(hf_weights, megatron_module)

    def megatron_to_hf(
        self, megatron_weights: Optional[torch.Tensor], megatron_module: Optional[torch.nn.Module]
    ) -> Dict[str, torch.Tensor]:
        scale_export = getattr(self._base_mapping, "megatron_to_hf_scale", None)
        row_block_size = self.scale_block_size
        if callable(scale_export) and isinstance(row_block_size, int):
            out = scale_export(megatron_weights, megatron_module, row_block_size=row_block_size)
        else:
            out = self._base_mapping.megatron_to_hf(megatron_weights, megatron_module)
        if not out:
            return out
        # Append suffix to every exported HF parameter name.
        return {f"{k}{self._suffix}": v for k, v in out.items()}


def _get_pg_collection_from_model(megatron_model: MegatronModule | List[MegatronModule] | None):
    """Return the ProcessGroupCollection attached to a Megatron model, if any.

    Megatron-Core's ``LanguageModule`` (parent of ``GPTModel``) stores the
    ``ProcessGroupCollection`` it was constructed with on ``self.pg_collection``.
    When users wire process groups manually via ``HyperCommGrid`` (decentralized
    PG path) the global ``parallel_state`` singleton is never populated, but the
    same groups are available through the model. Return ``None`` when no model
    is supplied or no ``pg_collection`` is reachable so callers can fall back to
    ``parallel_state``.
    """
    if megatron_model is None:
        return None
    models_list = megatron_model if isinstance(megatron_model, list) else [megatron_model]
    if not models_list:
        return None
    unwrapped = unwrap_model(models_list[0])
    # VLM wrappers expose the language model on ``.language_model``; the
    # ``pg_collection`` lives on the inner ``LanguageModule``.
    inner = getattr(unwrapped, "language_model", None)
    if inner is not None:
        unwrapped = inner
    return getattr(unwrapped, "pg_collection", None)


def _get_pp_group(megatron_model: MegatronModule | List[MegatronModule] | None):
    """Return the pipeline-parallel group, preferring the model's pg_collection."""
    pg = _get_pg_collection_from_model(megatron_model)
    pp_group = getattr(pg, "pp", None) if pg is not None else None
    if pp_group is not None:
        return pp_group
    return parallel_state.get_pipeline_model_parallel_group()


def _get_pp_rank(megatron_model: MegatronModule | List[MegatronModule] | None) -> int:
    """Return the pipeline-parallel rank, preferring the model's pg_collection."""
    pg = _get_pg_collection_from_model(megatron_model)
    pp_group = getattr(pg, "pp", None) if pg is not None else None
    if pp_group is not None:
        return torch.distributed.get_rank(group=pp_group)
    return parallel_state.get_pipeline_model_parallel_rank()


def _get_embedding_group(megatron_model: MegatronModule | List[MegatronModule] | None):
    """Return the embedding group, preferring the model's pg_collection."""
    pg = _get_pg_collection_from_model(megatron_model)
    embd_group = getattr(pg, "embd", None) if pg is not None else None
    if embd_group is not None:
        return embd_group
    return parallel_state.get_embedding_group()


def _get_ep_group(megatron_model: MegatronModule | List[MegatronModule] | None):
    """Return the expert-parallel group, preferring the model's pg_collection.

    Returns ``None`` if the model has a ``pg_collection`` whose ``ep`` is ``None``
    (typical for dense models in the decentralized PG path), so callers must
    treat ``None`` as "EP size 1".
    """
    pg = _get_pg_collection_from_model(megatron_model)
    if pg is not None:
        return getattr(pg, "ep", None)
    return parallel_state.get_expert_model_parallel_group()


def _megatron_local_name_to_global(
    models: MegatronModule | List[MegatronModule],
    config: TransformerConfig,
    param_name: str,
    vp_stage: Optional[int] = None,
) -> str:
    """Adjust layer number and expert number from local to global numbering."""
    # PP
    pp_group = _get_pp_group(models)
    if "layers." in param_name and get_pg_size(pp_group) > 1:
        match = re.match(r"^(.+?\.layers\.\d+)", param_name)
        assert match is not None
        layer_prefix = match.group(1)
        _, layer_module = get_module_and_param_from_name(
            models=models, param_name=layer_prefix, vp_stage=vp_stage
        )

        local_layer_number = int(param_name.split("layers.")[1].split(".")[0])
        if isinstance(layer_module, MegatronModule):
            global_layer_number = layer_module.layer_number - 1
            param_name = param_name.replace(
                f"layers.{local_layer_number}.", f"layers.{global_layer_number}."
            )

    # EP — fetched lazily because dense models may not have an EP group at all
    # (and for the decentralized PG path, ``pg_collection.ep`` may be ``None``).
    is_grouped_expert_param = ".experts.linear_fc" in param_name
    is_local_expert_param = ".experts.local_experts." in param_name
    is_expert_param = is_grouped_expert_param or is_local_expert_param
    ep_group = _get_ep_group(models) if is_expert_param else None
    if is_expert_param and ep_group is not None and get_pg_size(ep_group) > 1:
        num_experts = config.num_moe_experts
        num_experts_per_rank = num_experts // ep_group.size()

        def _update_grouped_expert_number(param_name: str, param_type: str) -> str:
            """Update expert number from local to global for weight or bias parameters."""
            expert_match = re.search(rf"\.{param_type}(\d+)(?=$|\.)", param_name)
            if expert_match is None:
                return param_name
            local_expert_number = int(expert_match.group(1))
            global_expert_number = num_experts_per_rank * ep_group.rank() + local_expert_number
            return f"{param_name[: expert_match.start(1)]}{global_expert_number}{param_name[expert_match.end(1) :]}"

        if is_local_expert_param:
            local_experts_match = re.search(r"\.local_experts\.(\d+)\.", param_name)
            if local_experts_match:
                local_expert_number = int(local_experts_match.group(1))
                global_expert_number = num_experts_per_rank * ep_group.rank() + local_expert_number
                param_name = param_name.replace(
                    f".local_experts.{local_expert_number}.",
                    f".local_experts.{global_expert_number}.",
                    1,
                )
        # Grouped MoE uses numeric suffixes for per-expert weight and bias
        # parameters, e.g. .weight0 and .bias1. Shared quantizer buffers such
        # as .weight_quantizer._amax must not go through EP renumbering.
        elif re.search(r"\.weight\d+(?=$|\.)", param_name):
            param_name = _update_grouped_expert_number(param_name, "weight")
        elif re.search(r"\.bias\d+(?=$|\.)", param_name):
            param_name = _update_grouped_expert_number(param_name, "bias")

    return param_name


class MegatronModelBridge(
    MegatronQuantizationBridge, Generic[HFPreTrained, ModelProviderTarget, MegatronModel]
):
    """
    High-level orchestrator for HuggingFace ↔ Megatron model conversions.

    This abstract base class provides the framework for converting models between
    HuggingFace and Megatron formats. It acts as an orchestrator that coordinates
    the conversion process without directly handling the complex details of
    tensor parallelism or weight transformations.

    The bridge pattern separates concerns:
    - MegatronModelBridge: Orchestrates the overall conversion process
    - MegatronMappingRegistry: Manages parameter name mappings
    - MegatronParamMapping: Handles actual weight transformations and distribution

    Key responsibilities:
    1. Build conversion tasks that map each parameter to its appropriate bridge
    2. Execute tasks with proper error handling and progress tracking
    3. Provide utilities for configuration translation
    4. Handle virtual pipeline parallelism (VP) complexities

    To implement a bridge for a new model architecture:

    1. Create a subclass decorated with @MegatronModelBridge.register_bridge:

        .. code-block:: python

            @MegatronModelBridge.register_bridge(source=LlamaForCausalLM, target=GPTModel)
            class MegatronCausalLlamaBridge(MegatronModelBridge):
                pass

    2. Implement provider_bridge to create Megatron configurations:

        .. code-block:: python

            def provider_bridge(self, hf_pretrained) -> GPTModelProvider:
                return GPTModelProvider(
                    num_layers=hf_pretrained.config.num_hidden_layers,
                    hidden_size=hf_pretrained.config.hidden_size,
                    ...
                )

    3. Implement mapping_registry to define weight mappings:

        .. code-block:: python

            def mapping_registry(self) -> MegatronMappingRegistry:
                return MegatronMappingRegistry(
                    AutoMapping(
                        megatron_param="embedding.word_embeddings.weight",
                        hf_param="model.embed_tokens.weight"
                    ),
                    ...
                )

    Example:
        .. code-block:: python

            # The bridge is typically not instantiated directly
            # Instead, use AutoBridge or AutoBridge which handle this
            bridge = AutoBridge.from_hf_pretrained("meta-llama/Meta-Llama-3-8B")
            provider = bridge.to_megatron_provider()

    Note:
        This class uses generic type parameters to ensure type safety:
        - HFPreTrained: The HuggingFace model type
        - ModelProviderTarget: The Megatron model provider type
        - MegatronModel: The Megatron model type
    """

    # These are extension points, not a setup checklist. Most new model bridges should keep all defaults below.

    # Set to False only when this bridge cannot produce a standalone Hugging Face checkpoint.
    SUPPORTS_HF_PRETRAINED_EXPORT: ClassVar[bool] = True

    # None selects GPTModelProvider. Pass provider=... to @register_bridge only for a specialized
    # provider, such as one implementing MLA.
    PROVIDER_CLASS = None

    # Builder-backed construction is rolling out incrementally by model family, so it is opt-in.
    # None means the family has not migrated: get_model_config() raises and the provider path
    # stays the supported one. A family that has mapped its Hugging Face config for the builder
    # path sets this to BridgeGPTModelConfig, or to a specialized config when the standard GPT
    # model config cannot represent the model. Subclasses of a migrated bridge inherit the
    # opt-in. The selected model config owns its nested transformer config class.
    MODEL_CONFIG_CLASS: ClassVar[type[ModelConfig] | None] = None

    # Conversion still uses the provider-backed construction path unless a model family has
    # explicitly migrated its conversion lifecycle to ModelBuilder. Keeping this separate
    # from MODEL_CONFIG_CLASS lets a family expose get_model_config() without silently
    # changing how its checkpoints are constructed. Requires MODEL_CONFIG_CLASS to be set.
    USE_MODEL_CONFIG_FOR_CONVERSION: ClassVar[bool] = False

    # Source modules intentionally absent from every target of this bridge.
    # Strict export still requires all other source tensors. Prefixes must name
    # complete modules (nonempty, ending in a dot), never a wildcard or all keys.
    HF_EXPORT_IGNORED_SOURCE_PREFIXES: ClassVar[tuple[str, ...]] = ()

    # Leave unset unless HF export must copy nonstandard files in addition to the usual artifacts,
    # for example ``["*reasoning_parser.py"]``.
    ADDITIONAL_FILE_PATTERNS = None

    # Source checkpoint key suffixes that import consumes and export never writes back, such as the
    # ``*_scale_inv`` companions of FP8 weights a bridge dequantizes on import. Saved artifacts drop
    # ``quantization_config``, so such a bridge exports plain weights; strict saving then ignores
    # these source keys instead of reporting them as missing.
    DEQUANTIZED_SOURCE_KEY_SUFFIXES: ClassVar[tuple[str, ...]] = ()

    def postprocess_hf_export_artifacts(self, path: Path) -> None:
        """Apply model-specific fixes after Hugging Face artifacts are saved.

        Args:
            path: Directory containing the exported Hugging Face artifacts.
        """

    def postprocess_hf_export_weights(self, path: Path) -> None:
        """Apply model-specific fixes after Hugging Face weights are saved.

        Args:
            path: Directory containing the complete exported Hugging Face checkpoint.
        """

    # HuggingFace PretrainedConfig, set by register_bridge_implementation dispatch.
    # Available in mapping_registry(), stream_weights_*(), and build_conversion_tasks().
    hf_config = None

    # Whether the Hugging Face source may run code from its repository. AutoBridge sets it from its
    # source, so a bridge whose model loads modules from the checkpoint can record it on the model config.
    trust_remote_code: bool = False

    # Optional MIMO conversion metadata. Subclasses set this when the standard
    # bridge registry can be split into MIMO component routes.
    mimo_source_prefixes: ClassVar[Mapping[str, str] | None] = None

    # Common bidirectional config field name mapping: (hf_name, megatron_name)
    # Some mappings may not be used by all models - that's fine, unused fields are skipped
    CONFIG_MAPPING = [
        # Core architecture
        ("num_hidden_layers", "num_layers"),
        ("hidden_size", "hidden_size"),
        ("intermediate_size", "ffn_hidden_size"),
        ("num_attention_heads", "num_attention_heads"),
        ("num_key_value_heads", "num_query_groups"),
        ("head_dim", "kv_channels"),
        ("vocab_size", "vocab_size"),
        ("max_position_embeddings", "seq_length"),
        ("rms_norm_eps", "layernorm_epsilon"),
        ("initializer_range", "init_method_std"),
        # Attention and dropout
        ("attention_dropout", "attention_dropout"),
        ("hidden_dropout", "hidden_dropout"),
        ("tie_word_embeddings", "share_embeddings_and_output_weights"),
        ("attention_bias", "add_qkv_bias"),
        ("mlp_bias", "add_bias_linear"),
        ("use_qk_norm", "qk_layernorm"),
        # RoPE
        ("rope_theta", "rotary_base"),
        ("partial_rotary_factor", "rotary_percent"),
        # MoE
        ("num_experts", "num_moe_experts"),
        ("num_local_experts", "num_moe_experts"),
        ("num_experts_per_tok", "moe_router_topk"),
        ("moe_intermediate_size", "moe_ffn_hidden_size"),
        ("aux_loss_alpha", "moe_aux_loss_coeff"),
        ("router_aux_loss_coef", "moe_aux_loss_coeff"),
        ("scoring_func", "moe_router_score_function"),
        ("n_routed_experts", "num_moe_experts"),
        ("n_group", "moe_router_num_groups"),
        ("topk_group", "moe_router_group_topk"),
        ("routed_scaling_factor", "moe_router_topk_scaling_factor"),
        # MLA
        ("q_lora_rank", "q_lora_rank"),
        ("kv_lora_rank", "kv_lora_rank"),
        ("qk_nope_head_dim", "qk_head_dim"),
        ("qk_rope_head_dim", "qk_pos_emb_head_dim"),
        ("v_head_dim", "v_head_dim"),
        # MTP
        ("num_nextn_predict_layers", "mtp_num_layers"),
        ("mtp_num_hidden_layers", "mtp_num_layers"),
    ]

    # YARN rope scaling field mapping for GPT models: (hf_rope_scaling_key, megatron_yarn_param)
    # These are only applied when rope_scaling.type == "yarn" and provider is GPTModelProvider
    # Uses yarn_ prefix (e.g., yarn_mscale, yarn_rotary_scaling_factor)
    YARN_ROPE_SCALING_MAPPING = [
        ("factor", "yarn_rotary_scaling_factor"),
        ("original_max_position_embeddings", "yarn_original_max_position_embeddings"),
        ("beta_fast", "yarn_beta_fast"),
        ("beta_slow", "yarn_beta_slow"),
        ("mscale", "yarn_mscale"),
        ("mscale_all_dim", "yarn_mscale_all_dim"),
    ]

    # Hugging Face YaRN defaults (transformers' _compute_yarn_parameters) for keys a
    # checkpoint omits: (megatron_param, default)
    YARN_HF_DEFAULTS = [
        ("yarn_beta_fast", 32.0),
        ("yarn_beta_slow", 1.0),
        ("yarn_correction_range_round_to_int", True),
    ]

    # MLA rope scaling field mapping: (hf_rope_scaling_key, megatron_mla_param)
    # These are applied for MLA models (DeepSeek, Kimi, etc.) which use MLATransformerConfig
    # Uses direct field names without yarn_ prefix (e.g., mscale, rotary_scaling_factor)
    MLA_ROPE_SCALING_MAPPING = [
        ("factor", "rotary_scaling_factor"),
        ("original_max_position_embeddings", "original_max_position_embeddings"),
        ("beta_fast", "beta_fast"),
        ("beta_slow", "beta_slow"),
        ("mscale", "mscale"),
        ("mscale_all_dim", "mscale_all_dim"),
    ]

    @classmethod
    def hf_to_megatron_activation(cls, hidden_act: str):
        """Convert HF activation name string to Megatron activation function."""
        if hidden_act not in ACTIVATION_FUNC_MAP:
            raise ValueError(
                f"Unsupported activation function: {hidden_act}. Supported: {list(ACTIVATION_FUNC_MAP.keys())}"
            )
        return ACTIVATION_FUNC_MAP[hidden_act]

    @classmethod
    def megatron_to_hf_activation(cls, activation_func) -> str:
        """Convert Megatron activation function to HF activation name string."""
        for hf_name, megatron_func in ACTIVATION_FUNC_MAP.items():
            if activation_func is megatron_func:
                return hf_name
        raise ValueError(
            f"Unsupported activation function: {activation_func}. Supported: {list(ACTIVATION_FUNC_MAP.values())}"
        )

    def _should_map_hf_config_field(
        self, hf_config: Any, hf_name: str, megatron_name: str, value: Any
    ) -> bool:
        """Return whether an HF config field should be mapped to provider kwargs."""
        return True

    def _hf_config_to_megatron_kwargs(self, hf_config: PretrainedConfig) -> dict[str, Any]:
        """Convert an HF config to flat Megatron config kwargs.

        Args:
            hf_config: HuggingFace model configuration object

        Returns:
            Megatron fields derived from ``CONFIG_MAPPING``.
        """
        provider_kwargs = {}

        # Map config fields using CONFIG_MAPPING
        # Supports dot notation for nested dict access (e.g., "rope_scaling.factor")
        for hf_name, megatron_name in self.CONFIG_MAPPING:
            has_value = False
            value = None
            if "." in hf_name:
                # Nested dict access: "parent.child" -> getattr(config, parent).get(child)
                parts = hf_name.split(".", 1)
                parent = getattr(hf_config, parts[0], None)
                if parent is not None and isinstance(parent, dict):
                    if parts[1] in parent:
                        value = parent[parts[1]]
                        has_value = True
            else:
                value = getattr(hf_config, hf_name, None)
                has_value = hasattr(hf_config, hf_name)
                if not has_value and hf_name == "partial_rotary_factor":
                    # Transformers 5.20+ keeps partial_rotary_factor only in rope_parameters.
                    try:
                        value, has_value = partial_rotary_factor_from_hf(hf_config), True
                    except ValueError:
                        pass
            if (
                has_value
                and megatron_name not in provider_kwargs
                and self._should_map_hf_config_field(hf_config, hf_name, megatron_name, value)
            ):
                provider_kwargs[megatron_name] = value

        # Extract rotary_base via compat function (handles both legacy rope_theta
        # attribute and transformers 5.0+ rope_parameters dict)
        if "rotary_base" not in provider_kwargs:
            try:
                provider_kwargs["rotary_base"] = rope_theta_from_hf(hf_config)
            except ValueError:
                pass

        # Handle rope scaling: extract params from rope_scaling dict
        # HF configs use either "type" or "rope_type" key for the scaling type
        from megatron.bridge.models.mla_provider import MLAModelProvider

        # MLA rope fields belong to an MLA provider or to a model config that nests an MLA transformer config.
        model_transformer_config_class = getattr(
            self.MODEL_CONFIG_CLASS, "transformer_config_class", None
        )
        is_mla = (
            self.PROVIDER_CLASS is not None and issubclass(self.PROVIDER_CLASS, MLAModelProvider)
        ) or (
            isinstance(model_transformer_config_class, type)
            and issubclass(model_transformer_config_class, MLATransformerConfig)
        )
        rope_scaling = getattr(hf_config, "rope_scaling", None)

        rope_type = None
        if rope_scaling is not None and isinstance(rope_scaling, dict) and rope_scaling != {}:
            rope_type = rope_scaling.get("type") or rope_scaling.get("rope_type")

        # MLA attention implements DeepSeek's YaRN, which some MLA checkpoints (Sarvam MLA) name "deepseek_yarn".
        if rope_type == "yarn" or (is_mla and rope_type == "deepseek_yarn"):
            # MLA configs use direct field names; GPT configs use yarn_ prefixed field names
            if is_mla:
                # MLA models: use direct field names (mscale, rotary_scaling_factor, etc.)
                mla_params = {}
                for hf_key, megatron_key in self.MLA_ROPE_SCALING_MAPPING:
                    value = rope_scaling.get(hf_key)
                    if value is not None:
                        mla_params[megatron_key] = value
                if mla_params:
                    provider_kwargs["_mla_rope_params"] = mla_params
            else:
                # GPT models: use yarn_ prefixed field names (dataclass fields on GPTModelProvider)
                provider_kwargs["position_embedding_type"] = "yarn"
                for hf_key, megatron_key in self.YARN_ROPE_SCALING_MAPPING:
                    value = rope_scaling.get(hf_key)
                    if value is not None:
                        provider_kwargs[megatron_key] = value
                if "truncate" in rope_scaling:
                    provider_kwargs["yarn_correction_range_round_to_int"] = rope_scaling["truncate"]
                # Apply Hugging Face YaRN defaults for omitted keys. Megatron Core passes these
                # fields to YarnRotaryEmbedding explicitly, so None would override its defaults.
                for megatron_key, default in self.YARN_HF_DEFAULTS:
                    provider_kwargs.setdefault(megatron_key, default)
        elif is_mla:
            if rope_type not in (None, "default"):
                logger.warning(
                    f"Unsupported {rope_type=} for MLA model; "
                    "defaulting to no scaling (rotary_scaling_factor=1.0, mscale_all_dim=1.0). "
                    "Add explicit handling for this rope_type if the model requires it."
                )
            # Fill in 1.0 for any missing keys to avoid Megatron's MLATransformerConfig defaults
            # (rotary_scaling_factor=40, mscale_all_dim=0.0), which are hardcoded for DeepSeek-V3's
            # YaRN config and are wrong for models without yarn scaling.
            provider_kwargs["_mla_rope_params"] = {
                "rotary_scaling_factor": 1.0,
                "mscale_all_dim": 1.0,
            }

        # Handle vocab_size_divisible_by
        vocab_size = provider_kwargs.get("vocab_size")
        if vocab_size is not None:
            provider_kwargs["make_vocab_size_divisible_by"] = self.make_vocab_size_divisible_by(
                vocab_size
            )

        # Determine dtype
        params_dtype = self.dtype_from_hf(hf_config, default=torch.float32)
        provider_kwargs["fp16"] = params_dtype == torch.float16
        provider_kwargs["bf16"] = params_dtype == torch.bfloat16
        provider_kwargs["params_dtype"] = params_dtype

        # Convert activation function (some models use hidden_act, others use hidden_activation)
        hidden_act = getattr(hf_config, "hidden_act", None) or getattr(
            hf_config, "hidden_activation", "silu"
        )
        provider_kwargs["activation_func"] = self.hf_to_megatron_activation(hidden_act)

        return provider_kwargs

    def hf_config_to_provider_kwargs(self, hf_config: PretrainedConfig) -> dict[str, Any]:
        """Return kwargs for the legacy provider compatibility path."""
        return self._hf_config_to_megatron_kwargs(hf_config)

    def hf_config_to_model_config_kwargs(self, hf_config: PretrainedConfig) -> dict[str, Any]:
        """Return flat kwargs for a builder-backed model config.

        Model-family bridges should call ``super()`` and add their architecture
        defaults here. This path intentionally does not call the provider-era
        override hook.
        """
        config_kwargs = self._hf_config_to_megatron_kwargs(hf_config)
        mla_rope_params = config_kwargs.pop("_mla_rope_params", None)
        if mla_rope_params is not None:
            # MLA transformer configs declare the rotary scaling fields, and MLA attention applies
            # RoPE itself, so the model keeps "rope", as provider_bridge() sets for MLA providers.
            config_kwargs.update(mla_rope_params)
            config_kwargs["position_embedding_type"] = "rope"
        config_kwargs.update(
            attention_softmax_in_fp32=False,
            cross_entropy_loss_fusion=True,
            deallocate_pipeline_outputs=True,
            gradient_accumulation_fusion=fusions.can_enable_gradient_accumulation_fusion(),
        )
        return config_kwargs

    @staticmethod
    def _partition_model_config_kwargs(
        config_kwargs: Mapping[str, Any],
        model_config_class: type[ModelConfig],
        transformer_config_class: type[BridgeTransformerConfig],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Partition flat kwargs between outer and nested config dataclasses."""
        if not is_dataclass(model_config_class):
            raise TypeError(
                f"Model config class {model_config_class.__name__} must be a dataclass."
            )
        if not is_dataclass(transformer_config_class):
            raise TypeError(
                f"Transformer config class {transformer_config_class.__name__} must be a dataclass."
            )

        model_fields = {field.name for field in fields(model_config_class) if field.init} - {
            "transformer"
        }
        transformer_fields = {
            field.name for field in fields(transformer_config_class) if field.init
        }
        unknown_fields = set(config_kwargs) - model_fields - transformer_fields
        if unknown_fields:
            raise ValueError(
                "Cannot map Hugging Face config fields to "
                f"{model_config_class.__name__} + {transformer_config_class.__name__}: "
                f"{sorted(unknown_fields)}. Override hf_config_to_model_config() for model-specific fields."
            )

        model_kwargs = {
            name: value for name, value in config_kwargs.items() if name in model_fields
        }
        transformer_kwargs = {
            name: value
            for name, value in config_kwargs.items()
            if name not in model_fields and name in transformer_fields
        }
        return model_kwargs, transformer_kwargs

    def hf_config_to_model_config(self, hf_config: PretrainedConfig) -> ModelConfig:
        """Convert a Hugging Face config directly to a builder-backed config.

        This path is opt-in per model family. ``MODEL_CONFIG_CLASS`` defaults to
        ``None``; a migrated family sets it to ``BridgeGPTModelConfig`` or to a
        specialized config class.

        Args:
            hf_config: Hugging Face architecture configuration.

        Returns:
            Serializable model configuration linked to its builder.

        Raises:
            ModelConfigNotSupportedError: If the model family has not migrated to
                builder-backed configs (``MODEL_CONFIG_CLASS`` is ``None``).
            ValueError: If mapped fields do not belong to either config dataclass.
        """
        model_config_class = self.MODEL_CONFIG_CLASS
        if model_config_class is None:
            raise ModelConfigNotSupportedError(
                f"ModelConfig conversion is not implemented for {type(self).__name__}. "
                "This model family has not migrated to builder-backed configs "
                "(MODEL_CONFIG_CLASS is None); use AutoBridge.to_megatron_provider() instead."
            )

        config_kwargs = self.hf_config_to_model_config_kwargs(hf_config)
        if not issubclass(model_config_class, ModelConfigOverrideMixin):
            raise TypeError(
                f"{model_config_class.__name__} must inherit {ModelConfigOverrideMixin.__name__}."
            )
        model_config_with_overrides = cast(type[ModelConfigOverrideMixin], model_config_class)
        transformer_config_class = model_config_with_overrides.transformer_config_class
        if not isinstance(transformer_config_class, type) or not issubclass(
            transformer_config_class, BridgeTransformerConfig
        ):
            raise TypeError(
                f"{model_config_class.__name__}.transformer_config_class must be a "
                f"{BridgeTransformerConfig.__name__} subclass."
            )
        model_kwargs, transformer_kwargs = self._partition_model_config_kwargs(
            config_kwargs, model_config_class, transformer_config_class
        )
        transformer_config = transformer_config_class(**transformer_kwargs)
        return model_config_class(transformer=transformer_config, **model_kwargs)

    # Set by @register_bridge decorator
    SOURCE_NAME: str | None = None
    MODEL_TYPE: str | None = None

    def text_only_pretrained(self, hf_pretrained: PreTrainedCausalLM) -> PreTrainedCausalLM:
        """Select a standalone language checkpoint view for an opted-in family.

        Implementations must preserve language weights and tokenizer semantics
        and return a config registered with the corresponding text bridge.
        """
        raise ValueError(f"{type(self).__name__} does not support text_only=True.")

    def provider_bridge(self, hf_pretrained: HFPreTrained) -> ModelProviderTarget:
        """Create a Megatron model provider from HuggingFace configuration.

        Default implementation that:
        1. Converts HF config to provider kwargs using CONFIG_MAPPING
        2. Creates and returns a GPTModelProvider

        Subclasses should override this to add model-specific configuration
        by calling super().provider_bridge() then setting properties directly
        on the returned provider (e.g., provider.normalization = "RMSNorm").

        Args:
            hf_pretrained (HFPreTrained): HuggingFace model or configuration
                containing the source model's architecture details.

        Returns:
            ModelProviderTarget: A configured model provider instance
        """
        from megatron.bridge.models.gpt_provider import GPTModelProvider
        from megatron.bridge.models.mla_provider import MLAModelProvider

        if self.MODEL_CONFIG_CLASS is not None:
            warnings.warn(
                f"{type(self).__name__}.provider_bridge() and AutoBridge.to_megatron_provider() "
                "are deprecated for builder-backed models. Use AutoBridge.get_model_config() "
                "for training recipes or AutoBridge.get_model() to construct a model. "
                "Legacy provider support will be removed in a future release.",
                FutureWarning,
                stacklevel=2,
            )

        hf_config = hf_pretrained.config

        # Build base provider kwargs using CONFIG_MAPPING
        provider_kwargs = self.hf_config_to_provider_kwargs(hf_config)

        mla_rope_params = provider_kwargs.pop("_mla_rope_params", None)

        # Use specified provider class, defaulting to GPTModelProvider
        provider_class = (
            self.PROVIDER_CLASS if self.PROVIDER_CLASS is not None else GPTModelProvider
        )
        is_mla_provider = issubclass(provider_class, MLAModelProvider)
        # Filter kwargs to only fields the provider dataclass accepts, so that MLA-only None
        # values (q_lora_rank, kv_lora_rank, …) are silently dropped for non-MLA providers
        # while still being passed through for MLA providers that declare them as fields.
        valid_fields = provider_class.__dataclass_fields__
        provider = provider_class(**{k: v for k, v in provider_kwargs.items() if k in valid_fields})

        # Determine position_embedding_type from HF rope_scaling.
        # For GPT providers: rope_type=="yarn" → "yarn"; otherwise the type the bridge's
        # kwargs select (e.g. "none" for Nemotron-H), and "rope" when they select none.
        # For MLA providers: always "rope" — YaRN scaling parameters are applied
        # separately via mla_rope_params (rotary_scaling_factor, mscale, etc.) and
        # position_embedding_type="yarn" is not a valid MLA config value.
        hf_rope_scaling = getattr(hf_config, "rope_scaling", None)
        rope_type = None
        if hf_rope_scaling:
            rope_type = hf_rope_scaling.get("type") or hf_rope_scaling.get("rope_type")
        position_embedding_type = provider_kwargs.get("position_embedding_type")
        if is_mla_provider:
            provider.position_embedding_type = "rope"
        elif rope_type == "yarn" or position_embedding_type == "yarn":
            provider.position_embedding_type = "yarn"
        else:
            provider.position_embedding_type = position_embedding_type or "rope"

        # Apply MLA rope params via setattr (for MLA models like DeepSeek, Kimi)
        if mla_rope_params:
            for key, value in mla_rope_params.items():
                setattr(provider, key, value)

        return provider

    @classmethod
    def megatron_to_hf_config(cls, provider) -> dict:
        """Convert Megatron provider config to HuggingFace config dict.

        Default implementation that:
        1. Converts provider to HF config using CONFIG_MAPPING
        2. Handles YARN rope scaling parameters
        3. Converts activation function and dtype
        4. Adds architectures and model_type from decorator

        Subclasses should override this to add model-specific configuration
        by calling super().megatron_to_hf_config() then setting values directly
        on the returned dict (e.g., hf_config["rope_scaling"] = {...}).

        Args:
            provider: Megatron model provider instance

        Returns:
            dict: HuggingFace config dictionary
        """
        hf_config = {}

        # Map config fields using CONFIG_MAPPING (reverse direction)
        # Supports dot notation for nested dict building (e.g., "rope_scaling.factor")
        for hf_name, megatron_name in cls.CONFIG_MAPPING:
            has_value = hasattr(provider, megatron_name)
            value = getattr(provider, megatron_name, None)
            if has_value:
                if "." in hf_name:
                    # Nested dict: "parent.child" -> hf_config["parent"]["child"] = value
                    parts = hf_name.split(".", 1)
                    if parts[0] not in hf_config:
                        hf_config[parts[0]] = {}
                    hf_config[parts[0]][parts[1]] = value
                else:
                    hf_config[hf_name] = value

        # Handle YARN rope scaling: check if provider has yarn_* params and build rope_scaling dict
        yarn_rotary_scaling_factor = getattr(provider, "yarn_rotary_scaling_factor", None)
        if yarn_rotary_scaling_factor is not None:
            if "rope_scaling" not in hf_config:
                hf_config["rope_scaling"] = {}
            hf_config["rope_scaling"]["rope_type"] = "yarn"

            for hf_key, megatron_key in cls.YARN_ROPE_SCALING_MAPPING:
                value = getattr(provider, megatron_key, None)
                if value is not None:
                    hf_config["rope_scaling"][hf_key] = value

            yarn_correction_range_round_to_int = getattr(
                provider, "yarn_correction_range_round_to_int", None
            )
            if yarn_correction_range_round_to_int is not None:
                hf_config["rope_scaling"]["truncate"] = yarn_correction_range_round_to_int

        # Convert activation function back to HF format
        activation_func = getattr(provider, "activation_func", None)
        if activation_func is not None:
            hf_config["hidden_act"] = cls.megatron_to_hf_activation(activation_func)

        # Determine torch_dtype
        if getattr(provider, "bf16", False):
            hf_config["torch_dtype"] = "bfloat16"
        elif getattr(provider, "fp16", False):
            hf_config["torch_dtype"] = "float16"
        else:
            hf_config["torch_dtype"] = "float32"

        # Add architectures and model_type from decorator
        if cls.SOURCE_NAME is not None:
            hf_config["architectures"] = [cls.SOURCE_NAME]
        if cls.MODEL_TYPE is not None:
            hf_config["model_type"] = cls.MODEL_TYPE

        return hf_config

    @abc.abstractmethod
    def mapping_registry(self) -> MegatronMappingRegistry:
        """Define weight mappings between HuggingFace and Megatron formats.

        This abstract method must be implemented by subclasses to specify how
        parameters map between the two formats. The returned MegatronMappingRegistry
        contains all param mappings needed for the model architecture.

        Returns:
            MegatronMappingRegistry: MegatronMappingRegistry containing all weight
                mapping definitions.

        Example:
            .. code-block:: python

                def mapping_registry(self):
                    return MegatronMappingRegistry(
                        AutoMapping(
                            megatron_param="embedding.word_embeddings.weight",
                            hf_param="model.embed_tokens.weight"
                        ),
                        QKVMapping(
                            megatron_param="decoder.layers.*.self_attention.linear_qkv.weight",
                            q="model.layers.*.self_attn.q_proj.weight",
                            k="model.layers.*.self_attn.k_proj.weight",
                            v="model.layers.*.self_attn.v_proj.weight"
                        ),
                        # ... more param mappings
                    )
        """
        raise NotImplementedError("Subclass must implement mapping_registry method")

    @staticmethod
    def _prefix_mapping_registry(
        registry: MegatronMappingRegistry, megatron_prefix: str, *extra: MegatronParamMapping
    ) -> MegatronMappingRegistry:
        """Re-key a text-model mapping registry for a VLM wrapper.

        VLM bridges typically reuse a text bridge's ``mapping_registry`` but must
        re-prefix every Megatron parameter (e.g. ``embedding.*`` →
        ``language_model.embedding.*``) because the language model is a submodule of
        the VLM wrapper. Concatenation/delegation mappings that hold a nested
        ``_tp_mapping`` keep it in sync with the parent ``megatron_param``.

        Args:
            registry: The text-model registry returned by ``super().mapping_registry()``.
            megatron_prefix: Prefix prepended to each ``megatron_param``.
            *extra: Additional mappings appended verbatim (e.g. replicated vision
                weights); they are NOT prefixed.

        Returns:
            A new ``MegatronMappingRegistry`` with prefixed text mappings + extras.
        """
        mappings = list(registry.mappings)
        for mapping in mappings:
            mapping.megatron_param = megatron_prefix + mapping.megatron_param
            nested = getattr(mapping, "_tp_mapping", None)
            if nested is not None:
                nested.megatron_param = mapping.megatron_param
        return MegatronMappingRegistry(*mappings, *extra)

    def _megatron_global_param_names_all_pp_ranks(
        self,
        megatron_model: Union[MegatronModel, List[MegatronModel]],
        *,
        exclude_module_types: tuple[type[torch.nn.Module], ...] = (),
        exclude_param_patterns: tuple[str, ...] = (),
    ) -> List[str]:
        """Get all parameter names across all pipeline parallel ranks."""
        # Filtered discovery must neither reuse nor replace ordinary conversion's cache.
        if (
            not exclude_module_types
            and not exclude_param_patterns
            and hasattr(self, "_cached_param_names")
            and hasattr(self, "_cached_param_owners")
        ):
            return self._cached_param_names

        # Compute the result
        pp_group = _get_pp_group(megatron_model)
        model_config = unwrap_model(megatron_model)[0].config
        global_param_names = []

        # Ensure megatron_model is a list for consistent handling
        models_list = megatron_model if isinstance(megatron_model, list) else [megatron_model]

        for vp_stage, model in enumerate(models_list):
            for local_param_name, _ in _named_conversion_parameters(
                model, exclude_module_types, exclude_param_patterns=exclude_param_patterns
            ):
                if "_extra_state" in local_param_name:
                    continue
                local_param_name = self._unwrap_name(local_param_name)
                global_param_name = _megatron_local_name_to_global(
                    models_list, model_config, local_param_name, vp_stage
                )
                if self._should_skip_mtp_duplicate_embedding_export(global_param_name, model):
                    continue
                global_param_names.append(global_param_name)

        gathered_global_param_names = [None] * pp_group.size()
        torch.distributed.all_gather_object(
            gathered_global_param_names, global_param_names, group=pp_group
        )

        # Record the lowest PP rank that physically owns every parameter. Some
        # models legitimately duplicate a logical parameter across stages
        # (e.g. MTP/MLA/shared embeddings). Full export already chooses the
        # lowest owner as its broadcast source; PP-local export uses the same
        # canonical owner so stage streams remain non-overlapping.
        param_owners = {}
        for owner_rank, owner_names in enumerate(gathered_global_param_names):
            for name in owner_names:
                param_owners.setdefault(name, owner_rank)

        # Flatten the list, sort it and remove duplicates.
        # the order matters here, casually re-order will cause a hang.
        # e.g. decoder.layers.0.mlp.experts.linear_fc1.weight100
        flattened_names = list(param_owners)

        # the order cannot be changed, this sync for all ranks for conversion
        # change this might cause a hang
        gathered_global_param_names = sorted(flattened_names, key=extract_sort_key)

        if not exclude_module_types and not exclude_param_patterns:
            self._cached_param_names = gathered_global_param_names
            self._cached_param_owners = param_owners

        return gathered_global_param_names

    def build_modelopt_quantizer_recipe(
        self, hf_pretrained: HFPreTrained, megatron_model: Union[MegatronModel, List[MegatronModel]]
    ) -> dict[str, dict[str, Any]]:
        """Map each enabled ModelOpt quantizer to its HF names with its properties-only state.

        Names follow the same amax mappings (and q/k/v, gate/up, per-expert fan-out) as the
        exported ``_amax`` tensors. Collective: every rank must call it.

        Args:
            hf_pretrained: HF model or config whose checkpoint layout selects the mappings.
            megatron_model: Megatron model instance or list of instances.

        Raises:
            ValueError: If an enabled quantizer has no HF mapping or holds tensors other than
                ``_amax`` (which are not exported), a disabled quantizer still rotates or scales its
                input (which the recipe cannot carry), or ranks disagree on the state of a quantizer
                exported under the same HF name.
        """
        from modelopt.torch.quantization.nn import TensorQuantizer

        models_list = megatron_model if isinstance(megatron_model, list) else [megatron_model]
        model_config = unwrap_model(models_list)[0].config
        self.hf_pretrained = hf_pretrained
        self.hf_config = hf_pretrained.config if hasattr(hf_pretrained, "config") else hf_pretrained
        registry = self.mapping_registry()
        recipe, unsupported = {}, []
        for vp_stage, model in enumerate(models_list):
            for name, module in model.named_modules():
                if not isinstance(module, TensorQuantizer):
                    continue
                global_name = _megatron_local_name_to_global(
                    models_list, model_config, self._unwrap_name(name), vp_stage
                )
                if not module.is_enabled:
                    # Disabled quantizers are left out of the recipe, so they must leave their input unchanged.
                    rotates = module.rotate_is_enabled and not module.rotate_back_is_enabled
                    if rotates or module.pre_quant_scale is not None:
                        unsupported.append(
                            f"{global_name} (disabled but rotates or scales its input)"
                        )
                    continue
                mapping = registry.megatron_to_hf_lookup(f"{global_name}._amax")
                hf_names = (
                    amax_mapping_hf_names(mapping, model_config.num_moe_experts) if mapping else []
                )
                if not hf_names:
                    unsupported.append(f"{global_name} (no Hugging Face mapping)")
                buffers = sorted(n for n, _ in module.named_buffers(recurse=False) if n != "_amax")
                if buffers:
                    unsupported.append(f"{global_name} (unexported tensors {buffers})")
                state = module.get_modelopt_state(properties_only=True)
                # build_conversion_tasks attaches the model config to modules lacking one; not quantizer state.
                state.pop("config", None)
                recipe.update((hf_name.removesuffix("._amax"), state) for hf_name in hf_names)

        per_rank = [(recipe, unsupported)]
        if torch.distributed.is_initialized():
            per_rank = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(per_rank, (recipe, unsupported))
        unsupported = sorted({name for _, names in per_rank for name in names})
        if unsupported:
            raise ValueError(f"ModelOpt quantizers cannot be exported: {unsupported}")
        merged = {}
        for rank_recipe, _ in per_rank:
            for hf_name, state in rank_recipe.items():
                if merged.setdefault(hf_name, state) != state:
                    raise ValueError(f"Ranks disagree on the ModelOpt quantizer state of {hf_name}")
        return merged

    def _with_progress_tracking(self, tasks, description: str, show_progress: bool = True):
        """Helper method to wrap an iterable with progress tracking.

        Args:
            tasks: Iterable of tasks to process
            description: Description for the progress bar
            show_progress: Whether to show progress (defaults to True)

        Yields:
            Items from the tasks iterable while updating progress
        """
        is_main_rank = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
        bridge_name = self.__class__.__name__

        if show_progress:
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeRemainingColumn(),
                TextColumn("({task.completed}/{task.total})"),
                TextColumn("{task.fields[bridge]}"),
                disable=not is_main_rank,
            ) as progress:
                task_id = progress.add_task(description, total=len(tasks), bridge=bridge_name)

                for task in tasks:
                    yield task
                    progress.update(task_id, advance=1)
        else:
            # not using disable above because we notice it will dump some empty progress bar,
            # even when disable is set to True
            for task in tasks:
                yield task

    def maybe_modify_loaded_hf_weight(
        self, hf_param: str | dict[str, str], hf_state_dict: Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        """Load weights from HuggingFace state dict.
        This function can be overridden by subclasses to preprocess the HF weights before conversion, such as renaming
        certain parameters to avoid mapping conflicts, or dequantize the weights.

        Note that loading is done lazily before this function is called, so the weights are actually loaded in
        this function when hf_state_dict.__getitem__ is called.

        Args:
            hf_param: The parameter name or dictionary of parameter names to load.
            hf_state_dict: The HuggingFace state dictionary.

        Returns:
            The loaded weights.
        """
        if isinstance(hf_param, str):
            hf_weights = hf_state_dict[hf_param]
        else:
            hf_weights = {k: hf_state_dict[v] for k, v in hf_param.items()}
        return hf_weights

    @staticmethod
    def get_hf_import_param_names(
        hf_param: str | dict[str, str], available_hf_param_names: set[str] | None = None
    ) -> tuple[str, ...]:
        """Declare HF tensors that the import preprocessing hook may consume.

        When checkpoint keys are available, this includes common quantization
        companions and alternative packed representations. This keeps incremental
        import callers from running a task before scale sidecars have arrived.
        Bridges whose preprocessing hook reads other companion keys must override
        this method to declare them.

        Args:
            hf_param: Parameter name or role-to-name mapping from the conversion mapping.
            available_hf_param_names: Complete source schema, when known.

        Returns:
            Ordered, deduplicated source tensor names required for the task.
        """
        mapped_names = (hf_param,) if isinstance(hf_param, str) else tuple(hf_param.values())
        if available_hf_param_names is None:
            return tuple(dict.fromkeys(mapped_names))

        required_names: list[str] = []
        for name in mapped_names:
            if name in available_hf_param_names:
                required_names.append(name)
                companion_names = [f"{name}_scale_inv"]
                if name.endswith(".weight"):
                    companion_names.append(f"{name[: -len('.weight')]}.scale")
                required_names.extend(
                    companion_name
                    for companion_name in companion_names
                    if companion_name in available_hf_param_names
                )
                continue

            packed_representations = (
                (f"{name}_packed", f"{name}_scale", f"{name}_shape"),
                (f"{name}_packed", f"{name}_scale"),
                (f"{name}_blocks", f"{name}_scales"),
            )
            representation = next(
                (
                    candidate
                    for candidate in packed_representations
                    if all(part in available_hf_param_names for part in candidate)
                ),
                None,
            )
            if representation is not None:
                required_names.extend(representation)

        return tuple(dict.fromkeys(required_names))

    @staticmethod
    def _convert_loaded_hf_weight(
        task: WeightConversionTask, hf_weights: torch.Tensor | dict[str, torch.Tensor]
    ) -> Optional[torch.Tensor]:
        """Convert already-loaded HF inputs for one Bridge task."""
        if task.megatron_module is None:
            return None
        return task.mapping.hf_to_megatron(hf_weights, task.megatron_module)

    def maybe_modify_converted_hf_weight(
        self,
        task: WeightConversionTask,
        converted_weights_dict: Dict[str, torch.Tensor],
        hf_state_dict: Mapping[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Modify the converted weights after conversion. By default, no modification is done.
        This function can be overridden by subclasses to postprocess the converted weights, such as merging the
        weights of multiple experts or quantizing the weights.

        Args:
            task: The WeightConversionTask object.
            converted_weights_dict: The converted weights dictionary.
            hf_state_dict: The HuggingFace state dict accessor for expected-key checks.

        Returns:
            The modified weights dictionary.
        """
        return converted_weights_dict

    @staticmethod
    def _is_vocab_export_task(task: WeightConversionTask[Any]) -> bool:
        """Return whether a conversion task emits an HF token embedding or language-model head."""
        vocab_param_suffixes = (
            "embedding.word_embeddings.weight",
            "output_layer.weight",
            "output_layer.bias",
        )
        global_param_name = task.global_param_name.removesuffix("_scale_inv")
        if not global_param_name.endswith(vocab_param_suffixes):
            return False

        hf_param = getattr(task.mapping, "hf_param", None)
        vocab_hf_param_suffixes = (
            "embed_tokens.weight",
            "word_embeddings.weight",
            "lm_head.weight",
            "head.weight",
            "predictions.bias",
        )
        return isinstance(hf_param, str) and hf_param.endswith(vocab_hf_param_suffixes)

    def _truncate_vocab_padding(
        self,
        task: WeightConversionTask[Any],
        converted_weights_dict: Dict[str, torch.Tensor],
        *,
        scale_block_size: int | None = None,
    ) -> Dict[str, torch.Tensor]:
        """Remove Megatron-only vocabulary rows from an HF export tensor."""
        if not self._is_vocab_export_task(task):
            return converted_weights_dict

        hf_param = task.mapping.hf_param
        is_scale_task = task.global_param_name.endswith("_scale_inv")
        output_hf_param = f"{hf_param}_scale_inv" if is_scale_task else hf_param
        if output_hf_param not in converted_weights_dict:
            return converted_weights_dict

        hf_config = self.hf_config
        vocab_size = None
        config_candidates = [hf_config]
        visited_config_ids = set()
        while config_candidates:
            candidate = config_candidates.pop(0)
            if candidate is None or id(candidate) in visited_config_ids:
                continue
            visited_config_ids.add(id(candidate))

            candidate_vocab_size = getattr(candidate, "vocab_size", None)
            if isinstance(candidate_vocab_size, int) and candidate_vocab_size > 0:
                vocab_size = candidate_vocab_size
                break
            config_candidates.extend(
                getattr(candidate, nested_config_name, None)
                for nested_config_name in ("text_config", "llm_config", "thinker_config")
            )
        if vocab_size is None:
            return converted_weights_dict

        source_tensor = converted_weights_dict[output_hf_param]
        target_rows = vocab_size
        if is_scale_task:
            scale_block_size = getattr(task.mapping, "scale_block_size", None)
            local_scale = task.param_weight
            module_weight = getattr(task.megatron_module, "weight", None)
            if scale_block_size is None:
                if (
                    local_scale is None
                    or module_weight is None
                    or local_scale.ndim == 0
                    or module_weight.ndim == 0
                    or local_scale.shape[0] == 0
                    or module_weight.shape[0] == 0
                    or module_weight.shape[0] % local_scale.shape[0] != 0
                ):
                    return converted_weights_dict
                scale_block_size = module_weight.shape[0] // local_scale.shape[0]
        if scale_block_size is not None:
            target_rows = math.ceil(vocab_size / scale_block_size)

        if source_tensor.ndim == 0 or source_tensor.shape[0] <= target_rows:
            return converted_weights_dict

        padded_vocab_size = source_tensor.shape[0]
        result = {}
        for name, tensor in converted_weights_dict.items():
            if tensor.ndim == 0:
                result[name] = tensor
            elif scale_block_size is not None and name.endswith("_scale_inv"):
                result[name] = tensor[: math.ceil(vocab_size / scale_block_size)]
            elif tensor.shape[0] == padded_vocab_size:
                result[name] = tensor[:vocab_size]
            else:
                result[name] = tensor
        return result

    @staticmethod
    def _cast_export_weight_dtype(
        weights: Dict[str, torch.Tensor], weight_dtype: Optional[torch.dtype]
    ) -> Dict[str, torch.Tensor]:
        """Cast float export weights to ``weight_dtype`` (no-op if None; ints untouched)."""
        if weight_dtype is None:
            return weights
        return {
            name: (weight.to(weight_dtype) if weight.is_floating_point() else weight)
            for name, weight in weights.items()
        }

    def _accumulate_grouped_export(
        self,
        task: "WeightConversionTask",
        converted_weights_dict: Dict[str, torch.Tensor],
        model_config,
        grouped_buffers: Dict[str, Dict[int, torch.Tensor]],
        hf_state_dict: Mapping[str, torch.Tensor],
        grouped_sources: Optional[Dict[str, List[str]]] = None,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Accumulate per-expert weights for grouped export, return merged result when complete.

        For fused-expert MoE models where one HF tensor contains all experts, this method
        collects individual expert weights produced by per-expert ``megatron_to_hf`` calls
        and returns the stacked result once all experts have been accumulated.

        When ``grouped_sources`` is given, the ``param_name`` of every task folded into a
        group key is appended to ``grouped_sources[group_key]`` so the caller can report all
        contributing Megatron parameters for the packed tensor.

        Returns:
            Merged weights dict when the group is complete, ``None`` otherwise.
        """
        from megatron.bridge.utils.common_utils import extract_expert_number_from_param

        # Use the mapping's process group rather than the global parallel-state
        # singleton. Decentralized callers such as MegatronMIMO install
        # route-local groups directly on mappings, while MCore's cached global EP
        # world size may still describe another component.
        ep_size = task.mapping.ep_size
        num_experts = model_config.num_moe_experts
        experts_per_rank = num_experts // ep_size

        try:
            local_expert_number = (
                extract_expert_number_from_param(task.param_name) % experts_per_rank
            )
        except ValueError:
            return None

        merged_result: Dict[str, torch.Tensor] = {}
        for group_key, value in converted_weights_dict.items():
            if group_key not in grouped_buffers:
                grouped_buffers[group_key] = {}

            if ep_size == 1:
                grouped_buffers[group_key][local_expert_number] = value
            else:
                if value.ndim > 0 and value.shape[0] == ep_size:
                    for i in range(ep_size):
                        global_expert_number = local_expert_number + (i * experts_per_rank)
                        grouped_buffers[group_key][global_expert_number] = value[i]
                else:
                    grouped_buffers[group_key][local_expert_number] = value
            if grouped_sources is not None:
                grouped_sources.setdefault(group_key, []).append(task.param_name)

            if len(grouped_buffers[group_key]) != num_experts:
                continue

            grouped_tensors = [grouped_buffers[group_key][i] for i in range(num_experts)]
            try:
                merged = torch.stack(grouped_tensors, dim=0)
            except torch.OutOfMemoryError:
                if not any(getattr(tensor, "is_cuda", False) for tensor in grouped_tensors):
                    raise
                logger.warning(
                    "Grouped expert export ran out of CUDA memory while stacking %d experts; retrying on CPU.",
                    num_experts,
                )
                cpu_tensors = [tensor.cpu() for tensor in grouped_tensors]
                del grouped_tensors
                del grouped_buffers[group_key]
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                merged = torch.stack(cpu_tensors, dim=0)

            if getattr(task.mapping, "transpose_on_export", False):
                if group_key in hf_state_dict:
                    # Adaptive: only transpose when the stacked shape doesn't match the original HF
                    # shape but the transposed shape does.  This handles configurations where the
                    # per-expert weight is already in [out, in] (PyTorch) rather than [in, out]
                    # (TE) layout — e.g. when explicit_expert_comm=False (etp=1, ep=1).
                    expected = tuple(hf_state_dict[group_key].shape)
                    transposed = merged.transpose(-1, -2).contiguous()
                    if tuple(merged.shape) != expected and tuple(transposed.shape) == expected:
                        merged = transposed
                else:
                    merged = merged.transpose(-1, -2).contiguous()

            grouped_buffers.pop(group_key, None)
            merged_result[group_key] = merged

        return merged_result or None

    def load_weights_hf_to_megatron(
        self,
        hf_pretrained: HFPreTrained,
        megatron_model: Union[MegatronModel, List[MegatronModel]],
        allowed_mismatched_params: Optional[List[str]] = None,
    ) -> List[MegatronModel]:
        """Load HuggingFace weights into Megatron models.

        This method orchestrates the complete weight loading process from HuggingFace
        format to Megatron's distributed format. It builds a conversion task and
        executes it with proper progress tracking and error handling.

        The actual weight transformations and distribution are delegated to the
        appropriate MegatronParamMapping instances based on the state mappings.

        Args:
            hf_pretrained (HFPreTrained): HuggingFace model or state source containing the
                weights to load.
            megatron_model (Union[MegatronModel, List[MegatronModel]]): Megatron model instance
                or list of model instances (one per virtual pipeline stage).
            allowed_mismatched_params (Optional[List[str]]): List of parameter names or patterns
                to allow mismatch (skip instead of raise error).

        Returns:
            List[MegatronModel]: The input megatron_model as a list with loaded weights.

        Process:
        1. Build a task mapping each Megatron parameter to its source
        2. For each parameter in the task:
            - Fetch source weights from HuggingFace state
            - Apply format transformation via the param mapping
            - Distribute to appropriate TP/PP ranks
            - Copy into the Megatron parameter

        Example:
            .. code-block:: python

                hf_model = PreTrainedCausalLM.from_pretrained("gpt2")
                megatron_model = create_megatron_model()  # Single model or list
                bridge.load_weights_hf_to_megatron(hf_model, megatron_model)

        Note:
            Progress is shown only on rank 0 to avoid cluttered output in
            distributed environments.

        Raises:
            ValueError: If hf_pretrained doesn't have state attribute or if weight shapes don't match.
            AttributeError: If required HF weights are missing.
        """
        if not isinstance(megatron_model, list):
            megatron_model = [megatron_model]

        ddp_config = getattr(megatron_model[0], "ddp_config", None)
        use_megatron_fsdp = ddp_config is not None and ddp_config.use_megatron_fsdp
        if use_megatron_fsdp:
            original_megatron_model = megatron_model
        unwrapped_model_list = unwrap_model(megatron_model)
        # [ModelOpt]: Hide extra parameters registered in Distillation mode
        with contextlib.ExitStack() as stack:
            if hasattr(unwrapped_model_list[0], "hide_teacher_model"):
                stack.enter_context(unwrapped_model_list[0].hide_teacher_model())
            if hasattr(unwrapped_model_list[0], "hide_loss_modules"):
                stack.enter_context(unwrapped_model_list[0].hide_loss_modules())

            hf_to_megatron_tasks = self.build_conversion_tasks(hf_pretrained, unwrapped_model_list)
        hf_state_dict: Mapping[str, torch.Tensor] = (
            hf_pretrained.state if hasattr(hf_pretrained, "state") else {}
        )

        description = f"Loading from {hf_pretrained.model_name_or_path}"

        # if export_weight_dtype is FP8, we need save the unquantized state dict for initializing optimizer main params
        export_weight_dtype = getattr(self, "export_weight_dtype", "bf16")
        capture_unquantized_state_dict = export_weight_dtype == "fp8"
        # Optional: capture per-rank, per-parameter unquantized shards for initializing
        # for loading original weights when fp8_param=True.
        captured_state_dicts: dict[str, dict[str, torch.Tensor]] | None = None
        if capture_unquantized_state_dict:
            captured_state_dicts = {f"model{i}": {} for i in range(len(megatron_model))}
        self.unquantized_state_dict = None

        _hf_import_cache: Dict[str, torch.Tensor] = {}
        for task in self._with_progress_tracking(hf_to_megatron_tasks, description):
            # None means megatron module not on current rank, skip if this task is not going to happen
            if task.megatron_module is None:
                continue
            # 1) Fetch source tensor(s) from HF state dict, with caching for grouped mappings
            hf_param_key = str(task.mapping.hf_param)
            is_grouped = getattr(task.mapping, "is_grouped_export", False)
            if is_grouped and hf_param_key in _hf_import_cache:
                hf_weights = _hf_import_cache[hf_param_key]
            else:
                hf_weights = self.maybe_modify_loaded_hf_weight(
                    task.mapping.hf_param, hf_state_dict
                )
                if is_grouped:
                    _hf_import_cache[hf_param_key] = hf_weights

            # 2) Delegate conversion & distribution to the bridge
            converted_weights = self._convert_loaded_hf_weight(task, hf_weights)

            # 3) Copy into Megatron param if this rank received a shard
            if converted_weights is not None:
                # Assert that param_weight is not None for HF->Megatron tasks
                assert (
                    task.param_weight is not None
                ), "param_weight is required for HF->Megatron conversion"
                if isinstance(task.param_weight, DTensor):
                    sliced_converted_weights = converted_weights.reshape(-1)[
                        task.param_weight.megatron_fsdp_slice
                    ]
                    task.param_weight._local_tensor.reshape(-1).copy_(sliced_converted_weights)
                    continue

                # Check shape compatibility before copying
                expected_shape = _get_mapping_shape(task.param_weight)
                if converted_weights.shape != expected_shape:
                    # Check whitelist
                    is_whitelisted = False
                    if allowed_mismatched_params:
                        for pattern in allowed_mismatched_params:
                            if fnmatch.fnmatch(
                                task.mapping.megatron_param, pattern
                            ) or fnmatch.fnmatch(task.param_name, pattern):
                                is_whitelisted = True
                                break

                    if is_whitelisted:
                        print_rank_0(
                            f"WARNING: Shape mismatch for megatron param {task.mapping.megatron_param} allowed by whitelist. Skipping."
                        )
                        continue

                    raise ValueError(
                        f"Shape mismatch for megatron param {task.mapping.megatron_param}:\n"
                        f"  Expected shape: {expected_shape}\n"
                        f"  Got shape: {converted_weights.shape}\n"
                        f"  Bridge type: {type(task.mapping).__name__}\n"
                        f"  HF mapping: {task.mapping.hf_param}"
                    )
                converted_weights = _slice_gtp_weight(converted_weights, task.param_weight)
                if capture_unquantized_state_dict:
                    vp_stage = task.vp_stage if task.vp_stage is not None else 0
                    chunk_key = f"model{vp_stage}"
                    captured_state_dicts[chunk_key][task.param_name] = converted_weights.detach()
                # NOTE:
                # For fp8_param (blockwise), `task.param_weight` can be a TransformerEngine
                # Float8BlockwiseQTensor) that is a leaf with requires_grad=True.
                # In-place updates under grad mode will raise:
                # "a leaf Variable that requires grad is being used in an in-place operation."
                with (
                    torch.no_grad(),
                    _gtp_weight_load_context(task.param_weight, task.megatron_module),
                ):
                    task.param_weight.copy_(converted_weights)
        self.finalize_hf_import(megatron_model)
        if use_megatron_fsdp:
            for m in original_megatron_model:
                m.module.install_optimized_model_weights()
            return original_megatron_model
        if capture_unquantized_state_dict:
            # Keep Megatron's single-chunk convention: "model" instead of "model0".
            if len(megatron_model) == 1:
                self.unquantized_state_dict = {"model": captured_state_dicts["model0"]}
            else:
                self.unquantized_state_dict = captured_state_dicts
        return megatron_model

    def iter_local_hf_params(self, tasks: Iterable[WeightConversionTask]) -> Iterable[LocalHFParam]:
        """Yield local unquantized BF16 parameters as canonical HF views.

        The caller may reuse tasks from :meth:`build_conversion_tasks` across
        iterations to cache parameter topology. Each pass reads ``param_weight``
        again and yields live local views without Bridge collectives, dtype
        conversion, or quantization sidecars.

        Args:
            tasks: Conversion tasks in deterministic export order.

        Yields:
            Canonical local HF parameter views in task and mapping order after
            every locally owned task passes validation.

        Raises:
            ValueError: If a locally owned task has inconsistent metadata, is
                not backed by unquantized BF16 storage, or needs conversion that
                cannot be represented by local views.
        """
        params = []
        for task in tasks:
            if task.param_weight is None:
                continue
            if task.megatron_module is None:
                raise ValueError(
                    f"{task.global_param_name}: local parameter has no owning Megatron module"
                )
            params.extend(
                task.mapping.local_hf_params(
                    task.param_weight,
                    global_param_name=task.global_param_name,
                    megatron_module=task.megatron_module,
                )
            )
        yield from params

    def stream_weights_hf_to_megatron(
        self,
        hf_pretrained: HFPreTrained,
        megatron_model: Union[MegatronModel, List[MegatronModel]],
        conversion_tasks: Optional[List[WeightConversionTask]] = None,
        *,
        hf_state_dict: Mapping[str, torch.Tensor] | None = None,
    ) -> Iterable[MegatronWeightTuple]:
        """Generator variant of load_weights_hf_to_megatron for streaming weight conversion.

        This method provides a memory-efficient way to convert weights by yielding
        them one at a time instead of loading all at once. Useful for processing
        very large models or when implementing custom weight handling logic.

        Args:
            hf_pretrained (HFPreTrained): HuggingFace model or state source containing
                the weights.
            megatron_model (Union[MegatronModel, List[MegatronModel]]): Megatron model instance
                or list of model instances to extract configuration from.
            conversion_tasks (Optional[List[WeightConversionTask]]): Pre-built conversion tasks.
                If not provided, tasks will be built automatically from the models.
            hf_state_dict: Optional external HF-style state mapping. When omitted,
                weights are read from ``hf_pretrained.state``.

        Yields:
            MegatronWeightTuple: Named tuples containing:
                - vp_stage: Index of the model in megatron_model list
                - param_name: Name of the parameter
                - weight: Transformed weight tensor for this rank

        Example:
            .. code-block:: python

                # Process weights one by one
                for weight_tuple in bridge.stream_weights_hf_to_megatron(hf_model, megatron_model):
                    print(f"Processing {weight_tuple.param_name}: {weight_tuple.weight.shape}")
                    # Custom processing logic here

                # Or use pre-built conversion tasks
                tasks = bridge.build_conversion_tasks(hf_model, megatron_model)
                for weight_tuple in bridge.stream_weights_hf_to_megatron(hf_model, megatron_model, tasks):
                    print(f"Processing {weight_tuple.param_name}: {weight_tuple.weight.shape}")

        Note:
            Only yields weights that belong to the current rank after TP/PP distribution.

        Raises:
            ValueError: If input parameters are invalid.
        """

        if not isinstance(megatron_model, list):
            megatron_model = [megatron_model]

        # Use provided conversion tasks or build them
        if conversion_tasks is None:
            conversion_tasks = self.build_conversion_tasks(hf_pretrained, megatron_model)
        if hf_state_dict is None:
            hf_state_dict = hf_pretrained.state

        for task in conversion_tasks:
            # None means megatron module not on current rank, skip if this task is not going to happen
            if task.megatron_module is None:
                continue
            hf_weights = self.maybe_modify_loaded_hf_weight(task.mapping.hf_param, hf_state_dict)
            converted_weights = self._convert_loaded_hf_weight(task, hf_weights)
            if converted_weights is not None:
                converted_weights = _slice_gtp_weight(converted_weights, task.param_weight)
                # Assert that vp_stage is not None for HF->Megatron tasks
                yield MegatronWeightTuple(task.param_name, converted_weights, task.vp_stage)

    def _should_emit_hf_passthrough(
        self, megatron_model: MegatronModel | List[MegatronModel], *, current_pp_stage_only: bool
    ) -> bool:
        """Return whether this rank should emit source-only HF tensors.

        Source-only tensors have no Megatron parameter owner. In PP-local mode
        assign them to PP rank 0 so every pipeline group emits one deterministic
        copy while the union of stage streams remains complete.
        """
        return not current_pp_stage_only or _get_pp_rank(megatron_model) == 0

    @torch.no_grad()
    def _stream_weight_events_megatron_to_hf(
        self,
        megatron_model: Union[MegatronModel, List[MegatronModel]],
        hf_pretrained: HFPreTrained,
        cpu: bool = True,
        show_progress: bool = True,
        conversion_tasks: Optional[List[WeightConversionTask]] = None,
        weight_dtype: Optional[torch.dtype] = None,
        with_megatron_names: bool = False,
        current_pp_stage_only: bool = False,
    ) -> Iterable[HFWeightTuple | HFSourcedWeightTuple | None]:
        """Export weights and emit ``None`` after each conversion task.

        This method orchestrates the conversion of weights from Megatron's distributed
        format back to HuggingFace format. It handles gathering from tensor parallel
        ranks, broadcasting across pipeline parallel ranks, and format conversions.
        By default all ranks receive the full tensors; pipeline-stage-local mode instead
        yields only tensors owned by the current pipeline stage.

        The export order is determined automatically:
        - First tries safetensors order (if key_to_filename_map is available)
        - Falls back to HuggingFace state dict order

        Args:
            megatron_model (Union[MegatronModel, List[MegatronModel]]): Megatron model instance
                or list of model instances (one per virtual pipeline stage).
            hf_pretrained (HFPreTrained): HuggingFace model/config for metadata
                and mapping info.
            cpu (bool, optional): Whether to move tensors to CPU before yielding.
                Defaults to True.
            show_progress (bool, optional): Display progress bar during export.
                Defaults to True.
            conversion_tasks (Optional[List[WeightConversionTask]]): Pre-built conversion tasks.
                If not provided, tasks will be built automatically from the models.
            with_megatron_names (bool, optional): When True, yield :class:`HFSourcedWeightTuple`
                (``param_name``, ``weight``, ``megatron_param_names``) so callers can map each
                exported HF weight back to the Megatron parameter(s) it came from: one name for a
                directly converted weight, every contributing per-expert task for a grouped-expert
                export, and no names for HF-only passthrough tensors. Defaults to False, which
                keeps the two-field :class:`HFWeightTuple` output.
            current_pp_stage_only (bool, optional): When True, execute only conversion tasks whose
                Megatron parameter is owned by the current pipeline stage and suppress PP tensor and
                metadata broadcasts. TP/EP gathering and all HF layout conversions remain enabled.
                Defaults to False.

        Yields:
            HuggingFace weight tuples, optionally carrying source names, followed
            by one ``None`` boundary marker after every conversion task.

        Example:
            .. code-block:: python

                # Export weights
                for name, weight in bridge.stream_weights_megatron_to_hf(megatron_model, hf_config):
                    print(f"Exported {name}: {weight.shape}")

                # Or use pre-built conversion tasks
                tasks = bridge.build_conversion_tasks(hf_config, megatron_model)
                for name, weight in bridge.stream_weights_megatron_to_hf(
                    megatron_model, hf_config, conversion_tasks=tasks
                ):
                    print(f"Exported {name}: {weight.shape}")

        Raises:
            ValueError: If input parameters are invalid.

        Note:
            Unless ``current_pp_stage_only`` is enabled, all ranks yield the full tensors after
            gathering from distributed format.
        """

        if not isinstance(megatron_model, list):
            megatron_model = [megatron_model]

        unwrapped_model_list = unwrap_model(megatron_model)
        self.hf_pretrained = hf_pretrained
        self.hf_config = hf_pretrained.config if hasattr(hf_pretrained, "config") else hf_pretrained

        # Use provided conversion tasks or build them
        if conversion_tasks is None:
            conversion_tasks = self.build_conversion_tasks(
                hf_pretrained, unwrapped_model_list, weight_dtype=weight_dtype
            )
        elif weight_dtype is not None:
            # Prebuilt tasks bypass build_conversion_tasks (where weight_dtype is recorded);
            # fail loudly instead of silently dropping it (this also rejects the fp8-tasks combo).
            raise ValueError(
                "weight_dtype is not supported with caller-supplied conversion_tasks; "
                "omit conversion_tasks so the dtype can be recorded at task-build time."
            )

        if current_pp_stage_only:
            # Keep the globally deterministic task construction above, but only
            # execute tasks whose parameter is physically present on this PP
            # stage. If a logical parameter is duplicated across stages, use
            # the same lowest-rank canonical owner as full export. The mapping
            # policy below returns local values instead of entering a cross-PP
            # broadcast.
            if not hasattr(self, "_cached_param_owners"):
                self._megatron_global_param_names_all_pp_ranks(megatron_model)
            canonical_owners = self._cached_param_owners
            current_pp_rank = _get_pp_rank(megatron_model)

            def _canonical_owner(task: WeightConversionTask) -> int:
                global_name = task.global_param_name
                owner = canonical_owners.get(global_name)
                if owner is None and global_name.endswith("_scale_inv"):
                    owner = canonical_owners.get(global_name.removesuffix("_scale_inv"))
                return current_pp_rank if owner is None else owner

            conversion_tasks = [
                task
                for task in conversion_tasks
                if task.param_weight is not None and _canonical_owner(task) == current_pp_rank
            ]

        megatron_to_hf_tasks = conversion_tasks
        unwrapped_model = unwrapped_model_list[0]
        model_config = unwrapped_model.config
        embeddings_are_tied = self._share_embeddings_and_output_weights(unwrapped_model)
        tied_mapping_registry = self.mapping_registry() if embeddings_are_tied else None

        hf_state_dict: Mapping[str, torch.Tensor] = (
            hf_pretrained.state if hasattr(hf_pretrained, "state") else {}
        )

        _grouped_buffers: Dict[str, Dict[int, torch.Tensor]] = {}
        # Megatron params folded into each packed grouped tensor; only tracked when requested.
        _grouped_sources: Optional[Dict[str, List[str]]] = {} if with_megatron_names else None

        def _mapping_export_context():
            return (
                MegatronParamMapping.skip_pp_broadcast()
                if current_pp_stage_only
                else contextlib.nullcontext()
            )

        for task in self._with_progress_tracking(
            megatron_to_hf_tasks, "Converting to HuggingFace", show_progress
        ):
            merged_result = None
            tensor = None
            if isinstance(task.param_weight, DTensor):
                from megatron.core.distributed.fsdp.src.megatron_fsdp.uneven_dtensor import (
                    uneven_dtensor_to_full_tensor,
                )

                megatron_weights = uneven_dtensor_to_full_tensor(task.param_weight)
            else:
                megatron_weights = _gather_gtp_weight(task.param_weight)
            megatron_module = task.megatron_module
            if task.vp_stage is not None and self._should_skip_mtp_duplicate_embedding_export(
                task.global_param_name, megatron_model[task.vp_stage]
            ):
                megatron_weights = None
                megatron_module = None

            with _mapping_export_context():
                converted_weights_dict = task.mapping.megatron_to_hf(
                    megatron_weights, megatron_module
                )
            # --- Grouped export path: accumulate per-expert weights, yield when complete ---
            if getattr(task.mapping, "is_grouped_export", False):
                merged_result = self._accumulate_grouped_export(
                    task,
                    converted_weights_dict,
                    model_config,
                    _grouped_buffers,
                    hf_state_dict,
                    grouped_sources=_grouped_sources,
                )
                if merged_result is not None:
                    merged_result = self._cast_export_weight_dtype(merged_result, task.weight_dtype)
                    for hf_name, tensor in merged_result.items():
                        # Report every per-expert task packed into this tensor, not just the one
                        # that happened to complete the group.
                        group_sources = (
                            tuple(_grouped_sources.pop(hf_name, ()))
                            if _grouped_sources is not None
                            else None
                        )
                        yield from HFWeightTuple(hf_name, tensor).iter_finalized(
                            export_hook=task.export_hook,
                            cpu=cpu,
                            megatron_param_names=group_sources,
                        )
                # Generator locals otherwise keep the previous task's tensors alive
                # while the next task's conversion RHS allocates its outputs.
                del megatron_weights, converted_weights_dict, merged_result, tensor
                yield None
                continue

            # --- Standard export path ---
            converted_weights_dict = self.maybe_modify_converted_hf_weight(
                task, converted_weights_dict, hf_state_dict
            )  # dict will be none except for one expert;
            # Finalize the converted tensor for the current export rank
            converted_weights_dict = self._truncate_vocab_padding(task, converted_weights_dict)
            # Quantizer ranges keep their precision; ``weight_dtype`` applies to the weights only.
            if not isinstance(task.mapping, AmaxMapping):
                converted_weights_dict = self._cast_export_weight_dtype(
                    converted_weights_dict, task.weight_dtype
                )

            tied_output_hf_name = None
            if tied_mapping_registry is not None:
                tied_output_hf_name = self._get_tied_output_hf_name(task, tied_mapping_registry)

            for hf_name, tensor in converted_weights_dict.items():
                if tied_output_hf_name is not None and hf_name == task.mapping.hf_param:
                    emit_output_weight = isinstance(hf_pretrained, PretrainedConfig)
                    if hasattr(hf_pretrained, "state") and hasattr(hf_pretrained.state, "source"):
                        expected_keys = hf_pretrained.state.source.get_all_keys()
                        emit_output_weight = tied_output_hf_name in expected_keys

                    yield from HFWeightTuple(hf_name, tensor).iter_finalized(
                        export_hook=task.export_hook,
                        cpu=cpu,
                        megatron_param_names=(task.param_name,) if with_megatron_names else None,
                    )
                    if emit_output_weight:
                        yield from HFWeightTuple(tied_output_hf_name, tensor).iter_finalized(
                            export_hook=task.export_hook,
                            cpu=cpu,
                            clone_identity_output=True,
                            megatron_param_names=(
                                (task.param_name,) if with_megatron_names else None
                            ),
                        )
                elif embeddings_are_tied and (
                    task.global_param_name.endswith("output_layer.weight")
                    or hf_name == tied_output_hf_name
                ):
                    # This should not happen when embeddings are tied - assert error
                    raise ValueError(
                        f"Encountered {hf_name} when embeddings are tied. This indicates a mapping error."
                    )
                else:
                    # Regular case - yield the tensor normally
                    yield from HFWeightTuple(hf_name, tensor).iter_finalized(
                        export_hook=task.export_hook,
                        cpu=cpu,
                        megatron_param_names=(task.param_name,) if with_megatron_names else None,
                    )
            # Generator locals otherwise keep the previous task's tensors alive
            # while the next task's conversion RHS allocates its outputs.
            del megatron_weights, converted_weights_dict, merged_result, tensor
            yield None

    @torch.no_grad()
    def stream_weights_megatron_to_hf(
        self,
        megatron_model: Union[MegatronModel, List[MegatronModel]],
        hf_pretrained: HFPreTrained,
        cpu: bool = True,
        show_progress: bool = True,
        conversion_tasks: Optional[List[WeightConversionTask]] = None,
        weight_dtype: Optional[torch.dtype] = None,
        with_megatron_names: bool = False,
        current_pp_stage_only: bool = False,
    ) -> Iterable[HFWeightTuple | HFSourcedWeightTuple]:
        """Export Megatron weights as the existing flat HuggingFace stream."""
        for event in self._stream_weight_events_megatron_to_hf(
            megatron_model,
            hf_pretrained,
            cpu=cpu,
            show_progress=show_progress,
            conversion_tasks=conversion_tasks,
            weight_dtype=weight_dtype,
            with_megatron_names=with_megatron_names,
            current_pp_stage_only=current_pp_stage_only,
        ):
            if event is not None:
                yield event

    @torch.no_grad()
    def _stream_weight_groups_megatron_to_hf(
        self,
        megatron_model: Union[MegatronModel, List[MegatronModel]],
        hf_pretrained: HFPreTrained,
        cpu: bool = True,
        show_progress: bool = True,
        conversion_tasks: Optional[List[WeightConversionTask]] = None,
        weight_dtype: Optional[torch.dtype] = None,
        with_megatron_names: bool = False,
    ) -> Iterable[tuple[HFWeightTuple | HFSourcedWeightTuple, ...]]:
        """Export one materialized output tuple per conversion task."""
        outputs = []
        for event in self._stream_weight_events_megatron_to_hf(
            megatron_model,
            hf_pretrained,
            cpu=cpu,
            show_progress=show_progress,
            conversion_tasks=conversion_tasks,
            weight_dtype=weight_dtype,
            with_megatron_names=with_megatron_names,
        ):
            if event is None:
                yield tuple(outputs)
                outputs.clear()
            else:
                outputs.append(event)
        if outputs:
            raise RuntimeError("Internal error: unterminated weight conversion task")

    def dtype_from_hf(self, config, default=None):
        """Extract torch dtype from a HuggingFace config.

        This utility method handles the conversion of dtype specifications in
        HuggingFace configs to PyTorch dtype objects. Supports both direct
        torch.dtype objects and string representations.

        Args:
            config: HuggingFace configuration object with a torch_dtype attribute.
            default (Any, optional): Default value to return if torch_dtype is
                not str or torch.dtype. Defaults to None.

        Returns:
            torch.dtype: The corresponding PyTorch dtype.

        Raises:
            AssertionError: If config doesn't have torch_dtype attribute.
            ValueError: If torch_dtype is neither a string nor torch.dtype.

        Example:
            .. code-block:: python

                dtype = bridge.dtype_from_hf(hf_config)
                print(dtype)  # torch.float16
        """
        assert hasattr(config, "torch_dtype"), "Expected config to have attr `torch_dtype`"
        torch_dtype = config.torch_dtype
        if isinstance(torch_dtype, torch.dtype):
            return torch_dtype
        elif isinstance(torch_dtype, str):
            return self.dtype_from_str(torch_dtype)
        elif default is not None:
            return default

        raise ValueError("torch_dtype is not of type str/torch.dtype")

    def dtype_from_str(self, dtype: str) -> torch.dtype:
        """Convert a string precision identifier to equivalent torch dtype.

        Delegates to ``megatron.bridge.utils.activation_map.str_to_dtype``.
        Defaults to ``torch.float32`` for unrecognized strings.

        Args:
            dtype (str): String representation of dtype (e.g., "float16", "fp16",
                "bf16-mixed").

        Returns:
            torch.dtype: Corresponding PyTorch dtype (defaults to float32 if unknown).
        """
        from megatron.bridge.utils.activation_map import str_to_dtype

        try:
            return str_to_dtype(dtype)
        except ValueError:
            return torch.float32

    def make_vocab_size_divisible_by(self, vocab_size: int) -> int:
        """Calculate an appropriate divisor for vocabulary size padding.

        Megatron requires vocabulary sizes to be divisible by certain values for
        efficient tensor parallelism. This method finds the largest power of 2
        (up to 128) that evenly divides the vocabulary size.

        Args:
            vocab_size (int): Original vocabulary size from the model.

        Returns:
            int: Largest power of 2 (≤ 128) that divides vocab_size.

        Example:
            .. code-block:: python

                # For vocab_size=50257 (GPT-2)
                divisor = bridge.make_vocab_size_divisible_by(50257)
                print(divisor)  # 1 (50257 is prime)

                # For vocab_size=32000 (Llama)
                divisor = bridge.make_vocab_size_divisible_by(32000)
                print(divisor)  # 128

        Note:
            The returned value is used by Megatron to potentially pad the
            vocabulary to ensure efficient parallelization.
        """
        base = 128
        while vocab_size % base != 0:
            base //= 2
        return base

    def _get_provider_from_model(self, model: MegatronModule) -> ModelProviderTarget:
        """Extract provider/config from model."""
        model = unwrap_model(model)
        return model.config

    def _share_embeddings_and_output_weights(self, model: MegatronModule) -> bool:
        """Return whether the model ties its input embeddings and output layer.

        Provider-built models carry the flag on ``model.config``. A builder-backed model's
        ``config`` is only the nested ``TransformerConfig``, while the flag lives on the outer
        model config, so fall back to the flag stored on the built language model, which owns
        the output layer, and then on the model itself.

        Args:
            model: Unwrapped Megatron model chunk.

        Returns:
            True when the output layer reuses the input embedding weights.
        """
        model_config = getattr(model, "config", None)
        if hasattr(model_config, "share_embeddings_and_output_weights"):
            return model_config.share_embeddings_and_output_weights
        for module in (getattr(model, "language_model", None), model):
            if module is not None and hasattr(module, "share_embeddings_and_output_weights"):
                return bool(module.share_embeddings_and_output_weights)
        return False

    def _unwrap_name(self, name: str) -> str:
        """Unwrap name from DDP or other wrappers.

        Args:
            name: Parameter name that may have 'module.' prefixes

        Returns:
            Unwrapped parameter name with 'module.' prefixes removed

        Example:
            'module.module.decoder.weight' -> 'decoder.weight'
        """
        if not isinstance(name, str):
            raise ValueError(f"name must be a string, got {type(name)}")

        while name.startswith("module."):
            name = name[len("module.") :]
        return name

    def _broadcast_shared_embeddings(
        self, megatron_model: Union[MegatronModel, List[MegatronModel]]
    ) -> None:
        """Broadcast shared embeddings and output weights across embedding group.

        When embeddings and output weights are shared and pipeline parallelism is enabled,
        this method ensures all ranks in the embedding group have the same weights by
        broadcasting from rank 0.

        Args:
            megatron_model: Megatron model instance or list of model instances.
        """
        pp_group = _get_pp_group(megatron_model)
        is_first_pp_stage = is_pp_first_stage(pp_group)
        is_last_pp_stage = is_pp_last_stage(pp_group)

        unwrapped_models = unwrap_model(megatron_model)
        if not isinstance(unwrapped_models, list):
            unwrapped_models = [unwrapped_models]
        # VPP chunk 0 owns the input embedding; the final chunk owns the output layer.
        unwrapped_model = unwrapped_models[-1] if is_last_pp_stage else unwrapped_models[0]

        # hack for vlm to work properly
        if (
            hasattr(unwrapped_model, "language_model")
            and unwrapped_model.language_model is not None
        ):
            unwrapped_model = unwrapped_model.language_model
        model_config = unwrapped_model.config
        share_embeddings = self._share_embeddings_and_output_weights(unwrapped_model)

        if (share_embeddings and model_config.pipeline_model_parallel_size > 1) and (
            is_first_pp_stage or is_last_pp_stage
        ):
            # Broadcast embeddings and output weights from rank 0 to embedding group
            embd_group = _get_embedding_group(megatron_model)
            if embd_group is None:
                return
            embd_group_ranks = torch.distributed.get_process_group_ranks(embd_group)
            if torch.distributed.get_rank() in embd_group_ranks:
                # Get embeddings and output weights from rank 0
                if hasattr(unwrapped_model, "embedding") and hasattr(
                    unwrapped_model.embedding, "word_embeddings"
                ):
                    embd_weights = unwrapped_model.embedding.word_embeddings.weight.data
                else:
                    assert hasattr(unwrapped_model, "output_layer"), "Output layer not found"
                    embd_weights = torch.empty_like(unwrapped_model.output_layer.weight.data)
                torch.distributed.broadcast(embd_weights, src=embd_group_ranks[0], group=embd_group)
                if hasattr(unwrapped_model, "output_layer"):
                    unwrapped_model.output_layer.weight.data.copy_(embd_weights)

    def finalize_hf_import(self, megatron_model: Union[MegatronModel, List[MegatronModel]]) -> None:
        """Finalize tied parameters and any available parameter-derived caches after import."""
        from megatron.core import resharding

        self._broadcast_shared_embeddings(megatron_model)
        refresh_module_caches = getattr(resharding, "refresh_module_caches", None)
        if refresh_module_caches is not None:
            refresh_module_caches(megatron_model)

    @staticmethod
    def _should_skip_mtp_duplicate_embedding_export(global_param_name: str, model) -> bool:
        """Return whether ``model`` owns only the MTP copy of the input embedding.

        The MTP stage needs a local input embedding even when the model uses untied
        output weights. Megatron keeps that tensor synchronized with the canonical
        pre-process embedding, so it must not become a second HF parameter.
        """
        if not global_param_name.endswith("embedding.word_embeddings.weight"):
            return False

        model_chunk = unwrap_model(model)
        model_config = getattr(model_chunk, "config", None)
        if getattr(model_config, "pipeline_model_parallel_size", 1) <= 1:
            return False
        if getattr(model_chunk, "pre_process", False):
            return False

        inner_model = getattr(model_chunk, "language_model", model_chunk)
        return bool(getattr(inner_model, "mtp_process", False))

    def _get_tied_output_hf_name(
        self, task: WeightConversionTask, mapping_registry: MegatronMappingRegistry
    ) -> str | None:
        """Resolve the HF output-weight alias for a tied embedding task."""
        embedding_suffix = "embedding.word_embeddings.weight"
        if not task.global_param_name.endswith(embedding_suffix):
            return None
        if not isinstance(task.mapping.hf_param, str):
            raise ValueError(
                "Expected tied embedding mapping to resolve to one HuggingFace parameter, "
                f"got {task.mapping.hf_param!r}."
            )

        megatron_prefix = task.global_param_name[: -len(embedding_suffix)]
        output_mapping = mapping_registry.megatron_to_hf_lookup(
            f"{megatron_prefix}output_layer.weight"
        )
        if output_mapping is None:
            return None
        if not isinstance(output_mapping.hf_param, str):
            raise ValueError(
                "Expected tied output-layer mapping to resolve to one HuggingFace parameter, "
                f"got {output_mapping.hf_param!r}."
            )
        if output_mapping.hf_param == task.mapping.hf_param:
            return None
        return output_mapping.hf_param

    def build_conversion_tasks(
        self,
        hf_pretrained: HFPreTrained,
        megatron_model: List[MegatronModel],
        weight_dtype: Optional[torch.dtype] = None,
        *,
        exclude_module_types: tuple[type[torch.nn.Module], ...] = (),
        exclude_param_patterns: tuple[str, ...] = (),
    ) -> List[WeightConversionTask]:
        """Construct the conversion tasks between HF and megatron.

        Args:
            weight_dtype: Export dtype recorded on each task. Overrides must forward it.
            exclude_module_types: Module types whose parameters and persistent buffers
                are excluded from discovery. Remaining parameters still require mappings.
            exclude_param_patterns: Glob patterns for local parameter and persistent-buffer
                names to exclude before mapping validation. Defaults to no exclusions.

        The algorithm walks over every parameter of every destination model,
        asks the :class:`MegatronMappingRegistry` whether it has a mapping for that
        parameter and returns a concrete task describing exactly how that
        parameter will be populated. Missing mappings or source weights are
        conversion errors, not empty task slots.
        """

        has_hf_state = hasattr(hf_pretrained, "state") and hasattr(hf_pretrained.state, "source")
        if not isinstance(hf_pretrained, PretrainedConfig) and not has_hf_state:
            raise ValueError("hf_pretrained.state.source is required for weight ordering")

        self.hf_pretrained = hf_pretrained
        self.hf_config = hf_pretrained.config if hasattr(hf_pretrained, "config") else hf_pretrained

        hf_keys: Optional[Iterable[str]] = (
            hf_pretrained.state.source.get_all_keys() if has_hf_state else None
        )
        hf_key_set = set(hf_keys) if hf_keys is not None else None

        mapping_registry = self.mapping_registry()
        pg_collection = _get_pg_collection_from_model(megatron_model)
        mapping_registry.set_process_groups_from_pg_collection(pg_collection)
        unwrapped_model = unwrap_model(megatron_model)[0]
        model_config = unwrapped_model.config
        embeddings_are_tied = self._share_embeddings_and_output_weights(unwrapped_model)
        pp_rank = _get_pp_rank(megatron_model)
        sorted_global_param_names_all_pp_ranks = self._megatron_global_param_names_all_pp_ranks(
            megatron_model,
            **({"exclude_module_types": exclude_module_types} if exclude_module_types else {}),
            **(
                {"exclude_param_patterns": exclude_param_patterns} if exclude_param_patterns else {}
            ),
        )

        # Filter out the output_layer weight if embeddings are tied to it -- it doesn't exist
        # as a separate parameter in that case. Other `output_layer.*` parameters (e.g. a
        # standalone bias, as used by BERT-style masked-LM heads) are untouched by weight tying
        # and must still be converted.
        if embeddings_are_tied:
            sorted_global_param_names_all_pp_ranks = [
                name
                for name in sorted_global_param_names_all_pp_ranks
                if not name.endswith("output_layer.weight")
            ]

        mappings_by_global_name = self._validate_conversion_mappings(
            mapping_registry, sorted_global_param_names_all_pp_ranks, hf_key_set
        )

        global_names_index_dict = {
            name: idx for idx, name in enumerate(sorted_global_param_names_all_pp_ranks)
        }

        pending_tasks: list[WeightConversionTask | None] = [None] * len(
            sorted_global_param_names_all_pp_ranks
        )
        for vp_stage, model in enumerate(megatron_model):
            for local_name, _ in _named_conversion_parameters(
                model, exclude_module_types, exclude_param_patterns=exclude_param_patterns
            ):
                if "_extra_state" in local_name:
                    continue

                local_name = self._unwrap_name(local_name)
                global_name = _megatron_local_name_to_global(
                    megatron_model, model_config, local_name, vp_stage
                )
                # if name removed due to some reason, continue. e.g. embeddings_are_tied
                if global_name not in global_names_index_dict:
                    print_rank_0(f"WARNING: {global_name} not in global_names_index_dict")
                    continue
                global_name_idx = global_names_index_dict[global_name]
                mapping = mappings_by_global_name[global_name]

                local_module, local_weights = get_module_and_param_from_name(
                    megatron_model, local_name, vp_stage
                )
                if local_module is not None and not hasattr(local_module, "config"):
                    # If module is not a MegatronModule (e.g. torch.nn.Conv1d or a module list) we need
                    # to get the config from the model
                    setattr(local_module, "config", model_config)

                pending_tasks[global_name_idx] = WeightConversionTask(
                    pp_rank=pp_rank,
                    vp_stage=vp_stage,
                    param_name=local_name,
                    global_param_name=global_name,
                    megatron_module=local_module,
                    param_weight=local_weights,
                    mapping=mapping,
                    weight_dtype=weight_dtype,
                    required_hf_param_names=self.get_hf_import_param_names(
                        mapping.hf_param, hf_key_set
                    ),
                )

        # Fill the remaining ones for pp communications
        for idx, global_name in enumerate(sorted_global_param_names_all_pp_ranks):
            if pending_tasks[idx] is None:
                mapping = mappings_by_global_name[global_name]
                # This is an exception here we pass in global name
                # we are not using global_name to extract module and weights
                # only use it for param mapping auto dispatch checks
                pending_tasks[idx] = WeightConversionTask(
                    pp_rank=pp_rank,
                    vp_stage=None,
                    param_name=global_name,
                    global_param_name=global_name,
                    megatron_module=None,
                    param_weight=None,
                    mapping=mapping,
                    weight_dtype=weight_dtype,
                    required_hf_param_names=self.get_hf_import_param_names(
                        mapping.hf_param, hf_key_set
                    ),
                )

        return self._require_concrete_tasks(pending_tasks)

    def _validate_conversion_mappings(
        self,
        mapping_registry: MegatronMappingRegistry,
        global_param_names: Iterable[str],
        hf_keys: Iterable[str] | None = None,
    ) -> dict[str, MegatronParamMapping]:
        """Resolve and validate mappings for the full cross-PP parameter list."""
        mappings_by_global_name: dict[str, MegatronParamMapping] = {}
        missing_mappings: list[str] = []
        missing_hf_weights: list[tuple[str, str]] = []
        hf_key_set = set(hf_keys) if hf_keys is not None else None

        for global_name in global_param_names:
            mapping = mapping_registry.megatron_to_hf_lookup(global_name)
            if mapping is None:
                missing_mappings.append(global_name)
                continue

            mappings_by_global_name[global_name] = mapping
            if hf_key_set is None or mapping.allow_hf_name_mismatch:
                continue

            expected_hf_names = (
                [mapping.hf_param]
                if isinstance(mapping.hf_param, str)
                else list(mapping.hf_param.values())
            )
            missing_hf_weights.extend(
                (global_name, hf_name) for hf_name in expected_hf_names if hf_name not in hf_key_set
            )

        if missing_mappings:
            missing_names = "\n  ".join(missing_mappings)
            raise ValueError(
                "No mapping found for the following Megatron parameter(s):\n"
                f"  {missing_names}\n"
                "Every global Megatron parameter must have a concrete mapping so import and export remain strict."
            )

        if missing_hf_weights:
            missing_names = "\n  ".join(
                f"{global_name} -> {hf_name}" for global_name, hf_name in missing_hf_weights
            )
            raise ValueError(
                "Hugging Face checkpoint is missing mapped parameter(s):\n"
                f"  {missing_names}\n"
                "If the HF config determines whether the weight exists, register the mapping "
                "conditionally on that config instead. If it does not, and the name is synthesized "
                "or the weight is absent on only some layers, set allow_hf_name_mismatch on the "
                "mapping."
            )

        return mappings_by_global_name

    @staticmethod
    def _require_concrete_tasks(
        pending_tasks: Iterable[WeightConversionTask | None],
    ) -> list[WeightConversionTask]:
        """Return tasks after enforcing the internal no-empty-slot invariant."""
        tasks: list[WeightConversionTask] = []
        for task in pending_tasks:
            if task is None:
                raise RuntimeError(
                    "Internal error: conversion task construction left an empty slot"
                )
            tasks.append(task)
        return tasks

    def _detect_fp8_params(
        self,
        megatron_model: List[MegatronModel],
        model_config: TransformerConfig,
        sorted_global_param_names_all_pp_ranks: List[str],
        pp_group: Any,
        fp8_scale_inv_attr: str,
    ) -> Dict[str, FP8ExportLayout]:
        """Detect supported FP8 parameters and gather layouts across pipeline parallel ranks.

        This method scans all parameters in the megatron model to determine which ones are
        blockwise or MXFP8 tensors with valid scale metadata. It then gathers these layouts
        across all pipeline parallel ranks to ensure consistent decisions.

        Args:
            megatron_model: List of Megatron model instances
            model_config: Transformer configuration
            sorted_global_param_names_all_pp_ranks: Sorted list of global parameter names
            pp_group: Pipeline parallel group for distributed communication
            fp8_scale_inv_attr: Metadata key for the FP8 scale_inv tensor. A leading
                underscore is accepted for backward compatibility.

        Returns:
            Dictionary mapping global parameter names to FP8 export layouts.
        """
        local_fp8_layouts: Dict[str, FP8ExportLayout | Literal[-1]] = {}
        global_name_set = set(sorted_global_param_names_all_pp_ranks)

        for vp_stage, model in enumerate(megatron_model):
            for local_name, _ in itertools.chain(
                model.named_parameters(), persistent_buffers(model)
            ):
                if "_extra_state" in local_name:
                    continue

                local_name = self._unwrap_name(local_name)
                global_name = _megatron_local_name_to_global(
                    megatron_model, model_config, local_name, vp_stage
                )
                if global_name not in global_name_set:
                    continue

                _, local_weights = get_module_and_param_from_name(
                    megatron_model, local_name, vp_stage
                )
                if local_weights is None:
                    continue

                layout = detect_fp8_export_layout(
                    local_weights,
                    fp8_recipe=getattr(model_config, "fp8_recipe", None),
                    fp8_scale_inv_attr=fp8_scale_inv_attr,
                )
                if layout is not None:
                    # Reject on every PP rank before creating raw storage views,
                    # which do not retain the parameter's GTP sharding metadata.
                    local_fp8_layouts[global_name] = -1 if _is_gtp_param(local_weights) else layout

        # Gather across PP ranks to ensure consistent insertion decisions
        fp8_layouts_list: list[Dict[str, FP8ExportLayout | Literal[-1]] | None] = [
            None
        ] * get_pg_size(pp_group)
        torch.distributed.all_gather_object(fp8_layouts_list, local_fp8_layouts, group=pp_group)
        global_fp8_layouts: Dict[str, FP8ExportLayout] = {}
        for layouts in fp8_layouts_list:
            if not layouts:
                continue
            # Validate only after every PP rank has completed the collective.
            for global_name, layout in layouts.items():
                if layout == -1:
                    raise ValueError(
                        f"{global_name}: raw FP8 export does not support GTP sharding; "
                        "use gathered dequantized export or gathered quantized export instead"
                    )
                try:
                    layout.validate()
                except ValueError as error:
                    raise ValueError(f"{global_name}: {error}") from None
                global_fp8_layouts[global_name] = layout

        return global_fp8_layouts

    def build_export_fp8_tasks(
        self,
        hf_pretrained: HFPreTrained,
        megatron_model: List[MegatronModel],
        *,
        scale_inv_suffix: str = "_scale_inv",
        fp8_scale_inv_attr: str = "_rowwise_scale_inv",
    ) -> List[WeightConversionTask]:
        """
        Build Megatron→(export) conversion tasks, inserting extra *scale_inv* tasks for supported FP8 params.
        """

        # Ensure hf_pretrained has the required state structure (reuse existing ordering assumptions)
        if not (hasattr(hf_pretrained, "state") and hasattr(hf_pretrained.state, "source")):
            raise ValueError("hf_pretrained.state.source is required for weight ordering")

        # Some registries read the source checkpoint's names, so attach it first as build_conversion_tasks does.
        self.hf_pretrained = hf_pretrained
        self.hf_config = hf_pretrained.config if hasattr(hf_pretrained, "config") else hf_pretrained
        mapping_registry = self.mapping_registry()
        pg_collection = _get_pg_collection_from_model(megatron_model)
        mapping_registry.set_process_groups_from_pg_collection(pg_collection)
        unwrapped_model = unwrap_model(megatron_model)[0]
        model_config = unwrapped_model.config
        embeddings_are_tied = self._share_embeddings_and_output_weights(unwrapped_model)
        pp_rank = _get_pp_rank(megatron_model)
        pp_group = _get_pp_group(megatron_model)
        sorted_global_param_names_all_pp_ranks = self._megatron_global_param_names_all_pp_ranks(
            megatron_model
        )

        # Filter out the output_layer weight if embeddings are tied to it -- it doesn't exist
        # as a separate parameter in that case. Other `output_layer.*` parameters (e.g. a
        # standalone bias, as used by BERT-style masked-LM heads) are untouched by weight tying
        # and must still be converted.
        if embeddings_are_tied:
            sorted_global_param_names_all_pp_ranks = [
                name
                for name in sorted_global_param_names_all_pp_ranks
                if not name.endswith("output_layer.weight")
            ]

        mappings_by_global_name = self._validate_conversion_mappings(
            mapping_registry, sorted_global_param_names_all_pp_ranks
        )

        # 1) Determine which global params use a supported FP8 layout and gather layouts across PP ranks
        global_fp8_layouts = self._detect_fp8_params(
            megatron_model,
            model_config,
            sorted_global_param_names_all_pp_ranks,
            pp_group,
            fp8_scale_inv_attr,
        )

        # 2) Expand the global name list with `*.scale_inv` entries.
        #    This defines the final deterministic task ordering.
        expanded_global_names: list[str] = []
        for global_name in sorted_global_param_names_all_pp_ranks:
            expanded_global_names.append(global_name)
            if global_name in global_fp8_layouts:
                expanded_global_names.append(f"{global_name}{scale_inv_suffix}")

        global_names_index_dict = {name: idx for idx, name in enumerate(expanded_global_names)}

        tasks: list[None | WeightConversionTask] = [None] * len(expanded_global_names)

        # 3) Fill tasks for params that are local to this rank.
        for vp_stage, model in enumerate(megatron_model):
            for local_name, _ in itertools.chain(
                model.named_parameters(), persistent_buffers(model)
            ):
                if "_extra_state" in local_name:
                    continue

                local_name = self._unwrap_name(local_name)
                global_name = _megatron_local_name_to_global(
                    megatron_model, model_config, local_name, vp_stage
                )
                if self._should_skip_mtp_duplicate_embedding_export(global_name, model):
                    continue
                if global_name not in global_names_index_dict:
                    continue

                mapping = mappings_by_global_name[global_name]
                local_module, local_weights = get_module_and_param_from_name(
                    megatron_model, local_name, vp_stage
                )
                if local_module is not None and not hasattr(local_module, "config"):
                    setattr(local_module, "config", model_config)

                # Main (weight/bias) task
                export_weight_tensor = local_weights
                fp8_layout = global_fp8_layouts.get(global_name)
                if fp8_layout is not None:
                    export_weight_tensor, scale_tensor = get_fp8_export_tensors(
                        local_weights, fp8_scale_inv_attr=fp8_scale_inv_attr
                    )
                tasks[global_names_index_dict[global_name]] = WeightConversionTask(
                    pp_rank=pp_rank,
                    vp_stage=vp_stage,
                    param_name=local_name,
                    global_param_name=global_name,
                    megatron_module=local_module,
                    param_weight=export_weight_tensor,
                    mapping=mapping,
                )

                # Optional scale_inv task (only for globally-detected FP8 params)
                if fp8_layout is not None:
                    scale_global_name = f"{global_name}{scale_inv_suffix}"
                    scale_local_name = f"{local_name}{scale_inv_suffix}"
                    # Note:
                    # Do NOT reuse the same mapping instance as the base weight task.
                    # We clone via `resolve(())` which returns a new mapping instance
                    base_mapping_for_scale = mapping_registry.resolve_mapping(mapping, ())
                    tasks[global_names_index_dict[scale_global_name]] = WeightConversionTask(
                        pp_rank=pp_rank,
                        vp_stage=vp_stage,
                        param_name=scale_local_name,
                        global_param_name=scale_global_name,
                        megatron_module=local_module,
                        param_weight=scale_tensor,
                        mapping=_HFNameSuffixMapping(
                            base_mapping_for_scale, scale_inv_suffix, fp8_layout.block_shape[0]
                        ),
                    )

        # 4) Fill remaining placeholders (for PP ranks that don't own certain params).
        for idx, global_name in enumerate(expanded_global_names):
            if tasks[idx] is not None:
                continue

            # For scale_inv entries, reuse the base param's mapping type.
            if global_name.endswith(scale_inv_suffix):
                base_global_name = global_name[: -len(scale_inv_suffix)]
                base_mapping = mappings_by_global_name[base_global_name]
                # clone mapping instance to avoid sharing state across tasks.
                base_mapping_for_scale = mapping_registry.resolve_mapping(base_mapping, ())
                mapping = _HFNameSuffixMapping(
                    base_mapping_for_scale,
                    scale_inv_suffix,
                    global_fp8_layouts[base_global_name].block_shape[0],
                )
            else:
                mapping = mappings_by_global_name[global_name]

            tasks[idx] = WeightConversionTask(
                pp_rank=pp_rank,
                vp_stage=None,
                param_name=global_name,
                global_param_name=global_name,
                megatron_module=None,
                param_weight=None,
                mapping=mapping,
            )

        return self._require_concrete_tasks(tasks)

    @classmethod
    def register_bridge(
        cls,
        *,
        source: Type[PreTrainedModel] | str,
        target: Type[MegatronModel],
        provider: Type[ModelProviderTarget] | None = None,
        model_type: str | None = None,
    ) -> Callable[[_BridgeImplClass], _BridgeImplClass]:
        """Class decorator for registering bridge implementations.

        This decorator registers a MegatronModelBridge subclass with the dispatch
        system, enabling automatic routing of conversions based on the source
        HuggingFace model type and target Megatron model type.

        Args:
            source (Type[PreTrainedModel] | str): HuggingFace PreTrainedModel class
                (e.g., LlamaForCausalLM) or the class name as a string. Using a
                string allows registering bridges for architectures that are only
                available via auto_map.
            target (Type[MegatronModel]): Megatron model class (e.g., GPTModel).
            provider (Type[ModelProviderTarget], optional): Provider class to use
                for this model (e.g., DeepSeekModelProvider for MLA models).
                Defaults to GPTModelProvider if not specified.
            model_type (str, optional): HuggingFace model_type string (e.g., "llama").
                Used for megatron_to_hf_config conversion.

        Returns:
            Callable[[_BridgeImplClass], _BridgeImplClass]: Decorator function
                that registers the bridge implementation.

        Example:
            .. code-block:: python

                @MegatronModelBridge.register_bridge(
                    source=LlamaForCausalLM, target=GPTModel, model_type="llama"
                )
                class MegatronCausalLlamaBridge(MegatronModelBridge):
                    def provider_bridge(self, hf_pretrained):
                        # Implementation
                        pass

                    def mapping_registry(self):
                        # Implementation
                        pass

            String-based registration with custom provider:

            .. code-block:: python

                @MegatronModelBridge.register_bridge(
                    source="DeepseekV3ForCausalLM",
                    target=GPTModel,
                    provider=DeepSeekModelProvider,
                    model_type="deepseek_v3",
                )
                class MegatronDeepseekV3Bridge(MegatronModelBridge):
                    ...

        Note:
            The decorated class is registered with multiple dispatchers to handle
            different conversion scenarios. The registration is automatic when the
            class is defined.
        """

        return create_bridge_decorator(
            source=source, target=target, provider=provider, model_type=model_type
        )


# Core dispatch functions
@dispatch
def get_model_bridge(hf_architecture, hf_config=None) -> "MegatronModelBridge":
    """Get the appropriate model bridge for a given HuggingFace architecture."""
    ...


@dispatch
def stream_weights_megatron_to_hf(
    dispatch_instance: MegatronModel,
    megatron_model: Union[MegatronModel, List[MegatronModel]],
    hf_pretrained: HFPreTrained,
    cpu: bool = True,
    show_progress: bool = True,
    conversion_tasks: Optional[List[WeightConversionTask]] = None,
    current_pp_stage_only: bool = False,
) -> Iterable[HFWeightTuple]:
    """Bridge Megatron model state to HuggingFace format."""
    ...


@dispatch
def stream_weights_megatron_to_hf_quant(
    dispatch_instance: MegatronModel,
    megatron_model: Union[MegatronModel, List[MegatronModel]],
    hf_pretrained: HFPreTrained,
    quantization_checker: Callable[[str], bool],
    quant_fn: Callable[..., Tuple[torch.Tensor, torch.Tensor]],
    quant_block_size: Optional[Tuple[int, int]] = None,
    cpu: bool = True,
    show_progress: bool = True,
    conversion_tasks: Optional[List[WeightConversionTask]] = None,
) -> Iterable[HFWeightTuple]:
    """Bridge Megatron model state to HuggingFace format with quantization."""
    ...


def register_bridge_implementation(
    *,
    source: Type["PreTrainedModel"] | str,
    target: Type["MegatronModule"],
    bridge_class: Type["MegatronModelBridge"],
) -> None:
    """Register a bridge implementation with the dispatch system.

    Args:
        source: HuggingFace PreTrainedModel class or the class name as a string.
            Using a string allows registering bridges for architectures that are
            available only via auto_map.
        target: Megatron model class (e.g., GPTModel)
        bridge_class: MegatronModelBridge implementation class
    """
    bridge_class_name = bridge_class.__name__

    @get_model_bridge.impl(source)
    def _get_model_bridge_impl(_, hf_config=None) -> "MegatronModelBridge":
        bridge = bridge_class()
        if hf_config is not None:
            # `hf_config` may be a raw config or a full HF wrapper; normalize both onto the bridge.
            bridge.hf_pretrained = hf_config
            bridge.hf_config = hf_config.config if hasattr(hf_config, "config") else hf_config
        return bridge

    @stream_weights_megatron_to_hf.impl((source, target))
    def _megatron_to_hf_registered_impl(
        _,
        megatron_model: Union[MegatronModel, List[MegatronModel]],
        hf_pretrained: HFPreTrained,
        cpu: bool = True,
        show_progress: bool = True,
        conversion_tasks: Optional[List[WeightConversionTask]] = None,
        current_pp_stage_only: bool = False,
    ) -> Iterable[HFWeightTuple]:
        bridge = bridge_class()

        # allow bridge to access model config
        bridge.hf_pretrained = hf_pretrained
        bridge.hf_config = (
            hf_pretrained.config if hasattr(hf_pretrained, "config") else hf_pretrained
        )

        return bridge.stream_weights_megatron_to_hf(
            megatron_model,
            hf_pretrained,
            cpu=cpu,
            show_progress=show_progress,
            conversion_tasks=conversion_tasks,
            **({"current_pp_stage_only": True} if current_pp_stage_only else {}),
        )

    @stream_weights_megatron_to_hf_quant.impl((source, target))
    def _megatron_to_hf_quant_registered_impl(
        _,
        megatron_model: Union[MegatronModel, List[MegatronModel]],
        hf_pretrained: HFPreTrained,
        quantization_checker: Callable[[str], bool],
        quant_fn: Callable[..., Tuple[torch.Tensor, torch.Tensor]],
        quant_block_size: Optional[Tuple[int, int]] = None,
        cpu: bool = True,
        show_progress: bool = True,
        conversion_tasks: Optional[List[WeightConversionTask]] = None,
    ) -> Iterable[HFWeightTuple]:
        bridge = bridge_class()

        bridge.hf_config = (
            hf_pretrained.config if hasattr(hf_pretrained, "config") else hf_pretrained
        )

        return bridge.stream_weights_megatron_to_hf_quant(
            megatron_model,
            hf_pretrained,
            quantization_checker,
            quant_fn,
            quant_block_size=quant_block_size,
            cpu=cpu,
            show_progress=show_progress,
            conversion_tasks=conversion_tasks,
        )

    # Set meaningful names for debugging
    _get_model_bridge_impl.__name__ = f"_bridge_with_{bridge_class_name}"
    _megatron_to_hf_registered_impl.__name__ = f"_megatron_to_hf_with_{bridge_class_name}"


def create_bridge_decorator(
    *,
    source: Type["PreTrainedModel"] | str,
    target: Type["MegatronModule"],
    provider: Type["ModelProviderMixin"] | None = None,
    model_type: str | None = None,
) -> Callable[[Type["MegatronModelBridge"]], Type["MegatronModelBridge"]]:
    """Create a decorator for registering bridge implementations.

    Args:
        source: HuggingFace PreTrainedModel class or the class name as a string
            (useful for auto_map architectures)
        target: Megatron model class
        provider: Provider class to use for this model (e.g., DeepSeekModelProvider)
        model_type: HuggingFace model_type string (e.g., "llama", "deepseek_v3")

    Returns:
        Decorator function that registers the bridge implementation
    """

    def decorator(bridge_class: Type["MegatronModelBridge"]) -> Type["MegatronModelBridge"]:
        # Store source name for HF config generation
        bridge_class.SOURCE_NAME = source if isinstance(source, str) else source.__name__
        # Store model_type for HF config generation
        if model_type is not None:
            bridge_class.MODEL_TYPE = model_type
        # Set the provider class on the bridge
        if provider is not None:
            bridge_class.PROVIDER_CLASS = provider
        register_bridge_implementation(source=source, target=target, bridge_class=bridge_class)
        return bridge_class

    return decorator
