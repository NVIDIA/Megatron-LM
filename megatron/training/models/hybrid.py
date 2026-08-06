# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar, Literal, override

from megatron.core.distributed.distributed_data_parallel_config import DistributedDataParallelConfig
from megatron.core.enums import ModelType
from megatron.core.models.hybrid.hybrid_architecture import (
    HybridLayerPattern,
    resolve_hybrid_architecture,
)
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_inference_stack_spec
from megatron.core.models.hybrid.hybrid_layer_specs import (
    hybrid_stack_spec as default_hybrid_stack_spec,
)
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.pipeline_parallel.utils import (
    is_pp_first_stage,
    is_pp_last_stage,
    is_vp_first_stage,
    is_vp_last_stage,
)
from megatron.core.post_training.modelopt.hybrid.model_specs import get_hybrid_stack_modelopt_spec
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.module import Float16Module, MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.models.base import ModelBuilder, ModelConfig, compose_hooks
from megatron.training.models.dist_utils import unimodal_build_distributed_models
from megatron.training.vocab_utils import calculate_padded_vocab_size

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class HybridModelConfig(ModelConfig):
    """Configuration for a Megatron Core Hybrid model.

    This is purely a configuration object. All model construction
    logic lives in ``HybridModelBuilder``.

    Contains a ``TransformerConfig`` alongside Hybrid-specific parameters. Attributes
    on the embedded ``transformer`` config are accessible directly on this object
    via ``__getattr__``/``__setattr__`` proxying.

    Direct ``layer_specs`` are the preferred architecture API. The legacy
    ``hybrid_layer_pattern`` string remains supported for compatibility.

    Note:
        ``vocab_size`` must be set before passing this config to ``HybridModelBuilder``.
        ``hybrid_attention_ratio``,``hybrid_mlp_ratio``, and
        ``hybrid_override_pattern`` are deprecated and will be removed in a future release.
    """

    builder: ClassVar[str] = "megatron.training.models.hybrid.HybridModelBuilder"
    transformer: TransformerConfig
    fp16_lm_cross_entropy: bool = False
    parallel_output: bool = True
    share_embeddings_and_output_weights: bool = False
    hybrid_attention_ratio: float = 0.0
    hybrid_mlp_ratio: float = 0.0
    hybrid_override_pattern: str | None = None
    hybrid_layer_pattern: str | None = None
    layer_specs: HybridLayerPattern | None = field(default=None, repr=False)
    mtp_layer_specs: HybridLayerPattern | None = field(default=None, repr=False)
    seq_length: int = 8192
    # HybridModel with no attention has no need for position embeddings, so none is default
    position_embedding_type: Literal["learned_absolute", "rope", "none"] = "none"
    rotary_percent: float = 1.0
    rotary_base: int = 10000
    seq_len_interpolation_factor: float | None = None
    make_vocab_size_divisible_by: int = 128
    hybrid_stack_spec: ModuleSpec | None = None
    vocab_size: int | None = None
    should_pad_vocab: bool = False

    @property
    def mamba_stack_spec(self):
        """Deprecated alias for hybrid_stack_spec."""
        return self.hybrid_stack_spec

    @mamba_stack_spec.setter
    def mamba_stack_spec(self, value):
        self.hybrid_stack_spec = value

    @override
    def __getattr__(self, name: str, /) -> Any:
        # __getattr__ is only called when normal attribute lookup has already failed,
        # so use object.__getattribute__ to fetch `transformer` without recursing.
        try:
            transformer = object.__getattribute__(self, "transformer")
        except AttributeError:
            raise AttributeError(f"HybridModelConfig has no attribute '{name}'")
        if hasattr(transformer, name):
            return getattr(transformer, name)
        raise AttributeError(f"Neither HybridModelConfig nor TransformerConfig has any attribute '{name}'.")

    @override
    def __setattr__(self, name: str, value: Any, /) -> None:
        # Use object.__getattribute__ to avoid triggering __getattr__ while
        # `transformer` may not yet exist (e.g. during dataclass __init__).
        try:
            transformer = object.__getattribute__(self, "transformer")
        except AttributeError:
            # `transformer` not yet initialised; store the attribute on self.
            super().__setattr__(name, value)
            return
        if hasattr(transformer, name):
            setattr(transformer, name, value)
        else:
            super().__setattr__(name, value)

    def finalize(self) -> None:
        """One time validation to run once config is ready to be used by builder."""

        if hasattr(self.transformer, "finalize") and callable(self.transformer.finalize):
            self.transformer.finalize()

    @override
    def as_dict(self) -> dict[str, Any]:
        """Serialize scalar configuration without live Python architecture objects.

        The Python recipe must supply direct ``ModuleSpec`` trees again when resuming.
        """

        result = super().as_dict()
        result.pop("layer_specs", None)
        result.pop("mtp_layer_specs", None)
        return result


class HybridModelBuilder(ModelBuilder[HybridModel, HybridModelConfig]):
    """Builder to construct Megatron Core Hybrid models.

    Example:
        >>> transformer_cfg = TransformerConfig(num_layers=32, hidden_size=4096, ...)
        >>> model_cfg = HybridModelConfig(transformer=transformer_cfg, vocab_size=32000, seq_length=2048, ...)
        >>>
        >>> # Single stage (e.g. inference)
        >>> model = HybridModelBuilder(model_cfg).build_model(pg_collection)
        >>>
        >>> # Distributed training
        >>> models = HybridModelBuilder(model_cfg).build_distributed_models(pg_collection)
    """

    def __init__(self, model_config: HybridModelConfig):
        super().__init__(model_config)
        has_direct_specs = (
            model_config.layer_specs is not None or model_config.mtp_layer_specs is not None
        )
        # Only direct descriptors need topology resolution before distributed construction.
        # Legacy patterns retain the historical builder and runtime selector path.
        if has_direct_specs:
            self._hybrid_stack_spec = self._get_hybrid_stack_spec()
            self._resolved_architecture = resolve_hybrid_architecture(
                config=model_config.transformer,
                hybrid_stack_spec=self._hybrid_stack_spec,
                layer_specs=model_config.layer_specs,
                mtp_layer_specs=model_config.mtp_layer_specs,
                hybrid_layer_pattern=model_config.hybrid_layer_pattern,
            )

    @classmethod
    def prepare_config_for_distributed_init(
        cls, model_config: HybridModelConfig, args: Any
    ) -> bool:
        """Resolve split topology before Megatron initializes pipeline runtime state.

        Direct Python specs are not present while command-line arguments are
        validated, so their inferred VPP size must be copied to the runtime
        namespace before ``initialize_model_parallel`` runs. Returns whether
        direct architecture state was prepared.
        """

        if model_config.layer_specs is None and model_config.mtp_layer_specs is None:
            return False

        builder = cls(model_config)
        resolved_architecture = getattr(builder, "_resolved_architecture", None)
        if resolved_architecture is None:
            return False

        transformer = model_config.transformer
        pp_size = transformer.pipeline_model_parallel_size
        runtime_pp_size = getattr(args, "pipeline_model_parallel_size", pp_size)
        if runtime_pp_size != pp_size:
            raise ValueError(
                "HybridModelConfig.transformer.pipeline_model_parallel_size must match the "
                f"runtime pipeline topology; got {pp_size} != {runtime_pp_size}."
            )

        inferred_vp_size = transformer.virtual_pipeline_model_parallel_size
        runtime_vp_size = getattr(args, "virtual_pipeline_model_parallel_size", None)
        if runtime_vp_size is not None and runtime_vp_size != inferred_vp_size:
            raise ValueError(
                "Hybrid architecture splits disagree with the runtime virtual pipeline "
                f"topology; got {inferred_vp_size} != {runtime_vp_size}."
            )
        args.virtual_pipeline_model_parallel_size = inferred_vp_size

        # Argument validation temporarily disables interleaved-only options when
        # direct Python split nodes are not available yet. Restore the user's
        # requested settings now that those nodes have inferred VPP.
        if inferred_vp_size is not None and runtime_vp_size is None:
            requested_overlap = getattr(
                args,
                "_overlap_p2p_comm_before_direct_vpp",
                getattr(args, "overlap_p2p_comm", False),
            )
            requested_align = getattr(
                args,
                "_align_param_gather_before_direct_vpp",
                getattr(args, "align_param_gather", False),
            )
            if pp_size == 2 and not requested_overlap:
                raise ValueError(
                    "Direct PP2/VPP interleaving requires P2P communication overlap; "
                    "remove --no-overlap-p2p-communication."
                )
            args.overlap_p2p_comm = requested_overlap
            args.align_param_gather = requested_align
            if hasattr(args, "batch_p2p_comm"):
                args.batch_p2p_comm = not requested_overlap
            transformer.overlap_p2p_comm = requested_overlap
            transformer.batch_p2p_comm = not requested_overlap

        return True

    def _get_hybrid_stack_spec(self) -> ModuleSpec:
        """Select the stack implementation used by every local model chunk."""

        hybrid_stack_spec = self._model_config.hybrid_stack_spec
        if hybrid_stack_spec is not None:
            return hybrid_stack_spec
        if self._model_config.transformer.transformer_impl == "inference_optimized":
            return hybrid_inference_stack_spec
        if self._model_config.restore_modelopt_state:
            return get_hybrid_stack_modelopt_spec(
                local_core_attention=False,
                remap_te_layernorm=False,
            )
        return default_hybrid_stack_spec

    def build_model(
        self,
        pg_collection: ProcessGroupCollection,
        pre_process: bool | None = None,
        post_process: bool | None = None,
        vp_stage: int | None = None,
    ) -> HybridModel:
        """Build a single ``MCoreHybridModel`` stage.

        Args:
            pg_collection: Process groups for distributed training
            pre_process: Include embedding layer
            post_process: Include output layer
            vp_stage: Virtual pipeline stage

        Returns:
            The constructed model

        """
        # Re-resolve if a caller selected a different implementation spec after
        # builder construction (for example, modelopt or optimized inference).
        hybrid_stack_spec = self._get_hybrid_stack_spec()
        has_direct_specs = (
            self._model_config.layer_specs is not None
            or self._model_config.mtp_layer_specs is not None
        )
        if has_direct_specs and hybrid_stack_spec is not getattr(
            self, "_hybrid_stack_spec", None
        ):
            self._hybrid_stack_spec = hybrid_stack_spec
            self._resolved_architecture = resolve_hybrid_architecture(
                config=self._model_config.transformer,
                hybrid_stack_spec=hybrid_stack_spec,
                layer_specs=self._model_config.layer_specs,
                mtp_layer_specs=self._model_config.mtp_layer_specs,
                hybrid_layer_pattern=self._model_config.hybrid_layer_pattern,
            )

        assert self._model_config.vocab_size is not None, "vocab_size must be configured before calling build_model()"
        if self._model_config.should_pad_vocab:
            padded_vocab_size = calculate_padded_vocab_size(
                self._model_config.vocab_size,
                self._model_config.make_vocab_size_divisible_by,
                self._model_config.transformer.tensor_model_parallel_size,
            )
        else:
            padded_vocab_size = self._model_config.vocab_size

        resolved_architecture = getattr(self, "_resolved_architecture", None)
        is_direct = resolved_architecture is not None
        if is_direct:
            vp_size = self._model_config.transformer.virtual_pipeline_model_parallel_size
            if vp_size is not None and vp_stage is None:
                # A single-chunk build defaults to the first virtual stage, matching
                # ResolvedHybridArchitecture.select_segment's public semantics.
                vp_stage = 0
            pre_process = (
                pre_process
                if pre_process is not None
                else is_pp_first_stage(pg_collection.pp) and is_vp_first_stage(vp_stage, vp_size)
            )
            post_process = (
                post_process
                if post_process is not None
                else is_pp_last_stage(pg_collection.pp) and is_vp_last_stage(vp_stage, vp_size)
            )
        else:
            pre_process = (
                pre_process if pre_process is not None else is_pp_first_stage(pg_collection.pp)
            )
            post_process = (
                post_process if post_process is not None else is_pp_last_stage(pg_collection.pp)
            )

        direct_architecture_kwargs = (
            {"resolved_hybrid_architecture": resolved_architecture} if is_direct else {}
        )
        return HybridModel(
            config=self._model_config.transformer,
            hybrid_stack_spec=hybrid_stack_spec,
            vocab_size=padded_vocab_size,
            max_sequence_length=self._model_config.seq_length,
            hybrid_layer_pattern=self._model_config.hybrid_layer_pattern,
            fp16_lm_cross_entropy=self._model_config.fp16_lm_cross_entropy,
            parallel_output=self._model_config.parallel_output,
            share_embeddings_and_output_weights=self._model_config.share_embeddings_and_output_weights,
            position_embedding_type=self._model_config.position_embedding_type,
            rotary_percent=self._model_config.rotary_percent,
            rotary_base=self._model_config.rotary_base,
            seq_len_interpolation_factor=self._model_config.seq_len_interpolation_factor,
            pre_process=pre_process,
            post_process=post_process,
            pg_collection=pg_collection,
            vp_stage=vp_stage,
            **direct_architecture_kwargs,
        )

    def build_distributed_models(
        self,
        pg_collection: ProcessGroupCollection,
        ddp_config: DistributedDataParallelConfig | None = None,
        overlap_param_gather_with_optimizer_step: bool = False,
        use_megatron_fsdp: bool = False,
        use_torch_fsdp2: bool = False,
        wrap_with_ddp: bool = True,
        data_parallel_random_init: bool = False,
        mixed_precision_wrapper: Callable[[Any, MegatronModule], MegatronModule] | None = Float16Module,
        model_type: ModelType = ModelType.encoder_or_decoder,
    ) -> list[HybridModel]:
        """Build model stages and wrap for distributed training.

        Args:
            pg_collection: Model communication process groups.
            ddp_config: DistributedDataParallel configuration
            overlap_param_gather_with_optimizer_step: Whether to overlap parameter
                gather with optimizer step.
            use_megatron_fsdp: Whether to use Megatron FSDP
            use_torch_fsdp2: Whether to use Torch FSDP 2.0
            wrap_with_ddp: Set to False to skip the DDP/FSDP wrapper.
            data_parallel_random_init: Whether to use data parallel random initialization
            mixed_precision_wrapper: Mixed precision wrapper, e.g. ``Float16Module``
            model_type: Deprecated flag, only used for backwards compatibility.

        Returns:
            List of model stages.
        """
        transformer_config = self._model_config.transformer
        composed_pre_wrap_hook = compose_hooks(self._model_config.pre_wrap_hooks)
        model_list = unimodal_build_distributed_models(
            self.build_model,
            transformer_config,
            pg_collection,
            ddp_config,
            overlap_param_gather_with_optimizer_step,
            use_megatron_fsdp,
            use_torch_fsdp2,
            wrap_with_ddp,
            data_parallel_random_init,
            mixed_precision_wrapper,
            composed_pre_wrap_hook,
            model_type,
        )

        composed_post_wrap_hook = compose_hooks(self._model_config.post_wrap_hooks)
        _model = composed_post_wrap_hook(model_list)
        if _model is not None:
            model_list = _model
        else:
            logger.warning("Final post wrap hook returned None, skipping post wrap hooks.")

        return model_list
