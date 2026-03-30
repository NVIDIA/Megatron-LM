# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import logging
from dataclasses import dataclass
from typing import Any, Callable, ClassVar, Literal, override

from megatron.core.distributed.distributed_data_parallel_config import DistributedDataParallelConfig
from megatron.core.enums import ModelType
from megatron.core.models.mamba.mamba_model import MambaModel
from megatron.core.models.mamba.mamba_layer_specs import mamba_stack_spec as default_mamba_stack_spec
from megatron.core.pipeline_parallel.utils import is_pp_first_stage, is_pp_last_stage
from megatron.core.post_training.modelopt.mamba.model_specs import get_mamba_stack_modelopt_spec
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.module import Float16Module, MegatronModule
from megatron.core.transformer.transformer_config import  TransformerConfig

from megatron.training.models.base import ModelConfig, ModelBuilder, compose_hooks
from megatron.training.vocab_utils import calculate_padded_vocab_size
from megatron.training.models.base import unimodal_build_distributed_models

logger = logging.getLogger(__name__)


def transformer_engine_mamba_stack_spec() -> ModuleSpec:
    """Return the default Mamba stack spec with Transformer Engine layers.

    This is a named function (not a lambda) to allow proper serialization
    and reconstruction from checkpoints. Named functions can be imported
    via their module path, unlike lambdas.

    Returns:
        Default Mamba stack specification from megatron.core
    """
    return default_mamba_stack_spec


def modelopt_mamba_stack_spec() -> ModuleSpec:
    """Mamba stack specification for quantization with ModelOpt.

    Uses Norm instead of TENorm and ColumnParallelLinear/RowParallelLinear
    instead of TE layers to enable proper quantizer insertion by ModelOpt.

    Returns:
        ModuleSpec: Module specification for quantization-ready Mamba stack
    """
    return get_mamba_stack_modelopt_spec(
        local_core_attention=False,
        remap_te_layernorm=False,
    )


def get_default_mamba_stack_spec(config: "MambaModelConfig") -> ModuleSpec:
    """Determine the most appropriate Mamba stack specification based on configuration.

    Args:
        config: Mamba configuration object

    Returns:
        ModuleSpec: Appropriate module specification based on config
    """
    if config.restore_modelopt_state:
        return modelopt_mamba_stack_spec()
    else:
        return transformer_engine_mamba_stack_spec()


@dataclass(kw_only=True)
class MambaModelConfig(ModelConfig):
    """Configuration for a Megatron Core Mamba (SSM) model.

    This is purely a configuration object. All model construction
    logic lives in ``MambaModelBuilder``.

    Contains a ``TransformerConfig`` alongside Mamba-specific parameters. Attributes
    on the embedded ``transformer`` config are accessible directly on this object
    via ``__getattr__``/``__setattr__`` proxying.

    Supports hybrid SSM/attention architectures via ``hybrid_layer_pattern``

    Note:
        ``vocab_size`` must be set before passing this config to ``MambaModelBuilder``.
        ``hybrid_attention_ratio``,``hybrid_mlp_ratio``, and
        ``hybrid_override_pattern`` are deprecated and will be removed in a future release.
    """

    builder: ClassVar[str] = "megatron.training.models.mamba.MambaModelBuilder"
    transformer: TransformerConfig
    fp16_lm_cross_entropy: bool = False
    parallel_output: bool = True
    share_embeddings_and_output_weights: bool = False
    hybrid_attention_ratio: float = 0.0
    hybrid_mlp_ratio: float = 0.0
    hybrid_override_pattern: str | None = None
    hybrid_layer_pattern: str | None = None
    seq_length: int = 8192
    # Mamba with no attention has no need for position embeddings, so none is default
    position_embedding_type: Literal["learned_absolute", "rope", "none"] = "none"
    rotary_percent: float = 1.0
    rotary_base: int = 10000
    seq_len_interpolation_factor: float | None = None
    make_vocab_size_divisible_by: int = 128
    mamba_stack_spec: ModuleSpec | Callable[[], ModuleSpec] | Callable[["MambaModelConfig"], ModuleSpec] = (
        get_default_mamba_stack_spec
    )
    vocab_size: int | None = None
    should_pad_vocab: bool = False

    @override
    def __getattr__(self, name: str, /) -> Any:
        # __getattr__ is only called when normal attribute lookup has already failed,
        # so use object.__getattribute__ to fetch `transformer` without recursing.
        try:
            transformer = object.__getattribute__(self, "transformer")
        except AttributeError:
            raise AttributeError(f"MambaModelConfig has no attribute '{name}'")
        if hasattr(transformer, name):
            return getattr(transformer, name)
        raise AttributeError(f"Neither MambaModelConfig nor TransformerConfig has any attribute '{name}'.")

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


class MambaModelBuilder(ModelBuilder[MambaModel, MambaModelConfig]):
    """Builder to construct Megatron Core Mamba models.

    Example:
        >>> transformer_cfg = TransformerConfig(num_layers=32, hidden_size=4096, ...)
        >>> model_cfg = MambaModelConfig(transformer=transformer_cfg, vocab_size=32000, seq_length=2048, ...)
        >>>
        >>> # Single stage (e.g. inference)
        >>> model = MambaModelBuilder(model_cfg).build_model(pg_collection)
        >>>
        >>> # Distributed training
        >>> models = MambaModelBuilder(model_cfg).build_distributed_models(pg_collection)
    """

    def __init__(self, model_config: MambaModelConfig):
        super().__init__(model_config)

    def build_model(
        self,
        pg_collection: ProcessGroupCollection,
        pre_process: bool | None = None,
        post_process: bool | None = None,
        vp_stage: int | None = None,
    ) -> MambaModel:
        """Build a single ``MCoreMambaModel`` stage.

        Args:
            pg_collection: Process groups for distributed training
            pre_process: Include embedding layer
            post_process: Include output layer
            vp_stage: Virtual pipeline stage

        Returns:
            The constructed model

        Note:
            Virtual pipeline model parallelism is not supported for Mamba models.
        """
        mamba_stack_spec = self._model_config.mamba_stack_spec
        if not isinstance(mamba_stack_spec, ModuleSpec):
            # Check if the function accepts config parameter
            import inspect

            if len(inspect.signature(mamba_stack_spec).parameters) > 0:
                mamba_stack_spec = mamba_stack_spec(self._model_config)
            else:
                mamba_stack_spec = mamba_stack_spec()

        assert (
            getattr(self._model_config.transformer, "virtual_pipeline_model_parallel_size", None) is None
            and vp_stage is None
        ), (
            "Virtual pipeline model parallelism is temporarily unsupported in SSM/Mamba "
            "models due to upstream MCore MambaModel API dependency"
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

        pre_process = pre_process if pre_process is not None else is_pp_first_stage(pg_collection.pp)
        post_process = post_process if post_process is not None else is_pp_last_stage(pg_collection.pp)
        return MambaModel(
            config=self._model_config.transformer,
            mamba_stack_spec=mamba_stack_spec,
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
    ) -> list[MambaModel]:
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
