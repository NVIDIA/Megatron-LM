# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

import inspect
import logging
from typing import Any, Callable, ClassVar, Literal, override

from megatron.core.models.gpt.heterogeneous.heterogeneous_layer_specs import get_gpt_heterogeneous_layer_spec
from megatron.core.transformer.heterogeneous.heterogeneous_config import HeterogeneousTransformerConfig
import torch
from megatron.core.distributed.distributed_data_parallel_config import DistributedDataParallelConfig
from megatron.core.enums import ModelType
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.pipeline_parallel.utils import (
    is_pp_first_stage,
    is_pp_last_stage,
    is_vp_first_stage,
    is_vp_last_stage,
)
from megatron.core.post_training.modelopt.gpt.model_specs import get_gpt_modelopt_spec
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.module import Float16Module, MegatronModule
from megatron.core.transformer.dot_product_attention import DotProductAttention as MCoreDotProductAttention
from megatron.core.transformer.enums import AttnBackend
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_transformer_block_with_experimental_attention_variant_spec,
)

from megatron.training.models.base import ModelConfig, ModelBuilder, compose_hooks
from megatron.training.vocab_utils import calculate_padded_vocab_size
from megatron.training.models.dist_utils import unimodal_build_distributed_models

from megatron.core.transformer.transformer_config import  TransformerConfig


logger = logging.getLogger(__name__)


from dataclasses import dataclass

from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_decoder_block_spec,
    get_gpt_decoder_layer_specs,
    get_gpt_layer_with_inference_spec,
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
)


def default_layer_spec(config: "GPTModelConfig", vp_stage: int) -> ModuleSpec:
    """Determine the most appropriate layer specification based on availability."""
    transformer_cfg = config.transformer
    use_te = transformer_cfg.transformer_impl == "transformer_engine"
    if config.restore_modelopt_state:
        ## Layer specification for quantization with ModelOpt. ##

        # arbitrary attention mask is used for speculative decoding training
        # When context parallel > 1, only causal mask type is supported
        from megatron.core import parallel_state

        use_arbitrary_attention_mask = (
            config.use_arbitrary_attention_mask
            if config.use_arbitrary_attention_mask is not None
            else parallel_state.get_context_parallel_world_size() == 1
        )
        return get_gpt_modelopt_spec(
            config=config.transformer,
            local_core_attention=False,
            remap_te_layernorm=True,
            real_quant_cfg="None",
            use_arbitrary_attention_mask=use_arbitrary_attention_mask,
        )
    elif transformer_cfg.experimental_attention_variant is not None:
        return get_transformer_block_with_experimental_attention_variant_spec(config=transformer_cfg, vp_stage=vp_stage)
    elif transformer_cfg.num_moe_experts is not None:
        return get_gpt_decoder_block_spec(
            transformer_cfg,
            use_transformer_engine=use_te,
            normalization=transformer_cfg.normalization,
            qk_l2_norm=transformer_cfg.qk_l2_norm,
            vp_stage=vp_stage,
        )
    elif isinstance(transformer_cfg, HeterogeneousTransformerConfig):
        return get_gpt_heterogeneous_layer_spec(transformer_cfg, use_te)
    elif use_te:
        if "use_te_op_fuser" in inspect.signature(get_gpt_layer_with_transformer_engine_spec).parameters:
            kwargs = {"use_te_op_fuser": config.use_transformer_engine_op_fuser}
        else:
            kwargs = {}
        return get_gpt_layer_with_transformer_engine_spec(
            config.transformer.num_moe_experts,
            config.transformer.moe_grouped_gemm,
            config.transformer.qk_layernorm,
            config.transformer.multi_latent_attention,
            config.transformer.experimental_attention_variant,
            qk_l2_norm=config.transformer.qk_l2_norm,
            use_kitchen=config.transformer.use_kitchen,
            use_te_activation_func=config.transformer.use_te_activation_func,
            use_kitchen_attention=config.transformer.use_kitchen_attention,
            kitchen_attention_backend=config.transformer.kitchen_attention_backend,
            mla_down_proj_fusion=getattr(config.transformer, "mla_down_proj_fusion", False),
            **kwargs,
        )
    elif transformer_cfg.transformer_impl == "inference_optimized":
        return get_gpt_layer_with_inference_spec(
            transformer_cfg.qk_layernorm,
            transformer_cfg.multi_latent_attention,
            qk_l2_norm=transformer_cfg.qk_l2_norm,
        )
    else:
        return get_gpt_layer_local_spec(
            transformer_cfg.num_moe_experts,
            transformer_cfg.moe_grouped_gemm,
            transformer_cfg.qk_layernorm,
            transformer_cfg.multi_latent_attention,
            transformer_cfg.experimental_attention_variant,
            normalization=transformer_cfg.normalization,
            use_kitchen=transformer_cfg.use_kitchen,
            use_kitchen_attention=transformer_cfg.use_kitchen_attention,
            kitchen_attention_backend=transformer_cfg.kitchen_attention_backend,
        )



@dataclass(kw_only=True)
class GPTModelConfig(ModelConfig):
    """Configuration for a Megatron Core GPT model.

    This is purely a configuration object. All model construction
    logic lives in ``GPTModelBuilder``.

    Contains a ``TransformerConfig`` alongside GPT-specific parameters. Attributes
    on the embedded ``transformer`` config are accessible directly on this object
    via ``__getattr__``/``__setattr__`` proxying.

    Note:
        ``vocab_size`` must be set before passing this config to ``GPTModelBuilder``.
    """

    builder: ClassVar[str] = "megatron.training.models.gpt.GPTModelBuilder"
    transformer: TransformerConfig
    transformer_layer_spec: ModuleSpec | Callable[["GPTModelConfig"], ModuleSpec] | None = None

    ### vocab padding related ###
    vocab_size: int | None = None
    """This represents the unpadded vocab size. The padded vocab size is
    automatically calculated in the GPTModelBuilder.
    """
    make_vocab_size_divisible_by: int = 128
    should_pad_vocab: bool = False
    """Set if the tokenizer provides the vocab size. In this case, the vocab size will be padded.
    Controls whether vocab size should be padded for tensor parallelism.
    """

    ### GPT Model initialization ###
    seq_length: int = 1024
    fp16_lm_cross_entropy: bool = False
    parallel_output: bool = True
    share_embeddings_and_output_weights: bool = False
    position_embedding_type: Literal["learned_absolute", "rope", "mrope", "yarn", "none"] = "learned_absolute"
    rotary_percent: float = 1.0
    rotary_base: int = 10000
    rope_scaling: bool = False
    rope_scaling_factor: float = 8.0
    scatter_embedding_sequence_parallel: bool = True
    seq_len_interpolation_factor: float | None = None

    tp_comm_overlap_cfg: str | dict[str, Any] | None = None
    """Config file when tp_comm_overlap is enabled."""

    ### settings for default layer spec options ###
    use_transformer_engine_op_fuser: bool = False
    use_arbitrary_attention_mask: bool | None = None

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

        if self.transformer.cuda_graph_impl != "none":
            assert self.transformer.use_te_rng_tracker, (
                "Transformer engine's RNG tracker is required for cudagraphs, it can be "
                "enabled with use_te_rng_tracker=True'."
            )

        vp_size = self.transformer.virtual_pipeline_model_parallel_size
        is_pipeline_asymmetric = (
            self.transformer.account_for_embedding_in_pipeline_split
            or self.transformer.account_for_loss_in_pipeline_split
        )
        is_pipeline_asymmetric |= (
            self.transformer.num_layers_in_first_pipeline_stage or self.transformer.num_layers_in_last_pipeline_stage
        ) is not None
        is_flexible_pp_layout = is_pipeline_asymmetric or (self.transformer.pipeline_model_parallel_layout is not None)
        if vp_size and not is_flexible_pp_layout:
            p_size = self.transformer.pipeline_model_parallel_size
            assert (self.transformer.num_layers // p_size) % vp_size == 0, (
                "Make sure the number of model chunks is the same across all pipeline stages."
            )


class GPTModelBuilder(ModelBuilder[GPTModel, GPTModelConfig]):
    """Builder to construct Megatron Core GPT models.

    Example:
        >>> transformer_cfg = TransformerConfig(num_layers=32, hidden_size=4096, ...)
        >>> model_cfg = GPTModelConfig(transformer=transformer_cfg, vocab_size=32000, seq_length=2048, ...)
        >>>
        >>> # Single stage (e.g. inference)
        >>> model = GPTModelBuilder(model_cfg).build_model(pg_collection)
        >>>
        >>> # Distributed training
        >>> models = GPTModelBuilder(model_cfg).build_distributed_models(pg_collection)
    """

    def __init__(self, model_config: GPTModelConfig):
        super().__init__(model_config)

    def build_model(
        self,
        pg_collection: ProcessGroupCollection,
        pre_process: bool | None = None,
        post_process: bool | None = None,
        vp_stage: int | None = None,
    ) -> GPTModel:
        """Build a single ``MCoreGPTModel`` stage.

        Args:
            pg_collection: Process groups for distributed training
            pre_process: Include embedding layer
            post_process: Include output layer
            vp_stage: Virtual pipeline stage

        Returns:
            The constructed model
        """
        transformer_layer_spec = self._model_config.transformer_layer_spec
        if transformer_layer_spec is None:
            transformer_layer_spec = default_layer_spec(self._model_config, vp_stage)
        elif not isinstance(transformer_layer_spec, ModuleSpec) and callable(transformer_layer_spec):
            # Check if the transformer_layer_spec function accepts vp_stage parameter
            if "vp_stage" in inspect.signature(transformer_layer_spec).parameters:
                transformer_layer_spec = transformer_layer_spec(self._model_config, vp_stage=vp_stage)
            else:
                transformer_layer_spec = transformer_layer_spec(self._model_config)

        assert self._model_config.vocab_size is not None, "vocab_size must be configured before calling build_model()"
        if self._model_config.should_pad_vocab:
            padded_vocab_size = calculate_padded_vocab_size(
                self._model_config.vocab_size,
                self._model_config.make_vocab_size_divisible_by,
                self._model_config.transformer.tensor_model_parallel_size,
            )
        else:
            padded_vocab_size = self._model_config.vocab_size

        mtp_spec = mtp_block_spec(self._model_config, transformer_layer_spec, vp_stage=vp_stage)

        # override spec with local backend if configured
        if self._model_config.attention_backend == AttnBackend.local:
            if hasattr(transformer_layer_spec, "submodules"):
                transformer_layer_spec.submodules.self_attention.submodules.core_attention = MCoreDotProductAttention

        # Determine pre/post flags if not provided using vp + pp stage
        vp_size = self._model_config.virtual_pipeline_model_parallel_size
        if pre_process is None:
            pre_process = is_vp_first_stage(vp_stage=vp_stage, vp_size=vp_size) and is_pp_first_stage(pg_collection.pp)
        if post_process is None:
            post_process = is_vp_last_stage(vp_stage=vp_stage, vp_size=vp_size) and is_pp_last_stage(pg_collection.pp)

        model = GPTModel(
            config=self._model_config.transformer,
            transformer_layer_spec=transformer_layer_spec,
            mtp_block_spec=mtp_spec,
            vocab_size=padded_vocab_size,
            max_sequence_length=self._model_config.seq_length,
            fp16_lm_cross_entropy=self._model_config.fp16_lm_cross_entropy,
            parallel_output=self._model_config.parallel_output,
            share_embeddings_and_output_weights=self._model_config.share_embeddings_and_output_weights,
            position_embedding_type=self._model_config.position_embedding_type,
            rotary_percent=self._model_config.rotary_percent,
            rotary_base=self._model_config.rotary_base,
            rope_scaling=self._model_config.rope_scaling,
            rope_scaling_factor=self._model_config.rope_scaling_factor,
            seq_len_interpolation_factor=self._model_config.seq_len_interpolation_factor,
            scatter_embedding_sequence_parallel=self._model_config.scatter_embedding_sequence_parallel,
            pre_process=pre_process,
            post_process=post_process,
            pg_collection=pg_collection,
            vp_stage=vp_stage,
        )

        return model

    def build_distributed_models(
        self,
        pg_collection: ProcessGroupCollection,
        ddp_config: DistributedDataParallelConfig | None = None,
        overlap_param_gather_with_optimizer_step: bool = False,
        use_megatron_fsdp: bool = False,
        use_torch_fsdp2: bool = False,
        wrap_with_ddp: bool = True,
        data_parallel_random_init: bool = True,
        mixed_precision_wrapper: Callable[[Any, MegatronModule], MegatronModule] | None = Float16Module,
        model_type: ModelType = ModelType.encoder_or_decoder,
    ) -> list[GPTModel]:
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


def mtp_block_spec(
    config: "GPTModelConfig", transformer_layer_spec: ModuleSpec, vp_stage: int | None = None
) -> ModuleSpec | None:
    """Create MTP block spec if model has MTP layers.

    Args:
        config: full model config

    Returns:
        ModuleSpec: The MTP module specification
    """
    transformer_cfg = config.transformer
    use_te = config.transformer.transformer_impl == "transformer_engine"

    if config.transformer.mtp_num_layers is not None:
        from megatron.core.models.gpt.gpt_layer_specs import get_gpt_mtp_block_spec

        if hasattr(transformer_layer_spec, "layer_specs") and len(transformer_layer_spec.layer_specs) == 0:
            # Get the decoder layer spec explicitly if no decoder layer in the last stage,
            # Only happens with block spec (TransformerBlockSubmodules) when using MoE.
            spec = default_layer_spec(config, vp_stage)
        else:
            decoder_specs = get_gpt_decoder_layer_specs(transformer_cfg, use_transformer_engine=use_te, normalization=transformer_cfg.normalization, qk_l2_norm=transformer_cfg.qk_l2_norm, vp_stage=vp_stage)
            spec = decoder_specs[-1]

        return get_gpt_mtp_block_spec(transformer_cfg, spec, use_transformer_engine=use_te, vp_stage=vp_stage)
    else:
        return None
