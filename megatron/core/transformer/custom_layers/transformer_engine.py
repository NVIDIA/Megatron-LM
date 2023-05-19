import torch
import transformer_engine as te

from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.parallel_state import get_tensor_model_parallel_group
from megatron.core.tensor_parallel import get_cuda_rng_tracker

class TELayerNorm(te.pytorch.module.LayerNorm):
    """
    Wrapper for the Transformer-Engine's `LayerNorm`.
    """
    def __init__(self,
                 hidden_size: int,
                 eps: float = 1e-5,
                 sequence_parallel: bool = False,
                 **kwargs):
        super().__init__(
            hidden_size=hidden_size,
            eps=eps,
            sequence_parallel=sequence_parallel
        )

class TELinear(te.pytorch.module.Linear):
    """
    Wrapper for the Transformer-Engine's `Linear` layer but specialized similar
    to megatron's `RowParallelLinear` layer.
    """
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 config: TransformerConfig,
                 parallel_mode: str,
                 **kwargs):
        self.config = config
        super().__init__(
            in_features=input_size,
            out_features=output_size,
            sequence_parallel=self.config.sequence_parallel,
            fuse_wgrad_accumulation=self.config.gradient_accumulation_fusion,
            tp_group=get_tensor_model_parallel_group(),
            tp_size=self.config.tensor_model_parallel_size,
            get_rng_state_tracker=get_cuda_rng_tracker,
            init_method=self.config.init_method,
            params_dtype=self.config.params_dtype,
            parallel_mode=parallel_mode,
            **kwargs
        )

class TEColumnParallelLinear(TELinear):
    """
    Wrapper for the Transformer-Engine's `Linear` layer but specialized similar
    to megatron's `ColumnParallelLinear` layer.
    """
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 config: TransformerConfig,
                 **kwargs):
        self.config = config
        super().__init__(
            input_size=input_size,
            output_size=output_size,
            config=self.config,
            parallel_mode="column",
            **kwargs
        )

class TERowParallelLinear(TELinear):
    """
    Wrapper for the Transformer-Engine's `Linear` layer but specialized similar
    to megatron's `RowParallelLinear` layer.
    """
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 config: TransformerConfig,
                 **kwargs):
        self.config = config
        super().__init__(
            input_size=input_size,
            output_size=output_size,
            config=self.config,
            parallel_mode="row",
            **kwargs
        )

class TECoreAttention(te.pytorch.transformer.DotProductAttention):
    """
    Wrapper for the Transformer-Engine's `DotProductAttention` layer that also
    has "flash attention" enabled.
    """
    def __init__(self,
                 config: TransformerConfig,
                 layer_number: int = 1,
                 attn_mask_type: AttnMaskType = AttnMaskType.padding,
                 **kwargs):
        self.config = config
        super().__init__(
            num_attention_heads=self.config.num_attention_heads,
            kv_channels=self.config.kv_channels,
            attention_dropout=self.config.attention_dropout,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type.name,
            sequence_parallel=self.config.sequence_parallel,
            tp_size=self.config.tensor_model_parallel_size,
            get_rng_state_tracker=get_cuda_rng_tracker,
            tp_group=get_tensor_model_parallel_group(),
            **kwargs
        )
