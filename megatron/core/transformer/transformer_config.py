# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.init as init
from torch import Tensor


@dataclass
class TransformerConfig:
    """ Configuration object for megatron-core transformers.

        Attributes:

        # model architecture
        num_layers (int): Number of transformer layers in a transformer block.
        hidden_size (int): Transformer hidden size.
        ffn_hidden_size (int): Transformer Feed-Forward Network hidden size.
                                This is set to 4*hidden_size if not provided. Defaults to None.')
        num_attention_heads (int): Number of transformer attention heads.
        kv_channels (int): Projection weights dimension in multi-head attention.
                            This is set to hidden_size // num_attention_heads if not provided.
                            Defaults to None.
        hidden_dropout (float): Dropout probability for transformer hidden state. Defaults to 0.1.
        attention_dropout (float): Post attention dropout probability. Defaults to 0.1.
        fp32_residual_connection (bool): If true, move residual connections to fp32.
        apply_residual_connection_post_layernorm (bool): If true, uses the original BERT residule connection ordering.
                                                         Defaults to False.
        layernorm-epsilon (float): Layernorm epsilon. Defaults to 1e-5.


        # model parallelism
        tensor_model_parallel_size (int): Intra-layer model parallelism. Splits tensors across GPU ranks. Defaults to 1.
        pipeline_model_parallel_size (int): Inter-layer model parallelism. Splits transformer layers across GPU ranks. Defaults to 1.
        virtual_pipeline_model_parallel_size (int): Interleaved pipeline parallelism is used to improve performance by reducing the pipeline bubble.
                                           Considers a transformer block as a list of smaller transformer (virtual) blocks.
                                           The number of virtual blocks per pipeline model parallel rank is the virtual model parallel size.
                                           See Efficient Large-Scale Language Model Training on GPU Clusters
                                           Using Megatron-LM: https://arxiv.org/pdf/2104.04473.pdf for more details.
                                           Defaults to None.
        sequence_parallel_enabled (bool): Makes tensor parallelism more memory efficient for LLMs (20B+) by 
                                          parallelizing layer norms and dropout sequentially.
                                          See Reducing Activation Recomputation in Large Transformer Models: https://arxiv.org/abs/2205.05198 for more details. 
                                          Defaults to False.
        # weight initialization
        init_method (Any): Method to initialize weights. Note that bias is always set to zero.
                            Defaults to init.xavier_normal_
        init_method_std: (float): Standard deviation of the zero mean normal. Defaults to 0.02.
        use_cpu_initialization (bool): When set to False, we initialize the weights directly on the GPU.
                                        Transferring weights from CPU to GPU can take a significant amount
                                        of time for large models. Defaults to False.
        perform_initialization (bool): If true, weights are initialized. Defaults to True.
        params_dtype: (torch.dtype): dtype used when intializing the weights. Defaults to torch.float32

        # mixed-precision
        fp16 (bool): If true, train with O2 fp16 mixed precision training. Defaults to False.
        bf16 (bool): If true, train with O2 bf16 mixed precision training. Defaults to False.
        apply_query_key_layer_scaling (bool): If true, scale Q * K^T by 1 / layer-number. Defaults to True.
        attention_softmax_in_fp32 (bool): If true, run attention masking and softmax in fp32.
                                          This should be true if apply_query_key_layer_scaling is true.

        # communication
        async_tensor_model_parallel_allreduce (bool): If true, enables asynchronous execution of
                                                        tensor-model-parallel all-reduce with weight
                                                        gradient compuation of a column-linear layer.
                                                        Defaults to True.

        # fusion
        gradient_accumulation_fusion (bool): If true, fuses weight gradient accumulation to GEMMs. Defaults to False.
        bias_gelu_fustion (bool): If true, fuses bias and gelu. Defaults to False.
        masked_softmax_fusion (bool): If true, uses softmax fusion.
        persist_layer_norm (bool): If true, uses the persistent fused layer norm kernel.
                                   This kernel only supports a fixed set of hidden sizes.
                                   Defaults to False.
        bias_dropout_fusion (bool): If true, uses bias dropout fusion.

        # activation recomputation
        recompute_granularity (str): megatron-core supports 'selective' activation checkpointing where only the memory intensive part of attention is checkpointed.
                                     These memory intensive activations are also less compute intensive which makes activation checkpointing more efficient for LLMs (20B+).
                                     See Reducing Activation Recomputation in Large Transformer Models: https://arxiv.org/abs/2205.05198 for more details.
                                     'full' will checkpoint the entire transformer layer.
                                     Must be 'selective' or 'full'. Defaults to None.
        recompute_method (str): uniform will uniformly divide the total number of transformer layers in a transformer block and recompute the input activation of
                                each divided chunk at the specified granularity.
                                block will recompute the input activations for only a set number of transformer layers per pipeline stage.
                                The rest of the layers in the pipeline stage will not have any activations recomputed.
                                Must be 'uniform' or 'block'. Defaults to None.
        recompute_num_layers (int): When recompute_method is uniform, recompute_num_layers is the number of transformer layers in each uniformly divided
                                    recompute unit.
                                    When recompute_method is block, recompute_num_layers is the number of transformer layers to recompute within each pipeline stage.
                                    Defaults to None.
        distribute_saved_activations (bool): If true, distribute recomputed activations across the model parallel group. Defaults to None.
                            

    """

    # model architecture
    num_layers: int
    hidden_size: int
    num_attention_heads: int

    ffn_hidden_size: int = None
    kv_channels: int = None
    hidden_dropout: float = 0.1
    attention_dropout: float = 0.1
    fp32_residual_connection: bool = False
    # @jcasper should we keep this option?
    apply_residual_connection_post_layernorm: bool = False
    layernorm_epsilon: float = 1e-5

    # model parallelism
    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    virtual_pipeline_model_parallel_size: int = None
    sequence_parallel_enabled: bool = False

    # weight initialization
    init_method: Callable = init.xavier_normal_
    init_method_std: float = 0.02
    output_layer_init_method: Callable = init.xavier_normal_
    use_cpu_initialization: bool = False
    perform_initialization: bool = True
    params_dtype: torch.dtype = torch.float32

    # mixed-precision
    fp16: bool = False
    bf16: bool = False
    apply_query_key_layer_scaling: bool = True
    attention_softmax_in_fp32: bool = True

    # communication
    async_tensor_model_parallel_allreduce: bool = True

    # fusion
    gradient_accumulation_fusion: bool = False
    bias_gelu_fusion: bool = False
    masked_softmax_fusion: bool = False
    persist_layer_norm: bool = False
    bias_dropout_fusion: bool = False

    # activation recomputation
    recompute_granularity: str = None
    recompute_method: str = None
    recompute_num_layers: int = None
    distribute_saved_activations: bool = None

    def __post_init__(self):
        """ Python dataclass method that is used to modify attributes after initialization.
            See https://docs.python.org/3/library/dataclasses.html#post-init-processing for more details.
        """
        if self.fp16 and self.bf16:
            raise ValueError(f'Only one of self.fp16: {self.fp16} and self.bf16 {self.bf16} should be True.')

        if self.ffn_hidden_size is None:
            self.ffn_hidden_size = 4 * self.hidden_size

        if self.kv_channels is None:
            self.kv_channels = self.hidden_size // self.num_attention_heads

        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True

        if self.recompute_granularity is not None:
            if not self.recompute_granularity in ['full', 'selective']:
                raise ValueError(
                    f'When using recompute_granuarlity: {self.recompute_granularity} must be "full" or "selective".'
                )

            if self.recompute_method is not None:
                if not self.recompute_method in ['block', 'uniform']:
                    raise ValueError(f'recompute_method: {self.recompute_method} must be "block" or "uniform".')
            else:
                raise ValueError(
                    f'Using recompute_granularity: {self.recompute_granularity} so recompute_method must be "block" or "uniform"'
                )

            if self.recompute_num_layers is None:
                raise ValueError(
                    f'When using recompute_granularity: {self.recompute_granularity} so recompute_num_layers must be between '
                    f'1 and num_layers_per_pipeline_rank: {self.num_layers // self.pipeline_model_parallel_size}'
                )

            if self.distribute_saved_activations and self.sequence_parallel_enabled:
                raise ValueError(
                    f'distribute_saved_activations: {self.distribute_saved_activations} must be false when sequence parallel is enabled: {self.sequence_parallel_enabled}'
                )

            if self.virtual_pipeline_model_parallel_size is not None:
                if not self.num_layers % self.virtual_pipeline_model_parallel_size == 0:
                    raise ValueError(
                        f'num_layers: {self.num_layers} must be divisible by virtual_model_parallel_size {self.virtual_pipeline_model_parallel_size}'
                    )

