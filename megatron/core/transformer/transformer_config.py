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
        hidden_size (int): Transformer hidden size.
        ffn_hidden_size (int): Transformer Feed-Forward Network hidden size.
                                This is set to 4*hidden_size if not provided. Defaults to None.')
        num_attention_heads (int): Number of transformer attention heads.
        kv_channels (int): Projection weights dimension in multi-head attention.
                            This is set to hidden_size // num_attention_heads if not provided.
                            Defaults to None.
        
        attention_dropout (float): Post attention dropout probability. Defaults to 0.1.
        padded_vocab_size (int): Vocab size after padding.

        # model parallelism
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

    """

    # model architecture
    hidden_size: int
    num_attention_heads: int
    padded_vocab_size: int

    ffn_hidden_size: int = None
    kv_channels: int = None

    attention_dropout: float = 0.1

    # model parallelism
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
