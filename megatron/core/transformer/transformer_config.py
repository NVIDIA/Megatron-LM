# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import warnings
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union

import torch.nn.functional as F

from megatron.core.transformer.enums import AttnBackend

from ..model_parallel_config import ModelParallelConfig
from ..utils import get_te_version, init_method_normal, is_te_min_version, scaled_init_method_normal


@dataclass
class TransformerConfig(ModelParallelConfig):
    """Configuration object for megatron-core transformers.

    The initialization function has an argument for each parameter,
    including those in ModelParallelConfig.
    """

    ####################
    # model architecture
    ####################
    num_layers: int = 0
    """Number of transformer layers in a transformer block."""

    num_layers_in_first_pipeline_stage: Optional[int] = None
    """Number of transformer layers on first pipeline stage. 
    None implies equal layer division across PP ranks."""

    num_layers_in_last_pipeline_stage: Optional[int] = None
    """Number of transformer layers on last pipeline stage. 
    None implies equal layer division across PP ranks."""

    account_for_embedding_in_pipeline_split: bool = False
    """If set, the embedding layer will be treated as a standard transformer
    layer in the context of partition and placement for pipeline parallelism."""

    account_for_loss_in_pipeline_split: bool = False
    """If set, the loss layer will be treated as a standard transformer
    layer in the context of partition and placement for pipeline parallelism."""

    hidden_size: int = 0
    """Transformer hidden size."""

    num_attention_heads: int = 0
    """Number of transformer attention heads."""

    attention_backend: AttnBackend = AttnBackend.auto
    """Attention backend to run. By default we let transformer engine
    decide the best backend to run (except in the case of local).
    If attention backend is local we use the local pytorch implementation in mcore. 
    Users can specify exact backend by changing this config. """

    softmax_scale: Optional[float] = None
    """Softmax scale for attention scaling."""

    num_query_groups: Optional[int] = None
    """Number of query groups for group query attention. If None, normal attention is used."""

    ffn_hidden_size: Optional[int] = None
    """Transformer Feed-Forward Network hidden size. This is set to 4*hidden_size
    if not provided."""

    kv_channels: Optional[int] = None
    """Projection weights dimension in multi-head attention. This is set to hidden_size //
    num_attention_heads if not provided."""

    hidden_dropout: float = 0.1
    """Dropout probability for transformer hidden state."""

    attention_dropout: float = 0.1
    """Post attention dropout probability."""

    fp32_residual_connection: bool = False
    """If true, move residual connections to fp32."""

    # @jcasper should we keep this option?
    apply_residual_connection_post_layernorm: bool = False
    """If True, uses the original BERT residule connection ordering."""

    layernorm_epsilon: float = 1e-5
    """Epsilon value for any LayerNorm operations."""

    layernorm_zero_centered_gamma: bool = False
    """If set to True, the LayerNorm is adjusted to center the gamma values around 0. This improves
    numerical stability."""

    add_bias_linear: bool = True
    """Include a bias term in all linear layers (QKV projections, after core attention, and two in
    MLP layer)."""

    add_qkv_bias: bool = False
    """Add a bias term only for QKV projections."""

    gated_linear_unit: bool = False
    """Use a gated linear unit for the first linear layer in the MLP."""

    activation_func: Callable = F.gelu
    """Activation function to use for the non-linearity in the MLP."""

    activation_func_fp8_input_store: bool = False
    """Store the input of MLP activation function in FP8 for backprop to save memory.
    The stored input is casted back to the original precision before backprop compuatation."""

    num_moe_experts: Optional[int] = None
    """Number of experts to use for MoE layer. When set, it replaces MLP with MoE layer. Set to None
    for no MoE."""

    rotary_interleaved: bool = False
    """True is rotate pairs of even and odd dimensions (RoFormer style), False is rotate pairs of
    first half and second half (LLaMa style). Default to False."""

    window_size: Optional[Tuple[int, int]] = None
    """If not None, then will use sliding window attention. The size of the window is specified by
    the numbers inside the tuple; -1 is special value meaning "infinite window size"."""

    normalization: str = "LayerNorm"
    """Which norm to use for normalization layers, valid options are `LayerNorm` and `RMSNorm`."""

    qk_layernorm: bool = False
    """Whether to apply LayerNorm to the query and key embeddings."""

    test_mode: bool = False
    """Whether to run real-time tests."""

    calculate_per_token_loss: bool = False
    """Whether cross entropy loss is calculated over the actual number of non-padded tokens in the
    global batch, versus the default behavior of assuming all tokens are non-padded."""

    multi_latent_attention: bool = False
    """Whether to use multi-latent attention."""

    ####################
    # initialization
    ####################
    init_method: Optional[Callable] = None
    """Method to initialize weights. Note that bias is always set to zero. Should be a function that
    takes a single Tensor and initializes it. If None, will be set to
    megatron.core.utils.init_method_normal(init_method_std) which is torch nn init normal with
    mean=0.0 and std=init_method_std."""

    output_layer_init_method: Optional[Callable] = None
    """Method to initialize weights of the output layer of both attention and MLP blocks. If None,
    will be set to megatron.core.utils.scaled_init_method_normal(init_method_std) which is torch nn
    init normal with mean=0.0 and std=init_method_std / math.sqrt(2.0 * num_layers)."""

    init_method_std: float = 0.02
    """Standard deviation of the zero mean normal for the default initialization method, not used if
    init_method and output_layer_init_method are provided."""

    ####################
    # mixed-precision
    ####################
    apply_query_key_layer_scaling: bool = False
    """If true, scale Q * K^T by 1 / layer-number. This improve numeric stability when training with
    fp16."""

    attention_softmax_in_fp32: bool = True
    """If True, run attention masking and softmax in fp32. This should be True if
    apply_query_key_layer_scaling is True."""

    ####################
    # fusion
    ####################
    bias_activation_fusion: bool = False
    """If True, fuses bias addition and the activation function when possible."""

    masked_softmax_fusion: bool = False
    """If True, uses softmax fusion."""

    persist_layer_norm: bool = False
    """If True, uses the persistent fused layer norm kernel. This kernel only supports a fixed set
    of hidden sizes."""

    memory_efficient_layer_norm: bool = False
    """If True, and using local layers (not from TransformerEngine), tells Apex to use the memory
    efficient fused LayerNorm kernel. Ignored if not using LayerNorm."""

    bias_dropout_fusion: bool = False  # TODO: this should be bias_dropout_add_fusion?
    """If True, uses bias dropout fusion."""

    apply_rope_fusion: bool = False
    """If True, use fused RoPE kernel."""

    ####################
    # activation recomputation
    ####################
    recompute_granularity: Optional[str] = None
    """Determines which type of activation recompute to use.  Megatron-core supports 'selective'
    activation checkpointing where only the memory intensive part of attention is checkpointed.
    These memory intensive activations are also less compute intensive which makes activation
    checkpointing more efficient for LLMs (20B+).  See Reducing Activation Recomputation in Large
    Transformer Models (https://arxiv.org/abs/2205.05198) for more details.  'full' will checkpoint
    the entire transformer layer.  If None, no recompute is performed and all activations are saved.
    If set, must be 'selective' or 'full'. 'selective' always uses all layers.
    """

    recompute_method: Optional[str] = None
    """Determines which transformer layers will be recomputed. uniform will uniformly divide the
    total number of transformer layers in a transformer block and recompute the input activation of
    each divided chunk at the specified granularity.  block will recompute the input activations for
    only a set number of transformer layers per pipeline stage.  The rest of the layers in the
    pipeline stage will not have any activations recomputed.  If None, and recompute is enabled, all
    layers will do recomputation. If set, must be 'uniform' or 'block'."""

    recompute_num_layers: Optional[int] = None
    """When recompute_method is uniform, recompute_num_layers is the number of transformer layers in
    each uniformly divided recompute unit.  When recompute_method is block, recompute_num_layers is
    the number of transformer layers to recompute within each pipeline stage.  Must be None for
    'selective' activation checkpointing."""

    distribute_saved_activations: Optional[bool] = None
    """If True, distribute recomputed activations across the model parallel group."""

    ####################
    # fp8 related
    ####################
    fp8: Optional[str] = None
    """If set, enables the use of FP8 precision through Transformer Engine. There are 2 predefined
    choices (1) 'e4m3' uniformly uses e4m3 for all FP8 tensors, (2) 'hybrid' uses e4m3 for all FP8
    activation and weight tensors and e5m2 for all FP8 output activation gradient tensors."""

    fp8_margin: int = 0
    """Margin for the scaling factor computation."""

    fp8_interval: int = 1
    """DEPRECATED from TransformerEngine v1.8.0. This flag is ignored.
    Controls how often the scaling factor is recomputed.
    """

    fp8_amax_history_len: int = 1
    """The length of the amax history window used for scaling factor computation."""

    fp8_amax_compute_algo: str = "most_recent"
    """Algorithm used for choosing the `amax` value for the scaling factor computation. There are 2
    predefined choices: `max` chooses the largest `amax` in the history window, while `most_recent`
    always chooses the most recently seen value.

    """

    fp8_wgrad: bool = True
    """When set to False, override FP8 config options and do the wgrad computation
    in higher precision."""

    fp8_dot_product_attention: bool = False
    """When set to True, use the FP8 implementation of Dot Product Attention."""

    fp8_multi_head_attention: bool = False
    """When set to True, use the FP8 implementation of Multi Head Attention."""

    tp_only_amax_red: bool = False
    """When set to True, reduce the FP8 AMAX only in the TP or TP-CP domain"""

    ####################
    # MoE related
    ####################
    moe_shared_expert_intermediate_size: Optional[int] = None
    """Shared expert total ffn hidden size.
    It should be equal to 'num_shared_experts * ffn_size_of_each_shared_expert' if
    there are multiple shared experts.
    None means no shared expert."""

    moe_shared_expert_overlap: bool = False
    """Enable overlapping between shared expert computations and dispatcher communications.
    Without this, the shared epxerts execute after the routed experts."""

    moe_layer_freq: int = 1
    """Frequency between MoE layers and Dense layers. Accepts either:
    - An integer N: Represents a 1:N ratio, meaning one expert layer for every N-1 dense layers.
    - A string containing a Python list expression that defines a custom pattern, e.g.:
    "([1]*3+[0]*1)*3" evaluates to [1,1,1,0,1,1,1,0,1,1,1,0]
    where 1 indicates an expert layer and 0 indicates a dense layer."""

    moe_ffn_hidden_size: Optional[int] = None
    """MoE Feed-Forward Network hidden size"""

    moe_router_load_balancing_type: str = "aux_loss"
    """The load balancing strategy for the router. "aux_loss" corresponds to the load balancing loss 
    used in GShard and SwitchTransformer; "seq_aux_loss" corresponds to the loss used in DeepSeekV2, 
    which computes the loss for each individual sample; "sinkhorn" corresponds to the balancing 
    algorithm used in S-BASE, and "none" implies no load balancing. The default is "aux_loss"."""

    moe_router_topk: int = 2
    """Number of experts to route to for each token."""

    moe_router_topk_limited_devices: Optional[int] = None
    """Number of EP ranks to consider for each token in group-limited routing, 
    DEPRECATED and replaced by moe_router_num_groups and moe_router_group_topk.
    """

    moe_router_num_groups: Optional[int] = None
    """Number of groups to divide experts into for group-limited routing.
    When using group-limited routing:
    1. Experts are divided into 'moe_router_num_groups' equal-sized groups
    2. For each token, 'moe_router_group_topk' groups are selected based on routing scores
    (specifically, the sum of top-2 expert scores within each group)
    3. From these selected groups, 'moe_router_topk' individual experts are chosen
    Two common use cases:
    - Device-limited routing: Set 'moe_router_num_groups' equal to expert parallel size (EP)
    to limit each token to experts on a subset of devices
    (See DeepSeek-V2: https://arxiv.org/pdf/2405.04434)
    - Node-limited routing: Set 'moe_router_num_groups' equal to number of nodes in EP group
    to limit each token to experts on a subset of nodes
    (See DeepSeek-V3: https://arxiv.org/pdf/2412.19437)
    """

    moe_router_group_topk: Optional[int] = None
    """Number of selected groups for group-limited routing."""

    moe_router_pre_softmax: bool = False
    """Enable pre-softmax routing for MoE, which means softmax is before the top-k selection. 
    By default, softmax is done after top-k."""

    moe_router_topk_scaling_factor: Optional[float] = None
    """Scaling factor for routing score in top-k selection, only works when moe_router_pre_softmax 
    enabled. Defaults to None, which means no scaling."""

    moe_router_score_function: str = "softmax"
    """Score function for MoE routing. Can be "softmax" or "sigmoid"."""

    moe_router_enable_expert_bias: bool = False
    """TopK routing with dynamic per-expert bias in the aux-loss-free load balancing strategy.
    The routing decision is based on the sum of the routing scores and the expert bias.
    See https://arxiv.org/abs/2408.15664 for details."""

    moe_router_bias_update_rate: float = 1e-3
    """The expert bias is updated based on the number of assigned tokens to each expert 
    in a global batch, where the bias is increased for the experts with less assigned tokens
    and decreased for the experts with more assigned tokens. 
    The default value 1e-3 is same as that used in DeepSeekV3."""

    moe_grouped_gemm: bool = False
    """When there are multiple experts per rank, compress multiple local (potentially small) gemms
    in a single kernel launch to improve the utilization and performance by leveraging the Grouped
    GEMM feature introduced since CUTLASS 2.8 (https://github.com/fanshiqing/grouped_gemm).
    """

    moe_use_legacy_grouped_gemm: bool = False
    """Use legacy GroupedMLP rather than TEGroupedMLP.
    Note: The legacy one will be deprecated soon."""

    moe_aux_loss_coeff: float = 0  # 1e-2 would be a good start value for load balance loss.
    """Scaling coefficient for the aux loss. A starting value of 1e-2 is recommended."""

    moe_z_loss_coeff: Optional[float] = None  # 1e-3 would be a good start value for z-loss
    """Scaling coefficient for the z-loss. A starting value of 1e-3 is recommended."""

    moe_input_jitter_eps: Optional[float] = None
    """Add noise to the input tensor by applying jitter with a specified epsilon value."""

    moe_token_dropping: bool = False
    """This feature involves selectively dropping and padding tokens for each expert to achieve a
    specified capacity, similar to GShard, Switch-Transformer, and DeepSpeed-MoE. Note that this is
    currently unsupported so should remain False."""

    moe_token_dispatcher_type: str = "allgather"
    """The type of token dispatcher to use. The default is 'allgather'.
    Options are 'allgather' and 'alltoall'."""

    moe_per_layer_logging: bool = False
    """Enable per-layer logging for MoE, currently supports auxiliary loss and z loss."""

    moe_expert_capacity_factor: Optional[float] = None
    """moe_expert_capacity_factor (float): The capacity factor for each expert, None means no token
    will be dropped. The default is None."""

    moe_pad_expert_input_to_capacity: bool = False
    """moe_pad_expert_input_to_capacity (bool): If True, pads the input for each expert to match
    the expert capacity length, effective only after the moe_expert_capacity_factor is set. The
    default setting is False."""

    moe_token_drop_policy: str = 'probs'
    """The policy to drop tokens. Can be either "probs" or "position". If "probs", the tokens with
    the lowest probabilities will be dropped. If "position", tokens at the end of each batch will
    be dropped.
    """

    moe_layer_recompute: bool = False
    """Memory optimization: checkpointing moe_layer to save actiavtion memory."""

    moe_permute_fusion: bool = False
    """Fuse token rearrangement ops during token dispatching."""

    ##################
    # Context Parallel
    ##################
    cp_comm_type: Optional[Union[str, List[str]]] = None
    """Inter-gpu communication type for context parallelism.
    str: all layers share same communication type.
    List[str]: each layer has its separate communication type.
    cp_comm_type of each layer can be "p2p" or "all_gather" or "a2a" or "a2a+p2p".
    "p2p": Exchange KV chunks with P2P communications in ring topology. P2P is async and can be
    overlapped with attention compute.
    "all_gather": All-gather to get full sequence of KV before attention. The all-gather is not
    async, and cannot be overlapped.
    "a2a": Like DeepSpeed Ulysses, scatter attention heads across the CP group, and gather to get
    full sequence of QKV.
    "a2a+p2p": A hierarchical implementation of context parallelism to attention. 
    It uses A2A communications in low-level CP groups (e.g., via NVLink),
    and P2P communications in high-level CP groups (e.g., via IBLink).
    """

    ##################
    # Cuda Graphs
    ##################
    enable_cuda_graph: bool = False
    """When set to true, TransformerLayer layers are swapped with a CUDA graphed version."""

    cuda_graph_use_single_mempool: bool = False
    """When set to true, cudagraphs will be captured inside a single mempool, in which all 
    cudagraphs may only be used once per step. If false, cudagraphs may be reused across 
    microbatches. Enabling may reduce cudagraph memory overheads due to memory fragmentation, 
    however may greatly increase the number of cudagraphs created when the number of microbatches 
    is high."""

    cuda_graph_retain_backward_graph: bool = False
    """When set to true, cudagraph backward passes will be graph captured with 'retain_grad=True'
    This may enable cudagraphs for certain modules that are not completely cudagraph safe. For 
    more details, see: https://pytorch.org/docs/stable/generated/torch.Tensor.backward.html."""

    cuda_graph_warmup_steps: int = 3
    """Number of warmup steps for CUDA graphs"""

    external_cuda_graph: bool = False
    """When set to true, TransformerLayer layers are swapped with user provided CUDA graphs."""

    ####################
    # miscellaneous
    ####################
    clone_scatter_output_in_embedding: bool = True
    """When set to True, clone the output of scatter_to_sequence_parallel_region in embedding layer
    to facilitate garbage collection of input."""

    disable_parameter_transpose_cache: bool = False
    """When set to true, the parameter transposes are not cached for subsequent iterations."""

    config_logger_dir: str = ""
    """When non-empty, dumps entry-point configs to config_logger_dir"""

    flash_decode: bool = False
    """ Use the optimized flash decoding kernel during inference. """

    use_te_rng_tracker: bool = False
    """ Whether to use the TE or MCore version of the RNG tracker. """

    inference_rng_tracker: bool = False
    """ Whether we should instantiate a separate RNG tracker for inference. """

    def __post_init__(self):
        """Python dataclass method that is used to modify attributes after initialization.
        See https://docs.python.org/3/library/dataclasses.html#post-init-processing for more
        details.
        """
        super().__post_init__()
        if self.fp16 and self.bf16:
            raise ValueError(
                f'Only one of self.fp16: {self.fp16} and self.bf16 {self.bf16} should be True.'
            )

        if self.num_attention_heads % self.tensor_model_parallel_size != 0:
            raise ValueError(
                f"num_attention_heads ({self.num_attention_heads}) must be a multiple of "
                f"tensor_model_parallel_size ({self.tensor_model_parallel_size})."
            )

        if self.ffn_hidden_size is None:
            self.ffn_hidden_size = 4 * self.hidden_size

        if self.kv_channels is None:
            self.kv_channels = self.hidden_size // self.num_attention_heads

        if self.num_query_groups is None:
            self.num_query_groups = self.num_attention_heads

        if self.num_query_groups % self.tensor_model_parallel_size != 0:
            raise ValueError(
                f"num_query_groups ({self.num_query_groups}) must be a multiple of "
                f"tensor_model_parallel_size ({self.tensor_model_parallel_size})."
            )

        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True

        if self.expert_model_parallel_size > 1 and self.num_moe_experts is None:
            raise ValueError('num_moe_experts must be non None to use expert-parallel.')

        if self.num_moe_experts is not None and self.num_moe_experts <= 0:
            raise ValueError('num_moe_experts must be non-negative.')

        if self.moe_ffn_hidden_size is None:
            self.moe_ffn_hidden_size = self.ffn_hidden_size

        if self.moe_shared_expert_intermediate_size is not None:
            if self.moe_shared_expert_intermediate_size <= 0:
                raise ValueError(
                    f'moe_shared_expert_intermediate_size must be '
                    f'num_shared_experts * ffn_size_of_each_shared_expert, '
                    f'but got {self.moe_shared_expert_intermediate_size}'
                )
            if self.moe_shared_expert_overlap and self.moe_token_dispatcher_type not in [
                "alltoall"
            ]:
                raise ValueError(
                    f'moe_shared_expert_overlap only works with alltoall token dispatcher.'
                )

        if self.moe_expert_capacity_factor is not None:
            if self.moe_token_dispatcher_type not in ["alltoall", "alltoall_seq"]:
                raise ValueError(
                    'moe_expert_capacity_factor only works with alltoall token dispatcher'
                )
            if self.moe_expert_capacity_factor < 0:
                self.moe_expert_capacity_factor = None
            if self.moe_router_load_balancing_type not in ["aux_loss", "seq_aux_loss", "none"]:
                raise ValueError(
                    'moe_expert_capacity_factor only works with aux_loss or none load balancing'
                )

        if self.moe_pad_expert_input_to_capacity:
            if self.moe_expert_capacity_factor is None:
                raise ValueError(
                    'moe_expert_capacity_factor must be set to use moe_pad_expert_input_to_capacity'
                )

        if self.cpu_offloading and (
            self.cpu_offloading_num_layers < 0 or self.cpu_offloading_num_layers >= self.num_layers
        ):
            raise ValueError(
                f'CPU offloading can be done only for layers less than {self.num_layers}'
            )

        if self.cpu_offloading and self.pipeline_model_parallel_size > 1:
            raise ValueError(
                'Currently there is no support for Pipeline parallelism with CPU offloading'
            )

        if self.cpu_offloading and self.recompute_granularity is not None:
            raise ValueError(
                'CPU offloading does not work when activation recomputation is enabled'
            )

        if self.recompute_granularity is not None:
            if self.recompute_granularity not in ['full', 'selective']:
                raise ValueError(
                    f'When using recompute_granuarlity: {self.recompute_granularity} must be "full"'
                    'or "selective".'
                )

            if self.recompute_method is not None:
                if self.recompute_method not in ['block', 'uniform']:
                    raise ValueError(
                        f'recompute_method: {self.recompute_method} must be "block" or "uniform".'
                    )
            elif self.recompute_granularity != 'selective':
                raise ValueError(
                    f'Using recompute_granularity: {self.recompute_granularity} so '
                    'recompute_method must be "block" or "uniform"'
                )

            if self.recompute_granularity != 'selective' and self.recompute_num_layers is None:
                raise ValueError(
                    f'When using recompute_granularity: {self.recompute_granularity} '
                    'recompute_num_layers must be between '
                    '1 and num_layers_per_pipeline_rank: '
                    f'{self.num_layers // self.pipeline_model_parallel_size}'
                )
            elif (
                self.recompute_granularity == 'selective' and self.recompute_num_layers is not None
            ):
                raise ValueError(
                    f'When using recompute_granularity: {self.recompute_granularity} '
                    'recompute_num_layers must be None.'
                )

            if self.distribute_saved_activations and self.sequence_parallel:
                raise ValueError(
                    f'distribute_saved_activations: {self.distribute_saved_activations} must be '
                    f'false when sequence parallel is enabled: {self.sequence_parallel}'
                )

        if (
            self.num_layers_in_first_pipeline_stage is not None
            or self.num_layers_in_last_pipeline_stage is not None
        ) and (
            self.account_for_embedding_in_pipeline_split or self.account_for_loss_in_pipeline_split
        ):
            raise ValueError(
                'num_layers_in_first_pipeline_stage and num_layers_in_last_pipeline_stage cannot be'
                'set at the same time with account_for_embedding_in_pipeline_split'
                'and account_for_loss_in_pipeline_split'
            )

        if (
            self.num_layers_in_first_pipeline_stage is not None
            or self.num_layers_in_last_pipeline_stage is not None
        ):
            pipeline_parallel_size = self.pipeline_model_parallel_size
            num_layers = self.num_layers

            if self.num_layers_in_first_pipeline_stage is not None:
                if self.num_layers_in_first_pipeline_stage <= 0:
                    raise ValueError('num_layers_in_first_pipeline_stage must be larger than 0')

                if self.virtual_pipeline_model_parallel_size is not None:
                    if (
                        self.num_layers_in_first_pipeline_stage
                        % self.virtual_pipeline_model_parallel_size
                        != 0
                    ):
                        raise ValueError(
                            f'number of layers at first stage: '
                            f'{self.num_layers_in_first_pipeline_stage}'
                            f'must be divisible by virtual pipeline'
                            f'parallel degree {self.virtual_pipeline_model_parallel_size}'
                        )
                num_layers -= self.num_layers_in_first_pipeline_stage
                pipeline_parallel_size -= 1

            if self.num_layers_in_last_pipeline_stage is not None:
                if self.num_layers_in_last_pipeline_stage <= 0:
                    raise ValueError('num_layers_in_last_pipeline_stage must be larger than 0')

                if self.virtual_pipeline_model_parallel_size is not None:
                    if (
                        self.num_layers_in_last_pipeline_stage
                        % self.virtual_pipeline_model_parallel_size
                        != 0
                    ):
                        raise ValueError(
                            f'number of layers at last stage: '
                            f'{self.num_layers_in_last_pipeline_stage}'
                            f'must be divisible by virtual pipeline'
                            f'parallel degree {self.virtual_pipeline_model_parallel_size}'
                        )
                num_layers -= self.num_layers_in_last_pipeline_stage
                pipeline_parallel_size -= 1

            if not num_layers % pipeline_parallel_size == 0:
                raise ValueError(
                    f'number of layers at middle stage: {num_layers} must be divisible by'
                    f'the middle pipeline model parallel size {pipeline_parallel_size}'
                )

            if self.virtual_pipeline_model_parallel_size is not None:
                num_layers_per_middle_pipeline_rank = num_layers // pipeline_parallel_size
                if (
                    not num_layers_per_middle_pipeline_rank
                    % self.virtual_pipeline_model_parallel_size
                    == 0
                ):
                    raise ValueError(
                        f'number of layers on each middle pipeline rank:'
                        f'{num_layers_per_middle_pipeline_rank} must be divisible by virtual'
                        f'pipeline parallel degree {self.virtual_pipeline_model_parallel_size}'
                    )

        if self.account_for_embedding_in_pipeline_split or self.account_for_loss_in_pipeline_split:
            if self.virtual_pipeline_model_parallel_size is None:
                pipeline_parallel_size = self.pipeline_model_parallel_size

                if self.account_for_embedding_in_pipeline_split:
                    pipeline_parallel_size -= 1

                if self.account_for_loss_in_pipeline_split:
                    pipeline_parallel_size -= 1

                if not self.num_layers % pipeline_parallel_size == 0:
                    raise ValueError(
                        f'number of middle layers: {self.num_layers} must be divisible by '
                        f'middle pipeline_model_parallel_size {pipeline_parallel_size}'
                    )
            else:
                num_layers = self.num_layers
                if self.account_for_embedding_in_pipeline_split:
                    num_layers += 1

                if self.account_for_loss_in_pipeline_split:
                    num_layers += 1

                if not num_layers % self.pipeline_model_parallel_size == 0:
                    raise ValueError(
                        f'num_layers: {num_layers} after enable'
                        f'account_for_embedding_in_pipeline_split or '
                        f'account_for_loss_in_pipeline_split must be divisible'
                        f'by pipeline_model_parallel_size '
                        f'{self.pipeline_model_parallel_size}'
                    )

                num_layers_per_pipeline_rank = num_layers // self.pipeline_model_parallel_size
                if (
                    not num_layers_per_pipeline_rank % self.virtual_pipeline_model_parallel_size
                    == 0
                ):
                    raise ValueError(
                        f'number of layers on each pipeline rank: {num_layers_per_pipeline_rank}'
                        f'(after enable account_for_embedding_in_pipeline_split or '
                        f'account_for_loss_in_pipeline_split) must be divisible by'
                        f'virtual_pipeline_model_parallel_size'
                        f'{self.virtual_pipeline_model_parallel_size}'
                    )

        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True

        if self.bias_activation_fusion:
            if self.activation_func not in [F.gelu, F.silu]:
                raise ValueError(
                    "When bias_activation_fusion is True, activation function should be either "
                    "gelu or swiglu"
                )
            if (
                self.activation_func == F.gelu
                and not self.gated_linear_unit
                and not self.add_bias_linear
            ):
                raise ValueError(
                    "When bias_activation_fusion is True, gated_linear_unit is False, "
                    "and activation function is gelu, add_bias_linear must also be True."
                )

        if self.activation_func_fp8_input_store:
            if self.activation_func != F.silu or not self.gated_linear_unit:
                raise ValueError("Storing activation input in FP8 is supported only for SwiGLU.")

        if self.apply_rope_fusion:
            if self.rotary_interleaved:
                raise ValueError("rotary_interleaved does not work with apply_rope_fusion.")

            from megatron.core.models.common.embeddings.rope_utils import (
                fused_apply_rotary_pos_emb,
                fused_apply_rotary_pos_emb_thd,
            )

            if fused_apply_rotary_pos_emb is None and fused_apply_rotary_pos_emb_thd is None:
                raise ValueError(
                    "apply_rope_fusion is not available. Please install TE >= 1.4 or Apex."
                )

            if self.multi_latent_attention:
                raise ValueError("multi_latent_attention does not support apply_rope_fusion.")

        if self.multi_latent_attention and self.rotary_interleaved:
            raise ValueError("rotary_interleaved does not work with multi_latent_attention.")

        if self.init_method is None:
            self.init_method = init_method_normal(self.init_method_std)

        if self.output_layer_init_method is None:
            self.output_layer_init_method = scaled_init_method_normal(
                self.init_method_std, self.num_layers
            )

        if (
            self.moe_token_dispatcher_type == "alltoall_seq"
            and self.tensor_model_parallel_size != self.expert_tensor_parallel_size
        ):
            raise ValueError(
                "alltoall_seq dispatcher not support different TP size for MoE and Dense layer."
            )

        if self.moe_router_enable_expert_bias and self.moe_router_score_function != "sigmoid":
            raise ValueError(
                "Expert bias for aux-loss-free routing only supports sigmoid score function."
                "Please set --moe-router-score-function sigmoid for sigmoid score function."
            )

        if self.num_moe_experts and self.fp8:
            # TE version below 1.7.0 will raise Error when handle zeros tokens for expert
            if not is_te_min_version("1.7.0.dev0"):
                raise ValueError(
                    "Only transformer-engine>=1.7.0 supports MoE FP8 training, "
                    f"but your version is {get_te_version()}."
                )

            if self.moe_grouped_gemm and not is_te_min_version("1.11.0"):
                raise ValueError(
                    "Only transformer-engine>=1.11.0 supports FP8 grouped gemm, "
                    f"but your version is {get_te_version()}."
                )

        if (
            self.moe_router_topk == 1
            and self.moe_router_score_function == 'softmax'
            and not self.moe_router_pre_softmax
            and self.moe_router_load_balancing_type != 'sinkhorn'
        ):
            # Requires applying softmax before selecting the top-k when k is 1,
            # since softmax on a [num_tokens, 1] would yield a zero gradient.
            raise ValueError("Please use --moe-router-pre-softmax when topk is 1.")

        if self.moe_router_group_topk:
            if self.moe_router_topk_limited_devices:
                raise ValueError(
                    "moe_router_topk_limited_devices is deprecated and replaced by "
                    "moe_router_group_topk and moe_router_num_groups."
                )
            if not self.moe_router_num_groups:
                raise ValueError(
                    "When using group limited routing, moe_router_num_groups must be specified."
                )
            else:
                assert self.num_moe_experts % self.moe_router_num_groups == 0, (
                    f"num_moe_experts ({self.num_moe_experts}) should be divisible by "
                    f"moe_router_num_groups ({self.moe_router_num_groups})."
                )
                assert self.moe_router_group_topk <= self.moe_router_num_groups, (
                    f"moe_router_group_topk ({self.moe_router_group_topk}) should be smaller than "
                    f"moe_router_num_groups ({self.moe_router_num_groups})."
                )
        elif self.moe_router_topk_limited_devices:
            warnings.warn(
                "moe_router_topk_limited_devices is deprecated. Use moe_router_group_topk and "
                "moe_router_num_groups instead."
            )
            self.moe_router_group_topk = self.moe_router_topk_limited_devices
            self.moe_router_num_groups = self.expert_model_parallel_size

        if self.flash_decode and self.fp8:
            raise ValueError("FP8 inference is currently not support with flash decoding.")

        if self.enable_cuda_graph:
            if self.cpu_offloading:
                raise ValueError("CUDA graphs not supported with CPU offloading.")
            if self.recompute_granularity:
                raise ValueError("CUDA graphs not supported with activation recomputation.")

        if self.moe_token_dispatcher_type in ['allgather', 'alltoall_seq']:
            if self.variable_seq_lengths is True:
                raise ValueError(
                    f"Token dispatcher type: {self.moe_token_dispatcher_type} does not support "
                    f"variable sequence length, please use alltoall dispatcher instead."
                )

        if self.moe_permute_fusion:
            from megatron.core.transformer.moe.moe_utils import (
                fused_permute,
                fused_sort_chunks_by_index,
                fused_unpermute,
            )

            if (
                fused_permute is None
                or fused_sort_chunks_by_index is None
                or fused_unpermute is None
            ):
                raise ValueError("fused permutation is not available. Please install TE >= 2.1.0.")

        if self.cp_comm_type is not None:
            if isinstance(self.cp_comm_type, list):
                assert len(self.cp_comm_type) == self.num_layers, (
                    f"Length of cp_comm_type ({len(self.cp_comm_type)}) should equal to "
                    f"the total number of transformer layers ({self.num_layers})!"
                )
            else:
                assert isinstance(
                    self.cp_comm_type, str
                ), "Unsupported communication type for context parallelism!"

        assert (
            self.pipeline_model_parallel_size > 0
        ), f"Pipeline model parallel size must be larger than 0 \
            when enable --standalone-embedding-stage and --standalone-loss-stage"


@dataclass
class MLATransformerConfig(TransformerConfig):
    """Configuration object for megatron-core Multi-Latent Attention (MLA) transformers.

    The initialization function has an argument for each parameter, including those in
    ModelParallelConfig. Included YaRN RoPE parameters that is fused in MLA.
    """

    multi_latent_attention: bool = True
    """Whether to use Multi-Latent Attention."""

    q_lora_rank: int = 512
    """Rank of Query tensor's low rank representation."""

    kv_lora_rank: int = 512
    """Rank of Key and Value tensors' low rank representation."""

    qk_head_dim: int = 128
    """Dimension of the head in the QK projection. q_head_dim = qk_head_dim + qk_pos_emb_head_dim"""

    qk_pos_emb_head_dim: int = 64
    """Dimension of the position embedding in the QK projection."""

    v_head_dim: int = 128
    """Dimension of the head in the V projection."""

    rotary_base: float = 10000
    """Rotary base for the rotary embeddings."""

    rotary_scaling_factor: float = 40
    """Rotary scaling factor for the rotary embeddings."""

    normalization: str = "RMSNorm"
    """Default normalization layer for MLA models is RMSNorm."""

    max_position_embeddings: int = 163840
    """Maximum position embeddings for the original model."""

    beta_fast: float = 32
    """Beta fast for YaRN RoPE."""

    beta_slow: float = 1
    """Beta slow for YaRN RoPE."""

    mscale: float = 0.707
    """Mscale for YaRN RoPE in Multi-Latent Attention."""

    mscale_all_dim: float = 0.707
    """Mscale all dimensions for YaRN RoPE in Multi-Latent Attention."""
