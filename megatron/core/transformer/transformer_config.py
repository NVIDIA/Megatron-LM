# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import warnings
from dataclasses import dataclass
from typing import Callable, List, Literal, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from megatron.core.enums import Fp8Recipe
from megatron.core.quantization.quant_config import RecipeConfig
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.pipeline_parallel_layer_layout import PipelineParallelLayerLayout

from ..fusions.fused_bias_geglu import quick_gelu
from ..model_parallel_config import ModelParallelConfig
from ..utils import (
    get_te_version,
    init_method_normal,
    is_te_min_version,
    is_torch_min_version,
    scaled_init_method_normal,
)

try:
    from packaging.version import Version as PkgVersion

    HAVE_PACKAGING = True
except ImportError:
    HAVE_PACKAGING = False


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

    mtp_num_layers: Optional[int] = None
    """Number of Multi-Token Prediction (MTP) Layers."""

    mtp_loss_scaling_factor: Optional[float] = None
    """Weighting factor of Multi-Token Prediction (MTP) loss."""

    num_layers_in_first_pipeline_stage: Optional[int] = None
    """Number of transformer layers on first pipeline stage.
    None implies equal layer division across PP ranks."""

    num_layers_in_last_pipeline_stage: Optional[int] = None
    """Number of transformer layers on last pipeline stage.
    None implies equal layer division across PP ranks."""

    pipeline_model_parallel_layout: Optional[Union[str, list, PipelineParallelLayerLayout]] = None
    """Custom definition of the pipeline parallel partitioning.
    Support type:
    - str: e.g., 'Et*3|(tt|)*29,m|L'. Stages are split by '|', replicated stages or layers
    can be described with multiplication. Commas can be used cosmetically.
    - list: e.g., [['embedding', 'decoder'], ['decoder', 'decoder', 'decoder', 'loss']].
    - PipelineParallelLayerLayout: a PipelineParallelLayerLayout object.
    If given either a string or a list, it will be transferred into a PipelineParallelLayerLayout
    in post init. Let i = a * pp_size + b, then layout[i] gives a list of the layers 
    in the a-th vpp stage and the b-th pp stage, i.e., vpp(0)pp(0), vpp(0)pp(1), ..., 
    vpp(i)pp(j), vpp(i)pp(j+1), ..., vpp(-1)pp(-2), vpp(-1)pp(-1).
    In the inner lists of layers, 'embedding' or 'E' denotes the embedding layer, 'loss' or 'L'
    denotes the loss function, and 'decoder' or 't' denotes the transformer decoder layer.
    Examples:
        [['embedding', 'decoder'], ['decoder', 'decoder', 'decoder', 'loss']]:
        pp = 2, vpp = None
        pp rank 0 holds: embedding, decoder
        pp rank 1 holds: decoder*3, loss
        'E|(tt|)*2,(t|)*4,mL':
        pp = 2, vpp = 4
        vpp rank 0 pp rank 0 holds: embedding
        vpp rank 0 pp rank 1~2 holds: decoder*2
        vpp rank 0 pp rank 3 holds: decoder
        vpp rank 1 pp rank 0~2 holds: decoder
        vpp rank 1 pp rank 3 holds: mtp, loss"""

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

    softmax_type: Literal['vanilla', 'off-by-one', 'learnable'] = 'vanilla'
    """Applies modified softmax from https://www.evanmiller.org/attention-is-off-by-one.html. 
       Supports both TE FusedAttention and local unfused attention. Supports both a fixed offset and 
       and learnable offset."""

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

    glu_linear_offset: float = 0.0
    """Offset term in the GLU activation function: activation_func(x[0]) * (x[1] + offset). Only 
    used when gated_linear_unit is True"""

    activation_func_clamp_value: Optional[float] = None
    """Clamp the output of the linear_fc1 in the activation function. Only used when activation_func
    is quick_gelu."""

    num_moe_experts: Optional[int] = None
    """Number of experts to use for MoE layer. When set, it replaces MLP with MoE layer. Set to None
    for no MoE."""

    rotary_interleaved: bool = False
    """True is rotate pairs of even and odd dimensions (RoFormer style), False is rotate pairs of
    first half and second half (LLaMa style). Default to False."""

    window_size: Optional[Tuple[int, int]] = None
    """If not None, then will use sliding window attention. The size of the window is specified by
    the numbers inside the tuple; -1 is special value meaning "infinite window size"."""

    window_attn_skip_freq: Optional[Union[int, List[int]]] = None
    """Frequency of full attention layers among sliding window attention layers. Accepts either:
    - An integer N: Represents a (N-1):1 ratio, one full attention layer after (N-1) SWA layers.
    - A list that defines a custom pattern, e.g.: [1,1,1,1,0,0,0,0], where 1 represents SWA. """

    normalization: str = "LayerNorm"
    """Which norm to use for normalization layers, valid options are `LayerNorm` and `RMSNorm`."""

    qk_layernorm: bool = False
    """Whether to apply `normalization` type of normalization to the query and key embeddings."""

    test_mode: bool = False
    """Whether to run real-time tests."""

    calculate_per_token_loss: bool = False
    """Whether cross entropy loss is calculated over the actual number of non-padded tokens in the
    global batch, versus the default behavior of assuming all tokens are non-padded."""

    multi_latent_attention: bool = False
    """Whether to use multi-latent attention."""

    no_rope_freq: Optional[Union[int, List[int]]] = None
    """Controls which layers perform Rotary Position Embedding (RoPE). Accepts either:
    An integer N: Creates a pattern where RoPE is skipped every N-1 layers. For example,
    no_rope=4 means RoPE is applied for 3 layers, then skipped for 1 layer, repeating this pattern.
    A list of integers: Defines a custom pattern where 1 means skip RoPE and 0 means apply RoPE.
    For example, [0,1,1,0] means: apply RoPE, skip RoPE, skip RoPE, apply RoPE."""

    moe_deepep_num_sms: int = 20
    """Number of SMs to use for DeepEP."""

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

    embedding_init_method: Optional[Callable] = None
    """
    Method to initialize weights of the embedding layer. If None, will be set as described 
    in init_method above.
    """

    embedding_init_method_std: Optional[float] = None
    """
    Standard deviation of the zero mean normal for the default initialization method for the 
    embedding layer. If None, will be set to init_method_std.
    """

    init_model_with_meta_device: bool = False
    """
    If True, initializes the model with the meta device. This is helpful for
    training of very large models. This feature is only works when megatron fsdp is turned on.
    """

    ####################
    # mixed-precision
    ####################
    apply_query_key_layer_scaling: bool = False
    """If true, scale Q * K^T by 1 / layer-number. This improve numeric stability when training with
    fp16."""

    attention_softmax_in_fp32: bool = True
    """If True, run attention masking and softmax in fp32. This should be True if
    apply_query_key_layer_scaling is True."""

    disable_bf16_reduced_precision_matmul: bool = False
    """If True, sets torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False to
    prevent matmul from using reduced precision accumulation when using BF16."""

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

    use_fused_weighted_squared_relu: bool = False
    """If True, uses fused weighted squared relu kernel when using MoE."""

    fused_single_qkv_rope: bool = False
    """If set, avoid splitting QKV before ROPE forward and avoid concatenating ROPE dgrads."""

    ####################
    # activation recomputation
    ####################
    recompute_granularity: Optional[str] = None
    """Determines which type of activation recompute to use.  Megatron-core supports 'selective'
    activation checkpointing where the submodules set in --recompute-modules is checkpointed.
    The default is "core_attn" which is the memory intensive part of attention.
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

    recompute_modules: Optional[List[str]] = None
    """The submodules to recompute.
    choices: "core_attn", "moe_act", "layernorm", "mla_up_proj", "mlp", "moe", "shared_experts".
    default: ["core_attn"].
    "core_attn": recompute the core attention part of the transformer layer.
    "moe_act": recompute the MoE MLP activation function.
    "layernorm": recompute the input_layernorm and pre_mlp_layernorm.
    "mla_up_proj": recompute the MLA up projection and RoPE applying parts.
    "mlp": recompute the dense MLP submodule.
    "moe": recompute the MoE layer.
    "shared_experts": recompute the shared experts in the MoE layer.
    "moe_act", "layernorm", and "mla_up_proj" use output-discarding checkpointing,
    "core_attn", "mlp", "moe", and "shared_experts" use normal checkpointing.
    """

    ####################
    # fp8 related
    ####################
    fp8: Optional[str] = None
    """If set, enables the use of FP8 precision through Transformer Engine. There are 2 predefined
    choices (1) 'e4m3' uniformly uses e4m3 for all FP8 tensors, (2) 'hybrid' uses e4m3 for all FP8
    activation and weight tensors and e5m2 for all FP8 output activation gradient tensors."""

    fp8_recipe: Optional[str] = "delayed"
    """If set, enables the use of FP8 precision through Transformer Engine. There are 3 predefined
    choices (1) 'tensorwise' uses per tensor current scaling recipe, (2) 'delayed'
    uses delayed scaling recipe, 3) 'mxfp8' for Blackwell architecture only,
    4) 'blockwise' for blockwise scaling recipe."""

    fp8_param: bool = False
    """If set, keep the parameters in fp8 precision to save memory. This option must be used
    together with fp8 mode (i.e., TransformerConfig.fp8 is not None). Note that not all parameters
    will be converted to fp8; for example, biases will remain unchanged. The parameters affected are
    primarily the weights of GEMMs. The specific parameters that will be converted to fp8 are
    determined by TE."""

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

    first_last_layers_bf16: bool = False
    """If True, retains first and last N TransformerBlocks in BF16 as opposed to FP8."""

    num_layers_at_start_in_bf16: int = 1
    """Number of layers at the start of the model to keep in BF16 precision when
    first_last_layers_bf16 is True."""

    num_layers_at_end_in_bf16: int = 1
    """Number of layers at the end of the model to keep in BF16 precision when
    first_last_layers_bf16 is True."""

    use_kitchen: bool = False
    """Use the kitchen extension for transformer quantization."""

    ####################
    # fp4 related
    ####################
    fp4: Optional[str] = None
    """If set, enables the use of FP4 precision through Transformer Engine. Currently only 
    supports 'nvfp4' which uses NVFP4BlockScaling recipe (requires TE >= 2.7.0.dev0)."""

    fp4_recipe: Optional[str] = "nvfp4"
    """If set, enables the use of FP4 precision through Transformer Engine. Currently only
    'nvfp4' is supported which uses NVFP4BlockScaling recipe for Blackwell+ architecture."""

    fp4_param: bool = False
    """If set, keep the parameters in fp4 precision to save memory. This option must be used
    together with fp4 mode (i.e., TransformerConfig.fp4 is not None). Note that not all parameters
    will be converted to fp4; for example, biases will remain unchanged."""

    ####################
    # MoE related
    ####################
    moe_shared_expert_intermediate_size: Optional[int] = None
    """Shared expert total ffn hidden size.
    It should be equal to 'num_shared_experts * ffn_size_of_each_shared_expert' if
    there are multiple shared experts.
    None means no shared expert.
    By default, the shared experts execute before the router. However, when
    moe_shared_expert_overlap or overlap_moe_expert_parallel_comm is set,
    the shared experts execute after the router, before the routed experts.
    This makes the gradients from the router and the shared experts added in
    different orders to the hidden_states, causing minor numerical differences
    in the hidden_states gradient."""

    moe_shared_expert_overlap: bool = False
    """Enable overlapping between shared expert computations and dispatcher communications.
    Without this, the shared experts execute before the router."""

    moe_layer_freq: Union[int, List[int]] = 1
    """Frequency between MoE layers and Dense layers. Accepts either:
    - An integer N: Represents a 1:N ratio, meaning one expert layer for every N-1 dense layers.
    - A list that defines a custom pattern, e.g.: [1,1,1,0,1,1,1,0,1,1,1,0]"""

    moe_ffn_hidden_size: Optional[int] = None
    """MoE Feed-Forward Network hidden size"""

    moe_router_load_balancing_type: Union[str, List[str]] = "aux_loss"
    """The load balancing strategy for the router.
    Options:
    - "aux_loss": Load balancing loss used in GShard and SwitchTransformer, calculated at
    micro-batch level.
    - "seq_aux_loss": Load balancing loss used in DeepSeekV2 and DeepSeekV3, computes loss
    for each individual sample.
    - "global_aux_loss": Load balancing loss calculated at global batch level.
    - "sinkhorn": Balancing algorithm used in S-BASE.
    - "none": No load balancing.
    A list of strings can be provided to combine multiple aux-loss load balancing types.
    The default is "aux_loss".
    """

    moe_router_topk: int = 2
    """Number of experts to route to for each token."""

    moe_router_topk_limited_devices: Optional[int] = None
    """Number of EP ranks to consider for each token in group-limited routing,
    DEPRECATED and replaced by moe_router_num_groups and moe_router_group_topk.
    """

    moe_router_padding_for_fp8: Optional[bool] = False
    """Whether to pad the routing_map to make sure the number of tokens each expert received
    is a multiple of 16/32 for FP8 precision. This can remove the explicit padding in the
    GroupedMLP layer."""

    moe_router_num_groups: Optional[int] = None
    """Number of groups to divide experts into for group-limited routing.
    When using group-limited routing:
    1. Experts are divided into 'moe_router_num_groups' equal-sized groups
    2. For each token, 'moe_router_group_topk' groups are selected based on sum of
    top-('moe_router_topk'/'moe_router_group_topk') routing scores within each group
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
    """Enable pre-softmax(pre-sigmoid) routing for MoE, which means softmax is before the 
    top-k selection.
    By default, softmax is done after top-k."""

    moe_router_topk_scaling_factor: Optional[float] = None
    """Scaling factor for routing score in top-k selection, only works when moe_router_pre_softmax
    enabled. Defaults to None, which means no scaling."""

    moe_router_score_function: str = "softmax"
    """Score function for MoE routing. Can be "softmax" or "sigmoid"."""

    moe_router_dtype: Optional[str] = None
    """Data type for routing and expert output weighted averaging. Using fp32 or fp64 can
    improve stability especially when the number of experts is large (e.g. finegrained-moe).
    None means no changes for dtype."""

    moe_router_enable_expert_bias: bool = False
    """TopK routing with dynamic per-expert bias in the aux-loss-free load balancing strategy.
    The routing decision is based on the sum of the routing scores and the expert bias.
    See https://arxiv.org/abs/2408.15664 for details."""

    moe_router_bias_update_rate: float = 1e-3
    """The expert bias is updated based on the number of assigned tokens to each expert
    in a global batch, where the bias is increased for the experts with less assigned tokens
    and decreased for the experts with more assigned tokens.
    The default value 1e-3 is same as that used in DeepSeekV3."""

    moe_router_force_load_balancing: bool = False
    """[Experimental] Force load balancing with random logits for MoE router, supports naive topk 
    and group-limited topk. This is an experimental feature and only for benchmark."""

    moe_grouped_gemm: bool = False
    """When there are multiple experts per rank, compress multiple local (potentially small) gemms
    in a single kernel launch to improve the utilization and performance by leveraging the Grouped
    GEMM feature introduced since CUTLASS 2.8 (https://github.com/fanshiqing/grouped_gemm).
    """

    moe_use_legacy_grouped_gemm: bool = False
    """Use legacy GroupedMLP rather than TEGroupedMLP.
    Note: The legacy one will be deprecated soon."""

    moe_aux_loss_coeff: Union[float, List[float]] = 0.0
    """Scaling coefficient for the aux loss. A starting value of 1e-2 is recommended.
    If a list of load balancing types is provided for `moe_router_load_balancing_type`,
    a corresponding list of coefficients should be provided here."""

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
    Options are 'allgather','alltoall' and 'flex'."""

    moe_enable_deepep: bool = False
    """[Experimental] Enable DeepEP for efficient token dispatching and combine in MoE models."""

    moe_per_layer_logging: bool = False
    """Enable per-layer logging for MoE, currently supports auxiliary loss and z loss."""

    moe_expert_capacity_factor: Optional[float] = None
    """moe_expert_capacity_factor (float): The capacity factor for each expert, None means no token
    will be dropped. The default is None."""

    moe_pad_expert_input_to_capacity: bool = False
    """moe_pad_expert_input_to_capacity (bool): If True, pads the input for each expert to match
    the expert capacity length, effective only after the moe_expert_capacity_factor is set. The
    default setting is False."""

    moe_token_drop_policy: str = "probs"
    """The policy to drop tokens. Can be either "probs" or "position". If "probs", the tokens with
    the lowest probabilities will be dropped. If "position", tokens at the end of each batch will
    be dropped.
    """

    moe_layer_recompute: bool = False
    """Memory optimization: checkpointing moe_layer to save actiavtion memory."""

    moe_permute_fusion: bool = False
    """Fuse token rearrangement ops during token dispatching."""

    moe_router_fusion: bool = False
    """Fuse ops in routing and aux loss calculation."""

    moe_apply_probs_on_input: bool = False
    """Apply probs on input of experts instead of applying after activation and glu."""

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
    """DEPRECATED and replaced by cuda_graph_impl.
    When set to true, either partial CUDA graph (1/many CUDA graph per layer) or full iteration
    CUDA graph (1 CUDA graph for whole iteration excluding optimizer) is enabled. --cuda-graph-scope
    determines the scope of graph capture."""

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
    """DEPRECATED and replaced by cuda_graph_impl.
    When set to true, TransformerLayer layers are swapped with user provided CUDA graphs."""

    cuda_graph_impl: str = "none"
    """Determines the CUDA graph capture implementation.
    "none": no CUDA graph.
    "local": capture the CUDA graph using MCore local implementation. Either partial CUDA graph
    (1/many CUDA graph per layer) or full iteration CUDA graph (1 CUDA graph for whole iteration
    excluding optimizer) is enabled.
    "transformer_engine": capture the CUDA graph using TE make_graphed_callables()."""

    cuda_graph_scope: Optional[List[str]] = None
    """Determines the CUDA graphs capturing scope.
    When cuda_graph_impl is set to "transformer_engine", valid values are "attn", "mlp", "moe",
    "moe_router", "moe_preprocess", "mamba". None means ["attn", "mlp"].
    When cuda_graph_impl is set to "local", "full_iteration" can be specified as cuda_graph_scope
    to enable whole iteration CUDA graph. All other values enable layerwise CUDA graph."""

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

    use_te_activation_func: bool = False
    """Whether to use ffn activation functions implemented by TransformerEngine"""

    use_te_rng_tracker: bool = False
    """ Whether to use the TE or MCore version of the RNG tracker. """

    inference_rng_tracker: bool = False
    """ Whether we should instantiate a separate RNG tracker for inference. """

    inference_sampling_seed: int = 42
    """ Random seed to use for sampling during inference. """

    symmetric_ar_type: Optional[str] = None
    """Type of symmetric all reduce to use"""

    mrope_section: Optional[List[int]] = None
    """ Multimodal rope section is for channel dimension of temporal, height and width
    in rope calculation. """

    is_hybrid_model: bool = False
    """ Indicates whether this is a hybrid model. """

    mamba_state_dim: int = 128
    """The dimensionality of the state representation in Mamba layers."""

    mamba_head_dim: int = 64
    """The dimensionality of the heads in the Mamba layers."""

    mamba_num_groups: int = 8
    """The number of groups used in Mamba layers."""

    mamba_num_heads: Optional[int] = None
    """The number of heads used in Mamba layers. 
    If None, the number of heads will be hidden_size * expand // mamba_head_dim."""

    use_mamba_mem_eff_path: bool = True
    """If True, use the memory efficient path for Mamba layers."""

    mlp_chunks_for_prefill: int = 1
    """The number of chunks along the sequence dimension to use for MLP computation
    during prefill."""

    heterogeneous_block_specs: bool = False
    """Whether to use heterogeneous block specs (nemotron-nas architecture)."""

    hetereogenous_dist_checkpoint: bool = False
    """Whether to use heterogenous layers in distributed checkpoint."""

    ####################
    # Quantization
    ####################
    quant_recipe: Optional[RecipeConfig] = None
    """Configuration of any quantization to be applied to the model"""

    transformer_impl: str = "transformer_engine"
    """Transformer implementation to use.
    Options are 'transformer_engine' for Transformer Engine and 'local' for MCore."""

    def __post_init__(self):
        """Python dataclass method that is used to modify attributes after initialization.
        See https://docs.python.org/3/library/dataclasses.html#post-init-processing for more
        details.
        """
        super().__post_init__()
        if self.fp16 and self.bf16:
            raise ValueError(
                f"Only one of self.fp16: {self.fp16} and self.bf16 {self.bf16} should be True."
            )

        # Apply BF16 matmul precision setting if needed
        if self.bf16 and self.disable_bf16_reduced_precision_matmul:
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

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

        if self.fp8:
            # cannot support first last layer bf16 with delayed scaling
            if self.first_last_layers_bf16 and self.fp8_recipe == Fp8Recipe.delayed:
                raise ValueError("Delayed scaling does not support first / last layer in BF16.")

            # max bf16 layers per pipeline stage
            max_bf16_layers_per_pipeline_stage = (
                self.num_layers // self.pipeline_model_parallel_size
            )

            # check start/end bf16 layer counts are valid
            if self.first_last_layers_bf16:
                if (
                    self.num_layers_at_start_in_bf16 < 0
                    or self.num_layers_at_start_in_bf16 > max_bf16_layers_per_pipeline_stage
                ):
                    raise ValueError(
                        f"num_layers_at_start_in_bf16 ({self.num_layers_at_start_in_bf16}) must be "
                        f"between 0 and number of layers per pipeline stage "
                        f"({max_bf16_layers_per_pipeline_stage})."
                    )
                if (
                    self.num_layers_at_end_in_bf16 < 0
                    or self.num_layers_at_end_in_bf16 > max_bf16_layers_per_pipeline_stage
                ):
                    raise ValueError(
                        f"num_layers_at_end_in_bf16 ({self.num_layers_at_end_in_bf16}) must be "
                        f"between 0 and number of layers per pipeline stage "
                        f"({max_bf16_layers_per_pipeline_stage})."
                    )

        if self.fp8_param and not self.fp8:
            raise ValueError("fp8_param must be used together with fp8 mode.")

        # FP4 validation
        if self.fp4_param and not self.fp4:
            raise ValueError("fp4_param must be used together with fp4 mode.")

        if self.fp4 and self.fp8:
            raise ValueError("fp4 and fp8 cannot be used simultaneously. Please choose one.")

        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True

        if self.expert_model_parallel_size > 1 and self.num_moe_experts is None:
            raise ValueError("num_moe_experts must be non None to use expert-parallel.")

        if self.num_moe_experts is not None and self.num_moe_experts <= 0:
            raise ValueError("num_moe_experts must be non-negative.")

        if self.num_moe_experts is not None and self.moe_ffn_hidden_size is None:
            self.moe_ffn_hidden_size = self.ffn_hidden_size
            warnings.warn("moe_ffn_hidden_size is not set, using ffn_hidden_size instead.")

        if self.num_moe_experts is None:
            assert (
                self.moe_ffn_hidden_size is None
            ), "moe_ffn_hidden_size must be None when num_experts is not set."

        if self.moe_enable_deepep:
            if self.moe_token_dispatcher_type != "flex":
                raise ValueError("DeepEP backend is only supported with flex token dispatcher.")

        if self.moe_token_dispatcher_type == "flex":
            if self.moe_pad_expert_input_to_capacity:
                raise ValueError(
                    "Flex token dispatcher does not support moe_pad_expert_input_to_capacity"
                )

        if self.moe_shared_expert_intermediate_size is not None:
            if self.moe_shared_expert_intermediate_size <= 0:
                raise ValueError(
                    f"moe_shared_expert_intermediate_size must be "
                    f"num_shared_experts * ffn_size_of_each_shared_expert, "
                    f"but got {self.moe_shared_expert_intermediate_size}"
                )
            if self.moe_shared_expert_overlap and self.moe_token_dispatcher_type not in [
                "alltoall"
            ]:
                raise ValueError(
                    f"moe_shared_expert_overlap only works with alltoall token dispatcher."
                )

        if isinstance(self.moe_router_load_balancing_type, list):
            assert isinstance(self.moe_aux_loss_coeff, list) and len(
                self.moe_aux_loss_coeff
            ) == len(self.moe_router_load_balancing_type), (
                "moe_aux_loss_coeff must be a list of the same length as "
                "moe_router_load_balancing_type"
            )

        if self.moe_expert_capacity_factor is not None:
            if self.moe_expert_capacity_factor < 0:
                self.moe_expert_capacity_factor = None
            if isinstance(self.moe_router_load_balancing_type, list):
                for load_balancing_type in self.moe_router_load_balancing_type:
                    if load_balancing_type not in [
                        "aux_loss",
                        "seq_aux_loss",
                        "global_aux_loss",
                        "none",
                    ]:
                        raise ValueError(
                            "moe_expert_capacity_factor only works with aux_loss, "
                            "seq_aux_loss, global_aux_loss or none load balancing"
                        )
            elif self.moe_router_load_balancing_type not in [
                "aux_loss",
                "seq_aux_loss",
                "global_aux_loss",
                "none",
            ]:
                raise ValueError(
                    "moe_expert_capacity_factor only works with aux_loss, "
                    "seq_aux_loss, global_aux_loss or none load balancing"
                )

        if self.moe_pad_expert_input_to_capacity:
            if self.moe_expert_capacity_factor is None:
                raise ValueError(
                    "moe_expert_capacity_factor must be set to use moe_pad_expert_input_to_capacity"
                )

        if self.cpu_offloading and (
            self.cpu_offloading_num_layers < 0 or self.cpu_offloading_num_layers >= self.num_layers
        ):
            raise ValueError(
                f"CPU offloading can be done only for layers less than {self.num_layers}"
            )

        if self.cpu_offloading and self.pipeline_model_parallel_size > 1:
            raise ValueError(
                "Currently there is no support for Pipeline parallelism with CPU offloading"
            )

        if self.cpu_offloading and self.recompute_granularity is not None:
            raise ValueError(
                "CPU offloading does not work when activation recomputation is enabled"
            )

        if self.recompute_granularity is not None:
            if self.recompute_granularity not in ["full", "selective"]:
                raise ValueError(
                    f'When using recompute_granuarlity: {self.recompute_granularity} must be "full"'
                    'or "selective".'
                )

            if self.recompute_method is not None:
                if self.recompute_method not in ["block", "uniform"]:
                    raise ValueError(
                        f'recompute_method: {self.recompute_method} must be "block" or "uniform".'
                    )
            elif self.recompute_granularity != "selective":
                raise ValueError(
                    f"Using recompute_granularity: {self.recompute_granularity} so "
                    'recompute_method must be "block" or "uniform"'
                )

            if self.recompute_granularity != "selective" and self.recompute_num_layers is None:
                raise ValueError(
                    f"When using recompute_granularity: {self.recompute_granularity} "
                    "recompute_num_layers must be between "
                    "1 and num_layers_per_pipeline_rank: "
                    f"{self.num_layers // self.pipeline_model_parallel_size}"
                )
            elif (
                self.recompute_granularity == "selective" and self.recompute_num_layers is not None
            ):
                raise ValueError(
                    f"When using recompute_granularity: {self.recompute_granularity} "
                    "recompute_num_layers must be None."
                )

            if self.distribute_saved_activations and self.sequence_parallel:
                raise ValueError(
                    f"distribute_saved_activations: {self.distribute_saved_activations} must be "
                    f"false when sequence parallel is enabled: {self.sequence_parallel}"
                )

        if self.recompute_modules is None:
            self.recompute_modules = ["core_attn"]

        if self.recompute_granularity == "selective":
            if len(self.recompute_modules) > 0:
                allowed_modules = {
                    "core_attn",
                    "moe_act",
                    "layernorm",
                    "mla_up_proj",
                    "mlp",
                    "moe",
                    "shared_experts",
                }
                invalid_modules = set(self.recompute_modules) - allowed_modules
                assert not invalid_modules, (
                    f"Invalid choices for recompute_modules: {invalid_modules}. "
                    f"Allowed modules are: {allowed_modules}"
                )

            if "moe_act" in self.recompute_modules and not self.moe_grouped_gemm:
                raise ValueError(
                    "moe_act in recompute_modules is only supported with moe_grouped_gemm."
                )

            if "mla_up_proj" in self.recompute_modules and not self.multi_latent_attention:
                raise ValueError(
                    "mla_up_proj in recompute_modules is only supported with "
                    "multi_latent_attention."
                )

            if "core_attn" in self.recompute_modules:
                warnings.warn(
                    "If you are using transformer_engine as the transformer implementation, "
                    "the core_attn is from transformer_engine and may be the fused version. "
                    "For fused attention, you have no need to set 'core_attn' to recompute. "
                    "Please check that the core_attn recompute is really needed."
                )

            if "shared_experts" in self.recompute_modules:
                if (
                    self.moe_shared_expert_intermediate_size is not None
                    and self.moe_shared_expert_overlap
                ):
                    raise ValueError(
                        "shared_experts recompute cannot work with --moe-shared-expert-overlap."
                    )

            if self.fp8:
                if "moe_act" in self.recompute_modules or "layernorm" in self.recompute_modules:
                    if self.fp8_recipe == 'delayed':
                        raise ValueError(
                            "Delayed scaling does not support moe_act and layernorm recompute "
                            "for fp8."
                        )
                    if not is_te_min_version("2.6.0dev0"):
                        raise ValueError(
                            "moe_act and layernorm recompute for fp8 needs "
                            "transformer-engine>=2.6.0dev0, "
                            f"but your version is {get_te_version()}."
                        )

        if self.moe_layer_recompute:
            warnings.warn(
                "--moe-layer-recompute is deprecated. "
                "Use --recompute-granularity selective --recompute-modules moe_layer instead."
            )
            if self.recompute_granularity == "full":
                raise ValueError(
                    "Do not set --moe-layer-recompute with full recompute granularity. "
                )
            self.recompute_granularity = "selective"
            if "moe" not in self.recompute_modules:
                self.recompute_modules.append("moe")

        if (
            self.num_layers_in_first_pipeline_stage is not None
            or self.num_layers_in_last_pipeline_stage is not None
        ) and (
            self.account_for_embedding_in_pipeline_split or self.account_for_loss_in_pipeline_split
        ):
            raise ValueError(
                "num_layers_in_first_pipeline_stage and num_layers_in_last_pipeline_stage cannot be"
                "set at the same time with account_for_embedding_in_pipeline_split"
                "and account_for_loss_in_pipeline_split"
            )

        # PP layout
        if self.pipeline_model_parallel_layout is not None:
            # If pipeline layout is set, we will check the conflicts
            # with other pipeline layout arguments.
            any_conflict = (
                self.num_layers_in_first_pipeline_stage is not None
                or self.num_layers_in_last_pipeline_stage is not None
                or self.account_for_embedding_in_pipeline_split
                or self.account_for_loss_in_pipeline_split
            )
            if any_conflict:
                raise ValueError(
                    "pipeline_model_parallel_layout cannot be set"
                    " with other pipeline layout arguments."
                    f" {self.num_layers_in_first_pipeline_stage=},"
                    f" {self.num_layers_in_last_pipeline_stage=},"
                    f" {self.account_for_embedding_in_pipeline_split=},"
                    f" {self.account_for_loss_in_pipeline_split=}."
                )

            # Transfer pipeline_model_parallel_layout from str or list to
            # PipelineParallelLayerLayout
            if isinstance(self.pipeline_model_parallel_layout, str):
                self.pipeline_model_parallel_layout = PipelineParallelLayerLayout.from_str(
                    layout=self.pipeline_model_parallel_layout,
                    pipeline_model_parallel_size=self.pipeline_model_parallel_size,
                )
            elif isinstance(self.pipeline_model_parallel_layout, list):
                # Since list is not hashable, the initialization will not be cached.
                self.pipeline_model_parallel_layout = PipelineParallelLayerLayout(
                    layout=self.pipeline_model_parallel_layout,
                    pipeline_model_parallel_size=self.pipeline_model_parallel_size,
                )

            # Check whether the input VPP size conflicts with the PP layout
            detected_vpp_size = (
                self.pipeline_model_parallel_layout.virtual_pipeline_model_parallel_size
            )
            if self.virtual_pipeline_model_parallel_size is not None:
                assert self.virtual_pipeline_model_parallel_size == detected_vpp_size, (
                    f"virtual_pipeline_model_parallel_size conflicts with"
                    f" pipeline_model_parallel_layout,"
                    f" ({self.virtual_pipeline_model_parallel_size=}, "
                    f" {detected_vpp_size=})"
                )
            elif detected_vpp_size > 1:
                self.virtual_pipeline_model_parallel_size = detected_vpp_size

            # Check whether the layout is valid.
            self.pipeline_model_parallel_layout.validate_layer_layout(
                num_layers=self.num_layers, mtp_num_layers=self.mtp_num_layers
            )

        # Uneven PP
        elif (
            self.num_layers_in_first_pipeline_stage is not None
            or self.num_layers_in_last_pipeline_stage is not None
        ):
            pipeline_parallel_size = self.pipeline_model_parallel_size
            num_layers = self.num_layers

            if self.num_layers_in_first_pipeline_stage is not None:
                if self.num_layers_in_first_pipeline_stage <= 0:
                    raise ValueError("num_layers_in_first_pipeline_stage must be larger than 0")

                if self.virtual_pipeline_model_parallel_size is not None:
                    if (
                        self.num_layers_in_first_pipeline_stage
                        % self.virtual_pipeline_model_parallel_size
                        != 0
                    ):
                        raise ValueError(
                            f"number of layers at first stage: "
                            f"{self.num_layers_in_first_pipeline_stage}"
                            f"must be divisible by virtual pipeline"
                            f"parallel degree {self.virtual_pipeline_model_parallel_size}"
                        )
                num_layers -= self.num_layers_in_first_pipeline_stage
                pipeline_parallel_size -= 1

            if self.num_layers_in_last_pipeline_stage is not None:
                if self.num_layers_in_last_pipeline_stage <= 0:
                    raise ValueError("num_layers_in_last_pipeline_stage must be larger than 0")

                if self.virtual_pipeline_model_parallel_size is not None:
                    if (
                        self.num_layers_in_last_pipeline_stage
                        % self.virtual_pipeline_model_parallel_size
                        != 0
                    ):
                        raise ValueError(
                            f"number of layers at last stage: "
                            f"{self.num_layers_in_last_pipeline_stage}"
                            f"must be divisible by virtual pipeline"
                            f"parallel degree {self.virtual_pipeline_model_parallel_size}"
                        )
                num_layers -= self.num_layers_in_last_pipeline_stage
                pipeline_parallel_size -= 1

            # Here pipeline_parallel_size is the number of middle PP stages. If there are middle
            # PP stages, check number of layers at middle stage is divisible by middle PP size.
            if pipeline_parallel_size and not num_layers % pipeline_parallel_size == 0:
                raise ValueError(
                    f"number of layers at middle stage: {num_layers} must be divisible by"
                    f"the middle pipeline model parallel size {pipeline_parallel_size}"
                )

            # If there are middle PP stages, check number of layers
            # on each middle PP rank is divisible by VPP size.
            if pipeline_parallel_size and self.virtual_pipeline_model_parallel_size is not None:
                num_layers_per_middle_pipeline_rank = num_layers // pipeline_parallel_size
                if (
                    not num_layers_per_middle_pipeline_rank
                    % self.virtual_pipeline_model_parallel_size
                    == 0
                ):
                    raise ValueError(
                        f"number of layers on each middle pipeline rank:"
                        f"{num_layers_per_middle_pipeline_rank} must be divisible by virtual"
                        f"pipeline parallel degree {self.virtual_pipeline_model_parallel_size}"
                    )

        elif (
            self.account_for_embedding_in_pipeline_split or self.account_for_loss_in_pipeline_split
        ):
            if self.virtual_pipeline_model_parallel_size is None:
                num_layers = self.num_layers

                if self.account_for_embedding_in_pipeline_split:
                    num_layers += 1

                if self.account_for_loss_in_pipeline_split:
                    num_layers += 1

                if not num_layers % self.pipeline_model_parallel_size == 0:
                    raise ValueError(
                        f"number of middle layers: {num_layers} must be divisible by "
                        f"middle pipeline_model_parallel_size {self.pipeline_model_parallel_size}"
                    )
            else:
                num_layers = self.num_layers
                if self.account_for_embedding_in_pipeline_split:
                    num_layers += 1

                if self.account_for_loss_in_pipeline_split:
                    num_layers += 1

                if not num_layers % self.pipeline_model_parallel_size == 0:
                    raise ValueError(
                        f"num_layers: {num_layers} after enable"
                        f"account_for_embedding_in_pipeline_split or "
                        f"account_for_loss_in_pipeline_split must be divisible"
                        f"by pipeline_model_parallel_size "
                        f"{self.pipeline_model_parallel_size}"
                    )

                num_layers_per_pipeline_rank = num_layers // self.pipeline_model_parallel_size
                if (
                    not num_layers_per_pipeline_rank % self.virtual_pipeline_model_parallel_size
                    == 0
                ):
                    raise ValueError(
                        f"number of layers on each pipeline rank: {num_layers_per_pipeline_rank}"
                        f"(after enable account_for_embedding_in_pipeline_split or "
                        f"account_for_loss_in_pipeline_split) must be divisible by"
                        f"virtual_pipeline_model_parallel_size"
                        f"{self.virtual_pipeline_model_parallel_size}"
                    )

        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True

        if self.bias_activation_fusion:
            if self.activation_func not in [F.gelu, F.silu, quick_gelu]:
                raise ValueError(
                    "When bias_activation_fusion is True, activation function should be either "
                    "gelu, swiglu, or quick_geglu"
                )
            if (
                self.activation_func == F.gelu
                and not self.gated_linear_unit
                and not self.add_bias_linear
            ):
                raise ValueError(
                    "When bias_activation_fusion is True, gated_linear_unit is False "
                    "and activation function is gelu, add_bias_linear must also be True."
                )
            if self.activation_func == quick_gelu and not self.gated_linear_unit:
                raise ValueError(
                    "When bias_activation_fusion is True and activation function is quick_gelu, "
                    "gated_linear_unit must be True."
                )
            if self.glu_linear_offset != 0.0 and self.activation_func != quick_gelu:
                raise ValueError(
                    "When bias_activation_fusion is True and glu_linear_offset is non-zero, "
                    "activation function must be quick_gelu."
                )

            if self.use_te_activation_func:
                raise ValueError(
                    "bias_activation_fusion and use_te_activation_func cannot be both true. "
                    "If you use bias in MLP FC1, we recommend setting bias_activation_fusion "
                    "to True and use_te_activation_func to False."
                )

        if self.use_te_activation_func:
            if self.activation_func not in (F.gelu, F.silu, F.relu):
                raise ValueError(
                    "TransformerEngine only support gelu, geglu, silu, swiglu, relu, reglu. "
                    "If you don't want to use TransformerEngine activation function, set "
                    "use_te_activation_func to False"
                )

        if self.activation_func_fp8_input_store:
            if self.activation_func != F.silu or not self.gated_linear_unit:
                raise ValueError("Storing activation input in FP8 is supported only for SwiGLU.")

        if self.apply_rope_fusion:
            if self.multi_latent_attention:
                warnings.warn(
                    "apply_rope_fusion for multi-latent attention only supports training. "
                    "It is experimental and may change in future versions."
                )
            else:
                if self.rotary_interleaved:
                    if not is_te_min_version("2.3.0"):
                        raise ValueError(
                            "rotary_interleaved does not work with apply_rope_fusion for "
                            "TE < 2.3.0. Please install TE >= 2.3.0"
                        )

                from megatron.core.models.common.embeddings.rope_utils import (
                    fused_apply_rotary_pos_emb,
                    fused_apply_rotary_pos_emb_thd,
                )

                if fused_apply_rotary_pos_emb is None and fused_apply_rotary_pos_emb_thd is None:
                    raise ValueError(
                        "apply_rope_fusion is not available. Please install TE >= 1.4."
                    )

        if self.multi_latent_attention and self.rotary_interleaved:
            raise ValueError("rotary_interleaved does not work with multi_latent_attention.")

        # Set the embedding init method
        if self.embedding_init_method_std is None:
            # By default, use the same init std as you use for every other non-output layer.
            self.embedding_init_method_std = self.init_method_std

        if self.embedding_init_method is None:
            if self.init_method is None or (self.embedding_init_method_std != self.init_method_std):
                # In this case, we set both the init method and the embedding init method to
                #  whatever std value requested (or defaulted) for the embedding_init_layer
                self.embedding_init_method = init_method_normal(self.embedding_init_method_std)
            else:
                # Replicate the current behavior where if you are not changing the std of the
                #  embedding init differently and the init method is set, we fallback to the
                #  init method for this layer. Since we are here after an OR we know that
                #  init_method is not None
                self.embedding_init_method = self.init_method

        if self.init_method is None:
            self.init_method = init_method_normal(self.init_method_std)

        if self.output_layer_init_method is None:
            self.output_layer_init_method = scaled_init_method_normal(
                self.init_method_std,
                self.num_layers,
                multiplier=2.0 if not self.is_hybrid_model else 1.0,
            )

        if self.num_moe_experts is not None and self.add_bias_linear:
            assert (
                self.expert_tensor_parallel_size == 1
            ), "Bias in Moe is only supported when ETP==1"

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

        if self.moe_router_padding_for_fp8:
            if self.fp8 is None:
                raise ValueError("fp8 must be specified when moe_router_padding_for_fp8 is True.")

            if self.moe_token_dispatcher_type in ["allgather", "alltoall_seq"]:
                raise ValueError(
                    "allgather and alltoall_seq dispatcher does not support "
                    "moe_router_padding_for_fp8."
                )

        if (
            self.moe_router_topk == 1
            and self.moe_router_score_function == "softmax"
            and not self.moe_router_pre_softmax
            and self.moe_router_load_balancing_type != "sinkhorn"
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

        if self.enable_cuda_graph or self.external_cuda_graph:
            assert (
                self.cuda_graph_impl == "none"
            ), "Do not use enable_cuda_graph or external_cuda_graph with cuda_graph_impl."
            assert (
                not self.enable_cuda_graph or not self.external_cuda_graph
            ), "enable_cuda_graph and external_cuda_graph cannot be enabled at the same time."

            if self.enable_cuda_graph:
                warnings.warn('enable_cuda_graph is deprecated, use cuda_graph_impl=local instead.')
                self.cuda_graph_impl = "local"
            if self.external_cuda_graph:
                warnings.warn(
                    'external_cuda_graph is deprecated, '
                    'use cuda_graph_impl=transformer_engine instead.'
                )
                self.cuda_graph_impl = "transformer_engine"
        if self.cuda_graph_impl != "none":
            assert self.cuda_graph_impl in [
                "transformer_engine",
                "local",
            ], f"Invalid cuda graph implementation: {self.cuda_graph_impl}"
            if self.cpu_offloading:
                raise ValueError("CUDA graphs not supported with CPU offloading.")
            if self.cuda_graph_scope is None:
                if self.cuda_graph_impl == "transformer_engine":
                    if self.num_moe_experts is None or self.num_moe_experts <= 1:
                        self.cuda_graph_scope = ['attn', 'mlp']
                    elif self.moe_layer_freq == 1 or (
                        isinstance(self.moe_layer_freq, list) and 0 not in self.moe_layer_freq
                    ):
                        self.cuda_graph_scope = ['attn', 'moe']
                    else:
                        self.cuda_graph_scope = ['attn', 'mlp', 'moe']
                elif self.cuda_graph_impl == "local":
                    self.cuda_graph_scope = []
            elif 'full_iteration' in self.cuda_graph_scope:
                assert self.cuda_graph_impl == "local" and len(self.cuda_graph_scope) == 1, (
                    "Set cuda_graph_impl=local and cuda_graph_scope=[full_iteration] "
                    "for full iteration CUDA graph."
                )

            if self.recompute_granularity:
                if (
                    self.recompute_granularity != "selective"
                    or self.cuda_graph_impl != "transformer_engine"
                ):
                    raise ValueError("CUDA graphs not supported with activation recomputation.")
                else:
                    if "attn" in self.cuda_graph_scope:
                        for module in self.recompute_modules:
                            if module in ['core_attn', 'mla_up_proj']:
                                raise ValueError(
                                    f'attn cuda graph is not supported with {module} recompute.'
                                )
                    if "mlp" in self.cuda_graph_scope and "mlp" in self.recompute_modules:
                        raise ValueError(f'mlp cuda graph is not supported with mlp recompute.')
                    if "moe" in self.cuda_graph_scope:
                        for module in self.recompute_modules:
                            if module in ['moe_act', 'moe', 'shared_experts']:
                                raise ValueError(
                                    f'moe cuda graph is not supported with {module} recompute.'
                                )
                    if "moe_router" in self.cuda_graph_scope:
                        for module in self.recompute_modules:
                            if module in ['moe', 'shared_experts']:
                                raise ValueError(
                                    f'moe_router cuda graph is not supported with {module} '
                                    'recompute.'
                                )
                    if "layernorm" in self.recompute_modules:
                        if (
                            "attn" in self.cuda_graph_scope
                            and "mlp" in self.cuda_graph_scope
                            and (
                                "moe" in self.cuda_graph_scope
                                or "moe_router" in self.cuda_graph_scope
                            )
                        ):
                            raise ValueError(
                                'cuda graph is not supported with layernorm recompute.'
                            )
                        if "attn" in self.cuda_graph_scope:
                            warnings.warn(
                                "input_layernorm recompute is not supported with attention "
                                "cudagraph. Will only recompute the pre_mlp_layernorm."
                            )
                        if (
                            "mlp" in self.cuda_graph_scope
                            or "moe" in self.cuda_graph_scope
                            or "moe_router" in self.cuda_graph_scope
                        ):
                            warnings.warn(
                                "pre_mlp_layernorm recompute is not supported with mlp/moe "
                                "cudagraph. Will only recompute the input_layernorm."
                            )

        if self.moe_token_dispatcher_type in ["allgather"]:
            if self.variable_seq_lengths is True:
                raise ValueError(
                    f"Token dispatcher type: {self.moe_token_dispatcher_type} does not support "
                    f"variable sequence length, please use alltoall dispatcher instead."
                )

        if self.moe_permute_fusion:
            from megatron.core.transformer.moe.moe_utils import (
                fused_permute,
                fused_permute_with_probs,
                fused_sort_chunks_by_index,
                fused_sort_chunks_by_index_with_probs,
                fused_unpermute,
            )

            if (
                fused_permute is None
                or fused_permute_with_probs is None
                or fused_sort_chunks_by_index is None
                or fused_sort_chunks_by_index_with_probs is None
                or fused_unpermute is None
            ):
                raise ValueError("fused permutation is not available. Please install TE >= 2.1.0.")

        if self.overlap_moe_expert_parallel_comm:
            # TODO: remove this after we fix the hang issue with torch version < 2.6.0
            assert is_torch_min_version(
                "2.6.0"
            ), "A2A Overlap encounters hang issue with torch version < 2.6.0"
            if self.pipeline_model_parallel_size > 1:
                assert self.virtual_pipeline_model_parallel_size is not None, (
                    "If enabling EP A2A overlap, virtual_pipeline_model_parallel_size "
                    "must be specified when pipeline_model_parallel_size > 1"
                )
            # Expert model parallelism requirements
            assert (
                self.expert_model_parallel_size > 1
            ), 'overlap_moe_expert_parallel_comm is only supported with expert model parallelism'
            assert self.moe_token_dispatcher_type in [
                'alltoall',
                'flex',
            ], 'overlap_moe_expert_parallel_comm is supported with alltoall/flex token dispatcher'

            assert (
                self.recompute_granularity != 'full'
            ), 'disable full recomputation when enabling overlap_moe_expert_parallel_comm'
            assert (
                self.recompute_method is None
            ), 'disable recomputation method when enabling overlap_moe_expert_parallel_comm'
            assert (
                self.recompute_num_layers is None
            ), 'recompute_num_layers must be None when enabling overlap_moe_expert_parallel_comm'

            # Check if bf16 or fp16 is used
            assert (
                self.bf16 or self.fp16
            ), 'overlap_moe_expert_parallel_comm is only supported with bf16 or fp16 model'

            assert (
                not self.moe_shared_expert_overlap
            ), 'disable moe_shared_expert_overlap when enabling overlap_moe_expert_parallel_comm'
            assert (
                self.mtp_num_layers is None or self.mtp_num_layers == 1
            ), 'MTP layernum only supports 1 when enabling overlap_moe_expert_parallel_comm.'

        # Check delay_wgrad_compute compatibility
        if self.delay_wgrad_compute:
            assert (
                self.overlap_moe_expert_parallel_comm
            ), 'overlap_moe_expert_parallel_comm must be enabled when enabling delay_wgrad_compute'
            assert (
                not self.moe_use_legacy_grouped_gemm
            ), 'delay_wgrad_compute is not supported with legacy groupedgemm implementation'

        if self.context_parallel_size > 1 and self.cp_comm_type is not None:
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

        if (
            self.num_moe_experts is not None
            and self.num_moe_experts >= 32
            and not self.moe_router_dtype
        ):
            warnings.warn(
                "Using a large number of experts (e.g. >=32) without fp32 routing. "
                "Consider enabling moe_router_dtype for better numerical stability."
            )
        if self.symmetric_ar_type is not None:
            if not HAVE_PACKAGING:
                raise ImportError(
                    "packaging is not installed. Please install it with `pip install packaging`."
                )
            assert is_torch_min_version("2.7.0a0"), "Must have at least torch version 2.7 or higher"
            assert is_te_min_version("2.3.0") or get_te_version() == PkgVersion(
                "2.3.0.dev0+39c0e70"
            ), "Must have at least TE version 2.3 or higher to use symmetric memory all reduce"

        if self.no_rope_freq:
            assert not self.flash_decode, "flash_decode cannot be used with no_rope."
            if isinstance(self.no_rope_freq, int):
                assert self.num_layers % self.no_rope_freq == 0, (
                    f"no_rope_freq={self.no_rope_freq} should be "
                    f"divisible by num_layers={self.num_layers}."
                )
                # Convert integer pattern to list pattern
                # e.g. no_rope=4 with num_layers=8 becomes [0,0,0,1,0,0,0,1]
                pattern = [0] * (self.no_rope_freq - 1) + [1]
                self.no_rope_freq = pattern * (self.num_layers // self.no_rope_freq)
            else:
                assert len(self.no_rope_freq) == self.num_layers, (
                    f"Length of no_rope list ({len(self.no_rope_freq)}) must match "
                    f"the number of layers ({self.num_layers})"
                )


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

    normalization: str = "RMSNorm"
    """Default normalization layer for MLA models is RMSNorm."""

    rope_type: str = "yarn"
    """Type of RoPE to use. Default to yarn, options are rope and yarn."""

    rotary_base: float = 10000
    """Rotary base for the rotary embeddings, used by rope and yarn."""

    rotary_percent: float = 1.0
    """Rotary percent for the rotary embeddings, used by rope."""

    rotary_scaling_factor: float = 40
    """Rotary scaling factor for the rotary embeddings, used by yarn."""

    original_max_position_embeddings: int = 4096
    """Original maximum position embeddings for the original model, used by yarn."""

    beta_fast: float = 32
    """Beta fast for YaRN RoPE, used by yarn."""

    beta_slow: float = 1
    """Beta slow for YaRN RoPE, used by yarn."""

    mscale: float = 1.0
    """Mscale for YaRN RoPE in Multi-Latent Attention, used by yarn."""

    mscale_all_dim: float = 0.0
    """Mscale all dimensions for YaRN RoPE in Multi-Latent Attention, used by yarn."""

    cache_mla_latents: bool = False
    """Cache the low dimensional tensors for MLA rather than full KV cache.
       This is only for the dynamic inference backend and requires that 
       Flash MLA is installed."""

    def __post_init__(self):
        super().__post_init__()
        if self.multi_latent_attention and self.apply_rope_fusion and self.rope_type != "yarn":
            raise ValueError("apply_rope_fusion for MLA only works with YARN RoPE.")

        if self.cache_mla_latents:
            assert (
                self.apply_rope_fusion is False
            ), "Rope Fusion is not compatible with caching latents"
