# Megatron Core MoE

TODO: add a summary of the latest features and architectures.
Megatron-Core MoE provides comprehensive parallelism strategies for Mixture-of-Experts models, seamlessly integrating Expert Parallelism with tensor, data, sequence, and pipeline parallelism.

## What's New
For latest features and architectures, please refer to the [MCore dev roadmap](https://github.com/NVIDIA/Megatron-LM/issues/1729).

### 🔥 [MCore dev] (2026/01)
- 🚀 Pipeline-aware fine-grained activation offloading 
- 🚀 Qwen3-Next model support
- 🚀 Muon and Layer-wise distributed optimizer

### 🔥 [MCore v0.15] (2025/11)
- 🚀 Add HybridEP backend to Flex Dispatcher(GB200, B200, H100 supported)
- 🚀 Support FSDP with EP for MoE models

### 🔥 [MCore v0.14] (2025/09)
- 🚀 Batch-level overlapping to hide EP-A2A communication (--overlap-moe-expert-parallel-comm --delay-wgrad-compute)
- 🚀 FP8 support for Fine-grained Recomputations
- Router fusion kernels for MoE models (--moe-router-fusion)
- Context Parallelism (CP) support for MTP and MLA

### 🔥 [MCore v0.13] (2025/07)
- Support bf16 dtype for optimizer states to use precision-aware optimizer in TransformerEngine (--use-precision-aware-optimizer)
- Flexible Asymmetric Virtual Pipeline Parallelism with Custom Pipeline Layout (--pipeline-model-parallel-layout)
- Add Hybrid Shard Data-Parallel support for MoE models (--num-distributed-optimizer-instances)
- Fine-grained recomputation to reduce activation memory. (--recompute-modules with --recompute-granularity selective)
- Memory efficient token permutation by moving the probs multiplication from unpermutation to activation function of GroupedMLP.

### 🔥 [MCore v0.12] (2025/05)
- Support DeepSeek's DeepEP for efficient token dispatching (--moe-token-dispatcher-type flex --moe-enable-deepep)
- Support Multi-Token Prediction (MTP) (--mtp-num-layers 1)
- CUDA Graph support for dropless MoE models with attention only capture (--te-rng-track --external-cuda-graph --cuda-graph-scope attn)

## Overview of MCore MoE Supported Features and Architectures

### Model Support
- ✅ **DeepSeek**
  - ✅ DeepSeek-V2
  - ✅ DeepSeek-V3, including MTP
- ✅ **Qwen**
  - ✅ Qwen2-57B-A14B
  - ✅ Qwen3-30B-A3B
  - ✅ Qwen3-235B-A22B
- ✅ **Mixtral**
  - ✅ Mixtral-8x7B
  - ✅ Mixtral-8x22B

### Core MoE Functionality
- ✅ Token dropless MoE (dMoE) - Advanced routing without token dropping
- ✅ Top-K Router with flexible K selection
- ✅ Load balancing losses for expert utilization optimization

### Advanced Parallelism
- ✅ Expert Parallel (EP) with 3D parallelism integration
- ✅ Full parallelism combo: EP + DP + TP + PP + SP support
- ✅ Context Parallel (CP) for long sequence MoE training
- ✅ Parallel Folding Heterogeneous Parallelism Mappings for Efficient Large-Scale MoE Model Training
- ✅ Distributed Optimizer for MoE (ZeRO-1 equivalent)

### Performance Optimizations
- ✅ Memory Efficient token permutation
- ✅ Fine-grained Recomputations (mla, moe, mlp, moe_act, norm)
- ✅ MLA TP Support for Mixture of Linear Attention
- ✅ GroupedGEMM and GA Fusion
- ✅ DP/PP/TP Communication Overlapping
- ✅ Overlapped Shared Expert execution
- ✅ Router Fusion optimizations
- ✅ Token (un)permutation Fusion kernels
- ✅ cuDNN fused Attention integration

### Hardware & Precision Support
- ✅ DeepEP support for H100 and B200
- ✅ GroupedGEMM including FP8/MXFP8 support
- ✅ FP8 weights with BF16 optimizer states
- ✅ FP8 training full support

### Developer Experience
- ✅ MoE Model Zoo with pre-training best practices
- ✅ Distributed Checkpointing for MoE models
- ✅ Upcycling Support for model scaling
- ✅ MCore2HF Converter for ecosystem compatibility
- ✅ Layer-wise logging for detailed monitoring
- ✅ Runtime Upcycling capabilities

## Quick Start Guide

### Basic MoE Training in Megatron-LM

To train a top-2 MoE model with 8 experts and auxiliary loss, add the following arguments to your megatron training script:

```bash
## Set MoE Hidden site
--num-experts 8
--moe-shared-expert-intermediate-size: 2048
## Set router config
--moe-router-load-balancing-type aux_loss
--moe-router-topk 2
--moe-aux-loss-coeff 1e-2
## Set token dispatcher
--moe-token-dispatcher-type alltoall
```

Detailed documentation for each feature is available in the [Feature Documentation](#feature-documentation) section.

### Use the pre-defined config to train the popular MoE models
We have provided some pre-defined config to train the popular MoE models in the [Megatron-MoE-Model-Zoo](https://github.com/yanring/Megatron-MoE-ModelZoo/tree/main) repository. You can use them as a reference to configure your training script. Currently we have added the config for Mixtral 8x7B, Mixtral 8x22B, DeepSeek-V3, Qwen3-30B-A3B, Qwen3-235B-A22B.

## Best Practices to achieve high performance on MoE training

### General Performance Tips
#### Training arguments
The following flags are general performance flags that can help to achieve higher performance on almost all workloads. Check if you have enabled all of them in your training script.

```bash
## Enable DeepEP token dispatcher
--moe-token-dispatcher-type flex
--moe-flex-dispatcher-backend deepep
## Enable GroupedGEMM
--moe-grouped-gemm
## Enable fusion kernels
--moe-router-fusion
--moe-permute-fusion
--cross-entropy-loss-fusion
--cross-entropy-fusion-impl te

## Communication optimization
--use-distributed-optimizer
--overlap-param-gather
--overlap-grad-reduce
--tp-comm-overlap

## Enable manual gc to prevent python jitter
--manual-gc: true
--manual-gc-interval: 10
```
#### Environment variables

Below are some environment variables that can be useful.
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True # Enable expandable segments to prevent memory fragmentation
export NCCL_NVLS_ENABLE=0 # Disable NVLS to prevent memory overhead
```
#### Dependencies
- Use the latest version of [TransformerEngine](https://github.com/NVIDIA/TransformerEngine).
- Use the latest [NGC PyTorch Docker Image](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch)

### Best Practices to achieve high performance on MoE training

分布式训练涉及通信、显存、计算之间的各种tradeoff，这也使得找到一个最优的parallel mapping比较复杂，这里我们提供了一个相对通用的流程来帮助你找到一个最优的并行训练配置。

### Step 1: Find the feasible parallel mapping under the memory capacity of the GPU
To find the best parallel mapping, we need to first know the feasible parallel mapping for the model under the memory capacity of the GPU.
The consumption of memory consists of three parts:
- Activation memory
- Weight and gradient memory
- Optimizer states memory
Different parallel strategies will shard these tensor memory in different ways.
| Parallel Strategy | Peak Activation Memory          | Weight Memory  | Optimizer states                  | Communication (Per-Layer) |
|:-----------------:|:-------------------------------:|:--------------:|:---------------------------------:|:-------------------------:|
| TP                | 1/N (with SP on)                | 1/N            | 1/N                               |        High               |
| EP                | ~1 (varies with EP balancing)   | 1/N in MoELayer| 1/N                               |       Medium              |
| PP                | 1 (>1 with virtual pipeline)    | 1/N            | 1/N                               |       Medium              |
| CP                | 1/N                             | 1              | 1/N (with distributed optimizer)  |       Medium              |
| DP                | 1                               | 1              | 1/N (with distributed optimizer)  |        Low                |

We provide the argument of `--fake-init-process-group` to emulate distributed training on one GPU. This is useful to find the feasible parallel mapping under the memory capacity of the GPU. See https://github.com/NVIDIA/Megatron-LM/pull/2254 for detailed usage.

### Step 2: Select Optimal Parallelism Strategy
To find a good parallel mapping that help you achieve a high throughput of a new model, there are some general rule that could help. Here is an overview of properties in different aspects for each parallel strategy.

For a specific model, the best parallel mapping varies based on the model architecture, trained sequence length and the hardware platform.
Here we provide some general rules to get better performance:
1. Keep the model parallism size as small as possible. 
    - For the large language models, model parallism is often required to prevent OOM, but it will bring communication overhead and hurt performance. 
    - With distributed optimizer, master weights and optimizer states will be sharded across all DP ranks with slight communication overhead.
    So try to reduce the model parallism size and increase data parallism size when there are lots of free GPU memory during training.
2. Ensure the EPxTP communication winthin the NVLink domain.
    - Communications of EP and TP should remain within the NVLink domain as much as possible, as both are communication-intensive.
    - If the model is too large and requires scaling across multiple nodes, consider PP before TP and EP. See item 3 for details.
3. Use Pipeline Parallelism to scale the model further.
    - Enable Virtual Pipeline Parallelism(VPP) to reduce pp bubbles when PP_size >= 2 by setting `num_layers_per_virtual_pipeline_stage`.
    - VPP_size tuning: the legal values of vpp_size are all common divisors of num_layers/pp_size, E.g., num_layers=24, pp_size=4, then we can pick vpp_size from {1, 2, 3, 6}. The larger the vpp_size, the lower the pipeline bubbles, while the larger number of P2P communications between each PP stages. Empirically a value in the middle often gives the best trade-off. `VPP_size=num_layers / PP_size / num_layers_per_virtual_pipeline_stage`
4. Prefer EP over TP for the expert layer when possible:
    - TP saves more memory than EP, but EP can achieve better GEMM efficiency and less communication overhead than TP.
    - If EP size increased to the number of expert, the local token permutation/un-permutation for experts computation are omitted.
    - Simplify the computation graph of MoE layers, more convenient for performing potential comm-computation overlapping.
    - In practice, EP8TP1 is better than EP4TP2 for 8x7B.
5. Enable Context Parallelism for long context training.
    - The efficiency of CP largely depends on whether its communication can be overlapped with computation. 
    - Empirically, use CP when sequence length >= 8K.

### Step 3: Enable Performance Features based on your profiling bottlenecks

#### Memory Optimization

#### Communication Optimization
- **1F1B Overlap**: `--overlap-moe-expert-parallel-comm`
- **Shared Expert Overlap**: `--moe-shared-expert-overlap`
- **DeepEP**: `--moe-token-dispatcher-type flex --moe-enable-deepep`

#### Compute Optimization
- **Router Fusion**: `--moe-router-fusion`
- **GroupedGEMM**: `--moe-grouped-gemm`
- **Permute Fusion**: `--moe-permute-fusion`

#### Memory Optimization
- **Precision-aware Optimizer**: Automatic with distributed optimizer
- **Selective Recomputation**: `--recompute-granularity selective`
- **FP8 Training**: `--fp8 --moe-router-padding-for-fp8`

## Feature Documentation

### Router and Load Balancing

#### Router Types
- **Top-K Router**: Standard routing with configurable K
- **Group-limited Routing**: Node/device-aware routing
- **Aux-loss-free**: Dynamic bias-based load balancing

### Token Dispatching

#### Dispatcher Types
- **allgather**: Best for TP-only or large Top-K
- **alltoall**: Recommended for EP > 1
- **flex**: Advanced dispatcher with DeepEP support

#### Token Drop Policies
```bash
# Dropless (default, recommended)
# No additional flags needed

# With capacity factor
--moe-expert-capacity-factor 1.0
--moe-pad-expert-input-to-capacity  # Optional padding
```

#### Load Balancing Strategies
```bash
# Auxiliary loss (recommended for most cases)
--moe-router-load-balancing-type aux_loss
--moe-aux-loss-coeff 1e-2

# Sinkhorn algorithm
--moe-router-load-balancing-type sinkhorn

# Aux-loss-free (DeepSeek-V3 style)
--moe-router-load-balancing-type none
--moe-router-enable-expert-bias
--moe-router-bias-update-rate 1e-3
```

### Selective Computation

### Precision-aware Optimizer

The parallelism patterns of the shared experts follow the settings of the dense part, i.e., the attention module. The shared experts are not distributed but replicated in EP ranks.

We also have an experimental feature that tries to overlap the communications and computations in the shared experts and the dispatcher.
We can set `--moe-shared-expert-overlap` and use `alltoall` dispatcher to enable it.
The overlapping relies on the envirionment setting `CUDA_DEVICE_MAX_CONNECTIONS=1`.
The `AllGather` and `ReduceScatter` communications in the shared experts are overlapped with `permute`/`unpermute` in the dispatcher.
The `MLP` computation part in the shared experts are overlapped with the `AlltoAll` communications in the dispatcher.
Both the forward and the backward pass can overlap. But to get the overlapping in the backward pass, the PyTorch version should `>= 2.2.0`.

### Checkpointing
A new output-discarding checkpointing method is also supported. This method discards the output memory of certain submodules during the forward pass and recomputes them during the backward pass, which can save memory compared to standard checkpointing. This can be enabled for specific submodules using the `--recompute-granularity selective --recompute-modules [submodule1, submodule2, ...]` argument. The supported submodules are:

* `moe_act`: Recompute the GroupedMLP activation function.
* `layernorm`: Recompute the input_layernorm and pre_mlp_layernorm (when they are not `IdentityOp`).
* `mla_up_proj`: Recompute the MLA up projection and RoPE applying parts.
* `core_attn`: Recompute the core attention submodule (uses standard checkpointing rather than output-discarding).
* `mlp`: Recompute the dense MLP submodule (uses standard checkpointing rather than output-discarding) which is useful for hybrid-models like DeepSeek-V3.
* `moe`: Recompute the MoE layer submodule (uses standard checkpointing rather than output-discarding).

### Upcycling
Use `--moe-use-upcycling` to enable upcycling, which loads the dense model from the `--load` directory, converts it to an MoE model at runtime, and starts training. The converted model is saved to the `--save` path before training begins. Upcycling is built on distributed checkpointing, supporting parallel modes different from existing dense checkpoints, such as arbitrary expert parallelism during upcycling.

In addition to the default upcycling strategy, we also support granular upcycling strategy which is a more state-of-the-art upcycling strategy from [our recent research work](https://arxiv.org/abs/2410.07524). For the default upcycling strategy, we duplicate the existing MLP to multiple experts, with each expert starting from a copy of the MLP. For the granular upcycling strategy, we use `--moe-upcycling-granularity` to specify how many times smaller is the expert hidden size compared with the original dense FFN hidden size. For using granular upcycling strategy, please set `--moe-upcycling-granularity` as a positive integer. If this param is set to 1, it means using the default upcycling strategy.

Note: The MoE model structure is defined through script arguments. All MoE-related arguments (such as `--num-experts`) can be customized; however, other model structure arguments must be consistent with those of the dense model. For granular upcycling strategy, the moe's FFN hidden size should be set as dense FFN hidden size divided by `--moe-upcycling-granularity`.

### Leverage DeepSeek's DeepEP for High-Performance Cross-Node Token Dispatching
- [DeepSeek-DeepEP](https://github.com/deepseek-ai/deepep) provides a highly optimized implementation for MoE token dispatching and combining operations, specifically designed for large-scale MoE training scenarios.
- DeepEP is particularly recommended for training large-scale, fine-grained MoE architectures such as DeepSeek-V3 and other advanced MoE models.
- To enable DeepEP in your training configuration, simply set `--moe-token-dispatcher-type=flex` and `--moe-flex-dispatcher-backend=deepep` in your command line arguments.

### Integrate HybridEP for High-Performance Intra-Node Token Dispatching
- [HybridEP](https://github.com/deepseek-ai/DeepEP/tree/hybrid-ep) is developed by NVIDIA as an optimized solution for large-scale MoE (Mixture of Experts) all-to-all communication. It is designed to leverage NVIDIA GPU hardware capabilities, significantly reducing Streaming Multiprocessor (SM) resource usage.
- HybridEP currently supports intra-node and multi-node NVLink scenarios.
- To enable HybridEP, set `--moe-token-dispatcher-type=flex` and
  `--moe-flex-dispatcher-backend=hybridep` in your command line arguments.

### CUDA Graph Support
CUDA Graph functionality can be enabled through the `--cuda-graph-impl` option. There are two implementations:

1. `--cuda-graph-impl=local`: Captures cuda graphs using the MCore-internal cuda graph manager.
2. `--cuda-graph-impl=transformer_engine`: Captures cuda graphs using the TE `make_graphed_callables()` interface.

To use `--cuda-graph-impl=transformer_engine`, the user should call related methods `TECudaGraphHelper.create_cudagraphs()` and `TECudaGraphHelper.cuda_graph_set_manual_hooks()` in the training script. Please refer to the usage in `megatron/training/training.py`.

For MoE models, certain configurations may prevent CUDA Graph capture of MoE layers. Specifically, when `--moe-expert-capacity-factor` and `--moe-pad-expert-input-to-capacity` are not set, the resulting dynamic shapes make MoE layers uncapturable. In such cases, you can still leverage CUDA Graphs for the attention layers (operations in `TransformerLayer._forward_attention()`) by setting `--cuda-graph-scope=attn`, while leaving the MoE layers (operations in `TransformerLayer._forward_mlp()`) unmodified. See the argument description for more usage of `--cuda-graph-scope`.



### FP8 Training

```bash
# Basic FP8 training
--fp8
--fp8-param-gather

# MoE-specific FP8 optimizations
--moe-router-padding-for-fp8
```

### Fine-grained Activation Offloading (collaborated with rednote)
Offload the input activation at the granularity of modules

**Usage**
```bash
# Enable fine-grained activation offloading
--fine-grained-activation-offloading

# Specify which modules are going to offload its input
# Choices: "attn_norm", "core_attn", "attn_proj", "mlp_norm", "expert_fc1", "moe_act".
--offload-modules expert_fc1
```
For more details, please refer to the ```docs/source/api-guide/fine_grained_activation_offloading.md```

### MoE Related Arguments
| Item | Description |
| --- | --- |
| --num-experts | Number of Experts in MoE (None means no MoE) |
| --expert-model-parallel-size | Degree of expert model parallelism. Default is 1. |
| --moe-ffn-hidden-size | MoE Feed-Forward Network hidden size. Default is None. |

<details>
<summary>View all MoE arguments</summary>

### Core Arguments
| Argument | Description | Default |
|----------|-------------|---------|
| --num-experts | Number of Experts in MoE | None |
| --expert-model-parallel-size | Degree of expert model parallelism | 1 |
| --moe-ffn-hidden-size | MoE FFN hidden size | None |
| --expert-tensor-parallel-size | Expert layer tensor parallelism | Same as TP |

### Router Arguments
| Argument | Description | Default |
|----------|-------------|---------|
| --moe-router-load-balancing-type | Load balancing: aux_loss, sinkhorn, seq_aux_loss, none | aux_loss |
| --moe-router-topk | Number of experts per token | 2 |
| --moe-router-score-function | Score function: softmax, sigmoid | softmax |
| --moe-router-pre-softmax | Softmax before top-k | False |
| --moe-router-num-groups | Groups for group-limited routing | None |
| --moe-router-group-topk | Selected groups in group-limited routing | None |
| --moe-router-enable-expert-bias | Dynamic per-expert bias | False |
| --moe-router-bias-update-rate | Bias update rate | 1e-3 |
| --moe-router-fusion | Enable router fusion | False |
| --moe-router-dtype | Router precision: fp32, fp64 | None |
| --moe-router-padding-for-fp8 | Pad for FP8 alignment | False |

### Loss and Regularization
| Argument | Description | Default |
|----------|-------------|---------|
| --moe-aux-loss-coeff | Auxiliary loss coefficient | 0.0 |
| --moe-z-loss-coeff | Z-loss coefficient | None |
| --moe-input-jitter-eps | Input jitter epsilon | None |

### Token Dispatching
| Argument | Description | Default |
|----------|-------------|---------|
| --moe-token-dispatcher-type | Dispatcher: allgather, alltoall, flex | allgather |
| --moe-enable-deepep | Enable DeepEP (with flex) | False |
| --moe-expert-capacity-factor | Capacity factor | None |
| --moe-pad-expert-input-to-capacity | Pad to capacity | False |
| --moe-token-drop-policy | Drop policy: probs, position | probs |
| --moe-permute-fusion | Fuse permutation ops | False |

### Performance Optimization
| Argument | Description | Default |
|----------|-------------|---------|
| --moe-grouped-gemm | Use GroupedGEMM | False |
| --overlap-moe-expert-parallel-comm | Batch-level EP overlap | False |
| --delay-wgrad-compute | Split dgrad/wgrad compute | False |
| --moe-shared-expert-intermediate-size | Shared expert FFN size | None |
| --moe-shared-expert-overlap | Overlap shared expert | False |

### Memory and Checkpointing
| Argument | Description | Default |
|----------|-------------|---------|
| --moe-layer-recompute | Recompute MoE layer | False |
| --moe-use-upcycling | Enable upcycling | False |
| --moe-upcycling-granularity | Upcycling granularity | 1 |

### Miscellaneous
| Argument | Description | Default |
|----------|-------------|---------|
| --moe-layer-freq | MoE layer frequency pattern | 1 |
| --moe-per-layer-logging | Per-layer logging | False |
| --moe-router-force-load-balancing | Force load balancing (experimental) | False |

</details>

## Examples

### Training Script Example

<details>
<summary>Complete Mixtral 8x7B training script</summary>

```bash
#!/bin/bash

# Runs Mixtral 8x7B model on 32 H100/A100 GPUs

export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=8
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"6000"}
NNODES=${NNODES:-"4"}
NODE_RANK=${RANK:-"0"}
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

CHECKPOINT_PATH=$1
TOKENIZER_MODEL=$2
DATA_PATH=$3

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NNODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

MODEL_ARGS=(
    --disable-bias-linear
    --seq-length 4096
    --max-position-embeddings 32768
    --num-layers 32
    --hidden-size 4096
    --ffn-hidden-size 14336
    --num-attention-heads 32
    --init-method-std 0.01
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --normalization RMSNorm
    --position-embedding-type rope
    --swiglu
    --untie-embeddings-and-output-weights
    --group-query-attention
    --num-query-groups 8
    --no-masked-softmax-fusion
    --no-position-embedding
)

MOE_ARGS=(
    --num-experts 8
    --expert-model-parallel-size 8
    --moe-router-load-balancing-type aux_loss
    --moe-router-topk 2
    --moe-aux-loss-coeff 1e-2
    --moe-grouped-gemm
    --moe-permute-fusion
    --moe-token-dispatcher-type alltoall
)

DATA_ARGS=(
    --tokenizer-type Llama2Tokenizer
    --tokenizer-model ${TOKENIZER_MODEL}
    --data-path $DATA_PATH
    --split 99990,8,2
)

TRAINING_ARGS=(
    --micro-batch-size 1
    --global-batch-size 128
    --lr 1e-4
    --train-iters 500000
    --lr-decay-iters 320000
    --lr-decay-style cosine
    --min-lr 1.0e-5
    --weight-decay 0.1
    --lr-warmup-iters 500
    --clip-grad 1.0
    --bf16
    --overlap-grad-reduce
    --overlap-param-gather
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 4
    --num-layers-per-virtual-pipeline-stage 8
    --sequence-parallel
    --use-distributed-optimizer
)

LOGGING_ARGS=(
    --log-interval 1
    --save-interval 10000
    --eval-interval 1000
    --eval-iters 10
    --save $CHECKPOINT_PATH
    --load $CHECKPOINT_PATH
    --tensorboard-dir "${CHECKPOINT_PATH}/tensorboard"
    --ckpt-format torch_dist
    --auto-detect-ckpt-format
)

torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py \
    ${MODEL_ARGS[@]} \
    ${MOE_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${LOGGING_ARGS[@]}
```

</details>

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](../../../../CONTRIBUTING.md) for guidelines.

## Support

- GitHub Issues: [Report bugs or request features](https://github.com/NVIDIA/Megatron-LM/issues)
- Documentation: [Full documentation](https://docs.nvidia.com/megatron-core/developer-guide/latest/index.html)


## Citation

If you use Megatron-Core MoE in your research, please cite:

```bibtex
@article{megatron-lm,
  title={Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism},
  author={Shoeybi, Mohammad and Patwary, Mostofa and Puri, Raul and LeGresley, Patrick and Casper, Jared and Catanzaro, Bryan},
  journal={arXiv preprint arXiv:1909.08053},
  year={2019}
}

#### Advantages of MoE Parallel Folding
1. The CP and EP group are folded together by defualt, such that:
    1. It reduces the minimal required GPUs to turn on both CP and EP. For example, the traditional way with (CP=8, EP=8) needs at least 64 GPUs, for now it only requires 8 GPUs.
    2. The CP and EP communication can be both put in the NVLink domain.
2. We can set different TP sizes for Attention and MoE part.
    1. For MoE, EP is often more efficient than TP. But in the traditional way, only using EP can get OOM for most models.
    2. With MoE parallel folding, we can turn on TP for Attention part and setting TP=1 for MoE models, which often gets better MFU.

### End-to-End Training Practice
**Use the latest NVIDIA PyTorch or NeMo Docker Image**
- [NGC PyTorch Image](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch)
- [NGC NeMo Image](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo)

**Token Dispatcher Choices**
- Token Dispatcher sends tokens to the designated expert, involves tensor rearangement and communications.
- Dispatcher `allgather` is the default option. It achieves better performance and efficiency when only tensor parallelism is used or when the Top-k value is very large.
- Dispatcher `alltoall` is recommended if expert parallelism is applied.
- Dispatcher `flex` is a new dispatcher decouples communication group from model parallelism. It supports two backends(DeepEP and HybridEP) selectable via `--moe-flex-dispatcher-backend`.

**Enable Communication Overlap**
- Enable `--overlap-param-gather` and `--overlap-grad-reduce` with distributed optimizer.
- Enable `--tp-comm-overlap` when TP>1.
- Enable p2p comm overlap when PP > 1 by setting `num_layers_per_virtual_pipeline_stage`.

**Enable GroupedGEMM when num_local_experts>1 with `--moe-grouped-gemm`**
- GroupedGEMM has higher efficiency than vanilla sequential GEMMs for each expert.
- Recommend to use the TE version of Grouped GEMM (by upgrading to MCore v0.8 and TE v1.9), which support Gradient Accumulation Fusion and FP8 Training.

**OOM Caused by Token Distribution Imbalance when Training From Scratch**  
MoE suffers from a severe load imbalance issue when the router is under-trained, leading to the model easily running out of memory (OOM), which typically occurs in the first 100~300 steps when training from scratch. 
Therefore, there are two recommended ways during the first 200 steps to avoid the OOM problem, which can be removed after the token distribution is more stable:
1. Increase the `expert-tensor-parallel-size` and decrease `expert-model-parallel-size` to replace EP with TP in MoELayer, this can prevent the load imbalancing between EP ranks. Since current ETP implementation has some memeory overhead, you can further enable activation recomputation only for MoE Layer by adding `--moe-layer-recompute`.
2. Setting capacity factor to a relatively small number like 1.0 by adding `--moe-token-capacity-factor 1.0`.

**Leverage DeepSeek's DeepEP for High-Performance Cross-Node Token Dispatching**
- The primary advantage of DeepEP is its cross-node token communication efficiency, which delivers substantial performance improvements when deploying expert parallelism across multiple nodes with large TopK values.
- To enable DeepEP in your training configuration, simply set `--moe-token-dispatcher-type=flex` and `--moe-enable-deepep` in your command line arguments.

**FP8 Training Best Practice**
- Using latest version of [TransformerEngine](https://github.com/NVIDIA/TransformerEngine).
- Enable router padding with `--moe-router-padding-for-quantization` to reduce padding overhead.
- Enable native FP8 weights with `--fp8-param-gather` to reduce weights memory cost.

### Reference Best Parallel Mapping

Here are the reference parallel mappings of MCore v0.8 for Mixtral 8x7B and 8x22B models:
|        Model            | Vocab Size| Dispatcher | Precision | #GPUs | SEQ LEN | TP | EP | PP | VP | MBS | GBS |
|:-----------------------:|:---------:|:----------:|:---------:|:-----:|:-------:|:--:|:--:|:--:|:--:|:---:|:---:|
| Mixtral 8x7B(Dropless)  |   32K     | All-to-All | BF16      | 64    | 4096    | 1  | 8  | 4  | 8  | 1   | 256 |
| Mixtral 8x22B(Dropless) |   32K     | All-to-All | BF16      | 128   | 4096    | 4  | 2  | 8  | 7  | 1   | 256 |

Detailed Benchmark Information:  
Server:
- 8xH100 80GB HBM3 
- NVLink 4th Generation
- InfiniBand 8x400 Gbit/s

Docker Image:
- PyTorch 24.09 with TransformerEngine v1.11
