# Megatron Core MoE

Megatron Core MoE is a production-ready framework for training large-scale Mixture-of-Experts models, providing the foundational architecture, performance optimizations, and best practices that guide MoE framework development across the industry.

## What's New
For latest features and architectures, please refer to the [MCore dev roadmap](https://github.com/NVIDIA/Megatron-LM/issues/1729).

### 🔥 [MCore dev] (2026/01)
- 🚀 Pipeline-aware fine-grained activation offloading
- 🚀 Qwen3-Next model support
- 🚀 DeepSeek-V3.2 model support
- 🚀 Muon and Layer-wise distributed optimizer
- 🚀 CUDA Graph support with fine-grained scopes

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

## Best Practices to achieve high performance on MoE training

Distributed training involves complex trade-offs between **communication**, **memory**, and **computation**, making it challenging to find an optimal parallelism configuration. This section provides a systematic workflow to help you identify the best parallel mapping for your model and hardware.

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

The optimal parallelism configuration varies based on **model architecture**, **sequence length**, and **hardware platform**. Below are general guidelines to help you achieve high throughput.

#### Guideline 1: Minimize Model Parallelism, Maximize Data Parallelism

| Aspect | Recommendation |
|--------|----------------|
| **Goal** | Keep TP/EP/PP as small as possible while avoiding OOM |
| **Why** | Model parallelism introduces communication overhead that hurts performance |
| **How** | Use distributed optimizer (`--use-distributed-optimizer`) to shard optimizer states across DP ranks, freeing memory for larger DP size |

#### Guideline 2: Keep EP and TP Communication Within NVLink Domain

| Aspect | Recommendation |
|--------|----------------|
| **Goal** | Ensure EP×TP fits within a single node (typically 8 GPUs) |
| **Why** | EP and TP are communication-intensive; NVLink provides much higher bandwidth than cross-node interconnects |
| **Scaling** | When scaling beyond one node, prefer PP over expanding TP/EP across nodes |

**Note:**
For very large MoE models like DeepSeek-V3, the EP communication may exceed the NVLink bandwidth. In this case, consider using 1F1B A2A Overlap to overlap the EP communication.

#### Guideline 3: Use Pipeline Parallelism (PP) for Multi-Node Scaling

| Aspect | Recommendation |
|--------|----------------|
| **Goal** | Use PP to distribute layers across nodes while keeping EP×TP within NVLink |
| **VPP** | Enable Virtual Pipeline Parallelism to reduce pipeline bubbles when `PP ≥ 2` |
| **Config** | Set `--num-layers-per-virtual-pipeline-stage` to control VPP size |

**VPP Size Tuning:**
- Valid values: all divisors of `num_layers / PP_size`
- Example: `num_layers=24, PP=4` → valid VPP sizes: `{1, 2, 3, 6}`
- Trade-off: Larger VPP = fewer bubbles but more P2P communications
- Recommendation: A middle value often gives the best balance

#### Guideline 4: Prefer EP over TP for Expert Layers

| EP Advantages | Details |
|---------------|---------|
| **Better GEMM efficiency** | Larger local matrix sizes improve GPU utilization |
| **Lower communication** | EP has less communication overhead than TP for MoE layers |
| **Simpler computation graph** | Easier to overlap communication with computation |
| **Token permutation** | When `EP = num_experts`, local token permutation is eliminated |

**Example:** For Mixtral 8x7B, `EP8×TP1` outperforms `EP4×TP2`.

#### Guideline 5: Enable Context Parallelism (CP) for Long Sequences

| Aspect | Recommendation |
|--------|----------------|
| **When to use** | Sequence length ≥ 8K tokens |
| **Key factor** | CP efficiency depends on overlapping communication with computation |
| **Config** | Set `--context-parallel-size` to partition sequences across GPUs |

### Step 3: Enable Performance Features Based on Profiling Bottlenecks

After establishing a working parallel configuration, profile your training to identify bottlenecks and apply targeted optimizations.

#### Memory Bottleneck

**Symptom**: Forced to use full recomputation or excessively large parallelism degrees to avoid OOM.

**Solutions**:
| Optimization | Overhead | Config | Reference |
|--------------|----------|--------|---------|
| Selective Recomputation | Low | `--recompute-granularity selective --recompute-modules ...` | [Fine-grained Recomputation](#fine-grained-recomputation) |
| Activation Offloading | Medium | `--fine-grained-activation-offloading --offload-modules ...` | [Fine-grained Activation Offloading](#fine-grained-activation-offloading) |
| Optimizer Offloading | Medium | `--optimizer-cpu-offload` | --- |

#### Communication Bottleneck

**Symptom**: Profiling shows significant time spent in collective operations.

**Solutions**: Identify which communication is the bottleneck and enable corresponding overlap:
| Communication Type | Overlap Config |
|--------------------|----------------|
| DP gradient reduce | `--overlap-grad-reduce` |
| DP param gather    | `--overlap-param-gather` |
| TP communication   | `--tp-comm-overlap` |
| EP All-to-All      | `--overlap-moe-expert-parallel-comm --delay-wgrad-compute` |
| PP send/recv       | Enable VPP with `--num-layers-per-virtual-pipeline-stage` |

#### CPU Overhead Bottleneck

**Symptom**: Nsight Systems timeline shows gaps between GPU kernels where CPU cannot launch kernels fast enough.

**Solutions**:
| Optimization | Config |
|--------------|--------|
| Disable Python GC | `--manual-gc --manual-gc-interval 100` |
| Enable CUDA Graphs | `--cuda-graph-impl transformer_engine --cuda-graph-scope attn moe_router moe_preprocess` |
| Reduce kernel launches | Decrease TP size or increase micro-batch size |

#### Computation Bottleneck

**Symptom**: GPU utilization is low despite no communication or CPU bottlenecks.

**Solutions**:
| Optimization | Config |
|--------------|--------|
| Enable kernel fusions | `--moe-router-fusion --moe-grouped-gemm --moe-permute-fusion` |
| Use FP8 precision | `--fp8-format e4m3 --fp8-recipe blockwise` |


## Feature Documentation

### Router and Load Balancing

Routers determine which expert(s) handle each token. A lightweight MLP scores every token and applies `softmax` or `sigmoid` to compute routing probabilities. The router then selects the top-K experts for each token.

> **Note**: The router logits is better to remain in **FP32** or **FP64** rather than BF16 by --moe-router-dtype fp32. At high expert counts, FP32 precision yields better accuracy because output hidden states of experts are multiplied by router scores and accumulated to get the final output.

#### Router Types

| Router Types | Description | Config |
|-------------|-------------|----------|
| **Top-K Router** | Standard routing with configurable K, uses softmax for probability computation | --moe-router-topk 8 |
| **Group Top-K Router** | Selects top-K expert groups, then routes experts in selected groups | --moe-router-num-groups 8 --moe-router-group-topk 4 |
| **Router score function** | Score function to calculate the probs from output logits of router | --moe-router-score-function softmax/sigmoid |

#### Load Balancing Strategies

| Strategy | Description | Config |
|----------|-------------|--------|
| **aux_loss** | Auxiliary loss for balancing expert usage on a micro-batch | `--moe-router-load-balancing-type aux_loss` |
| **seq_aux_loss** | Sequence-level auxiliary loss for balancing expert usage on each sequence| `--moe-router-load-balancing-type seq_aux_loss` |
| **global_aux_loss** | Global auxiliary loss for balancing expert usage on a global batch across all ranks | `--moe-router-load-balancing-type global_aux_loss` |
| **sinkhorn** | Optimal transport formulation for balancing expert usage | `--moe-router-load-balancing-type sinkhorn` |
| **aux loss free** | Dynamic bias-based load balancing strategy without auxiliary loss | `--moe-router-enable-expert-bias --moe-router-bias-update-rate 1e-3`|
| **none** | No load balancing | `--moe-router-load-balancing-type none` |

### Token Dispatching

After routing, tokens are **dispatched** to the GPU hosting the assigned expert. After expert computation, tokens are sent back and **combined** to restore the original sequence.

| Dispatcher | Description | Best For | Config |
|------------|-------------|----------|--------|
| **alltoall** | NCCL-based All-to-All communication for token exchange | Standard EP > 1 setups | `--moe-token-dispatcher-type alltoall` |
| **FlexDispatcher with [DeepEP](https://github.com/deepseek-ai/DeepEP) backend** | Removes redundant tokens during cross-node communication, fuses intra/inter-node communication into single kernel | Cross-node EP, fine-grained MoE (DeepSeek-V3) | `--moe-token-dispatcher-type flex --moe-flex-dispatcher-backend deepep` |
| **FlexDispatcher with [HybridEP](https://github.com/deepseek-ai/DeepEP/tree/hybrid-ep) backend** | NVIDIA's optimized dispatcher using TMA and IBGDA, fewer SMs, native MNNVL support | GB200 NVL72, Multi-Node NVLink | `--moe-token-dispatcher-type flex --moe-flex-dispatcher-backend hybridep` |
| **allgather** | Gathers all tokens to each GPU, no inter-GPU token movement | TP-only setups, small EP, large Top-K | `--moe-token-dispatcher-type allgather` |

### Upcycling
Use `--moe-use-upcycling` to enable upcycling, which loads the dense model from the `--load` directory, converts it to an MoE model at runtime, and starts training. The converted model is saved to the `--save` path before training begins. Upcycling is built on distributed checkpointing, supporting parallel modes different from existing dense checkpoints, such as arbitrary expert parallelism during upcycling.

In addition to the default upcycling strategy, we also support granular upcycling strategy which is a more state-of-the-art upcycling strategy from [our recent research work](https://arxiv.org/abs/2410.07524). For the default upcycling strategy, we duplicate the existing MLP to multiple experts, with each expert starting from a copy of the MLP. For the granular upcycling strategy, we use `--moe-upcycling-granularity` to specify how many times smaller is the expert hidden size compared with the original dense FFN hidden size. For using granular upcycling strategy, please set `--moe-upcycling-granularity` as a positive integer. If this param is set to 1, it means using the default upcycling strategy.

Note: The MoE model structure is defined through script arguments. All MoE-related arguments (such as `--num-experts`) can be customized; however, other model structure arguments must be consistent with those of the dense model. For granular upcycling strategy, the moe's FFN hidden size should be set as dense FFN hidden size divided by `--moe-upcycling-granularity`.

## Training Optimizations
MoE training faces three fundamental performance bottlenecks: **Memory Wall**, **Communication Wall**, and **Compute Efficiency Wall**. The following optimizations address each of these challenges.

### MoE Parallel Folding
**The Problem with Traditional Approaches:**
- Prior MoE frameworks constrain **EP ≤ DP** (Expert Parallelism must be a sub-group of Data Parallelism), which severely limits scalability.
- Applying the same TP/CP to both attention and MoE is suboptimal:
  - High TP benefits attention but hurts MoE (small per-expert dims make TP overhead prohibitive)
  - High CP benefits long-context attention but is unnecessary for MoE (tokens processed independently)

**MoE Parallel Folding** is Megatron Core's solution that **decouples attention and MoE parallelism**:

| Parallelism Group | Attention Layers | MoE Layers |
|-------------------|------------------|------------|
| **Dimensions** | TP × CP × DP × PP | ETP × EP × EDP × PP |

#### Key Benefits

1. **Breaks the EP ≤ DP Constraint**
   - Traditional: TP=4, CP=2, DP=8, PP=4 → max EP=8
   - With Folding: Same attention config, but MoE uses ETP=1, EP=64, EDP=1 → 8× more expert parallelism

2. **Reduces Minimum GPU Requirements**
   - Traditional CP=8, EP=8 requires at least 64 GPUs
   - With Folding: CP and EP are folded together, only 8 GPUs needed

3. **Enables Independent Optimization**
   - Use high TP for attention (memory efficiency)
   - Use ETP=1 for MoE (better GEMM efficiency, less communication)

4. **Keeps High-Bandwidth Communication in NVLink Domain**
   - Both CP and EP communication can remain within NVLink domain

> **Reference**: [MoE Parallel Folding: Heterogeneous Parallelism Mappings for Efficient Large-Scale MoE Model Training](https://arxiv.org/abs/2504.14960)

### Memory Optimization

Memory optimization is critical for large-scale MoE training, as MoE models maintain all expert parameters even though only a subset is activated per token.

| Optimization | Description | Config |
|--------------|-------------|--------|
| **Fine-grained Recomputation** | Selectively recomputes specific modules (e.g., `mla_up_proj`, `layernorm`, `moe_act`) instead of full layers | `--recompute-granularity selective --recompute-modules mla_up_proj layernorm moe_act` |
| **Fine-grained Activation Offloading** | Offloads activations to CPU memory, overlapping D2H/H2D transfers with computation | See `docs/source/api-guide/fine_grained_activation_offloading.md` |
| **Precision-aware Optimizer** | Stores optimizer states (exp_avg, exp_avg_sq) in BF16 instead of FP32, reducing optimizer memory by 50% | `--use-precision-aware-optimizer --exp-avg-dtype bf16 --exp-avg-sq-dtype bf16` |
| **Optimizer Offloading** | Offloads optimizer states to CPU memory. | `--optimizer-cpu-offload` |

#### Fine-grained Recomputation
A new output-discarding checkpointing method is also supported. This method discards the output memory of certain submodules during the forward pass and recomputes them during the backward pass, which can save memory compared to standard checkpointing. This can be enabled for specific submodules using the `--recompute-granularity selective --recompute-modules [submodule1, submodule2, ...]` argument. The supported submodules are:

* `moe_act`: Recompute the GroupedMLP activation function.
* `layernorm`: Recompute the input_layernorm and pre_mlp_layernorm (when they are not `IdentityOp`).
* `mla_up_proj`: Recompute the MLA up projection and RoPE applying parts.
* `core_attn`: Recompute the core attention submodule (uses standard checkpointing rather than output-discarding).
* `mlp`: Recompute the dense MLP submodule (uses standard checkpointing rather than output-discarding) which is useful for hybrid-models like DeepSeek-V3.
* `moe`: Recompute the MoE layer submodule (uses standard checkpointing rather than output-discarding).

#### Fine-grained Activation Offloading

Unlike recomputation (which trades compute for memory), offloading trades **GPU-CPU bandwidth for memory**: activations are transferred to CPU during forward pass and retrieved during backward pass. The key is hiding transfer latency behind computation using asynchronous D2H/H2D transfers.

**Key Features:**
- **Module-level granularity**: Target specific modules rather than entire layers
- **Computation-offloading overlap**: Asynchronous transfers via independent CUDA streams
- **Compatible with PP/VPP**: Works with pipeline parallelism and fine-grained recomputation

**Usage**
```bash
--fine-grained-activation-offloading
--offload-modules expert_fc1 moe_act # Choices: attn_norm, core_attn, attn_proj, mlp_norm, expert_fc1, moe_act
```

For more details, see `docs/source/api-guide/fine_grained_activation_offloading.md`

### Communication Optimization

Distributed training introduces communication overhead from various parallelism strategies. Megatron Core supports overlapping communication with computation to hide latency and improve throughput.

#### Data Parallel (DP) Communication Overlap

With distributed optimizer, DP introduces **reduce-scatter** (gradients) and **all-gather** (parameters) communications, chunked by Transformer layer granularity.

| Optimization | Description | Config |
|--------------|-------------|--------|
| **Gradient Reduce Overlap** | Overlaps gradient reduce-scatter with backward computation | `--overlap-grad-reduce` |
| **Param Gather Overlap** | Overlaps parameter all-gather with forward computation | `--overlap-param-gather` |
| **BF16 Gradient Reduce** | Reduces gradients in BF16 instead of FP32 for better performance | `--grad-reduce-in-fp32 false` (via mixed precision config) |
| **FP8 Param Gather** | Conducts parameter all-gather in FP8, reducing overhead by 50% | `--fp8-param-gather` |

#### Tensor Parallel (TP) Communication Overlap

TP with sequence parallelism introduces activation all-gather and reduce-scatter operations. Communications are overlapped in **bulk** (no dependency) or **pipelined** (with dependency) fashion.

| Optimization | Description | Config |
|--------------|-------------|--------|
| **TP Comm Overlap** | Enables bulk and pipelined TP communication overlap | `--tp-comm-overlap` |

> **Requirements**: `tensor_model_parallel_size >= 2` and `--sequence-parallel`

#### Pipeline Parallel (PP) Communication Overlap

PP introduces P2P activation sends/receives between pipeline stages. Overlap is automatic in the 1F1B pipelining phase when VPP is enabled.

| Optimization | Description | Config |
|--------------|-------------|--------|
| **P2P Comm Overlap** | Overlaps PP P2P communications with non-dependent computations | `--overlap-p2p-comm` (auto-enabled with VPP) |
| **VPP for Better Overlap** | Increases overlap opportunities by reducing layers per virtual stage | `--num-layers-per-virtual-pipeline-stage` |

#### Expert Parallel (EP) Communication Overlap

EP All-to-All can consume 30-40% of training time without optimization. These features hide or reduce EP communication overhead.

| Optimization | Description | Config |
|--------------|-------------|--------|
| **EP A2A Overlap** | Overlaps All-to-All with computation by merging FWD-BWD passes of adjacent microbatches | `--overlap-moe-expert-parallel-comm --delay-wgrad-compute` |
| **Shared Expert Overlap** | Runs shared expert computation concurrently with EP token transfer | `--moe-shared-expert-overlap` |

> **Requirements for EP A2A Overlap**: `expert_model_parallel_size > 1`, CUDA_DEVICE_MAX_CONNECTIONS > 1.

### Compute Optimization

Fine-grained MoE produces many small operations that can underutilize GPU resources. These optimizations reduce kernel launch overhead and improve GPU utilization.

| Optimization | Description | Config |
|--------------|-------------|--------|
| **Grouped GEMM** | Batches multiple expert GEMM operations into a single kernel call, improving GPU utilization | `--moe-grouped-gemm` |
| **Router Fusion** | Fuses router projection, top-k selection, softmax, and auxiliary loss into fewer kernels | `--moe-router-fusion` |
| **Permute Fusion** | Fuses token permutation/unpermutation operations into optimized single kernels | `--moe-permute-fusion` |
| **FP8 Training** | Uses FP8 Tensor Core operations for faster GEMMs on Hopper/Blackwell GPUs | `--fp8 --fp8-recipe blockwise` |


### FP8 Training

FP8 training provides benefits across all three performance walls:

| Wall | FP8 Benefit | Impact |
|------|-------------|--------|
| **Memory** | 50% activation reduction | Stores linear layer inputs in FP8 instead of BF16 |
| **Memory** | Eliminate BF16 weight copies | Native FP8 casts directly from FP32 to FP8 |
| **Communication** | 50% EP dispatch volume | Dispatches tokens in FP8 instead of BF16 |
| **Communication** | 50% parameter all-gather | With FP8 primary weights (except MXFP8) |
| **Compute** | Faster Tensor Core GEMMs | FP8 ops on Hopper/Blackwell are faster than BF16 |

#### FP8 Recipes

| Recipe | Scaling Granularity | Format | Platform | Use Case |
|--------|---------------------|--------|----------|----------|
| **Per-tensor** | Whole tensor | E4M3/E5M2 hybrid | Hopper, Blackwell | Conservative, initial experimentation |
| **Blockwise** | 1×128 (activations), 128×128 (weights) | E4M3 | Hopper | **Production-proven** (DeepSeek-V3, Minimax-M2) |
| **MXFP8** | 1×32 | E4M3 + E8M0 scaling | Blackwell | Native hardware support on GB200 |

> **Recommendation**: Use **blockwise FP8** on Hopper for production training. It has been validated at scale on DeepSeek-V3 class models.

#### MoE-Specific FP8 Optimizations

| Optimization | Description | Config |
|--------------|-------------|--------|
| **Routing Map Padding** | Pads routing map (not tokens) to align M dimension to 16/32, avoiding per-tensor padding overhead | `--moe-router-padding-for-fp8` |
| **FP8 Primary Weights** | Casts FP32 master weights directly to FP8, eliminating BF16 intermediate copy | `--fp8-param-gather` (Need additional `--reuse-grad-buf-for-mxfp8-param-ag` for MXFP8) |


#### Example Configuration

```bash
# Blockwise FP8 on Hopper (recommended for production)
--fp8-format e4m3
--fp8-recipe blockwise
--fp8-param-gather
--moe-router-padding-for-fp8

# MXFP8 on Blackwell
--fp8-format e4m3
--fp8-recipe mxfp8
--moe-router-padding-for-fp8
--fp8-param-gather 
--reuse-grad-buf-for-mxfp8-param-ag
```

> **Note**: For blockwise and MXFP8 recipes with current scaling, training loss curves show negligible difference compared to BF16 baselines.


### CUDA Graph
CUDA Graph functionality can be enabled through the `--cuda-graph-impl` option. There are two implementations:

1. `--cuda-graph-impl=local`: Captures cuda graphs using the MCore-internal cuda graph manager.
2. `--cuda-graph-impl=transformer_engine`: Captures cuda graphs using the TE `make_graphed_callables()` interface.

To use `--cuda-graph-impl=transformer_engine`, the user should call related methods `TECudaGraphHelper.create_cudagraphs()` and `TECudaGraphHelper.cuda_graph_set_manual_hooks()` in the training script. Please refer to the usage in `megatron/training/training.py`.

For MoE models, certain configurations may prevent CUDA Graph capture of MoE layers. Specifically, when `--moe-expert-capacity-factor` and `--moe-pad-expert-input-to-capacity` are not set, the resulting dynamic shapes make MoE layers uncapturable. In such cases, you can still leverage CUDA Graphs for the attention layers (operations in `TransformerLayer._forward_attention()`) by setting `--cuda-graph-scope=attn`, while leaving the MoE layers (operations in `TransformerLayer._forward_mlp()`) unmodified. See the argument description for more usage of `--cuda-graph-scope`.

## MoE Arguments Reference
### Core Arguments
| Argument | Description | Default |
|----------|-------------|---------|
| --num-experts | Number of Experts in MoE | None |
| --expert-model-parallel-size | Degree of expert model parallelism | 1 |
| --moe-ffn-hidden-size | MoE FFN hidden size | FFN hidden size of the dense model |
| --expert-tensor-parallel-size | Expert layer tensor parallelism | Same as TP(Recommeded to set to 1 for fine-grained MoE models) |
| --moe-layer-freq | MoE layer frequency pattern | 1 |

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
| --moe-per-layer-logging | Per-layer logging | False |
| --moe-router-force-load-balancing | Force load balancing (experimental) | False |

## Examples
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

We welcome contributions! Please see [CONTRIBUTING.md](https://github.com/NVIDIA/Megatron-LM/blob/main/CONTRIBUTING.md) for guidelines.

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

@article{moe-parallel-folding,
    title={MoE Parallel Folding: Heterogeneous Parallelism Mappings for Efficient Large-Scale MoE Model Training with Megatron Core}, 
    author={Liu, Dennis and Yan, Zijie and Yao, Xin and Liu, Tong and Korthikanti, Vijay and Wu, Evan and Fan, Shiqing and Deng, Gao and Bai, Hongxiao and Chang, Jianbin and Aithal, Ashwath and Andersch, Michael and Shoeybi, Mohammad and Yao, Jiajie and Zhou, Chandler and Wu, David and Li, Xipeng and Yang, June},
    year={2025},
    journal={arXiv preprint arXiv:2504.14960},
}
```
