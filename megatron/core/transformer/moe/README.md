# Megatron Core MoE

Megatron-Core MoE provides comprehensive parallelism strategies, seamlessly integrating Expert Parallelism with tensor, data, sequence, and pipeline parallelism. With MCore v0.9, we've achieved remarkable performance of **468 TFLOPS** for Mixtral 8X7B bf16 training. Additionally, we support state-of-the-art MoE model architectures including DeepSeek-V3 and Qwen-MoE.

### What's New
- **Support for DeepSeek-V3 architecture**
  - Enable TP for MLA and DeepSeek-V3
  - Support aux-loss-free load balancing strategy
  - Support node-limited routing
  - Support Multi-Token Prediction (MTP)
- **Support DeepSeek's DeepEP for efficient token dispatching and combining**
- Add fusion for token permutation and unpermutation
- Support Uneven virtual pipeline parallel split
- Support output-discarding checkpointing on some submodules

### Parallelism
- **Expert Parallelism**
    - A specific method of parallelism for MoE models, where experts are partitioned onto different workers and each worker processes a different batch of training samples, each worker process one or more experts for each MoE layer.
- **3D Parallelism**: Data Parallelism, Tensor Parallelism, Pipeline Parallelism
    - Note: When using MoE with expert parallelism and tensor parallelism, sequence parallelism must be enabled.
- **Context Parallelism**:
    - Split the sequence dimension to support long context training.
- **Richer parallel mappings**: EP can be combined with DP/TP/PP/CP for handling larger MoE variants.
- **MoE Parallel Folding**: Support for setting different parallelism strategies for Attention and MoE components, enabling more flexible and efficient model sharding. See detailed documentation below.
- **Full distributed optimizer support.**

### Router and Load Balancing
- Router type:
    - Top-K MLP router
- Load Balancing algorithms:
    - Sinkhorn (S-BASE)
    - Aux loss / Load balancing loss
    - Aux-loss-free load balancing strategy

### Performance Optimizations
- (Experimental) **DeepEP** is integrated for efficient token communication in large-scale MoE training.
- GroupedGEMM when num local experts > 1
    - Supported dtype: bf16
    - Performance improvements for larger MoE models
- Enable `--tp-comm-overlap` for MoE
- FP8 training support

### Token Dispatch Mechanism
- Dropless / No token drop
- Token drop, with or without padding to capacity
- Token permutation / Unpermutation fusion

### Ease of use
- Checkpoint converter for Mixtral models, see the [example](https://github.com/NVIDIA/Megatron-LM/tree/main/examples/mixtral) for details.
- MoE Layer Frequency to customize the hybrid MoE/Dense layer architecture
- Distributed checkpoining
- Per-layer logging
- Upcycling Support

## Upcoming features
- TopK Router Fusion
- Multi-token Prediction

# User Guide

## Usage

### Quick Start
To train a top-2 MoE model with 8 experts and auxiliary loss, include the following arguments:

```bash
--num-experts 8
--expert-model-parallel-size 8
--moe-grouped-gemm
--moe-permute-fusion
--moe-router-load-balancing-type aux_loss # options: aux_loss, sinkhorn, none. Default is aux_loss.
--moe-router-topk 2
--moe-aux-loss-coeff 1e-2
--use-distributed-optimizer
--moe-token-dispatcher-type alltoall
```

To enable the token drop mechanism, such as GShard and SwitchTransformer, include the following arguments:

```bash
--moe-expert-capacity-factor 1.0
--moe-pad-expert-input-to-capacity # Optional
```

The following figure illustrates differenting dropping strategies in MCore:
<!-- This image is uncommented for now as Sphinx cannot resolve this path. Sphinx imports this markdown file, and from the imported location this relative path does not exist anymore. Ideally, this markdown should not live here but rather in the `docs/` directory that Sphinx uses. -->
<!-- ![Token Droppling Strategies](../../../../docs/source/images/moe/token_drop.png) -->

1. The default dropless strategy will not drop or pad any token.
2. By setting `--moe-expert-capacity-factor`, the tokens exceed the capacity of expert will be dropped based on their selected probabilities. 
   The dropping is performed before the token exchange operation between EP ranks when EP > 1. 
   The formula of capacity is `capacity = num_tokens_per_rank * topk * capacity_factor / num_experts`.
3. By setting `--moe-pad-expert-input-to-capacity`, the experts with tokens less than capacity will be padded to the capacity.

### Fine-tuning Mixtral Models
Megatron-Core has full support for Mixtral MoE models, and we provide the checkpoint converter for Mixtral models from huggingface format to MCore format. 
<!-- See more details in the [mixtral example](../../../../examples/mixtral/README.md). -->

### Distributed Checkpointing
MCore v0.7 introduced fully parallel and asynchronous saving capabilities to distributed checkpointing, 
which addresses the issues of low efficiency in the traditional checkpoint saving methods. 
It also solved the problem of incompatibility between checkpoints of different parallel mappings in the traditional format.
With the new distributed checkpointing solution, MCore can achieve flexible parallelism configurations by saving and loading the unified format checkpoints.
Compared to native PyTorch solution, MCore achieves up to 50x reduction in checkpointing overhead.

From MCore v0.8, MoE supports Distributed Checkpointing, which means users can save and load with any combination of parallelism and it is currently available, including expert parallel.
1. Loading weight and distributed optimizer states with TPxCPxEPxPP resharding with SequentialMLP is supported in version 0.8.
2. GroupedMLP weight resharding is supported in version 0.8.0 and optimizer state resharding is supported in version 0.10.0. Switching between GroupedMLP/SequentialMLP when loading and saving is partially supported.
3. TEGroupedMLP has fully support on distributed checkpointing and is fully exchangable with SequentialMLP in version 0.9.0.
4. Optimizer state resharding cannot do across EP=1 with EP>1 due to the different optimizer type.

Usage
- `--ckpt-format torch_dist` The main argument, it will attempt to save and load using distributed checkpointing.
- `--auto-detect-ckpt-format` With this, it can load both distributed checkpointing and legacy checkpointing.

Checkpoint compatibility across SequentialMLP, GroupedMLP, and TEGroupedMLP:
```text
    ┌───────────────┐          ┌───────────────┐          ┌───────────────┐     
    │   GroupedMLP  │          │ SequentialMLP │          │ TEGroupedMLP  │     
    │               │          │               │          │               │     
    │               │          │               │          │               │     
    │ ┌───────────┐ │          │ ┌───────────┐ │          │ ┌───────────┐ │     
    │ │legacy ckpt│ │          │ │legacy ckpt│ │          │ │legacy ckpt│ │     
    │ └─────┬─────┘ │          │ └─────┬─────┘ │          │ └─────┬─────┘ │     
    │       ▼       │          │       ▼       │          │       ▼       │     
    │  ┌─────────┐  │          │  ┌─────────┐  │          │  ┌─────────┐  │     
    │  │dist ckpt│  │          │  │dist ckpt│  │          │  │dist ckpt│  │     
┌──►│  │ weight  │  │◄────────►│  │ weight  │  │◄────────►│  │ weight  │  │◄──┐ 
│   │  └─────────┘  │          │  └─────────┘  │          │  └─────────┘  │   │ 
└───┼───────────────┼──────────┼───────────────┼──────────┼───────────────┼───┘ 
    │┌─────────────┐│          │┌─────────────┐│          │┌─────────────┐│     
    ││  dist ckpt  ││          ││  dist ckpt  ││          ││  dist ckpt  ││     
    ││optim states ││          ││optim states ││◄────────►││optim states ││     
    │└─────────────┘│          │└─────────────┘│          │└─────────────┘│     
    └───────────────┘          └───────────────┘          └───────────────┘     
```

Best practices for distributed checkpointing:
1. Convert a legacy checkpoint to a distributed checkpoint. To achieve this, we can add both `--ckpt-format torch_dist --auto-detect-ckpt-format`, then it will load the legacy one and save as the distributed checkpoint format later when the training progress tries to save checkpoints.
2. Convert checkpoint of the legacy GroupedMLP to TEGroupedMLP. This is only supported for the weight parts. To achieve this, we can use the above method to convert the legacy checkpoint to a distributed checkpoint of the legacy GroupedMLP. After updating the libraries and using TEGroupedMLP, we can directly load the previously saved checkpoint by adding argument `--no-load-optim`.

### Shared Experts
MCore v0.9 introduced the shared expert feature. We can enable this feature by setting suitable `--moe-shared-expert-intermediate-size`.

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
- To enable DeepEP in your training configuration, simply set `--moe-token-dispatcher-type=flex` and `--moe-enable-deepep` in your command line arguments.

### CUDA Graph Support
CUDA Graph functionality can be enabled through two options:

1. `--enable-cuda-graph`: Automatically captures graphs during runtime (just-in-time)
2. `--external-cuda-graph`: Requires manual graph capture before runtime (ahead-of-time)

Note: These two options cannot be enabled simultaneously.

For manual capture with `--external-cuda-graph`, refer to the `cuda_graph_capture()` and `cuda_graph_set_manual_hooks()` functions in `megatron/training/training.py`.

For MoE models, certain configurations may prevent CUDA Graph capture of MoE layers. Specifically, when `--moe-expert-capacity-factor` and `--moe-pad-expert-input-to-capacity` are not set, the resulting dynamic shapes make MoE layers uncapturable. In such cases, you can still leverage CUDA Graphs for attention layers by setting `--cuda-graph-scope=attn`, while leaving MoE layers unmodified. Note that the `--cuda-graph-scope` parameter is only applicable when using `--external-cuda-graph` mode.

### MoE Related Arguments
| Item | Description |
| --- | --- |
| --num-experts | Number of Experts in MoE (None means no MoE) |
| --expert-model-parallel-size | Degree of expert model parallelism. Default is 1. |
| --moe-ffn-hidden-size | MoE Feed-Forward Network hidden size. Default is None. |

<details>
<summary> View all MoE related arguments. </summary>

| Item | Description |
| --- | --- |
| --num-experts | Number of Experts in MoE (None means no MoE) |
| --expert-model-parallel-size | Degree of expert model parallelism. Default is 1. |
| --moe-ffn-hidden-size | MoE Feed-Forward Network hidden size. Default is None. |
| --expert-tensor-parallel-size | Degree of tensor model parallelism of expert layer. Default is same to --tensor-model-parallel-size. |
| --moe-layer-freq | Frequency between MoE layers and Dense layers. Accepts either: 1) An integer N for 1:N ratio (one expert layer for every N-1 dense layers), 2) A string "N" for the same ratio, or 3) A string with Python list expression for custom patterns like `([1]*3+[0]*1)*3` which gives [1,1,1,0,1,1,1,0,1,1,1,0] where 1=expert layer and 0=dense layer. Examples: `([0]+[1]*23)` for 1 dense layer followed by 23 experts layers, `([1]*3+[0]*2)*2` for three expert layers followed by two dense layers, repeated twice. Default is 1. |
| --moe-grouped-gemm | When there are multiple experts per rank, launch multiple local GEMM kernels in multiple streams to improve the utilization and performance with GroupedLinear in TransformerEngine. |
| --moe-router-load-balancing-type | Determines the load balancing strategy for the router. "aux_loss" corresponds to the load balancing loss used in GShard and SwitchTransformer; "seq_aux_loss" corresponds to the load balancing loss used in DeepSeekV2 and DeepSeekV3, which computes the loss for each individual sample; "sinkhorn" corresponds to the balancing algorithm used in S-BASE, and "none" implies no load balancing. The default is "aux_loss". |
| --moe-router-dtype | Data type for routing computation and expert output weighted averaging. Options are 'fp32' and 'fp64'. This can improve numerical stability, particularly when using a large number of experts. The throughput/memory impact should be negligible when used with --moe-permute-fusion. Default is None (no dtype promotion). |
| --moe-router-topk | Number of experts to route to for each token. The default is 2. |  
| --moe-router-score-function | Score function for MoE routing. Can be "softmax" or "sigmoid". Default is "softmax". |
| --moe-router-pre-softmax | Enable pre-softmax routing for MoE, which means softmax is before the top-k selection. By default, softmax is done after top-k. |
| --moe-router-num-groups | Number of groups to divide experts into for group-limited routing. When using group-limited routing: 1) Experts are divided into equal-sized groups, 2) For each token, a subset of groups are selected based on routing scores (sum of top-2 expert scores within each group), 3) From these selected groups, moe_router_topk experts are chosen.  Two common use cases: 1) Device-limited routing: Set equal to expert parallel size (EP) to limit each token to experts on a subset of devices (See DeepSeek-V2: https://arxiv.org/pdf/2405.04434) 2) Node-limited routing: Set equal to number of nodes in EP group to limit each token to experts on a subset of nodes (See DeepSeek-V3: https://arxiv.org/pdf/2412.19437)) |
| --moe-router-group-topk | Number of selected groups for group-limited routing. |
| --moe-router-topk-scaling-factor | Scaling factor for routing score in top-k selection, only works when --moe-router-pre-softmax enabled. Defaults to None, which means no scaling. |
| --moe-router-enable-expert-bias | TopK routing with dynamic per-expert bias in the aux-loss-free load balancing strategy. The routing decision is based on the sum of the routing scores and the expert bias. See https://arxiv.org/abs/2408.15664 for details. |
| --moe-router-bias-update-rate | The expert bias is updated based on the number of assigned tokens to each expert in a global batch, where the bias is increased for experts with less assigned tokens and decreased for experts with more assigned tokens. Default is 1e-3 same as that used in DeepSeekV3. |
| --moe-router-force-load-balancing | (Experimental) Force override routing to balance token distribution using random logits for MoE routers, supporting naive top-k and group-limited top-k. This experimental feature is for benchmarking purposes only! |
| --moe-router-padding-for-fp8 | Pad the routing_map to make sure the number of tokens each expert received is a multiple of 16/32 for FP8 precision. It is suggested to enable this for dropless training with FP8 precision when num_local_experts > 1. This is a more efficient way to pad for FP8 which eliminates the explicit padding in the GroupedMLP layer. |
| --moe-aux-loss-coeff | Scaling coefficient for the aux loss: a starting value of 1e-2 is recommended. Default is 0.0. |
| --moe-z-loss-coeff | Scaling coefficient for the z-loss: a starting value of 1e-3 is recommended. Default is None. |
| --moe-input-jitter-eps | Add noise to the input tensor by applying jitter with a specified epsilon value. Default is None. |
| --moe-token-dispatcher-type | Determines the token dispatcher type. Choices are "allgather", "alltoall". Default is "allgather". We recommend using 'alltoall' if expert parallelism is applied. We have upgraded the "alltoall" dispatcher in place during MCore v0.9, while the original implementation renamed as "alltoall_seq" is retained until MCore v0.13.|
| --moe-enable-deepep | (Experimental) Enable DeepSeek/DeepEP for efficient token dispatching and combine in MoE models. Only works with flex token dispatcher by setting --moe-token-dispatcher-type=flex. |
| --moe-per-layer-logging | Enable per-layer logging for MoE, currently supports auxiliary loss and z loss. |
| --moe-expert-capacity-factor | The capacity factor for each expert, None means no token will be dropped. Default is None. |
| --moe-pad-expert-input-to-capacity | Pads the input for each expert to match the expert capacity length, effective only after the --moe-expert-capacity-factor is set. |
| --moe-token-drop-policy | The policy to drop tokens. Can be either "probs" or "position". If "probs", the tokens with the lowest probabilities will be dropped. If "position", tokens at the end of each batch will be dropped. |
| --moe-layer-recompute | Enable activation checkpointing for moe_layer, should be used when memory is not sufficient. |
| --moe-permute-fusion | Fuse token rearrangement ops during token dispatching. |
| --moe-shared-expert-intermediate-size | Set shared expert total ffn hidden size. It should be equal to `num_shared_experts * ffn_size_of_each_shared_expert` if there are multiple shared experts. None means no shared expert. |
| --moe-shared-expert-overlap | (Experimental, may change) If this is set, the communications/computations in the shared experts and the dispatcher will overlap (The `alltoall` dispatcher is needed.) Otherwise, the shared expert runs after the routed experts. |
| --moe-use-upcycling | Load the dense model checkpoint, convert it into an MoE model at runtime and start training. The converted model will be saved to the path specified by `--save` before training begins. Upcycling is implemented on the top of distributed checkpointing, so it supports parallel modes different from the dense model.|
| --pipeline-model-parallel-layout | (Experimental, may change) A string containing a Python list expression that defines a custom pipeline model parallel layout. |
| --moe-upcycling-granularity | This param sepecifics how many times smaller is the expert hidden size compared with the original dense FFN hidden size. For using granular upcycling strategy, please set this param as a positive integer. If this param is set to 1, it means using the default upcycling strategy.|

</details>

## MoE training example:
<details>
<summary>Click here. </summary>

```bash
#!/bin/bash

# Runs Mixtral 8x7B model on 32 H100/A100 GPUs
# The Dropless MoE suffers from an imbalanced token distribution at the early stage of training (the first few hundred iterations), which may lead to poor performance and out-of-memory (OOM) issues.
# To check the performance of a Dropless MoE model, we should run the model for at least 500 iterations or resume from trained checkpoints.

export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"6000"}
NNODES=${NNODES:-"1"}
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
    --moe-router-load-balancing-type aux_loss # options: aux_loss, sinkhorn, None. Default is aux_loss.
    --moe-router-topk 2
    --moe-aux-loss-coeff 1e-2
    --moe-grouped-gemm
    --moe-permute-fusion
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
    --log-interval 1 \
    --save-interval 10000 \
    --eval-interval 1000 \
    --eval-iters 10 \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH \
    --tensorboard-dir "${CHECKPOINT_PATH}/tensorboard" \
    --no-load-optim \
    --no-load-rng
)

if [ -n "${WANDB_API_KEY}" ]; then
    LOGGING_ARGS+=(
        --wandb-project ${WANDB_PROJECT:-"Mixtral-Finetuning"}
        --wandb-exp-name ${WANDB_NAME:-"Mixtral_8x7B"} 
    )
fi

torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py \
    ${MODEL_ARGS[@]} \
    ${MOE_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${LOGGING_ARGS[@]}
```
</details>

# Performance Best Practice

### Tuning Guide of Parallel Mappings

To find a good parallel mapping that help you achieve a high throughput of a new model, there are some general rule that could help. Here is an overview of properties in different aspects for each parallel strategy.

| Parallel Strategy | Peak Activation Memory          | Weight Memory  | Optimizer states                  | Communication (Per-Layer) |
|:-----------------:|:-------------------------------:|:--------------:|:---------------------------------:|:-------------------------:|
| TP                | 1/N (with SP on)                | 1/N            | 1/N                               |        High               |
| EP                | 1                               | 1/N in MoELayer| 1/N                               |       Medium              |
| PP                | 1 (>1 with virtual pipeline)    | 1/N            | 1/N                               |       Medium              |
| CP                | 1/N                             | 1              | 1/N (with distributed optimizer)  |       Medium              |
| DP                | 1                               | 1              | 1/N (with distributed optimizer)  |        Low                |

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

### MoE Parallel Folding

MoE Parallel Folding separates the MoE related parallel groups from Dense groups.
1. Traditional MoE parallel groups are entangled with dense by using a 5-dimension parallel group generator with default order `tp-cp-ep-dp-pp`. The EP group in MoE is a sub-group of DP in Attention.
2. With MoE Parallel Folding, we use a parallel group generator with `tp-cp-dp-pp` for Attention, and another with `tp-ep-dp-pp` for MoE. The EPxTP group in MoE is a sub-group of DPxCPxTP in Attention.

By setting `--expert-tensor-parallel-size`, we can set MoE-specific TP size.

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
- Dispatcher `flex` is a new dispatcher decouples communication group from model parallelism. Currently, only the DeepEP backend is supported for by setting `--moe-enable-deepep`.

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
- Enable router padding with `--moe-router-padding-for-fp8` to reduce padding overhead.
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