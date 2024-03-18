# Megatron Core MoE Key Features

### Parallelism

- **Expert Parallel**
    - A specific method of parallelism for MoE models, where experts are partitioned onto different workers and each worker processes a different batch of training samples, each worker process one or more experts for each MoE layer.
- **3D Parallel**: Data Parallel , Tensor Parallel, Pipeline Parallel, Sequence Parallel
    - Note: When using MoE with expert parallelism and tensor parallelism, sequence parallelism must be used.
- **Richer parallel mappings**: EP can be combined with DP/TP/PP/SP for handling larger MoE variants.
- **Distributed optimizer.**

### Router and Load Balancing

- Router type:
    - Top-K MLP router
    - Expert Choice router (coming soon)
- Load Balancing algorithms:
    - Sinkhorn (S-BASE)
    - Aux loss / Load balancing loss

### Performance Optimizations

- GroupedGEMM when num local experts > 1
    - Supported dtype: bf16

### Token Dispatch Mechanism

- Dropless / No token drop.
- Token drop. (coming soon)

### Ease of use
- Checkpoint converter (coming soon)

## Upcoming features

- Enhanced cutlass GroupedGEMM kernels
    - Reduced host-device syncs.
    - More supported dtype: fp32/bf16/fp16
    - Kernel heuristics tuned for A100/A10/L40S
    - BWD cutlass GroupedGEMM kernels supported
- Token permutation / unpermutation fusion
- Fused Sinkhorn Kernel
- Context Parallel with MoE
- FP8 training support
- Enable ’--tp-comm-overlap‘ for MoE
- Distributed optimizer for MoE params.

# User Guide

### MoE Related Arguments

| Item | Description |
| --- | --- |
| num-experts | Number of Experts in MoE (None means no MoE) |
| expert-model-parallel-size | Degree of expert model parallelism. |
| moe-grouped-gemm | When there are multiple experts per rank, compress multiple local gemms into a single kernel launch to improve the utilization and performance by leveraging the Grouped GEMM feature introduced since CUTLASS 2.8 |
| moe-router-load-balancing-type | Determines the load balancing strategy for the router. "aux_loss" corresponds to the load balancing loss used in GShard and SwitchTransformer, "sinkhorn" corresponds to the balancing algorithm used in S-BASE, and "none" implies no load balancing. The default is "aux_loss". |
| moe-router-topk | Number of experts to route to for each token. The default is 2. |
| moe-aux-loss-coeff | Scaling coefficient for the aux loss: a starting value of 1e-2 is recommended. |
| moe-z-loss-coeff | Scaling coefficient for the z-loss: a starting value of 1e-3 is recommended. |
| moe-input-jitter-eps | Add noise to the input tensor by applying jitter with a specified epsilon value. |
| moe-token-dropping | This feature involves selectively dropping and padding tokens for each expert to achieve a specified capacity, similar to GShard, Switch-Transformer, and DeepSpeed-MoE. Note: Currently unsupported. |

### Example

To train a top-2 MoE model with an auxiliary loss, include the following arguments:

```python
--num-experts 8
--expert-model-parallel-size 8
--moe-grouped-gemm
--moe-router-load-balancing-type aux_loss # options: aux_loss, sinkhorn, none. Default is aux_loss.
--moe-router-topk 2
--moe-aux-loss-coeff 1e-2
--use-distributed-optimizer
```
## A detailed MoE script:
<details>
<summary>Click here. </summary>
    
```bash
#!/bin/bash

# Runs Mixtral 8x7B model on 16 A100 GPUs

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
    --use-mcore-models
    --disable-bias-linear
    --seq-length 2048
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
    --expert-model-parallel-size 4
    --moe-router-load-balancing-type aux_loss # options: aux_loss, sinkhorn, None. Default is aux_loss.
    --moe-router-topk 2
    --moe-aux-loss-coeff 1e-2
    --moe-grouped-gemm
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
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 4
    --pipeline-model-parallel-size 1
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
