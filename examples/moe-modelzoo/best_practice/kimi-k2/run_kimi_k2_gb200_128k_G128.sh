# GB200 Kimi-K2 128k TP16 PP2 EP64 benchmark
export CLUSTER=${CLUSTER:-}
export CONTAINER_IMAGE=${CONTAINER_IMAGE:-}   # see dockers/GB200.Dockerfile
export GB200_CLUSTER=1

# bindpcie is recommended for GB200 CPU/memory affinity binding.
# wget https://raw.githubusercontent.com/NVIDIA/mlperf-common/refs/heads/main/client/bindpcie
# chmod +x bindpcie && export BINDPCIE_PATH=/path/to/bindpcie
export BINDPCIE_PATH=${BINDPCIE_PATH:-}

export MEGATRON_PATH=${MEGATRON_PATH:-}        # /path/to/Megatron-LM
export MCORE_RELEASE_VERSION=${MCORE_RELEASE_VERSION:-0.16}

export MODEL=Kimi-K2
export RUN_NAME=${RUN_NAME:-"${MODEL}-gb200-128k"}
export OUTPUT_PATH=${OUTPUT_PATH:-"${PWD}/outputs/${RUN_NAME}"}
export WANDB_PROJECT=${WANDB_PROJECT:-megatron-moe-benchmarking}
export WANDB_API_KEY=${WANDB_API_KEY:-}

export PROFILE=${PROFILE:-0}
export PRETRAIN=1
export MBS=1
export MOE_GROUPED_GEMM=true
export RUN_TIME=${RUN_TIME:-00:30:00}
export COMMENT=${COMMENT:-gb200-128k}
export DISPATCHER=hybridep
export OPTIMIZER_OFFLOAD=0
export A2A_OVERLAP=0
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

export DATA_PATH=${DATA_PATH:-mock}
export TOKENIZER_MODEL=${TOKENIZER_MODEL:-mock}
export SEGMENT=${SEGMENT:-16}

OPTIMIZER_OFFLOAD=${OPTIMIZER_OFFLOAD} A2A_OVERLAP=${A2A_OVERLAP} DISPATCHER=${DISPATCHER} MODEL=Kimi-K2 PP=2 TP=16 EP=64 CP=1 NNODES=32 GBS=128 SEQ_LEN=131072 PR=mxfp8 bash sbatch_benchmarking.sh \
    --recompute-granularity selective \
    --recompute-modules mla_up_proj mlp layernorm moe_act \
    --pipeline-model-parallel-layout "'Et|t|(tt|)*29tL'" \
    --fine-grained-activation-offloading \
    --offload-modules expert_fc1 \
    --te-rng-tracker \
    --offload-optimizer-states \
    --moe-permute-fusion-into-hybridep \
    --moe-router-force-load-balancing \
    --mock-data \
    --tokenizer-type NullTokenizer --vocab-size 163840 \
    --train-iters ${TRAIN_ITERS:-10} \
    --eval-iters 0
