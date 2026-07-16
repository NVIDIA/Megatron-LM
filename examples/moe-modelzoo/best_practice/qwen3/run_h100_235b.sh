# Cluster variables
export CONTAINER_IMAGE=${CONTAINER_IMAGE:-}
export ACCOUNT=${ACCOUNT:-}
export MEGATRON_PATH=${MEGATRON_PATH:-}
export PARTITION=${PARTITION:-}
export CONTAINER_MOUNTS=${CONTAINER_MOUNTS:-}
export CLUSTER=${CLUSTER:-}

# Model selection parameters
export MODEL=${MODEL:-Qwen3-235B-A22B}
export RUN_NAME=${RUN_NAME:-"${MODEL}-benchmarking"}
export WANDB_PROJECT=${WANDB_PROJECT:-}
export OUTPUT_PATH=${OUTPUT_PATH:-}

# # Training parameters
export PROFILE=0 # whether to profile the model with nsys profile
export PRETRAIN=0 # whether train the model from scratch
export MBS=1
export SEQ_LEN=4096
export MOE_GROUPED_GEMM=true

export RUN_TIME=00:30:00
export COMMENT=baseline
export PR=${PR:-"bf16"}

# H100 config, 320 TFLOPS
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_GRAPH_REGISTER=0 DISPATCHER=deepep A2A_OVERLAP=1 TP=2 PP=8 VPP=4 EP=32 NNODES=32 GBS=2048 bash ./sbatch_benchmarking.sh --recompute-granularity selective --recompute-modules moe_act layernorm --moe-router-force-load-balancing --cuda-graph-impl transformer_engine --cuda-graph-scope moe_router moe_preprocess
