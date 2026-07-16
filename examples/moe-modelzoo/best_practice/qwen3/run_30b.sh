# Cluster variables
export CONTAINER_IMAGE=
export ACCOUNT=
export MEGATRON_PATH=
export PARTITION=
export RUN_NAME="${MODEL}-benchmarking"
export CONTAINER_MOUNTS=
export CLUSTER=CW

# Model selection parameters
export MODEL=Qwen3-30B-A3B
export WANDB_PROJECT=
export OUTPUT_PATH=

# # Training parameters
export PROFILE=0 # whether to profile the model with nsys profile
export PRETRAIN=0 # whether train the model from scratch
export MBS=1
export SEQ_LEN=4096
export MOE_GROUPED_GEMM=true

export RUN_TIME=00:30:00
export COMMENT=baseline

# EP + DP solution
PP=1 VPP=1 TP=1 EP=8 NNODES=4 bash ./sbatch_benchmarking.sh --moe-router-force-load-balancing

# EP + full recompute solution
COMMENT="full_recompute" PP=1 VPP=1 TP=1 EP=8 MBS=4 NNODES=4 bash ./sbatch_benchmarking.sh --moe-router-force-load-balancing --recompute-granularity full --recompute-method uniform --recompute-num-layers 1

# config for 1x8 B200 nodes
PP=1 TP=1 EP=8 MBS=4 NNODES=1 bash ./sbatch_benchmarking.sh --moe-router-force-load-balancing --recompute-granularity selective --recompute-modules moe_act layernorm
