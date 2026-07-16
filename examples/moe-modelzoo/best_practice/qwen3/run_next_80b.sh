# Cluster variables
export CONTAINER_IMAGE=
export ACCOUNT=
export MEGATRON_PATH=
export PARTITION=
export RUN_NAME="${MODEL}-benchmarking"
export CONTAINER_MOUNTS=
export CLUSTER=

# Model selection parameters
export MODEL=Qwen3-Next-80B-A3B
export WANDB_PROJECT=
export OUTPUT_PATH=

# # Training parameters
export PROFILE=0 # whether to profile the model with nsys profile
export PRETRAIN=0 # whether train the model from scratch
export MBS=1
export SEQ_LEN=4096
export MOE_GROUPED_GEMM=true
export PR=bf16

export RUN_TIME=00:30:00
export COMMENT=baseline


# H100 baseline config
PP=4 VPP=3 TP=1 EP=32 CP=1 NNODES=16 GBS=256 bash ./sbatch_benchmarking.sh --recompute-granularity selective --recompute-modules moe_act shared_experts layernorm --moe-router-force-load-balancing
