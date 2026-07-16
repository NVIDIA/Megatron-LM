# H100
export CLUSTER=
export CONTAINER_IMAGE=
export MEGATRON_PATH=
export MCORE_RELEASE_VERSION=
export GB200_CLUSTER=0

export MODEL=DeepSeek-V3
export WANDB_PROJECT=
export RUN_NAME="${MODEL}-benchmarking"

export BINDPCIE_PATH=${BINDPCIE_PATH:-""}

# # Training parameters
export PROFILE=0 # whether to profile the model with nsys profile
export PRETRAIN=0 # whether train the model from scratch
export MBS=1
export SEQ_LEN=4096
export MOE_GROUPED_GEMM=true

export RUN_TIME=00:30:00
export COMMENT=


# H100 config. best config on 1024 H100 GPUs with 4096 sequence lengths
A2A_OVERLAP=1 PP=8 VPP=4 TP=2 EP=64 NNODES=128 GBS=8192 PR=fp8 bash sbatch_benchmarking.sh \
    --recompute-granularity selective \
    --recompute-modules mla_up_proj mlp \
    --mtp-num-layers 1 --mtp-loss-scaling-factor 0.1 \
    --pipeline-model-parallel-layout "Et|(tt|)*30mL"

# B200 config with 256 GPUs
PR=mxfp8 A2A_OVERLAP=1 TP=1 PP=8 EP=32 NNODES=32 GBS=2048 bash sbatch_benchmarking.sh \
    --recompute-granularity selective \
    --recompute-modules mla_up_proj mlp \
    --mtp-num-layers 1 --mtp-loss-scaling-factor 0.1 \
    --pipeline-model-parallel-layout "Et|(tt|)*30mL"



# Long Context
## 256 H100, 4k sequence length
OPTIMIZER_OFFLOAD=1 A2A_OVERLAP=0 MODEL=DeepSeek-V3 PP=8 VPP=4 TP=1 EP=32 CP=1 NNODES=32 GBS=8192 PR=fp8 bash sbatch_benchmarking.sh --recompute-granularity selective --recompute-modules mla_up_proj moe mlp layernorm --pipeline-model-parallel-layout "Et*2|(tt|)*22t|(tt|)*7mL" --mtp-num-layers 1 --mtp-loss-scaling-factor 0.1

## 256 H100, 16k sequence length
OPTIMIZER_OFFLOAD=1 A2A_OVERLAP=0 MODEL=DeepSeek-V3 PP=8 VPP=4 TP=4 EP=32 CP=1 NNODES=32 GBS=3840 SEQ_LEN=16384 PR=fp8 bash sbatch_benchmarking.sh --recompute-granularity selective --recompute-modules mla_up_proj moe mlp layernorm --pipeline-model-parallel-layout "Et*2|(tt|)*22t|(tt|)*7mL" --mtp-num-layers 1 --mtp-loss-scaling-factor 0.1

## 256 H100, 32k sequence length
OPTIMIZER_OFFLOAD=1 A2A_OVERLAP=0 MODEL=DeepSeek-V3 PP=8 VPP=4 TP=8 EP=32 CP=1 NNODES=32 GBS=1920 SEQ_LEN=32768 PR=fp8 bash sbatch_benchmarking.sh --recompute-granularity selective --recompute-modules mla_up_proj moe mlp layernorm --pipeline-model-parallel-layout "Et*2|(tt|)*22t|(tt|)*7mL" --mtp-num-layers 1 --mtp-loss-scaling-factor 0.1

## 256 H100, 64k sequence length
OPTIMIZER_OFFLOAD=1 A2A_OVERLAP=0 MODEL=DeepSeek-V3 PP=8 VPP=4 TP=16 EP=32 CP=1 NNODES=32 GBS=960 SEQ_LEN=65536 PR=fp8 bash sbatch_benchmarking.sh --recompute-granularity selective --recompute-modules mla_up_proj moe mlp layernorm --pipeline-model-parallel-layout "Et*2|(tt|)*22t|(tt|)*7mL" --mtp-num-layers 1 --mtp-loss-scaling-factor 0.1

## 256 H100, 128k sequence length
OPTIMIZER_OFFLOAD=1 A2A_OVERLAP=0 MODEL=DeepSeek-V3 PP=8 VPP=4 TP=32 EP=32 CP=1 NNODES=32 GBS=480 SEQ_LEN=131072 PR=fp8 bash sbatch_benchmarking.sh --recompute-granularity selective --recompute-modules mla_up_proj moe mlp layernorm --pipeline-model-parallel-layout "Et*2|(tt|)*22t|(tt|)*7mL" --mtp-num-layers 1 --mtp-loss-scaling-factor 0.1

## 512 H100, 256k sequence length
OPTIMIZER_OFFLOAD=1 A2A_OVERLAP=0 MODEL=DeepSeek-V3 PP=8 VPP=4 TP=64 EP=32 CP=1 NNODES=64 GBS=240 SEQ_LEN=262144 PR=fp8 bash sbatch_benchmarking.sh --recompute-granularity selective --recompute-modules mla_up_proj moe mlp layernorm --pipeline-model-parallel-layout "Et*2|(tt|)*22t|(tt|)*7mL" --mtp-num-layers 1 --mtp-loss-scaling-factor 0.1

# Note: Add (remove) `--moe-router-force-load-balancing` to test forced load balancing (dropless).
