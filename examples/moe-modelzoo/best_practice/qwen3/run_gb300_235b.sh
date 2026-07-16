# Cluster variables
export CONTAINER_IMAGE=${CONTAINER_IMAGE:-}
export ACCOUNT=${ACCOUNT:-}
export MEGATRON_PATH=${MEGATRON_PATH:-}
export PARTITION=${PARTITION:-}
export CONTAINER_MOUNTS=${CONTAINER_MOUNTS:-}
export CLUSTER=${CLUSTER:-}
export BINDPCIE_PATH=${BINDPCIE_PATH:-}

# GB env
export GB200_CLUSTER=1
export NVLINK_DOMAIN_SIZE=72
## HybridEP settings, automatically set if using latest HybridEP
# export USE_MNNVL=1
# export NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN=32 # should be similar as EP

# Model selection parameters
export MODEL=${MODEL:-Qwen3-235B-A22B}
export RUN_NAME=${RUN_NAME:-"${MODEL}-benchmarking"}
export WANDB_PROJECT=${WANDB_PROJECT:-}
export OUTPUT_PATH=${OUTPUT_PATH:-}

# # Training parameters
export PROFILE=0 # whether to profile the model with nsys profile
export PRETRAIN=0 # whether train the model from scratch
export MBS=${MBS:-1}
export SEQ_LEN=${SEQ_LEN:-4096}
export MOE_GROUPED_GEMM=true

export RUN_TIME=00:30:00
export COMMENT=baseline
export PR=${PR:-"mxfp8"}

# GB300 config, MXFP8, 1055 TFLOPS
PR=mxfp8 MBS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_GRAPH_REGISTER=0 DISPATCHER=hybridep SEGMENT=16 A2A_OVERLAP=1 TP=1 PP=1 VPP=1 EP=64 NNODES=16 GBS=8192 bash ./sbatch_benchmarking.sh --moe-router-force-load-balancing --cuda-graph-impl transformer_engine --cuda-graph-scope attn moe_router moe_preprocess

# GB300 config, MXFP8 + paged_stash + full-iteration CUDA graph + HybridEP, 128 GPUs, ~1324 TFLOPS
## Requires the fix from https://github.com/NVIDIA/Megatron-LM/pull/4226 (commit b169e131).
## Image: TransformerEngine >= 2.16, nvidia-cudnn-frontend >= 1.23.0.
PR=mxfp8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_GRAPH_REGISTER=0 DISPATCHER=hybridep A2A_OVERLAP=1 NVTE_CUTEDSL_FUSED_GROUPED_MLP=1 NVTE_NORM_FWD_USE_CUDNN=0 NVTE_NORM_BWD_USE_CUDNN=0 TP=1 PP=1 VPP=1 EP=64 SEGMENT=16 NNODES=32 MBS=1 GBS=8192 bash ./sbatch_benchmarking.sh --moe-router-force-load-balancing --moe-paged-stash --moe-expert-rank-capacity-factor 1.2 --moe-paged-stash-page-size 64 --moe-paged-stash-buffer-size-factor-cuda 1.0 --use-transformer-engine-op-fuser --moe-pad-experts-for-cuda-graph-inference --moe-mlp-glu-interleave-size 32 --cuda-graph-impl local --cuda-graph-scope full_iteration --cuda-graph-warmup-steps 2 --no-check-for-nan-in-loss-and-grad --te-rng-tracker
                                                                                            