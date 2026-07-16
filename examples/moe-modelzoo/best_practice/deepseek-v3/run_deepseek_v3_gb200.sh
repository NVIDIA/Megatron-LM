# ── Cluster setup (GB200 NVL72) ───────────────────────────────────────────────
export CLUSTER=${CLUSTER:-}
export CONTAINER_IMAGE=${CONTAINER_IMAGE:-}   # see dockers/GB200.Dockerfile
export GB200_CLUSTER=1

# ── bindpcie (CPU/memory affinity binding, recommended for GB200) ─────────────
# wget https://raw.githubusercontent.com/NVIDIA/mlperf-common/refs/heads/main/client/bindpcie
# chmod +x bindpcie && export BINDPCIE_PATH=/path/to/bindpcie
export BINDPCIE_PATH=${BINDPCIE_PATH:-}

# ── Paths (fill in) ───────────────────────────────────────────────────────────
export MEGATRON_PATH=${MEGATRON_PATH:-}        # /path/to/Megatron-LM (moe_dev branch; --offload-optimizer-states requires dev branch)
export MCORE_RELEASE_VERSION=${MCORE_RELEASE_VERSION:-0.16}

# ── Common training defaults ──────────────────────────────────────────────────
export MODEL=${MODEL:-DeepSeek-V3}
export DATASET=${DATASET:-}                # your dataset name
export MBS=${MBS:-1}
export SEQ_LEN=${SEQ_LEN:-4096}
export MOE_GROUPED_GEMM=${MOE_GROUPED_GEMM:-true}
export PRETRAIN=${PRETRAIN:-0}           # 0 = finetune from checkpoint, 1 = from scratch
export DISPATCHER=${DISPATCHER:-hybridep}
export A2A_OVERLAP=${A2A_OVERLAP:-1}
export NVTE_CPU_OFFLOAD_V1=${NVTE_CPU_OFFLOAD_V1:-1}
export RUN_TIME=${RUN_TIME:-00:30:00}
export WANDB_API_KEY=${WANDB_API_KEY:-}    # set to enable WandB; leave empty to disable
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# HybridEP: topology is auto-detected via NVLINK_DOMAIN_SIZE=72 (set by sbatch_benchmarking.sh).
# NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN and USE_MNNVL do NOT need to be set.
# SEGMENT is a Slurm arg for NVLink-aware node allocation: SEGMENT = EP * TP / 4.

# ── Configs ───────────────────────────────────────────────────────────────────

# 64 nodes (256 GPUs), 4k seq, mxfp8, HybridEP
PR=mxfp8 TP=1 PP=4 EP=64 VPP=4 NNODES=64 GBS=8192 SEGMENT=16 bash sbatch_benchmarking.sh \
  --recompute-granularity selective --recompute-modules mla_up_proj \
  --cuda-graph-impl transformer_engine --cuda-graph-scope attn moe_router moe_preprocess --te-rng-tracker --cuda-graph-warmup-steps 1 \
  --pipeline-model-parallel-layout "Et*4|(tttt|)*14tmL" \
  --mtp-num-layers 1 --mtp-loss-scaling-factor 0.1 \
  --fine-grained-activation-offloading --offload-modules expert_fc1 --delay-offload-until-cuda-graph \
  --moe-router-pre-softmax \
  --offload-optimizer-states \
  --moe-router-force-load-balancing

# Note: add/remove --moe-router-force-load-balancing to test forced balance vs dropless.
