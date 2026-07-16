# ── Cluster setup (B200 x86) ──────────────────────────────────────────────────
export CLUSTER=${CLUSTER:-}
export CONTAINER_IMAGE=${CONTAINER_IMAGE:-}   # see dockers/B200.Dockerfile
export GB200_CLUSTER=0

# ── Paths (fill in) ───────────────────────────────────────────────────────────
export MEGATRON_PATH=${MEGATRON_PATH:-}
export MCORE_RELEASE_VERSION=${MCORE_RELEASE_VERSION:-0.16}

# ── Common training defaults ──────────────────────────────────────────────────
export MODEL=${MODEL:-DeepSeek-V3}
export DATASET=${DATASET:-}                # your dataset name
export MBS=${MBS:-1}
export SEQ_LEN=${SEQ_LEN:-4096}
export MOE_GROUPED_GEMM=${MOE_GROUPED_GEMM:-true}
export PRETRAIN=${PRETRAIN:-0}           # 0 = finetune from checkpoint, 1 = from scratch
export DISPATCHER=${DISPATCHER:-deepep}
export RUN_TIME=${RUN_TIME:-00:30:00}
export WANDB_API_KEY=${WANDB_API_KEY:-}    # set to enable WandB; leave empty to disable

# ── Configs ───────────────────────────────────────────────────────────────────

# 256x B200, 4k seq, mxfp8, DeepEP
PR=mxfp8 TP=1 PP=8 EP=32 NNODES=32 GBS=8192 bash sbatch_benchmarking.sh \
  --recompute-granularity selective --recompute-modules mlp mla_up_proj \
  --pipeline-model-parallel-layout "Et*4|(tttt|)*14tmL" \
  --mtp-num-layers 1 --mtp-loss-scaling-factor 0.1 \
  --moe-deepep-num-sms 24 \
  --moe-router-force-load-balancing

# Note: add/remove --moe-router-force-load-balancing to test forced balance vs dropless.
