#!/bin/bash
# Shared configuration and helpers for the fsdp_dtensor -> torch_dist reverse
# converter validation harness (see ./README.md).
#
# Dual-use — this is the single source of truth for the tiny-model args and the
# converter/resume flags:
#   * `source common.sh` from a driver (validate_resume.sh / validate_reshard.sh)
#     to get COMMON_ARGS, DETERMINISTIC_ENV, FSDP_TRAIN_FLAGS, CLASSIC_LOAD_FLAGS,
#     and the model registry (load_model / list_models).
#   * run it as a small CLI so the Python bit-exact tool reads the exact same arg
#     vector instead of re-declaring it:
#       bash common.sh list-models
#       bash common.sh emit-args <model>
#       bash common.sh emit-load-flags

# Resolve directories relative to THIS file (no absolute paths baked in).
_VAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
_MODELS_DIR="$_VAL_DIR/models"
# tools/checkpoint/fsdp_dtensor_to_torch_dist_validation -> repo root is three levels up.
REPO_ROOT="$(cd "$_VAL_DIR/../../.." && pwd)"

# The reverse converter CLI.
INSPECTOR="$REPO_ROOT/tools/checkpoint/checkpoint_inspector.py"

# Train/convert/resume outputs (checkpoints + logs) land here. Gitignored.
# Override with RESULTS_DIR=/somewhere.
RESULTS_ROOT="${RESULTS_DIR:-$_VAL_DIR/results}"

# --- tiny shared model config (identical across every validation model) ------
# 12L/512H bf16 deterministic mock-data run. Per-model files set NUM_LAYERS and
# append architecture flags via ARCH. Caller-specific flags (--train-iters,
# --save, --save-interval, --load) are appended by each driver, NOT here.
COMMON_ARGS=(
  --hidden-size 512 --num-attention-heads 8
  --seq-length 1024 --max-position-embeddings 1024
  --micro-batch-size 4 --global-batch-size 32
  --eval-interval 1000 --eval-iters 5 --split 949,50,1
  --lr 1.5e-4 --min-lr 1e-5 --lr-decay-style cosine --lr-warmup-fraction 0.01
  --weight-decay 1e-2 --clip-grad 1.0 --use-checkpoint-opt_param-scheduler
  --transformer-impl transformer_engine --bf16 --deterministic-mode
  --no-gradient-accumulation-fusion --seed 1234
  --tokenizer-type NullTokenizer --vocab-size 131073 --mock-data --log-interval 1
  --tensor-model-parallel-size 1
)

# Deterministic env for FSDP train + classic resume. Omitting
# NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 with --deterministic-mode + TE fails the
# model build. Do NOT set CUDA_DEVICE_MAX_CONNECTIONS=1 for single-rank FSDP.
DETERMINISTIC_ENV=(
  CUBLAS_WORKSPACE_CONFIG=:4096:8
  NVTE_ALLOW_NONDETERMINISTIC_ALGO=0
  NCCL_ALGO=Ring
)

# Produce the fsdp_dtensor source checkpoint.
FSDP_TRAIN_FLAGS=(
  --use-megatron-fsdp --use-distributed-optimizer --ckpt-format fsdp_dtensor
  --data-parallel-sharding-strategy optim_grads_params
)

# Resume a classic (non-FSDP) job from the converted torch_dist checkpoint.
# --dist-ckpt-optim-fully-reshardable is the only distributed-optimizer on-disk
# layout that is per-parameter and model-shaped (what the converter emits);
# log_all strictness drops the omitted TE _extra_state keys instead of erroring.
CLASSIC_LOAD_FLAGS=(
  --use-distributed-optimizer --ckpt-format torch_dist
  --dist-ckpt-optim-fully-reshardable --dist-ckpt-strictness log_all --no-load-rng
)

# --- model registry ----------------------------------------------------------
list_models() {
  local f
  for f in "$_MODELS_DIR"/*.sh; do
    [ -e "$f" ] || continue
    basename "$f" .sh
  done
}

# Source models/<name>.sh, populating MODEL_LABEL / MODEL_TRANSFORM / NUM_LAYERS /
# ARCH / RESHARD_LAYOUTS / EXTRA_SETUP. Exits non-zero (listing choices) if unknown.
load_model() {
  local name="$1" f
  f="$_MODELS_DIR/$name.sh"
  if [ ! -f "$f" ]; then
    echo "error: unknown model '$name'. Available models:" >&2
    list_models | sed 's/^/  /' >&2
    return 2
  fi
  # Reset to defaults so a sparse model file can rely on them.
  MODEL_LABEL="" MODEL_TRANSFORM="" NUM_LAYERS=12 ARCH="" RESHARD_LAYOUTS="" EXTRA_SETUP=""
  # shellcheck disable=SC1090
  source "$f"
}

# Full training/eval arg vector for a model (COMMON_ARGS + --num-layers + ARCH),
# space-joined on one line for the Python bit-exact tool. ARCH is intentionally
# unquoted (word-split into individual flags).
emit_args() {
  load_model "$1" || return $?
  # shellcheck disable=SC2086
  printf '%s ' "${COMMON_ARGS[@]}" --num-layers "$NUM_LAYERS" $ARCH
  printf '\n'
}

emit_load_flags() {
  printf '%s ' "${CLASSIC_LOAD_FLAGS[@]}"
  printf '\n'
}

# --- preflight helper --------------------------------------------------------
# Remove a stale nvidia-resiliency-ext (<0.6.0) that breaks `import megatron.core`
# in older dev images. Harmless if already current; checkpoint validation does
# not use nvrx async-save. A freshly-built dev image needs no fix.
preflight_remove_stale_nvrx() {
  local d
  for d in /usr/local/lib/python3.12/dist-packages \
           /opt/venv/lib/python3.12/site-packages \
           /usr/lib/python3/dist-packages; do
    rm -rf "$d"/nvidia_resiliency_ext* 2>/dev/null || true
  done
}

# Number of visible GPUs (respects CUDA_VISIBLE_DEVICES).
gpu_count() {
  if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    awk -F, '{print NF}' <<<"$CUDA_VISIBLE_DEVICES"
  else
    nvidia-smi -L 2>/dev/null | grep -c '^GPU' || echo 0
  fi
}

# --- CLI dispatch (only when executed directly, not when sourced) ------------
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
  set -uo pipefail
  case "${1:-}" in
    list-models)     list_models ;;
    emit-args)       emit_args "${2:?usage: common.sh emit-args <model>}" ;;
    emit-load-flags) emit_load_flags ;;
    preflight)       preflight_remove_stale_nvrx ;;
    gpu-count)       gpu_count ;;
    *) echo "usage: common.sh {list-models|emit-args <model>|emit-load-flags|preflight|gpu-count}" >&2
       exit 2 ;;
  esac
fi
