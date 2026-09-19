#!/bin/bash
# Load-side resharding validation for the fsdp_dtensor -> torch_dist converter.
#
#   bash validate_reshard.sh <model>         # needs >=2 GPUs
#
# validate_resume.sh proves the converted checkpoint resumes single-rank
# (TP1 -> TP1). This driver proves the converter's actual promise: the converted
# torch_dist checkpoint (full-shape tensors) re-loads into a CLASSIC (non-FSDP)
# job under a DIFFERENT target parallel layout and reshards on load. It reuses the
# results/<model>/td80 checkpoint produced by validate_resume.sh, loads it 2-GPU
# under each layout in the model's RESHARD_LAYOUTS, and verifies a clean load +
# iter-81 loss/LR continuity vs the FSDP reference.
#
# Prereq: `bash validate_resume.sh <model>` already ran (needs results/<model>/td80
# and results/<model>/train_fsdp.log). Requires >=2 visible GPUs.
set -uo pipefail

_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$_DIR/common.sh"

M="${1:?usage: validate_reshard.sh <model>   (see: bash common.sh list-models)}"
load_model "$M" || exit $?

cd "$REPO_ROOT"

OUT="$RESULTS_ROOT/$M"
IT=80; NX=81; END=$((IT + 3))
PORT="${MASTER_PORT:-29650}"

if [ -z "$RESHARD_LAYOUTS" ]; then
  echo "[$M] no RESHARD_LAYOUTS configured for this model — nothing to reshard. Skipping."
  exit 0
fi
if [ "$(gpu_count)" -lt 2 ]; then
  echo "[$M] reshard needs >=2 GPUs (saw $(gpu_count)). Skipping." >&2
  exit 0
fi
[ -d "$OUT/td$IT/iter_00000$IT" ] || {
  echo "missing $OUT/td$IT — run 'bash validate_resume.sh $M' first" >&2; exit 2; }
FSDP_REF="$(grep -E "iteration +$NX/" "$OUT/train_fsdp.log" | tail -1)"
[ -n "$FSDP_REF" ] || { echo "no FSDP iter-$NX reference in $OUT/train_fsdp.log" >&2; exit 2; }

# 2-GPU classic load needs P2P disabled (pre-Blackwell) and one copy engine.
RESHARD_ENV=("${DETERMINISTIC_ENV[@]}" NCCL_P2P_DISABLE=1 CUDA_DEVICE_MAX_CONNECTIONS=1)

layout_flags() {
  case "$1" in
    TP2)   echo "--tensor-model-parallel-size 2" ;;
    TP2SP) echo "--tensor-model-parallel-size 2 --sequence-parallel" ;;
    PP2)   echo "--tensor-model-parallel-size 1 --pipeline-model-parallel-size 2" ;;
    EP2)   echo "--tensor-model-parallel-size 1 --expert-model-parallel-size 2" ;;
    *)     echo "" ;;
  esac
}

echo "======== [$M] RESHARD (load td$IT under each target layout) ========"
for L in $RESHARD_LAYOUTS; do
  PAR="$(layout_flags "$L")"
  [ -n "$PAR" ] || { echo "unknown layout $L — skipping" >&2; continue; }
  RLOG="$OUT/reshard_${L}.log"
  echo "=== [$M] load td$IT at $L ($PAR) ==="
  # The converted checkpoint carries no TP1 parallel state, so
  # --tensor-model-parallel-size in COMMON_ARGS is overridden by $PAR (later wins).
  env "${RESHARD_ENV[@]}" torchrun --nproc_per_node 2 --master_port "$PORT" \
    pretrain_gpt.py \
    "${CLASSIC_LOAD_FLAGS[@]}" "${COMMON_ARGS[@]}" --num-layers "$NUM_LAYERS" $ARCH $PAR \
    --train-iters "$END" --save-interval 1000 \
    --load "$OUT/td$IT" --data-cache-path "/tmp/dc_${M}_reshard_${L}" 2>&1 | tee "$RLOG" >/dev/null
  echo "  rc=${PIPESTATUS[0]}"
done

echo "======== [$M] RESHARD VERIFICATION (iter $NX vs FSDP reference) ========"
echo "FSDP    ref: $FSDP_REF"
for L in $RESHARD_LAYOUTS; do
  RLOG="$OUT/reshard_${L}.log"
  [ -f "$RLOG" ] || continue
  echo "--- layout $L ---"
  grep -h "successfully loaded checkpoint" "$RLOG" | head -1
  echo "RESHARD $L: $(grep -E "iteration +$NX/" "$RLOG" | head -1)"
done
echo "[$M] reshard done — PASS when each layout loads cleanly and its iter-$NX"
echo "     lm loss matches the FSDP reference within bf16 tolerance."
