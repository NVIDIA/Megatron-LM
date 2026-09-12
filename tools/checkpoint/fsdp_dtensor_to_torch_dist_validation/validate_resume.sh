#!/bin/bash
# Resume-continuity validation for the fsdp_dtensor -> torch_dist reverse converter.
#
#   bash validate_resume.sh <model>          # e.g. dense, moe_grouped, gdn_hybrid
#   bash common.sh list-models               # see the available models
#
# For <model>:
#   1. train Megatron-FSDP from scratch to iter 100, --save-interval 20
#      (fsdp_dtensor checkpoints at iters 20/40/60/80/100)
#   2. reverse-convert iter_0000060 -> td60 and iter_0000080 -> td80
#   3. resume a CLASSIC (non-FSDP) job from td60 (-> iter 63) and td80 (-> iter 83)
#   4. print the VERIFICATION block: the two [Convert] lines, the two
#      "successfully loaded ... at iteration {60,80}" lines, and — for each of
#      iters 61 and 81 — the FSDP reference line next to the resumed line
#      (compare `lm loss:` and `learning rate:`).
#
# Runs single-rank (torchrun --nproc_per_node 1). The converter is
# parallelism-agnostic (it reads each tensor's global shape from DCP), so world
# size 1 fully exercises it and avoids the NCCL contention that can deadlock
# multi-rank collectives when a GPU is shared. Needs 1 GPU. Outputs land under
# results/<model>/ (gitignored). See ./README.md for how to read the results.
set -uo pipefail

_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$_DIR/common.sh"

M="${1:?usage: validate_resume.sh <model>   (see: bash common.sh list-models)}"
load_model "$M" || exit $?

# pretrain_gpt.py and its imports resolve relative to the repo root.
cd "$REPO_ROOT"

OUT="$RESULTS_ROOT/$M"
LOG="$OUT/_run.log"
PORT="${MASTER_PORT:-29600}"

# One-time per-model setup (e.g. GDN's flash-linear-attention).
if [ -n "$EXTRA_SETUP" ]; then
  echo "=== [$M] extra setup: $EXTRA_SETUP ==="
  eval "$EXTRA_SETUP"
fi

rm -rf "$OUT"; mkdir -p "$OUT"
echo "=== [$M] $MODEL_LABEL — train -> iter 100 (save @20/40/60/80/100) ===" | tee "$LOG"

env "${DETERMINISTIC_ENV[@]}" torchrun --nproc_per_node 1 --master_port "$PORT" \
  pretrain_gpt.py \
  "${FSDP_TRAIN_FLAGS[@]}" "${COMMON_ARGS[@]}" --num-layers "$NUM_LAYERS" $ARCH \
  --train-iters 100 --save-interval 20 \
  --save "$OUT/fsdp" --data-cache-path "/tmp/dc_${M}_train" 2>&1 | tee "$OUT/train_fsdp.log"

for IT in 60 80; do
  TAG="td$IT"
  echo "=== [$M] convert iter_00000$IT -> $TAG ===" | tee -a "$LOG"
  python "$INSPECTOR" convert-fsdp-dtensor-to-torch-dist \
    "$OUT/fsdp/iter_00000$IT" "$OUT/$TAG/iter_00000$IT" 2>&1 | tee "$OUT/convert_$IT.log"
  echo "$IT" > "$OUT/$TAG/latest_checkpointed_iteration.txt"

  END=$((IT + 3))
  echo "=== [$M] classic resume from $TAG (iter $((IT + 1)) -> $END) ===" | tee -a "$LOG"
  env "${DETERMINISTIC_ENV[@]}" torchrun --nproc_per_node 1 --master_port "$((PORT + 1))" \
    pretrain_gpt.py \
    "${CLASSIC_LOAD_FLAGS[@]}" "${COMMON_ARGS[@]}" --num-layers "$NUM_LAYERS" $ARCH \
    --train-iters "$END" --save-interval 1000 \
    --load "$OUT/$TAG" --data-cache-path "/tmp/dc_${M}_resume$IT" 2>&1 | tee "$OUT/resume_$IT.log"
done

echo "======== [$M] VERIFICATION ========" | tee -a "$LOG"
for IT in 60 80; do
  NX=$((IT + 1))
  echo "--- checkpoint $IT (compare resumed iter $NX vs FSDP iter $NX) ---" | tee -a "$LOG"
  grep -h "\[Convert\]" "$OUT/convert_$IT.log" | tee -a "$LOG"
  grep -h "successfully loaded checkpoint" "$OUT/resume_$IT.log" | tee -a "$LOG"
  echo "FSDP    iter $NX: $(grep -E "iteration +$NX/" "$OUT/train_fsdp.log" | tail -1)" | tee -a "$LOG"
  echo "RESUMED iter $NX: $(grep -E "iteration +$NX/" "$OUT/resume_$IT.log" | head -1)" | tee -a "$LOG"
done
echo "[$M] done — outputs under $OUT" | tee -a "$LOG"
echo "[$M] PASS when: loads at 60/80, first resumed lm loss ~= FSDP loss at iter 61/81"
echo "     (bf16 tolerance; FP8 tracks ~1% looser by design), and LR matches exactly."
