#!/bin/bash
# Source-side sharding validation for the fsdp_dtensor -> torch_dist converter.
#
#   bash validate_source_sharding.sh <model> [DP2|TP2|PP2|EP2]   # needs >=2 GPUs
#
# validate_resume.sh trains the FSDP source single-rank (DP1) — an *unsharded* DCP
# store. This driver trains the source on >=2 GPUs under a real sharded layout, so
# the converter's source-side gather (multi-shard DTensor reassembly, and the
# TP-reshard `nd_reformulated_orig_global_shape` path in checkpoint_inspector.py)
# is actually exercised, not just asserted. It then converts (single-process,
# unchanged) and resumes a CLASSIC single-rank job, checking iter-61/81 continuity
# against the sharded FSDP run's own log.
#
# This is the source-side counterpart to validate_reshard.sh (which shards the
# *load* side): together they cover both ends of the parallelism-agnostic claim.
#
# Layout (default DP2):
#   DP2  - FSDP shards params + optimizer across 2 data-parallel ranks (the plain
#          "trained on >1 GPU" case; TP1/PP1/EP1)
#   TP2  - tensor-parallel source
#   PP2  - pipeline-parallel source (exercises _stack_layers layer-offset contiguity)
#   EP2  - expert-parallel source (MoE models only; exercises the expert gather)
#
# Needs >=2 visible GPUs. Outputs under results/<model>__src_<layout>/ (gitignored).
set -uo pipefail

_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$_DIR/common.sh"

M="${1:?usage: validate_source_sharding.sh <model> [DP2|TP2|PP2|EP2]}"
LAYOUT="${2:-DP2}"
load_model "$M" || exit $?

cd "$REPO_ROOT"

if [ "$(gpu_count)" -lt 2 ]; then
  echo "[$M/$LAYOUT] source sharding needs >=2 GPUs (saw $(gpu_count)). Skipping." >&2
  exit 0
fi

case "$LAYOUT" in
  DP2) SRC_PAR="" ;;
  TP2) SRC_PAR="--tensor-model-parallel-size 2" ;;
  PP2) SRC_PAR="--pipeline-model-parallel-size 2" ;;
  EP2) SRC_PAR="--expert-model-parallel-size 2"
       case "$ARCH" in
         *--num-experts*) ;;
         *) echo "[$M/$LAYOUT] EP2 requires an MoE model (no --num-experts in ARCH)" >&2; exit 2 ;;
       esac ;;
  *) echo "unknown layout '$LAYOUT' (use DP2|TP2|PP2|EP2)" >&2; exit 2 ;;
esac

OUT="$RESULTS_ROOT/${M}__src_${LAYOUT}"
LOG="$OUT/_run.log"
PORT="${MASTER_PORT:-29700}"

if [ -n "$EXTRA_SETUP" ]; then
  echo "=== [$M/$LAYOUT] extra setup: $EXTRA_SETUP ==="
  eval "$EXTRA_SETUP"
fi

# 2-GPU FSDP training: pre-Blackwell wants P2P disabled to avoid NCCL hangs;
# Megatron-FSDP must NOT set CUDA_DEVICE_MAX_CONNECTIONS=1.
TRAIN_ENV=("${DETERMINISTIC_ENV[@]}" NCCL_P2P_DISABLE=1)

rm -rf "$OUT"; mkdir -p "$OUT"
echo "=== [$M/$LAYOUT] $MODEL_LABEL — train SHARDED source (2 GPUs, $LAYOUT) -> iter 100 (save @20) ===" | tee "$LOG"

env "${TRAIN_ENV[@]}" torchrun --nproc_per_node 2 --master_port "$PORT" \
  pretrain_gpt.py \
  "${FSDP_TRAIN_FLAGS[@]}" "${COMMON_ARGS[@]}" --num-layers "$NUM_LAYERS" $ARCH $SRC_PAR \
  --train-iters 100 --save-interval 20 \
  --save "$OUT/fsdp" --data-cache-path "/tmp/dc_${M}_src${LAYOUT}_train" 2>&1 | tee "$OUT/train_fsdp.log"

for IT in 60 80; do
  TAG="td$IT"
  echo "=== [$M/$LAYOUT] convert iter_00000$IT -> $TAG (single-process CPU) ===" | tee -a "$LOG"
  python "$INSPECTOR" convert-fsdp-dtensor-to-torch-dist \
    "$OUT/fsdp/iter_00000$IT" "$OUT/$TAG/iter_00000$IT" 2>&1 | tee "$OUT/convert_$IT.log"
  echo "$IT" > "$OUT/$TAG/latest_checkpointed_iteration.txt"

  END=$((IT + 3))
  echo "=== [$M/$LAYOUT] classic single-rank resume from $TAG (iter $((IT + 1)) -> $END) ===" | tee -a "$LOG"
  env "${DETERMINISTIC_ENV[@]}" torchrun --nproc_per_node 1 --master_port "$((PORT + 1))" \
    pretrain_gpt.py \
    "${CLASSIC_LOAD_FLAGS[@]}" "${COMMON_ARGS[@]}" --num-layers "$NUM_LAYERS" $ARCH \
    --train-iters "$END" --save-interval 1000 \
    --load "$OUT/$TAG" --data-cache-path "/tmp/dc_${M}_src${LAYOUT}_resume$IT" 2>&1 | tee "$OUT/resume_$IT.log"
done

echo "======== [$M/$LAYOUT] VERIFICATION (sharded source -> convert -> classic resume) ========" | tee -a "$LOG"
for IT in 60 80; do
  NX=$((IT + 1))
  echo "--- checkpoint $IT (resumed iter $NX vs sharded-FSDP iter $NX) ---" | tee -a "$LOG"
  grep -h "\[Convert\]" "$OUT/convert_$IT.log" | tee -a "$LOG"
  grep -h "successfully loaded checkpoint" "$OUT/resume_$IT.log" | tee -a "$LOG"
  echo "FSDP($LAYOUT) iter $NX: $(grep -E "iteration +$NX/" "$OUT/train_fsdp.log" | tail -1)" | tee -a "$LOG"
  echo "RESUMED       iter $NX: $(grep -E "iteration +$NX/" "$OUT/resume_$IT.log" | head -1)" | tee -a "$LOG"
done
echo "[$M/$LAYOUT] done — outputs under $OUT" | tee -a "$LOG"
echo "[$M/$LAYOUT] PASS when: the sharded source converts, the classic job loads at"
echo "     60/80, and iter-61/81 lm loss matches the sharded-FSDP loss within bf16 tol (LR exact)."
