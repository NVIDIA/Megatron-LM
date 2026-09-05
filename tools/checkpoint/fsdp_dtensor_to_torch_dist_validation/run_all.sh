#!/bin/bash
# Run the reverse-converter validation across the model coverage set.
#
#   bash run_all.sh                          # resume-continuity for the 6-family set
#   bash run_all.sh --with-bitexact          # also the bit-exact diff (iter 80) per model
#   bash run_all.sh --with-reshard           # also the load-side reshard sweep (>=2 GPUs)
#   bash run_all.sh --with-source-sharding   # also train a DP2-sharded source (>=2 GPUs)
#   bash run_all.sh --models "dense mtp"     # a custom subset (any of: list-models)
#
# The default set is the six converter-transform families. The two extra
# archetypes (moe_mla_mtp, dense_fp8) are not run by default — pass them via
# --models. Each model runs independently; a roll-up prints at the end.
#
# NOTE: a roll-up entry of "ok" means the scripts ran to completion (checkpoint
# loaded, no crash). The numeric verdicts — resumed lm-loss ~= FSDP loss within
# bf16 tolerance, LR exact, and an empty bit-exact diff — are in each model's
# VERIFICATION block and diff output. See ./README.md.
set -uo pipefail

_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$_DIR/common.sh"

MODELS="dense dense_swiglu moe_grouped moe_gated mtp gdn_hybrid"
WITH_BITEXACT=0
WITH_RESHARD=0
WITH_SOURCE=0

while [ $# -gt 0 ]; do
  case "$1" in
    --models)              MODELS="$2"; shift 2 ;;
    --with-bitexact)       WITH_BITEXACT=1; shift ;;
    --with-reshard)        WITH_RESHARD=1; shift ;;
    --with-source-sharding) WITH_SOURCE=1; shift ;;
    -h|--help)             grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

declare -A STATUS
for M in $MODELS; do
  echo ""
  echo "################## $M ##################"
  if bash "$_DIR/validate_resume.sh" "$M"; then
    STATUS[$M]="resume:ok"
  else
    STATUS[$M]="resume:ERROR"; continue
  fi
  if [ "$WITH_BITEXACT" = 1 ]; then
    if RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 \
         python "$_DIR/validate_bitexact.py" "$M" --iter 80; then
      STATUS[$M]="${STATUS[$M]} bitexact:ran"
    else
      STATUS[$M]="${STATUS[$M]} bitexact:ERROR"
    fi
  fi
  if [ "$WITH_RESHARD" = 1 ]; then
    if bash "$_DIR/validate_reshard.sh" "$M"; then
      STATUS[$M]="${STATUS[$M]} reshard:ran"
    else
      STATUS[$M]="${STATUS[$M]} reshard:ERROR"
    fi
  fi
  if [ "$WITH_SOURCE" = 1 ]; then
    if bash "$_DIR/validate_source_sharding.sh" "$M" DP2; then
      STATUS[$M]="${STATUS[$M]} src_dp2:ran"
    else
      STATUS[$M]="${STATUS[$M]} src_dp2:ERROR"
    fi
  fi
done

echo ""
echo "======== ROLL-UP (see each model's VERIFICATION block for the numeric verdict) ========"
for M in $MODELS; do
  printf '  %-14s %s\n' "$M" "${STATUS[$M]:-not-run}"
done
