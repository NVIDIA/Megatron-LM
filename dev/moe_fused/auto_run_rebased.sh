#!/usr/bin/env bash
# Unattended runner for the main-rebased 12-gate benchmark.
#
# Lives on the login node (nohup'd) so it outlives any local agent session.
# Waits for the cog session's Slurm allocation, health-checks the node's Lustre
# client before spending the window, stages the repo at a fresh path, then runs
# the standard e2e benchmark and records the result.
#
# The node health check is not optional: REBASE-S11 burned a full allocation on
# nvl72169-T17, whose Lustre client listed files in directories but could not
# stat or open them, so every import died on a different "missing" module.
set -uo pipefail

BASE=/lustre/fsw/portfolios/coreai/users/shanmugamr/agents-space
HANDLE="${HANDLE:-qwen-rebase}"
S=$BASE/sessions/$HANDLE
SRC=$BASE/workspaces/megatron_lm/3a58dc8a1dfd076f/repo
OUT=$BASE/auto_rebased
mkdir -p "$OUT"
LOG=$OUT/runner.log

log() { echo "[$(date -u +%FT%TZ)] $*" >> "$LOG"; }

submit_exec() {
  # $1 = workdir, $2 = command; echoes the request id
  local wd="$1" cmd="$2" rid t
  rid=$(python3 -c 'import uuid;print(uuid.uuid4().hex)')
  t=$S/exec/.staging-$rid
  mkdir -p "$t"
  {
    echo '#!/usr/bin/env bash'
    echo 'set -uo pipefail'
    echo 'exec_root="${COG_EXEC_ROOT}"'
    echo 'run_root="$exec_root/run"'
    echo 'mkdir -p "$run_root/logs/torchrun"'
    echo 'export RUN_DIR="$run_root"'
    echo 'export TORCHRUN_LOG_DIR="$run_root/logs/torchrun"'
    echo "$cmd"
  } > "$t/command.sh"
  chmod +x "$t/command.sh"
  printf '%s' "$wd" > "$t/workdir"
  python3 -c 'import time;print(int(time.time()*1000))' > "$t/submitted_at"
  mv "$t" "$S/exec/pending/$rid"
  echo "$rid"
}

wait_exec() {
  # $1 = request id, $2 = timeout seconds
  local rid="$1" limit="$2" waited=0
  while [ "$waited" -lt "$limit" ]; do
    [ -f "$S/exec/runs/$rid/exit_code" ] && return 0
    sleep 15
    waited=$((waited + 15))
  done
  return 1
}

log "runner started, waiting for allocation of session $HANDLE"

# ---- 1. wait for the allocation ------------------------------------------
JOB=""
for i in $(seq 1 5760); do   # up to 24h at 15s
  JOB=$(squeue -u "$USER" -h -o '%i %T %j' | awk -v h="cog:session:$HANDLE" '$3==h && $2=="RUNNING"{print $1}' | head -1)
  [ -n "$JOB" ] && break
  sleep 15
done
if [ -z "$JOB" ]; then log "FATAL: no running allocation within 24h"; exit 1; fi
NODE=$(squeue -j "$JOB" -h -o '%N')
log "allocation live: job=$JOB node=$NODE"

# controller needs a moment to come up
sleep 60

# ---- 2. health-check the node's Lustre client ----------------------------
log "health-checking node filesystem"
HRID=$(submit_exec "$SRC" 'python3 -c "
import os
paths = [
    \"megatron/core/transformer/transformer_layer.py\",
    \"megatron/core/inference/contexts/dynamic_context.py\",
    \"megatron/core/inference/moe/vllm_fused_moe.py\",
]
bad = [p for p in paths if not os.path.isfile(p)]
print(\"FSHEALTH\", \"OK\" if not bad else \"BROKEN \" + str(bad))
"')
if ! wait_exec "$HRID" 600; then log "FATAL: health check timed out"; exit 1; fi
HEALTH=$(grep -o 'FSHEALTH .*' "$S/exec/runs/$HRID/stdout.log" 2>/dev/null)
log "health result: ${HEALTH:-<none>}"
if ! echo "$HEALTH" | grep -q 'FSHEALTH OK'; then
  log "FATAL: node $NODE has a broken Lustre client; not spending the window. Requeue on another node."
  exit 2
fi

# ---- 3. stage the repo at a fresh path ----------------------------------
# A fresh path avoids stale dentries if the source tree was rewritten in place.
RUNWS=$BASE/workspaces/megatron_lm/auto_rebased_$(date +%s)/repo
mkdir -p "$(dirname "$RUNWS")"
cp -a "$SRC" "$RUNWS"
find "$RUNWS" -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null
log "staged repo at $RUNWS"

# ---- 4. run the benchmark (unmodified script, for comparability) --------
GATES="MCORE_FUSE_FC1_ACT=1 MCORE_MOE_FUSED_ALIGN=1 MCORE_MOE_GEMM_TUNE=1 MCORE_MOE_FUSED_COUNT=1 MCORE_MOE_SUM_FAST=1 MCORE_ROUTER_FUSED_TOPK=1 MCORE_MOE_FUSED_SCATTER=1 MCORE_INFER_INCR_ATTN_STATE=1 MCORE_INFER_VEC_UPDATE_REQS=1 MCORE_INFER_FAST_POST_PROCESS=1 MCORE_FUSED_QK_NORM=1 MCORE_FUSED_ADD_NORM=1 MCORE_FUSED_ADD_NORM_QKV=1 MCORE_FLASH_ATTN_VERSION=2 MCORE_NVLS_RS_BF16=1"
CMD="SESSION_HANDLE=$HANDLE TAG=rebased12 OSL=1024 BS=256 NITERS=5 NWARMUP=2 $GATES bash dev/moe_fused/run_e2e_cfg.sh"
log "submitting benchmark"
BRID=$(submit_exec "$RUNWS" "$CMD")
log "benchmark request=$BRID stdout=$S/exec/runs/$BRID/stdout.log"
echo "$S/exec/runs/$BRID/stdout.log" > "$OUT/latest_stdout_path"

if ! wait_exec "$BRID" 7200; then log "benchmark did not finish within 2h"; exit 1; fi
log "benchmark finished exit=$(cat "$S/exec/runs/$BRID/exit_code" 2>/dev/null)"
cp "$S/exec/runs/$BRID/stdout.log" "$OUT/rebased12_stdout.log" 2>/dev/null
grep -iE 'throughput|tokens/s|tok/s|avg_latency|TPOT|COHERENCE|PROMPT' \
  "$S/exec/runs/$BRID/stdout.log" > "$OUT/rebased12_summary.txt" 2>/dev/null
log "results at $OUT/rebased12_summary.txt"
