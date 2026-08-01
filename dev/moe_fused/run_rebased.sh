#!/usr/bin/env bash
# Run the standard in-session e2e benchmark on the main-rebased tree.
#
# The pinned session venv ships a partially-installed sympy (its smtlib module
# exists but sympy.functions.elementary._trigonometric_special does not). That
# was latent until the rebase: current main's megatron/core/utils.py imports
# dist_checkpointing, which pulls torch.distributed.checkpoint ->
# torch.fx.experimental.symbolic_shapes -> sympy at import time, so every rank
# now dies before argument parsing. Shadow the broken copy with a good one from
# an overlay dir on PYTHONPATH rather than mutating the shared venv.
set -uo pipefail

VENV=/lustre/fsw/portfolios/coreai/users/shanmugamr/agents-space/envs/megatron_lm/dd356431262b5db4/.venv
PYBIN=$VENV/bin/python
# Node-local, not lustre: installing sympy's ~1300 small files onto the shared
# filesystem fails partway with [Errno 5] Input/output error, which is almost
# certainly how the venv's copy ended up truncated to begin with. All four ranks
# share this node, so /tmp is visible to every one of them.
FIX=/tmp/cog_fixpkgs

# Which packages are truncated varies by *node*, not just by venv: the same
# Lustre files that import cleanly on one node raise ModuleNotFoundError for a
# submodule on another. Enumerating them by hand cost several cycles (sympy, then
# mpmath, then pydantic, then onnxscript), so converge automatically instead:
# import, read the missing module out of the error, reinstall that distribution
# into the overlay, repeat. Seeds are the ones already known to break, including
# jaraco.functools, which is reached through setuptools' vendored distutils and so
# cannot be routed around on Python 3.12. setuptools stays under 82 because this
# torch build requires that.
mkdir -p "$FIX"
export PYTHONPATH="$FIX:${PYTHONPATH:-}"

_pip_into_overlay() {
  $PYBIN -m pip install --no-cache-dir --target="$FIX" --upgrade "$@" 2>&1 | tail -3
}

if ! $PYBIN -c 'import megatron.core' 2>/dev/null; then
  echo "===== seeding overlay in $FIX ====="
  _pip_into_overlay 'mpmath>=1.3' 'sympy>=1.13' 'networkx>=3.0' 'setuptools<82' \
    'jaraco.functools' 'pydantic>=2.9'
fi

# Bounded, because an unbounded loop on a genuinely absent module would spin
# until the arm's timeout and leave no useful log.
for _attempt in 1 2 3 4 5 6; do
  err=$($PYBIN -c 'import megatron.core' 2>&1) && break
  missing=$(printf '%s\n' "$err" | sed -n "s/.*No module named '\([A-Za-z0-9_]*\).*/\1/p" | tail -1)
  if [ -z "$missing" ]; then
    echo "===== overlay cannot repair this import (not a missing module) ====="
    printf '%s\n' "$err" | tail -25
    exit 1
  fi
  echo "===== overlay repair $_attempt: reinstalling '$missing' ====="
  _pip_into_overlay "$missing" || true
done

echo "===== overlay check ====="
$PYBIN -c 'import megatron.core; print("megatron.core import OK")' || {
  echo "MEGATRON IMPORT FAILED after overlay repair - traceback above"
  exit 1
}

exec bash dev/moe_fused/run_e2e_cfg.sh
