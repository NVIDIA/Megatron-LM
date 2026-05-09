#!/bin/bash
set -euo pipefail
PKG_DIR=${1:?usage: install_python_deps.sh PKG_DIR}
MARKER=$PKG_DIR/.deps_installed_v5
LOCK=$PKG_DIR/.install.lockdir

mkdir -p "$PKG_DIR"
if [ -f "$MARKER" ]; then exit 0; fi

while ! mkdir "$LOCK" 2>/dev/null; do
    sleep 5
    if [ -f "$MARKER" ]; then exit 0; fi
done
trap 'rmdir "$LOCK" 2>/dev/null || true' EXIT

if [ -f "$MARKER" ]; then exit 0; fi

if command -v uv >/dev/null 2>&1; then
    INSTALL="uv pip install --quiet"
else
    INSTALL="pip install --quiet"
fi

$INSTALL --target="$PKG_DIR" transformers wandb omegaconf
$INSTALL --no-deps --target="$PKG_DIR" \
    'git+https://github.com/NVIDIA-NeMo/Emerging-Optimizers.git@v0.2.0'
# flash-linear-attention provides chunk_delta_rule (Schlag DeltaNet),
# chunk_gated_delta_rule (GDN), and chunk_kda (Kimi Delta Attention).
# --no-deps to avoid pulling huggingface-hub/transformers conflicts.
$INSTALL --no-deps --target="$PKG_DIR" 'flash-linear-attention==0.5.0'

touch "$MARKER"
