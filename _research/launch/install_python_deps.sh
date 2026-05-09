#!/bin/bash
set -euo pipefail
PKG_DIR=${1:?usage: install_python_deps.sh PKG_DIR}
MARKER=$PKG_DIR/.deps_installed_v7
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
# fla-core is the actual implementation package (the meta-package
# flash-linear-attention only ships fla/layers/ and fla/models/; the kernels
# fla/{modules,ops,__init__.py,utils.py} live in fla-core==0.5.0). With
# --no-deps the meta-package would leave fla broken, so we pin fla-core
# directly. Provides chunk_delta_rule (Schlag DeltaNet), chunk_gated_delta_rule
# (GDN), and chunk_kda (Kimi Delta Attention).
$INSTALL --no-deps --target="$PKG_DIR" 'fla-core==0.5.0'
# tilelang provides an alternate backend FLA uses on Hopper to dodge a known
# Triton 3.4+ bug in chunk_bwd_dqkwg that produces incorrect gradients for
# chunk_gated_delta_rule. Without tilelang the backward raises at iter 1.
# See fla-org/flash-linear-attention#640.
$INSTALL --target="$PKG_DIR" 'tilelang'

touch "$MARKER"
