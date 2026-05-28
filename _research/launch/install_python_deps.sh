#!/bin/bash
set -euo pipefail
PKG_DIR=${1:?usage: install_python_deps.sh PKG_DIR}
MARKER=$PKG_DIR/.deps_installed_v12
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

# $PKG_DIR is prepended to PYTHONPATH, so anything installed here shadows the
# container's matched torch/transformers/hf_hub stack and breaks imports. Install
# only container-absent packages; --no-deps everything below so none drags torch.
$INSTALL --target="$PKG_DIR" wandb omegaconf
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
$INSTALL --no-deps --target="$PKG_DIR" 'tilelang'

# mamba_ssm + causal_conv1d for Mamba 2 / Mamba 3 baselines (state-spaces/mamba).
# v2.3.1 ships Mamba 3 (March 2026 release). Both packages compile CUDA kernels
# at install time on first use; on ARM64 GH200 this can take 5-10 min and may
# need to fall back to PyTorch-only paths if wheels aren't available. Used by
# pretrain_hybrid.py + hybrid_stack_spec; ignored by pretrain_gpt.py runs.
# `|| true` so a Mamba install failure doesn't block the deltanet runs that
# don't need it.
# MAMBA_SKIP_CUDA_BUILD=TRUE lets the install skip CUDA extension compilation
# (which fails on ARM64 GH200 due to missing prebuilt wheels). The package
# falls back to PyTorch/Triton paths inside the layer.
# Capture stderr to a log file so failures aren't silently swallowed.
MAMBA_LOG=$PKG_DIR/.mamba_install.log
{
    echo "=== causal_conv1d ==="
    CAUSAL_CONV1D_SKIP_CUDA_BUILD=TRUE CAUSAL_CONV1D_FORCE_BUILD=FALSE \
        pip install --no-deps --target="$PKG_DIR" 'causal_conv1d>=1.5.0' 2>&1 || echo "[install failed: causal_conv1d]"
    echo "=== mamba_ssm ==="
    # --no-build-isolation: mamba_ssm's setup.py imports torch to read CUDA version,
    # which fails inside pip's isolated build env (no torch there). Use system torch.
    MAMBA_SKIP_CUDA_BUILD=TRUE MAMBA_FORCE_BUILD=FALSE \
        pip install --no-build-isolation --no-deps --target="$PKG_DIR" 'mamba_ssm>=2.3.1' 2>&1 || echo "[install failed: mamba_ssm]"
} > "$MAMBA_LOG" 2>&1
echo "Mamba install log: $MAMBA_LOG (tail -20):"
tail -20 "$MAMBA_LOG" || true

touch "$MARKER"
