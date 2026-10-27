#!/usr/bin/env bash
# One-shot parity run: fixture -> HF vs Megatron comparison.
#
#   run_parity.sh <run-dir> [extra run_parity.py args]
#
# Env:
#   QWEN4_TRANSFORMERS   extra PYTHONPATH entry holding a transformers build with qwen4_exp
#   REUSE_FIXTURE=1      skip fixture generation and reuse <run-dir>
#
# Run this inside the training container on one GPU. It does not allocate anything itself:
# submit it with srun/sbatch, do not hold an interactive allocation open around it.
set -uo pipefail

RUN="${1:?usage: run_parity.sh <run-dir> [args...]}"; shift || true
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MLM="$(cd "$HERE/../../.." && pwd)"

# ---------------------------------------------------------------------------------------
# These two are not optional.
#
# TransformerEngine asks cuBLAS for CUBLAS_COMPUTE_32F_FAST_TF32 on every pure-fp32 GEMM, and
# Triton's tl.dot defaults to TF32 for fp32 inputs. Either one puts ~1e-3 of noise under every
# linear layer -- three orders of magnitude above the 1e-6 this comparison is trying to resolve,
# so the run would "pass" its loose checks and be meaningless.
#
# They must be set for the HF process too, not just Megatron's: when `fla` is importable, HF's
# gated-delta-rule dispatches to the same Triton kernel. Missing this once cost half a day of
# chasing an 8.5e-4 "divergence" in a GDN layer that was pure TF32 noise.
# ---------------------------------------------------------------------------------------
export NVIDIA_TF32_OVERRIDE=0
export TRITON_F32_DEFAULT=ieee
export CUDA_DEVICE_MAX_CONNECTIONS=1

export PYTHONPATH="${QWEN4_TRANSFORMERS:+$QWEN4_TRANSFORMERS:}$MLM"
cd "$MLM" || exit 2

mkdir -p "$RUN"
echo "=== environment ==="
python3 -c "import torch, transformers; print('torch', torch.__version__, '| transformers', transformers.__version__)"
python3 -c "
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpForCausalLM
print('transformers provides qwen4_exp: OK')
" || {
  echo "This transformers build has no qwen4_exp. Install one that does and point" >&2
  echo "QWEN4_TRANSFORMERS at it (see README)." >&2
  exit 3
}
python3 -c "import megatron.core; print('megatron.core OK')" || exit 4
git -C "$MLM" log -1 --format='megatron %h %s' 2>/dev/null || true

if [ "${REUSE_FIXTURE:-0}" != "1" ]; then
  echo "=== fixture ==="
  python3 "$HERE/make_fixture.py" --out "$RUN" || exit 5
fi

echo "=== parity ==="
python3 "$HERE/run_parity.py" --fixture "$RUN" --out "$RUN/out" "$@"
rc=$?
echo "=== parity rc=$rc ==="
exit $rc
