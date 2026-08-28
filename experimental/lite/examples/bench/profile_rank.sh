#!/usr/bin/env bash
set -euo pipefail

script_dir="$(readlink -m "$(dirname "${BASH_SOURCE[0]}")")"
global_rank="${RANK:-${SLURM_PROCID:-0}}"

if [[ "${global_rank}" != "0" ]]; then
  exec python3 "${script_dir}/bench.py" "$@"
fi

output_dir="$(readlink -m "${NSYS_OUTPUT_DIR:-${script_dir}/nsys}")"
mkdir -p "${output_dir}"
report_base="${output_dir}/${NSYS_REPORT_NAME:-mlite-bench-rank0}"

export MLITE_NSYS_CAPTURE=1
export MLITE_STEP_NVTX=1
nsys profile \
  --trace=cuda,nvtx \
  --sample=none \
  --cpuctxsw=none \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --force-overwrite=true \
  --output="${report_base}" \
  python3 "${script_dir}/bench.py" "$@"

test -s "${report_base}.nsys-rep"
nsys export \
  --type=sqlite \
  --force-overwrite=true \
  --output="${report_base}.sqlite" \
  "${report_base}.nsys-rep"
test -s "${report_base}.sqlite"
