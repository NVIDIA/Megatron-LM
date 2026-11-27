#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail
cd /opt/megatron-lm
export UV_PROJECT_ENVIRONMENT=/opt/venv
if [ "${MFSDP_BASELINE_ONLY:-false}" = "true" ]; then
  result=/workspace/results/github-baseline-50
  mkdir -p "$result"
  echo '{"label":"github-baseline-50","steps":50,"trace":false}' > "$result/request.json"
  uv run --no-sync python /workspace/payload.py "$result/request.json" > "$result/payload.log" 2>&1
  cat "$result/runtime.json"
  cat "$result/result.json"
  exit
fi
if [ "${MFSDP_HEURISTIC_SCAN:-false}" = "true" ]; then
  mkdir -p /workspace/results/heuristic-scan
  g++ -std=c++17 -O2 /workspace/scan_heuristics.cpp -o /workspace/scan-heuristics \
    -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcublasLt -lcudart
  /workspace/scan-heuristics /workspace/results/heuristic-scan/choices.tsv
  sha256sum /usr/local/cuda/lib64/libcublasLt.so.13 > /workspace/results/heuristic-scan/library-sha256.txt
  exit
fi
prefix=${MFS_DP_TRACE_PREFIX:-github}
snapshot="/workspace/results/$prefix-traced-4/collectives/gemm-snapshot-rank0"
if [ -f "$snapshot/manifest.json" ]; then
  CUDA_VISIBLE_DEVICES=0 uv run --no-sync python /workspace/replay_gemm.py "$snapshot"
  exit
fi
for label in "$prefix-baseline-4" "$prefix-traced-4"; do
  uv run --no-sync python - "$label" <<'PYCODE'
import json
import sys
from pathlib import Path
label = sys.argv[1]
result = Path('/workspace/results') / label
result.mkdir(parents=True, exist_ok=False)
(result / 'request.json').write_text(json.dumps({
    'label': label, 'steps': 4, 'trace': '-traced-' in label,
    'env': {'MFSDP_CAPTURE_GEMM': 'general_grouped_gemm:14'} if '-traced-' in label else {},
}))
PYCODE
  uv run --no-sync python /workspace/payload.py \
    "/workspace/results/$label/request.json" > "/workspace/results/$label/payload.log" 2>&1
  cat "/workspace/results/$label/result.json"
done
CUDA_VISIBLE_DEVICES=0 uv run --no-sync python /workspace/replay_gemm.py \
  "/workspace/results/$prefix-traced-4/collectives/gemm-snapshot-rank0"
