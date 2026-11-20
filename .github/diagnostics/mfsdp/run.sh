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
prefix=${MFS_DP_TRACE_PREFIX:-github}
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
}))
PYCODE
  uv run --no-sync python /workspace/payload.py \
    "/workspace/results/$label/request.json" > "/workspace/results/$label/payload.log" 2>&1
  cat "/workspace/results/$label/result.json"
done
