#!/usr/bin/env bash
# Adapted from https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/08035723e7a86f94fc54a7f4473bd0519f2ac941/scripts/conversion/convert.sh
# Source license: Apache-2.0.
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

executor=local
expect_executor_value=false
for argument in "$@"; do
    if [[ ${expect_executor_value} == true ]]; then
        executor=${argument}
        expect_executor_value=false
        continue
    fi
    case "${argument}" in
        --executor)
            expect_executor_value=true
            ;;
        --executor=*)
            executor=${argument#--executor=}
            ;;
    esac
done

if [[ ${expect_executor_value} == true ]]; then
    echo "--executor requires a value" >&2
    exit 2
fi

if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    # Use NeMo Run from the pre-populated environment (for example, the CI container).
    UV_ARGS=(--no-project --active)
elif [[ ${executor} == slurm ]]; then
    # The Slurm head-node launcher only needs NeMo Run; model dependencies live
    # in the submitted container.
    UV_ARGS=(--no-project --with nemo-run==0.10.0)
else
    # Local execution runs the conversion worker in the project environment.
    UV_ARGS=(--with nemo-run==0.10.0)
fi

exec uv run "${UV_ARGS[@]}" python "${SCRIPT_DIR}/setup_conversion.py" "$@"
