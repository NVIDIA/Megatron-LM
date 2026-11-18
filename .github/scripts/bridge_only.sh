#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# An empty, unavailable, or invalid diff must keep CI enabled.
set -euo pipefail
bridge_only=false
if [[ "${BASE_SHA:-}" =~ ^[0-9a-f]{40}$ && "${HEAD_SHA:-}" =~ ^[0-9a-f]{40}$ ]]; then
    diff_file=$(mktemp)
    trap 'rm -f "$diff_file"' EXIT
    separator=..
    if [[ "${USE_MERGE_BASE:-false}" == true ]]; then
        separator=...
    fi
    # Disable rename detection so moving code into Bridge still tests its removal
    # from the original location. NUL delimiters preserve unusual filenames.
    if git diff --name-only --no-renames -z "$BASE_SHA$separator$HEAD_SHA" -- > "$diff_file"; then
        seen=false
        outside=false
        while IFS= read -r -d '' path; do
            seen=true
            case "$path" in
                examples/bridge/*|megatron/bridge/*) ;;
                *) outside=true ;;
            esac
        done < "$diff_file"
        if [[ "$seen" == true && "$outside" == false ]]; then
            bridge_only=true
        fi
    fi
fi
echo "bridge_only=$bridge_only" >> "$GITHUB_OUTPUT"
echo "Bridge-only changes: $bridge_only"
