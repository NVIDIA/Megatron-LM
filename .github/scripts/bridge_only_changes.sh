#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -eu

# Disable rename detection so a move across the boundary counts on both sides.
diff=(git diff --quiet --no-renames --ignore-submodules=none "$1" "$2" --)
bridge_only=false
if "${diff[@]}" ':(exclude,glob)examples/bridge/**' ':(exclude,glob)megatron/bridge/**'; then
  # Distinguish a nonempty diff (1) from an empty diff (0) or a Git error (>1).
  status=0
  "${diff[@]}" || status=$?
  if [[ $status == 1 ]]; then
    bridge_only=true
  fi
fi
echo "$bridge_only"
