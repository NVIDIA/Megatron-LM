#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -eu

config=.github/workflows/config/pr-ci-skip-paths.txt
base=$1
head=$2
target=$3
# Use the target branch's current policy, including for PRs that branched earlier.
if ! patterns=$(git show "$target:$config"); then
  echo false
  exit 0
fi
exclusions=()
while IFS= read -r pattern; do
  [[ -z "$pattern" || "$pattern" == \#* ]] && continue
  exclusions+=(":(top,exclude,glob)$pattern")
done <<< "$patterns"

# Disable rename detection so a move across the boundary counts on both sides.
diff=(git diff --quiet --no-renames --ignore-submodules=none "$base" "$head" --)
skip=false
if (( ${#exclusions[@]} )) && "${diff[@]}" "${exclusions[@]}"; then
  # Distinguish a nonempty diff (1) from an empty diff (0) or a Git error (>1).
  status=0
  "${diff[@]}" || status=$?
  if [[ $status == 1 ]]; then
    skip=true
  fi
fi
echo "$skip"
