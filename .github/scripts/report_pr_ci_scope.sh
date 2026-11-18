#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -eu

target=$1
[[ "$target" =~ ^[0-9a-f]{40}$ ]] || exit 0

# Compare the full PR even when the latest push changes only a subset of files.
base=$(gh api "repos/$GITHUB_REPOSITORY/compare/$target...$GITHUB_SHA?per_page=1" \
  --jq '.merge_base_commit.sha') || exit 0
[[ "$base" =~ ^[0-9a-f]{40}$ ]] || exit 0

if git fetch --no-tags --depth=1 --filter=blob:none origin "$base" "$target"; then
  skip=$(bash .github/scripts/pr_ci_skip_paths.sh "$base" "$GITHUB_SHA" "$target")
  echo "skip_builds_and_tests=$skip" >> "$GITHUB_OUTPUT"
else
  echo "::notice::Diff base unavailable. Running normal CI."
fi
