#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

classifier="$(cd "$(dirname "$0")" && pwd)/bridge_only_changes.sh"
repo=$(mktemp -d)
trap 'rm -rf "$repo"' EXIT
cd "$repo"
git init -q
git config user.name "CI test"
git config user.email "ci@example.org"
git config commit.gpgSign false
git config core.hooksPath /dev/null

commit() {
  git add -A
  git commit -qm "$1"
  git rev-parse HEAD
}

expect() {
  local expected=$1 base=$2 head=$3 actual
  actual=$(bash "$classifier" "$base" "$head")
  if [[ "$actual" != "$expected" ]]; then
    echo "Expected $expected for $base..$head, got '$actual'" >&2
    exit 1
  fi
}

echo "shared code" > shared.py
base=$(commit "Create baseline")
expect false "$base" "$base"
expect false 0000000000000000000000000000000000000000 "$base"

mkdir -p examples/bridge/nested megatron/bridge
echo "example" > $'examples/bridge/nested/line\nbreak.py'
example=$(commit "Add Bridge example")
expect true "$base" "$example"
echo "model" > megatron/bridge/model.py
model=$(commit "Add Bridge model")
expect true "$example" "$model"
expect true "$base" "$model"

mkdir -p megatron/bridge_extra
git mv megatron/bridge/model.py megatron/bridge_extra/model.py
outside=$(commit "Move model outside Bridge")
expect false "$model" "$outside"
git mv megatron/bridge_extra/model.py examples/bridge/model.py
inside=$(commit "Move model into Bridge")
expect false "$outside" "$inside"
expect true "$model" "$inside"

echo "changed" >> shared.py
shared=$(commit "Change shared code")
git rm -q examples/bridge/model.py
deleted=$(commit "Remove Bridge model")
expect true "$shared" "$deleted"
expect false "$inside" "$deleted"
