#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

classifier="$(cd "$(dirname "$0")" && pwd)/pr_ci_skip_paths.sh"
config=.github/workflows/config/pr-ci-skip-paths.txt
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
  local expected=$1 base=$2 head=$3 target=${4:-$2} actual
  actual=$(bash "$classifier" "$base" "$head" "$target")
  if [[ "$actual" != "$expected" ]]; then
    echo "Expected $expected for $base..$head with policy at $target, got '$actual'" >&2
    exit 1
  fi
}

echo "shared code" > shared.py
missing=$(commit "Create baseline without a skip list")
mkdir -p examples/bridge/nested megatron/bridge
echo "example" > examples/bridge/fixture.py
without_list=$(commit "Change only Bridge before the list exists")
expect false "$missing" "$without_list"
mkdir -p .github/workflows/config
: > "$config"
empty=$(commit "Create an empty skip list")
echo "updated example" >> examples/bridge/fixture.py
with_empty_list=$(commit "Change only Bridge with an empty list")
expect false "$empty" "$with_empty_list"
printf '%s\n' 'examples/bridge/**' 'megatron/bridge/**' > "$config"
base=$(commit "Configure Bridge paths")
expect false "$base" "$base"
expect false 0000000000000000000000000000000000000000 "$base"
expect false "$base" 0000000000000000000000000000000000000000

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

# The proposed list would match both changes. Only a later PR may use it.
printf '\n# More skipped paths\nfuture component/**\n%s\n' "$config" >> "$config"
mkdir -p 'future component'
echo "new component" > 'future component/example.py'
expanded=$(commit "Expand the skip list and add a component")
expect false "$deleted" "$expanded"

# An existing PR picks up the target's new policy without including unrelated
# target-branch changes in the PR diff.
echo "unrelated target change" >> shared.py
target=$(commit "Update shared code on target")
git checkout -q --detach "$deleted"
mkdir -p 'future component'
echo "another example" > 'future component/another.py'
topic=$(commit "Add an example on a topic branch")
expect false "$deleted" "$topic"
expect true "$deleted" "$topic" "$target"
