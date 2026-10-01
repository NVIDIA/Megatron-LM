#!/usr/bin/env bash
set -euo pipefail

compute_keys() {
  local base_ref=$1
  local cache_variant=$2
  local pr_number=$3
  local merge_group_head_ref=$4
  local github_ref=$5
  local event_name=$6
  local ref_name=$7

  BASE_REF="${base_ref#refs/heads/}"
  CACHE_VARIANT=$cache_variant
  CACHE_NAMESPACE=$(printf '%s' "$BASE_REF" | tr '/:@' '-' | tr -cd '[:alnum:]_.-')
  if [ -z "$CACHE_NAMESPACE" ]; then
    return 1
  fi

  PR_NUMBER=$pr_number
  if [ "$PR_NUMBER" = "0" ] && [ -n "$merge_group_head_ref" ]; then
    PR_NUMBER=$(printf '%s' "$merge_group_head_ref" | sed -nE 's#.*pr-([0-9]+)-.*#\1#p')
  fi

  BASELINE_KEY="${CACHE_NAMESPACE}-${CACHE_VARIANT}-baseline"
  if [ -n "$PR_NUMBER" ] && [ "$PR_NUMBER" != "0" ]; then
    KEY="${CACHE_NAMESPACE}-${CACHE_VARIANT}-${PR_NUMBER}"
  elif [ "$github_ref" = "refs/heads/$BASE_REF" ] || [ "$event_name" = "schedule" ]; then
    KEY="$BASELINE_KEY"
  else
    BRANCH_NAMESPACE=$(printf '%s' "$ref_name" | tr '/:@' '-' | tr -cd '[:alnum:]_.-')
    KEY="${CACHE_NAMESPACE}-${CACHE_VARIANT}-${BRANCH_NAMESPACE}"
  fi

  if [ "${#KEY}" -gt 100 ]; then
    KEY="${KEY:0:83}-$(printf '%s' "$KEY" | sha256sum | cut -c1-16)"
  fi
  if [ "${#BASELINE_KEY}" -gt 100 ]; then
    BASELINE_KEY="${BASELINE_KEY:0:83}-$(printf '%s' "$BASELINE_KEY" | sha256sum | cut -c1-16)"
  fi
}

assert_keys() {
  local expected_key=$1
  local expected_baseline=$2
  shift 2
  compute_keys "$@"
  test "$KEY" = "$expected_key"
  test "$BASELINE_KEY" = "$expected_baseline"
}

assert_keys main-dev-6077 main-dev-baseline refs/heads/main dev 0 pr-6077-ff6b refs/heads/gh-readonly-queue/main/pr-6077 merge_group gh-readonly-queue/main/pr-6077
assert_keys dev-dev-6072 dev-dev-baseline refs/heads/dev dev 0 pr-6072-95e4 refs/heads/gh-readonly-queue/dev/pr-6072 merge_group gh-readonly-queue/dev/pr-6072
assert_keys main-lts-6159 main-lts-baseline main lts 6159 '' refs/heads/pull-request/6159 push pull-request/6159
assert_keys main-dev-baseline main-dev-baseline main dev 0 '' refs/heads/main schedule main
assert_keys main-dev-deploy-release-1.2 main-dev-baseline main dev 0 '' refs/heads/deploy-release/1.2 push deploy-release/1.2
