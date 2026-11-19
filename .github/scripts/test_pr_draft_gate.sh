#!/usr/bin/env bash
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

set -euo pipefail

cd "$(dirname "$0")"
export GITHUB_REPOSITORY=NVIDIA/Megatron-LM PR_NUMBER=123

gh() { cat; }
export -f gh

pr='{"state":"OPEN","isDraft":false,"headRefOid":"latest","labels":{"nodes":[]},
     "reviews":{"nodes":[]},"timelineItems":{"totalCount":0}}'
approved='.reviews.nodes = [{"state":"APPROVED","commit":{"oid":"latest"}}]'
exempt='.labels.nodes = [{"name":"skip-bot-approval"}]'

check() {
    local action=$1 expected=$2 fixture=$3 actual
    actual=$(jq -c "($fixture) | {data: {repository: {pullRequest: .}}}" <<< "$pr" |
        PR_ACTION="$action" bash pr_draft_gate.sh)
    if [[ "$actual" != "$expected" ]]; then
        echo "$action $fixture: expected $expected, got $actual" >&2
        exit 1
    fi
}

check opened draft "."
check opened draft "$exempt"
check opened ignore ".isDraft = true"
check opened ignore ".timelineItems.totalCount = 1"
check ready_for_review draft "."
check ready_for_review ready "$approved"
check ready_for_review draft "$approved | .headRefOid = \"updated\""
check ready_for_review draft "$approved | .reviews.nodes[0].state = \"DISMISSED\""
check ready_for_review draft "$approved | .reviews.nodes[0].state = \"CHANGES_REQUESTED\""
check ready_for_review ready "$exempt"
check ready_for_review ready "., ($exempt)"
check ready_for_review ignore "$exempt | .isDraft = true"
check ready_for_review ignore "$approved | .state = \"CLOSED\""
echo "Draft gate cases passed."
