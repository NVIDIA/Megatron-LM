#!/usr/bin/env bash
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

set -euo pipefail

# shellcheck disable=SC2016
decision=$(gh api graphql --paginate -F owner="${GITHUB_REPOSITORY%/*}" -F repo="${GITHUB_REPOSITORY#*/}" \
    -F number="$PR_NUMBER" -f query='
    query($owner: String!, $repo: String!, $number: Int!, $endCursor: String) {
      repository(owner: $owner, name: $repo) {
        pullRequest(number: $number) {
          state
          isDraft
          headRefOid
          labels(first: 100, after: $endCursor) {
            nodes { name }
            pageInfo { hasNextPage endCursor }
          }
          reviews(author: "svcnvidia-nemo-ci", states: [APPROVED, CHANGES_REQUESTED, DISMISSED], last: 1) {
            nodes { state commit { oid } }
          }
          timelineItems(itemTypes: [READY_FOR_REVIEW_EVENT], last: 1) { totalCount }
        }
      }
    }' | jq -ser --arg action "$PR_ACTION" '
    map(.data.repository.pullRequest) as $pages
    | $pages[-1] as $pr
    | $pr.reviews.nodes[0] as $review
    | if $pr == null then error("Missing pull request")
      elif $pr.state != "OPEN" or $pr.isDraft then "ignore"
      elif $action == "opened" then
        if $pr.timelineItems.totalCount == 0 then "draft" else "ignore" end
      elif any($pages[].labels.nodes[]; .name == "skip-bot-approval") then "ready"
      elif $review.state == "APPROVED" and $review.commit.oid == $pr.headRefOid then "ready"
      else "draft" end')

if [[ "$PR_ACTION" != ready_for_review || "$decision" != draft ]]; then
    echo "$decision"
    exit 0
fi

team_pages=$(GH_TOKEN="${MEMBERS_TOKEN:?}" gh api --paginate \
    "orgs/${GITHUB_REPOSITORY%/*}/teams/mcore-engineers/members?per_page=100")
jq -ser --argjson author_id "${PR_AUTHOR_ID:?}" '
    if length == 0 or any(.[]; type != "array") then error("Missing team membership data")
    elif any(.[][]; .id == $author_id) then "draft"
    else "ready" end' <<< "$team_pages"
