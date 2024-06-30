#!/bin/bash

set -exou pipefail

collect_jet_jobs () {
  PAGE=1
  PER_PAGE=100
  RESULTS="[]"

  while true; do
    # Fetch the paginated results
    RESPONSE=$(curl \
                  -s \
                  --globoff \
                  --header "PRIVATE-TOKEN: $RW_API_TOKEN" \
                  "${ENDPOINT}/pipelines/${JET_PIPELINE_ID}/jobs?page=$PAGE&per_page=$PER_PAGE"
              )
    # Combine the results
    RESULTS=$(jq -s '.[0] + .[1]' <<< "$RESULTS $RESPONSE")

    # Check if there are more pages
    if [[ $(jq 'length' <<< "$RESPONSE") -lt $PER_PAGE ]]; then
      break
    fi

    # Increment the page number
    PAGE=$((PAGE + 1))
  done

  echo "$RESULTS"
}

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <jet-ci-pipeline-id>"
    exit 1
elif [[ -z "${RW_API_TOKEN}" ]]; then
    echo "RW_API_TOKEN empty, get one at https://gitlab-master.nvidia.com/-/user_settings/personal_access_tokens"
    exit 1
fi

CI_PIPELINE_ID=$1
CI_PROJECT_ID=${CI_PROJECT_ID:-19378}

# Fetch Elastic logs
set +x
PIPELINE_JSON=$(curl \
                  --fail \
                  --silent \
                  --header "PRIVATE-TOKEN: ${RW_API_TOKEN}" \
                  "https://gitlab-master.nvidia.com/api/v4/projects/${CI_PROJECT_ID}/pipelines/${CI_PIPELINE_ID}/bridges?per_page=100"
                ) || ret_code=$?
set -x
if [[ ${ret_code:-0} -ne 0 ]]; then
    echo CI_PIPELINE_ID=$CI_PIPELINE_ID does not exist
    exit 1
fi

# Fetch GitLab logs of JET downstream pipeline
DOWNSTREAM_PIPELINE_ID=$(jq '.[0].downstream_pipeline.id' <<< "$PIPELINE_JSON")
set +x
JET_PIPELINE_JSON=$(curl \
                      --fail \
                      --silent \
                      --header "PRIVATE-TOKEN: ${RW_API_TOKEN}" \
                      "${ENDPOINT}/pipelines/${DOWNSTREAM_PIPELINE_ID}/bridges?per_page=100"
                    )
set -x
JET_PIPELINE_ID=$(jq '.[0].downstream_pipeline.id' <<< "$JET_PIPELINE_JSON")

set +x
JET_LOGS=$(collect_jet_jobs)
set -x

LAST_STAGE_TEST_JOBS=$(jq \
  --arg ENDPOINT ${ENDPOINT} '[
    .[] 
    | select(.name | contains("3 logs_after"))
    | select(.name | startswith("build/") | not)
    | {
        name, 
        retry_url: ($ENDPOINT + "/jobs/" + (.id | tostring) + "/retry")
      }
  ] | unique_by(.name)' <<< "$JET_LOGS"
)

NUM_LAST_STAGE_TEST_JOBS=$(jq length <<< $LAST_STAGE_TEST_JOBS)

set +x
i=1
for retry_url in $(jq -r '.[].retry_url' <<< "$LAST_STAGE_TEST_JOBS"); do
  RES=$(curl \
          --silent \
          --request POST \
          --header "PRIVATE-TOKEN: $RW_API_TOKEN" \
          "$retry_url"
        ) || ret_code=$?
  if [[ ${ret_code:-0} -ne 0 ]]; then
      echo "Failed to retry $retry_url"
      exit 1
  fi
  echo "($i / $NUM_LAST_STAGE_TEST_JOBS) Retried $retry_url successfully"
  i=$(($i + 1))
done
set -x

# Wait until all jobs completed
count_active_jobs () {
  JET_LOGS=$(collect_jet_jobs)

  echo $(jq '[.[] | select((.status == "running") or (.status == "pending"))] | length' <<< "$JET_LOGS")
}

set +x
while true; do
  active_jobs=$(count_active_jobs)
  echo "Active jobs $active_jobs"

  if [[ "$active_jobs" -eq 0 ]]; then
    break
  fi
  sleep 15
done
set -x