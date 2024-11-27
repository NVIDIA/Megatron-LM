set -euxo pipefail

collect_jobs () {
  PAGE=1
  PER_PAGE=100
  RESULTS="[]"

  while true; do
    # Fetch the paginated results
    RESPONSE=$(curl \
                  -s \
                  --globoff \
                  --header "PRIVATE-TOKEN: $RO_API_TOKEN" \
                  "https://${GITLAB_ENDPOINT}/api/v4/projects/${CI_PROJECT_ID}/pipelines/${CI_PIPELINE_ID}/jobs?page=$PAGE&per_page=$PER_PAGE"
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

CI_PIPELINE_ID=${1:-16595865}
CI_PROJECT_ID=${CI_PROJECT_ID:-19378}
PIPELINE_URL=https://${GITLAB_ENDPOINT}/ADLR/megatron-lm/-/pipelines/$CI_PIPELINE_ID
JOB_URL=https://${GITLAB_ENDPOINT}/ADLR/megatron-lm/-/jobs/
CONTEXT="unit-tests-extended"

# Fetch Elastic logs
set +x
UNIT_TESTS_JOBS=$(collect_jobs | jq '[.[] | select(.name | startswith("test:pyt"))]')
set -x
if [[ ${ret_code:-0} -ne 0 ]]; then
    echo CI_PIPELINE_ID=$CI_PIPELINE_ID does not exist
    exit 1
fi

if [[ $UNIT_TESTS_JOBS == null ]]; then
    FAILED_JOBS=$(curl \
                    --fail \
                    --silent \
                    --header "PRIVATE-TOKEN: ${RO_API_TOKEN}" \
                    "https://${GITLAB_ENDPOINT}/api/v4/projects/${CI_PROJECT_ID}/pipelines/${CI_PIPELINE_ID}/jobs?per_page=100" \
                  | jq --arg JOB_URL "$JOB_URL" '[.[] | select(.status == "failed") | ("<" + $JOB_URL + (.id | tostring) + "|" + .name + ">")] | join("\n• Job: ")' | tr -d '"')
    curl \
        -X POST \
        -H "Content-type: application/json" \
        --data '
            {
                "blocks": [
                    {                
                        "type": "section",
                        "text": {            
                            "type": "mrkdwn",
                            "text": "<'$PIPELINE_URL'|Report of '$DATE' ('$CONTEXT')>:\n"   
                        }
                    },
                    {                
                        "type": "section",
                        "text": {            
                            "type": "mrkdwn",
                            "text": "\n• Job: '"$FAILED_JOBS"'"   
                        }
                    },
                ]
            
            }' \
        $WEBHOOK_URL

else
    FAILED_JOBS=$(echo -E "$UNIT_TESTS_JOBS" \
                | jq --arg GITLAB_ENDPOINT "$GITLAB_ENDPOINT" --arg JOB_URL "$JOB_URL" '[
                    .[] 
                    | select(.status != "success")
                    | {
                        name,
                        id,
                        "url": ($JOB_URL + (.id | tostring)),
                    }
                ]'
            ) 
    set -x

    for row in $(echo "${FAILED_JOBS}" | jq -r '.[] | @base64'); do
        _jq() {
        echo ${row} | base64 --decode | jq -r ${1}
        }
        JOB_ID=$(_jq '.id')
        FULL_LOG=$(curl \
            --location \
            --header "PRIVATE-TOKEN: ${RO_API_TOKEN}" \
            "https://${GITLAB_ENDPOINT}/api/v4/projects/${CI_PROJECT_ID}/jobs/${JOB_ID}/trace")
        
        if [[ "$FULL_LOG" == *exception* ]]; then 
            LAST_EXCEPTION_POS=$(echo "$FULL_LOG" | grep -o -b 'exception' | tail -1 | cut -d: -f1)
            SHORT_LOG=${FULL_LOG:$LAST_EXCEPTION_POS-500:499}
        else
            SHORT_LOG=${FULL_LOG: -1000}
        fi

        FAILED_JOBS=$(echo "$FAILED_JOBS" \
                    | jq \
                        --argjson JOB_ID "$JOB_ID" \
                        --arg SLURM_FAILURE "$SHORT_LOG" '
                            .[] |= ((select(.id==$JOB_ID) += {
                                "slurm_failure_reason": $SLURM_FAILURE}))
                    ')
    done

    NUM_FAILED=$(echo "$FAILED_JOBS" | jq 'length')
    NUM_TOTAL=$(echo "$UNIT_TESTS_JOBS" | jq 'length')

    if [[ $NUM_FAILED -eq 0 ]]; then
        BLOCKS='[
            {                
                "type": "section",
                "text": {            
                    "type": "mrkdwn",
                    "text": ":doge3d: <'$PIPELINE_URL'|Report of '$DATE' ('$CONTEXT')>: All '$NUM_TOTAL' passed"
                }
            }
        ]'
    else
        BLOCKS=$(echo "$FAILED_JOBS" \
                    | jq --arg DATE "$DATE" --arg CONTEXT "$CONTEXT" --arg URL "$PIPELINE_URL" --arg NUM_FAILED "$NUM_FAILED" --arg NUM_TOTAL "$NUM_TOTAL" '
                        [
                            {                
                                "type": "section",
                                "text": {            
                                    "type": "mrkdwn",
                                    "text": (":doctorge: <" + $URL + "|Report of " + $DATE + " (" + $CONTEXT + ")>: " + $NUM_FAILED + " of " + $NUM_TOTAL + " failed")
                                }
                            }
                        ] + [
                            .[] 
                            | {                
                                "type": "section",
                                "text": {            
                                    "type": "mrkdwn",
                                    "text": (                               
                                        "• Job: <" +.url + "|" + .name + ">"
                                        + "\n    SLURM failure reason: \n```" + .slurm_failure_reason + "```"
                                        
                                    )
                                }
                            }
                        ] + [
                            {                
                                "type": "section",
                                "text": {            
                                    "type": "mrkdwn",
                                    "text": ("===============================================")
                                }
                            }
                        ]'
        )
    fi

    for row in $(echo "${BLOCKS}" | jq -r '.[] | @base64'); do
        _jq() {
            echo ${row} | base64 --decode
        }

        curl \
            -X POST \
            -H "Content-type: application/json" \
            --data '{"blocks": '["$(_jq)"]'}' \
            $WEBHOOK_URL
    done

fi