set -euxo pipefail

collect_jet_jobs () {
  PAGE=1
  PER_PAGE=100
  RESULTS="[]"

  while true; do
    # Fetch the paginated results
    RESPONSE=$(curl \
                  -s \
                  --globoff \
                  --header "PRIVATE-TOKEN: $RO_API_TOKEN" \
                  "https://${GITLAB_ENDPOINT}/api/v4/projects/70847/pipelines/${JET_PIPELINE_ID}/jobs?page=$PAGE&per_page=$PER_PAGE"
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

# Fetch Elastic logs
set +x
PIPELINE_JSON=$(curl \
                  --fail \
                  --silent \
                  --header "PRIVATE-TOKEN: ${RO_API_TOKEN}" \
                  "https://${GITLAB_ENDPOINT}/api/v4/projects/${CI_PROJECT_ID}/pipelines/${CI_PIPELINE_ID}/bridges?per_page=100"
                ) || ret_code=$?
set -x
if [[ ${ret_code:-0} -ne 0 ]]; then
    echo CI_PIPELINE_ID=$CI_PIPELINE_ID does not exist
    exit 1
fi

# Fetch GitLab logs of JET downstream pipeline
DOWNSTREAM_PIPELINE_ID=$(jq '.[0].downstream_pipeline.id' <<< "$PIPELINE_JSON")

PIPELINE_URL=https://${GITLAB_ENDPOINT}/ADLR/megatron-lm/-/pipelines/$CI_PIPELINE_ID
JOB_URL=https://${GITLAB_ENDPOINT}/ADLR/megatron-lm/-/jobs/

if [[ $DOWNSTREAM_PIPELINE_ID == null ]]; then
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
    set +x
    JET_PIPELINE_JSON=$(curl \
                        --fail \
                        --silent \
                        --header "PRIVATE-TOKEN: ${RO_API_TOKEN}" \
                        "https://${GITLAB_ENDPOINT}/api/v4/projects/70847/pipelines/${DOWNSTREAM_PIPELINE_ID}/bridges?per_page=100"
                        )
    set -x
    JET_PIPELINE_ID=$(jq '.[0].downstream_pipeline.id' <<< "$JET_PIPELINE_JSON")

    set +x
    JET_LOGS=$(echo "$(collect_jet_jobs)" \
                | jq '[
                    .[] 
                    | select(.name | startswith("build/") | not)
                    | select(.name | contains("3 logs_after") | not)
                    | select(.name | contains("1 logs_before") | not)
                ]'
            ) 

    FAILED_JET_LOGS=$(echo "$JET_LOGS" \
                | jq --arg GITLAB_ENDPOINT "$GITLAB_ENDPOINT" '[
                    .[] 
                    | select(.status != "success")
                    | {
                        "name": (.name[6:] | split(" ")[0]),
                        id,
                        "url": ("https://" + $GITLAB_ENDPOINT + "/dl/jet/ci/-/jobs/" + (.id | tostring)),
                    }
                ]'
            ) 
    set -x

    for row in $(echo "${FAILED_JET_LOGS}" | jq -r '.[] | @base64'); do
        _jq() {
        echo ${row} | base64 --decode | jq -r ${1}
        }
        JOB_ID=$(_jq '.id')
        SLURM_FAILURE=$(jet \
                                -c -df json -th logs query --raw \
                                -c "obj_status.s_message" \
                                --eq obj_ci.l_job_id "$JOB_ID" \
                            | jq '.[0].obj_status.s_message' \
                            | tr -d '"'
                        )
        FAILED_JET_LOGS=$(echo "$FAILED_JET_LOGS" \
                            | jq \
                                --argjson JOB_ID "$JOB_ID" \
                                --arg SLURM_FAILURE "$SLURM_FAILURE" '
                                    .[] |= ((select(.id==$JOB_ID) += {
                                        "slurm_failure_reason": $SLURM_FAILURE}))
                            ')
    done

    NUM_FAILED=$(echo "$FAILED_JET_LOGS" | jq 'length')
    NUM_TOTAL=$(echo "$JET_LOGS" | jq 'length')

    if [[ $NUM_FAILED -eq 0 ]]; then
        BLOCKS='[
            {                
                "type": "section",
                "text": {            
                    "type": "mrkdwn",
                    "text": "<'$PIPELINE_URL'|Report of '$DATE' ('$CONTEXT')>: All '$NUM_TOTAL' passed :doge3d:"
                }
            },
            {                
                "type": "section",
                "text": {            
                    "type": "mrkdwn",
                    "text": "==============================================="
                }
            }
        ]'
    else
        BLOCKS=$(echo -e "$FAILED_JET_LOGS" \
                    | jq --arg DATE "$DATE" --arg CONTEXT "$CONTEXT" --arg URL "$PIPELINE_URL" --arg NUM_FAILED "$NUM_FAILED" --arg NUM_TOTAL "$NUM_TOTAL" '
                        [
                            {                
                                "type": "section",
                                "text": {            
                                    "type": "mrkdwn",
                                    "text": ("<" + $URL + "|Report of " + $DATE + " (" + $CONTEXT + ")>: " + $NUM_FAILED + " of " + $NUM_TOTAL + " failed :doctorge:")
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
                                        + "\n    SLURM failure reason: \n```" + .slurm_failure_reason[-2000:] + "```"
                                        
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