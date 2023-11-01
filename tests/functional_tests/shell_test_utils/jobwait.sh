#! /bin/bash

JOBID=$1
echo "Job id : $JOBID"

if [[ $JOBID -eq "" ]]; then
  exit 1
fi

sleep 10s

while true; do
    export STATE=`sacct -j $JOBID --format State --parsable2 --noheader |& head -n 1`
    case "${STATE}" in
        PENDING|RUNNING|REQUEUED)
            echo "Job is still in $STATE"
            sleep 15s
            ;;
        *)
            sleep 30s
            echo "Exiting with SLURM job status '${STATE}'"
            exit 0
            ;;
    esac
done
