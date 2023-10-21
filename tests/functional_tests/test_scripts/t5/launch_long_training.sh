SCRIPT_PATH="/lustre/fsw/joc/huvu/codes/T5_mcore/megatron-lm-updated/megatron-lm/tests/functional_tests/test_scripts/t5/sbatch_t5_distributed.sh"
EXPERIMENT_NAME="t5-sbatch_final_pile_multinodes_fullPile_checkpoint"

# first job
jobname=${EXPERIMENT_NAME}-1
jobid=$(sbatch --account=llmservice_dev_mcore --job-name=llmservice_dev_mcore-run:${jobname} ${SCRIPT_PATH})
prev_jobname=$jobname
echo "Submitted"
echo $jobname
echo $jobid

# subsequent jobs
for i in {2..5}; do
        jobname=${EXPERIMENT_NAME}-${i}
        jobid=$(sbatch --account=llmservice_dev_mcore --job-name=llmservice_dev_mcore-run:${jobname} --dependency=afternotok:${jobid##* } ${SCRIPT_PATH})
        echo "Submitted"
        echo $jobname
        echo $jobid
        done