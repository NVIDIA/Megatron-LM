#! /bin/bash

# step 1 : OBTAINING THE COMMAND LINE ARGUMENTS
echo "------- ARGUMENTS LIST --------"
for ARGUMENT in "$@"
do
   KEY=$(echo $ARGUMENT | cut -f1 -d=)

   KEY_LENGTH=${#KEY}
   VALUE="${ARGUMENT:$KEY_LENGTH+1}"

   export "$KEY"="$VALUE"
   echo "$KEY=$VALUE"
done
echo "---------------------------------"

export BUILD_DIR=`pwd` #Path to megatron-lm repo

# step 2 : SETTING RUN NAME
export RUN_NAME=resume_${RUN_MODEL}_tp${TP_SIZE}_pp${PP_SIZE}_${NUM_NODES}nodes
echo "----------------- DEBUG FOLDER INFORMATION ---------------------------"
echo "In case of error check ${SELENE_ADLR_CI_PATH}/${CI_PIPELINE_ID}/${RUN_NAME}/debug for result logs."
echo "Run name is $RUN_NAME"
echo "----------------------------------------------------------------------"

# step 3 : CREATING REQUIRED DIRECTORIES
mkdir -p $SELENE_ADLR_CI_PATH/$CI_PIPELINE_ID/$RUN_NAME/checkpoints
mkdir -p $SELENE_ADLR_CI_PATH/$CI_PIPELINE_ID/$RUN_NAME/tensorboard_logs
mkdir -p $SELENE_ADLR_CI_PATH/$CI_PIPELINE_ID/$RUN_NAME/debug
rm -rf $SELENE_ADLR_CI_PATH/$CI_PIPELINE_ID/$RUN_NAME/checkpoints/*
rm -rf $SELENE_ADLR_CI_PATH/$CI_PIPELINE_ID/$RUN_NAME/tensorboard_logs/*
rm -rf $SELENE_ADLR_CI_PATH/$CI_PIPELINE_ID/$RUN_NAME/debug/*

# step 4 : EXPORTING SOME ENV VARIABLES 
export BASE_DIR=$SELENE_ADLR_CI_PATH/$CI_PIPELINE_ID/$RUN_NAME
export LOGS_DIR=$BASE_DIR/tensorboard_logs
export OMP_NUM_THREADS=2
export GOTO_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2

# step 5 : CREATING A COPY OF THE SBATCH SCRIPT THAT WILL BE RUN FOR DEBUGGING
envsubst '$BASE_DIR $PYTORCH_IMAGE $BUILD_DIR $DATA_DIR $MBS $GBS $TP_SIZE $PP_SIZE $VP_SIZE $NUM_NODES $MAX_STEPS'  <$BUILD_DIR/tests/functional_tests/test_scripts/$RUN_MODEL/sbatch_${RUN_MODEL}_distributed_resume_checkpoint_test.sh > $SELENE_ADLR_CI_PATH/$CI_PIPELINE_ID/$RUN_NAME/debug/sbatch_${RUN_MODEL}_distributed_resume_checkpoint_test.sh

# step 6 : SUBMITTING THE JOB
sbatch_submission=`sbatch -t $TIME_LIMIT $BUILD_DIR/tests/functional_tests/test_scripts/$RUN_MODEL/sbatch_${RUN_MODEL}_distributed_resume_checkpoint_test.sh --export=BASE_DIR,BUILD_DIR,DATA_DIR,TP_SIZE,PP_SIZE,VP_SIZE,NUM_NODES,PYTORCH_IMAGE`
export SLURM_JOBID=$(echo $sbatch_submission| grep 'Submitted batch job' | awk '{ print $4 }');

# step 7 : WAITING FOR JOB TO COMPLETE AND PRINTING JOB INFO
bash $BUILD_DIR/tests/functional_tests/shell_test_utils/jobwait.sh $SLURM_JOBID
echo "--------------- JOB INFO ---------------"
scontrol show job=$SLURM_JOBID
echo "---------------------------------------"
# Gitlab logs collapsible section markers
echo -e "\e[0Ksection_end:`date +%s`:slurm_setup\r\e[0K"
# Follow output of the job
echo "Finished job"
export SLURM_STATE=$(sacct -j "${SLURM_JOBID}" --format State --parsable2 --noheader |& head -n 1)
echo "Slurm job state $SLURM_STATE"
if [[ "$SLURM_STATE" != "COMPLETED" ]]; then echo "Slurm job did not complete. See ${SELENE_ADLR_CI_PATH}/${CI_PIPELINE_ID}/${RUN_NAME}/debug directory for result logs. Skipping pytest."; exit 1; fi

# step 8 : COMPARING THE GROUND TRUTH VALUES TO THE OBTAINED VALUES FROM THE JOB
source $PYTHON_VIRTUAL_ENV
PYTEST_EXIT=0
pytest $BUILD_DIR/tests/functional_tests/python_test_utils/test_resume_checkpoint_pipeline.py || PYTEST_EXIT=$?
if [[ $PYTEST_EXIT == 0 ]]; then echo "Pytest succeded"; else echo "Pytest failed. See ${SELENE_ADLR_CI_PATH}/${CI_PIPELINE_ID}/${RUN_NAME}/debug directory for result logs"; exit $PYTEST_EXIT; fi
