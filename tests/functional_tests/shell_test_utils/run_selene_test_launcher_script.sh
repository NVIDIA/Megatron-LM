#! /bin/bash

# step 1 : OBTAINING THE COMMAND LINE ARGUMENTS
echo "------ARGUMENTS LIST --------"
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
if [[ $USE_CORE -eq 1 && $USE_TE -eq 1 ]]; then
    echo "Cannot run megatron core and transformer engine together"
    exit 1
fi

# step 2 : SETTING RUN NAME
RUN_NAME=${RUN_MODEL}_tp${TP_SIZE}_pp${PP_SIZE}_${NUM_NODES}nodes_${MAX_STEPS}steps
if [[ $USE_TE == 1 ]]; then RUN_NAME=${RUN_NAME}_te_enabled; fi
if [[ $USE_CORE == 1 ]]; then RUN_NAME=${RUN_NAME}_core_enabled; fi
if [[ -n $METADATA ]]; then RUN_NAME=${RUN_NAME}_${METADATA}; fi
export $RUN_NAME
echo "----------------- DEBUG FOLDER INFORMATION ---------------------------"
echo "In case of error check ${SELENE_ADLR_CI_PATH}/${CI_PIPELINE_ID}/${RUN_NAME}/debug directory for result logs."
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
envsubst '$BASE_DIR $PYTORCH_IMAGE $BUILD_DIR $DATA_DIR $VP_SIZE $MBS $GBS $ADDITIONAL_PARAMS $USE_TE $TP_SIZE $PP_SIZE $NUM_NODES $MAX_STEPS $USE_CORE' <$BUILD_DIR/tests/functional_tests/test_scripts/$RUN_MODEL/sbatch_${RUN_MODEL}_distributed_test.sh > $SELENE_ADLR_CI_PATH/$CI_PIPELINE_ID/$RUN_NAME/debug/sbatch_${RUN_MODEL}_distributed_test.sh

# step 6 : SUBMITTING THE JOB
sbatch_submission=`sbatch $BUILD_DIR/tests/functional_tests/test_scripts/$RUN_MODEL/sbatch_${RUN_MODEL}_distributed_test.sh --export=BASE_DIR,BUILD_DIR,DATA_DIR,USE_TE,TP_SIZE,PP_SIZE,NUM_NODES,MAX_STEPS,VP_SIZE,MBS,GBS,PYTORCH_IMAGE,ADDITIONAL_PARAMS`
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
echo "Slurm log dump start ------------------------------------------------------------"
cat $SELENE_ADLR_CI_PATH/$CI_PIPELINE_ID/$RUN_NAME/debug/slurm*
echo "Slurm log dump end --------------------------------------------------------------"
python3 $BUILD_DIR/tests/functional_tests/python_test_utils/check_slurm_job_completion.py $SLURM_JOBID
if [ $? -ne 0 ]; then echo "Slurm job did not complete. See ${SELENE_ADLR_CI_PATH}/${CI_PIPELINE_ID}/${RUN_NAME}/debug directory for result logs. Skipping pytest."; exit 1; fi

# step 8 : DISPLAYING THE GROUND TRUTH INFO FOR DEBUGGING OR UPDATING GROUND TRUTH VALUES
source $PYTHON_VIRTUAL_ENV
if [[ "$DISPLAY_OUTPUT" == "True" ]]; then
    python3 $BUILD_DIR/tests/functional_tests/python_test_utils/get_test_results_from_tensorboard_logs.py $LOGS_DIR $RUN_NAME
fi

# step 9 : COMPARING THE GROUND TRUTH VALUES TO THE OBTAINED VALUES FROM THE JOB
export EXPECTED_METRICS_FILE=$BUILD_DIR/tests/functional_tests/test_results/$RUN_MODEL/$RUN_NAME.json
PYTEST_EXIT=0
pytest $BUILD_DIR/tests/functional_tests/python_test_utils/test_ci_pipeline.py || PYTEST_EXIT=$?
if [[ $PYTEST_EXIT == 0 ]]; then echo "Pytest succeded"; else echo "Pytest failed. See ${SELENE_ADLR_CI_PATH}/${CI_PIPELINE_ID}/${RUN_NAME}/debug directory for result logs"; exit $PYTEST_EXIT; fi