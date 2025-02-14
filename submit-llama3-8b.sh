#!/bin/bash

#SBATCH --account=a-a06
#SBATCH --time=00:19:59
#SBATCH --job-name=llama-8b
#SBATCH --output=/iopsstor/scratch/cscs/%u/Megatron-LM/logs/slurm/training/%x-%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/Megatron-LM/logs/slurm/training/%x-%j.err
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --mem=460000
#SBATCH --environment=/capstor/store/cscs/swissai/a06/containers/NGC-PyTorch/ngc_pt_jan.toml	# Vanilla 25.01 PyTorch NGC Image 
#SBATCH --no-requeue	# Prevent Slurm to requeue the job if the execution crashes (e.g. node failure) so we don't loose the logs

echo "START TIME: $(date)"

################ Configs ################
# Use the FineWeb Edu dataset
DATASETS="/capstor/store/cscs/swissai/a06/datasets_tokenized/nemo/Llama-3.1-70B/fineweb-edu-full-merge"

# 128 * 4096 = 524_288 tokens per batch
# 128 / 1 = 128 forward passes with 2 replicas per node requires 64 nodes such that
# each replica does batch accumulation of 1 and 16 nodes for batch accumulation of 4.
MBS=1
GBS=128
SEQ_LEN=4096 
TRAINING_STEPS=5000
CHECKPOINT_STEPS=1000

#### Debugging ####
LOG_NCCL=false # Log NCCL_DEBUG=info. Every process will dump the logging into separate files, check `NCCL_DEBUG_FILE`
NSYS_PROFILER=false # Turn on the NSYS profiler # NOTE(tj.solergibert) When using the profiler, stdout gets blocked
MOCK_DATA=false # Set to `true` to use mock data
###################

# Megatron source and dataset cache WARNING (!) MUST BE ON IOPSSTOR (!)
MEGATRON_LM_DIR=/iopsstor/scratch/cscs/$USER/Megatron-LM
DATASET_CACHE_DIR=/iopsstor/scratch/cscs/$USER/datasets/cache
# Overwrite SRC_DIR/Megatron-LM with the current codebase
BACKUP_CODEBASE=false  

# Logging directories & artifacts
PROJECT_NAME=Megatron-Clariden
EXP_NAME=llama3-8b-${SLURM_NNODES}n-${SEQ_LEN}sl-${GBS}gbsz
PROJECT_DIR=$MEGATRON_LM_DIR/logs/Meg-Runs/$PROJECT_NAME

#########################################

EXP_DIR=$PROJECT_DIR/$EXP_NAME
CKPT_DIR=$EXP_DIR/checkpoints
TRIGGER_DIR=$EXP_DIR/triggers
DEBUG_DIR=$EXP_DIR/debug/$SLURM_JOB_ID
COMPUTE_ENVIRONMENT_DIR=$DEBUG_DIR/compute_environment.txt
GPU_MEM_LOGGING=$DEBUG_DIR/memory_logging.txt
LOGGING_DIR=$EXP_DIR/logging
TENSORBOARD_DIR=$LOGGING_DIR/tensorboard
SRC_DIR=$EXP_DIR/src

# Set up directories
mkdir -p $CKPT_DIR
mkdir -p $PROJECT_DIR
mkdir -p $TRIGGER_DIR
mkdir -p $DEBUG_DIR
mkdir -p $LOGGING_DIR
mkdir -p $SRC_DIR

# Clean triggers
rm -f $TRIGGER_DIR/save
rm -f $TRIGGER_DIR/exit

# Copy current source code to SRC_DIR if asked to or if the SRC_DIR is empty
if [ "$BACKUP_CODEBASE" == true ] || [ -z "$(ls -A "$SRC_DIR")" ]; then
  echo "Copying current codebase to $SRC_DIR..."
  rsync -av --exclude-from=$MEGATRON_LM_DIR/.gitignore $MEGATRON_LM_DIR/ $SRC_DIR/
  # Save current commit hash
  cd ~/Megatron-LM
else
  echo "In order to backup the latest codebase you can turn BACKUP_CODEBASE to true"
fi
echo "The codebase path for this job: $SRC_DIR"

# Set up ENV
cd $SRC_DIR
export PYTHONPATH=$SRC_DIR:$PYTHONPATH
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
MASTER_ADDR=$(hostname)
MASTER_PORT=25678
export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK/SLURM_GPUS_PER_NODE))

srun -l bash -c 'echo $(hostname) $(nvidia-smi | grep -o "|\\s*[0-9]*MiB")' > $GPU_MEM_LOGGING
ulimit -c 0

#### Megatron Args #### Check megatron/training/arguments.py
# Based on the Llama 3.1 8B model.
TRANSFORMER_ENGINE_ARGS=(
	--transformer-impl transformer_engine
	--use-precision-aware-optimizer
	--main-grads-dtype bf16
)

NETWORK_SIZE_ARGS=(
	--num-layers 32
	--hidden-size 4096
	--ffn-hidden-size 14336
	--num-attention-heads 32
	--group-query-attention
	--num-query-groups 8
	--max-position-embeddings $SEQ_LEN
	--position-embedding-type rope
	--rotary-base 500000
	--use-rope-scaling
	--rope-scaling-factor 8
	--make-vocab-size-divisible-by 128
	--normalization RMSNorm
	--swiglu
	--untie-embeddings-and-output-weights
)

LOGGING_ARGS=(
	--log-throughput
	--log-progress
	--tensorboard-dir $TENSORBOARD_DIR
	--log-timers-to-tensorboard
	--no-log-loss-scale-to-tensorboard
	--log-memory-to-tensorboard
)

REGULARIZATION_ARGS=(
	--attention-dropout 0.0
	--hidden-dropout 0.0
	--weight-decay 0.1
	--clip-grad 1.0
	--adam-beta1 0.9
	--adam-beta2 0.95
)

TRAINING_ARGS=(
	--micro-batch-size $MBS
	--global-batch-size $GBS
	--no-check-for-nan-in-loss-and-grad
	--train-iters $TRAINING_STEPS
	--log-interval 1
	--cross-entropy-loss-fusion
	--disable-bias-linear
	--optimizer adam
	--dataloader-type single
	--manual-gc
	--manual-gc-interval 5000
)

INITIALIZATION_ARGS=(
	--seed 28
	--init-method-std 0.008944
)

# NOTE(tj.solergibert) Check all the arguments in megatron/training/arguments.py#L1548 or https://github.com/NVIDIA/Megatron-LM/blob/0dd78ddcdb117ce4f2e9761449274d87af717674/megatron/training/arguments.py#L1548-L1606
LEARNING_RATE_ARGS=(
	--lr 0.00022
	--min-lr 0.000022  # x10 reduction
	--lr-decay-style cosine
	--lr-warmup-iters 2000
)

CHECKPOINTING_ARGS=(
	--save $CKPT_DIR
	--save-interval $CHECKPOINT_STEPS
	--ckpt-format torch_dist
	--load $CKPT_DIR  # delete this to NOT reload from the latest checkpoint
	--async-save
)

MIXED_PRECISION_ARGS=(
	--bf16
)

DISTRIBUTED_ARGS=(
	--tensor-model-parallel-size 1
	--pipeline-model-parallel-size 1
	--use-distributed-optimizer
    --overlap-grad-reduce
    --overlap-param-gather
)

TOKENIZER_ARGS=(
	--tokenizer-type HuggingFaceTokenizer
	--tokenizer-model nvidia/OpenMath2-Llama3.1-8B
)

DATA_ARGS=(
	--split 100,0,0
	--seq-length $SEQ_LEN
	--num-workers 1
	--num-dataset-builder-threads 1
)

# Data Args
if [ "$MOCK_DATA" = true ]; then
  DATA_ARGS="${DATA_ARGS[@]} --mock-data"
else
  DATA_ARGS="${DATA_ARGS[@]} --data-path $(python3 $SRC_DIR/scripts/tools/create_data_config.py -p $DATASETS) --data-cache-path $DATASET_CACHE_DIR"
fi

TORCHRUN_ARGS=(
    --nproc-per-node $SLURM_GPUS_PER_NODE
	--nnodes $SLURM_NNODES
	--rdzv_endpoint $MASTER_ADDR:$MASTER_PORT
	--rdzv_backend c10d
	--max_restarts 0
	--tee 3
)

CMD_PREFIX="numactl --membind=0-3"

TRAINING_CMD="torchrun ${TORCHRUN_ARGS[@]} $SRC_DIR/pretrain_gpt.py \
    ${TRANSFORMER_ENGINE_ARGS[@]} \
    ${NETWORK_SIZE_ARGS[@]} \
    ${LOGGING_ARGS[@]} \
    ${REGULARIZATION_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${INITIALIZATION_ARGS[@]} \
    ${LEARNING_RATE_ARGS[@]} \
    ${CHECKPOINTING_ARGS[@]} \
    ${MIXED_PRECISION_ARGS[@]} \
    ${DISTRIBUTED_ARGS[@]} \
    ${TOKENIZER_ARGS[@]} \
    $DATA_ARGS"

# WANDB Logging
if [ -n "$WANDB_API_KEY" ]; then
  echo "[$(date)] WANDB API key detected. Enabling WANDB logging."
  # Sync any previous run data if present
  if [ -d "$LOGGING_DIR/wandb/latest-run" ]; then
    echo "[$(date)] Syncing WANDB from previous run"
    wandb sync "$LOGGING_DIR/wandb/latest-run"
  fi
  # Add wandb-related args to TRAINING_CMD
  TRAINING_CMD="$TRAINING_CMD \
    --wandb-save-dir $LOGGING_DIR \
    --wandb-project $PROJECT_NAME \
    --wandb-exp-name $EXP_NAME-$SLURM_JOB_ID"
else
  export WANDB_MODE=disabled
  echo "[$(date)] No WANDB API key found. WANDB logging disabled."
fi

# NCCL Debug
if [ "$LOG_NCCL" = true ]; then
  CMD_PREFIX="NCCL_DEBUG=INFO NCCL_DEBUG_FILE=$DEBUG_DIR/nccl-info-procid-\$SLURM_PROCID.txt $CMD_PREFIX"
fi

# NSYS profiler
if [ "$NSYS_PROFILER" = true ]; then
    NSYS_LAUNCHER="nsys profile -s none --trace='nvtx,cudnn,cublas,cuda' --output=$DEBUG_DIR/nsys-trace.nsys-rep --force-overwrite true --capture-range=cudaProfilerApi --capture-range-end=stop"
    TRAINING_CMD="$NSYS_LAUNCHER $TRAINING_CMD --profile"
fi

# Save sbatch script
cp $0 $DEBUG_DIR

# Checkpoint Compute Environment
echo -e "$(date)" > $COMPUTE_ENVIRONMENT_DIR 
printf '=%.0s' {1..100} >> $COMPUTE_ENVIRONMENT_DIR 
echo -e "\nCMD: $CMD_PREFIX $TRAINING_CMD" >> $COMPUTE_ENVIRONMENT_DIR
printf '=%.0s' {1..100} >> $COMPUTE_ENVIRONMENT_DIR 
echo -e "\nSlurm file: $0\n" >> $COMPUTE_ENVIRONMENT_DIR
cat $0 >> $COMPUTE_ENVIRONMENT_DIR
echo -e "" >> $COMPUTE_ENVIRONMENT_DIR
printf '=%.0s' {1..100} >> $COMPUTE_ENVIRONMENT_DIR 
echo -e "\nTOML file: $SLURM_SPANK__SLURM_SPANK_OPTION_pyxis_environment\n" >> $COMPUTE_ENVIRONMENT_DIR
cat $SLURM_SPANK__SLURM_SPANK_OPTION_pyxis_environment >> $COMPUTE_ENVIRONMENT_DIR
echo -e "" >> $COMPUTE_ENVIRONMENT_DIR
printf '=%.0s' {1..100} >> $COMPUTE_ENVIRONMENT_DIR 
echo -e "\nNODES: $(scontrol show hostnames $SLURM_JOB_NODELIST)" >> $COMPUTE_ENVIRONMENT_DIR
printf '=%.0s' {1..100} >> $COMPUTE_ENVIRONMENT_DIR 
echo -e "\nMegatron path: $SRC_DIR ($(git -C $SRC_DIR rev-parse --verify HEAD))" >> $COMPUTE_ENVIRONMENT_DIR
printf '=%.0s' {1..100} >> $COMPUTE_ENVIRONMENT_DIR 
echo -e "\n$(pip list)" >> $COMPUTE_ENVIRONMENT_DIR
printf '=%.0s' {1..100} >> $COMPUTE_ENVIRONMENT_DIR 
echo -e "\n$(nvidia-smi)" >> $COMPUTE_ENVIRONMENT_DIR # CUDA Version & Driver
printf '=%.0s' {1..100} >> $COMPUTE_ENVIRONMENT_DIR 
echo -e "\nEnvironment Variables:\n\n$(printenv)" >> $COMPUTE_ENVIRONMENT_DIR
printf '=%.0s' {1..100} >> $COMPUTE_ENVIRONMENT_DIR 

srun -lu --cpus-per-task $SLURM_CPUS_PER_TASK --wait 60 bash -c "$CMD_PREFIX $TRAINING_CMD"

echo "END TIME: $(date)"