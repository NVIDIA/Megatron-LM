# TODO: Where to put eval output?
# TODO: Support nodes>1 maybe?
# TODO: Currently tp*pp>1 hangs :(
# TODO: Need to update completions to use MCoreEngine

# Define default variables.
DEF_MEGATRON_PATH=$(dirname $(dirname $( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )))  # Grandparent of current file location.
DEF_LOGS_DIR=$PWD/"eval-logs"
DEF_SBATCH_PATH=$PWD/evaluate.sbatch
DEF_CONTAINER_PATH="/capstor/store/cscs/swissai/a06/containers/NeMo/nemo-latest.toml"
DEF_LM_HARNESS_PATH="/capstor/store/cscs/swissai/a06/users/ahernnde/workspace/lm-evaluation-harness"
DEF_ACCOUNT=a-a06
DEF_TOKENIZER=/users/fsimin/tokenizer_nemo/

DEF_IT=latest
DEF_TASKS=arc_easy
DEF_LIMIT=1000

# Usage function.
usage () {
	echo "Usage: submit_evaluation.sh <checkpoint-path> [options...]"
	echo "Submits a slurm sbatch to run evaluation of the specified path. Specify either --size or --tp and --pp in order to convert the checkpoint to a more efficient distributed config for inference."
	echo "In addition, there are a few environment variables you can set to specify some paths (optional)."
	echo ""
	echo "Arguments:"
	echo "  <checkpoint-path>: Path of the megatron checkpoint to evaluate."
	echo ""
	echo "Options:"
	echo "  --help: Prints this message and exits."
	echo "  --size (choices={1, 3, 8, 70}): The size of the checkpoint to evaluate. If not set, --tp and --pp should be specified."
	echo "  --tasks: lm-eval-harness tasks to run (default=$DEF_TASKS)."
	echo "  --limit (int>0 or 'null'): lm-eval-harness limit samples per task (default=$DEF_LIMIT)."
	echo "  --tp (int>0): Target TP size for inference. Ignored if --size is set, required otherwise."
	echo "  --pp (int>0): Target PP size for inference. Ignored if --size is set, required otherwise."
	echo "  --iteration (int>0 | 'latest'): What iteration to evaluate (default=$DEF_IT)"
	echo "  --wandb-project"
	echo "  --wandb-entity"
	echo "  --wandb-id"
	echo ""
	echo "Variables:"
	echo "  MEGATRON_PATH: Megatron root (default=$DEF_MEGATRON_PATH)."
	echo "  LOGS_DIR: Slurm logs directry (default=$DEF_LOGS_DIR)."
	echo "  SBATCH_PATH: Where to save the sbatch file (default=$DEF_SBATCH_PATH)."
	echo "  CONTAINER_PATH: Container path (default=$DEF_CONTAINER_PATH)."
	echo "  LM_HARNESS_PATH: lm-eval-harness root (default=$DEF_LM_HARNESS_PATH)."
	echo "  ACCOUNT: Slurm account (default=$DEF_ACCOUNT)."
	echo "  TOKENIZER: Huggingface tokenizer (default=$DEF_TOKENIZER)."
}

# Set variables.
if [ -z ${MEGATRON_PATH+x} ]; then
	MEGATRON_PATH=$DEF_MEGATRON_PATH
fi
if [ -z ${LOGS_DIR+x} ]; then
	LOGS_DIR=$DEF_LOGS_DIR
fi
if [ -z ${SBATCH_PATH+x} ]; then
	SBATCH_PATH=$DEF_SBATCH_PATH
fi
if [ -z ${CONTAINER_PATH+x} ]; then
	CONTAINER_PATH=$DEF_CONTAINER_PATH
fi
if [ -z ${ACCOUNT+x} ]; then
	ACCOUNT=$DEF_ACCOUNT
fi
if [ -z ${LM_HARNESS_PATH+x} ]; then
	LM_HARNESS_PATH=$DEF_LM_HARNESS_PATH
fi
if [ -z ${TOKENIZER+x} ]; then
	TOKENIZER=$DEF_TOKENIZER
fi

# Parse args.
if [[ $# -eq 0 ]]; then
	echo Invalid argument count: $# >&2
	usage
	exit 1
fi

CHECKPOINT_PATH=$1
shift
while [[ $# -gt 0 ]]; do
	case $1 in
		--help)
			usage; exit 0;;
		--size)
			if [ $2 -eq 1 ] || [ $2 -eq 3 ] || [ $2 -eq 8 ]; then
				TP=1
				PP=1
			elif [ $2 -eq 70 ]; then
				TP=4
				PP=1
			else
				echo Unknown size $2. Choices={1, 8, 70}. >&2
				exit 1
			fi
			shift 2;;

		--tasks)
			TASKS=$2; shift 2;;
		--limit)
			LIMIT=$2; shift 2;;
		--iteration)
			ITERATION=$2; shift 2;;
		--wandb-project)
			WANDB_PROJECT=$2; shift 2;;
		--wandb-entity)
			WANDB_ENTITY=$2; shift 2;;
		--wandb-id)
			WANDB_ID=$2; shift 2;;

		--tp)
			if [ -z ${TP+x} ]; then  # check if undef to ignore --tp if --size is set.
				TP=$2
			fi
			shift 2;;

		--pp)
			if [ -z ${PP+x} ]; then  # check if undef to ignore --pp if --size is set.
				PP=$2
			fi
			shift 2;;

		*)
			echo "Unexpected argument $1" >&2
			usage
			exit 1
	esac
done

if [ -z ${TP+x} ]; then
	echo Neither --size nor --tp was specified. >&2
	exit 1
fi

if [ -z ${PP+x} ]; then
	echo Neither --size nor --pp was specified. >&2
	exit 1
fi

if [ -z ${ITERATION+x} ]; then
	ITERATION=$DEF_IT
fi
if [ -z ${TASKS+x} ]; then
	TASKS=$DEF_TASKS
fi
if [ -z ${LIMIT+x} ]; then
	LIMIT=$DEF_LIMIT
fi

# Build eval args depending on this scripts args.
if [ $LIMIT != null ]; then
	LIMIT_ARGS="--limit=$LIMIT"
fi
if [ $ITERATION = latest ]; then
	ITERATION=$(cat $CHECKPOINT_PATH/latest_checkpointed_iteration.txt)
fi

if [ ! -z ${WANDB_ENTITY+x} ] || [ ! -z ${WANDB_PROJECT+x} ] || [ ! -z ${WANDB_ID+x} ]; then
	if [ -z ${WANDB_ENTITY+x} ] || [ -z ${WANDB_PROJECT+x} ] || [ -z ${WANDB_ID+x} ]; then
		echo "Either all --wandb-entity, --wandb-project and --wandb-id should be set, or neither" >&2
		exit 1
	fi
	WANDB_ARGS="--wandb_args entity=$WANDB_ENTITY,project=$WANDB_PROJECT,id=$WANDB_ID,resume=allow,step=$ITERATION"
fi

# Make sure TP and PP settings make sense.
GPUS_PER_NODE=4
WORLD_SIZE=$(($TP*PP))
if (( WORLD_SIZE > GPUS_PER_NODE )); then
	echo "tp*pp > gpus_per_node not supported yet" >&2
	exit 1
fi
if [ $WORLD_SIZE -eq 0 ]; then  # TODO: fix this optimization, not used for now as WORLD_SIZE>0 always.
	echo "tp*pp = 1. DP optimization enabled"
	NPROC_PER_NODE=$GPUS_PER_NODE
	DP=$GPUS_PER_NODE
else
	NPROC_PER_NODE=$WORLD_SIZE
	DP=1
fi

# Some useful variables.
JOBNAME=evaluate_$(echo $CHECKPOINT_PATH | tr / _)
ENDPOINT_PORT=5000
MBS=1
TORCHRUN_ARGS=(
	--nproc-per-node $NPROC_PER_NODE
	--nnodes 1
	--master-addr localhost
	--master-port $(($ENDPOINT_PORT + 1000))
)

# Now let's prepare the sbatch.
cat > $SBATCH_PATH <<- EOM
#!/bin/bash
#SBATCH --account=a-a06
#SBATCH --cpus-per-task=288
#SBATCH --gres=gpu:4
#SBATCH --environment=$CONTAINER_PATH
#SBATCH --job-name=$JOBNAME
#SBATCH --mem=460000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=$LOGS_DIR/${JOBNAME}_%j.out
#SBATCH --error=$LOGS_DIR/${JOBNAME}_%j.err
#SBATCH --time=12:00:00
#SBATCH --exclusive
#SBATCH --dependency=singleton

# Step 0: Some useful logs.
export MASTER_ADDR=\$(hostname)
echo "Using nodes: \$SLURM_JOB_NODELIST"
srun -l bash -c 'echo \$(hostname) \$(nvidia-smi | grep -o "|\\s*[0-9]*MiB")'

srun -l --unbuffered numactl --membind=0-3 bash -c "
	export CUDA_DEVICE_MAX_CONNECTIONS=1
	set -e

	# Step 1: Launch inference endpoint.
	echo Spinning inference endpoint
	cd $MEGATRON_PATH
	torchrun ${TORCHRUN_ARGS[@]} tools/run_text_generation_server.py --tensor-model-parallel-size=$TP --pipeline-model-parallel-size=$PP --use-checkpoint-args --load=$CHECKPOINT_PATH --bf16 --micro-batch-size=$MBS --max-batch-size=$DP --tokenizer-type=HuggingFaceTokenizer --tokenizer-model=$TOKENIZER --seed=42 --port=$ENDPOINT_PORT --ckpt-step=$ITERATION --finetune &

	# Step 2: Launch lm-harness.
	echo Running lm-eval-harness.
	cd $LM_HARNESS_PATH
	python lm_eval/__main__.py --model local-completions --tasks $TASKS --model_args base_url=http://localhost:5000/completions,tokenized_requests=False,tokenizer=$TOKENIZER,num_concurrent=8,timeout=43200 --batch_size=$DP --output=eval_\$SLURM_JOBID $LIMIT_ARGS $WANDB_ARGS
"
EOM

echo Saved sbatch to $SBATCH_PATH
OUT=$(sbatch $SBATCH_PATH)
echo $OUT

IFS=' ' read -ra CHUNKS <<< $OUT
JOBID=${CHUNKS[-1]}
echo Logs go to: $LOGS_DIR/${JOBNAME}_$JOBID.out
