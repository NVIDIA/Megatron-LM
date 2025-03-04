# Define default variables :D.
GPUS_PER_NODE=4

DEF_MEGATRON_PATH=$(dirname $(dirname $( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )))  # Grandparent of current file location.
DEF_LOGS_ROOT=$PWD/eval-logs
DEF_CONTAINER_PATH=/capstor/store/cscs/swissai/a06/containers/NGC-PyTorch/ngc_pt_jan.toml
DEF_ACCOUNT=a-a06
DEF_TOKENIZER=alehc/swissai-tokenizer

ITERATION=latest
TASKS=scripts/evaluation/swissai_eval
LIMIT=null
BS=1
CONVERT_TO_HF=false

TASK_GROUPS="mmlu_continuation global_mmlu"

# Usage function.
usage () {
	echo "Usage: submit_evaluation.sh <checkpoint-path> [options...]"
	echo "Submits a slurm sbatch to run evaluation of the specified path. The checkpoint-path should either be a megatron checkpoint or a huggingface model. This is determining by looking for checkpoint-path/latest_checkpointed_iteration.txt."
	echo "If a megatron checkpoint is given, you can add --convert-to-hf to convert the checkpoint to HF for faster inference (highly recommended if there is a HF implementation available). Otherwise the slower api endpoint will be used." 
	echo "On all cases, specify either --size or --tp and --pp in order to convert the checkpoint to a more efficient distributed config for inference."
	echo "If a hf checkpoint is given (or --convert-to-hf is set), DP will be enabled as long as TP*PP<NUM_GPUS_PER_NODE."
	echo "You will need to specify a way to compute the consumed tokens at the iteration you are evaluating. This is either by specifying --iteration and --tokens-per-iter at the same time, or alternatively by --consumed-tokens (the latter only available in HF checkpoints)."
	echo "In addition, there are a few environment variables you can set to specify some paths (optional)."
	echo "If setting any of the --wandb arguments, make sure to also export your WANDB_API_KEY."
	echo ""
	echo "Arguments:"
	echo "  <checkpoint-path>: Path of the megatron checkpoint to evaluate."
	echo ""
	echo "Options:"
	echo "  --help: Prints this message and exits."
	echo "  --name: Name of the eval run. If not set, the path will be used as name"
	echo "  --size (choices={1, 3, 8, 70}): The size of the checkpoint to evaluate. If not set, --tp and --pp should be specified. This only sets --tp and --pp for you."
	echo "  --convert-to-hf: When set, if a megatron checkpoint is given, the model will be converted to HF."
	echo "  --tasks: lm-eval-harness tasks to run (default=$TASKS)."
	echo "  --limit (int>0 or 'null'): lm-eval-harness limit samples per task (default=$LIMIT)."
	echo "  --tp (int>0): Target TP size for inference. Ignored if --size is set, required otherwise."
	echo "  --pp (int>0): Target PP size for inference. Ignored if --size is set, required otherwise."
	echo "  --bs (int>0): Batch size used for inference (default=$BS)."
	echo "  --iteration (int>0 | 'latest'): What iteration to evaluate (default=$ITERATION)"
	echo "  --revision: Only used when input is HF checkpoint. Sets the HF revision."
	echo "  --tokens-per-iter (int>0): If specified with --iteration, the total consumed_tokens will be calculated by iteration*tokens_per_iter. Cannot be specified if consumed_tokens is also specified"
	echo "  --consumed-tokens (int>0): When --iteration or --tokens-per-iter are not set, you need to specify this to set the total number of tokens the checkpoint has seen up to now."
	echo "  --wandb-project"
	echo "  --wandb-entity"
	echo "  --wandb-id"
	echo ""
	echo "Variables:"
	echo "  MEGATRON_PATH: Megatron root (default=$DEF_MEGATRON_PATH)."
	echo "  LOGS_DIR: Logs root directry where wandb logs, slurm logs and .sbatches will go (default=$DEF_LOGS_ROOT)."
	echo "  CONTAINER_PATH: Container path (default=$DEF_CONTAINER_PATH)."
	echo "  LM_HARNESS_PATH: lm-eval-harness root (default=$DEF_LM_HARNESS_PATH)."
	echo "  ACCOUNT: Slurm account (default=$DEF_ACCOUNT)."
	echo "  TOKENIZER: Huggingface tokenizer (default=$DEF_TOKENIZER)."
}

# Set variables.
if [ -z ${MEGATRON_PATH+x} ]; then
	MEGATRON_PATH=$DEF_MEGATRON_PATH
fi
if [ -z ${LOGS_ROOT+x} ]; then
	LOGS_ROOT=$DEF_LOGS_ROOT
fi
if [ -z ${CONTAINER_PATH+x} ]; then
	CONTAINER_PATH=$DEF_CONTAINER_PATH
fi
if [ -z ${ACCOUNT+x} ]; then
	ACCOUNT=$DEF_ACCOUNT
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
		--bs)
			BS=$2; shift 2;;
		--limit)
			LIMIT=$2; shift 2;;
		--name)
			NAME=$2; shift 2;;
		--convert-to-hf)
			CONVERT_TO_HF=true; shift;;
		--iteration)
			ITERATION=$2; shift 2;;
		--revision)
			REVISION=$2; shift 2;;
		--tokens-per-iter)
			TOKENS_PER_ITER=$2; shift 2;;
		--consumed-tokens)
			CONSUMED_TOKENS=$2; shift 2;;
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

# Build eval args depending on this scripts args.
if [ $LIMIT != null ]; then
	LIMIT_ARGS="--limit=$LIMIT"
fi
if [ $ITERATION = latest ]; then
	if [ -f $CHECKPOINT_PATH/latest_checkpointed_iteration.txt ]; then
		ITERATION=$(cat $CHECKPOINT_PATH/latest_checkpointed_iteration.txt)
	else
		ITERATION=1
	fi
fi

# Determine the CONSUMED_TOKENS.
if [ -f $CHECKPOINT_PATH/latest_checkpointed_iteration.txt ]; then
	# In the case of Megatron checkpoints, CONSUMED_TOKENS is exlusively computed using --iteration and --tokens-per-iter.
	if [ -z ${TOKENS_PER_ITER+x} ]; then
		echo When using megatron checkpoints, you need to set --tokens-per-iter to accurately compute the consumed tokens >&2
		exit 1
	fi
	if [ ! -z ${CONSUMED_TOKENS+x} ]; then
		echo When using megatron checkpoints, you cannot set --consumed-tokens, please set --tokens-per-iter instead >&2
		exit 1
	fi
	CONSUMED_TOKENS=$((ITERATION*TOKENS_PER_ITER))
else
	# The huggingface checkpoints can get CONSUMED_TOKENS either by --tokens-per-iter or --consumed-tokens
	if [ -z ${CONSUMED_TOKENS+x} ]; then
		if [ -z ${TOKENS_PER_ITER+x} ]; then
			echo Neither of --consumed-tokens or --tokens-per-iter set, aborting >&2
			exit 1
		fi
		CONSUMED_TOKENS=$((ITERATION*TOKENS_PER_ITER))
	fi
fi

mkdir -p $LOGS_ROOT/slurm $LOGS_ROOT/sbatch $LOGS_ROOT/lmharness
if [ -z ${NAME+x} ]; then
	NAME=$(echo $CHECKPOINT_PATH | tr / _)
fi
#NAME=${NAME}_it$ITERATION
LOGS_DIR=$LOGS_ROOT/slurm
SBATCH_PATH=$LOGS_ROOT/sbatch/$NAME.sbatch
WANDB_DIR=$LOGS_ROOT
EVAL_DIR=$LOGS_ROOT/lmharness

if [ ! -z ${WANDB_ENTITY+x} ] || [ ! -z ${WANDB_PROJECT+x} ] || [ ! -z ${WANDB_ID+x} ]; then
	if [ -z ${WANDB_ENTITY+x} ] || [ -z ${WANDB_PROJECT+x} ] || [ -z ${WANDB_ID+x} ]; then
		echo "Either all --wandb-entity, --wandb-project and --wandb-id should be set, or none" >&2
		exit 1
	fi
	WANDB_ARGS="--wandb_args entity=$WANDB_ENTITY,project=$WANDB_PROJECT,id=$WANDB_ID,resume=allow,step=$ITERATION,consumed_tokens=$CONSUMED_TOKENS"
	read -r -d '' WANDB_COMMAND <<- EOM
	# Wandb sync just in case wandb died in lm-harness.
	for path in $WANDB_DIR/wandb/run-*-$WANDB_ID; do
		WANDB_RESUME=allow wandb sync -e $WANDB_ENTITY -p $WANDB_PROJECT --id $WANDB_ID \\\$path
	done

	# Update eval_table.
	cd $MEGATRON_PATH
	WANDB_RESUME=allow python scripts/evaluation/update_wandb_eval_table.py --entity=$WANDB_ENTITY --project=$WANDB_PROJECT --runid=$WANDB_ID --groups $TASK_GROUPS
	EOM
fi

# Some useful variables.
JOBNAME=evaluate_$NAME
ENDPOINT_PORT=5000

COMMON_EVAL_ARGS="--trust_remote_code --batch_size=$BS --tasks=$TASKS --output=$EVAL_DIR/eval_\$SLURM_JOBID $LIMIT_ARGS $WANDB_ARGS"
if [ -f $CHECKPOINT_PATH/latest_checkpointed_iteration.txt ] && [ $CONVERT_TO_HF != true ]; then
	echo Megatron checkpoint detected!
	echo "WARNING! No conversion to HF will be done. Please specify HF implementation if available, otherwise evaluation will be slower."
	WORLD_SIZE=$(($TP*PP))
	if (( WORLD_SIZE > GPUS_PER_NODE )); then
		echo "tp*pp > gpus_per_node not supported yet" >&2
		exit 1
	fi
	read -r -d '' CMD_SERVER <<- EOM
	# Launch inference endpoint.
	echo Spinning inference endpoint
	torchrun --nproc-per-node=$WORLD_SIZE --master-addr=localhost --master-port=$(($ENDPOINT_PORT + 1000)) tools/run_text_generation_server.py --tensor-model-parallel-size=$TP --pipeline-model-parallel-size=$PP --use-checkpoint-args --load=$CHECKPOINT_PATH --bf16 --micro-batch-size=$BS --max-batch-size=$BS --tokenizer-type=HuggingFaceTokenizer --tokenizer-model=$TOKENIZER --seed=42 --port=$ENDPOINT_PORT --ckpt-step=$ITERATION --finetune --max-tokens-to-oom=4194304 &

	EOM
	CMD_EVAL="WANDB_RESUME=allow lm_eval --model=local-completions --model_args=base_url=http://localhost:5000/completions,tokenized_requests=False,tokenizer=$TOKENIZER,num_concurrent=0,timeout=43200,max_retries=1,max_length=4096 $COMMON_EVAL_ARGS"
else
	if [ -f $CHECKPOINT_PATH/latest_checkpointed_iteration.txt ]; then
		echo Megatron checkpoint detected!
		echo Checkpoint will be converted to HF

		read -r -d '' CMD_CONVERT <<- EOM
		# Convert from megatron to HF.
		cd $MEGATRON_PATH
		export PYTHONPATH=$MEGATRON_PATH:\\\$PYTHONPATH
		torchrun scripts/conversion/torchdist_2_torch.py --bf16 --load=$CHECKPOINT_PATH --ckpt-step=$ITERATION --ckpt-convert-save=\\\$TORCH_NODIST_PATH
		python tools/checkpoint/convert.py --model-type=GPT --loader=core --saver=llama_hf --load-dir=\\\$TORCH_NODIST_PATH/torch --save-dir=\\\$HF_TEMP_PATH --hf-tokenizer=$TOKENIZER
		EOM
		HF_CHECKPOINT_PATH=\\\$HF_TEMP_PATH
	else
		echo Huggingface checkpoint detected!
		HF_CHECKPOINT_PATH=$CHECKPOINT_PATH
		if [ ! -z ${REVISION+x} ]; then
			MAYBE_REVISION=",revision=$REVISION"
		fi
	fi

	CMD_EVAL="WANDB_RESUME=allow accelerate launch -m lm_eval --model=hf --model_args=pretrained=$HF_CHECKPOINT_PATH,tokenizer=$TOKENIZER,max_length=4096$MAYBE_REVISION $COMMON_EVAL_ARGS"
fi

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
#SBATCH --time=01:00:00
#SBATCH --exclusive
#SBATCH --dependency=singleton

# Step 0: Some useful logs.
export MASTER_ADDR=\$(hostname)
echo "Using nodes: \$SLURM_JOB_NODELIST"
srun -l bash -c 'echo \$(hostname) \$(nvidia-smi | grep -o "|\\s*[0-9]*MiB")'

srun -l --unbuffered numactl --membind=0-3 bash -c "
	export CUDA_DEVICE_MAX_CONNECTIONS=1
	export WANDB_DIR=$WANDB_DIR
	export HF_HOME=$SCRATCH/huggingface
	set -e

	# Create tempdirs.
	cd
	mkdir -p $SCRATCH/.tmp
	HF_TEMP_PATH=\\\$(mktemp -d -p $SCRATCH/.tmp)  # To store hf conversion (if needed).
	TORCH_NODIST_PATH=\\\$(mktemp -d -p $SCRATCH/.tmp)  # To store torch no dist checkpoint converted (if needed).
	REPOS_PATH=\\\$(mktemp -d -p $SCRATCH/.tmp)  # To git clone repos.
	function cleanup {
		rm -rf \\\$HF_TEMP_PATH
		rm -rf \\\$TORCH_NODIST_PATH
		rm -rf \\\$REPOS_PATH
	}
	trap cleanup EXIT

	# Install custom transformers and lm-harness.
	cd \\\$REPOS_PATH
	git clone https://github.com/swiss-ai/transformers.git
	cd transformers
	git checkout swissai-model
	python -m pip install -e .
	cd ..
	git clone https://github.com/AleHD/lm-evaluation-harness.git
	cd lm-evaluation-harness
	python -m pip install -e .[api]

	$CMD_CONVERT
	$CMD_SERVER

	# Launch eval.
	cd $MEGATRON_PATH
	$CMD_EVAL

	$WANDB_COMMAND
"
EOM

echo Saved sbatch to $SBATCH_PATH
OUT=$(sbatch $SBATCH_PATH)
echo $OUT

IFS=' ' read -ra CHUNKS <<< $OUT
JOBID=${CHUNKS[-1]}
echo Logs go to: $LOGS_DIR/${JOBNAME}_$JOBID.out
