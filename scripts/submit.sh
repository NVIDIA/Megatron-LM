# tp1:
# name_parallel_factor: tensor([1, 1, 1, 1], device='cuda:0')
# shapes: tensor([4718592, 6291456, 4718592, 7864320], device='cuda:0')


#= Prelude =#
# Some constants.
SCRIPT_VERSION=v1
SEQ_LEN=4096
TOKENIZER=alehc/swissai-tokenizer
DATA_DIR=/capstor/store/cscs/swissai/a06/datasets_tokenized/megatron/sai/swissai-fineweb-edu-filterrobots-merge
CODE_PATH=$STORE/users/ahernnde/workspace/AleHD-Megatron-LM

DEF_TOKENS=50
TOKENS=$DEF_TOKENS
NODES=1

# Usage function.
usage () {
	echo "Usage: llama.sh <size> [options...]"
	echo "<size>: 300/1"
	echo "Options:"
	echo " --debug: Displays this message"
	echo " --nodes <nodes>: How many nodes to use"
	echo " --fp8: Enables fp8"
	echo " --fp8-dpa: Enables fp8 DPA"
	echo " --tokens <int> (default=$DEF_TOKENS): Amount of tokens to train with (in B)."
	echo " --extra-name <name>: Add a suffix to the name"
}

if [[ $# -eq 0 ]]; then
	>&2 echo Invalid argument count: $#
	usage
	exit 1
fi

# Define some variables depending on size.
EXTRA_ARGS=()
TP=1
PP=1
if [[ $1 -eq 300 ]]; then 
	# batch_size: ~0.52M.
	# tok/sec/gpu: ~83k  (2nodes, bf16).
	# 50B ETA: ~21h (2nodes, bf16).
	# ckpt freq: ~2h.
	LAYERS=16
	HIDDEN_SIZE=1024
	FFN_SIZE=4096
	NUM_HEADS=8
	NUM_QUERY_GROUPS=4
	MBS=16
	GBS=128
	ITERS=2000  # 1BT.
	LR=0.001
	INIT_STD=0.001
	SIZE=320M
	SAVE_FREQ=10000
elif [[ $1 -eq 1 ]]; then 
	# batch_size: ~1.05M.
	# tok/sec/gpu: 36.5k (8nodes, bf16).
	# 50B ETA: ~12h (8nodes, bf16).
	# ckpt freq: ~1h15m (8nodes, bf16).
	LAYERS=16
	HIDDEN_SIZE=2048
	FFN_SIZE=8192
	NUM_HEADS=16
	NUM_QUERY_GROUPS=8
	MBS=4
	GBS=256
	ITERS=1000  # 1BT.
	LR=0.0005
	INIT_STD=0.001
	SIZE=1B
	SAVE_FREQ=5000
else
	>&2 echo "Invalid llama size: $1"
	usage
	exit 1
fi
shift

# Now get the general options.
SUFFIX=""
TIME=1-00:00:00
while [[ $# -gt 0 ]]; do
	case $1 in
		--debug)
			SCRIPT_VERSION=$SCRIPT_VERSION-debug;
			TIME=00:30:00
			shift;;
		--nodes)
			NODES=$2; shift 2;;
		--fp8)
			FP8=true; shift;;
		--fp8-dpa)
			FP8DPA=true; shift;;
		--extra-name)
			EXTRA_NAME="-$2"; shift 2;;
		--tokens)
			TOKENS=$2; shift 2;;
		*)
			echo "Unexpected argument $1"
			usage
			exit 1
	esac
done

#= MIDDLE: Set up arguments. =#
FP8_ARGS=()
if [[ $FP8 = true ]]; then
	SUFFIX=$SUFFIX-fp8
	FP8_ARGS+=(--fp8-margin 0 --fp8-format hybrid --fp8-amax-history-len 1024 --fp8-amax-compute-algo max)
	if [[ $FP8DPA = true ]]; then
		SUFFIX=$SUFFIX-fp8dpa
		FP8_ARGS+=(--fp8-dot-product-attention)
	fi
fi

WARMUP=$((5*ITERS/2))  # 5BT.
EVAL_INTERVAL=$((ITERS/2))  # 500MT.
EVAL_ITERS=$((ITERS/100))  # 10MT.
if [[ $TOKENS != $DEF_TOKENS ]]; then
	SUFFIX=$SUFFIX-${TOKENS}BT
fi
ITERS=$((ITERS*TOKENS))

if [[ $NODES -ne 1 ]]; then
	SUFFIX=$SUFFIX-nodes$NODES
fi

# Final preparations.
SUFFIX=$SUFFIX$EXTRA_NAME
NAME=llama${SIZE}$SUFFIX
ROOT_PATH=$SCRATCH/op$SCRIPT_VERSION/$NAME
SAVE_PATH=$ROOT_PATH/checkpoints
mkdir -p $SAVE_PATH

#= WRAPPING UP: Set up the _ARGS variables that are going to be used in the end =#
ENVS="CUDA_DEVICE_MAX_CONNECTIONS=1 OMP_NUM_THREADS=\\\$SLURM_CPUS_PER_TASK WANDB_RESUME=allow WANDB_RUN_ID=$NAME"
WANDB_PROJECT=op_$SCRIPT_VERSION

LLAMA_ARGS=(
	--tensor-model-parallel-size $TP
	--pipeline-model-parallel-size $PP
	--seq-length $SEQ_LEN
	--max-position-embeddings $SEQ_LEN
	--tokenizer-type HuggingFaceTokenizer
	--tokenizer-model $TOKENIZER
	--normalization RMSNorm
	--position-embedding-type rope
	--attention-softmax-in-fp32
	--disable-bias-linear
	--transformer-impl transformer_engine
	--num-layers $LAYERS
	--hidden-size $HIDDEN_SIZE
	--group-query-attention
	--num-query-groups $NUM_QUERY_GROUPS
	--ffn-hidden-size $FFN_SIZE
	--num-attention-heads $NUM_HEADS
	--attention-dropout 0.0
	--hidden-dropout 0.0
	--rotary-base 500000
	--rotary-percent 1.0
	--use-rope-scaling
	--bf16
	--adam-eps 0.00000001
	--norm-epsilon 0.00001
)

TRAINING_ARGS=(
	--micro-batch-size $MBS
	--global-batch-size $GBS
	--train-iters $ITERS
	--weight-decay 0.1
	--adam-beta1 0.9 
	--adam-beta2 0.95
	--init-method-std $INIT_STD
	--clip-grad 1.0 
	--lr $LR
	--min-lr 1e-8
)

DISTRIBUTED_ARGS=(
	--master-addr \$MASTER_ADDR
	--node-rank \\\$SLURM_PROCID
	--nproc_per_node 4
	--nnodes $NODES
	--master_port 25678
)

DATA_ARGS=(
	--data-path \$DATA_PATHS
	--data-cache-path $SCRATCH/data/cache
	--split 99,1,0
)

LOGGING=(
	--log-interval 1
	--save-interval $SAVE_FREQ
	--save $SAVE_PATH
	--load $SAVE_PATH
	--tensorboard-dir $ROOT_PATH/tensorboard
	--eval-interval $EVAL_INTERVAL
	--eval-iters $EVAL_ITERS
	--wandb-project $WANDB_PROJECT
	--wandb-exp-name $NAME
	--wandb-save-dir $ROOT_PATH/wandb
	--timing-log-level 1
	--tensorboard-log-interval 1
	--log-progress
	--log-throughput
	--log-timers-to-tensorboard
	--log-validation-ppl-to-tensorboard
	--log-params-norm-per-param
	--log-num-zeros-in-grad
	--log-intermediate-metrics mean rms kurtosis
	--log-params-norm
	--log-memory-to-tensorboard
)

SCHEDULER_ARGS=(
	--lr-decay-style WSD
	--lr-wsd-decay-style 1-sqrt
	--lr-wsd-decay-iters $(($ITERS/5))
	--lr-warmup-iters $WARMUP
)

EXTRA_ARGS+=(
	--use-distributed-optimizer
	--overlap-grad-reduce
	--overlap-param-gather
	--async-save
	--sequence-parallel
)

ARGS="${LLAMA_ARGS[@]} ${TRAINING_ARGS[@]} ${SCHEDULER_ARGS[@]} ${DATA_ARGS[@]} ${LOGGING[@]} ${EXTRA_ARGS[@]} ${FP8_ARGS[@]}"

#= RUNNING: Prepare and launch a slurm script =#
CMD="torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py $ARGS"

mkdir -p $ROOT_PATH
cat > $ROOT_PATH/submission.sbatch <<- EOM
#!/bin/bash
#SBATCH --account=a-a06
#SBATCH --time=$TIME
#SBATCH --job-name=$NAME
#SBATCH --output=$ROOT_PATH/slurmlogs/${NAME}_%j.out
#SBATCH --error=$ROOT_PATH/slurmlogs/${NAME}_%j.err
#SBATCH --nodes=$NODES
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=288
#SBATCH --mem=460000
#SBATCH --environment=/iopsstor/scratch/cscs/ahernnde/ncg_pt.toml
#SBATCH --exclusive
#SBATCH --dependency=singleton

echo Using nodes: \$SLURM_JOB_NODELIST
srun -l bash -c 'echo \$(hostname) \$(nvidia-smi | grep -o "|\\s*[0-9]*MiB")'

DATA_PATHS=\$(find $DATA_DIR -type f | sed -E 's/\.[^.]+$//' | sort -u | tr '\n' ' ')

export WANDB_API_KEY=\$(cat $STORE/users/ahernnde/.keys/wandb.txt)
export MASTER_ADDR=\$(hostname)

srun -l --unbuffered numactl --membind=0-3 bash -c "
	cd $CODE_PATH
	export PYTHONPATH=\$PWD
	eval \"$ENVS\" $CMD
"

EOM
echo "Saved sbatch to $ROOT_PATH/submission.sbatch"
sbatch $ROOT_PATH/submission.sbatch
