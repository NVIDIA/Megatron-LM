#= Prelude =#
# Some constants.
SCRIPT_VERSION=v1
SEQ_LEN=4096
TOKENIZER=/capstor/store/cscs/swissai/a06/users/ahernnde/swissai-tokenizer/
DATA_DIR=/capstor/store/cscs/swissai/a06/datasets_tokenized/megatron/sai/swissai-fineweb-edu-filterrobots-merge
CODE_PATH=$STORE/users/ahernnde/workspace/AleHD-Megatron-LM


ACTIVATION=swiglu
QK_IMPL=apex
DYT_ALPHA=1.0
NODES=1
TIME=1-00:00:00
MARGIN=0
FP8_FIRST_AND_LAST=0
PARTITION=normal
PRECISION=hybrid
FP8LEN=1024
WEIGHT_DECAY=0.1
MIN_LR=1e-8

# Usage function.
usage () {
	echo "Usage: llama.sh <size> [options...]"
	echo "<size>: 300/1/8"
	echo "Options:"
	# Misc settings.
	echo " --debug: Displays this message"
	echo " --nodes <nodes>: How many nodes to use"
	echo " --extra-name <name>: Add a suffix to the name"
	echo " --time (default=$TIME): Change the sbatch time limit"
	# FP8 settings.
	echo " --fp8: Enables fp8"
	echo " --fp8-dpa: Enables fp8 DPA"
	echo " --fp8-tensorwise: Use tensorwise scaling"
	echo " --fp8-first-and-last-bf16 <int>: Specifies how many first and last layers are done in bf16"
	echo " --fp8-margin: fp8 margin"
	echo " --fp8-dpa-nobwd: Disables fp8 DPA for backward"
	echo " --fp8-e4m3: Use e4m3 fp8 precision instead of hybrid"
	echo " --fp8-len <int>: fp8 history length"
	# Training settings..
	echo " --tokens <int> ): Amount of tokens to train with (in B)."
	echo " --lr <float>: Learning rate."
	echo " --cooldown-wd: When set weight decay will be cooldown."
	# Architecture settings.
	echo " --init <float>: Change init std."
	echo " --no-prenorm: Disables pre-layernorm."
	echo " --postnorm: Enables post-layernorm."
	echo " --qknorm: Enables qk norm."
	echo " --qkimpl <te|apex|torch>: QK Implementation."
	echo " --qkinit <float>: Sets qk norm initialization to this value."
	echo " --qk-dyt: Enables QK DyT (instead of RMS)"
	echo " --dyt-alpha <float>: QK DyT alpha initialization"
	echo " --softmax-scale: Sets attention softmax scale."
	echo " --layerscale: Enables layerscale."
	echo " --layerscale-value <float>: Layerscale init value."
	echo " --input-upscale: Upscales input by 1/std."
	echo " --alpha: MLP alpha."
	echo " --activation (default=$ACTIVATION): MLP activation. Choices=[swiglu, gelu, xielu]."
	# Opt settings.
	echo " --decay-qkgains: Decay qk layernorm gains"
	echo " --no-train-qk-gains: Don't train QK layernorm gains"
	# Logs.
	echo " --log-grad: Log individual grad norms."
	echo " --wandb-name <str>: Specify wandb name."
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
UNTIE=true
INTERMEDIATE_METRICS_INTERVAL=1
if [[ $1 -eq 300 ]]; then 
	# batch_size: ~0.52M.
	# tok/sec/gpu: ~59.5k  (4nodes, bf16).
	# 50B ETA: ~14h30m. (4nodes, bf16).
	# ckpt freq: ~1h30m.
	LAYERS=16
	HIDDEN_SIZE=1024
	FFN_SIZE=4096
	NUM_HEADS=8
	NUM_QUERY_GROUPS=4
	MBS=8
	GBS=128
	ITERS=2000  # 1BT.
	LR=0.001
	INIT_STD=0.001
	SIZE=390M
	SAVE_FREQ=10000
	DEF_TOKENS=50
	UNTIE=false
elif [[ $1 -eq 1 ]]; then 
	# batch_size: ~1.05M.
	# tok/sec/gpu: ~38.5k (8nodes, bf16).
	# 125B ETA: ~29h (8nodes, bf16).
	# ckpt freq: ~1h10m (8nodes, bf16).
	LAYERS=16
	HIDDEN_SIZE=2048
	FFN_SIZE=8192
	NUM_HEADS=16
	NUM_QUERY_GROUPS=8
	MBS=4
	GBS=256
	ITERS=1000  # 1BT.
	LR=0.0005
	INIT_STD=0.02
	SIZE=1.5B
	SAVE_FREQ=5000
	DEF_TOKENS=125
	INTERMEDIATE_METRICS_INTERVAL=100
elif [[ $1 -eq 8 ]]; then 
	# batch_size: ~1.1M.
	TP=4  # TODO: TP=1 is faster but we need DP>=64 for this.
	LAYERS=32
	HIDDEN_SIZE=4096
	FFN_SIZE=14336
	NUM_HEADS=32
	NUM_QUERY_GROUPS=8
	MBS=4
	GBS=512
	ITERS=500  # 1BT.
	LR=0.0005  # TODO: Previously baseline lr=0.00005, OP lr=0.0003.
	INIT_STD=0.02  # TODO: Most likely OP will need larger.
	SIZE=8B
	SAVE_FREQ=2500
	DEF_TOKENS=250
	INTERMEDIATE_METRICS_INTERVAL=1000
else
	>&2 echo "Invalid llama size: $1"
	usage
	exit 1
fi
shift

# Now get the general options.
TOKENS=$DEF_TOKENS
ENVS=""
SUFFIX=""
while [[ $# -gt 0 ]]; do
	case $1 in
		--debug)
			SCRIPT_VERSION=$SCRIPT_VERSION-debug;
			TIME=00:30:00
			DEBUG=true
			shift;;
		--nodes)
			NODES=$2; shift 2;;
		--time)
			TIME=$2; shift 2;;
		--fp8)
			FP8=true; shift;;
		--fp8-tensorwise)
			TENSORWISE=true; shift;;
		--fp8-first-and-last-bf16)
			FP8_FIRST_AND_LAST=$2; shift 2;;
		--fp8-dpa)
			FP8DPA=true; shift;;
		--fp8-margin)
			MARGIN=$2; shift 2;;
		--fp8-dpa-nobwd)
			FP8DPA_NOBWD=true; shift;;
		--fp8-e4m3)
			PRECISION=e4m3; shift;;
		--fp8-len)
			FP8LEN=$2; shift 2;;
		--extra-name)
			EXTRA_NAME="-$2"; shift 2;;
		--tokens)
			TOKENS=$2; shift 2;;
		--init)
			NEW_INIT_STD=$2; shift 2;;
		--lr)
			LR=$2; 
			CHANGED_LR=true
			shift 2;;
		--cooldown-wd)
			COOLDOWN_WD=true; shift;;
		--no-prenorm)
			PRENORM=false; shift;;
		--postnorm)
			POSTNORM=true; shift;;
		--qknorm)
			QKNORM=true; shift;;
		--qkimpl)
			QK_IMPL=$2; shift 2;;
		--qkinit)
			QK_INIT=$2; shift 2;;
		--qk-dyt)
			QK_DYT=true; shift;;
		--dyt-alpha)
			DYT_ALPHA=$2; shift 2;;
		--softmax-scale)
			SOFTMAX_SCALE=$2; shift 2;;
		--layerscale)
			LAYERSCALE=true; shift;;
		--layerscale-value)
			LAYERSCALE_VALUE=$2; shift 2;;
		--input-upscale)
			INPUT_UPSCALE=true; shift;;
		--activation)
			ACTIVATION=$2; shift 2;;
		--alpha)
			ALPHA=$2; shift 2;;
		--decay-qkgains)
			DECAY_GAINS=true; shift;;
		--no-train-qk-gains)
			NOTRAIN_GAINS=true; shift;;
		--log-grad)
			LOG_GRADS=true; shift;;
		--wandb-name)
			WANDB_NAME=$2; shift 2;;
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
	FP8_ARGS+=(--fp8-margin $MARGIN --fp8-format $PRECISION --fp8-amax-history-len $FP8LEN --fp8-amax-compute-algo max)
	if [[ $FP8DPA = true ]]; then
		SUFFIX=$SUFFIX-fp8dpa
		FP8_ARGS+=(--fp8-dot-product-attention)
		if [[ $FP8DPA_NOBWD = true ]]; then
			SUFFIX=$SUFFIX-dpaNObwd
			ENVS="$ENVS NVTE_FP8_DPA_BWD=0"
		fi
	fi
	if [[ $TENSORWISE = true ]]; then
		SUFFIX=$SUFFIX-fp8TW
		FP8_ARGS+=(--fp8-recipe tensorwise)
		MAYBE_INSTALL_TE="NVTE_FRAMEWORK=pytorch pip install git+https://github.com/NVIDIA/TransformerEngine.git@release_v2.2"
	elif [[ $MARGIN -ne 0 ]]; then
		SUFFIX=$SUFFIX-fp8margin$MARGIN
	fi
	if [[ $PRECISION != hybrid ]]; then
		SUFFIX=$SUFFIX-fp8$PRECISION
	fi
	if [[ $FP8LEN -ne 1024 ]]; then
		SUFFIX=$SUFFIX-fp8len$FP8LEN
	fi
	if [[ $FP8_FIRST_AND_LAST -ge 0 ]]; then
		SUFFIX=$SUFFIX-fp8safe$FP8_FIRST_AND_LAST
		FP8_ARGS+=(--first-last-layers-bf16 --num-layers-at-start-in-bf16 $FP8_FIRST_AND_LAST --num-layers-at-end-in-bf16 $FP8_FIRST_AND_LAST)
	fi
fi

ARCH_ARGS=()
if [[ ! -z ${NEW_INIT_STD+x} ]]; then
	SUFFIX=$SUFFIX-std$NEW_INIT_STD
	INIT_STD=$NEW_INIT_STD
fi
if [[ $PRENORM = false ]]; then
	SUFFIX=$SUFFIX-nopre
	ARCH_ARGS+=(--no-pre-layer-norm)
fi
if [[ $POSTNORM = true ]]; then
	SUFFIX=$SUFFIX-postln
	ARCH_ARGS+=(--post-layer-norm)
fi
if [[ $QKNORM = true ]]; then
	ARCH_ARGS+=(--qk-layernorm)
	if [[ $QK_DYT = true ]]; then
		SUFFIX=$SUFFIX-qkDyT
		if [[ $DYT_ALPHA != 1.0 ]]; then
			SUFFIX=$SUFFIX-DyTalpha$DYT_ALPHA
		fi
		ARCH_ARGS+=(--qk-dyt --dyt-alpha-init $DYT_ALPHA)
	else
		SUFFIX=$SUFFIX-qknorm
		if [[ $QK_IMPL != apex ]]; then
			SUFFIX=$SUFFIX-qk$QK_IMPL
		fi
		ARCH_ARGS+=(--qknorm-impl $QK_IMPL)
		if [[ $QK_IMPL = torch ]]; then
			ARCH_ARGS+=(--no-persist-layer-norm)
		fi
	fi

	if [[ ! -z ${QK_INIT+x} ]]; then
		SUFFIX=$SUFFIX-qkinit$QK_INIT
		ARCH_ARGS+=(--qknorm-init $QK_INIT)
	fi
fi
if [[ ! -z ${SOFTMAX_SCALE+x} ]]; then
	SUFFIX=$SUFFIX-softmax$SOFTMAX_SCALE
	ARCH_ARGS+=(--softmax-scale $SOFTMAX_SCALE)
fi
if [[ $LAYERSCALE = true ]]; then
	SUFFIX=$SUFFIX-ls
	if [[ ! -z ${LAYERSCALE_VALUE+x} ]]; then
		BETA=$LAYERSCALE_VALUE
		SUFFIX=$SUFFIX$BETA
	else
		BETA=$(echo "print(1/$LAYERS**0.5)" | python3)
	fi
	if [[ $POSTNORM = true ]] && [[ $PRENORM = false ]]; then
		ARCH_ARGS+=(--layernorm-init $BETA)
	else
		ARCH_ARGS+=(--layer-scale $BETA)
	fi
fi
if [[ $INPUT_UPSCALE = true ]]; then
	SUFFIX=$SUFFIX-is
	MULT=$(echo "print(1/$INIT_STD)" | python3)
	ARCH_ARGS+=(--input-embeddings-multiplier $MULT)
fi

if [[ $ACTIVATION != swiglu ]] && [[ $ACTIVATION != gelu ]] && [[ $ACTIVATION != xielu ]]; then
	>&2 echo Unknown activation: $ACTIVATION
	exit 1
fi
if [[ $ACTIVATION != swiglu ]]; then
	SUFFIX=$SUFFIX-$ACTIVATION
	FFN_SIZE=$((3*$FFN_SIZE/2))
fi
if [[ $ACTIVATION != gelu ]]; then
	ARCH_ARGS+=(--$ACTIVATION)
fi
if [[ ! -z ${ALPHA+x} ]]; then
	SUFFIX=$SUFFIX-mlp$ALPHA
	ARCH_ARGS+=(--mlp-alpha $ALPHA)
fi

OPT_ARGS=()
if [[ $DECAY_GAINS = true ]]; then
	SUFFIX=$SUFFIX-decayQKgains
	OPT_ARGS+=(--weight-decay-qk-gains)
fi
if [[ $NOTRAIN_GAINS = true ]]; then
	SUFFIX=$SUFFIX-ntQKgain
	OPT_ARGS+=(--no-train-qk-gains)
fi

if [[ $CHANGED_LR = true ]]; then
	SUFFIX=$SUFFIX-lr$LR
fi

WARMUP=$((5*ITERS/2))  # 2.5BT.
EVAL_INTERVAL=$((ITERS/2))  # 500MT.
EVAL_ITERS=$((ITERS/100))  # 10MT.
if [[ $TOKENS != $DEF_TOKENS ]]; then
	SUFFIX=$SUFFIX-${TOKENS}BT
fi
ITERS=$((ITERS*TOKENS))
DECAY_ITERS=$(($ITERS/5))

if [[ $COOLDOWN_WD = true ]]; then
	SUFFIX=$SUFFIX-coolWD
	MIN_COOLDOWN=$(echo "print($MIN_LR*$WEIGHT_DECAY/$LR)" | python3)
	OPT_ARGS+=(--end-weight-decay-cooldown $MIN_COOLDOWN --weight-decay-cooldown-iters $DECAY_ITERS --weight-decay-cooldown-style 1-sqrt)
fi

EXTRA_LOGS=()
EXTRA_OPT=()
if [[ $LOG_GRADS = true ]]; then
	EXTRA_LOGS+=(--log-indiv-grad-norm)
else
	EXTRA_OPT+=(--use-distributed-optimizer --overlap-param-gather)
fi

# Final preparations.
SUFFIX=$SUFFIX$EXTRA_NAME
NAME=llama${SIZE}$SUFFIX
JOB_NAME=$NAME
ROOT_PATH=$SCRATCH/op$SCRIPT_VERSION/$NAME
SAVE_PATH=$ROOT_PATH/checkpoints
DIFFS_PATH=$ROOT_PATH/diffs
mkdir -p $SAVE_PATH
mkdir -p $DIFFS_PATH

if [[ $NODES -eq 1 ]] && [[ $DEBUG = true ]]; then
	PARTITION=debug
fi
if [[ $DEBUG = true ]]; then
	JOB_NAME=$NAME-debug
fi
if [[ -z ${WANDB_NAME+x} ]]; then
	WANDB_NAME=$NAME
fi

#= WRAPPING UP: Set up the _ARGS variables that are going to be used in the end =#
ENVS="$ENVS CUDA_DEVICE_MAX_CONNECTIONS=1 OMP_NUM_THREADS=\\\$SLURM_CPUS_PER_TASK WANDB_RESUME=allow WANDB_RUN_ID=$NAME"
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
if [[ $UNTIE = true ]]; then
	LLAMA_ARGS+=(--untie-embeddings-and-output-weights)
fi

TRAINING_ARGS=(
	--micro-batch-size $MBS
	--global-batch-size $GBS
	--train-iters $ITERS
	--weight-decay $WEIGHT_DECAY
	--adam-beta1 0.9 
	--adam-beta2 0.95
	--init-method-std $INIT_STD
	--clip-grad 1.0 
	--lr $LR
	--min-lr $MIN_LR
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
	--wandb-exp-name $WANDB_NAME
	--wandb-save-dir $ROOT_PATH/wandb
	--timing-log-level 1
	--tensorboard-log-interval 1
	--log-progress
	--log-throughput
	--log-timers-to-tensorboard
	--log-validation-ppl-to-tensorboard
	--log-intermediate-metrics mean rms kurtosis
	--log-intermediate-metrics-interval $INTERMEDIATE_METRICS_INTERVAL
	--log-params-norm-per-param
	--log-num-zeros-in-grad
	--log-params-norm
	--log-memory-to-tensorboard
	--log-weight-decay
)
LOGGING=(${LOGGING[@]} ${EXTRA_LOGS[@]})

SCHEDULER_ARGS=(
	--lr-decay-style WSD
	--lr-wsd-decay-style 1-sqrt
	--lr-wsd-decay-iters $DECAY_ITERS
	--lr-warmup-iters $WARMUP
)

EXTRA_ARGS+=(
	--overlap-grad-reduce
	--async-save
	--sequence-parallel
)
EXTRA_ARGS=(${EXTRA_ARGS[@]} ${EXTRA_OPT[@]})

ARGS="${LLAMA_ARGS[@]} ${TRAINING_ARGS[@]} ${SCHEDULER_ARGS[@]} ${DATA_ARGS[@]} ${LOGGING[@]} ${EXTRA_ARGS[@]} ${FP8_ARGS[@]} ${ARCH_ARGS[@]} ${OPT_ARGS[@]}"

#= RUNNING: Prepare and launch a slurm script =#
CMD="torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py $ARGS"

mkdir -p $ROOT_PATH
cat > $ROOT_PATH/submission.sbatch <<- EOM
#!/bin/bash
#SBATCH --account=a-a06
#SBATCH --time=$TIME
#SBATCH --job-name=$JOB_NAME
#SBATCH --output=$ROOT_PATH/slurmlogs/%j.out
#SBATCH --error=$ROOT_PATH/slurmlogs/%j.err
#SBATCH --nodes=$NODES
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=288
#SBATCH --mem=460000
#SBATCH --environment=/iopsstor/scratch/cscs/ahernnde/ncg_pt.toml
#SBATCH --exclusive
#SBATCH --dependency=singleton
#SBATCH --partition=$PARTITION

echo Using nodes: \$SLURM_JOB_NODELIST
srun -l bash -c 'echo \$(hostname) \$(nvidia-smi | grep -o "|\\s*[0-9]*MiB")'

# Log git status.
cd $CODE_PATH
echo ---------
echo git status:
git status
echo git log:
git log -n 1
echo ---------
git diff > $DIFFS_PATH/\$SLURM_JOBID.diff

DATA_PATHS=\$(find $DATA_DIR -type f | sed -E 's/\.[^.]+$//' | sort -u | tr '\n' ' ')

export WANDB_API_KEY=\$(cat $STORE/users/ahernnde/.keys/wandb.txt)
export MASTER_ADDR=\$(hostname)

srun -l --unbuffered numactl --membind=0-3 bash -c "
	cd $CODE_PATH
	$MAYBE_INSTALL_TE
	export PYTHONPATH=\$PWD
	eval \"$ENVS\" $CMD
"

EOM
echo "Saved sbatch to $ROOT_PATH/submission.sbatch"
sbatch $ROOT_PATH/submission.sbatch
