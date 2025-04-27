#!/bin/bash

export GPU_MAX_HW_QUEUES=${GPU_MAX_HW_QUEUES:-2}
export TORCH_NCCL_HIGH_PRIORITY=${TORCH_NCCL_HIGH_PRIORITY:-1}
export NCCL_CHECKS_DISABLE=${NCCL_CHECKS_DISABLE:-1}
NCCL_IB_HCA_LIST=$(rdma link -j | python3 -c "import sys, json; links=json.load(sys.stdin);names=[links[i]['ifname'] for i in range(8)]; print(*names,sep=',')")
export NCCL_IB_HCA=${NCCL_IB_HCA:-$NCCL_IB_HCA_LIST}
export NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-3}
export NCCL_CROSS_NIC=${NCCL_CROSS_NIC:-0}
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export NCCL_PROTO=${NCCL_PROTO:-Simple}
export RCCL_MSCCL_ENABLE=${RCCL_MSCCL_ENABLE:-0}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export HSA_NO_SCRATCH_RECLAIM=${HSA_NO_SCRATCH_RECLAIM:-1}


CURRENT_DIR="$( cd "$( dirname "$0" )" && pwd )"
MEGATRON_PATH=$( dirname $( dirname ${CURRENT_DIR}))


GPUS_PER_NODE=`python3 -c "import torch; print(torch.cuda.device_count())"`
EXPERIMENT="mixtral"


RUN_ENV="${RUN_ENV:-cluster}"
if [ $RUN_ENV = cluster ]; then

MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-27777}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"

elif [ $RUN_ENV = slurm ]; then

MASTER_ADDR=${SLURM_MASTER_ADDR}
MASTER_PORT=${SLURM_MASTER_PORT}
NNODES=$SLURM_NNODES
NODE_RANK=${SLURM_NODEID}

fi

WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

MODEL_SIZE=${MODEL_SIZE:-"8x7B"}



TP_SIZE=${TP_SIZE:-8}
PP_SIZE=${PP_SIZE:-1}
EP_SIZE=${EP_SIZE:-8}
ETP_SIZE=${ETP_SIZE:-1}
CP_SIZE=${CP_SIZE:-1}
SP=${SP:-true}
GBS=${GBS:-64}
MBS=${MBS:-4}
SEQLEN=${SEQLEN:-4096}
PR=${PR:-bf16} # bf16, fp16, fp8

TRAIN_ITERS=${TRAIN_ITERS:-10}


LR_WARMUP_ITERS=2
LR_DECAY_ITERS=$(( ${TRAIN_ITERS} - ${LR_WARMUP_ITERS}))


# tokenizer
TOKENIZER_MODEL=${TOKENIZER_MODEL:-"/path/to/tokenizer.model"}

# data
DATA_DIR=${DATA_DIR:-"/workspace/dev/data"}
DATASET_PATH=${DATA_DIR}/wudao_mistralbpe_content_document
VALID_DATASET_PATH=${DATA_DIR}/wudao_mistralbpe_content_document



EXTRA_ARGS=(
    --use-flash-attn
    --no-gradient-accumulation-fusion
    --moe-layer-freq 1
)

# do profile
PROFILE=${PROFILE:-false}
PROFILE_SYNC=${PROFILE_SYNC:-false}
PROFILE_START=${PROFILE_START:-6}
PROFILE_END=${PROFILE_END:-7}
FORCE_BALANCE=${FORCE_BALANCE:-false}
echo "PROFILE: $PROFILE"
echo "PROFILE_START: $PROFILE_START"
echo "PROFILE_END: $PROFILE_END"
echo "FORCE_BALANCE: $FORCE_BALANCE"
echo ""

NAME="${RUN_ENV}-mcore-${MODEL_SIZE}-gbs-${GBS}-mbs-${MBS}-seqlen-${SEQLEN}-pr-${PR}-tp-${TP_SIZE}-pp-${PP_SIZE}-etp-${ETP_SIZE}-ep-${EP_SIZE}-cp-${CP_SIZE}-ac-${AC}-do-${DO}-sp-${SP}-profile${PROFILE}-sync${PROFILE_SYNC}-${TIMESTAMP}"
OUTPUT_BASEPATH=output/${EXPERIMENT}-${NAME}

TENSORBOARD_DIR="${OUTPUT_BASEPATH}/tensorboard/"
CHECKPOINT_PATH="${MEGATRON_PATH}/checkpoint"
TRAIN_LOG=${OUTPUT_BASEPATH}/log/${EXPERIMENT}-${NAME}.log
mkdir -p "${OUTPUT_BASEPATH}/checkpoint/"
mkdir -p "${OUTPUT_BASEPATH}/log/"
mkdir -p ${TENSORBOARD_DIR}
echo "OUTPUT_BASEPATH: $OUTPUT_BASEPATH"
echo "TENSORBOARD_DIR: $TENSORBOARD_DIR"
echo "CHECKPOINT_PATH: $CHECKPOINT_PATH"
echo "TRAIN_LOG: $TRAIN_LOG"
echo ""



if [ $PROFILE = true ]; then
    # blocking kernel
    if [ $PROFILE_SYNC = true ]; then
        export HIP_LAUNCH_BLOCKING=1
    fi

    EXTRA_ARGS+=(
            --profile 
            --profile-ranks 0 
            --use-pytorch-profiler 
            --profile-step-start ${PROFILE_START} 
            --profile-step-end ${PROFILE_END} 
            --moe-router-force-load-balancing
            )
elif [ $FORCE_BALANCE = true ]; then
    EXTRA_ARGS+=(
            --moe-router-force-load-balancing
        )
fi


ENABLE_PROFILING="${ENABLE_PROFILING:-0}" #enable pytorch profiling
if [$ENABLE_PROFILING = true]; then
    EXTRA_ARGS+=(
        --profile
        --use-pytorch-profiler
        --profile-ranks 0
        --profile-step-start 3
        --profile-step-end 4
        --moe-router-force-load-balancing
    )
fi

if [ $PR = fp16 ]; then
    pr_options=(
		    --fp16 
            --apply-query-key-layer-scaling)
    export NVTE_APPLY_QK_LAYER_SCALING=1
elif [ $PR = bf16 ]; then
    pr_options=(
        --bf16)
elif [ $PR = fp8 ]; then
    TRANSFORMER_IMPL=transformer_engine
    pr_options=(
        --bf16
        --fp8-format hybrid 
        --fp8-amax-compute-algo max 
        --fp8-amax-history-len 1024)
else
    profile_options=()
fi

# perf options
OPTIMIZER_OFFLOAD=false
GEMM_TUNING="${GEMM_TUNING:-1}"
USE_GROUPED_GEMM="${USE_GROUPED_GEMM:-true}"
MOE_USE_LEGACY_GROUPED_GEMM="${MOE_USE_LEGACY_GROUPED_GEMM:-true}"
NVTE_CK_USES_BWD_V3="${NVTE_CK_USES_BWD_V3:-1}"
GPT_LAYER_IN_TE="${GPT_LAYER_IN_TE:-true}"
echo "GEMM_TUING: $GEMM_TUNING"
echo "USE_GROUPED_GEMM: $USE_GROUPED_GEMM"
echo "MOE_USE_LEGACY_GROUPED_GEMM: $MOE_USE_LEGACY_GROUPED_GEMM"
echo "NVTE_CK_USES_BWD_V3: $NVTE_CK_USES_BWD_V3"
echo "GPT_LAYER_IN_TE: $GPT_LAYER_IN_TE"
echo ""


# memory options
AC="${AC:-none}" #none #sel #full
export RECOMPUTE_METHOD=${RECOMPUTE_METHOD:-block} # block uniform
export RECOMPUTE_NUM_LAYERS=${RECOMPUTE_NUM_LAYERS:-1}
echo "AC: $AC"
echo "RECOMPUTE_METHOD: $RECOMPUTE_METHOD"
echo "RECOMPUTE_NUM_LAYERS: $RECOMPUTE_NUM_LAYERS"
echo ""


if [ $AC = full ]; then
    activation_checkpoint_options=(
		    --recompute-method ${RECOMPUTE_METHOD} 
		    --recompute-granularity full 
            --recompute-num-layers ${RECOMPUTE_NUM_LAYERS})
elif [ $AC = sel ]; then
    activation_checkpoint_options=(
        --recompute-activations)
elif [ $AC = none ]; then
    activation_checkpoint_options=()
    
fi

if [ $USE_GROUPED_GEMM = true ]; then
    USE_GROUPED_GEMM_OPTION="--moe-grouped-gemm"
else
    USE_GROUPED_GEMM_OPTION=""
fi

if [ $MOE_USE_LEGACY_GROUPED_GEMM = true ]; then
    USE_LEGACY_GROUPED_GEMM_OPTION="--moe-use-legacy-grouped-gemm"
else
    USE_LEGACY_GROUPED_GEMM_OPTION=""
    # disable gemm tuning when using TE Group GEMM.
    GEMM_TUNING=0
    echo "[WARN] GEMM tuning is disabled when using TransformerEngine Group GEMM."
fi

# gemm tuning, https://github.com/ROCm/TransformerEngine
if [ "$GEMM_TUNING" -eq 1 ]; then
   export TE_HIPBLASLT_TUNING_RUN_COUNT=10
   export TE_HIPBLASLT_TUNING_ALGO_COUNT=50
else
   unset TE_HIPBLASLT_TUNING_RUN_COUNT
   unset TE_HIPBLASLT_TUNING_ALGO_COUNT
fi

if [ $GPT_LAYER_IN_TE = true ]; then
    TRANSFORMER_IMPL=transformer_engine
else
    TRANSFORMER_IMPL=local
fi



if [ -z ${MP_VP} ]; then
    vp_options=()
else
    vp_options=(
        --num-layers-per-virtual-pipeline-stage ${MP_VP} 
        )
fi

if [ "$NVTE_CK_USES_BWD_V3" -eq 1 ]; then
    echo "Using BWD FAv3"
    export NVTE_CK_USES_BWD_V3=1    #by default 0, if set to 1, some cases will call the bwd v3 dqdkdv kernel;
    export NVTE_CK_V3_ATOMIC_FP32=0 #by default 1, if set to 0 will use atomic fp16/bf16(w/o convert_dq kernel) when NVTE_CK_USES_BWD_V3 is set to 1;
    export NVTE_CK_V3_SPEC=1        #by default 0, if set to 1 will call the specialized v3 kernel when NVTE_CK_USES_BWD_V3 is set to 1;
else
    echo "Disabling BWD FAv3"
    export NVTE_CK_USES_BWD_V3=0
    export NVTE_CK_V3_ATOMIC_FP32=1
    export NVTE_CK_V3_SPEC=0
fi



DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NNODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)


if [ "$MODEL_SIZE" = "8x7B" ]; then
    NUM_LAYERS=${NUM_LAYERS:-32}
elif [ "$MODEL_SIZE" = "8x22B" ]; then
    NUM_LAYERS=${NUM_LAYERS:-56}
else
    echo "invalid model size"
    exit 1
fi

if [ "$MODEL_SIZE" = "8x7B" ]; then
MODEL_ARGS=(
    --use-mcore-models
    --disable-bias-linear
    --seq-length ${SEQLEN}
    --max-position-embeddings 32768
    --num-layers $NUM_LAYERS
    --hidden-size 4096
    --ffn-hidden-size 14336
    --num-attention-heads 32
    --init-method-std 0.01
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --normalization RMSNorm
    --position-embedding-type rope
    --swiglu
    --untie-embeddings-and-output-weights
    --group-query-attention
    --num-query-groups 8
    --no-masked-softmax-fusion
    --no-position-embedding
    --rotary-base 1000000
)
elif [ "$MODEL_SIZE" = "8x22B" ]; then
MODEL_ARGS=(
    --use-mcore-models
    --disable-bias-linear
    --seq-length ${SEQLEN}
    --max-position-embeddings 65536
    --num-layers $NUM_LAYERS
    --hidden-size 6144
    --ffn-hidden-size 16384
    --num-attention-heads 48
    --init-method-std 0.01
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --normalization RMSNorm
    --position-embedding-type rope
    --swiglu
    --untie-embeddings-and-output-weights
    --group-query-attention
    --num-query-groups 8
    --no-masked-softmax-fusion
    --no-position-embedding
    --rotary-base 1000000
)
else
    echo "invalid model size"
    exit 1
fi

MOE_ARGS=(
    --num-experts 8
    --moe-router-topk 2
    --moe-router-load-balancing-type aux_loss
    --moe-aux-loss-coeff 1e-2
    --transformer-impl ${TRANSFORMER_IMPL} 
    $USE_GROUPED_GEMM_OPTION 
    $USE_LEGACY_GROUPED_GEMM_OPTION 
    --moe-token-dispatcher-type alltoall
    --overlap-grad-reduce
    --overlap-param-gather
)

MOCK_DATA="${MOCK_DATA:-1}" # 1: use mock data, 0: use real data

# For multi-node runs DATA_CACHE_PATH should point to a common path accessible by all the nodes (for eg, an NFS directory)
DATA_CACHE_PATH="${DATA_CACHE_PATH:-../.cache}"


DATA_ARGS=(
    --tokenizer-type Llama2Tokenizer
    --tokenizer-model ${TOKENIZER_MODEL}
    --split 9900,80,20
)

if [ $MOCK_DATA = 1 ];then
	EXTRA_ARGS+=( 
	--mock-data 
	--data-cache-path $DATA_CACHE_PATH
	)
else
	EXTRA_ARGS+=( 
	--data-path $DATASET_PATH 
	--data-cache-path ${DATA_CACHE_PATH} 
	)
    echo "DATASET_PATH: $DATASET_PATH"
    echo "VALID_DATASET_PATH: $VALID_DATASET_PATH"
    echo ""
fi


TRAINING_ARGS=(
    --micro-batch-size ${MBS}
    --global-batch-size ${GBS}
    --lr 1e-4
    --train-iters ${TRAIN_ITERS}
    --lr-decay-iters ${LR_DECAY_ITERS}
    --lr-decay-style cosine
    --min-lr 1.0e-5
    --weight-decay 0.1
    --lr-warmup-iters ${LR_WARMUP_ITERS}
    --clip-grad 1.0
    --seed 1234
    --ckpt-format torch 
)
    #--save ${CHECKPOINT_PATH} 
    #--load ${CHECKPOINT_PATH} 

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size ${TP_SIZE}
    --pipeline-model-parallel-size ${PP_SIZE}
    --expert-model-parallel-size ${EP_SIZE}
    --expert-tensor-parallel-size ${ETP_SIZE}
    --context-parallel-size ${CP_SIZE} 
    --use-distributed-optimizer
)
if [ $SP = true ] && [ "$TP_SIZE" -gt 1 ]; then
    MODEL_PARALLEL_ARGS+=( --sequence-parallel )
fi

LOGGING_ARGS=(
    --log-interval 1
    --save-interval 500
    --eval-interval 1000000
    --eval-iters -1
    --log-throughput
    --tensorboard-log-interval 1
    --tensorboard-dir ${TENSORBOARD_DIR}/tensorboard
)


if [ -n "${WANDB_API_KEY}" ]; then
    LOGGING_ARGS+=(
        --wandb-project ${WANDB_PROJECT:-"Mixtral"}
        --wandb-exp-name ${WANDB_NAME:-${MODEL_SIZE}}
        --wandb-save-dir logs/wandb 
    )
fi

command="torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py \
    ${MODEL_ARGS[@]} \
    ${pr_options[@]} \
    ${activation_checkpoint_options[@]} \
    ${MOE_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${LOGGING_ARGS[@]} \
    ${EXTRA_ARGS[@]}"

command="$command  2>&1 | tee $TRAIN_LOG"

echo "launch_command=${command}"
eval ${command}
echo ""


if [ "$RUN_ENV" = "cluster" ] || ( [ "$RUN_ENV" = "slurm" ] && [ "$SLURM_NODEID" = "$((NNODES - 1))" ] ); then
echo 'import argparse
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                        prog="Process Log")
    parser.add_argument("filename")
    args = parser.parse_args()

    with open(args.filename) as f:
        lines = f.readlines()
    lines = lines[1:-1]
    lines = [float(a) for a in lines]
    mean = np.mean(np.array(lines))
    print(mean)' > mean_log_value.py


echo '============================================================================================================'
grep -Eo 'throughput per GPU [^|]*' $TRAIN_LOG | sed -E 's/.*throughput per GPU \(TFLOP\/s\/GPU\): ([0-9\.]+).*/\1/' > tmp.txt
echo "throughput per GPU: $(python mean_log_value.py tmp.txt)" |& tee -a $TRAIN_LOG
THROUGHPUT=$(python mean_log_value.py tmp.txt)
rm tmp.txt

echo '============================================================================================================'
grep -Eo 'elapsed time per iteration [^|]*' $TRAIN_LOG | sed -E 's/.*elapsed time per iteration \(ms\): ([0-9\.]+).*/\1/' > tmp.txt
echo "elapsed time per iteration: $(python mean_log_value.py tmp.txt)" |& tee -a $TRAIN_LOG

TIME_PER_ITER=$(python mean_log_value.py tmp.txt 2>/dev/null | awk '{printf "%.6f", $0}')
PERFORMANCE=$(awk -v bs="$GBS" -v sl="$SEQLEN" -v tpi="$TIME_PER_ITER" -v ws="$((NNODES * GPUS_PER_NODE))" 'BEGIN {printf "%.6f", bs * sl * 1000/ (tpi * ws)}')
echo "tokens/GPU/s: $PERFORMANCE" |& tee -a $TRAIN_LOG
rm tmp.txt

fi
