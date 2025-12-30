#!/bin/bash

# set -euo pipefail
export DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PYTHONPATH=

export NCCL_IB_SL=1
export TOKENIZERS_PARALLELISM="false"

#export NVTE_DEBUG=1
#export NVTE_DEBUG_LEVEL=2

USER="wuguohao"

# Auto-detect batch or interactive mode.
which srun
BATCH=$((1-$?))

DEBUG=0
USE_TILING=1
USE_CP=0
USE_TE_CE=1
USE_FLASH_ATTN=0
USE_FSDP=0
PROFILE=0
PROFILE_MEMORY=0
PROFILE_MEMORY_PATH="/m2v_model/wuguohao03/nv_teamwork/Megatron-LM/logs/output/interactive_hybrid_cp/profile"
# PROFILE_RANKS=[0,1,2,3,4,5,6,7,8]
TRAIN_ITERS=10
USE_MOCK_DATA=1
MASTER_PORT=6103
TP=1
PP=8
PP_l=
MIN_CP=1
MAX_CP=8
NUM_LAYERS=8

MBZ=1
BZ=2048
HIDDEN_SIZE=5120
FFN_HIDDEN_SIZE=13824
HEAD_DIM=128
NUM_HEAD=$((HIDDEN_SIZE / HEAD_DIM))
SEQ_LEN=131072 #131072 #81920 #65536 # 32768 #16384
MIN_SEQ_LEN=256
MAX_SEQLEN_PER_DP_CP_RANK=65536
NW=16
AD=0.0
HD=0.0
LI=1
EXTRA_ARGS=""
NONDETERMINISTIC_ATTN=1
# NUM_GPU=8

# Remember to update model and job name if running in batch mode!!
# if [[ $BATCH -eq 0 ]]; then
#     DATETIME=`date +'%y-%m-%d-%H-%M-%S'`
#     MODEL_NAME="interactive_hybrid_cp"
#     WORKSPACE="/home/tailaim//work_data/megatron-lm/logs"
#     SOURCE="/home/tailaim/work_data/megatron-lm"
#     TOKENIZER="/home/tailaim/work_data/megatron-moe-scripts/Nemotron-H-4B-Instruct"
# else
#     MODEL_NAME="interactive_hybrid_cp"
#     WORKSPACE="/lustre/fsw/portfolios/coreai/users/tailaim/work_data/megatron-lm/logs"
#     SOURCE="/lustre/fsw/portfolios/coreai/users/tailaim/work_data/megatron-lm"
#     TOKENIZER="/lustre/fsw/portfolios/llmservice/users/kezhik/images/Nemotron-H-4B-Instruct"
# fi

HOSTFILE=${HOSTFILE:-}
if [ -f /etc/mpi/hostfile ]; then
    if [ ! -f /etc/mpi/hostfile_seq -a -z "$HOSTFILE" ]; then
        echo "Please use kai_launch to generate /etc/mpi/hostfile_seq"
        exit 1
    fi
    HOSTFILE=${HOSTFILE:-/etc/mpi/hostfile_seq}
fi

if [ -n "$HOSTFILE" ]; then
    # 多机任务
    if [ -z "${MY_NODE_IP:-}" ]; then echo "Variable MY_NODE_IP does not exist."; exit 1; fi
    if ! ifconfig | grep " $MY_NODE_IP " >/dev/null; then echo "MY_NODE_IP \"$MY_NODE_IP\" is not contained in \`ifconfig\`."; exit 1; fi
    MASTER_ADDR=$MY_NODE_IP
    if [ ! -f "$HOSTFILE" ]; then echo "Hostfile \"$HOSTFILE\" does not exist."; exit 1; fi
    NP=${NP:-$(cat "$HOSTFILE" | grep -v '^#' | grep -oP 'slots=\K\d+' | awk '{sum += $1} END {print sum}')}
else
    # 单机任务
    MASTER_ADDR=127.0.0.1
    NP=${NP:-$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)}
fi
# NP=16

function check_str() {
    if [ ! -v "$1" ]; then echo "Variable $1 is not set."; exit 1; fi
    if [[ -z "${!1}" ]]; then echo "Variable $1 is not a string."; exit 1; fi
}

# PLM_RSH_ARGS
if [ -v TARGET_IP_PORT_FILE ]; then check_str TARGET_IP_PORT_FILE; PLM_RSH_ARGS="-F $TARGET_IP_PORT_FILE";
elif [ ! -v TARGET_IP_PORT_FILE -a -n "$HOSTFILE" ]; then PORT=$(cat /etc/ssh/ssh_config | grep 'Port' | cut -d'"' -f2); check_integer PORT; PLM_RSH_ARGS="-p $PORT";
else PLM_RSH_ARGS=;
fi


MODEL_NAME="interactive_hybrid_cp"
TOKENIZER=None

WORKSPACE="/m2v_model/wuguohao03/nv_teamwork/Megatron-LM/logs"
SOURCE="/m2v_model/wuguohao03/nv_teamwork/Megatron-LM"
OUTPUT_BASE="${WORKSPACE}/output"
OUTPUT="${OUTPUT_BASE}/${MODEL_NAME}"

FINETUNE_DIR=${OUTPUT}/checkpoints
LOGS_DIR="${OUTPUT}/logs"
TENSORBOARD_DIR="${OUTPUT}/tensorboard"
DATACACHE_DIR="${OUTPUT}/data_cache"

export HF_DATASETS_CACHE="${OUTPUT}/hf_datasets_cache"

DATA_TRAIN="/home/tailaim/data/thd_formatted_100k.jsonl"


# if [[ $DEBUG -eq 1 ]]; then
#     MBZ=1
#     BZ=256
#     NW=4
#     AD=0.0
#     HD=0.0
#     LI=1

#     # EXTRA_ARGS="--deterministic-mode --use-cpu-initialization"

#     NONDETERMINISTIC_ATTN=1

#     NUM_GPU=8
#     export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

#     #export NCCL_ALGO=Tree
#     #export CUBLAS_WORKSPACE_CONFIG=:4096:8
# else
# fi


if [[ $USE_TE_CE -eq 1 ]]; then
    EXTRA_ARGS+=" --cross-entropy-loss-fusion --cross-entropy-fusion-impl te"
fi

if [[ $PROFILE -eq 1 ]]; then
    EXTRA_ARGS+=" --profile --profile-step-start 2 --profile-step-end 6 "
fi

echo $USE_MOCK_DATA
if [[ $USE_MOCK_DATA -eq 1 ]]; then
    # EXTRA_ARGS+=" --mock-data --sft-mock-dataset-config-json '{\"mode\":\"file\",\"path\":\"path/to/file\"}'"
    if [[ $BATCH -eq 0 ]]; then
    # EXTRA_ARGS+=" --mock-data --sft-mock-dataset-config-json {\"mode\":\"distribution\",\"type\":\"lognormal\",\"min_seq_len\":256,\"max_seq_len\":$SEQ_LEN,\"mean_seq_len\":16384,\"lognormal_sigma\":1.1} "
    # EXTRA_ARGS+=" --mock-data --sft-mock-dataset-config-json {\"mode\":\"distribution\",\"type\":\"linear\",\"min_seq_len\":1024,\"max_seq_len\":32768} "
    EXTRA_ARGS+=" --mock-data --sft-mock-dataset-config-json {\"mode\":\"file\",\"path\":\"/m2v_model/wuguohao03/dataset/github/github_subset_2.csv\",\"min_seq_len\":$MIN_SEQ_LEN,\"max_seq_len\":$SEQ_LEN} "
    # EXTRA_ARGS+=" --mock-data --sft-mock-dataset-config-json {\"mode\":\"file\",\"path\":\"/m2v_model/wuguohao03/dataset/commoncrawl/commoncrawl_subset_2.csv\",\"min_seq_len\":$MIN_SEQ_LEN,\"max_seq_len\":$SEQ_LEN} "
    # EXTRA_ARGS+=" --mock-data --sft-mock-dataset-config-json {\"mode\":\"file\",\"path\":\"/m2v_model/wuguohao03/dataset/wikipedia/wikipedia_subset_2.csv\",\"min_seq_len\":$MIN_SEQ_LEN,\"max_seq_len\":$SEQ_LEN} "
    # EXTRA_ARGS+=" --mock-data --sft-mock-dataset-config-json {\"mode\":\"indexed_file\",\"path\":\"${DATA_TRAIN}\",\"type\":\"lognormal\",\"min_seq_len\":1024,\"max_seq_len\":32768,\"mean_seq_len\":8192,\"lognormal_sigma\":1.1} "
    else
    # EXTRA_ARGS+=" --mock-data --sft-mock-dataset-config-json '{\"mode\":\"distribution\",\"type\":\"lognormal\",\"min_seq_len\":256,\"max_seq_len\":$SEQ_LEN,\"mean_seq_len\":16384,\"lognormal_sigma\":1.1}' "
    # EXTRA_ARGS+=" --mock-data --sft-mock-dataset-config-json {\"mode\":\"distribution\",\"type\":\"linear\",\"min_seq_len\":1024,\"max_seq_len\":32768} "
    EXTRA_ARGS+=" --mock-data --sft-mock-dataset-config-json {\"mode\":\"file\",\"path\":\"/m2v_model/wuguohao03/dataset/github/github_subset_2.csv\",\"min_seq_len\":$MIN_SEQ_LEN,\"max_seq_len\":$SEQ_LEN} "
    # EXTRA_ARGS+=" --mock-data --sft-mock-dataset-config-json {\"mode\":\"file\",\"path\":\"/m2v_model/wuguohao03/dataset/commoncrawl/commoncrawl_subset_2.csv\",\"min_seq_len\":$MIN_SEQ_LEN,\"max_seq_len\":$SEQ_LEN} "
    # EXTRA_ARGS+=" --mock-data --sft-mock-dataset-config-json {\"mode\":\"file\",\"path\":\"/m2v_model/wuguohao03/dataset/wikipedia/wikipedia_subset_2.csv\",\"min_seq_len\":$MIN_SEQ_LEN,\"max_seq_len\":$SEQ_LEN} "
    # EXTRA_ARGS+=" --mock-data --sft-mock-dataset-config-json {\"mode\":\"indexed_file\",\"path\":\"${DATA_TRAIN}\",\"type\":\"lognormal\",\"min_seq_len\":1024,\"max_seq_len\":32768,\"mean_seq_len\":8192,\"lognormal_sigma\":1.1} "
    fi
else
    EXTRA_ARGS+=" --data-path ${DATA_TRAIN} --tokenizer-model ${TOKENIZER} "
fi

if [[ $USE_FSDP -eq 1 ]]; then
    #  --ckpt-format fsdp_dtensor 
    EXTRA_ARGS+="--ckpt-format fsdp_dtensor --use-megatron-fsdp --data-parallel-sharding-strategy optim_grads_params --no-gradient-accumulation-fusion --use-distributed-optimizer "
    unset CUDA_DEVICE_MAX_CONNECTIONS
else
    export CUDA_DEVICE_MAX_CONNECTIONS=1
fi


    # --profile-ranks $PROFILE_RANKS \

    # --use-gpu-timer \
    # --gpu-timer-interval 1 \
    # 
    # --hybrid-context-parallel-scheduler only_packing_no_scheduling \
    # --recompute-activations \
    # --disable-gloo-process-groups \
    # --add-qkv-bias \
    # --disable-gloo-process-groups \
    # --hybrid-context-parallel \

OPTIONS=" \
    `if [ $PROFILE_MEMORY == 1 ]; then echo --profile-memory; fi` \
    `if [ $PROFILE_MEMORY == 1 ]; then echo --profile-memory-path $PROFILE_MEMORY_PATH; fi` \
    --log-throughput \
    --log-energy \
    --no-check-for-nan-in-loss-and-grad \
    --recompute-granularity full \
    --recompute-method uniform \
    --recompute-num-layers 1 \
    --timing-log-level 1 \
    --timing-log-option minmax \
    --hybrid-context-parallel \
    --min-hybrid-context-parallel-size $MIN_CP \
    --max-hybrid-context-parallel-size $MAX_CP \
    --sft-sequence-packing \
    --max-seqlen-per-dp-cp-rank $MAX_SEQLEN_PER_DP_CP_RANK \
    --sft \
    --vocab-size $SEQ_LEN \
    --tokenizer-type NullTokenizer \
    --legacy-tokenizer \
    --use-distributed-optimizer \
    --disable-bias-linear \
    --sft-tokenizer-prompt-format nemotron-h-aligned \
    --transformer-impl transformer_engine \
    --normalization RMSNorm \
    --norm-epsilon 1e-06 \
    --attention-dropout ${AD} \
    --hidden-dropout ${HD} \
    --untie-embeddings-and-output-weights \
    --position-embedding-type rope \
    --rotary-percent 1.0 \
    --rotary-base 1000000 \
    --swiglu \
    --tensor-model-parallel-size ${TP}  \
    --pipeline-model-parallel-size ${PP} \
    ${PP_l:+--num-layers-per-virtual-pipeline-stage $PP_l} \
    --rerun-mode disabled \
    --num-layers $NUM_LAYERS \
    --hidden-size $HIDDEN_SIZE \
    --ffn-hidden-size $FFN_HIDDEN_SIZE \
    --num-attention-heads $NUM_HEAD \
    --num-workers ${NW} \
    --exit-duration-in-mins 230 \
    --seq-length ${SEQ_LEN} \
    --max-position-embeddings ${SEQ_LEN} \
    --train-iters $TRAIN_ITERS \
    --lr-warmup-samples 0 \
    --micro-batch-size ${MBZ} \
    --global-batch-size ${BZ} \
    --lr 2e-5 \
    --min-lr 0.0 \
    --lr-decay-style cosine \
    --log-interval ${LI} \
    --eval-iters 0 \
    --eval-interval 999999 \
    --save-interval 1000 \
    --data-cache-path ${DATACACHE_DIR} \
    --use-mcore-models \
    --no-create-attention-mask-in-dataloader \
    --no-mmap-bin-files \
    --split 100,0,0 \
    --clip-grad 1.0 \
    --weight-decay 0.05 \
    --adam-beta1 0.9 \
    --adam-beta2 0.999 \
    --init-method-std 0.014 \
    --bf16 \
    --tensorboard-dir ${TENSORBOARD_DIR} \
    ${EXTRA_ARGS} \
    --distributed-timeout-minutes 60 \
    --calculate-per-token-loss \
    --attention-backend flash \
    --use-dist-ckpt \
"

# PROFILE_WRAPPER
if [ $PROFILE == 1 ]; then PROFILE_WRAPPER="$SCRIPT_DIR/nsys_profile_rank.sh";
else PROFILE_WRAPPER=; fi

# Interactive or batch mode
# if [[ $BATCH -eq 0 ]]; then
#     DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`
#     # if [[ $PROFILE -eq 1 ]]; then
#     #     nsys profile -w true -t cublas,cuda,nvtx,osrt -s cpu -c cudaProfilerApi -o gpt_sft_hetero_cp_iter7_8_flash_global_64 torchrun --nproc_per_node ${NUM_GPU} pretrain_gpt.py ${OPTIONS}
#     # else
#     #     torchrun --nproc_per_node ${NUM_GPU} /home/tailaim/work_data/megatron-lm/pretrain_gpt.py ${OPTIONS}
#     # fi
#     echo "MASTER_ADDR = ${MASTER_ADDR}, NP = ${NP}, NODE_RANK = ${NODE_RANK}, NUM_GPU = ${NUM_GPU} "
#     $PROFILE_WRAPPER torchrun --master_addr ${MASTER_ADDR} --master_port=12345 --nnodes ${NP} --node_rank ${NODE_RANK} --nproc_per_node ${NUM_GPU} /m2v_model/wuguohao03/nv_teamwork/Megatron-LM/pretrain_gpt.py ${OPTIONS} | tee ${LOGS_DIR}/$DATETIME.log
# else
#     if [[ $PROFILE -eq 1 ]]; then
#         run_cmd="cd ${SOURCE}; nsys profile -w true -t cublas,cuda,nvtx,osrt -s cpu -c cudaProfilerApi --capture-range-end stop -o without_hetero_cp_global_%q{SLURM_PROCID} python -u pretrain_gpt.py ${OPTIONS}"
#     else
#         run_cmd="cd ${SOURCE}; python -u pretrain_gpt.py ${OPTIONS}"
#     fi

#     DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`
#     echo "run_cmd: ${run_cmd}"
#     srun -l --verbose \
#     --container-image /lustre/fsw/portfolios/coreai/users/tailaim/work_data/megatron-moe-scripts/mcore-moe-pytorch25.06.sqsh \
#     --container-mounts "/lustre" \
#     --no-container-mount-home \
#     --output=${LOGS_DIR}/%x_%j_$DATETIME.log \
#     sh -c "${run_cmd}"

#     set +x
# fi

exec &> >(tee "${LOGS_DIR}/$DATETIME.log")
# echo "HOSTFILE = ${HOSTFILE} MASTER_ADDR = ${MASTER_ADDR}, NP = ${NP}, NUM_GPU = ${NUM_GPU} "

cat $HOSTFILE

set -x

# mpirun --hostfile hostfile -np 24 cat $HOSTFILE

        # -x NVTE_DEBUG=1 \
        # -x NVTE_DEBUG_LEVEL=2 \
        # -x NCCL_ALGO=^NVLS,NVLSTree \
        # -x CUDA_DEVICE_MAX_CONNECTIONS=1 \


mpirun --allow-run-as-root \
        ${HOSTFILE:+--hostfile "$HOSTFILE"} \
        --np $NP \
        --bind-to none --map-by slot \
        --mca plm_rsh_args "$PLM_RSH_ARGS" \
        --mca btl self,tcp \
        --mca pml ob1 \
        -mca plm_rsh_num_concurrent 600 \
        -mca routed_radix 600 \
        -mca btl_tcp_if_include bond0,eth01 \
        -mca oob_tcp_if_include bond0,eth01 \
        -mca btl_openib_allow_ib false \
        -mca opal_set_max_sys_limits 1 \
        -x HOROVOD_MPI_THREADS_DISABLE=1 \
        -x MPI_THREAD_SINGLE=1 \
        -x NCCL_IB_DISABLE=0 \
        -x NCCL_IB_GID_INDEX=3 \
        -x NCCL_IB_HCA=mlx5 \
        -x NCCL_IB_QPS_PER_CONNECTION=16 \
        -x NCCL_IB_TIMEOUT=20 \
        -x NCCL_ALGO=^NVLS,NVLSTree \
        -x NCCL_PROTO=^LL128 \
        -x KML_ID \
        -x TASK_ID \
        -x DATETIME \
        -x CREATOR \
        -x TASK_RECORD_URL \
        -x HOSTNAME \
        -x TRAIN_MODE=True \
        -x PATH \
        -x PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True\
        ${LD_LIBRARY_PATH:+-x LD_LIBRARY_PATH} \
        -x PYTHONPATH="$/m2v_model/wuguohao03/nv_teamwork/Megatron-LM":"/m2v_model/wangchenyu05/hot_switch/TransformerEngine":$PYTHONPATH \
        -x NCCL_DEBUG=WARN \
        -x http_proxy=http://oversea-squid2.ko.txyun:11080 \
        -x https_proxy=http://oversea-squid2.ko.txyun:11080 \
        -x no_proxy=localhost,127.0.0.1,localaddress,localdomain.com,internal,corp.kuaishou.com,test.gifshow.com,staging.kuaishou.com \
    $PROFILE_WRAPPER \
    with_nccl_local_env \
    python -u /m2v_model/wuguohao03/nv_teamwork/Megatron-LM/pretrain_gpt.py \
        ${OPTIONS} \
        --distributed-backend nccl \
        --master-addr ${MASTER_ADDR}:${MASTER_PORT}


exit 1
