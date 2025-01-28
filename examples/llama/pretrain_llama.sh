#!/bin/bash

# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company.

set -ex

# Distributed training variables
LAUNCHER_TYPE=${HL_LAUNCHER_TYPE:-mpirun}
DATA_DIR=${HL_DATA_DIR_ROOT:-/data/datasets/red_pajama}
DATA_CACHE_DIR=${HL_DATA_CACHE_DIR:-}
DATA_FILE_PREFIX=${HL_DATA_FILE_PREFIX:-redpajama}
TOKENIZER_MODEL=${HL_TOKENIZER_MODEL:-}
TRANSFORMER_IMPL=${HL_TRANSFORMER_IMPL:-transformer_engine}
# Parallelism variables
NUM_NODES=${HL_NUM_NODES:-1}
DP=${HL_DP:-2}
TP=${HL_TP:-2}
PP=${HL_PP:-2}
MICRO_BATCH_SIZE=${HL_MICRO_BATCH:-1} # batch_size
EXIT_INTERVAL=${HL_EXIT_INTERVAL:-0}
OUTPUT_DIR=${HL_RESULTS_DIR:-}
OUTPUT_DIR_PREFIX=${HL_RESULTS_DIR_PREFIX:-.}
CHECKPOINT_SAVE=${HL_SAVE:-1}
SAVE_INTERVAL=${HL_SAVE_INTERVAL:-2000}
DIST_CKPT_FORMAT=${HL_DIST_CKPT_FORMAT:-torch_dist}
USE_DISTRIBUTED_OPTIMIZER=${HL_USE_DISTRIBUTED_OPTIMIZER:-1}
USE_DIST_CKPT=${HL_USE_DIST_CKPT:-0}
LOAD_DIR=${HL_LOAD_DIR:-}
CHECKPOINTS_DIR=${HL_CHECKPOINTS_DIR:-}
VERIFY_CKPT=${HL_VERIFY_CKPT:-1}
TENSORBOARD_DIR=${HL_TENSORBOARD_DIR:-}
KILL_SWITCH_FILE=${HL_KILL_SWITCH_FILE:-}
HOSTSFILE=${HL_HOSTSFILE:-}
CKP_ACT=${HL_CKP_ACT:-0}
RECOMPUTE_NUM_LAYERS=${HL_RECOMPUTE_NUM_LAYERS:-1}
LOG_INTERVAL=${HL_LOG_INTERVAL:-10}
LLAMA_VER=${HL_LLAMA_VER:-3.1} # 1 for LLaMA, 2 for LLaMA 2 and 3.1 for LLaMA 3.1
LLAMA_MODEL_SIZE=${HL_LLAMA_MODEL_SIZE:-8}
DEVICES_PER_NODE=${HL_DEVICES_PER_NODE:-8}
SEQ_PARALLEL=${HL_SEQ_PARALLEL:-1}
OPTIMIZER=${HL_OPTIMIZER:-fusedadamw}
DROPOUT=${HL_DROPOUT:-0.0}
EVAL_ITERS=${HL_EVAL_ITERS:-100}
EVAL_INTERVAL=${HL_EVAL_INTERVAL:-1000}
USE_FUSED_SDPA=${HL_USE_FUSED_SDPA:-1}
USE_FUSED_SDPA_WITH_RECOMPUTE=${HL_USE_FUSED_SDPA_WITH_RECOMPUTE:-0}
USE_FAST_SOFTMAX=${HL_USE_FAST_SOFTMAX:-1}
USE_FUSED_RMSNORM=${HL_USE_FUSED_RMSNORM:-1}
PROFILE_TYPE=${HL_PROFILE_TYPE:-} # provide either of pt, pt-full, hltv
PROFILE_STEP_START=${HL_PROFILE_STEP_START:-3}
PROFILE_STEP_END=${HL_PROFILE_STEP_END:-4}
PROFILE_RANKS=${HL_PROFILE_RANKS:-"0"} # "0 1 4 7"
REDIRECT_LOGS=${HL_REDIRECT_LOGS:-0}
DETERMINISTIC_MODE=${HL_DETERMINISTIC_MODE:-1}
FP8=${HL_FP8:-0}
FP8_FORMAT=${HL_FP8_FORMAT:-hybrid} # hybrid or e5m2
FP8_MARGIN=${HL_FP8_MARGIN:-0}
FP8_AMAX_COMPUTE_ALGO=${HL_FP8_AMAX_COMPUTE_ALGO:-max} # max or most_recent
USE_TORCH_COMPILE=${HL_USE_TORCH_COMPILE:-0}
USE_LAZY_MODE=${HL_USE_LAZY_MODE:-1}
SKIP_TRAIN=${HL_SKIP_TRAIN:-0}
NUM_WORKERS=${HL_NUM_WORKERS:-2}
FP8_COVERAGE=${HL_FP8_COVERAGE:-"mlp_row_parallel=False attention=False"}
OVERLAP_GRAD_REDUCE=${HL_OVERLAP_GRAD_REDUCE:-0}
LOG_ZEROS_IN_GRAD=${HL_LOG_ZEROS_IN_GRAD:-0}

if [[ -z "${MEGATRON_LM_ROOT}" ]]; then
    MEGATRON_LM_ROOT=$(realpath "$(dirname "$0")"/../../)
fi

if [[ $((NUM_NODES*DEVICES_PER_NODE)) -ne $((DP*TP*PP)) ]]; then
    echo "NUM_NODES*DEVICES_PER_NODE != DP*TP*PP"
    exit 1
fi

if [[ "${TRANSFORMER_IMPL}" = "local" && "${FP8}" -eq 1 ]]; then
    echo "fp8 is not supported with local transformer implementation"
    exit 1
fi

if [[ "$USE_LAZY_MODE" = 1 && "$USE_TORCH_COMPILE" = 1 ]]; then
    echo "Cannot use lazy(HL_USE_LAZY_MODE) and torch.compile(HL_USE_TORCH_COMPILE) modes together"
    exit 1
fi

# Network size variables
if [[ "${LLAMA_VER}" = "1" ]]; then
    TOKENIZER_TYPE=${HL_TOKENIZER_TYPE:-GPTSentencePieceTokenizer}
    GLOBAL_BATCH_SIZE=${HL_GBS:-2048} # microbatches in the pipeline (computed as `GLOBAL_BATCH / (DP * MICRO_BATCH)`) should be divisible by the PP
    MAX_SEQ_LEN=${HL_SEQ_LEN:-2048}
    TRAIN_ITERS=${HL_TRAIN_ITERS:-250000}
    ADAM_EPS=1e-8
    LR_WARMUP_ITERS=2000
    ROTARY_BASE=10000
    if [[ "${LLAMA_MODEL_SIZE}" = "7" ]]; then
        HIDDEN_SIZE=${HL_HIDDEN_SIZE:-4096}
        NUM_HEADS=${HL_NUM_HEADS:-32}
        NUM_QUERY_GROUPS=${HL_NUM_QUERY_GROUPS:-32}
        NUM_LAYERS=${HL_NUM_LAYERS:-32}
        FFN_HIDDEN_SIZE=${HL_FFN_HIDDEN_SIZE:-11008}
        LR=3e-4
        MIN_LR=3e-5
    elif [[ "${LLAMA_MODEL_SIZE}" = "13" ]]; then
        HIDDEN_SIZE=${HL_HIDDEN_SIZE:-5120}
        NUM_HEADS=${HL_NUM_HEADS:-40}
        NUM_QUERY_GROUPS=${HL_NUM_QUERY_GROUPS:-40}
        NUM_LAYERS=${HL_NUM_LAYERS:-40}
        FFN_HIDDEN_SIZE=${HL_FFN_HIDDEN_SIZE:-13824}
        LR=3e-4
        MIN_LR=3e-5
    elif [[ "${LLAMA_MODEL_SIZE}" = "65" ]]; then
        HIDDEN_SIZE=${HL_HIDDEN_SIZE:-8192}
        NUM_HEADS=${HL_NUM_HEADS:-64}
        NUM_QUERY_GROUPS=${HL_NUM_QUERY_GROUPS:-64}
        NUM_LAYERS=${HL_NUM_LAYERS:-80}
        FFN_HIDDEN_SIZE=${HL_FFN_HIDDEN_SIZE:-22016}
        LR=1.5e-4
        MIN_LR=1.5e-5
    else
        echo "invalid LLAMA_MODEL_SIZE: ${LLAMA_MODEL_SIZE}"
        exit 1
    fi
elif [[ "${LLAMA_VER}" = "2" ]]; then
    TOKENIZER_TYPE=${HL_TOKENIZER_TYPE:-GPTSentencePieceTokenizer}
    GLOBAL_BATCH_SIZE=${HL_GBS:-1024} # microbatches in the pipeline (computed as `GLOBAL_BATCH / (DP * MICRO_BATCH)`) should be divisible by the PP
    MAX_SEQ_LEN=${HL_SEQ_LEN:-4096}
    TRAIN_ITERS=${HL_TRAIN_ITERS:-500000}
    ADAM_EPS=1e-8
    LR_WARMUP_ITERS=2000
    ROTARY_BASE=10000
    if [[ "${LLAMA_MODEL_SIZE}" = "7" ]]; then
        HIDDEN_SIZE=${HL_HIDDEN_SIZE:-4096}
        NUM_HEADS=${HL_NUM_HEADS:-32}
        NUM_QUERY_GROUPS=${HL_NUM_QUERY_GROUPS:-32}
        NUM_LAYERS=${HL_NUM_LAYERS:-32}
        FFN_HIDDEN_SIZE=${HL_FFN_HIDDEN_SIZE:-11008}
        LR=3e-4
        MIN_LR=3e-5
    elif [[ "${LLAMA_MODEL_SIZE}" = "13" ]]; then
        HIDDEN_SIZE=${HL_HIDDEN_SIZE:-5120}
        NUM_HEADS=${HL_NUM_HEADS:-40}
        NUM_QUERY_GROUPS=${HL_NUM_QUERY_GROUPS:-40}
        NUM_LAYERS=${HL_NUM_LAYERS:-40}
        FFN_HIDDEN_SIZE=${HL_FFN_HIDDEN_SIZE:-13824}
        LR=3e-4
        MIN_LR=3e-5
    elif [[ "${LLAMA_MODEL_SIZE}" = "34" ]]; then
        HIDDEN_SIZE=${HL_HIDDEN_SIZE:-8192}
        NUM_HEADS=${HL_NUM_HEADS:-64}
        NUM_QUERY_GROUPS=${HL_NUM_QUERY_GROUPS:-8}
        NUM_LAYERS=${HL_NUM_LAYERS:-48}
        FFN_HIDDEN_SIZE=${HL_FFN_HIDDEN_SIZE:-22016}
        LR=1.5e-4
        MIN_LR=1.5e-5
    elif [[ "${LLAMA_MODEL_SIZE}" = "70" ]]; then
        HIDDEN_SIZE=${HL_HIDDEN_SIZE:-8192}
        NUM_HEADS=${HL_NUM_HEADS:-64}
        NUM_QUERY_GROUPS=${HL_NUM_QUERY_GROUPS:-8}
        NUM_LAYERS=${HL_NUM_LAYERS:-80}
        FFN_HIDDEN_SIZE=${HL_FFN_HIDDEN_SIZE:-28672}
        LR=1.5e-4
        MIN_LR=1.5e-5
    else
        echo "invalid LLAMA_MODEL_SIZE: ${LLAMA_MODEL_SIZE}"
        exit 1
    fi
elif [[ "${LLAMA_VER}" = "3.1" ]]; then
    TOKENIZER_TYPE=${HL_TOKENIZER_TYPE:-Llama3Tokenizer}
    GLOBAL_BATCH_SIZE=${HL_GBS:-2048} # microbatches in the pipeline (computed as `GLOBAL_BATCH / (DP * MICRO_BATCH)`) should be divisible by the PP
    MAX_SEQ_LEN=${HL_SEQ_LEN:-8192}
    TRAIN_ITERS=${HL_TRAIN_ITERS:-937500}
    ADAM_EPS=1e-5
    LR_WARMUP_ITERS=8000
    ROTARY_BASE=500000
    if [[ "${LLAMA_MODEL_SIZE}" = "8" ]]; then
        # LLaMA3.1-8B model architecture
        HIDDEN_SIZE=${HL_HIDDEN_SIZE:-4096}
        NUM_HEADS=${HL_NUM_HEADS:-32} # must be divisible by TP
        NUM_QUERY_GROUPS=${HL_NUM_QUERY_GROUPS:-8} # must be divisible by TP
        NUM_LAYERS=${HL_NUM_LAYERS:-32} # must be divisible by PP
        FFN_HIDDEN_SIZE=${HL_FFN_HIDDEN_SIZE:-14336}
        LR=3e-4
        MIN_LR=3e-6
    elif [[ "${LLAMA_MODEL_SIZE}" = "70" ]]; then
        # LLaMA3.1-70B model architecture
        HIDDEN_SIZE=${HL_HIDDEN_SIZE:-8192}
        NUM_HEADS=${HL_NUM_HEADS:-64} # must be divisible by TP
        NUM_QUERY_GROUPS=${HL_NUM_QUERY_GROUPS:-8} # must be divisible by TP
        NUM_LAYERS=${HL_NUM_LAYERS:-80} # must be divisible by PP
        FFN_HIDDEN_SIZE=${HL_FFN_HIDDEN_SIZE:-28672}
        LR=1.5e-4
        MIN_LR=1.5e-6
    elif [[ "${LLAMA_MODEL_SIZE}" = "405" ]]; then
        # LLaMA3.1-405B model architecture
        HIDDEN_SIZE=${HL_HIDDEN_SIZE:-16384}
        NUM_HEADS=${HL_NUM_HEADS:-128} # must be divisible by TP
        NUM_QUERY_GROUPS=${HL_NUM_QUERY_GROUPS:-8} # must be divisible by TP
        NUM_LAYERS=${HL_NUM_LAYERS:-126} # must be divisible by PP
        FFN_HIDDEN_SIZE=${HL_FFN_HIDDEN_SIZE:-53248}
        LR=8e-5
        MIN_LR=8e-7
    else
        echo "incorrect HL_LLAMA_MODEL_SIZE=${LLAMA_MODEL_SIZE} is set"
        exit 1
    fi
else
    echo "invalid LLAMA_VER: ${LLAMA_VER}"
    exit 1
fi

if [[ $(( NUM_LAYERS % PP )) -ne 0 ]]; then
    echo 'HL_NUM_LAYERS must be divisible by PP'
    exit 1
fi

# Paths
SRC_PATH="${MEGATRON_LM_ROOT}/pretrain_gpt.py"
DATA_PATH=${DATA_DIR}/${DATA_FILE_PREFIX}

if [[ -z "${TOKENIZER_MODEL}" ]]; then
    TOKENIZER_MODEL="${DATA_DIR}/tokenizer.model"
fi

NUM_DEVICES=$((DEVICES_PER_NODE*NUM_NODES))

RUNTIME=$(date +"%Y%m%d_%H%M")
# Experiment name
if [[ -z "${EXP_NAME}" ]]; then
    EXP_NAME="default"
fi
# output paths
if [[ -z "${OUTPUT_DIR}" ]]; then
    data_type="bf16"
    if [[ "${FP8}" -eq 1 ]]; then
        data_type="fp8"
    fi
    OUTPUT_DIR=${OUTPUT_DIR_PREFIX}/out/llama${LLAMA_VER}_${LLAMA_MODEL_SIZE}b/${data_type}_${TRANSFORMER_IMPL}_${EXP_NAME}_nl${NUM_LAYERS}_hs${HIDDEN_SIZE}_ffn${FFN_HIDDEN_SIZE}_gb${GLOBAL_BATCH_SIZE}_mb${MICRO_BATCH_SIZE}_sp${SEQ_PARALLEL}_D${DP}_T${TP}_P${PP}_devices${NUM_DEVICES}_${RUNTIME}
fi
if [[ -z "${CHECKPOINTS_DIR}" ]]; then
    CHECKPOINTS_DIR=${OUTPUT_DIR}/checkpoints
fi
if [[ -z "${LOAD_DIR}" ]]; then
    LOAD_DIR=${CHECKPOINTS_DIR}
fi

if [[ -z "${TENSORBOARD_DIR}" ]]; then
    TENSORBOARD_DIR=${OUTPUT_DIR}/tensorboard
fi
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${CHECKPOINTS_DIR}"
mkdir -p "${TENSORBOARD_DIR}"

if [[ "${NUM_NODES}" -ne "1" ]] && [[ -z "${HOSTSFILE}" ]]; then
    HOSTSFILE=${MEGATRON_LM_ROOT}/examples/hostsfile
    if [[ -f "${HOSTSFILE}" ]]; then
        cat /dev/null > "${HOSTSFILE}"
    fi
    etc_mpi_hostfile="/etc/mpi/hostfile"
    if [[ ! -f ${etc_mpi_hostfile} ]]; then
        echo "${etc_mpi_hostfile} not available, set HL_HOSTSFILE"
        exit 1
    fi
    cat ${etc_mpi_hostfile} | xargs -I{} echo {} slots=8 >> "${HOSTSFILE}"
fi

# Setting the environment variables
PT_HPU_GPU_MIGRATION=1
PT_TE_ENFORCE_BF16_AMAX_REDUCTION=${HL_FP8_ENFORCE_BF16_AMAX_REDUCTION:-1}

# Set training command
CMD=""
if [[ "${LAUNCHER_TYPE}" = "mpirun" ]]; then
    CMD="${CMD} mpirun"
    CMD="${CMD} --allow-run-as-root"
    CMD="${CMD} -n ${NUM_DEVICES}"
    CMD="${CMD} --bind-to none"
    CMD="${CMD} -x PT_HPU_GPU_MIGRATION=${PT_HPU_GPU_MIGRATION}"
    CMD="${CMD} -x PT_TE_ENFORCE_BF16_AMAX_REDUCTION=${PT_TE_ENFORCE_BF16_AMAX_REDUCTION}"
    if [[ "${NUM_NODES}" -ne "1" ]]; then
        CMD="${CMD} -hostfile ${HOSTSFILE}"
        CMD="${CMD} -x MASTER_ADDR=$(head -n 1 "${HOSTSFILE}" | sed -n s/[[:space:]]slots.*//p)"
    else
        CMD="${CMD} -x MASTER_ADDR=localhost"
    fi
    CMD="${CMD} -x MASTER_PORT=12345"
elif [[ "${LAUNCHER_TYPE}" = "torchrun" ]]; then
    if [[ "${NUM_NODES}" -ne "1" ]]; then
        echo "NUM_NODES greater than 1 not supported by torchrun"
        exit 1
    fi
    export PT_HPU_GPU_MIGRATION=${PT_HPU_GPU_MIGRATION}
    CMD="${CMD} torchrun"
    CMD="${CMD} --nnodes ${NUM_NODES}"
    CMD="${CMD} --nproc-per-node ${DEVICES_PER_NODE}"
    CMD="${CMD} --no-python"
    CMD="${CMD} --node-rank 0"
    CMD="${CMD} --master-addr localhost"
    CMD="${CMD} --master-port 12345"
else
    echo "Unsupported launcher type = ${LAUNCHER_TYPE}"
    exit 2
fi

if [ "$USE_LAZY_MODE" = "0" ]; then
    CMD="${CMD} -x PT_HPU_LAZY_MODE=0"
fi

CMD="${CMD} \
    python ${SRC_PATH} \
    --transformer-impl ${TRANSFORMER_IMPL} \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --distributed-backend nccl \
    --seq-length ${MAX_SEQ_LEN} \
    --num-layers ${NUM_LAYERS} \
    --hidden-size ${HIDDEN_SIZE} \
    --num-attention-heads ${NUM_HEADS} \
    --group-query-attention \
    --num-query-groups ${NUM_QUERY_GROUPS} \
    --ffn-hidden-size ${FFN_HIDDEN_SIZE} \
    --position-embedding-type rope \
    --rotary-base ${ROTARY_BASE} \
    --max-position-embeddings ${MAX_SEQ_LEN} \
    --normalization RMSNorm \
    --swiglu \
    --untie-embeddings-and-output-weights \
    --attention-dropout ${DROPOUT} \
    --hidden-dropout ${DROPOUT} \
    --weight-decay 1e-1 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --adam-eps ${ADAM_EPS} \
    --lr ${LR} \
    --lr-decay-style cosine \
    --lr-warmup-iters ${LR_WARMUP_ITERS} \
    --min-lr ${MIN_LR} \
    --use-torch-compile=${USE_TORCH_COMPILE} \
    --use-fused-sdpa-with-recompute ${USE_FUSED_SDPA_WITH_RECOMPUTE} \
    --use-fused-sdpa ${USE_FUSED_SDPA} \
    --use-fused-rmsnorm ${USE_FUSED_RMSNORM} \
    --use-fast-softmax ${USE_FAST_SOFTMAX} \
    --micro-batch-size ${MICRO_BATCH_SIZE} \
    --global-batch-size ${GLOBAL_BATCH_SIZE} \
    --train-iters ${TRAIN_ITERS} \
    --log-interval ${LOG_INTERVAL} \
    --log-throughput \
    --disable-bias-linear \
    --optimizer ${OPTIMIZER} \
    --no-gradient-accumulation-fusion \
    --no-masked-softmax-fusion \
    --use-mcore-models \
    --bf16 \
    --exit-interval ${EXIT_INTERVAL} \
    --tensorboard-dir ${TENSORBOARD_DIR} \
    --log-validation-ppl-to-tensorboard \
    --log-batch-size-to-tensorboard \
    --log-timers-to-tensorboard \
    --load ${LOAD_DIR} \
    --eval-interval ${EVAL_INTERVAL} \
    --eval-iters ${EVAL_ITERS} \
    --data-path ${DATA_PATH} \
    --num-workers ${NUM_WORKERS} \
    "

##     --OVERLAP_GRAD_REDUCE
if [[ "${OVERLAP_GRAD_REDUCE}" -eq 1 ]]; then
    CMD="${CMD} --overlap-grad-reduce"
fi

##     --LOG_ZEROS_IN_GRAD
if [[ "${LOG_ZEROS_IN_GRAD}" -eq 1 ]]; then
    CMD="${CMD} -log-num-zeros-in-grad"
fi

if [[ "${SEQ_PARALLEL}" -eq 1 ]]; then
    CMD="${CMD} --sequence-parallel"
fi

if [[ "${USE_FUSED_SDPA}" = "1" || "${USE_FUSED_SDPA_WITH_RECOMPUTE}" = "1" ]]; then
    CMD="${CMD} --no-create-attention-mask-in-dataloader"
fi

if [[ "${SKIP_TRAIN}" -eq 1 ]]; then
    CMD="${CMD} --skip-train"
fi

if [[ "${CKP_ACT}" -eq 1 ]]; then
    CMD="${CMD} --recompute-granularity=full"
    CMD="${CMD} --recompute-method uniform"
    CMD="${CMD} --recompute-num-layers ${RECOMPUTE_NUM_LAYERS}"
elif [[ "${CKP_ACT}" -eq 2 ]]; then
    CMD="${CMD} --recompute-granularity selective"
elif [[ "${CKP_ACT}" -ne 0 ]]; then
    echo "incorrect HL_CKP_ACT=${CKP_ACT} is set"
    exit 1
fi

if [[ "${USE_DISTRIBUTED_OPTIMIZER}" -eq 1 ]]; then
    CMD="${CMD} --use-distributed-optimizer"
fi

if [[ "${DETERMINISTIC_MODE}" -eq 1 ]]; then
    CMD="${CMD} --deterministic-mode"
fi

# fp8 args
if [[ "${TRANSFORMER_IMPL}" = "transformer_engine" && "${FP8}" -eq 1 ]]; then

    FP8_MEASURE_INTERVAL=${HL_FP8_MEASURE_INTERVAL:-$(( GLOBAL_BATCH_SIZE / MICRO_BATCH_SIZE / DP ))}
    FP8_AMAX_HISTORY_LEN=${HL_FP8_AMAX_HISTORY_LEN:-$((( GLOBAL_BATCH_SIZE / MICRO_BATCH_SIZE / DP + 15 ) / 16 ))}
    FP8_AMAX_REDUCE=${HL_FP8_AMAX_REDUCE:-1}

    CMD="${CMD} --fp8-interval ${FP8_MEASURE_INTERVAL}"
    CMD="${CMD} --fp8-margin ${FP8_MARGIN}"
    CMD="${CMD} --fp8-amax-compute-algo ${FP8_AMAX_COMPUTE_ALGO}"
    CMD="${CMD} --fp8-amax-history-len ${FP8_AMAX_HISTORY_LEN}"
    CMD="${CMD} --fp8-format ${FP8_FORMAT}"
    CMD="${CMD} --fp8-coverage ${FP8_COVERAGE}"

    if [[ "${FP8_AMAX_REDUCE}" -eq 1 ]]; then
        CMD="${CMD} --fp8-amax-reduce"
    fi
fi

# handle kill switch file argument
if [[ -n "${KILL_SWITCH_FILE}" ]]; then
    CMD="${CMD} --kill-switch-file ${KILL_SWITCH_FILE}"
fi

if [[ -n "${PROFILE_TYPE}" ]]; then
    CMD="${CMD} --profile-type ${PROFILE_TYPE}"
    CMD="${CMD} --profile-step-start ${PROFILE_STEP_START}"
    CMD="${CMD} --profile-step-end ${PROFILE_STEP_END}"
    CMD="${CMD} --profile-ranks ${PROFILE_RANKS}"
fi

if [[ "${CHECKPOINT_SAVE}" -eq 1 ]]; then
    CMD="${CMD} --save ${CHECKPOINTS_DIR}"
    CMD="${CMD} --save-interval ${SAVE_INTERVAL}"
    CMD="${CMD} --dist-ckpt-format ${DIST_CKPT_FORMAT}"
    if [[ "${USE_DIST_CKPT}" -eq 1 ]]; then
        CMD="${CMD} --use-dist-ckpt"
    fi
    if [[ "${VERIFY_CKPT}" -eq 1 ]]; then
        CMD="${CMD} --verify-checkpoint"
        CMD="${CMD} --verify-checkpoint-model-type LLAMA"
    fi
fi

if [[ "${TOKENIZER_TYPE}" = "GPTSentencePieceTokenizer" || "${TOKENIZER_TYPE}" = "Llama3Tokenizer" ]]; then
    CMD="${CMD} --tokenizer-type ${TOKENIZER_TYPE}"
    CMD="${CMD} --tokenizer-model ${TOKENIZER_MODEL}"
elif [[ "${TOKENIZER_TYPE}" = "GPT2BPETokenizer" ]]; then
    CMD="${CMD} --tokenizer-type GPT2BPETokenizer"
    CMD="${CMD} --vocab-file ${DATA_DIR}/gpt2-vocab.json"
    CMD="${CMD} --merge-file ${DATA_DIR}/gpt2-merges.txt"
else
    echo "incorrect HL_TOKENIZER_TYPE=${TOKENIZER_TYPE} is set"
    exit 1
fi

if [[ -n "${DATA_CACHE_DIR}" ]]; then
    CMD="${CMD} --data-cache-path ${DATA_CACHE_DIR}"
fi

if [[ "${REDIRECT_LOGS}" -eq 1 ]]; then
    ${CMD} 2>&1 | tee "${OUTPUT_DIR}"/log_"${EXP_NAME}"_"${RUNTIME}".txt
else
    ${CMD}
fi
