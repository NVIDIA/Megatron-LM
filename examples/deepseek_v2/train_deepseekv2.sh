#!/bin/bash
###############################################################################
# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
#################################################################################
set -e

TOKENIZER_MODEL="deepseek-ai/DeepSeek-V2-Lite"

CURRENT_DIR="$( cd "$( dirname "$0" )" && pwd )"
MEGATRON_PATH=$( dirname $( dirname ${CURRENT_DIR}))
export PYTHONPATH=${MEGATRON_PATH}:$PYTHONPATH
export CUDA_DEVICE_MAX_CONNECTIONS=1
echo $CURRENT_DIR


# network envs
export GPU_MAX_HW_QUEUES=${GPU_MAX_HW_QUEUES:-2}
export TORCH_NCCL_HIGH_PRIORITY=${TORCH_NCCL_HIGH_PRIORITY:-1}
export NCCL_CHECKS_DISABLE=${NCCL_CHECKS_DISABLE:-1}
NCCL_IB_HCA_LIST=$(rdma link -j | python3 -c "import sys, json; links=json.load(sys.stdin);names=[links[i]['ifname'] for i in range(8)]; print(*names,sep=',')")
export NCCL_IB_HCA=${NCCL_IB_HCA:-$NCCL_IB_HCA_LIST}
export NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-3}
export NCCL_CROSS_NIC=${NCCL_CROSS_NIC:-0}
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1} # Reducing to 1 ensures no PCIE traffic (even on single node)
export NCCL_PROTO=${NCCL_PROTO:-Simple}
export RCCL_MSCCL_ENABLE=${RCCL_MSCCL_ENABLE:-0}


ENV=dsw

# Here are some configs controled by env
if [ -z ${MP_DATASET_TYPE} ];then
    MP_DATASET_TYPE="idxmap"
fi

if [ -z ${MP_AC_LAYERS} ];then
    MP_AC_LAYERS=1
fi

EXPERIMENT_DIR="experiment"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)


MODEL_NAME=DeepSeek-V2-Lite

DATA_DIR=${DATA_DIR:-"/root/deepseek-datasets"}
MODEL=deepseek-ai/${MODEL_NAME}

MODEL_SIZE=16B
BATCH_SIZE="${MBS:-4}"
GLOBAL_BATCH_SIZE="${GBS:-256}"
LR=1e-5
MIN_LR=1e-6
SEQ_LEN="${SEQ_LEN:-2048}"
PAD_LEN="${PAD_LEN:-2048}"
PR="${PR:-bf16}" #fp8 or bf16
TP=1
PP=1  
CP=1
EP=8  
SP=true
DO=true
FL=true
SFT=false   
AC="${AC:-sel}" #full, sel, none
OPTIMIZER_OFFLOAD=false
SAVE_INTERVAL=5000
TRAIN_ITERS="${TRAIN_ITERS:-20}"
LR_WARMUP_ITERS=2
LR_DECAY_ITERS=$(( ${TRAIN_ITERS} - ${LR_WARMUP_ITERS}))
OUTPUT_BASEPATH=${EXPERIMENT_DIR}/deepseek-ckpts/test_ft
GEMM_TUNING="${GEMM_TUNING:-0}"

TRAIN_LOG=${EXPERIMENT_DIR}/MI300X-$MODEL_NAME-${PR}-seq${SEQ_LEN}-tp${TP}pp${PP}ep${EP}-mbs${MBS}gbs${GBS}-ac_${AC}-do_${DO}-fa_${FL}-sp_${SP}-${TIMESTAMP}.log

MOCK_DATA="${MOCK_DATA:-1}"

# For multi-node runs DATA_CACHE_PATH should point to a common path accessible by all the nodes (for eg, an NFS directory)
DATA_CACHE_PATH=${DATA_CACHE_PATH:-"/root/cache"}

if [ "$MOCK_DATA" -eq 1 ]; then
    echo Using mock data.
    data_args="--mock-data --data-cache-path ${DATA_CACHE_PATH}"
else
    echo Using data from $DATA_DIR

    data_args="--train-data-path ${DATA_DIR}/mmap_deepseekv2_datasets_text_document \
	        --valid-data-path ${DATA_DIR}/mmap_deepseekv2_datasets_text_document \
	        --test-data-path ${DATA_DIR}/mmap_deepseekv2_datasets_text_document
	      "
fi


# gemm tuning, https://github.com/ROCm/TransformerEngine
if [ "$GEMM_TUNING" -eq 1 ]; then
   export TE_HIPBLASLT_TUNING_RUN_COUNT=10
   export TE_HIPBLASLT_TUNING_ALGO_COUNT=50
else
   unset TE_HIPBLASLT_TUNING_RUN_COUNT
   unset TE_HIPBLASLT_TUNING_ALGO_COUNT
fi


if [ $ENV = dsw ]; then
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
MASTER_ADDR=localhost
MASTER_PORT=$(shuf -n 1 -i 10000-65535)
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=`python3 -c "import torch; print(torch.cuda.device_count())"`
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
elif [ $ENV = dlc ]; then

NNODES=${WORLD_SIZE}
NODE_RANK=${RANK}
GPUS_PER_NODE=${KUBERNETES_CONTAINER_RESOURCE_GPU}

fi


if [ "${NNODES:-1}" -gt 1 ]; then
    export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-ens51np0}"
    export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-ens51np0}"
    echo "NCCL and GLOO socket interfaces set."
else
    echo "Single node setup, skipping NCCL and GLOO socket interface settings."
fi

if [ -z ${MP_VP} ]; then
    vp_options=""
else
    vp_options=" \
        --num-layers-per-virtual-pipeline-stage ${MP_VP}"
fi

if [ -z ${MP_SFT_PACKING} ]; then
    MP_SFT_PACKING=false
fi

if [ $MODEL_SIZE = 236B ]; then

HIDDEN_SIZE=5120
NUM_ATTN_HEADS=128
NUM_LAYERS=60
INTERMEDIATE_SIZE=12288
MOE_INTERMEDIATE_SIZE=1536
MAX_POSITION_EMBEDDINGS=${SEQ_LEN}
EXTRA_VOCAB_SIZE=2400
Q_LORA_RANK=1536
KV_LORA_RANK=512
QK_NOPE_HEAD_DIM=128
QK_ROPE_HEAD_DIM=64
V_HEAD_DIM=128
ROPE_THETA=10000
SCALE_FACTOR=40
NUM_EXPERTS=160
ROUTER_TOPK=6
NUM_SHARED_EXPERTS=2
MOE_LAYER_FREQ=1

moe_options=" \
    --multi-latent-attention \
    --qk-layernorm \
    --moe-ffn-hidden-size ${MOE_INTERMEDIATE_SIZE} \
    --enable-shared-expert \
    --moe-layer-freq ${MOE_LAYER_FREQ} \
    --num-shared-experts ${NUM_SHARED_EXPERTS} \
    --moe-router-topk ${ROUTER_TOPK} \
    --num-experts ${NUM_EXPERTS} \
    --moe-aux-loss-coeff 1e-2 \
    --expert-model-parallel-size ${EP} \
    --q-lora-rank ${Q_LORA_RANK} \
    --kv-lora-rank ${KV_LORA_RANK} \
    --qk-nope-head-dim ${QK_NOPE_HEAD_DIM} \
    --qk-rope-head-dim ${QK_ROPE_HEAD_DIM} \
    --v-head-dim ${V_HEAD_DIM} \
    --moe-router-load-balancing-type aux_loss"


elif [ $MODEL_SIZE = 16B ]; then

HIDDEN_SIZE=2048
NUM_ATTN_HEADS=16
NUM_LAYERS=27
INTERMEDIATE_SIZE=10944
MOE_INTERMEDIATE_SIZE=1408
MAX_POSITION_EMBEDDINGS=${SEQ_LEN}
EXTRA_VOCAB_SIZE=2400
KV_LORA_RANK=512
QK_NOPE_HEAD_DIM=128
QK_ROPE_HEAD_DIM=64
V_HEAD_DIM=128
ROPE_THETA=10000
SCALE_FACTOR=40
NUM_EXPERTS=64
ROUTER_TOPK=6
NUM_SHARED_EXPERTS=2
MOE_LAYER_FREQ=1

moe_options=" \
    --multi-latent-attention \
    --qk-layernorm \
    --moe-ffn-hidden-size ${MOE_INTERMEDIATE_SIZE} \
    --enable-shared-expert \
    --moe-layer-freq ${MOE_LAYER_FREQ} \
    --num-shared-experts ${NUM_SHARED_EXPERTS} \
    --moe-router-topk ${ROUTER_TOPK} \
    --num-experts ${NUM_EXPERTS} \
    --moe-aux-loss-coeff 1e-2 \
    --expert-model-parallel-size ${EP} \
    --kv-lora-rank ${KV_LORA_RANK} \
    --qk-nope-head-dim ${QK_NOPE_HEAD_DIM} \
    --qk-rope-head-dim ${QK_ROPE_HEAD_DIM} \
    --v-head-dim ${V_HEAD_DIM} \
    --moe-router-load-balancing-type aux_loss"

fi

TP_COMM_OVERLAP=$(( ($TP > 1) ? 1 : 0 ))
comm_overlap_option="\
    --overlap-grad-reduce \
    --overlap-param-gather"

if [ $TP_COMM_OVERLAP -eq 1 ]; then
    comm_overlap_option="\
        --tp-comm-overlap \
        --overlap-grad-reduce \
        --overlap-param-gather"
fi

if [ $AC = full ]; then
    activation_checkpoint_options=" \
		    --recompute-method uniform \
		    --recompute-granularity full \
            --recompute-num-layers 27"
elif [ $AC = sel ]; then
    activation_checkpoint_options=" \
        --recompute-activations"
elif [ $AC = none ]; then
    activation_checkpoint_options=" \
    "
fi
echo "PR: $PR"
if [ $PR = fp16 ]; then
    pr_options=" \
		    --fp16 \
            --apply-query-key-layer-scaling"
    export NVTE_APPLY_QK_LAYER_SCALING=1
elif [ $PR = bf16 ]; then
    pr_options=" \
        --bf16"
elif [ $PR = fp8 ]; then
    pr_options=" \
        --bf16
        --fp8-format hybrid \
        --fp8-amax-compute-algo max \
        --fp8-amax-history-len 1024 \
        --transformer-impl transformer_engine"
fi

if [ $OPTIMIZER_OFFLOAD != false ] && [ $DO = false ]; then
    echo "Offload optimizer is valid only if \$DO=true"
    DO=true
fi

if [ $DO = true ]; then
    do_options=" \
		    --use-distributed-optimizer"

elif [ $DO = false ]; then
    do_options=" \
                    "
fi

if [ $SP = true ] && [ $TP -gt 1 ]; then
    sp_options=" \
		    --sequence-parallel"

elif [ $SP = false ]; then
    sp_options=" \
                    "
fi

if [ $FL = true ]; then
    flash_options=" \
		    --use-flash-attn"

elif [ $FL = false ]; then
    flash_options=" \
                    "
fi


if [ $OPTIMIZER_OFFLOAD = 'static' ]; then
    offload_option=" \
        --optimizer hybridadam \
        --optimizer-offload-policy static \
        --optimizer-offload-fraction 1.0"
elif [ $OPTIMIZER_OFFLOAD = 'auto' ]; then
    offload_option=" \
        --optimizer hybridadam \
        --optimizer-offload-policy auto"
else
    offload_option=""
fi

sft_option="--train-mode pretrain"

NAME="${ENV}-finetune-mcore-deepseek-${MODEL_SIZE}-lr-${LR}-bs-${BATCH_SIZE}-seqlen-${SEQ_LEN}-pr-${PR}-tp-${TP}-pp-${PP}-ac-${AC}-do-${DO}-sp-${SP}"
mkdir -p "${OUTPUT_BASEPATH}/tensorboard/"
mkdir -p "${OUTPUT_BASEPATH}/checkpoint/"
mkdir -p "${OUTPUT_BASEPATH}/log/"
current_time=$(date "+%Y.%m.%d-%H.%M.%S")
TENSORBOARD_DIR="${OUTPUT_BASEPATH}/tensorboard/${NAME}_${current_time}"
mkdir -p ${TENSORBOARD_DIR}

SAVE="${SAVE:-0}"

if [ "$SAVE" -eq 1 ]; then
    SAVED_PRETRAIN_CHECKPOINT_PATH="${OUTPUT_BASEPATH}/checkpoint/${NAME}"
    save_args="--save ${SAVED_PRETRAIN_CHECKPOINT_PATH}"
else
    save_args=""
fi

SAVED_PRETRAIN_CHECKPOINT_PATH="${OUTPUT_BASEPATH}/checkpoint/${NAME}"

megatron_options="  \
	--log-throughput \
	--no-gradient-accumulation-fusion \
	--no-async-tensor-model-parallel-allreduce \
        --lr ${LR} \
        --min-lr ${MIN_LR} \
        --lr-decay-style cosine \
        --adam-beta1 0.9 \
        --adam-beta2 0.95 \
        --weight-decay 0.1 \
        --clip-grad 1.0 \
        --init-method-std 0.008 \
        --attention-dropout 0.0 \
        --hidden-dropout 0.0 \
        --dataloader-type cyclic \
        --lr-decay-iters ${LR_DECAY_ITERS} \
        --lr-warmup-iters ${LR_WARMUP_ITERS} \
        --train-iters ${TRAIN_ITERS} \
        --micro-batch-size ${BATCH_SIZE} \
        --global-batch-size ${GLOBAL_BATCH_SIZE} \
        --num-layers ${NUM_LAYERS} \
        --hidden-size ${HIDDEN_SIZE} \
        --num-attention-heads ${NUM_ATTN_HEADS} \
        --ffn-hidden-size ${INTERMEDIATE_SIZE} \
        --seq-length ${SEQ_LEN} \
        --max-position-embeddings ${MAX_POSITION_EMBEDDINGS} \
        --max-padding-length ${PAD_LEN} \
        --log-interval 1 \
        --eval-interval 10000 \
        --eval-iters -1 \
        --exit-interval 50 \
        --save-interval ${SAVE_INTERVAL} \
        --tensorboard-queue-size 1 \
        --tensorboard-dir ${TENSORBOARD_DIR} \
        --log-timers-to-tensorboard \
        --log-batch-size-to-tensorboard \
        --log-validation-ppl-to-tensorboard \
        --tensor-model-parallel-size ${TP} \
        --pipeline-model-parallel-size ${PP} \
        --no-load-optim \
        --no-load-rng \
        --num-workers 8 \
        --extra-vocab-size ${EXTRA_VOCAB_SIZE} \
        --tokenizer-type DeepSeekV2Tokenizer \
        --tokenizer-model ${TOKENIZER_MODEL}\
        --dataset LLama-Pretrain-Idxmap \
        --swiglu \
        --normalization RMSNorm \
        --norm-epsilon 1e-06 \
        --use-rotary-position-embeddings \
        --no-bias-swiglu-fusion \
        --no-rope-fusion \
        --position-embedding-type rope \
        --untie-embeddings-and-output-weights \
        --disable-bias-linear \
        --use-mcore-models \
        --moe-grouped-gemm \
        --moe-use-legacy-grouped-gemm \
        --ckpt-format torch \
        --rotary-base ${ROPE_THETA} \
        --rotary-scaling-factor ${SCALE_FACTOR} \
        --transformer-impl transformer_engine \
        --eod-mask-loss
        "

#Replace with --load ${PRETRAIN_CHECKPOINT_PATH} to load from checkpoint

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

run_cmd="torchrun $DISTRIBUTED_ARGS examples/deepseek_v2/pretrain_deepseek.py 
 ${megatron_options} ${data_args} ${pr_options} ${activation_checkpoint_options} \
 ${do_options} ${sp_options} ${moe_options} ${offload_option} ${sft_option} ${vp_options} ${flash_options} ${save_args}"

run_cmd="$run_cmd | tee $TRAIN_LOG"
echo ${run_cmd}
eval ${run_cmd}
set +x

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
grep -Eo 'elapsed time per iteration [^|]*' $TRAIN_LOG | sed -E 's/.*elapsed time per iteration \(ms\): ([0-9\.]+).*/\1/' > tmp.txt
echo "elapsed time per iteration: $(python mean_log_value.py tmp.txt)" |& tee -a $TRAIN_LOG

TIME_PER_ITER=$(python3 mean_log_value.py tmp.txt 2>/dev/null | awk '{printf "%.6f", $0}')
TGS=$(awk -v bs="$GLOBAL_BATCH_SIZE" -v sl="$SEQ_LEN" -v tpi="$TIME_PER_ITER" -v ws="$WORLD_SIZE" 'BEGIN {printf "%.6f", bs * sl * 1000/ (tpi * ws)}')
echo "tokens/GPU/s: $TGS" |& tee -a $TRAIN_LOG
rm tmp.txt

# Extract memory usage
grep -Eo 'mem usages: [^|]*' "$TRAIN_LOG" | sed -E 's/.*mem usages: ([0-9\.]+).*/\1/' > tmp.txt
MEMUSAGE=$(python3 mean_log_value.py tmp.txt)
echo "mem usages: $MEMUSAGE" |& tee -a "$TRAIN_LOG"
rm tmp.txt
