#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES=0
export NVTE_FUSED_ATTN=1

MODEL=${MODEL:-"DeepSeek-V3"}
# Defaults to repo root; this script lives under examples/moe-modelzoo/ckpt_convert_scripts/DeepSeek-V3/
MEGATRON_PATH=${MEGATRON_PATH:-"$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../../../.." && pwd)"} # Path to Megatron-LM
export PYTHONPATH=${MEGATRON_PATH}:${PYTHONPATH}

if [ ${MODEL} == "DeepSeek-V3" ]; then
    TP=${TP:-1}
    PP=${PP:-16}
    EP=${EP:-64}

    SOURCE_CKPT_PATH=${SOURCE_CKPT_PATH:-"your_own_load_path/HuggingFace/${MODEL}"} # Path to load HuggingFace checkpoint
    TARGET_CKPT_PATH=${TARGET_CKPT_PATH:-"your_own_save_path/${MODEL}-to-mcore-TP${TP}-PP${PP}-EP${EP}"} # Path to save Megatron checkpoint

    NUM_LAYERS=61
    HIDDEN_SIZE=7168
    FFN_HIDDEN_SIZE=18432
    NUM_ATTN_HEADS=128
    KV_CHANNELS=128
    MAX_POSITION_EMBEDDINGS=4096
    Q_RANK=1536
    KV_LORA_RANK=512
    QK_NOPE_HEAD_DIM=128
    QK_ROPE_HEAD_DIM=64
    V_HEAD_DIM=128
    ROPE_THETA=10000
    ROPE_SCALING_FACTOR=40
    MSCALE=1.0
    MSCALE_ALL_DIM=1.0
    NUM_EXPERTS=256
    MOE_LAYER_FREQ=([0]*3+[1]*58)
    MOE_FFN_HIDDEN_SIZE=2048
    ROUTER_TOPK=8
    NUM_SHARED_EXPERTS=1
    SHARED_INTERMEDIATE_SIZE=$((MOE_FFN_HIDDEN_SIZE * NUM_SHARED_EXPERTS))
    ROUTER_TOPK_SCALING_FACTOR=2.5

else
    echo "Error: The valid model name is DeepSeek-V3. Current model name is ${MODEL}."
    exit 1
fi

# Check if the source and target paths exist
if [ ! -d ${SOURCE_CKPT_PATH} ]; then
    echo "Error: SOURCE_CKPT_PATH ($SOURCE_CKPT_PATH) does not exist."
    exit 1
fi

if [ ! -d ${TARGET_CKPT_PATH} ]; then
    echo "Error: TARGET_CKPT_PATH ($TARGET_CKPT_PATH) does not exist."
    exit 1
fi

distributed_options=" \
    --target-tensor-model-parallel-size ${TP} \
    --target-pipeline-model-parallel-size ${PP} \
    --target-decoder-first-pipeline-num-layers 4 \
    --target-decoder-last-pipeline-num-layers 1 \
    --expert-model-parallel-size 1 \
    --target-expert-model-parallel-size ${EP}"

training_options=" \
    --use-mcore-models \
    --micro-batch-size 1 \
    --disable-bias-linear \
    --no-bias-swiglu-fusion \
    --no-rope-fusion \
    --no-async-tensor-model-parallel-allreduce \
    --use-cpu-initialization"

transformer_engine_options=" \
    --transformer-impl transformer_engine"

data_options=" \
    --seq-length 4096 \
    --tokenizer-type  HuggingFaceTokenizer \
    --tokenizer-model  deepseek-ai/DeepSeek-V3"
    
network_size_options=" \
    --num-layers ${NUM_LAYERS} \
    --hidden-size ${HIDDEN_SIZE} \
    --ffn-hidden-size ${FFN_HIDDEN_SIZE} \
    --num-attention-heads ${NUM_ATTN_HEADS} \
    --kv-channels ${KV_CHANNELS} \
    --max-position-embeddings ${MAX_POSITION_EMBEDDINGS} \
    --position-embedding-type rope \
    --rotary-base ${ROPE_THETA} \
    --make-vocab-size-divisible-by 3232 \
    --normalization RMSNorm \
    --norm-epsilon 1e-6 \
    --swiglu \
    --untie-embeddings-and-output-weights \
    --multi-latent-attention \
    --mtp-num-layers 1 \
    --mtp-loss-scaling-factor 0.1"

regularization_options=" \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --clip-grad 1.0 \
    --weight-decay 0.01 \
    --qk-layernorm"

moe_options=" \
    --num-experts ${NUM_EXPERTS} \
    --moe-layer-freq ${MOE_LAYER_FREQ} \
    --moe-ffn-hidden-size ${MOE_FFN_HIDDEN_SIZE} \
    --moe-shared-expert-intermediate-size ${SHARED_INTERMEDIATE_SIZE} \
    --moe-router-load-balancing-type seq_aux_loss \
    --moe-aux-loss-coeff 1e-3 \
    --moe-router-topk ${ROUTER_TOPK} \
    --moe-token-dispatcher-type alltoall \
    --moe-router-pre-softmax \
    --moe-grouped-gemm \
    --moe-router-topk-scaling-factor ${ROUTER_TOPK_SCALING_FACTOR} \
    --moe-router-group-topk 4 \
    --moe-router-num-groups 8 \
    --moe-router-score-function sigmoid \
    --moe-router-enable-expert-bias \
    --moe-router-bias-update-rate 1e-3 \
    --moe-router-dtype fp32"

mla_options=" \
    --q-lora-rank ${Q_RANK} \
    --kv-lora-rank ${KV_LORA_RANK} \
    --qk-head-dim ${QK_NOPE_HEAD_DIM} \
    --qk-pos-emb-head-dim ${QK_ROPE_HEAD_DIM} \
    --v-head-dim ${V_HEAD_DIM} \
    --rotary-scaling-factor ${ROPE_SCALING_FACTOR}"

checkpointing_options=" \
    --load ${SOURCE_CKPT_PATH} \
    --save ${TARGET_CKPT_PATH} \
    --save-interval 1"

mix_precision_options=" \
    --bf16"

# Path to current working directory
WORKSPACE=$(dirname "$(readlink -f "$0")")
python ${WORKSPACE}/deepseek_v3_hf_to_mg.py \
    ${distributed_options} \
    ${training_options} \
    ${transformer_engine_options} \
    ${data_options} \
    ${network_size_options} \
    ${regularization_options} \
    ${moe_options} \
    ${mla_options} \
    ${checkpointing_options} \
    ${mix_precision_options}
