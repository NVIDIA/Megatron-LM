#!/bin/bash

#SBATCH -A coreai_dlalgo_genai
# DFW: batch
# OCI-NRT: batch_block1
# OCI-IAD: batch_block1,batch_block3,batch_block4,backfill_block1,backfill_block2,backfill_block3,backfill_block4
#SBATCH -p batch
#SBATCH -t 00:20:00
#SBATCH --mem=0
#SBATCH --ntasks-per-node=8
#SBATCH --nodes=4
#SBATCH --exclusive
#SBATCH --gpus-per-node=8
#SBATCH --job-name=hetero_cp_global

export NCCL_IB_SL=1
export TOKENIZERS_PARALLELISM="false"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

#export NVTE_DEBUG=1
#export NVTE_DEBUG_LEVEL=2

USER=$SLURM_JOB_USER

# Auto-detect batch or interactive mode.
which srun
BATCH=$((1-$?))

DEBUG=0
USE_TILING=1
USE_CP=1
USE_TE_CE=0
USE_FLASH_ATTN=0
USE_FSDP=0
USE_CUSTOM_FSDP=0
PROFILE=0

# Remember to update model and job name if running in batch mode!!
if [[ $BATCH -eq 0 ]]; then
    DATETIME=`date +'%y-%m-%d-%H-%M-%S'`
    MODEL_NAME="interactive_hybrid_cp"
    DEBUG=1
else
    MODEL_NAME="interactive_hybrid_cp"
fi

WORKSPACE="/lustre/fsw/portfolios/coreai/users/pmannan/workspace"
SOURCE=`pwd`
OUTPUT_BASE="${WORKSPACE}/output"
OUTPUT="${OUTPUT_BASE}/${MODEL_NAME}"

FINETUNE_DIR=${OUTPUT}/checkpoints
LOGS_DIR="${OUTPUT}/logs"
TENSORBOARD_DIR="${OUTPUT}/tensorboard"
DATACACHE_DIR="${OUTPUT}/data_cache"

export HF_DATASETS_CACHE="${OUTPUT}/hf_datasets_cache"

DATA_TRAIN="/lustre/fs1/portfolios/llmservice/users/adithyare/sft/nano_v2_fake_packed_131072_10000_rndm//stage1_stage2_multiling_128k_seq_packed.empty_assist_filtered.shuf.jsonl"

SEQ_LEN=131072 #131072 #81920 #65536

if [[ $DEBUG -eq 1 ]]; then
    MBZ=1
    BZ=64
    NW=4
    AD=0.0
    HD=0.0
    LI=1

    # EXTRA_ARGS="--deterministic-mode --use-cpu-initialization"

    NONDETERMINISTIC_ATTN=1

    NUM_GPU=8
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

    #export NCCL_ALGO=Tree
    #export CUBLAS_WORKSPACE_CONFIG=:4096:8
else
    MBZ=1
    BZ=64
    NW=8
    AD=0.0
    HD=0.0
    LI=1
    EXTRA_ARGS=""
    NONDETERMINISTIC_ATTN=1
    NUM_GPU=8
fi

if [[ $USE_CP -eq 1 ]]; then
    if [[ $BATCH -eq 1 ]]; then
        CP_SIZE=8
    else
        CP_SIZE=8
    fi
    EXTRA_ARGS+=" --context-parallel-size ${CP_SIZE} "
fi

if [[ $USE_TE_CE -eq 1 ]]; then
    EXTRA_ARGS+=" --enable-te-ce --cross-entropy-loss-fusion "
fi

if [[ $PROFILE -eq 1 ]]; then
    EXTRA_ARGS+="--profile --profile-step-start 7 --profile-step-end 8 --profile-ranks 0 16 "
fi

# CHECKPOINT_DIR="/lustre/fsw/portfolios/llmservice/users/trintamaki/workspace/output/video_sft_stage2_qwen_2p5_7b_radio_research_cp_0429_tp2/checkpoints"
TP=1
EXTRA_ARGS+=" --ckpt-format torch_dist --use-distributed-optimizer "
# EXTRA_ARGS+=" --overlap-param-gather --overlap-grad-reduce "
export CUDA_DEVICE_MAX_CONNECTIONS=1

EXTRA_ARGS+=" --use-precision-aware-optimizer --main-grads-dtype bf16 --main-params-dtype fp16 --exp-avg-dtype fp16 --exp-avg-sq-dtype fp16 "

OPTIONS=" \
    --disable-bias-linear \
    --sft \
    --legacy-tokenizer \
    --tokenizer-type SFTTokenizer \
    --sft-tokenizer-prompt-format nemotron-h-aligned \
    --tokenizer-model /lustre/fsw/portfolios/llmservice/users/kezhik/images/Nemotron-H-4B-Instruct \
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
    --pipeline-model-parallel-size 1  \
    --rerun-mode disabled \
    --num-layers 28 \
    --hidden-size 4096 \
    --ffn-hidden-size 14336 \
    --add-qkv-bias \
    --num-attention-heads 32 \
    --num-workers ${NW} \
    --exit-duration-in-mins 230 \
    --seq-length ${SEQ_LEN} \
    --max-position-embeddings ${SEQ_LEN} \
    --train-samples 100000 \
    --lr-warmup-samples 20000 \
    --micro-batch-size ${MBZ} \
    --global-batch-size ${BZ} \
    --lr 2e-5 \
    --min-lr 0.0 \
    --lr-decay-style cosine \
    --log-interval ${LI} \
    --eval-iters 10 \
    --eval-interval 999999 \
    --data-path ${DATA_TRAIN} \
    --save-interval 1000 \
    --save ${FINETUNE_DIR} \
    --load ${FINETUNE_DIR} \
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
    --group-query-attention \
    --num-query-groups 8 \
    --disable-gloo-process-groups \
    --use-dist-ckpt \
    --hybrid-context-parallel \
    --max-seqlen-per-cp-rank 8192 \
"

# --recompute-method block \
#     --recompute-num-layers 14 \
#     --recompute-granularity full \

# Interactive or batch mode
if [[ $BATCH -eq 0 ]]; then
    if [[ $PROFILE -eq 1 ]]; then
        nsys profile -w true -t cublas,cuda,nvtx,osrt -s cpu -c cudaProfilerApi -o gpt_sft_hetero_cp_iter7_8_flash_global_64 torchrun --nproc_per_node ${NUM_GPU} pretrain_gpt.py ${OPTIONS}
    else
        torchrun --nproc_per_node ${NUM_GPU} pretrain_gpt.py ${OPTIONS}
    fi
else
    if [[ $PROFILE -eq 1 ]]; then
        run_cmd="cd ${SOURCE}; nsys profile -w true -t cublas,cuda,nvtx,osrt -s cpu -c cudaProfilerApi --capture-range-end stop -o llama38b_hybrid_cp_%q{SLURM_PROCID} python -u pretrain_gpt.py ${OPTIONS}"
    else
        run_cmd="cd ${SOURCE}; python -u pretrain_gpt.py ${OPTIONS}"
    fi

    DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`

    echo "run_cmd: ${run_cmd}"

    srun -l --verbose \
    --container-image /lustre/fsw/portfolios/llmservice/users/matthieul/docker/megatron-dev-img-05142025-pytorch-dev-te-cd37379-editable-energon-mamba-fix-vlmeval-pad-conv.sqsh \
    --container-mounts "/lustre" \
    --output=${LOGS_DIR}/%x_%j_$DATETIME.log \
    sh -c "${run_cmd}"

    set +x
fi