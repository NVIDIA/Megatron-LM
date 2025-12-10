#!/bin/bash

#SBATCH -A coreai_devtech_all
# DFW: batch
# OCI-NRT: batch_block1
# OCI-IAD: batch_block1,batch_block3,batch_block4,backfill_block1,backfill_block2,backfill_block3,backfill_block4
#SBATCH -p batch
#SBATCH -t 00:30:00
#SBATCH --mem=0
#SBATCH --ntasks-per-node=8
#SBATCH --nodes=1
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
USE_CP=0
USE_TE_CE=1
USE_FLASH_ATTN=0
USE_FSDP=1
PROFILE=0
USE_MOCK_DATA=1
TP=1

# Remember to update model and job name if running in batch mode!!
if [[ $BATCH -eq 0 ]]; then
    DATETIME=`date +'%y-%m-%d-%H-%M-%S'`
    MODEL_NAME="interactive_hybrid_cp"
    WORKSPACE="/home/tailaim//work_data/megatron-lm/logs"
    SOURCE="/home/tailaim/work_data/megatron-lm"
    TOKENIZER="/home/tailaim/work_data/megatron-moe-scripts/Nemotron-H-4B-Instruct"
else
    MODEL_NAME="interactive_hybrid_cp"
    WORKSPACE="/lustre/fsw/portfolios/coreai/users/tailaim/work_data/megatron-lm/logs"
    SOURCE="/lustre/fsw/portfolios/coreai/users/tailaim/work_data/megatron-lm"
    TOKENIZER="/lustre/fsw/portfolios/llmservice/users/kezhik/images/Nemotron-H-4B-Instruct"
fi

WORKSPACE="/lustre/fsw/portfolios/coreai/users/tailaim/work_data/megatron-lm/logs"
SOURCE="/lustre/fsw/portfolios/coreai/users/tailaim/work_data/megatron-lm"
OUTPUT_BASE="${WORKSPACE}/output"
OUTPUT="${OUTPUT_BASE}/${MODEL_NAME}"

FINETUNE_DIR=${OUTPUT}/checkpoints
LOGS_DIR="${OUTPUT}/logs"
TENSORBOARD_DIR="${OUTPUT}/tensorboard"
DATACACHE_DIR="${OUTPUT}/data_cache"

export HF_DATASETS_CACHE="${OUTPUT}/hf_datasets_cache"

DATA_TRAIN="/home/tailaim/data/thd_formatted_100k.jsonl"

SEQ_LEN=16384 #131072 #81920 #65536

if [[ $DEBUG -eq 1 ]]; then
    MBZ=1
    BZ=256
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
    BZ=256
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
        CP_SIZE=4
    else
        CP_SIZE=4
    fi
    EXTRA_ARGS+=" --context-parallel-size ${CP_SIZE} "
fi

if [[ $USE_TE_CE -eq 1 ]]; then
    EXTRA_ARGS+=" --cross-entropy-loss-fusion --cross-entropy-fusion-impl te"
fi

if [[ $PROFILE -eq 1 ]]; then
    EXTRA_ARGS+="--profile --profile-step-start 7 --profile-step-end 8 "
fi

if [[ $USE_MOCK_DATA -eq 1 ]]; then
    # EXTRA_ARGS+=" --mock-data --sft-mock-dataset-config-json '{\"mode\":\"file\",\"path\":\"path/to/file\"}'"
    if [[ $BATCH -eq 0 ]]; then
    EXTRA_ARGS+=" --mock-data --sft-mock-dataset-config-json {\"mode\":\"distribution\",\"type\":\"lognormal\",\"min_seq_len\":1024,\"max_seq_len\":16384,\"mean_seq_len\":8192,\"lognormal_sigma\":1.1} --tokenizer-type NullTokenizer --vocab-size 131072 "
    else
    EXTRA_ARGS+=" --mock-data --sft-mock-dataset-config-json '{\"mode\":\"distribution\",\"type\":\"lognormal\",\"min_seq_len\":1024,\"max_seq_len\":16384,\"mean_seq_len\":8192,\"lognormal_sigma\":1.1}' --tokenizer-type NullTokenizer --vocab-size 131072 "
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



OPTIONS=" \
    --hybrid-context-parallel \
    --sft-sequence-packing \
    --max-seqlen-per-dp-cp-rank 4096 \
    --sft \
    --tokenizer-type SFTTokenizer \
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
    --pipeline-model-parallel-size 1 \
    --rerun-mode disabled \
    --num-layers 4 \
    --hidden-size 2048 \
    --ffn-hidden-size 8192 \
    --add-qkv-bias \
    --num-attention-heads 16 \
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
    --disable-gloo-process-groups \
    --use-dist-ckpt \
"


# Interactive or batch mode
if [[ $BATCH -eq 0 ]]; then
    if [[ $PROFILE -eq 1 ]]; then
        nsys profile -w true -t cublas,cuda,nvtx,osrt -s cpu -c cudaProfilerApi -o gpt_sft_hetero_cp_iter7_8_flash_global_64 torchrun --nproc_per_node ${NUM_GPU} pretrain_gpt.py ${OPTIONS}
    else
        torchrun --nproc_per_node ${NUM_GPU} /home/tailaim/work_data/megatron-lm/pretrain_gpt.py ${OPTIONS}
    fi
else
    if [[ $PROFILE -eq 1 ]]; then
        run_cmd="cd ${SOURCE}; nsys profile -w true -t cublas,cuda,nvtx,osrt -s cpu -c cudaProfilerApi --capture-range-end stop -o without_hetero_cp_global_%q{SLURM_PROCID} python -u pretrain_gpt.py ${OPTIONS}"
    else
        run_cmd="cd ${SOURCE}; python -u pretrain_gpt.py ${OPTIONS}"
    fi

    DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`
    echo "run_cmd: ${run_cmd}"
    srun -l --verbose \
    --container-image /lustre/fsw/portfolios/coreai/users/tailaim/work_data/megatron-moe-scripts/mcore-moe-pytorch25.06.sqsh \
    --container-mounts "/lustre" \
    --no-container-mount-home \
    --output=${LOGS_DIR}/%x_%j_$DATETIME.log \
    sh -c "${run_cmd}"

    set +x
fi
