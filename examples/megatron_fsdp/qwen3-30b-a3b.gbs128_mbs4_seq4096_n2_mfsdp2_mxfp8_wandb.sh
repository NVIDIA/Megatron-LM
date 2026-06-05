#!/bin/bash

#SBATCH --job-name=qwen3-30b-a3b.gbs128_mbs4_seq4096_n2_mfsdp2_mxfp8
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --partition=gb200
#SBATCH --account=<your_slurm_account>
#SBATCH --time=2:00:00
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --dependency=singleton

export NCCL_IB_SL=1
export NCCL_IB_TIMEOUT=19
export NVTE_FWD_LAYERNORM_SM_MARGIN=16
export NVTE_BWD_LAYERNORM_SM_MARGIN=16
export NCCL_P2P_NET_CHUNKSIZE=2097152
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_DEVICE_MAX_CONNECTIONS=8
export WANDB_API_KEY=${WANDB_API_KEY:-""}
export WANDB_ENTITY=${WANDB_ENTITY:-"nvidia"}

MEGATRON_PATH=${MEGATRON_PATH:-"/path/to/Megatron-LM"}
OUTPUT_PATH=${OUTPUT_PATH:-"${MEGATRON_PATH}/output"}
NAME=qwen3-30b-a3b.gbs128_mbs4_seq4096_n2_mfsdp2_mxfp8_wandb
mkdir -p "$OUTPUT_PATH/checkpoints/$NAME"
CHECKPOINT_DIR=${CHECKPOINT_DIR:-"${MEGATRON_PATH}/checkpoints"}
DATA_PATH=${DATA_PATH:-"/path/to/data/c4/en/c4-train.en_6_text_document"}
TOKENIZER_MODEL=${TOKENIZER_MODEL:-"/path/to/data/c4/en/tokenizer"}

PRETRAIN_ARGS=(
    --seq-length 4096
    --max-position-embeddings 4096
    --tensor-model-parallel-size 1
    --context-parallel-size 1
    --pipeline-model-parallel-size 1
    --expert-tensor-parallel-size 1
    --use-mcore-models
    --use-flash-attn
    --no-check-for-nan-in-loss-and-grad
    --manual-gc
    --manual-gc-interval 10
    --recompute-granularity selective
    --recompute-modules moe_act
    --transformer-impl transformer_engine
    --apply-layernorm-1p
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model ${TOKENIZER_MODEL}
    --data-path ${DATA_PATH}
    --data-cache-path $MEGATRON_PATH/data_cache
    --num-layers 48
    --hidden-size 2048
    --ffn-hidden-size 6144
    --num-attention-heads 32
    --norm-epsilon 1e-06
    --normalization RMSNorm
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --position-embedding-type rope
    --rotary-base 1000000
    --swiglu
    --untie-embeddings-and-output-weights
    --disable-bias-linear
    --group-query-attention
    --num-query-groups 4
    --kv-channels 128
    --make-vocab-size-divisible-by 1187
    --qk-layernorm
    --attention-backend fused
    --rotary-percent 1.0
    --rotary-seq-len-interpolation-factor 1
    --clip-grad 1.0
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.95
    --lr 0.00012
    --min-lr 1.2e-05
    --lr-decay-samples 255126953
    --lr-warmup-samples 162761
    --lr-decay-style cosine
    --init-method-std 0.02
    --ckpt-format fsdp_dtensor
    --train-samples 268554688
    --exit-duration-in-mins 230
    --split 99,1,0
    --eval-iters 32
    --eval-interval 100
    --log-interval 1
    --distributed-timeout-minutes 60
    --no-mmap-bin-files
    --no-create-attention-mask-in-dataloader
    --log-throughput
    --bf16
    --num-workers 6
    --enable-experimental
    --auto-detect-ckpt-format
    --save $OUTPUT_PATH/checkpoints/$NAME
    --save-interval 500
    --load $OUTPUT_PATH/checkpoints/$NAME
    --dist-ckpt-strictness log_all
    --fp8-recipe mxfp8
    --fp8-format e4m3
    --overlap-grad-reduce
    --overlap-param-gather
    --use-megatron-fsdp
    --data-parallel-sharding-strategy optim_grads_params
    --calculate-per-token-loss
    --init-model-with-meta-device
    --grad-reduce-in-bf16
    --use-distributed-optimizer
    --num-experts 128
    --moe-ffn-hidden-size 768
    --moe-router-load-balancing-type aux_loss
    --moe-router-topk 8
    --moe-aux-loss-coeff 0.001
    --moe-grouped-gemm
    --moe-router-dtype fp32
    --moe-permute-fusion
    --moe-router-fusion
    --moe-router-force-load-balancing
    --moe-token-dispatcher-type alltoall
    --sequence-parallel
    --cross-entropy-loss-fusion
    --cross-entropy-fusion-impl te
    --wandb-project megatron-fsdp
    --wandb-exp-name ${NAME}
    --wandb-save-dir $OUTPUT_PATH/wandb
    --log-timers-to-tensorboard
    --log-memory-to-tensorboard
    --log-num-zeros-in-grad
    --log-params-norm
    --log-validation-ppl-to-tensorboard
    --tensorboard-dir $OUTPUT_PATH/tensorboard
    --micro-batch-size 4
    --fp8-param-gather
    --expert-model-parallel-size 4
    --global-batch-size 128
    --use-megatron-fsdp-v2
)

run_cmd="
cd $MEGATRON_PATH;
git rev-parse HEAD;
export PYTHONPATH=$MEGATRON_PATH:$PYTHONPATH;
python3 -u pretrain_gpt.py ${PRETRAIN_ARGS[@]}"

srun --mpi=pmix -l \
--container-image nvcr.io/nvidia/nemo:26.04 \
--container-mounts "/home:/home,/lustre:/lustre" --no-container-mount-home \
bash -x -c "${run_cmd}"
