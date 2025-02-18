#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=8

DATA_PATH=${DATA_PATH:-"/mnt/cluster/deepseek-ai/data/deepseek_content_document"}

TOKENIZER_PATH=${TOKENIZER_PATH:-"/mnt/cluster/deepseek-ai/DeepSeek-V2/"}

CHECKPOINT_PATH=${CHECKPOINT_PATH:-"/mnt/cluster/deepseek-ai/DeepSeek_V2_tp16pp1ep8/"}

TENSORBOARD_PATH=${TENSORBOARD_PATH:-"/mnt/cluster/aiak-training-llm/tensorboard-log/deepseek-v2"}

NSYS_PATH=${NSYS_PATH:-"/mnt/cluster/deepseek-ai/nsys/"}

# Change for multinode config
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"6000"}
NNODES=${WORLD_SIZE:-"1"}
NODE_RANK=${RANK:-"0"}

# NSYS_ARGS="nsys profile \
# 	   -s none -t nvtx,cuda \
#            --force-overwrite true \
#            -o $NSYS_PATH/DeepSeek_V2_tp16pp1ep8_${OMPI_COMM_WORLD_SIZE}_${OMPI_COMM_WORLD_RANK}"

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NNODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

# def deepseek_v2():
#     """deepseek-v2"""
#     return DeepseekConfig(
#         num_layers=60,
#         hidden_size=5120,
#         ffn_hidden_size=12288,
#         num_attention_heads=128,
#         num_query_groups=128,
#         num_experts=160,
#         moe_ffn_hidden_size=1536,
#         moe_shared_expert_intermediate_size=3072,
#         q_lora_rank=1536,
#         kv_lora_rank=512,
#         qk_nope_head_dim=128,
#         qk_rope_head_dim=64,
#         v_head_dim=128,
#         moe_layer_freq=1,
#         first_k_dense_replace=1,
#         qk_layernorm=True,
#         vocab_size_in_config_file=102400,
#         moe_router_num_groups=8,
#         moe_router_group_topk=3
#     )

DEEPSEEK_MODEL_ARGS=(
    --num-layers 60
    --hidden-size 5120
    --ffn-hidden-size 12288
    --num-attention-heads 128
    --num-query-groups 128
    --num-experts 160
    --moe-ffn-hidden-size 1536
    --moe-shared-expert-intermediate-size 3072
    --q-lora-rank 1536
    --kv-lora-rank 512
    --qk-nope-head-dim 128
    --qk-rope-head-dim 64
    --v-head-dim 128
    --moe-layer-freq 1
    --qk-layernorm True
    --vocab-size 102400
    --moe-router-num-groups 8
    --moe-router-group-topk 3
)

DATA_ARGS=(
    --tokenizer-type DeepSeekV2Tokenizer
    --tokenizer-model $TOKENIZER_PATH
    --data-path $DATA_PATH
    --split 99990,8,2
)

TRAINING_ARGS=(
    --training-phase pretrain
    --seq-length 2048
    --max-position-embeddings 2048
    --init-method-std 0.01
    --no-masked-softmax-fusion
    --micro-batch-size 1
    --global-batch-size 16
    --lr 1e-4
    --train-iters 500000
    --lr-decay-iters 320000
    --lr-decay-style cosine
    --min-lr 1.0e-5
    --weight-decay 0.1
    --lr-warmup-iters 500
    --clip-grad 1.0
    --bf16
    --load $CHECKPOINT_PATH
#    --save $CHECKPOINT_PATH
    --save-interval 10000
    --eval-interval 1000
    --eval-iters 10
    --no-load-optim
    --no-load-rng
    --multi-latent-attention
)

MOE_ARGS=(
    --moe-router-load-balancing-type aux_loss
    --moe-router-topk 6
    --moe-aux-loss-coeff 1e-3
    #--moe-grouped-gemm
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 16
    --pipeline-model-parallel-size 1
    --expert-model-parallel-size 8
    --sequence-parallel
    --moe-token-dispatcher-type alltoall
    --use-distributed-optimizer
)

LOGGING_ARGS=(
    --log-interval 1
    --tensorboard-dir ${TENSORBOARD_PATH}
    --log-timers-to-tensorboard
)

torchrun ${DISTRIBUTED_ARGS[@]} pretrain_deepseek_v2.py \
    ${DEEPSEEK_MODEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MOE_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${LOGGING_ARGS[@]}