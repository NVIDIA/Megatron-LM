#!/bin/bash
cd /lustre/fsw/joc/huvu/codes/T5_mcore/megatron-lm-updated/megatron-lm
pip install -e .

export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

# CHECKPOINT_PATH="/lustre/fsw/joc/huvu/data/t5/trained_models/test10"
# VOCAB_FILE="/lustre/fsw/joc/huvu/data/t5/vocab/bert-large-cased-vocab.txt"
# DATA_PATH="/lustre/fsw/joc/huvu/data/t5/training_data/bc_rn_owt_sto_wiki_dedup_shuf_cleaned_0.7_mmap"
# TENSORBOARD_DIR=$CHECKPOINT_PATH

# # Pile dataset partial (original path: /lustre/fsw/joc/big_nlp/t5/dataset/Pile/)
# CHECKPOINT_PATH="/lustre/fsw/joc/huvu/data/t5/trained_models/sbatch_pile_testcheckpoint_test1"
# VOCAB_FILE="/lustre/fsw/joc/big_nlp/t5/dataset/Pile/bert-large-cased-vocab.txt"
# DATA_PATH="/lustre/fsw/joc/huvu/data/t5/training_data/my-t5_00_bert_tokenizer_text_document" # [can't be used unless having the right vocab file and right tokenizer]
# TENSORBOARD_DIR=$CHECKPOINT_PATH

# Pile dataset full (original path: /lustre/fsw/joc/big_nlp/t5/dataset/Pile/)
CHECKPOINT_PATH="/lustre/fsw/joc/huvu/data/t5/trained_models/test28"
VOCAB_FILE="/lustre/fsw/joc/big_nlp/t5/dataset/Pile/bert-large-cased-vocab.txt"
DATA_PATH=""
for k in {00..29}; do
    DATA_PATH+=" 0.033 /lustre/fsw/joc/huvu/data/t5/training_data/symlinks/my-t5_${k}_bert_tokenizer_text_document"
done
TEST_NAME=transformer_engine
TENSORBOARD_DIR=$CHECKPOINT_PATH/$TEST_NAME


DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"


# original run
T5_ARGS="
    --num-layers 12 \
    --hidden-size 768 \
    --num-attention-heads 12 \
    --kv-channels 64 \
    --ffn-hidden-size 3072 \
    --encoder-seq-length 512 \
    --decoder-seq-length 128 \
    --max-position-embeddings 512 \
    --micro-batch-size 64 \
    --global-batch-size 512 \
    --lr 0.0001 \
    --train-iters 1000000 \
    --lr-decay-iters 1000000 \
    --lr-decay-style linear \
    --min-lr 0.00001 \
    --weight-decay 1e-2 \
    --lr-warmup-fraction .01 \
    --clip-grad 1.0 \
    --bf16 \
    --vocab-extra-ids 100 \
    --init-method-std 0.015 \
    --transformer-impl transformer_engine
"

## TP-DP-PP (mainly TP)
T5_ARGS="
    --num-layers 12 \
    --hidden-size 768 \
    --num-attention-heads 12 \
    --kv-channels 64 \
    --ffn-hidden-size 3072 \
    --encoder-seq-length 512 \
    --decoder-seq-length 128 \
    --tensor-model-parallel-size 2 \
    --pipeline-model-parallel-size 1 \
    --pipeline-model-parallel-split-rank 1 \
    --max-position-embeddings 512 \
    --micro-batch-size 64 \
    --global-batch-size 512 \
    --lr 0.0001 \
    --train-iters 1000000 \
    --lr-decay-iters 1000000 \
    --lr-decay-style linear \
    --min-lr 0.00001 \
    --weight-decay 1e-2 \
    --lr-warmup-fraction .01 \
    --clip-grad 1.0 \
    --bf16 \
    --vocab-extra-ids 100 \
    --init-method-std 0.015 \
    --transformer-impl transformer_engine
"

# ## use flash-attention
# T5_ARGS="
#     --num-layers 12 \
#     --hidden-size 768 \
#     --num-attention-heads 12 \
#     --kv-channels 64 \
#     --ffn-hidden-size 3072 \
#     --encoder-seq-length 512 \
#     --decoder-seq-length 128 \
#     --tensor-model-parallel-size 1 \
#     --pipeline-model-parallel-size 1 \
#     --pipeline-model-parallel-split-rank 1 \
#     --max-position-embeddings 512 \
#     --micro-batch-size 64 \
#     --global-batch-size 512 \
#     --lr 0.0001 \
#     --train-iters 1000000 \
#     --lr-decay-iters 1000000 \
#     --lr-decay-style linear \
#     --min-lr 0.00001 \
#     --weight-decay 1e-2 \
#     --lr-warmup-fraction .01 \
#     --clip-grad 1.0 \
#     --bf16 \
#     --vocab-extra-ids 100 \
#     --init-method-std 0.015 \
#     --transformer-impl transformer_engine \
#     --use-flash-attn
# "

# distributed optimizer
T5_ARGS="
    --num-layers 12 \
    --hidden-size 768 \
    --num-attention-heads 12 \
    --kv-channels 64 \
    --ffn-hidden-size 3072 \
    --encoder-seq-length 512 \
    --decoder-seq-length 128 \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --max-position-embeddings 512 \
    --micro-batch-size 64 \
    --global-batch-size 512 \
    --lr 0.0001 \
    --train-iters 1000000 \
    --lr-decay-iters 1000000 \
    --lr-decay-style linear \
    --min-lr 0.00001 \
    --weight-decay 1e-2 \
    --lr-warmup-fraction .01 \
    --clip-grad 1.0 \
    --bf16 \
    --vocab-extra-ids 100 \
    --init-method-std 0.015 \
    --transformer-impl transformer_engine \
    --use-distributed-optimizer
"

## use rope embeddings
T5_ARGS="
    --num-layers 12 \
    --hidden-size 768 \
    --num-attention-heads 12 \
    --kv-channels 64 \
    --ffn-hidden-size 3072 \
    --encoder-seq-length 512 \
    --decoder-seq-length 128 \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --pipeline-model-parallel-split-rank 1 \
    --max-position-embeddings 512 \
    --micro-batch-size 64 \
    --global-batch-size 512 \
    --lr 0.0001 \
    --train-iters 1000000 \
    --lr-decay-iters 1000000 \
    --lr-decay-style linear \
    --min-lr 0.00001 \
    --weight-decay 1e-2 \
    --lr-warmup-fraction .01 \
    --clip-grad 1.0 \
    --bf16 \
    --vocab-extra-ids 100 \
    --init-method-std 0.015 \
    --transformer-impl transformer_engine \
    --position-embedding-type rope
"


## not use transformer-engine
T5_ARGS="
    --num-layers 12 \
    --hidden-size 768 \
    --num-attention-heads 12 \
    --kv-channels 64 \
    --ffn-hidden-size 3072 \
    --encoder-seq-length 512 \
    --decoder-seq-length 128 \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --pipeline-model-parallel-split-rank 1 \
    --max-position-embeddings 512 \
    --micro-batch-size 64 \
    --global-batch-size 512 \
    --lr 0.0001 \
    --train-iters 1000000 \
    --lr-decay-iters 1000000 \
    --lr-decay-style linear \
    --min-lr 0.00001 \
    --weight-decay 1e-2 \
    --lr-warmup-fraction .01 \
    --clip-grad 1.0 \
    --bf16 \
    --vocab-extra-ids 100 \
    --init-method-std 0.015 \
    --transformer-impl transformer_engine \
"

tests:
 - use TE
 - TP
 - FA
 - total:(TE-DO-TP) transformer-engine / distributed optimizer / tensor parallel
    + 0-1-0: yes - resume: yes
    + 0-1-1: yes - resume: yes
    + 0-0-0: yes - resume: yes
    + 0-0-1: yes - resume: yes
    + 1-1-0: yes - resume: yes
    + 1-1-1: yes - resume: yes
    + 1-0-0: yes - resume: yes
    + 1-0-1: yes - resume: yes


# export NVTE_FLASH_ATTN=1
# export NVTE_FUSED_ATTN=1
T5_ARGS="
    --num-layers 12 \
    --hidden-size 768 \
    --num-attention-heads 12 \
    --kv-channels 64 \
    --ffn-hidden-size 3072 \
    --encoder-seq-length 512 \
    --decoder-seq-length 128 \
    --tensor-model-parallel-size 2 \
    --pipeline-model-parallel-size 1 \
    --pipeline-model-parallel-split-rank 1 \
    --max-position-embeddings 512 \
    --micro-batch-size 64 \
    --global-batch-size 512 \
    --lr 0.0001 \
    --train-iters 1000000 \
    --lr-decay-iters 1000000 \
    --lr-decay-style linear \
    --min-lr 0.00001 \
    --weight-decay 1e-2 \
    --lr-warmup-fraction .01 \
    --clip-grad 1.0 \
    --bf16 \
    --vocab-extra-ids 100 \
    --init-method-std 0.015 \
    --transformer-impl transformer_engine
"

no use-distributed-optimizer: 24637MiB
use-distributed-optimizer: 23301MiB


# # original
# T5_ARGS="
#     --num-layers 12 \
#     --hidden-size 768 \
#     --num-attention-heads 12 \
#     --kv-channels 64 \
#     --ffn-hidden-size 3072 \
#     --encoder-seq-length 512 \
#     --decoder-seq-length 128 \
#     --max-position-embeddings 512 \
#     --micro-batch-size 64 \
#     --global-batch-size 512 \
#     --lr 0.0001 \
#     --train-iters 1000000 \
#     --lr-decay-iters 1000000 \
#     --lr-decay-style linear \
#     --min-lr 0.00001 \
#     --weight-decay 1e-2 \
#     --lr-warmup-fraction .01 \
#     --clip-grad 1.0 \
#     --fp16 \
#     --vocab-extra-ids 100
# "

# # run with bf16
# T5_ARGS="
#     --num-layers 12 \
#     --hidden-size 768 \
#     --num-attention-heads 12 \
#     --kv-channels 64 \
#     --ffn-hidden-size 3072 \
#     --encoder-seq-length 512 \
#     --decoder-seq-length 128 \
#     --max-position-embeddings 512 \
#     --micro-batch-size 64 \
#     --global-batch-size 512 \
#     --lr 0.0001 \
#     --train-iters 1000000 \
#     --lr-decay-iters 1000000 \
#     --lr-decay-style linear \
#     --min-lr 0.00001 \
#     --weight-decay 1e-2 \
#     --lr-warmup-fraction .01 \
#     --clip-grad 1.0 \
#     --bf16 \
#     --vocab-extra-ids 100
# "



# # continue training of /lustre/fsw/joc/huvu/data/t5/trained_models/sbatch_pile_test1
# T5_ARGS="
#     --num-layers 12 \
#     --hidden-size 768 \
#     --num-attention-heads 12 \
#     --kv-channels 64 \
#     --ffn-hidden-size 3072 \
#     --encoder-seq-length 512 \
#     --decoder-seq-length 128 \
#     --max-position-embeddings 512 \
#     --micro-batch-size 64 \
#     --global-batch-size 512 \
#     --lr 0.0001 \
#     --train-iters 1000000 \
#     --lr-decay-iters 1000000 \
#     --lr-decay-style linear \
#     --min-lr 0.00001 \
#     --weight-decay 1e-2 \
#     --lr-warmup-fraction .01 \
#     --clip-grad 1.0 \
#     --fp16 \
#     --vocab-extra-ids 100
# "


# ## running with bf16 instead of fp16
# T5_ARGS="
#     --num-layers 12 \
#     --hidden-size 768 \
#     --num-attention-heads 12 \
#     --kv-channels 64 \
#     --ffn-hidden-size 3072 \
#     --encoder-seq-length 512 \
#     --decoder-seq-length 128 \
#     --max-position-embeddings 512 \
#     --micro-batch-size 64 \
#     --global-batch-size 512 \
#     --lr 0.0001 \
#     --train-iters 1000000 \
#     --lr-decay-iters 1000000 \
#     --lr-decay-style linear \
#     --min-lr 0.00001 \
#     --weight-decay 1e-2 \
#     --lr-warmup-fraction .01 \
#     --clip-grad 1.0 \
#     --bf16 \
#     --vocab-extra-ids 100
# "


# ## different batch-size
# T5_ARGS="
#     --num-layers 12 \
#     --hidden-size 768 \
#     --num-attention-heads 12 \
#     --kv-channels 64 \
#     --ffn-hidden-size 3072 \
#     --encoder-seq-length 512 \
#     --decoder-seq-length 128 \
#     --max-position-embeddings 512 \
#     --micro-batch-size 128 \
#     --global-batch-size 1024 \
#     --lr 0.0001 \
#     --train-iters 1000000 \
#     --lr-decay-iters 1000000 \
#     --lr-decay-style linear \
#     --min-lr 0.00001 \
#     --weight-decay 1e-2 \
#     --lr-warmup-fraction .01 \
#     --clip-grad 1.0 \
#     --fp16 \
#     --vocab-extra-ids 100
# "


# ## TP-DP-PP
# T5_ARGS="
#     --num-layers 12 \
#     --hidden-size 768 \
#     --num-attention-heads 12 \
#     --kv-channels 64 \
#     --ffn-hidden-size 3072 \
#     --encoder-seq-length 512 \
#     --decoder-seq-length 128 \
#     --max-position-embeddings 512 \
#     --micro-batch-size 16 \
#     --tensor-model-parallel-size 2 \
#     --pipeline-model-parallel-size 4 \
#     --pipeline-model-parallel-split-rank 3 \
#     --lr 0.0001 \
#     --train-iters 1000000 \
#     --lr-decay-iters 1000000 \
#     --lr-decay-style linear \
#     --min-lr 0.00001 \
#     --weight-decay 1e-2 \
#     --lr-warmup-fraction .01 \
#     --clip-grad 1.0 \
#     --fp16 \
#     --vocab-extra-ids 100
# "


# ## fp8 (check core/transformer/transformer_config.py) - only work on H100
# T5_ARGS="
#     --num-layers 12 \
#     --hidden-size 768 \
#     --num-attention-heads 12 \
#     --kv-channels 64 \
#     --ffn-hidden-size 3072 \
#     --encoder-seq-length 512 \
#     --decoder-seq-length 128 \
#     --max-position-embeddings 512 \
#     --micro-batch-size 16 \
#     --global-batch-size 128 \
#     --lr 0.0001 \
#     --train-iters 1000000 \
#     --lr-decay-iters 1000000 \
#     --lr-decay-style linear \
#     --min-lr 0.00001 \
#     --weight-decay 1e-2 \
#     --lr-warmup-fraction .01 \
#     --clip-grad 1.0 \
#     --fp8-format hybrid \
#     --vocab-extra-ids 100
# "

# ## different encoder-seq-length and decoder-seq-length
# T5_ARGS="
#     --num-layers 12 \
#     --hidden-size 768 \
#     --num-attention-heads 12 \
#     --kv-channels 64 \
#     --ffn-hidden-size 3072 \
#     --encoder-seq-length 512 \
#     --decoder-seq-length 128 \
#     --max-position-embeddings 512 \
#     --micro-batch-size 128 \
#     --global-batch-size 1024 \
#     --lr 0.0001 \
#     --train-iters 1000000 \
#     --lr-decay-iters 1000000 \
#     --lr-decay-style linear \
#     --min-lr 0.00001 \
#     --weight-decay 1e-2 \
#     --lr-warmup-fraction .01 \
#     --clip-grad 1.0 \
#     --fp16 \
#     --vocab-extra-ids 100
# "

# ## rope relative positional encoding
# T5_ARGS="
#     --num-layers 12 \
#     --hidden-size 768 \
#     --num-attention-heads 12 \
#     --kv-channels 64 \
#     --ffn-hidden-size 2048 \
#     --encoder-seq-length 512 \
#     --decoder-seq-length 128 \
#     --position-embedding-type learned_absolute \
#     --max-position-embeddings 512 \
#     --micro-batch-size 16 \
#     --global-batch-size 128 \
#     --lr 0.0001 \
#     --train-iters 1000000 \
#     --lr-decay-iters 1000000 \
#     --lr-decay-style linear \
#     --min-lr 0.00001 \
#     --weight-decay 1e-2 \
#     --lr-warmup-fraction .01 \
#     --clip-grad 1.0 \
#     --fp16 \
#     --vocab-extra-ids 100
# "

# # old version
# DATA_ARGS="
#     --data-path $DATA_PATH \
#     --vocab-file $VOCAB_FILE \
#     --data-impl mmap \
#     --tokenizer-type BertWordPieceCase \
#     --split 99982,9,9 \
# "

DATA_ARGS="
    --data-path $DATA_PATH \
    --vocab-file $VOCAB_FILE \
    --tokenizer-type BertWordPieceCase \
    --split 99982,9,9 \
"


OUTPUT_ARGS="
    --log-interval 100 \
    --tensorboard-dir ${TENSORBOARD_DIR} \
    --save-interval 500 \
    --eval-interval 1000 \
    --eval-iters 10
"

# cd /lustre/fsw/joc/huvu/codes/T5_mcore/megatron-lm-updated/megatron-lm
# pip install -e .

mkdir $CHECKPOINT_PATH
torchrun $DISTRIBUTED_ARGS pretrain_t5_core.py \
    $T5_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH \
