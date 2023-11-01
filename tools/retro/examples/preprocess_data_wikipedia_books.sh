#!/bin/bash

set -u

unset NCCL_DEBUG

######## Megatron, Retro dirs. ########

REPO_DIR="/lustre/fs4/portfolios/adlr/users/boxinw/github-version/retro/Megatron-LM"
RETRO_WORKDIR="/lustre/fs4/portfolios/adlr/users/boxinw/workdirs/wiki2"

######## Task (e.g., db, index, query). ########

#RETRO_TASKS="db-build"
# RETRO_TASKS="index-train"
# RETRO_TASKS="index-add"
# RETRO_TASKS="query-pretraining-neighbors"
RETRO_TASKS=$1

######## Data. ########

DATA_HOME="/lustre/fs4/portfolios/adlr/users/boxinw/pretraining_data/"

B3="${DATA_HOME}/MTNLG/Books3_shuf_text_document"
WIK="${DATA_HOME}/MTNLG/Wikipedia_shuf_text_document"


DATA_BLEND=" \
  0.5 ${WIK} \
  0.5 ${B3} \
"

######## Index. ########

RETRO_INDEX_STR="OPQ32_64,IVF65536_HNSW8,PQ32"
RETRO_INDEX_NTRAIN=1000000
RETRO_INDEX_TRAIN_LOAD_FRACTION=0.97
RETRO_INDEX_ADD_LOAD_FRACTION=0.95

######## GPT. ########

RETRO_GPT_SEED=1234
RETRO_GPT_SPLIT="98,2,0"
RETRO_GPT_DATA_PATH=${DATA_BLEND}
RETRO_GPT_DATALOADER_TYPE=single
RETRO_GPT_EVAL_INTERVAL=2000
RETRO_GPT_EVAL_ITERS=50
RETRO_GPT_TRAIN_SAMPLES=200000
RETRO_GPT_LR_DECAY_SAMPLES=175000
RETRO_GPT_LR_WARMUP_SAMPLES=10000
RETRO_GPT_SEQ_LENGTH=512
RETRO_GPT_GLOBAL_BATCH_SIZE=256
RETRO_GPT_CHUNK_LENGTH=64

######## Query. ########

RETRO_QUERY_NUM_NEIGHBORS_QUERY=200 RETRO_QUERY_NUM_NEIGHBORS_SAVE=20
RETRO_QUERY_EF_SEARCH=32
RETRO_QUERY_NPROBE=4096

######## Args. ########

ARGS=" \
    --distributed-timeout-minutes 600 \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --num-layers 24 \
    --hidden-size 1024 \
    --num-attention-heads 16 \
    --micro-batch-size 1 \
    --global-batch-size ${RETRO_GPT_GLOBAL_BATCH_SIZE} \
    --seq-length 512 \
    --max-position-embeddings 512 \
    --load /lustre/fsw/portfolios/adlr/users/lmcafee/bert-23/checkpoints \
    --exit-on-missing-checkpoint \
    --no-load-optim \
    --no-load-rng \
    --data-path ${RETRO_GPT_DATA_PATH} \
    --tokenizer-type BertWordPieceLowerCase \
    --vocab-file  /lustre/fsw/portfolios/adlr/users/lmcafee/retro/misc/vocab/bert-large-uncased-vocab.txt \
    --split ${RETRO_GPT_SPLIT} \
    --distributed-backend nccl \
    --lr 0.0001 \
    --lr-decay-style linear \
    --min-lr 1.0e-5 \
    --train-samples ${RETRO_GPT_TRAIN_SAMPLES} \
    --lr-decay-samples ${RETRO_GPT_LR_DECAY_SAMPLES} \
    --lr-warmup-samples ${RETRO_GPT_LR_WARMUP_SAMPLES} \
    --weight-decay 1e-2 \
    --clip-grad 1.0 \
    --eval-interval ${RETRO_GPT_EVAL_INTERVAL} \
    --eval-iters ${RETRO_GPT_EVAL_ITERS} \
    --fp16 \
    --dataloader-type ${RETRO_GPT_DATALOADER_TYPE} \
    --no-data-sharding \
    --no-gradient-accumulation-fusion \
    --no-async-tensor-model-parallel-allreduce \
    --bert-embedder-type megatron \
    --output-bert-embeddings \
    \
    --retro-workdir ${RETRO_WORKDIR} \
    --retro-tasks ${RETRO_TASKS} \
    --retro-return-doc-ids \
    --retro-bert-vocab-file  /lustre/fsw/portfolios/adlr/users/lmcafee/retro/misc/vocab/bert-large-uncased-vocab.txt \
    --retro-bert-tokenizer-type BertWordPieceLowerCase \
    --retro-gpt-seed ${RETRO_GPT_SEED} \
    --retro-gpt-tokenizer-type GPTSentencePieceTokenizer \
    --retro-gpt-tokenizer-model /lustre/fsw/portfolios/adlr/users/lmcafee/retro/misc/next-llm-tokenizer/mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model \
    --retro-gpt-seq-length ${RETRO_GPT_SEQ_LENGTH} \
    --retro-gpt-chunk-length ${RETRO_GPT_CHUNK_LENGTH} \
    --retro-gpt-global-batch-size ${RETRO_GPT_GLOBAL_BATCH_SIZE} \
    --retro-gpt-eval-interval ${RETRO_GPT_EVAL_INTERVAL} \
    --retro-gpt-eval-iters ${RETRO_GPT_EVAL_ITERS} \
    --retro-gpt-split ${RETRO_GPT_SPLIT} \
    --retro-gpt-data-path ${RETRO_GPT_DATA_PATH} \
    --retro-index-str ${RETRO_INDEX_STR} \
    --retro-index-ntrain ${RETRO_INDEX_NTRAIN} \
    --retro-index-train-load-fraction ${RETRO_INDEX_TRAIN_LOAD_FRACTION} \
    --retro-index-add-load-fraction ${RETRO_INDEX_ADD_LOAD_FRACTION} \
    --retro-index-no-delete-training-embeddings \
    --retro-index-no-delete-added-codes \
    --retro-query-num-neighbors-query ${RETRO_QUERY_NUM_NEIGHBORS_QUERY} \
    --retro-query-num-neighbors-save ${RETRO_QUERY_NUM_NEIGHBORS_SAVE} \
    --retro-query-ef-search ${RETRO_QUERY_EF_SEARCH} \
    --retro-query-nprobe ${RETRO_QUERY_NPROBE} \
"

######## Command. ########

NPROCS=8 # Number of GPUs.
NODE_RANK=0
MASTER_ADDR=localhost
CMD="\
    cd ${REPO_DIR} && pwd && \
    export PYTHONPATH=$PYTHONPATH:${REPO_DIR} && \
    python -m torch.distributed.run \
    --nproc_per_node ${NPROCS} \
    --nnodes 1 \
    --node_rank ${NODE_RANK} \
    --master_addr ${MASTER_ADDR} \
    --master_port 6000 \
    tools/retro/main.py ${ARGS} \
"
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo "CMD = '$CMD'."
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~"
eval $CMD
