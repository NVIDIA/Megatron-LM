#!/bin/bash

# Finetune a pretrained language model to generate the corresponding response
# The input is the dialogue context and knowledge, and the output is the response
# The size of the pretrained language model is 357M

WORLD_SIZE=8

DISTRIBUTED_ARGS="--nproc_per_node $WORLD_SIZE \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr localhost \
                  --master_port 6000"

CHECKPOINT_PATH=<PATH_OF_THE_LANGUAGE_MODEL>
OUTPUT_MODEL_PATH=<PATH_OF_THE_SAVED_MODEL>
VOCAB_PATH=<PATH_OF_THE_VOCAB_FILE>
MERGE_PATH=<PATH_OF_THE_MERGE_FILE>
TRAIN_PATH=<PATH_OF_THE_TRAINING_DATASET>
TEST_PATH=<PATH_OF_THE_TEST_DATASET>

python -m torch.distributed.launch $DISTRIBUTED_ARGS ./tasks/main.py \
        --num-layers 24 \
        --hidden-size 1024 \
        --num-attention-heads 16 \
        --seq-length 2048 \
        --max-position-embeddings 2048 \
        --micro-batch-size 4 \
        --global-batch-size 64 \
        --train-samples 142000 \
        --lr-decay-samples 10000 \
        --lr-warmup-samples 3000 \
        --lr 1.0e-5 \
        --min-lr 5.0e-6 \
        --lr-decay-style cosine \
        --vocab-file ${VOCAB_PATH} \
        --merge-file ${MERGE_PATH} \
        --save-interval 10000 \
        --save ${OUTPUT_MODEL_PATH} \
        --pretrained-checkpoint ${CHECKPOINT_PATH} \
        --weight-decay 0.1 \
        --adam-beta2 0.95 \
        --log-params-norm \
        --log-num-zeros-in-grad \
        --fp16 \
        --DDP-impl torch \
        --checkpoint-activations \
        --epochs 3 \
        --task KNWL-DIALO-FINETUNE \
        --module response \
        --spec-toks [SEP],[CTRL],[PAD] \
        --train-data ${TRAIN_PATH} \
        --test-data ${TEST_PATH} \
        --tokenizer-type GPT2BPETokenizer
