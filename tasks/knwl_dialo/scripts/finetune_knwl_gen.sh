#!/bin/bash

# Finetune a pretrained language model to generate the context-relevant knowledge
# The input is the dialogue context, and output is the relevant knowledge
# The size of the pretrained language model is 357M

WORLD_SIZE=8

DISTRIBUTED_ARGS="--nproc_per_node $WORLD_SIZE \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr localhost \
                  --master_port 6000"

CHECKPOINT_PATH=<Specify path for the language model>
OUTPUT_MODEL_PATH=<Specify path for the saved model>
VOCAB_PATH=<Specify path for the vocab file>
MERGE_PATH=<Specify path for the merge file>
TRAIN_PATH=<Specify path for the training dataset>
TEST_PATH=<Specify path for the test dataset>

python -m torch.distributed.launch $DISTRIBUTED_ARGS ./tasks/main.py \
        --num-layers 24 \
        --hidden-size 1024 \
        --num-attention-heads 16 \
        --seq-length 2048 \
        --max-position-embeddings 2048 \
        --micro-batch-size 4 \
        --global-batch-size 64 \
        --train-samples 61000 \
        --lr-decay-samples 50000 \
        --lr-warmup-samples 5000 \
        --lr 1.5e-5 \
        --min-lr 1.0e-5 \
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
        --epochs 4 \
        --task KNWL-DIALO-FINETUNE \
        --module knowledge \
        --spec-toks [SEP],[CTRL],[PAD] \
        --train-data ${TRAIN_PATH} \
        --test-data ${TEST_PATH} \
        --tokenizer-type GPT2BPETokenizer
