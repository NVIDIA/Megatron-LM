#!/bin/bash

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
        --tensor-model-parallel-size 1 \
        --pipeline-model-parallel-size 1 \
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
        --log-interval 100 \
        --vocab-file ${VOCAB_PATH} \
        --merge-file ${MERGE_PATH} \
        --save-interval 10000 \
        --save ${OUTPUT_MODEL_PATH} \
        --pretrained-checkpoint ${CHECKPOINT_PATH} \
        --clip-grad 1.0 \
        --weight-decay 0.1 \
        --adam-beta1 0.9 \
        --adam-beta2 0.95 \
        --init-method-std 0.02 \
        --log-params-norm \
        --log-num-zeros-in-grad \
        --fp16 \
        --DDP-impl torch \
        --checkpoint-activations \
        --epochs 3 \
        --task KNWL-DIALO-FINETUNE \
        --module response \
        --spec-toks [SEP],[CTRL],[PAD] \
        --train-data-path ${TRAIN_PATH} \
        --test-data-path ${TEST_PATH} \
        --max-seq-len 1024 \
        --tokenizer-type GPT2BPETokenizer
