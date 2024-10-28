#! /bin/bash

# Pre-trains ViT based image inpainting model

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_IB_SL=1

# Training and validation paths should each point to a folder where each
# sub-folder contains a collection of images in jpg or png format
# e.g. If using imagenet, one train image might be, train_data/n01688243/n01688243_11301.JPEG
DATA_PATH_TRAIN=<Specify train data path>
DATA_PATH_VAL=<Specify validation data path>

CHECKPOINT_PATH=<Specify path>

INPAINT_ARGS="
        --vision-pretraining-type inpaint \
   	   --tensor-model-parallel-size 1 \
        --num-layers 12 \
        --hidden-size 768 \
        --num-attention-heads 12 \
        --patch-dim 4 \
        --seq-length 3136 \
        --max-position-embeddings 3136 \
        --img-h 224 \
        --img-w 224 \
        --mask-factor 1.0 \
        --fp16 \
        --train-iters 750000 \
        --lr-decay-style cosine \
        --micro-batch-size 4 \
        --global-batch-size 1024 \
        --lr 0.0005 \
        --min-lr 0.00001 \
        --attention-dropout 0.0 \
        --weight-decay 0.05 \
        --lr-warmup-iters 12500 \
        --clip-grad 1.0 \
        --no-gradient-accumulation-fusion \
        --num-workers 4 \
        --DDP-impl torch "

DATA_ARGS="
     --tokenizer-type NullTokenizer \
     --vocab-size 0 \
     --data-path $DATA_PATH_TRAIN $DATA_PATH_VAL \
     --no-data-sharding \
     --split 949,50,1 \
"

OUTPUT_ARG="
     --log-interval 32 \
     --save-interval 10000 \
     --eval-interval 2500 \
     --eval-iters 100 \
     --tensorboard-dir ${CHECKPOINT_PATH} \
"

torchrun pretrain_vision_inpaint.py \
     $INPAINT_ARGS \
     $DATA_ARGS \
     $OUTPUT_ARGS \
     --save $CHECKPOINT_PATH \
     --load $CHECKPOINT_PATH

