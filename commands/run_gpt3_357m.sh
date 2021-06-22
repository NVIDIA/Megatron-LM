#!/bin/bash

#SBATCH -p interactive -A gpu_adlr_nlp -t 1:00:00 --nodes=1 --exclusive --mem=0 --overcommit --ntasks-per-node=16 --gres=gpu:16,gpfs:circe --dependency=singleton --job-name=adlr-nlp-largelm:gpt3-357m

NAME="gpt3-357m"
DIR=`pwd`
DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`
mkdir -p $DIR/logs

TENSORBOARD_DIR="${DIR}/tensorboard/${NAME}"
mkdir -p ${TENSORBOARD_DIR}

DATA_PATH=/gpfs/fs1/projects/gpu_adlr/datasets/nlp/gpt2_indexed_dataset/roberta_dataset/rn_owt_sto_wiki_dedup_shuf_cleaned_0.7_text_document

options=" \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --num-layers 24 \
    --hidden-size 1024 \
    --num-attention-heads 16 \
    --seq-length 2048 \
    --max-position-embeddings 2048 \
    --micro-batch-size 2 \
    --global-batch-size 256 \
    --rampup-batch-size 32 32 1953125 \
    --train-samples 192000000 \
    --lr-decay-samples 166400000 \
    --lr-warmup-samples 162761 \
    --lr 3.0e-4 \
    --min-lr 3.0e-5 \
    --lr-decay-style cosine \
    --log-interval 100 \
    --eval-iters 50 \
    --eval-interval 2000 \
    --data-path ${DATA_PATH} \
    --vocab-file /gpfs/fs1/projects/gpu_adlr/datasets/nlp/gpt2_indexed_dataset/bpe/gpt2-vocab.json \
    --merge-file /gpfs/fs1/projects/gpu_adlr/datasets/nlp/gpt2_indexed_dataset/bpe/gpt2-merges.txt \
    --save-interval 10000 \
    --exit-interval 100 \
    --save /gpfs/fs1/projects/gpu_adlr/datasets/zihanl/checkpoints/gpt3/${NAME} \
    --load /gpfs/fs1/projects/gpu_adlr/datasets/zihanl/checkpoints/gpt3/${NAME} \
    --split 98,2,0 \
    --clip-grad 1.0 \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --init-method-std 0.02 \
    --log-params-norm \
    --log-num-zeros-in-grad \
    --fp16 \
    --DDP-impl torch \
    --tensorboard-dir ${TENSORBOARD_DIR} \
    --checkpoint-activations "

run_cmd="python ${DIR}/pretrain_gpt.py ${options}"

srun -l \
     --container-image "gitlab-master.nvidia.com/adlr/megatron-lm/pytorch-nlp-retriever-faiss:20.12-py3-devel" \
     --container-mounts "/gpfs/fs1/projects/gpu_adlr/datasets:/gpfs/fs1/projects/gpu_adlr/datasets,/home/zihanl:/home/zihanl" \
     --output=$DIR/logs/%x_%j_$DATETIME.log sh -c "${run_cmd}"

set +x

