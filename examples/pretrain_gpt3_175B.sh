#!/bin/bash


#SBATCH <SLURM OPTIONS> --nodes=128 --exclusive --ntasks-per-node=8 --job-name=megatron_gpt3_175b


DIR=`pwd`
DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`
mkdir -p $DIR/logs


DATASET_1="<PATH TO THE FIRST DATASET>"
DATASET_2="<PATH TO THE SECOND DATASET>"
DATASET_3="<PATH TO THE THIRD DATASET>"
DATASET="0.2 ${DATASET_1} 0.3 ${DATASET_2} 0.5 ${DATASET_3}"


options=" \
	--tensor-model-parallel-size 8 \
	--pipeline-model-parallel-size 16 \
        --num-layers 96 \
        --hidden-size 12288 \
        --num-attention-heads 96 \
        --seq-length 2048 \
        --max-position-embeddings 2048 \
	--micro-batch-size 1 \
	--global-batch-size 1536 \
	--rampup-batch-size 16 16 5859375 \
	--train-samples 146484375 \
       	--lr-decay-samples 126953125 \
        --lr-warmup-samples 183105 \
        --lr 6.0e-5 \
	--min-lr 6.0e-6 \
        --lr-decay-style cosine \
        --log-interval 10 \
        --eval-iters 40 \
        --eval-interval 1000 \
	--data-path ${DATASET} \
	--vocab-file <PATH TO gpt-vocab.json> \
	--merge-file <PATH TO gpt-merges.txt> \
	--save-interval 1000 \
	--save <PATH TO CHECKPOINTS DIRECTORY> \
	--load <PATH TO CHECKPOINTS DIRECTORY> \
        --split 98,2,0 \
        --clip-grad 1.0 \
	--weight-decay 0.1 \
	--adam-beta1 0.9 \
	--adam-beta2 0.95 \
	--init-method-std 0.006 \
	--tensorboard-dir <TENSORBOARD DIRECTORY> \
        --fp16 \
	--checkpoint-activations "


run_cmd="python -u ${DIR}/pretrain_gpt.py $@ ${options}"


srun -l \
     --container-image "nvcr.io/nvidia/pytorch:20.12-py3" \
     --container-mounts "<DIRECTORIES TO MOUNT>" \
     --output=$DIR/logs/%x_%j_$DATETIME.log sh -c "${run_cmd}"


set +x

