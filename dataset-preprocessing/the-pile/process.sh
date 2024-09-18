#!/bin/bash
#SBATCH -J data_process
#SBATCH -p general


export HF_DATASETS_OFFLINE=0
export HF_DATASETS_CACHE="/N/scratch/jindjia/.cache/huggingface/datasets"
MASTER_PORT=6000
NODE_RANK=$SLURM_NODEID
NNODES=1
PROC_PER_NODE=64

cd /N/scratch/jindjia/thepile

echo "data process start"
date

srun -n $PROC_PER_NODE python Megatron-LM/tools/preprocess_data_dist.py \
       --input pile.jsonl \
       --split train \
       --columns text \
       --output-prefix pile \
       --vocab-file vocab.json \
       --merge-file merges.txt \
       --dataset-impl mmap \
       --tokenizer-type GPT2BPETokenizer \
       --append-eod \
       --torch-backend mpi

echo "data process finish"
date