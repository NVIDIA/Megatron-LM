#!/bin/bash
#SBATCH -J data_process

mamba init
source ~/.bashrc
mamba activate megatron
set -x

cd PATH # TODO cd to saving path folder
echo "data process start"
date

python Megatron-LM/tools/preprocess_data.py \ # TODO set to your megatron-LM path
       --input /N/slate/jindjia/LLM/data/wikitext-2-v1/wikitext-2-v1.json \
       --output-prefix wikitext-2-v1 \
       --vocab-file vocab.json \
       --dataset-impl mmap \
       --tokenizer-type GPT2BPETokenizer \
       --merge-file merges.txt \
       --json-keys text \
       --append-eod \
       --workers 4 \

echo "data process finish"
date