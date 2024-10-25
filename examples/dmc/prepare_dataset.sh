#!/bin/bash

TOKENIZER_MODEL_PATH=$1

mkdir -p datasets
HF_HOME=$(pwd)"/datasets/hf_cache" python examples/dmc/download_dataset.py

python -u tools/preprocess_data.py \
   --input=datasets/dmc_demo/hf_wiki_20231101_en_train.jsonl \
   --json-keys=text \
   --tokenizer-type=Llama2Tokenizer \
   --tokenizer-model=$TOKENIZER_MODEL_PATH \
   --output-prefix=datasets/dmc_demo/llama2_tokenized/hf_wiki_20231101_en_train \
   --append-eod \
   --workers 96 \
   --partitions 1 \
   --log-interval 1000

echo "Finished dataset preprocessing"

