#!/bin/bash

python tools/preprocess_data.py \
    --input /workspace/dataset/wikitext103/text/single.json \
    --output-prefix output_prefix/my-gpt2-cased \
    --vocab-file vocab/gpt2-vocab.json \
    --dataset-impl mmap \
    --tokenizer-type GPT2BPETokenizer \
    --merge-file vocab/gpt2-merges.txt \
    --append-eod \
    --workers 8
    