#!/bin/bash

python tools/preprocess_data.py \
    --input /workspace/dataset/wikitext103/text/single.json \
    --output-prefix output_prefix/my-t5-uncased \
    --vocab-file vocab/bert-large-uncased-vocab.txt \
    --tokenizer-type BertWordPieceLowerCase \
    --split-sentences \
    --workers 8
    