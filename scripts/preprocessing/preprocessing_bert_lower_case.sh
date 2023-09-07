#!/bin/bash

python tools/preprocess_data.py \
    --input /workspace/dataset/wikitext103/text/single.json \
    --output-prefix output_prefix/my-bert-cased \
    --vocab-file vocab/bert-large-cased-vocab.txt \
    --tokenizer-type BertWordPieceCase \
    --split-sentences \
    --workers 8
    