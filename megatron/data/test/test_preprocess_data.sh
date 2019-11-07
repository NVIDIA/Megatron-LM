#!/bin/bash

python ../preprocess_data.py \
       --input test_samples.json \
       --vocab vocab.txt \
       --output-prefix test_samples \
       --workers 1 \
       --log-interval 2
