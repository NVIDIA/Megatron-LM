#!/bin/bash

DATA_DIR=./dataset/glue

(
    python scripts/utils/download_glue.py \
    --data_dir $DATA_DIR \
    --task CoLA
) &
(
    python scripts/utils/download_glue.py \
    --data_dir $DATA_DIR \
    --task SST
) &
(
    python scripts/utils/download_glue.py \
    --data_dir $DATA_DIR \
    --task QQP
) &
(
    python scripts/utils/download_glue.py \
    --data_dir $DATA_DIR \
    --task STS
) &
(
    python scripts/utils/download_glue.py \
    --data_dir $DATA_DIR \
    --task MNLI
) &
(
    python scripts/utils/download_glue.py \
    --data_dir $DATA_DIR \
    --task QNLI
) &
(
    python scripts/utils/download_glue.py \
    --data_dir $DATA_DIR \
    --task RTE
) &

(
    python scripts/utils/download_glue.py \
    --data_dir $DATA_DIR \
    --task WNLI
) &
(
    python scripts/utils/download_glue.py \
    --data_dir $DATA_DIR \
    --task diagnostic
) &
(
    python scripts/utils/download_glue.py \
    --data_dir $DATA_DIR \
    --task MRPC \
    --path_to_mrpc $DATA_DIR
) &

wait