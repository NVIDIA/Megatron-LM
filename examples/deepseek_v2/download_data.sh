#!/bin/bash
EXPERIMENT_DIR="experiment"
export EXPERIMENT_DIR
mkdir -p $EXPERIMENT_DIR

DATA_DIR="${EXPERIMENT_DIR}/data/"
export DATA_DIR
ls experiment
if ! [ -d "${DATA_DIR}" ]; then
  mkdir -p $DATA_DIR
  cd $DATA_DIR
else
  cd $DATA_DIR
fi


TRAIN_SET_DIR="deepseekv2-train-datasets"
if ! [ -d "${TRAIN_SET_DIR}" ]; then
  mkdir -p $TRAIN_SET_DIR
  cd $TRAIN_SET_DIR
else
  cd $TRAIN_SET_DIR
fi

files=("mmap_deepseekv2_datasets_text_document.bin" "mmap_deepseekv2_datasets_text_document.idx" "SlimPajama.json" "alpaca_zh-train.json" "alpaca_zh-valid.json")
for file in "${files[@]}"; do
    if [ ! -e "$file" ]; then
    wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/deepseek-datasets/$file
  else
    echo "File exists."
    fi
done
