#!/bin/bash

# 引数から NODE_RANK の数を取得
NUM_RANKS=$1

# ベースとなる YAML テンプレート
TEMPLATE="./generator/job-template.yaml"

rm -rf jobs
mkdir -p jobs

# NODE_RANK ごとに YAML ファイルを生成
for (( i=1; i<$NUM_RANKS; i++ ))
do
  # 出力ファイル名
  OUTPUT_FILE="jobs-${i}.yaml"

  # テンプレートファイルから新しい YAML ファイルを作成し、NODE_RANK を置換
  sed -e "s/value: \"__NUM_RANK_VALUE__\"/value: \"${i}\"/" \
      -e "s/__DEPLOYMENT_NAME__/megatron-lm-worker-${i}/" $TEMPLATE > jobs/$OUTPUT_FILE

  echo "Generated $OUTPUT_FILE with NODE_RANK=${i}"
done

CONFIG_FILE="./resource/config.yaml"
sed -i "s/NNODES: \"[0-9]*\"/NNODES: \"${NUM_RANKS}\"/" $CONFIG_FILE
