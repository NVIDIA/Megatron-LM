#!/bin/bash

# Check input
if [ -z ${TOKENIZER_MODEL+x} ]; then
    echo "TOKENIZER_MODEL required"
    exit 1
fi

# Setup variables
DATA_DIR="${DATA_DIR:-data}"
DATASET="${DATASET:-bookcorpus}" # one of wiki,bookcorpus,fineweb
DATASET_PATH="${DATA_DIR}/data.jsonl"
TOKENIZER_TYPE="${TOKENIZER_TYPE:-HuggingFaceTokenizer}"

# Create directories
mkdir -p ${DATA_DIR}

# Download and preprocess dataset
echo "Downloading '${DATASET}' dataset to '${DATASET_PATH}'..."
if [ "$DATASET" == "wiki" ]; then
    wget -nc -O "$DATASET_PATH" https://a3s.fi/lumi-llm-scaling/wikipedia_20220301.en.train.jsonl
elif [ "$DATASET" == "fineweb" ]; then
    python examples/llama/fineweb/download.py --out-dir ${DATA_DIR}
    if [ $? -ne 0 ]; then
        echo "Download failed"
        exit 1
    fi
elif [ "$DATASET" == "bookcorpus" ]; then
    python3 examples/llama/bookcorpus/download.py --out-dir ${DATA_DIR}
    if [ $? -ne 0 ]; then
        echo "Download failed"
        exit 1
    fi
else
    echo "Invalid DATASET=${DATASET}, only 'wiki', 'bookcorpus' and 'fineweb' supported"
    exit 1
fi

# Preprocess dataset
echo "Preprocessing '${DATASET}' dataset using '${TOKENIZER_MODEL}' tokenizer..."
python3 tools/preprocess_data.py \
    --input "${DATASET_PATH}"  \
    --tokenizer-type $TOKENIZER_TYPE \
    --tokenizer-model "${TOKENIZER_MODEL}" \
    --output-prefix "${DATA_DIR}/data" \
    --workers `nproc`
