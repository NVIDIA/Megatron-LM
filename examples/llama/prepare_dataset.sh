TMP_DIR="tmp"
mkdir -p $TMP_DIR
mkdir -p ${TMP_DIR}/data

DATA_PATH="${TMP_DIR}/data"
TOKENIZER_MODEL=${TMP_DIR}/tokenizer.model
DATASET="${DATASET:-wiki}" #wiki; bookcorpus

# Download the tokenizer model
if ! [ -f "$TOKENIZER_MODEL" ]; then
wget -O $TOKENIZER_MODEL https://huggingface.co/NousResearch/Llama-2-7b-chat-hf/resolve/main/tokenizer.model
fi

if [ "$DATASET" == "wiki" ]; then
    DATASET_PATH="${DATA_PATH}/wiki"
    echo "Downloading wikipedia-en dataset to ${DATASET_PATH}..."

    wget -P "$DATASET_PATH" https://a3s.fi/lumi-llm-scaling/wikipedia_20220301.en.train.jsonl
    wget -P "$DATASET_PATH" https://a3s.fi/lumi-llm-scaling/wikipedia_20220301.en.valid.jsonl
    wget -P "$DATASET_PATH" https://huggingface.co/gpt2/raw/main/vocab.json
    wget -P "$DATASET_PATH" https://huggingface.co/gpt2/raw/main/merges.txt

    for f in wikipedia_20220301.en.{train,valid}.jsonl; do
        python tools/preprocess_data.py \
            --input "$f" \
            --output "$DATASET_PATH/$(basename $f .jsonl)" \
            --tokenizer-type GPT2BPETokenizer \
            --vocab-file "$DATASET_PATH/vocab.json" \
            --merge-file "$DATASET_PATH/merges.txt" \
            --append-eod \
            --workers 128
    done
fi

if [ "$DATASET" == "bookcorpus" ]; then
    DATASET_PATH="${DATA_PATH}/bookcorpus"
    echo "Downloading bookcorpus dataset to ${DATASET_PATH}..."
    python3 examples/llama/prepare_bookcorpus_megatron_dataset.py --out-dir ${DATASET_PATH}
    python3 tools/preprocess_data.py --input ${DATASET_PATH}/bookcorpus_megatron.json  --tokenizer-type GPTSentencePieceTokenizer \
    --tokenizer-model ${TOKENIZER_MODEL} --output-prefix ${DATASET_PATH}/bookcorpus --workers `nproc` --split-sentences --partitions 2
fi

echo "Finishing data preparation!"
