TMP_DIR="tmp"
mkdir -p $TMP_DIR
mkdir -p ${TMP_DIR}/data

DATA_PATH="${TMP_DIR}/data"
TOKENIZER_MODEL=${TMP_DIR}/tokenizer.model

# Download the tokenizer model
if ! [ -f "$TOKENIZER_MODEL" ]; then
wget -O $TOKENIZER_MODEL https://huggingface.co/NousResearch/Llama-2-7b-chat-hf/resolve/main/tokenizer.model
fi

python3 prepare_bookcorpus_megatron_dataset.py --out-dir ${DATA_PATH}
python3 tools/preprocess_data.py --input ${DATA_PATH}/bookcorpus_megatron.json  --tokenizer-type GPTSentencePieceTokenizer \
--tokenizer-model ${TOKENIZER_MODEL} --output-prefix ${DATA_PATH}/bookcorpus --workers `nproc` --split-sentences

python3 tools/preprocess_data.py --input ${DATA_PATH}/bookcorpus_megatron.json  --tokenizer-type GPTSentencePieceTokenizer \
--tokenizer-model ${TOKENIZER_MODEL} --output-prefix ${DATA_PATH}/bookcorpus --workers `nproc` --split-sentences
