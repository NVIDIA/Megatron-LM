TOK_MODEL="../tokenizer_v5_improved/ar_en.model"
VOCAB_FILE="../tokenizer_v5_improved/ar_en.vocab"
TOK_TYPE="Llama2Tokenizer"
INPUT_FOLDER_PATH="../RAW_DATA_FOLDER/SlimPajama-627B/test/chunk1/example_holdout_1002.jsonl"
BIN_IDX_PATH="../RAW_DATA_FOLDER/SlimPajama-627B/tok_v5_improved_bin_idx"
mkdir -p $BIN_IDX_PATH
MATCHING_PREFIX_NAME="en_slim_pajama_"

python examples/data-processing/tokenize_shards.py \
--input-folder-path "$INPUT_FOLDER_PATH" \
--bin-idx-folder-path "$BIN_IDX_PATH" \
--tokenizer-module "nemo" \
--tokenizer-model "$TOK_MODEL" \
--vocab-file "$VOCAB_FILE" \
--tokenizer-type 'sentencepiece' \
--num-proc 16 \
--compute-target 'local'