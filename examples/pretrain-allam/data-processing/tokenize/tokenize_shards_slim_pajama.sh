TOK_MODEL=""
TOK_TYPE="Llama2Tokenizer"
INPUT_FOLDER_PATH="../RAW_DATA_FOLDER/SlimPajama-627B/export/test.jsonl"
BIN_IDX_PATH="../RAW_DATA_FOLDER/SlimPajama-627B/tok_v5_improved_bin_idx"
mkdir $BIN_IDX_PATH
MATCHING_PREFIX_NAME="en_dolma_"

python examples/data-processing/tokenize_shards.py \
--input-folder-path "$INPUT_FOLDER_PATH" \
--bin-idx-folder-path "$BIN_IDX_PATH" \
--tokenizer-module "nemo" \
--tokenizer-model "$TOK_MODEL" \
--vocab-file "" \
--tokenizer-type 'sentencepiece' \
--num-proc 16 \
--compute-target 'local'