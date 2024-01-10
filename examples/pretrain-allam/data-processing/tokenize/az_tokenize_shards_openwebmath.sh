TOK_MODEL="https://allamllmuksstandard.blob.core.windows.net/llm-data/openwebmath/openwebmath_tokenize_by_v5tok.improved/tokenizer_v5_improved/ar_en.model"
TOK_TYPE="Llama2Tokenizer"
INPUT_FOLDER_PATH="https://allamllmuksstandard.blob.core.windows.net/llm-data/openwebmath/openwebmath_merged_shards/"
BIN_IDX_PATH="https://allamllmuksstandard.blob.core.windows.net/llm-data/openwebmath/openwebmath_tokenize_by_v5tok.improved/meglm_tok_v5_improved_bin_idx/"
AZJOB_FILE="examples/data-processing/az_templates/template_nemo_tokenize.yaml"
MATCHING_PREFIX_NAME=""

python examples/data-processing/tokenize_shards.py \
--input-folder-path "$INPUT_FOLDER_PATH" \
--bin-idx-folder-path "$BIN_IDX_PATH" \
--tokenizer-model "$TOK_MODEL" \
--tokenizer-type $TOK_TYPE \
--az-sample-yaml-job-file $AZJOB_FILE \
--az-configs "examples/configs/azure_login_configs.json" \
--num-proc 16 \
--compute-target 'azure'