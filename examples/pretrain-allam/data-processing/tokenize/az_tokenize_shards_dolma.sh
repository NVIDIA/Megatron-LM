TOK_MODEL="https://allamllmuksstandard.blob.core.windows.net/vocab-expanded-training-data/tokenizer_v5_improved/ar_en.model"
TOK_TYPE="sentencepiece"
INPUT_FOLDER_PATH="https://allamllmuksstandard.blob.core.windows.net/vocab-expanded-training-data/dolma/merged_shards"
BIN_IDX_PATH="https://allamllmuksstandard.blob.core.windows.net/vocab-expanded-training-data/dolma/tok_v5_improved_bin_idx"
AZJOB_FILE="examples/configs/azure_login_configs.json"
MATCHING_PREFIX_NAME="en_dolma_"

python examples/data-processing/tokenize_shards.py \
--input-folder-path "$INPUT_FOLDER_PATH" \
--bin-idx-folder-path "$BIN_IDX_PATH" \
--tokenizer-module "nemo" \
--tokenizer-model "$TOK_MODEL" \
--tokenizer-type $TOK_TYPE \
--az-configs  $AZJOB_FILE \
--num-proc 16 \
--compute-target 'azure'