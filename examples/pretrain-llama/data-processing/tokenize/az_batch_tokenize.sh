# TOK_MODEL="https://allamllmuksstandard.blob.core.windows.net/vocab-expanded-training-data/allam_data_2-1_splits-llama2-indexed_data/tokenizer/tokenizer.model"
# TOK_TYPE="Llama2Tokenizer"
# INPUT_FOLDER_PATH="https://allamllmuksstandard.blob.core.windows.net/vocab-expanded-training-data/processed/merged_shards"
# BIN_IDX_PATH="https://allamllmuksstandard.blob.core.windows.net/vocab-expanded-training-data/allam_data_2-1_splits-llama2-indexed_data/llama2_bin_idx_trans/"
# AZJOB_FILE="examples/pretrain-llama/data-processing/tokenize/tokenize.yaml"
# MATCHING_PREFIX_NAME="ar_translated_"

# python examples/pretrain-llama/data-processing/tokenize/az_batch_tokenize.py \
# --input-folder-path "$INPUT_FOLDER_PATH" \
# --bin-idx-folder-path "$BIN_IDX_PATH" \
# --tokenizer-model "$TOK_MODEL" \
# --tokenizer-type $TOK_TYPE \
# --az-configs "examples/pretrain-llama/data-processing/tokenize/azure_login_configs.json" \
# --num-proc 16
