AZ_LOGIN_CONF="examples/configs/azure_login_configs.json"
AZJOB_FILE="examples/data-processing/az_templates/template_nemo_tokenize.yaml"
TOK_TYPE="Llama2Tokenizer"

TOK_MODEL="https://allamllmuksstandard.blob.core.windows.net/vocab-expanded-training-data/allam_data_2-1_splits-llama2-indexed_data/tokenizer/tokenizer.model"
INPUT_FOLDER_PATH="https://allamllmuksstandard.blob.core.windows.net/llm-data/allam-v2-data/merged_shards/"
OUTPUT_BIN_IDX_PATH="https://allamllmuksstandard.blob.core.windows.net/vocab-expanded-training-data/allam_data_2-1_splits-llama2-indexed_data/llama2_bin_idx_with_trans/"

python examples/data-processing/tokenize_shards.py \
--input-folder-path "$INPUT_FOLDER_PATH" \
--bin-idx-folder-path "$OUTPUT_BIN_IDX_PATH" \
--tokenizer-model "$TOK_MODEL" \
--tokenizer-type $TOK_TYPE \
--az-configs $AZ_LOGIN_CONF \
--az-sample-yaml-job-file "$AZJOB_FILE" \
--num-proc 16 

TOK_MODEL="https://allamllmuksstandard.blob.core.windows.net/vocab-expanded-training-data/allam_data_2-1_splits-llama2-VE-indexed_data/tokenizer/tokenizer.model"
INPUT_FOLDER_PATH="https://allamllmuksstandard.blob.core.windows.net/llm-data/allam-v2-data/merged_shards/"
OUTPUT_BIN_IDX_PATH="https://allamllmuksstandard.blob.core.windows.net/vocab-expanded-training-data/allam_data_2-1_splits-llama2-VE-indexed_data/llama2_bin_idx_with_trans/"

python examples/data-processing/tokenize_shards.py \
--input-folder-path "$INPUT_FOLDER_PATH" \
--bin-idx-folder-path "$OUTPUT_BIN_IDX_PATH" \
--tokenizer-model "$TOK_MODEL" \
--tokenizer-type $TOK_TYPE \
--az-configs $AZ_LOGIN_CONF \
--az-sample-yaml-job-file "$AZJOB_FILE" \
--num-proc 16 
