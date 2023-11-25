AZ_SUBS="c7209a17-0d9f-41df-8e45-e0172343698d"
AZ_RESOURCE_GROUP="LLM-Test"
AZ_WORKSPACE="Provisioning-Test"
SAS_TOKEN="sp=racwdli&st=2023-11-25T11:38:52Z&se=2023-11-25T19:38:52Z&spr=https&sv=2022-11-02&sr=c&sig=yZQU8wwWZSrZARG8vftWZqfdnTr2HzFCmgKXL05kQow%3D"
AZJOB_FILE="examples/pretrain-llama/data-processing/merge_shard/template_merge_shard.yaml"
TOK_MODEL="https://allamllmuksstandard.blob.core.windows.net/vocab-expanded-training-data/allam_data_2-1_splits-llama2-indexed_data/tokenizer/tokenizer.model"
TOK_TYPE="Llama2Tokenizer"
INPUT_FOLDER_PATH="https://allamllmuksstandard.blob.core.windows.net/vocab-expanded-training-data/processed/merged_shards"
BIN_IDX_PATH="https://allamllmuksstandard.blob.core.windows.net/vocab-expanded-training-data/allam_data_2-1_splits-llama2-indexed_data/llama2_bin_idx_trans/"
AZJOB_FILE="examples/pretrain-llama/data-processing/tokenize/tokenize.yaml"
MATCHING_PREFIX_NAME="ar_translated_"

python examples/pretrain-llama/data-processing/tokenize/az_batch_tokenize.py \
--az-subscription "$AZ_SUBS" \
--az-resource-group "$AZ_RESOURCE_GROUP" \
--az-workspace-name "$AZ_WORKSPACE" \
--az-blob-input-folder-path "$INPUT_FOLDER_PATH" \
--az-blob-bin-idx-folder-path "$BIN_IDX_PATH" \
--az-tokenizer-model "$TOK_MODEL" \
--tokenizer-type $TOK_TYPE \
--az-sas-token "$SAS_TOKEN" \
--sample-yaml-job-file "$AZJOB_FILE" \
--az-num-proc 16
