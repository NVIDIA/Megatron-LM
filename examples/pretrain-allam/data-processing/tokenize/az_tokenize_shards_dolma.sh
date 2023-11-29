TOK_MODEL="azureml://subscriptions/c7209a17-0d9f-41df-8e45-e0172343698d/resourcegroups/llm-test/workspaces/Provisioning-Test/datastores/tokenizer/paths/tokenizer_v5_improved/tokenizer.model"
TOK_TYPE="Llama2Tokenizer"
INPUT_FOLDER_PATH="https://allamllmuksstandard.blob.core.windows.net/vocab-expanded-training-data/dolma/merged_shards"
BIN_IDX_PATH="https://allamllmuksstandard.blob.core.windows.net/vocab-expanded-training-data/dolma/tok_v5_improved_bin_idx"
AZJOB_FILE="examples/data-processing/tokenize/tokenize.yaml"
MATCHING_PREFIX_NAME="en_dolma_"

python examples/data-processing/tokenize_shards.py \
--input-folder-path "$INPUT_FOLDER_PATH" \
--bin-idx-folder-path "$BIN_IDX_PATH" \
--tokenizer-module "nemo" \
--tokenizer-model "$TOK_MODEL" \
--tokenizer-type 'sentencepiece' \
--az-configs "examples/configs/azure_login_configs.json" \
--num-proc 16 \
--compute-target 'local'