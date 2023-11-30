TOK_MODEL="https://allamllmstorageuks.blob.core.windows.net/tokenizer/tokenizer_v5_improved/tokenizer.model"
TOK_TYPE="sentencepiece"
INPUT_FOLDER_PATH="https://allamllmuksstandard.blob.core.windows.net/vocab-expanded-training-data/the_pile"
BIN_IDX_PATH="https://allamllmuksstandard.blob.core.windows.net/vocab-expanded-training-data/the_pile_bin_idx"
MATCHING_PREFIX_NAME="en_pile_"

python examples/data-processing/tokenize_shards.py \
--input-folder-path "$INPUT_FOLDER_PATH" \
--bin-idx-folder-path "$BIN_IDX_PATH" \
--tokenizer-module "nemo" \
--tokenizer-model "$TOK_MODEL" \
--tokenizer-type 'sentencepiece' \
--az-configs "examples/configs/azure_login_configs.json" \
--num-proc 16 \
--compute-target 'azure' \
--az-sample-yaml-job-file 'examples/data-processing/az_templates/template_nemo_tokenize.yaml'

