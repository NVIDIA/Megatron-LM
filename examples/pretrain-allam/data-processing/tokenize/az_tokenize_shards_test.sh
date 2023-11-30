TOK_MODEL="https://allamllmuksstandard.blob.core.windows.net/vocab-expanded-training-data/tokenizer_v5_improved/ar_en.model"
VOCAB_FILE="https://allamllmuksstandard.blob.core.windows.net/vocab-expanded-training-data/tokenizer_v5_improved/ar_en.vocab"
TOK_TYPE="sentencepiece"
INPUT_FOLDER_PATH="https://allamllmuksstandard.blob.core.windows.net/vocab-expanded-training-data/test_dataset"
BIN_IDX_PATH="https://allamllmuksstandard.blob.core.windows.net/vocab-expanded-training-data/test_dataset/nemo_bin_idx"
MATCHING_PREFIX_NAME="en_test_"

python examples/data-processing/tokenize_shards.py \
--input-folder-path "$INPUT_FOLDER_PATH" \
--bin-idx-folder-path "$BIN_IDX_PATH" \
--tokenizer-module "nemo" \
--tokenizer-model "$TOK_MODEL" \
--tokenizer-type 'sentencepiece' \
--vocab-file $VOCAB_FILE \
--az-configs "examples/configs/azure_login_configs.json" \
--num-proc 16 \
--compute-target 'azure' \
--az-sample-yaml-job-file 'examples/data-processing/az_templates/template_nemo_tokenize.yaml'


TOK_MODEL="https://allamllmuksstandard.blob.core.windows.net/vocab-expanded-training-data/tokenizer_v5_improved/ar_en.model"
VOCAB_FILE="https://allamllmuksstandard.blob.core.windows.net/vocab-expanded-training-data/tokenizer_v5_improved/ar_en.vocab"
TOK_TYPE="Llama2Tokenizer"
INPUT_FOLDER_PATH="https://allamllmuksstandard.blob.core.windows.net/vocab-expanded-training-data/test_dataset"
BIN_IDX_PATH="https://allamllmuksstandard.blob.core.windows.net/vocab-expanded-training-data/test_dataset/megatron_bin_idx/"
MATCHING_PREFIX_NAME="en_test_"

python examples/data-processing/tokenize_shards.py \
--input-folder-path "$INPUT_FOLDER_PATH" \
--bin-idx-folder-path "$BIN_IDX_PATH" \
--tokenizer-module "megatron" \
--tokenizer-model "$TOK_MODEL" \
--tokenizer-type 'Llama2Tokenizer' \
--vocab-file $VOCAB_FILE \
--az-configs "examples/configs/azure_login_configs.json" \
--num-proc 16 \
--compute-target 'azure' \
--az-sample-yaml-job-file 'examples/data-processing/az_templates/template_nemo_tokenize.yaml'