set -e
source scripts/envs/env_vars.sh || true

AZURE_TOK_SAS_TOKEN="https://provisioningte0256624006.blob.core.windows.net/sbmaruf/vocab_expansion_assets/llama2_tokenizer/tokenizer.model?sp=r&st=2023-11-09T06:51:37Z&se=2025-12-31T14:51:37Z&spr=https&sv=2022-11-02&sr=b&sig=zVenodwlsVSX1YmuRpemG53fU59JB4aNCgL8Xvb3llo%3D"
AZURE_DATA_SAS_TOKEN="https://provisioningte0256624006.blob.core.windows.net/noufs/allam_data_2-1_splits/*?sp=rl&st=2023-11-09T06:57:45Z&se=2024-12-31T14:57:45Z&spr=https&sv=2022-11-02&sr=c&sig=%2FWH5uOcWYMMX1JbcB%2FMcLC5NqC1Qy6rRxCzbieOG8FI%3D"
DATA_PATH=$RAW_DATA_FOLDER'/allam_data_2-1_splits/'

OUTPUT_PATH="$DUMPED_FOLDER/allam_data_2-1_splits-indexed_data/"
BIN_IDX_FOLDER=$OUTPUT_PATH"/llama2_bin_idx"
TOK_PATH="$OUTPUT_PATH/tokenizer/tokenizer.model"
mkdir -p $OUTPUT_PATH
mkdir -p $BIN_IDX_FOLDER

if [ ! -d $DATA_PATH ]; then
    azcopy copy  $AZURE_DATA_SAS_TOKEN $DATA_PATH --overwrite=false --recursive
fi
if [ ! -f $TOK_PATH ]; then
    mkdir -p $OUTPUT_PATH'/tokenizer'
    azcopy copy  $AZURE_TOK_SAS_TOKEN $TOK_PATH --overwrite=false
fi

python examples/pretrain-llama/data-processing/multiprocess_runner.py \
 --glob-input-path "$DATA_PATH/*.jsonl" \
 --output-folder $BIN_IDX_FOLDER \
 --tokenizer-model $TOK_PATH \
 --tokenizer-type 'Llama2Tokenizer' \
 --per-file-workers 1 \
 --num-file-workers 168