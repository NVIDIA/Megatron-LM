set -e
source scripts/envs/env_vars.sh || true

AZURE_TOK_SAS_TOKEN="https://provisioningte0256624006.blob.core.windows.net/sbmaruf/vocab_expansion_assets/llama2_tokenizer/tokenizer.model?sp=r&st=2023-11-09T06:51:37Z&se=2025-12-31T14:51:37Z&spr=https&sv=2022-11-02&sr=b&sig=zVenodwlsVSX1YmuRpemG53fU59JB4aNCgL8Xvb3llo%3D"
AZURE_DATA_SAS_TOKEN="https://provisioningte0256624006.blob.core.windows.net/noufs/allam_data_2-1_splits/*?sp=rl&st=2023-11-09T06:57:45Z&se=2024-12-31T14:57:45Z&spr=https&sv=2022-11-02&sr=c&sig=%2FWH5uOcWYMMX1JbcB%2FMcLC5NqC1Qy6rRxCzbieOG8FI%3D"
DATA_PATH=$RAW_DATA_FOLDER'/allam_data_2-1_splits'
# DATA_PATH=$RAW_DATA_FOLDER'/temp'

OUTPUT_PATH="$DUMPED_FOLDER/allam_data_2-1_splits-indexed_data/"
TOK_PATH="$OUTPUT_PATH/tokenizer/tokenizer.model"
mkdir -p $OUTPUT_PATH

if [ ! -d $DATA_PATH ]; then
    azcopy copy  $AZURE_DATA_SAS_TOKEN $DATA_PATH --overwrite=false --recursive
fi
if [ ! -f $TOK_PATH ]; then
    mkdir -p $OUTPUT_PATH'/tokenizer'
    azcopy copy  $AZURE_TOK_SAS_TOKEN $TOK_PATH --overwrite=false
fi

for _inp in $DATA_PATH/*.jsonl;
do
    # echo 'Indexing '$_inp'...'
    FILENAME=${_inp##*/}
    FILENAME_WITHOUT_EXT=${FILENAME%.*}
    TEMP_FOLDER=$DUMPED_FOLDER'/temp_bin_idx/temp_'$FILENAME_WITHOUT_EXT
    INDEX_FOLDER=$OUTPUT_PATH'/bin_idx/'$FILENAME_WITHOUT_EXT
    if [ -d $INDEX_FOLDER ]; then
        echo $INDEX_FOLDER
        continue
    fi
    mkdir -p $TEMP_FOLDER
    mkdir -p $INDEX_FOLDER
    
    echo 'Copying file from '$_inp' to '$TEMP_FOLDER
    cp $_inp $TEMP_FOLDER

    python tools/preprocess_data.py \
    --input $TEMP_FOLDER/$FILENAME \
    --json-keys 'text' \
    --partitions 16 \
    --tokenizer-type 'Llama2Tokenizer' \
    --tokenizer-model $TOK_PATH \
    --workers 128 \
    --output-prefix $TEMP_FOLDER/$FILENAME_WITHOUT_EXT \
    --keep-sequential-samples
    
    mv $TEMP_FOLDER/$FILENAME_WITHOUT_EXT'_text_document.bin' $INDEX_FOLDER
    mv $TEMP_FOLDER/$FILENAME_WITHOUT_EXT'_text_document.idx' $INDEX_FOLDER
    rm -rf $TEMP_FOLDER/

done