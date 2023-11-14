#!/bin/bash
set -e

AR_FILE="allam_data_2-1_splits-llama2-indexed_data/llama2_bin_idx/ar_encyclopedias_split_00_text_document"
AR_BIN="${AR_FILE}.bin"
AR_IDX="${AR_FILE}.idx"

EN_FILE="allam_data_2-1_splits-llama2-indexed_data/llama2_bin_idx/en_books_books_split_02_text_document"
EN_BIN="${EN_FILE}.bin"
EN_IDX="${EN_FILE}.idx"

OUTPUT_DATA="outputs/data"
TOK_PATH="outputs/tokenizer"
TOK_FILE="${TOK_PATH}/tokenizer.model"
FINAL_OUTPUT="outputs/results"

if [ ! -d $OUTPUT_DATA ]; then
    mkdir -p ${OUTPUT_DATA}
fi

if [ ! -d $TOK_PATH ]; then
    mkdir -p ${TOK_PATH}
fi

if [ ! -d $FINAL_OUTPUT ]; then
    mkdir -p ${FINAL_OUTPUT}
fi

if [ ! -f $TOK_FILE ]; then
    AZURE_TOK="allam_data_2-1_splits-llama2-indexed_data/tokenizer/tokenizer.model"
    AZURE_PATH="https://allamllmuksstandard.blob.core.windows.net/vocab-expanded-training-data/${AZURE_TOK}?sp=racwdl&st=2023-11-13T10:57:16Z&se=2025-04-17T18:57:16Z&spr=https&sv=2022-11-02&sr=c&sig=Gg2nHVydrJozaXGrcst0iF3mVgKkhRWO05Mt%2Bfp%2F69Y%3D"
    azcopy copy "${AZURE_PATH}" "${TOK_FILE}"
fi

FILE_PATHS=($AR_BIN $AR_IDX $EN_BIN $EN_IDX)

for FILE_PATH in "${FILE_PATHS[@]}"
do
    OUTPUT_PATH="${OUTPUT_DATA}/${FILE_PATH}"
    if [ ! -f $OUTPUT_PATH ]; then
        AZURE_PATH="https://allamllmuksstandard.blob.core.windows.net/vocab-expanded-training-data/${FILE_PATH}?sp=racwdl&st=2023-11-13T10:57:16Z&se=2025-04-17T18:57:16Z&spr=https&sv=2022-11-02&sr=c&sig=Gg2nHVydrJozaXGrcst0iF3mVgKkhRWO05Mt%2Bfp%2F69Y%3D"
        azcopy copy "${AZURE_PATH}" "${OUTPUT_PATH}"
    fi
done

python detokenize.py --tokenizer_model "${TOK_FILE}" --file_path "${OUTPUT_DATA}/${AR_FILE}" --output "${FINAL_OUTPUT}/ar_output.jsonl"
python detokenize.py --tokenizer_model "${TOK_FILE}" --file_path "${OUTPUT_DATA}/${EN_FILE}" --output "${FINAL_OUTPUT}/en_output.jsonl"