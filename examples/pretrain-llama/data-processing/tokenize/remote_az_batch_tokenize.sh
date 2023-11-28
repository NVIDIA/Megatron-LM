
SHARD_NAME=$1
SHARD_NAME_WITHOUT_EXT="${SHARD_NAME%.*}"
AZ_INPUT_FOLDER=$2
AZ_OUTPUT_FOLDER=$3
TOK_TYPE=$4
AZ_TOK_MODEL=$5
TOK_NAME=$(basename "$AZ_TOK_MODEL")
# echo $TOK_NAME >> out.txt
NUM_PROC=$6
LOG_INTERVAL=$7
TOKE_MODULE=$8
SAS_TOKEN=$9

echo "Running azcopy copy \"$AZ_TOK_MODEL\?$SAS_TOKEN\" \".\"" >> out.txt
azcopy copy "$AZ_TOK_MODEL?$SAS_TOKEN" "."

echo "Running azcopy copy \"$AZ_INPUT_FOLDER/$SHARD_NAME?$SAS_TOKEN\" \".\"" >> out.txt
azcopy copy "$AZ_INPUT_FOLDER/$SHARD_NAME?$SAS_TOKEN" "."

echo "Running $TOKE_MODULE tokenization ..."

if [ "$TOKE_MODULE" -eq 'megatron' ]; then

python tools/preprocess_data.py \
--input $SHARD_NAME \
--output-prefix $SHARD_NAME_WITHOUT_EXT \
--tokenizer-type $TOK_TYPE \
--tokenizer-model $TOK_NAME \
--workers $NUM_PROC \
--append-eod \
--log-interval $LOG_INTERVAL

elif [ "$ENG_LANG_PROB" -eq 'nemo' ]; then

python /workspace/nemo/scripts/nlp_language_modeling/preprocess_data_for_megatron.py \
--input $SHARD_NAME \
--output-prefix $SHARD_NAME_WITHOUT_EXT \
--tokenizer-library $TOK_TYPE \
--tokenizer-model $TOK_NAME \
--workers $NUM_PROC \
--vocab-file \ 
--append-eod \
--log-interval $LOG_INTERVAL

fi

echo "Uploading bin ..."
echo "azcopy copy $SHARD_NAME_WITHOUT_EXT\"_text_document.bin\"  \"$AZ_OUTPUT_FOLDER?$SAS_TOKEN\""  >> out.txt
azcopy copy $SHARD_NAME_WITHOUT_EXT"_text_document.bin"  "$AZ_OUTPUT_FOLDER?$SAS_TOKEN"

echo "Uploading idx ..."
echo "azcopy copy $SHARD_NAME_WITHOUT_EXT\"_text_document.idx\" \"$AZ_OUTPUT_FOLDER?$SAS_TOKEN\"" >> out.txt
azcopy copy $SHARD_NAME_WITHOUT_EXT"_text_document.idx" "$AZ_OUTPUT_FOLDER?$SAS_TOKEN"