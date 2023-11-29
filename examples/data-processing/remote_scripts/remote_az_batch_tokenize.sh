
SHARD_NAME=$1
SHARD_NAME_WITHOUT_EXT="${SHARD_NAME%.*}"
AZ_INPUT_FOLDER=$2
AZ_OUTPUT_FOLDER=$3
TOK_MODULE=$4
TOK_TYPE=$5
AZ_TOK_MODEL=$6
TOK_NAME=$(basename "$AZ_TOK_MODEL")
# echo $TOK_NAME >> out.txt
NUM_PROC=$7
LOG_INTERVAL=$8
SAS_TOKEN=$9


echo "Running azcopy copy \"$AZ_TOK_MODEL\?$SAS_TOKEN\" \".\"" >> out.txt
azcopy copy "$AZ_TOK_MODEL?$SAS_TOKEN" "."

echo "Running azcopy copy \"$AZ_INPUT_FOLDER/$SHARD_NAME?$SAS_TOKEN\" \".\"" >> out.txt
azcopy copy "$AZ_INPUT_FOLDER/$SHARD_NAME?$SAS_TOKEN" "."

echo "Running tokenization ..."
python tools/preprocess_data.py \
--input $SHARD_NAME \
--output-prefix $SHARD_NAME_WITHOUT_EXT \
--tokenizer-type $TOK_TYPE \
--tokenizer-model $TOK_NAME \
--workers $NUM_PROC \
--append-eod \
--log-interval $LOG_INTERVAL

echo "Uploading bin ..."
echo "azcopy copy $SHARD_NAME_WITHOUT_EXT\"_text_document.bin\"  \"$AZ_OUTPUT_FOLDER?$SAS_TOKEN\""  >> out.txt
azcopy copy $SHARD_NAME_WITHOUT_EXT"_text_document.bin"  "$AZ_OUTPUT_FOLDER?$SAS_TOKEN"

echo "Uploading idx ..."
echo "azcopy copy $SHARD_NAME_WITHOUT_EXT\"_text_document.idx\" \"$AZ_OUTPUT_FOLDER?$SAS_TOKEN\"" >> out.txt
azcopy copy $SHARD_NAME_WITHOUT_EXT"_text_document.idx" "$AZ_OUTPUT_FOLDER?$SAS_TOKEN"