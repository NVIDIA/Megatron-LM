
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
SAS_TOKEN=$8

# # echo "azcopy copy \"$AZ_TOK_MODEL\?$SAS_TOKEN\" \".\"" >> out.txt
# azcopy copy "$AZ_TOK_MODEL?$SAS_TOKEN" "."

# # echo "azcopy copy \"$AZ_INPUT_FOLDER/$SHARD_NAME?$SAS_TOKEN\" \".\"" >> out.txt
# azcopy copy "$AZ_INPUT_FOLDER/$SHARD_NAME?$SAS_TOKEN" "."

# python tools/preprocess_data.py \
# --input $SHARD_NAME \
# --output-prefix $SHARD_NAME_WITHOUT_EXT \
# --tokenizer-type $TOK_TYPE \
# --tokenizer-model $TOK_NAME \
# --workers $NUM_PROC \
# --append-eod \
# --log-interval $LOG_INTERVAL

# echo "azcopy copy $SHARD_NAME_WITHOUT_EXT\"_text_document.bin\"  \"$AZ_OUTPUT_FOLDER?$SAS_TOKEN\""  >> out.txt
azcopy copy $SHARD_NAME_WITHOUT_EXT"_text_document.bin"  "$AZ_OUTPUT_FOLDER?$SAS_TOKEN"

# echo "azcopy copy $SHARD_NAME_WITHOUT_EXT\"_text_document.idx\" \"$AZ_OUTPUT_FOLDER?$SAS_TOKEN\"" >> out.txt
azcopy copy $SHARD_NAME_WITHOUT_EXT"_text_document.idx" "$AZ_OUTPUT_FOLDER?$SAS_TOKEN"