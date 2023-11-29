
SHARD_NAME=$1
SHARD_NAME_WITHOUT_EXT="${SHARD_NAME%.*}"
INPUT_FOLDER=$2
OUTPUT_FOLDER=$3
TOK_MODULE=$4
TOK_TYPE=$5
TOK_MODEL=$6
TOK_NAME=$(basename "$TOK_MODEL")
# echo $TOK_NAME >> out.txt
VOCAB_FILE=$7
MERGE_FILE=$8
NUM_PROC=$9
LOG_INTERVAL=$8
SAS_TOKEN=$9

echo "Running azcopy copy \"$INPUT_FOLDER/$SHARD_NAME?$SAS_TOKEN\" \".\"" >> out.txt
azcopy copy "$INPUT_FOLDER/$SHARD_NAME?$SAS_TOKEN" "."

if [ "$TOK_MODULE" = "megatron" ]; then
    echo "Running tokenization via Megatron-LM module..."
    python tools/preprocess_data.py \
    --input $SHARD_NAME \
    --output-prefix $SHARD_NAME_WITHOUT_EXT \
    --tokenizer-type $TOK_TYPE \
    --tokenizer-model $TOK_NAME \
    --workers $NUM_PROC \
    --append-eod \
    --log-interval $LOG_INTERVAL

elif [ "$TOK_MODULE" = "nemo" ]; then
    echo "Running tokenization via NeMo-Megatron module..."
    python tools/preprocess_data_for_megatron.py \
    --input $SHARD_NAME \
    --output-prefix $SHARD_NAME_WITHOUT_EXT \
    --tokenizer-library $TOK_TYPE \
    --tokenizer-model $TOK_NAME \
    --workers $NUM_PROC \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --append-eod \
    --log-interval $LOG_INTERVAL
else
    echo "Tokenizer module not implemented."
    exit 1
fi

echo "Uploading bin ..."
echo "azcopy copy $SHARD_NAME_WITHOUT_EXT\"_text_document.bin\"  \"$OUTPUT_FOLDER?$SAS_TOKEN\""  >> out.txt
azcopy copy $SHARD_NAME_WITHOUT_EXT"_text_document.bin"  "$OUTPUT_FOLDER?$SAS_TOKEN"

echo "Uploading idx ..."
echo "azcopy copy $SHARD_NAME_WITHOUT_EXT\"_text_document.idx\" \"$OUTPUT_FOLDER?$SAS_TOKEN\"" >> out.txt
azcopy copy $SHARD_NAME_WITHOUT_EXT"_text_document.idx" "$OUTPUT_FOLDER?$SAS_TOKEN"