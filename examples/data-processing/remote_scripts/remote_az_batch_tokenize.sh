set -e 

SHARD_NAME=$1
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
LOG_INTERVAL=${10}
SAS_TOKEN=${11}

echo "SHARD_NAME " $SHARD_NAME
echo "INPUT_FOLDER " $INPUT_FOLDER
echo "OUTPUT_FOLDER " $OUTPUT_FOLDER
echo "TOK_MODULE " $TOK_MODULE
echo "TOK_TYPE " $TOK_TYPE
echo "TOK_MODEL " $TOK_MODEL
echo "VOCAB_FILE " $VOCAB_FILE
echo "MERGE_FILE " $MERGE_FILE
echo "NUM_PROC " $NUM_PROC
echo "LOG_INTERVAL " $LOG_INTERVAL
echo "SAS_TOKEN " $SAS_TOKEN


echo "Running azcopy copy \"$INPUT_FOLDER/$SHARD_NAME?$SAS_TOKEN\" \".\""
azcopy copy "$INPUT_FOLDER/$SHARD_NAME?$SAS_TOKEN" "."

echo "Running azcopy copy \"$TOK_MODEL?$SAS_TOKEN\" \".\""
azcopy copy "$TOK_MODEL?$SAS_TOKEN" "."

if [ "$VOCAB_FILE" != "None" ]; then
echo "Running azcopy copy \"$VOCAB_FILE?$SAS_TOKEN\" \".\"" 
azcopy copy "$VOCAB_FILE?$SAS_TOKEN" "."
fi

if [ "$MERGE_FILE" != "None" ]; then
echo "Running azcopy copy \"$MERGE_FILE?$SAS_TOKEN\" \".\""
azcopy copy "$MERGE_FILE?$SAS_TOKEN" "."
fi

SHARD_NAME=$(basename $SHARD_NAME)
SHARD_NAME_WITHOUT_EXT="${SHARD_NAME%.*}"

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
    --append-eod \
    --log-interval $LOG_INTERVAL
else
    echo "Tokenizer module not implemented."
    exit 1
fi

python examples/data-processing/count_token_and_rename_bin_idx.py \
  --source-prefix-paths $SHARD_NAME_WITHOUT_EXT"_text_document.bin" \
  --megatron-path '.'

echo "Uploading bin ..."
echo "azcopy copy *.bin\"  \"$OUTPUT_FOLDER?$SAS_TOKEN\""
azcopy copy *.bin  "$OUTPUT_FOLDER?$SAS_TOKEN"

echo "Uploading idx ..."
echo "azcopy copy *.idx\" \"$OUTPUT_FOLDER?$SAS_TOKEN\""
azcopy copy *.idx "$OUTPUT_FOLDER?$SAS_TOKEN"
