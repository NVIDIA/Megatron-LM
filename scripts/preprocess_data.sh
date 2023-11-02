OUTPUT_PATH="/mnt/deva100/sbmaruf/OpenAlphaCodeAssets/indexed_data"
mkdir -p $OUTPUT_PATH

DATA_PATH="/mnt/deva100/sbmaruf/OpenAlphaCodeAssets/prompted_for_training_bigdata_v2/chat_ml_*.jsonl"
for _inp in $DATA_PATH
do
python tools/preprocess_data.py \
 --input $_inp \
 --json-keys 'prompted_sample' \
 --partitions 1 \
 --tokenizer-type 'Llama2Tokenizer' \
 --tokenizer-model '/mnt/deva100/sbmaruf/OpenAlphaCodeAssets/CodeLlama-7b-hf/tokenizer.model' \
 --workers 24 \
 --output-prefix $OUTPUT_PATH'/openalphacode_' 
done
