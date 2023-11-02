OUTPUT_PATH="/mnt/deva100/sbmaruf/OpenAlphaCodeAssets/prompted_for_training_bigdata_v2/"
mkdir -p $OUTPUT_PATH

DATA_PATH="/mnt/deva100/sbmaruf/OpenAlphaCodeAssets/prompted_for_training_bigdata_v2/*.jsonl"
for _inp in $DATA_PATH; do
    python tools/project_chatml.py \
    --input "$_inp" \
    --json-keys 'inputs' 'targets' \
    --tokenizer-type 'Llama2Tokenizer' \
    --tokenizer-model '/mnt/deva100/sbmaruf/OpenAlphaCodeAssets/CodeLlama-7b-hf/tokenizer.model' \
    --workers 24 \
    --output-prefix $OUTPUT_PATH'/chat_ml_' \
    --chatml-template '{bos_token}[INST]{inputs}[/INST]\n{targets}\n{eos_token}' \
    --output-json-key 'prompted_sample'
done
