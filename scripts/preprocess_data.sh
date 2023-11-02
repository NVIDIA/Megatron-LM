OUTPUT_PATH="/mnt/deva100/sbmaruf/OpenAlphaCodeAssets/indexed_data"
mkdir -p $OUTPUT_PATH

DATA_PATH="/mnt/deva100/sbmaruf/OpenAlphaCodeAssets/prompted_for_training_bigdata_v2/program_synthesis_codecontest.jsonl"
for _inp in $DATA_PATH
do
python tools/preprocess_data_sft.py \
 --input $_inp \
 --json-keys 'inputs' 'targets' \
 --partitions 1 \
 --tokenizer-type 'Llama2Tokenizer' \
 --tokenizer-model '/mnt/deva100/sbmaruf/OpenAlphaCodeAssets/CodeLlama-7b-hf/tokenizer.model' \
 --worker 24 \
 --output-prefix $OUTPUT_PATH'/openalphacode_' \
 --chatml-template '{bos_token}[INST]{inputs}[/INST]\n{targets}\n{eos_token}' \
 --output-json-key 'prompted_sample'
done
