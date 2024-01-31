WEST_EUROPE_SAS_TOKEN="sv=2023-01-03&ss=btqf&srt=sco&st=2024-01-31T14%3A30%3A08Z&se=2024-11-30T14%3A30%3A00Z&sp=rwdxftlacup&sig=gk3anfhh%2F%2FsjS1TcYVVcViji%2BcV%2B9zGA6GwdvIptiEM%3D"
UK_SOUTH_SAS_TOKEN="sv=2023-01-03&ss=btqf&srt=sco&st=2024-01-31T15%3A22%3A43Z&se=2025-02-28T15%3A22%3A00Z&sp=rwdftlacup&sig=o4NWSxHuvuf2xaKl3%2F7F7MeVSXtziizdZZ2ov8ot%2Fw4%3D"

OUTPUT_DIR="../DUMPED/allam-7bv5-en"
mkdir $OUTPUT_DIR

azcopy copy "https://provisioningte0256624006.blob.core.windows.net/sbmaruf/allam_assets/Checkpoints/allam-7b-alpha/?$WEST_EUROPE_SAS_TOKEN" "$OUTPUT_DIR/" --recursive --overwrite=false

azcopy copy "https://allamllmstorageuks.blob.core.windows.net/tokenizer/tokenizer_v5_improved/?$UK_SOUTH_SAS_TOKEN" "$OUTPUT_DIR/" --recursive  --overwrite=false

TOK_MODEL_PATH="../DUMPED/allam-7bv5-en/tokenizer_v5_improved/ar_en.model"
INPUT_DIR=../DUMPED/allam-7bv5-en/allam-7b-alpha/hf/combined_iter_0105000_iter_0200000_iter_0460000_iter_0615000_hf
OUTPUT_DIR="../DUMPED/allam-7bv5-en/allam-7b-alpha/megatron/combined_iter_0105000_iter_0200000_iter_0460000_iter_0615000"

TP=1
PP=1
python tools/checkpoint/util.py \
--load-dir $INPUT_DIR \
--save-dir ${OUTPUT_DIR}_tp"$TP"_pp"$PP"/ \
--model-type "GPT" \
--loader "llama2_hf" \
--saver "megatron" \
--target-tensor-parallel-size $TP \
--target-pipeline-parallel-size $PP \
--no-checking \
--max-queue-size 1 \
--megatron-path "." \
--tokenizer-model $TOK_MODEL_PATH


TP=2
PP=2
python tools/checkpoint/util.py \
--load-dir $INPUT_DIR \
--save-dir ${OUTPUT_DIR}_tp"$TP"_pp"$PP"/ \
--model-type "GPT" \
--loader "llama2_hf" \
--saver "megatron" \
--target-tensor-parallel-size $TP \
--target-pipeline-parallel-size $PP \
--no-checking \
--max-queue-size 1 \
--megatron-path "." \
--tokenizer-model $TOK_MODEL_PATH


azcopy copy "../DUMPED/allam-7bv5-en/allam-7b-alpha/megatron" "https://provisioningte025662
4006.blob.core.windows.net/sbmaruf/allam_assets/Checkpoints/allam-7b-alpha/?$WEST_EUROPE_SAS_TOKEN" --recursive --overwrite=false