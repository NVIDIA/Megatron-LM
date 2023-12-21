azcopy copy "https://provisioningte0256624006.blob.core.windows.net/sbmaruf/vocab_expansion_assets/expanded_models/Llama-2-7b-hf-ExpTok-32K_10M_init_VE_with_llama_avg/?sv=2023-01-03&ss=btqf&srt=sco&st=2023-12-21T15%3A54%3A29Z&se=2023-12-22T15%3A54%3A29Z&sp=rwdxftlacup&sig=Fehd%2BAPkS3sJBAvPsPRO0V45JPduwI6OU7CUZanVEtU%3D" "../DUMPED/" --recursive --overwrite=false

CKPT_PATH="../DUMPED/Llama-2-7b-hf-ExpTok-32K_10M_init_VE_with_llama_avg/"
OUT_PATH="../DUMPED/Meg-LM-Llama-2-7b-hf-ExpTok-32K_10M_init_VE_with_llama_avg/"

python tools/checkpoint/util.py \
--load-dir $CKPT_PATH \
--save-dir $OUT_PATH \
--model-type "GPT" \
--loader "llama2_hf" \
--saver "megatron" \
--tokenizer-model "$CKPT_PATH/tokenizer.model" \
--target-tensor-parallel-size 2 \
--target-pipeline-parallel-size 2 \
--no-checking \
--max-queue-size 1 \
--megatron-path "."

azcopy copy ../DUMPED/Meg-LM-Llama-2-7b-hf-ExpTok-32K_10M_init_VE_with_llama_avg "https://provisioningte0256624006.blob.core.windows.net/sbmaruf/vocab_expansion_assets/megatronlm_models/?sv=2023-01-03&ss=btqf&srt=sco&st=2023-12-21T16%3A12%3A54Z&se=2023-12-22T16%3A12%3A54Z&sp=rwdxftlacup&sig=cRsaaOnhJPGo2Xfg%2BdTtunOzmyEmSPvU6CQSiU%2BFovw%3D" --recursive --overwrite=false