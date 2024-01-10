
azcopy copy "https://provisioningte0256624006.blob.core.windows.net/sbmaruf/vocab_expansion_assets/expanded_models/Llama-2-70b-hf-ExpTok-32K_10M_VE_init/?sv=2023-01-03&ss=btqf&srt=sco&st=2023-12-25T06%3A25%3A17Z&se=2024-09-26T06%3A25%3A00Z&sp=rwdxftlacup&sig=vlLjVfzHFXr%2BYspTH81Wn88zqKGRli3AUMyWD1Rkoak%3D" "../DUMPED/" --recursive --overwrite=false

azcopy copy "https://provisioningte0256624006.blob.core.windows.net/sbmaruf/vocab_expansion_assets/merged_tokenizers/llama2_ar_32K_10M/?sv=2023-01-03&ss=btqf&srt=sco&st=2023-12-25T06%3A25%3A17Z&se=2024-09-26T06%3A25%3A00Z&sp=rwdxftlacup&sig=vlLjVfzHFXr%2BYspTH81Wn88zqKGRli3AUMyWD1Rkoak%3D" "../DUMPED/" --recursive --overwrite=false

TP=4
PP=2

python tools/checkpoint/util.py \
--load-dir "../DUMPED/Llama-2-70b-hf-ExpTok-32K_10M_VE_init/" \
--save-dir "../DUMPED/Meg-LM-Llama-2-70b-hf-ExpTok-32K_10M_VE_with_llama_avg_tp$TP"_pp"$PP/" \
--model-type "GPT" \
--loader "llama2_hf" \
--saver "megatron" \
--target-tensor-parallel-size $TP \
--target-pipeline-parallel-size $PP \
--no-checking \
--max-queue-size 1 \
--megatron-path "." \
--tokenizer-model ../DUMPED/llama2_ar_32K_10M/spm/tokenizer.model

azcopy copy "../DUMPED/Meg-LM-Llama-2-70b-hf-ExpTok-32K_10M_VE_with_llama_avg_tp*" "https://provisioningte0256624006.blob.core.windows.net/sbmaruf/vocab_expansion_assets/megatronlm_models/?sv=2023-01-03&ss=btqf&srt=sco&st=2023-12-25T06%3A25%3A17Z&se=2024-09-26T06%3A25%3A00Z&sp=rwdxftlacup&sig=vlLjVfzHFXr%2BYspTH81Wn88zqKGRli3AUMyWD1Rkoak%3D" --recursive --overwrite=false
rm -rf "../DUMPED/Meg-LM-Llama-2-70b-hf-ExpTok-32K_10M_VE_with_llama_avg_tp*"

TP=2
PP=4

python tools/checkpoint/util.py \
--load-dir "../DUMPED/Llama-2-70b-hf-ExpTok-32K_10M_VE_init/" \
--save-dir "../DUMPED/Meg-LM-Llama-2-70b-hf-ExpTok-32K_10M_VE_with_llama_avg_tp$TP"_pp"$PP/" \
--model-type "GPT" \
--loader "llama2_hf" \
--saver "megatron" \
--target-tensor-parallel-size $TP \
--target-pipeline-parallel-size $PP \
--no-checking \
--max-queue-size 1 \
--megatron-path "." \
--tokenizer-model ../DUMPED/llama2_ar_32K_10M/spm/tokenizer.model

azcopy copy "../DUMPED/Meg-LM-Llama-2-70b-hf-ExpTok-32K_10M_VE_with_llama_avg_tp*" "https://provisioningte0256624006.blob.core.windows.net/sbmaruf/vocab_expansion_assets/megatronlm_models/?sv=2023-01-03&ss=btqf&srt=sco&st=2023-12-25T06%3A25%3A17Z&se=2024-09-26T06%3A25%3A00Z&sp=rwdxftlacup&sig=vlLjVfzHFXr%2BYspTH81Wn88zqKGRli3AUMyWD1Rkoak%3D" --recursive --overwrite=false
rm -rf ../DUMPED/Meg-LM-Llama-2-70b-hf-ExpTok-32K_10M_VE_with_llama_avg_tp*

TP=8
PP=1

python tools/checkpoint/util.py \
--load-dir "../DUMPED/Llama-2-70b-hf-ExpTok-32K_10M_VE_init/" \
--save-dir "../DUMPED/Meg-LM-Llama-2-70b-hf-ExpTok-32K_10M_VE_with_llama_avg_tp$TP"_pp"$PP/" \
--model-type "GPT" \
--loader "llama2_hf" \
--saver "megatron" \
--target-tensor-parallel-size $TP \
--target-pipeline-parallel-size $PP \
--no-checking \
--max-queue-size 1 \
--megatron-path "." \
--tokenizer-model ../DUMPED/llama2_ar_32K_10M/spm/tokenizer.model

azcopy copy "../DUMPED/Meg-LM-Llama-2-70b-hf-ExpTok-32K_10M_VE_with_llama_avg_tp*" "https://provisioningte0256624006.blob.core.windows.net/sbmaruf/vocab_expansion_assets/megatronlm_models/?sv=2023-01-03&ss=btqf&srt=sco&st=2023-12-25T06%3A25%3A17Z&se=2024-09-26T06%3A25%3A00Z&sp=rwdxftlacup&sig=vlLjVfzHFXr%2BYspTH81Wn88zqKGRli3AUMyWD1Rkoak%3D" --recursive --overwrite=false
rm -rf ../DUMPED/Meg-LM-Llama-2-70b-hf-ExpTok-32K_10M_VE_with_llama_avg_tp*

TP=1
PP=8

python tools/checkpoint/util.py \
--load-dir "../DUMPED/Llama-2-70b-hf-ExpTok-32K_10M_VE_init/" \
--save-dir "../DUMPED/Meg-LM-Llama-2-70b-hf-ExpTok-32K_10M_VE_with_llama_avg_tp$TP"_pp"$PP/" \
--model-type "GPT" \
--loader "llama2_hf" \
--saver "megatron" \
--target-tensor-parallel-size $TP \
--target-pipeline-parallel-size $PP \
--no-checking \
--max-queue-size 1 \
--megatron-path "." \
--tokenizer-model ../DUMPED/llama2_ar_32K_10M/spm/tokenizer.model

azcopy copy "../DUMPED/Meg-LM-Llama-2-70b-hf-ExpTok-32K_10M_VE_with_llama_avg_tp*" "https://provisioningte0256624006.blob.core.windows.net/sbmaruf/vocab_expansion_assets/megatronlm_models/?sv=2023-01-03&ss=btqf&srt=sco&st=2023-12-25T06%3A25%3A17Z&se=2024-09-26T06%3A25%3A00Z&sp=rwdxftlacup&sig=vlLjVfzHFXr%2BYspTH81Wn88zqKGRli3AUMyWD1Rkoak%3D" --recursive --overwrite=false
rm -rf ../DUMPED/Meg-LM-Llama-2-70b-hf-ExpTok-32K_10M_VE_with_llama_avg_tp*

TP=4
PP=4

python tools/checkpoint/util.py \
--load-dir "../DUMPED/Llama-2-70b-hf-ExpTok-32K_10M_VE_init/" \
--save-dir "../DUMPED/Meg-LM-Llama-2-70b-hf-ExpTok-32K_10M_VE_with_llama_avg_tp$TP"_pp"$PP/" \
--model-type "GPT" \
--loader "llama2_hf" \
--saver "megatron" \
--target-tensor-parallel-size $TP \
--target-pipeline-parallel-size $PP \
--no-checking \
--max-queue-size 1 \
--megatron-path "." \
--tokenizer-model ../DUMPED/llama2_ar_32K_10M/spm/tokenizer.model

azcopy copy "../DUMPED/Meg-LM-Llama-2-70b-hf-ExpTok-32K_10M_VE_with_llama_avg_tp*" "https://provisioningte0256624006.blob.core.windows.net/sbmaruf/vocab_expansion_assets/megatronlm_models/?sv=2023-01-03&ss=btqf&srt=sco&st=2023-12-25T06%3A25%3A17Z&se=2024-09-26T06%3A25%3A00Z&sp=rwdxftlacup&sig=vlLjVfzHFXr%2BYspTH81Wn88zqKGRli3AUMyWD1Rkoak%3D" --recursive --overwrite=false
rm -rf ../DUMPED/Meg-LM-Llama-2-70b-hf-ExpTok-32K_10M_VE_with_llama_avg_tp*

TP=8
PP=2

python tools/checkpoint/util.py \
--load-dir "../DUMPED/Llama-2-70b-hf-ExpTok-32K_10M_VE_init/" \
--save-dir "../DUMPED/Meg-LM-Llama-2-70b-hf-ExpTok-32K_10M_VE_with_llama_avg_tp$TP"_pp"$PP/" \
--model-type "GPT" \
--loader "llama2_hf" \
--saver "megatron" \
--target-tensor-parallel-size $TP \
--target-pipeline-parallel-size $PP \
--no-checking \
--max-queue-size 1 \
--megatron-path "." \
--tokenizer-model ../DUMPED/llama2_ar_32K_10M/spm/tokenizer.model

azcopy copy "../DUMPED/Meg-LM-Llama-2-70b-hf-ExpTok-32K_10M_VE_with_llama_avg_tp*" "https://provisioningte0256624006.blob.core.windows.net/sbmaruf/vocab_expansion_assets/megatronlm_models/?sv=2023-01-03&ss=btqf&srt=sco&st=2023-12-25T06%3A25%3A17Z&se=2024-09-26T06%3A25%3A00Z&sp=rwdxftlacup&sig=vlLjVfzHFXr%2BYspTH81Wn88zqKGRli3AUMyWD1Rkoak%3D" --recursive --overwrite=false
rm -rf ../DUMPED/Meg-LM-Llama-2-70b-hf-ExpTok-32K_10M_VE_with_llama_avg_tp*

TP=2
PP=8

python tools/checkpoint/util.py \
--load-dir "../DUMPED/Llama-2-70b-hf-ExpTok-32K_10M_VE_init/" \
--save-dir "../DUMPED/Meg-LM-Llama-2-70b-hf-ExpTok-32K_10M_VE_with_llama_avg_tp$TP"_pp"$PP/" \
--model-type "GPT" \
--loader "llama2_hf" \
--saver "megatron" \
--target-tensor-parallel-size $TP \
--target-pipeline-parallel-size $PP \
--no-checking \
--max-queue-size 1 \
--megatron-path "." \
--tokenizer-model ../DUMPED/llama2_ar_32K_10M/spm/tokenizer.model

azcopy copy "../DUMPED/Meg-LM-Llama-2-70b-hf-ExpTok-32K_10M_VE_with_llama_avg_tp*" "https://provisioningte0256624006.blob.core.windows.net/sbmaruf/vocab_expansion_assets/megatronlm_models/?sv=2023-01-03&ss=btqf&srt=sco&st=2023-12-25T06%3A25%3A17Z&se=2024-09-26T06%3A25%3A00Z&sp=rwdxftlacup&sig=vlLjVfzHFXr%2BYspTH81Wn88zqKGRli3AUMyWD1Rkoak%3D" --recursive --overwrite=false
rm -rf ../DUMPED/Meg-LM-Llama-2-70b-hf-ExpTok-32K_10M_VE_with_llama_avg_tp*

TP=4
PP=8

python tools/checkpoint/util.py \
--load-dir "../DUMPED/Llama-2-70b-hf-ExpTok-32K_10M_VE_init/" \
--save-dir "../DUMPED/Meg-LM-Llama-2-70b-hf-ExpTok-32K_10M_VE_with_llama_avg_tp$TP"_pp"$PP/" \
--model-type "GPT" \
--loader "llama2_hf" \
--saver "megatron" \
--target-tensor-parallel-size $TP \
--target-pipeline-parallel-size $PP \
--no-checking \
--max-queue-size 1 \
--megatron-path "." \
--tokenizer-model ../DUMPED/llama2_ar_32K_10M/spm/tokenizer.model

azcopy copy "../DUMPED/Meg-LM-Llama-2-70b-hf-ExpTok-32K_10M_VE_with_llama_avg_tp*" "https://provisioningte0256624006.blob.core.windows.net/sbmaruf/vocab_expansion_assets/megatronlm_models/?sv=2023-01-03&ss=btqf&srt=sco&st=2023-12-25T06%3A25%3A17Z&se=2024-09-26T06%3A25%3A00Z&sp=rwdxftlacup&sig=vlLjVfzHFXr%2BYspTH81Wn88zqKGRli3AUMyWD1Rkoak%3D" --recursive --overwrite=false
rm -rf ../DUMPED/Meg-LM-Llama-2-70b-hf-ExpTok-32K_10M_VE_with_llama_avg_tp*

TP=8
PP=4
python tools/checkpoint/util.py \
--load-dir "../DUMPED/Llama-2-70b-hf-ExpTok-32K_10M_VE_init/" \
--save-dir "../DUMPED/Meg-LM-Llama-2-70b-hf-ExpTok-32K_10M_VE_with_llama_avg_tp$TP"_pp"$PP/" \
--model-type "GPT" \
--loader "llama2_hf" \
--saver "megatron" \
--target-tensor-parallel-size $TP \
--target-pipeline-parallel-size $PP \
--no-checking \
--max-queue-size 1 \
--megatron-path "." \
--tokenizer-model ../DUMPED/llama2_ar_32K_10M/spm/tokenizer.model

azcopy copy "../DUMPED/Meg-LM-Llama-2-70b-hf-ExpTok-32K_10M_VE_with_llama_avg_tp*" "https://provisioningte0256624006.blob.core.windows.net/sbmaruf/vocab_expansion_assets/megatronlm_models/?sv=2023-01-03&ss=btqf&srt=sco&st=2023-12-25T06%3A25%3A17Z&se=2024-09-26T06%3A25%3A00Z&sp=rwdxftlacup&sig=vlLjVfzHFXr%2BYspTH81Wn88zqKGRli3AUMyWD1Rkoak%3D" --recursive --overwrite=false
rm -rf ../DUMPED/Meg-LM-Llama-2-70b-hf-ExpTok-32K_10M_VE_with_llama_avg_tp*

TP=8
PP=8
python tools/checkpoint/util.py \
--load-dir "../DUMPED/Llama-2-70b-hf-ExpTok-32K_10M_VE_init/" \
--save-dir "../DUMPED/Meg-LM-Llama-2-70b-hf-ExpTok-32K_10M_VE_with_llama_avg_tp$TP"_pp"$PP/" \
--model-type "GPT" \
--loader "llama2_hf" \
--saver "megatron" \
--target-tensor-parallel-size $TP \
--target-pipeline-parallel-size $PP \
--no-checking \
--max-queue-size 1 \
--megatron-path "." \
--tokenizer-model ../DUMPED/llama2_ar_32K_10M/spm/tokenizer.model

azcopy copy "../DUMPED/Meg-LM-Llama-2-70b-hf-ExpTok-32K_10M_VE_with_llama_avg_tp*" "https://provisioningte0256624006.blob.core.windows.net/sbmaruf/vocab_expansion_assets/megatronlm_models/?sv=2023-01-03&ss=btqf&srt=sco&st=2023-12-25T06%3A25%3A17Z&se=2024-09-26T06%3A25%3A00Z&sp=rwdxftlacup&sig=vlLjVfzHFXr%2BYspTH81Wn88zqKGRli3AUMyWD1Rkoak%3D" --recursive --overwrite=false
rm -rf ../DUMPED/Meg-LM-Llama-2-70b-hf-ExpTok-32K_10M_VE_with_llama_avg_tp*