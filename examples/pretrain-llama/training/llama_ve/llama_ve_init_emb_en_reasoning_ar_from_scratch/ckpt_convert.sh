azcopy copy "https://provisioningte0256624006.blob.core.windows.net/sbmaruf/vocab_expansion_assets/megatronlm_models/Meg-LM-Llama-2-7b-hf-ExpTok-32K_10M_init_VE_with_llama_avg/?sv=2023-01-03&ss=btqf&srt=sco&st=2024-01-31T14%3A30%3A08Z&se=2024-11-30T14%3A30%3A00Z&sp=rwdxftlacupsig=gk3anfhh%2F%2FsjS1TcYVVcViji%2BcV%2B9zGA6GwdvIptiEM%3D" "./" --recursive

python tools/checkpoint/util.py \
--load-dir "/tmp/Exps/DUMPED/Meg-LM-Llama-2-7b-hf-ExpTok-32K_10M_init_VE_with_llama_avg" \
--save-dir "../DUMPED/Meg-LM-Llama-2-7b-hf-ExpTok-32K_10M_init_VE_with_llama_avg_tp1_pp1/" \
--model-type "GPT" \
--loader "megatron" \
--saver "megatron" \
--target-tensor-parallel-size 1 \
--target-pipeline-parallel-size 1 \
--no-checking \
--max-queue-size 1 \
--megatron-path "."