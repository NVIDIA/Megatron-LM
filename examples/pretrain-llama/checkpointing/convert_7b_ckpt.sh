# sample script, may need modifications.

python tools/checkpoint/util.py \
--load-dir "../DUMPED/Llama-2-7b-hf-ExpTok-32K_10M_tp2_pp2_seq4096_gb48/" \
--save-dir "../DUMPED/tp1_pp1/" \
--model-type "GPT" \
--loader "megatron" \
--saver "megatron" \
--target-tensor-parallel-size 1 \
--target-pipeline-parallel-size 1 \
--no-checking \
--max-queue-size 1 \
--megatron-path "."


python tools/checkpoint/util.py \
--load-dir "/tmp/Exps/dumped/Meg-LM-Llama-2-7b-hf-ExpTok-32K_10M_init_VE_with_llama_avg" \
--save-dir "/tmp/Exps/dumped/Meg-LM-Llama-2-7b-hf-ExpTok-32K_10M_init_VE_with_llama_avg_tp1_pp1/" \
--model-type "GPT" \
--loader "megatron" \
--saver "megatron" \
--target-tensor-parallel-size 1 \
--target-pipeline-parallel-size 1 \
--no-checking \
--max-queue-size 1 \
--megatron-path "."