if [ ! -f "examples/pretrain-llama/training/llama_ve/llama_ve_en_reasoning_ar_lr_hyp_tune/data.json" ]; then
    python examples/data-processing/remote_list.py \
    --az-configs "examples/configs/azure_login_configs.json" \
    --input-folder-path "https://allamllmuksstandard.blob.core.windows.net/llm-data/data_repo/tokenize_by_llama2-ve/meglm_llama2-v2_bin_idx/" \
    --export-data-signature "examples/pretrain-llama/training/llama_ve/llama_ve_en_reasoning_ar_lr_hyp_tune/data.json"
fi

python examples/data-processing/data_ratio_from_file.py \
 --prefix-paths-from-json "examples/pretrain-llama/training/llama_ve/llama_ve_en_reasoning_ar_lr_hyp_tune/data.json" \
 --domain-ratio-from-json "examples/pretrain-llama/training/llama_ve/llama_ve_en_reasoning_ar_lr_hyp_tune/data_ratio.json" \
 --lang-select-prob-json "examples/pretrain-llama/training/llama_ve/llama_ve_en_reasoning_ar_lr_hyp_tune/lang_prob.json" \
 --total-token 10_000_000_000 \
 --exclude-iterator-json "examples/pretrain-llama/training/llama_ve/llama_ve_en_reasoning_ar_lr_hyp_tune/exclude_iterator.json"