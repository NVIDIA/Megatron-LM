set -e 
TOTAL_NUM_TOKENS=10_000_000_000
for ENG_TOK in {2..9}
do
    
    AR_TOK=$((10 - $ENG_TOK))

    echo "{\"en\": $ENG_TOK,\"ar\": $AR_TOK}" > examples/pretrain-allam/training/allam-7bv3-0-tune/lang_prob.json

    if [ ! -f "examples/pretrain-allam/training/allam-7bv3-0-tune/data.json" ]; then
        python examples/data-processing/remote_list.py \
        --az-configs "examples/configs/azure_login_configs.json" \
        --input-folder-path "https://allamllmuksstandard.blob.core.windows.net/llm-data/data_repo/tokenize_by_v5_improved/meglm_tok_v5_improved_bin_idx/" \
        --export-data-signature "examples/pretrain-allam/training/allam-7bv3-0-tune/data.json"
    fi

    python examples/data-processing/data_ratio_from_file.py \
    --prefix-paths-from-json "examples/pretrain-allam/training/allam-7bv3-0-tune/data.json" \
    --domain-ratio-from-json "examples/pretrain-allam/training/allam-7bv3-0-tune/data_ratio.json" \
    --lang-select-prob-json "examples/pretrain-allam/training/allam-7bv3-0-tune/lang_prob.json" \
    --total-token $TOTAL_NUM_TOKENS \
    --exclude-iterator-json "examples/pretrain-allam/training/allam-7bv3-0-tune/exclude_iterator.json" \
    --prefix-for-file-path "\$BIN_IDX_PATH/" \
    --export-script "examples/pretrain-allam/training/allam-7bv3-0-tune/iterator_prob.sh"

    sed "s/\${{ENG_TOK}}/$ENG_TOK/g; s/allam-7bv3-0_dolma-tune/allam-7bv3-0_dolma-tune_en-$ENG_TOK"_ar-"$AR_TOK/g" examples/pretrain-allam/training/allam-7bv3-0-tune/azureml_conf.yaml > examples/pretrain-allam/training/allam-7bv3-0-tune/temp.yaml
    cat examples/pretrain-allam/training/allam-7bv3-0-tune/temp.yaml
    az ml job create --subscription c7209a17-0d9f-41df-8e45-e0172343698d \
     --resource-group LLM-Test \
     --workspace-name Provisioning-Test \
     --file examples/pretrain-allam/training/allam-7bv3-0-tune/temp.yaml
    rm examples/pretrain-allam/training/allam-7bv3-0-tune/temp.yaml
done