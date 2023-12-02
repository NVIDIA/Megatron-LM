python examples/data-processing/process_shard.py \
    --az-configs examples/configs/azure_login_configs.json \
    --az-sample-yaml-job-file examples/data-processing/az_templates/template_count_token_and_rename_bin_idx.yaml \
    --input-folder-path "https://allamllmuksstandard.blob.core.windows.net/vocab-expanded-training-data/dolma/meg_lm_tok_v5_improved_bin_idx" \
    --output-folder-path "https://allamllmuksstandard.blob.core.windows.net/vocab-expanded-training-data/dolma/meg_lm_tok_v5_improved_bin_idx_with_meta" \
    --config-path-for-process-function example/shard_process_count_token_and_rename_bin_idx.json \
    --compute-target azure \
    --dry-run