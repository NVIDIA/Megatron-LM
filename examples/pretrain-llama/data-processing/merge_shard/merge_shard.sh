# python examples/pretrain-llama/data-processing/merge_shard/merge_shard.py \
# --az-subscription "c7209a17-0d9f-41df-8e45-e0172343698d" \
# --az-resource-group "LLM-Test" \
# --az-workspace-name "Provisioning-Test" \
# --az-blob-input-folder "https://allamllmuksstandard.blob.core.windows.net/vocab-expanded-training-data/processed/ar/translated/en2ar_books_corpus_formatted" \
# --az-blob-output-folder-path "https://allamllmuksstandard.blob.core.windows.net/vocab-expanded-training-data/processed/merged_shards" \
# --az-sas-token "sp=racwdli&st=2023-11-25T01:04:32Z&se=2023-11-25T09:04:32Z&spr=https&sv=2022-11-02&sr=c&sig=htlh%2B5OEmWy0ikTxgrAJny0F7g3PdgXxYFIQwsEHYh0%3D" \
# --sample-yaml-job-file "examples/pretrain-llama/data-processing/merge_shard/template_merge_shard.yaml" \
# --prefix-name "ar_translated_books3_" \
# --shard-size 96636764160 

python examples/pretrain-llama/data-processing/merge_shard/merge_shard.py \
--az-subscription "c7209a17-0d9f-41df-8e45-e0172343698d" \
--az-resource-group "LLM-Test" \
--az-workspace-name "Provisioning-Test" \
--az-blob-input-folder "https://allamllmuksstandard.blob.core.windows.net/vocab-expanded-training-data/processed/ar/translated/en2ar_books_corpus_formatted" \
--az-blob-output-folder-path "https://allamllmuksstandard.blob.core.windows.net/vocab-expanded-training-data/processed/merged_shards" \
--az-sas-token "sp=racwdli&st=2023-11-25T01:04:32Z&se=2023-11-25T09:04:32Z&spr=https&sv=2022-11-02&sr=c&sig=htlh%2B5OEmWy0ikTxgrAJny0F7g3PdgXxYFIQwsEHYh0%3D" \
--sample-yaml-job-file "examples/pretrain-llama/data-processing/merge_shard/template_merge_shard.yaml" \
--prefix-name "ar_translated_books3_" \
--shard-size 96636764160 

python examples/pretrain-llama/data-processing/merge_shard/merge_shard.py \
--az-subscription "c7209a17-0d9f-41df-8e45-e0172343698d" \
--az-resource-group "LLM-Test" \
--az-workspace-name "Provisioning-Test" \
--az-blob-input-folder "https://allamllmuksstandard.blob.core.windows.net/vocab-expanded-training-data/processed/ar/translated/en2ar_peS2o" \
--az-blob-output-folder-path "https://allamllmuksstandard.blob.core.windows.net/vocab-expanded-training-data/processed/merged_shards" \
--az-sas-token "sp=racwdli&st=2023-11-25T01:04:32Z&se=2023-11-25T09:04:32Z&spr=https&sv=2022-11-02&sr=c&sig=htlh%2B5OEmWy0ikTxgrAJny0F7g3PdgXxYFIQwsEHYh0%3D" \
--sample-yaml-job-file "examples/pretrain-llama/data-processing/merge_shard/template_merge_shard.yaml" \
--prefix-name "ar_translated_books3_" \
--shard-size 96636764160