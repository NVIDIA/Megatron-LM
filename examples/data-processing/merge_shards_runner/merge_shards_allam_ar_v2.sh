AZJOB_FILE="examples/data-processing/merge_shard/az_templates/template_merge_shard.yaml"
AZ_OUTPUT_FOLDER="https://allamllmuksstandard.blob.core.windows.net/vocab-expanded-training-data/processed/merged_shards"
NUM_TOKEN=43486543872

python examples/data-processing/merge_shards.py \
--input-folder-path "https://allamllmuksstandard.blob.core.windows.net/llm-data/processed/ar/translated/en2ar_books_corpus_formatted" \
--output-folder-path "$AZ_OUTPUT_FOLDER" \
--sample-yaml-job-file "$AZJOB_FILE" \
--prefix-name "trans-ar_books3_" \
--shard-size $NUM_TOKEN 

python examples/data-processing/merge_shards.py \
--input-folder-path "https://allamllmuksstandard.blob.core.windows.net/llm-data/processed/ar/translated/en2ar_c4" \
--output-folder-path "$AZ_OUTPUT_FOLDER" \
--sample-yaml-job-file "$AZJOB_FILE" \
--prefix-name "trans-ar_c4_" \
--shard-size $NUM_TOKEN 

python examples/data-processing/merge_shards.py \
--input-folder-path "https://allamllmuksstandard.blob.core.windows.net/llm-data/processed/ar/translated/en2ar_peS2o/arabic_text_only" \
--output-folder-path "$AZ_OUTPUT_FOLDER" \
--sample-yaml-job-file "$AZJOB_FILE" \
--prefix-name "trans-ar_peS20_" \
--shard-size $NUM_TOKEN  \
--az-configs "examples/configs/azure_login_configs.json" \

python examples/data-processing/merge_shards.py \
--input-folder-path "https://allamllmuksstandard.blob.core.windows.net/llm-data/processed/ar/translated/en2ar_wikipedia" \
--output-folder-path "$AZ_OUTPUT_FOLDER" \
--sample-yaml-job-file "$AZJOB_FILE" \
--prefix-name "trans-ar_wiki_" \
--shard-size $NUM_TOKEN \
--az-configs "examples/configs/azure_login_configs.json"