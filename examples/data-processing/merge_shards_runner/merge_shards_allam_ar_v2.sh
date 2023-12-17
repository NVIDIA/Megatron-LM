AZJOB_FILE="examples/data-processing/az_templates/template_merge_shard.yaml"
AZJOB_CONFIG="examples/configs/azure_login_configs.json"
AZ_OUTPUT_FOLDER="https://allamllmuksstandard.blob.core.windows.net/llm-data/allam-v2-data/merged_shards"
NUM_TOKEN=40486543872

python examples/data-processing/merge_shards.py \
--input-folder-path "https://allamllmuksstandard.blob.core.windows.net/llm-data/allam-v2-data/ar/translated/en2ar_books_corpus_formatted/" \
--output-folder-path "$AZ_OUTPUT_FOLDER" \
--az-sample-yaml-job-file "$AZJOB_FILE" \
--prefix-name "trans-ar_books3_" \
--shard-size $NUM_TOKEN \
--az-configs $AZJOB_CONFIG 

python examples/data-processing/merge_shards.py \
--input-folder-path "https://allamllmuksstandard.blob.core.windows.net/llm-data/allam-v2-data/ar/translated/en2ar_c4/" \
--output-folder-path "$AZ_OUTPUT_FOLDER" \
--az-sample-yaml-job-file "$AZJOB_FILE" \
--prefix-name "trans-ar_c4_" \
--shard-size $NUM_TOKEN \
--az-configs $AZJOB_CONFIG 

python examples/data-processing/merge_shards.py \
--input-folder-path "https://allamllmuksstandard.blob.core.windows.net/llm-data/allam-v2-data/ar/translated/en2ar_peS2o/" \
--output-folder-path "$AZ_OUTPUT_FOLDER" \
--az-sample-yaml-job-file "$AZJOB_FILE" \
--prefix-name "trans-ar_peS20_" \
--shard-size $NUM_TOKEN  \
--az-configs $AZJOB_CONFIG 

python examples/data-processing/merge_shards.py \
--input-folder-path "https://allamllmuksstandard.blob.core.windows.net/llm-data/allam-v2-data/ar/translated/en2ar_wikipedia/" \
--output-folder-path "$AZ_OUTPUT_FOLDER" \
--az-sample-yaml-job-file "$AZJOB_FILE" \
--prefix-name "trans-ar_wiki_" \
--shard-size $NUM_TOKEN \
--az-configs $AZJOB_CONFIG