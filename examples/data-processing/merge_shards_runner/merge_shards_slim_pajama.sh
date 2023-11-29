
NUM_SIZE=4348654387

# # train
# python examples/data-processing/merge_shard.py \
# --input-folder-path "../RAW_DATA_FOLDER/SlimPajama-627B/train/" \
# --output-folder-path "../RAW_DATA_FOLDER/SlimPajama-627B/merged_shards" \
# --prefix-name "en_slim_pajama_" \
# --compute-target 'local' \
# --shard-size $NUM_TOKEN 

# # validation
# python examples/data-processing/merge_shard.py \
# --input-folder-path "../RAW_DATA_FOLDER/SlimPajama-627B/validation/" \
# --output-folder-path "../RAW_DATA_FOLDER/SlimPajama-627B/merged_shards" \
# --prefix-name "en_slim_pajama_" \
# --compute-target 'local' \
# --shard-size $NUM_TOKEN 

# # test
# python examples/data-processing/merge_shard.py \
# --input-folder-path $PATH_TO_SLIM_PAJAMA/test.jsonl \
# --output-folder-path "$PATH_TO_SLIM_PAJAMA/merged_shards" \
# --prefix-name "en_slim_pajama_" \
# --compute-target 'local' \
# --shard-size $NUM_SIZE 