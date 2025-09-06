python "tools/preprocess_data.py" \
       --input './dataset/dolma/**/*.json.gz' \
       --workers 32 \
       --partitions 8 \
       --output-prefix ./dataset/dolma_processed \
       --tokenizer-type HuggingFaceTokenizer \
       --tokenizer-model ./model/llama3/ \
       --append-eod
