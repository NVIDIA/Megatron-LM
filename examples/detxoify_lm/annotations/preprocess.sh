VOCAB_FILE=pt2-vocab.json
MERGE_FILE=gpt2-merges.txt

python3 tools/preprocess_data.py \
    --input $1 \
    --output-prefix $2 \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --tokenizer-type GPT2BPETokenizer \
    --append-eod  --workers 20 --chunk-size 25




