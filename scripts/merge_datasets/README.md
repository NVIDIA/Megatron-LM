As showed in [#12](https://github.com/swiss-ai/Megatron-LM/pull/12), when using datasets split into hundreds of file prefixes, we may encounter CPU OOM errors. To address this, it is necessary to merge multiple files into a single large file. 

For this purpose, we have developed this pipeline, which simply requires specifying the `TOKENIZER_NAME` and `DATASET_NAME` previously used for tokenizing the data, and it will merge every dump folder (which by default will have 20 file prefixes—check `NUMBER_OF_DATATROVE_TASKS` in `scripts/tokenization/submit_tokenization.sh`) into a single file.

