PATH_TO_SLIM_PAJAMA='../RAW_DATA_FOLDER/SlimPajama-627B'
mkdir -p ../RAW_DATA_FOLDER/
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/cerebras/SlimPajama-627B $PATH_TO_SLIM_PAJAMA
cd $PATH_TO_SLIM_PAJAMA
git lfs pull
rm -rf .git


rm -rf ../RAW_DATA_FOLDER/SlimPajama-627B/export/
mkdir -p $PATH_TO_SLIM_PAJAMA/export/

find $PATH_TO_SLIM_PAJAMA/train -type f -name '*.jsonl' | sort | xargs -I {} cat {} >> $PATH_TO_SLIM_PAJAMA/export/train.jsonl
find $PATH_TO_SLIM_PAJAMA/validation -type f -name '*.jsonl' | sort | xargs -I {} cat {} >> $PATH_TO_SLIM_PAJAMA/export/validation.jsonl
find $PATH_TO_SLIM_PAJAMA/test -type f -name '*.jsonl' | sort | xargs -I {} cat {} >> $PATH_TO_SLIM_PAJAMA/export/test.jsonl

split --line-bytes=43G --additional-suffix=.jsonl -d -a 4 $PATH_TO_SLIM_PAJAMA/export/train.jsonl $PATH_TO_SLIM_PAJAMA/export/train_

azcopy copy ../RAW_DATA_FOLDER/SlimPajama-627B "https://allamllmuksstandard.blob.core.windows.net/llm-data/?sv=2023-01-03&ss=btqf&srt=sco&st=2023-12-03T21%3A43%3A17Z&se=2025-05-30T21%3A43%3A00Z&sp=rwdxftlacup&sig=51chDeU9Xqnk7GrTSA3u2gEfdgCUQIq9SDAJEQCPQxE%3D"  --recursive --overwrite=false

PATH_TO_SLIM_PAJAMA="../RAW_DATA_FOLDER/SlimPajama-627B"

mkdir -p "$PATH_TO_SLIM_PAJAMA/data_by_domain"
python examples/data-processing/sample_json_data_from_meta.py \
--input-folder-path "$PATH_TO_SLIM_PAJAMA/export/" \
--output-folder-path "$PATH_TO_SLIM_PAJAMA/data_by_domain" \
--meta-keys 'meta' 'redpajama_set_name'

cd $PATH_TO_SLIM_PAJAMA/data_by_domain
mkdir -p ../merged_shards

split --line-bytes=85G --additional-suffix=.jsonl -d -a 4 RedPajamaC4.jsonl ../merged_shards/en_SlimPajamaC4_
split --line-bytes=85G --additional-suffix=.jsonl -d -a 4 RedPajamaStackExchange.jsonl ../merged_shards/en_SlimPajamaStackExchange_
split --line-bytes=70G --additional-suffix=.jsonl -d -a 4 RedPajamaGithub.jsonl ../merged_shards/en_SlimPajamaGithub_
split --line-bytes=70G --additional-suffix=.jsonl -d -a 4 RedPajamaWikipedia.jsonl ../merged_shards/en_SlimPajamaWikipedia_
split --line-bytes=70G --additional-suffix=.jsonl -d -a 4 RedPajamaBook.jsonl ../merged_shards/en_SlimPajamaBook_
split --line-bytes=85G --additional-suffix=.jsonl -d -a 4 RedPajamaCommonCrawl.jsonl ../merged_shards/en_SlimPajamaCommonCrawl_
split --line-bytes=85G --additional-suffix=.jsonl -d -a 4 RedPajamaArXiv.jsonl ../merged_shards/en_SlimPajamaArXiv_

cd ../../
azcopy copy SlimPajama-627B/merged_shards "https://allamllmuksstandard.blob.core.windows.net/llm-data/SlimPajama-627B/?sv=2023-01-03&ss=btqf&srt=sco&st=2023-12-03T21%3A43%3A17Z&se=2025-05-30T21%3A43%3A00Z&sp=rwdxftlacup&sig=51chDeU9Xqnk7GrTSA3u2gEfdgCUQIq9SDAJEQCPQxE%3D" --recursive --overwrite=false

