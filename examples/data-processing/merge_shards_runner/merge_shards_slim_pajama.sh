
NUM_SIZE=4348654387

bash examples/data-processing/merge_shards_runner/slim_pajama_process.sh

azcopy copy "https://allamllmuksstandard.blob.core.windows.net/vocab-expanded-training-data/SlimPajama-627B/export/?sv=2023-01-03&ss=btqf&srt=sco&st=2023-12-02T21%3A07%3A05Z&se=2023-12-03T21%3A07%3A05Z&sp=rwdxftlacup&sig=xaz92d%2B4V7bJO8jcDhm1BbybLMsA4j%2FL6S0XO4XJgoA%3D" ../RAW_DATA_FOLDER/SlimPajama-627B/ --recursive --overwrite=false

SLIM_PAJAMA_ROOT="../RAW_DATA_FOLDER/SlimPajama-627B"

mkdir -p "$SLIM_PAJAMA_ROOT/data_by_domain"
python examples/data-processing/sample_json_data_from_meta.py \
--input-folder-path "$SLIM_PAJAMA_ROOT/export/" \
--output-folder-path "$SLIM_PAJAMA_ROOT/data_by_domain" \
--meta-keys 'meta' 'redpajama_set_name'

cd $SLIM_PAJAMA_ROOT/data_by_domain
split --line-bytes=85G --additional-suffix=.jsonl -d -a 4 RedPajamaCommonCrawl.jsonl ../merged_shards/RedPajamaC4_
split --line-bytes=70G --additional-suffix=.jsonl -d -a 4 RedPajamaGithub.jsonl ../merged_shards/RedPajamaGithub_
split --line-bytes=70G --additional-suffix=.jsonl -d -a 4 RedPajamaWikipedia.jsonl ../merged_shards/RedPajamaWikipedia_
split --line-bytes=70G --additional-suffix=.jsonl -d -a 4 RedPajamaBook.jsonl ../merged_shards/RedPajamaBook_
split --line-bytes=85G --additional-suffix=.jsonl -d -a 4 RedPajamaCommonCrawl.jsonl ../merged_shards/RedPajamaCommonCrawl_

cp RedPajamaArXiv.jsonl $SLIM_PAJAMA_ROOT/merged_shards/RedPajamaArXiv_0001.jsonl

cd $SLIM_PAJAMA_ROOT/merged_shards
ls -l | awk '{print $9}' | xargs -I {} mv {} en_{}

azcopy copy SlimPajama-627B "https://allamllmuksstandard.blob.core.windows.net/llm-data/?sv=2023-01-03&ss=btqf&srt=sco&st=2023-12-03T21%3A43%3A17Z&se=2025-05-30T21%3A43%3A00Z&sp=rwdxftlacup&sig=51chDeU9Xqnk7GrTSA3u2gEfdgCUQIq9SDAJEQCPQxE%3D" --recursive --overwrite=false

