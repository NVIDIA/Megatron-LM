
NUM_SIZE=4348654387

bash examples/data-processing/merge_shards_runner/slim_pajama_process.sh

azcopy copy "https://allamllmuksstandard.blob.core.windows.net/vocab-expanded-training-data/SlimPajama-627B/export/?sv=2023-01-03&ss=btqf&srt=sco&st=2023-12-02T21%3A07%3A05Z&se=2023-12-03T21%3A07%3A05Z&sp=rwdxftlacup&sig=xaz92d%2B4V7bJO8jcDhm1BbybLMsA4j%2FL6S0XO4XJgoA%3D" ../RAW_DATA_FOLDER/SlimPajama-627B/ --recursive --overwrite=false

mkdir -p "../RAW_DATA_FOLDER/SlimPajama-627B/data_by_domain"
python examples/data-processing/sample_json_data_from_meta.py \
--input-folder-path "../RAW_DATA_FOLDER/SlimPajama-627B/export/" \
--output-folder-path "../RAW_DATA_FOLDER/SlimPajama-627B/data_by_domain" \
--meta-keys 'meta' 'redpajama_set_name'

cd ../RAW_DATA_FOLDER/SlimPajama-627B/data_by_domain
split --line-bytes=85G --additional-suffix=.jsonl -d -a 4 RedPajamaCommonCrawl.jsonl ../shards/RedPajamaC4_
split --line-bytes=70G --additional-suffix=.jsonl -d -a 4 RedPajamaGithub.jsonl ../shards/RedPajamaGithub.jsonl
split --line-bytes=70G --additional-suffix=.jsonl -d -a 4 RedPajamaWikipedia.jsonl ../shards/RedPajamaWikipedia_
split --line-bytes=70G --additional-suffix=.jsonl -d -a 4 RedPajamaBook.jsonl ../shards/RedPajamaBook_
split --line-bytes=85G --additional-suffix=.jsonl -d -a 4 RedPajamaCommonCrawl.jsonl ../shards/RedPajamaCommonCrawl_