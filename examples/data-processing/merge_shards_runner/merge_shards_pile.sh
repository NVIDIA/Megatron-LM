
PILE_PATH="../RAW_DATA_FOLDER/the_pile/"
python examples/data-processing/sample_json_data_from_meta.py \
--input-folder-path "$PILE_PATH/data/" \
--output-folder-path "$PILE_PATH/data_by_domain" \
--meta-keys meta pile_set_name

for file in $PILE_PATH/data_by_domain/*; do mv "$file" "${file// /_}"; done
cd $PILE_PATH/data_by_domain
for file in *; do mv "$file" "${file//_/-}"; done
cd -


ls -l $PILE_PATH/data/* | awk '{print $9}' | xargs -I {}  wc -l "{}" | awk '{sum += $1} END {print sum}' # 210607728
ls -l $PILE_PATH/data_by_domain/* | awk '{print $9}' | xargs -I {}  wc -l "{}" | awk '{sum += $1} END {print sum}' # 210607728

mkdir -p $PILE_PATH/meged_shards/
mkdir -p $PILE_PATH/temp/

split --line-bytes=85G --additional-suffix=.jsonl -d -a 4 $PILE_PATH/data_by_domain/ArXiv.jsonl $PILE_PATH/meged_shards/Pile-ArXiv_
cat $PILE_PATH/data_by_domain/BookCorpus2.jsonl > $PILE_PATH/temp/books2-books3-pgtbrg.jsonl
cat $PILE_PATH/data_by_domain/Books3.jsonl >> $PILE_PATH/temp/books2-books3-pgtbrg.jsonl
cat $PILE_PATH"/data_by_domain/Gutenberg-(PG-19).jsonl" >> $PILE_PATH/temp/books2-books3-pgtbrg.jsonl
split --line-bytes=85G --additional-suffix=.jsonl -d -a 4 $PILE_PATH/temp/books2-books3-pgtbrg.jsonl $PILE_PATH/meged_shards/Pile-books2-books3-pgtbrg_

cat $PILE_PATH/data_by_domain/Enron-Emails.jsonl > $PILE_PATH/temp/enron-hn-nexpt-phil-ubuntu-yt-ep.jsonl
cat $PILE_PATH/data_by_domain/HackerNews.jsonl > $PILE_PATH/temp/enron-hn-nexpt-phil-ubuntu-y-ep.jsonl
cat $PILE_PATH/data_by_domain/NIH-ExPorter.jsonl > $PILE_PATH/temp/enron-hn-nexpt-phil-ubuntu-yt-ep.jsonl
cat $PILE_PATH/data_by_domain/PhilPapers.jsonl > $PILE_PATH/temp/enron-hn-nexpt-phil-ubuntu-yt-ep.jsonl
cat $PILE_PATH/data_by_domain/Ubuntu-IRC.jsonl > $PILE_PATH/temp/enron-hn-nexpt-phil-ubuntu-yt-ep.jsonl
cat $PILE_PATH/data_by_domain/YoutubeSubtitles.jsonl > $PILE_PATH/temp/enron-hn-nexpt-phil-ubuntu-yt-ep.jsonl
cat $PILE_PATH/data_by_domain/EuroParl.jsonl > $PILE_PATH/temp/enron-hn-nexpt-phil-ubuntu-yt-ep.jsonl
split --line-bytes=85G --additional-suffix=.jsonl -d -a 4 $PILE_PATH/temp/enron-hn-nexpt-phil-ubuntu-yt-ep.jsonl $PILE_PATH/meged_shards/Pile-enron-hn-nexpt-phil-ubuntu-yt-ep_

split --line-bytes=85G --additional-suffix=.jsonl -d -a 4 $PILE_PATH/data_by_domain/DM-Mathematics.jsonl $PILE_PATH/meged_shards/Pile-DM-Mathematics_
split --line-bytes=85G --additional-suffix=.jsonl -d -a 4 $PILE_PATH/data_by_domain/FreeLaw.jsonl $PILE_PATH/meged_shards/Pile-FreeLaw_
split --line-bytes=85G --additional-suffix=.jsonl -d -a 4 $PILE_PATH/data_by_domain/Github.jsonl $PILE_PATH/meged_shards/Pile-Github_
split --line-bytes=85G --additional-suffix=.jsonl -d -a 4 $PILE_PATH/data_by_domain/OpenSubtitles.jsonl $PILE_PATH/meged_shards/Pile-OpenSubtitles_
split --line-bytes=85G --additional-suffix=.jsonl -d -a 4 $PILE_PATH/data_by_domain/OpenWebText2.jsonl $PILE_PATH/meged_shards/Pile-OpenWebText2_
split --line-bytes=85G --additional-suffix=.jsonl -d -a 4 $PILE_PATH/data_by_domain/Pile-CC.jsonl $PILE_PATH/meged_shards/Pile-CC_
split --line-bytes=85G --additional-suffix=.jsonl -d -a 4 $PILE_PATH/data_by_domain/PubMed-Abstracts.jsonl $PILE_PATH/meged_shards/Pile-PubMed-Abstracts_
split --line-bytes=85G --additional-suffix=.jsonl -d -a 4 $PILE_PATH/data_by_domain/PubMed-Central.jsonl $PILE_PATH/meged_shards/Pile-PubMed-Central_
split --line-bytes=85G --additional-suffix=.jsonl -d -a 4 $PILE_PATH/data_by_domain/StackExchange.jsonl $PILE_PATH/meged_shards/Pile-StackExchange_
split --line-bytes=85G --additional-suffix=.jsonl -d -a 4 $PILE_PATH/data_by_domain/USPTO-Backgrounds.jsonl $PILE_PATH/meged_shards/Pile-USPTO-Backgrounds_
split --line-bytes=85G --additional-suffix=.jsonl -d -a 4 $PILE_PATH"/data_by_domain/Wikipedia-(en).jsonl" $PILE_PATH/meged_shards/Pile-Wikipedia-en_