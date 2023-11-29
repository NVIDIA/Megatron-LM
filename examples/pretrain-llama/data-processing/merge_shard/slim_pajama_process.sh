PATH_TO_SLIM_PAJAMA='../RAW_DATA_FOLDER/SlimPajama-627B'
rm -rf ../RAW_DATA_FOLDER/SlimPajama-627B/export/
mkdir -p $PATH_TO_SLIM_PAJAMA/export/

find $PATH_TO_SLIM_PAJAMA/train -type f -name '*.jsonl' | sort | xargs -I {} cat {} >> $PATH_TO_SLIM_PAJAMA/export/train.jsonl
find $PATH_TO_SLIM_PAJAMA/validation -type f -name '*.jsonl' | sort | xargs -I {} cat {} >> $PATH_TO_SLIM_PAJAMA/export/validation.jsonl
find $PATH_TO_SLIM_PAJAMA/test -type f -name '*.jsonl' | sort | xargs -I {} cat {} >> $PATH_TO_SLIM_PAJAMA/export/test.jsonl

split --line-bytes=43G --additional-suffix=.jsonl -d -a 4 $PATH_TO_SLIM_PAJAMA/export/train.jsonl $PATH_TO_SLIM_PAJAMA/export/train_
rm $PATH_TO_SLIM_PAJAMA/export/train.jsonl
