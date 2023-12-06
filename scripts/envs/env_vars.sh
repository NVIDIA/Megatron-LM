export DUMPED_FOLDER="../DUMPED"
export RAW_DATA_FOLDER="../RAW_DATA_FOLDER"
export CACHE_DATA_FOLDER="../cache"
export TRANSFORMERS_CACHE="/cache/"

mkdir -p $DUMPED_FOLDER
mkdir -p $RAW_DATA_FOLDER
mkdir -p $CACHE_DATA_FOLDER

git config --global credential.helper store
huggingface-cli login --token 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
