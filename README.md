Megatron is a large, powerful transformer. This repo is for ongoing research on training large, powerful transformer language models at scale. Currently, we support multinode training of [BERT](https://arxiv.org/pdf/1810.04805.pdf) in mixed precision. Our codebase is capable of training BERT Large on 64 V100 GPUs in 3 days. We achieved a final language modeling perplexity of 3.15 and SQuAD F1-score of 90.7.

# Setup
We officially support only python3.6.

To use this repo please install the latest supported versions of PyTorch with GPU support. 

Additionally, part of this codebase leverages tensorflow-cpu to perform dataloading of TFRecords. We recommend creating a virtual environment (to avoid breaking existing tf installations) and install our `reuirements.txt`.

```
python -m pip install virtualenv
virtualenv bert_env
source bert_env/bin/activate
pip install -r requirements.txt
```


# Usage
We've provided 4 scripts that pretrain BERT. All saved checkpoints can be used for finetuning according to [existing implementations](https://github.com/huggingface). Save model checkpoints with `--save`.

## BERT Pretraining
`bash scripts/pretrain_bert.sh`

This script runs single gpu BERT pretraining and is mainly for debugging purposes.

To use this script place your `--train-data` in loose json format with one json per line. The text field of your json dictionaries should correspond to `--text-key`. 

```
python pretrain_bert.py \
    --batch-size 4 \
    --tokenizer-type BertWordPieceTokenizer \
    --cache-dir temp_cache_dir \
    --tokenizer-model-type bert-large-uncased \
    --vocab-size 30522 \
    --train-data wikipedia \
    --loose-json \
    --text-key text \
    --split 1000,1,1 \
    --lazy-loader \
    --max-preds-per-seq 80 \
    --seq-length 512 \
    --max-position-embeddings 512 \
    --num-layers 24 \
    --hidden-size 1024 \
    --intermediate-size 4096 \
    --num-attention-heads 16 \
    --hidden-dropout 0.1 \
    --attention-dropout 0.1 \
    --train-iters 1000000 \
    --lr 0.0001 \
    --lr-decay-style linear \
    --lr-decay-iters 990000 \
    --warmup .01 \
    --weight-decay 1e-2 \
    --clip-grad 1.0 \
    --fp16 \
    --fp32-layernorm \
    --fp32-embedding \
    --hysteresis 2 \
    --num-workers 2 
```

## Distributed BERT Pretraining
`bash scripts/pretrain_bert_distributed.sh`

To use this script, follow the same data preparation procedure as in [earlier sections](#bert-pretraining). This script uses the pytorch distributed launcher to launch distributed training. As such, multinode training can be achieved by properly setting environment variables for the `env://` init method. See the official pytorch [documentation](https://pytorch.org/docs/stable/distributed.html#launch-utility) for further description of these [environment variables](https://pytorch.org/docs/stable/distributed.html#environment-variable-initialization). By default multinode training uses the nccl distributed backend.

## Distributed BERT Pretraining with TFRecords
`bash scripts/pretrain_bert_tfrecords_distributed.sh`

This script takes advantage of TensorFlow BERT's [`create_pretraining.py`](https://github.com/NVIDIA/DeepLearningExamples/blob/master/TensorFlow/LanguageModeling/BERT/create_pretraining_data.py) script to pre-cache the dataset in the TFRecord format. To convert the data to pytorch tensors we use a `TFRecordDataset` and tensorflow eager mode to turn the TFRecords into numpy matrices before loading them into pytorch gpu tensors. This greatly reduces the overhead of dataprocessing and speeds up training. Pass a whitespace-separated list of TFRecord paths to `--train-data` and enable the `--use-tfrecords` flag. Multinode training can be achieved as described in the [previous section](#distributed-bert-pretraining).

## Train Custom Sentence Piece Tokenizer and Pretrain BERT
`bash scripts/pretrain_bert_sentencepiece.sh`

This script runs BERT pretraining with a `sentencepiece` tokenizer. If no sentencepiece tokenizer exists at `--tokenizer-path` one will be trained automatically. The sentencepiece tokenizer can be used with the previous scripts (NOTE: sentencepiece training can only happen during single gpu pretraining). `<--tokenizer-path>.vocab` can be used with [`create_pretraining_data.py`](https://github.com/NVIDIA/DeepLearningExamples/blob/master/TensorFlow/LanguageModeling/BERT/create_pretraining_data.py) to make a TFRecord dataset with the given tokenization.


# Collecting Wikipedia Training Data
We recommend following the wikipedia data extraction process specified by google research: "the recommended pre-processing is to download [the latest dump](https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2), extract the text with [WikiExtractor.py](https://github.com/attardi/wikiextractor), and then apply any necessary cleanup to convert it into plain text." 

We recommend using the `--json` argument when using WikiExtractor, which will dump the wikipedia data into loose json format (one json per line), making it more manageable and readily consumable by our codebase.

Once the json dataset is ready make sure to set the path in line 27 of `data_utils/corpora.py`.

If your system is memory limited we also recommend running pretraining with the `--lazy-loader` argument as we've done. After preprocessing the dataset once, this will allow the dataset to be lazily loaded from disk, as opposed to storing it in memory.
