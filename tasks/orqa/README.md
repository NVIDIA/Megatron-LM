We present below the steps on show how to run unsupervised and supervised trainining and evaluation for retriever for [open domain question answering](https://arxiv.org/abs/2101.00408).

## End-to-End Training of Neural Retrievers for Open-Domain Question Answering

We use two stages for retriever pretraining and finetuning, (i) unsupervised pretraining, and (ii) supervised finetuning. 

### Unsupervised pretraining
1. We use the following to preprocess dataset for Inverse Cloze Task (ICT) task, we call unsupervised pretraining. Having a corpus in loose JSON format with the intension of creating a collection of fixed-size blocks of text as the fundamental units of data. For a corpus like Wikipedia, this will mean multiple sentences per block but also multiple blocks per document. Run `tools/preprocess_data.py` to construct one or more indexed datasets with the `--split-sentences` argument to make sentences the basic unit. We construct two datasets, one with the title of every document, and another with the body.

<pre>
python tools/preprocess_data.py \
    --input /path/to/corpus.json \
    --json-keys text title \
    --split-sentences \
    --tokenizer-type BertWordPieceLowerCase \
    --vocab-file /path/to/vocab.txt \
    --output-prefix corpus_indexed \
    --workers 10
</pre>

2. The `examples/pretrain_ict.sh` script runs single GPU 217M parameter biencoder model for ICT retriever training. Single GPU training is primarily intended for debugging purposes, as the code is developed for distributed training. The script uses pretrained BERT model with batch size of 4096 (hence need data parallel world size of 32).

<pre>

PRETRAINED_BERT_PATH="Specify path of pretrained BERT model"
TEXT_DATA_PATH="Specify path and file prefix of the text data"
TITLE_DATA_PATH="Specify path and file prefix od the titles"
CHECKPOINT_PATH="Specify path"

python pretrain_ict.py \
        --num-layers 12 \
        --hidden-size 768 \
        --num-attention-heads 12 \
        --tensor-model-parallel-size 1 \
        --micro-batch-size 32 \
        --seq-length 256 \
        --max-position-embeddings 512 \
        --train-iters 100000 \
        --vocab-file bert-vocab.txt \
        --tokenizer-type BertWordPieceLowerCase \
        --DDP-impl torch \
        --bert-load ${PRETRAINED_BERT_PATH} \
        --log-interval 100 \
        --eval-interval 1000 \
        --eval-iters 10 \
        --retriever-report-topk-accuracies 1 5 10 20 100 \
        --retriever-score-scaling \
        --load $CHECKPOINT_PATH \
        --save $CHECKPOINT_PATH \
        --data-path ${TEXT_DATA_PATH} \
        --titles-data-path ${TITLE_DATA_PATH} \
        --lr 0.0001 \
        --lr-decay-style linear \
        --weight-decay 1e-2 \
        --clip-grad 1.0 \
        --lr-warmup-fraction 0.01 \
        --save-interval 4000 \
        --exit-interval 8000 \
        --query-in-block-prob 0.1 \
        --fp16
</pre>

