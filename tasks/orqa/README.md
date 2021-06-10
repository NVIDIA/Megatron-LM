The following steps show how to run unsupervised and supervised trainining and evaluation for retriever for open domain question answering.

<a id="realm"></a>
## REALM Pipeline
The following sections (will) reflect the three stages of training a REALM system. For now it's just the ICT code.
Loosely, they are pretraining the retriever modules, then jointly training the language model and the retriever, and then finetuning a question answering head on the language model with fixed retriever.

### Inverse Cloze Task (ICT) Pretraining
1. Have a corpus in loose JSON format with the intention of creating a collection of fixed-size blocks of text as the fundamental units of data. For a corpus like Wikipedia, this will mean multiple sentences per block but also multiple blocks per document.
Run `tools/preprocess_data.py` to construct one or more indexed datasets with the `--split-sentences` argument to make sentences the basic unit. For the original REALM system, we construct two datasets, one with the title of every document, and another with the body.
Refer to the following script
<pre>
python preprocess_data.py \
    --input /path/to/corpus.json \
    --json-keys text title \
    --split-sentences \
    --tokenizer-type BertWordPieceLowerCase \
    --vocab-file /path/to/vocab.txt \
    --output-prefix corpus_indexed \
    --workers 5  # works well for 10 CPU cores. Scale up accordingly.
</pre>

2. Use a custom samples mapping function in place of `megatron/data/realm_dataset_utils.get_block_samples_mapping` if required. To do this, you will need to implement a new function in C++ inside of `megatron/data/helpers.cpp`. The samples mapping data structure is used to select the data that will constitute every training sample in advance of the training loop.
 The samples mapping is responsible for holding all of the required metadata needed to construct the sample from one or more indexed datasets. In REALM, the samples mapping contains the start and end sentence indices, as well as the document index (to find the correct title for a body) and a unique ID for every block.
3. Pretrain a BERT language model using `pretrain_bert.py`, with the sequence length equal to the block size in token ids. This model should be trained on the same indexed dataset that is used to supply the blocks for the information retrieval task.
In REALM, this is an uncased bert base model trained with the standard hyperparameters.
4. Use `pretrain_ict.py` to train an `ICTBertModel` which uses two BERT-based encoders to encode queries and blocks to perform retrieval with.
The script below trains the ICT model from REALM. It refrences a pretrained BERT model (step 3) in the `--bert-load` argument. The batch size used in the paper is 4096, so this would need to be run with data parallel world size 32.
<pre>
python pretrain_ict.py \
    --num-layers 12 \
    --num-attention-heads 12 \
    --hidden-size 768 \
    --batch-size 128 \
    --seq-length 256 \
    --max-position-embeddings 256 \
    --ict-head-size 128 \
    --train-iters 100000 \
    --checkpoint-activations \
    --bert-load /path/to/pretrained_bert \
    --load checkpoints \
    --save checkpoints \
    --data-path /path/to/indexed_dataset \
    --titles-data-path /path/to/titles_indexed_dataset \
    --vocab-file /path/to/vocab.txt \
    --lr 0.0001 \
    --num-workers 2 \
    --lr-decay-style linear \
    --weight-decay 1e-2 \
    --clip-grad 1.0 \
    --warmup .01 \
    --save-interval 3000 \
    --query-in-block-prob 0.1 \
    --fp16

</pre>

