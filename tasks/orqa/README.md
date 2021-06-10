## End-to-End Training of Neural Retrievers for Open-Domain Question Answering

Below we present the steps to run unsupervised and supervised trainining and evaluation of the retriever for [open domain question answering](https://arxiv.org/abs/2101.00408).

### Unsupervised pretraining
1. Use `tools/preprocess_data.py` to preprocess the dataset for Inverse Cloze Task (ICT) task, which we call unsupervised pretraining. This script takes as input a corpus in loose JSON format and creates fixed-size blocks of text as the fundamental units of data. For a corpus like Wikipedia, this will mean multiple sentences per block and multiple blocks per document. Run [`tools/preprocess_data.py`](../../tools/preprocess_data.py) to construct one or more indexed datasets with the `--split-sentences` argument to make sentences the basic unit. We construct two datasets, one with the title of every document and another with the body.

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

2. The [`examples/pretrain_ict.sh`](../../examples/pretrain_ict.sh) script runs a single GPU 217M parameter biencoder model for ICT retriever training. Single GPU training is primarily intended for debugging purposes, as the code is developed for distributed training. The script uses a pretrained BERT model with a batch size of 4096 (hence the need for a data parallel world size of 32).

3. Evaluate the pretrained ICT model using [`examples/evaluate_retriever_nq.sh`](../../examples/evaluate_retriever_nq.sh) for natural question answering dataset.

### Supervised finetuning

1. Use the above pretrained ICT model to finetune using [Google's natural question answering dataset](https://ai.google.com/research/NaturalQuestions/). The script [`examples/finetune_retriever_distributed.sh`](../../examples/finetune_retriever_distributed.sh) provides an example for how to do this. Our finetuning consists of score scaling, longer training (80 epochs), and hard negative examples.

2. Evaluate the finetuned model using the same evaluation script as mentioned above for the unsupervised model.

More details on the retriever are available in [our paper](https://arxiv.org/abs/2101.00408).

The reader component will be available soon.
 
