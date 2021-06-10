
## End-to-End Training of Neural Retrievers for Open-Domain Question Answering

We present below the steps on show how to run unsupervised and supervised trainining and evaluation for retriever for [open domain question answering](https://arxiv.org/abs/2101.00408).

### Unsupervised pretraining
1. We use the following to preprocess dataset for Inverse Cloze Task (ICT) task, we call unsupervised pretraining. Having a corpus in loose JSON format with the intension of creating a collection of fixed-size blocks of text as the fundamental units of data. For a corpus like Wikipedia, this will mean multiple sentences per block but also multiple blocks per document. Run [`tools/preprocess_data.py`](../../tools/preprocess_data.py) to construct one or more indexed datasets with the `--split-sentences` argument to make sentences the basic unit. We construct two datasets, one with the title of every document, and another with the body.

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

2. The [`examples/pretrain_ict.sh`](../../examples/pretrain_ict.sh) script runs single GPU 217M parameter biencoder model for ICT retriever training. Single GPU training is primarily intended for debugging purposes, as the code is developed for distributed training. The script uses pretrained BERT model with batch size of 4096 (hence need data parallel world size of 32).

3. Evaluate the pretrained ICT model using [`examples/evaluate_retriever_nq.sh`](../../examples/evaluate_retriever_nq.sh) for natural question answering dataset.

### Supervised finetuning

1. We use the above pretrained ICT model to finetune using [Google's natural question answering dataset](https://ai.google.com/research/NaturalQuestions/). We use the script [`examples/finetune_retriever_distributed.sh`](../../examples/finetune_retriever_distributed.sh) for this purpose. Our finetuning consists of score scaling, longer training (80 epochs), and hard negative examples.

2. We evaluate the finetuned model using the same evaluation script as mentioned above for the unsupervised model.


More details on the retriever are available in [our paper](https://arxiv.org/abs/2101.00408).

The reader component will be available soon.
 
