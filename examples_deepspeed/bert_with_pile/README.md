This ```bert_with_pile``` folder includes examples about BERT pre-training (using [the public Pile data](https://github.com/EleutherAI/the-pile) or user's own data) with DeepSpeed integration. We also provide scripts about preprocessing Pile data and MNLI finetuning.

## Data preprocessing
```prepare_pile_data.py``` is the script for downloading, decompressing, and preprocessing [the public Pile data](https://github.com/EleutherAI/the-pile). Users can also modify this script to preprocess their own training data.

## BERT pre-training
```ds_pretrain_bert.sh``` is the script for BERT pre-training integrated with DeepSpeed, supporting [ZeRO](https://www.deepspeed.ai/tutorials/zero/) together with Megatron's tensor-slicing model parallelism. The training hyperparameters follow the [Megatron paper](https://arxiv.org/abs/1909.08053). Note that the pipeline parallelism is currently not supported: DeepSpeed's pipeline parallelism is only integrated with the GPT case, and currently DeepSpeed is not integrated with Megatron's own pipeline parallelism.

As a reference performance number, our measurements show that our example is able to achieve a throughput up to 145 TFLOPs per GPU when pre-training a 1.3B BERT model (with ZeRO stage-1, without model parallelism, with 64 NVIDIA A100 GPUs, with batch size 4096 (64 per GPU), with activation checkpointing).

One thing to note is that this pre-training recipe is NOT a strict reproduction of the [original BERT paper](https://arxiv.org/abs/1810.04805): the Pile data is larger than the data used in original BERT (and the data used by Megatron paper); Megatron-LM introduces some changes to the BERT model (see details in [Megatron paper](https://arxiv.org/abs/1909.08053)); the training hyperparameters are also different. Overall these differences lead to longer training time but also better model quality than original BERT (see MNLI score below), and supporting large model scale by the combination of ZeRO and model parallelism. If you don't have enough computation budget, we recommend to reduce the total training iterations (```train_iters``` in the script) and potentially increase the learning rate at the same time. If you want to strictly reproduce original BERT, we recommend to use our [another BERT example](https://github.com/microsoft/DeepSpeedExamples/tree/master/bing_bert).

## BERT MNLI fine-tuning
```ds_finetune_bert_mnli.sh``` is the script for BERT MNLI fine-tuning, following the hyperparameters in the [Megatron paper](https://arxiv.org/abs/1909.08053). As a reference, table below present the scores using the model pre-trained based on the script above, comparing with the scores of original BERT and Megatron paper's BERT. Our BERT-Large's score is slightly lower than Megatron paper's, mainly due to the different data we used (Pile data is much diverse and larger than the data in Megatron paper, which potentially has negative effect on small million-scale models).

| MNLI dev set accuracy | **MNLI-m** | **MNLI-mm** |
| ---------- |---------- |---------- |
| BERT-Base, [original BERT](https://arxiv.org/abs/1810.04805) | 84.6 | 83.4 |
| BERT-Base, ours (median on 5 seeds) | 86.1 | 86.1 |
| BERT-Large, [original BERT](https://arxiv.org/abs/1810.04805) | 86.7 | 85.9 |
| BERT-Large, [Megatron paper](https://arxiv.org/abs/1909.08053) | 89.7 | 90.0 |
| BERT-Large, ours (median on 5 seeds) | 89.1 | 89.6 |

