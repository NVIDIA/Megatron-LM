This directory includes GPT-3/BERT pretraining example scripts for DeepSpeed Data Efficiency Library technologies (curriculum learning, random-LTD, and the two composed together).

You need to install updated DeepSpeed version (>=0.8.0), which contains the DeepSpeed Data Efficiency Library.

Additional tutorial can be found at [DeepSpeed website](https://www.deepspeed.ai/tutorials/data-efficiency/).

Additional technical details can be found in our [random-LTD paper](https://arxiv.org/abs/2211.11586) and [data efficiency paper](https://arxiv.org/abs/2212.03597).

## GPT-3 pretraining and evaluation
Inside ``gpt`` folder, first the ``ds_analyze_gpt_data_map.sh`` and ``ds_analyze_gpt_data_reduce.sh`` are used for curriculum learning's offline data analysis and indexing.

``gpt/pretrain`` includes the pretraining example scripts. You can choose a setup to run by uncommenting one block in ``ds_pretrain_gpt_1.3B_dense_run.sh``. One thing to note is that in our [random-LTD paper](https://arxiv.org/abs/2211.11586) we did not scale peak learning rate when using less than 100% data, while in our later [data efficiency paper](https://arxiv.org/abs/2212.03597) we find that scaling LR based on used percentage of data helps improve model quality.

``gpt/eval`` includes the zero-/few-shot evaluation example scripts. ``ds_evalharness_parallel_run.sh`` is for zero-shot, and ``ds_evalharness_parallel_run_10shot.sh`` is for 10-shot.

## BERT pretraining and finetuning
Inside ``bert`` folder, first the ``pile_data_download_preprocess.py`` can be used to download and preprocess the public Pile dataset.

The ``ds_analyze_bert_data_map.sh`` and ``ds_analyze_bert_data_reduce.sh`` are used for curriculum learning's offline data analysis and indexing.

``bert/pretrain`` includes the pretraining example scripts. You can choose a setup to run by uncommenting one block in ``ds_pretrain_bert_336M_run.sh``. One thing to note is that in our [random-LTD paper](https://arxiv.org/abs/2211.11586) we did not scale peak learning rate when using less than 100% data, while in our later [data efficiency paper](https://arxiv.org/abs/2212.03597) we find that scaling LR based on used percentage of data helps improve model quality.

``bert/finetune`` includes the MNLI/QQP/RACE finetuning example scripts following the [Megatron-LM paper](https://arxiv.org/abs/1909.08053). However, we found that the RACE task's accuracy is not very stable and the Megatron-LM paper used a very long number of epochs for MNLI/QQP which is not necessary. Thus we added capability of finetuning other GLUE tasks, and switched to follow the hyperparameters of the [original BERT paper](https://arxiv.org/abs/1810.04805). The corresponding scripts are at ``bert/finetune_glue``, which we recommend to use instead of ``bert/finetune``. Our [data efficiency paper](https://arxiv.org/abs/2212.03597) also uses the scripts under ``bert/finetune_glue`` for GLUE finetuning.