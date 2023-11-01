# InstructRetro: Instruction Tuning post Retrieval-Augmented Pretraining

InstructRetro is an innovative extension of the large language model (LLM) architecture, aimed at advancing the state of LLM capabilities. By augmenting the pretraining phase with a retrieval mechanism, InstructRetro showcases notable improvements in terms of perplexity and factual accuracy, thus opening new avenues for enhanced instruction tuning and zero-shot generalization.

This README provides an end-to-end tutorial to reproduce InstructRetro.   

## Citations

See more details from our paper:

[Shall we Pretrain Autoregressive Language Models with Retrieval? A Comprehensive Study.](https://arxiv.org/abs/2304.06762)

_Boxin Wang, Wei Ping, Peng Xu, Lawrence McAfee, Zihan Liu, Mohammad Shoeybi, Yi Dong, Oleksii Kuchaiev, Bo Li, Chaowei Xiao, Anima Anandkumar, Bryan Catanzaro._ (EMNLP 2023)

[InstructRetro: Instruction Tuning post Retrieval-Augmented Pretraining.](https://arxiv.org/abs/2310.07713) 

_Boxin Wang, Wei Ping, Lawrence McAfee, Peng Xu, Bo Li, Mohammad Shoeybi, Bryan Catanzaro._ 

Please cite the paper as follows if you use the data or code from this repo:

```bibtex
@inproceedings{wang2023shall,
    title   = {Shall We Pretrain Autoregressive Language Models with Retrieval? A Comprehensive Study},
    author  = {Boxin Wang and Wei Ping and Peng Xu and Lawrence McAfee and Zihan Liu and Mohammad Shoeybi and Yi Dong and Oleksii Kuchaiev and Bo Li and Chaowei Xiao and Anima Anandkumar and Bryan Catanzaro},
    journal = {The 2023 Conference on Empirical Methods in Natural Language Processing},
    year    = {2023}
}

@article{wang2023instructretro,
    title   = {InstructRetro: Instruction Tuning post Retrieval-Augmented Pretraining},
    author  = {Boxin Wang and Wei Ping and Lawrence McAfee and Peng Xu and Bo Li and Mohammad Shoeybi and Bryan Catanzaro},
    year    = {2023},
    journal = {arXiv preprint arXiv: 2310.07713}
}
```

# End-to-end Reproduction Guide

In this README, we provide an end-to-end reproduction guide for InstructRetro, covering from large-scale retrieval construction, pretraining, perplexity evaluation, instruction tuning, to downstream task evaluation. 

## Step 0: Prepare the environment

We recommend using a docker environment  to run the code.

### Docker image

[//]: # (We provide docker images for the reproduction. )

[//]: # ()
[//]: # (```bash)

[//]: # (```)

We provide a [docker build file](https://github.com/NVIDIA/Megatron-LM/blob/main/tools/retro/examples/Dockerfile) for the reproduction. The docker image is based on `nvcr.io/nvidia/pytorch:23.04-py3`.


### Install dependencies

If docker is not available, we recommend start from a clean conda environment, including:
- Python 3.8
- NVIDIA CUDA® 12.1.0
- NVIDIA cuBLAS 12.1.3
- NVIDIA cuDNN 8.9.0
- NVIDIA NCCL 2.17.1 
- PyTorch 2.1.0a0+fe05266f

Then install Retro-specific dependencies, including:
```bash
pip install -U faiss-gpu
pip install -U transformers
pip install -U sentencepiece
pip install -U h5py
pip install -U nltk
pip install -U einops
```



## Step 1: Build retrieval database

In this step, we build a large-scale retrieval database for InstructRetro through [Faiss](https://github.com/facebookresearch/faiss) to retrieve from trillions of tokens, and preprocess (and save) the retrieval neighbors for the pretraining step.

Please refer to [build_db.md]() for more details.

## Step 2: Pretraining

*Please strictly follow the Step 1 to build the retrieval database before pretraining to make sure the preprocessed retrieval neighbors match the pretraining corpus.*

In the pretraining step, we support both pretraining from scratch and continued pretraining from a pretrained GPT model.

We provide a template pretraining script to pretrain 800M Retro from scratch. Prepare your own arguments and update our templates in `tools/retro/examples/pretrain_model.sh`. Please note that the data path should be exactly matching the one used in Step 1 to make sure the preprocessed retrieval neighbors match the pretraining corpus.

[//]: # (Take the example of the Wikipedia corpus)

```bash
bash tools/retro/examples/pretrain_model.sh
```
After pretraining, the model checkpoints will be saved in the `--save` directory if you specified the arg in `pretrain_model.sh`.

To continue pretraining with retrieval from a pretrained GPT model, please specify `--load` in `pretrain_model.sh` to load the pretrained GPT model checkpoint (the architecture of GPT, including hidden size, number of layers, and activation methods, should be exactly the same as the one used for Retro). You should also specify   `--no-load-optim --finetune` to make sure the optimizer state is not loaded from the pretrained GPT model and the continued pretraining with retrieval is from a clean start.

## Step 3: Perplexity evaluation

During pretraining, we will automatically evaluate the model perplexity on the specified validation corpus every `--eval-interval` steps. The validation corpus should be exactly the same as the one used in Step 1 to make sure the preprocessed retrieval neighbors match the pretraining corpus.

To evaluate the perplexity of a pretrained model, please add `--skip-train` in `pretrain_model.sh` to skip the pretraining step and only evaluate the perplexity of the model specified in `--load` on the validation corpus. Run the above command again to evaluate the perplexity of a pretrained model:

```bash
bash tools/retro/examples/pretrain_model.sh
```

## Step 4: Instruction tuning

In this step, we fine-tune the pretrained model on the downstream task with instructions. We provide a template instruction tuning script to fine-tune 800M Retro on an open-source blend of instruction tuning datasets. The dataset is available to download through the Google Drive link. The blendable dataset consists of the following open-source instruction tuning datasets:

### Dataset Breakdown
| Dataset                |Samples|Epochs|Sampling Prob|
|------------------------|------:|-----:|------------:|
| soda                   |      2560 |  0.005| 0.020|
| eli5                   |      1536 |  0.017| 0.012|
| eli5                   |       604 |  0.019| 0.005|
| eli5                   |       421 |  0.019| 0.003|
| self_instruct_short    |      1280 |  0.043| 0.010|
| self_instruct_long     |      2560 |  0.333| 0.020|
| unnatural-instructions |      2560 |  0.024| 0.020|
| flan_cot               |      1280 |  0.093| 0.010|
| dolly                  |      6400 |  0.938| 0.050|
| oasst-skip-noncode     |    104558 |  1.839| 0.817|
| oasst-skip-code        |      4243 |  1.839| 0.033|
### Instruction tuning script
Download the blendable dataset in your data home directory `$DATA_HOME` and update our templates in `tools/retro/sft/sft_retro_lm.sh`.

An example command to run instruction tuning on 800M Retro is as follows:
```bash
                                      [blend-dataset-name] [model-size] [batch-size]  [lr]    [checkpoints]
bash tools/retro/sft/sft_retro_lm.sh         sft               843m            128    5e-6  <path/to/pretrained/retro>  
```

The checkpoints will be saved in the `--save` directory. For example, it will be saved to 
`<SFT_HOME>/checkpoints/applications/retro-sft_pp1_same_format_ctx1_843m_128_5e-6`.

## Step 5: Downstream task evaluation

In this step, we demonstrate how to run InstructRetro for zero-shot evaluation on downstream question answering (QA) tasks. 


```bash
bash tools/retro/text_generation/retro_generate.sh nq 43b greedy test  0 20000 1000 5 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/github-version/retro/Megatron-LM/checkpoints/applications/retro-open_inst_pp1_same_format_ctx1_43b_128_5e-6 2
bash tools/retro/text_generation/retro_generate.sh nq 43b greedy test  0 20000 1000 5 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/github-version/retro/Megatron-LM/checkpoints/applications/retro-qc_pp1_same_format_ctx1_43b_128_5e-6 2
bash tools/retro/text_generation/retro_generate.sh nq 43b greedy test  0 20000 1000 5 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/github-version/retro/Megatron-LM/checkpoints/applications/retro-sft_pp1_same_format_ctx1_43b_128_5e-6 2

bash tools/retro/text_generation/retro_generate.sh nq 843m greedy test  0 20000 500 5 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/github-version/retro/Megatron-LM/checkpoints/applications/retro-sft_pp1_same_format_ctx1_843m_128_5e-6 2
```