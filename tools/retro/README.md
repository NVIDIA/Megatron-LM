# Retro and InstructRetro

Retro [(Borgeaud et al., 2022)](https://arxiv.org/abs/2112.04426) is an autoregressive decoder-only language model (LM) pretrained with retrieval-augmentation. 
Retro features practical scalibility to support large-scale pretraining from scratch by retrieving from trillions of token.
Pretraining with retrieval provides a more efficient storage mechanism of factual knowledge, when compared to storing factual knowledge implicitly within the network's parameters, thus largely reducing model parameters while achieving lower perplexity than standard GPT. 
Retro also provides the flexibility to update the
knowledge stored in LMs [(Wang et al., 2023a)](https://arxiv.org/abs/2304.06762)
by updating the retrieval database without training LMs again.

InstructRetro [(Wang et al., 2023b)](https://arxiv.org/abs/2310.07713) further scales up the size of Retro to 48B, featuring the largest LLM pretrained with retrieval (as of December 2023). 
The obtained foundation model, Retro 48B, largely outperforms the GPT counterpart in terms of perplexity.
With instruction tuning on Retro, InstructRetro demonstrates significant improvement over the instruction tuned GPT on downstream tasks in the zero-shot setting. Specifically, the average improvement of InstructRetro is 7% over its GPT counterpart across 8 short-form QA tasks, and 10% over GPT across 4 challenging long-form QA tasks. We also find that one can ablate the encoder from InstructRetro architecture and directly use the InstructRetro decoder backbone as GPT, while achieving comparable results.

This README provides an end-to-end tutorial to reproduce Retro and InstructRetro.

# Contents
  * [End-to-end Reproduction Guide](#end-to-end-reproduction-guide)
     * [Step 0: Prepare the environment](#step-0-prepare-the-environment)
        * [Docker image](#docker-image)
        * [Install dependencies](#install-dependencies)
     * [Step 1: Build retrieval database](#step-1-build-retrieval-database)
     * [Step 2: Pretraining](#step-2-pretraining)
     * [Step 3: Perplexity evaluation](#step-3-perplexity-evaluation)
     * [Step 4: Instruction tuning](#step-4-instruction-tuning)
     * [Step 5: Downstream task evaluation](#step-5-downstream-task-evaluation)
  * [Citations](#citations)

# End-to-end Reproduction Guide

In this README, we provide an end-to-end reproduction guide for InstructRetro, covering from large-scale retrieval construction, pretraining, perplexity evaluation, instruction tuning, to downstream task evaluation. 


## Step 0: Prepare the environment

We recommend using docker environment to run the code.

### Docker image


We provide a docker build file in [tools/retro/examples/Dockerfile](examples/Dockerfile) for the reproduction. The docker image is based on `nvcr.io/nvidia/pytorch:23.09-py3`.


### Install dependencies

Clone the Megatron repo:

```bash
git clone --branch InstructRetro https://github.com/NVIDIA/Megatron-LM.git
```

If docker is not available, we recommend starting from a clean conda environment with the following runtime dependencies:

- Python 3.10
- NVIDIA CUDA® 12.2.1
- NVIDIA cuBLAS 12.2.5.6
- NVIDIA cuDNN 8.9.5
- NVIDIA NCCL 2.18.5
- PyTorch 2.1.0a0+32f93b1

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

Please refer to [tools/retro/build_db.md](build_db.md) for more details.

## Step 2: Pretraining

*Please strictly follow Step 1 to build the retrieval database before pretraining to make sure the preprocessed retrieval neighbors match the pretraining corpus.*

In the pretraining step, we support both pretraining from scratch and continued pretraining from a pretrained GPT model.

We provide a template pretraining script to pretrain 843M Retro from scratch. Prepare your own arguments and update our templates in [tools/retro/examples/pretrain_model.sh](examples/pretrain_model.sh). Please note that the data path should be exactly matching the one used in Step 1 to make sure the preprocessed retrieval neighbors match the pretraining corpus.

[//]: # (Take the example of the Wikipedia corpus)

```bash
bash tools/retro/examples/pretrain_model.sh
```
After pretraining, the model checkpoints will be saved in the `--save` directory if you specified the arg in `pretrain_model.sh`.

To continue pretraining with retrieval from a pretrained GPT model, please specify `--load` in `pretrain_model.sh` to load the pretrained GPT model checkpoint (the architecture of GPT, including hidden size, number of layers, and activation methods, should be exactly the same as the one used for Retro). You should also specify  `--no-load-optim --finetune` to make sure the optimizer state is not loaded from the pretrained GPT model and the continued pretraining with retrieval is from a clean start. After the first job / the first run, you will continue pretraining with retrieval from your last checkpoint. In the follow-up jobs, you should launch the pretraining without the flags `--no-load-optim --finetune` to make sure the optimizer state is correctly loaded from your last job.


## Step 3: Perplexity evaluation

During pretraining, we will automatically evaluate the model perplexity on the specified validation corpus every `--eval-interval` steps. The validation corpus should be exactly the same as the one used in Step 1 to make sure the preprocessed retrieval neighbors match the pretraining corpus.

To evaluate the perplexity of a pretrained model, please add `--skip-train` in `pretrain_model.sh` to skip the pretraining step and only evaluate the perplexity of the model specified in `--load` on the validation corpus. Run the above command again to evaluate the perplexity of a pretrained model:

```bash
bash tools/retro/examples/pretrain_model.sh
```

## Step 4: Instruction tuning

In this step, we fine-tune the pretrained model on the downstream task with instructions. We provide a template instruction tuning script to fine-tune 843M Retro.

We also provide an open-source blend of instruction tuning datasets. The dataset is available to download through [here](https://drive.google.com/file/d/1nzKwwYf8lYb9gN3P4YO8pFNU_B2nMYe1/view?usp=sharing). The blendable dataset consists of the following open-source instruction tuning datasets:

### Instruction Tuning Dataset Breakdown
| Dataset                                                    | Samples | Epochs | Sampling Prob |
|------------------------------------------------------------|--------:|-------:|--------------:|
| [soda](https://arxiv.org/abs/2212.10465)                   |    2560 |  0.005 |         0.020 |
| [eli5](https://arxiv.org/abs/1907.09190)                   |    2561 |  0.055 |         0.020 |
| [self_instruct_short](https://arxiv.org/abs/2212.10560)    |    1280 |  0.043 |         0.010 |
| [self_instruct_long](https://arxiv.org/abs/2212.10560)     |    2560 |  0.333 |         0.020 |
| [unnatural-instructions](https://arxiv.org/abs/2212.09689) |    2560 |  0.024 |         0.020 |
| [flan_cot](https://arxiv.org/abs/2210.11416)               |    1280 |  0.093 |         0.010 |
| [dolly](https://arxiv.org/abs/2305.13735)                  |    6400 |  0.938 |         0.050 |
| [oasst-skip-noncode](https://open-assistant.io/)           |  104558 |  1.839 |         0.817 |
| [oasst-skip-code](https://open-assistant.io/)              |    4243 |  1.839 |         0.033 |

Refer to the paper links above for more details about each instruction tuning dataset.

*We note that the provided instruction tuning dataset is all from open-source instruction tuning datasets. It is slightly different from what we use in [InstructRetro](https://arxiv.org/abs/2310.07713), which contains private and proprietary datasets. Thus a 1-2% accuracy difference in downstream tasks may be expected.*  

### Instruction tuning script
Download the [blended instruction tuning dataset](https://drive.google.com/file/d/1nzKwwYf8lYb9gN3P4YO8pFNU_B2nMYe1/view?usp=sharing) in your data home directory `$DATA_HOME` and update our templates in [tools/retro/sft/sft_retro_lm.sh](sft/sft_retro_lm.sh).

An example command to run instruction tuning on 843M Retro is as follows:
```bash
                                      [blend-dataset-name] [model-size] [batch-size]  [lr]    [checkpoints]
bash tools/retro/sft/sft_retro_lm.sh       open_inst               843m            128    5e-6  <path/to/pretrained/retro>  
```

The `blend_dataset_name` argument will blend all the datasets within the `$DATA_HOME` following the weights and configurations specified in the `${blend_dataset_name}.sh` ([open_inst.sh](sft/open_inst.sh) in the example above).
The checkpoints will be saved in the `--save` directory. For example, it will be saved to 
`<SFT_HOME>/checkpoints/applications/retro-sft_pp1_same_format_ctx1_843m_128_5e-6`. 

## Step 5: Downstream task evaluation

In this step, we demonstrate how to run InstructRetro for zero-shot evaluation on downstream question answering (QA) tasks. 

We present an example command to run retro generation given the InstructRetro checkpoints and the Natural Question (NQ) task. The example command is for the 843m InstructRetro obtained in Step 4. Please specify the directory for the NQ dataset and update the command accordingly for other checkpoints.  

```bash
bash tools/retro/text_generation/retro_generate.sh nq 843m greedy test  0 20000 1000 5 pp1 <SFT_HOME>/checkpoints/applications/retro-sft_pp1_same_format_ctx1_843m_128_5e-6 2
```

The generated responses will be saved in the corresponding checkpoint directory. For example, for the 843m InstructRetro, it will be saved to 
`<SFT_HOME>/checkpoints/applications/retro-sft_pp1_same_format_ctx1_843m_128_5e-6/retro-generate-nq_5_2_843m_test_greedy_0_20000_1000.txt`.

To evaluate the F1 / Exact Match (EM) scores of the generated responses, we provide an example script to run the evaluation on the NQ dataset. Please specify the directory for the NQ dataset and update the command accordingly for other checkpoints and downstream tasks.  

```bash
python3 tools/retro/text_generation/evaluate.py
```

# Citations

See more details from our papers:

[Shall we Pretrain Autoregressive Language Models with Retrieval? A Comprehensive Study.](https://arxiv.org/abs/2304.06762)

_Boxin Wang, Wei Ping, Peng Xu, Lawrence McAfee, Zihan Liu, Mohammad Shoeybi, Yi Dong, Oleksii Kuchaiev, Bo Li, Chaowei Xiao, Anima Anandkumar, Bryan Catanzaro._ (EMNLP 2023)

[InstructRetro: Instruction Tuning post Retrieval-Augmented Pretraining.](https://arxiv.org/abs/2310.07713) 

_Boxin Wang, Wei Ping, Lawrence McAfee, Peng Xu, Bo Li, Mohammad Shoeybi, Bryan Catanzaro._ 

Please cite the papers as follows if you use the data or code from this repo:

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
