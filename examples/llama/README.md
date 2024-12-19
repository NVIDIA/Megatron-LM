# LLaMa for PyTorch

This directory provides examples of the GPT-based LLaMa models training in the Megatron-LM repository on Intel® Gaudi® 2 & Gaudi® 3 AI accelerators.
Before you get started, make sure to review the [Supported Configuration](../../README.md#supported-configuration).

## Table of Contents
* [Setup](#setup)
* [Training Script Settings](#training-script-settings)
* [LLaMA Training and Examples](#llama-training-and-examples)


# Setup
Please follow the instructions provided in the [Intel Gaudi Installation Guide](https://docs.habana.ai/en/latest/Installation_Guide/index.html)
to set up the environment including the `$PYTHON` environment variable. To achieve the best performance, please follow the methods outlined in the [Optimizing Training Platform guide](https://docs.habana.ai/en/latest/PyTorch/Model_Optimization_PyTorch/Optimization_in_Training_Platform.html).
The guides will walk you through the process of setting up your system to run the model on Gaudi 2 & Gaudi 3.

## How to Use
Users bear sole liability and responsibility to follow and comply with any third party licenses, and Habana Labs disclaims and will bear no liability with respect to users’ use or compliance with third party licenses.
* Third-Party Models
  * In the course of using Megatron-LM, users may choose to download models created and distributed by third parties after reviewing background information about the models and agreeing to the license governing those models.
  * Notice: Intel does not create the content and does not warrant its accuracy or quality. By accessing the third-party content, or using materials trained on or with such content, you are indicating your acceptance of the terms associated with that content and warranting that your use complies with the applicable license.
  * Intel expressly disclaims the accuracy, adequacy, or completeness of any such third-party content, and is not liable for any errors, omissions, or defects in the content, or for any reliance on the content. You agree Intel is not liable for any liability or damages relating to your use of third-party content.
  * Intel’s identification of these resources does not expand or otherwise alter Intel’s applicable published warranties or warranty disclaimers for Intel products or solutions, and you agree that no additional obligations, indemnifications, or liabilities arise from Intel identifying such resources. Intel reserves the right, without notice, to make corrections, enhancements, improvements, and other changes to its materials.
  * The table below contains links to the licenses for certain third-party models and detailed information about the capabilities, limitations, and best practices for those models.

    | Model/Component        | Framework         | Mode                | Detailed Information | License |
    | ---------------------- | ----------------- | ------------------- | -------------------- | ------- |
    | LLaMA 2                | PyTorch           | Pretraining         | [Use Policy](https://github.com/meta-llama/llama-models/blob/main/models/llama2/USE_POLICY.md) | [License](https://github.com/meta-llama/llama-models/blob/main/models/llama2/LICENSE) |
    | LLaMA 3.1              | PyTorch           | Pretraining         | [Use Policy](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/USE_POLICY.md) | [License](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/LICENSE) |
    | Meta LLaMA 3 Tokenizer | PyTorch           | Tokenizer Support   | [Use Policy](https://github.com/meta-llama/llama3/blob/main/USE_POLICY.md) | [License](https://github.com/meta-llama/llama3/blob/main/LICENSE) |

## Prerequisites
* When creating Docker container, set the shared memory size as 10 GB through the Docker run command:
  ```bash
  --shm-size=10g
  ```

## Clone Intel Gaudi Megatron-LM
In the Docker container, clone this repository and switch to the branch that matches your Intel Gaudi software version.
You can run the [`hl-smi`](https://docs.habana.ai/en/latest/System_Management_Tools_Guide/System_Management_Tools.html#hl-smi-utility-options) utility to determine the Intel Gaudi software version.
```bash
git clone -b [Intel Gaudi software version] https://github.com/HabanaAI/Megatron-LM
```
Set the required environment variables as shown below:
```
export MEGATRON_LM_ROOT=/path/to/Megatron-LM
export PYTHONPATH=$MEGATRON_LM_ROOT:$PYTHONPATH
```
## Install LLaMA Requirements
* In the Docker container, go to the Megatron-LM directory:
  ```bash
  cd $MEGATRON_LM_ROOT
  ```

* Install the required packages using pip:
  ```bash
  pip install -r megatron/core/requirements.txt
  ```
* Review and accept the [LLaMA 3 tokenizer license conditions](https://github.com/meta-llama/llama3/blob/main/LICENSE) before using it
  ```bash
  pip install -r examples/llama/requirements.txt
  ```

* To run training on more than 128 cards, apply the below configuration changes:
  ```bash
  echo '*    soft nofile  unlimited' >> /etc/security/limits.conf
  echo '*    hard nofile  unlimited' >> /etc/security/limits.conf
  echo 'root soft nofile  unlimited' >> /etc/security/limits.conf
  echo 'root hard nofile  unlimited' >> /etc/security/limits.conf
  ```

## Dataset Preparation
Follow the instructions in https://github.com/bigscience-workshop/bigscience/tree/master/data/oscar to download oscar-en full dataset. Note that the dataset takes around 550G of disk space. This dataset is used for training LLaMA & LLaMA 2.
### Dataset Preparation Example
The below provides the steps required to prepare your dataset. It is based on instructions in https://github.com/bigscience-workshop/bigscience/tree/master/data/oscar. This example uses the `zh` dataset.
### Step 0 :
```bash
pip install -r $MEGATRON_LM_ROOT/examples/llama/requirements_preprocess.txt
git clone https://github.com/bigscience-workshop/bigscience.git
cd bigscience/data/oscar
# Edit the `oscar-to-jsonl.py` in the list language_subsets and remove the comment on unshuffled_deduplicated_zh and comment out unshuffled_deduplicated_en
vi oscar-to-jsonl.py
```
### Step 1 :
```bash
# -s can be added for subset of data
$PYTHON oscar-to-jsonl.py
```
### Step 2 :
  ```bash
mkdir -p zh
mv oscar*.jsonl zh
cd zh
  ```
### Step 3 :
Use one of the three methods below to tokenize the dataset. You can use any number of workers based on the CPU cores.
*  Tokenize the dataset using GPT2BPETokenizer:
    ```bash
    # download gpt2 vocab and merge files
    wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json
    wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt

    # tokenize individual jsonl files
    # loop count will change based on number of files for a given dataset
    mkdir zh_tokenized
    for i in $(seq 0 4);
    do
      $PYTHON $MEGATRON_LM_ROOT/tools/preprocess_data.py --input oscar-${i}.jsonl --output-prefix zh_tokenized/tokenized${i} --tokenizer-type GPT2BPETokenizer --vocab-file gpt2-vocab.json --merge-file gpt2-merges.txt --append-eod --workers 80
    done
    ```
  * Tokenize the dataset using GPTSentencePieceTokenizer:
    ```bash
    # download tokenizer.model based on model trying to train
    # tokenize individual jsonl files
    # loop count will change based on number of files for a given dataset
    mkdir zh_tokenized
    for i in $(seq 0 4);
    do
      $PYTHON $MEGATRON_LM_ROOT/tools/preprocess_data.py --input oscar-${i}.jsonl --output-prefix zh_tokenized/tokenized${i} --tokenizer-type GPTSentencePieceTokenizer --tokenizer-model /path/to/tokenizer.model --append-eod --workers 80
    done
    ```
### Step 4 :
 * Merge multiple tokenized dataset files into a single file using the below method:
    ```bash
    # merge tokenized files
    mkdir zh_tokenized_merged
    $PYTHON $MEGATRON_LM_ROOT/tools/merge_datasets.py --input zh_tokenized --output-prefix zh_tokenized_merged/tokenized_text_document
    # use the tokenized files generated from above command to train
    ```

# Training Script Settings
* Based on the tokenization method, update the tokenizer type:
  ```
  HL_TOKENIZER_TYPE=GPT2BPETokenizer
  ```
* Run custom tokenizer code from local path using HFTokenizer method:
  ```
  HL_TRUST_REMOTE_CODE=1
  ```
* Update data root dir with the path of your choice:
  ```
  HL_DATA_DIR_ROOT=/data/bigscience/oscar-en
  ```
* Update data file prefix(*.bin and *.idx) based on file name in data root dir:
  ```
  HL_DATA_FILE_PREFIX=tokenized_text_document
  ```
* Update tokenizer.model file path if it is not in data root dir, required for any sentence piece based tokenizer:
  ```
  HL_TOKENIZER_MODEL=path/to/tokenizer.model
  ```

Note: For the training commands, make sure to change the IP addresses in hostsfile according to your setup.
`HL_RESULTS_DIR` and `HL_DATA_DIR_ROOT` must be shared writable across all nodes and launchers when running training on more than 8 cards.
The same applies to `HL_CHECKPOINTS_DIR`, `HL_TENSORBOARD_DIR` and `HL_KILL_SWITCH` if specified.
If `HL_DATA_DIR_ROOT` is not writable, then `HL_DATA_CACHE_DIR` must be set to a writable location and
must be shared and accessible across all nodes and launchers when running training on more than 8 cards.


# LLaMA Training and Examples
* Training of LLaMA is based on https://arxiv.org/abs/2302.13971
* Training of LLaMA 2 is based on https://arxiv.org/pdf/2307.09288
* Training of LLaMA 3.1 is based on https://ai.meta.com/research/publications/the-llama-3-herd-of-models/

## Multi-Card Training Examples
### LLaMA 3.1 Recipes
#### Sequence Length 8192
* Run LLaMA 3.1 8B on 8 HPUs with BF16 & FP8 precision:

  ```bash
  # Retain default settings for optimal performance.

  # FP8 config
  HL_FP8=1 \
  HL_TRANSFORMER_IMPL=transformer_engine \
  HL_SEQ_PARALLEL=0 \
  HL_TOKENIZER_TYPE=Llama3Tokenizer \
  HL_CKP_ACT=2 \
  HL_LLAMA_VER=3.1 \
  HL_DP=4 \
  HL_TP=2 \
  HL_PP=1 \
  HL_LLAMA_MODEL_SIZE=8 \
  $MEGATRON_LM_ROOT/examples/llama/pretrain_llama.sh

  # BF16 config
  HL_TOKENIZER_TYPE=Llama3Tokenizer \
  HL_CKP_ACT=2 \
  HL_LLAMA_VER=3.1 \
  HL_DP=4 \
  HL_TP=2 \
  HL_PP=1 \
  HL_LLAMA_MODEL_SIZE=8 \
  $MEGATRON_LM_ROOT/examples/llama/pretrain_llama.sh
  ```

* Run LLaMA 3.1 70B on 64 HPUs with BF16 & FP8 precision:
  ```bash
  # Retain default settings for optimal performance.

  # FP8 config
  HL_FP8=1 \
  HL_TRANSFORMER_IMPL=transformer_engine \
  HL_SEQ_PARALLEL=0 \
  HL_TOKENIZER_TYPE=Llama3Tokenizer \
  HL_CKP_ACT=2 \
  HL_NUM_NODES=8 \
  HL_LLAMA_VER=3.1 \
  HL_DP=8 \
  HL_TP=8 \
  HL_PP=1 \
  HL_LLAMA_MODEL_SIZE=70 \
  $MEGATRON_LM_ROOT/examples/llama/pretrain_llama.sh

  # BF16 config
  HL_TOKENIZER_TYPE=Llama3Tokenizer \
  HL_CKP_ACT=2 \
  HL_NUM_NODES=8 \
  HL_LLAMA_VER=3.1 \
  HL_DP=8 \
  HL_TP=8 \
  HL_PP=1 \
  HL_LLAMA_MODEL_SIZE=70 \
  $MEGATRON_LM_ROOT/examples/llama/pretrain_llama.sh

#### Sequence Length 32768
* Run LLaMA 3.1 8B on 8 HPUs with BF16 precision:
  ```bash
  # Retain default settings for optimal performance.

  HL_NUM_WORKERS=0 \
  HL_SEQ_LEN=32768 \
  HL_TOKENIZER_TYPE=Llama3Tokenizer \
  HL_CKP_ACT=2 \
  HL_LLAMA_VER=3.1 \
  HL_LLAMA_MODEL_SIZE=8 \
  HL_DP=2 \
  HL_TP=4 \
  HL_PP=1 \
  $MEGATRON_LM_ROOT/examples/llama/pretrain_llama.sh
  ```

* Run LLaMA 3.1 70B on 32 HPUs with BF16 precision:
  ```bash
  # Retain default settings for optimal performance.

  HL_NUM_WORKERS=0 \
  HL_SEQ_LEN=32768 \
  HL_TOKENIZER_TYPE=Llama3Tokenizer \
  HL_CKP_ACT=1 \
  HL_NUM_NODES=4 \
  HL_LLAMA_VER=3.1 \
  HL_LLAMA_MODEL_SIZE=70 \
  HL_DP=4 \
  HL_TP=8 \
  HL_PP=1 \
  $MEGATRON_LM_ROOT/examples/llama/pretrain_llama.sh

### LLaMA 2 Recipes
* Run LLaMA 2 7B on 8 HPUs with BF16 precision:
  ```
  HL_LLAMA_VER=2 HL_NUM_NODES=1 HL_PP=2 HL_TP=2 HL_DP=2 examples/llama/pretrain_llama.sh
  ```

* Run LLaMA 2 7B on 64 HPUs with BF16 precision:
  ```
  HL_LLAMA_VER=2 HL_HOSTSFILE=examples/hostsfile HL_NUM_NODES=8 HL_PP=2 HL_TP=2 HL_DP=16 examples/llama/pretrain_llama.sh
  ```

* Run LLaMA 2 70B on 32 HPUs with BF16 precision:
  ```
  HL_LLAMA_VER=2 HL_HOSTSFILE=examples/hostsfile HL_LLAMA_MODEL_SIZE=70 HL_NUM_NODES=4 HL_PP=4 HL_TP=8 HL_DP=1 examples/llama/pretrain_llama.sh
  ```


# Known Issues
* Only scripts and configurations mentioned in this README are supported and verified.
* HL_PP > 1 configurations in training with fp8 precision are not supported.
* Using HL_USE_FUSED_SDPA_WITH_RECOMPUTE=1 in training with fp8 precision is not supported.
* Sporadic numerical instability can occur when training with fp8 precision.