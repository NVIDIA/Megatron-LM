# Mixtral for PyTorch

This directory provides examples of the GPT-based Mixtral models training in the Megatron-LM repository on Intel® Gaudi® 2 AI accelerator.
Before you get started, make sure to review the [Supported Configuration](../../README.md#supported-configuration).

## Table of Contents
* [Setup](#setup)
* [Training Script Settings](#training-script-settings)
* [Mixtral Training and Examples](#mixtral-training-and-examples)


# Setup
Please follow the instructions provided in the [Intel Gaudi Installation Guide](https://docs.habana.ai/en/latest/Installation_Guide/index.html)
to set up the environment including the `$PYTHON` environment variable. To achieve the best performance, please follow the methods outlined in the [Optimizing Training Platform guide](https://docs.habana.ai/en/latest/PyTorch/Model_Optimization_PyTorch/Optimization_in_Training_Platform.html).
The guides will walk you through the process of setting up your system to run the model on Gaudi 2.

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
    | Mixtral 8x7B                | PyTorch           | Pretraining         | [Model Card](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1) | [License](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md) |

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
## Install Mixtral Requirements
* In the Docker container, go to the Megatron-LM directory:
  ```bash
  cd $MEGATRON_LM_ROOT
  ```

* Install the required packages using pip:
  ```bash
  pip install -r megatron/core/requirements.txt
  pip install -r examples/mixtral/requirements.txt
  ```

* To run training on more than 128 cards, apply the below configuration changes:
  ```bash
  echo '*    soft nofile  unlimited' >> /etc/security/limits.conf
  echo '*    hard nofile  unlimited' >> /etc/security/limits.conf
  echo 'root soft nofile  unlimited' >> /etc/security/limits.conf
  echo 'root hard nofile  unlimited' >> /etc/security/limits.conf
  ```

## Dataset
Follow the instructions in https://github.com/togethercomputer/RedPajama-Data/tree/main to recreate RedPajama dataset.


# Training Script Settings
* Based on the tokenization method, update the tokenizer type:
  ```
  HL_TOKENIZER_TYPE=GPTSentencePieceTokenizer
  ```
* Update data root dir with the path of your choice:
  ```
  HL_DATA_DIR_ROOT=/data/bigscience/red_pajama
  ```
* Update data file prefix(*.bin and *.idx) based on file name in data root dir:
  ```
  HL_DATA_FILE_PREFIX=sample
  ```
* Update tokenizer.model file path if it is not in data root dir, required for any sentence piece based tokenizer:
  ```
  HL_TOKENIZER_MODEL=path/to/tokenizer.model
  ```

Note: For the training commands, make sure to change the IP addresses in hostsfile according to your setup.
`HL_RESULTS_DIR` and `HL_DATA_DIR_ROOT` must be shared writable across all nodes and launchers when running training on more than 8 cards.
The same applies to `HL_CHECKPOINTS_DIR`, `HL_TENSORBOARD_DIR` and `HL_KILL_SWITCH` if specified.
If `HL_DATA_DIR_ROOT` is not writable, then `HL_CACHE_PATH` must be set to a writable location and
must be shared and accessible across all nodes and launchers when running training on more than 8 cards.

### Mixtral Training and Examples
* Training of Mixtral is based on https://arxiv.org/abs/2401.04088

### Multi-Card Training Examples
Configure the following for the Mixtral examples below:
* Set the correct path for `HL_DATA_DIR_ROOT`.
* Set the correct values for `HL_TOKENIZER_TYPE` and `HL_DATA_FILE_PREFIX`.
* Add `HL_DATA_CACHE_DIR` and/or `HL_TOKENIZER_MODEL` if necessary.

Refer to [training script settings](#training-script-settings) for details.

* Run Mixtral 8x7b on 32 HPUs, Lazy mode, with BF16 precision, sequence length 32k:
  ```
  HL_HOSTSFILE=$MEGATRON_LM_ROOT/examples/hostsfile \
  HL_TOKEN_DISPATCHER_TYPE='alltoall' \
  HL_DIST_OPTIMIZER=1 \
  HL_MOE_NUM_CAPACITY_BINS=10 \
  HL_NUM_NODES=4 \
  HL_DP=4 \
  HL_TP=8 \
  HL_MOE_EP=1 \
  HL_SEQ_PARALLEL=1 \
  HL_CKP_ACT=3 \
  HL_USE_FUSED_SDPA_WITH_RECOMPUTE=1 \
  $MEGATRON_LM_ROOT/examples/mixtral/pretrain_mixtral.sh
  ```

### Validated Configurations
The following configurations have been validated to be functioning with Gaudi 2:
* DP+TP+SP
* DP+TP+SP+allgather+ExTP
* DP+EP
* DP+EP+allgather+ExTP

Pipeline parallelism (PP) was not validated, therefore it may not work correctly.

# Supported Configuration
| Validated on  | Intel Gaudi Software Version | PyTorch Version | Mode     |
|---------------|------------------------------|-----------------|----------|
| Gaudi 2       | 1.19.0                       | 2.5.1           | Training |

# Known Issues
* Only scripts and configurations mentioned in this README are supported and verified.
