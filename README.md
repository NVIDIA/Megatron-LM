<h1 align="center">
<p>Megatron-LM
</h1>

<h3 align="center">
    <p>Enhanced launcher for Megatron-LM tailored to train LLMs at scale in Slurm-based clusters</p>
</h3>

<!-- TOC -->

- [Introduction](#introduction)
- [Setup](#setup)
    - [Submit a run](#submit-a-run)
- [Data](#data)
    - [Tokenization](#tokenization)
    - [Set the Datasets in Megatron](#set-the-datasets-in-megatron)
    - [Data mixtures](#data-mixtures)
- [Checkpointing](#checkpointing)
    - [Resuming from a checkpoint](#resuming-from-a-checkpoint)
- [Contribute](#contribute)
    - [Copy your wandb logs to a new project.](#copy-your-wandb-logs-to-a-new-project)

<!-- /TOC -->

# Introduction
This repository contains scripts to train large language models (LLMs) at scale using Megatron-LM, specifically tailored for Slurm clusters. It's tailored for large-scale training runs and include mechanism for improved fault tolerance, automatic resumption after crashes or interruptions, automated evaluation submission, and enhanced WANDB logging. Additionally, it provides configurations optimized for peak performance on the Alps Supercomputer.

We also provide scripts to tokenize data using datatrove at scale and tools to convert model weights to the HuggingFace format.

# Setup

1. Clone the Megatron-LM repository to your `iopsstor`.
  ```
  cd /iopsstor/scratch/cscs/$USER/
  git clone https://github.com/swiss-ai/Megatron-LM.git
  ```

2. You need to set the weights & biases environment variable with your key. Ideally you add it `~/.bashrc`.
  ```
  export WANDB_API_KEY=57b215a80f....
  ```
  Now re-login or run `source ~/.bashrc` to make sure it is set. 

## Submit a Run

1. Start the llama 3 8B baseline run by submitting the `submit-llama3-8B.sh` sbatch script. 
  ```
  sbatch submit-llama3-8B.sh
  ```
  You can find all available arguments here: [https://github.com/swiss-ai/Megatron-LM/blob/main/megatron/training/arguments.py](https://github.com/swiss-ai/Megatron-LM/blob/main/megatron/training/arguments.py)
  (We will add <1B, 3B, and 70B baseline runs using the llama3 architecture very soon.)

2. The main logs will be visible ~2min after launching the script in the `Megatron-Clariden` Project on your personal weights and biases page.

3. The local logs will be at `/iopsstor/scratch/cscs/$USER/Megatron-LM/logs` with the slurm outputs and error files at `/iopsstor/scratch/cscs/$USER/Megatron-LM/logs/slurm/training`.

# Data
>[!NOTE]
> On the Alps supercomputer, you can find tokenized datasets in `/capstor/store/cscs/swissai/a06/datasets_tokenized/nemo/sai`

## Tokenization
While Megatron provides data tokenization tools, we use `datatrove`, which enables us reading data in multiple formats (`json`, `parquet`, `csv`...), easy parallelization across multiple nodes, and efficient filtering and deduplication our data.

We have extended `datatrove` ([PR](https://github.com/huggingface/datatrove/pull/304)) to include the [`MegatronDocumentTokenizer`](https://github.com/huggingface/datatrove/blob/22606036e92c8d83268f313f462ee98eceb3fa0b/src/datatrove/pipeline/tokens/megatron_tokenizer.py#L144) pipeline stage, allowing the generation of files containing tokenized documents compatible with NeMo/Megatron.

All Megatron tokenized files, also referred to as *file prefixes*, consist of two file types:
- The `.bin` files contain raw tokens, where each token is either 4 bytes (for vocabularies > 65535) or 2 bytes.
- The `.idx` files contain metadata about the corresponding `.bin` files. For more details on creating these files, refer to [this example](https://github.com/huggingface/datatrove/blob/22606036e92c8d83268f313f462ee98eceb3fa0b/src/datatrove/pipeline/tokens/megatron_tokenizer.py#L59-L72).
>[!CAUTION]
> 🚨 On the Alps Supercomputer, ensure you **DO NOT WRITE**  under **`/capstor/store`**. Always use `/iopsstor/scratch`! 🚨

We include a launcher to tokenize data at scale using multiple nodes in `scripts/tokenization`. Start by preparing your workspace using the `scripts/tokenization/prepare_dumps.py` script, which identifies parquet files in the specified directory, filters them based on criteria (check `--filter-in` and `--filter-out`), and splits the workload evenly across `--n-dumps`.  This process generates *.txt* files for each dump, specifying the files to process.

Once the workspace is ready, configure the tokenization job by setting the tokenizer, the number of datatrove parallel workers per node, and the directory we prepared with `scripts/tokenization/prepare_dumps.py` in `scripts/tokenization/submit_tokenization.sh`. Running this script will submit multiple Slurm jobs, with each job responsible for processing one dump on a single node.
>[!CAUTION]
> 🚨 Ensure the `SBATCH --environment` flag in `scripts/tokenization/tokenize.sh` is correctly configured for your environment 🚨

Before running large-scale jobs, it is recommended to optimize the number of datatrove parallel workers  (`datatrove`'s [`LocalPipelineExecutor`](https://github.com/huggingface/datatrove/blob/22606036e92c8d83268f313f462ee98eceb3fa0b/src/datatrove/executor/local.py#L15) `tasks` & `workers`) and the input file size. You can easily modify the later with `datatrove` using the [`max_file_size`](https://github.com/huggingface/datatrove/blob/22606036e92c8d83268f313f462ee98eceb3fa0b/src/datatrove/pipeline/writers/parquet.py#L21) configuration of the `ParquetWriter`.

For example, on the Alps supercomputer, the best configuration involved processing parquet files of 500 MB with Snappy compression and using 28 datatrove workers per node, achieving a throughput of ~70 million tokens per second per node. More details [here](https://docs.google.com/presentation/d/1t12axPhvjpuxGQWr1xJIioazKeVZ212ewuyil5uWMnQ/edit#slide=id.p).

## Set the Datasets in Megatron
In Megatron, we will specify datasets using the `--data-path` configuration. This field expects a list of tokenized file prefixes, and optionally a weight for each file. If the weights are not specified, Megatron will automatically compute them based on the number of sequences it can extract from each file prefix. To simplify this process, we will use the `scripts/create_data_config.py` script to automatically generate the list of file prefixes by recursively searching for all file prefixes in the directories specified in the `DATASETS` variable in the launcher script.

## Data mixtures

As we just mentioned in the previous section, we only need to specify a directory with file prefixes, and Megatron will assign a weight to each prefix based on the number of samples it contains.  

For data mixtures, we will follow the same workflow but add an additional step where we create a folder for our data mixture containing symlinks to the original datasets.  

To achieve this, we have developed the `scripts/tools/create_data_mixture.py` script, where we only need to specify the `--folders` for the mixture, the `--weights` of each folder, and an `--output` directory to store the created data mixture.  

Upon successfully creating a mixture, we will see its statistics, such as the number of tokens, the number of file prefixes per dataset, and the total size of the mixture.  

Keep in mind that the mixture will be created **without repetition**. This means that we will construct the mixture while respecting the weights until a dataset is exhausted.

# Checkpointing
>[!CAUTION]
> 🚨 On the Alps Supercomputer, ensure you **DO NOT WRITE**  under **`/capstor/store`**. Use `/iopsstor/scratch` instead 🚨

Checkpointing is a critical component of LLM training. It must be fast to minimize disruption to training and complete, meaning it captures not only the model weights but also the optimizer states, DataLoader states and RNG states.

We use the PyTorch distributed checkpointing backend (`--ckpt-format torch_dist`) leveraging the asynchronous checkpointing option and parallelizing both storing and loading checkpoints within all the devices. The checkpoints are topology-agnostic, allowing them to be loaded with a different topology from the one used to store them. Each process will store 2 files (Default value for `thread_count` [[1](https://github.com/NVIDIA/Megatron-LM/blob/55cdfc1e8bfe116f54dbe6e48ff70cc92c9f4a91/megatron/core/dist_checkpointing/strategies/torch.py#L614)], [[2](https://pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.FileSystemWriter)]) containing the state of the run.

In Alps, writing a Llama3-70B checkpoint to `/iopsstor/scratch` blocks training for approximately 40 seconds. Be aware that these checkpoints are huge: A Llama3-70B model checkpoint takes **920 GB** and the Llama3-8B version takes **105 GB** (model weights in bf16 and optimizer states in fp32).

In the launcher set `CHECKPOINT_STEPS` as the frequency every how many steps you want to store a checkpoint.
## Resuming from a checkpoint
In Megatron, we use `--save` and `--load` to specify where checkpoints are read from and saved to.  

With each checkpoint save, the file `latest_checkpointed_iteration.txt` will be updated in the `--save` directory, containing a reference to the last saved checkpoint. There is currently no feature to just keep the last `N` checkpoints.

When resuming a run, the application will attempt to load the checkpoint referenced in the `latest_checkpointed_iteration.txt` file from the `--load` directory. To load a checkpoint from a different iteration, you will need to manually modify the reference inside `latest_checkpointed_iteration.txt`.

# Contribute

You can submit issues and create branches on `https://github.com/swiss-ai/Megatron-LM`. The main branch is protected so you won't be able to directly commit to it.

1. Pull the latest version
  ```
  git pull
  ```

2. Create a branch with a descriptive name
  ```
  git checkout -b my-contribution-branch
  ```

3. Make your changes locally and commit them. Make sure your latest commit works as expected. Do not commit unnecessary files. 

4. sync with the latest main
  ```
  git pull origin main
  ```

5. Push your commit to your branch
  ```
  git push origin my-contribution-branch
  ```

6. Go to `https://github.com/swiss-ai/Megatron-LM. 

7. GitHub will have a pop-up proposing to you to create a new PR. Click on that green button. 

8. IMPORTANT: Change the base repository to `swiss-ai/Megatron-LM` or your PR will be submitted to the official Nvidia repo!

9. In your PR explain what you did and why. 

10. If you make a contribution with a run, either use the wandb web UI to move your run into a new folder (select the run in the runs table and a move button will appear) or use the `scripts/copy_wandb_project.py` to create a copy in a new fresh wandb project that contains only the relevant run and the respective baseline (but in this case all runs will start from step 0). See instructions in the next section on how to use the copy script.

11. If you want to archive your logs, move your log folder `~/Megatron-LM/logs/Meg-Runs/your_project_name/your_experiment_name` to `/capstor/store/cscs/swissai/a06/megatron_runs` and share the path in your PR.

12. Add `ischlag` or `TJ-Solergibert` to review your PR.

13. Use this command to return to main branch.
  ```
  git checkout main
  ```

## Copy Your Wandb Logs to a New Project 

To submit your contribution you have to provide a fresh wandb project with the relevant logs. 

1. Go to your `wandb.ai` page and create a new public project (e.g. `contrib_linear_attention`)

2. Use the `copy_wandb_project.py` to create a copy of your run from your personal wandb project. Make sure it contains the relevant ablations and baselines.
  ```
  python scripts/copy_wandb_project.py -se myusername -de myusername -sp Megatron-Clariden -dp contrib_linear_attention -r f5v94x1q  
  ```
